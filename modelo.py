"""Carga de artefactos, preprocesamiento, inferencia ONNX y mapas de activación.

El preprocesamiento replica exactamente el del notebook:
  escala de grises → 256×256 bilineal → 224×224 bilineal → 3 canales → [0, 1] → normalización ImageNet.
Cualquier diferencia aquí degradaría las predicciones sin dar error, por eso vive en un solo lugar.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image

from config import DIR_ARTEFACTOS, DIR_EJEMPLOS, RUTA_DB, RUTA_METADATOS, RUTA_PESOS_CAM


# ── Contexto compartido por todas las vistas ─────────────────────────────────
@dataclass
class Contexto:
    meta: dict | None = None
    sesion: object | None = None
    pesos_cam: np.ndarray | None = None
    ejemplos: list[Path] = field(default_factory=list)
    hay_db: bool = False
    umbral: float = 0.5
    faltantes: list[str] = field(default_factory=list)

    @property
    def listo(self) -> bool:
        return not self.faltantes

    @property
    def principal(self) -> dict:
        return self.meta["metricas_principal"]

    @property
    def base(self) -> dict:
        return self.meta["metricas_base"]


@st.cache_data(show_spinner=False)
def _leer_metadatos(ruta: str, _mtime: float) -> dict:
    with open(ruta, encoding="utf-8") as f:
        return json.load(f)


@st.cache_resource(show_spinner=False)
def _abrir_sesion(ruta: str, _mtime: float):
    import onnxruntime as ort
    opciones = ort.SessionOptions()
    opciones.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(ruta, sess_options=opciones, providers=["CPUExecutionProvider"])


def construir_contexto() -> Contexto:
    ctx = Contexto()
    if not RUTA_METADATOS.exists():
        ctx.faltantes.append(str(RUTA_METADATOS.relative_to(RUTA_METADATOS.parents[1])))
        return ctx
    ctx.meta = _leer_metadatos(str(RUTA_METADATOS), RUTA_METADATOS.stat().st_mtime)
    ruta_onnx = DIR_ARTEFACTOS / ctx.meta["modelo"]["archivo"]
    for ruta in (ruta_onnx, RUTA_PESOS_CAM):
        if not ruta.exists():
            ctx.faltantes.append(f"artefactos/{ruta.name}")
    if ctx.faltantes:
        return ctx
    ctx.sesion = _abrir_sesion(str(ruta_onnx), ruta_onnx.stat().st_mtime)
    ctx.pesos_cam = np.load(RUTA_PESOS_CAM)
    ctx.ejemplos = sorted(DIR_EJEMPLOS.glob("*.jpeg")) if DIR_EJEMPLOS.exists() else []
    ctx.hay_db = RUTA_DB.exists()
    ctx.umbral = float(ctx.meta["umbral"])
    return ctx


# ── Preprocesamiento ─────────────────────────────────────────────────────────
def cargar_imagen_gris(origen, tam: int = 256) -> Image.Image:
    """Misma función del notebook: decodificación reducida (draft) + escala de grises + redimensión."""
    img = Image.open(origen)
    img.draft("L", (tam, tam))
    return img.convert("L").resize((tam, tam), Image.BILINEAR)


def parece_radiografia(origen) -> tuple[bool, float]:
    """Heurística simple: una radiografía casi no tiene saturación de color.

    Devuelve (es_probable, saturación media en [0, 1]). No reemplaza un detector de dominio,
    solo evita que alguien suba una foto a color y crea que el resultado tiene sentido.
    """
    img = Image.open(origen).convert("RGB")
    img.thumbnail((128, 128))
    hsv = np.asarray(img.convert("HSV"), dtype=np.float32)
    saturacion = float(hsv[..., 1].mean() / 255)
    return saturacion < 0.12, saturacion


def preprocesar(img_gris: Image.Image, meta: dict) -> np.ndarray:
    p = meta["preprocesamiento"]
    tam = p["tam_entrada"]
    if img_gris.size != (p["tam_cache"], p["tam_cache"]):
        img_gris = img_gris.convert("L").resize((p["tam_cache"], p["tam_cache"]), Image.BILINEAR)
    arr = np.asarray(img_gris.resize((tam, tam), Image.BILINEAR), dtype=np.float32) / 255.0
    media = np.array(p["media"], dtype=np.float32)[:, None, None]
    desv = np.array(p["desviacion"], dtype=np.float32)[:, None, None]
    x = (np.repeat(arr[None], 3, axis=0) - media) / desv
    return x[None].astype(np.float32)                       # (1, 3, 224, 224)


# ── Inferencia ───────────────────────────────────────────────────────────────
@dataclass
class Resultado:
    prob_neumonia: float
    cam: np.ndarray            # (224, 224) en [0, 1]
    milisegundos: float


def predecir(ctx: Contexto, img_gris: Image.Image) -> Resultado:
    x = preprocesar(img_gris, ctx.meta)
    t0 = time.perf_counter()
    logits, mapas = ctx.sesion.run(None, {ctx.meta["modelo"]["entrada"]: x})
    ms = (time.perf_counter() - t0) * 1000
    z = logits[0] - logits[0].max()
    prob = float(np.exp(z)[1] / np.exp(z).sum())
    return Resultado(prob, mapa_activacion(ctx, mapas[0], ctx.meta["preprocesamiento"]["tam_entrada"]), ms)


def mapa_activacion(ctx: Contexto, mapas: np.ndarray, tam: int, clase: int = 1) -> np.ndarray:
    """CAM exacto para GAP + capa lineal: suma de mapas ponderada por los pesos de la clase."""
    cam = np.maximum(np.tensordot(ctx.pesos_cam[clase], mapas, axes=(0, 0)), 0).astype(np.float32)
    cam = np.asarray(Image.fromarray(cam).resize((tam, tam), Image.BILINEAR))
    rango = cam.max() - cam.min()
    return (cam - cam.min()) / rango if rango > 1e-8 else np.zeros_like(cam)


def _calor(v: np.ndarray) -> np.ndarray:
    """Rampa ámbar → naranja → rojo marcador: el calor sube hacia el color de la clase NEUMONÍA."""
    paradas = np.array([[0.95, 0.76, 0.30], [0.93, 0.45, 0.18], [0.85, 0.21, 0.21]], dtype=np.float32)
    v = np.clip(v, 0, 1)[..., None]
    bajo = paradas[0] + (paradas[1] - paradas[0]) * np.clip(v * 2, 0, 1)
    alto = paradas[1] + (paradas[2] - paradas[1]) * np.clip(v * 2 - 1, 0, 1)
    return np.where(v < 0.5, bajo, alto)


def superponer(img_gris: Image.Image, cam: np.ndarray, opacidad: float = 0.45) -> Image.Image:
    """Mezcla el mapa de calor solo donde hay activación: las zonas frías conservan la radiografía."""
    tam = cam.shape[0]
    base = np.asarray(img_gris.convert("L").resize((tam, tam), Image.BILINEAR), dtype=np.float32) / 255
    base = np.repeat(base[..., None], 3, axis=-1)
    alfa = (opacidad * np.clip(cam * 1.5 - 0.25, 0, 1))[..., None]
    mezcla = ((1 - alfa) * base + alfa * _calor(cam)).clip(0, 1)
    return Image.fromarray((mezcla * 255).astype(np.uint8))


def clase_predicha(prob: float, umbral: float) -> str:
    return "NEUMONIA" if prob >= umbral else "NORMAL"
