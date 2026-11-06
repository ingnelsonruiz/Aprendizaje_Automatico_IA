"""Pestaña «Pipeline»: preprocesamiento y laboratorio interactivo de aumento de datos."""

import pandas as pd
import streamlit as st
from PIL import Image, ImageEnhance, ImageOps

from config import CYAN, GREEN, PINK
from modelo import cargar_imagen_gris, clase_predicha, predecir
from ui import imagen, kpi, panel, seccion, tabla


JUSTIFICACION = ["Variaciones de encuadre y distancia del equipo",
                 "Posicionamiento imperfecto del paciente pediátrico",
                 "Tórax no siempre centrado",
                 "Diferencias de exposición entre equipos de rayos X",
                 "El corazón está a la izquierda: voltear crea anatomía irreal (dextrocardia)"]


def _aumentar(img, rotacion, zoom, dx, dy, brillo, contraste, voltear):
    tam = img.size[0]
    if zoom > 1.0:                                     # recorte central = acercamiento
        lado = int(tam / zoom)
        o = (tam - lado) // 2
        img = img.crop((o, o, o + lado, o + lado)).resize((tam, tam), Image.BILINEAR)
    img = img.rotate(rotacion, resample=Image.BILINEAR,
                     translate=(int(dx * tam), int(dy * tam)), fillcolor=0)
    img = ImageEnhance.Brightness(img).enhance(brillo)
    img = ImageEnhance.Contrast(img).enhance(contraste)
    return ImageOps.mirror(img) if voltear else img


def render(ctx):
    p = ctx.meta["preprocesamiento"]
    seccion("PREPROCESAMIENTO (IDÉNTICO EN ENTRENAMIENTO Y EN ESTA APP)")
    pasos = '<span class="flecha">➜</span>'.join(
        f'<div class="step"><b>Paso {i}</b><span>{t}</span></div>' for i, t in enumerate(p["pasos"], 1))
    st.markdown(f'<div class="flow">{pasos}</div>', unsafe_allow_html=True)
    panel("Las radiografías originales miden entre ~400 y ~2 900 píxeles de ancho. El notebook las guarda en "
          "memoria a 256×256 en escala de grises (decodificación JPEG reducida con <code>draft</code>) y la red "
          "recibe 224×224×3, porque EfficientNet fue preentrenada en ImageNet con imágenes a color normalizadas "
          "con esa media y desviación. Esta app usa la misma función de carga para no introducir diferencias.")

    seccion("AUMENTO DE DATOS APLICADO EN ENTRENAMIENTO")
    tabla(pd.DataFrame(list(zip(ctx.meta["entrenamiento"]["aumento_datos"], JUSTIFICACION)),
                       columns=["Transformación", "Justificación"]))

    seccion("LABORATORIO: ¿CAMBIA LA PREDICCIÓN SI ALTERO LA IMAGEN?")
    if not ctx.ejemplos:
        st.info("No hay radiografías de ejemplo en artefactos/ejemplos.")
        return
    nombre = st.selectbox("Radiografía de prueba", [e.name for e in ctx.ejemplos], key="lab_img")
    original = cargar_imagen_gris(next(e for e in ctx.ejemplos if e.name == nombre), p["tam_cache"])

    c1, c2, c3 = st.columns(3)
    rot = c1.slider("Rotación (°)", -30, 30, 0, help="En entrenamiento se usó ±10°.")
    zoom = c1.slider("Acercamiento", 1.0, 1.5, 1.0, 0.05, help="En entrenamiento: recortes del 80–100 %.")
    dx = c2.slider("Desplazamiento horizontal", -0.15, 0.15, 0.0, 0.01)
    dy = c2.slider("Desplazamiento vertical", -0.15, 0.15, 0.0, 0.01)
    brillo = c3.slider("Brillo", 0.5, 1.5, 1.0, 0.05, help="En entrenamiento: ±20 %.")
    contraste = c3.slider("Contraste", 0.5, 1.5, 1.0, 0.05, help="En entrenamiento: ±20 %.")
    voltear = st.toggle("Volteo horizontal (NO usado en entrenamiento)", value=False)
    aumentada = _aumentar(original, rot, zoom, dx, dy, brillo, contraste, voltear)

    r0, r1 = predecir(ctx, original), predecir(ctx, aumentada)
    a, b = st.columns(2)
    with a:
        imagen(original, "Original")
        st.markdown(kpi(f"{r0.prob_neumonia*100:.1f}%", f"P(neumonía) · {clase_predicha(r0.prob_neumonia, ctx.umbral)}",
                        CYAN), unsafe_allow_html=True)
    with b:
        imagen(aumentada, "Aumentada")
        delta = (r1.prob_neumonia - r0.prob_neumonia) * 100
        color = GREEN if abs(delta) < 10 else PINK
        st.markdown(kpi(f"{r1.prob_neumonia*100:.1f}%", f"P(neumonía) · {clase_predicha(r1.prob_neumonia, ctx.umbral)} "
                        f"· Δ {delta:+.1f} pp", color), unsafe_allow_html=True)
    st.caption("Un modelo robusto mantiene la predicción estable dentro de los rangos usados en el entrenamiento; "
               "fuera de ellos (por ejemplo, rotaciones de 30°) es esperable que cambie.")
