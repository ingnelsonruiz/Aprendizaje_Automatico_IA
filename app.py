"""Radiografía Inteligente — detección de neumonía con redes neuronales convolucionales.

Proyecto ACA 2 · Aprendizaje Automático · Especialización en Inteligencia Artificial (CUN)

Este archivo solo orquesta: configura la página, construye el contexto (modelo ONNX + metadatos)
y entrega cada pestaña a su módulo. Para agregar una pestaña: crear `vista_mi_vista.py` con
`render(ctx)` y registrarla en `PESTANAS`.
"""

import base64
import io

import numpy as np
import streamlit as st
from PIL import Image

st.set_page_config(
    page_title="Radiografía Inteligente · Neumonía",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded",
)

import estilos                                                                   # noqa: E402
from ayuda import ayuda, expansor_indicadores                                    # noqa: E402
from config import (AMBER, ASIGNATURA, CODIGOS, CYAN, DOCENTE, EQUIPO, GREEN,     # noqa: E402
                    GRUPO, PINK, PROYECTO, SUBTITULO, TITULO_APP, URL_APP,
                    URL_REPO, URL_VIDEO, VIOLET)
from modelo import cargar_imagen_gris, construir_contexto, predecir, superponer  # noqa: E402
from ui import kpi, panel                                                        # noqa: E402
import vista_arquitectura, vista_dataset, vista_documentacion, vista_entrenamiento  # noqa: E402
import vista_equipo, vista_historia, vista_metricas, vista_pipeline              # noqa: E402
import vista_prediccion, vista_sql, vista_umbral                                 # noqa: E402

PESTANAS = [
    ("La historia", vista_historia),
    ("El dataset", vista_dataset),
    ("Consultas SQL", vista_sql),
    ("Pipeline", vista_pipeline),
    ("Arquitectura", vista_arquitectura),
    ("Entrenamiento", vista_entrenamiento),
    ("Métricas", vista_metricas),
    ("Umbral clínico", vista_umbral),
    ("Predicción", vista_prediccion),
    ("Equipo", vista_equipo),
    ("Documentación", vista_documentacion),
]


def barra_lateral(ctx):
    with st.sidebar:
        st.markdown("### Visor")
        if ctx.listo:
            ctx.umbral = st.slider("Umbral de decisión sobre P(neumonía)", 0.05, 0.95,
                                   float(ctx.meta["umbral"]), 0.01, help=ayuda("Umbral de decisión"))
            st.caption(f"Umbral óptimo en validación: **{ctx.meta['umbral']:.2f}**")
            st.markdown("---")
            m = ctx.meta["modelo"]
            st.markdown(f"**Modelo:** EfficientNet-B0  \n**Parámetros:** {m['parametros']:,}  \n"
                        f"**Formato:** ONNX en CPU  \n**Entrenado:** {ctx.meta['fecha_entrenamiento']}")
        st.markdown("---")
        st.markdown(f"**{PROYECTO}**  \n{ASIGNATURA}  \nDocente: {DOCENTE}  \n"
                    f"{CODIGOS[0]}, {CODIGOS[1].lower()}, {CODIGOS[2]}")
        st.markdown(f"**{GRUPO}**  \n" + "  \n".join(n for n, _ in EQUIPO))
        enlaces = [(t, u) for t, u in [("Aplicación", URL_APP), ("Video", URL_VIDEO), ("Repositorio", URL_REPO)] if u]
        if enlaces:
            st.markdown("---")
            st.markdown("  \n".join(f"[{t}]({u})" for t, u in enlaces))


def _png_base64(img) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


@st.cache_data(show_spinner=False)
def _pelicula_portada(ruta: str, _mtime: float):
    """Radiografía de prueba con su mapa de activación, para la portada."""
    ctx = construir_contexto()
    img = cargar_imagen_gris(ruta, ctx.meta["preprocesamiento"]["tam_cache"])
    r = predecir(ctx, img)
    base = img.convert("L").resize((448, 448), Image.BILINEAR)
    cam = np.asarray(Image.fromarray((r.cam * 255).astype(np.uint8)).resize((448, 448), Image.BILINEAR)) / 255
    h, w = r.cam.shape
    # Proporción del mapa que cae dentro de la región torácica central (no en bordes ni marcadores)
    masa_central = float(r.cam[int(.15 * h):int(.85 * h), int(.18 * w):int(.82 * w)].sum() / (r.cam.sum() + 1e-8))
    return _png_base64(base), _png_base64(superponer(base, cam, 0.62)), r.prob_neumonia, masa_central


def _elegir_portada(ctx):
    """Neumonía de ejemplo detectada con alta confianza cuyo mapa cae más dentro de los pulmones."""
    candidatos = [e for e in ctx.ejemplos if not e.name.startswith("normal")]
    if not candidatos:
        return None
    datos = {e: _pelicula_portada(str(e), e.stat().st_mtime) for e in candidatos}
    seguros = [e for e, d in datos.items() if d[2] >= 0.9] or list(datos)
    return datos[max(seguros, key=lambda e: datos[e][3])][:3]


def _coma(v, dec=1):
    return f"{v:.{dec}f}".replace(".", ",")


def encabezado(ctx):
    if not ctx.listo:
        st.markdown(f'<div class="hero"><div><h1>{TITULO_APP}</h1><p class="bajada">{SUBTITULO}</p></div></div>',
                    unsafe_allow_html=True)
        return
    r = ctx.principal
    n_img = r["vp"] + r["fp"] + r["vn"] + r["fn"]
    n_pac = next((p["pacientes"] for p in ctx.meta["dataset"]["particiones"] if p["split"] == "prueba"), None)
    portada = _elegir_portada(ctx)

    lectura = "".join(
        f'<div class="{clase}" title="{ayuda(clave)}"><div class="v">{valor}</div><div class="l">{etiqueta}</div></div>'
        for valor, etiqueta, clave, clase in [
            (f"{_coma(r['recall_sensibilidad'] * 100)}&#8239;%", "de las neumonías detectadas", "Sensibilidad (recall)", "rojo"),
            (f"{_coma(r['especificidad'] * 100)}&#8239;%", "de los sanos bien descartados", "Especificidad", ""),
            (_coma(r["f1_macro"], 3), "F1 macro", "F1", ""),
            (_coma(r["roc_auc"], 3), "Área bajo la curva ROC", "ROC-AUC", ""),
        ])
    pelicula = ""
    if portada:
        base64_base, base64_cam, prob = portada
        pelicula = f'''
<div>
  <div class="negatoscopio">
    <div class="pelicula" role="img" aria-label="Radiografía de tórax con el mapa de activación del modelo">
      <img src="data:image/png;base64,{base64_base}" alt="">
      <img class="cam" src="data:image/png;base64,{base64_cam}" alt="">
      <div class="barrido"></div>
      <div class="anot si"><b>Tórax AP</b><br>Paciente pediátrico<br>Conjunto de prueba</div>
      <div class="anot sd"><b>EfficientNet-B0</b><br>Entrada 224 × 224<br>ONNX en CPU</div>
      <div class="anot ii">Mapa de activación<br>clase neumonía</div>
      <div class="anot id">P(neumonía) <span class="alerta">{_coma(prob, 3)}</span><br>Umbral {_coma(ctx.meta["umbral"], 2)}</div>
    </div>
  </div>
  <p class="pie-pelicula">Radiografía real que el modelo nunca vio. El color marca las regiones que más
  aumentaron la probabilidad de neumonía.</p>
</div>'''
    nota = (f"Resultados sobre {n_img:,} radiografías de prueba".replace(",", ".")
            + (f" de {n_pac:,} pacientes".replace(",", ".") if n_pac else "")
            + " que no participaron en el entrenamiento, con el umbral elegido en validación.")
    st.markdown(f'''
<div class="hero">
  <div>
    <h1>{TITULO_APP}</h1>
    <p class="bajada">Una red neuronal convolucional lee radiografías de tórax de niños, señala las que
    muestran neumonía y marca en la imagen la zona que motivó su decisión.</p>
    <div class="lectura">{lectura}</div>
    <p class="nota">{nota}</p>
  </div>
  {pelicula}
</div>''', unsafe_allow_html=True)
    expansor_indicadores(["Sensibilidad (recall)", "Especificidad", "F1", "ROC-AUC"])


def main():
    estilos.aplicar()
    with st.spinner("Cargando modelo y metadatos..."):
        ctx = construir_contexto()
    barra_lateral(ctx)
    encabezado(ctx)

    if not ctx.listo:
        panel("⚠️ <b>Faltan artefactos del modelo:</b> " + ", ".join(ctx.faltantes) +
              ".<br>Ejecute el notebook <code>ACA2_Neumonia_CNN.ipynb</code> en Colab, descargue "
              "<code>artefactos_aca2.zip</code> y descomprímalo en la raíz del repositorio.")
        st.write("")
        vista_documentacion.render(ctx)
        return

    pestanas = st.tabs([etiqueta for etiqueta, _ in PESTANAS])
    for contenedor, (_, modulo) in zip(pestanas, PESTANAS):
        with contenedor:
            modulo.render(ctx)

    st.markdown(f'<div class="aviso">{TITULO_APP}, {PROYECTO}, {GRUPO}. Prototipo académico: no sustituye '
                f'el diagnóstico de un profesional de la salud.</div>', unsafe_allow_html=True)


main()
