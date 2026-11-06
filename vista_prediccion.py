"""Pestaña «Predicción»: inferencia en tiempo real con mapa de activación."""

import plotly.graph_objects as go
import streamlit as st

from config import AMBER, CYAN, GREEN, PINK
from modelo import cargar_imagen_gris, clase_predicha, parece_radiografia, predecir, superponer
from ui import estilo, imagen, mostrar, panel, seccion


def _medidor(prob, umbral):
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=prob * 100, number=dict(suffix=" %", font=dict(size=40)),
        title=dict(text="Probabilidad de neumonía"),
        gauge=dict(axis=dict(range=[0, 100]), bar=dict(color=PINK if prob >= umbral else GREEN),
                   steps=[dict(range=[0, umbral * 100], color="rgba(29,143,122,0.14)"),
                          dict(range=[umbral * 100, 100], color="rgba(217,54,54,0.14)")],
                   threshold=dict(line=dict(color=AMBER, width=4), value=umbral * 100))))
    return estilo(fig, alto=300)


def render(ctx):
    seccion("ANALIZAR UNA RADIOGRAFÍA")
    fuente = st.radio("Origen de la imagen", ["Ejemplos del conjunto de prueba", "Subir radiografía"],
                      horizontal=True)
    if fuente == "Subir radiografía":
        archivo = st.file_uploader("Radiografía de tórax AP (JPG o PNG)", type=["jpg", "jpeg", "png"])
        if archivo is None:
            st.info("Suba una imagen para analizarla, o use los ejemplos.")
            return
        es_rx, sat = parece_radiografia(archivo)
        if not es_rx:
            st.warning(f"La imagen tiene color (saturación media {sat:.0%}): no parece una radiografía. "
                       "El resultado no tendrá validez.")
        archivo.seek(0)
        origen, etiqueta_real = archivo, None
    else:
        if not ctx.ejemplos:
            st.info("No hay ejemplos disponibles en artefactos/ejemplos.")
            return
        nombre = st.selectbox("Radiografía", [e.name for e in ctx.ejemplos])
        origen = next(e for e in ctx.ejemplos if e.name == nombre)
        etiqueta_real = "NORMAL" if nombre.startswith("normal") else "NEUMONIA"

    img = cargar_imagen_gris(origen, ctx.meta["preprocesamiento"]["tam_cache"])
    r = predecir(ctx, img)
    clase = clase_predicha(r.prob_neumonia, ctx.umbral)

    izq, centro, der = st.columns([1, 1, 1])
    with izq:
        imagen(img, "Radiografía (preprocesada)")
    with centro:
        opacidad = st.slider("Opacidad del mapa de calor", 0.0, 1.0, 0.5, 0.05)
        imagen(superponer(img, r.cam, opacidad), "Mapa de activación (CAM) para «neumonía»")
    with der:
        mostrar(_medidor(r.prob_neumonia, ctx.umbral))
        css = "veredicto-m" if clase == "NEUMONIA" else "veredicto-b"
        st.markdown(f'<div class="{css}">{"NEUMONÍA" if clase == "NEUMONIA" else "NORMAL"}</div>',
                    unsafe_allow_html=True)
        st.caption(f"Umbral activo {ctx.umbral:.2f} · inferencia {r.milisegundos:.0f} ms en CPU")
        if etiqueta_real:
            acierto = etiqueta_real == clase
            st.markdown(f"**Etiqueta real:** {etiqueta_real} {'✅' if acierto else '❌'}")

    margen = abs(r.prob_neumonia - ctx.umbral)
    if margen < 0.10:
        st.warning("La probabilidad está muy cerca del umbral: caso dudoso que debería priorizarse para lectura humana.")
    panel("<b>Cómo leer el mapa:</b> las zonas en ámbar y rojo son las que más aumentaron la probabilidad de "
          "neumonía. En un caso positivo se espera que coincidan con opacidades en los campos pulmonares. "
          "Si el calor aparece en bordes, letras o fuera del tórax, desconfíe del resultado.<br><br>"
          "⚠️ <b>Prototipo académico.</b> Entrenado con radiografías pediátricas (1 a 5 años) de un único hospital; "
          "no es un dispositivo médico ni reemplaza la lectura de un radiólogo.")
