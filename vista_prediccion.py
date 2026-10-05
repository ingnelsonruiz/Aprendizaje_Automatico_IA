"""Pestaña «Predicción»: simulador de diagnóstico en tiempo real."""

import plotly.graph_objects as go
import streamlit as st

from config import AMBER, GREEN, PINK
from ui import estilo, mostrar, seccion


def _controles(ctx):
    """Dibuja un slider por cada variable «media» y devuelve los valores elegidos."""
    valores = {}
    grid = st.columns(3)
    for i, nombre in enumerate(ctx.columnas_media()):
        minimo = float(ctx.X[nombre].min())
        maximo = float(ctx.X[nombre].max())
        medio = float(ctx.X[nombre].mean())
        paso = (maximo - minimo) / 200 if maximo > minimo else 0.01
        with grid[i % 3]:
            valores[nombre] = st.slider(nombre, minimo, maximo, medio, paso)
    return valores


def _velocimetro(p_mal):
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=p_mal * 100,
        number=dict(suffix="%", font=dict(size=46)),
        title=dict(text="PROBABILIDAD DE MALIGNIDAD", font=dict(size=15)),
        gauge=dict(axis=dict(range=[0, 100], tickcolor="#667788"),
                   bar=dict(color=PINK if p_mal >= 0.5 else GREEN),
                   bgcolor="rgba(13,27,42,0.4)",
                   steps=[dict(range=[0, 50], color="rgba(0,255,136,0.12)"),
                          dict(range=[50, 100], color="rgba(255,45,120,0.12)")],
                   threshold=dict(line=dict(color=AMBER, width=4), thickness=0.8, value=50))))
    return estilo(fig, alto=360)


def render(ctx):
    seccion("SIMULADOR DE DIAGNÓSTICO EN TIEMPO REAL")
    st.caption("Ajusta las 10 características «media» del núcleo. Las otras 20 variables se mantienen "
               "en su promedio.")

    valores = _controles(ctx)
    p_mal = ctx.probabilidad_maligno(valores)

    st.write("")
    izq, der = st.columns(2)
    with izq:
        mostrar(_velocimetro(p_mal))
    with der:
        st.write("")
        st.write("")
        clase = "veredicto-m" if p_mal >= 0.5 else "veredicto-b"
        texto = "⚠ MALIGNO" if p_mal >= 0.5 else "✔ BENIGNO"
        st.markdown(f'<div class="{clase}">{texto}</div>', unsafe_allow_html=True)
        st.write("")
        st.markdown(f'<div class="panel">Modelo: <b>{ctx.modelo_sel}</b><br>'
                    f'Benigno: <b>{(1-p_mal)*100:.1f}%</b> · Maligno: <b>{p_mal*100:.1f}%</b></div>',
                    unsafe_allow_html=True)
    st.caption("⚠ Herramienta académica. No reemplaza el diagnóstico médico profesional.")
