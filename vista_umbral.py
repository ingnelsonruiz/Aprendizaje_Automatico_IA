"""Pestaña «Umbral clínico»: cómo mover el punto de corte cambia los errores."""

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from config import AMBER, CYAN, GREEN, PINK
from ui import estilo, kpi, mostrar, panel, seccion


def _conteos(proba, y_real, umbral):
    pred = (proba >= umbral).astype(int)
    vp = int(((pred == 1) & (y_real == 1)).sum())
    fn = int(((pred == 0) & (y_real == 1)).sum())
    fp = int(((pred == 1) & (y_real == 0)).sum())
    sens = vp / (vp + fn) if (vp + fn) else 0.0
    prec = vp / (vp + fp) if (vp + fp) else 1.0
    return vp, fn, fp, sens, prec


def render(ctx):
    seccion("AJUSTE DEL UMBRAL DE DECISIÓN")
    panel("En salud, un <b>falso negativo</b> (un tumor maligno clasificado como benigno) es el error más "
          "grave. Si bajas el umbral, el modelo detecta más casos malignos: sube la sensibilidad, pero "
          "aparecen más falsas alarmas.")

    umbral = st.slider("Umbral de probabilidad para clasificar como MALIGNO", 0.05, 0.95, 0.50, 0.01)
    proba = ctx.activo["proba"]
    _, fn, fp, sens, prec = _conteos(proba, ctx.y_prueba, umbral)

    a, b, c, d = st.columns(4)
    a.markdown(kpi(f"{sens*100:.1f}%", "Sensibilidad", GREEN), unsafe_allow_html=True)
    b.markdown(kpi(f"{prec*100:.1f}%", "Precisión", CYAN), unsafe_allow_html=True)
    c.markdown(kpi(f"{fn}", "Malignos no detectados", PINK), unsafe_allow_html=True)
    d.markdown(kpi(f"{fp}", "Falsas alarmas", AMBER), unsafe_allow_html=True)

    cortes = np.linspace(0.05, 0.95, 91)
    curvas = [_conteos(proba, ctx.y_prueba, t) for t in cortes]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=cortes, y=[c[3] for c in curvas], name="Sensibilidad",
                             line=dict(color=GREEN, width=3)))
    fig.add_trace(go.Scatter(x=cortes, y=[c[4] for c in curvas], name="Precisión",
                             line=dict(color=CYAN, width=3)))
    fig.add_vline(x=umbral, line=dict(color=AMBER, dash="dash", width=2))
    fig.update_layout(xaxis_title="Umbral", yaxis_title="Valor", yaxis=dict(range=[0, 1.05]))
    mostrar(estilo(fig, alto=400, titulo="SENSIBILIDAD VS. PRECISIÓN SEGÚN EL UMBRAL"))
