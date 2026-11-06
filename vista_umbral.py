"""Pestaña «Umbral clínico»: efecto del punto de corte (curvas calculadas en validación)."""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from ayuda import ayuda
from config import AMBER, CYAN, GREEN, PINK
from ui import estilo, kpi, mostrar, panel, seccion


def render(ctx):
    seccion("AJUSTE DEL UMBRAL DE DECISIÓN")
    panel("En tamizaje, un <b>falso negativo</b> (neumonía no detectada) es el error más grave. Bajar el umbral "
          "aumenta la sensibilidad a costa de más falsas alarmas. Las curvas se calcularon en el conjunto de "
          "<b>validación</b>, que es donde se elige el umbral; el conjunto de prueba se reserva para la evaluación "
          "final. Mueva el control <b>Umbral de decisión</b> de la barra lateral: también cambia la pestaña "
          "Predicción.")

    b = pd.DataFrame(ctx.meta["barrido_umbral"])
    fila = b.iloc[(b["umbral"] - ctx.umbral).abs().idxmin()]
    c = st.columns(4)
    c[0].markdown(kpi(f"{ctx.umbral:.2f}", "Umbral activo", AMBER, ayuda("Umbral de decisión")), unsafe_allow_html=True)
    c[1].markdown(kpi(f"{fila['recall_neumonia']*100:.1f}%", "Sensibilidad (val.)", PINK,
                      ayuda("Sensibilidad (recall)")), unsafe_allow_html=True)
    c[2].markdown(kpi(f"{fila['especificidad']*100:.1f}%", "Especificidad (val.)", GREEN,
                      ayuda("Especificidad")), unsafe_allow_html=True)
    c[3].markdown(kpi(f"{fila['f1_macro']:.3f}", "F1 macro (val.)", CYAN, ayuda("F1")), unsafe_allow_html=True)

    fig = go.Figure()
    for col, nombre, color in [("recall_neumonia", "Sensibilidad", PINK), ("especificidad", "Especificidad", GREEN),
                               ("f1_macro", "F1 macro", CYAN)]:
        fig.add_trace(go.Scatter(x=b["umbral"], y=b[col], name=nombre, line=dict(color=color, width=3)))
    fig.add_vline(x=ctx.umbral, line=dict(color=AMBER, dash="dash", width=2), annotation_text="activo")
    fig.add_vline(x=ctx.meta["umbral"], line=dict(color="#0F2A3D", dash="dot", width=1), annotation_text="óptimo F1",
                  annotation_position="bottom right")
    fig.update_layout(xaxis_title="Umbral sobre P(neumonía)", yaxis=dict(range=[0, 1.05], title="Valor"))
    mostrar(estilo(fig, alto=440, titulo="SENSIBILIDAD, ESPECIFICIDAD Y F1 SEGÚN EL UMBRAL"))

    alto_recall = b[b["recall_neumonia"] >= 0.98]
    if not alto_recall.empty:
        u = alto_recall.iloc[-1]
        st.caption(f"Para garantizar ≥ 98 % de sensibilidad en validación el umbral máximo es {u['umbral']:.2f}, "
                   f"con especificidad de {u['especificidad']:.1%}.")
