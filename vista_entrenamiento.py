"""Pestaña «Entrenamiento»: curvas de aprendizaje por época."""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from ayuda import expansor_indicadores
from config import COLOR_MODELO
from ui import estilo, mostrar, panel, seccion, tabla

METRICAS = {
    "Pérdida": ("train_loss", "val_loss"),
    "Exactitud": ("train_acc", "val_acc"),
    "F1 macro (validación)": (None, "val_f1"),
    "ROC-AUC (validación)": (None, "val_auc"),
}


def render(ctx):
    h = pd.DataFrame(ctx.meta["historia"])
    fin_fase1 = ctx.meta["entrenamiento"]["epoca_fin_fase1"]
    seccion("CURVAS DE APRENDIZAJE")
    metrica = st.radio("Métrica", list(METRICAS), horizontal=True)
    col_ent, col_val = METRICAS[metrica]

    fig = go.Figure()
    for modelo, sub in h.groupby("modelo", sort=False):
        color = COLOR_MODELO.get(modelo, "#0F2A3D")
        if col_ent:
            fig.add_trace(go.Scatter(x=sub["epoca"], y=sub[col_ent], name=f"{modelo} · entrenamiento",
                                     line=dict(color=color, dash="dash", width=2)))
        fig.add_trace(go.Scatter(x=sub["epoca"], y=sub[col_val], name=f"{modelo} · validación",
                                 mode="lines+markers", line=dict(color=color, width=3)))
        mejor = sub.loc[sub["val_f1"].idxmax()]
        fig.add_trace(go.Scatter(x=[mejor["epoca"]], y=[mejor[col_val]], mode="markers", showlegend=False,
                                 marker=dict(symbol="star", size=16, color=color, line=dict(color="white", width=1)),
                                 hovertext=f"{modelo}: mejor época (F1 val {mejor['val_f1']:.4f})"))
    fig.add_vline(x=fin_fase1 + 0.5, line=dict(color="#9FB2C0", dash="dot"),
                  annotation_text="EffNet: fin fase 1", annotation_position="top")
    fig.update_layout(xaxis_title="Época", yaxis_title=metrica)
    mostrar(estilo(fig, alto=460))
    st.caption("★ = época seleccionada (mejor F1 macro en validación); sus pesos son los que se exportaron.")

    panel("<b>Cómo leer las curvas:</b> si la pérdida de entrenamiento sigue bajando mientras la de validación "
          "sube, el modelo memoriza en lugar de generalizar (sobreajuste); el <i>early stopping</i> detiene el "
          "entrenamiento cuando el F1 de validación deja de mejorar. En EfficientNet, el salto tras la línea "
          "punteada muestra el beneficio de descongelar la red en la fase 2.")
    expansor_indicadores(["Pérdida (loss)", "F1", "ROC-AUC"])

    with st.expander("Tabla completa por época"):
        tabla(h.round(4))
