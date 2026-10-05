"""Pestaña «Modelos»: comparación de los tres algoritmos entrenados."""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import confusion_matrix

from config import COLOR_MODELO, CYAN, NOMBRES_METRICAS, VIOLET
from ui import estilo, mostrar, seccion, tabla

METRICAS = list(NOMBRES_METRICAS.keys())
ETIQUETAS = list(NOMBRES_METRICAS.values())


def _tabla_metricas(ctx):
    return pd.DataFrame({n: {k: r[k] for k in METRICAS} for n, r in ctx.resultados.items()}).T


def _barras_y_radar(df):
    izq, der = st.columns(2)
    with izq:
        fig = go.Figure()
        for nombre in df.index:
            fig.add_trace(go.Bar(name=nombre, x=ETIQUETAS, y=df.loc[nombre].values,
                                 marker_color=COLOR_MODELO[nombre],
                                 text=[f"{v:.3f}" for v in df.loc[nombre].values],
                                 textposition="outside"))
        fig.update_layout(barmode="group", yaxis=dict(range=[0.85, 1.03]))
        mostrar(estilo(fig, alto=420, titulo="MÉTRICAS POR MODELO"))
    with der:
        fig = go.Figure()
        for nombre in df.index:
            vals = df.loc[nombre].values.tolist()
            fig.add_trace(go.Scatterpolar(r=vals + [vals[0]], theta=ETIQUETAS + [ETIQUETAS[0]],
                                          name=nombre, fill="toself", opacity=0.45,
                                          line=dict(color=COLOR_MODELO[nombre], width=2)))
        fig.update_layout(polar=dict(bgcolor="rgba(13,27,42,0.35)",
                                     radialaxis=dict(range=[0.88, 1.0], gridcolor="rgba(0,212,255,0.2)"),
                                     angularaxis=dict(gridcolor="rgba(0,212,255,0.2)")))
        mostrar(estilo(fig, alto=420, titulo="RADAR DE DESEMPEÑO"))


def _roc_y_confusion(ctx):
    izq, der = st.columns(2)
    with izq:
        fig = go.Figure()
        for nombre, r in ctx.resultados.items():
            fig.add_trace(go.Scatter(x=r["fpr"], y=r["tpr"], mode="lines",
                                     name=f"{nombre} · AUC {r['auc']:.3f}",
                                     line=dict(color=COLOR_MODELO[nombre], width=3)))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Azar",
                                 line=dict(color="#667788", dash="dash")))
        fig.update_layout(xaxis_title="Tasa de falsos positivos", yaxis_title="Sensibilidad",
                          legend=dict(x=0.45, y=0.08))
        mostrar(estilo(fig, alto=420, titulo="CURVAS ROC"))
    with der:
        cm = confusion_matrix(ctx.y_prueba, ctx.activo["pred"])
        fig = px.imshow(cm, text_auto=True, x=["Benigno", "Maligno"], y=["Benigno", "Maligno"],
                        color_continuous_scale=[[0, "#0D1B2A"], [1, COLOR_MODELO[ctx.modelo_sel]]])
        fig.update_traces(textfont=dict(size=30))
        fig.update_layout(xaxis_title="Predicción del modelo", yaxis_title="Diagnóstico real",
                          coloraxis_showscale=False)
        mostrar(estilo(fig, alto=420, titulo=f"MATRIZ DE CONFUSIÓN · {ctx.modelo_sel.upper()}"))


def _importancias(ctx):
    seccion("IMPORTANCIA DE LAS VARIABLES · RANDOM FOREST")
    rf = ctx.resultados["Random Forest"]["modelo"]
    imp = pd.Series(rf.feature_importances_, index=ctx.columnas).sort_values(ascending=True).tail(15)
    fig = go.Figure(go.Bar(x=imp.values, y=imp.index, orientation="h",
                           marker=dict(color=imp.values, colorscale=[[0, VIOLET], [1, CYAN]]),
                           text=[f"{v:.3f}" for v in imp.values], textposition="outside"))
    fig.update_layout(xaxis_title="Importancia relativa")
    mostrar(estilo(fig, alto=520))


def render(ctx):
    seccion("COMPARACIÓN DE MODELOS")
    df = _tabla_metricas(ctx)
    _barras_y_radar(df)
    _roc_y_confusion(ctx)
    _importancias(ctx)
    seccion("TABLA DE RESULTADOS")
    tabla(df.rename(columns=NOMBRES_METRICAS).round(4).reset_index().rename(columns={"index": "Modelo"}))
