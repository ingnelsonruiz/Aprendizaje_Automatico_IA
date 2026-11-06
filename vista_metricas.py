"""Pestaña «Métricas»: evaluación en el conjunto de prueba."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from ayuda import ayuda, expansor_indicadores
from config import AMBER, COLOR_MODELO, CYAN, GREEN, NARANJA, PINK, VIOLET
from ui import estilo, kpi, mostrar, panel, seccion, tabla

COLUMNAS = {"accuracy": "Exactitud", "precision": "Precisión", "recall_sensibilidad": "Sensibilidad",
            "especificidad": "Especificidad", "f1_neumonia": "F1 neumonía", "f1_macro": "F1 macro",
            "roc_auc": "ROC-AUC", "vp": "VP", "fp": "FP", "vn": "VN", "fn": "FN"}


def render(ctx):
    r = ctx.principal
    seccion(f"EFFICIENTNET-B0 · CONJUNTO DE PRUEBA · UMBRAL {ctx.meta['umbral']:.2f}")
    c = st.columns(6)
    for col, (v, nombre, color, clave) in zip(c, [
        (r["accuracy"], "Exactitud", CYAN, "Exactitud (accuracy)"),
        (r["recall_sensibilidad"], "Sensibilidad", PINK, "Sensibilidad (recall)"),
        (r["especificidad"], "Especificidad", GREEN, "Especificidad"),
        (r["precision"], "Precisión", VIOLET, "Precisión (VPP)"),
        (r["f1_macro"], "F1 macro", AMBER, "F1"),
        (r["roc_auc"], "ROC-AUC", CYAN, "ROC-AUC"),
    ]):
        col.markdown(kpi(f"{v:.3f}", nombre, color, ayuda(clave)), unsafe_allow_html=True)
    expansor_indicadores(["Exactitud (accuracy)", "Sensibilidad (recall)", "Especificidad",
                          "Precisión (VPP)", "F1", "ROC-AUC", "VP, FP, VN, FN"])

    izq, der = st.columns(2)
    with izq:
        seccion("MATRIZ DE CONFUSIÓN")
        cm = np.array(ctx.meta["matriz_confusion"])
        pct = cm / cm.sum(axis=1, keepdims=True)
        texto = [[f"<b>{cm[i, j]}</b><br>{pct[i, j]:.1%}<br>{e}" for j, e in enumerate(fila)]
                 for i, fila in enumerate([["VN", "FP"], ["FN", "VP"]])]
        fig = go.Figure(go.Heatmap(z=pct, x=["Pred. NORMAL", "Pred. NEUMONÍA"], y=["Real NORMAL", "Real NEUMONÍA"],
                                   text=texto, texttemplate="%{text}", colorscale=[[0, "#F4F8FB"], [1, "#8DB4D3"]],
                                   showscale=False, textfont=dict(size=16, color="#0F2A3D")))
        fig.update_yaxes(autorange="reversed")
        mostrar(estilo(fig, alto=400))
        st.caption(f"{r['fn']} neumonías no detectadas y {r['fp']} falsas alarmas sobre "
                   f"{r['vp'] + r['fn'] + r['vn'] + r['fp']} radiografías de prueba.")
    with der:
        seccion("CURVA ROC")
        fig = go.Figure()
        for nombre, roc in ctx.meta["roc"].items():
            auc = ctx.principal["roc_auc"] if nombre == "EffNet-B0" else ctx.base["roc_auc"]
            fig.add_trace(go.Scatter(x=roc["fpr"], y=roc["tpr"], name=f"{nombre} (AUC {auc:.3f})",
                                     line=dict(color=COLOR_MODELO.get(nombre), width=3)))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], name="Azar", line=dict(color="#9FB2C0", dash="dash")))
        fig.update_layout(xaxis_title="1 − especificidad", yaxis_title="Sensibilidad",
                          legend=dict(x=0.98, y=0.04, xanchor="right", yanchor="bottom"))
        mostrar(estilo(fig, alto=400))

    seccion("COMPARACIÓN DE MODELOS Y UMBRALES")
    t = pd.DataFrame(ctx.meta["metricas_prueba"]).T.rename(columns=COLUMNAS)
    t.index.name = "Modelo"
    tabla(t.reset_index().round(4))

    seccion("DESEMPEÑO POR SUBTIPO")
    s = pd.DataFrame(ctx.meta["metricas_por_subtipo"])
    fig = go.Figure(go.Bar(x=s["subtipo"].str.capitalize(), y=s["tasa_acierto"],
                           marker_color=[{"normal": GREEN, "bacteria": PINK, "virus": NARANJA}.get(t, AMBER) for t in s["subtipo"]],
                           text=[f"{v:.1%}<br>n = {n}" for v, n in zip(s["tasa_acierto"], s["imagenes"])],
                           textposition="outside"))
    fig.update_layout(yaxis=dict(range=[0, 1.12], title="Tasa de acierto"))
    mostrar(estilo(fig, alto=360))
    panel("La neumonía viral suele producir patrones intersticiales más sutiles que la consolidación lobar típica "
          "de la bacteriana; si su tasa de acierto es menor, es coherente con la dificultad clínica de "
          "distinguirla en la radiografía.")
