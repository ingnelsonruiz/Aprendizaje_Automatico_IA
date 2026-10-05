"""Pestaña «Exploración»: distribuciones, correlaciones y vista previa."""

import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from config import CYAN, PINK
from ui import estilo, mostrar, seccion, tabla

MAPA_COLOR = {"Benigno": CYAN, "Maligno": PINK}


def render(ctx):
    izq, der = st.columns(2)
    with izq:
        seccion("DISTRIBUCIÓN DE CLASES")
        cuenta = ctx.etiqueta.value_counts()
        fig = go.Figure(go.Pie(labels=cuenta.index, values=cuenta.values, hole=0.62,
                               marker=dict(colors=[CYAN, PINK], line=dict(color="#05070f", width=3)),
                               textinfo="label+percent", textfont=dict(size=15)))
        fig.add_annotation(text=f"<b>{len(ctx.X)}</b><br>casos", showarrow=False,
                           font=dict(size=22, color="#fff"))
        mostrar(estilo(fig, alto=380))

    with der:
        seccion("DISTRIBUCIÓN POR VARIABLE")
        var = st.selectbox("Variable", list(ctx.X.columns), index=0, key="exp_var")
        df = ctx.X[[var]].copy()
        df["Diagnóstico"] = ctx.etiqueta.values
        fig = px.histogram(df, x=var, color="Diagnóstico", barmode="overlay", nbins=40,
                           opacity=0.7, color_discrete_map=MAPA_COLOR)
        fig.update_layout(yaxis_title="Cantidad de casos")
        mostrar(estilo(fig, alto=330))

    seccion("MAPA DE CORRELACIÓN · VARIABLES «MEDIA»")
    fig = px.imshow(ctx.X[ctx.columnas_media()].corr(), text_auto=".2f", aspect="auto",
                    color_continuous_scale=[[0, PINK], [0.5, "#0D1B2A"], [1, CYAN]], zmin=-1, zmax=1)
    mostrar(estilo(fig, alto=600))
    st.caption("Valores cercanos a 1 indican variables que crecen juntas (por ejemplo, radio, "
               "perímetro y área).")

    seccion("VISTA PREVIA DE LOS DATOS")
    prev = ctx.X.copy()
    prev.insert(0, "Diagnóstico", ctx.etiqueta.values)
    tabla(prev.head(15))
