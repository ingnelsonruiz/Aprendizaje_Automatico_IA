"""Pestaña «Universo 3D»: reducción de dimensionalidad con PCA."""

import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from config import CYAN, GREEN, PINK, SEMILLA, VIOLET
from ui import estilo, kpi, mostrar, panel, seccion

EJES = ["Componente 1", "Componente 2", "Componente 3"]


@st.cache_data(show_spinner=False)
def _proyectar(_X):
    pca = PCA(n_components=3, random_state=SEMILLA)
    Z = pca.fit_transform(StandardScaler().fit_transform(_X))
    return pd.DataFrame(Z, columns=EJES), pca.explained_variance_ratio_


def render(ctx):
    seccion("REDUCCIÓN DE DIMENSIONALIDAD · DE 30D A 3D")
    panel("Cada punto es una paciente. El análisis de componentes principales (PCA) comprime las 30 "
          "variables en 3 componentes: puedes <b>rotar, acercar y explorar</b> el espacio para ver cómo "
          "se separan los tumores benignos de los malignos.")

    df, varianza = _proyectar(ctx.X)
    df = df.copy()
    df["Diagnóstico"] = ctx.etiqueta.values

    fig = px.scatter_3d(df, x=EJES[0], y=EJES[1], z=EJES[2], color="Diagnóstico",
                        color_discrete_map={"Benigno": CYAN, "Maligno": PINK}, opacity=0.85)
    fig.update_traces(marker=dict(size=4, line=dict(width=0.4, color="rgba(255,255,255,0.4)")))
    ejes = dict(backgroundcolor="rgba(0,0,0,0)", gridcolor="rgba(0,212,255,0.15)")
    fig.update_layout(scene=dict(xaxis=ejes, yaxis=ejes, zaxis=ejes))
    mostrar(estilo(fig, alto=620))

    a, b, c = st.columns(3)
    a.markdown(kpi(f"{varianza[0]*100:.1f}%", "Varianza componente 1", CYAN), unsafe_allow_html=True)
    b.markdown(kpi(f"{varianza[1]*100:.1f}%", "Varianza componente 2", VIOLET), unsafe_allow_html=True)
    c.markdown(kpi(f"{varianza.sum()*100:.1f}%", "Varianza total en 3D", GREEN), unsafe_allow_html=True)
