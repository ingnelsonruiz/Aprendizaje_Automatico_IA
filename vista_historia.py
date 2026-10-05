"""Pestaña «La historia»: qué se busca predecir y qué variables cuentan esa historia."""

import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from config import AMBER, CYAN, GREEN, NARANJA, PINK, POR_QUE, VIOLET
from datos import base_de
from ui import estilo, historia, mostrar, panel, seccion, tarjeta

PALETA_TOP = [PINK, CYAN, VIOLET, GREEN, AMBER, NARANJA]


def _relato():
    seccion("UNA PACIENTE, UNA DUDA, UNA DECISIÓN")
    historia(
        "Una mujer encuentra un bulto en su mama. Va al médico y el médico le ordena una "
        "<b>punción con aguja fina</b>: una aguja muy delgada entra en la masa y extrae unas pocas "
        "células. Esas células van al laboratorio, se ponen bajo el microscopio y se fotografían. "
        "Un patólogo mira la imagen y debe responder una sola pregunta, la que lo cambia todo:",
        CYAN, "¿Este tumor es benigno o es cáncer?")
    historia(
        "Responder esa pregunta a ojo depende de la experiencia de quien mira. Por eso, en 1993, un "
        "equipo de la Universidad de Wisconsin hizo algo distinto: en vez de solo mirar, <b>midieron</b>. "
        "Marcaron el contorno de cada núcleo celular en la imagen y calcularon su tamaño, su forma y qué "
        "tan irregular era su borde. Convirtieron una impresión visual en <b>30 números</b>. Luego "
        "juntaron 569 casos en los que el diagnóstico ya estaba confirmado por biopsia.",
        VIOLET)
    historia(
        "Ese es exactamente el problema que resuelve esta aplicación. El modelo recibe los 30 números de "
        "un núcleo celular y devuelve una probabilidad de que el tumor sea maligno. No reemplaza al "
        "médico: le da una <b>segunda opinión cuantitativa</b>, construida a partir de cientos de casos "
        "previos cuyo desenlace ya se conoce.",
        GREEN, "Variable objetivo: Benigno (0) o Maligno (1)")

    st.write("")
    a, b, c = st.columns(3)
    a.markdown(tarjeta("LO QUE ENTRA", "30 medidas del núcleo celular: tamaño, forma e irregularidad del "
                                       "borde. Todas numéricas y sin valores faltantes.", CYAN),
               unsafe_allow_html=True)
    b.markdown(tarjeta("LO QUE SALE", "Una probabilidad entre 0 % y 100 % de que el tumor sea maligno, "
                                      "más la clasificación final.", VIOLET), unsafe_allow_html=True)
    c.markdown(tarjeta("LO QUE ESTÁ EN JUEGO", "Un falso negativo deja a una paciente con cáncer sin "
                                               "tratamiento. Por eso la sensibilidad pesa más que la "
                                               "exactitud.", PINK), unsafe_allow_html=True)


def _variables_clave(ctx):
    seccion("LAS VARIABLES QUE CUENTAN LA HISTORIA")
    historia(
        "No todas las medidas pesan igual. El Random Forest puede decirnos <b>cuánto usa cada variable</b> "
        "para decidir. Abajo están las 6 más importantes, con lo que significan en el microscopio y cómo "
        "cambian entre un tumor benigno y uno maligno. Hay un patrón claro: las que más pesan hablan de "
        "<b>tamaño</b> y de <b>bordes irregulares</b>.",
        AMBER)

    import pandas as pd
    rf = ctx.resultados["Random Forest"]["modelo"]
    top = pd.Series(rf.feature_importances_, index=ctx.columnas).sort_values(ascending=False).head(6)

    for i in range(0, len(top), 2):
        fila = st.columns(2)
        for j, col in enumerate(fila):
            if i + j >= len(top):
                break
            var, peso = top.index[i + j], top.iloc[i + j]
            m_ben = ctx.X.loc[ctx.y == 0, var].mean()
            m_mal = ctx.X.loc[ctx.y == 1, var].mean()
            delta = (m_mal / m_ben - 1) * 100 if m_ben else 0
            texto = (f"{POR_QUE.get(base_de(var), '')}<br><br>"
                     f"<b>Importancia:</b> {peso*100:.1f} % · <b>Benigno:</b> {m_ben:.3f} · "
                     f"<b>Maligno:</b> {m_mal:.3f} (<b>{delta:+.0f} %</b>)")
            col.markdown(tarjeta(f"#{i+j+1} · {var.upper()}", texto, PALETA_TOP[i + j]),
                         unsafe_allow_html=True)
        st.write("")
    return list(top.index)


def _comparacion(ctx, opciones):
    seccion("CÓMO SE VE ESA DIFERENCIA")
    var = st.selectbox("Elige una variable para ver su historia en los datos", opciones,
                       index=0, key="story_var")
    izq, der = st.columns([3, 2])
    with izq:
        df = ctx.X[[var]].copy()
        df["Diagnóstico"] = ctx.etiqueta.values
        fig = px.histogram(df, x=var, color="Diagnóstico", barmode="overlay", nbins=45, opacity=0.72,
                           color_discrete_map={"Benigno": CYAN, "Maligno": PINK})
        fig.update_layout(yaxis_title="Cantidad de pacientes")
        mostrar(estilo(fig, alto=400, titulo=f"DISTRIBUCIÓN · {var.upper()}"))
    with der:
        fig = go.Figure()
        for nombre, clase, color in [("Benigno", 0, CYAN), ("Maligno", 1, PINK)]:
            fig.add_trace(go.Box(y=ctx.X.loc[ctx.y == clase, var], name=nombre, marker_color=color,
                                 boxmean=True, line=dict(width=2)))
        fig.update_layout(showlegend=False, yaxis_title=var)
        mostrar(estilo(fig, alto=400, titulo="COMPARACIÓN DIRECTA"))
    panel("Donde las dos distribuciones casi no se tocan, la variable separa bien los dos grupos por sí "
          "sola. Donde se superponen, el modelo necesita combinarla con otras para decidir. Esa "
          "combinación es, en el fondo, todo lo que hace el aprendizaje automático.")


def render(ctx):
    _relato()
    opciones = _variables_clave(ctx)
    _comparacion(ctx, opciones)
