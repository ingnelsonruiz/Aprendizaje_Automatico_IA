"""Pestaña «El dataset»: de dónde salen los datos y qué significa cada variable."""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from config import AMBER, CARACTERISTICAS, CITA, CYAN, GREEN, PINK, VIOLET
from datos import nombre_es
from ui import estilo, kpi, mostrar, panel, seccion, tabla, tarjeta

FLUJO = [
    ("1 · PUNCIÓN", "Se extrae una muestra de la masa con aguja fina (FNA)"),
    ("2 · IMAGEN", "La muestra se digitaliza con un microscopio y una cámara"),
    ("3 · SEGMENTACIÓN", "Se delimita el contorno de cada núcleo celular"),
    ("4 · MEDICIÓN", "Se calculan 10 características de cada núcleo"),
    ("5 · RESUMEN", "Cada una se resume en 3 estadísticos → 30 variables"),
    ("6 · DIAGNÓSTICO", "Se registra el resultado confirmado: Benigno o Maligno"),
]


def _resumen(ctx):
    seccion("¿QUÉ ES ESTE DATASET?")
    panel("<b>Breast Cancer Wisconsin (Diagnostic)</b> es uno de los conjuntos de datos más usados para "
          "aprender clasificación en Machine Learning. Cada fila es una <b>paciente</b> a la que se le hizo "
          "una <b>punción con aguja fina (FNA)</b> en una masa de la mama. De esa muestra se obtuvo una "
          "imagen digitalizada y, con ella, se midieron las características de los <b>núcleos de las "
          "células</b>. Cada caso tiene además un <b>diagnóstico confirmado</b>: <b>Benigno</b> (no es "
          "cáncer) o <b>Maligno</b> (es cáncer). El objetivo del modelo es aprender a predecir ese "
          "diagnóstico usando solo las medidas del núcleo.")
    st.write("")
    k = st.columns(5)
    k[0].markdown(kpi(f"{len(ctx.X)}", "Pacientes", CYAN), unsafe_allow_html=True)
    k[1].markdown(kpi(f"{ctx.X.shape[1]}", "Variables", VIOLET), unsafe_allow_html=True)
    k[2].markdown(kpi(f"{ctx.n_benignos}", "Benignos", GREEN), unsafe_allow_html=True)
    k[3].markdown(kpi(f"{ctx.n_malignos}", "Malignos", PINK), unsafe_allow_html=True)
    k[4].markdown(kpi(f"{int(ctx.X.isnull().sum().sum())}", "Valores faltantes", AMBER),
                  unsafe_allow_html=True)


def _origen():
    seccion("¿CÓMO SE OBTUVIERON LOS DATOS?")
    pasos = "".join(f'<div class="step"><b>{t}</b><span>{d}</span></div>' for t, d in FLUJO)
    st.markdown(f'<div class="flow">{pasos}</div>', unsafe_allow_html=True)

    seccion("DE 10 CARACTERÍSTICAS A 30 VARIABLES")
    a, b, c = st.columns(3)
    a.markdown(tarjeta("MEDIA", "Valor promedio de la característica entre todos los núcleos de la imagen. "
                                "Ejemplo: <i>Radio (media)</i>.", CYAN, "×10"), unsafe_allow_html=True)
    b.markdown(tarjeta("ERROR ESTÁNDAR", "Qué tanto varía la característica de un núcleo a otro dentro de "
                                         "la misma imagen. Ejemplo: <i>Radio (error est.)</i>.", VIOLET,
                       "×10"), unsafe_allow_html=True)
    c.markdown(tarjeta("PEOR VALOR", "Promedio de los tres valores más extremos de la imagen: captura al "
                                     "núcleo más «anormal». Ejemplo: <i>Radio (peor)</i>.", PINK, "×10"),
               unsafe_allow_html=True)
    st.caption("10 características × 3 estadísticos = 30 variables numéricas. El archivo original trae "
               "además un identificador de paciente y el diagnóstico, que no se usan como predictoras.")


def _diccionario(ctx):
    seccion("DICCIONARIO DE VARIABLES")
    filas = []
    for orig, es, mide, aspecto, _ in CARACTERISTICAS:
        col = nombre_es(orig + "1")
        m_ben = ctx.X.loc[ctx.y == 0, col].mean()
        m_mal = ctx.X.loc[ctx.y == 1, col].mean()
        filas.append({"Característica": es, "Nombre original (UCI)": orig, "Qué mide": mide,
                      "Describe": aspecto, "Media en benignos": round(m_ben, 4),
                      "Media en malignos": round(m_mal, 4),
                      "Variación": f"{(m_mal / m_ben - 1) * 100:+.0f}%"})
    tabla(pd.DataFrame(filas))
    st.caption("Las medias se calculan en vivo con los datos reales (variables «media»).")

    seccion("¿QUÉ CARACTERÍSTICAS SEPARAN MEJOR LOS DOS GRUPOS?")
    dif = []
    for orig, es, _, _, _ in CARACTERISTICAS:
        col = nombre_es(orig + "1")
        dif.append((es, (ctx.X.loc[ctx.y == 1, col].mean() / ctx.X.loc[ctx.y == 0, col].mean() - 1) * 100))
    dif = pd.DataFrame(dif, columns=["Característica", "Diferencia"]).sort_values("Diferencia")
    fig = go.Figure(go.Bar(x=dif["Diferencia"], y=dif["Característica"], orientation="h",
                           marker=dict(color=dif["Diferencia"], colorscale=[[0, CYAN], [1, PINK]]),
                           text=[f"{v:+.0f}%" for v in dif["Diferencia"]], textposition="outside"))
    fig.update_layout(xaxis_title="Diferencia porcentual de la media: maligno vs. benigno")
    mostrar(estilo(fig, alto=430, titulo="MALIGNO VS. BENIGNO · DIFERENCIA DE MEDIAS"))


def _ficha():
    seccion("FICHA TÉCNICA Y CITA")
    a, b = st.columns(2)
    a.markdown(tarjeta("FICHA", "Repositorio: UCI Machine Learning Repository (ID 17)<br>"
                                "Autores: W. Wolberg, O. Mangasarian, N. Street y W. Street<br>"
                                "Donado: 31/10/1995 · Área: Salud y medicina<br>"
                                "Tarea: clasificación binaria · Licencia: CC BY 4.0", CYAN),
               unsafe_allow_html=True)
    b.markdown(tarjeta("ARTÍCULO DE ORIGEN", "«Nuclear feature extraction for breast tumor diagnosis» — "
                                             "Street, Wolberg y Mangasarian (1993), Electronic Imaging.<br>"
                                             "DOI del dataset: 10.24432/C5DW2B", VIOLET),
               unsafe_allow_html=True)
    st.code(CITA, language=None)


def render(ctx):
    _resumen(ctx)
    _origen()
    _diccionario(ctx)
    _ficha()
