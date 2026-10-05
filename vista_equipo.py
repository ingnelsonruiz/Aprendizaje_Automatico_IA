"""Pestaña «Equipo»: integrantes, alcance y stack del proyecto."""

import streamlit as st

from config import (ASIGNATURA, CODIGOS, CYAN, EQUIPO, GREEN, GRUPO,
                           PINK, PROGRAMA, PROYECTO, SUBTITULO, VIOLET)
from ui import kpi, miembro, panel, seccion, tarjeta


def render(ctx):
    seccion(PROYECTO.upper())
    panel(f"<b>{SUBTITULO}</b><br>"
          f"Asignatura: {ASIGNATURA} · {' · '.join(CODIGOS).title()}<br>{PROGRAMA}")

    seccion(f"{GRUPO.upper()} · INTEGRANTES")
    cols = st.columns(2)
    for i, (nombre, color) in enumerate(EQUIPO):
        cols[i % 2].markdown(miembro(nombre, color, PROGRAMA), unsafe_allow_html=True)

    seccion("ALCANCE DEL PROYECTO")
    a, b, c = st.columns(3)
    a.markdown(tarjeta("MODELADO", "Tres algoritmos clásicos entrenados y comparados: regresión "
                                   "logística, Random Forest y SVM con kernel RBF.", CYAN),
               unsafe_allow_html=True)
    b.markdown(tarjeta("EVALUACIÓN", "Exactitud, precisión, sensibilidad, F1, curvas ROC, matrices de "
                                     "confusión y análisis del umbral de decisión.", VIOLET),
               unsafe_allow_html=True)
    c.markdown(tarjeta("DESPLIEGUE", "Aplicación web interactiva en Streamlit Community Cloud, con "
                                     "laboratorio didáctico y simulador de predicción.", GREEN),
               unsafe_allow_html=True)

    seccion("STACK TECNOLÓGICO")
    s = st.columns(4)
    s[0].markdown(kpi("Python", "Lenguaje", CYAN), unsafe_allow_html=True)
    s[1].markdown(kpi("Scikit-learn", "Modelado", VIOLET), unsafe_allow_html=True)
    s[2].markdown(kpi("Plotly", "Visualización", GREEN), unsafe_allow_html=True)
    s[3].markdown(kpi("Streamlit", "Despliegue", PINK), unsafe_allow_html=True)
