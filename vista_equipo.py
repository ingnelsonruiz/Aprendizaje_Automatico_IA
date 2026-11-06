"""Pestaña «Equipo»: integrantes, alcance y stack del proyecto."""

import streamlit as st

from config import (ASIGNATURA, CODIGOS, CYAN, DOCENTE, EQUIPO, GREEN, GRUPO, INSTITUCION,
                    PINK, PROGRAMA, PROYECTO, SUBTITULO, URL_APP, URL_REPO, URL_VIDEO, VIOLET)
from ui import kpi, miembro, panel, seccion, tarjeta


def render(ctx):
    seccion(PROYECTO.upper())
    panel(f"<b>{SUBTITULO}</b><br>{INSTITUCION} · {PROGRAMA}<br>"
          f"Asignatura: {ASIGNATURA}, {CODIGOS[0]}, {CODIGOS[1].lower()}, {CODIGOS[2]}<br>Docente: {DOCENTE}")

    seccion(f"{GRUPO.upper()} · INTEGRANTES")
    cols = st.columns(2)
    for i, (nombre, color) in enumerate(EQUIPO):
        cols[i % 2].markdown(miembro(nombre, color, PROGRAMA), unsafe_allow_html=True)

    seccion("ALCANCE")
    a, b, c = st.columns(3)
    a.markdown(tarjeta("MODELADO", "CNN propia como línea base y EfficientNet-B0 preentrenada con fine-tuning en "
                       "dos fases, entrenadas en GPU con precisión mixta.", CYAN), unsafe_allow_html=True)
    b.markdown(tarjeta("EVALUACIÓN", "Curvas de aprendizaje, sensibilidad, especificidad, F1, ROC-AUC, matriz de "
                       "confusión, ajuste de umbral y mapas de activación.", VIOLET), unsafe_allow_html=True)
    c.markdown(tarjeta("DESPLIEGUE", "Modelo exportado a ONNX y servido en CPU en Streamlit Community Cloud, con "
                       "consultas SQL sobre DuckDB.", GREEN), unsafe_allow_html=True)

    seccion("STACK TECNOLÓGICO")
    s = st.columns(5)
    for col, (v, l, color) in zip(s, [("PyTorch", "Entrenamiento", CYAN), ("ONNX Runtime", "Inferencia", VIOLET),
                                      ("DuckDB", "Gestión SQL", GREEN), ("Plotly", "Visualización", PINK),
                                      ("Streamlit", "Despliegue", CYAN)]):
        col.markdown(kpi(v, l, color), unsafe_allow_html=True)

    enlaces = [(t, u) for t, u in [("Aplicación", URL_APP), ("Video de sustentación", URL_VIDEO),
                                   ("Repositorio", URL_REPO)] if u]
    if enlaces:
        seccion("ENLACES")
        st.markdown("  \n".join(f"- [{t}]({u})" for t, u in enlaces))
