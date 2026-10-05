"""Diagnóstico Inteligente — Breast Cancer Wisconsin (Diagnostic).

Propuesta del Proyecto ACA 1 · Grupo 3
Aprendizaje Automático · 54ES2 · Segundo bloque · 26ES4

Este archivo solo orquesta: configura la página, construye el contexto de datos
y entrega cada pestaña a su módulo.

Módulos de apoyo: config, estilos, ui, datos, entrenamiento.
Una vista por pestaña: vista_*.py, cada una con una función render(ctx).

Para agregar una pestaña nueva:
  1. Crea `vista_mi_vista.py` con una función `render(ctx)`.
  2. Impórtala abajo y añade una entrada a la lista `PESTANAS`.
"""

import streamlit as st

st.set_page_config(
    page_title="Diagnóstico Inteligente · Cáncer de Mama",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

import estilos                                          # noqa: E402
from config import (AMBER, ASIGNATURA, CITA, CODIGOS, CYAN, EQUIPO,    # noqa: E402
                           GREEN, GRUPO, PINK, PROGRAMA, PROYECTO,
                           SUBTITULO, VIOLET)
from datos import construir_contexto                          # noqa: E402
from ui import kpi                                            # noqa: E402
import vista_dataset, vista_equipo, vista_exploracion            # noqa: E402
import vista_historia, vista_laboratorio, vista_metricas         # noqa: E402
import vista_modelos, vista_prediccion, vista_umbral             # noqa: E402
import vista_universo3d                                          # noqa: E402

# Orden de las pestañas: (etiqueta, módulo). Reordenar aquí reordena la app.
PESTANAS = [
    ("📖 La historia", vista_historia),
    ("📘 El dataset", vista_dataset),
    ("🧪 Laboratorio", vista_laboratorio),
    ("🌌 Universo 3D", vista_universo3d),
    ("📊 Exploración", vista_exploracion),
    ("🤖 Modelos", vista_modelos),
    ("🎯 Umbral clínico", vista_umbral),
    ("🔮 Predicción", vista_prediccion),
    ("📏 Métricas", vista_metricas),
    ("👥 Equipo", vista_equipo),
]


def barra_lateral(opciones_modelo):
    """Dibuja la barra lateral y devuelve el modelo seleccionado."""
    with st.sidebar:
        st.markdown("### 🧬 CENTRO DE CONTROL")
        seleccionado = st.selectbox("Modelo activo", opciones_modelo, index=1)
        st.markdown("---")
        marcador = st.empty()
        st.markdown("---")
        st.markdown(f"**{PROYECTO}**  \n{SUBTITULO}")
        st.markdown(f"**Asignatura**  \n{ASIGNATURA}  \n{' · '.join(CODIGOS).title()}")
        st.markdown(f"**{GRUPO}**  \n" + "  \n".join(n for n, _ in EQUIPO))
        st.markdown("---")
        st.caption("Dataset: UCI ML Repository · ID 17  \nDOI: 10.24432/C5DW2B")
    return seleccionado, marcador


def encabezado(ctx):
    """Dibuja el bloque superior con el título y los indicadores."""
    chips = "".join(f'<span class="chip">{c}</span>' for c in CODIGOS)
    st.markdown(f"""
<div class="hero">
    <div class="hero-tag">{PROYECTO.upper()} · {GRUPO.upper()}</div>
    <div class="chips"><span class="chip">{ASIGNATURA.upper()}</span>{chips}</div>
    <h1>DIAGNÓSTICO INTELIGENTE</h1>
    <p>{SUBTITULO}</p>
</div>
""", unsafe_allow_html=True)

    r = ctx.activo
    cols = st.columns(5)
    cols[0].markdown(kpi(f"{len(ctx.X)}", "Pacientes", CYAN), unsafe_allow_html=True)
    cols[1].markdown(kpi(f"{ctx.X.shape[1]}", "Variables", VIOLET), unsafe_allow_html=True)
    cols[2].markdown(kpi(f"{r['Accuracy']*100:.1f}%", "Exactitud", GREEN), unsafe_allow_html=True)
    cols[3].markdown(kpi(f"{r['Recall']*100:.1f}%", "Sensibilidad", PINK), unsafe_allow_html=True)
    cols[4].markdown(kpi(f"{r['auc']:.3f}", "Área bajo ROC", AMBER), unsafe_allow_html=True)


def main():
    estilos.aplicar()

    with st.spinner("⚡ Conectando con UCI y entrenando los modelos..."):
        ctx = construir_contexto()

    ctx.modelo_sel, marcador = barra_lateral(list(ctx.resultados.keys()))
    marcador.markdown(f"**Entrenamiento:** {ctx.n_train} casos  \n"
                      f"**Prueba:** {ctx.n_test} casos  \n"
                      f"**Variables:** {ctx.X.shape[1]}")

    encabezado(ctx)
    st.write("")

    pestanas = st.tabs([etiqueta for etiqueta, _ in PESTANAS])
    for contenedor, (_, modulo) in zip(pestanas, PESTANAS):
        with contenedor:
            modulo.render(ctx)

    st.markdown(f'<div class="aviso">Diagnóstico Inteligente · {PROYECTO} · {GRUPO} · '
                f'{ASIGNATURA} {CODIGOS[0]} · {CODIGOS[-1]}</div>', unsafe_allow_html=True)


main()
