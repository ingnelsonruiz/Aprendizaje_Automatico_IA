"""Pestaña «La historia»: problema, pregunta orientadora y flujo metodológico."""

import streamlit as st

from config import AMBER, CYAN, GREEN, PINK, VIOLET
from ui import historia, seccion, tarjeta

PASOS = [
    ("1. Datos y SQL", "Inventario en DuckDB, EDA y deduplicación por hash", "20 % de la rúbrica"),
    ("2. Pipeline", "Partición por paciente, caché y aumento de datos", "20 % de la rúbrica"),
    ("3. Modelado", "CNN base frente a EfficientNet-B0 con transfer learning", "20 % de la rúbrica"),
    ("4. Evaluación", "Curvas, F1, AUC, matriz de confusión, umbral y CAM", "15 % de la rúbrica"),
    ("5. Despliegue", "Modelo ONNX servido en esta aplicación Streamlit", "15 % de la rúbrica"),
]


def render(ctx):
    seccion("EL PROBLEMA")
    historia(
        "La neumonía es una de las principales causas de muerte en menores de cinco años en el mundo. "
        "La radiografía de tórax es el examen de imagen más usado para confirmarla, pero su lectura "
        "requiere un radiólogo, y en zonas rurales o dispersas ese especialista no siempre está "
        "disponible a tiempo. <b>Un sistema que priorice las radiografías sospechosas</b> puede acortar "
        "el tiempo hasta el tratamiento.",
        color=CYAN,
        pregunta="¿Puede una red neuronal convolucional distinguir radiografías normales de radiografías "
                 "con neumonía, priorizando no dejar pasar casos positivos y mostrando en qué se basa?")

    seccion("CÓMO LO ABORDAMOS")
    pasos = '<span class="flecha">➜</span>'.join(
        f'<div class="step"><b>{t}</b><span>{d}</span><small>{r}</small></div>' for t, d, r in PASOS)
    st.markdown(f'<div class="flow">{pasos}</div>', unsafe_allow_html=True)

    seccion("DECISIONES QUE MARCAN LA DIFERENCIA")
    d = ctx.meta["dataset"]
    a, b, c, e = st.columns(4)
    a.markdown(tarjeta("SIN DUPLICADOS", f"El dataset original trae {d['duplicados']:,} archivos repetidos. "
                       "Se eliminaron por hash para que ninguna imagen esté a la vez en entrenamiento y prueba.",
                       PINK), unsafe_allow_html=True)
    b.markdown(tarjeta("POR PACIENTE", "Un mismo niño puede tener varias radiografías. La partición se hizo por "
                       "paciente: las métricas reflejan pacientes nuevos, no imágenes memorizadas.", VIOLET),
               unsafe_allow_html=True)
    c.markdown(tarjeta("SENSIBILIDAD PRIMERO", "Pesos de clase en la pérdida y umbral ajustado en validación: "
                       "un falso negativo es el error más costoso.", GREEN), unsafe_allow_html=True)
    e.markdown(tarjeta("EXPLICABLE", "Cada predicción trae un mapa de activación (CAM) que muestra qué zonas "
                       "pulmonares motivaron la decisión.", AMBER), unsafe_allow_html=True)
