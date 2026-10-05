"""Pestaña «Métricas»: los cuatro resultados posibles y las cuatro métricas."""

import streamlit as st
from sklearn.metrics import confusion_matrix

from config import AMBER, CYAN, GREEN, PINK, VIOLET
from ui import panel, seccion, tarjeta


def _resultados(ctx):
    seccion(f"LOS 4 RESULTADOS POSIBLES · {ctx.modelo_sel.upper()}")
    vn, fp, fn, vp = confusion_matrix(ctx.y_prueba, ctx.activo["pred"]).ravel()
    a, b, c, d = st.columns(4)
    a.markdown(tarjeta("VERDADERO POSITIVO", "El modelo dijo <b>maligno</b> y realmente lo era. "
                                             "Acierto clave.", GREEN, int(vp)), unsafe_allow_html=True)
    b.markdown(tarjeta("VERDADERO NEGATIVO", "El modelo dijo <b>benigno</b> y realmente lo era. Acierto.",
                       CYAN, int(vn)), unsafe_allow_html=True)
    c.markdown(tarjeta("FALSO POSITIVO", "El modelo dijo <b>maligno</b> pero era benigno. Falsa alarma: "
                                         "genera estrés y exámenes extra.", AMBER, int(fp)),
               unsafe_allow_html=True)
    d.markdown(tarjeta("FALSO NEGATIVO", "El modelo dijo <b>benigno</b> pero era maligno. El error más "
                                         "grave: la paciente quedaría sin tratamiento.", PINK, int(fn)),
               unsafe_allow_html=True)
    return int(vn), int(fp), int(fn), int(vp)


def _formulas(ctx, vn, fp, fn, vp):
    seccion("LAS 4 MÉTRICAS, CON TUS PROPIOS RESULTADOS")
    r = ctx.activo
    a, b = st.columns(2)
    a.markdown(tarjeta(f"EXACTITUD · {r['Accuracy']*100:.2f}%",
                       f"¿Qué porcentaje acertó en total?<br><b>(VP + VN) / Total</b> = "
                       f"({vp} + {vn}) / {vp+vn+fp+fn}<br>Engaña si las clases están desbalanceadas.",
                       CYAN), unsafe_allow_html=True)
    b.markdown(tarjeta(f"PRECISIÓN · {r['Precision']*100:.2f}%",
                       f"De los que dijo «maligno», ¿cuántos lo eran?<br><b>VP / (VP + FP)</b> = "
                       f"{vp} / ({vp} + {fp})<br>Mide cuántas falsas alarmas genera.", VIOLET),
               unsafe_allow_html=True)
    st.write("")
    c, d = st.columns(2)
    c.markdown(tarjeta(f"SENSIBILIDAD (RECALL) · {r['Recall']*100:.2f}%",
                       f"De los malignos reales, ¿cuántos detectó?<br><b>VP / (VP + FN)</b> = "
                       f"{vp} / ({vp} + {fn})<br>La métrica prioritaria en salud.", GREEN),
               unsafe_allow_html=True)
    d.markdown(tarjeta(f"F1 · {r['F1']*100:.2f}%",
                       "Balance entre precisión y sensibilidad.<br><b>2 · P · R / (P + R)</b><br>"
                       "Útil cuando importan ambas a la vez.", AMBER), unsafe_allow_html=True)


def render(ctx):
    vn, fp, fn, vp = _resultados(ctx)
    _formulas(ctx, vn, fp, fn, vp)
    seccion("UNA ANALOGÍA PARA NO CONFUNDIRSE")
    panel("Imagina una <b>alarma de incendios</b>.<br>"
          "• <b>Sensibilidad:</b> de todos los incendios que ocurrieron, ¿cuántos detectó la alarma? "
          "Mira la realidad.<br>"
          "• <b>Precisión:</b> de todas las veces que sonó, ¿cuántas había fuego de verdad? Mira lo que "
          "el modelo predijo.<br>"
          "• <b>Exactitud:</b> de todas las situaciones, ¿cuántas veces acertó, sonando con fuego y "
          "callando sin él?<br><br>"
          "En diagnóstico médico preferimos una alarma que a veces suene de más antes que una que deje "
          "pasar un incendio real. Por eso la <b>sensibilidad</b> es la métrica más importante.")
