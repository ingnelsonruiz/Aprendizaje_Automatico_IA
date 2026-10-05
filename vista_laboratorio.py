"""Pestaña «Laboratorio»: cómo aprende un modelo, paso a paso.

Esta vista solo dibuja. Los cálculos viven en `entrenamiento.py`.
Para agregar un experimento nuevo: escribe su función allí y añade aquí
una sección que la consuma.
"""

import numpy as np
import plotly.graph_objects as go
import streamlit as st

import entrenamiento as ent
from config import AMBER, CYAN, GREEN, PINK, VIOLET
from ui import estilo, historia, kpi, mostrar, panel, seccion

ESCALA_PROB = [[0, "rgba(0,212,255,0.30)"],
               [0.5, "rgba(10,14,26,0.1)"],
               [1, "rgba(255,45,120,0.30)"]]


def _figura_frontera(xs, ys, zz, Z, etiquetas, titulo):
    """Mapa de probabilidad + línea de decisión + nube de puntos."""
    fig = go.Figure()
    fig.add_trace(go.Heatmap(x=xs, y=ys, z=zz, zmin=0, zmax=1,
                             showscale=False, colorscale=ESCALA_PROB))
    fig.add_trace(go.Contour(x=xs, y=ys, z=zz, showscale=False,
                             contours=dict(start=0.5, end=0.5, size=0.1, coloring="lines"),
                             line=dict(color=AMBER, width=3)))
    for nombre, clase, color in [("Benigno", 0, CYAN), ("Maligno", 1, PINK)]:
        msk = etiquetas == clase
        fig.add_trace(go.Scatter(
            x=Z[msk, 0], y=Z[msk, 1], mode="markers", name=nombre,
            marker=dict(size=6, color=color,
                        line=dict(width=0.5, color="rgba(255,255,255,0.5)"))))
    fig.update_layout(xaxis_title="Componente 1", yaxis_title="Componente 2")
    return estilo(fig, alto=460, titulo=titulo)


# ── Secciones ────────────────────────────────────────────────────────────────
def _pasos(ctx):
    seccion("A · LOS CUATRO PASOS DEL ENTRENAMIENTO")
    c1, c2, c3, c4 = st.columns(4)
    from ui import tarjeta
    c1.markdown(tarjeta("1 · DIVIDIR",
                        f"Se apartan <b>{ctx.n_test} casos</b> que el modelo <b>nunca verá</b> al entrenar. "
                        f"Con los otros <b>{ctx.n_train}</b> aprende. Así la evaluación es un examen con "
                        "preguntas nuevas.", CYAN), unsafe_allow_html=True)
    c2.markdown(tarjeta("2 · ESCALAR",
                        "El área vale miles y la suavidad vale décimas. Se estandarizan todas a la misma "
                        "escala para que ninguna domine solo por su unidad.", VIOLET), unsafe_allow_html=True)
    c3.markdown(tarjeta("3 · AJUSTAR",
                        "El modelo arranca con parámetros al azar, mide qué tan mal predice (la <b>pérdida</b>) "
                        "y los corrige. Repite eso miles de veces.", GREEN), unsafe_allow_html=True)
    c4.markdown(tarjeta("4 · EVALUAR",
                        "Se le muestran los casos apartados y se cuentan aciertos y errores con la matriz "
                        "de confusión.", AMBER), unsafe_allow_html=True)


def _gradiente():
    seccion("B · EL MODELO APRENDIENDO, ÉPOCA POR ÉPOCA")
    historia(
        "Aquí entrenamos una regresión logística sobre 2 dimensiones (para poder dibujarla) usando "
        "<b>descenso de gradiente</b>. Mueve el control de épocas: una época es una pasada completa por los "
        "datos de entrenamiento. Verás cómo la <b>pérdida baja</b> y cómo la <b>frontera de decisión</b> "
        "gira hasta acomodarse entre los dos grupos.",
        VIOLET,
    )

    with st.spinner("Entrenando paso a paso..."):
        Z, etiquetas, hist = ent.descenso_gradiente()

    epoca = st.slider("Época de entrenamiento", 1, len(hist), 1, key="lab_epoca")
    h = hist[epoca - 1]

    g1, g2, g3, g4 = st.columns(4)
    g1.markdown(kpi(f"{epoca}", "Época", CYAN), unsafe_allow_html=True)
    g2.markdown(kpi(f"{h['perdida']:.4f}", "Pérdida", PINK), unsafe_allow_html=True)
    g3.markdown(kpi(f"{h['acc_tr']*100:.1f}%", "Acierto entrenamiento", VIOLET), unsafe_allow_html=True)
    g4.markdown(kpi(f"{h['acc_te']*100:.1f}%", "Acierto prueba", GREEN), unsafe_allow_html=True)

    st.write("")
    izq, der = st.columns(2)
    with izq:
        xs, ys, zz = ent.malla_sigmoide(Z, h["w"], h["b"])
        mostrar(_figura_frontera(xs, ys, zz, Z, etiquetas,
                                 f"FRONTERA DE DECISIÓN EN LA ÉPOCA {epoca}"))
    with der:
        ep = [x["epoca"] for x in hist]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=ep, y=[x["perdida"] for x in hist], name="Pérdida",
                                 line=dict(color=PINK, width=3)))
        fig.add_trace(go.Scatter(x=[epoca], y=[h["perdida"]], mode="markers", showlegend=False,
                                 marker=dict(size=14, color=AMBER, line=dict(width=2, color="#fff"))))
        fig.update_layout(xaxis_title="Época", yaxis_title="Pérdida (log loss)")
        mostrar(estilo(fig, alto=215, titulo="LA PÉRDIDA BAJA EN CADA ÉPOCA"))

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=ep, y=[x["acc_tr"] for x in hist], name="Entrenamiento",
                                 line=dict(color=VIOLET, width=3)))
        fig.add_trace(go.Scatter(x=ep, y=[x["acc_te"] for x in hist], name="Prueba",
                                 line=dict(color=GREEN, width=3)))
        fig.add_vline(x=epoca, line=dict(color=AMBER, dash="dash", width=2))
        fig.update_layout(xaxis_title="Época", yaxis_title="Acierto")
        mostrar(estilo(fig, alto=215, titulo="EL ACIERTO SUBE"))

    panel("La <b>pérdida</b> es el castigo por equivocarse: entre más confiado esté el modelo en una "
          "respuesta equivocada, más alta es. Entrenar es simplemente <b>bajar esa pérdida</b>. "
          "La línea amarilla es la frontera donde el modelo duda, con 50 % de probabilidad para cada clase.")


def _fronteras():
    seccion("C · CADA ALGORITMO DIBUJA LA FRONTERA A SU MANERA")
    historia(
        "Todos los modelos buscan lo mismo: separar los dos grupos. Pero cada uno lo hace con una forma "
        "distinta. Cambia el algoritmo y su parámetro para ver la diferencia: la regresión logística traza "
        "una <b>línea recta</b>, el árbol hace <b>cortes rectangulares</b>, el bosque los <b>promedia</b> "
        "y el SVM dibuja <b>curvas suaves</b>.",
        GREEN,
    )

    izq, der = st.columns([1, 2])
    with izq:
        algo = st.selectbox("Algoritmo", list(ent.ALGORITMOS.keys()), key="lab_algo")
        cfg = ent.ALGORITMOS[algo]
        if cfg["tipo"] == "slider":
            param = st.slider(cfg["parametro"], cfg["minimo"], cfg["maximo"],
                              cfg["defecto"], key=f"lab_p_{algo}")
        else:
            param = st.select_slider(cfg["parametro"], options=cfg["opciones"],
                                     value=cfg["defecto"], key=f"lab_p_{algo}")
        st.caption(cfg["ayuda"])

    with st.spinner("Reentrenando el modelo..."):
        F = ent.frontera(algo, param)

    with izq:
        st.markdown(kpi(f"{F['acc_tr']*100:.1f}%", "Acierto entrenamiento", VIOLET), unsafe_allow_html=True)
        st.write("")
        st.markdown(kpi(f"{F['acc_te']*100:.1f}%", "Acierto prueba", GREEN), unsafe_allow_html=True)

    with der:
        mostrar(_figura_frontera(F["xs"], F["ys"], F["zz"], F["Ztr"], F["ytr"],
                                 f"FRONTERA · {algo.upper()}"))


def _sobreajuste():
    seccion("D · EL PELIGRO DE MEMORIZAR: SOBREAJUSTE")
    historia(
        "Un modelo puede llegar a <b>100 % de acierto en entrenamiento</b> y seguir siendo malo. Eso pasa "
        "cuando en vez de aprender el patrón, memoriza los casos. En la gráfica, mira qué pasa cuando el "
        "árbol se hace más profundo: el acierto en entrenamiento sube hasta el tope, pero el de prueba se "
        "estanca. Esa separación entre las dos líneas es el sobreajuste.",
        PINK,
    )
    profundidades, acc_tr, acc_te = ent.curva_sobreajuste()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=profundidades, y=acc_tr, name="Entrenamiento (lo que ya vio)",
                             line=dict(color=VIOLET, width=3)))
    fig.add_trace(go.Scatter(x=profundidades, y=acc_te, name="Prueba (casos nuevos)",
                             line=dict(color=GREEN, width=3)))
    mejor = profundidades[int(np.argmax(acc_te))]
    fig.add_vline(x=mejor, line=dict(color=AMBER, dash="dash", width=2))
    fig.update_layout(xaxis_title="Profundidad máxima del árbol", yaxis_title="Acierto")
    mostrar(estilo(fig, alto=420, titulo="SOBREAJUSTE SEGÚN LA COMPLEJIDAD DEL MODELO"))
    panel(f"La línea amarilla marca la profundidad <b>{mejor}</b>, donde el acierto en casos nuevos es "
          "mayor. Pasado ese punto, agregar complejidad ya no ayuda: solo hace que el modelo memorice. "
          "Por eso <b>siempre</b> se evalúa con datos que el modelo nunca vio.")


# ── Punto de entrada de la vista ─────────────────────────────────────────────
def render(ctx):
    seccion("LABORATORIO · ¿QUÉ PASA REALMENTE CUANDO UN MODELO «APRENDE»?")
    historia(
        "Entrenar un modelo suena a magia, pero no lo es. Es un proceso con pasos concretos que se pueden "
        "<b>ver</b>. En este laboratorio vas a mover los controles y observar, en vivo, cómo el modelo pasa "
        "de no saber nada a separar correctamente los tumores benignos de los malignos.",
        CYAN,
    )
    _pasos(ctx)
    _gradiente()
    _fronteras()
    _sobreajuste()
