"""Pestaña «Entrenamiento en vivo»: ver a un modelo aprender en tiempo real.

El estudiante elige el algoritmo y sus parámetros, y el modelo se entrena de
verdad mientras mira: la frontera se mueve, las curvas crecen y una bitácora
cuenta, paso por paso, qué hizo el modelo.

Dos modos:
  · Automático: se reproduce solo, a la velocidad elegida.
  · Paso a paso: el estudiante avanza con el botón «Siguiente paso».

Los cálculos viven en `entrenamiento.py` (`pasos_en_vivo`, `evaluar_en_prueba`).
"""

import time

import plotly.graph_objects as go
import streamlit as st

import entrenamiento as ent
from config import AMBER, CYAN, GREEN, PINK, VIOLET
from ui import estilo, historia, kpi, panel, seccion, tarjeta

ESCALA_PROB = [[0, "rgba(0,212,255,0.30)"],
               [0.5, "rgba(10,14,26,0.1)"],
               [1, "rgba(255,45,120,0.30)"]]
VELOCIDADES = {"🐢 Lenta": 1.2, "🚶 Normal": 0.5, "🚀 Rápida": 0.12}


# ── Dibujo ───────────────────────────────────────────────────────────────────
def _grafico(fig, clave):
    """plotly_chart con clave única: cada fotograma es un elemento nuevo."""
    try:
        st.plotly_chart(fig, width="stretch", key=clave)
    except TypeError:
        st.plotly_chart(fig, use_container_width=True, key=clave)


def _figura_frontera(f, g, titulo):
    fig = go.Figure()
    fig.add_trace(go.Heatmap(x=f["xs"], y=f["ys"], z=f["zz"], zmin=0, zmax=1,
                             showscale=False, colorscale=ESCALA_PROB, hoverinfo="skip"))
    fig.add_trace(go.Contour(x=f["xs"], y=f["ys"], z=f["zz"], showscale=False,
                             hoverinfo="skip",
                             contours=dict(start=0.5, end=0.5, size=0.1, coloring="lines"),
                             line=dict(color=AMBER, width=3), name="Frontera"))
    Z, y = g["Ztr"], g["ytr"]
    for nombre, clase, color in [("Benigno", 0, CYAN), ("Maligno", 1, PINK)]:
        msk = y == clase
        fig.add_trace(go.Scatter(x=Z[msk, 0], y=Z[msk, 1], mode="markers", name=nombre,
                                 marker=dict(size=6, color=color,
                                             line=dict(width=0.5, color="rgba(255,255,255,0.7)"))))
    err = f["errores"]
    if err.any():
        fig.add_trace(go.Scatter(x=Z[err, 0], y=Z[err, 1], mode="markers",
                                 name=f"Mal clasificados ({int(err.sum())})",
                                 marker=dict(size=13, symbol="x-thin-open", color="#ffffff",
                                             line=dict(width=2, color="#ffffff"))))
    fig.update_layout(xaxis_title="Componente 1", yaxis_title="Componente 2",
                      legend=dict(orientation="h", y=-0.18))
    return estilo(fig, alto=500, titulo=titulo)


def _figura_curvas(hist, total, unidad, con_perdida):
    x = list(range(1, len(hist) + 1))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=[h["acc_tr"] for h in hist], name="Acierto entrenamiento",
                             mode="lines+markers", line=dict(color=VIOLET, width=3)))
    fig.add_trace(go.Scatter(x=x, y=[h["acc_val"] for h in hist], name="Acierto validación",
                             mode="lines+markers", line=dict(color=AMBER, width=3)))
    if con_perdida:
        fig.add_trace(go.Scatter(x=x, y=[h["perdida"] for h in hist], name="Pérdida",
                                 mode="lines", yaxis="y2", line=dict(color=PINK, width=2, dash="dot")))
        fig.update_layout(yaxis2=dict(title="Pérdida", overlaying="y", side="right",
                                      showgrid=False, rangemode="tozero"))
    fig.update_layout(xaxis=dict(title=f"{unidad} (paso)", range=[0.5, total + 0.5]),
                      yaxis=dict(title="Acierto", range=[0.4, 1.02], tickformat=".0%"),
                      legend=dict(orientation="h", y=-0.25))
    return estilo(fig, alto=500, titulo="ASÍ VA APRENDIENDO")


def _bitacora(hist):
    filas = "".join(
        f'<div style="border-left:3px solid {AMBER if j == 0 else "rgba(123,47,255,0.5)"};'
        f'padding:6px 12px;margin-bottom:8px;{"" if j == 0 else "opacity:0.65;"}">'
        f'<b style="color:{AMBER if j == 0 else "#cfe3f0"};font-family:Orbitron,sans-serif;'
        f'font-size:0.85rem">PASO {h["i"]} · {h["etiqueta"].upper()}</b><br>{h["relato"]}</div>'
        for j, h in enumerate(reversed(hist[-6:])))
    return f'<div class="panel" style="max-height:420px;overflow-y:auto">{filas}</div>'


def _interpretar(h, previo):
    """Comentario pedagógico automático sobre el estado actual."""
    brecha = h["acc_tr"] - h["acc_val"]
    msgs = []
    if previo is not None:
        d = h["acc_val"] - previo["acc_val"]
        if d > 0.005:
            msgs.append(f"✅ El acierto en validación <b>subió {d*100:.1f} puntos</b>: el "
                        "modelo está generalizando mejor.")
        elif d < -0.005:
            msgs.append(f"⚠️ El acierto en validación <b>bajó {abs(d)*100:.1f} puntos</b> "
                        "aunque siguió entrenando. Más entrenamiento no siempre es mejor.")
        else:
            msgs.append("➖ La validación casi no cambió: el modelo se está estabilizando.")
    if brecha > 0.05:
        msgs.append(f"🔎 Acierta {brecha*100:.1f} puntos más en entrenamiento que en "
                    "validación: señal de <b>sobreajuste</b> (empieza a memorizar).")
    n_err = int(h["errores"].sum())
    msgs.append(f"❌ Quedan <b>{n_err} pacientes de entrenamiento mal clasificados</b> "
                "(marcados con ✕ en el mapa).")
    return "<br>".join(msgs)


def _dibujar(zona, hist, cfg, g, clave):
    """Pinta un fotograma completo dentro de un contenedor reemplazable."""
    h = hist[-1]
    previo = hist[-2] if len(hist) > 1 else None
    with zona.container():
        st.progress(h["i"] / h["total"],
                    text=f"Paso {h['i']} de {h['total']} · {h['etiqueta']}")
        k = st.columns(4)
        k[0].markdown(kpi(h["etiqueta"], "Avance", CYAN), unsafe_allow_html=True)
        k[1].markdown(kpi(f"{h['acc_tr']*100:.1f}%", "Acierto entrenamiento", VIOLET),
                      unsafe_allow_html=True)
        k[2].markdown(kpi(f"{h['acc_val']*100:.1f}%", "Acierto validación", AMBER),
                      unsafe_allow_html=True)
        perd = f"{h['perdida']:.3f}" if h["perdida"] is not None else f"{h['rec_val']*100:.1f}%"
        k[3].markdown(kpi(perd, "Pérdida" if h["perdida"] is not None else "Sensibilidad val.",
                          PINK if h["perdida"] is not None else GREEN), unsafe_allow_html=True)
        st.write("")
        izq, der = st.columns([3, 2])
        with izq:
            _grafico(_figura_frontera(h, g, f"FRONTERA · {h['etiqueta'].upper()}"),
                     f"{clave}_f")
        with der:
            _grafico(_figura_curvas(hist, h["total"], cfg["unidad"],
                                    h["perdida"] is not None), f"{clave}_c")
        a, b = st.columns([3, 2])
        with a:
            st.markdown("**📓 Bitácora: lo que hizo el modelo**")
            st.markdown(_bitacora(hist), unsafe_allow_html=True)
        with b:
            st.markdown("**🧠 ¿Qué significa?**")
            panel(_interpretar(h, previo))


def _examen_final(h):
    r = ent.evaluar_en_prueba(h["modelo"])
    seccion("EXAMEN FINAL · CONJUNTO DE PRUEBA (EL MODELO NUNCA LO VIO)")
    c = st.columns(4)
    c[0].markdown(kpi(f"{r['acc']*100:.1f}%", "Exactitud en prueba", GREEN), unsafe_allow_html=True)
    c[1].markdown(kpi(f"{r['rec']*100:.1f}%", "Sensibilidad en prueba", PINK), unsafe_allow_html=True)
    c[2].markdown(kpi(f"{r['fn']}", "Malignos no detectados", AMBER), unsafe_allow_html=True)
    c[3].markdown(kpi(f"{r['fp']}", "Falsas alarmas", CYAN), unsafe_allow_html=True)
    panel(f"De los malignos reales, detectó <b>{r['vp']}</b> y se le escaparon <b>{r['fn']}</b>. "
          f"De los benignos, acertó <b>{r['vn']}</b> y alarmó de más en <b>{r['fp']}</b>. "
          "Compara la exactitud de prueba con la de validación: si son parecidas, el modelo "
          "generaliza; si la de prueba es mucho menor, se ajustó demasiado a lo que vio.")


# ── Estado ───────────────────────────────────────────────────────────────────
def _firma(algoritmo, n_pasos, tasa):
    return f"{algoritmo}|{n_pasos}|{tasa}"


def _reproducir_hasta(algoritmo, n_pasos, tasa, hasta):
    """Recalcula de forma determinista los primeros `hasta` pasos (sin animación)."""
    hist = []
    for f in ent.pasos_en_vivo(algoritmo, n_pasos, tasa):
        hist.append(f)
        if len(hist) >= hasta:
            break
    return hist


# ── Punto de entrada ─────────────────────────────────────────────────────────
def render(ctx):
    seccion("ENTRENAMIENTO EN VIVO · MIRA A UN MODELO APRENDER EN TIEMPO REAL")
    historia(
        "Aquí el modelo <b>se entrena de verdad, delante de ti</b>. Elige un algoritmo, "
        "ajusta sus parámetros y dale ▶. En cada paso verás tres cosas: cómo se mueve la "
        "<b>frontera de decisión</b> sobre los pacientes, cómo crecen las <b>curvas de "
        "acierto y pérdida</b>, y una <b>bitácora</b> que cuenta en palabras qué hizo el "
        "modelo. Para poder dibujarlo, los 30 rasgos se resumen en 2 componentes "
        "principales (PCA).",
        GREEN, pregunta="¿Qué cambia dentro del modelo en cada paso?")

    st.markdown('<div class="flow">' + "".join(
        f'<div class="step"><b>{a}</b><span>{b}</span></div>' for a, b in [
            ("1", "Elige el algoritmo"), ("2", "Ajusta pasos y velocidad"),
            ("3", "Entrena: automático o paso a paso"), ("4", "Lee la bitácora"),
            ("5", "Examen final en prueba")]) + "</div>", unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns([2, 1.3, 1.3, 1.3])
    algoritmo = c1.selectbox("Algoritmo", list(ent.MODELOS_VIVO.keys()), key="vivo_algo")
    cfg = ent.MODELOS_VIVO[algoritmo]
    if algoritmo == "SVM (RBF)":
        n_pasos = cfg["pasos_max"]
        c2.markdown(f"**Pasos**  \n{n_pasos} (fijos)")
    else:
        n_pasos = c2.slider(f"Pasos ({cfg['unidad'].lower()}s)", 3, cfg["pasos_max"],
                            cfg["pasos_def"], key=f"vivo_pasos_{algoritmo}")
    if cfg["usa_tasa"]:
        tasa = c3.select_slider("Tasa de aprendizaje", [0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0],
                                value=cfg["tasa_def"], key=f"vivo_tasa_{algoritmo}")
    else:
        tasa = 0.0
        c3.markdown("**Tasa de aprendizaje**  \nNo aplica a este modelo")
    velocidad = c4.radio("Velocidad", list(VELOCIDADES.keys()), index=1, key="vivo_vel")

    a, b = st.columns(2)
    a.markdown(tarjeta("CÓMO APRENDE", cfg["idea"], VIOLET), unsafe_allow_html=True)
    b.markdown(tarjeta("QUÉ OBSERVAR", cfg["mirar"], AMBER), unsafe_allow_html=True)
    st.write("")

    firma = _firma(algoritmo, n_pasos, tasa)
    if st.session_state.get("vivo_firma") != firma:      # cambió la configuración
        st.session_state.vivo_firma = firma
        st.session_state.vivo_paso = 0

    b1, b2, b3, _ = st.columns([1.3, 1.3, 1, 2])
    auto = b1.button("▶ Entrenar en vivo", type="primary", key="vivo_play")
    siguiente = b2.button("⏭ Siguiente paso", key="vivo_next",
                          disabled=st.session_state.vivo_paso >= n_pasos)
    if b3.button("↺ Reiniciar", key="vivo_reset"):
        st.session_state.vivo_paso = 0

    g = ent.proyeccion_2d()
    zona = st.empty()

    if auto:
        hist = []
        for f in ent.pasos_en_vivo(algoritmo, n_pasos, tasa):
            hist.append(f)
            _dibujar(zona, hist, cfg, g, f"vivo_{f['i']}")
            time.sleep(VELOCIDADES[velocidad])
        st.session_state.vivo_paso = len(hist)
    else:
        if siguiente:
            st.session_state.vivo_paso += 1
        paso = st.session_state.vivo_paso
        if paso == 0:
            zona.info("Pulsa **▶ Entrenar en vivo** para verlo completo, o "
                      "**⏭ Siguiente paso** para avanzar tú mismo, un paso a la vez.")
            return
        with st.spinner("Entrenando..."):
            hist = _reproducir_hasta(algoritmo, n_pasos, tasa, paso)
        _dibujar(zona, hist, cfg, g, f"vivo_m_{paso}")

    if hist and hist[-1]["i"] == hist[-1]["total"]:
        st.success(f"🏁 Entrenamiento terminado: {hist[-1]['etiqueta'].lower()}. "
                   "Ahora el examen real.")
        _examen_final(hist[-1])
