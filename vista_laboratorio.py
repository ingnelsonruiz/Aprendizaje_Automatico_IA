"""Pestaña «Laboratorio»: cómo aprende un modelo, paso a paso y con trazabilidad.

Esta vista solo dibuja. Los cálculos viven en `entrenamiento.py`.
Para agregar un experimento: escribe su función allí y añade aquí una
sección `_mi_experimento()` que la consuma, llamada desde `render(ctx)`.

Secciones:
  A · El reparto de los datos
  B · El modelo aprendiendo paso a paso (con elección de algoritmo)
  C · Validación cruzada
  D · ¿Cuántos datos hacen falta?
  E · Sobreajuste
  F · La traza de una decisión
  G · Evaluar pacientes nuevos
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import entrenamiento as ent
from config import AMBER, CYAN, GREEN, NARANJA, PINK, VIOLET
from ui import estilo, historia, kpi, mostrar, panel, seccion, tabla, tarjeta

ESCALA_PROB = [[0, "rgba(0,212,255,0.30)"],
               [0.5, "rgba(10,14,26,0.1)"],
               [1, "rgba(255,45,120,0.30)"]]
MAPA_CLASE = {"Benigno": CYAN, "Maligno": PINK}


def _figura_frontera(xs, ys, zz, nubes, titulo, alto=470):
    """Mapa de probabilidad + línea de decisión + una o más nubes de puntos.

    `nubes` es una lista de (nombre, Z, etiquetas, símbolo).
    """
    fig = go.Figure()
    fig.add_trace(go.Heatmap(x=xs, y=ys, z=zz, zmin=0, zmax=1,
                             showscale=False, colorscale=ESCALA_PROB))
    fig.add_trace(go.Contour(x=xs, y=ys, z=zz, showscale=False,
                             contours=dict(start=0.5, end=0.5, size=0.1, coloring="lines"),
                             line=dict(color=AMBER, width=3)))
    for etiqueta_nube, Z, etiquetas, simbolo in nubes:
        for nombre, clase, color in [("Benigno", 0, CYAN), ("Maligno", 1, PINK)]:
            msk = etiquetas == clase
            if not msk.any():
                continue
            fig.add_trace(go.Scatter(
                x=Z[msk, 0], y=Z[msk, 1], mode="markers",
                name=f"{nombre} · {etiqueta_nube}",
                marker=dict(size=7 if simbolo != "circle" else 6, color=color,
                            symbol=simbolo,
                            line=dict(width=1 if simbolo != "circle" else 0.5,
                                      color="rgba(255,255,255,0.75)"))))
    fig.update_layout(xaxis_title="Componente 1", yaxis_title="Componente 2")
    return estilo(fig, alto=alto, titulo=titulo)


# ═══════════════════════════════════════════════════════════════════════════
# A · EL REPARTO DE LOS DATOS
# ═══════════════════════════════════════════════════════════════════════════
def _reparto():
    seccion("A · EL REPARTO DE LOS DATOS")
    historia(
        "Antes de entrenar nada hay que repartir los pacientes en tres montones. Es la "
        "decisión más importante de todo el proceso y la que más se equivoca la gente: "
        "si el modelo ve los mismos datos con los que luego lo evalúas, el resultado es "
        "mentira.",
        CYAN)

    p = ent.particion_triple()
    n_tr, n_val, n_te = len(p["Xtr"]), len(p["Xval"]), len(p["Xte"])
    total = n_tr + n_val + n_te

    a, b, c = st.columns(3)
    a.markdown(tarjeta("ENTRENAMIENTO", f"Con estos el modelo <b>aprende</b>: ajusta sus "
                                        f"pesos o construye sus reglas. Es el único montón "
                                        f"que el modelo «ve».<br><br>"
                                        f"<b>{n_tr/total*100:.0f} %</b> del total", CYAN, n_tr),
               unsafe_allow_html=True)
    b.markdown(tarjeta("VALIDACIÓN", "Sirve para <b>afinar</b>: elegir profundidad, número de "
                                     "árboles, umbral. El modelo no aprende de ellos, pero sí "
                                     "nos guían.<br><br>"
                                     f"<b>{n_val/total*100:.0f} %</b> del total", AMBER, n_val),
               unsafe_allow_html=True)
    c.markdown(tarjeta("PRUEBA", "Se guarda bajo llave hasta el final. Es el <b>examen</b> "
                                 "real: solo se usa una vez, para reportar el desempeño."
                                 f"<br><br><b>{n_te/total*100:.0f} %</b> del total", GREEN, n_te),
               unsafe_allow_html=True)

    st.write("")
    izq, der = st.columns([3, 2])
    with izq:
        fig = go.Figure()
        for nombre, n, color in [("Entrenamiento", n_tr, CYAN), ("Validación", n_val, AMBER),
                                 ("Prueba", n_te, GREEN)]:
            fig.add_trace(go.Bar(y=["Pacientes"], x=[n], name=f"{nombre} ({n})",
                                 orientation="h", marker_color=color,
                                 text=f"{nombre}<br>{n}", textposition="inside",
                                 insidetextanchor="middle"))
        fig.update_layout(barmode="stack", showlegend=False,
                          xaxis_title=f"{total} pacientes en total", yaxis_title="")
        mostrar(estilo(fig, alto=210, titulo="CÓMO SE REPARTEN LOS 569 PACIENTES"))

    with der:
        filas = []
        for nombre, yy in [("Entrenamiento", p["ytr"]), ("Validación", p["yval"]),
                           ("Prueba", p["yte"])]:
            filas.append({"Conjunto": nombre,
                          "Benignos": int((yy == 0).sum()),
                          "Malignos": int((yy == 1).sum()),
                          "% maligno": f"{(yy == 1).mean()*100:.1f}%"})
        tabla(pd.DataFrame(filas))

    panel("Fíjate en la última columna: el porcentaje de malignos es casi idéntico en los "
          "tres montones. Eso no es casualidad, se llama <b>estratificación</b>. Si repartes "
          "al azar sin estratificar, puede tocarte un conjunto de prueba con muy pocos "
          "malignos y la evaluación deja de ser representativa.")


# ═══════════════════════════════════════════════════════════════════════════
# B · EL MODELO APRENDIENDO PASO A PASO
# ═══════════════════════════════════════════════════════════════════════════
def _paso_a_paso():
    seccion("B · EL MODELO APRENDIENDO, PASO A PASO")
    historia(
        "Elige un algoritmo y míralo construirse. Cada uno avanza de una manera distinta, "
        "así que el control cambia de significado según el modelo: para la regresión "
        "logística son <b>épocas</b>, para el árbol son <b>niveles de profundidad</b>, para "
        "el bosque son <b>árboles</b> y para el SVM son <b>iteraciones</b>. "
        "Los puntos redondos son datos de entrenamiento; los diamantes, de validación: "
        "el modelo nunca los usó para aprender.",
        VIOLET)

    algoritmo = st.selectbox("Algoritmo a observar", list(ent.MODELOS_LAB.keys()),
                             key="lab_algoritmo")

    with st.spinner(f"Entrenando {algoritmo} paso a paso..."):
        exp = ent.entrenamiento_progresivo(algoritmo)
    cfg, hist = exp["cfg"], exp["historial"]

    panel(f"<b>¿Qué avanza en cada paso?</b> {cfg['avanza']}<br><br>"
          f"<i>{cfg['nota']}</i>")

    indice = st.slider(f"Avance · {cfg['unidad']}", 1, len(hist), 1, key="lab_paso")
    h = hist[indice - 1]
    valor = "sin límite" if h["paso"] == -1 else h["paso"]

    k = st.columns(4)
    k[0].markdown(kpi(f"{valor}", cfg["unidad"], CYAN), unsafe_allow_html=True)
    k[1].markdown(kpi(f"{h['acc_tr']*100:.1f}%", "Acierto entrenamiento", VIOLET),
                  unsafe_allow_html=True)
    k[2].markdown(kpi(f"{h['acc_val']*100:.1f}%", "Acierto validación", AMBER),
                  unsafe_allow_html=True)
    k[3].markdown(kpi(f"{h['rec_val']*100:.1f}%", "Sensibilidad validación", GREEN),
                  unsafe_allow_html=True)

    st.write("")
    izq, der = st.columns([3, 2])
    with izq:
        mostrar(_figura_frontera(
            exp["xs"], exp["ys"], h["zz"],
            [("entrenamiento", exp["Ztr"], exp["ytr"], "circle"),
             ("validación", exp["Zval"], exp["yval"], "diamond")],
            f"FRONTERA CON {valor} {cfg['unidad'].upper()}"))

    with der:
        pasos_x = list(range(1, len(hist) + 1))
        if cfg["tiene_perdida"]:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=pasos_x, y=[x["perdida"] for x in hist],
                                     name="Pérdida", line=dict(color=PINK, width=3)))
            fig.add_trace(go.Scatter(x=[indice], y=[h["perdida"]], mode="markers",
                                     showlegend=False,
                                     marker=dict(size=14, color=AMBER,
                                                 line=dict(width=2, color="#fff"))))
            fig.update_layout(xaxis_title=cfg["unidad"].capitalize(),
                              yaxis_title="Pérdida (log loss)")
            mostrar(estilo(fig, alto=215, titulo="EL ERROR BAJA EN CADA PASO"))
            alto_acc = 215
        else:
            alto_acc = 440

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=pasos_x, y=[x["acc_tr"] for x in hist],
                                 name="Entrenamiento", line=dict(color=VIOLET, width=3)))
        fig.add_trace(go.Scatter(x=pasos_x, y=[x["acc_val"] for x in hist],
                                 name="Validación", line=dict(color=AMBER, width=3)))
        fig.add_vline(x=indice, line=dict(color=AMBER, dash="dash", width=2))
        fig.update_layout(xaxis_title=cfg["unidad"].capitalize(), yaxis_title="Acierto")
        mostrar(estilo(fig, alto=alto_acc, titulo="ENTRENAMIENTO VS. VALIDACIÓN"))

    brecha = h["acc_tr"] - h["acc_val"]
    if brecha > 0.05:
        panel(f"⚠ En este punto el modelo acierta <b>{brecha*100:.1f} puntos más</b> en "
              "entrenamiento que en validación. Esa brecha es la señal de que está "
              "empezando a memorizar en vez de generalizar.")
    else:
        panel("Las dos líneas van juntas: el modelo está aprendiendo el patrón, no "
              "memorizando. Cuando la línea morada se despega hacia arriba y la amarilla "
              "se queda plana, aparece el sobreajuste.")


# ═══════════════════════════════════════════════════════════════════════════
# C · VALIDACIÓN CRUZADA
# ═══════════════════════════════════════════════════════════════════════════
def _cruzada():
    seccion("C · VALIDACIÓN CRUZADA: NO CONFIAR EN UNA SOLA PARTICIÓN")
    historia(
        "Un solo reparto puede salir con suerte. La <b>validación cruzada</b> parte los "
        "datos en 5 pliegues y entrena 5 veces: en cada vuelta, un pliegue distinto hace "
        "de validación y los otros cuatro de entrenamiento. Al final se promedia. Si los "
        "5 resultados se parecen, el modelo es estable.",
        GREEN)

    algoritmo = st.selectbox("Algoritmo a validar", list(ent.MODELOS_LAB.keys()),
                             key="lab_cv_algo")
    with st.spinner("Entrenando 5 veces..."):
        pliegues, n_total = ent.validacion_cruzada(algoritmo)

    accs = [p["acc"] for p in pliegues]
    recs = [p["rec"] for p in pliegues]
    k = st.columns(4)
    k[0].markdown(kpi(f"{np.mean(accs)*100:.2f}%", "Acierto promedio", CYAN),
                  unsafe_allow_html=True)
    k[1].markdown(kpi(f"±{np.std(accs)*100:.2f}", "Variación entre pliegues", VIOLET),
                  unsafe_allow_html=True)
    k[2].markdown(kpi(f"{np.mean(recs)*100:.2f}%", "Sensibilidad promedio", GREEN),
                  unsafe_allow_html=True)
    k[3].markdown(kpi(f"{n_total}", "Pacientes repartidos", AMBER), unsafe_allow_html=True)

    st.write("")
    izq, der = st.columns([3, 2])
    with izq:
        fig = go.Figure()
        etiquetas = [f"Pliegue {p['pliegue']}" for p in pliegues]
        fig.add_trace(go.Bar(x=etiquetas, y=accs, name="Acierto", marker_color=CYAN,
                             text=[f"{v*100:.1f}%" for v in accs], textposition="outside"))
        fig.add_trace(go.Bar(x=etiquetas, y=recs, name="Sensibilidad", marker_color=GREEN,
                             text=[f"{v*100:.1f}%" for v in recs], textposition="outside"))
        fig.add_hline(y=float(np.mean(accs)), line=dict(color=AMBER, dash="dash", width=2))
        fig.update_layout(barmode="group", yaxis=dict(range=[0.80, 1.05]),
                          yaxis_title="Valor")
        mostrar(estilo(fig, alto=400, titulo=f"5 PLIEGUES · {algoritmo.upper()}"))
    with der:
        tabla(pd.DataFrame([{
            "Pliegue": p["pliegue"], "Entrena con": p["n_train"], "Valida con": p["n_val"],
            "Acierto": f"{p['acc']*100:.2f}%", "Sensibilidad": f"{p['rec']*100:.2f}%",
        } for p in pliegues]))
        panel(f"La línea amarilla es el promedio. Una variación de "
              f"<b>±{np.std(accs)*100:.2f} puntos</b> indica qué tan dependiente es el "
              "resultado del reparto que te tocó.")


# ═══════════════════════════════════════════════════════════════════════════
# D · CUÁNTOS DATOS HACEN FALTA
# ═══════════════════════════════════════════════════════════════════════════
def _cuantos_datos():
    seccion("D · ¿CUÁNTOS PACIENTES HACEN FALTA PARA APRENDER?")
    historia(
        "Esta curva responde una pregunta muy práctica: ¿vale la pena conseguir más datos? "
        "Se entrena el mismo modelo con cada vez más pacientes y se mide siempre contra la "
        "misma validación. Si la línea amarilla se aplana, más datos ya no ayudan; si "
        "sigue subiendo, conseguir más casos sí mejoraría el modelo.",
        AMBER)

    algoritmo = st.selectbox("Algoritmo", list(ent.MODELOS_LAB.keys()), key="lab_curva_algo")
    with st.spinner("Entrenando con distintas cantidades de datos..."):
        filas = ent.curva_aprendizaje(algoritmo)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[f["n"] for f in filas], y=[f["acc_tr"] for f in filas],
                             name="Entrenamiento", line=dict(color=VIOLET, width=3),
                             mode="lines+markers"))
    fig.add_trace(go.Scatter(x=[f["n"] for f in filas], y=[f["acc_val"] for f in filas],
                             name="Validación", line=dict(color=AMBER, width=3),
                             mode="lines+markers"))
    fig.update_layout(xaxis_title="Pacientes usados para entrenar", yaxis_title="Acierto")
    mostrar(estilo(fig, alto=400, titulo=f"CURVA DE APRENDIZAJE · {algoritmo.upper()}"))

    primero, ultimo = filas[0], filas[-1]
    mejora = (ultimo["acc_val"] - primero["acc_val"]) * 100
    panel(f"Pasando de <b>{primero['n']}</b> a <b>{ultimo['n']}</b> pacientes, el acierto en "
          f"validación cambió <b>{mejora:+.1f} puntos</b>. Si ese número es pequeño, el "
          "modelo ya aprovechó lo que estos datos tenían para dar.")


# ═══════════════════════════════════════════════════════════════════════════
# E · SOBREAJUSTE
# ═══════════════════════════════════════════════════════════════════════════
def _sobreajuste():
    seccion("E · EL PELIGRO DE MEMORIZAR: SOBREAJUSTE")
    historia(
        "Un modelo puede llegar a <b>100 % de acierto en entrenamiento</b> y seguir siendo "
        "malo. Mira qué pasa cuando el árbol se hace más profundo: el acierto en "
        "entrenamiento sube hasta el tope, pero el de validación se estanca. Esa separación "
        "entre las dos líneas es el sobreajuste.",
        PINK)

    profundidades, acc_tr, acc_val = ent.curva_sobreajuste()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=profundidades, y=acc_tr, name="Entrenamiento (ya lo vio)",
                             line=dict(color=VIOLET, width=3)))
    fig.add_trace(go.Scatter(x=profundidades, y=acc_val, name="Validación (casos nuevos)",
                             line=dict(color=AMBER, width=3)))
    mejor = profundidades[int(np.argmax(acc_val))]
    fig.add_vline(x=mejor, line=dict(color=GREEN, dash="dash", width=2))
    fig.update_layout(xaxis_title="Profundidad máxima del árbol", yaxis_title="Acierto")
    mostrar(estilo(fig, alto=400, titulo="SOBREAJUSTE SEGÚN LA COMPLEJIDAD DEL MODELO"))
    panel(f"La línea verde marca la profundidad <b>{mejor}</b>, donde el acierto en casos "
          "nuevos es mayor. Pasado ese punto, agregar complejidad ya no ayuda: solo hace "
          "que el modelo memorice.")


# ═══════════════════════════════════════════════════════════════════════════
# F · LA TRAZA DE UNA DECISIÓN
# ═══════════════════════════════════════════════════════════════════════════
def _traza(ctx):
    seccion("F · LA TRAZA: CÓMO SE LLEGA A UNA DECISIÓN CONCRETA")
    historia(
        "Hasta aquí vimos el modelo entrenarse. Ahora vamos al revés: tomamos <b>un "
        "paciente</b> y seguimos su recorrido completo, número por número, hasta la "
        "probabilidad final. Nada de caja negra.",
        NARANJA)

    muestra = ent.muestra_de_prueba()
    etiquetas = [f"{r.id_paciente} · diagnóstico real: {r.Diagnóstico}"
                 for r in muestra.itertuples()]
    elegido = st.selectbox("Paciente del conjunto de prueba", etiquetas, key="lab_traza_pac")
    i = etiquetas.index(elegido)
    fila = muestra.loc[[i], ctx.columnas]
    real = muestra.loc[i, "Diagnóstico"]

    # ── Paso 1: estandarización ─────────────────────────────────────────────
    st.markdown("##### Paso 1 · De la unidad original al valor estandarizado")
    panel("Cada variable se convierte en «cuántas desviaciones estándar está por encima o "
          "por debajo del promedio del entrenamiento». Así el área, que vale cientos, y la "
          "suavidad, que vale décimas, pasan a ser comparables.")
    df_z, z = ent.traza_escalado(fila, ctx.escalador, ctx.columnas)
    df_z_vista = df_z.copy()
    df_z_vista["Operación"] = [
        f"({v:.3f} − {m:.3f}) ÷ {s:.3f}"
        for v, m, s in zip(df_z["Valor del paciente"], df_z["Promedio del entrenamiento"],
                           df_z["Desviación estándar"])]
    tabla(df_z_vista.round(4).head(10))
    st.caption("Se muestran las primeras 10 de las 30 variables.")

    # ── Paso 2: según el algoritmo ──────────────────────────────────────────
    st.markdown("##### Paso 2 · Cómo decide cada algoritmo")
    algoritmo = st.selectbox("Algoritmo a trazar",
                             ["Regresión Logística", "Árbol de decisión", "Random Forest",
                              "SVM (RBF)"], key="lab_traza_algo")
    fila_escalada = ctx.escalador.transform(fila)

    if algoritmo == "Regresión Logística":
        modelo = ctx.resultados["Regresión Logística"]["modelo"]
        df_ap, total, sesgo, prob = ent.traza_logistica(modelo, z, ctx.columnas)
        panel("La regresión logística multiplica cada valor estandarizado por un "
              "<b>peso aprendido</b> y suma todo. Un aporte positivo empuja hacia maligno; "
              "uno negativo, hacia benigno.")
        izq, der = st.columns([2, 3])
        with izq:
            tabla(df_ap.head(12))
            st.caption("Las 12 variables de mayor aporte absoluto.")
        with der:
            top = df_ap.head(12).iloc[::-1]
            fig = go.Figure(go.Bar(
                x=top["Aporte (w × z)"], y=top["Variable"], orientation="h",
                marker_color=[PINK if v > 0 else CYAN for v in top["Aporte (w × z)"]],
                text=[f"{v:+.2f}" for v in top["Aporte (w × z)"]], textposition="outside"))
            fig.update_layout(xaxis_title="Aporte a la decisión")
            mostrar(estilo(fig, alto=420, titulo="QUIÉN EMPUJA HACIA DÓNDE"))
        panel(f"<b>Suma de aportes + sesgo:</b> {total - sesgo:+.3f} {sesgo:+.3f} = "
              f"<b>{total:+.3f}</b><br>"
              f"Ese número pasa por la función sigmoide, que lo aplasta entre 0 y 1: "
              f"<b>probabilidad de maligno = {prob*100:.1f} %</b>")
        prob_final = prob

    elif algoritmo == "Árbol de decisión":
        arbol, sc_arbol, cols_arbol = ent.arbol_interpretable()
        z_arbol = sc_arbol.transform(fila)[0]
        reglas, hoja = ent.traza_arbol(arbol, z_arbol, cols_arbol, sc_arbol)
        panel("El árbol hace una secuencia de preguntas de sí o no. Cada respuesta lo manda "
              "por una rama. Al final llega a una <b>hoja</b>, y la probabilidad es "
              "simplemente qué proporción de los pacientes de entrenamiento que cayeron en "
              "esa misma hoja eran malignos.")
        tabla(reglas)
        panel(f"<b>Hoja final:</b> allí cayeron <b>{hoja['n_hoja']}</b> pacientes del "
              f"entrenamiento, de los cuales <b>{hoja['malignos']}</b> eran malignos y "
              f"<b>{hoja['benignos']}</b> benignos.<br>"
              f"<b>Probabilidad de maligno = {hoja['malignos']}/{hoja['n_hoja']} = "
              f"{hoja['probabilidad']*100:.1f} %</b>")
        st.caption("Se usa un árbol de profundidad 4 para que las reglas sean legibles.")
        prob_final = hoja["probabilidad"]

    elif algoritmo == "Random Forest":
        bosque = ctx.resultados["Random Forest"]["modelo"]
        v = ent.traza_bosque(bosque, fila_escalada)
        panel(f"El bosque no decide solo: le pregunta a sus <b>{v['total']} árboles</b> y "
              "cuenta los votos. La probabilidad es la proporción de árboles que dijeron "
              "«maligno».")
        izq, der = st.columns([2, 3])
        with izq:
            st.markdown(kpi(f"{v['malignos']}", "Votan maligno", PINK), unsafe_allow_html=True)
            st.write("")
            st.markdown(kpi(f"{v['benignos']}", "Votan benigno", CYAN), unsafe_allow_html=True)
            st.write("")
            st.markdown(kpi(f"{v['probabilidad']*100:.1f}%", "Probabilidad", AMBER),
                        unsafe_allow_html=True)
        with der:
            fig = go.Figure(go.Bar(x=["Benigno", "Maligno"], y=[v["benignos"], v["malignos"]],
                                   marker_color=[CYAN, PINK],
                                   text=[v["benignos"], v["malignos"]], textposition="outside"))
            fig.update_layout(yaxis_title="Árboles que votaron así")
            mostrar(estilo(fig, alto=380, titulo=f"VOTACIÓN DE LOS {v['total']} ÁRBOLES"))
        panel(f"<b>Probabilidad de maligno = {v['malignos']}/{v['total']} = "
              f"{v['probabilidad']*100:.1f} %</b>. Cuando la votación es muy pareja, el "
              "modelo está genuinamente dudando.")
        prob_final = v["probabilidad"]

    else:
        modelo = ctx.resultados["SVM (RBF)"]["modelo"]
        t = ent.traza_svm(modelo, fila_escalada)
        prob_final = float(modelo.predict_proba(fila_escalada)[0, 1])
        panel("El SVM traza una frontera y mide <b>a qué distancia</b> cae el paciente. El "
              "signo indica el lado; la magnitud, qué tan lejos está de la zona de duda.")
        k = st.columns(3)
        k[0].markdown(kpi(f"{t['distancia']:+.3f}", "Distancia a la frontera",
                          PINK if t["distancia"] > 0 else CYAN), unsafe_allow_html=True)
        k[1].markdown(kpi(t["lado"], "Lado de la frontera", VIOLET), unsafe_allow_html=True)
        k[2].markdown(kpi(f"{prob_final*100:.1f}%", "Probabilidad calibrada", AMBER),
                      unsafe_allow_html=True)
        panel("Un valor cercano a cero significa que el paciente cae casi encima de la "
              "frontera: el modelo no está seguro. La probabilidad se obtiene calibrando "
              "esa distancia con el método de Platt.")

    # ── Paso 3: veredicto ───────────────────────────────────────────────────
    st.markdown("##### Paso 3 · El veredicto, comparado con la realidad")
    predicho = "Maligno" if prob_final >= 0.5 else "Benigno"
    acerto = predicho == real
    izq, der = st.columns(2)
    izq.markdown(tarjeta("PREDICCIÓN DEL MODELO",
                         f"{predicho} · <b>{prob_final*100:.1f} %</b> de probabilidad de maligno",
                         PINK if predicho == "Maligno" else CYAN), unsafe_allow_html=True)
    der.markdown(tarjeta("DIAGNÓSTICO REAL (BIOPSIA)",
                         f"{real} · {'✔ el modelo acertó' if acerto else '✘ el modelo se equivocó'}",
                         GREEN if acerto else AMBER), unsafe_allow_html=True)
    if not acerto and real == "Maligno":
        panel("⚠ Este es un <b>falso negativo</b>: el caso más grave. Es exactamente el tipo "
              "de error que se busca reducir bajando el umbral de decisión.")


# ═══════════════════════════════════════════════════════════════════════════
# G · EVALUAR PACIENTES NUEVOS
# ═══════════════════════════════════════════════════════════════════════════
def _evaluar(ctx):
    seccion("G · EVALUAR PACIENTES NUEVOS DESDE UN ARCHIVO")
    historia(
        "Un modelo entrenado sirve para algo solo si lo puedes usar con <b>datos que nunca "
        "vio</b>. Aquí puedes cargar un archivo CSV con uno o varios pacientes y obtener su "
        "diagnóstico predicho. Si no tienes archivo, usa los pacientes reales de ejemplo.",
        CYAN)

    izq, der = st.columns(2)
    with izq:
        plantilla = ent.plantilla_csv(ctx.columnas, ctx.X)
        st.download_button("⬇ Descargar plantilla CSV vacía",
                           plantilla.to_csv(index=False).encode("utf-8"),
                           "plantilla_pacientes.csv", "text/csv",
                           help="Trae las 30 columnas con los promedios del dataset. "
                                "Edita los valores y vuelve a cargarla.")
    with der:
        muestra = ent.muestra_de_prueba(8)
        st.download_button("⬇ Descargar 8 pacientes reales de prueba",
                           muestra.to_csv(index=False).encode("utf-8"),
                           "pacientes_reales.csv", "text/csv",
                           help="Pacientes del conjunto de prueba con su diagnóstico "
                                "confirmado, para comparar con la predicción.")

    st.write("")
    c1, c2 = st.columns([2, 1])
    with c1:
        archivo = st.file_uploader("Carga un CSV con pacientes", type=["csv"],
                                   key="lab_csv")
    with c2:
        usar_ejemplo = st.checkbox("Usar los 8 pacientes de ejemplo", value=False,
                                   key="lab_usar_ejemplo")
        umbral = st.slider("Umbral de maligno", 0.05, 0.95, 0.50, 0.01, key="lab_umbral_csv")

    datos = None
    if usar_ejemplo:
        datos = ent.muestra_de_prueba(8)
    elif archivo is not None:
        try:
            datos = pd.read_csv(archivo)
        except Exception as e:
            st.error(f"No pude leer el archivo: {e}")
            return

    if datos is None:
        panel("Carga un archivo o marca la casilla de ejemplo para ver los resultados. "
              "El CSV acepta los nombres en español (<i>Radio (media)</i>) o los originales "
              "de UCI (<i>radius1</i>). Puede traer columnas extra como "
              "<code>id_paciente</code>: se conservan en la salida.")
        return

    datos, faltantes, ignoradas = ent.normalizar_columnas(datos, ctx.columnas)
    if faltantes:
        st.error(f"Al archivo le faltan {len(faltantes)} columnas obligatorias. "
                 f"Las primeras: {', '.join(faltantes[:5])}")
        st.caption("Descarga la plantilla de arriba para ver los nombres exactos.")
        return
    if ignoradas:
        st.caption(f"Columnas no usadas para predecir (se conservan si son identificadores): "
                   f"{', '.join(ignoradas[:6])}")

    modelo = ctx.activo["modelo"]
    resultados, descartadas = ent.evaluar_pacientes(datos, modelo, ctx.escalador,
                                                    ctx.columnas, umbral)
    if resultados.empty:
        st.error("Ninguna fila tenía las 30 variables en formato numérico.")
        return
    if descartadas:
        st.warning(f"Se descartaron {descartadas} filas con valores vacíos o no numéricos.")

    n_mal = int((resultados["Diagnóstico predicho"] == "Maligno").sum())
    k = st.columns(4)
    k[0].markdown(kpi(f"{len(resultados)}", "Pacientes evaluados", CYAN), unsafe_allow_html=True)
    k[1].markdown(kpi(f"{n_mal}", "Predichos malignos", PINK), unsafe_allow_html=True)
    k[2].markdown(kpi(f"{len(resultados)-n_mal}", "Predichos benignos", GREEN),
                  unsafe_allow_html=True)
    k[3].markdown(kpi(ctx.modelo_sel, "Modelo usado", VIOLET), unsafe_allow_html=True)

    st.write("")
    izq, der = st.columns([3, 2])
    with izq:
        tabla(resultados)
    with der:
        fig = px.histogram(resultados, x="Probabilidad de maligno (%)", nbins=20,
                           color="Diagnóstico predicho", color_discrete_map=MAPA_CLASE)
        fig.add_vline(x=umbral * 100, line=dict(color=AMBER, dash="dash", width=2))
        fig.update_layout(yaxis_title="Pacientes")
        mostrar(estilo(fig, alto=330, titulo="DISTRIBUCIÓN DE PROBABILIDADES"))

    if "Diagnóstico real" in resultados.columns:
        aciertos = int((resultados["Diagnóstico real"] ==
                        resultados["Diagnóstico predicho"]).sum())
        panel(f"El archivo traía el diagnóstico real: el modelo acertó en "
              f"<b>{aciertos} de {len(resultados)}</b> casos con el umbral actual. "
              "Mueve el umbral y mira cómo cambia.")

    st.download_button("⬇ Descargar resultados",
                       resultados.to_csv(index=False).encode("utf-8"),
                       "resultados_prediccion.csv", "text/csv")
    st.caption("⚠ Herramienta académica. No reemplaza el diagnóstico médico profesional.")


# ═══════════════════════════════════════════════════════════════════════════
# PUNTO DE ENTRADA
# ═══════════════════════════════════════════════════════════════════════════
def render(ctx):
    seccion("LABORATORIO · ¿QUÉ PASA REALMENTE CUANDO UN MODELO «APRENDE»?")
    historia(
        "Entrenar un modelo suena a magia, pero no lo es. Es un proceso con pasos concretos "
        "que se pueden <b>ver</b>, con datos que se pueden <b>seguir</b> y con decisiones "
        "que se pueden <b>auditar</b>. Recorre las siete secciones en orden: al final vas a "
        "poder explicar, número por número, por qué el modelo dijo lo que dijo.",
        CYAN)

    pasos = [("A", "Reparto de datos"), ("B", "Entrenamiento paso a paso"),
             ("C", "Validación cruzada"), ("D", "Cuántos datos hacen falta"),
             ("E", "Sobreajuste"), ("F", "Traza de una decisión"),
             ("G", "Evaluar pacientes nuevos")]
    st.markdown('<div class="flow">' + "".join(
        f'<div class="step"><b>{a}</b><span>{b}</span></div>' for a, b in pasos) + "</div>",
        unsafe_allow_html=True)

    _reparto()
    _paso_a_paso()
    _cruzada()
    _cuantos_datos()
    _sobreajuste()
    _traza(ctx)
    _evaluar(ctx)
