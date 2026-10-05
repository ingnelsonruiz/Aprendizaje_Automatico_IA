"""Motor del laboratorio: solo cálculo, sin interfaz.

Separar el cálculo del dibujo permite agregar experimentos nuevos aquí y
consumirlos desde `vista_laboratorio.py` sin tocar nada más.

Bloques:
  1. Partición didáctica (entrenamiento / validación / prueba)
  2. Entrenamiento progresivo: cada modelo avanza a su manera
  3. Validación cruzada (K-fold)
  4. Curva de aprendizaje por cantidad de datos
  5. Sobreajuste
  6. Trazas: cómo un modelo llega a una decisión concreta
  7. Evaluación de pacientes cargados por el usuario
"""

import warnings

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import accuracy_score, log_loss, recall_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from config import SEMILLA
from datos import nombre_es, particionar

RESOLUCION = 70          # rejilla de las fronteras; subirla cuesta memoria
MARGEN = 0.6


# ═══════════════════════════════════════════════════════════════════════════
# 1 · PARTICIÓN DIDÁCTICA
# ═══════════════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner=False)
def particion_triple():
    """Divide en entrenamiento / validación / prueba.

    El conjunto de prueba es el mismo que usa la app; el de validación se
    recorta del de entrenamiento, que es como se hace en la práctica.
    """
    Xtr_full, Xte, ytr_full, yte = particionar()
    Xtr, Xval, ytr, yval = train_test_split(
        Xtr_full, ytr_full, test_size=0.25, random_state=SEMILLA, stratify=ytr_full)
    return dict(Xtr=Xtr, ytr=ytr, Xval=Xval, yval=yval, Xte=Xte, yte=yte,
                n_total=len(Xtr_full) + len(Xte))


@st.cache_resource(show_spinner=False)
def proyeccion_2d():
    """Proyecta a 2 componentes principales para poder dibujar las fronteras.

    El PCA se ajusta SOLO con el entrenamiento: validación y prueba se
    transforman con esa misma proyección, nunca al revés.
    """
    p = particion_triple()
    sc = StandardScaler().fit(p["Xtr"])
    pca = PCA(n_components=2, random_state=SEMILLA).fit(sc.transform(p["Xtr"]))
    return dict(
        Ztr=pca.transform(sc.transform(p["Xtr"])), ytr=p["ytr"].to_numpy(),
        Zval=pca.transform(sc.transform(p["Xval"])), yval=p["yval"].to_numpy(),
        Zte=pca.transform(sc.transform(p["Xte"])), yte=p["yte"].to_numpy(),
        escalador=sc, pca=pca,
    )


# ═══════════════════════════════════════════════════════════════════════════
# 2 · ENTRENAMIENTO PROGRESIVO
# ═══════════════════════════════════════════════════════════════════════════
# Cada algoritmo «avanza» de una manera distinta. Aquí se declara qué
# significa un paso para cada uno, y por eso sí se puede elegir el modelo.
MODELOS_LAB = {
    "Regresión Logística": dict(
        unidad="época",
        avanza="Cada paso es una pasada completa por los datos. El modelo mide su error y "
               "corrige sus pesos un poco: eso es el descenso de gradiente.",
        pasos=list(range(1, 61)),
        tiene_perdida=True,
        nota="Es el único que aprende de verdad «poco a poco». Los demás se reconstruyen "
             "enteros en cada paso.",
    ),
    "Árbol de decisión": dict(
        unidad="nivel de profundidad",
        avanza="Cada paso le permite al árbol hacer un corte más. Con profundidad 1 solo "
               "puede partir los datos en dos; con 10 puede dibujar un mosaico fino.",
        pasos=list(range(1, 16)),
        tiene_perdida=False,
        nota="El árbol no «itera»: se reconstruye completo cada vez, pero con más permiso "
             "para ramificarse.",
    ),
    "Random Forest": dict(
        unidad="árboles en el bosque",
        avanza="Cada paso agrega un árbol más. Todos votan y el promedio de sus votos "
               "suaviza la frontera.",
        pasos=[1, 2, 3, 5, 8, 12, 18, 25, 35, 50, 70, 100],
        tiene_perdida=False,
        nota="Aquí el progreso no es aprender mejor, sino promediar más opiniones.",
    ),
    "SVM (RBF)": dict(
        unidad="iteraciones del optimizador",
        avanza="Cada paso le da más iteraciones al optimizador para colocar la frontera lo "
               "más lejos posible de ambos grupos.",
        pasos=[1, 2, 5, 10, 25, 50, 100, 250, 500, 1000, 3000, -1],
        tiene_perdida=False,
        nota="El último valor, -1, significa «sin límite»: el optimizador corre hasta "
             "converger.",
    ),
}


def _rejilla(Z):
    xs = np.linspace(Z[:, 0].min() - MARGEN, Z[:, 0].max() + MARGEN, RESOLUCION)
    ys = np.linspace(Z[:, 1].min() - MARGEN, Z[:, 1].max() + MARGEN, RESOLUCION)
    gx, gy = np.meshgrid(xs, ys)
    return xs, ys, np.c_[gx.ravel(), gy.ravel()], gx.shape


def _superficie(modelo, puntos, forma):
    """Probabilidad de malignidad en cada punto de la rejilla."""
    if hasattr(modelo, "predict_proba"):
        z = modelo.predict_proba(puntos)[:, 1]
    else:                                   # SVM sin calibrar: distancia al margen
        z = 1 / (1 + np.exp(-modelo.decision_function(puntos)))
    return z.reshape(forma)


@st.cache_resource(show_spinner=False)
def entrenamiento_progresivo(nombre):
    """Entrena el modelo elegido paso a paso y guarda la traza de cada paso.

    Devuelve la geometría (rejilla y nubes de puntos) y una lista `historial`
    con, por cada paso: la superficie de probabilidad, los aciertos en
    entrenamiento y validación, y la pérdida cuando el modelo la expone.
    """
    g = proyeccion_2d()
    Ztr, ytr, Zval, yval = g["Ztr"], g["ytr"], g["Zval"], g["yval"]
    xs, ys, puntos, forma = _rejilla(Ztr)
    cfg = MODELOS_LAB[nombre]

    historial, sgd = [], None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")     # el SVM avisa cuando corta por max_iter
        for paso in cfg["pasos"]:
            if nombre == "Regresión Logística":
                if sgd is None:
                    sgd = SGDClassifier(loss="log_loss", learning_rate="constant",
                                        eta0=0.03, random_state=SEMILLA, penalty=None)
                sgd.partial_fit(Ztr, ytr, classes=np.array([0, 1]))
                modelo = sgd
            elif nombre == "Árbol de decisión":
                modelo = DecisionTreeClassifier(max_depth=paso,
                                                random_state=SEMILLA).fit(Ztr, ytr)
            elif nombre == "Random Forest":
                modelo = RandomForestClassifier(n_estimators=paso,
                                                random_state=SEMILLA).fit(Ztr, ytr)
            else:
                modelo = SVC(kernel="rbf", max_iter=paso,
                             random_state=SEMILLA).fit(Ztr, ytr)

            perdida = None
            if cfg["tiene_perdida"]:
                p = np.clip(modelo.predict_proba(Ztr)[:, 1], 1e-9, 1 - 1e-9)
                perdida = log_loss(ytr, p)

            historial.append(dict(
                paso=paso,
                zz=_superficie(modelo, puntos, forma).astype(np.float32),
                acc_tr=accuracy_score(ytr, modelo.predict(Ztr)),
                acc_val=accuracy_score(yval, modelo.predict(Zval)),
                rec_val=recall_score(yval, modelo.predict(Zval), zero_division=0),
                perdida=perdida,
            ))

    return dict(xs=xs, ys=ys, Ztr=Ztr, ytr=ytr, Zval=Zval, yval=yval,
                historial=historial, cfg=cfg)


# ═══════════════════════════════════════════════════════════════════════════
# 3 · VALIDACIÓN CRUZADA
# ═══════════════════════════════════════════════════════════════════════════
def _modelo_completo(nombre):
    """Instancia el algoritmo con sus 30 variables (no la versión 2D)."""
    if nombre == "Árbol de decisión":
        return DecisionTreeClassifier(max_depth=5, random_state=SEMILLA)
    if nombre == "Random Forest":
        return RandomForestClassifier(n_estimators=200, random_state=SEMILLA)
    if nombre == "SVM (RBF)":
        return SVC(kernel="rbf", probability=True, random_state=SEMILLA)
    return LogisticRegression(max_iter=1000, random_state=SEMILLA)


@st.cache_resource(show_spinner=False)
def validacion_cruzada(nombre, k=5):
    """K-fold sobre el entrenamiento: cada pliegue se usa una vez como validación."""
    p = particion_triple()
    X = pd.concat([p["Xtr"], p["Xval"]])
    y = pd.concat([p["ytr"], p["yval"]])
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=SEMILLA)

    pliegues = []
    for i, (idx_tr, idx_val) in enumerate(skf.split(X, y), start=1):
        sc = StandardScaler().fit(X.iloc[idx_tr])
        modelo = _modelo_completo(nombre).fit(sc.transform(X.iloc[idx_tr]), y.iloc[idx_tr])
        pred = modelo.predict(sc.transform(X.iloc[idx_val]))
        pliegues.append(dict(
            pliegue=i, n_train=len(idx_tr), n_val=len(idx_val),
            acc=accuracy_score(y.iloc[idx_val], pred),
            rec=recall_score(y.iloc[idx_val], pred, zero_division=0),
        ))
    return pliegues, len(X)


# ═══════════════════════════════════════════════════════════════════════════
# 4 · CURVA DE APRENDIZAJE POR CANTIDAD DE DATOS
# ═══════════════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner=False)
def curva_aprendizaje(nombre, fracciones=(0.1, 0.2, 0.3, 0.45, 0.6, 0.8, 1.0)):
    """¿Cuánto mejora el modelo si le damos más pacientes para entrenar?"""
    p = particion_triple()
    Xtr, ytr, Xval, yval = p["Xtr"], p["ytr"], p["Xval"], p["yval"]
    sc = StandardScaler().fit(Xtr)
    Xtr_s, Xval_s = sc.transform(Xtr), sc.transform(Xval)

    filas = []
    for f in fracciones:
        n = max(20, int(len(Xtr_s) * f))
        if n >= len(Xtr_s):
            idx = np.arange(len(Xtr_s))
        else:
            idx, _ = train_test_split(np.arange(len(Xtr_s)), train_size=n,
                                      random_state=SEMILLA, stratify=ytr)
        modelo = _modelo_completo(nombre).fit(Xtr_s[idx], ytr.iloc[idx])
        filas.append(dict(
            n=len(idx),
            acc_tr=accuracy_score(ytr.iloc[idx], modelo.predict(Xtr_s[idx])),
            acc_val=accuracy_score(yval, modelo.predict(Xval_s)),
        ))
    return filas


# ═══════════════════════════════════════════════════════════════════════════
# 5 · SOBREAJUSTE
# ═══════════════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner=False)
def curva_sobreajuste(profundidad_maxima=20):
    """Acierto en entrenamiento y validación según la profundidad del árbol."""
    p = particion_triple()
    sc = StandardScaler().fit(p["Xtr"])
    Xtr_s, Xval_s = sc.transform(p["Xtr"]), sc.transform(p["Xval"])

    profundidades, acc_tr, acc_val = list(range(1, profundidad_maxima + 1)), [], []
    for d in profundidades:
        m = DecisionTreeClassifier(max_depth=d, random_state=SEMILLA).fit(Xtr_s, p["ytr"])
        acc_tr.append(accuracy_score(p["ytr"], m.predict(Xtr_s)))
        acc_val.append(accuracy_score(p["yval"], m.predict(Xval_s)))
    return profundidades, acc_tr, acc_val


# ═══════════════════════════════════════════════════════════════════════════
# 6 · TRAZAS: CÓMO SE LLEGA A UNA DECISIÓN
# ═══════════════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner=False)
def arbol_interpretable(profundidad=4):
    """Un árbol pequeño, con las 30 variables, para poder leer sus reglas."""
    p = particion_triple()
    sc = StandardScaler().fit(p["Xtr"])
    arbol = DecisionTreeClassifier(max_depth=profundidad, random_state=SEMILLA)
    arbol.fit(sc.transform(p["Xtr"]), p["ytr"])
    return arbol, sc, list(p["Xtr"].columns)


def traza_escalado(fila, escalador, columnas):
    """Paso 1 de cualquier traza: valor original -> valor estandarizado."""
    z = escalador.transform(fila)[0]
    return pd.DataFrame({
        "Variable": columnas,
        "Valor del paciente": fila.iloc[0].to_numpy(),
        "Promedio del entrenamiento": escalador.mean_,
        "Desviación estándar": escalador.scale_,
        "Valor estandarizado (z)": z,
    }), z


def traza_logistica(modelo, z, columnas):
    """Aporte de cada variable a la decisión de una regresión logística."""
    w = modelo.coef_[0]
    aportes = w * z
    sesgo = float(modelo.intercept_[0])
    total = float(aportes.sum()) + sesgo
    df = pd.DataFrame({
        "Variable": columnas,
        "Peso aprendido (w)": w.round(4),
        "Valor estandarizado (z)": z.round(4),
        "Aporte (w × z)": aportes.round(4),
    })
    df = df.reindex(df["Aporte (w × z)"].abs().sort_values(ascending=False).index)
    return df, total, sesgo, float(1 / (1 + np.exp(-total)))


def traza_arbol(arbol, z, columnas, escalador):
    """Reglas que el árbol fue aplicando, expresadas en unidades originales."""
    t = arbol.tree_
    nodo, reglas = 0, []
    while t.children_left[nodo] != -1:
        j = int(t.feature[nodo])
        umbral_z = t.threshold[nodo]
        umbral_real = umbral_z * escalador.scale_[j] + escalador.mean_[j]
        valor_real = z[j] * escalador.scale_[j] + escalador.mean_[j]
        izquierda = z[j] <= umbral_z
        reglas.append({
            "Paso": len(reglas) + 1,
            "Pregunta del árbol": f"¿{columnas[j]} ≤ {umbral_real:.3f}?",
            "Valor del paciente": round(float(valor_real), 3),
            "Respuesta": "Sí" if izquierda else "No",
            "Sigue por": "Rama izquierda" if izquierda else "Rama derecha",
        })
        nodo = t.children_left[nodo] if izquierda else t.children_right[nodo]

    conteo = t.value[nodo][0]
    total = float(conteo.sum())
    # value puede venir como proporción o como conteo según la versión de scikit-learn
    proporcion = float(conteo[1] / total) if total else 0.0
    n_nodo = int(t.n_node_samples[nodo])
    return pd.DataFrame(reglas), dict(
        n_hoja=n_nodo,
        benignos=int(round((1 - proporcion) * n_nodo)),
        malignos=int(round(proporcion * n_nodo)),
        probabilidad=proporcion,
    )


def traza_bosque(bosque, fila_escalada):
    """Voto de cada árbol del bosque para un paciente."""
    votos = np.array([int(a.predict(fila_escalada)[0]) for a in bosque.estimators_])
    return dict(total=len(votos), malignos=int(votos.sum()),
                benignos=int((votos == 0).sum()), probabilidad=float(votos.mean()),
                votos=votos)


def traza_svm(modelo, fila_escalada):
    """Distancia al margen de un SVM."""
    d = float(modelo.decision_function(fila_escalada)[0])
    return dict(distancia=d, lado="Maligno" if d > 0 else "Benigno", confianza=abs(d))


# ═══════════════════════════════════════════════════════════════════════════
# 7 · EVALUACIÓN DE PACIENTES CARGADOS
# ═══════════════════════════════════════════════════════════════════════════
def plantilla_csv(columnas, X):
    """CSV de ejemplo con los promedios del dataset, para que el usuario lo edite."""
    fila = {c: round(float(X[c].mean()), 4) for c in columnas}
    df = pd.DataFrame([fila, fila])
    df.insert(0, "id_paciente", ["PACIENTE-001", "PACIENTE-002"])
    return df


def normalizar_columnas(df, columnas_esperadas):
    """Acepta nombres en español o los originales en inglés de UCI.

    Devuelve (DataFrame renombrado, faltantes, ignoradas).
    """
    df = df.rename(columns={c: nombre_es(c) for c in df.columns})
    faltantes = [c for c in columnas_esperadas if c not in df.columns]
    ignoradas = [c for c in df.columns if c not in columnas_esperadas]
    return df, faltantes, ignoradas


def evaluar_pacientes(df, modelo, escalador, columnas, umbral=0.5):
    """Predice sobre un lote de pacientes y devuelve (tabla, filas descartadas)."""
    X = df[columnas].apply(pd.to_numeric, errors="coerce")
    validas = X.notna().all(axis=1)
    descartadas = int((~validas).sum())
    X = X[validas]
    if X.empty:
        return pd.DataFrame(), descartadas

    proba = modelo.predict_proba(escalador.transform(X))[:, 1]
    salida = pd.DataFrame({
        "Probabilidad de maligno (%)": (proba * 100).round(1),
        "Diagnóstico predicho": np.where(proba >= umbral, "Maligno", "Benigno"),
    }, index=X.index)

    for col in ("id_paciente", "id", "Diagnóstico"):
        if col in df.columns:
            salida.insert(0, col if col != "Diagnóstico" else "Diagnóstico real",
                          df.loc[X.index, col].to_numpy())
    if salida.shape[1] == 2:
        salida.insert(0, "Fila del archivo", X.index + 2)
    return salida, descartadas


@st.cache_data(show_spinner=False)
def muestra_de_prueba(n=8):
    """Pacientes reales del conjunto de prueba, con su diagnóstico confirmado."""
    p = particion_triple()
    Xte, yte = p["Xte"], p["yte"]
    idx = list(yte[yte == 0].index[: n // 2]) + list(yte[yte == 1].index[: n - n // 2])
    muestra = Xte.loc[idx].copy()
    muestra.insert(0, "id_paciente", [f"REAL-{i:03d}" for i in range(1, len(idx) + 1)])
    muestra.insert(1, "Diagnóstico", yte.loc[idx].map({0: "Benigno", 1: "Maligno"}).to_numpy())
    return muestra.reset_index(drop=True)


# ═══════════════════════════════════════════════════════════════════════════
# 8 · ENTRENAMIENTO EN VIVO
# ═══════════════════════════════════════════════════════════════════════════
# A diferencia del bloque 2, aquí NO se usa caché: el modelo se entrena de
# verdad mientras el estudiante mira. `pasos_en_vivo` es un generador: entrega
# un «fotograma» por paso, con lo que el modelo acaba de hacer contado en
# lenguaje sencillo. La vista decide si lo reproduce solo o paso a paso.
from sklearn.neural_network import MLPClassifier  # noqa: E402

MODELOS_VIVO = {
    "Regresión Logística": dict(
        unidad="Época", pasos_def=40, pasos_max=150, usa_tasa=True, tasa_def=0.001,
        idea="Busca una <b>línea recta</b> que separe benignos de malignos. Empieza con "
             "pesos casi en cero y, en cada época, mide cuánto se equivoca (la "
             "<b>pérdida</b>) y empuja los pesos en la dirección que reduce ese error: "
             "eso es el <b>descenso de gradiente</b>.",
        mirar="Mira cómo la línea amarilla gira y se desplaza hasta acomodarse, y cómo la "
              "curva de pérdida baja rápido al principio y luego se aplana.",
    ),
    "Red neuronal (MLP)": dict(
        unidad="Época", pasos_def=60, pasos_max=200, usa_tasa=True, tasa_def=0.01,
        idea="Una red de <b>8 neuronas ocultas</b>. Cada neurona aprende una línea propia y "
             "la red las combina, así que la frontera puede <b>curvarse</b>. Aprende igual "
             "que la logística: época a época, con descenso de gradiente "
             "(retropropagación).",
        mirar="Al inicio la frontera es casi recta; con las épocas aparecen curvas. Si la "
              "tasa de aprendizaje es muy alta, la pérdida «salta» en vez de bajar.",
    ),
    "Árbol de decisión": dict(
        unidad="Profundidad", pasos_def=10, pasos_max=15, usa_tasa=False,
        idea="Hace <b>preguntas de sí/no</b> sobre una variable a la vez "
             "(«¿Componente 1 ≤ 0,8?»). Cada nivel de profundidad le permite hacer una "
             "pregunta más en cada rama, y eso parte el plano en rectángulos.",
        mirar="La frontera solo tiene cortes horizontales y verticales. Fíjate en la bitácora: "
              "te dice qué pregunta nueva aprendió en cada nivel. Con mucha profundidad "
              "el acierto de entrenamiento llega a 100 % pero el de validación no: memoriza.",
    ),
    "Random Forest": dict(
        unidad="Árbol", pasos_def=30, pasos_max=100, usa_tasa=False,
        idea="Entrena <b>muchos árboles</b>, cada uno con una muestra distinta de pacientes, "
             "y los pone a <b>votar</b>. Un árbol solo es inestable; el promedio de muchos "
             "es firme.",
        mirar="Con 1 árbol la frontera es tosca; a medida que se suman árboles se suaviza. "
              "La bitácora compara el acierto del árbol nuevo (solo) contra el del bosque "
              "completo: el bosque casi siempre gana.",
    ),
    "SVM (RBF)": dict(
        unidad="Iteración", pasos_def=12, pasos_max=12, usa_tasa=False,
        idea="Busca la frontera que deja el <b>mayor margen</b> posible entre los dos grupos. "
             "Solo le importan los pacientes cercanos a la frontera: los "
             "<b>vectores de soporte</b>. El kernel RBF le permite curvarse.",
        mirar="Con pocas iteraciones el optimizador no ha terminado y la frontera es mala. "
              "Mira cómo baja el número de vectores de soporte cuando la solución se afina.",
    ),
}
_ITER_SVM = [1, 2, 3, 5, 8, 12, 20, 35, 60, 120, 400, -1]


def _desc_logistica(m, paso, perdida, previo):
    w1, w2 = (float(v) for v in m.coef_[0])
    b = float(m.intercept_[0])
    txt = (f"Recorrió los pacientes de entrenamiento, calculó la pérdida "
           f"(<b>{perdida:.4f}</b>) y ajustó sus pesos.<br>"
           f"Ecuación actual: <code>z = {w1:+.3f}·C1 {w2:+.3f}·C2 {b:+.3f}</code>")
    if previo is not None:
        txt += (f"<br>Cambio en esta época: Δw1 = {w1 - previo[0]:+.4f}, "
                f"Δw2 = {w2 - previo[1]:+.4f}, Δb = {b - previo[2]:+.4f}")
    return txt, (w1, w2, b)


def _desc_arbol(m, profundidad):
    t = m.tree_
    prof = np.zeros(t.node_count, dtype=int)
    for n in range(t.node_count):
        for h in (t.children_left[n], t.children_right[n]):
            if h != -1:
                prof[h] = prof[n] + 1
    nuevas = [n for n in range(t.node_count)
              if prof[n] == profundidad - 1 and t.children_left[n] != -1]
    hojas = int((t.children_left == -1).sum())
    if not nuevas:
        return (f"No encontró ninguna pregunta nueva que mejore: todas las ramas ya son "
                f"puras o tienen muy pocos pacientes. El árbol dejó de crecer "
                f"(tiene {hojas} hojas).")
    preguntas = "".join(
        f"<li>¿Componente {int(t.feature[n]) + 1} ≤ {t.threshold[n]:.3f}? "
        f"(decide sobre {int(t.n_node_samples[n])} pacientes)</li>" for n in nuevas[:6])
    extra = f"<li>… y {len(nuevas) - 6} preguntas más</li>" if len(nuevas) > 6 else ""
    return (f"Aprendió <b>{len(nuevas)} pregunta(s) nueva(s)</b> en el nivel {profundidad}:"
            f"<ul>{preguntas}{extra}</ul>Ahora el plano está dividido en <b>{hojas} "
            f"regiones</b> (hojas).")


def pasos_en_vivo(nombre, n_pasos, tasa=0.03):
    """Entrena `nombre` paso a paso y entrega un fotograma por paso.

    Cada fotograma trae: superficie de probabilidad, métricas en entrenamiento y
    validación, pérdida (si aplica), máscara de pacientes mal clasificados en el
    entrenamiento y un relato en HTML de lo que el modelo hizo en ese paso.
    """
    g = proyeccion_2d()
    Ztr, ytr, Zval, yval = g["Ztr"], g["ytr"], g["Zval"], g["yval"]
    xs, ys, puntos, forma = _rejilla(Ztr)
    clases = np.array([0, 1])

    if nombre == "SVM (RBF)":
        secuencia = _ITER_SVM[:n_pasos]
    else:
        secuencia = list(range(1, n_pasos + 1))

    modelo, previo = None, None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for i, paso in enumerate(secuencia, start=1):
            perdida = None
            if nombre == "Regresión Logística":
                if modelo is None:
                    modelo = SGDClassifier(loss="log_loss", learning_rate="constant",
                                           eta0=tasa, random_state=SEMILLA, penalty=None)
                modelo.partial_fit(Ztr, ytr, classes=clases)
                perdida = log_loss(ytr, np.clip(modelo.predict_proba(Ztr)[:, 1], 1e-9, 1 - 1e-9))
                relato, previo = _desc_logistica(modelo, paso, perdida, previo)
                etiqueta = f"Época {paso}"
            elif nombre == "Red neuronal (MLP)":
                if modelo is None:
                    modelo = MLPClassifier(hidden_layer_sizes=(8,), learning_rate_init=tasa,
                                           random_state=SEMILLA)
                modelo.partial_fit(Ztr, ytr, classes=clases)
                perdida = log_loss(ytr, np.clip(modelo.predict_proba(Ztr)[:, 1], 1e-9, 1 - 1e-9))
                delta = "" if previo is None else f" (cambio {perdida - previo:+.4f})"
                relato = (f"Pasó los datos hacia adelante por las 8 neuronas, midió el error "
                          f"(pérdida <b>{perdida:.4f}</b>{delta}) y lo propagó hacia atrás "
                          f"para corregir los {modelo.coefs_[0].size + modelo.coefs_[1].size} "
                          f"pesos de la red.")
                previo = perdida
                etiqueta = f"Época {paso}"
            elif nombre == "Árbol de decisión":
                modelo = DecisionTreeClassifier(max_depth=paso, random_state=SEMILLA).fit(Ztr, ytr)
                relato = _desc_arbol(modelo, paso)
                etiqueta = f"Profundidad {paso}"
            elif nombre == "Random Forest":
                if modelo is None:
                    modelo = RandomForestClassifier(n_estimators=1, warm_start=True,
                                                    random_state=SEMILLA)
                else:
                    modelo.set_params(n_estimators=paso)
                modelo.fit(Ztr, ytr)
                solo = accuracy_score(yval, modelo.estimators_[-1].predict(Zval).astype(int))
                bosque = accuracy_score(yval, modelo.predict(Zval))
                relato = (f"Plantó el árbol #{paso} con una muestra al azar de pacientes. "
                          f"Ese árbol <b>solo</b> acierta {solo*100:.1f} % en validación; "
                          f"el <b>bosque completo votando</b> ({paso} árboles) acierta "
                          f"{bosque*100:.1f} %.")
                etiqueta = f"{paso} árbol(es)"
            else:
                modelo = SVC(kernel="rbf", max_iter=paso, random_state=SEMILLA).fit(Ztr, ytr)
                n_sv = int(modelo.n_support_.sum())
                tope = "sin límite (hasta converger)" if paso == -1 else f"máximo {paso}"
                relato = (f"Se le permitió al optimizador {tope} iteración(es). La frontera "
                          f"quedó apoyada en <b>{n_sv} vectores de soporte</b> "
                          f"({n_sv/len(Ztr)*100:.0f} % de los pacientes de entrenamiento).")
                if paso == -1:
                    relato += " El optimizador convergió: esta es la solución final."
                etiqueta = "Sin límite" if paso == -1 else f"{paso} iteraciones"

            pred_tr = modelo.predict(Ztr)
            pred_val = modelo.predict(Zval)
            yield dict(
                i=i, total=len(secuencia), etiqueta=etiqueta,
                zz=_superficie(modelo, puntos, forma).astype(np.float32),
                acc_tr=accuracy_score(ytr, pred_tr),
                acc_val=accuracy_score(yval, pred_val),
                rec_val=recall_score(yval, pred_val, zero_division=0),
                perdida=perdida, errores=pred_tr != ytr, relato=relato,
                modelo=modelo, xs=xs, ys=ys,
            )


def evaluar_en_prueba(modelo):
    """Examen final del modelo entrenado en vivo sobre el conjunto de prueba (2D)."""
    g = proyeccion_2d()
    pred = modelo.predict(g["Zte"])
    yte = g["yte"]
    return dict(
        acc=accuracy_score(yte, pred),
        rec=recall_score(yte, pred, zero_division=0),
        vp=int(((pred == 1) & (yte == 1)).sum()), fn=int(((pred == 0) & (yte == 1)).sum()),
        vn=int(((pred == 0) & (yte == 0)).sum()), fp=int(((pred == 1) & (yte == 0)).sum()),
    )
