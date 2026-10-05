"""Experimentos didácticos del laboratorio: solo cálculo, sin interfaz.

Separar el cálculo del dibujo permite agregar experimentos nuevos aquí y
consumirlos desde `vista_laboratorio.py` sin tocar nada más.
"""

import numpy as np
import streamlit as st
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import accuracy_score, log_loss
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from config import SEMILLA
from datos import particionar

# Algoritmos disponibles en el experimento de fronteras.
# Para agregar uno nuevo basta con añadir una entrada aquí y su constructor
# en `_construir_modelo`.
ALGORITMOS = {
    "Regresión Logística": dict(
        parametro="Parámetro C",
        opciones=[0.001, 0.01, 0.1, 1.0, 10.0], defecto=1.0, tipo="select",
        ayuda="C regula qué tanto se le permite al modelo ajustarse a los datos."),
    "Árbol de decisión": dict(
        parametro="Profundidad máxima",
        minimo=1, maximo=15, defecto=3, tipo="slider",
        ayuda="Más profundidad = cortes más finos. Demasiada = memoriza el ruido."),
    "Random Forest": dict(
        parametro="Número de árboles",
        minimo=1, maximo=150, defecto=50, tipo="slider",
        ayuda="Cada árbol vota. Más árboles suavizan la frontera y estabilizan la decisión."),
    "SVM (RBF)": dict(
        parametro="Parámetro C",
        opciones=[0.01, 0.1, 1.0, 10.0, 100.0], defecto=1.0, tipo="select",
        ayuda="C alto = tolera menos errores y curva más la frontera; C bajo = frontera más simple."),
}


def _construir_modelo(nombre, parametro):
    """Devuelve una instancia del algoritmo con el hiperparámetro indicado."""
    if nombre == "Árbol de decisión":
        return DecisionTreeClassifier(max_depth=int(parametro), random_state=SEMILLA)
    if nombre == "Random Forest":
        return RandomForestClassifier(n_estimators=int(parametro), random_state=SEMILLA)
    if nombre == "SVM (RBF)":
        return SVC(kernel="rbf", C=float(parametro), probability=True, random_state=SEMILLA)
    return LogisticRegression(C=float(parametro), max_iter=1000, random_state=SEMILLA)


@st.cache_resource(show_spinner=False)
def proyeccion_2d():
    """Proyecta los datos a 2 componentes principales, para poder dibujarlos."""
    Xtr, Xte, ytr, yte = particionar()
    sc = StandardScaler().fit(Xtr)
    pca = PCA(n_components=2, random_state=SEMILLA).fit(sc.transform(Xtr))
    Ztr = pca.transform(sc.transform(Xtr))
    Zte = pca.transform(sc.transform(Xte))
    return Ztr, ytr.to_numpy(), Zte, yte.to_numpy()


@st.cache_resource(show_spinner=False)
def descenso_gradiente(epocas=60, tasa=0.03):
    """Entrena una regresión logística época por época y guarda el historial.

    Devuelve (Ztr, ytr, historial). Cada entrada del historial trae la pérdida,
    los aciertos y los pesos de esa época, para poder reconstruir la frontera.
    """
    Ztr, ytr, Zte, yte = proyeccion_2d()
    clf = SGDClassifier(loss="log_loss", learning_rate="constant", eta0=tasa,
                        random_state=SEMILLA, penalty=None)
    historial = []
    for e in range(1, epocas + 1):
        clf.partial_fit(Ztr, ytr, classes=np.array([0, 1]))
        p_tr = np.clip(clf.predict_proba(Ztr)[:, 1], 1e-9, 1 - 1e-9)
        historial.append(dict(
            epoca=e,
            perdida=log_loss(ytr, p_tr),
            acc_tr=accuracy_score(ytr, clf.predict(Ztr)),
            acc_te=accuracy_score(yte, clf.predict(Zte)),
            w=clf.coef_[0].copy(),
            b=float(clf.intercept_[0]),
        ))
    return Ztr, ytr, historial


def malla_sigmoide(Z, w, b, resolucion=90, margen=0.6):
    """Rejilla de probabilidades de un modelo lineal, a partir de sus pesos."""
    xs = np.linspace(Z[:, 0].min() - margen, Z[:, 0].max() + margen, resolucion)
    ys = np.linspace(Z[:, 1].min() - margen, Z[:, 1].max() + margen, resolucion)
    gx, gy = np.meshgrid(xs, ys)
    zz = 1 / (1 + np.exp(-(w[0] * gx + w[1] * gy + b)))
    return xs, ys, zz


@st.cache_resource(show_spinner=False)
def frontera(nombre_algoritmo, parametro, resolucion=110, margen=0.6):
    """Entrena un algoritmo en 2D y devuelve su rejilla de probabilidades."""
    Ztr, ytr, Zte, yte = proyeccion_2d()
    modelo = _construir_modelo(nombre_algoritmo, parametro).fit(Ztr, ytr)

    xs = np.linspace(Ztr[:, 0].min() - margen, Ztr[:, 0].max() + margen, resolucion)
    ys = np.linspace(Ztr[:, 1].min() - margen, Ztr[:, 1].max() + margen, resolucion)
    gx, gy = np.meshgrid(xs, ys)
    zz = modelo.predict_proba(np.c_[gx.ravel(), gy.ravel()])[:, 1].reshape(gx.shape)

    return dict(xs=xs, ys=ys, zz=zz, Ztr=Ztr, ytr=ytr, Zte=Zte, yte=yte,
                acc_tr=accuracy_score(ytr, modelo.predict(Ztr)),
                acc_te=accuracy_score(yte, modelo.predict(Zte)))


@st.cache_resource(show_spinner=False)
def curva_sobreajuste(profundidad_maxima=20):
    """Acierto en entrenamiento y prueba según la profundidad del árbol."""
    Xtr, Xte, ytr, yte = particionar()
    sc = StandardScaler().fit(Xtr)
    Xtr_s, Xte_s = sc.transform(Xtr), sc.transform(Xte)

    profundidades, acc_tr, acc_te = list(range(1, profundidad_maxima + 1)), [], []
    for p in profundidades:
        m = DecisionTreeClassifier(max_depth=p, random_state=SEMILLA).fit(Xtr_s, ytr)
        acc_tr.append(accuracy_score(ytr, m.predict(Xtr_s)))
        acc_te.append(accuracy_score(yte, m.predict(Xte_s)))
    return profundidades, acc_tr, acc_te
