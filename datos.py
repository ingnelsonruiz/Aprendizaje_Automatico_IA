"""Carga del dataset y entrenamiento de los modelos de producción.

Este módulo es la única fuente de verdad de los datos. Las vistas reciben un
objeto `Contexto` y nunca vuelven a cargar ni a entrenar por su cuenta.
"""

import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, auc, f1_score, precision_score,
                             recall_score, roc_curve)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from ucimlrepo import fetch_ucirepo

from config import BASE_ES, SEMILLA, SUFIJO_ES, TEST_SIZE, UCI_ID


# ── Traducción de nombres de columnas ────────────────────────────────────────
def nombre_es(col):
    """`radius1` -> `Radio (media)` · `texture3` -> `Textura (peor)`."""
    m = re.match(r"^(.*?)(\d)$", col)
    if not m:
        return col
    base, suf = m.groups()
    return f"{BASE_ES.get(base, base)} ({SUFIJO_ES.get(suf, suf)})"


def base_de(col):
    """`Radio (media)` -> `Radio`."""
    return col.split(" (")[0]


# ── Carga ────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def cargar_datos():
    """Descarga el dataset de UCI y devuelve (X con nombres en español, y binaria)."""
    ds = fetch_ucirepo(id=UCI_ID)
    X = ds.data.features.copy()
    X.columns = [nombre_es(c) for c in X.columns]
    y = ds.data.targets.iloc[:, 0].map({"M": 1, "B": 0}).astype(int)
    return X, y


@st.cache_data(show_spinner=False)
def particionar():
    """Divide en entrenamiento y prueba de forma estratificada y reproducible."""
    X, y = cargar_datos()
    return train_test_split(X, y, test_size=TEST_SIZE, random_state=SEMILLA, stratify=y)


# ── Entrenamiento ────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def entrenar():
    """Entrena los tres modelos principales y devuelve resultados y escalador."""
    Xtr, Xte, ytr, yte = particionar()
    sc = StandardScaler().fit(Xtr)
    Xtr_s, Xte_s = sc.transform(Xtr), sc.transform(Xte)

    modelos = {
        "Regresión Logística": LogisticRegression(random_state=SEMILLA, max_iter=1000),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=SEMILLA),
        "SVM (RBF)": SVC(kernel="rbf", probability=True, random_state=SEMILLA),
    }
    resultados = {}
    for nombre, m in modelos.items():
        m.fit(Xtr_s, ytr)
        pred = m.predict(Xte_s)
        proba = m.predict_proba(Xte_s)[:, 1]
        fpr, tpr, _ = roc_curve(yte, proba)
        resultados[nombre] = dict(
            modelo=m, pred=pred, proba=proba, fpr=fpr, tpr=tpr, auc=auc(fpr, tpr),
            Accuracy=accuracy_score(yte, pred),
            Precision=precision_score(yte, pred),
            Recall=recall_score(yte, pred),
            F1=f1_score(yte, pred),
        )
    # El orden de columnas es crítico: el escalador valida nombre y posición.
    return resultados, sc, yte.to_numpy(), len(Xtr), len(Xte), list(Xtr.columns)


# ── Contexto compartido ──────────────────────────────────────────────────────
@dataclass
class Contexto:
    """Todo lo que una vista necesita. Se construye una vez en `app.py`."""
    X: pd.DataFrame
    y: pd.Series
    etiqueta: pd.Series
    resultados: dict
    escalador: object
    y_prueba: np.ndarray
    n_train: int
    n_test: int
    columnas: list
    modelo_sel: str = "Random Forest"

    @property
    def activo(self):
        """Diccionario de resultados del modelo seleccionado en la barra lateral."""
        return self.resultados[self.modelo_sel]

    @property
    def n_benignos(self):
        return int((self.y == 0).sum())

    @property
    def n_malignos(self):
        return int((self.y == 1).sum())

    def columnas_media(self):
        """Las 10 variables «media», en el orden con que se entrenó."""
        return [c for c in self.columnas if c.endswith("(media)")]

    def fila_para_predecir(self, valores):
        """Arma una fila respetando el orden exacto de columnas del entrenamiento.

        Es lo que evita el error de validación de nombres de scikit-learn.
        """
        datos = [[valores.get(c, float(self.X[c].mean())) for c in self.columnas]]
        return pd.DataFrame(datos, columns=self.columnas)

    def probabilidad_maligno(self, valores):
        """Devuelve la probabilidad de malignidad para un conjunto de valores."""
        fila = self.fila_para_predecir(valores)
        return float(self.activo["modelo"].predict_proba(self.escalador.transform(fila))[0, 1])


def construir_contexto(modelo_sel="Random Forest"):
    """Punto de entrada único: carga, entrena y empaqueta todo en un `Contexto`."""
    X, y = cargar_datos()
    resultados, sc, y_prueba, n_tr, n_te, columnas = entrenar()
    return Contexto(
        X=X, y=y,
        etiqueta=y.map({0: "Benigno", 1: "Maligno"}),
        resultados=resultados, escalador=sc, y_prueba=y_prueba,
        n_train=n_tr, n_test=n_te, columnas=columnas, modelo_sel=modelo_sel,
    )
