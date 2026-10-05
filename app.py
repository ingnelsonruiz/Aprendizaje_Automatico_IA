import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)

# ── CONFIGURACIÓN DE PÁGINA ───────────────────────────────────────────────────
st.set_page_config(
    page_title="Breast Cancer ML - Wisconsin",
    page_icon="🔬",
    layout="wide"
)

st.title("🔬 Machine Learning — Breast Cancer Wisconsin (Diagnostic)")
st.markdown("""
**Dataset:** UCI Machine Learning Repository — ID 17  
**Tarea:** Clasificación binaria — Maligno (M) / Benigno (B)  
**Fuente:** [https://archive.ics.uci.edu/dataset/17](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic)
""")

st.divider()

# ── CARGA DE DATOS (con caché para no recargar) ───────────────────────────────
@st.cache_data
def cargar_datos():
    dataset = fetch_ucirepo(id=17)
    X = dataset.data.features
    y = dataset.data.targets
    return X, y

@st.cache_resource
def entrenar_modelos(X, y):
    y_encoded = (y['Diagnosis'] == 'M').astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    modelos = {
        'Regresión Logística': LogisticRegression(random_state=42, max_iter=1000),
        'Random Forest':       RandomForestClassifier(n_estimators=100, random_state=42),
        'SVM (RBF)':           SVC(kernel='rbf', probability=True, random_state=42)
    }

    resultados = {}
    for nombre, modelo in modelos.items():
        modelo.fit(X_train_sc, y_train)
        y_pred = modelo.predict(X_test_sc)
        resultados[nombre] = {
            'Accuracy':  accuracy_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred),
            'Recall':    recall_score(y_test, y_pred),
            'F1-Score':  f1_score(y_test, y_pred),
            'y_pred':    y_pred,
            'modelo':    modelo
        }

    return modelos, resultados, scaler, X_train, X_test, y_train, y_test

with st.spinner("Cargando dataset desde UCI..."):
    X, y = cargar_datos()

with st.spinner("Entrenando modelos..."):
    modelos, resultados, scaler, X_train, X_test, y_train, y_test = entrenar_modelos(X, y)

st.success(f"✅ Dataset cargado: {X.shape[0]} instancias, {X.shape[1]} variables")

# ── TABS PRINCIPALES ──────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Exploración", "⚙️ Modelos", "📈 Evaluación", "🔮 Predicción", "ℹ️ Acerca de"
])

# ── TAB 1: EXPLORACIÓN ────────────────────────────────────────────────────────
with tab1:
    st.subheader("Exploración del Dataset")

    col1, col2, col3 = st.columns(3)
    col1.metric("Total instancias", "569")
    col2.metric("Variables predictoras", "30")
    col3.metric("Clases", "2 (B / M)")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Distribución de clases**")
        conteo = y['Diagnosis'].value_counts()
        fig, ax = plt.subplots(figsize=(5, 4))
        colores = ['#4C9BE8', '#E85C5C']
        ax.bar(['Benigno (B)', 'Maligno (M)'], conteo.values, color=colores)
        ax.set_ylabel('Cantidad')
        ax.set_title('Distribución de Clases')
        for i, v in enumerate(conteo.values):
            ax.text(i, v + 3, f'{v}\n({v/len(y)*100:.1f}%)', ha='center', fontweight='bold')
        st.pyplot(fig)
        plt.close()

    with col2:
        st.markdown("**Proporción de clases**")
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.pie(conteo.values, labels=['Benigno (B)', 'Maligno (M)'],
               autopct='%1.1f%%', colors=colores, startangle=90,
               explode=(0, 0.05))
        ax.set_title('Proporción')
        st.pyplot(fig)
        plt.close()

    st.markdown("**Vista previa del dataset**")
    df_preview = X.copy()
    df_preview["Diagnosis"] = y.values.flatten()
    st.dataframe(df_preview.head(10), use_container_width=True)

    st.markdown("**Estadísticas descriptivas**")
    st.dataframe(X.describe().round(3), use_container_width=True)

    st.markdown("**Matriz de correlación — Primeras 10 variables**")
    fig, ax = plt.subplots(figsize=(12, 7))
    sns.heatmap(X.iloc[:, :10].corr(), annot=True, fmt='.2f',
                cmap='coolwarm', center=0, ax=ax)
    ax.set_title('Correlación entre variables')
    st.pyplot(fig)
    plt.close()

# ── TAB 2: MODELOS ────────────────────────────────────────────────────────────
with tab2:
    st.subheader("Modelos Entrenados")

    st.markdown("""
    Se entrenaron **3 modelos de clasificación** con los mismos datos (80% train / 20% test):

    | Modelo | Descripción |
    |---|---|
    | **Regresión Logística** | Modelo lineal, simple e interpretable. Línea base. |
    | **Random Forest** | Ensemble de árboles de decisión. Robusto y da importancia de variables. |
    | **SVM (RBF)** | Máquina de soporte vectorial con kernel radial. Efectivo en espacios de alta dimensión. |
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**División de datos**")
        st.info(f"🟦 Train: {X_train.shape[0]} muestras (80%)\n\n🟩 Test: {X_test.shape[0]} muestras (20%)")

    with col2:
        st.markdown("**Preprocesamiento**")
        st.info("✅ StandardScaler aplicado\n\n✅ Stratify activado (mantiene proporción de clases)")

    st.markdown("**Importancia de variables — Random Forest**")
    rf_model = modelos['Random Forest']
    importancias = pd.Series(rf_model.feature_importances_, index=X.columns)
    importancias = importancias.sort_values(ascending=False).head(15)

    fig, ax = plt.subplots(figsize=(10, 6))
    importancias.plot(kind='barh', color='seagreen', edgecolor='white', ax=ax)
    ax.invert_yaxis()
    ax.set_title('Top 15 Variables más Importantes — Random Forest')
    ax.set_xlabel('Importancia relativa')
    st.pyplot(fig)
    plt.close()

# ── TAB 3: EVALUACIÓN ─────────────────────────────────────────────────────────
with tab3:
    st.subheader("Evaluación de Modelos")

    # Tabla de métricas
    metricas_df = pd.DataFrame({
        nombre: {
            'Accuracy':  round(r['Accuracy'],  4),
            'Precision': round(r['Precision'], 4),
            'Recall':    round(r['Recall'],    4),
            'F1-Score':  round(r['F1-Score'],  4)
        }
        for nombre, r in resultados.items()
    }).T

    st.markdown("**Comparación de métricas**")
    st.dataframe(
        metricas_df.style.highlight_max(axis=0, color='#90EE90'),
        use_container_width=True
    )
    st.caption("🟢 Verde = mejor valor por columna | En salud: priorizar **Recall**")

    # Gráfico comparativo
    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(metricas_df.columns))
    width = 0.25
    colores = ['#4C9BE8', '#2ECC71', '#E74C3C']
    for i, (nombre, row) in enumerate(metricas_df.iterrows()):
        bars = ax.bar(x + i * width, row.values, width, label=nombre,
                      color=colores[i], alpha=0.85)
        for bar, val in zip(bars, row.values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=8)
    ax.set_xticks(x + width)
    ax.set_xticklabels(metricas_df.columns)
    ax.set_ylim(0.85, 1.05)
    ax.set_ylabel('Valor')
    ax.set_title('Comparación de Modelos')
    ax.legend()
    ax.axhline(y=0.95, color='gray', linestyle='--', alpha=0.5, label='Umbral 95%')
    st.pyplot(fig)
    plt.close()

    # Matrices de confusión
    st.markdown("**Matrices de confusión**")
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    for ax, (nombre, r) in zip(axes, resultados.items()):
        cm = confusion_matrix(y_test, r['y_pred'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=['Benigno', 'Maligno'],
                    yticklabels=['Benigno', 'Maligno'])
        ax.set_title(nombre)
        ax.set_xlabel('Predicción')
        ax.set_ylabel('Real')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # Reporte del mejor modelo
    mejor = metricas_df['Recall'].idxmax()
    st.markdown(f"**Reporte detallado — Mejor modelo por Recall: `{mejor}`**")
    reporte = classification_report(
        y_test, resultados[mejor]['y_pred'],
        target_names=['Benigno', 'Maligno']
    )
    st.code(reporte)

# ── TAB 4: PREDICCIÓN INTERACTIVA ─────────────────────────────────────────────
with tab4:
    st.subheader("🔮 Predicción en tiempo real")
    st.markdown("Ajusta los valores de las variables y el modelo predecirá si el tumor es **Benigno** o **Maligno**.")

    modelo_sel = st.selectbox("Selecciona el modelo:", list(modelos.keys()))

    st.markdown("**Variables del núcleo celular (valores medios)**")
    col1, col2, col3 = st.columns(3)

    vars_principales = X.columns[:10].tolist()
    valores = {}

    for i, col_name in enumerate(vars_principales):
        col = [col1, col2, col3][i % 3]
        min_v = float(X[col_name].min())
        max_v = float(X[col_name].max())
        mean_v = float(X[col_name].mean())
        with col:
            valores[col_name] = st.slider(
                col_name, min_value=round(min_v, 4),
                max_value=round(max_v, 4),
                value=round(mean_v, 4),
                step=round((max_v - min_v) / 100, 4)
            )

    # Completar las 30 variables con la media para las no mostradas
    input_completo = []
    for col_name in X.columns:
        if col_name in valores:
            input_completo.append(valores[col_name])
        else:
            input_completo.append(float(X[col_name].mean()))

    input_array = np.array(input_completo).reshape(1, -1)
    input_scaled = scaler.transform(input_array)

    modelo_activo = modelos[modelo_sel]
    prediccion = modelo_activo.predict(input_scaled)[0]
    probabilidad = modelo_activo.predict_proba(input_scaled)[0]

    st.divider()
    col1, col2, col3 = st.columns(3)

    with col1:
        if prediccion == 1:
            st.error("🔴 MALIGNO")
        else:
            st.success("🟢 BENIGNO")

    with col2:
        st.metric("Probabilidad Benigno", f"{probabilidad[0]*100:.1f}%")

    with col3:
        st.metric("Probabilidad Maligno", f"{probabilidad[1]*100:.1f}%")

    # Barra de probabilidad
    fig, ax = plt.subplots(figsize=(8, 1.5))
    ax.barh([''], [probabilidad[0]], color='#4C9BE8', label='Benigno')
    ax.barh([''], [probabilidad[1]], left=[probabilidad[0]], color='#E85C5C', label='Maligno')
    ax.set_xlim(0, 1)
    ax.set_xlabel('Probabilidad')
    ax.legend(loc='upper right')
    ax.set_title('Distribución de probabilidad')
    st.pyplot(fig)
    plt.close()

    st.caption("⚠️ Esta herramienta es solo para fines académicos. No reemplaza el diagnóstico médico profesional.")

# ── TAB 5: ACERCA DE ─────────────────────────────────────────────────────────
with tab5:
    st.subheader("Acerca del Proyecto")
    st.markdown("""
    ### Breast Cancer Wisconsin (Diagnostic)

    **Descripción del dataset:**
    Las características fueron calculadas a partir de imágenes digitalizadas de aspiración con aguja fina (FNA) de una masa mamaria.
    Describen las características de los núcleos celulares presentes en la imagen.

    **Variables:** Para cada núcleo celular se calculan 10 características reales:
    - Radio, Textura, Perímetro, Área, Suavidad
    - Compacidad, Concavidad, Puntos cóncavos, Simetría, Dimensión fractal

    Cada una con **media**, **error estándar** y **peor valor** → 30 variables en total.

    **Métricas clave:**

    | Métrica | Fórmula | Descripción |
    |---|---|---|
    | Accuracy | (VP + VN) / Total | % total de aciertos |
    | Precision | VP / (VP + FP) | De los predichos malignos, cuántos lo eran |
    | Recall | VP / (VP + FN) | De los malignos reales, cuántos detectó |
    | F1-Score | 2 × (P × R) / (P + R) | Balance entre Precision y Recall |

    > En salud, el **Recall de la clase Maligna** es la métrica prioritaria porque minimiza los falsos negativos (malignos no detectados).

    **Citación:**
    > Wolberg, W., Mangasarian, O., Street, N., & Street, W. (1993).
    > Breast Cancer Wisconsin (Diagnostic) [Dataset].
    > UCI Machine Learning Repository. https://doi.org/10.24432/C5DW2B
    """)
