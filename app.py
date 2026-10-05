import re
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc
)

# ═════════════════════════════════════════════════════════════════════════════
# CONFIGURACIÓN Y ESTILO
# ═════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Diagnóstico Inteligente · Cáncer de Mama",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

CYAN, VIOLET, GREEN, PINK, AMBER = "#00D4FF", "#7B2FFF", "#00FF88", "#FF2D78", "#FFD700"
COLOR_MODELO = {"Regresión Logística": CYAN, "Random Forest": VIOLET, "SVM (RBF)": GREEN}

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@500;700;900&family=Rajdhani:wght@400;500;600&display=swap');

.stApp {
    background:
        radial-gradient(circle at 15% 0%, rgba(123,47,255,0.22) 0%, transparent 40%),
        radial-gradient(circle at 90% 10%, rgba(0,212,255,0.18) 0%, transparent 45%),
        #05070f;
    font-family: 'Rajdhani', sans-serif;
}
html, body, [class*="css"] { font-family: 'Rajdhani', sans-serif; }
#MainMenu, footer { visibility: hidden; }
.block-container { padding-top: 1.5rem; max-width: 1400px; }

/* HERO */
.hero {
    position: relative; padding: 2rem 2.5rem; margin-bottom: 1.2rem;
    border: 1px solid rgba(0,212,255,0.35); border-radius: 18px;
    background: linear-gradient(135deg, rgba(13,27,42,0.9), rgba(10,14,26,0.6));
    box-shadow: 0 0 40px rgba(0,212,255,0.12), inset 0 0 60px rgba(123,47,255,0.08);
    overflow: hidden;
}
.hero::before {
    content: ""; position: absolute; top: 0; left: -100%; width: 60%; height: 2px;
    background: linear-gradient(90deg, transparent, #00D4FF, transparent);
    animation: scan 3.5s linear infinite;
}
@keyframes scan { 0% { left: -60%; } 100% { left: 110%; } }
.hero-tag {
    display: inline-block; padding: 4px 14px; border-radius: 20px; font-size: 0.78rem;
    letter-spacing: 3px; color: #00D4FF; border: 1px solid #00D4FF;
    background: rgba(0,212,255,0.08); margin-bottom: 0.6rem;
}
.chips { margin: 0.2rem 0 0.6rem; }
.chip {
    display: inline-block; padding: 3px 12px; margin: 0 8px 6px 0; border-radius: 6px;
    font-size: 0.82rem; letter-spacing: 2px; font-weight: 600; color: #cfe3f0;
    border: 1px solid rgba(123,47,255,0.6); background: rgba(123,47,255,0.15);
}
.hero h1 {
    font-family: 'Orbitron', sans-serif; font-weight: 900; font-size: 2.4rem; margin: 0.2rem 0;
    background: linear-gradient(90deg, #00D4FF, #7B2FFF, #00FF88, #00D4FF);
    background-size: 300% 100%; -webkit-background-clip: text; background-clip: text;
    -webkit-text-fill-color: transparent; animation: flow 6s linear infinite;
}
@keyframes flow { 0% { background-position: 0% 50%; } 100% { background-position: 300% 50%; } }
.hero p { color: #8AAABB; font-size: 1.1rem; margin: 0; }

/* KPI */
.kpi {
    position: relative; padding: 1.1rem 1rem; border-radius: 14px; text-align: center;
    background: linear-gradient(160deg, rgba(255,255,255,0.05), rgba(255,255,255,0.01));
    border: 1px solid var(--c); box-shadow: 0 0 22px color-mix(in srgb, var(--c) 25%, transparent);
    transition: transform .25s, box-shadow .25s; backdrop-filter: blur(6px);
}
.kpi:hover { transform: translateY(-6px); box-shadow: 0 0 38px var(--c); }
.kpi::before { content:""; position:absolute; top:0; left:15%; width:70%; height:3px; background:var(--c); border-radius:3px; }
.kpi-v { font-family: 'Orbitron', sans-serif; font-size: 1.9rem; font-weight: 700; color: var(--c); }
.kpi-l { color: #9fb7c9; font-size: 0.88rem; letter-spacing: 2px; text-transform: uppercase; margin-top: 4px; }

/* TARJETAS DE TEXTO */
.card {
    padding: 1.1rem 1.2rem; border-radius: 14px; height: 100%;
    border: 1px solid var(--c); background: rgba(13,27,42,0.55);
    box-shadow: 0 0 18px color-mix(in srgb, var(--c) 18%, transparent);
}
.card h4 { font-family: 'Orbitron', sans-serif; font-size: 0.95rem; color: var(--c); margin: 0 0 6px; letter-spacing: 1px; }
.card p { color: #b9cddb; font-size: 1.02rem; margin: 0; line-height: 1.35; }
.card .n { font-family: 'Orbitron', sans-serif; font-size: 1.8rem; color: var(--c); }

/* FLUJO */
.flow { display: flex; flex-wrap: wrap; gap: 10px; align-items: stretch; margin: 0.6rem 0 0.4rem; }
.step {
    flex: 1 1 150px; padding: 0.9rem 0.8rem; border-radius: 12px; text-align: center;
    border: 1px solid rgba(0,212,255,0.4); background: rgba(0,212,255,0.06); position: relative;
}
.step b { display: block; color: #00D4FF; font-family: 'Orbitron', sans-serif; font-size: 0.8rem; letter-spacing: 1px; margin-bottom: 4px; }
.step span { color: #b9cddb; font-size: 0.95rem; line-height: 1.25; }

/* TABS */
.stTabs [data-baseweb="tab-list"] { gap: 6px; border-bottom: 1px solid rgba(0,212,255,0.25); flex-wrap: wrap; }
.stTabs [data-baseweb="tab"] {
    height: 46px; padding: 0 16px; border-radius: 10px 10px 0 0;
    background: rgba(255,255,255,0.03); color: #8AAABB; font-weight: 600; letter-spacing: 1px;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(180deg, rgba(0,212,255,0.2), rgba(0,212,255,0.02)) !important;
    color: #00D4FF !important; border-bottom: 2px solid #00D4FF;
}

/* SECCIONES */
.sec { font-family: 'Orbitron', sans-serif; font-size: 1.1rem; color: #fff; margin: 1.3rem 0 0.5rem; letter-spacing: 1px; }
.sec span { color: #00D4FF; }
.panel {
    padding: 1rem 1.3rem; border-radius: 14px; border: 1px solid rgba(123,47,255,0.35);
    background: rgba(13,27,42,0.55); color: #b9cddb; font-size: 1.05rem; line-height: 1.4;
}
.veredicto-m, .veredicto-b {
    padding: 1.2rem; border-radius: 14px; text-align: center;
    font-family: 'Orbitron', sans-serif; font-size: 1.6rem; font-weight: 700; letter-spacing: 3px;
}
.veredicto-m { color: #FF2D78; border: 1px solid #FF2D78; background: rgba(255,45,120,0.08); box-shadow: 0 0 30px rgba(255,45,120,0.35); }
.veredicto-b { color: #00FF88; border: 1px solid #00FF88; background: rgba(0,255,136,0.07); box-shadow: 0 0 30px rgba(0,255,136,0.3); }
.aviso { color: #6d8599; font-size: 0.85rem; text-align: center; margin-top: 2rem; }

section[data-testid="stSidebar"] { background: rgba(8,12,24,0.95); border-right: 1px solid rgba(0,212,255,0.2); }
</style>
""", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# TRADUCCIÓN DE VARIABLES AL ESPAÑOL
# ═════════════════════════════════════════════════════════════════════════════
# (nombre original, nombre en español, qué mide, aspecto que describe)
CARACTERISTICAS = [
    ("radius", "Radio",
     "Promedio de las distancias desde el centro del núcleo hasta los puntos de su borde.",
     "Tamaño"),
    ("texture", "Textura",
     "Desviación estándar de los tonos de gris dentro del núcleo en la imagen.",
     "Aspecto interno"),
    ("perimeter", "Perímetro",
     "Longitud total del contorno del núcleo.",
     "Tamaño"),
    ("area", "Área",
     "Superficie que ocupa el núcleo en la imagen.",
     "Tamaño"),
    ("smoothness", "Suavidad",
     "Variación local entre las longitudes del radio: qué tan liso es el borde.",
     "Regularidad del borde"),
    ("compactness", "Compacidad",
     "Perímetro² / área − 1,0. Indica qué tan compacta o alargada es la forma.",
     "Forma"),
    ("concavity", "Concavidad",
     "Severidad (profundidad) de los hundimientos del contorno.",
     "Irregularidad del borde"),
    ("concave_points", "Puntos cóncavos",
     "Cantidad de porciones cóncavas (entrantes) del contorno.",
     "Irregularidad del borde"),
    ("symmetry", "Simetría",
     "Qué tan simétrico es el núcleo.",
     "Forma"),
    ("fractal_dimension", "Dimensión fractal",
     "Aproximación de la «línea costera» − 1: mide la complejidad del borde.",
     "Complejidad del borde"),
]
BASE_ES = {orig: es for orig, es, _, _ in CARACTERISTICAS}
SUFIJO_ES = {"1": "media", "2": "error est.", "3": "peor"}


def nombre_es(col):
    """radius1 -> Radio (media) · texture3 -> Textura (peor) ..."""
    m = re.match(r"^(.*?)(\d)$", col)
    if not m:
        return col
    base, suf = m.groups()
    return f"{BASE_ES.get(base, base)} ({SUFIJO_ES.get(suf, suf)})"


# ═════════════════════════════════════════════════════════════════════════════
# UTILIDADES
# ═════════════════════════════════════════════════════════════════════════════
def mostrar(fig):
    """Gráfico Plotly ajustado al ancho (compatible con versiones nuevas y viejas)."""
    try:
        st.plotly_chart(fig, width="stretch")
    except TypeError:
        st.plotly_chart(fig, use_container_width=True)


def tabla(df):
    try:
        st.dataframe(df, width="stretch", hide_index=True)
    except TypeError:
        st.dataframe(df, use_container_width=True, hide_index=True)


def estilo(fig, alto=420, titulo=None):
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(13,27,42,0.35)",
        height=alto,
        margin=dict(l=20, r=20, t=55 if titulo else 25, b=20),
        font=dict(family="Rajdhani, sans-serif", size=14, color="#cfe3f0"),
        title=dict(text=titulo, font=dict(family="Orbitron, sans-serif", size=15, color="#ffffff")) if titulo else None,
        legend=dict(bgcolor="rgba(0,0,0,0)"),
    )
    fig.update_xaxes(gridcolor="rgba(0,212,255,0.08)", zerolinecolor="rgba(0,212,255,0.15)")
    fig.update_yaxes(gridcolor="rgba(0,212,255,0.08)", zerolinecolor="rgba(0,212,255,0.15)")
    return fig


def kpi(valor, etiqueta, color):
    return f'<div class="kpi" style="--c:{color}"><div class="kpi-v">{valor}</div><div class="kpi-l">{etiqueta}</div></div>'


def tarjeta(titulo, texto, color, numero=None):
    n = f'<div class="n">{numero}</div>' if numero is not None else ""
    return f'<div class="card" style="--c:{color}">{n}<h4>{titulo}</h4><p>{texto}</p></div>'


def seccion(texto):
    st.markdown(f'<div class="sec"><span>▌</span> {texto}</div>', unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# DATOS Y MODELOS
# ═════════════════════════════════════════════════════════════════════════════
@st.cache_data(show_spinner=False)
def cargar_datos():
    ds = fetch_ucirepo(id=17)
    X = ds.data.features.copy()
    X.columns = [nombre_es(c) for c in X.columns]
    y = ds.data.targets.iloc[:, 0].map({"M": 1, "B": 0}).astype(int)
    return X, y


@st.cache_resource(show_spinner=False)
def entrenar():
    X, y = cargar_datos()
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    sc = StandardScaler().fit(Xtr)
    Xtr_s, Xte_s = sc.transform(Xtr), sc.transform(Xte)

    modelos = {
        "Regresión Logística": LogisticRegression(random_state=42, max_iter=1000),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
        "SVM (RBF)": SVC(kernel="rbf", probability=True, random_state=42),
    }
    res = {}
    for nombre, m in modelos.items():
        m.fit(Xtr_s, ytr)
        pred = m.predict(Xte_s)
        proba = m.predict_proba(Xte_s)[:, 1]
        fpr, tpr, _ = roc_curve(yte, proba)
        res[nombre] = dict(
            modelo=m, pred=pred, proba=proba, fpr=fpr, tpr=tpr, auc=auc(fpr, tpr),
            Accuracy=accuracy_score(yte, pred), Precision=precision_score(yte, pred),
            Recall=recall_score(yte, pred), F1=f1_score(yte, pred),
        )
    return res, sc, yte.to_numpy(), len(Xtr), len(Xte)


with st.spinner("⚡ Conectando con UCI y entrenando los modelos..."):
    X, y = cargar_datos()
    RES, SC, YTE, N_TR, N_TE = entrenar()

etiqueta = y.map({0: "Benigno", 1: "Maligno"})
N_BEN, N_MAL = int((y == 0).sum()), int((y == 1).sum())

# ═════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### 🧬 CENTRO DE CONTROL")
    modelo_sel = st.selectbox("Modelo activo", list(RES.keys()), index=1)
    st.markdown("---")
    st.markdown(f"**Entrenamiento:** {N_TR} casos  \n**Prueba:** {N_TE} casos  \n**Variables:** {X.shape[1]}")
    st.markdown("---")
    st.markdown("**Asignatura**  \nAprendizaje Automático  \n54ES2 · Segundo bloque · 26ES4  \n"
                "Especialización en Inteligencia Artificial")
    st.markdown("---")
    st.caption("Dataset: UCI ML Repository · ID 17  \nDOI: 10.24432/C5DW2B")

# ═════════════════════════════════════════════════════════════════════════════
# ENCABEZADO
# ═════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="hero">
    <div class="hero-tag">ESPECIALIZACIÓN EN INTELIGENCIA ARTIFICIAL</div>
    <div class="chips">
        <span class="chip">APRENDIZAJE AUTOMÁTICO</span>
        <span class="chip">54ES2</span>
        <span class="chip">SEGUNDO BLOQUE</span>
        <span class="chip">26ES4</span>
    </div>
    <h1>DIAGNÓSTICO INTELIGENTE</h1>
    <p>Cáncer de mama de Wisconsin (Diagnóstico) · Clasificación de tumores benignos y malignos con Machine Learning</p>
</div>
""", unsafe_allow_html=True)

c1, c2, c3, c4, c5 = st.columns(5)
c1.markdown(kpi(f"{len(X)}", "Pacientes", CYAN), unsafe_allow_html=True)
c2.markdown(kpi(f"{X.shape[1]}", "Variables", VIOLET), unsafe_allow_html=True)
c3.markdown(kpi(f"{RES[modelo_sel]['Accuracy']*100:.1f}%", "Exactitud", GREEN), unsafe_allow_html=True)
c4.markdown(kpi(f"{RES[modelo_sel]['Recall']*100:.1f}%", "Sensibilidad (recall)", PINK), unsafe_allow_html=True)
c5.markdown(kpi(f"{RES[modelo_sel]['auc']:.3f}", "Área bajo la curva ROC", AMBER), unsafe_allow_html=True)

st.write("")
(t_data, t_3d, t_exp, t_mod, t_umb, t_pred, t_met) = st.tabs([
    "📘 El dataset", "🌌 Universo 3D", "📊 Exploración", "🤖 Modelos",
    "🎯 Umbral clínico", "🔮 Predicción", "📏 Métricas",
])

# ═════════════════════════════════════════════════════════════════════════════
# TAB · EL DATASET
# ═════════════════════════════════════════════════════════════════════════════
with t_data:
    seccion("¿QUÉ ES ESTE DATASET?")
    st.markdown("""
<div class="panel">
<b>Breast Cancer Wisconsin (Diagnostic)</b> es uno de los conjuntos de datos más usados para aprender clasificación
en Machine Learning. Cada fila es una <b>paciente</b> a la que se le hizo una <b>punción con aguja fina (FNA)</b>
en una masa de la mama. De esa muestra se obtuvo una imagen digitalizada y, con ella, se midieron las
características de los <b>núcleos de las células</b>. Cada caso tiene además un <b>diagnóstico confirmado</b>:
<b>Benigno</b> (no es cáncer) o <b>Maligno</b> (es cáncer). El objetivo del modelo es aprender a predecir ese
diagnóstico usando solo las medidas del núcleo.
</div>
""", unsafe_allow_html=True)

    st.write("")
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.markdown(kpi(f"{len(X)}", "Pacientes", CYAN), unsafe_allow_html=True)
    k2.markdown(kpi(f"{X.shape[1]}", "Variables", VIOLET), unsafe_allow_html=True)
    k3.markdown(kpi(f"{N_BEN}", "Benignos", GREEN), unsafe_allow_html=True)
    k4.markdown(kpi(f"{N_MAL}", "Malignos", PINK), unsafe_allow_html=True)
    k5.markdown(kpi(f"{int(X.isnull().sum().sum())}", "Valores faltantes", AMBER), unsafe_allow_html=True)

    seccion("¿CÓMO SE OBTUVIERON LOS DATOS?")
    st.markdown("""
<div class="flow">
  <div class="step"><b>1 · PUNCIÓN</b><span>Se extrae una muestra de la masa con aguja fina (FNA)</span></div>
  <div class="step"><b>2 · IMAGEN</b><span>La muestra se digitaliza con un microscopio y una cámara</span></div>
  <div class="step"><b>3 · SEGMENTACIÓN</b><span>Se delimita el contorno de cada núcleo celular</span></div>
  <div class="step"><b>4 · MEDICIÓN</b><span>Se calculan 10 características de cada núcleo</span></div>
  <div class="step"><b>5 · RESUMEN</b><span>Cada una se resume en 3 estadísticos → 30 variables</span></div>
  <div class="step"><b>6 · DIAGNÓSTICO</b><span>Se registra el resultado confirmado: Benigno o Maligno</span></div>
</div>
""", unsafe_allow_html=True)

    seccion("DE 10 CARACTERÍSTICAS A 30 VARIABLES")
    a, b, c = st.columns(3)
    a.markdown(tarjeta("MEDIA", "Valor promedio de la característica entre todos los núcleos de la imagen. "
                                "Ejemplo: <i>Radio (media)</i>.", CYAN, "×10"), unsafe_allow_html=True)
    b.markdown(tarjeta("ERROR ESTÁNDAR", "Qué tanto varía la característica de un núcleo a otro dentro de la misma "
                                         "imagen. Ejemplo: <i>Radio (error est.)</i>.", VIOLET, "×10"), unsafe_allow_html=True)
    c.markdown(tarjeta("PEOR VALOR", "Promedio de los tres valores más extremos (más grandes) de la imagen: "
                                     "captura al núcleo más «anormal». Ejemplo: <i>Radio (peor)</i>.", PINK, "×10"),
               unsafe_allow_html=True)
    st.caption("10 características × 3 estadísticos = 30 variables numéricas. Además, el archivo original trae un "
               "identificador de paciente y el diagnóstico, que no se usan como variables predictoras.")

    seccion("DICCIONARIO DE VARIABLES")
    filas = []
    for orig, es, mide, aspecto in CARACTERISTICAS:
        col_media = nombre_es(orig + "1")
        m_ben = X.loc[y == 0, col_media].mean()
        m_mal = X.loc[y == 1, col_media].mean()
        filas.append({
            "Característica": es,
            "Nombre original (UCI)": orig,
            "Qué mide": mide,
            "Describe": aspecto,
            "Media en benignos": round(m_ben, 4),
            "Media en malignos": round(m_mal, 4),
            "Variación": f"{(m_mal / m_ben - 1) * 100:+.0f}%",
        })
    tabla(pd.DataFrame(filas))
    st.caption("Las medias de benignos y malignos se calculan en vivo con los datos reales (variables «media»).")

    seccion("¿QUÉ CARACTERÍSTICAS SEPARAN MEJOR LOS DOS GRUPOS?")
    dif = []
    for orig, es, _, _ in CARACTERISTICAS:
        col_media = nombre_es(orig + "1")
        dif.append((es, (X.loc[y == 1, col_media].mean() / X.loc[y == 0, col_media].mean() - 1) * 100))
    dif = pd.DataFrame(dif, columns=["Característica", "Diferencia"]).sort_values("Diferencia")
    fig = go.Figure(go.Bar(
        x=dif["Diferencia"], y=dif["Característica"], orientation="h",
        marker=dict(color=dif["Diferencia"], colorscale=[[0, CYAN], [1, PINK]]),
        text=[f"{v:+.0f}%" for v in dif["Diferencia"]], textposition="outside",
    ))
    fig.update_layout(xaxis_title="Diferencia porcentual de la media: maligno vs. benigno")
    mostrar(estilo(fig, alto=430, titulo="MALIGNO VS. BENIGNO · DIFERENCIA DE MEDIAS"))
    st.markdown('<div class="panel">Cuanto más larga es la barra, más distintos son en promedio los núcleos malignos '
                'de los benignos en esa característica. Eso es justamente lo que el modelo aprovecha para clasificar.'
                '</div>', unsafe_allow_html=True)

    seccion("FICHA TÉCNICA Y CITA")
    f1_, f2_ = st.columns(2)
    f1_.markdown(tarjeta("FICHA", "Repositorio: UCI Machine Learning Repository (ID 17)<br>"
                                  "Autores: W. Wolberg, O. Mangasarian, N. Street y W. Street<br>"
                                  "Donado: 31/10/1995 · Área: Salud y medicina<br>"
                                  "Tarea: clasificación binaria · Licencia: CC BY 4.0", CYAN),
                  unsafe_allow_html=True)
    f2_.markdown(tarjeta("ARTÍCULO DE ORIGEN", "«Nuclear feature extraction for breast tumor diagnosis» — "
                                               "Street, Wolberg y Mangasarian (1993), Electronic Imaging.<br>"
                                               "DOI del dataset: 10.24432/C5DW2B", VIOLET), unsafe_allow_html=True)
    st.code("Wolberg, W., Mangasarian, O., Street, N., & Street, W. (1993). Breast Cancer Wisconsin (Diagnostic) "
            "[Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5DW2B", language=None)

# ═════════════════════════════════════════════════════════════════════════════
# TAB · UNIVERSO 3D (PCA)
# ═════════════════════════════════════════════════════════════════════════════
with t_3d:
    seccion("REDUCCIÓN DE DIMENSIONALIDAD · DE 30D A 3D")
    st.markdown('<div class="panel">Cada punto es una paciente. El análisis de componentes principales (PCA) comprime las '
                '30 variables en 3 componentes: puedes <b>rotar, acercar y explorar</b> el espacio para ver cómo se '
                'separan los tumores benignos de los malignos.</div>', unsafe_allow_html=True)

    pca = PCA(n_components=3, random_state=42)
    Z = pca.fit_transform(StandardScaler().fit_transform(X))
    dfz = pd.DataFrame(Z, columns=["Componente 1", "Componente 2", "Componente 3"])
    dfz["Diagnóstico"] = etiqueta.values

    fig = px.scatter_3d(
        dfz, x="Componente 1", y="Componente 2", z="Componente 3", color="Diagnóstico",
        color_discrete_map={"Benigno": CYAN, "Maligno": PINK}, opacity=0.85,
    )
    fig.update_traces(marker=dict(size=4, line=dict(width=0.4, color="rgba(255,255,255,0.4)")))
    fig.update_layout(scene=dict(
        xaxis=dict(backgroundcolor="rgba(0,0,0,0)", gridcolor="rgba(0,212,255,0.15)"),
        yaxis=dict(backgroundcolor="rgba(0,0,0,0)", gridcolor="rgba(0,212,255,0.15)"),
        zaxis=dict(backgroundcolor="rgba(0,0,0,0)", gridcolor="rgba(0,212,255,0.15)"),
    ))
    mostrar(estilo(fig, alto=620))

    var = pca.explained_variance_ratio_
    a, b, c = st.columns(3)
    a.markdown(kpi(f"{var[0]*100:.1f}%", "Varianza componente 1", CYAN), unsafe_allow_html=True)
    b.markdown(kpi(f"{var[1]*100:.1f}%", "Varianza componente 2", VIOLET), unsafe_allow_html=True)
    c.markdown(kpi(f"{var.sum()*100:.1f}%", "Varianza total en 3D", GREEN), unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════════════════════════
# TAB · EXPLORACIÓN
# ═════════════════════════════════════════════════════════════════════════════
with t_exp:
    col1, col2 = st.columns([1, 1])
    with col1:
        seccion("DISTRIBUCIÓN DE CLASES")
        cuenta = etiqueta.value_counts()
        fig = go.Figure(go.Pie(
            labels=cuenta.index, values=cuenta.values, hole=0.62,
            marker=dict(colors=[CYAN, PINK], line=dict(color="#05070f", width=3)),
            textinfo="label+percent", textfont=dict(size=15),
        ))
        fig.add_annotation(text=f"<b>{len(X)}</b><br>casos", showarrow=False, font=dict(size=22, color="#fff"))
        mostrar(estilo(fig, alto=380))

    with col2:
        seccion("DISTRIBUCIÓN POR VARIABLE")
        var_sel = st.selectbox("Variable", list(X.columns), index=0)
        dfh = X[[var_sel]].copy()
        dfh["Diagnóstico"] = etiqueta.values
        fig = px.histogram(dfh, x=var_sel, color="Diagnóstico", barmode="overlay", nbins=40, opacity=0.7,
                           color_discrete_map={"Benigno": CYAN, "Maligno": PINK})
        fig.update_layout(yaxis_title="Cantidad de casos")
        mostrar(estilo(fig, alto=330))

    seccion("MAPA DE CORRELACIÓN · VARIABLES «MEDIA»")
    cols_media = [c for c in X.columns if c.endswith("(media)")]
    top = X[cols_media].corr()
    fig = px.imshow(top, text_auto=".2f", aspect="auto",
                    color_continuous_scale=[[0, PINK], [0.5, "#0D1B2A"], [1, CYAN]], zmin=-1, zmax=1)
    mostrar(estilo(fig, alto=600))
    st.caption("Valores cercanos a 1 indican variables que crecen juntas (por ejemplo, radio, perímetro y área).")

    seccion("VISTA PREVIA DE LOS DATOS")
    prev = X.copy()
    prev.insert(0, "Diagnóstico", etiqueta.values)
    tabla(prev.head(15))

# ═════════════════════════════════════════════════════════════════════════════
# TAB · MODELOS
# ═════════════════════════════════════════════════════════════════════════════
with t_mod:
    seccion("COMPARACIÓN DE MODELOS")
    nombres_met = {"Accuracy": "Exactitud", "Precision": "Precisión", "Recall": "Sensibilidad", "F1": "F1"}
    metricas = list(nombres_met.keys())
    etiquetas_met = list(nombres_met.values())
    df_m = pd.DataFrame({n: {k: r[k] for k in metricas} for n, r in RES.items()}).T

    col1, col2 = st.columns(2)
    with col1:
        fig = go.Figure()
        for nombre in df_m.index:
            fig.add_trace(go.Bar(name=nombre, x=etiquetas_met, y=df_m.loc[nombre].values,
                                 marker_color=COLOR_MODELO[nombre],
                                 text=[f"{v:.3f}" for v in df_m.loc[nombre].values], textposition="outside"))
        fig.update_layout(barmode="group", yaxis=dict(range=[0.85, 1.03]))
        mostrar(estilo(fig, alto=420, titulo="MÉTRICAS POR MODELO"))

    with col2:
        fig = go.Figure()
        for nombre in df_m.index:
            vals = df_m.loc[nombre].values.tolist()
            fig.add_trace(go.Scatterpolar(r=vals + [vals[0]], theta=etiquetas_met + [etiquetas_met[0]], name=nombre,
                                          fill="toself", opacity=0.45, line=dict(color=COLOR_MODELO[nombre], width=2)))
        fig.update_layout(polar=dict(bgcolor="rgba(13,27,42,0.35)",
                                     radialaxis=dict(range=[0.88, 1.0], gridcolor="rgba(0,212,255,0.2)"),
                                     angularaxis=dict(gridcolor="rgba(0,212,255,0.2)")))
        mostrar(estilo(fig, alto=420, titulo="RADAR DE DESEMPEÑO"))

    col3, col4 = st.columns(2)
    with col3:
        fig = go.Figure()
        for nombre, r in RES.items():
            fig.add_trace(go.Scatter(x=r["fpr"], y=r["tpr"], mode="lines", name=f"{nombre} · AUC {r['auc']:.3f}",
                                     line=dict(color=COLOR_MODELO[nombre], width=3)))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Azar",
                                 line=dict(color="#667788", dash="dash")))
        fig.update_layout(xaxis_title="Tasa de falsos positivos", yaxis_title="Sensibilidad (recall)",
                          legend=dict(x=0.45, y=0.08))
        mostrar(estilo(fig, alto=420, titulo="CURVAS ROC"))

    with col4:
        cm = confusion_matrix(YTE, RES[modelo_sel]["pred"])
        fig = px.imshow(cm, text_auto=True, x=["Benigno", "Maligno"], y=["Benigno", "Maligno"],
                        color_continuous_scale=[[0, "#0D1B2A"], [1, COLOR_MODELO[modelo_sel]]])
        fig.update_traces(textfont=dict(size=30))
        fig.update_layout(xaxis_title="Predicción del modelo", yaxis_title="Diagnóstico real", coloraxis_showscale=False)
        mostrar(estilo(fig, alto=420, titulo=f"MATRIZ DE CONFUSIÓN · {modelo_sel.upper()}"))

    seccion("IMPORTANCIA DE LAS VARIABLES · RANDOM FOREST")
    rf = RES["Random Forest"]["modelo"]
    imp = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=True).tail(15)
    fig = go.Figure(go.Bar(x=imp.values, y=imp.index, orientation="h",
                           marker=dict(color=imp.values, colorscale=[[0, VIOLET], [1, CYAN]]),
                           text=[f"{v:.3f}" for v in imp.values], textposition="outside"))
    fig.update_layout(xaxis_title="Importancia relativa")
    mostrar(estilo(fig, alto=520))

    seccion("TABLA DE RESULTADOS")
    tabla(df_m.rename(columns=nombres_met).round(4).reset_index().rename(columns={"index": "Modelo"}))

# ═════════════════════════════════════════════════════════════════════════════
# TAB · UMBRAL CLÍNICO
# ═════════════════════════════════════════════════════════════════════════════
with t_umb:
    seccion("AJUSTE DEL UMBRAL DE DECISIÓN")
    st.markdown('<div class="panel">En salud, un <b>falso negativo</b> (un tumor maligno clasificado como benigno) es '
                'el error más grave. Si bajas el umbral, el modelo detecta más casos malignos: sube la sensibilidad, '
                'pero aparecen más falsas alarmas.</div>', unsafe_allow_html=True)

    umbral = st.slider("Umbral de probabilidad para clasificar como MALIGNO", 0.05, 0.95, 0.50, 0.01)
    proba = RES[modelo_sel]["proba"]
    pred_u = (proba >= umbral).astype(int)
    tp = int(((pred_u == 1) & (YTE == 1)).sum())
    fn = int(((pred_u == 0) & (YTE == 1)).sum())
    fp = int(((pred_u == 1) & (YTE == 0)).sum())
    rec = tp / (tp + fn) if (tp + fn) else 0
    pre = tp / (tp + fp) if (tp + fp) else 0

    a, b, c, d = st.columns(4)
    a.markdown(kpi(f"{rec*100:.1f}%", "Sensibilidad", GREEN), unsafe_allow_html=True)
    b.markdown(kpi(f"{pre*100:.1f}%", "Precisión", CYAN), unsafe_allow_html=True)
    c.markdown(kpi(f"{fn}", "Malignos no detectados", PINK), unsafe_allow_html=True)
    d.markdown(kpi(f"{fp}", "Falsas alarmas", AMBER), unsafe_allow_html=True)

    ts = np.linspace(0.05, 0.95, 91)
    recs, pres = [], []
    for t in ts:
        p_ = (proba >= t).astype(int)
        tp_ = ((p_ == 1) & (YTE == 1)).sum()
        fn_ = ((p_ == 0) & (YTE == 1)).sum()
        fp_ = ((p_ == 1) & (YTE == 0)).sum()
        recs.append(tp_ / (tp_ + fn_) if (tp_ + fn_) else 0)
        pres.append(tp_ / (tp_ + fp_) if (tp_ + fp_) else 1)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ts, y=recs, name="Sensibilidad", line=dict(color=GREEN, width=3)))
    fig.add_trace(go.Scatter(x=ts, y=pres, name="Precisión", line=dict(color=CYAN, width=3)))
    fig.add_vline(x=umbral, line=dict(color=AMBER, dash="dash", width=2))
    fig.update_layout(xaxis_title="Umbral", yaxis_title="Valor", yaxis=dict(range=[0, 1.05]))
    mostrar(estilo(fig, alto=400, titulo="SENSIBILIDAD VS. PRECISIÓN SEGÚN EL UMBRAL"))

# ═════════════════════════════════════════════════════════════════════════════
# TAB · PREDICCIÓN
# ═════════════════════════════════════════════════════════════════════════════
with t_pred:
    seccion("SIMULADOR DE DIAGNÓSTICO EN TIEMPO REAL")
    st.caption("Ajusta las 10 características «media» del núcleo. Las otras 20 variables se mantienen en su promedio.")

    cols_pred = [c for c in X.columns if c.endswith("(media)")]
    valores = {}
    grid = st.columns(3)
    for i, nombre in enumerate(cols_pred):
        mn, mx, md = float(X[nombre].min()), float(X[nombre].max()), float(X[nombre].mean())
        paso = (mx - mn) / 200 if mx > mn else 0.01
        with grid[i % 3]:
            valores[nombre] = st.slider(nombre, mn, mx, md, paso)

    fila = np.array([valores.get(c, float(X[c].mean())) for c in X.columns]).reshape(1, -1)
    fila_df = pd.DataFrame(fila, columns=X.columns)
    p_mal = float(RES[modelo_sel]["modelo"].predict_proba(SC.transform(fila_df))[0, 1])

    st.write("")
    izq, der = st.columns([1, 1])
    with izq:
        fig = go.Figure(go.Indicator(
            mode="gauge+number", value=p_mal * 100, number=dict(suffix="%", font=dict(size=46)),
            title=dict(text="PROBABILIDAD DE MALIGNIDAD", font=dict(size=15)),
            gauge=dict(
                axis=dict(range=[0, 100], tickcolor="#667788"),
                bar=dict(color=PINK if p_mal >= 0.5 else GREEN),
                bgcolor="rgba(13,27,42,0.4)",
                steps=[dict(range=[0, 50], color="rgba(0,255,136,0.12)"),
                       dict(range=[50, 100], color="rgba(255,45,120,0.12)")],
                threshold=dict(line=dict(color=AMBER, width=4), thickness=0.8, value=50),
            ),
        ))
        mostrar(estilo(fig, alto=360))
    with der:
        st.write("")
        st.write("")
        if p_mal >= 0.5:
            st.markdown('<div class="veredicto-m">⚠ MALIGNO</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="veredicto-b">✔ BENIGNO</div>', unsafe_allow_html=True)
        st.write("")
        st.markdown(f'<div class="panel">Modelo: <b>{modelo_sel}</b><br>'
                    f'Benigno: <b>{(1-p_mal)*100:.1f}%</b> · Maligno: <b>{p_mal*100:.1f}%</b></div>',
                    unsafe_allow_html=True)
    st.caption("⚠ Herramienta académica. No reemplaza el diagnóstico médico profesional.")

# ═════════════════════════════════════════════════════════════════════════════
# TAB · MÉTRICAS
# ═════════════════════════════════════════════════════════════════════════════
with t_met:
    seccion(f"LOS 4 RESULTADOS POSIBLES · {modelo_sel.upper()}")
    tn_, fp_, fn_, tp_ = confusion_matrix(YTE, RES[modelo_sel]["pred"]).ravel()
    a, b, c, d = st.columns(4)
    a.markdown(tarjeta("VERDADERO POSITIVO", "El modelo dijo <b>maligno</b> y realmente lo era. Acierto clave.",
                       GREEN, int(tp_)), unsafe_allow_html=True)
    b.markdown(tarjeta("VERDADERO NEGATIVO", "El modelo dijo <b>benigno</b> y realmente lo era. Acierto.",
                       CYAN, int(tn_)), unsafe_allow_html=True)
    c.markdown(tarjeta("FALSO POSITIVO", "El modelo dijo <b>maligno</b> pero era benigno. Falsa alarma: genera "
                                         "estrés y exámenes extra.", AMBER, int(fp_)), unsafe_allow_html=True)
    d.markdown(tarjeta("FALSO NEGATIVO", "El modelo dijo <b>benigno</b> pero era maligno. El error más grave: "
                                         "la paciente quedaría sin tratamiento.", PINK, int(fn_)),
               unsafe_allow_html=True)

    seccion("LAS 4 MÉTRICAS, EXPLICADAS CON TUS PROPIOS RESULTADOS")
    r = RES[modelo_sel]
    m1, m2 = st.columns(2)
    m1.markdown(tarjeta(
        f"EXACTITUD (ACCURACY) · {r['Accuracy']*100:.2f}%",
        f"¿Qué porcentaje acertó en total?<br><b>(VP + VN) / Total</b> = ({tp_} + {tn_}) / {tp_+tn_+fp_+fn_}<br>"
        "Engaña si las clases están desbalanceadas.", CYAN), unsafe_allow_html=True)
    m2.markdown(tarjeta(
        f"PRECISIÓN · {r['Precision']*100:.2f}%",
        f"De los que dijo «maligno», ¿cuántos lo eran?<br><b>VP / (VP + FP)</b> = {tp_} / ({tp_} + {fp_})<br>"
        "Mide cuántas falsas alarmas genera.", VIOLET), unsafe_allow_html=True)
    st.write("")
    m3, m4 = st.columns(2)
    m3.markdown(tarjeta(
        f"SENSIBILIDAD (RECALL) · {r['Recall']*100:.2f}%",
        f"De los malignos reales, ¿cuántos detectó?<br><b>VP / (VP + FN)</b> = {tp_} / ({tp_} + {fn_})<br>"
        "La métrica prioritaria en salud.", GREEN), unsafe_allow_html=True)
    m4.markdown(tarjeta(
        f"F1 · {r['F1']*100:.2f}%",
        "Balance entre precisión y sensibilidad.<br><b>2 · P · R / (P + R)</b><br>"
        "Útil cuando importan ambas a la vez.", AMBER), unsafe_allow_html=True)

    seccion("UNA ANALOGÍA PARA NO CONFUNDIRSE")
    st.markdown("""
<div class="panel">
Imagina una <b>alarma de incendios</b>.<br>
• <b>Sensibilidad (recall):</b> de todos los incendios que ocurrieron, ¿cuántos detectó la alarma? Mira la realidad.<br>
• <b>Precisión:</b> de todas las veces que sonó la alarma, ¿cuántas veces había fuego de verdad? Mira lo que el modelo predijo.<br>
• <b>Exactitud:</b> de todas las situaciones, ¿cuántas veces acertó (sonar cuando había fuego y callar cuando no)?<br><br>
En diagnóstico médico preferimos una alarma que a veces suene de más (falsa alarma) antes que una que deje
pasar un incendio real (falso negativo). Por eso la <b>sensibilidad</b> es la métrica más importante.
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="aviso">Diagnóstico Inteligente · Aprendizaje Automático 54ES2 · 26ES4 · Python · Scikit-learn · '
            'Plotly · Streamlit</div>', unsafe_allow_html=True)
