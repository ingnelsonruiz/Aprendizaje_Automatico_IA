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
    page_title="NEURAL DIAGNOSTICS · Breast Cancer ML",
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
    position: relative; padding: 2.2rem 2.5rem; margin-bottom: 1.2rem;
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
    display: inline-block; padding: 4px 14px; border-radius: 20px; font-size: 0.75rem;
    letter-spacing: 3px; color: #00D4FF; border: 1px solid #00D4FF;
    background: rgba(0,212,255,0.08); margin-bottom: 0.8rem;
}
.hero h1 {
    font-family: 'Orbitron', sans-serif; font-weight: 900; font-size: 2.6rem; margin: 0.2rem 0;
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
.kpi-v { font-family: 'Orbitron', sans-serif; font-size: 2rem; font-weight: 700; color: var(--c); }
.kpi-l { color: #9fb7c9; font-size: 0.9rem; letter-spacing: 2px; text-transform: uppercase; margin-top: 4px; }

/* TABS */
.stTabs [data-baseweb="tab-list"] { gap: 6px; border-bottom: 1px solid rgba(0,212,255,0.25); }
.stTabs [data-baseweb="tab"] {
    height: 46px; padding: 0 18px; border-radius: 10px 10px 0 0;
    background: rgba(255,255,255,0.03); color: #8AAABB; font-weight: 600; letter-spacing: 1px;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(180deg, rgba(0,212,255,0.2), rgba(0,212,255,0.02)) !important;
    color: #00D4FF !important; border-bottom: 2px solid #00D4FF;
}

/* SECCIONES */
.sec { font-family: 'Orbitron', sans-serif; font-size: 1.15rem; color: #fff; margin: 1.2rem 0 0.4rem; letter-spacing: 1px; }
.sec span { color: #00D4FF; }
.panel {
    padding: 1rem 1.3rem; border-radius: 14px; border: 1px solid rgba(123,47,255,0.35);
    background: rgba(13,27,42,0.55); color: #b9cddb; font-size: 1.05rem;
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
# UTILIDADES
# ═════════════════════════════════════════════════════════════════════════════
def mostrar(fig):
    """Muestra un gráfico Plotly ajustado al ancho (compatible con versiones nuevas y viejas)."""
    try:
        st.plotly_chart(fig, width="stretch")
    except TypeError:
        st.plotly_chart(fig, use_container_width=True)


def tabla(df):
    try:
        st.dataframe(df, width="stretch")
    except TypeError:
        st.dataframe(df, use_container_width=True)


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


def seccion(texto):
    st.markdown(f'<div class="sec"><span>▌</span> {texto}</div>', unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# DATOS Y MODELOS
# ═════════════════════════════════════════════════════════════════════════════
@st.cache_data(show_spinner=False)
def cargar_datos():
    ds = fetch_ucirepo(id=17)
    X = ds.data.features.copy()
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


with st.spinner("⚡ Conectando con UCI y entrenando modelos..."):
    X, y = cargar_datos()
    RES, SC, YTE, N_TR, N_TE = entrenar()

etiqueta = y.map({0: "Benigno", 1: "Maligno"})

# ═════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### 🧬 CENTRO DE CONTROL")
    modelo_sel = st.selectbox("Modelo activo", list(RES.keys()), index=1)
    st.markdown("---")
    st.markdown(f"**Entrenamiento:** {N_TR} casos  \n**Prueba:** {N_TE} casos  \n**Variables:** {X.shape[1]}")
    st.markdown("---")
    st.caption("Dataset: UCI ML Repository · ID 17  \nDOI: 10.24432/C5DW2B")

# ═════════════════════════════════════════════════════════════════════════════
# HERO
# ═════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="hero">
    <div class="hero-tag">MACHINE LEARNING · CLASIFICACIÓN BINARIA</div>
    <h1>NEURAL DIAGNOSTICS</h1>
    <p>Breast Cancer Wisconsin (Diagnostic) · Detección de tumores benignos y malignos con Inteligencia Artificial</p>
</div>
""", unsafe_allow_html=True)

c1, c2, c3, c4, c5 = st.columns(5)
c1.markdown(kpi(f"{len(X)}", "Pacientes", CYAN), unsafe_allow_html=True)
c2.markdown(kpi(f"{X.shape[1]}", "Dimensiones", VIOLET), unsafe_allow_html=True)
c3.markdown(kpi(f"{RES[modelo_sel]['Accuracy']*100:.1f}%", "Accuracy", GREEN), unsafe_allow_html=True)
c4.markdown(kpi(f"{RES[modelo_sel]['Recall']*100:.1f}%", "Recall maligno", PINK), unsafe_allow_html=True)
c5.markdown(kpi(f"{RES[modelo_sel]['auc']:.3f}", "AUC · ROC", AMBER), unsafe_allow_html=True)

st.write("")
tabs = st.tabs(["🌌 Universo 3D", "📊 Exploración", "🤖 Modelos", "🎯 Umbral clínico", "🔮 Predicción", "ℹ️ Acerca de"])

# ═════════════════════════════════════════════════════════════════════════════
# TAB 1 · UNIVERSO 3D (PCA)
# ═════════════════════════════════════════════════════════════════════════════
with tabs[0]:
    seccion("REDUCCIÓN DE DIMENSIONALIDAD · 30D → 3D")
    st.markdown('<div class="panel">Cada punto es un paciente. PCA comprime las 30 variables en 3 componentes '
                'principales: se puede <b>rotar, acercar y explorar</b> el espacio para ver cómo se separan '
                'los tumores benignos de los malignos.</div>', unsafe_allow_html=True)

    pca = PCA(n_components=3, random_state=42)
    Z = pca.fit_transform(StandardScaler().fit_transform(X))
    dfz = pd.DataFrame(Z, columns=["PC1", "PC2", "PC3"])
    dfz["Diagnóstico"] = etiqueta.values

    fig = px.scatter_3d(
        dfz, x="PC1", y="PC2", z="PC3", color="Diagnóstico",
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
    a.markdown(kpi(f"{var[0]*100:.1f}%", "Varianza PC1", CYAN), unsafe_allow_html=True)
    b.markdown(kpi(f"{var[1]*100:.1f}%", "Varianza PC2", VIOLET), unsafe_allow_html=True)
    c.markdown(kpi(f"{var.sum()*100:.1f}%", "Varianza total 3D", GREEN), unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════════════════════════
# TAB 2 · EXPLORACIÓN
# ═════════════════════════════════════════════════════════════════════════════
with tabs[1]:
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
        mostrar(estilo(fig, alto=330))

    seccion("MAPA DE CORRELACIÓN")
    top = X.iloc[:, :12].corr()
    fig = px.imshow(top, text_auto=".2f", aspect="auto",
                    color_continuous_scale=[[0, PINK], [0.5, "#0D1B2A"], [1, CYAN]], zmin=-1, zmax=1)
    mostrar(estilo(fig, alto=560))

    seccion("VISTA PREVIA DE DATOS")
    prev = X.copy()
    prev.insert(0, "Diagnóstico", etiqueta.values)
    tabla(prev.head(15))

# ═════════════════════════════════════════════════════════════════════════════
# TAB 3 · MODELOS
# ═════════════════════════════════════════════════════════════════════════════
with tabs[2]:
    seccion("COMPARACIÓN DE MODELOS")
    metricas = ["Accuracy", "Precision", "Recall", "F1"]
    df_m = pd.DataFrame({n: {k: r[k] for k in metricas} for n, r in RES.items()}).T

    col1, col2 = st.columns(2)
    with col1:
        fig = go.Figure()
        for nombre in df_m.index:
            fig.add_trace(go.Bar(name=nombre, x=metricas, y=df_m.loc[nombre].values,
                                 marker_color=COLOR_MODELO[nombre],
                                 text=[f"{v:.3f}" for v in df_m.loc[nombre].values], textposition="outside"))
        fig.update_layout(barmode="group", yaxis=dict(range=[0.85, 1.03]))
        mostrar(estilo(fig, alto=420, titulo="MÉTRICAS POR MODELO"))

    with col2:
        fig = go.Figure()
        for nombre in df_m.index:
            vals = df_m.loc[nombre].values.tolist()
            fig.add_trace(go.Scatterpolar(r=vals + [vals[0]], theta=metricas + [metricas[0]], name=nombre,
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
        fig.update_layout(xaxis_title="Falsos positivos (FPR)", yaxis_title="Recall (TPR)",
                          legend=dict(x=0.45, y=0.08))
        mostrar(estilo(fig, alto=420, titulo="CURVAS ROC"))

    with col4:
        cm = confusion_matrix(YTE, RES[modelo_sel]["pred"])
        fig = px.imshow(cm, text_auto=True, x=["Benigno", "Maligno"], y=["Benigno", "Maligno"],
                        color_continuous_scale=[[0, "#0D1B2A"], [1, COLOR_MODELO[modelo_sel]]])
        fig.update_traces(textfont=dict(size=30))
        fig.update_layout(xaxis_title="Predicción", yaxis_title="Real", coloraxis_showscale=False)
        mostrar(estilo(fig, alto=420, titulo=f"MATRIZ DE CONFUSIÓN · {modelo_sel.upper()}"))

    seccion("IMPORTANCIA DE VARIABLES · RANDOM FOREST")
    rf = RES["Random Forest"]["modelo"]
    imp = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=True).tail(15)
    fig = go.Figure(go.Bar(x=imp.values, y=imp.index, orientation="h",
                           marker=dict(color=imp.values, colorscale=[[0, VIOLET], [1, CYAN]]),
                           text=[f"{v:.3f}" for v in imp.values], textposition="outside"))
    mostrar(estilo(fig, alto=520))

    seccion("TABLA DE RESULTADOS")
    tabla(df_m.round(4))

# ═════════════════════════════════════════════════════════════════════════════
# TAB 4 · UMBRAL CLÍNICO
# ═════════════════════════════════════════════════════════════════════════════
with tabs[3]:
    seccion("AJUSTE DEL UMBRAL DE DECISIÓN")
    st.markdown('<div class="panel">En salud, un <b>falso negativo</b> (maligno clasificado como benigno) es el error más '
                'grave. Baja el umbral para detectar más casos malignos: sube el Recall, pero aparecen más falsas '
                'alarmas.</div>', unsafe_allow_html=True)

    umbral = st.slider("Umbral de probabilidad para clasificar como MALIGNO", 0.05, 0.95, 0.50, 0.01)
    proba = RES[modelo_sel]["proba"]
    pred_u = (proba >= umbral).astype(int)
    tp = int(((pred_u == 1) & (YTE == 1)).sum())
    fn = int(((pred_u == 0) & (YTE == 1)).sum())
    fp = int(((pred_u == 1) & (YTE == 0)).sum())
    tn = int(((pred_u == 0) & (YTE == 0)).sum())
    rec = tp / (tp + fn) if (tp + fn) else 0
    pre = tp / (tp + fp) if (tp + fp) else 0

    a, b, c, d = st.columns(4)
    a.markdown(kpi(f"{rec*100:.1f}%", "Recall", GREEN), unsafe_allow_html=True)
    b.markdown(kpi(f"{pre*100:.1f}%", "Precisión", CYAN), unsafe_allow_html=True)
    c.markdown(kpi(f"{fn}", "Falsos negativos", PINK), unsafe_allow_html=True)
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
    fig.add_trace(go.Scatter(x=ts, y=recs, name="Recall", line=dict(color=GREEN, width=3)))
    fig.add_trace(go.Scatter(x=ts, y=pres, name="Precisión", line=dict(color=CYAN, width=3)))
    fig.add_vline(x=umbral, line=dict(color=AMBER, dash="dash", width=2))
    fig.update_layout(xaxis_title="Umbral", yaxis_title="Valor", yaxis=dict(range=[0, 1.05]))
    mostrar(estilo(fig, alto=400, titulo="RECALL vs PRECISIÓN SEGÚN EL UMBRAL"))

# ═════════════════════════════════════════════════════════════════════════════
# TAB 5 · PREDICCIÓN
# ═════════════════════════════════════════════════════════════════════════════
with tabs[4]:
    seccion("SIMULADOR DE DIAGNÓSTICO EN TIEMPO REAL")
    st.caption("Ajusta las 10 variables principales (valores medios). Las otras 20 se mantienen en su promedio.")

    cols_pred = list(X.columns[:10])
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
# TAB 6 · ACERCA DE
# ═════════════════════════════════════════════════════════════════════════════
with tabs[5]:
    seccion("SOBRE EL DATASET")
    st.markdown("""
<div class="panel">
Las características se calculan a partir de imágenes digitalizadas de una <b>aspiración con aguja fina (FNA)</b>
de una masa mamaria y describen los núcleos celulares: radio, textura, perímetro, área, suavidad, compacidad,
concavidad, puntos cóncavos, simetría y dimensión fractal. Cada una con su <b>media</b>, <b>error estándar</b>
y <b>peor valor</b>, para un total de 30 variables.
</div>
""", unsafe_allow_html=True)

    seccion("MÉTRICAS CLAVE")
    tabla(pd.DataFrame({
        "Métrica": ["Accuracy", "Precision", "Recall", "F1-Score"],
        "Fórmula": ["(VP + VN) / Total", "VP / (VP + FP)", "VP / (VP + FN)", "2·P·R / (P + R)"],
        "Responde a": ["¿Qué % acertó en total?", "De los predichos malignos, ¿cuántos lo eran?",
                       "De los malignos reales, ¿cuántos detectó?", "Balance entre Precision y Recall"],
    }))

    seccion("CITA")
    st.code("Wolberg, W., Mangasarian, O., Street, N., & Street, W. (1993). Breast Cancer Wisconsin (Diagnostic) "
            "[Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5DW2B", language=None)

st.markdown('<div class="aviso">NEURAL DIAGNOSTICS · Python · Scikit-learn · Plotly · Streamlit</div>',
            unsafe_allow_html=True)
