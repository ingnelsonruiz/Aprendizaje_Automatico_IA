"""Hoja de estilos de la aplicación. Editar aquí cambia el aspecto de todas las vistas."""

import streamlit as st

CSS = """
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

/* TARJETAS */
.card {
    padding: 1.1rem 1.2rem; border-radius: 14px; height: 100%;
    border: 1px solid var(--c); background: rgba(13,27,42,0.55);
    box-shadow: 0 0 18px color-mix(in srgb, var(--c) 18%, transparent);
}
.card h4 { font-family: 'Orbitron', sans-serif; font-size: 0.95rem; color: var(--c); margin: 0 0 6px; letter-spacing: 1px; }
.card p { color: #b9cddb; font-size: 1.02rem; margin: 0; line-height: 1.35; }
.card .n { font-family: 'Orbitron', sans-serif; font-size: 1.8rem; color: var(--c); }

/* MIEMBRO DEL EQUIPO */
.member {
    display: flex; align-items: center; gap: 14px; padding: 1rem 1.2rem; border-radius: 14px;
    border: 1px solid var(--c); background: rgba(13,27,42,0.55); margin-bottom: 10px;
    box-shadow: 0 0 18px color-mix(in srgb, var(--c) 15%, transparent);
    transition: transform .2s;
}
.member:hover { transform: translateX(6px); }
.avatar {
    flex: 0 0 52px; width: 52px; height: 52px; border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-family: 'Orbitron', sans-serif; font-size: 1.1rem; font-weight: 700;
    color: var(--c); border: 2px solid var(--c); background: rgba(255,255,255,0.04);
}
.member .nom { color: #e8f2f8; font-size: 1.15rem; font-weight: 600; }
.member .rol { color: #8AAABB; font-size: 0.92rem; letter-spacing: 1px; }

/* HISTORIA */
.story {
    padding: 1.4rem 1.6rem; border-radius: 16px; margin-bottom: 0.8rem;
    border-left: 4px solid var(--c); background: rgba(13,27,42,0.6);
    color: #c3d7e4; font-size: 1.12rem; line-height: 1.55;
}
.story b { color: #ffffff; }
.story .q {
    display: block; margin-top: 0.8rem; font-family: 'Orbitron', sans-serif;
    font-size: 1.05rem; color: var(--c); letter-spacing: 1px;
}

/* FLUJO */
.flow { display: flex; flex-wrap: wrap; gap: 10px; align-items: stretch; margin: 0.6rem 0 0.4rem; }
.step {
    flex: 1 1 150px; padding: 0.9rem 0.8rem; border-radius: 12px; text-align: center;
    border: 1px solid rgba(0,212,255,0.4); background: rgba(0,212,255,0.06);
}
.step b { display: block; color: #00D4FF; font-family: 'Orbitron', sans-serif; font-size: 0.8rem; letter-spacing: 1px; margin-bottom: 4px; }
.step span { color: #b9cddb; font-size: 0.95rem; line-height: 1.25; }

/* TABS — override agresivo del naranja de Streamlit */
div[data-baseweb="tab-list"],
.stTabs div[data-baseweb="tab-list"],
[data-testid="stTabs"] div[data-baseweb="tab-list"] {
    background-color: #080c18 !important;
    background: #080c18 !important;
    border-bottom: 1px solid rgba(0,212,255,0.3) !important;
    padding: 4px 6px 0 !important;
    gap: 4px !important;
}

/* cada pestaña */
div[data-baseweb="tab"],
.stTabs div[data-baseweb="tab"] {
    background: rgba(255,255,255,0.04) !important;
    color: #b0cfe0 !important;
    font-weight: 600 !important;
    letter-spacing: 1px !important;
    border-radius: 8px 8px 0 0 !important;
    height: 44px !important;
    padding: 0 14px !important;
}
div[data-baseweb="tab"]:hover {
    background: rgba(0,212,255,0.1) !important;
    color: #e0f4ff !important;
}
div[data-baseweb="tab"][aria-selected="true"],
.stTabs div[data-baseweb="tab"][aria-selected="true"] {
    background: linear-gradient(180deg, rgba(0,212,255,0.22), rgba(0,212,255,0.04)) !important;
    color: #00D4FF !important;
    border-bottom: 2px solid #00D4FF !important;
}

/* texto dentro de la pestaña siempre visible */
div[data-baseweb="tab"] p,
div[data-baseweb="tab"] span {
    color: inherit !important;
}

/* flecha de navegación (overflow) — siempre visible */
div[data-baseweb="tab-list"] button,
.stTabs div[data-baseweb="tab-list"] > button {
    background: rgba(0,212,255,0.15) !important;
    border: 1px solid rgba(0,212,255,0.45) !important;
    color: #00D4FF !important;
    opacity: 1 !important;
    visibility: visible !important;
    border-radius: 6px !important;
    min-width: 32px !important;
    min-height: 32px !important;
    align-self: center !important;
}
div[data-baseweb="tab-list"] button:hover {
    background: rgba(0,212,255,0.3) !important;
}
div[data-baseweb="tab-list"] button svg {
    fill: #00D4FF !important;
    stroke: #00D4FF !important;
}

/* panel de contenido */
div[data-baseweb="tab-panel"] {
    background: transparent !important;
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

section[data-testid="stSidebar"] {
    background: rgba(8,12,24,0.95);
    border-right: 1px solid rgba(0,212,255,0.2);
}
section[data-testid="stSidebar"] * {
    color: #e0eaf2 !important;
}
section[data-testid="stSidebar"] .stMarkdown p,
section[data-testid="stSidebar"] .stMarkdown strong,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] .stSelectbox label,
section[data-testid="stSidebar"] small,
section[data-testid="stSidebar"] span {
    color: #e0eaf2 !important;
}
section[data-testid="stSidebar"] h3 {
    color: #00D4FF !important;
    font-family: 'Orbitron', sans-serif;
    font-size: 0.95rem;
    letter-spacing: 2px;
}
section[data-testid="stSidebar"] hr {
    border-color: rgba(0,212,255,0.25) !important;
}
section[data-testid="stSidebar"] .stSelectbox > div > div {
    background: rgba(13,27,42,0.8) !important;
    border: 1px solid rgba(0,212,255,0.4) !important;
    color: #e0eaf2 !important;
}
</style>
"""


def aplicar():
    """Inyecta la hoja de estilos en la página."""
    st.markdown(CSS, unsafe_allow_html=True)
