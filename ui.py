"""Componentes reutilizables de interfaz.

Todas las vistas usan estas funciones, así que un cambio aquí se refleja
en toda la aplicación sin tocar cada vista.
"""

import streamlit as st

from config import CYAN


# ── Envoltorios tolerantes a cambios de API de Streamlit ─────────────────────
def mostrar(fig):
    """Dibuja una figura de Plotly ocupando todo el ancho disponible."""
    try:
        st.plotly_chart(fig, width="stretch")
    except TypeError:                       # versiones anteriores de Streamlit
        st.plotly_chart(fig, use_container_width=True)


def tabla(df, ocultar_indice=True):
    """Dibuja un DataFrame ocupando todo el ancho disponible."""
    try:
        st.dataframe(df, width="stretch", hide_index=ocultar_indice)
    except TypeError:
        st.dataframe(df, use_container_width=True, hide_index=ocultar_indice)


# ── Estilo común de las figuras ──────────────────────────────────────────────
def estilo(fig, alto=420, titulo=None):
    """Aplica el tema oscuro del proyecto a una figura de Plotly."""
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(13,27,42,0.35)",
        height=alto,
        margin=dict(l=20, r=20, t=55 if titulo else 25, b=20),
        font=dict(family="Rajdhani, sans-serif", size=14, color="#cfe3f0"),
        title=dict(text=titulo,
                   font=dict(family="Orbitron, sans-serif", size=15, color="#ffffff")) if titulo else None,
        legend=dict(bgcolor="rgba(0,0,0,0)"),
    )
    fig.update_xaxes(gridcolor="rgba(0,212,255,0.08)", zerolinecolor="rgba(0,212,255,0.15)")
    fig.update_yaxes(gridcolor="rgba(0,212,255,0.08)", zerolinecolor="rgba(0,212,255,0.15)")
    return fig


# ── Bloques de contenido ─────────────────────────────────────────────────────
def kpi(valor, etiqueta, color):
    """Devuelve el HTML de una tarjeta de indicador. Usar con unsafe_allow_html=True."""
    return (f'<div class="kpi" style="--c:{color}">'
            f'<div class="kpi-v">{valor}</div>'
            f'<div class="kpi-l">{etiqueta}</div></div>')


def tarjeta(titulo, texto, color, numero=None):
    """Devuelve el HTML de una tarjeta con título, texto y número opcional."""
    n = f'<div class="n">{numero}</div>' if numero is not None else ""
    return f'<div class="card" style="--c:{color}">{n}<h4>{titulo}</h4><p>{texto}</p></div>'


def miembro(nombre, color, rol):
    """Devuelve el HTML de una ficha de integrante del equipo."""
    iniciales = "".join(p[0] for p in nombre.split()[:2]).upper()
    return (f'<div class="member" style="--c:{color}">'
            f'<div class="avatar">{iniciales}</div>'
            f'<div><div class="nom">{nombre}</div>'
            f'<div class="rol">{rol}</div></div></div>')


def seccion(texto):
    """Escribe un encabezado de sección."""
    st.markdown(f'<div class="sec"><span>▌</span> {texto}</div>', unsafe_allow_html=True)


def panel(texto):
    """Escribe un bloque de texto explicativo."""
    st.markdown(f'<div class="panel">{texto}</div>', unsafe_allow_html=True)


def historia(texto, color=CYAN, pregunta=None):
    """Escribe un bloque narrativo con barra lateral de color."""
    q = f'<span class="q">{pregunta}</span>' if pregunta else ""
    st.markdown(f'<div class="story" style="--c:{color}">{texto}{q}</div>', unsafe_allow_html=True)
