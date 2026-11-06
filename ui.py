"""Componentes reutilizables de interfaz (heredados del ACA 1).

Todas las vistas usan estas funciones, así que un cambio aquí se refleja
en toda la aplicación sin tocar cada vista.
"""

import streamlit as st

from config import CYAN  # noqa: F401


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
    """Aplica el tema «negatoscopio» (claro, tinta azul película) a una figura de Plotly."""
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#FFFFFF",
        height=alto,
        margin=dict(l=20, r=20, t=55 if titulo else 25, b=20),
        font=dict(family="Archivo, system-ui, sans-serif", size=14, color="#0F2A3D"),
        title=dict(text=oracion(titulo), font=dict(size=17, color="#0F2A3D")) if titulo else None,
        legend=dict(bgcolor="rgba(255,255,255,0.85)", bordercolor="#C9D6DF", borderwidth=1),
        hoverlabel=dict(bgcolor="#0F2A3D", font_color="#FFFFFF"),
    )
    fig.update_xaxes(gridcolor="#E1E9EF", zerolinecolor="#C9D6DF", linecolor="#C9D6DF")
    fig.update_yaxes(gridcolor="#E1E9EF", zerolinecolor="#C9D6DF", linecolor="#C9D6DF")
    return fig


# ── Texto en tipo oración ────────────────────────────────────────────────────
# Las vistas escriben algunos títulos en mayúsculas; aquí se convierten a tipo oración
# conservando siglas y nombres propios.
_PROPIOS = {
    "sql": "SQL", "duckdb": "DuckDB", "cnn": "CNN", "f1": "F1", "roc": "ROC", "auc": "AUC",
    "roc-auc": "ROC-AUC", "cam": "CAM", "efficientnet-b0": "EfficientNet-B0", "efficientnet": "EfficientNet",
    "aca": "ACA", "ia": "IA", "onnx": "ONNX", "mbconv": "MBConv", "mbconv1": "MBConv1", "mbconv6": "MBConv6",
    "gap": "GAP", "imagenet": "ImageNet", "streamlit": "Streamlit", "softmax": "Softmax",
    "stem": "Stem", "cun": "CUN", "1×1": "1×1",
}


def oracion(texto):
    if not texto or not texto.isupper():
        return texto
    palabras = []
    for p in texto.lower().split(" "):
        nucleo = p.strip("(),.:")
        palabras.append(p.replace(nucleo, _PROPIOS[nucleo]) if nucleo in _PROPIOS else p)
    salida = " ".join(palabras).replace(" · ", ": ", 1).replace(" · ", ", ")
    return salida[:1].upper() + salida[1:]


# ── Bloques de contenido ─────────────────────────────────────────────────────
def kpi(valor, etiqueta, color, ayuda=None):
    """Devuelve el HTML de una tarjeta de indicador. Usar con unsafe_allow_html=True.

    `ayuda` aparece como texto emergente al pasar el cursor sobre la tarjeta.
    """
    titulo = f' title="{ayuda}"' if ayuda else ""
    return (f'<div class="kpi" style="--c:{color}"{titulo}>'
            f'<div class="kpi-v">{valor}</div>'
            f'<div class="kpi-l">{etiqueta}</div></div>')


def tarjeta(titulo, texto, color, numero=None):
    """Devuelve el HTML de una tarjeta con título, texto y número opcional."""
    n = f'<div class="n">{numero}</div>' if numero is not None else ""
    return f'<div class="card" style="--c:{color}">{n}<h4>{oracion(titulo)}</h4><p>{texto}</p></div>'


def miembro(nombre, color, rol):
    """Devuelve el HTML de una ficha de integrante del equipo."""
    iniciales = "".join(p[0] for p in nombre.split()[:2]).upper()
    return (f'<div class="member" style="--c:{color}">'
            f'<div class="avatar">{iniciales}</div>'
            f'<div><div class="nom">{nombre}</div>'
            f'<div class="rol">{rol}</div></div></div>')


def seccion(texto):
    """Escribe un encabezado de sección."""
    st.markdown(f'<div class="sec">{oracion(texto)}</div>', unsafe_allow_html=True)


def panel(texto):
    """Escribe un bloque de texto explicativo."""
    st.markdown(f'<div class="panel">{texto}</div>', unsafe_allow_html=True)


def historia(texto, color=CYAN, pregunta=None):
    """Escribe un bloque narrativo con barra lateral de color."""
    q = f'<span class="q">{pregunta}</span>' if pregunta else ""
    st.markdown(f'<div class="story" style="--c:{color}">{texto}{q}</div>', unsafe_allow_html=True)


def imagen(img, pie=None):
    """Muestra una imagen ocupando el ancho de su columna."""
    try:
        st.image(img, caption=pie, width="stretch")
    except TypeError:
        st.image(img, caption=pie, use_container_width=True)
