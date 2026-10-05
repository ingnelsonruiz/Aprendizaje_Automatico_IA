"""Constantes del proyecto: paleta, metadatos académicos y diccionario de variables.

Editar aquí cambia la identidad visual y los textos de todo el proyecto.
"""

# ── Paleta ───────────────────────────────────────────────────────────────────
CYAN = "#00D4FF"
VIOLET = "#7B2FFF"
GREEN = "#00FF88"
PINK = "#FF2D78"
AMBER = "#FFD700"
NARANJA = "#FF6B35"

COLOR_MODELO = {
    "Regresión Logística": CYAN,
    "Random Forest": VIOLET,
    "SVM (RBF)": GREEN,
}

# ── Metadatos académicos ─────────────────────────────────────────────────────
PROYECTO = "Propuesta del Proyecto ACA 1"
SUBTITULO = "Modelado con algoritmos clásicos de Machine Learning y despliegue interactivo en Streamlit"
ASIGNATURA = "Aprendizaje Automático"
CODIGOS = ["54ES2", "SEGUNDO BLOQUE", "26ES4"]
PROGRAMA = "Especialización en Inteligencia Artificial"
GRUPO = "Grupo 3"

EQUIPO = [
    ("Lina María Cabezas Alvear", CYAN),
    ("Nelson Javier Ruiz Lozano", VIOLET),
    ("Yehimy Alexandra Cabrera", GREEN),
    ("Eliana Andrea Medina Forero", PINK),
]

# ── Dataset ──────────────────────────────────────────────────────────────────
UCI_ID = 17
SEMILLA = 42
TEST_SIZE = 0.2

CITA = ("Wolberg, W., Mangasarian, O., Street, N., & Street, W. (1993). "
        "Breast Cancer Wisconsin (Diagnostic) [Dataset]. UCI Machine Learning Repository. "
        "https://doi.org/10.24432/C5DW2B")

# (original, español, qué mide, aspecto que describe, por qué importa clínicamente)
CARACTERISTICAS = [
    ("radius", "Radio",
     "Promedio de las distancias desde el centro del núcleo hasta los puntos de su borde.",
     "Tamaño",
     "Las células cancerosas pierden el control de su ciclo de división y sus núcleos crecen. "
     "Un radio grande es una de las primeras señales de alarma."),
    ("texture", "Textura",
     "Desviación estándar de los tonos de gris dentro del núcleo en la imagen.",
     "Aspecto interno",
     "Un núcleo sano se ve parejo. Cuando el ADN se desorganiza, el interior del núcleo se ve "
     "moteado y la variación de grises sube."),
    ("perimeter", "Perímetro",
     "Longitud total del contorno del núcleo.",
     "Tamaño",
     "Crece junto con el radio, pero crece todavía más rápido si el borde además es irregular."),
    ("area", "Área",
     "Superficie que ocupa el núcleo en la imagen.",
     "Tamaño",
     "Es la medida de tamaño más directa. Un núcleo maligno puede ocupar más del doble de "
     "superficie que uno benigno."),
    ("smoothness", "Suavidad",
     "Variación local entre las longitudes del radio: qué tan liso es el borde.",
     "Regularidad del borde",
     "Un tumor benigno tiende a tener bordes lisos y redondeados; el maligno los tiene rugosos."),
    ("compactness", "Compacidad",
     "Perímetro² / área − 1,0. Indica qué tan compacta o alargada es la forma.",
     "Forma",
     "Un círculo perfecto es lo más compacto posible. Entre más se aleje de un círculo, "
     "más sospechosa es la forma."),
    ("concavity", "Concavidad",
     "Severidad (profundidad) de los hundimientos del contorno.",
     "Irregularidad del borde",
     "Mide qué tan profundas son las «mordidas» del contorno. Los núcleos malignos suelen "
     "tener hendiduras marcadas."),
    ("concave_points", "Puntos cóncavos",
     "Cantidad de porciones cóncavas (entrantes) del contorno.",
     "Irregularidad del borde",
     "Cuenta cuántas hendiduras hay. Es una de las variables que mejor separa benigno de "
     "maligno en este dataset."),
    ("symmetry", "Simetría",
     "Qué tan simétrico es el núcleo.",
     "Forma",
     "El crecimiento descontrolado rompe la simetría del núcleo."),
    ("fractal_dimension", "Dimensión fractal",
     "Aproximación de la «línea costera» − 1: mide la complejidad del borde.",
     "Complejidad del borde",
     "Un borde muy enrevesado, como una costa recortada, tiene mayor dimensión fractal."),
]

BASE_ES = {orig: es for orig, es, _, _, _ in CARACTERISTICAS}
POR_QUE = {es: porque for _, es, _, _, porque in CARACTERISTICAS}
SUFIJO_ES = {"1": "media", "2": "error est.", "3": "peor"}

NOMBRES_METRICAS = {
    "Accuracy": "Exactitud",
    "Precision": "Precisión",
    "Recall": "Sensibilidad",
    "F1": "F1",
}
