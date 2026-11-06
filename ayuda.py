"""Textos de ayuda: significado de cada indicador y manual de uso descargable.

Centralizado aquí para que las tarjetas (tooltips), la pestaña de documentación
y el manual en Markdown digan siempre lo mismo.
"""

import streamlit as st

from config import ASIGNATURA, CITA_DATASET, CITA_KAGGLE, DOCENTE, PROGRAMA, PROYECTO, SUBTITULO

# Indicador → (definición, cómo leerlo en este proyecto)
GLOSARIO = {
    "Exactitud (accuracy)": (
        "Proporción de radiografías clasificadas correctamente sobre el total.",
        "Con clases desbalanceadas (≈ 3 neumonías por cada normal) puede engañar: un modelo que "
        "dijera siempre «neumonía» tendría ~73 % de exactitud sin aprender nada."),
    "Sensibilidad (recall)": (
        "De todos los pacientes que realmente tienen neumonía, qué porcentaje detecta el modelo. "
        "VP / (VP + FN).",
        "Es la métrica clínica prioritaria en tamizaje: cada punto que falta es un niño con neumonía "
        "que el sistema dejaría pasar."),
    "Especificidad": (
        "De todos los pacientes sanos, qué porcentaje clasifica correctamente como normales. "
        "VN / (VN + FP).",
        "Una especificidad baja significa muchas falsas alarmas: más radiografías que un "
        "radiólogo debe revisar sin necesidad."),
    "Precisión (VPP)": (
        "De todas las radiografías que el modelo marca como neumonía, qué porcentaje lo es realmente. "
        "VP / (VP + FP).",
        "Responde: «si el sistema dice neumonía, ¿qué tanto le creo?»."),
    "F1": (
        "Media armónica entre precisión y sensibilidad. El F1 macro promedia el F1 de ambas clases "
        "con el mismo peso.",
        "Es el criterio usado para seleccionar el mejor modelo y el umbral, porque castiga "
        "descuidar la clase minoritaria (NORMAL)."),
    "ROC-AUC": (
        "Área bajo la curva ROC: probabilidad de que el modelo asigne mayor puntaje a un caso con "
        "neumonía que a uno normal elegido al azar. 0,5 = azar; 1,0 = perfecto.",
        "No depende del umbral: mide la capacidad de ordenar los casos. Es útil para comparar "
        "modelos antes de fijar el punto de corte."),
    "Umbral de decisión": (
        "Probabilidad a partir de la cual la radiografía se clasifica como NEUMONÍA.",
        "Se eligió en el conjunto de validación (no en prueba) maximizando el F1 macro. Bajarlo "
        "aumenta la sensibilidad y las falsas alarmas; subirlo hace lo contrario."),
    "Pérdida (loss)": (
        "Entropía cruzada ponderada por clase: penaliza las predicciones seguras y equivocadas.",
        "En las curvas de aprendizaje, si la pérdida de entrenamiento baja y la de validación sube, "
        "el modelo está sobreajustando."),
    "Mapa de activación (CAM)": (
        "Mapa de calor que indica qué regiones de la imagen empujaron la predicción hacia neumonía.",
        "Debe concentrarse en los campos pulmonares. Activaciones en bordes, letras o electrodos "
        "sugieren que el modelo usa atajos y no hallazgos clínicos."),
    "VP, FP, VN, FN": (
        "Verdaderos positivos, falsos positivos, verdaderos negativos y falsos negativos: las cuatro "
        "celdas de la matriz de confusión.",
        "El FN (neumonía no detectada) es el error más costoso en este contexto."),
}


def ayuda(nombre: str) -> str:
    """Texto corto para tooltips."""
    definicion, lectura = GLOSARIO[nombre]
    return f"{definicion} {lectura}".replace('"', "'")


def expansor_indicadores(nombres: list[str], titulo: str = "¿Qué significa cada indicador?"):
    with st.expander(titulo):
        for n in nombres:
            definicion, lectura = GLOSARIO[n]
            st.markdown(f'<div class="ayuda-kpi"><b>{n}.</b> {definicion} <i>{lectura}</i></div>',
                        unsafe_allow_html=True)
            st.write("")


def manual_markdown(meta: dict | None) -> str:
    umbral = f"{meta['umbral']:.2f}" if meta else "(definido por el notebook)"
    glosario = "\n".join(f"- **{n}.** {d} {l}" for n, (d, l) in GLOSARIO.items())
    return f"""# Manual de uso — {PROYECTO}

**{SUBTITULO}**
{ASIGNATURA} · {PROGRAMA} · Docente: {DOCENTE}

## 1. ¿Para qué sirve esta herramienta?
Clasifica radiografías de tórax pediátricas como **NORMAL** o **NEUMONÍA** con una red neuronal
convolucional (EfficientNet-B0 ajustada por transfer learning) y muestra un **mapa de activación**
que indica en qué zonas de la imagen se apoyó la decisión. Es un prototipo académico: **no es un
dispositivo médico ni sustituye el criterio de un radiólogo.**

## 2. Pestañas
| Pestaña | Qué encuentra |
|---|---|
| La historia | Problema, pregunta orientadora y flujo metodológico |
| El dataset | Origen, distribución de clases, duplicados y partición por paciente |
| Consultas SQL | Exploración de la base DuckDB con consultas predefinidas o propias (solo lectura) |
| Pipeline | Preprocesamiento y laboratorio interactivo de aumento de datos |
| Arquitectura | CNN base vs. EfficientNet-B0, transfer learning en dos fases y CAM |
| Entrenamiento | Curvas de pérdida, exactitud, F1 y AUC por época |
| Métricas | Resultados en el conjunto de prueba, matriz de confusión, ROC y desempeño por subtipo |
| Umbral clínico | Efecto de mover el punto de corte sobre sensibilidad y especificidad |
| Predicción | Cargue una radiografía (o use un ejemplo) y obtenga probabilidad, decisión y mapa de calor |
| Documentación | Este manual y el glosario |

## 3. Cómo hacer una predicción
1. Vaya a **Predicción**.
2. Elija una radiografía de ejemplo (del conjunto de prueba, nunca vista en entrenamiento) o suba
   un archivo JPG/PNG de una radiografía de tórax anteroposterior.
3. Revise la probabilidad de neumonía, la decisión según el umbral activo (barra lateral; por
   defecto {umbral}) y el mapa de activación.
4. Ajuste la opacidad del mapa para comparar con la imagen original.

**Advertencias:** el modelo fue entrenado con niños de 1 a 5 años de un único hospital. En adultos,
proyecciones laterales, fotografías de pantalla o imágenes a color los resultados no son confiables;
la app avisa si la imagen no parece una radiografía.

## 4. Glosario de indicadores
{glosario}

## 5. Ejecución local
```bash
pip install -r requirements.txt
streamlit run app.py
```
Requiere la carpeta `artefactos/` (y opcionalmente `chest_xray.duckdb`) generada por el notebook
`ACA2_Neumonia_CNN.ipynb`. Para reentrenar: ejecutar el notebook en Colab con GPU, descargar
`artefactos_aca2.zip` y descomprimirlo en la raíz del repositorio.

## 6. Estructura del código
| Archivo | Responsabilidad |
|---|---|
| `app.py` | Orquestador: página, barra lateral y pestañas |
| `config.py` | Paleta, textos del curso, integrantes, rutas y enlaces |
| `modelo.py` | Carga de artefactos, preprocesamiento, inferencia ONNX y CAM |
| `ayuda.py` | Glosario y este manual |
| `ui.py`, `estilos.py` | Componentes visuales y CSS (heredados del ACA 1) |
| `vista_*.py` | Una pestaña por archivo, cada una con `render(ctx)` |

## 7. Referencias
- {CITA_DATASET}
- {CITA_KAGGLE}
- Tan, M., & Le, Q. V. (2019). EfficientNet: Rethinking model scaling for convolutional neural networks. *ICML*, 6105–6114.
- Zhou, B., Khosla, A., Lapedriza, A., Oliva, A., & Torralba, A. (2016). Learning deep features for discriminative localization. *CVPR*, 2921–2929.
"""
