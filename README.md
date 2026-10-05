# Diagnóstico Inteligente — Breast Cancer Wisconsin

**Propuesta del Proyecto ACA 1 · Grupo 3**
Aprendizaje Automático · 54ES2 · Segundo bloque · 26ES4
Especialización en Inteligencia Artificial

Aplicación web que clasifica tumores de mama como benignos o malignos usando el
dataset *Breast Cancer Wisconsin (Diagnostic)* del UCI Machine Learning Repository.

## Ejecutar

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Estructura

```
app.py                     Orquestador: configura la página y despacha las pestañas
requirements.txt

nucleo/                    Lógica compartida (sin interfaz, salvo ui.py)
  config.py                Paleta, metadatos del curso, diccionario de variables
  estilos.py               Hoja de estilos CSS
  ui.py                    Componentes: kpi, tarjeta, sección, panel, gráficos
  datos.py                 Carga, traducción de columnas, entrenamiento, Contexto
  entrenamiento.py         Experimentos didácticos del laboratorio (solo cálculo)

vistas/                    Una pestaña por archivo, cada una con render(ctx)
  historia.py              Qué se busca predecir y variables clave
  dataset.py               Origen de los datos y diccionario
  laboratorio.py           Cómo aprende un modelo, paso a paso
  universo3d.py            PCA en 3 dimensiones
  exploracion.py           Distribuciones y correlaciones
  modelos.py               Comparación de los tres algoritmos
  umbral.py                Ajuste del punto de corte clínico
  prediccion.py            Simulador en tiempo real
  metricas.py              Explicación de las métricas
  equipo.py                Integrantes y alcance
```

## Cómo modificar sin romper nada

| Quiero... | Toco solo... |
|---|---|
| Mejorar el laboratorio | `nucleo/entrenamiento.py` (cálculo) y `vistas/laboratorio.py` (dibujo) |
| Cambiar colores o tipografía | `nucleo/config.py` y `nucleo/estilos.py` |
| Agregar o quitar un modelo de producción | `nucleo/datos.py`, función `entrenar()` |
| Cambiar textos del curso o integrantes | `nucleo/config.py` |
| Agregar una pestaña | Crear `vistas/mi_vista.py` con `render(ctx)` y registrarla en `PESTANAS` de `app.py` |

### Agregar un experimento al laboratorio

1. En `nucleo/entrenamiento.py`, escribe una función que **solo calcule** y devuelva datos.
   Decórala con `@st.cache_resource` si es costosa.
2. En `vistas/laboratorio.py`, crea una función `_mi_experimento()` que la consuma
   y dibuje, y llámala desde `render(ctx)`.

Ningún otro archivo necesita cambiar.

### Agregar un algoritmo al comparador de fronteras

En `nucleo/entrenamiento.py`: añade una entrada al diccionario `ALGORITMOS` con su
control (`slider` o `select`) y su constructor en `_construir_modelo()`. La vista lo
toma automáticamente.

## El objeto `ctx`

Todas las vistas reciben un `Contexto` (definido en `nucleo/datos.py`) con:

| Atributo | Contenido |
|---|---|
| `ctx.X`, `ctx.y` | Variables y objetivo, con nombres en español |
| `ctx.etiqueta` | Serie con «Benigno» / «Maligno» |
| `ctx.resultados` | Dict por modelo: métricas, predicciones, ROC |
| `ctx.activo` | Resultados del modelo elegido en la barra lateral |
| `ctx.escalador` | `StandardScaler` ya ajustado |
| `ctx.columnas` | Orden exacto de columnas del entrenamiento |
| `ctx.columnas_media()` | Las 10 variables «media» |
| `ctx.probabilidad_maligno(valores)` | Predicción segura para el simulador |

## Dataset

Wolberg, W., Mangasarian, O., Street, N., & Street, W. (1993).
*Breast Cancer Wisconsin (Diagnostic)* [Dataset].
UCI Machine Learning Repository. https://doi.org/10.24432/C5DW2B
