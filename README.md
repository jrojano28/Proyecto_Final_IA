# 🤖 Aplicación Web de Machine Learning — Clasificación con Regresión Logística

> **Proyecto Final — Inteligencia Artificial**
> Aplicación web educativa para entrenar, evaluar y usar modelos de clasificación supervisada basados en Regresión Logística.

---

## 🌐 Demo en vivo

🔗 **URL del despliegue:** _[Pendiente — se actualizará tras el despliegue en Render]_

🐙 **Repositorio GitHub:** [https://github.com/jrojano28/Proyecto_Final_IA](https://github.com/jrojano28/Proyecto_Final_IA)

---

## 📌 Descripción del Proyecto

Esta aplicación web permite demostrar de forma práctica e interactiva los conceptos fundamentales del aprendizaje automático supervisado:

- Cómo se **estructura un problema de clasificación** supervisada
- Cómo se **separan los datos** en entrenamiento (80%) y prueba (20%)
- Cómo se **entrena un modelo** de Regresión Logística
- Cómo se **interpretan los coeficientes** del modelo
- Cómo se **evalúa el modelo** usando métricas de clasificación (Accuracy, Precision, Recall, F1)
- Cómo se **realiza una predicción** con nuevos datos
- Cómo se **despliega una aplicación web** en un entorno cloud

---

## 🧠 Tecnologías utilizadas

| Tecnología | Uso |
|------------|-----|
| **Python 3** | Lenguaje principal |
| **Flask** | Framework web (servidor y rutas) |
| **Scikit-Learn** | Entrenamiento y evaluación del modelo (`LogisticRegression`) |
| **Pandas** | Carga y preprocesamiento de datos |
| **NumPy** | Operaciones numéricas sobre arrays |
| **Matplotlib** | Generación de la matriz de confusión como imagen |
| **Joblib** | Serialización y persistencia del modelo entrenado |
| **Bootstrap 5** | Diseño visual responsivo |
| **Gunicorn** | Servidor WSGI para producción (cloud) |

---

## 📁 Estructura del proyecto

```
Proyecto_Final_IA/
│
├── app.py                  # Aplicación Flask principal (rutas y lógica ML)
├── Procfile                # Configuración para despliegue en Render/Heroku
├── requirements.txt        # Dependencias del proyecto
├── README.md               # Este archivo
│
├── templates/              # Páginas HTML (Jinja2)
│   ├── index.html          # Página principal / Dashboard
│   ├── dataset.html        # Vista del dataset Iris
│   ├── dataset_wine.html   # Vista del dataset Wine
│   ├── entrenar.html       # Entrenamiento con Iris + métricas
│   ├── entrenar_csv.html   # Entrenamiento con CSV propio
│   ├── archivo.html        # Carga de archivo CSV
│   ├── prediccion.html     # Formulario de predicción
│   ├── matriz.html         # Visualización de matriz de confusión (Iris)
│   ├── matriz_csv.html     # Visualización de matriz de confusión (CSV)
│   ├── ia_generativa.html  # Documentación del uso de IA generativa
│   ├── acerca.html         # Información del proyecto y tecnologías
│   └── error.html          # Página de errores amigable
│
├── static/
│   ├── css/
│   │   └── style.css       # Estilos premium personalizados
│   ├── matriz.png          # Imagen de la matriz de confusión (Iris)
│   └── matriz_csv.png      # Imagen de la matriz de confusión (CSV)
│
├── models/                 # Modelos y metadatos serializados (.pkl)
│   ├── modelo_iris.pkl
│   ├── modelo_csv.pkl
│   ├── columnas.pkl
│   ├── encoders.pkl
│   └── metrics.pkl
│
├── uploads/                # Archivos CSV cargados por el usuario
└── utils/
    ├── preprocessing.py    # Funciones de preprocesamiento
    ├── training.py         # Funciones de entrenamiento
    └── evaluation.py       # Funciones de evaluación
```

---

## ⚙️ Instalación y ejecución local

### 1. Clonar el repositorio

```bash
git clone <URL_DEL_REPOSITORIO>
cd Proyecto_Final_IA
```

### 2. Crear y activar el entorno virtual

```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 4. Ejecutar la aplicación

```bash
flask run
```

La aplicación estará disponible en: `http://127.0.0.1:5000`

---

## 🔄 Flujo de uso de la aplicación

```
1. Inicio (/)
   │
   ├── Datasets precargados
   │   ├── /dataset       → Vista del dataset Iris (150 registros, 4 variables)
   │   └── /dataset_wine  → Vista del dataset Wine (178 registros, 13 variables)
   │
   ├── /entrenar          → Entrenar modelo con Iris (automático, 80/20)
   │   └── /matriz        → Ver matriz de confusión
   │
   ├── /archivo           → Cargar tu propio archivo CSV
   │   └── /entrenar_csv  → Configurar variables y entrenar modelo
   │       └── /prediccion → Hacer predicciones con nuevos datos
   │
   ├── /ia_generativa     → Documentación de IA generativa usada
   └── /acerca            → Tecnologías y métricas explicadas
```

---

## 📊 Métricas de evaluación disponibles

| Métrica | Descripción |
|---------|-------------|
| **Accuracy** | Porcentaje total de predicciones correctas |
| **Precision** | Capacidad de evitar falsos positivos (promedio ponderado) |
| **Recall** | Capacidad de detectar todos los positivos reales |
| **F1 Score** | Media armónica entre Precision y Recall |
| **Matriz de Confusión** | Visualización gráfica de aciertos y errores por clase |

---

## 🤖 Uso de IA Generativa

Este proyecto fue desarrollado con el apoyo de herramientas de IA generativa (**Antigravity** y **ChatGPT**). La documentación detallada sobre qué fue generado, qué fue corregido manualmente y qué se aprendió en el proceso está disponible en la sección **[IA Generativa](http://localhost:5000/ia_generativa)** de la aplicación.

---

## 🚀 Despliegue en la nube (Render)

El proyecto está configurado para desplegarse en **[Render.com](https://render.com)**:

1. El archivo `Procfile` contiene: `web: gunicorn app:app`
2. El archivo `requirements.txt` incluye todas las dependencias, incluido `gunicorn`
3. En Render, crear un **Web Service** apuntando al repositorio de GitHub

---

## 👥 Integrantes del grupo

| Nombre | Rol |
|--------|-----|
| _(Nombre del estudiante 1)_ | Desarrollo y ML |
| _(Nombre del estudiante 2)_ | Frontend y despliegue |

---

## 📅 Fecha de entrega

**14 de Junio de 2026 — 11:59 pm**

---

*Proyecto Final — Asignatura de Inteligencia Artificial · 2026*
