---
title: Proyecto Final IA - Clasificacion con Regresion Logistica
emoji: 🤖
colorFrom: indigo
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
---

# 🤖 Aplicación Web de Machine Learning — Clasificación con Regresión Logística

> **Proyecto Final — Inteligencia Artificial**
> Aplicación web educativa para entrenar, evaluar y usar modelos de clasificación supervisada basados en Regresión Logística.

---

## 🚀 Demo en vivo

🔗 **Repositorio GitHub:** [https://github.com/jrojano28/Proyecto_Final_IA](https://github.com/jrojano28/Proyecto_Final_IA)

---

## 📋 Descripción del Proyecto

Esta aplicación web permite demostrar de forma práctica e interactiva los conceptos fundamentales del aprendizaje automático supervisado:

- Cómo se **estructura un problema de clasificación** supervisada
- Cómo se **separan los datos** en entrenamiento (80%) y prueba (20%)
- Cómo se **entrena un modelo** de Regresión Logística
- Cómo se **interpretan los coeficientes** del modelo
- Cómo se **evalúa el modelo** usando métricas de clasificación (Accuracy, Precision, Recall, F1)
- Cómo se **realiza una predicción** con nuevos datos
- Cómo se **despliega una aplicación web** en un entorno cloud

---

## 🛠 Tecnologías utilizadas

| Tecnología | Uso |
|------------|-----|
| **Python 3** | Lenguaje principal |
| **Flask** | Framework web (servidor y rutas) |
| **Scikit-Learn** | Entrenamiento y evaluación del modelo (`LogisticRegression`) |
| **Pandas** | Carga y preprocesamiento de datos |
| **NumPy** | Operaciones numéricas sobre arrays |
| **Matplotlib** | Generación de gráficas y matriz de confusión |
| **Joblib** | Serialización y persistencia del modelo entrenado |
| **Bootstrap 5** | Diseño visual responsivo |
| **Gunicorn** | Servidor WSGI para producción (cloud) |
| **Docker** | Contenedorización para despliegue en Hugging Face Spaces |

---

## 📁 Estructura del proyecto

```
Proyecto_Final_IA/
│
├── app.py                  # Aplicación Flask principal (rutas y lógica ML)
├── Dockerfile              # Configuración Docker para Hugging Face Spaces
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
│   └── error.html          # Página de errores amigable
│
├── static/
│   └── css/
│       └── style.css       # Estilos premium personalizados
│
├── models/                 # Modelos y metadatos serializados (.pkl)
│
├── uploads/                # Archivos CSV cargados por el usuario
│
└── utils/
    ├── preprocessing.py    # Funciones de preprocesamiento
    ├── training.py         # Funciones de entrenamiento
    └── evaluation.py       # Funciones de evaluación
```

---

## ⚙️ Instalación y ejecución local

### 1. Clonar el repositorio

```bash
git clone https://github.com/jrojano28/Proyecto_Final_IA.git
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

## 📊 Métricas de evaluación disponibles

| Métrica | Descripción |
|---------|-------------|
| **Accuracy** | Porcentaje total de predicciones correctas |
| **Precision** | Capacidad de evitar falsos positivos (promedio ponderado) |
| **Recall** | Capacidad de detectar todos los positivos reales |
| **F1 Score** | Media armónica entre Precision y Recall |
| **Matriz de Confusión** | Visualización gráfica de aciertos y errores por clase |

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

*Proyecto Final — Asignatura de Inteligencia Artificial © 2026*
