"""
Proyecto Final de Inteligencia Artificial
==========================================
Aplicación web Flask para clasificación supervisada con Regresión Logística.
Permite cargar datasets CSV, entrenar modelos, evaluar métricas y realizar predicciones.

Autores: Grupo Proyecto Final IA
Tecnologías: Flask, Scikit-Learn, Pandas, NumPy, Matplotlib
"""

from flask import Flask, render_template, request
from sklearn.datasets import load_iris, load_wine
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)
import matplotlib
matplotlib.use('Agg')  # Backend sin GUI, necesario para Flask (evita errores de tkinter en hilos)
import matplotlib.pyplot as plt
import joblib
import os
from flask import Flask, render_template, request, redirect, url_for, flash

app = Flask(__name__)
app.secret_key = 'supersecretkey'  # Needed for flash messages

import threading
matplotlib_lock = threading.Lock()


# ─────────────────────────────────────────────
# RUTA: Inicio / Dashboard principal
# ─────────────────────────────────────────────
@app.route("/")
def inicio():
    return render_template("index.html")


# ─────────────────────────────────────────────
# RUTA: Vista del Dataset Iris (precargado)
# ─────────────────────────────────────────────
@app.route("/dataset")
def dataset():
    """Muestra una vista previa del famoso dataset de flores Iris."""

    iris = load_iris()

    df = pd.DataFrame(
        iris.data,
        columns=iris.feature_names
    )

    # Mapear los números de clase a nombres de especies
    species_map = {0: "setosa", 1: "versicolor", 2: "virginica"}
    df["species"] = [species_map[t] for t in iris.target]

    tabla = df.head(20).to_html(
        classes="table table-striped table-hover",
        index=False
    )

    return render_template(
        "dataset.html",
        tabla=tabla,
        n_filas=df.shape[0],
        n_columnas=df.shape[1]
    )


# ─────────────────────────────────────────────
# RUTA: Vista del Dataset Wine (precargado)
# ─────────────────────────────────────────────
@app.route("/dataset_wine")
def dataset_wine():
    """Muestra una vista previa del dataset de vinos italianos."""

    wine = load_wine()

    df = pd.DataFrame(
        wine.data,
        columns=wine.feature_names
    )
    df["target"] = wine.target

    tabla = df.head(20).to_html(
        classes="table table-striped table-hover",
        index=False
    )

    return render_template(
        "dataset_wine.html",
        tabla=tabla,
        n_filas=df.shape[0],
        n_columnas=df.shape[1]
    )


# ─────────────────────────────────────────────
# RUTA: Entrenar modelo con Dataset Iris
# ─────────────────────────────────────────────
@app.route("/entrenar")
def entrenar():
    """
    Entrena un clasificador de Regresión Logística sobre el dataset Iris.
    Calcula métricas, genera la matriz de confusión y gráficos adicionales.
    """

    iris = load_iris()
    X = iris.data
    y = iris.target
    feature_names = iris.feature_names
    clases_iris = ["setosa", "versicolor", "virginica"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    modelo = LogisticRegression(max_iter=200)
    modelo.fit(X_train, y_train)

    os.makedirs("models", exist_ok=True)
    os.makedirs("static", exist_ok=True)
    joblib.dump(modelo, "models/modelo_iris.pkl")

    predicciones = modelo.predict(X_test)
    accuracy  = accuracy_score(y_test, predicciones)
    precision = precision_score(y_test, predicciones, average="weighted", zero_division=0)
    recall    = recall_score(y_test, predicciones, average="weighted", zero_division=0)
    f1        = f1_score(y_test, predicciones, average="weighted", zero_division=0)

    # ─── Coeficientes del modelo ───
    avg_coef = np.mean(np.abs(modelo.coef_), axis=0)
    coeficientes = [
        {"variable": name, "coeficiente": round(float(c), 4)}
        for name, c in zip(feature_names, avg_coef)
    ]
    intercepto = [round(float(i), 4) for i in modelo.intercept_]

    # ─── GRÁFICA 0: Matriz de confusión ───
    matriz = confusion_matrix(y_test, predicciones)
    plt.figure(figsize=(6, 5))
    plt.imshow(matriz, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Matriz de Confusión - Iris")
    plt.colorbar()
    tick_marks = range(len(clases_iris))
    plt.xticks(tick_marks, clases_iris, rotation=45)
    plt.yticks(tick_marks, clases_iris)
    plt.xlabel("Predicción")
    plt.ylabel("Valor Real")
    for i in range(len(matriz)):
        for j in range(len(matriz)):
            plt.text(j, i, str(matriz[i, j]), ha="center", va="center",
                     color="white" if matriz[i, j] > matriz.max() / 2.0 else "black")
    plt.tight_layout()
    plt.savefig("static/matriz.png", bbox_inches='tight')
    plt.close()

    # ─── GRÁFICA 1: Curva Sigmoide ───
    z = np.linspace(-8, 8, 300)
    sigmoid = 1 / (1 + np.exp(-z))
    plt.figure(figsize=(7, 4))
    plt.plot(z, sigmoid, color='#4f46e5', linewidth=2.5, label='σ(z) = 1 / (1 + e⁻ᶻ)')
    plt.axhline(0.5, color='#ef4444', linestyle='--', linewidth=1.5, label='Umbral de decisión (0.5)')
    plt.axvline(0, color='#94a3b8', linestyle=':', linewidth=1)
    plt.fill_between(z, sigmoid, 0.5, where=(sigmoid > 0.5), alpha=0.12, color='#4f46e5', label='Zona clase positiva')
    plt.fill_between(z, sigmoid, 0.5, where=(sigmoid < 0.5), alpha=0.12, color='#ef4444', label='Zona clase negativa')
    plt.xlabel('z = β₀ + β₁x₁ + ... + βₙxₙ', fontsize=11)
    plt.ylabel('Probabilidad P(y=1)', fontsize=11)
    plt.title('Función Sigmoide — Corazón de la Regresión Logística', fontsize=12, fontweight='bold')
    plt.legend(fontsize=9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("static/grafica_sigmoide.png", bbox_inches='tight', dpi=120)
    plt.close()

    # ─── GRÁFICA 2: Scatter Plot Petal Length vs Petal Width ───
    colores = ['#f97316', '#4f46e5', '#10b981']
    plt.figure(figsize=(7, 5))
    for cls, color, nombre in zip([0, 1, 2], colores, clases_iris):
        idx = y == cls
        plt.scatter(X[idx, 2], X[idx, 3], c=color, label=nombre, alpha=0.75,
                    edgecolors='white', linewidth=0.5, s=65)
    plt.axhline(0.8,  color='#f97316', linestyle='--', linewidth=1.5, alpha=0.7, label='Frontera Setosa')
    plt.axvline(4.75, color='#4f46e5', linestyle='--', linewidth=1.5, alpha=0.7, label='Frontera Versicolor/Virginica')
    plt.xlabel('Petal Length (cm)', fontsize=11)
    plt.ylabel('Petal Width (cm)', fontsize=11)
    plt.title('Dispersión de Clases — Iris\n(Petal Length vs Petal Width)', fontsize=12, fontweight='bold')
    plt.legend(fontsize=9)
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig("static/grafica_dispersion.png", bbox_inches='tight', dpi=120)
    plt.close()

    # ─── GRÁFICA 3: Curva de Aprendizaje ───
    train_sizes = np.linspace(0.1, 1.0, 10)
    train_acc_vals, test_acc_vals, sizes_label = [], [], []
    for size in train_sizes:
        n = max(int(size * len(X_train)), 6)
        m_tmp = LogisticRegression(max_iter=200)
        m_tmp.fit(X_train[:n], y_train[:n])
        train_acc_vals.append(accuracy_score(y_train[:n], m_tmp.predict(X_train[:n])) * 100)
        test_acc_vals.append(accuracy_score(y_test, m_tmp.predict(X_test)) * 100)
        sizes_label.append(n)
    plt.figure(figsize=(7, 4))
    plt.plot(sizes_label, train_acc_vals, 'o-', color='#4f46e5', linewidth=2, markersize=6, label='Accuracy Entrenamiento')
    plt.plot(sizes_label, test_acc_vals,  's--', color='#10b981', linewidth=2, markersize=6, label='Accuracy Prueba')
    plt.fill_between(sizes_label, train_acc_vals, test_acc_vals, alpha=0.08, color='#f97316', label='Brecha generalización')
    plt.xlabel('Número de muestras de entrenamiento', fontsize=11)
    plt.ylabel('Accuracy (%)', fontsize=11)
    plt.title('Curva de Aprendizaje — Regresión Logística (Iris)', fontsize=12, fontweight='bold')
    plt.legend(fontsize=9)
    plt.ylim([50, 105])
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("static/grafica_aprendizaje.png", bbox_inches='tight', dpi=120)
    plt.close()

    return render_template(
        "entrenar.html",
        accuracy=round(accuracy * 100, 2),
        precision=round(precision * 100, 2),
        recall=round(recall * 100, 2),
        f1=round(f1 * 100, 2),
        n_train=X_train.shape[0],
        n_test=X_test.shape[0],
        coeficientes=coeficientes,
        intercepto=intercepto,
        variables_usadas=list(feature_names)
    )


# ─────────────────────────────────────────────
# RUTA: Cargar archivo CSV personalizado
# ─────────────────────────────────────────────
@app.route("/archivo", methods=["GET", "POST"])
def archivo():
    """Permite al usuario subir su propio dataset CSV y ver una vista previa."""

    tabla = None
    filas = None
    columnas_n = None
    nombre_archivo = None
    columnas_lista = None
    ruta_csv = "uploads/dataset.csv"

    if request.method == "POST":
        archivo = request.files["archivo"]
        if archivo:
            os.makedirs("uploads", exist_ok=True)
            nombre_archivo = archivo.filename
            archivo.save(ruta_csv)
            # Guardar el nombre del archivo para no perderlo al recargar la página
            with open("uploads/metadata.txt", "w", encoding="utf-8") as f:
                f.write(nombre_archivo)
            # Borrar modelos y métricas anteriores para evitar inconsistencias con el nuevo dataset
            if os.path.exists("models"):
                for f_name in ["modelo_csv.pkl", "columnas.pkl", "target.pkl", "encoders.pkl", "encoder_target.pkl", "predictoras.pkl", "metrics.pkl", "coeficientes.pkl", "intercepto.pkl"]:
                    f_path = os.path.join("models", f_name)
                    if os.path.exists(f_path):
                        try:
                            os.remove(f_path)
                        except Exception:
                            pass


    if os.path.exists(ruta_csv):
        try:
            df = pd.read_csv(ruta_csv)
            filas = df.shape[0]
            columnas_n = df.shape[1]
            columnas_lista = df.columns.tolist()
            tabla = df.head(10).to_html(
                classes="table table-striped table-hover",
                index=False
            )
            if not nombre_archivo:
                if os.path.exists("uploads/metadata.txt"):
                    with open("uploads/metadata.txt", "r", encoding="utf-8") as f:
                        nombre_archivo = f.read().strip()
                else:
                    nombre_archivo = "dataset.csv"
        except Exception:
            pass

    return render_template(
        "archivo.html",
        tabla=tabla,
        filas=filas,
        columnas=columnas_n,
        columnas_lista=columnas_lista,
        nombre_archivo=nombre_archivo
    )



# ─────────────────────────────────────────────
# RUTA: Entrenar modelo con Dataset CSV
# ─────────────────────────────────────────────
@app.route("/entrenar_csv", methods=["GET", "POST"])
def entrenar_csv():
    """
    Entrena un clasificador de Regresión Logística sobre cualquier CSV cargado.
    Permite seleccionar la columna target y las variables predictoras.
    Detecta y codifica automáticamente columnas categóricas.
    """

    ruta_csv = "uploads/dataset.csv"

    if not os.path.exists(ruta_csv):
        return render_template("error.html", mensaje="Primero debes cargar un archivo CSV en la sección 'Cargar CSV'.")

    df = pd.read_csv(ruta_csv)
    columnas = df.columns.tolist()

    # Variables de resultado (None hasta que se entrene)
    accuracy = precision = recall = f1 = None
    n_train = n_test = None
    coeficientes = []
    intercepto = []
    variables_usadas = []
    target_usado = None

    # Intentar cargar métricas guardadas si ya existe un entrenamiento para la vista GET
    if request.method == "GET" and os.path.exists("models/metrics.pkl") and os.path.exists("models/modelo_csv.pkl"):
        try:
            metrics_saved = joblib.load("models/metrics.pkl")
            accuracy = metrics_saved.get("accuracy")
            precision = metrics_saved.get("precision")
            recall = metrics_saved.get("recall")
            f1 = metrics_saved.get("f1")
            n_train = metrics_saved.get("n_train")
            n_test = metrics_saved.get("n_test")

            target_usado = joblib.load("models/target.pkl") if os.path.exists("models/target.pkl") else None
            variables_usadas = joblib.load("models/predictoras.pkl") if os.path.exists("models/predictoras.pkl") else []

            modelo = joblib.load("models/modelo_csv.pkl")
            if modelo.coef_.ndim == 2:
                raw_coefs = modelo.coef_[0]
            else:
                raw_coefs = modelo.coef_

            coeficientes = []
            for var, val in zip(variables_usadas, raw_coefs):
                coeficientes.append({
                    "variable": var,
                    "coeficiente": round(float(val), 4),
                    "impacto": "alto" if abs(val) > 0.5 else "bajo"
                })
            coeficientes = sorted(coeficientes, key=lambda x: abs(x["coeficiente"]), reverse=True)

            if hasattr(modelo.intercept_, "tolist"):
                intercepto = [round(float(x), 4) for x in modelo.intercept_.tolist()]
            else:
                intercepto = round(float(modelo.intercept_), 4)
        except Exception:
            pass

    if request.method == "POST":


        # ─── 1. Obtener configuración del formulario ───
        target = request.form["target"]
        target_usado = target

        # Variables predictoras: checkboxes seleccionados por el usuario
        predictoras = request.form.getlist("features")
        if not predictoras:
            flash('Debes seleccionar al menos una columna predictora.', 'warning')
            return redirect(url_for('entrenar_csv'))

        # ─── 2. Limpieza: eliminar filas nulas en el target ───
        df_clean = df.dropna(subset=[target])

        # Preparar X usando solo las columnas elegidas
        X = df_clean[predictoras].copy()
        y = df_clean[target].copy()

        # Guardar la lista de columnas predictoras para usar en predicción
        joblib.dump(predictoras, 'models/predictoras.pkl')

        # ─── 3. Preprocesamiento de columnas ───
        encoders = {}

        for columna in X.columns:
            # Detectar columnas categóricas (texto, booleano, categoría)
            if (
                X[columna].dtype == "object"
                or X[columna].dtype == "category"
                or X[columna].dtype == "bool"
                or not pd.api.types.is_numeric_dtype(X[columna])
            ):
                # Rellenar nulos con "Unknown" y codificar con LabelEncoder
                X[columna] = X[columna].fillna("Unknown").astype(str)
                encoder = LabelEncoder()
                X[columna] = encoder.fit_transform(X[columna])
                encoders[columna] = encoder
            else:
                # Columna numérica: convertir y rellenar nulos con la mediana
                X[columna] = pd.to_numeric(X[columna], errors="coerce")
                mediana = X[columna].median()
                X[columna] = X[columna].fillna(mediana if not pd.isna(mediana) else 0.0)

        # ─── 4. Codificar target si es categórico o convertir si es continuo ───
        encoder_target = None
        from sklearn.utils.multiclass import type_of_target

        # Si el target es numérico, verificar si es continuo (float) o discreto (int)
        if pd.api.types.is_numeric_dtype(y):
            unique_vals = y.nunique()
            tot_vals    = len(y)
            y_type      = type_of_target(y.dropna())

            if y_type == "continuous":
                # Columna continua: convertir a categorías numéricas discretas usando rangos
                # Usamos 5 bins si hay muchos valores únicos, o el nº de únicos si hay pocos
                n_bins = min(5, unique_vals)
                try:
                    y = pd.cut(y, bins=n_bins, labels=False, duplicates='drop')
                    y = y.fillna(0).astype(int)
                    flash(
                        f'⚠️ La columna "{target}" tiene valores numéricos continuos. '
                        f'Se agrupó automáticamente en {n_bins} rangos/clases para poder aplicar clasificación. '
                        f'Para mejores resultados, elige una columna con categorías discretas.',
                        'warning'
                    )
                except Exception:
                    return render_template(
                        "error.html",
                        mensaje=(
                            f'La columna "{target}" tiene valores numéricos continuos y no pudo '
                            f'convertirse en clases. Por favor selecciona una columna con categorías '
                            f'o clases discretas (texto, booleano, o enteros con pocos valores únicos) '
                            f'como variable objetivo.'
                        )
                    )
        else:
            # Columna de texto/categoría: codificar con LabelEncoder
            encoder_target = LabelEncoder()
            y = encoder_target.fit_transform(y.fillna("Unknown").astype(str))

        # Validar que queden al menos 2 clases después del procesamiento
        if len(set(y)) < 2:
            return render_template(
                "error.html",
                mensaje=f'La columna "{target}" solo tiene una clase única. El modelo necesita al menos 2 clases distintas para clasificar.'
            )

        # ─── 5. División entrenamiento / prueba ───
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
        except Exception:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
        n_train = X_train.shape[0]
        n_test  = X_test.shape[0]

        # ─── 6. Entrenamiento del modelo ───
        modelo = LogisticRegression(max_iter=1000)
        modelo.fit(X_train, y_train)


        # Guardar coeficientes e intercepto para visualización
        joblib.dump(modelo.coef_, 'models/coeficientes.pkl')
        joblib.dump(modelo.intercept_, 'models/intercepto.pkl')

        predicciones = modelo.predict(X_test)

        # ─── 7. Persistencia del modelo y metadatos ───
        os.makedirs("models", exist_ok=True)
        joblib.dump(modelo,          "models/modelo_csv.pkl")
        joblib.dump(X.columns.tolist(), "models/columnas.pkl")
        joblib.dump(target,          "models/target.pkl")
        joblib.dump(encoders,        "models/encoders.pkl")
        joblib.dump(encoder_target,  "models/encoder_target.pkl")

        # ─── 8. Métricas de evaluación ───
        accuracy  = round(accuracy_score(y_test, predicciones) * 100, 2)
        precision = round(precision_score(y_test, predicciones, average="weighted", zero_division=0) * 100, 2)
        recall    = round(recall_score(y_test, predicciones, average="weighted", zero_division=0) * 100, 2)
        f1        = round(f1_score(y_test, predicciones, average="weighted", zero_division=0) * 100, 2)

        # ─── 9. Guardar métricas para persistir entre recargas ───
        joblib.dump({
            "accuracy":  accuracy,
            "precision": precision,
            "recall":    recall,
            "f1":        f1,
            "n_train":   n_train,
            "n_test":    n_test,
        }, "models/metrics.pkl")


        # ─── 10. Matriz de confusión ───
        matriz = confusion_matrix(y_test, predicciones)

        # Etiquetas de clases
        if encoder_target is not None:
            clases_labels = encoder_target.classes_
        else:
            clases_labels = sorted(list(set(list(y_test) + list(predicciones))))

        matplotlib_lock.acquire()
        try:
            plt.figure(figsize=(max(5, len(clases_labels)), max(4, len(clases_labels) - 1)))
            plt.imshow(matriz, interpolation='nearest', cmap=plt.cm.Blues)
            plt.title("Matriz de Confusión")
            plt.colorbar()

            if len(clases_labels) <= 12:
                tick_marks = range(len(clases_labels))
                plt.xticks(tick_marks, clases_labels, rotation=45, ha='right')
                plt.yticks(tick_marks, clases_labels)

            plt.xlabel("Predicción")
            plt.ylabel("Valor Real")

            if len(matriz) <= 15:
                for i in range(len(matriz)):
                    for j in range(len(matriz)):
                        plt.text(
                            j, i, str(matriz[i, j]),
                            ha="center", va="center",
                            color="white" if matriz[i, j] > matriz.max() / 2.0 else "black"
                        )

            os.makedirs("static", exist_ok=True)
            plt.savefig("static/matriz_csv.png", bbox_inches='tight')
            plt.close()

            # ─── GRÁFICA 1: Curva Sigmoide para CSV ───
            try:
                z = np.linspace(-8, 8, 300)
                sigmoid = 1 / (1 + np.exp(-z))
                plt.figure(figsize=(7, 4))
                plt.plot(z, sigmoid, color='#4f46e5', linewidth=2.5, label='σ(z) = 1 / (1 + e⁻ᶻ)')
                plt.axhline(0.5, color='#ef4444', linestyle='--', linewidth=1.5, label='Umbral de decisión (0.5)')
                plt.axvline(0, color='#94a3b8', linestyle=':', linewidth=1)
                plt.fill_between(z, sigmoid, 0.5, where=(sigmoid > 0.5), alpha=0.12, color='#4f46e5', label='Zona clase positiva')
                plt.fill_between(z, sigmoid, 0.5, where=(sigmoid < 0.5), alpha=0.12, color='#ef4444', label='Zona clase negativa')
                plt.xlabel('z = β₀ + β₁x₁ + ... + βₙxₙ', fontsize=11)
                plt.ylabel('Probabilidad P(y=1)', fontsize=11)
                plt.title('Función Sigmoide — Regresión Logística', fontsize=12, fontweight='bold')
                plt.legend(fontsize=9)
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig("static/grafica_sigmoide_csv.png", bbox_inches='tight', dpi=120)
                plt.close()
            except Exception as e:
                plt.close()
                print(f"Error generando curva sigmoide CSV: {e}")

            # ─── GRÁFICA 2: Dispersión de Clases para CSV ───
            try:
                plt.figure(figsize=(7, 5))
                feat_x = predictoras[0]
                feat_y = predictoras[1] if len(predictoras) > 1 else predictoras[0]
                
                unique_classes = sorted(list(set(y)))
                colores_disp = ['#f97316', '#4f46e5', '#10b981', '#a855f7', '#ec4899', '#06b6d4', '#eab308']
                
                for idx_c, cls_val in enumerate(unique_classes):
                    idx = (y == cls_val)
                    color = colores_disp[idx_c % len(colores_disp)]
                    
                    if encoder_target is not None:
                        nombre_clase = encoder_target.inverse_transform([cls_val])[0]
                    else:
                        nombre_clase = f"Clase {cls_val}"
                    
                    if len(predictoras) == 1:
                        x_vals = X.loc[idx, feat_x].values
                        y_vals = np.array([cls_val] * len(x_vals)) + np.random.normal(0, 0.08, len(x_vals))
                        plt.scatter(x_vals, y_vals, c=color, label=str(nombre_clase), alpha=0.75,
                                    edgecolors='white', linewidth=0.5, s=65)
                    else:
                        plt.scatter(X.loc[idx, feat_x].values, X.loc[idx, feat_y].values, c=color, label=str(nombre_clase), alpha=0.75,
                                    edgecolors='white', linewidth=0.5, s=65)
                
                plt.xlabel(feat_x, fontsize=11)
                plt.ylabel(feat_y if len(predictoras) > 1 else f"{target} (con jitter)", fontsize=11)
                plt.title(f'Dispersión de Clases — CSV\n({feat_x} vs {feat_y if len(predictoras) > 1 else target})', fontsize=12, fontweight='bold')
                plt.legend(fontsize=9)
                plt.grid(True, alpha=0.25)
                plt.tight_layout()
                plt.savefig("static/grafica_dispersion_csv.png", bbox_inches='tight', dpi=120)
                plt.close()
            except Exception as e:
                plt.close()
                print(f"Error generando gráfica de dispersión CSV: {e}")

            # ─── GRÁFICA 3: Curva de Aprendizaje para CSV ───
            try:
                train_sizes = np.linspace(0.1, 1.0, 10)
                train_acc_vals, test_acc_vals, sizes_label = [], [], []
                for size in train_sizes:
                    n = max(int(size * len(X_train)), 6)
                    n = min(n, len(X_train))
                    if n < 2:
                        continue
                    
                    try:
                        m_tmp = LogisticRegression(max_iter=1000)
                        m_tmp.fit(X_train[:n], y_train[:n])
                        train_acc_vals.append(accuracy_score(y_train[:n], m_tmp.predict(X_train[:n])) * 100)
                        test_acc_vals.append(accuracy_score(y_test, m_tmp.predict(X_test)) * 100)
                        sizes_label.append(n)
                    except Exception:
                        if train_acc_vals:
                            train_acc_vals.append(train_acc_vals[-1])
                            test_acc_vals.append(test_acc_vals[-1])
                            sizes_label.append(n)
                
                if len(sizes_label) >= 2:
                    plt.figure(figsize=(7, 4))
                    plt.plot(sizes_label, train_acc_vals, 'o-', color='#4f46e5', linewidth=2, markersize=6, label='Accuracy Entrenamiento')
                    plt.plot(sizes_label, test_acc_vals,  's--', color='#10b981', linewidth=2, markersize=6, label='Accuracy Prueba')
                    plt.fill_between(sizes_label, train_acc_vals, test_acc_vals, alpha=0.08, color='#f97316', label='Brecha generalización')
                    plt.xlabel('Número de muestras de entrenamiento', fontsize=11)
                    plt.ylabel('Accuracy (%)', fontsize=11)
                    plt.title('Curva de Aprendizaje — Regresión Logística (CSV)', fontsize=12, fontweight='bold')
                    plt.legend(fontsize=9)
                    plt.grid(True, alpha=0.3)
                    plt.tight_layout()
                    plt.savefig("static/grafica_aprendizaje_csv.png", bbox_inches='tight', dpi=120)
                    plt.close()
            except Exception as e:
                plt.close()
                print(f"Error generando curva de aprendizaje CSV: {e}")
        finally:
            matplotlib_lock.release()

        # Procesar coeficientes para visualización educativa
        if modelo.coef_.ndim == 2:
            raw_coefs = modelo.coef_[0]
        else:
            raw_coefs = modelo.coef_

        coef_list = []
        for var, val in zip(predictoras, raw_coefs):
            coef_list.append({
                "variable": var,
                "coeficiente": round(float(val), 4),
                "impacto": "alto" if abs(val) > 0.5 else "bajo"
            })
        coef_list = sorted(coef_list, key=lambda x: abs(x["coeficiente"]), reverse=True)

        if hasattr(modelo.intercept_, "tolist"):
            intercepto_val = [round(float(x), 4) for x in modelo.intercept_.tolist()]
        else:
            intercepto_val = round(float(modelo.intercept_), 4)

        # Renderizar resultados con información adicional
        return render_template(
            "entrenar_csv.html",
            columnas=columnas,
            variables_usadas=predictoras,
            target_usado=target,
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1=f1,
            n_train=n_train,
            n_test=n_test,
            coeficientes=coef_list,
            intercepto=intercepto_val
        )


    return render_template(
        "entrenar_csv.html",
        columnas=columnas,
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1=f1,
        n_train=n_train,
        n_test=n_test,
        coeficientes=coeficientes,
        intercepto=intercepto,
        variables_usadas=variables_usadas,
        target_usado=target_usado
    )


# ─────────────────────────────────────────────
# RUTA: Predicción dinámica con el modelo CSV
# ─────────────────────────────────────────────
@app.route("/prediccion", methods=["GET", "POST"])
def prediccion():
    """
    Formulario dinámico de predicción. Genera inputs automáticamente
    según las columnas del modelo entrenado. Muestra la clase predicha
    y las probabilidades de pertenencia a cada clase.
    """

    resultado = None
    probabilidades = []

    if not os.path.exists("models/modelo_csv.pkl"):
        return render_template(
            "error.html",
            mensaje="Primero debes entrenar un modelo CSV en la sección 'Entrenar CSV'."
        )

    modelo        = joblib.load("models/modelo_csv.pkl")
    encoders      = joblib.load("models/encoders.pkl")
    encoder_target = joblib.load("models/encoder_target.pkl")
    predictoras   = joblib.load('models/predictoras.pkl') if os.path.exists('models/predictoras.pkl') else []

    if request.method == "POST":
        # Preparar datos de entrada en el mismo orden que durante entrenamiento
        datos = []
        for columna in predictoras:
            valor = request.form.get(columna, "")
            if columna in encoders:
                try:
                    valor = encoders[columna].transform([valor])[0]
                except Exception:
                    # Valor desconocido, usar primera clase
                    if len(encoders[columna].classes_) > 0:
                        valor = encoders[columna].transform([encoders[columna].classes_[0]])[0]
                    else:
                        valor = 0
            else:
                try:
                    valor = float(valor)
                except Exception:
                    valor = 0.0
            datos.append(valor)

        # Predicción de clase
        pred = modelo.predict([datos])[0]
        # Predicción de probabilidad
        probas = modelo.predict_proba([datos])[0]

        if encoder_target is not None:
            try:
                pred_label = encoder_target.inverse_transform([pred])[0]
            except Exception:
                pred_label = str(pred)
        else:
            pred_label = str(pred)

        resultado = pred_label

        # Obtener nombres de todas las clases
        if encoder_target is not None:
            classes_labels = encoder_target.classes_
        else:
            classes_labels = [str(c) for c in modelo.classes_]

        for class_label, prob_val in zip(classes_labels, probas):
            probabilidades.append({
                "clase": class_label,
                "porcentaje": round(float(prob_val) * 100, 2),
                "es_predicha": class_label == pred_label
            })
        
        # Ordenar de mayor a menor probabilidad
        probabilidades = sorted(probabilidades, key=lambda x: x["porcentaje"], reverse=True)

        return render_template(
            "prediccion.html",
            columnas=predictoras,
            encoders=encoders,
            resultado=resultado,
            probabilidades=probabilidades
        )

    return render_template(
        "prediccion.html",
        columnas=predictoras,
        encoders=encoders,
        resultado=resultado,
        probabilidades=probabilidades
    )



# ─────────────────────────────────────────────
# RUTA: Visualizar Matriz de Confusión - Iris
# ─────────────────────────────────────────────
@app.route("/matriz")
def matriz():
    return render_template("matriz.html")


# ─────────────────────────────────────────────
# RUTA: Visualizar Matriz de Confusión - CSV
# ─────────────────────────────────────────────
@app.route("/matriz_csv")
def matriz_csv():
    return render_template("matriz_csv.html")


# ─────────────────────────────────────────────
# RUTA: Módulo de IA Generativa
# ─────────────────────────────────────────────
@app.route("/ia_generativa")
def ia_generativa():
    """Página que documenta qué partes del proyecto fueron generadas con IA y qué aprendió el estudiante."""
    return render_template("ia_generativa.html")


# ─────────────────────────────────────────────
# RUTA: Acerca del Proyecto
# ─────────────────────────────────────────────
@app.route("/acerca")
def acerca():
    return render_template("acerca.html")


# ─────────────────────────────────────────────
# PUNTO DE ENTRADA
# ─────────────────────────────────────────────
if __name__ == "__main__":
    # Use host='0.0.0.0' para despliegue en la nube
    app.run(host='0.0.0.0', debug=True)
