# utils/training.py
"""Utility functions for training a Logistic Regression model.

The workflow mirrors the logic previously embedded in ``app.py`` but is
encapsulated in reusable functions so that the Flask routes stay thin and
the core machine‑learning pipeline can be unit‑tested.
"""

import os
from typing import Tuple, Dict, Any

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

from .preprocessing import impute_missing, encode_categorical


def train_model(
    csv_path: str,
    target_column: str,
    selected_features: list,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[LogisticRegression, Dict[str, Any]]:
    """Train a Logistic Regression model.

    Parameters
    ----------
    csv_path: str
        Path to the uploaded CSV file.
    target_column: str
        Name of the column to be used as the target.
    selected_features: list
        List of column names to be used as predictors.
    test_size: float, optional
        Fraction of data to reserve for testing. Default 0.2.
    random_state: int, optional
        Seed for reproducibility. Default 42.

    Returns
    -------
    model: LogisticRegression
        Trained model instance.
    artifacts: dict
        Dictionary containing auxiliary objects and metrics needed for later
        inference and reporting.
    """
    # Load raw data
    df_raw = pd.read_csv(csv_path)

    # Separate target and features
    y = df_raw[target_column]
    X = df_raw[selected_features]

    # Preprocess
    X_filled = impute_missing(X)
    X_enc, encoders = encode_categorical(X_filled)
    y_filled = impute_missing(pd.DataFrame(y))
    y_enc, encoder_target = encode_categorical(y_filled, {})

    # Train / test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_enc, y_enc.squeeze(), test_size=test_size, random_state=random_state, stratify=y_enc.squeeze()
    )

    # Model training
    modelo = LogisticRegression(max_iter=1000)
    modelo.fit(X_train, y_train)

    # Predictions
    pred_test = modelo.predict(X_test)

    # Metrics
    metrics = {
        "accuracy": round(accuracy_score(y_test, pred_test) * 100, 2),
        "precision": round(
            precision_score(y_test, pred_test, average="weighted", zero_division=0
        )
        * 100,
        "recall": round(
            recall_score(y_test, pred_test, average="weighted", zero_division=0
        )
        * 100,
        "f1": round(f1_score(y_test, pred_test, average="weighted", zero_division=0) * 100, 2),
        "n_train": len(X_train),
        "n_test": len(X_test),
    }

    # Save artifacts – paths are relative to project root for consistency with Flask code
    os.makedirs("models", exist_ok=True)
    joblib.dump(modelo, "models/modelo_csv.pkl")
    joblib.dump(selected_features, "models/predictoras.pkl")
    joblib.dump(encoders, "models/encoders.pkl")
    joblib.dump(encoder_target, "models/encoder_target.pkl")
    joblib.dump(metrics, "models/metrics.pkl")

    # Return model and a compact artifact dictionary for the caller (e.g., Flask route)
    artifacts = {
        "metrics": metrics,
        "encoders": encoders,
        "encoder_target": encoder_target,
        "selected_features": selected_features,
    }
    return modelo, artifacts
