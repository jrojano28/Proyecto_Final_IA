# utils/evaluation.py
"""Utility functions for evaluating a trained model.

- calculate_metrics: returns accuracy, precision, recall, f1.
- plot_confusion_matrix: saves a PNG of the confusion matrix.
"""
import os
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


def calculate_metrics(y_true, y_pred, average='weighted'):
    """Calculate common classification metrics.
    Returns a dict with accuracy, precision, recall, f1 (percentage).
    """
    return {
        'accuracy': round(accuracy_score(y_true, y_pred) * 100, 2),
        'precision': round(precision_score(y_true, y_pred, average=average, zero_division=0) * 100, 2),
        'recall': round(recall_score(y_true, y_pred, average=average, zero_division=0) * 100, 2),
        'f1': round(f1_score(y_true, y_pred, average=average, zero_division=0) * 100, 2),
    }


def plot_confusion_matrix(y_true, y_pred, classes, output_path='static/confusion.png'):
    """Generate and save a confusion matrix plot.
    Args:
        y_true, y_pred: true and predicted labels.
        classes: list of class labels (strings or ints).
        output_path: where to save the PNG.
    """
    cm = confusion_matrix(y_true, y_pred)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.figure(figsize=(max(5, len(classes)), max(4, len(classes) - 1)))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Matriz de Confusión')
    plt.colorbar()
    if len(classes) <= 12:
        tick_marks = range(len(classes))
        plt.xticks(tick_marks, classes, rotation=45, ha='right')
        plt.yticks(tick_marks, classes)
    plt.xlabel('Predicción')
    plt.ylabel('Valor Real')
    # Annotate cells
    if cm.shape[0] <= 15:
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, str(cm[i, j]),
                         ha='center', va='center',
                         color='white' if cm[i, j] > cm.max() / 2.0 else 'black')
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    return output_path
