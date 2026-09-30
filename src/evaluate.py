"""Shared evaluation helpers: metrics, confusion matrices, comparison plots."""

import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # no GUI needed - save figures to files
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix,
)

RESULTS_DIR = "results"
CM_DIR = os.path.join(RESULTS_DIR, "confusion_matrices")
PLOT_DIR = os.path.join(RESULTS_DIR, "plots")


def evaluate_model(y_true, y_pred):
    """Accuracy + macro-averaged precision/recall/F1 (fair for 4 balanced classes)."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
    }


def evaluate_all(models, X_valid, y_valid):
    """Evaluate every model on the SAME validation set. Returns comparison DataFrame."""
    rows = []
    for name, model in models.items():
        y_pred = model.predict(X_valid)
        m = evaluate_model(y_valid, y_pred)
        print(f"\n--- {name} ---")
        print(classification_report(y_valid, y_pred, digits=4))
        rows.append({"model": name, "accuracy": m["accuracy"],
                     "precision": m["precision"], "recall": m["recall"], "f1": m["f1"]})
    return pd.DataFrame(rows, columns=["model", "accuracy", "precision", "recall", "f1"])


def save_confusion_matrices(models, X_valid, y_valid):
    """Save one confusion-matrix heatmap PNG per model."""
    os.makedirs(CM_DIR, exist_ok=True)
    for name, model in models.items():
        y_pred = model.predict(X_valid)
        cm = confusion_matrix(y_valid, y_pred, labels=[0, 1, 2, 3])
        plt.figure(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=[0, 1, 2, 3], yticklabels=[0, 1, 2, 3])
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title(f"Confusion Matrix - {name}")
        path = os.path.join(CM_DIR, f"{name}.png")
        plt.tight_layout()
        plt.savefig(path)
        plt.close()
        print(f"Saved {path}")


def save_comparison_plots(comparison):
    """Bar charts comparing accuracy / precision / recall / F1 across models."""
    os.makedirs(PLOT_DIR, exist_ok=True)
    for metric in ["accuracy", "precision", "recall", "f1"]:
        plt.figure(figsize=(8, 4))
        sns.barplot(data=comparison, x="model", y=metric)
        plt.xticks(rotation=15)
        plt.ylim(0, 1)
        plt.title(f"Model comparison - {metric}")
        path = os.path.join(PLOT_DIR, f"{metric}_comparison.png")
        plt.tight_layout()
        plt.savefig(path)
        plt.close()
        print(f"Saved {path}")
