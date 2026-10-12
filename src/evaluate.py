"""Evaluation utilities for the oral lesion project.

This module provides the evaluation skeleton for metrics, confusion matrices, and reports.
It is intentionally prepared before the actual dataset is available so the experiment flow is
ready when the data arrives.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn import metrics


def compute_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, labels: list[str] | None = None):
    """Compute standard classification metrics."""
    if labels is None:
        labels = sorted(set(y_true).union(set(y_pred)))

    accuracy = metrics.accuracy_score(y_true, y_pred)
    precision = metrics.precision_score(y_true, y_pred, average="macro", zero_division=0)
    recall = metrics.recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1 = metrics.f1_score(y_true, y_pred, average="macro", zero_division=0)
    report = metrics.classification_report(y_true, y_pred, target_names=labels, zero_division=0)
    confusion = metrics.confusion_matrix(y_true, y_pred, labels=list(range(len(labels))))
    return {
        "accuracy": accuracy,
        "precision_macro": precision,
        "recall_macro": recall,
        "f1_macro": f1,
        "classification_report": report,
        "confusion_matrix": confusion,
    }


def save_confusion_matrix(cm: np.ndarray, labels: list[str], output_path: str | Path):
    """Save a confusion matrix plot."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(output, dpi=300)
    plt.close()


if __name__ == "__main__":
    print("Evaluation utilities are ready for the actual test set after dataset validation.")
