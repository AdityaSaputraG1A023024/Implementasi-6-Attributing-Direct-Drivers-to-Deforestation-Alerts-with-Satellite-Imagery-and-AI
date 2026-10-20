"""Evaluasi metrik per kelas, biaya inferensi, kelas tertukar, dan abstensi."""

from __future__ import annotations

import json
from time import perf_counter
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)


def evaluate_predictions(
    model_name: str,
    truth: np.ndarray,
    probabilities: np.ndarray,
    class_names: list[str],
    inference_seconds: float,
    training_seconds: float,
    parameter_count: int | float,
    seed: int,
    split_name: str = "test",
) -> dict[str, Any]:
    """Menghitung metrik per kelas, makro, confusion matrix, dan pasangan salah."""
    predictions = probabilities.argmax(axis=1)
    labels = np.arange(len(class_names))
    precision, recall, f1, support = precision_recall_fscore_support(
        truth, predictions, labels=labels, zero_division=0
    )
    matrix = confusion_matrix(truth, predictions, labels=labels)
    off_diagonal = matrix.copy()
    np.fill_diagonal(off_diagonal, 0)
    pairs: list[dict[str, Any]] = []
    for true_id, predicted_id in np.argwhere(off_diagonal > 0):
        pairs.append(
            {
                "actual": class_names[int(true_id)],
                "predicted": class_names[int(predicted_id)],
                "count": int(matrix[true_id, predicted_id]),
            }
        )
    pairs.sort(key=lambda pair: pair["count"], reverse=True)
    per_class = {
        class_names[index]: {
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(f1[index]),
            "support": int(support[index]),
        }
        for index in labels
    }
    return {
        "model": model_name,
        "split": split_name,
        "seed": int(seed),
        "accuracy": float(accuracy_score(truth, predictions)),
        "precision_macro": float(precision.mean()),
        "recall_macro": float(recall.mean()),
        "f1_macro": float(f1.mean()),
        "training_seconds": float(training_seconds),
        "inference_seconds": float(inference_seconds),
        "parameter_count": (
            int(parameter_count)
            if np.isfinite(float(parameter_count))
            else None
        ),
        "per_class_metrics": json.dumps(per_class, ensure_ascii=False),
        "confusion_matrix": json.dumps(matrix.tolist()),
        "most_confused_pairs": json.dumps(pairs[:10], ensure_ascii=False),
        "predictions": predictions,
        "probabilities": probabilities,
    }


def confidence_coverage(
    model_name: str,
    truth: np.ndarray,
    probabilities: np.ndarray,
    thresholds: list[float],
    seed: int,
) -> list[dict[str, Any]]:
    """Mengukur coverage dan accuracy di antara prediksi yang diterima."""
    confidence = probabilities.max(axis=1)
    predictions = probabilities.argmax(axis=1)
    rows: list[dict[str, Any]] = []
    for threshold in thresholds:
        accepted = confidence >= threshold
        rows.append(
            {
                "model": model_name,
                "seed": int(seed),
                "threshold": float(threshold),
                "coverage": float(accepted.mean()),
                "selective_accuracy": (
                    float(accuracy_score(truth[accepted], predictions[accepted]))
                    if accepted.any()
                    else float("nan")
                ),
                "accepted_count": int(accepted.sum()),
                "total_count": int(len(truth)),
            }
        )
    return rows


def timed_predict(function: Any, *args: Any, **kwargs: Any) -> tuple[np.ndarray, float]:
    """Mengukur inferensi satu batch/model dan mengembalikan skor keluaran."""
    start = perf_counter()
    probabilities = function(*args, **kwargs)
    return np.asarray(probabilities), perf_counter() - start
