"""Melatih baseline mayoritas dan Random Forest pada fitur citra teragregasi."""

from __future__ import annotations

from time import perf_counter
from typing import Any

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier

from src.features import extract_image_features


def build_random_forest(settings: dict[str, Any], seed: int) -> RandomForestClassifier:
    """Membuat Random Forest dengan parameter eksplisit dari config."""
    return RandomForestClassifier(
        n_estimators=int(settings["n_estimators"]),
        max_depth=settings.get("max_depth"),
        min_samples_leaf=int(settings["min_samples_leaf"]),
        max_features=settings["max_features"],
        n_jobs=int(settings["n_jobs"]),
        random_state=seed,
    )


def train_rf_model(
    train_images: np.ndarray,
    train_labels: np.ndarray,
    validation_images: np.ndarray,
    settings: dict[str, Any],
    seed: int,
) -> tuple[RandomForestClassifier, dict[str, float]]:
    """Melatih RF hanya pada fitur split latih dan mengukur waktu proses."""
    start = perf_counter()
    train_features = extract_image_features(train_images)
    validation_features = extract_image_features(validation_images)
    feature_seconds = perf_counter() - start
    model = build_random_forest(settings, seed)
    start = perf_counter()
    model.fit(train_features, train_labels)
    train_seconds = perf_counter() - start
    return model, {
        "feature_extraction_seconds": feature_seconds,
        "training_seconds": train_seconds,
        "parameter_count": float("nan"),
        "model_complexity_count": float(
            sum(tree.tree_.node_count for tree in model.estimators_)
        ),
        "tree_count": float(len(model.estimators_)),
        "input_feature_count": float(train_features.shape[1]),
    }


def majority_predictor(train_labels: np.ndarray) -> DummyClassifier:
    """Membuat baseline yang selalu memilih kelas terbanyak pada data latih."""
    return DummyClassifier(strategy="most_frequent").fit(
        np.zeros((len(train_labels), 1)), train_labels
    )
