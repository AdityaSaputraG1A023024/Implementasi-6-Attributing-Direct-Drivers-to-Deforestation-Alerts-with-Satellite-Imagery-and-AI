"""Menjalankan eksperimen RF-versus-CNN, ringkasan, dan visual laporan."""

from __future__ import annotations

import argparse
import json
import platform
import sys
from time import perf_counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import sklearn
import torch
import yaml
from sklearn.model_selection import train_test_split

from src.data_loader import (
    PROJECT_ROOT,
    load_config,
    load_eurosat,
    stratified_split_indices,
)
from src.evaluate import confidence_coverage, evaluate_predictions
from src.features import extract_image_features
from src.plot_results import create_figures
from src.train_cnn import predict_cnn, train_cnn_model
from src.train_rf import majority_predictor, train_rf_model


def _subset_indices(
    indices: np.ndarray, labels: np.ndarray, fraction: float, seed: int
) -> np.ndarray:
    """Memilih subset latih berstrata untuk setiap titik learning curve."""
    if fraction >= 1.0:
        return np.sort(indices)
    subset, _ = train_test_split(
        indices,
        train_size=fraction,
        random_state=seed,
        stratify=labels[indices],
    )
    return np.sort(subset)


def _rf_probabilities(model: Any, images: np.ndarray) -> tuple[np.ndarray, float]:
    """Menghitung fitur uji dan probabilitas RF sebagai satu inferensi terukur."""
    from time import perf_counter

    start = perf_counter()
    features = extract_image_features(images)
    observed = model.predict_proba(features)
    probabilities = np.zeros((len(images), len(CLASS_NAMES)), dtype=np.float64)
    probabilities[:, model.classes_.astype(int)] = observed
    return probabilities, perf_counter() - start


def _evaluate_model(
    model_name: str,
    labels: np.ndarray,
    probabilities: np.ndarray,
    seed: int,
    training_seconds: float,
    inference_seconds: float,
    parameter_count: int | float,
    split_name: str,
    training_fraction: float,
) -> dict[str, Any]:
    """Menyatukan hasil evaluator dan menyimpan tingkat data latih."""
    row = evaluate_predictions(
        model_name,
        labels,
        probabilities,
        CLASS_NAMES,
        inference_seconds,
        training_seconds,
        parameter_count,
        seed,
        split_name,
    )
    row["training_fraction"] = float(training_fraction)
    row.pop("predictions")
    row.pop("probabilities")
    return row


CLASS_NAMES: list[str] = []


def run_experiment(
    config: dict[str, Any],
    subset_size: int | None = None,
    seed_limit: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Melatih/evaluasi baseline, RF, CNN dan kurva data untuk beberapa seed."""
    global CLASS_NAMES
    dataset = load_eurosat(config, subset_size=subset_size)
    CLASS_NAMES = dataset.class_names
    settings = config["dataset"]
    experiment = config["experiment"]
    seeds = [int(seed) for seed in experiment["seeds"]]
    if seed_limit is not None:
        if seed_limit < 1:
            raise ValueError("--seeds harus minimal satu.")
        seeds = seeds[:seed_limit]
    fractions = [float(value) for value in experiment["learning_curve_fractions"]]
    fractions = sorted(set(fractions + [1.0]))
    result_rows: list[dict[str, Any]] = []
    coverage_rows: list[dict[str, Any]] = []
    split_metadata: list[dict[str, Any]] = []
    rf_settings = config["random_forest"]
    cnn_settings = config["cnn"]
    torch.set_num_threads(int(experiment["cpu_threads"]))

    for seed in seeds:
        splits = stratified_split_indices(
            dataset.labels,
            seed,
            float(settings["train_fraction"]),
            float(settings["validation_fraction"]),
            float(settings["test_fraction"]),
        )
        train_indices = splits["train"]
        validation_indices = splits["validation"]
        test_indices = splits["test"]
        split_metadata.append({
            "seed": seed,
            **{f"{name}_count": int(len(indices)) for name, indices in splits.items()},
            "train_ids": dataset.sample_ids[train_indices].tolist(),
            "validation_ids": dataset.sample_ids[validation_indices].tolist(),
            "test_ids": dataset.sample_ids[test_indices].tolist(),
        })
        x_test = dataset.images[test_indices]
        y_test = dataset.labels[test_indices]
        x_validation = dataset.images[validation_indices]
        y_validation = dataset.labels[validation_indices]

        # Baseline mayoritas hanya memakai label bagian latih.
        majority = majority_predictor(dataset.labels[train_indices])
        start = perf_counter()
        dummy_probabilities = majority.predict_proba(
            np.zeros((len(test_indices), 1))
        )
        dummy_inference = perf_counter() - start
        result_rows.append(_evaluate_model(
            "majority_baseline", y_test, dummy_probabilities, seed, 0.0, dummy_inference,
            0, "test", 1.0,
        ))

        for fraction in fractions:
            partial_indices = _subset_indices(
                train_indices, dataset.labels, fraction, seed + int(fraction * 1000)
            )
            x_train = dataset.images[partial_indices]
            y_train = dataset.labels[partial_indices]

            # Random Forest memakai ringkasan statistik pita yang sama antar split.
            rf, rf_times = train_rf_model(
                x_train, y_train, x_validation, rf_settings, seed
            )
            rf_probabilities, rf_inference = _rf_probabilities(rf, x_test)
            rf_row = _evaluate_model(
                "random_forest",
                y_test,
                rf_probabilities,
                seed,
                rf_times["feature_extraction_seconds"] + rf_times["training_seconds"],
                rf_inference,
                rf_times["parameter_count"],
                "test" if fraction == 1.0 else "learning_curve",
                fraction,
            )
            rf_row["feature_extraction_seconds"] = rf_times["feature_extraction_seconds"]
            rf_row["model_fit_seconds"] = rf_times["training_seconds"]
            rf_row["model_complexity_count"] = rf_times["model_complexity_count"]
            rf_row["tree_count"] = rf_times["tree_count"]
            rf_row["input_feature_count"] = rf_times["input_feature_count"]
            result_rows.append(rf_row)
            if fraction == 1.0:
                curve_row = rf_row.copy()
                curve_row["split"] = "learning_curve"
                result_rows.append(curve_row)
            if fraction == 1.0:
                coverage_rows.extend(confidence_coverage(
                    "random_forest",
                    y_test,
                    rf_probabilities,
                    [float(item) for item in experiment["confidence_thresholds"]],
                    seed,
                ))

            # CNN dan augmentasi hanya melihat subset data latih; validasi tetap terpisah.
            cnn_result = train_cnn_model(
                x_train,
                y_train,
                x_validation,
                y_validation,
                cnn_settings,
                len(dataset.class_names),
                seed,
            )
            cnn_probabilities, cnn_inference = predict_cnn(
                cnn_result.model, x_test, int(cnn_settings["batch_size"])
            )
            cnn_row = _evaluate_model(
                "cnn",
                y_test,
                cnn_probabilities,
                seed,
                cnn_result.training_seconds,
                cnn_inference,
                cnn_result.parameter_count,
                "test" if fraction == 1.0 else "learning_curve",
                fraction,
            )
            cnn_row["epochs_trained"] = cnn_result.epochs_trained
            cnn_row["best_epoch"] = cnn_result.best_epoch
            cnn_row["best_validation_loss"] = cnn_result.best_validation_loss
            cnn_row["model_complexity_count"] = cnn_result.parameter_count
            result_rows.append(cnn_row)
            if fraction == 1.0:
                curve_row = cnn_row.copy()
                curve_row["split"] = "learning_curve"
                result_rows.append(curve_row)
            if fraction == 1.0:
                coverage_rows.extend(confidence_coverage(
                    "cnn",
                    y_test,
                    cnn_probabilities,
                    [float(item) for item in experiment["confidence_thresholds"]],
                    seed,
                ))
            print(
                f"Seed {seed}, data latih {fraction:.0%}: "
                f"RF dan CNN selesai ({len(partial_indices)} citra latih)."
            )

    raw = pd.DataFrame(result_rows)
    coverage = pd.DataFrame(coverage_rows)
    summary_columns = [
        "accuracy", "precision_macro", "recall_macro", "f1_macro",
        "training_seconds", "inference_seconds", "parameter_count",
        "feature_extraction_seconds", "model_fit_seconds", "epochs_trained",
        "best_epoch", "best_validation_loss",
        "model_complexity_count", "tree_count", "input_feature_count",
    ]
    available_columns = [column for column in summary_columns if column in raw.columns]
    summary = raw.groupby(
        ["model", "split", "training_fraction"], dropna=False
    )[available_columns].agg(["mean", "std"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    summary = summary.reset_index()
    metadata = {
        **dataset.metadata,
        "python_version": sys.version.split()[0],
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "scikit_learn_version": sklearn.__version__,
        "pytorch_version": torch.__version__,
        "device": "cpu",
        "random_seeds_used": seeds,
        "split_fractions": {
            "train": float(settings["train_fraction"]),
            "validation": float(settings["validation_fraction"]),
            "test": float(settings["test_fraction"]),
        },
        "learning_curve_fractions": fractions,
        "rf_hyperparameters": rf_settings,
        "cnn_hyperparameters": cnn_settings,
        "cnn_parameter_count": int(raw.loc[raw["model"] == "cnn", "parameter_count"].iloc[0]),
        "split_indices": split_metadata,
        "class_names": dataset.class_names,
        "warning": (
            "EuroSAT RGB mengukur klasifikasi tutupan lahan, bukan atribusi pemicu "
            "deforestasi; hasil tidak dibandingkan dengan angka seminar."
        ),
    }
    return raw, coverage, summary, metadata


def _parse_args() -> argparse.Namespace:
    """Menerima subset dan pembatas seed untuk smoke test lokal."""
    parser = argparse.ArgumentParser(description="Bandingkan RF dan CNN pada EuroSAT RGB.")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config.yaml")
    parser.add_argument(
        "--subset-size",
        type=int,
        default=None,
        help="Jumlah citra untuk uji cepat; default diambil dari config.yaml.",
    )
    parser.add_argument("--seeds", type=int, default=None, help="Jumlah seed awal untuk uji cepat.")
    return parser.parse_args()


def main() -> None:
    """Menyimpan laporan mentah, ringkasan, metadata, dan grafik."""
    arguments = _parse_args()
    config = load_config(arguments.config)
    raw, coverage, summary, metadata = run_experiment(
        config, arguments.subset_size, arguments.seeds
    )
    report_dir = PROJECT_ROOT / "reports"
    figure_dir = report_dir / "figures"
    report_dir.mkdir(exist_ok=True)
    raw.drop(columns=["per_class_metrics", "confusion_matrix", "most_confused_pairs"]).to_csv(
        report_dir / "hasil_per_seed.csv", index=False, encoding="utf-8-sig"
    )
    summary.to_csv(report_dir / "ringkasan.csv", index=False, encoding="utf-8-sig")
    coverage.to_csv(report_dir / "cakupan_vs_akurasi.csv", index=False, encoding="utf-8-sig")
    # Detail string-JSON membantu melihat metrik per kelas dan kelas tertukar.
    raw[[
        "model", "split", "seed", "training_fraction", "per_class_metrics",
        "confusion_matrix", "most_confused_pairs",
    ]].to_csv(report_dir / "detail_evaluasi.csv", index=False, encoding="utf-8-sig")
    (report_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (report_dir / "dataset_metadata.json").write_text(
        json.dumps({key: value for key, value in metadata.items() if key != "split_indices"},
                   indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    create_figures(raw, coverage, metadata["class_names"], figure_dir)
    print("\nRingkasan laporan final:")
    final_rows = summary[
        (summary["split"] == "test") & (summary["training_fraction"] == 1.0)
    ]
    print(final_rows.to_string(index=False))
    print(f"Jumlah hasil evaluasi: {len(raw)}; metadata: {report_dir / 'metadata.json'}")


if __name__ == "__main__":
    main()
