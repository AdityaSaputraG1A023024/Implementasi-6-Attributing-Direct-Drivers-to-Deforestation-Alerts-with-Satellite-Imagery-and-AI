"""Membuat grafik laporan dari keluaran eksperimen yang benar-benar dijalankan."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def _style() -> None:
    """Menetapkan gaya grafik yang sederhana dan konsisten."""
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams["figure.dpi"] = 120


def plot_confusion_matrices(
    results: pd.DataFrame,
    class_names: list[str],
    output_dir: Path,
) -> None:
    """Menggabungkan confusion matrix model pada rerata seed di dataset penuh."""
    _style()
    models = [name for name in ("random_forest", "cnn") if name in results.model.unique()]
    figure, axes = plt.subplots(1, len(models), figsize=(7 * len(models), 6), squeeze=False)
    for axis, model_name in zip(axes[0], models):
        rows = results[
            (results["model"] == model_name)
            & (results["split"] == "test")
            & (results["training_fraction"] == 1.0)
        ]
        matrices = [np.asarray(json.loads(value), dtype=float) for value in rows["confusion_matrix"]]
        mean_matrix = np.mean(matrices, axis=0)
        sns.heatmap(
            mean_matrix,
            ax=axis,
            cmap="Blues",
            annot=True,
            fmt=".1f",
            xticklabels=class_names,
            yticklabels=class_names,
        )
        axis.set_title(f"Confusion matrix rata-rata — {model_name}")
        axis.set_xlabel("Prediksi")
        axis.set_ylabel("Label sebenarnya")
        axis.tick_params(axis="x", rotation=45)
        axis.tick_params(axis="y", rotation=0)
    figure.tight_layout()
    figure.savefig(output_dir / "confusion_matrix.png", bbox_inches="tight")
    plt.close(figure)


def plot_f1_per_class(results: pd.DataFrame, class_names: list[str], output_dir: Path) -> None:
    """Menampilkan F1 tiap kelas dengan simpangan baku lintas seed."""
    _style()
    records: list[dict[str, Any]] = []
    rows = results[
        (results["split"] == "test")
        & (results["training_fraction"] == 1.0)
        & (results["model"].isin(["random_forest", "cnn"]))
    ]
    for _, row in rows.iterrows():
        per_class = json.loads(row["per_class_metrics"])
        for class_name in class_names:
            records.append(
                {
                    "model": row["model"],
                    "kelas": class_name,
                    "F1": per_class[class_name]["f1"],
                    "seed": row["seed"],
                }
            )
    frame = pd.DataFrame(records)
    summary = frame.groupby(["model", "kelas"], as_index=False).agg(
        F1=("F1", "mean"), simpangan_baku=("F1", "std")
    )
    figure, axis = plt.subplots(figsize=(13, 6))
    sns.barplot(data=summary, x="kelas", y="F1", hue="model", ax=axis, errorbar=None)
    axis.set_ylim(0, 1)
    axis.set_xlabel("Kelas tutupan lahan")
    axis.set_ylabel("F1 rata-rata")
    axis.set_title("F1 per kelas — dataset penuh")
    axis.tick_params(axis="x", rotation=35)
    figure.tight_layout()
    figure.savefig(output_dir / "f1_per_kelas.png", bbox_inches="tight")
    plt.close(figure)


def plot_learning_curve(results: pd.DataFrame, output_dir: Path) -> None:
    """Membandingkan F1 makro pada fraksi data latih yang berbeda."""
    _style()
    rows = results[
        (results["split"] == "learning_curve")
        & results["model"].isin(["random_forest", "cnn"])
    ]
    summary = rows.groupby(["model", "training_fraction"], as_index=False).agg(
        F1=("f1_macro", "mean"), simpangan_baku=("f1_macro", "std")
    )
    figure, axis = plt.subplots(figsize=(9, 5))
    sns.lineplot(
        data=summary,
        x="training_fraction",
        y="F1",
        hue="model",
        marker="o",
        ax=axis,
    )
    axis.set_xticks(sorted(summary["training_fraction"].unique()))
    axis.set_xlabel("Proporsi data latih yang dipakai")
    axis.set_ylabel("F1 makro uji")
    axis.set_ylim(0, 1)
    axis.set_title("Kurva ukuran data latih")
    figure.tight_layout()
    figure.savefig(output_dir / "kurva_data_latih.png", bbox_inches="tight")
    plt.close(figure)


def plot_coverage_curve(coverage: pd.DataFrame, output_dir: Path) -> None:
    """Menampilkan akurasi pada prediksi diterima terhadap coverage."""
    _style()
    summary = coverage.groupby(["model", "threshold"], as_index=False).agg(
        coverage=("coverage", "mean"),
        accuracy=("selective_accuracy", "mean"),
        accuracy_std=("selective_accuracy", "std"),
    )
    figure, axis = plt.subplots(figsize=(8, 5))
    sns.lineplot(
        data=summary,
        x="coverage",
        y="accuracy",
        hue="model",
        marker="o",
        ax=axis,
    )
    axis.set_xlabel("Cakupan prediksi yang diterima")
    axis.set_ylabel("Akurasi pada prediksi yang diterima")
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.set_title("Usulan tambahan: keyakinan versus cakupan")
    figure.tight_layout()
    figure.savefig(output_dir / "cakupan_vs_akurasi.png", bbox_inches="tight")
    plt.close(figure)


def create_figures(
    results: pd.DataFrame,
    coverage: pd.DataFrame,
    class_names: list[str],
    output_dir: Path,
) -> None:
    """Membuat semua grafik laporan dalam direktori keluaran."""
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_confusion_matrices(results, class_names, output_dir)
    plot_f1_per_class(results, class_names, output_dir)
    plot_learning_curve(results, output_dir)
    plot_coverage_curve(coverage, output_dir)
