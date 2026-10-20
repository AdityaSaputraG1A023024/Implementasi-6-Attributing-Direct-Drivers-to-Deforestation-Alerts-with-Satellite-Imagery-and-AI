"""Mengunduh, membaca, dan membagi citra EuroSAT RGB secara terstratifikasi."""

from __future__ import annotations

import argparse
import io
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests
import yaml
from PIL import Image
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class EuroSATData:
    """Contoh citra, label, nama kelas, ID stabil, dan metadata dataset."""

    images: np.ndarray
    labels: np.ndarray
    class_names: list[str]
    sample_ids: np.ndarray
    metadata: dict[str, Any]


def load_config(path: Path | None = None) -> dict[str, Any]:
    """Membaca config YAML dari direktori proyek."""
    config_path = path or PROJECT_ROOT / "config.yaml"
    with config_path.open("r", encoding="utf-8-sig") as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict):
        raise ValueError(f"Konfigurasi tidak valid: {config_path}")
    return config


def ensure_dataset(config: dict[str, Any], force_download: bool = False) -> Path:
    """Mengambil arsip Parquet resmi mirror bila berkas belum ada."""
    settings = config["dataset"]
    archive = PROJECT_ROOT / settings["parquet_path"]
    if archive.is_file() and not force_download:
        return archive
    archive.parent.mkdir(parents=True, exist_ok=True)
    temporary = archive.with_suffix(archive.suffix + ".part")
    try:
        with requests.get(settings["parquet_url"], stream=True, timeout=(20, 120)) as response:
            response.raise_for_status()
            with temporary.open("wb") as output:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        output.write(chunk)
        temporary.replace(archive)
    except requests.RequestException as error:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(
            "Unduhan EuroSAT gagal. Periksa internet atau unduh arsip Parquet "
            f"dari {settings['source_url']} ke {archive}."
        ) from error
    return archive


def _decode_image(image_value: Any) -> np.ndarray:
    """Mendekode objek gambar Parquet dan memastikan RGB 8-bit."""
    if isinstance(image_value, dict):
        image_bytes = image_value.get("bytes")
        image_path = image_value.get("path")
        if image_bytes is not None:
            source: Any = io.BytesIO(image_bytes)
        elif image_path:
            source = image_path
        else:
            raise ValueError("Kolom citra tidak berisi bytes atau path.")
    elif isinstance(image_value, (bytes, bytearray, memoryview)):
        source = io.BytesIO(image_value)
    else:
        raise TypeError(f"Tipe kolom citra tidak didukung: {type(image_value).__name__}")
    with Image.open(source) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def load_eurosat(
    config: dict[str, Any],
    subset_size: int | None = None,
    force_download: bool = False,
) -> EuroSATData:
    """Memuat dataset RGB, memilih subset seimbang, dan mencatat metadata nyata."""
    import pyarrow.parquet as parquet

    archive = ensure_dataset(config, force_download)
    table = parquet.read_table(archive)
    frame = table.to_pandas()
    required = {"image", "label"}
    if not required.issubset(frame.columns):
        raise ValueError(
            f"Parquet harus memiliki kolom {sorted(required)}; ditemukan {list(frame.columns)}."
        )
    info_path = archive.parent.parent / "dataset_infos.json"
    # Nama label berasal dari metadata upstream, bukan urutan yang ditebak.
    info_url = "https://huggingface.co/datasets/nielsr/eurosat-demo/resolve/main/dataset_infos.json"
    if info_path.is_file():
        info = json.loads(info_path.read_text(encoding="utf-8"))
    else:
        try:
            info_response = requests.get(info_url, timeout=30)
            info_response.raise_for_status()
        except requests.RequestException as error:
            raise RuntimeError(
                "Metadata nama kelas tidak dapat diunduh dan tidak tersedia di cache."
            ) from error
        info_path.write_text(info_response.text, encoding="utf-8")
        info = json.loads(info_response.text)
    class_names = list(info["nielsr--eurosat-demo"]["features"]["label"]["names"])
    raw_labels = frame["label"].to_numpy(dtype=np.int64)
    if raw_labels.min(initial=0) < 0 or raw_labels.max(initial=0) >= len(class_names):
        raise ValueError("Label dalam Parquet di luar daftar kelas metadata.")

    # Subset dipilih berstrata agar semua kelas tetap terwakili.
    indices = np.arange(len(frame))
    configured_subset = int(
        subset_size if subset_size is not None else config["dataset"]["subset_size"]
    )
    if configured_subset < 1:
        raise ValueError("Ukuran subset harus lebih besar dari nol.")
    if configured_subset > len(indices):
        raise ValueError(
            f"Ukuran subset {configured_subset} melebihi jumlah citra {len(indices)}."
        )
    if configured_subset < len(indices):
        indices, _ = train_test_split(
            indices,
            train_size=configured_subset,
            random_state=int(config["dataset"]["seed"]),
            stratify=raw_labels,
        )
        indices = np.sort(indices)
    sampled_frame = frame.iloc[indices].reset_index(drop=True)
    images = np.stack([_decode_image(value) for value in sampled_frame["image"]])
    labels = raw_labels[indices]
    if images.ndim != 4 or images.shape[-1] != 3:
        raise ValueError(f"Dataset harus citra RGB NHWC; bentuk aktual {images.shape}.")

    counts = np.bincount(labels, minlength=len(class_names))
    metadata: dict[str, Any] = {
        "dataset": "EuroSAT RGB",
        "source": config["dataset"]["source"],
        "source_url": config["dataset"]["source_url"],
        "mirror_dataset_version": "nielsr/eurosat-demo (revision 61bab8317ee7f13bea1831f2ec3a7fa571a01a81)",
        "archive_name": archive.name,
        "archive_bytes": archive.stat().st_size,
        "original_image_count": int(len(frame)),
        "subset_image_count": int(len(images)),
        "subset_seed": int(config["dataset"]["seed"]),
        "image_height": int(images.shape[1]),
        "image_width": int(images.shape[2]),
        "bands": ["Red", "Green", "Blue"],
        "band_count": 3,
        "ndvi_available": False,
        "ndvi_note": "NDVI tidak dihitung karena versi RGB tidak memiliki pita NIR.",
        "class_counts_original": {
            class_names[index]: int((raw_labels == index).sum())
            for index in range(len(class_names))
        },
        "class_counts_subset": {
            class_names[index]: int(counts[index])
            for index in range(len(class_names))
        },
        "class_names": class_names,
        "image_mode": "RGB",
        "array_dtype": str(images.dtype),
    }
    return EuroSATData(
        images=images,
        labels=labels,
        class_names=class_names,
        sample_ids=indices.astype(np.int64),
        metadata=metadata,
    )


def stratified_split_indices(
    labels: np.ndarray,
    seed: int,
    train_fraction: float,
    validation_fraction: float,
    test_fraction: float,
) -> dict[str, np.ndarray]:
    """Membuat split stratified 70/15/15 dengan ID citra tak tumpang tindih."""
    fractions = np.array([train_fraction, validation_fraction, test_fraction])
    if not np.isclose(fractions.sum(), 1.0) or np.any(fractions <= 0):
        raise ValueError("Proporsi split harus positif dan jumlahnya 1.")
    indices = np.arange(len(labels))
    train_indices, remainder = train_test_split(
        indices,
        train_size=train_fraction,
        random_state=seed,
        stratify=labels,
    )
    validation_relative = validation_fraction / (validation_fraction + test_fraction)
    validation_indices, test_indices = train_test_split(
        remainder,
        train_size=validation_relative,
        random_state=seed + 1,
        stratify=labels[remainder],
    )
    splits = {
        "train": np.sort(train_indices),
        "validation": np.sort(validation_indices),
        "test": np.sort(test_indices),
    }
    assert_disjoint_splits(splits)
    return splits


def assert_disjoint_splits(splits: dict[str, np.ndarray]) -> None:
    """Menggagalkan alur bila satu ID citra muncul di lebih dari satu bagian."""
    names = list(splits)
    for first_index, first_name in enumerate(names):
        for second_name in names[first_index + 1 :]:
            overlap = np.intersect1d(splits[first_name], splits[second_name])
            if overlap.size:
                raise ValueError(
                    f"Kebocoran citra: split {first_name}/{second_name} "
                    f"berbagi {overlap.size} ID."
                )


def main() -> None:
    """Menyediakan perintah eksplisit untuk mengunduh dan memeriksa metadata."""
    parser = argparse.ArgumentParser(description="Unduh dan periksa EuroSAT RGB.")
    parser.add_argument("--download", action="store_true", help="Unduh dataset sekarang.")
    parser.add_argument("--subset-size", type=int, default=None)
    args = parser.parse_args()
    config = load_config()
    data = load_eurosat(config, subset_size=args.subset_size, force_download=args.download)
    reports = PROJECT_ROOT / "reports"
    reports.mkdir(exist_ok=True)
    output = reports / "dataset_metadata.json"
    output.write_text(json.dumps(data.metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(data.metadata, indent=2, ensure_ascii=False))
    print(f"Metadata disimpan: {output}")


if __name__ == "__main__":
    main()
