"""Ekstraksi fitur tabular sederhana dari pita RGB untuk Random Forest."""

from __future__ import annotations

import numpy as np


def extract_image_features(images: np.ndarray, histogram_bins: int = 8) -> np.ndarray:
    """Menghitung statistik per pita dan histogram RGB yang dinormalisasi."""
    if images.ndim != 4 or images.shape[-1] not in (3, 13):
        raise ValueError("Masukan harus berbentuk NHWC dengan 3 pita RGB atau 13 pita.")
    pixels = images.astype(np.float32) / 255.0
    features: list[np.ndarray] = []
    for image in pixels:
        per_band: list[float] = []
        for band in range(image.shape[-1]):
            values = image[..., band].reshape(-1)
            per_band.extend(
                [
                    float(values.mean()),
                    float(values.std()),
                    *np.quantile(values, [0.10, 0.25, 0.50, 0.75, 0.90]).astype(float),
                ]
            )
            histogram, _ = np.histogram(
                values, bins=histogram_bins, range=(0.0, 1.0), density=False
            )
            per_band.extend((histogram / histogram.sum().clip(min=1)).astype(float))
        # Jangan menghitung NDVI tanpa pita NIR yang tersedia.
        features.append(np.asarray(per_band, dtype=np.float32))
    return np.stack(features)
