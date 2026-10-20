"""Uji split tanpa kebocoran, bentuk fitur, model, dan determinisme seed."""

from __future__ import annotations

import unittest

import numpy as np
import torch

from src.data_loader import assert_disjoint_splits, stratified_split_indices
from src.evaluate import evaluate_predictions
from src.features import extract_image_features
from src.train_cnn import SmallCNN, set_seed
from src.train_rf import build_random_forest


class PipelineTests(unittest.TestCase):
    """Memastikan komponen inti klasifikasi dapat dipakai dan direproduksi."""

    def setUp(self) -> None:
        self.images = np.random.default_rng(23).integers(
            0, 256, size=(20, 64, 64, 3), dtype=np.uint8
        )
        self.labels = np.tile(np.arange(10), 10)

    def test_split_is_disjoint_and_stratified(self) -> None:
        split = stratified_split_indices(
            self.labels, seed=11, train_fraction=0.6,
            validation_fraction=0.2, test_fraction=0.2,
        )
        assert_disjoint_splits(split)
        self.assertEqual(sum(len(indices) for indices in split.values()), len(self.labels))
        for indices in split.values():
            class_counts = np.bincount(self.labels[indices], minlength=10)
            self.assertLessEqual(int(class_counts.max() - class_counts.min()), 1)

    def test_feature_shape_for_rgb_statistics_and_histograms(self) -> None:
        features = extract_image_features(self.images[:3], histogram_bins=8)
        self.assertEqual(features.shape, (3, 45))
        self.assertTrue(np.isfinite(features).all())

    def test_model_output_dimensions_and_rf_parameters(self) -> None:
        cnn = SmallCNN(class_count=10, base_channels=8)
        outputs = cnn(torch.zeros((2, 3, 64, 64)))
        self.assertEqual(tuple(outputs.shape), (2, 10))
        forest = build_random_forest(
            {
                "n_estimators": 5,
                "max_depth": 4,
                "min_samples_leaf": 1,
                "max_features": "sqrt",
                "n_jobs": 1,
            },
            seed=7,
        )
        self.assertEqual(forest.n_estimators, 5)
        self.assertEqual(forest.random_state, 7)

    def test_same_seed_produces_same_cnn_initialization(self) -> None:
        set_seed(19, cpu_threads=1)
        first = SmallCNN(class_count=10, base_channels=4).state_dict()
        set_seed(19, cpu_threads=1)
        second = SmallCNN(class_count=10, base_channels=4).state_dict()
        for key in first:
            self.assertTrue(torch.equal(first[key], second[key]), key)

    def test_random_forest_predictions_repeat_with_same_seed(self) -> None:
        features = extract_image_features(self.images)
        labels = np.tile(np.arange(10), 2)
        settings = {
            "n_estimators": 5,
            "max_depth": 3,
            "min_samples_leaf": 1,
            "max_features": "sqrt",
            "n_jobs": 1,
        }
        first = build_random_forest(settings, seed=31).fit(features, labels)
        second = build_random_forest(settings, seed=31).fit(features, labels)
        np.testing.assert_array_equal(first.predict(features), second.predict(features))

    def test_metrics_include_per_class_and_misclassification_pairs(self) -> None:
        probabilities = np.eye(10)[np.array([0, 2, 1, 3])]
        truth = np.array([0, 1, 2, 3])
        result = evaluate_predictions(
            "test", truth, probabilities, [f"kelas_{index}" for index in range(10)],
            inference_seconds=0.1, training_seconds=1.0, parameter_count=5, seed=1,
        )
        self.assertEqual(result["accuracy"], 0.5)
        self.assertEqual(result["f1_macro"], 0.2)
        self.assertIn("kelas_1", result["per_class_metrics"])
        self.assertTrue(result["most_confused_pairs"])


if __name__ == "__main__":
    unittest.main()
