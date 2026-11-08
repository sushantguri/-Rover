"""
Unit tests for UncertaintyEstimator and Shannon Entropy calculation.
"""
import unittest
import torch
import numpy as np

from src.uncertainty import UncertaintyEstimator


class TestUncertaintyEstimator(unittest.TestCase):
    def setUp(self):
        self.estimator = UncertaintyEstimator(high_uncertainty_threshold=0.65)

    def test_zero_entropy_on_confident_prediction(self):
        """When one class has 100% confidence, normalized entropy must be 0.0."""
        # Logits strongly favoring class 0
        logits = torch.tensor([[[[50.0]], [[-50.0]], [[-50.0]], [[-50.0]]]]) # (1, 4, 1, 1)
        ent = self.estimator.compute_entropy_map(logits)

        self.assertAlmostEqual(float(ent[0, 0]), 0.0, places=3)

    def test_maximum_entropy_on_uniform_logits(self):
        """When all 4 classes have equal logits (0.0), normalized entropy must be exactly 1.0."""
        logits = torch.zeros(1, 4, 2, 2)
        ent = self.estimator.compute_entropy_map(logits)

        self.assertAlmostEqual(float(ent[0, 0]), 1.0, places=3)
        self.assertAlmostEqual(float(np.mean(ent)), 1.0, places=3)

    def test_margin_map_bounds(self):
        """Margin uncertainty must stay bounded in [0.0, 1.0]."""
        logits = torch.randn(1, 4, 8, 8)
        margin = self.estimator.compute_margin_map(logits)

        self.assertGreaterEqual(float(np.min(margin)), 0.0)
        self.assertLessEqual(float(np.max(margin)), 1.0)

    def test_telemetry_high_ambiguity_flag(self):
        """High entropy across the frame must trigger HIGH_PERCEPTION_AMBIGUITY status."""
        uniform_entropy = np.ones((32, 32), dtype=np.float32)
        telem = self.estimator.evaluate_uncertainty_telemetry(uniform_entropy)

        self.assertEqual(telem["confidence_status"], "HIGH_PERCEPTION_AMBIGUITY")
        self.assertEqual(telem["high_uncertainty_pixel_pct"], 100.0)


if __name__ == "__main__":
    unittest.main()
