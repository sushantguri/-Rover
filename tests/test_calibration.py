"""
Unit tests for Temperature-Scaled Probability Calibration & Conformal Safety Margin Engine.
"""
import unittest
import numpy as np
import torch

from src.calibration import ProbabilityCalibrator


class TestProbabilityCalibrator(unittest.TestCase):
    def setUp(self):
        self.calibrator = ProbabilityCalibrator(temperature=1.25, conformal_sand_threshold=0.25)

    def test_temperature_scaling_softmax_sums_to_one(self):
        """Calibrated probabilities must strictly sum to 1.0 across all 4 classes."""
        logits = torch.randn(1, 4, 32, 32)
        probs = self.calibrator.calibrate_probs(logits)
        self.assertEqual(probs.shape, (4, 32, 32))
        np.testing.assert_allclose(probs.sum(axis=0), np.ones((32, 32), dtype=np.float32), atol=1e-5)

    def test_borderline_sand_escalation_catches_subthreshold_ripples(self):
        """Ambiguous pixels where Bedrock barely beats Sand must be flagged as borderline sand."""
        h, w = 40, 40
        logits = torch.zeros(1, 4, h, w)
        # Bedrock marginally higher than Sand (diff=0.25 -> after T=1.25, P(sand) > 0.30)
        logits[0, 1, :, :] = 1.0
        logits[0, 2, :, :] = 0.75
        logits[0, 0, :, :] = -2.0
        logits[0, 3, :, :] = -2.0

        pred_mask = np.ones((h, w), dtype=np.int64)  # Model predicted all Bedrock (1)
        summary, borderline_mask = self.calibrator.evaluate_safety_margin(
            logits, pred_mask, sand_threshold_pct=25.0
        )

        self.assertEqual(summary["raw_sand_pct"], 0.0)
        self.assertGreater(summary["borderline_sand_pct"], 90.0)
        self.assertEqual(summary["conformal_alert"], "CONSERVATIVE_SAND_ESCALATION")
        self.assertTrue(np.all(borderline_mask))

    def test_ece_perfect_predictions_low_error(self):
        """High-confidence accurate predictions must exhibit near-zero ECE."""
        probs = np.zeros((4, 16, 16), dtype=np.float32)
        probs[0, :, :] = 0.99
        probs[1, :, :] = 0.01
        targets = np.zeros((16, 16), dtype=np.int64)
        ece = self.calibrator.compute_ece(probs, targets)
        self.assertLess(ece, 0.02)

    def test_safety_margin_overlay_preserves_rgb_shape(self):
        """Safety margin overlay renders valid RGB uint8 image matching input shape."""
        img = np.random.randint(60, 180, (32, 32, 3), dtype=np.uint8)
        pred_mask = np.ones((32, 32), dtype=np.int64)
        borderline = np.zeros((32, 32), dtype=bool)
        borderline[:10, :10] = True
        overlay = self.calibrator.render_safety_margin_overlay(img, pred_mask, borderline)
        self.assertEqual(overlay.shape, (32, 32, 3))
        self.assertEqual(overlay.dtype, np.uint8)


if __name__ == "__main__":
    unittest.main()
