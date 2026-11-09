"""
Unit tests for Test-Time Augmentation (TTA) Multi-View Ensemble Engine.
"""
import unittest
import torch
import torch.nn as nn

from src.tta import TestTimeAugmenter


class DummySymmetricModel(nn.Module):
    """Simple 1x1 conv mock model that depends strictly on local pixel intensity."""
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 4, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class TestTTA(unittest.TestCase):
    def setUp(self):
        self.model = DummySymmetricModel()
        self.tta_fast = TestTimeAugmenter(self.model, mode="fast")
        self.tta_full = TestTimeAugmenter(self.model, mode="full")

    def test_fast_tta_probs_sum_to_one(self):
        """Probabilities output by TTA fast mode must sum strictly to 1.0 across classes."""
        dummy_input = torch.randn(2, 3, 64, 64)
        probs = self.tta_fast.predict_probs(dummy_input)

        self.assertEqual(probs.shape, (2, 4, 64, 64))
        class_sums = probs.sum(dim=1)
        self.assertTrue(torch.allclose(class_sums, torch.ones_like(class_sums), atol=1e-5))

    def test_full_tta_probs_sum_to_one(self):
        """Probabilities output by TTA full (4-view) mode must sum strictly to 1.0."""
        dummy_input = torch.randn(1, 3, 32, 32)
        probs = self.tta_full.predict_probs(dummy_input)

        self.assertEqual(probs.shape, (1, 4, 32, 32))
        class_sums = probs.sum(dim=1)
        self.assertTrue(torch.allclose(class_sums, torch.ones_like(class_sums), atol=1e-5))

    def test_pseudo_logits_compatibility(self):
        """Pseudo-logits must match argmax of probabilities."""
        dummy_input = torch.randn(1, 3, 32, 32)
        probs = self.tta_fast.predict_probs(dummy_input)
        logits = self.tta_fast.predict_pseudo_logits(dummy_input)

        self.assertEqual(logits.shape, (1, 4, 32, 32))
        self.assertTrue(torch.equal(torch.argmax(probs, dim=1), torch.argmax(logits, dim=1)))


if __name__ == "__main__":
    unittest.main()
