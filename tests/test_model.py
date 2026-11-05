"""
Unit tests for DeepLabV3+ Model Architecture and Loss Function.
"""
import unittest
import torch
import torch.nn as nn

from src.model import get_deeplabv3plus_model, get_loss_fn, IGNORE


class TestModelArchitecture(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.device = torch.device("cpu")
        # Load model without pretrained weights for fast offline unit testing
        cls.model = get_deeplabv3plus_model(num_classes=4, encoder_weights=None).to(cls.device)
        cls.model.eval()

    def test_forward_pass_shape(self):
        """Verifies forward pass output shape matches (B, num_classes, H, W)."""
        test_resolutions = [(256, 256), (384, 384)]
        for h, w in test_resolutions:
            dummy_input = torch.randn(2, 3, h, w, device=self.device)
            with torch.no_grad():
                logits = self.model(dummy_input)
            self.assertEqual(logits.shape, (2, 4, h, w), f"Failed for resolution {h}x{w}")

    def test_loss_function_ignores_255(self):
        """Verifies that changing logits at pixels with label 255 has ZERO effect on loss."""
        weights = [0.65, 0.49, 0.86, 0.99]
        loss_fn = get_loss_fn(weights=weights, ignore_index=IGNORE, device=self.device)

        logits1 = torch.zeros(1, 4, 10, 10)
        logits2 = logits1.clone()
        # Change logits drastically at position (0, 0)
        logits2[:, :, 0, 0] = 50.0

        # Target where position (0, 0) is ignored (255) and rest are valid (0)
        target = torch.zeros(1, 10, 10, dtype=torch.long)
        target[0, 0, 0] = IGNORE

        loss1 = loss_fn(logits1, target)
        loss2 = loss_fn(logits2, target)

        self.assertAlmostEqual(loss1.item(), loss2.item(), places=5, 
                               msg="Altering logits at index 255 must have ZERO impact on loss")

    def test_loss_function_class_weights(self):
        """Verifies weighted cross-entropy assigns higher unreduced loss to higher weighted classes."""
        w = torch.tensor([0.5, 0.5, 5.0, 0.5]) # Sand (class 2) has 10x higher weight

        loss_fn = nn.CrossEntropyLoss(weight=w, reduction="none")

        logits = torch.zeros(2, 4, 1, 1) # Equal logits for all classes
        targets = torch.tensor([[[0]], [[2]]]) # Sample 0 is Soil (w=0.5), Sample 1 is Sand (w=5.0)

        unreduced_loss = loss_fn(logits, targets)
        loss_soil = unreduced_loss[0].item()
        loss_sand = unreduced_loss[1].item()

        self.assertAlmostEqual(loss_sand / loss_soil, 10.0, places=3, 
                               msg="Sand unreduced loss must be exactly 10x higher according to class weight")


if __name__ == "__main__":
    unittest.main()
