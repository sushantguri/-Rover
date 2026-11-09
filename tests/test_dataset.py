"""
Unit tests for AI4Mars dataset loading, preprocessing, and masking.
"""
import unittest
import numpy as np
from PIL import Image
from src.dataset import prepare_row, IGNORE


class TestDatasetPipeline(unittest.TestCase):
    def test_prepare_row_with_masks(self):
        dummy_img = Image.fromarray(np.full((100, 100, 3), 128, dtype=np.uint8))
        dummy_lbl = np.zeros((100, 100), dtype=np.uint8)  # all soil (0)

        # Rover mask in top-left (10x10)
        rover_m = np.zeros((100, 100), dtype=np.uint8)
        rover_m[:10, :10] = 1

        # Range mask in top-right (10x10)
        range_m = np.zeros((100, 100), dtype=np.uint8)
        range_m[:10, 90:] = 1

        row = {
            "image": dummy_img,
            "label_mask": dummy_lbl,
            "rover_mask": rover_m,
            "range_mask": range_m,
            "has_labels": True,
        }

        img, lbl, _, _ = prepare_row(row)
        self.assertEqual(img.shape, (100, 100, 3))
        self.assertEqual(lbl.shape, (100, 100))
        # Rover area should be IGNORE
        self.assertTrue((lbl[:10, :10] == IGNORE).all())
        # Range area should be IGNORE
        self.assertTrue((lbl[:10, 90:] == IGNORE).all())
        # Unmasked area should still be 0 (soil)
        self.assertTrue((lbl[50:60, 50:60] == 0).all())

    def test_mask_nearest_interpolation(self):
        """
        Ensures that when masks are resized, ONLY original discrete classes exist,
        with no interpolated floating artifacts.
        """
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[20:40, 20:40] = 1  # bedrock
        mask[60:80, 60:80] = 2  # sand
        mask[:10, :] = 255      # ignored

        pil_mask = Image.fromarray(mask)
        resized = pil_mask.resize((256, 256), Image.NEAREST)
        arr = np.array(resized)

        unique_vals = set(np.unique(arr))
        allowed_vals = {0, 1, 2, 255}
        self.assertTrue(
            unique_vals.issubset(allowed_vals),
            f"Unexpected values in resized mask: {unique_vals - allowed_vals}",
        )


if __name__ == "__main__":
    unittest.main()
