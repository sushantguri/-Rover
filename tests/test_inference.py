"""
Integration tests for MarsTerrainSegmenter and Telemetry Logic.
"""
import unittest
import numpy as np
from PIL import Image

from src.inference import MarsTerrainSegmenter, CLASSES


class TestInferencePipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Initialize segmenter (uses existing checkpoint if available or base model)
        cls.segmenter = MarsTerrainSegmenter(device="cpu")

    def test_predict_on_synthetic_image(self):
        """Verifies prediction returns all expected overlays and valid telemetry schema."""
        dummy_img = np.random.randint(50, 200, (128, 128, 3), dtype=np.uint8)
        
        result = self.segmenter.predict(dummy_img, img_size=128, sand_threshold=25.0)

        # 1. Check returned keys
        expected_keys = {
            "prediction_mask", 
            "overlay_multiclass", 
            "overlay_sand_hazard", 
            "overlay_costmap", 
            "overlay_uncertainty", 
            "telemetry"
        }
        self.assertTrue(expected_keys.issubset(result.keys()))

        # 2. Check shapes
        self.assertEqual(result["prediction_mask"].shape, (128, 128))
        self.assertEqual(result["overlay_multiclass"].shape, (128, 128, 3))
        self.assertEqual(result["overlay_sand_hazard"].shape, (128, 128, 3))
        self.assertEqual(result["overlay_costmap"].shape, (128, 128, 3))
        self.assertEqual(result["overlay_uncertainty"].shape, (128, 128, 3))

        # 3. Check telemetry schema
        telemetry = result["telemetry"]
        self.assertIn("traversability_index", telemetry)
        self.assertIn("safety_status", telemetry)
        self.assertIn("autonomous_drive_action", telemetry)
        self.assertIn("terrain_distribution_pct", telemetry)
        self.assertIn("terramechanics", telemetry)
        self.assertIn("power_budget", telemetry)
        self.assertIn("uncertainty", telemetry)

        # 4. Check percentage sum adds to 100%
        pct_sum = sum(telemetry["terrain_distribution_pct"].values())
        self.assertAlmostEqual(pct_sum, 100.0, places=1)

    def test_predict_with_tta(self):
        """Verifies TTA inference mode executes properly and updates telemetry metadata."""
        dummy_img = np.random.randint(50, 200, (128, 128, 3), dtype=np.uint8)
        result = self.segmenter.predict(dummy_img, img_size=128, use_tta=True, tta_mode="fast")
        self.assertEqual(result["telemetry"]["tta_mode"], "fast")

    def test_traversability_index_bounds(self):
        """Verifies traversability index stays strictly within [0.0, 100.0]."""
        dummy_img = Image.new("RGB", (64, 64), color=(180, 120, 80))
        result = self.segmenter.predict(dummy_img, img_size=64)
        t_index = result["telemetry"]["traversability_index"]
        self.assertGreaterEqual(t_index, 0.0)
        self.assertLessEqual(t_index, 100.0)


if __name__ == "__main__":
    unittest.main()
