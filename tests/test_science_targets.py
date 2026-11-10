"""
Unit tests for NASA AEGIS-Inspired Geological Science Target Selector.
"""
import unittest
import numpy as np

from src.science_targets import AEGISTargetSelector


class TestAEGISTargetSelector(unittest.TestCase):
    def setUp(self):
        self.selector = AEGISTargetSelector(top_k=3)

    def test_detects_bedrock_and_stratigraphic_contact(self):
        """Mixed soil + bedrock scene yields ranked geological targets with valid coordinates."""
        mask = np.zeros((128, 128), dtype=np.int64)
        mask[50:115, 20:95] = 1  # Large firm bedrock outcrop + contact with surrounding soil

        res = self.selector.select_targets(mask)
        self.assertEqual(res["aegis_status"], "TARGETS_LOCKED")
        self.assertGreaterEqual(res["target_count"], 1)
        categories = {t["category"] for t in res["targets"]}
        self.assertTrue(
            "BEDROCK_OUTCROP" in categories or "STRATIGRAPHIC_CONTACT" in categories
        )

    def test_standoff_warning_on_sand_surrounded_rock(self):
        """A float rock surrounded by loose sand must require standoff laser rather than drive approach."""
        mask = np.full((128, 128), 2, dtype=np.int64)  # Sand dune field
        mask[60:80, 50:70] = 3  # Boulder isolated inside sand

        res = self.selector.select_targets(mask)
        self.assertEqual(res["aegis_status"], "TARGETS_LOCKED")
        self.assertEqual(res["targets"][0]["category"], "FLOAT_ROCK_SPECIMEN")
        self.assertEqual(res["targets"][0]["approach_safety"], "STANDOFF_LASER_ONLY")

    def test_pure_sand_scene_yields_no_targets(self):
        """Pure loose sand dune offers zero exposed geological targets."""
        mask = np.full((100, 100), 2, dtype=np.int64)
        res = self.selector.select_targets(mask)
        self.assertEqual(res["target_count"], 0)
        self.assertEqual(res["aegis_status"], "NO_EXPOSED_GEOLOGY_FOUND")

    def test_render_target_reticles_preserves_shape(self):
        """Reticle overlay must return uint8 RGB array matching original frame shape."""
        img = np.random.randint(60, 190, (128, 128, 3), dtype=np.uint8)
        mask = np.ones((128, 128), dtype=np.int64)
        target_data = self.selector.select_targets(mask)
        rendered = self.selector.render_target_reticles(img, target_data)

        self.assertEqual(rendered.shape, img.shape)
        self.assertEqual(rendered.dtype, np.uint8)


if __name__ == "__main__":
    unittest.main()
