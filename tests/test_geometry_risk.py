"""
Unit tests for Horizon Range-Zone & Boulder Geometry Risk Analyzer.
"""
import unittest
import numpy as np

from src.geometry_risk import TerrainGeometryAnalyzer


class TestTerrainGeometryAnalyzer(unittest.TestCase):
    def setUp(self):
        self.analyzer = TerrainGeometryAnalyzer()

    def test_nearfield_vs_horizon_zone_separation(self):
        """Sand at horizon vs clear near-field must map to exact zone breakdowns."""
        mask = np.ones((100, 100), dtype=np.int64)  # All bedrock
        mask[:25, :] = 2  # Top 25% (far horizon 15-30m) is pure sand
        analysis = self.analyzer.analyze(mask)

        self.assertGreater(analysis["horizon_sand_pct"], 80.0)
        self.assertEqual(analysis["nearfield_sand_pct"], 0.0)
        self.assertEqual(analysis["immediate_zone_alert"], "NEARFIELD_CORRIDOR_CLEAR")

    def test_nearfield_large_boulder_high_centering_critical(self):
        """A wide contiguous boulder cluster in the near-field must flag CRITICAL risk."""
        mask = np.zeros((100, 100), dtype=np.int64)
        mask[72:96, 35:65] = 3  # 30% width boulder inside near-field (>65 row threshold)
        analysis = self.analyzer.analyze(mask)

        self.assertGreaterEqual(analysis["boulder_cluster_count"], 1)
        self.assertGreaterEqual(analysis["nearfield_boulder_count"], 1)
        self.assertEqual(analysis["high_centering_risk"], "CRITICAL")
        self.assertEqual(analysis["immediate_zone_alert"], "IMMEDIATE_NEARFIELD_OBSTRUCTION")

    def test_roughness_and_overlay_shape(self):
        """Verifies terrain roughness index stays in [0, 100] and radar overlay preserves RGB shape."""
        mask = np.zeros((120, 160), dtype=np.int64)
        mask[::2, ::2] = 1
        img = np.random.randint(60, 200, (120, 160, 3), dtype=np.uint8)

        analysis = self.analyzer.analyze(mask)
        self.assertGreaterEqual(analysis["terrain_roughness_index"], 0.0)
        self.assertLessEqual(analysis["terrain_roughness_index"], 100.0)

        overlay = self.analyzer.render_zone_radar_overlay(img, mask, analysis)
        self.assertEqual(overlay.shape, (120, 160, 3))
        self.assertEqual(overlay.dtype, np.uint8)


if __name__ == "__main__":
    unittest.main()
