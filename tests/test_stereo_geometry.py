"""
Unit tests for RoverStereoGeometryEngine (Stereo Disparity & 3D Step Hazard Detection).
"""
import unittest
import numpy as np

from src.stereo_geometry import (
    RoverStereoGeometryEngine,
    CRITICAL_STEP_HEIGHT_CM,
    DEFAULT_STEREO_BASELINE_M,
    DEFAULT_FOCAL_LENGTH_PX,
)


class TestRoverStereoGeometryEngine(unittest.TestCase):
    def setUp(self):
        self.engine = RoverStereoGeometryEngine(
            baseline_m=DEFAULT_STEREO_BASELINE_M,
            focal_length_px=DEFAULT_FOCAL_LENGTH_PX,
        )

    def test_depth_and_disparity_computation(self):
        """Verifies synthetic depth increases with distance and disparity follows triangulation."""
        mask = np.zeros((100, 100), dtype=np.int64)  # All soil
        depth = self.engine.synthesize_metric_depth(mask)
        self.assertEqual(depth.shape, (100, 100))

        # Bottom row (near rover) must have smaller depth than top row (horizon)
        self.assertLess(depth[-1, 50], depth[0, 50])
        self.assertGreater(depth[0, 50], 20.0)
        self.assertLess(depth[-1, 50], 3.0)

        # Disparity calculation: d = f * b / Z
        disp = self.engine.compute_disparity_map(depth)
        self.assertEqual(disp.shape, (100, 100))
        # Near field (large disparity) vs far field (small disparity)
        self.assertGreater(disp[-1, 50], disp[0, 50])

    def test_smooth_flat_terrain_has_nominal_step(self):
        """Verifies flat bedrock/soil produces small steps and permits nominal drive."""
        mask = np.ones((100, 100), dtype=np.int64)  # Pure bedrock
        depth = self.engine.synthesize_metric_depth(mask)
        hazards = self.engine.analyze_3d_hazards(mask, depth)

        self.assertEqual(hazards["step_hazard_status"], "NOMINAL_TERRAIN_RELIEF")
        self.assertTrue(hazards["traverse_permitted"])
        self.assertLess(hazards["max_positive_step_cm"], 10.0)
        self.assertGreater(hazards["clearance_margin_pct"], 80.0)

    def test_large_boulder_triggers_step_hazard(self):
        """Verifies large near-field rock clusters escalate to step obstacle alerts."""
        mask = np.zeros((100, 100), dtype=np.int64)
        # Place large boulder in near-field (rows 70 to 95)
        mask[70:95, 30:70] = 3  # Class 3: Big Rock
        depth = self.engine.synthesize_metric_depth(mask)
        hazards = self.engine.analyze_3d_hazards(mask, depth)

        self.assertIn(
            hazards["step_hazard_status"],
            ["CRITICAL_UNCLIMBABLE_WALL", "ELEVATED_STEP_OBSTACLE"],
        )
        self.assertGreaterEqual(hazards["max_positive_step_cm"], 15.0)

    def test_elevation_overlay_shape_and_dtype(self):
        """Verifies elevation overlay renders a valid RGB image matching original shape."""
        orig = np.random.randint(60, 200, (64, 64, 3), dtype=np.uint8)
        mask = np.zeros((64, 64), dtype=np.int64)
        mask[40:50, 20:30] = 3
        depth = self.engine.synthesize_metric_depth(mask)
        hazards = self.engine.analyze_3d_hazards(mask, depth)

        overlay = self.engine.render_elevation_overlay(
            orig, mask, depth, hazards["max_positive_step_cm"], alpha=0.60
        )
        self.assertEqual(overlay.shape, (64, 64, 3))
        self.assertEqual(overlay.dtype, np.uint8)


if __name__ == "__main__":
    unittest.main()
