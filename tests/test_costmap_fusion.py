"""
Unit tests for Multi-Layer AutoNav Costmap Fusion Engine.
"""
import unittest
import numpy as np

from src.costmap_fusion import AutoNavCostmapFuser


class TestAutoNavCostmapFuser(unittest.TestCase):
    def setUp(self):
        self.fuser = AutoNavCostmapFuser(obstacle_inflation_radius=2)

    def test_semantic_base_costs(self):
        """Terrain with no obstacles, slope, or uncertainty preserves exact base semantic costs."""
        mask = np.array([
            [1, 0],
            [2, 0],
        ], dtype=np.int64)
        costmap, telemetry = self.fuser.fuse_layers(mask)

        self.assertEqual(costmap[0, 0], 1.0)    # Bedrock = 1.0
        self.assertEqual(costmap[0, 1], 2.0)    # Soil = 2.0
        self.assertEqual(costmap[1, 0], 18.0)   # Sand = 18.0
        self.assertFalse(telemetry["slope_cost_active"])
        self.assertFalse(telemetry["uncertainty_buffer_active"])

    def test_slope_and_tilt_increases_navigable_cost(self):
        """Uphill pitch increases forward terrain navigation cost without altering lethal obstacles."""
        mask = np.zeros((30, 30), dtype=np.int64)  # All soil (base = 2.0)
        mask[10, 10] = 3                          # Obstacle rock

        cost_flat, _ = self.fuser.fuse_layers(mask, pitch_deg=0.0, roll_deg=0.0)
        cost_tilt, tele = self.fuser.fuse_layers(mask, pitch_deg=18.0, roll_deg=4.0)

        self.assertTrue(tele["slope_cost_active"])
        # Navigable soil cost must be strictly higher under slope resistance
        self.assertGreater(cost_tilt[0, 0], cost_flat[0, 0])
        # Rock obstacle remains 1000.0
        self.assertEqual(cost_tilt[10, 10], 1000.0)

    def test_epistemic_uncertainty_buffer_penalizes_ambiguous_cells(self):
        """High prediction entropy adds cost penalty to guide planner through certain paths."""
        mask = np.zeros((20, 20), dtype=np.int64)
        entropy = np.zeros((20, 20), dtype=np.float32)
        entropy[:, :10] = 0.90  # High ambiguity in left half

        costmap, tele = self.fuser.fuse_layers(mask, entropy_map=entropy)
        self.assertTrue(tele["uncertainty_buffer_active"])
        self.assertGreater(costmap[5, 5], costmap[5, 15])

    def test_ventifact_wear_penalty_applied_to_bedrock(self):
        """High ventifact roughness adds protective cost penalty to sharp bedrock."""
        mask = np.array([[1, 0]], dtype=np.int64)  # Bedrock, Soil
        cost_smooth, _ = self.fuser.fuse_layers(mask, ventifact_roughness_index=10.0)
        cost_sharp, tele = self.fuser.fuse_layers(mask, ventifact_roughness_index=90.0)

        self.assertTrue(tele["ventifact_protection_active"])
        self.assertGreater(cost_sharp[0, 0], cost_smooth[0, 0])
        # Soil cell is unaffected by ventifact roughness
        self.assertEqual(cost_sharp[0, 1], cost_smooth[0, 1])

    def test_render_fused_costmap_overlay_preserves_shape(self):
        """Fused costmap overlay returns RGB uint8 array matching input shape."""
        img = np.random.randint(50, 200, (40, 50, 3), dtype=np.uint8)
        costmap = np.full((40, 50), 2.0, dtype=np.float32)
        overlay = self.fuser.render_fused_costmap_overlay(img, costmap)
        self.assertEqual(overlay.shape, (40, 50, 3))
        self.assertEqual(overlay.dtype, np.uint8)


if __name__ == "__main__":
    unittest.main()
