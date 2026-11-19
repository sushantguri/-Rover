"""
Unit tests for NASA Rover Dynamic Window Approach (DWA) Local Reactive Planner.
"""
import unittest
import numpy as np
from src.dwa_local_planner import RoverDWALocalPlanner


class TestRoverDWALocalPlanner(unittest.TestCase):
    def setUp(self):
        self.planner = RoverDWALocalPlanner(
            max_speed_cm_s=4.5,
            min_speed_cm_s=0.0,
            max_yaw_rate_deg_s=12.0,
            lookahead_m=3.5,
        )

    def test_dynamic_window_bounds(self):
        """Verifies computed dynamic window satisfies motor acceleration envelopes."""
        vs_min, vs_max, ws_min, ws_max = self.planner.compute_dynamic_window(
            current_v_cm_s=2.0, current_w_deg_s=0.0
        )
        self.assertGreaterEqual(vs_min, 0.0)
        self.assertLessEqual(vs_max, 4.5)
        self.assertGreaterEqual(ws_min, -12.0)
        self.assertLessEqual(ws_max, 12.0)

    def test_plan_local_twist_on_clear_terrain(self):
        """Clear bedrock/soil terrain should choose forward velocity and high clearance."""
        mask = np.zeros((128, 128), dtype=np.int64)  # all soil (0)
        result = self.planner.plan_local_twist(
            mask, target_heading_deg=0.0, current_v_cm_s=3.0, current_w_deg_s=0.0
        )
        self.assertEqual(result["dwa_status"], "DWA_TRAJECTORY_OPTIMAL")
        self.assertGreater(result["selected_v_cm_s"], 0.0)
        self.assertEqual(result["clearance_margin_pct"], 100.0)
        self.assertIn("STRAIGHT", result["steer_action"])

    def test_obstacle_steer_around(self):
        """A rock directly in front should induce steering away to clear obstacle."""
        mask = np.zeros((128, 128), dtype=np.int64)
        # Place big rock directly ahead in the center line
        h, w = 128, 128
        mask[int(h * 0.6) : int(h * 0.8), int(w * 0.45) : int(w * 0.55)] = 3  # Big Rock (3)

        result = self.planner.plan_local_twist(
            mask, target_heading_deg=0.0, current_v_cm_s=2.0, current_w_deg_s=0.0
        )
        # Should either steer or find non-colliding alternative
        self.assertIn(result["dwa_status"], ["DWA_TRAJECTORY_OPTIMAL", "DWA_EMERGENCY_STOP"])
        if result["dwa_status"] == "DWA_TRAJECTORY_OPTIMAL":
            # The selected w_deg_s should be non-zero (turning to avoid front boulder)
            self.assertNotEqual(result["selected_w_deg_s"], 0.0)

    def test_render_dwa_overlay_shape_and_dtype(self):
        """Verifies DWA multi-arc overlay produces a valid RGB image matching input dimensions."""
        img = np.random.randint(50, 200, (64, 64, 3), dtype=np.uint8)
        mask = np.zeros((64, 64), dtype=np.int64)
        res = self.planner.plan_local_twist(mask)
        overlay = self.planner.render_dwa_overlay(img, mask, res)

        self.assertEqual(overlay.shape, (64, 64, 3))
        self.assertEqual(overlay.dtype, np.uint8)


if __name__ == "__main__":
    unittest.main()
