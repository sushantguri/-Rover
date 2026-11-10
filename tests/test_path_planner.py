"""
Unit tests for 8-Connected A* Autonomous Mars Path Planner.
"""
import unittest
import numpy as np

from src.path_planner import MarsPathPlanner


class TestMarsPathPlanner(unittest.TestCase):
    def setUp(self):
        self.planner = MarsPathPlanner(obstacle_inflation_radius=2)

    def test_clear_bedrock_path_acquired(self):
        """Clear bedrock terrain acquires a straight low-cost trajectory."""
        mask = np.ones((128, 128), dtype=np.int64)  # Bedrock everywhere
        res = self.planner.plan_path(mask, planning_resolution=64)

        self.assertEqual(res["status"], "PATH_ACQUIRED")
        self.assertGreater(res["waypoint_count"], 10)
        self.assertGreater(res["total_path_cost"], 0.0)

    def test_path_avoids_central_boulder_obstacle(self):
        """Planner must route cleanly around a large boulder placed on the center line."""
        mask = np.zeros((128, 128), dtype=np.int64)  # Soil
        mask[55:75, 45:83] = 3  # Big Rock barrier in middle center

        res = self.planner.plan_path(mask, planning_resolution=64)
        self.assertEqual(res["status"], "PATH_ACQUIRED")

        # Verify no waypoint enters the interior of the boulder
        for r, c in res["waypoints"]:
            self.assertNotEqual(mask[min(127, r), min(127, c)], 3)

    def test_wall_to_wall_rock_blocks_path(self):
        """Complete wall of Big Rocks across frame returns PATH_BLOCKED_BY_OBSTACLES."""
        mask = np.zeros((128, 128), dtype=np.int64)
        mask[60:70, :] = 3  # Full width boulder barrier

        res = self.planner.plan_path(mask, planning_resolution=64)
        self.assertEqual(res["status"], "PATH_BLOCKED_BY_OBSTACLES")
        self.assertEqual(res["waypoint_count"], 0)

    def test_render_path_overlay_shape(self):
        """Overlay rendering preserves input image shape and RGB channels."""
        img = np.random.randint(50, 200, (128, 128, 3), dtype=np.uint8)
        mask = np.ones((128, 128), dtype=np.int64)
        res = self.planner.plan_path(mask, planning_resolution=64)
        overlay = self.planner.render_path_overlay(img, res)

        self.assertEqual(overlay.shape, img.shape)
        self.assertEqual(overlay.dtype, np.uint8)


if __name__ == "__main__":
    unittest.main()
