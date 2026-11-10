"""
Unit tests for Autonomous Rover Steering Kinematics & Path Smoothing.
"""
import unittest
import numpy as np

from src.kinematics import RoverKinematicPlanner, rdp_simplify


class TestRoverKinematics(unittest.TestCase):
    def setUp(self):
        self.planner = RoverKinematicPlanner(field_depth_m=25.0)

    def test_rdp_simplify_reduces_collinear_points(self):
        """Straight vertical path must simplify cleanly down to start & end anchors."""
        pts = [(100 - i, 50) for i in range(95)]
        simplified = rdp_simplify(pts, epsilon=2.0)
        self.assertEqual(len(simplified), 2)
        self.assertEqual(simplified[0], (100, 50))
        self.assertEqual(simplified[-1], (6, 50))

    def test_straight_path_maneuver_and_duration(self):
        """Nominal straight trajectory yields STRAIGHT_CRUISE and physically scaled drive time."""
        path_data = {
            "status": "PATH_ACQUIRED",
            "waypoints": [(200 - i * 4, 100) for i in range(48)],
            "start": (200, 100),
            "goal": (12, 100),
        }
        profile = self.planner.profile_trajectory(
            path_data=path_data,
            image_shape=(200, 200),
            effective_speed_cm_s=5.0,
        )
        self.assertEqual(profile["kinematic_status"], "TRAJECTORY_FEASIBLE")
        self.assertEqual(profile["steering_maneuver"], "STRAIGHT_CRUISE")
        self.assertAlmostEqual(profile["initial_heading_deg"], 0.0, places=1)
        self.assertGreater(profile["estimated_distance_m"], 20.0)
        self.assertGreater(profile["estimated_drive_time_s"], 400.0)

    def test_blocked_path_returns_halt(self):
        """Blocked A* result must yield HALT_NO_PATH cleanly."""
        profile = self.planner.profile_trajectory(
            path_data={"status": "PATH_BLOCKED_BY_OBSTACLES", "waypoints": []},
            image_shape=(128, 128),
            effective_speed_cm_s=4.5,
        )
        self.assertEqual(profile["kinematic_status"], "NO_VIABLE_ARC")
        self.assertEqual(profile["steering_maneuver"], "HALT_NO_PATH")
        self.assertEqual(profile["estimated_distance_m"], 0.0)


if __name__ == "__main__":
    unittest.main()
