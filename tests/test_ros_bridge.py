"""
Unit tests for ROS 2 Navigation Bridge message conversion.
"""
import unittest
import numpy as np

from src.ros_bridge import Ros2NavBridge


class TestRos2NavBridge(unittest.TestCase):
    def setUp(self):
        self.bridge = Ros2NavBridge(map_resolution_meters=0.05, frame_id="navcam_optical_frame")

    def test_create_occupancy_grid_msg(self):
        """Verifies OccupancyGrid message structure and values."""
        mask = np.array([
            [1, 0],    # Bedrock, Soil
            [2, 3],    # Sand, Big Rock
            [255, 1]   # Ignored, Bedrock
        ], dtype=np.int64)

        msg = self.bridge.create_occupancy_grid_msg(mask)

        # Check ROS header & info
        self.assertEqual(msg["header"]["frame_id"], "navcam_optical_frame")
        self.assertEqual(msg["info"]["width"], 2)
        self.assertEqual(msg["info"]["height"], 3)
        self.assertEqual(msg["info"]["resolution"], 0.05)

        # Check cost values: Bedrock=5, Soil=15, Sand=80, Rock=100, Ignored=-1
        expected = [5, 15, 80, 100, -1, 5]
        self.assertEqual(msg["data"], expected)

    def test_create_terrain_telemetry_msg(self):
        """Verifies telemetry message correctly reports hazard tripwire status."""
        class_pcts = {"soil": 20.0, "bedrock": 40.0, "sand": 35.0, "big rock": 5.0}
        msg = self.bridge.create_terrain_telemetry_msg(class_pcts, traversability_index=62.5)

        self.assertEqual(msg["traversability_index"], 62.5)
        self.assertTrue(msg["hazard_tripwire_triggered"]) # Sand > 25%
        self.assertAlmostEqual(msg["sand_ratio"], 0.35)

    def test_create_nav_path_and_cmd_vel(self):
        """Verifies nav_msgs/Path and geometry_msgs/Twist serialization."""
        waypoints = [(100, 50), (50, 40), (10, 30)]
        path_msg = self.bridge.create_nav_path_msg(waypoints, image_shape=(100, 100))

        self.assertEqual(path_msg["header"]["frame_id"], "base_link")
        self.assertEqual(len(path_msg["poses"]), 3)
        self.assertGreater(path_msg["poses"][-1]["pose"]["position"]["x"], path_msg["poses"][0]["pose"]["position"]["x"])

        twist = self.bridge.create_cmd_vel_msg(
            {"steering_maneuver": "SLIGHT_RIGHT_ARC", "initial_heading_deg": 15.0},
            commanded_speed_cm_s=4.0,
        )
        self.assertAlmostEqual(twist["linear"]["x"], 0.04, places=4)
        self.assertLess(twist["angular"]["z"], 0.0)

        halt_twist = self.bridge.create_cmd_vel_msg({"steering_maneuver": "HALT_NO_PATH"}, commanded_speed_cm_s=4.5)
        self.assertEqual(halt_twist["linear"]["x"], 0.0)
        self.assertEqual(halt_twist["angular"]["z"], 0.0)

    def test_create_aegis_markers_msg(self):
        """Verifies MarkerArray structure for RViz2 visualization."""
        targets = {
            "targets": [
                {"rank": 1, "target_id": "AEGIS-TGT-1", "category": "BEDROCK_OUTCROP", "pixel_xy": [64, 80]}
            ]
        }
        markers_msg = self.bridge.create_aegis_markers_msg(targets, image_shape=(128, 128))
        self.assertEqual(len(markers_msg["markers"]), 1)
        self.assertEqual(markers_msg["markers"][0]["ns"], "aegis_geology")
        self.assertEqual(markers_msg["markers"][0]["id"], 1)


if __name__ == "__main__":
    unittest.main()

