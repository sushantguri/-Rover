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


if __name__ == "__main__":
    unittest.main()

