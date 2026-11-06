"""
ROS 2 / ROS Bridge and Message Serialization Module.
Transforms DeepLabV3+ terrain predictions and costmaps into standard ROS 2 navigation messages:
- nav_msgs/msg/OccupancyGrid (for Nav2 costmap layers)
- sensor_msgs/msg/Image (classified terrain maps)
- std_msgs/msg/Float32MultiArray (terrain distribution telemetry)
Enables seamless integration with NASA AutoNav, ROS 2 Nav2, and Gazebo Mars rover simulations.
"""
import time
import numpy as np
from typing import Dict, Any, List, Tuple


class Ros2NavBridge:
    """
    Serializes DeepLabV3+ predictions into ROS 2 navigation message payloads.
    Compatible with rclpy and rosbridge_suite (JSON websocket bridges).
    """
    def __init__(self, map_resolution_meters: float = 0.05, frame_id: str = "rover_navcam_optical_frame"):
        self.resolution = map_resolution_meters  # 5 cm per cell
        self.frame_id = frame_id

    def create_occupancy_grid_msg(self, pred_mask: np.ndarray) -> Dict[str, Any]:
        """
        Converts terrain segmentation mask into a standard nav_msgs/msg/OccupancyGrid.
        
        Occupancy grid standards:
          - -1 : Unknown / Unlabeled (our 255 IGNORE class)
          - 0-20 : Free space / High traction (Bedrock / Soil)
          - 75-85 : High slip hazard (Sand)
          - 100 : Lethal obstacle (Big Rock)
        """
        h, w = pred_mask.shape
        grid = np.zeros((h, w), dtype=np.int8)

        # Mapping rules
        grid[pred_mask == 1] = 5    # Bedrock: 5% cost
        grid[pred_mask == 0] = 15   # Soil: 15% cost
        grid[pred_mask == 2] = 80   # Sand: 80% cost (Wheel slip warning)
        grid[pred_mask == 3] = 100  # Big Rock: 100% obstacle
        grid[pred_mask == 255] = -1 # Ignored / Beyond 30m: -1 (Unknown)

        # Flatten in row-major order as expected by ROS 2
        flattened_data = grid.flatten().tolist()

        now_sec = time.time()
        sec = int(now_sec)
        nanosec = int((now_sec - sec) * 1e9)

        msg = {
            "header": {
                "stamp": {"sec": sec, "nanosec": nanosec},
                "frame_id": self.frame_id
            },
            "info": {
                "map_load_time": {"sec": sec, "nanosec": nanosec},
                "resolution": self.resolution,
                "width": w,
                "height": h,
                "origin": {
                    "position": {"x": - (w * self.resolution) / 2.0, "y": 0.0, "z": 0.0},
                    "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}
                }
            },
            "data": flattened_data
        }
        return msg

    def create_terrain_telemetry_msg(self, class_pcts: Dict[str, float], traversability_index: float) -> Dict[str, Any]:
        """
        Generates telemetry dictionary formatted for ROS 2 custom message / diagnostics.
        """
        now_sec = time.time()
        return {
            "header": {
                "stamp": {"sec": int(now_sec), "nanosec": int((now_sec - int(now_sec)) * 1e9)},
                "frame_id": "base_link"
            },
            "traversability_index": traversability_index,
            "soil_ratio": class_pcts.get("soil", 0.0) / 100.0,
            "bedrock_ratio": class_pcts.get("bedrock", 0.0) / 100.0,
            "sand_ratio": class_pcts.get("sand", 0.0) / 100.0,
            "big_rock_ratio": class_pcts.get("big rock", 0.0) / 100.0,
            "hazard_tripwire_triggered": class_pcts.get("sand", 0.0) > 25.0
        }
