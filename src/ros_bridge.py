"""
ROS 2 / ROS Bridge and Message Serialization Module.
Transforms DeepLabV3+ terrain predictions, trajectories, and AEGIS science targets into standard ROS 2 navigation messages:
- nav_msgs/msg/OccupancyGrid (for Nav2 costmap layers)
- nav_msgs/msg/Path (for Nav2 global/local waypoints)
- geometry_msgs/msg/Twist (for /cmd_vel base mobility control)
- visualization_msgs/msg/MarkerArray (for RViz2 AEGIS geological targets)
- std_msgs/msg/Float32MultiArray / custom terrain telemetry diagnostics
Enables seamless integration with NASA AutoNav, ROS 2 Nav2, and Gazebo Mars rover simulations.
"""
import math
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

    def _stamp(self) -> Dict[str, int]:
        now_sec = time.time()
        sec = int(now_sec)
        nanosec = int((now_sec - sec) * 1e9)
        return {"sec": sec, "nanosec": nanosec}

    def create_nav_path_msg(
        self,
        waypoints: List[Tuple[int, int]],
        image_shape: Tuple[int, int],
        field_depth_m: float = 25.0,
        frame_id: str = "base_link"
    ) -> Dict[str, Any]:
        """
        Converts pixel waypoints [(r, c), ...] into a standard ROS 2 nav_msgs/msg/Path
        projected into rover base_link coordinates (+x forward, +y left in meters).
        """
        h, w = image_shape[:2]
        stamp = self._stamp()
        poses = []

        for r, c in waypoints:
            # Map bottom of image -> x=0m forward; top -> x=field_depth_m forward
            forward_x = float(((h - r) / max(1, h)) * field_depth_m)
            # Standard REP-103 ROS coordinate convention: +y is left, image center -> y=0
            lateral_y = float(((w / 2.0 - c) / max(1, w)) * field_depth_m * 0.8)
            poses.append({
                "header": {"stamp": stamp, "frame_id": frame_id},
                "pose": {
                    "position": {"x": round(forward_x, 3), "y": round(lateral_y, 3), "z": 0.0},
                    "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
                },
            })

        return {
            "header": {"stamp": stamp, "frame_id": frame_id},
            "poses": poses,
        }

    def create_cmd_vel_msg(
        self,
        kinematics: Dict[str, Any],
        commanded_speed_cm_s: float
    ) -> Dict[str, Dict[str, float]]:
        """
        Translates kinematic steering maneuver & speed into geometry_msgs/msg/Twist (/cmd_vel).
        """
        maneuver = kinematics.get("steering_maneuver", "HALT_NO_PATH")
        if maneuver == "HALT_NO_PATH" or commanded_speed_cm_s <= 0.05:
            return {
                "linear": {"x": 0.0, "y": 0.0, "z": 0.0},
                "angular": {"x": 0.0, "y": 0.0, "z": 0.0},
            }

        linear_x_m_s = float(commanded_speed_cm_s) / 100.0
        # Image-right (+heading_deg) maps to negative yaw rate around +z in REP-103
        heading_deg = float(kinematics.get("initial_heading_deg", 0.0))
        angular_z_rad_s = float(np.clip(-math.radians(heading_deg) * 0.5, -0.45, 0.45))

        return {
            "linear": {"x": round(linear_x_m_s, 4), "y": 0.0, "z": 0.0},
            "angular": {"x": 0.0, "y": 0.0, "z": round(angular_z_rad_s, 4)},
        }

    def create_aegis_markers_msg(
        self,
        science_targets: Dict[str, Any],
        image_shape: Tuple[int, int],
        field_depth_m: float = 25.0,
        frame_id: str = "base_link"
    ) -> Dict[str, Any]:
        """
        Converts AEGIS autonomous geological targets into visualization_msgs/msg/MarkerArray.
        """
        h, w = image_shape[:2]
        stamp = self._stamp()
        markers = []

        color_map = {
            "BEDROCK_OUTCROP": {"r": 0.22, "g": 0.74, "b": 0.97, "a": 0.95},
            "STRATIGRAPHIC_CONTACT": {"r": 0.98, "g": 0.80, "b": 0.08, "a": 0.95},
            "FLOAT_ROCK_SPECIMEN": {"r": 0.96, "g": 0.45, "b": 0.71, "a": 0.95},
        }

        for tgt in science_targets.get("targets", []):
            px_x, px_y = tgt["pixel_xy"]
            forward_x = float(((h - px_y) / max(1, h)) * field_depth_m)
            lateral_y = float(((w / 2.0 - px_x) / max(1, w)) * field_depth_m * 0.8)
            col = color_map.get(tgt["category"], {"r": 1.0, "g": 1.0, "b": 1.0, "a": 0.9})
            markers.append({
                "header": {"stamp": stamp, "frame_id": frame_id},
                "ns": "aegis_geology",
                "id": int(tgt["rank"]),
                "type": 2,  # SPHERE
                "action": 0,  # ADD
                "pose": {
                    "position": {"x": round(forward_x, 3), "y": round(lateral_y, 3), "z": 0.25},
                    "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
                },
                "scale": {"x": 0.4, "y": 0.4, "z": 0.4},
                "color": col,
                "text": f"{tgt['target_id']}:{tgt['category']}",
            })

        return {"markers": markers}

