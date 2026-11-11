"""
CLI Utility: Export complete ROS 2 Nav2 / rosbridge JSON telemetry bundle for a Mars rover image.
Generates OccupancyGrid, nav_msgs/Path, geometry_msgs/Twist (/cmd_vel), AEGIS MarkerArray,
and terrain diagnostics ready for ROS 2 / RViz2 replay.

Usage:
  python scripts/export_ros_packet.py --image sample_images/test_mixed_terrain.jpg --output outputs/ros2_packet.json
"""
import os
import sys
import json
import argparse

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.inference import MarsTerrainSegmenter
from src.path_planner import MarsPathPlanner
from src.kinematics import RoverKinematicPlanner
from src.ros_bridge import Ros2NavBridge


def export_ros_packet(image_path: str, output_json: str, grid_resolution: int = 128, img_size: int = 512):
    os.makedirs(os.path.dirname(os.path.abspath(output_json)), exist_ok=True)

    segmenter = MarsTerrainSegmenter()
    res = segmenter.predict(image_path, img_size=img_size)
    pred_mask = res["prediction_mask"]
    telemetry = res["telemetry"]

    planner = MarsPathPlanner(obstacle_inflation_radius=2)
    plan_data = planner.plan_path(pred_mask, planning_resolution=grid_resolution)

    kin_planner = RoverKinematicPlanner()
    eff_speed = telemetry["power_budget"]["effective_ground_speed_cm_s"]
    kinematics = kin_planner.profile_trajectory(
        plan_data, image_shape=pred_mask.shape, effective_speed_cm_s=eff_speed, pred_mask=pred_mask
    )

    bridge = Ros2NavBridge(map_resolution_meters=round(0.05 * (pred_mask.shape[0] / max(1, grid_resolution)), 4))
    # Use planning resolution mask for compact JSON OccupancyGrid payload
    from PIL import Image
    import numpy as np
    small_mask = np.array(
        Image.fromarray(pred_mask.astype(np.uint8)).resize((grid_resolution, grid_resolution), Image.NEAREST),
        dtype=np.int64,
    )

    waypoints_for_ros = kinematics["smoothed_waypoints"] or plan_data["waypoints"]
    packet = {
        "topics": {
            "/autonav/costmap": bridge.create_occupancy_grid_msg(small_mask),
            "/autonav/global_plan": bridge.create_nav_path_msg(waypoints_for_ros, pred_mask.shape),
            "/cmd_vel": bridge.create_cmd_vel_msg(kinematics, eff_speed),
            "/aegis/markers": bridge.create_aegis_markers_msg(telemetry["science_targets"], pred_mask.shape),
            "/autonav/telemetry": bridge.create_terrain_telemetry_msg(
                telemetry["terrain_distribution_pct"], telemetry["traversability_index"]
            ),
        },
        "meta": {
            "source_image": os.path.basename(image_path),
            "safety_status": telemetry["safety_status"],
            "steering_maneuver": kinematics["steering_maneuver"],
        },
    }

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(packet, f, indent=2)

    size_kb = os.path.getsize(output_json) / 1024.0
    print(f"✓ Exported ROS 2 Nav2 packet to '{output_json}' ({size_kb:.1f} KB)")
    return packet


def main():
    parser = argparse.ArgumentParser(description="Export ROS 2 Nav2 JSON telemetry packet")
    parser.add_argument("--image", "-i", type=str, default="sample_images/test_mixed_terrain.jpg")
    parser.add_argument("--output", "-o", type=str, default="outputs/ros2_packet.json")
    parser.add_argument("--grid-res", type=int, default=96)
    args = parser.parse_args()

    export_ros_packet(args.image, args.output, grid_resolution=args.grid_res)


if __name__ == "__main__":
    main()

