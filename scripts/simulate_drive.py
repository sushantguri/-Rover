"""
Mars Rover Autonomous Navigation Simulation.
Takes an image, runs DeepLabV3+ semantic segmentation, generates the costmap,
and computes an optimal collision-free A* drive trajectory avoiding sand dunes and rock hazards.

Usage:
  python scripts/simulate_drive.py --image sample_images/test_mixed_terrain.jpg --output-dir outputs/
"""
import os
import sys
import argparse
import json
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.inference import MarsTerrainSegmenter
from src.path_planner import MarsPathPlanner
from src.kinematics import RoverKinematicPlanner


def main():
    parser = argparse.ArgumentParser(description="Simulate autonomous Mars drive planning")
    parser.add_argument("--image", "-i", type=str, default="sample_images/test_mixed_terrain.jpg", help="Path to input image")
    parser.add_argument("--output-dir", "-o", type=str, default="outputs/drive_sim", help="Output directory")
    parser.add_argument("--checkpoint", "-c", type=str, default="checkpoints/best_deeplabv3plus.pth", help="Model checkpoint")
    parser.add_argument("--pitch", type=float, default=0.0, help="Rover longitudinal pitch angle (deg)")
    parser.add_argument("--roll", type=float, default=0.0, help="Rover lateral roll angle (deg)")
    parser.add_argument("--tta", action="store_true", help="Enable Test-Time Augmentation ensemble")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(args.image))[0]

    print(f"1. Loading image: {args.image}")
    orig_img = Image.open(args.image).convert("RGB")

    print("2. Running DeepLabV3+ terrain segmentation & costmap generation...")
    segmenter = MarsTerrainSegmenter(checkpoint_path=args.checkpoint)
    res = segmenter.predict(orig_img, img_size=512, use_tta=args.tta, pitch_deg=args.pitch, roll_deg=args.roll)
    pred_mask = res["prediction_mask"]
    telemetry = res["telemetry"]
    overlay_multiclass = res["overlay_multiclass"]

    print("3. Executing A* path planning & Ackermann steering kinematics...")
    planner = MarsPathPlanner(obstacle_inflation_radius=2)
    path_data = planner.plan_path(pred_mask, planning_resolution=128)
    kin_planner = RoverKinematicPlanner()
    eff_speed = telemetry["power_budget"]["effective_ground_speed_cm_s"]
    kinematics = kin_planner.profile_trajectory(
        path_data,
        image_shape=pred_mask.shape,
        effective_speed_cm_s=eff_speed,
        pred_mask=pred_mask,
    )

    print(f"   Status: {path_data['status']} | Maneuver: {kinematics['steering_maneuver']} ({kinematics['initial_heading_deg']:+.1f} deg)")
    print(f"   Waypoints: {path_data['waypoint_count']} points | Est. Arc: {kinematics['estimated_distance_m']:.1f} m")
    print(f"   Slope Attitude: {telemetry['slope_stability']['stability_status']} | Coupled Slip: {telemetry['slope_stability']['slope_coupled_slip_pct']:.1f}%")

    print("4. Rendering trajectory overlay...")
    path_overlay = planner.render_path_overlay(overlay_multiclass, path_data)

    out_path = os.path.join(args.output_dir, f"{base_name}_planned_path.jpg")
    Image.fromarray(path_overlay).save(out_path, quality=95)
    
    json_path = os.path.join(args.output_dir, f"{base_name}_trajectory.json")
    with open(json_path, "w") as f:
        save_data = {
            "status": path_data["status"],
            "waypoint_count": path_data["waypoint_count"],
            "total_path_cost": path_data["total_path_cost"],
            "start": list(path_data["start"]),
            "goal": list(path_data["goal"]),
            "waypoints": [list(pt) for pt in path_data["waypoints"]],
            "kinematics": {
                k: ([list(p) for p in v] if k == "smoothed_waypoints" else v)
                for k, v in kinematics.items()
            },
            "slope_stability": telemetry["slope_stability"],
            "calibration_safety": telemetry["calibration_safety"],
            "power_budget": telemetry["power_budget"],
            "science_targets": telemetry["science_targets"],
        }
        json.dump(save_data, f, indent=2)

    print(f"\n✓ Drive simulation complete! Saved visualization to {out_path}")
    print(f"✓ Waypoints JSON saved to {json_path}")


if __name__ == "__main__":
    main()

