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


def main():
    parser = argparse.ArgumentParser(description="Simulate autonomous Mars drive planning")
    parser.add_argument("--image", "-i", type=str, default="sample_images/test_mixed_terrain.jpg", help="Path to input image")
    parser.add_argument("--output-dir", "-o", type=str, default="outputs/drive_sim", help="Output directory")
    parser.add_argument("--checkpoint", "-c", type=str, default="checkpoints/best_deeplabv3plus.pth", help="Model checkpoint")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(args.image))[0]

    print(f"1. Loading image: {args.image}")
    orig_img = Image.open(args.image).convert("RGB")

    print("2. Running DeepLabV3+ terrain segmentation & costmap generation...")
    segmenter = MarsTerrainSegmenter(checkpoint_path=args.checkpoint)
    res = segmenter.predict(orig_img, img_size=512)
    pred_mask = res["prediction_mask"]
    overlay_multiclass = res["overlay_multiclass"]

    print("3. Executing A* path planning over terrain costmap...")
    planner = MarsPathPlanner(obstacle_inflation_radius=2)
    path_data = planner.plan_path(pred_mask, planning_resolution=128)

    print(f"   Status: {path_data['status']}")
    print(f"   Waypoints: {path_data['waypoint_count']} points")
    print(f"   Total Traversal Cost: {path_data['total_path_cost']}")

    print("4. Rendering trajectory overlay...")
    path_overlay = planner.render_path_overlay(overlay_multiclass, path_data)

    out_path = os.path.join(args.output_dir, f"{base_name}_planned_path.jpg")
    Image.fromarray(path_overlay).save(out_path, quality=95)
    
    json_path = os.path.join(args.output_dir, f"{base_name}_trajectory.json")
    with open(json_path, "w") as f:
        # Convert path_data for json serialization
        save_data = {
            "status": path_data["status"],
            "waypoint_count": path_data["waypoint_count"],
            "total_path_cost": path_data["total_path_cost"],
            "start": list(path_data["start"]),
            "goal": list(path_data["goal"]),
            "waypoints": [list(pt) for pt in path_data["waypoints"]]
        }
        json.dump(save_data, f, indent=2)

    print(f"\n✓ Drive simulation complete! Saved visualization to {out_path}")
    print(f"✓ Waypoints JSON saved to {json_path}")


if __name__ == "__main__":
    main()
