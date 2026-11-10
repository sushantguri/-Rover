"""
CLI Tool: Generate a NASA Sol Drive Engineering Field Report (HTML) for any Rover Image.

Usage:
  python scripts/generate_mission_report.py --image sample_images/test_mixed_terrain.jpg --output outputs/sol_report.html --tta
"""
import os
import sys
import argparse
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.inference import MarsTerrainSegmenter
from src.path_planner import MarsPathPlanner
from src.kinematics import RoverKinematicPlanner
from src.report_generator import MissionReportGenerator


def main():
    parser = argparse.ArgumentParser(description="Generate standalone NASA Sol Drive Field Report")
    parser.add_argument("--image", "-i", type=str, default="sample_images/test_mixed_terrain.jpg", help="Input rover image")
    parser.add_argument("--output", "-o", type=str, default="outputs/sol_4118_report.html", help="Output HTML report path")
    parser.add_argument("--sol", type=int, default=4118, help="Mars Sol mission number")
    parser.add_argument("--tta", action="store_true", help="Enable Test-Time Augmentation ensemble")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    print(f"1. Loading image '{args.image}'...")
    orig_img = Image.open(args.image).convert("RGB")
    orig_np = np.array(orig_img)

    print(f"2. Running terrain segmentation (TTA={'Enabled' if args.tta else 'Disabled'})...")
    segmenter = MarsTerrainSegmenter()
    res = segmenter.predict(orig_img, img_size=512, use_tta=args.tta)

    print("3. Computing A* obstacle-avoidance path & steering kinematics...")
    planner = MarsPathPlanner(obstacle_inflation_radius=2)
    path_data = planner.plan_path(res["prediction_mask"], planning_resolution=128)
    path_overlay = planner.render_path_overlay(res["overlay_multiclass"], path_data)

    kin_planner = RoverKinematicPlanner()
    eff_speed = res["telemetry"]["power_budget"]["effective_ground_speed_cm_s"]
    res["telemetry"]["kinematics"] = kin_planner.profile_trajectory(
        path_data,
        image_shape=res["prediction_mask"].shape,
        effective_speed_cm_s=eff_speed,
        pred_mask=res["prediction_mask"],
    )

    print("4. Compiling HTML Field Report...")
    generator = MissionReportGenerator()
    html_content = generator.generate_html_report(
        telemetry=res["telemetry"],
        orig_img_np=orig_np,
        overlay_multiclass=res["overlay_multiclass"],
        overlay_sand=res["overlay_sand_hazard"],
        overlay_path=path_overlay,
        path_data=path_data,
        sol_number=args.sol,
        overlay_zone_radar=res["overlay_zone_radar"],
        overlay_science=res["overlay_science_targets"],
    )

    with open(args.output, "w", encoding="utf-8") as f:
        f.write(html_content)

    filesize_kb = os.path.getsize(args.output) / 1024
    print(f"\n✓ Mission Report successfully saved to '{args.output}' ({filesize_kb:.1f} KB)")
    print("  Open this file in any web browser to inspect or print to PDF.")


if __name__ == "__main__":
    main()
