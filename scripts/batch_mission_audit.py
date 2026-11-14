"""
CLI Fleet Audit Tool: Batch-audits Mars Rover camera frames across perception,
horizon hazard zones, A* steering kinematics, MMRTG energy cost, and AEGIS science targeting.

Usage:
  python scripts/batch_mission_audit.py --input-dir sample_images/ --output-dir outputs/mission_audit/
"""
import os
import sys
import glob
import csv
import json
import argparse

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.inference import MarsTerrainSegmenter
from src.path_planner import MarsPathPlanner
from src.kinematics import RoverKinematicPlanner


def run_batch_audit(input_dir: str, output_dir: str, img_size: int = 384, limit: int = 0):
    os.makedirs(output_dir, exist_ok=True)
    extensions = ("*.jpg", "*.jpeg", "*.png")
    image_paths = []
    for ext in extensions:
        image_paths.extend(glob.glob(os.path.join(input_dir, ext)))
    image_paths = sorted(image_paths)
    if limit > 0:
        image_paths = image_paths[:limit]

    if not image_paths:
        raise FileNotFoundError(f"No images found in '{input_dir}'")

    segmenter = MarsTerrainSegmenter()
    planner = MarsPathPlanner(obstacle_inflation_radius=2)
    kin_planner = RoverKinematicPlanner()

    records = []
    print(f"Auditing {len(image_paths)} Mars rover scenes from '{input_dir}' on {segmenter.device}...")

    for idx, img_path in enumerate(image_paths, start=1):
        fname = os.path.basename(img_path)
        res = segmenter.predict(img_path, img_size=img_size)
        mask = res["prediction_mask"]
        t = res["telemetry"]
        geom = t["geometry_risk"]
        pwr = t["power_budget"]
        sci = t["science_targets"]
        wear = t.get("wheel_wear", {})
        solar = t.get("solar_thermal", {})
        fus = t.get("costmap_fusion", {})

        plan_data = planner.plan_path(mask, planning_resolution=96)
        kin = kin_planner.profile_trajectory(
            plan_data,
            image_shape=mask.shape,
            effective_speed_cm_s=pwr["effective_ground_speed_cm_s"],
            pred_mask=mask,
        )

        top_target = sci["targets"][0]["category"] if sci["targets"] else "NONE"
        top_score = sci["targets"][0]["priority_score"] if sci["targets"] else 0.0

        row = {
            "image": fname,
            "traversability_index": t["traversability_index"],
            "safety_status": t["safety_status"],
            "sand_pct": round(t["terrain_distribution_pct"]["sand"], 2),
            "nearfield_sand_pct": geom["nearfield_sand_pct"],
            "high_centering_risk": geom["high_centering_risk"],
            "path_status": plan_data["status"],
            "steering_maneuver": kin["steering_maneuver"],
            "energy_cost_wh_per_m": pwr["energy_cost_wh_per_m"],
            "power_status": pwr["power_status"],
            "wheel_wear_status": wear.get("wear_status", "N/A"),
            "peak_contact_stress_mpa": wear.get("peak_contact_stress_mpa", 0.0),
            "puncture_risk_pct": wear.get("puncture_risk_pct", 0.0),
            "solar_flux_w_m2": solar.get("total_solar_flux_w_m2", 0.0),
            "heater_power_w": solar.get("survival_heater_power_w", 0.0),
            "mean_nav_cost": fus.get("mean_navigable_cost", 0.0),
            "top_aegis_target": top_target,
            "top_aegis_score": top_score,
        }
        records.append(row)
        print(
            f"  [{idx}/{len(image_paths)}] {fname:<34} | Trav={row['traversability_index']:>5.1f} | "
            f"{row['safety_status']:<16} | Maneuver={row['steering_maneuver']:<18} | AEGIS={top_target}"
        )

    json_path = os.path.join(output_dir, "mission_audit_summary.json")
    with open(json_path, "w", encoding="utf-8") as jf:
        json.dump(records, jf, indent=2)

    csv_path = os.path.join(output_dir, "mission_audit_summary.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as cf:
        writer = csv.DictWriter(cf, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)

    print(f"\n✓ Audit complete! Saved JSON & CSV summaries to '{output_dir}'")
    return records


def main():
    parser = argparse.ArgumentParser(description="Batch Mission Fleet Audit across Mars Rover frames")
    parser.add_argument("--input-dir", "-i", type=str, default="sample_images/")
    parser.add_argument("--output-dir", "-o", type=str, default="outputs/mission_audit/")
    parser.add_argument("--resolution", "-r", type=int, default=384)
    parser.add_argument("--limit", type=int, default=0, help="Max images to audit (0 = all)")
    args = parser.parse_args()

    run_batch_audit(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        img_size=args.resolution,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
