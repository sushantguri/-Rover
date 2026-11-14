"""
CLI Fleet Tool: Audits Mars Rover Aluminum Wheel Skin Wear and Ventifact Puncture Hazards.
Evaluates razor-sharp ventifact micro-relief across exposed bedrock/rock scenes,
accumulated aluminum thickness loss (um), peak contact stress (MPa), and traction control modes.

Usage:
  python scripts/audit_wheel_wear.py --input-dir sample_images/ --output-dir outputs/wheel_wear_audit/
"""
import os
import sys
import glob
import csv
import json
import argparse
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.inference import MarsTerrainSegmenter


def run_wheel_wear_audit(
    input_dir: str,
    output_dir: str,
    img_size: int = 384,
    limit: int = 0,
    traverse_leg_m: float = 20.0,
    save_overlays: bool = False,
):
    os.makedirs(output_dir, exist_ok=True)
    if save_overlays:
        overlay_dir = os.path.join(output_dir, "overlays")
        os.makedirs(overlay_dir, exist_ok=True)

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
    records = []
    accumulated_loss_um = 0.0

    print("=" * 86)
    print("🔴 NASA JPL MARS ROVER FLEET WHEEL WEAR & VENTIFACT PUNCTURE AUDIT")
    print(f"Target: {len(image_paths)} scenes | Leg: {traverse_leg_m}m | Device: {segmenter.device}")
    print("=" * 86)

    for idx, img_path in enumerate(image_paths, start=1):
        fname = os.path.basename(img_path)
        res = segmenter.predict(img_path, img_size=img_size)
        t = res["telemetry"]
        wear = t["wheel_wear"]

        accumulated_loss_um += wear["leg_skin_loss_um"]

        row = {
            "image": fname,
            "safety_status": t["safety_status"],
            "bedrock_pct": round(t["terrain_distribution_pct"]["bedrock"], 2),
            "rock_pct": round(t["terrain_distribution_pct"]["big rock"], 2),
            "ventifact_roughness_index": wear["ventifact_roughness_index"],
            "peak_contact_stress_mpa": wear["peak_contact_stress_mpa"],
            "aluminum_yield_ratio": wear["aluminum_yield_ratio"],
            "puncture_risk_pct": wear["puncture_risk_pct"],
            "wear_rate_um_per_km": wear["wear_rate_um_per_km"],
            "leg_skin_loss_um": wear["leg_skin_loss_um"],
            "cumulative_skin_loss_um": round(accumulated_loss_um, 3),
            "wear_status": wear["wear_status"],
            "traction_control_mode": wear["traction_control_mode"],
        }
        records.append(row)

        if save_overlays:
            overlay_path = os.path.join(output_dir, "overlays", f"{os.path.splitext(fname)[0]}_wear.jpg")
            Image.fromarray(res["overlay_wheel_wear"]).save(overlay_path, quality=90)

        status_tag = wear["wear_status"].split("_")[0]
        print(
            f"  [{idx:>2}/{len(image_paths):<2}] {fname:<32} | "
            f"Stress={wear['peak_contact_stress_mpa']:>5.1f} MPa ({wear['aluminum_yield_ratio']:>4.2f}x) | "
            f"Puncture={wear['puncture_risk_pct']:>4.1f}% | {status_tag:<9} | {wear['traction_control_mode']}"
        )

    json_path = os.path.join(output_dir, "wheel_wear_audit_summary.json")
    with open(json_path, "w", encoding="utf-8") as jf:
        json.dump(records, jf, indent=2)

    csv_path = os.path.join(output_dir, "wheel_wear_audit_summary.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as cf:
        writer = csv.DictWriter(cf, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)

    total_distance_km = (len(image_paths) * traverse_leg_m) / 1000.0
    max_stress = max(r["peak_contact_stress_mpa"] for r in records)
    mean_wear_rate = np.mean([r["wear_rate_um_per_km"] for r in records])

    print("-" * 86)
    print(f"✓ Wheel Wear Audit Complete:")
    print(f"  • Total Projected Traverse:  {total_distance_km:.2f} km across {len(records)} legs")
    print(f"  • Peak Ventifact Stress:     {max_stress:.1f} MPa (Yield: 276.0 MPa)")
    print(f"  • Cumulative Skin Thickness Loss: {accumulated_loss_um:.3f} μm (out of 750 μm skin)")
    print(f"  • Mean Fleet Abrasive Rate:  {mean_wear_rate:.1f} μm/km")
    print(f"  • Output Summaries:          '{json_path}' and '{csv_path}'")
    print("=" * 86)

    return records


def main():
    parser = argparse.ArgumentParser(description="Fleet Wheel Wear & Ventifact Puncture Hazard Audit CLI")
    parser.add_argument("--input-dir", "-i", type=str, default="sample_images/", help="Input rover imagery directory")
    parser.add_argument("--output-dir", "-o", type=str, default="outputs/wheel_wear_audit/", help="Output directory")
    parser.add_argument("--resolution", "-r", type=int, default=384, help="Inference resolution")
    parser.add_argument("--limit", "-l", type=int, default=0, help="Max images to audit (0 = all)")
    parser.add_argument("--traverse-leg-m", type=float, default=20.0, help="Traverse distance per scene in meters")
    parser.add_argument("--save-overlays", action="store_true", help="Save ruby-red wheel puncture overlays")

    args = parser.parse_args()
    run_wheel_wear_audit(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        img_size=args.resolution,
        limit=args.limit,
        traverse_leg_m=args.traverse_leg_m,
        save_overlays=args.save_overlays,
    )


if __name__ == "__main__":
    main()
