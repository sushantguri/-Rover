"""
CLI Fleet Tool: Audits Mars Rover Navcam Stereo 3D Elevation Step Hazards.
Evaluates positive vertical step obstacles (>25cm critical threshold), negative drop-off
trenches (>35cm rollover risk), belly ground clearance margins (60cm rocker-bogie envelope),
and stereo disparity ranges across candidate Martian navigation frames.

Usage:
  python scripts/audit_stereo_geometry.py --input-dir sample_images/ --output-dir outputs/stereo_audit/
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


def run_stereo_geometry_audit(
    input_dir: str,
    output_dir: str,
    img_size: int = 384,
    limit: int = 0,
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

    print("=" * 90)
    print("🔴 NASA JPL MARS ROVER NAVCAM STEREO 3D ELEVATION & STEP HAZARD AUDIT")
    print(f"Target: {len(image_paths)} scenes | Focal: 1400px | Baseline: 0.424m | Device: {segmenter.device}")
    print("=" * 90)

    for idx, img_path in enumerate(image_paths, start=1):
        fname = os.path.basename(img_path)
        res = segmenter.predict(img_path, img_size=img_size)
        t = res["telemetry"]
        stereo = t.get("stereo_geometry", {})

        row = {
            "image": fname,
            "safety_status": t["safety_status"],
            "traversability_index": t["traversability_index"],
            "max_positive_step_cm": stereo.get("max_positive_step_cm", 0.0),
            "max_negative_dropoff_cm": stereo.get("max_negative_dropoff_cm", 0.0),
            "belly_clearance_cm": stereo.get("belly_clearance_cm", 0.0),
            "clearance_margin_pct": stereo.get("clearance_margin_pct", 0.0),
            "step_hazard_status": stereo.get("step_hazard_status", "UNKNOWN"),
            "traverse_permitted": stereo.get("traverse_permitted", True),
            "disparity_min_px": stereo.get("disparity_min_px", 0.0),
            "disparity_max_px": stereo.get("disparity_max_px", 0.0),
            "mean_depth_m": stereo.get("mean_depth_m", 0.0),
        }
        records.append(row)

        if save_overlays and "overlay_stereo_elevation" in res:
            overlay_path = os.path.join(output_dir, "overlays", f"{os.path.splitext(fname)[0]}_stereo.jpg")
            Image.fromarray(res["overlay_stereo_elevation"]).save(overlay_path, quality=90)

        step_tag = row["step_hazard_status"]
        print(
            f"  [{idx:>2}/{len(image_paths):<2}] {fname:<32} | "
            f"Step={row['max_positive_step_cm']:>5.1f}cm | Drop={row['max_negative_dropoff_cm']:>5.1f}cm | "
            f"BellyClear={row['belly_clearance_cm']:>5.1f}cm ({row['clearance_margin_pct']:>3.0f}%) | {step_tag:<22}"
        )

    json_path = os.path.join(output_dir, "stereo_geometry_audit_summary.json")
    with open(json_path, "w", encoding="utf-8") as jf:
        json.dump(records, jf, indent=2)

    csv_path = os.path.join(output_dir, "stereo_geometry_audit_summary.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as cf:
        writer = csv.DictWriter(cf, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)

    print(f"\n✓ Stereo 3D audit complete! Saved {len(records)} records to '{output_dir}'")
    return records


def parse_args():
    parser = argparse.ArgumentParser(
        description="Batch audit Mars rover scenes for 3D step obstacles and stereo disparity hazards."
    )
    parser.add_argument("--input-dir", type=str, default="sample_images/", help="Directory containing rover images")
    parser.add_argument("--output-dir", type=str, default="outputs/stereo_audit/", help="Directory to save audit summaries")
    parser.add_argument("--img-size", type=int, default=384, help="Inference resolution")
    parser.add_argument("--limit", type=int, default=0, help="Max images to process (0 = all)")
    parser.add_argument("--save-overlays", action="store_true", help="Save 3D stereo elevation overlays")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_stereo_geometry_audit(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        img_size=args.img_size,
        limit=args.limit,
        save_overlays=args.save_overlays,
    )
