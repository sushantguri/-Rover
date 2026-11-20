"""
CLI Visual Odometry (VO) & Wheel Slip Audit Tool:
Evaluates FAST/gradient peak visual keypoint tracking, ground displacement,
terrain grounding (bedrock/soil vs sand deception), slip ratio, and confidence score
across single images or directory batches of Mars rover optical frames.

Usage:
  python scripts/audit_visual_odometry.py --input sample_images/ --output-dir outputs/vo_audit/
  python scripts/audit_visual_odometry.py --input sample_images/edr_sample_01.jpg --commanded-distance 1.5
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
from src.visual_odometry import RoverVisualOdometryModel


def audit_single_image(
    segmenter: MarsTerrainSegmenter,
    vo_model: RoverVisualOdometryModel,
    image_path: str,
    commanded_distance_m: float = 1.0,
    resolution: int = 512,
    save_overlay: bool = False,
    output_dir: str = None,
) -> dict:
    """Audits visual odometry metrics for a single image file."""
    fname = os.path.basename(image_path)
    img_pil = Image.open(image_path).convert("RGB")
    orig_np = np.array(img_pil)

    res = segmenter.predict(
        image_path,
        img_size=resolution,
        commanded_distance_m=commanded_distance_m,
    )
    mask = res["prediction_mask"]
    t = res["telemetry"]
    vo_t = t.get("visual_odometry", {})
    mech = t.get("terramechanics", {})

    record = {
        "image": fname,
        "resolution": resolution,
        "commanded_distance_m": commanded_distance_m,
        "actual_distance_m": vo_t.get("actual_distance_m", commanded_distance_m),
        "odometry_error_m": vo_t.get("odometry_error_m", 0.0),
        "total_keypoints": vo_t.get("total_keypoints", 0),
        "firm_anchor_count": vo_t.get("firm_anchor_count", 0),
        "bedrock_features": vo_t.get("bedrock_features", 0),
        "soil_features": vo_t.get("soil_features", 0),
        "sand_features": vo_t.get("sand_features", 0),
        "vo_confidence_pct": vo_t.get("vo_confidence_pct", 0.0),
        "wheel_slip_pct": vo_t.get("wheel_slip_pct", 0.0),
        "expected_slip_pct": mech.get("expected_wheel_slip_pct", 0.0),
        "vo_status": vo_t.get("vo_status", "N/A"),
        "slip_warning": vo_t.get("slip_warning", "NONE"),
        "safety_status": t.get("safety_status", "N/A"),
    }

    if save_overlay and output_dir:
        os.makedirs(output_dir, exist_ok=True)
        overlay_np = res.get("overlay_visual_odometry")
        if overlay_np is not None:
            out_img_path = os.path.join(output_dir, f"vo_overlay_{os.path.splitext(fname)[0]}.jpg")
            Image.fromarray(overlay_np).save(out_img_path, quality=90)
            record["overlay_path"] = out_img_path

    return record


def run_vo_audit(
    input_path: str,
    output_dir: str = None,
    commanded_distance_m: float = 1.0,
    resolution: int = 512,
    save_overlays: bool = False,
    limit: int = 0,
) -> list:
    """Executes VO slip audit across a single image file or directory of images."""
    if os.path.isfile(input_path):
        image_paths = [input_path]
    elif os.path.isdir(input_path):
        extensions = ("*.jpg", "*.jpeg", "*.png")
        image_paths = []
        for ext in extensions:
            image_paths.extend(glob.glob(os.path.join(input_path, ext)))
        image_paths = sorted(image_paths)
    else:
        raise FileNotFoundError(f"Input path '{input_path}' not found")

    if not image_paths:
        raise FileNotFoundError(f"No valid images found at '{input_path}'")

    if limit > 0:
        image_paths = image_paths[:limit]

    segmenter = MarsTerrainSegmenter()
    vo_model = RoverVisualOdometryModel()

    records = []
    print(f"Auditing Visual Odometry for {len(image_paths)} scenes (Cmd Dist: {commanded_distance_m:.1f}m)...")

    for idx, img_path in enumerate(image_paths, start=1):
        rec = audit_single_image(
            segmenter=segmenter,
            vo_model=vo_model,
            image_path=img_path,
            commanded_distance_m=commanded_distance_m,
            resolution=resolution,
            save_overlay=save_overlays,
            output_dir=output_dir,
        )
        records.append(rec)
        print(
            f"  [{idx}/{len(image_paths)}] {rec['image']:<32} | "
            f"KPs: {rec['total_keypoints']:>3} (Anchors: {rec['firm_anchor_count']:>3}) | "
            f"Slip: {rec['wheel_slip_pct']:>5.1f}% | Trust: {rec['vo_confidence_pct']:>5.1f}% | {rec['vo_status']}"
        )

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        json_path = os.path.join(output_dir, "vo_audit_summary.json")
        with open(json_path, "w", encoding="utf-8") as jf:
            json.dump(records, jf, indent=2)

        csv_path = os.path.join(output_dir, "vo_audit_summary.csv")
        with open(csv_path, "w", newline="", encoding="utf-8") as cf:
            writer = csv.DictWriter(cf, fieldnames=list(records[0].keys()))
            writer.writeheader()
            writer.writerows(records)

        print(f"\n✓ VO Slip Audit complete! Output saved to '{output_dir}'")

    return records


def main():
    parser = argparse.ArgumentParser(description="Audit Visual Odometry feature tracking & wheel slip across Mars frames")
    parser.add_argument("--input", "-i", type=str, required=True, help="Input image file or directory")
    parser.add_argument("--output-dir", "-o", type=str, default=None, help="Directory to save summary JSON, CSV, and overlays")
    parser.add_argument("--commanded-distance", "-d", type=float, default=1.0, help="Commanded wheel drive distance in meters")
    parser.add_argument("--resolution", "-r", type=int, default=512, help="Inference resolution")
    parser.add_argument("--save-overlays", action="store_true", help="Save annotated VO tracking overlay images")
    parser.add_argument("--limit", type=int, default=0, help="Max images to audit in directory mode (0 = all)")
    args = parser.parse_args()

    run_vo_audit(
        input_path=args.input,
        output_dir=args.output_dir,
        commanded_distance_m=args.commanded_distance,
        resolution=args.resolution,
        save_overlays=args.save_overlays,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
