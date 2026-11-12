"""
CLI Script for Mars Rover Terrain Segmentation & Telemetry Generation.
Run inference on single images or entire folders.

Usage:
  python scripts/predict.py --input sample_images/test_rocky_terrain.jpg --output-dir outputs/
  python scripts/predict.py --input sample_images/ --output-dir outputs/ --resolution 512
"""
import os
import sys
import glob
import json
import argparse
from PIL import Image

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.inference import MarsTerrainSegmenter


def main():
    parser = argparse.ArgumentParser(description="Run Mars Rover Terrain Segmentation & Telemetry")
    parser.add_argument("--input", "-i", type=str, required=True, help="Path to an image file or directory of images")
    parser.add_argument("--output-dir", "-o", type=str, default="outputs", help="Directory to save output overlays and JSON telemetry")
    parser.add_argument("--checkpoint", "-c", type=str, default="checkpoints/best_deeplabv3plus.pth", help="Model checkpoint path")
    parser.add_argument("--resolution", "-r", type=int, default=512, help="Inference image resolution")
    parser.add_argument("--sand-threshold", type=float, default=25.0, help="Sand threshold for emergency halt alert (%)")
    parser.add_argument("--opacity", type=float, default=0.60, help="Overlay opacity")
    parser.add_argument("--tta", action="store_true", help="Enable Test-Time Augmentation ensemble")
    parser.add_argument("--auto-tta", action="store_true", help="Auto-enable TTA when optical watchdog flags glare/haze/darkness")
    parser.add_argument("--pitch", type=float, default=0.0, help="Rover longitudinal pitch angle (deg)")
    parser.add_argument("--roll", type=float, default=0.0, help="Rover lateral roll angle (deg)")
    parser.add_argument("--device", type=str, default=None, help="Compute device (mps, cuda, cpu)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Collect images
    if os.path.isdir(args.input):
        valid_exts = (".jpg", ".jpeg", ".png")
        image_paths = [p for p in glob.glob(os.path.join(args.input, "*")) if p.lower().endswith(valid_exts)]
    else:
        image_paths = [args.input]

    if not image_paths:
        print(f"No valid images found in {args.input}")
        return

    print(f"Initializing Mars Terrain Segmenter on {args.device or 'auto'}...")
    segmenter = MarsTerrainSegmenter(checkpoint_path=args.checkpoint, device=args.device)
    print(f"Found {len(image_paths)} image(s) to process. Outputs will be saved to '{args.output_dir}'\n")

    for idx, img_path in enumerate(image_paths, 1):
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        print(f"[{idx}/{len(image_paths)}] Processing: {base_name}...")

        res = segmenter.predict(
            image_input=img_path,
            img_size=args.resolution,
            sand_threshold=args.sand_threshold,
            opacity=args.opacity,
            use_tta=args.tta,
            auto_tta=args.auto_tta,
            pitch_deg=args.pitch,
            roll_deg=args.roll,
        )

        telemetry = res["telemetry"]
        
        # Save output images
        multiclass_path = os.path.join(args.output_dir, f"{base_name}_terrain.jpg")
        sand_hazard_path = os.path.join(args.output_dir, f"{base_name}_sand_hazard.jpg")
        costmap_path = os.path.join(args.output_dir, f"{base_name}_costmap.jpg")
        safety_margin_path = os.path.join(args.output_dir, f"{base_name}_safety_margin.jpg")
        telemetry_path = os.path.join(args.output_dir, f"{base_name}_telemetry.json")

        Image.fromarray(res["overlay_multiclass"]).save(multiclass_path, quality=95)
        Image.fromarray(res["overlay_sand_hazard"]).save(sand_hazard_path, quality=95)
        Image.fromarray(res["overlay_costmap"]).save(costmap_path, quality=95)
        Image.fromarray(res["overlay_safety_margin"]).save(safety_margin_path, quality=95)

        with open(telemetry_path, "w") as f:
            json.dump(telemetry, f, indent=2)

        # Print summary
        dist = telemetry["terrain_distribution_pct"]
        sensor = telemetry["sensor_health"]
        cal = telemetry["calibration_safety"]
        slope = telemetry["slope_stability"]
        print(f"  ✓ Traversability Index: {telemetry['traversability_index']}/100.0 | Status: {telemetry['safety_status']}")
        print(f"  ✓ Distribution: Bedrock={dist['bedrock']:.1f}% | Soil={dist['soil']:.1f}% | Sand={dist['sand']:.1f}% | BigRock={dist['big rock']:.1f}%")
        print(f"  ✓ Watchdog: {sensor['sensor_health_status']} (Trust {sensor['optical_trust_score']:.0f}%) | Conformal Sand: {cal['conservative_sand_pct']:.1f}% ({cal['conformal_alert']})")
        print(f"  ✓ Attitude: {slope['stability_status']} ({slope['combined_tilt_deg']:.1f}°) | Drive Command: {telemetry['autonomous_drive_action']} ({telemetry['recommended_speed_cm_s']} cm/s)\n")

    print(f"All {len(image_paths)} image(s) successfully processed! Saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
