"""
Sequential Drive Stream Processor & Animated Telemetry HUD Generator.
Processes an ordered sequence of Mars rover camera frames, simulates continuous AutoNav traversal,
computes terramechanics telemetry across the drive, and compiles an animated drive HUD (GIF).

Usage:
  python scripts/process_drive_sequence.py --input-dir sample_images/ --output-dir outputs/drive_sequence/ --fps 2
"""
import os
import sys
import glob
import json
import argparse
from typing import List
from PIL import Image, ImageDraw, ImageFont
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.inference import MarsTerrainSegmenter
from src.path_planner import MarsPathPlanner
from src.terramechanics import RoverTerramechanicsEngine


def draw_hud_overlay(frame_np: np.ndarray, telemetry: dict, mechanics: dict, frame_idx: int, total_frames: int) -> Image.Image:
    """
    Renders space-grade telemetry HUD overlay directly onto the driving frame.
    """
    pil_img = Image.fromarray(frame_np).convert("RGBA")
    hud = Image.new("RGBA", pil_img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(hud)
    w, h = pil_img.size

    # Top Mission Banner
    draw.rectangle([(0, 0), (w, 52)], fill=(10, 15, 26, 210))
    draw.line([(0, 52), (w, 52)], fill=(239, 68, 68, 255), width=2)

    # Frame Counter & Header
    draw.text((16, 10), "🔴 NASA JPL MARS ROVER AUTONAV", fill=(255, 255, 255, 255))
    draw.text((16, 28), f"FRAME {frame_idx:02d}/{total_frames:02d} • SOL 4118 DRIVE TELEMETRY", fill=(148, 163, 184, 255))

    # Safety Status Badge
    status = telemetry["safety_status"]
    if "CRITICAL" in status:
        badge_color = (239, 68, 68, 255)
        badge_text = "HALT & REPLAN"
    elif "CAUTION" in status:
        badge_color = (245, 158, 11, 255)
        badge_text = "SLOW CREEP"
    else:
        badge_color = (16, 185, 129, 255)
        badge_text = "DRIVE NOMINAL"

    draw.rounded_rectangle([(w - 180, 10), (w - 16, 42)], radius=4, fill=badge_color)
    draw.text((w - 165, 18), badge_text, fill=(255, 255, 255, 255))

    # Bottom Telemetry HUD Box
    box_h = 80
    draw.rectangle([(0, h - box_h), (w, h)], fill=(10, 15, 26, 220))
    draw.line([(0, h - box_h), (w, h - box_h)], fill=(51, 65, 85, 255), width=1)

    # Telemetry Columns
    trav_idx = telemetry["traversability_index"]
    draw.text((20, h - 70), "TRAVERSABILITY", fill=(148, 163, 184, 255))
    draw.text((20, h - 50), f"{trav_idx:.1f} / 100", fill=(255, 255, 255, 255))

    sinkage = mechanics["mean_wheel_sinkage_cm"]
    draw.text((180, h - 70), "WHEEL SINKAGE", fill=(148, 163, 184, 255))
    draw.text((180, h - 50), f"{sinkage:.1f} cm", fill=(255, 152, 0, 255) if sinkage > 4.0 else (255, 255, 255, 255))

    slip = mechanics["expected_wheel_slip_pct"]
    draw.text((340, h - 70), "WHEEL SLIP", fill=(148, 163, 184, 255))
    draw.text((340, h - 50), f"{slip:.1f}%", fill=(239, 68, 68, 255) if slip > 35.0 else (255, 255, 255, 255))

    speed = telemetry["recommended_speed_cm_s"]
    draw.text((490, h - 70), "TARGET SPEED", fill=(148, 163, 184, 255))
    draw.text((490, h - 50), f"{speed:.1f} cm/s", fill=(255, 255, 255, 255))

    dist = telemetry["terrain_distribution_pct"]
    dist_str = f"Bedrock: {dist['bedrock']:.0f}%  |  Soil: {dist['soil']:.0f}%  |  Sand: {dist['sand']:.0f}%  |  Rock: {dist['big rock']:.0f}%"
    draw.text((20, h - 22), dist_str, fill=(203, 213, 225, 255))

    combined = Image.alpha_composite(pil_img, hud).convert("RGB")
    return combined


def process_sequence(input_dir: str, output_dir: str, max_frames: int = 8, fps: int = 2):
    os.makedirs(output_dir, exist_ok=True)
    valid_exts = (".jpg", ".jpeg", ".png")
    all_files = sorted([p for p in glob.glob(os.path.join(input_dir, "*")) if p.lower().endswith(valid_exts)])

    if not all_files:
        print(f"No image files found in {input_dir}")
        return

    selected_files = all_files[:max_frames]
    print(f"===========================================================")
    print(f"  NASA Mars Rover Drive Stream Processor")
    print(f"  Ingesting {len(selected_files)} sequential frames from '{input_dir}'")
    print(f"===========================================================\n")

    segmenter = MarsTerrainSegmenter()
    planner = MarsPathPlanner(obstacle_inflation_radius=2)
    mechanics_engine = RoverTerramechanicsEngine()

    processed_frames: List[Image.Image] = []
    traverse_history: List[dict] = []

    for idx, img_path in enumerate(selected_files, 1):
        base_name = os.path.basename(img_path)
        print(f"[{idx}/{len(selected_files)}] Processing {base_name}...")

        # 1. Inference & Terramechanics
        res = segmenter.predict(img_path, img_size=512)
        telemetry = res["telemetry"]
        dist = telemetry["terrain_distribution_pct"]
        mechanics = mechanics_engine.evaluate_terrain_mechanics(dist)

        # 2. Path Planning trajectory
        path_data = planner.plan_path(res["prediction_mask"], planning_resolution=128)
        trajectory_overlay = planner.render_path_overlay(res["overlay_multiclass"], path_data)

        # 3. Render HUD
        hud_frame = draw_hud_overlay(trajectory_overlay, telemetry, mechanics, idx, len(selected_files))
        processed_frames.append(hud_frame)

        # Record log entry
        log_entry = {
            "frame_index": idx,
            "filename": base_name,
            "traversability_index": telemetry["traversability_index"],
            "safety_status": telemetry["safety_status"],
            "wheel_sinkage_cm": mechanics["mean_wheel_sinkage_cm"],
            "wheel_slip_pct": mechanics["expected_wheel_slip_pct"],
            "terrain_distribution": dist,
            "path_status": path_data["status"],
            "total_path_cost": path_data["total_path_cost"]
        }
        traverse_history.append(log_entry)

    # Compile Animated Drive GIF
    gif_path = os.path.join(output_dir, "mars_rover_drive_hud.gif")
    duration_ms = int(1000.0 / fps)
    print(f"\nCompiling {len(processed_frames)} frames into animated sequence ({fps} FPS)...")
    
    # Save GIF
    processed_frames[0].save(
        gif_path,
        save_all=True,
        append_images=processed_frames[1:],
        duration=duration_ms,
        loop=0,
        optimize=True
    )
    gif_size_kb = os.path.getsize(gif_path) / 1024
    print(f"  ✓ Saved Animated Drive HUD -> {gif_path} ({gif_size_kb:.0f} KB)")

    # Save Traverse History JSON
    json_path = os.path.join(output_dir, "traverse_telemetry_log.json")
    with open(json_path, "w") as f:
        json.dump(traverse_history, f, indent=2)
    print(f"  ✓ Saved Traverse Telemetry Log -> {json_path}")
    print("\nDrive sequence compilation successfully completed!")


def main():
    parser = argparse.ArgumentParser(description="Process sequential rover drive frames into animated HUD")
    parser.add_argument("--input-dir", "-i", type=str, default="sample_images", help="Folder containing sequence images")
    parser.add_argument("--output-dir", "-o", type=str, default="outputs/drive_sequence", help="Output directory")
    parser.add_argument("--max-frames", "-n", type=int, default=6, help="Maximum frames to process")
    parser.add_argument("--fps", type=int, default=2, help="Playback frame rate")
    args = parser.parse_args()

    process_sequence(args.input_dir, args.output_dir, max_frames=args.max_frames, fps=args.fps)


if __name__ == "__main__":
    main()
