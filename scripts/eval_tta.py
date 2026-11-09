"""
Benchmark & Evaluation Utility: Single-Pass vs Multi-View Test-Time Augmentation (TTA).
Quantifies mIoU, Per-Class IoU (especially Sand Hazard IoU), and inference latency
trade-offs across NASA JPL AI4Mars expert splits.

Usage:
  python scripts/eval_tta.py --split test_min3 --mode fast --max-samples 50
"""
import os
import sys
import time
import json
import argparse
import numpy as np
from PIL import Image
import torch
from datasets import load_dataset

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.dataset import prepare_row, CLASSES, IGNORE
from src.model import get_deeplabv3plus_model
from src.metrics import SegmentationMetrics
from src.tta import TestTimeAugmenter


def run_tta_benchmark(
    split_name: str = "test_min3",
    tta_mode: str = "fast",
    img_size: int = 512,
    max_samples: int = 50,
    checkpoint_path: str = "checkpoints/best_deeplabv3plus.pth",
    output_json: str = "checkpoints/tta_benchmark_summary.json"
):
    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running Single-Pass vs TTA ({tta_mode}) Benchmark on {device}...")

    parquet_path = os.path.join("data", f"{split_name}.parquet")
    if not os.path.exists(parquet_path):
        raise FileNotFoundError(f"Split file not found: {parquet_path}")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    ckpt = torch.load(checkpoint_path, map_location=device)
    model = get_deeplabv3plus_model(num_classes=4).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    tta_engine = TestTimeAugmenter(model, mode=tta_mode)
    ds = load_dataset("parquet", data_files={split_name: parquet_path})[split_name]

    limit = len(ds) if max_samples <= 0 else min(len(ds), max_samples)
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1).to(device)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1).to(device)

    metrics_single = SegmentationMetrics(num_classes=4, ignore_index=IGNORE, class_names=CLASSES)
    metrics_tta = SegmentationMetrics(num_classes=4, ignore_index=IGNORE, class_names=CLASSES)

    single_latencies_ms = []
    tta_latencies_ms = []
    evaluated_count = 0

    for idx in range(limit):
        row = ds[idx]
        img_np, label_np, _, _ = prepare_row(row)
        if label_np is None:
            continue

        orig_h, orig_w = label_np.shape
        img_pil = Image.fromarray(img_np).resize((img_size, img_size), Image.BILINEAR)
        img_arr = np.array(img_pil, dtype=np.float32) / 255.0
        img_tensor = torch.from_numpy(img_arr).permute(2, 0, 1).unsqueeze(0).to(device)
        img_tensor = (img_tensor - mean) / std

        with torch.no_grad():
            t0 = time.perf_counter()
            logits_single = model(img_tensor)
            pred_single_small = torch.argmax(logits_single, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
            single_latencies_ms.append((time.perf_counter() - t0) * 1000.0)

            t1 = time.perf_counter()
            logits_tta = tta_engine.predict_pseudo_logits(img_tensor)
            pred_tta_small = torch.argmax(logits_tta, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
            tta_latencies_ms.append((time.perf_counter() - t1) * 1000.0)

        pred_single = np.array(Image.fromarray(pred_single_small).resize((orig_w, orig_h), Image.NEAREST), dtype=np.int64)
        pred_tta = np.array(Image.fromarray(pred_tta_small).resize((orig_w, orig_h), Image.NEAREST), dtype=np.int64)

        metrics_single.update(pred_single, label_np)
        metrics_tta.update(pred_tta, label_np)
        evaluated_count += 1

    single_res = metrics_single.summary_dict()
    tta_res = metrics_tta.summary_dict()
    single_ms = float(np.mean(single_latencies_ms)) if single_latencies_ms else 0.0
    tta_ms = float(np.mean(tta_latencies_ms)) if tta_latencies_ms else 0.0

    print("\n" + "=" * 72)
    print(f"TTA ABLATION ({split_name} | {evaluated_count} images | mode={tta_mode} | res={img_size})")
    print("=" * 72)
    print(f"{'Metric':<20} | {'Single Pass':>14} | {'TTA Ensemble':>14} | {'Delta':>12}")
    print("-" * 72)
    for label, key in [("Mean IoU", "mean_iou"), ("Pixel Accuracy", "pixel_accuracy")]:
        s_val = single_res[key] * 100.0
        t_val = tta_res[key] * 100.0
        diff = t_val - s_val
        print(f"{label:<20} | {s_val:>13.2f}% | {t_val:>13.2f}% | {diff:>+11.2f}%")
    for c in CLASSES:
        s_val = single_res["per_class_iou"][c] * 100.0
        t_val = tta_res["per_class_iou"][c] * 100.0
        diff = t_val - s_val
        print(f"{(c + ' IoU'):<20} | {s_val:>13.2f}% | {t_val:>13.2f}% | {diff:>+11.2f}%")
    print("-" * 72)
    print(f"{'Mean Latency (ms)':<20} | {single_ms:>11.1f} ms | {tta_ms:>11.1f} ms | {tta_ms - single_ms:>+9.1f} ms")
    print("=" * 72)

    payload = {
        "split": split_name,
        "tta_mode": tta_mode,
        "resolution": img_size,
        "samples_evaluated": evaluated_count,
        "device": device,
        "single_pass": {
            **single_res,
            "mean_latency_ms": round(single_ms, 2)
        },
        "tta_ensemble": {
            **tta_res,
            "mean_latency_ms": round(tta_ms, 2)
        }
    }
    os.makedirs(os.path.dirname(os.path.abspath(output_json)), exist_ok=True)
    with open(output_json, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nSaved TTA benchmark summary to {output_json}")
    return payload


def main():
    parser = argparse.ArgumentParser(description="Evaluate Single-Pass vs TTA Segmentation Accuracy & Latency")
    parser.add_argument("--split", type=str, default="test_min3", choices=["test_min3", "test_min1"])
    parser.add_argument("--mode", type=str, default="fast", choices=["fast", "full"])
    parser.add_argument("--img-size", type=int, default=512)
    parser.add_argument("--max-samples", type=int, default=50, help="0 = evaluate entire split")
    parser.add_argument("--output", type=str, default="checkpoints/tta_benchmark_summary.json")
    args = parser.parse_args()

    run_tta_benchmark(
        split_name=args.split,
        tta_mode=args.mode,
        img_size=args.img_size,
        max_samples=args.max_samples,
        output_json=args.output
    )


if __name__ == "__main__":
    main()
