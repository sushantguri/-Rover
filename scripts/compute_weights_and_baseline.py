"""
Step 4: Compute class weights.
Step 5: Build all-soil baseline and evaluate on test_min3 and test_min1.
"""
import os
import json
import numpy as np
from datasets import load_dataset
from src.dataset import prepare_row, CLASSES, IGNORE
from src.metrics import SegmentationMetrics, compute_ai4mars_class_weights

print("=== Step 4: Computing Class Distribution and Weights ===")
ds = load_dataset("parquet", data_files={
    "train": "data/train_shard0.parquet",
    "test_min3": "data/test_min3.parquet",
    "test_min1": "data/test_min1.parquet"
})

train_split = ds["train"]
stats = compute_ai4mars_class_weights(train_split, max_samples=len(train_split))
counts = stats["counts"]
shares = stats["shares"]
weights = stats["weights"]

print("\nClass Statistics on Training Data:")
print(f"{'Class':<12} | {'Pixel Count':<14} | {'Share (%)':<10} | {'Weight (1 - share)':<18}")
print("-" * 62)
for i, name in enumerate(CLASSES):
    print(f"{name:<12} | {counts[i]:>14,d} | {shares[i]*100:>8.2f}% | {weights[i]:>18.4f}")
print("-" * 62)
print(f"Total labeled pixels analyzed: {counts.sum():,d}")
print(f"Computed weights tensor: {weights.tolist()}")

# Save weights to json for training script
os.makedirs("checkpoints", exist_ok=True)
with open("checkpoints/class_weights.json", "w") as f:
    json.dump({
        "classes": CLASSES,
        "counts": counts.tolist(),
        "shares": shares.tolist(),
        "weights": weights.tolist()
    }, f, indent=2)
print("Saved class weights to checkpoints/class_weights.json")

print("\n=== Step 5: Evaluating 'All Soil' Baseline ===")
# Test baseline on both test_min3 (unanimous experts) and test_min1
for test_name in ["test_min3", "test_min1"]:
    test_ds = ds[test_name]
    metrics = SegmentationMetrics(num_classes=4, ignore_index=IGNORE, class_names=CLASSES)
    
    for row in test_ds:
        _, label, _, _ = prepare_row(row)
        if label is None:
            continue
        # Baseline predicts "soil" (class 0) for every pixel
        pred_soil = np.zeros_like(label, dtype=np.int64)
        metrics.update(pred_soil, label)
        
    print(f"\n{metrics.format_table(title=f'Baseline (All-Soil) on {test_name}')}")
    
    summary = metrics.summary_dict()
    print(f"Key insight for {test_name}:")
    print(f"  Pixel Accuracy is {summary['pixel_accuracy']*100:.2f}%, yet Sand IoU is {summary['per_class_iou']['sand']*100:.2f}% and Mean IoU is only {summary['mean_iou']*100:.2f}%.")
    print("  This demonstrates why pixel accuracy alone is misleading for autonomous driving safety!")

    with open(f"checkpoints/baseline_{test_name}_metrics.json", "w") as f:
        json.dump(summary, f, indent=2)

