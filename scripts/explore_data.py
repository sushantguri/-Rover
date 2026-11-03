"""
Step 1: Explore AI4Mars data.
Loads samples from train and test splits, analyzes the proportion of unlabeled pixels,
and generates multi-panel visual comparisons with rover/range exclusion masks.
"""
import os
import numpy as np
from datasets import load_dataset
from src.dataset import prepare_row, CLASSES, IGNORE
from src.visualization import plot_data_exploration, COLOR_MAP, HEX_COLORS

os.makedirs("figures", exist_ok=True)

print("=== AI4Mars Data Exploration ===")
print("Loading test_min3 and train_shard0 from local parquet...")

ds = load_dataset("parquet", data_files={
    "train": "data/train_shard0.parquet",
    "test_min3": "data/test_min3.parquet",
    "test_min1": "data/test_min1.parquet"
})

for split_name in ["train", "test_min3"]:
    split_data = ds[split_name]
    print(f"\nAnalyzing split: {split_name} ({len(split_data)} samples)...")
    
    unlabeled_ratios = []
    class_pixel_counts = np.zeros(4, dtype=np.int64)
    has_labels_count = 0
    saved_figures = 0
    
    for idx, row in enumerate(split_data):
        if not row.get("has_labels", True) or row.get("label_mask") is None:
            continue
            
        has_labels_count += 1
        img, final_lbl, rover_m, range_m = prepare_row(row)
        if final_lbl is None:
            continue
            
        total_px = final_lbl.size
        unlabeled_px = (final_lbl == IGNORE).sum()
        unlabeled_pct = (unlabeled_px / total_px) * 100.0
        unlabeled_ratios.append(unlabeled_pct)
        
        # Count labeled pixels
        labeled_mask = (final_lbl != IGNORE) & (final_lbl >= 0) & (final_lbl < 4)
        if labeled_mask.sum() > 0:
            class_pixel_counts += np.bincount(final_lbl[labeled_mask], minlength=4)[:4]
            
        # Save exploratory plots for the first 3 samples of each split
        if saved_figures < 3:
            raw_lbl = np.array(row["label_mask"])
            raw_lbl = raw_lbl[..., 0] if raw_lbl.ndim == 3 else raw_lbl
            save_path = f"figures/exploration_{split_name}_sample_{saved_figures+1}.png"
            plot_data_exploration(
                image=img,
                raw_label=raw_lbl,
                rover_mask=rover_m,
                range_mask=range_m,
                final_mask=final_lbl,
                sample_idx=idx,
                save_path=save_path
            )
            saved_figures += 1

    avg_unlabeled = np.mean(unlabeled_ratios) if unlabeled_ratios else 0.0
    median_unlabeled = np.median(unlabeled_ratios) if unlabeled_ratios else 0.0
    print(f"Results for {split_name}:")
    print(f"  Samples with labels: {has_labels_count} / {len(split_data)}")
    print(f"  Average unlabeled area: {avg_unlabeled:.2f}% (Median: {median_unlabeled:.2f}%)")
    print("  Class pixel distribution among labeled pixels:")
    tot_lbl = class_pixel_counts.sum()
    for c_i, c_name in enumerate(CLASSES):
        pct = (class_pixel_counts[c_i] / tot_lbl * 100) if tot_lbl > 0 else 0
        print(f"    - {c_name:<10}: {class_pixel_counts[c_i]:>12,d} px ({pct:>5.2f}%)")

print("\nData exploration complete! Check figures/ folder for generated plots.")

