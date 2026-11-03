"""
Step 8: Sand Error Analysis.
Identifies images where the model misses sand (the critical hazard that trapped the Spirit rover),
analyzes failure modes (shadows, lighting, ripple texture absence), and generates diagnostic figures.
"""
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
from datasets import load_dataset

from src.dataset import prepare_row, CLASSES, IGNORE
from src.model import get_deeplabv3plus_model
from src.visualization import mask_to_rgb, create_overlay

os.makedirs("figures", exist_ok=True)


def analyze_sand_errors(model, test_dataset, device, img_size=512, top_k=5):
    """
    Scans test dataset for sand hazard errors.
    Sand class ID: 2.
    """
    SAND_CLS = 2
    
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
    
    records = []
    
    for idx, row in enumerate(test_dataset):
        img_np, label_np, rover_m, range_m = prepare_row(row)
        if label_np is None:
            continue
            
        orig_h, orig_w = label_np.shape
        
        # Check if sand exists in ground truth
        sand_true = (label_np == SAND_CLS)
        sand_true_px = sand_true.sum()
        
        # Model prediction
        img_pil = Image.fromarray(img_np).resize((img_size, img_size), Image.BILINEAR)
        img_arr = np.array(img_pil, dtype=np.float32) / 255.0
        img_tensor = torch.from_numpy(img_arr).permute(2, 0, 1).unsqueeze(0).to(device)
        img_tensor = (img_tensor - mean) / std
        
        with torch.no_grad():
            logits = model(img_tensor)
            pred_small = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
            
        pred_pil = Image.fromarray(pred_small).resize((orig_w, orig_h), Image.NEAREST)
        pred_np = np.array(pred_pil, dtype=np.int64)
        
        # Only evaluate labeled pixels
        valid = (label_np != IGNORE) & (label_np >= 0) & (label_np < 4)
        sand_pred = (pred_np == SAND_CLS) & valid
        
        tp = np.logical_and(sand_true, sand_pred).sum()
        fn = np.logical_and(sand_true, ~sand_pred).sum()  # Missed sand (danger!)
        fp = np.logical_and(~sand_true & valid, sand_pred).sum()  # False alarm
        union = tp + fn + fp
        
        iou = tp / union if union > 0 else 0.0
        
        # What did missed sand get confused with?
        confused_with = {}
        if fn > 0:
            missed_targets = pred_np[sand_true & (~sand_pred)]
            for c_id, c_name in enumerate(CLASSES):
                confused_with[c_name] = int((missed_targets == c_id).sum())
                
        records.append({
            "idx": idx,
            "sand_true_px": int(sand_true_px),
            "tp": int(tp),
            "fn": int(fn),
            "fp": int(fp),
            "iou": float(iou),
            "confused_with": confused_with,
            "image": img_np,
            "label": label_np,
            "pred": pred_np
        })

    # Filter to images with significant sand in ground truth (> 5,000 pixels)
    sand_images = [r for r in records if r["sand_true_px"] > 5000]
    
    # Sort by lowest IoU (worst misses)
    sand_images.sort(key=lambda x: x["iou"])
    
    print(f"\nAnalyzed {len(records)} test images. Found {len(sand_images)} images with prominent sand hazards (>5,000 px).")
    print(f"Generating top {top_k} sand error diagnostic figures...")
    
    top_misses = sand_images[:top_k]
    
    for rank, item in enumerate(top_misses):
        idx = item["idx"]
        img = item["image"]
        lbl = item["label"]
        prd = item["pred"]
        
        fig, axes = plt.subplots(1, 4, figsize=(22, 5.5))
        
        # 1. Rover Camera
        axes[0].imshow(img)
        axes[0].set_title(f"Mars Rover Navcam #{idx}", fontsize=12, fontweight="bold")
        axes[0].axis("off")
        
        # 2. Expert Ground Truth
        axes[1].imshow(create_overlay(img, lbl, alpha=0.55))
        axes[1].set_title("Expert Ground Truth\n(Orange = Sand)", fontsize=12, fontweight="bold")
        axes[1].axis("off")
        
        # 3. Model Prediction
        axes[2].imshow(create_overlay(img, prd, alpha=0.55))
        axes[2].set_title(f"DeepLabV3+ Prediction\n(Sand IoU: {item['iou']*100:.1f}%)", fontsize=12, fontweight="bold")
        axes[2].axis("off")
        
        # 4. Error Map: Green = TP Sand, Red = Missed Sand (FN), Yellow = False Alarm Sand (FP)
        error_map = np.zeros((*lbl.shape, 3), dtype=np.uint8)
        # Background dark grey
        error_map[:] = [35, 35, 35]
        
        valid = (lbl != IGNORE) & (lbl >= 0) & (lbl < 4)
        sand_true = (lbl == SAND_CLS)
        sand_pred = (prd == SAND_CLS) & valid
        
        tp_mask = sand_true & sand_pred
        fn_mask = sand_true & (~sand_pred)
        fp_mask = (~sand_true) & sand_pred
        
        error_map[tp_mask] = [0, 230, 118]    # Bright Green: Correct Sand
        error_map[fn_mask] = [255, 23, 68]    # Bright Red: MISSED SAND (DANGER!)
        error_map[fp_mask] = [255, 214, 0]    # Amber/Yellow: False Sand Alarm
        
        axes[3].imshow(error_map)
        axes[3].set_title(f"Sand Hazard Error Map\nRed: Missed ({item['fn']:,} px) | Green: Hit", fontsize=12, fontweight="bold", color="crimson")
        axes[3].axis("off")
        
        plt.tight_layout()
        save_path = f"figures/sand_error_case_{rank+1}_img_{idx}.png"
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"  [Case {rank+1}] Image #{idx}: Sand IoU = {item['iou']*100:.1f}%, Missed {item['fn']:,} px. Confused with: {item['confused_with']}")

    # Compile findings
    total_missed_as_soil = sum(r["confused_with"].get("soil", 0) for r in sand_images)
    total_missed_as_bedrock = sum(r["confused_with"].get("bedrock", 0) for r in sand_images)
    total_missed_as_rock = sum(r["confused_with"].get("big rock", 0) for r in sand_images)
    tot_miss = total_missed_as_soil + total_missed_as_bedrock + total_missed_as_rock
    
    findings = {
        "total_sand_images": len(sand_images),
        "mean_sand_iou": float(np.mean([r["iou"] for r in sand_images])),
        "missed_as_soil_pct": float(total_missed_as_soil / tot_miss * 100) if tot_miss > 0 else 0,
        "missed_as_bedrock_pct": float(total_missed_as_bedrock / tot_miss * 100) if tot_miss > 0 else 0,
        "missed_as_bigrock_pct": float(total_missed_as_rock / tot_miss * 100) if tot_miss > 0 else 0,
        "top_miss_cases": [
            {
                "rank": r_i + 1,
                "image_idx": m["idx"],
                "sand_iou": m["iou"],
                "sand_true_px": m["sand_true_px"],
                "missed_px": m["fn"],
                "confused_with": m["confused_with"]
            }
            for r_i, m in enumerate(top_misses)
        ]
    }
    
    with open("checkpoints/sand_error_analysis.json", "w") as f:
        json.dump(findings, f, indent=2)
    print("\nSand error analysis report saved to checkpoints/sand_error_analysis.json")


def main():
    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading best model checkpoint onto {device}...")
    
    ckpt_path = "checkpoints/best_deeplabv3plus.pth"
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}.")
        
    ckpt = torch.load(ckpt_path, map_location=device)
    model = get_deeplabv3plus_model(num_classes=4).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    
    print("Loading test_min3 (Expert consensus >= 3)...")
    ds = load_dataset("parquet", data_files={"test_min3": "data/test_min3.parquet"})
    analyze_sand_errors(model, ds["test_min3"], device, img_size=512, top_k=4)


if __name__ == "__main__":
    main()

