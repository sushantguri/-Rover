"""
Step 7: Evaluation script.
Evaluates DeepLabV3+ model on expert test sets (test_min3 and test_min1).
Generates confusion matrices, per-class IoU, mean IoU, pixel accuracy,
and direct comparison with the All-Soil baseline.
"""
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import torch
from datasets import load_dataset
from PIL import Image

from src.dataset import prepare_row, CLASSES, IGNORE
from src.model import get_deeplabv3plus_model
from src.metrics import SegmentationMetrics
from src.visualization import COLOR_MAP, mask_to_rgb, create_overlay

os.makedirs("figures", exist_ok=True)


def plot_confusion_matrix(cm, class_names, title, save_path):
    """
    Plots normalized confusion matrix heatmap.
    """
    # Normalize by true class row sums
    row_sums = cm.sum(axis=1, keepdims=True)
    norm_cm = np.divide(cm.astype(float), row_sums, out=np.zeros_like(cm, dtype=float), where=(row_sums > 0))
    
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(norm_cm, cmap="Blues", vmin=0, vmax=1)
    
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, fontsize=11, fontweight="bold")
    ax.set_yticklabels(class_names, fontsize=11, fontweight="bold")
    ax.set_xlabel("Predicted Class", fontsize=12, fontweight="bold")
    ax.set_ylabel("True Expert Class", fontsize=12, fontweight="bold")
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    
    # Overlay text annotations with counts and percentages
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            color = "white" if norm_cm[i, j] > 0.5 else "black"
            text = f"{norm_cm[i, j]*100:.1f}%\n({cm[i, j]:,d})"
            ax.text(j, i, text, ha="center", va="center", color=color, fontsize=9)
            
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved confusion matrix plot: {save_path}")


def evaluate_split(model, dataset_split, split_name, device, img_size=512, save_samples=3):
    """
    Evaluates model over an entire dataset split at original or test resolution.
    """
    print(f"\n=======================================================")
    print(f"Evaluating on {split_name} ({len(dataset_split)} samples)...")
    print(f"=======================================================")
    
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
    
    metrics = SegmentationMetrics(num_classes=4, ignore_index=IGNORE, class_names=CLASSES)
    saved_count = 0
    
    for idx, row in enumerate(dataset_split):
        img_np, label_np, rover_m, range_m = prepare_row(row)
        if label_np is None:
            continue
            
        orig_h, orig_w = label_np.shape
        
        # Prepare tensor input
        img_pil = Image.fromarray(img_np).resize((img_size, img_size), Image.BILINEAR)
        img_arr = np.array(img_pil, dtype=np.float32) / 255.0
        img_tensor = torch.from_numpy(img_arr).permute(2, 0, 1).unsqueeze(0).to(device)
        img_tensor = (img_tensor - mean) / std
        
        with torch.no_grad():
            logits = model(img_tensor)
            pred_small = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
            
        # Resize prediction back to original image size using NEAREST interpolation
        pred_pil = Image.fromarray(pred_small).resize((orig_w, orig_h), Image.NEAREST)
        pred_np = np.array(pred_pil, dtype=np.int64)
        
        # Update metrics (strictly excluding 255 labeled pixels)
        metrics.update(pred_np, label_np)
        
        # Save qualitative prediction comparisons
        if saved_count < save_samples:
            fig, axes = plt.subplots(1, 4, figsize=(20, 5))
            axes[0].imshow(img_np)
            axes[0].set_title(f"Mars Rover Camera #{idx+1}", fontweight="bold")
            axes[0].axis("off")
            
            axes[1].imshow(create_overlay(img_np, label_np, alpha=0.55))
            axes[1].set_title("Expert Ground Truth", fontweight="bold")
            axes[1].axis("off")
            
            axes[2].imshow(create_overlay(img_np, pred_np, alpha=0.55))
            axes[2].set_title("DeepLabV3+ Prediction", fontweight="bold")
            axes[2].axis("off")
            
            # Highlight Sand Hazard overlay
            axes[3].imshow(create_overlay(img_np, pred_np, alpha=0.6, highlight_sand=True))
            axes[3].set_title("Sand Hazard Alert Overlay", fontweight="bold", color="darkorange")
            axes[3].axis("off")
            
            plt.tight_layout()
            pred_path = f"figures/prediction_{split_name}_sample_{saved_count+1}.png"
            plt.savefig(pred_path, dpi=200, bbox_inches="tight")
            plt.close()
            saved_count += 1

    # Print markdown evaluation table
    table_str = metrics.format_table(title=f"DeepLabV3+ Performance on {split_name}")
    print(table_str)
    
    # Save confusion matrix plot
    cm_path = f"figures/confusion_matrix_{split_name}.png"
    plot_confusion_matrix(metrics.confusion_matrix, CLASSES,
                          title=f"Confusion Matrix: DeepLabV3+ on {split_name}",
                          save_path=cm_path)
                          
    summary = metrics.summary_dict()
    with open(f"checkpoints/eval_{split_name}_results.json", "w") as f:
        json.dump(summary, f, indent=2)
        
    return summary


def main():
    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading best model checkpoint onto {device}...")
    
    ckpt_path = "checkpoints/best_deeplabv3plus.pth"
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}. Run training first.")
        
    ckpt = torch.load(ckpt_path, map_location=device)
    model = get_deeplabv3plus_model(num_classes=4).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Model loaded successfully from epoch {ckpt['epoch']} (Saved Val mIoU: {ckpt['val_mean_iou']*100:.2f}%)")
    
    # Load test splits
    print("Loading test splits...")
    ds = load_dataset("parquet", data_files={
        "test_min3": "data/test_min3.parquet",
        "test_min1": "data/test_min1.parquet"
    })
    
    # Evaluate test_min3 (expert consensus >= 3)
    res_min3 = evaluate_split(model, ds["test_min3"], "test_min3", device, img_size=512)
    
    # Evaluate test_min1 (expert consensus >= 1)
    res_min1 = evaluate_split(model, ds["test_min1"], "test_min1", device, img_size=512)
    
    # Load baselines for comparative report
    print("\n=======================================================")
    print("COMPARATIVE SUMMARY: BASELINE VS DEEPLABV3+")
    print("=======================================================")
    for s_name, res in [("test_min3", res_min3), ("test_min1", res_min1)]:
        base_path = f"checkpoints/baseline_{s_name}_metrics.json"
        if os.path.exists(base_path):
            with open(base_path) as bf:
                base = json.load(bf)
            print(f"\n--- {s_name.upper()} ---")
            print(f"{'Metric':<18} | {'All-Soil Baseline':<20} | {'DeepLabV3+':<18} | {'Delta':<12}")
            print("-" * 75)
            print(f"{'Mean IoU':<18} | {base['mean_iou']*100:>18.2f}% | {res['mean_iou']*100:>16.2f}% | {('+' if res['mean_iou']>=base['mean_iou'] else '') + f'{(res['mean_iou']-base['mean_iou'])*100:.2f}%':>12}")
            print(f"{'Pixel Accuracy':<18} | {base['pixel_accuracy']*100:>18.2f}% | {res['pixel_accuracy']*100:>16.2f}% | {('+' if res['pixel_accuracy']>=base['pixel_accuracy'] else '') + f'{(res['pixel_accuracy']-base['pixel_accuracy'])*100:.2f}%':>12}")
            for c in CLASSES:
                b_iou = base['per_class_iou'][c]
                m_iou = res['per_class_iou'][c]
                print(f"{c + ' IoU':<18} | {b_iou*100:>18.2f}% | {m_iou*100:>16.2f}% | {('+' if m_iou>=b_iou else '') + f'{(m_iou-b_iou)*100:.2f}%':>12}")


if __name__ == "__main__":
    main()

