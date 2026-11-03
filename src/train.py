"""
Training pipeline for DeepLabV3+ Mars Rover Terrain Segmentation.
Supports Apple Silicon MPS, NVIDIA CUDA, and CPU.
"""
import os
import time
import json
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset import load_ai4mars_splits, AI4MarsDataset, CLASSES, IGNORE
from src.model import get_deeplabv3plus_model, get_loss_fn
from src.metrics import SegmentationMetrics


def train_one_epoch(model, dataloader, optimizer, loss_fn, device, epoch, num_epochs):
    model.train()
    running_loss = 0.0
    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]", leave=False)
    
    for images, masks in pbar:
        images = images.to(device)
        masks = masks.to(device)
        
        optimizer.zero_grad()
        logits = model(images)
        loss = loss_fn(logits, masks)
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        
    avg_loss = running_loss / max(len(dataloader), 1)
    return avg_loss


@torch.no_grad()
def evaluate_model(model, dataloader, loss_fn, device, desc="Val"):
    model.eval()
    running_loss = 0.0
    metrics = SegmentationMetrics(num_classes=4, ignore_index=IGNORE, class_names=CLASSES)
    
    pbar = tqdm(dataloader, desc=f"Evaluating [{desc}]", leave=False)
    for images, masks in pbar:
        images = images.to(device)
        masks_cpu = masks.cpu().numpy()
        
        logits = model(images)
        loss = loss_fn(logits, masks.to(device))
        running_loss += loss.item()
        
        preds = torch.argmax(logits, dim=1).cpu().numpy()
        metrics.update(preds, masks_cpu)
        
    avg_loss = running_loss / max(len(dataloader), 1)
    results = metrics.summary_dict()
    results["loss"] = avg_loss
    return results, metrics


def main():
    parser = argparse.ArgumentParser(description="Train DeepLabV3+ on AI4Mars")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--img-size", type=int, default=512, help="Image resolution (H=W)")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="Validation set fraction")
    parser.add_argument("--output-dir", type=str, default="checkpoints", help="Output directory")
    parser.add_argument("--device", type=str, default="auto", help="Compute device: auto, mps, cuda, cpu")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    
    # 1. Device selection
    if args.device == "auto":
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    print(f"Using compute device: {device}")

    # 2. Load dataset splits
    print("Loading AI4Mars dataset splits...")
    splits = load_ai4mars_splits(data_dir="data", val_ratio=args.val_ratio)
    train_hf = splits["train"]
    val_hf = splits["val"]
    print(f"Train split size: {len(train_hf)}, Val split size: {len(val_hf)}")

    train_ds = AI4MarsDataset(train_hf, img_size=(args.img_size, args.img_size), is_train=True)
    val_ds = AI4MarsDataset(val_hf, img_size=(args.img_size, args.img_size), is_train=False)
    print(f"Valid labeled samples: {len(train_ds)} train, {len(val_ds)} val")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # 3. Load class weights
    weights_path = os.path.join(args.output_dir, "class_weights.json")
    if os.path.exists(weights_path):
        with open(weights_path, "r") as f:
            w_data = json.load(f)
            weights = w_data["weights"]
            print(f"Loaded class weights from {weights_path}: {weights}")
    else:
        # Fallback to balanced weights
        weights = [0.6551, 0.4909, 0.8634, 0.9906]
        print(f"Using default AI4Mars weights: {weights}")

    # 4. Model, Loss, Optimizer
    print("Initializing DeepLabV3Plus with ResNet-50 encoder...")
    model = get_deeplabv3plus_model(num_classes=4).to(device)
    loss_fn = get_loss_fn(weights=weights, ignore_index=IGNORE, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # 5. Training Loop
    best_miou = 0.0
    history = []
    print(f"\nStarting training for {args.epochs} epochs (Resolution: {args.img_size}x{args.img_size})...")

    start_time = time.time()
    for epoch in range(args.epochs):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device, epoch, args.epochs)
        scheduler.step()
        
        # Validation
        val_results, val_metrics = evaluate_model(model, val_loader, loss_fn, device, desc=f"Val Ep {epoch+1}")
        val_miou = val_results["mean_iou"]
        val_acc = val_results["pixel_accuracy"]
        val_loss = val_results["loss"]
        sand_iou = val_results["per_class_iou"]["sand"]
        
        elapsed = time.time() - t0
        print(f"Epoch {epoch+1:02d}/{args.epochs:02d} ({elapsed:.1f}s) | "
              f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
              f"Val mIoU: {val_miou*100:.2f}% | Val Acc: {val_acc*100:.2f}% | "
              f"Sand IoU: {sand_iou*100:.2f}%")

        record = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_mean_iou": val_miou,
            "val_pixel_accuracy": val_acc,
            "per_class_iou": val_results["per_class_iou"]
        }
        history.append(record)

        # Save best model
        if val_miou > best_miou:
            best_miou = val_miou
            best_ckpt = os.path.join(args.output_dir, "best_deeplabv3plus.pth")
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_mean_iou": val_miou,
                "val_results": val_results,
                "class_weights": weights,
                "classes": CLASSES,
                "img_size": args.img_size
            }, best_ckpt)
            print(f"  ★ New best model saved! (Val mIoU: {val_miou*100:.2f}%)")

    total_time = time.time() - start_time
    print(f"\nTraining completed in {total_time/60:.1f} min. Best Val mIoU: {best_miou*100:.2f}%")

    # Save training history
    with open(os.path.join(args.output_dir, "training_history.json"), "w") as f:
        json.dump(history, f, indent=2)


if __name__ == "__main__":
    main()

