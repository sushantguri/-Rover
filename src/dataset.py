"""
AI4Mars Dataset loader, preprocessing, masking, and PyTorch Dataset implementation.
"""
import os
import random
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from datasets import load_dataset
import torchvision.transforms.functional as TF

CLASSES = ["soil", "bedrock", "sand", "big rock"]
IGNORE = 255  # Unlabeled / ignored pixels


def prepare_row(row):
    """
    Extracts image and builds the cleaned ground truth label mask.
    Applies both rover_mask (rover body) and range_mask (> 30 m) by setting
    excluded regions to IGNORE (255).
    """
    if row.get("label_mask") is None:
        return None, None, None, None
    
    image = np.array(row["image"].convert("RGB"))
    label = np.array(row["label_mask"])
    label = label[..., 0] if label.ndim == 3 else label
    label = label.astype(np.int64)
    
    rover_m = None
    range_m = None
    
    if row.get("rover_mask") is not None:
        rover_m = np.array(row["rover_mask"])
        rover_m = rover_m[..., 0] if rover_m.ndim == 3 else rover_m
        label[rover_m > 0] = IGNORE
        
    if row.get("range_mask") is not None:
        range_m = np.array(row["range_mask"])
        range_m = range_m[..., 0] if range_m.ndim == 3 else range_m
        label[range_m > 0] = IGNORE
        
    return image, label, rover_m, range_m


class AI4MarsDataset(Dataset):
    """
    PyTorch Dataset wrapper for AI4Mars dataset.
    
    Args:
        hf_dataset: HuggingFace dataset or list of rows
        img_size: Tuple (height, width) for model input (default: 512, 512)
        is_train: Whether to apply data augmentations (flips, color jitter)
        filter_empty: If True, skips images with no labeled pixels
    """
    def __init__(self, hf_dataset, img_size=(512, 512), is_train=False, filter_empty=True):
        self.img_size = img_size
        self.is_train = is_train
        
        # Pre-filter valid samples
        self.valid_indices = []
        for i in range(len(hf_dataset)):
            row = hf_dataset[i]
            if not row.get("has_labels", True) or row.get("label_mask") is None:
                continue
            self.valid_indices.append(i)
            
        self.dataset = hf_dataset
        
        # ImageNet normalization stats
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        real_idx = self.valid_indices[idx]
        row = self.dataset[real_idx]
        
        img_np, label_np, _, _ = prepare_row(row)
        
        # Convert to PIL for transformations
        img_pil = Image.fromarray(img_np)
        mask_pil = Image.fromarray(label_np.astype(np.uint8))
        
        # Resize: Bilinear for image, NEAREST for mask to avoid corrupting discrete class IDs
        if self.img_size is not None:
            img_pil = TF.resize(img_pil, self.img_size, interpolation=TF.InterpolationMode.BILINEAR)
            mask_pil = TF.resize(mask_pil, self.img_size, interpolation=TF.InterpolationMode.NEAREST)
            
        # Data Augmentations (training only)
        if self.is_train:
            # 1. Random horizontal flip (applied to BOTH image and mask)
            if random.random() > 0.5:
                img_pil = TF.hflip(img_pil)
                mask_pil = TF.hflip(mask_pil)
                
            # 2. Photometric jitter (applied ONLY to image)
            if random.random() > 0.3:
                # Slight brightness and contrast jitter mimicking Martian sun angles
                brightness_factor = random.uniform(0.8, 1.2)
                contrast_factor = random.uniform(0.8, 1.2)
                img_pil = TF.adjust_brightness(img_pil, brightness_factor)
                img_pil = TF.adjust_contrast(img_pil, contrast_factor)
                
        # Convert to tensors
        img_tensor = TF.to_tensor(img_pil)  # Float in [0, 1], shape (3, H, W)
        img_tensor = (img_tensor - self.mean) / self.std  # Normalized
        
        mask_np = np.array(mask_pil, dtype=np.int64)
        mask_tensor = torch.from_numpy(mask_np)  # (H, W), dtype int64
        
        return img_tensor, mask_tensor


def load_ai4mars_splits(data_dir="data", val_ratio=0.1, seed=42):
    """
    Loads train, validation holdout (10% of train), test_min3, and test_min1 splits.
    """
    train_path = os.path.join(data_dir, "train_shard0.parquet")
    test_min3_path = os.path.join(data_dir, "test_min3.parquet")
    test_min1_path = os.path.join(data_dir, "test_min1.parquet")
    
    files = {
        "train": train_path,
        "test_min3": test_min3_path,
        "test_min1": test_min1_path
    }
    
    for name, path in files.items():
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing {name} file at {path}. Run scripts/download_data.py first.")
            
    raw_ds = load_dataset("parquet", data_files=files)
    
    # 90% train / 10% validation holdout split from training images
    full_train = raw_ds["train"]
    split_ds = full_train.train_test_split(test_size=val_ratio, seed=seed, shuffle=True)
    
    return {
        "train": split_ds["train"],
        "val": split_ds["test"],
        "test_min3": raw_ds["test_min3"],
        "test_min1": raw_ds["test_min1"]
    }

