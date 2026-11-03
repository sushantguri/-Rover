"""
Visualization utilities for Mars Rover Terrain Segmentation.
Provides color maps, legend generation, and overlay functions.
"""
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Class metadata
CLASSES = ["soil", "bedrock", "sand", "big rock"]
CLASS_LABELS = {
    0: "soil",
    1: "bedrock",
    2: "sand",
    3: "big rock",
    255: "unlabeled/ignored"
}

# Distinct RGB colors for each class
# 0: Soil (warm Martian tan)
# 1: Bedrock (slate blue-grey)
# 2: Sand (hazard amber/orange)
# 3: Big rock (hazard crimson red)
# 255: Ignored (dark neutral)
COLOR_MAP = {
    0: (218, 165, 32),    # Goldenrod / Soil
    1: (100, 149, 237),   # Cornflower Blue / Bedrock
    2: (255, 120, 0),     # Bright Orange / Sand
    3: (220, 20, 60),     # Crimson Red / Big Rock
    255: (40, 40, 40)     # Dark Charcoal / Ignored
}

HEX_COLORS = {
    0: "#DAA520",
    1: "#6495ED",
    2: "#FF7800",
    3: "#DC143C",
    255: "#282828"
}


def mask_to_rgb(mask: np.ndarray, color_map=COLOR_MAP) -> np.ndarray:
    """
    Converts a 2D integer segmentation mask to an RGB image.
    Mask shape: (H, W). Returns: (H, W, 3) uint8.
    """
    h, w = mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for cls_val, color in color_map.items():
        idx = (mask == cls_val)
        if np.any(idx):
            rgb[idx] = color
    return rgb


def create_overlay(image: np.ndarray, mask: np.ndarray, alpha: float = 0.5,
                   highlight_sand: bool = False, color_map=COLOR_MAP) -> np.ndarray:
    """
    Blends the segmentation mask onto the original RGB image.
    Unlabeled pixels (255) remain 100% original image (no tint).
    If highlight_sand=True, sand gets full opacity/vibrancy while other classes are subdued.
    """
    if image.dtype != np.uint8:
        image = (np.clip(image, 0, 1) * 255).astype(np.uint8)
    
    rgb_mask = mask_to_rgb(mask, color_map=color_map)
    labeled_pixels = (mask != 255) & (mask >= 0) & (mask < len(CLASSES))
    
    if highlight_sand:
        sand_pixels = (mask == 2)
        other_labeled = labeled_pixels & (~sand_pixels)
        
        blended = image.copy().astype(np.float32)
        # Subdued other classes
        blended[other_labeled] = (
            (1.0 - alpha * 0.4) * image[other_labeled] +
            (alpha * 0.4) * rgb_mask[other_labeled]
        )
        # High-visibility sand overlay
        blended[sand_pixels] = (
            (1.0 - min(0.85, alpha + 0.3)) * image[sand_pixels] +
            min(0.85, alpha + 0.3) * rgb_mask[sand_pixels]
        )
        return np.clip(blended, 0, 255).astype(np.uint8)
    
    blended = image.copy().astype(np.float32)
    blended[labeled_pixels] = (
        (1.0 - alpha) * image[labeled_pixels] +
        alpha * rgb_mask[labeled_pixels]
    )
    return np.clip(blended, 0, 255).astype(np.uint8)


def plot_data_exploration(image: np.ndarray, raw_label: np.ndarray,
                          rover_mask: np.ndarray, range_mask: np.ndarray,
                          final_mask: np.ndarray, sample_idx: int,
                          save_path: str = None):
    """
    Creates a 5-panel figure displaying:
    1. Raw Rover Image
    2. Crowdsourced / Expert Label
    3. Rover & Range Exclusion Masks
    4. Clean Filtered Mask (Rover + Range -> 255)
    5. Terrain Overlay on Mars Rover Image
    """
    fig, axes = plt.subplots(1, 5, figsize=(22, 5))
    
    # 1. Image
    axes[0].imshow(image)
    axes[0].set_title("1. Rover Camera (RGB)", fontsize=12, fontweight="bold")
    axes[0].axis("off")
    
    # 2. Raw label
    raw_rgb = mask_to_rgb(raw_label)
    axes[1].imshow(raw_rgb)
    axes[1].set_title("2. Raw Annotations", fontsize=12, fontweight="bold")
    axes[1].axis("off")
    
    # 3. Rover & Range Masks
    exclusion = np.zeros_like(raw_label, dtype=np.uint8)
    if rover_mask is not None:
        exclusion[rover_mask > 0] = 1
    if range_mask is not None:
        exclusion[range_mask > 0] = 2
    excl_rgb = np.zeros((*exclusion.shape, 3), dtype=np.uint8)
    excl_rgb[exclusion == 1] = [0, 255, 255]   # Cyan: Rover body
    excl_rgb[exclusion == 2] = [255, 0, 255]   # Magenta: > 30m horizon
    axes[2].imshow(excl_rgb)
    axes[2].set_title("3. Rover & Range Masks", fontsize=12, fontweight="bold")
    axes[2].axis("off")
    
    # 4. Filtered Mask
    final_rgb = mask_to_rgb(final_mask)
    unlabeled_pct = (final_mask == 255).sum() / final_mask.size * 100
    axes[3].imshow(final_rgb)
    axes[3].set_title(f"4. Clean Mask\n({unlabeled_pct:.1f}% Unlabeled)", fontsize=12, fontweight="bold")
    axes[3].axis("off")
    
    # 5. Overlay
    overlay = create_overlay(image, final_mask, alpha=0.55)
    axes[4].imshow(overlay)
    axes[4].set_title("5. Clean Terrain Overlay", fontsize=12, fontweight="bold")
    axes[4].axis("off")
    
    # Add legend
    legend_patches = [
        mpatches.Patch(color=np.array(COLOR_MAP[0])/255., label="0: Soil"),
        mpatches.Patch(color=np.array(COLOR_MAP[1])/255., label="1: Bedrock"),
        mpatches.Patch(color=np.array(COLOR_MAP[2])/255., label="2: Sand (Hazard)"),
        mpatches.Patch(color=np.array(COLOR_MAP[3])/255., label="3: Big Rock (Hazard)"),
        mpatches.Patch(color=np.array(COLOR_MAP[255])/255., label="255: Unlabeled/Ignored"),
    ]
    fig.legend(handles=legend_patches, loc="lower center", ncol=5, fontsize=11, frameon=True, bbox_to_anchor=(0.5, -0.05))
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"Saved figure to {save_path}")
    plt.close()

