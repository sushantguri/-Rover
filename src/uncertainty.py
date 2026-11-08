"""
Perception Uncertainty & Prediction Entropy Estimation Module.
Computes per-pixel Shannon entropy and margin uncertainty from DeepLabV3+ softmax probabilities.
Identifies ambiguous terrain boundaries, shadowing artifacts, and flags uncertain regions for human inspection.
"""
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from typing import Dict, Any, Tuple

# Custom uncertainty color ramp: 0.0 (Cool Blue/Transparent) -> 0.5 (Amber) -> 1.0 (Bright Crimson/Magenta)
UNCERTAINTY_COLOR_MAP = np.array([
    [30, 41, 59],      # Low uncertainty (Dark Slate)
    [59, 130, 246],    # 0.25 (Blue)
    [245, 158, 11],    # 0.50 (Amber)
    [239, 68, 68],     # 0.75 (Red)
    [217, 70, 239]     # 1.00 (Bright Magenta / Alert)
], dtype=np.float32)


class UncertaintyEstimator:
    """
    Computes information-theoretic prediction uncertainty metrics on segmentation logits.
    """
    def __init__(self, high_uncertainty_threshold: float = 0.65):
        self.high_threshold = high_uncertainty_threshold
        # Maximum possible entropy for 4 classes: log(4) ≈ 1.38629
        self.max_entropy = np.log(4.0)

    def compute_entropy_map(self, logits_tensor: torch.Tensor) -> np.ndarray:
        """
        Calculates normalized Shannon Entropy H(x) in [0.0, 1.0] from raw logits.
        logits_tensor shape: (1, 4, H, W) or (4, H, W)
        """
        if logits_tensor.ndim == 4:
            logits_tensor = logits_tensor.squeeze(0)

        # Softmax probabilities P(c|x)
        probs = F.softmax(logits_tensor, dim=0).cpu().numpy()  # (4, H, W)
        
        # Clamp to avoid log(0)
        eps = 1e-12
        probs_clamped = np.clip(probs, eps, 1.0)
        
        # Shannon Entropy: -sum(p * log(p))
        entropy = -np.sum(probs_clamped * np.log(probs_clamped), axis=0)
        
        # Normalize to [0.0, 1.0]
        norm_entropy = entropy / self.max_entropy
        return np.clip(norm_entropy, 0.0, 1.0).astype(np.float32)

    def compute_margin_map(self, logits_tensor: torch.Tensor) -> np.ndarray:
        """
        Computes margin uncertainty: 1.0 - (P_top1 - P_top2).
        High values indicate boundary ambiguity between two competing classes.
        """
        if logits_tensor.ndim == 4:
            logits_tensor = logits_tensor.squeeze(0)

        probs = F.softmax(logits_tensor, dim=0).cpu().numpy()
        sorted_probs = np.sort(probs, axis=0)  # Ascending
        top1 = sorted_probs[-1]
        top2 = sorted_probs[-2]

        margin = 1.0 - (top1 - top2)
        return np.clip(margin, 0.0, 1.0).astype(np.float32)

    def generate_uncertainty_heatmap(
        self, 
        orig_img_np: np.ndarray, 
        entropy_map: np.ndarray, 
        alpha: float = 0.65
    ) -> np.ndarray:
        """
        Creates a colorized heatmap overlay where high entropy regions glow in vibrant amber/magenta.
        """
        orig_h, orig_w, _ = orig_img_np.shape
        ent_h, ent_w = entropy_map.shape

        if (orig_h, orig_w) != (ent_h, ent_w):
            pil_ent = Image.fromarray((entropy_map * 255).astype(np.uint8)).resize(
                (orig_w, orig_h), Image.BILINEAR
            )
            entropy_full = np.array(pil_ent, dtype=np.float32) / 255.0
        else:
            entropy_full = entropy_map

        # Map entropy [0, 1] to 5-color ramp
        scaled = entropy_full * 4.0
        idx_low = np.clip(np.floor(scaled).astype(int), 0, 3)
        idx_high = idx_low + 1
        t = (scaled - idx_low)[:, :, np.newaxis]

        color_low = UNCERTAINTY_COLOR_MAP[idx_low]
        color_high = UNCERTAINTY_COLOR_MAP[idx_high]
        colored_heat = (color_low * (1.0 - t) + color_high * t).astype(np.uint8)

        # Dynamic alpha mask: Low uncertainty pixels are almost transparent
        blend_mask = np.clip((entropy_full - 0.15) / 0.70, 0.0, 1.0)[:, :, np.newaxis] * alpha
        blended = (orig_img_np * (1.0 - blend_mask) + colored_heat * blend_mask).astype(np.uint8)
        return blended

    def evaluate_uncertainty_telemetry(self, entropy_map: np.ndarray) -> Dict[str, Any]:
        """
        Computes summary statistics on prediction confidence for rover decision making.
        """
        mean_entropy = float(np.mean(entropy_map))
        median_entropy = float(np.median(entropy_map))
        p90_entropy = float(np.percentile(entropy_map, 90))
        
        high_uncertainty_ratio = float(np.mean(entropy_map >= self.high_threshold)) * 100.0

        if high_uncertainty_ratio > 20.0 or mean_entropy > 0.50:
            status = "HIGH_PERCEPTION_AMBIGUITY"
            recommendation = "Low optical contrast or severe shadowing. Slow drive speed and capture multi-angle Mastcam confirmation."
        elif high_uncertainty_ratio > 8.0:
            status = "MODERATE_BOUNDARY_UNCERTAINTY"
            recommendation = "Minor class boundary uncertainty detected (e.g. thin sand veneers on bedrock). Nominal drive permitted."
        else:
            status = "HIGH_CONFIDENCE_CLEAR"
            recommendation = "Clear terrain definition with crisp semantic boundaries. Full autonomous trust level."

        return {
            "mean_prediction_entropy": round(mean_entropy, 3),
            "median_prediction_entropy": round(median_entropy, 3),
            "p90_prediction_entropy": round(p90_entropy, 3),
            "high_uncertainty_pixel_pct": round(high_uncertainty_ratio, 1),
            "confidence_status": status,
            "perception_recommendation": recommendation
        }
