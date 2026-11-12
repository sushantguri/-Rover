"""
Temperature-Scaled Probability Calibration & Conformal Sand Safety Margin Engine.
Mitigates overconfident argmax predictions and catches ambiguous thin-sand veneers
over bedrock (reducing false-negative Spirit-trap wheel sinkage risk).
"""
from typing import Dict, Any, Tuple
import numpy as np
import torch
import torch.nn.functional as F

from src.visualization import create_overlay


class ProbabilityCalibrator:
    """
    Applies temperature scaling, computes Expected Calibration Error (ECE),
    and evaluates conformal sand recall safety margins.
    """
    def __init__(self, temperature: float = 1.25, conformal_sand_threshold: float = 0.25):
        if temperature <= 0.0:
            raise ValueError("Temperature must be strictly positive.")
        self.temperature = float(temperature)
        self.conformal_sand_threshold = float(conformal_sand_threshold)

    def calibrate_probs(self, logits_tensor: torch.Tensor) -> np.ndarray:
        """
        Applies temperature scaling softmax(logits / T) -> (C, H, W) float32 numpy array.
        """
        if logits_tensor.ndim == 4:
            logits_tensor = logits_tensor.squeeze(0)
        scaled = logits_tensor.float() / self.temperature
        probs = F.softmax(scaled, dim=0).cpu().numpy().astype(np.float32)
        return probs

    def compute_ece(
        self,
        probs: np.ndarray,
        target_mask: np.ndarray,
        n_bins: int = 10,
        ignore_index: int = 255,
    ) -> float:
        """
        Computes Expected Calibration Error (ECE) over valid pixels (!= ignore_index).
        probs shape: (C, H, W), target_mask shape: (H, W).
        """
        valid = target_mask != ignore_index
        if not np.any(valid):
            return 0.0

        confidences = np.max(probs, axis=0)[valid]
        predictions = np.argmax(probs, axis=0)[valid]
        targets = target_mask[valid]
        accuracies = (predictions == targets).astype(np.float32)

        bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
        ece = 0.0
        n_valid = float(confidences.size)

        for i in range(n_bins):
            low, high = bin_edges[i], bin_edges[i + 1]
            if i == n_bins - 1:
                in_bin = (confidences >= low) & (confidences <= high)
            else:
                in_bin = (confidences >= low) & (confidences < high)

            count = int(np.sum(in_bin))
            if count > 0:
                bin_acc = float(np.mean(accuracies[in_bin]))
                bin_conf = float(np.mean(confidences[in_bin]))
                ece += (count / n_valid) * abs(bin_acc - bin_conf)

        return round(float(ece), 4)

    def evaluate_safety_margin(
        self,
        logits_tensor: torch.Tensor,
        pred_mask: np.ndarray,
        sand_threshold_pct: float = 25.0,
    ) -> Tuple[Dict[str, Any], np.ndarray]:
        """
        Evaluates conformal safety recall for sand hazards and returns:
          (telemetry_dict, borderline_sand_boolean_mask).
        """
        if logits_tensor.ndim == 4:
            logits_tensor = logits_tensor.squeeze(0)

        scaled = logits_tensor.float() / self.temperature
        probs_t = F.softmax(scaled, dim=0)

        h, w = pred_mask.shape[:2]
        if probs_t.shape[1:] != (h, w):
            probs_t = F.interpolate(
                probs_t.unsqueeze(0), size=(h, w), mode="bilinear", align_corners=False
            ).squeeze(0)

        sand_prob = probs_t[2].cpu().numpy().astype(np.float32)
        valid = pred_mask != 255
        total_valid = max(1, int(np.sum(valid)))

        raw_sand_mask = (pred_mask == 2) & valid
        borderline_mask = (pred_mask != 2) & valid & (sand_prob >= self.conformal_sand_threshold)
        conservative_sand_mask = raw_sand_mask | borderline_mask

        raw_sand_pct = float(np.sum(raw_sand_mask) / total_valid * 100.0)
        borderline_sand_pct = float(np.sum(borderline_mask) / total_valid * 100.0)
        conservative_sand_pct = float(np.sum(conservative_sand_mask) / total_valid * 100.0)

        mean_sand_conf = (
            float(np.mean(sand_prob[raw_sand_mask])) if np.any(raw_sand_mask) else 0.0
        )
        max_sand_prob = float(np.max(sand_prob[valid])) if np.any(valid) else 0.0

        if raw_sand_pct < sand_threshold_pct <= conservative_sand_pct:
            conformal_alert = "CONSERVATIVE_SAND_ESCALATION"
        elif borderline_sand_pct >= 3.0:
            conformal_alert = "BORDERLINE_SAND_DETECTED"
        else:
            conformal_alert = "CALIBRATED_NOMINAL"

        summary = {
            "temperature": self.temperature,
            "conformal_sand_threshold": self.conformal_sand_threshold,
            "raw_sand_pct": round(raw_sand_pct, 2),
            "borderline_sand_pct": round(borderline_sand_pct, 2),
            "conservative_sand_pct": round(conservative_sand_pct, 2),
            "mean_sand_confidence": round(mean_sand_conf, 3),
            "max_sand_probability": round(max_sand_prob, 3),
            "conformal_alert": conformal_alert,
        }
        return summary, borderline_mask

    def render_safety_margin_overlay(
        self,
        orig_img_np: np.ndarray,
        pred_mask: np.ndarray,
        borderline_mask: np.ndarray,
        alpha: float = 0.65,
    ) -> np.ndarray:
        """
        Renders confirmed sand in glowing amber/orange and borderline conformal
        sand-ambiguity zones (e.g., thin dust mantles over bedrock) in bright magenta.
        """
        base = create_overlay(orig_img_np, pred_mask, alpha=0.45, highlight_sand=True).astype(np.float32)
        if np.any(borderline_mask):
            magenta = np.array([217.0, 70.0, 239.0], dtype=np.float32)
            blend = min(0.90, alpha + 0.2)
            base[borderline_mask] = (1.0 - blend) * orig_img_np[borderline_mask] + blend * magenta
        return np.clip(base, 0, 255).astype(np.uint8)
