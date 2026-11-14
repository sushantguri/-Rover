"""
Multi-Layer AutoNav Costmap Fusion Engine.
Synthesizes semantic terrain traversability, near-field obstacle inflation,
slope/attitude resistance, epistemic uncertainty risk, and wheel-wear ventifact protection
into a unified navigation costmap compatible with A*, D* Lite, and ROS 2 Nav2.
"""
import numpy as np
from typing import Dict, Any, Tuple
from PIL import Image

from src.path_planner import TERRAIN_COST_WEIGHTS

# Default layer fusion weights
DEFAULT_WEIGHT_SEMANTIC = 1.0
DEFAULT_WEIGHT_SLOPE = 0.35
DEFAULT_WEIGHT_UNCERTAINTY = 12.0
DEFAULT_WEIGHT_VENTIFACT_WEAR = 8.0

# 5-stage costmap visualization color ramp
# 1.0 (Optimal Bedrock: Cyan) -> 2.0 (Soil: Green) -> 20.0 (Sand: Orange) -> 80.0 (High Risk: Amber/Magenta) -> 500+ (Obstacle: Crimson)
FUSED_COST_PALETTE = np.array([
    [33, 150, 243],    # 0: Low cost / Bedrock (Cyan)
    [76, 175, 80],     # 1: Safe Soil (Green)
    [255, 152, 0],     # 2: Sand Hazard (Orange)
    [217, 70, 239],    # 3: Uncertainty / Ventifact Buffer (Magenta)
    [239, 68, 68],     # 4: Lethal Obstacle (Crimson)
], dtype=np.float32)


class AutoNavCostmapFuser:
    """
    Fuses multiple perception and physical risk layers into a composite 2D navigation costmap.
    """
    def __init__(
        self,
        weight_semantic: float = DEFAULT_WEIGHT_SEMANTIC,
        weight_slope: float = DEFAULT_WEIGHT_SLOPE,
        weight_uncertainty: float = DEFAULT_WEIGHT_UNCERTAINTY,
        weight_ventifact_wear: float = DEFAULT_WEIGHT_VENTIFACT_WEAR,
        obstacle_inflation_radius: int = 2,
    ):
        self.w_sem = weight_semantic
        self.w_slope = weight_slope
        self.w_uncert = weight_uncertainty
        self.w_wear = weight_ventifact_wear
        self.inflation_radius = obstacle_inflation_radius

    def fuse_layers(
        self,
        pred_mask: np.ndarray,
        entropy_map: np.ndarray | None = None,
        pitch_deg: float = 0.0,
        roll_deg: float = 0.0,
        ventifact_roughness_index: float = 0.0,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Computes composite costmap C(r, c) over the image grid.
        Returns:
            composite_costmap: (H, W) float32 array
            telemetry: summary metrics dictionary
        """
        h, w = pred_mask.shape[:2]
        costmap = np.zeros((h, w), dtype=np.float32)

        # 1. Base Semantic Cost Layer
        for cls_id, base_cost in TERRAIN_COST_WEIGHTS.items():
            costmap[pred_mask == cls_id] = base_cost * self.w_sem

        # Ignored boundary (255) treated as lethal
        costmap[pred_mask == 255] = 1000.0

        # 2. Slope & Attitude Layer
        # Positive pitch (uphill) increases forward driving cost; roll tilt increases cross-slope drift cost
        tilt_magnitude = float(np.hypot(pitch_deg, roll_deg))
        if tilt_magnitude > 2.0:
            # Upper image rows (forward horizon) penalized more when climbing uphill
            row_vals = np.linspace(1.2, 0.4, h, dtype=np.float32)
            row_ramp = np.repeat(row_vals[:, np.newaxis], w, axis=1)
            slope_cost = (tilt_magnitude * self.w_slope) * row_ramp
            # Do not penalize absolute obstacles
            navigable = costmap < 500.0
            costmap[navigable] += slope_cost[navigable]

        # 3. Epistemic Uncertainty Buffer Layer
        if entropy_map is not None:
            if entropy_map.shape != (h, w):
                pil_ent = Image.fromarray((entropy_map * 255).astype(np.uint8)).resize(
                    (w, h), Image.BILINEAR
                )
                ent_resized = np.array(pil_ent, dtype=np.float32) / 255.0
            else:
                ent_resized = entropy_map

            navigable = costmap < 500.0
            costmap[navigable] += ent_resized[navigable] * self.w_uncert

        # 4. Wheel-Wear Sharp Ventifact Protection Layer
        # If bedrock has sharp micro-relief, add modest penalty so rover prefers smooth soil when available
        if ventifact_roughness_index > 25.0:
            bedrock_cells = (pred_mask == 1) & (costmap < 500.0)
            wear_penalty = (ventifact_roughness_index / 100.0) * self.w_wear
            costmap[bedrock_cells] += wear_penalty

        # 5. Obstacle Inflation (Big Rocks, class 3)
        if self.inflation_radius > 0:
            rock_coords = np.argwhere(pred_mask == 3)
            for r, c in rock_coords:
                r_min = max(0, r - self.inflation_radius)
                r_max = min(h, r + self.inflation_radius + 1)
                c_min = max(0, c - self.inflation_radius)
                c_max = min(w, c + self.inflation_radius + 1)
                costmap[r_min:r_max, c_min:c_max] = np.maximum(
                    costmap[r_min:r_max, c_min:c_max], 500.0
                )

        # Retain strict lethal cap
        costmap[pred_mask == 3] = 1000.0
        costmap[pred_mask == 255] = 1000.0

        navigable_mask = costmap < 500.0
        nav_cells = int(np.sum(navigable_mask))
        mean_nav_cost = float(np.mean(costmap[navigable_mask])) if nav_cells > 0 else 1000.0
        lethal_pct = float(np.mean(costmap >= 500.0) * 100.0)

        telemetry = {
            "mean_navigable_cost": round(mean_nav_cost, 2),
            "lethal_obstacle_pct": round(lethal_pct, 1),
            "navigable_cell_count": nav_cells,
            "slope_cost_active": bool(tilt_magnitude > 2.0),
            "uncertainty_buffer_active": bool(entropy_map is not None),
            "ventifact_protection_active": bool(ventifact_roughness_index > 25.0),
        }
        return costmap, telemetry

    def render_fused_costmap_overlay(
        self,
        orig_img_np: np.ndarray,
        fused_costmap: np.ndarray,
        alpha: float = 0.65,
    ) -> np.ndarray:
        """
        Renders a continuous colorized heatmap of the fused costmap onto the rover camera image.
        """
        h, w = fused_costmap.shape[:2]
        # Logarithmic normalization: cost values range 1.0 to 1000.0
        log_cost = np.log1p(np.clip(fused_costmap, 0.0, 1000.0))
        # log(2) ~ 0.69, log(1001) ~ 6.91
        norm_cost = np.clip((log_cost - 0.69) / (6.91 - 0.69), 0.0, 1.0)

        scaled = norm_cost * 4.0
        idx_low = np.clip(np.floor(scaled).astype(int), 0, 3)
        idx_high = idx_low + 1
        t = (scaled - idx_low)[:, :, np.newaxis]

        color_low = FUSED_COST_PALETTE[idx_low]
        color_high = FUSED_COST_PALETTE[idx_high]
        colored_cost = (color_low * (1.0 - t) + color_high * t).astype(np.uint8)

        blended = (orig_img_np * (1.0 - alpha) + colored_cost * alpha).astype(np.uint8)
        return blended
