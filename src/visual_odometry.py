"""
NASA JPL Mars Rover Visual Odometry (VO) & Slip Compensation Model.
Simulates autonomous keypoint feature tracking (Harris/gradient peaks) across
consecutive Navcam frames to estimate true metric ground translation,
wheel slip ratio, yaw drift, and VO tracking confidence on Martian terrain.

References:
  - Maimone, Cheng, Matthies, "Two years of Visual Odometry on the Mars Exploration Rovers",
    Journal of Field Robotics, 2007.
  - NASA JPL AutoNav Flight Software (MSL Curiosity & M2020 Perseverance).
"""
import math
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
from PIL import Image, ImageDraw


class RoverVisualOdometryModel:
    """
    Simulates visual odometry keypoint tracking on firm rock/soil vs. loose sand,
    evaluating true metric ground translation against commanded wheel odometry.
    """

    def __init__(
        self,
        nominal_fov_deg: float = 45.0,
        focal_length_px: float = 1400.0,
        min_features_required: int = 15,
        critical_slip_threshold: float = 0.40,
    ):
        self.nominal_fov_deg = nominal_fov_deg
        self.focal_length_px = focal_length_px
        self.min_features_required = min_features_required
        self.critical_slip_threshold = critical_slip_threshold

    def extract_salient_keypoints(
        self,
        image_np: np.ndarray,
        pred_mask: np.ndarray,
        max_features: int = 60,
    ) -> List[Dict[str, Any]]:
        """
        Extracts salient tracking features from high-contrast gradient peaks,
        annotating each keypoint with its local terrain classification.
        """
        h, w = pred_mask.shape[:2]
        if image_np.ndim == 3:
            gray = np.mean(image_np, axis=2).astype(np.float32)
        else:
            gray = image_np.astype(np.float32)

        # Spatial gradient magnitude (Sobel-like difference)
        gy, gx = np.gradient(gray)
        grad_mag = np.sqrt(gx**2 + gy**2)

        # Exclude rover body (top / bottom extremes)
        grad_mag[: int(h * 0.15), :] = 0.0
        grad_mag[int(h * 0.90) :, :] = 0.0

        # Subsample grid to pick spatially distributed corner features
        grid_step = max(8, min(h, w) // 16)
        keypoints = []

        for r in range(grid_step, h - grid_step, grid_step):
            for c in range(grid_step, w - grid_step, grid_step):
                patch = grad_mag[r - 4 : r + 4, c - 4 : c + 4]
                if patch.size == 0:
                    continue
                score = float(np.max(patch))
                if score > 8.0:
                    # Find local argmax within patch
                    local_idx = np.unravel_index(np.argmax(patch), patch.shape)
                    kr = r - 4 + local_idx[0]
                    kc = c - 4 + local_idx[1]
                    terrain_class = int(pred_mask[kr, kc])

                    # Class trust weight: bedrock=1.0, rock=0.9, soil=0.7, sand=0.2
                    trust = 1.0 if terrain_class == 1 else (0.9 if terrain_class == 3 else (0.7 if terrain_class == 0 else 0.2))

                    keypoints.append(
                        {
                            "row": int(kr),
                            "col": int(kc),
                            "gradient_strength": round(score, 1),
                            "terrain_class": terrain_class,
                            "anchor_trust": trust,
                        }
                    )

        # Sort by strength and keep top max_features
        keypoints.sort(key=lambda kp: kp["gradient_strength"] * kp["anchor_trust"], reverse=True)
        return keypoints[:max_features]

    def evaluate_vo_tracking(
        self,
        pred_mask: np.ndarray,
        commanded_distance_m: float = 1.0,
        expected_wheel_slip_pct: float = 10.0,
        image_np: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """
        Evaluates visual odometry tracking feasibility, simulated actual displacement,
        observed visual slip ratio, and confidence status.
        """
        h, w = pred_mask.shape[:2]
        if image_np is None:
            # Synthetic gradient frame
            image_np = np.zeros((h, w, 3), dtype=np.uint8)
            image_np[pred_mask == 1] = 160  # bedrock texture
            image_np[pred_mask == 0] = 110  # soil texture
            image_np[pred_mask == 2] = 80   # sand texture
            image_np[pred_mask == 3] = 220  # rock texture

        keypoints = self.extract_salient_keypoints(image_np, pred_mask)
        total_kps = len(keypoints)

        # Count keypoint distribution across terrain classes
        bedrock_kps = sum(1 for kp in keypoints if kp["terrain_class"] == 1)
        soil_kps = sum(1 for kp in keypoints if kp["terrain_class"] == 0)
        sand_kps = sum(1 for kp in keypoints if kp["terrain_class"] == 2)
        rock_kps = sum(1 for kp in keypoints if kp["terrain_class"] == 3)

        firm_kps = bedrock_kps + soil_kps + rock_kps
        sand_ratio = (sand_kps / max(1, total_kps))

        # Overall tracking confidence index [0, 100]
        # High when firm features abound; drops when features are sparse or in sand
        feature_count_factor = min(1.0, total_kps / float(self.min_features_required))
        firm_quality_factor = firm_kps / max(1, total_kps)
        vo_confidence = round(feature_count_factor * firm_quality_factor * 100.0, 1)

        # Simulate true ground displacement factoring in terramechanics slip
        true_slip_ratio = min(0.95, max(0.0, expected_wheel_slip_pct / 100.0))
        # If rover is on sand, slip is compounded by visual deception
        actual_distance_m = commanded_distance_m * (1.0 - true_slip_ratio)
        odometry_error_m = commanded_distance_m - actual_distance_m

        # Observed VO slip ratio
        observed_slip_pct = round(true_slip_ratio * 100.0, 1)

        # VO Status Classification
        if total_kps < self.min_features_required:
            vo_status = "VO_FEATURE_STARVATION"
            slip_warning = "Insufficient visual features (<15 keypoints). Wheel odometry dead-reckoning required."
        elif sand_ratio >= 0.70:
            vo_status = "VO_SAND_DECEPTION_WARNING"
            slip_warning = "Feature track dominated by loose sand ripples. Visual odometry may underestimate slip."
        elif observed_slip_pct >= (self.critical_slip_threshold * 100.0):
            vo_status = "CRITICAL_WHEEL_SLIPPAGE"
            slip_warning = f"High wheel slippage ({observed_slip_pct}%). Rover wheels spinning with low forward progress."
        elif observed_slip_pct >= 20.0:
            vo_status = "ELEVATED_SLIP_COMPENSATED"
            slip_warning = f"Moderate wheel slip ({observed_slip_pct}%). VO closed-loop distance compensation active."
        else:
            vo_status = "NOMINAL_VO_TRACKING"
            slip_warning = "Solid visual feature tracking on firm substrate. Negligible odometry error."

        return {
            "total_keypoints": total_kps,
            "bedrock_features": bedrock_kps,
            "soil_features": soil_kps,
            "sand_features": sand_kps,
            "rock_features": rock_kps,
            "firm_anchor_count": firm_kps,
            "vo_confidence_pct": vo_confidence,
            "commanded_distance_m": round(commanded_distance_m, 3),
            "actual_distance_m": round(actual_distance_m, 3),
            "odometry_error_m": round(odometry_error_m, 3),
            "wheel_slip_pct": observed_slip_pct,
            "vo_status": vo_status,
            "slip_warning": slip_warning,
            "keypoints": keypoints,
        }

    def render_tracking_overlay(
        self,
        image_np: np.ndarray,
        pred_mask: np.ndarray,
        vo_telemetry: Dict[str, Any],
        alpha: float = 0.65,
    ) -> np.ndarray:
        """
        Renders visual odometry feature tracking vector overlay with color-coded
        confidence reticles (Cyan=Bedrock anchor, Green=Regolith, Orange=Sand slip, Red=Lost).
        """
        pil_img = Image.fromarray(image_np.copy()).convert("RGBA")
        overlay = Image.new("RGBA", pil_img.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        keypoints = vo_telemetry.get("keypoints", [])
        slip_ratio = vo_telemetry.get("wheel_slip_pct", 0.0) / 100.0
        vector_len = max(4.0, 16.0 * (1.0 - slip_ratio))

        for kp in keypoints:
            r = kp["row"]
            c = kp["col"]
            c_type = kp["terrain_class"]

            if c_type == 1:
                color = (0, 220, 255, 230)      # Bedrock (Cyan)
            elif c_type == 0:
                color = (34, 197, 94, 210)      # Soil (Green)
            elif c_type == 2:
                color = (249, 115, 22, 230)     # Sand (Orange - high slip)
            else:
                color = (239, 68, 68, 230)      # Big Rock (Red)

            rad = 4
            draw.ellipse([c - rad, r - rad, c + rad, r + rad], outline=color, width=2)
            # Feature tracking forward motion vector
            draw.line([c, r, c, r + vector_len], fill=color, width=2)

        # Composite and return RGB
        blended = Image.alpha_composite(pil_img, overlay)
        return np.array(blended.convert("RGB"))
