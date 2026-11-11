"""
Optical Sensor Health & Atmospheric Fault Diagnostics (Watchdog).
Evaluates Mars Rover Navcam/Mastcam RGB imagery for solar glare saturation, dust storm
contrast brownout, nighttime/deep-shadow underexposure, and lens blur prior to trusting
neural segmentation predictions.
"""
import numpy as np
from typing import Dict, Any


class OpticalSensorWatchdog:
    """
    Analyzes raw camera luminance, contrast, clipping, and spatial sharpness metrics.
    """
    def __init__(
        self,
        glare_clip_threshold_pct: float = 12.0,
        underexposed_clip_threshold_pct: float = 35.0,
        low_contrast_std_threshold: float = 18.0,
        blur_gradient_threshold: float = 3.0
    ):
        self.glare_clip_threshold_pct = glare_clip_threshold_pct
        self.underexposed_clip_threshold_pct = underexposed_clip_threshold_pct
        self.low_contrast_std_threshold = low_contrast_std_threshold
        self.blur_gradient_threshold = blur_gradient_threshold

    def inspect_frame(self, img_np: np.ndarray) -> Dict[str, Any]:
        """
        Inspects an RGB numpy array (H, W, 3) and classifies sensor health state.
        """
        if img_np.ndim == 2:
            gray = img_np.astype(np.float32)
        else:
            # Standard ITU-R BT.601 luminance conversion
            gray = (
                0.299 * img_np[:, :, 0].astype(np.float32)
                + 0.587 * img_np[:, :, 1].astype(np.float32)
                + 0.114 * img_np[:, :, 2].astype(np.float32)
            )

        mean_lum = float(np.mean(gray))
        contrast_std = float(np.std(gray))
        overexposed_pct = float(np.mean(gray >= 248.0) * 100.0)
        underexposed_pct = float(np.mean(gray <= 12.0) * 100.0)

        # Edge sharpness proxy via mean L1 spatial gradient magnitude
        if gray.shape[0] > 1 and gray.shape[1] > 1:
            dx = np.abs(gray[:, 1:] - gray[:, :-1])
            dy = np.abs(gray[1:, :] - gray[:-1, :])
            sharpness_score = float(0.5 * (np.mean(dx) + np.mean(dy)))
        else:
            sharpness_score = 0.0

        flags = []
        if overexposed_pct >= self.glare_clip_threshold_pct:
            flags.append("SOLAR_GLARE_CLIPPING")
        if underexposed_pct >= self.underexposed_clip_threshold_pct or mean_lum < 25.0:
            flags.append("DEEP_SHADOW_UNDEREXPOSURE")
        if contrast_std < self.low_contrast_std_threshold:
            flags.append("LOW_CONTRAST_DUST_HAZE")
        if sharpness_score < self.blur_gradient_threshold:
            flags.append("LENS_BLUR_OR_OCCULTATION")

        if "DEEP_SHADOW_UNDEREXPOSURE" in flags and "LOW_CONTRAST_DUST_HAZE" in flags:
            status = "SENSOR_FAULT_DARK_FRAME"
            recommend_tta = True
            trust_weight = 0.25
        elif "SOLAR_GLARE_CLIPPING" in flags:
            status = "GLARE_SATURATION_WARNING"
            recommend_tta = True
            trust_weight = 0.65
        elif "LOW_CONTRAST_DUST_HAZE" in flags or "LENS_BLUR_OR_OCCULTATION" in flags:
            status = "ATMOSPHERIC_HAZE_ADVISORY"
            recommend_tta = True
            trust_weight = 0.70
        elif "DEEP_SHADOW_UNDEREXPOSURE" in flags:
            status = "SHADOW_CONTRAST_WARNING"
            recommend_tta = True
            trust_weight = 0.75
        else:
            status = "NOMINAL_OPTICAL"
            recommend_tta = False
            trust_weight = 1.0

        return {
            "sensor_health_status": status,
            "optical_trust_score": round(trust_weight * 100.0, 1),
            "mean_luminance": round(mean_lum, 1),
            "contrast_std": round(contrast_std, 1),
            "overexposed_pct": round(overexposed_pct, 2),
            "underexposed_pct": round(underexposed_pct, 2),
            "sharpness_gradient": round(sharpness_score, 2),
            "active_fault_flags": flags,
            "recommend_tta_boost": recommend_tta,
        }
