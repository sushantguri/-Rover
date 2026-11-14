"""
Curiosity/Perseverance Aluminum Wheel Ventifact Wear & Puncture Fatigue Mechanics.
Simulates localized contact stress, puncture fatigue, and abrasive wear on 0.75 mm aluminum rover
wheels traversing sharp Martian ventifacts (wind-sculpted bedrock ridges and embedded basalt shards).
"""
import numpy as np
from typing import Dict, Any, Tuple
from PIL import Image

# Physical wheel parameters for MSL Curiosity / Mars 2020 Perseverance
ALUMINUM_SKIN_THICKNESS_MM = 0.75      # 750 um thin machined aerospace aluminum
ALUMINUM_YIELD_STRENGTH_MPA = 276.0    # 6061-T6 aluminum yield strength
MAX_GROUSER_FATIGUE_LIFETIME_KM = 35.0 # Projected baseline operational wheel lifetime


class RoverWheelWearModel:
    """
    Evaluates puncture fatigue and abrasive wear mechanics across segmented Martian terrain.
    """
    def __init__(
        self,
        skin_thickness_mm: float = ALUMINUM_SKIN_THICKNESS_MM,
        yield_strength_mpa: float = ALUMINUM_YIELD_STRENGTH_MPA,
    ):
        self.skin_thickness_mm = skin_thickness_mm
        self.yield_strength_mpa = yield_strength_mpa

    def compute_ventifact_roughness(
        self,
        gray_image: np.ndarray,
        pred_mask: np.ndarray,
    ) -> float:
        """
        Measures high-frequency surface gradient micro-relief specifically over exposed
        bedrock (class 1) and big rock (class 3) regions where pointed ventifacts protrude.
        Returns a normalized roughness index in [0.0, 100.0].
        """
        h, w = gray_image.shape[:2]
        if h < 3 or w < 3:
            return 0.0

        # Mask of hard rock/bedrock surfaces where punctures occur
        hard_surface = (pred_mask == 1) | (pred_mask == 3)
        if not np.any(hard_surface):
            return 0.0

        # Discrete spatial gradient
        gray_f = gray_image.astype(np.float32)
        dx = np.abs(gray_f[:, 1:] - gray_f[:, :-1])
        dy = np.abs(gray_f[1:, :] - gray_f[:-1, :])

        # Pad to match original dimensions
        dx_padded = np.pad(dx, ((0, 0), (0, 1)), mode="edge")
        dy_padded = np.pad(dy, ((0, 1), (0, 0)), mode="edge")
        gradient_mag = 0.5 * (dx_padded + dy_padded)

        # Average gradient specifically on hard rock pixels
        ventifact_gradient = float(np.mean(gradient_mag[hard_surface]))
        # Normalize: typical Mars Navcam gradients range 2.0 (smooth) to 25.0+ (razor-sharp caprock)
        roughness_index = float(np.clip((ventifact_gradient - 2.0) / 18.0 * 100.0, 0.0, 100.0))
        return round(roughness_index, 1)

    def evaluate_wheel_wear(
        self,
        terrain_distribution_pct: Dict[str, float],
        terramechanics_data: Dict[str, Any] | None = None,
        ventifact_roughness_index: float = 20.0,
        drive_distance_m: float = 20.0,
    ) -> Dict[str, Any]:
        """
        Estimates contact point stress, skin puncture probability, and gouging rate.
        """
        mech = terramechanics_data or {}
        bedrock_pct = float(terrain_distribution_pct.get("bedrock", 0.0))
        rock_pct = float(terrain_distribution_pct.get("big rock", 0.0))
        sand_pct = float(terrain_distribution_pct.get("sand", 0.0))
        soil_pct = float(terrain_distribution_pct.get("soil", 0.0))

        slip_pct = float(mech.get("expected_wheel_slip_pct", 5.0))
        hard_terrain_ratio = (bedrock_pct + rock_pct * 1.5) / 100.0

        # Point contact stress model:
        # On sand/soil, wheel distributes load over 40x25cm contact patch (~15-35 kPa).
        # On pointed ventifact pyramids, load concentrates on 1-3 mm asperities (up to 200+ MPa).
        base_contact_stress = 25.0  # MPa on flat regolith
        roughness_factor = ventifact_roughness_index / 100.0
        slip_abrasion_multiplier = 1.0 + (slip_pct / 40.0) * 1.5

        peak_contact_stress_mpa = base_contact_stress + (
            hard_terrain_ratio * (180.0 * roughness_factor + 45.0) * slip_abrasion_multiplier
        )
        peak_contact_stress_mpa = float(np.clip(peak_contact_stress_mpa, 15.0, 380.0))

        # Yield safety ratio: contact stress vs 6061-T6 aluminum yield strength
        yield_ratio = peak_contact_stress_mpa / self.yield_strength_mpa

        # Puncture risk per 100m of traverse: non-linear spike when stress approaches/exceeds yield
        if yield_ratio >= 1.0:
            puncture_risk_pct = float(np.clip(45.0 + (yield_ratio - 1.0) * 80.0, 45.0, 95.0))
        elif yield_ratio >= 0.70:
            puncture_risk_pct = float(np.clip((yield_ratio - 0.70) / 0.30 * 40.0 + 5.0, 5.0, 45.0))
        else:
            puncture_risk_pct = float(np.clip(yield_ratio * 7.0, 0.1, 5.0))

        # Abrasive thickness loss in microns per kilometer
        # Soft sand causes mild polishing (10-25 um/km); sharp ventifact scraping causes severe gouging (150-500 um/km)
        wear_rate_um_per_km = (
            (soil_pct / 100.0) * 18.0 +
            (sand_pct / 100.0) * 32.0 +
            (bedrock_pct / 100.0) * (65.0 + roughness_factor * 220.0 * slip_abrasion_multiplier) +
            (rock_pct / 100.0) * (120.0 + roughness_factor * 350.0)
        )
        wear_rate_um_per_km = float(np.clip(wear_rate_um_per_km, 10.0, 650.0))

        # Skin thickness loss for this specific traverse leg (microns)
        leg_skin_loss_um = wear_rate_um_per_km * (drive_distance_m / 1000.0)

        # Operational status and traction control advisory
        if peak_contact_stress_mpa >= self.yield_strength_mpa or puncture_risk_pct >= 50.0:
            wear_status = "CRITICAL_PUNCTURE_HAZARD"
            advisory = (
                f"Peak ventifact contact stress ({peak_contact_stress_mpa:.0f} MPa) exceeds aluminum yield threshold ({self.yield_strength_mpa:.0f} MPa). "
                "Engage JPL Traction Control (limit slip < 4%) and detour around sharp caprock."
            )
            traction_mode = "TRACTION_CONTROL_ACTIVE_STRICT"
        elif peak_contact_stress_mpa >= 180.0 or ventifact_roughness_index >= 60.0:
            wear_status = "ELEVATED_VENTIFACT_STRESS"
            advisory = (
                f"Sharp basaltic micro-relief detected (Roughness index {ventifact_roughness_index:.0f}/100). "
                "Wheel skin wear rate elevated. Avoid high-slip turning on bedrock."
            )
            traction_mode = "REDUCE_WHEEL_SLIP_MODE"
        elif sand_pct > 30.0:
            wear_status = "ABRASIVE_SAND_SCOUR"
            advisory = "Fine basaltic sand scour dominates. Low structural puncture risk; monitor wheel sinkage."
            traction_mode = "TORQUE_BALANCED_SAND"
        else:
            wear_status = "NOMINAL_WHEEL_WEAR"
            advisory = "Smooth regolith/pavement. Contact stress well below aluminum yield limit."
            traction_mode = "STANDARD_CRUISE_TORQUE"

        return {
            "ventifact_roughness_index": round(ventifact_roughness_index, 1),
            "peak_contact_stress_mpa": round(peak_contact_stress_mpa, 1),
            "aluminum_yield_ratio": round(yield_ratio, 2),
            "puncture_risk_pct": round(puncture_risk_pct, 1),
            "wear_rate_um_per_km": round(wear_rate_um_per_km, 1),
            "leg_skin_loss_um": round(leg_skin_loss_um, 3),
            "wear_status": wear_status,
            "wear_advisory": advisory,
            "traction_control_mode": traction_mode,
        }

    def render_wear_hazard_overlay(
        self,
        orig_img_np: np.ndarray,
        pred_mask: np.ndarray,
        ventifact_roughness_index: float,
        alpha: float = 0.60,
    ) -> np.ndarray:
        """
        Renders an overlay pinpointing wheel-puncture ventifact zones in high-contrast ruby-red
        over bedrock and rocks, while subduing safe driving regolith.
        """
        orig_h, orig_w = orig_img_np.shape[:2]
        pil_img = Image.fromarray(orig_img_np).convert("RGBA")
        overlay = Image.new("RGBA", (orig_w, orig_h), (0, 0, 0, 0))
        overlay_np = np.zeros((orig_h, orig_w, 4), dtype=np.uint8)

        # Highlight sharp bedrock and big rocks proportionally to roughness
        hard_mask = (pred_mask == 1) | (pred_mask == 3)
        intensity = int(np.clip(160 + ventifact_roughness_index * 0.9, 160, 255))
        alpha_val = int(np.clip(alpha * 255 * (0.6 + ventifact_roughness_index / 200.0), 80, 240))

        # Crimson / Ruby red color for wheel puncture hazard
        overlay_np[hard_mask] = [intensity, 20, 60, alpha_val]

        # Convert back
        tint_img = Image.fromarray(overlay_np, mode="RGBA")
        combined = Image.alpha_composite(pil_img, tint_img).convert("RGB")
        return np.array(combined)
