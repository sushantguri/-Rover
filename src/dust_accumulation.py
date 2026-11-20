"""
Martian Atmospheric Dust Deposition, Optical Transmission Degradation & Convective Cleaning Model.
Models sol-by-sol dust settling from suspension onto Mars rover camera windows and calibration targets,
accounting for atmospheric dust optical depth (tau), veiling glare contrast attenuation,
and convective vortex ("dust devil") cleaning events.

References:
  - Lorenz, "Martian dust devils: Review of observations, dynamics, and hazards", Icarus, 2020.
  - Landis et al., "Dust deposition on the Mars Exploration Rover solar arrays", Acta Astronautica, 2006.
  - Bell et al., "The Mars Science Laboratory Curiosity Mast Camera (Mastcam) Investigation", 2017.
"""
import math
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
from PIL import Image, ImageEnhance, ImageDraw


class MartianDustAccumulationModel:
    """
    Simulates optical window dust deposition, contrast degradation,
    and convective vortex cleaning events across Martian mission sols.
    """

    def __init__(
        self,
        base_settling_rate_per_sol: float = 0.0018,
        min_optical_transmittance: float = 0.35,
        critical_cleaning_threshold: float = 0.70,
    ):
        self.base_settling_rate = base_settling_rate_per_sol
        self.min_optical_transmittance = min_optical_transmittance
        self.critical_cleaning_threshold = critical_cleaning_threshold

    def evaluate_dust_deposition(
        self,
        sol_number: int = 4125,
        last_cleaning_sol: int = 4000,
        dust_tau: float = 0.5,
        wind_speed_m_s: float = 8.0,
    ) -> Dict[str, Any]:
        """
        Calculates optical transmittance, accumulated dust layer thickness (microns),
        contrast reduction, and convective vortex cleaning probability.
        """
        sols_since_clean = max(0, sol_number - last_cleaning_sol)

        # Higher tau accelerates atmospheric dust fallout
        tau_acceleration = 1.0 + 0.40 * (dust_tau - 0.5)
        # Settling loss accumulated
        cumulative_dust_loss = self.base_settling_rate * sols_since_clean * tau_acceleration

        transmittance = max(self.min_optical_transmittance, 1.0 - cumulative_dust_loss)
        transmittance_pct = round(transmittance * 100.0, 1)

        # Effective dust layer thickness (microns) ~ 0.025 um per 1% transmission loss
        loss_pct = 100.0 - transmittance_pct
        dust_thickness_um = round(loss_pct * 0.028, 2)

        # Veiling glare contrast loss factor
        contrast_loss_pct = round((1.0 - transmittance) * 75.0, 1)

        # Convective cleaning vortex ("dust devil") probability
        # Higher in warm afternoon winds; estimated daily probability ~ 1.5% to 8%
        wind_factor = min(2.5, max(0.5, wind_speed_m_s / 8.0))
        cleaning_prob_pct = round(min(12.0, 1.8 * wind_factor * (1.0 + 0.2 * math.sin(sol_number % 668))), 1)

        # Lens mitigation status
        if transmittance < self.critical_cleaning_threshold:
            mitigation_status = "CRITICAL_DUST_OBSCURATION"
            advisory = (
                f"Severe dust layer ({dust_thickness_um} um, {transmittance_pct}% trans). "
                f"Perform Mastcam photometric calibration target reference & engage software de-hazing."
            )
            software_contrast_boost = 1.35
        elif transmittance < 0.85:
            mitigation_status = "MODERATE_DUST_HAZE"
            advisory = f"Noticeable dust accumulation ({dust_thickness_um} um). Contrast attenuated by {contrast_loss_pct}%."
            software_contrast_boost = 1.15
        else:
            mitigation_status = "NOMINAL_OPTICAL_CLARITY"
            advisory = "Optical window clean. Nominal photometric transmission."
            software_contrast_boost = 1.0

        return {
            "sol_number": sol_number,
            "sols_since_clean": sols_since_clean,
            "dust_optical_depth_tau": dust_tau,
            "optical_transmittance_pct": transmittance_pct,
            "dust_thickness_um": dust_thickness_um,
            "contrast_loss_pct": contrast_loss_pct,
            "convective_vortex_prob_pct": cleaning_prob_pct,
            "software_contrast_boost": software_contrast_boost,
            "mitigation_status": mitigation_status,
            "lens_cleaning_advisory": advisory,
        }

    def render_dust_overlay(
        self,
        image_np: np.ndarray,
        dust_telemetry: Dict[str, Any],
        alpha: float = 0.65,
    ) -> np.ndarray:
        """
        Renders simulated optical window dust particulate haze and circular vignetting
        representing physical camera window dust obscuration.
        """
        h, w = image_np.shape[:2]
        pil_img = Image.fromarray(image_np.copy()).convert("RGBA")

        # Reduce contrast according to dust loss
        trans = dust_telemetry.get("optical_transmittance_pct", 100.0) / 100.0
        contrast_factor = max(0.4, trans)
        enhancer = ImageEnhance.Contrast(pil_img)
        hazed_img = enhancer.enhance(contrast_factor)

        # Create reddish-ochre dust vignette overlay
        overlay = Image.new("RGBA", pil_img.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        # Dust tint color: Martian ferric oxide ochre (198, 120, 60)
        dust_opacity = int(min(140, max(10, (1.0 - trans) * 180)))
        overlay_color = (198, 115, 60, dust_opacity)

        # Corner particulate vignetting
        center_x, center_y = w // 2, h // 2
        max_dist = math.sqrt(center_x**2 + center_y**2)

        # Subtle peripheral vignette gradient
        for step in range(4):
            inset = step * (min(h, w) // 16)
            oval_box = [inset, inset, w - inset, h - inset]
            draw.ellipse(oval_box, outline=(210, 130, 70, max(0, dust_opacity // 3)), width=3)

        blended = Image.alpha_composite(hazed_img, overlay)
        return np.array(blended.convert("RGB"))
