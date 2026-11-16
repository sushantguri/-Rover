"""
Martian Atmospheric Boundary-Layer Aeolian Saltation & Active Sand Drift Forecaster.
Simulates wind shear friction velocity (u*), Bagnold sand mass flux in 6 mbar CO2,
and evaluates unstable dune lee-side slipface avalanche entrapment hazards.
"""
import math
import numpy as np
from typing import Dict, Any, Tuple
from PIL import Image, ImageDraw

# Mars atmospheric boundary layer constants
MARS_ATMOSPHERIC_DENSITY_KG_M3 = 0.015   # ~6 mbar cold CO2 atmosphere
MARS_GRAVITY_M_S2 = 3.721                # Surface acceleration of gravity
THRESHOLD_FRICTION_VELOCITY_M_S = 1.45   # Saltation threshold (Greeley-Iversen / Kok et al.)
WIND_FRICTION_FACTOR = 0.045             # u* / v_10m ratio over rough regolith
DUNE_ANGLE_OF_REPOSE_DEG = 34.0          # Angle of repose for dry basaltic sand


class MartianAeolianDriftModel:
    """
    Evaluates wind shear stress, active sand transport, and dune slipface entrapment risk.
    """
    def __init__(
        self,
        density_kg_m3: float = MARS_ATMOSPHERIC_DENSITY_KG_M3,
        gravity_m_s2: float = MARS_GRAVITY_M_S2,
        u_star_threshold: float = THRESHOLD_FRICTION_VELOCITY_M_S,
    ):
        self.density = density_kg_m3
        self.gravity = gravity_m_s2
        self.u_star_t = u_star_threshold

    def compute_saltation_flux(self, wind_speed_m_s: float = 8.0) -> Dict[str, float]:
        """
        Computes friction velocity u* and Bagnold-White sand mass transport flux q.
        """
        u_star = wind_speed_m_s * WIND_FRICTION_FACTOR
        if u_star > self.u_star_t:
            # Bagnold-White formulation for low-density saltating flux
            excess = u_star - self.u_star_t
            # Mass flux q in g / (m * s)
            flux_kg_m_s = 1.8 * (self.density / self.gravity) * excess * ((u_star + self.u_star_t) ** 2)
            flux_g_m_s = float(flux_kg_m_s * 1000.0)
            status = "ACTIVE_SALTATION_DRIFT"
        elif u_star >= self.u_star_t * 0.70:
            flux_g_m_s = float((u_star / self.u_star_t) * 0.15)
            status = "THRESHOLD_CREEP_WARMUP"
        else:
            flux_g_m_s = 0.0
            status = "STABLE_QUIESCENT_REGOLITH"

        return {
            "friction_velocity_m_s": round(float(u_star), 3),
            "threshold_velocity_m_s": round(float(self.u_star_t), 3),
            "mass_flux_g_m_s": round(flux_g_m_s, 2),
            "saltation_status": status,
        }

    def evaluate_slipface_hazard(
        self,
        sand_coverage_pct: float,
        wind_speed_m_s: float = 8.0,
        wind_azimuth_deg: float = 270.0,
        rover_heading_deg: float = 0.0,
    ) -> Dict[str, Any]:
        """
        Calculates dune slipface collapse risk based on relative wind direction,
        predicting treacherous loose lee-slope avalanches vs packed windward stoss slopes.
        """
        saltation = self.compute_saltation_flux(wind_speed_m_s)
        u_star = saltation["friction_velocity_m_s"]

        # Relative angle between rover heading and wind origin (meteorological source azimuth)
        rel_angle = abs((rover_heading_deg - wind_azimuth_deg + 180.0) % 360.0 - 180.0)

        # Driving downwind (rel_angle near 180°: moving in same direction as wind) forces descent onto loose lee slipfaces
        # Driving upwind (rel_angle near 0°: heading directly into wind) ascends packed windward stoss slopes
        downwind_factor = (1.0 - math.cos(math.radians(rel_angle))) * 0.5  # 1.0 downwind, 0.0 upwind

        # Sand cover scaling
        sand_scale = min(1.0, sand_coverage_pct / 30.0)

        # Risk index [0, 100]
        base_risk = sand_scale * (35.0 + downwind_factor * 45.0)
        if saltation["saltation_status"] == "ACTIVE_SALTATION_DRIFT":
            base_risk += 20.0

        slipface_risk_index = float(np.clip(base_risk, 0.0, 100.0))

        # Recompute optimal traverse heading: perpendicular to wind (cross-wind along crest)
        optimal_crossing_heading = (wind_azimuth_deg + 90.0) % 360.0

        # Operational status and advisory
        if slipface_risk_index >= 70.0:
            hazard_level = "CRITICAL_SLIPFACE_AVALANCHE"
            advisory = (
                f"Severe downwind dune approach (Rel angle {rel_angle:.0f}°). "
                f"Active saltation flux ({saltation['mass_flux_g_m_s']:.1f} g/m/s) creates uncompacted lee slipfaces near {DUNE_ANGLE_OF_REPOSE_DEG}° angle of repose. "
                f"Replan crossing heading to {optimal_crossing_heading:.0f}° to traverse stable crest lines."
            )
        elif slipface_risk_index >= 35.0:
            hazard_level = "ELEVATED_AEOLIAN_DRIFT"
            advisory = (
                f"Moderate sand drift activity (Wind {wind_speed_m_s:.1f} m/s @ {wind_azimuth_deg:.0f}°). "
                "Engage differential creep when crossing transverse sand ripples."
            )
        else:
            hazard_level = "NOMINAL_STABLE_DUNES"
            advisory = "Wind stress below saltation threshold. Sand ripples mechanically cohesive."

        return {
            "wind_speed_m_s": round(float(wind_speed_m_s), 1),
            "wind_azimuth_deg": round(float(wind_azimuth_deg), 1),
            "relative_wind_heading_deg": round(float(rel_angle), 1),
            "optimal_crossing_heading_deg": round(float(optimal_crossing_heading), 1),
            "friction_velocity_m_s": saltation["friction_velocity_m_s"],
            "mass_flux_g_m_s": saltation["mass_flux_g_m_s"],
            "saltation_status": saltation["saltation_status"],
            "slipface_risk_index": round(slipface_risk_index, 1),
            "aeolian_hazard_level": hazard_level,
            "aeolian_advisory": advisory,
        }

    # Alias for flexibility
    evaluate_dune_slipface_hazard = evaluate_slipface_hazard

    def render_drift_overlay(
        self,
        orig_img_np: np.ndarray,
        pred_mask: np.ndarray,
        wind_azimuth_deg: float,
        slipface_risk_index: float,
        alpha: float = 0.60,
    ) -> np.ndarray:
        """
        Renders an aeolian sand drift overlay with directional airflow streamlines
        and amber hazard highlighting across mobile sand dunes.
        """
        orig_h, orig_w = orig_img_np.shape[:2]
        pil_img = Image.fromarray(orig_img_np).convert("RGBA")
        overlay_np = np.zeros((orig_h, orig_w, 4), dtype=np.uint8)

        # Highlight sand regions in amber or warning red depending on slipface risk
        sand_mask = pred_mask == 2
        if np.any(sand_mask):
            r_col = int(np.clip(220 + slipface_risk_index * 0.35, 220, 255))
            g_col = int(np.clip(160 - slipface_risk_index * 1.1, 40, 160))
            overlay_np[sand_mask] = [r_col, g_col, 20, int(alpha * 200)]

        tint_img = Image.fromarray(overlay_np, mode="RGBA")
        draw = ImageDraw.Draw(tint_img)

        # Draw wind direction streamlines
        # Project 2D streamline vector from wind azimuth (0=North/Up, 90=East/Right, 180=South/Down, 270=West/Left)
        rad = math.radians(wind_azimuth_deg)
        dx = -math.sin(rad) * 36.0  # Wind blowing towards
        dy = math.cos(rad) * 36.0

        grid_spacing = max(36, orig_w // 8)
        arrow_col = (255, 255, 255, 220) if slipface_risk_index < 70.0 else (239, 68, 68, 240)

        for y in range(grid_spacing // 2, orig_h - 20, grid_spacing):
            for x in range(grid_spacing // 2, orig_w - 20, grid_spacing):
                if pred_mask[min(y, orig_h - 1), min(x, orig_w - 1)] == 2:
                    x2 = int(np.clip(x + dx, 0, orig_w - 1))
                    y2 = int(np.clip(y + dy, 0, orig_h - 1))
                    draw.line([(x, y), (x2, y2)], fill=arrow_col, width=2)
                    draw.ellipse([(x - 2, y - 2), (x + 2, y + 2)], fill=arrow_col)

        combined = Image.alpha_composite(pil_img, tint_img).convert("RGB")
        return np.array(combined)
