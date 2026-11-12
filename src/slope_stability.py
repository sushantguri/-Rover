"""
Rover Attitude, Slope Stability & Gravity-Coupled Slip Analyzer.
Models pitch/roll tilt effects on Bekker-Wong wheel traction, static rollover stability margins,
grade climbing resistance, and Martian downhill braking distance (g = 3.721 m/s^2).
"""
import math
from typing import Dict, Any
import numpy as np

from src.terramechanics import (
    ROVER_WEIGHT_MARS_N,
    MARS_GRAVITY_M_S2,
)

# Curiosity/Perseverance geometric rollover limit is ~45 deg;
# JPL operational tilt safety limit on loose Martian slopes is 30 deg.
GEOMETRIC_ROLLOVER_LIMIT_DEG = 45.0
OPERATIONAL_MAX_TILT_DEG = 30.0


class SlopeStabilityAnalyzer:
    """
    Evaluates rover pitch/roll attitude stability and slope-coupled slip mechanics.
    """
    def __init__(
        self,
        max_operational_tilt_deg: float = OPERATIONAL_MAX_TILT_DEG,
        geometric_rollover_deg: float = GEOMETRIC_ROLLOVER_LIMIT_DEG,
    ):
        self.max_operational_tilt_deg = max_operational_tilt_deg
        self.geometric_rollover_deg = geometric_rollover_deg

    def analyze_attitude(
        self,
        pitch_deg: float = 0.0,
        roll_deg: float = 0.0,
        terramechanics_data: Dict[str, Any] | None = None,
        commanded_speed_cm_s: float = 4.5,
    ) -> Dict[str, Any]:
        """
        Computes combined 3D tilt, rollover safety margin, grade resistance,
        slope-amplified wheel slip, and braking distance.
        """
        mech = terramechanics_data or {}
        base_slip_pct = float(mech.get("expected_wheel_slip_pct", 8.0))
        base_dp_n = float(mech.get("net_drawbar_pull_n", 1800.0))

        pitch_rad = math.radians(pitch_deg)
        roll_rad = math.radians(roll_deg)

        # True 3D surface tilt magnitude: atan(sqrt(tan(pitch)^2 + tan(roll)^2))
        combined_tilt_rad = math.atan(math.hypot(math.tan(pitch_rad), math.tan(roll_rad)))
        combined_tilt_deg = math.degrees(combined_tilt_rad)

        # Static rollover safety margin relative to operational tilt envelope
        rollover_margin_pct = max(
            0.0,
            min(100.0, (1.0 - (combined_tilt_deg / max(1e-5, self.geometric_rollover_deg))) * 100.0),
        )

        # Gravity grade resistance: positive = climbing uphill drag; negative = downhill push
        grade_resistance_n = ROVER_WEIGHT_MARS_N * math.sin(pitch_rad)
        lateral_crab_force_n = ROVER_WEIGHT_MARS_N * math.sin(abs(roll_rad))

        # Effective drawbar pull accounting for normal load reduction cos(tilt) & grade drag
        net_slope_drawbar_pull_n = (base_dp_n * math.cos(combined_tilt_rad)) - max(0.0, grade_resistance_n)

        # Slope-coupled slip amplification (uphill + cross-slope regolith shear)
        slope_factor = 1.0 + 2.25 * math.sin(combined_tilt_rad) + 0.65 * math.sin(abs(roll_rad))
        slope_coupled_slip_pct = float(np.clip(base_slip_pct * slope_factor, 1.0, 98.0))

        # Martian downhill stopping distance estimate (d = v^2 / (2 * mu * g - 2 * g * sin(downhill)))
        speed_m_s = max(0.0, float(commanded_speed_cm_s)) / 100.0
        effective_friction = max(0.08, 0.55 * (1.0 - slope_coupled_slip_pct / 120.0))
        downhill_slope_sin = math.sin(max(0.0, -pitch_rad))
        decel_m_s2 = max(0.25, MARS_GRAVITY_M_S2 * (effective_friction * math.cos(combined_tilt_rad) - downhill_slope_sin))
        braking_dist_cm = ((speed_m_s ** 2) / (2.0 * decel_m_s2)) * 100.0

        # Attitude hazard classification
        if combined_tilt_deg >= self.max_operational_tilt_deg:
            status = "TIPOVER_ENVELOPE_EXCEEDED"
            advisory = (
                f"Combined tilt ({combined_tilt_deg:.1f} deg) exceeds {self.max_operational_tilt_deg:.0f} deg JPL operational safety envelope. "
                "Immediate attitude recovery required."
            )
        elif net_slope_drawbar_pull_n <= 0.0 or slope_coupled_slip_pct >= 75.0:
            status = "STEEP_SLOPE_SLIP_CRITICAL"
            advisory = (
                f"Slope-coupled slip ({slope_coupled_slip_pct:.1f}%) exhausts net drawbar pull margin ({net_slope_drawbar_pull_n:.0f} N). "
                "Abort direct ascent and switchback cross-slope."
            )
        elif combined_tilt_deg >= 15.0 or slope_coupled_slip_pct >= 35.0:
            status = "SLOPE_CAUTION_CRAB_RISK"
            advisory = (
                f"Moderate grade/side-slope ({combined_tilt_deg:.1f} deg). Lateral crabbing force {lateral_crab_force_n:.0f} N; throttle drive velocity."
            )
        else:
            status = "NOMINAL_ATTITUDE"
            advisory = "Rover attitude within nominal stability envelope; positive drawbar traction reserve."

        return {
            "pitch_deg": round(float(pitch_deg), 1),
            "roll_deg": round(float(roll_deg), 1),
            "combined_tilt_deg": round(combined_tilt_deg, 1),
            "rollover_safety_margin_pct": round(rollover_margin_pct, 1),
            "grade_resistance_n": round(grade_resistance_n, 1),
            "lateral_crab_force_n": round(lateral_crab_force_n, 1),
            "net_slope_drawbar_pull_n": round(net_slope_drawbar_pull_n, 1),
            "slope_coupled_slip_pct": round(slope_coupled_slip_pct, 1),
            "braking_distance_cm": round(braking_dist_cm, 2),
            "stability_status": status,
            "stability_advisory": advisory,
        }
