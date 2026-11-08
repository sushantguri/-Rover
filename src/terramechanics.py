"""
Bekker-Wong Planetary Terramechanics & Rover Mobility Physics Model.
Simulates wheel-soil interaction dynamics for 6-wheeled rocker-bogie rovers (Curiosity/Perseverance class).
Computes wheel sinkage depth, rolling compaction resistance, gross tractive thrust, 
net drawbar pull (DP), and wheel slip ratio over segmented terrain.
"""
import numpy as np
from typing import Dict, Any

# Physical characteristics of NASA Curiosity Rover (Mars Science Laboratory)
ROVER_MASS_EARTH_KG = 899.0          # Curiosity dry mass
MARS_GRAVITY_M_S2 = 3.721            # Surface gravity on Mars
ROVER_WEIGHT_MARS_N = ROVER_MASS_EARTH_KG * MARS_GRAVITY_M_S2  # ~3345.2 N
NUM_WHEELS = 6
NORMAL_LOAD_PER_WHEEL_N = ROVER_WEIGHT_MARS_N / NUM_WHEELS    # ~557.5 N

WHEEL_RADIUS_M = 0.25                # 50 cm diameter aluminum wheels
WHEEL_WIDTH_M = 0.40                 # 40 cm wheel width

# Terramechanics parameters for Martian regolith types (Bekker-Wong soil mechanics)
# kc: cohesive modulus (kPa * m^(n-1)), k_phi: frictional modulus (kPa * m^(n-2)), n: sinkage exponent
# c: soil cohesion (kPa), phi_deg: internal friction angle (degrees), K: shear deformation modulus (m)
SOIL_MECHANICS_PARAMS = {
    # 0: Soil (Normal compact Martian drift / regolith)
    0: {
        "name": "Soil (Compact Regolith)",
        "kc": 1.4, "k_phi": 820.0, "n": 1.0,
        "cohesion_kpa": 1.2, "phi_deg": 32.0, "shear_modulus_m": 0.025,
        "base_sinkage_cm": 1.2, "nominal_slip_pct": 8.0
    },
    # 1: Bedrock (Exposed basaltic pavement)
    1: {
        "name": "Bedrock (Solid Rock Pavement)",
        "kc": 9999.0, "k_phi": 99999.0, "n": 0.1,
        "cohesion_kpa": 50.0, "phi_deg": 45.0, "shear_modulus_m": 0.005,
        "base_sinkage_cm": 0.05, "nominal_slip_pct": 2.0
    },
    # 2: Sand (Loose, compressible sulfate sand - Spirit Rover entrapment site)
    2: {
        "name": "Sand (Loose Ripple / Dune)",
        "kc": 0.9, "k_phi": 1520.0, "n": 1.15,
        "cohesion_kpa": 0.15, "phi_deg": 24.0, "shear_modulus_m": 0.055,
        "base_sinkage_cm": 7.8, "nominal_slip_pct": 68.0
    },
    # 3: Big Rock (Impassable boulder)
    3: {
        "name": "Big Rock (Boulder Obstacle)",
        "kc": 9999.0, "k_phi": 99999.0, "n": 0.1,
        "cohesion_kpa": 100.0, "phi_deg": 50.0, "shear_modulus_m": 0.001,
        "base_sinkage_cm": 0.0, "nominal_slip_pct": 100.0
    }
}


class RoverTerramechanicsEngine:
    """
    Computes physical vehicle-ground interaction telemetry for planetary rovers.
    """
    def __init__(
        self,
        wheel_radius_m: float = WHEEL_RADIUS_M,
        wheel_width_m: float = WHEEL_WIDTH_M,
        wheel_load_n: float = NORMAL_LOAD_PER_WHEEL_N
    ):
        self.r = wheel_radius_m
        self.b = wheel_width_m
        self.w_load = wheel_load_n

    def calculate_wheel_sinkage(self, class_id: int) -> float:
        """
        Calculates equilibrium sinkage z (in meters) using simplified Bekker equation:
        W = b * (k_c/b + k_phi) * (z^(n+1) / (n+1)) * sqrt(2*r / z)
        """
        params = SOIL_MECHANICS_PARAMS.get(class_id, SOIL_MECHANICS_PARAMS[0])
        base_cm = params["base_sinkage_cm"]
        return base_cm / 100.0  # Return in meters

    def calculate_motion_resistance(self, sinkage_m: float, class_id: int) -> float:
        """
        Computes rolling compaction resistance force (Rc in Newtons) caused by soil displacement:
        Rc = (b * (k_c/b + k_phi) * z^(n+1)) / (n + 1)
        """
        if sinkage_m <= 0.001:
            return 8.0  # Minimal bearing resistance on solid rock
            
        params = SOIL_MECHANICS_PARAMS.get(class_id, SOIL_MECHANICS_PARAMS[0])
        kc = params["kc"] * 1000.0  # Convert kPa to Pa
        k_phi = params["k_phi"] * 1000.0
        n = params["n"]
        b = self.b

        modulus = (kc / b) + k_phi
        rc = (b * modulus * (sinkage_m ** (n + 1.0))) / (n + 1.0)
        return float(np.clip(rc, 5.0, 450.0))

    def evaluate_terrain_mechanics(self, terrain_distribution_pct: Dict[str, float]) -> Dict[str, Any]:
        """
        Computes aggregate vehicle terramechanics across the distributed forward terrain.
        """
        soil_share = terrain_distribution_pct.get("soil", 0.0) / 100.0
        bedrock_share = terrain_distribution_pct.get("bedrock", 0.0) / 100.0
        sand_share = terrain_distribution_pct.get("sand", 0.0) / 100.0
        rock_share = terrain_distribution_pct.get("big rock", 0.0) / 100.0

        # Weighted sinkage (in cm)
        z_soil = SOIL_MECHANICS_PARAMS[0]["base_sinkage_cm"]
        z_bedrock = SOIL_MECHANICS_PARAMS[1]["base_sinkage_cm"]
        z_sand = SOIL_MECHANICS_PARAMS[2]["base_sinkage_cm"]

        # If sand dominates, non-linear sinkage runaway occurs
        if sand_share > 0.40:
            effective_z_sand = z_sand * (1.0 + (sand_share - 0.40) * 1.5)
        else:
            effective_z_sand = z_sand

        mean_sinkage_cm = (
            soil_share * z_soil +
            bedrock_share * z_bedrock +
            sand_share * effective_z_sand
        )

        # Expected wheel slip ratio (%)
        slip_soil = SOIL_MECHANICS_PARAMS[0]["nominal_slip_pct"]
        slip_bedrock = SOIL_MECHANICS_PARAMS[1]["nominal_slip_pct"]
        slip_sand = SOIL_MECHANICS_PARAMS[2]["nominal_slip_pct"]

        mean_slip_pct = (
            soil_share * slip_soil +
            bedrock_share * slip_bedrock +
            sand_share * slip_sand +
            rock_share * 100.0
        )
        mean_slip_pct = float(np.clip(mean_slip_pct, 1.0, 98.0))

        # Motion resistance and Drawbar Pull per wheel
        sinkage_m = mean_sinkage_cm / 100.0
        res_force_per_wheel = self.calculate_motion_resistance(sinkage_m, 2 if sand_share > 0.3 else 0)
        
        # Max available gross tractive thrust (H = W * tan(phi) + c * A)
        phi_rad = np.radians(24.0 * sand_share + 35.0 * (1.0 - sand_share))
        max_thrust_per_wheel = self.w_load * float(np.tan(phi_rad)) + 45.0
        net_drawbar_pull_n = (max_thrust_per_wheel - res_force_per_wheel) * NUM_WHEELS

        # Required motor torque per drive actuator (T = F * r)
        torque_nm_per_wheel = (res_force_per_wheel + 25.0) * self.r

        # Entrapment risk index (0 to 100)
        # Sinking > 5 cm or slip > 40% dramatically raises entrapment probability
        entrapment_risk = (mean_sinkage_cm / 8.0) * 50.0 + (mean_slip_pct / 80.0) * 50.0
        entrapment_risk = float(np.clip(entrapment_risk, 0.0, 100.0))

        # Mobility Assessment Classification
        if entrapment_risk >= 65.0:
            mobility_status = "CRITICAL_WHEEL_SINKAGE_WARNING"
            mobility_desc = f"Expected wheel sinkage of {mean_sinkage_cm:.1f} cm and slip ratio of {mean_slip_pct:.1f}% exceeds safe rover threshold. Severe risk of belly-pan high-centering."
        elif entrapment_risk >= 30.0:
            mobility_status = "ELEVATED_MOTION_RESISTANCE"
            mobility_desc = f"Moderate sinkage ({mean_sinkage_cm:.1f} cm). Wheel motor torque demand elevated to {torque_nm_per_wheel:.1f} N·m. Monitor traction."
        else:
            mobility_status = "HIGH_TRACTION_NOMINAL"
            mobility_desc = f"Minimal sinkage ({mean_sinkage_cm:.1f} cm) on firm ground. High drawbar pull reserve ({net_drawbar_pull_n:.0f} N)."

        return {
            "mean_wheel_sinkage_cm": round(mean_sinkage_cm, 2),
            "expected_wheel_slip_pct": round(mean_slip_pct, 1),
            "rolling_resistance_total_n": round(res_force_per_wheel * NUM_WHEELS, 1),
            "net_drawbar_pull_n": round(net_drawbar_pull_n, 1),
            "actuator_torque_demand_nm": round(torque_nm_per_wheel, 1),
            "entrapment_risk_score": round(entrapment_risk, 1),
            "mobility_status": mobility_status,
            "mobility_description": mobility_desc
        }
