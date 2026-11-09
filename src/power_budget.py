"""
MMRTG Power Budget & Mobility Energy Consumption Estimator.
Simulates electrical power draw and battery energy impact for MMRTG-powered Mars rovers
(Curiosity/Perseverance class) based on terramechanics actuator torque and wheel slip.
"""
import numpy as np
from typing import Dict, Any

# Physical constants for Curiosity/Perseverance MMRTG & Mobility Subsystem
MMRTG_CONTINUOUS_OUTPUT_W = 110.0     # Continuous RTG thermal-electric power (W)
BATTERY_CAPACITY_WH = 1200.0          # Dual Li-Ion buffer batteries (~1.2 kWh usable)
AVIONICS_BASELINE_LOAD_W = 55.0       # Base flight computer, thermal heating & cameras
DRIVETRAIN_EFFICIENCY = 0.72          # Planetary gearboxes + brushless motor efficiency
NUM_DRIVE_WHEELS = 6
WHEEL_RADIUS_M = 0.25


class RoverPowerBudgetEstimator:
    """
    Calculates specific energy cost (Wh/m) and battery consumption for planetary traverses.
    """
    def __init__(
        self,
        mmrtg_watts: float = MMRTG_CONTINUOUS_OUTPUT_W,
        battery_wh: float = BATTERY_CAPACITY_WH,
        drivetrain_efficiency: float = DRIVETRAIN_EFFICIENCY
    ):
        self.mmrtg_watts = mmrtg_watts
        self.battery_wh = battery_wh
        self.efficiency = drivetrain_efficiency

    def estimate_traverse_energy(
        self,
        terramechanics_data: Dict[str, Any],
        commanded_speed_cm_s: float = 4.5,
        planned_distance_m: float = 20.0
    ) -> Dict[str, Any]:
        """
        Estimates electrical energy draw based on torque demand and slip ratio.
        """
        torque_per_wheel = terramechanics_data.get("actuator_torque_demand_nm", 15.0)
        slip_pct = terramechanics_data.get("expected_wheel_slip_pct", 10.0)
        slip_ratio = np.clip(slip_pct / 100.0, 0.0, 0.96)

        # If rover is commanded to halt (0 cm/s), only baseline avionics power is consumed
        if commanded_speed_cm_s <= 0.05:
            return {
                "effective_ground_speed_cm_s": 0.0,
                "mobility_power_draw_w": 0.0,
                "total_system_power_w": AVIONICS_BASELINE_LOAD_W,
                "energy_cost_wh_per_m": 0.0,
                "traverse_20m_energy_wh": 0.0,
                "battery_impact_pct": 0.0,
                "power_status": "STANDBY_HALTED",
                "thermal_motor_warning": False
            }

        # Wheel angular velocity omega = v_cmd / r (rad/s)
        v_cmd_m_s = commanded_speed_cm_s / 100.0
        omega_rad_s = v_cmd_m_s / WHEEL_RADIUS_M

        # Effective ground speed reduced by wheel slip: v_eff = v_cmd * (1 - slip)
        v_eff_m_s = max(0.002, v_cmd_m_s * (1.0 - slip_ratio))
        v_eff_cm_s = v_eff_m_s * 100.0

        # Mechanical power across 6 wheels: P_mech = 6 * torque * omega
        mech_power_w = NUM_DRIVE_WHEELS * torque_per_wheel * omega_rad_s
        # Electrical mobility power accounting for drivetrain efficiency
        elec_mobility_w = mech_power_w / self.efficiency
        total_power_w = AVIONICS_BASELINE_LOAD_W + elec_mobility_w

        # Specific Energy Cost (Wh per meter of true ground progress)
        # Time to travel 1 meter = 1 / v_eff_m_s seconds = (1 / (v_eff_m_s * 3600)) hours
        hours_per_meter = 1.0 / (v_eff_m_s * 3600.0)
        energy_wh_per_m = total_power_w * hours_per_meter

        # Energy required for standard Sol traverse distance (e.g. 20 meters)
        traverse_energy_wh = energy_wh_per_m * planned_distance_m
        battery_impact_pct = (traverse_energy_wh / self.battery_wh) * 100.0

        # Assess thermal & energy efficiency state
        if energy_wh_per_m > 3.0 or slip_pct > 50.0:
            power_status = "EXCESSIVE_ENERGY_DRAIN"
            thermal_warning = True
        elif energy_wh_per_m > 1.2 or total_power_w > self.mmrtg_watts:
            power_status = "NET_BATTERY_DISCHARGE"
            thermal_warning = False
        else:
            power_status = "RTG_SUSTAINABLE_DRIVE"
            thermal_warning = False

        return {
            "effective_ground_speed_cm_s": round(float(v_eff_cm_s), 2),
            "mobility_power_draw_w": round(float(elec_mobility_w), 1),
            "total_system_power_w": round(float(total_power_w), 1),
            "energy_cost_wh_per_m": round(float(energy_wh_per_m), 2),
            "traverse_20m_energy_wh": round(float(traverse_energy_wh), 1),
            "battery_impact_pct": round(float(battery_impact_pct), 2),
            "power_status": power_status,
            "thermal_motor_warning": thermal_warning
        }
