"""
Multi-Waypoint Sol Mission Battery SoC, Odometry Drift & Watchdog Simulator.
Simulates sequential Mars Rover autonomous driving across multiple Sol legs, accounting for
MMRTG continuous 110 W generation, Li-Ion battery discharge/recharge cycles, slip-induced
wheel odometry drift, AEGIS standoff science pauses, and safety abort interlocks.
"""
from typing import List, Dict, Any
from src.power_budget import (
    MMRTG_CONTINUOUS_OUTPUT_W,
    AVIONICS_BASELINE_LOAD_W,
    BATTERY_CAPACITY_WH,
)


class SolMissionSimulator:
    """
    Stateful multi-leg Mars Sol traverse simulator.
    """
    def __init__(
        self,
        initial_soc_pct: float = 90.0,
        min_reserve_soc_pct: float = 30.0,
        max_consecutive_halts: int = 3,
        default_leg_distance_m: float = 15.0,
        aegis_dwell_seconds: float = 45.0,
    ):
        self.initial_soc_pct = initial_soc_pct
        self.min_reserve_soc_pct = min_reserve_soc_pct
        self.max_consecutive_halts = max_consecutive_halts
        self.default_leg_distance_m = default_leg_distance_m
        self.aegis_dwell_seconds = aegis_dwell_seconds
        # Net RTG recharge power when drive motors are idle (110 W - 55 W = 55 W)
        self.net_rtg_idle_surplus_w = MMRTG_CONTINUOUS_OUTPUT_W - AVIONICS_BASELINE_LOAD_W

    def simulate_sol_traverse(self, step_telemetries: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Simulates a sequence of telemetry packets from sequential Navcam frames.
        Each item may optionally carry `"kinematics"`, `"terramechanics"`, `"power_budget"`,
        and `"science_targets"`.
        """
        battery_wh = BATTERY_CAPACITY_WH * (self.initial_soc_pct / 100.0)
        reserve_wh = BATTERY_CAPACITY_WH * (self.min_reserve_soc_pct / 100.0)

        total_dist_m = 0.0
        total_time_s = 0.0
        total_drift_m = 0.0
        aegis_stops = 0
        consecutive_halts = 0
        mission_status = "SOL_DRIVE_COMPLETED"
        timeline: List[Dict[str, Any]] = []

        for step_idx, step in enumerate(step_telemetries, start=1):
            safety = step.get("safety_status", "NOMINAL_DRIVE")
            power = step.get("power_budget", {})
            mech = step.get("terramechanics", {})
            kin = step.get("kinematics", {})
            sci = step.get("science_targets", {})

            slip_ratio = min(0.95, max(0.0, mech.get("expected_wheel_slip_pct", 5.0) / 100.0))
            eff_speed_cm_s = power.get("effective_ground_speed_cm_s", step.get("recommended_speed_cm_s", 4.0))
            wh_per_m = power.get("energy_cost_wh_per_m", 0.8)

            # Check if halted by hazard perception or thermal warning
            is_halted = ("CRITICAL" in safety) or eff_speed_cm_s <= 0.05 or kin.get("steering_maneuver") == "HALT_NO_PATH"

            if is_halted:
                consecutive_halts += 1
                leg_dist_m = 0.0
                # Hover/replan cooldown (~30s) recharges battery from MMRTG surplus
                leg_time_s = 30.0
                energy_delta_wh = (self.net_rtg_idle_surplus_w * leg_time_s) / 3600.0
                drift_m = 0.0
                leg_action = "HAZARD_HALT_RTG_RECHARGE"
            else:
                consecutive_halts = 0
                raw_dist = kin.get("estimated_distance_m", 0.0)
                leg_dist_m = raw_dist if raw_dist > 0.5 else self.default_leg_distance_m
                speed_m_s = eff_speed_cm_s / 100.0
                drive_s = leg_dist_m / max(0.002, speed_m_s)

                # Mobility power draw balanced against continuous MMRTG 110 W output
                total_sys_w = power.get("total_system_power_w", AVIONICS_BASELINE_LOAD_W + wh_per_m * speed_m_s * 3600.0)
                net_load_w = total_sys_w - MMRTG_CONTINUOUS_OUTPUT_W
                drive_energy_wh = (net_load_w * drive_s) / 3600.0

                # Autonomous AEGIS dwell pause if geology targets locked on firm terrain
                aegis_recharge_wh = 0.0
                if sci.get("target_count", 0) > 0:
                    aegis_stops += 1
                    drive_s += self.aegis_dwell_seconds
                    aegis_recharge_wh = (self.net_rtg_idle_surplus_w * self.aegis_dwell_seconds) / 3600.0

                leg_time_s = drive_s
                energy_delta_wh = aegis_recharge_wh - drive_energy_wh
                # Visual odometry / wheel slip drift scales nonlinearly above 15% slip
                drift_m = leg_dist_m * (slip_ratio ** 1.5) * 0.35
                leg_action = kin.get("steering_maneuver", "NOMINAL_TRAVERSE")

            battery_wh = min(BATTERY_CAPACITY_WH, max(0.0, battery_wh + energy_delta_wh))
            soc_pct = (battery_wh / BATTERY_CAPACITY_WH) * 100.0

            total_dist_m += leg_dist_m
            total_time_s += leg_time_s
            total_drift_m += drift_m

            timeline.append({
                "step": step_idx,
                "action": leg_action,
                "leg_distance_m": round(leg_dist_m, 2),
                "leg_duration_s": round(leg_time_s, 1),
                "net_energy_delta_wh": round(energy_delta_wh, 2),
                "battery_soc_pct": round(soc_pct, 2),
                "cumulative_distance_m": round(total_dist_m, 2),
                "cumulative_drift_m": round(total_drift_m, 3),
            })

            if battery_wh <= reserve_wh:
                mission_status = "MISSION_ABORT_LOW_BATTERY"
                break
            if consecutive_halts >= self.max_consecutive_halts:
                mission_status = "MISSION_HALT_CONSECUTIVE_HAZARDS"
                break

        final_soc_pct = (battery_wh / BATTERY_CAPACITY_WH) * 100.0
        return {
            "mission_status": mission_status,
            "steps_executed": len(timeline),
            "total_distance_m": round(total_dist_m, 2),
            "total_duration_s": round(total_time_s, 1),
            "estimated_odometry_drift_m": round(total_drift_m, 3),
            "aegis_science_stops": aegis_stops,
            "initial_soc_pct": round(self.initial_soc_pct, 2),
            "final_soc_pct": round(final_soc_pct, 2),
            "net_battery_delta_pct": round(final_soc_pct - self.initial_soc_pct, 2),
            "timeline": timeline,
        }
