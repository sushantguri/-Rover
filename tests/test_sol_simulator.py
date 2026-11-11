"""
Unit tests for Multi-Waypoint Sol Mission Battery SoC & Odometry Simulator.
"""
import unittest

from src.sol_simulator import SolMissionSimulator


class TestSolMissionSimulator(unittest.TestCase):
    def test_nominal_sol_traverse_accumulates_distance_and_aegis_stops(self):
        """Sequential nominal legs accumulate traverse distance, SoC delta, and AEGIS stops."""
        sim = SolMissionSimulator(initial_soc_pct=90.0)
        steps = [
            {
                "safety_status": "NOMINAL_DRIVE",
                "terramechanics": {"expected_wheel_slip_pct": 4.0},
                "power_budget": {"effective_ground_speed_cm_s": 4.5, "total_system_power_w": 95.0},
                "kinematics": {"estimated_distance_m": 20.0, "steering_maneuver": "STRAIGHT_CRUISE"},
                "science_targets": {"target_count": 2},
            },
            {
                "safety_status": "CAUTION_ADVISORY",
                "terramechanics": {"expected_wheel_slip_pct": 25.0},
                "power_budget": {"effective_ground_speed_cm_s": 1.4, "total_system_power_w": 165.0},
                "kinematics": {"estimated_distance_m": 15.0, "steering_maneuver": "SLIGHT_LEFT_ARC"},
                "science_targets": {"target_count": 0},
            },
        ]
        res = sim.simulate_sol_traverse(steps)
        self.assertEqual(res["mission_status"], "SOL_DRIVE_COMPLETED")
        self.assertEqual(res["steps_executed"], 2)
        self.assertAlmostEqual(res["total_distance_m"], 35.0, places=1)
        self.assertEqual(res["aegis_science_stops"], 1)
        self.assertGreater(res["estimated_odometry_drift_m"], 0.0)

    def test_consecutive_hazards_trigger_mission_abort(self):
        """Three consecutive CRITICAL_HAZARD frames must trip watchdog abort cleanly."""
        sim = SolMissionSimulator(max_consecutive_halts=3)
        hazard_step = {
            "safety_status": "CRITICAL_HAZARD",
            "power_budget": {"effective_ground_speed_cm_s": 0.0},
        }
        res = sim.simulate_sol_traverse([hazard_step, hazard_step, hazard_step, hazard_step])
        self.assertEqual(res["mission_status"], "MISSION_HALT_CONSECUTIVE_HAZARDS")
        self.assertEqual(res["steps_executed"], 3)
        self.assertEqual(res["total_distance_m"], 0.0)

    def test_low_battery_floor_triggers_hibernation_abort(self):
        """Dropping below reserve SoC triggers MISSION_ABORT_LOW_BATTERY."""
        sim = SolMissionSimulator(initial_soc_pct=31.0, min_reserve_soc_pct=30.0)
        heavy_step = {
            "safety_status": "CAUTION_ADVISORY",
            "terramechanics": {"expected_wheel_slip_pct": 60.0},
            "power_budget": {"effective_ground_speed_cm_s": 1.0, "total_system_power_w": 350.0},
            "kinematics": {"estimated_distance_m": 25.0, "steering_maneuver": "HARD_EVASION_LEFT"},
        }
        res = sim.simulate_sol_traverse([heavy_step, heavy_step])
        self.assertEqual(res["mission_status"], "MISSION_ABORT_LOW_BATTERY")
        self.assertEqual(res["steps_executed"], 1)


if __name__ == "__main__":
    unittest.main()
