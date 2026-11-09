"""
Unit tests for MMRTG Power Budget & Mobility Energy Estimator.
"""
import unittest
from src.power_budget import RoverPowerBudgetEstimator


class TestPowerBudget(unittest.TestCase):
    def setUp(self):
        self.estimator = RoverPowerBudgetEstimator()

    def test_bedrock_low_energy_cost(self):
        """Driving over solid bedrock must yield sustainable power draw and low Wh/m."""
        mechanics = {
            "actuator_torque_demand_nm": 8.0,
            "expected_wheel_slip_pct": 2.0
        }
        res = self.estimator.estimate_traverse_energy(mechanics, commanded_speed_cm_s=4.5)

        self.assertLess(res["energy_cost_wh_per_m"], 1.0)
        self.assertEqual(res["power_status"], "RTG_SUSTAINABLE_DRIVE")
        self.assertFalse(res["thermal_motor_warning"])

    def test_high_slip_sand_excessive_energy_drain(self):
        """High torque and high slip sand must trigger excessive energy drain and thermal warning."""
        mechanics = {
            "actuator_torque_demand_nm": 90.0,
            "expected_wheel_slip_pct": 70.0
        }
        res = self.estimator.estimate_traverse_energy(mechanics, commanded_speed_cm_s=2.0)

        self.assertGreater(res["energy_cost_wh_per_m"], 3.0)
        self.assertEqual(res["power_status"], "EXCESSIVE_ENERGY_DRAIN")
        self.assertTrue(res["thermal_motor_warning"])

    def test_halted_rover_standby_power(self):
        """When commanded speed is 0.0, mobility draw must be 0.0 W."""
        mechanics = {"actuator_torque_demand_nm": 50.0, "expected_wheel_slip_pct": 50.0}
        res = self.estimator.estimate_traverse_energy(mechanics, commanded_speed_cm_s=0.0)

        self.assertEqual(res["mobility_power_draw_w"], 0.0)
        self.assertEqual(res["power_status"], "STANDBY_HALTED")


if __name__ == "__main__":
    unittest.main()
