"""
Unit tests for RoverTerramechanicsEngine.
"""
import unittest
from src.terramechanics import RoverTerramechanicsEngine


class TestTerramechanics(unittest.TestCase):
    def setUp(self):
        self.engine = RoverTerramechanicsEngine()

    def test_bedrock_minimal_sinkage(self):
        """Bedrock must produce virtually zero sinkage and low slip."""
        dist = {"soil": 0.0, "bedrock": 100.0, "sand": 0.0, "big rock": 0.0}
        res = self.engine.evaluate_terrain_mechanics(dist)

        self.assertLess(res["mean_wheel_sinkage_cm"], 0.2)
        self.assertLess(res["expected_wheel_slip_pct"], 5.0)
        self.assertEqual(res["mobility_status"], "HIGH_TRACTION_NOMINAL")

    def test_sand_high_sinkage_and_slip(self):
        """Sand dune must trigger critical entrapment warning with high sinkage."""
        dist = {"soil": 0.0, "bedrock": 0.0, "sand": 100.0, "big rock": 0.0}
        res = self.engine.evaluate_terrain_mechanics(dist)

        self.assertGreater(res["mean_wheel_sinkage_cm"], 7.0)
        self.assertGreater(res["expected_wheel_slip_pct"], 60.0)
        self.assertGreater(res["entrapment_risk_score"], 65.0)
        self.assertEqual(res["mobility_status"], "CRITICAL_WHEEL_SINKAGE_WARNING")

    def test_resistance_positive_and_finite(self):
        """Rolling resistance and torque must always be positive finite values."""
        dist = {"soil": 40.0, "bedrock": 40.0, "sand": 20.0, "big rock": 0.0}
        res = self.engine.evaluate_terrain_mechanics(dist)

        self.assertGreater(res["rolling_resistance_total_n"], 0.0)
        self.assertGreater(res["actuator_torque_demand_nm"], 0.0)
        self.assertLess(res["actuator_torque_demand_nm"], 200.0)


if __name__ == "__main__":
    unittest.main()
