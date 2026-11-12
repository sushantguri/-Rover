"""
Unit tests for Rover Attitude, Slope Stability & Gravity-Coupled Slip Analyzer.
"""
import unittest

from src.slope_stability import SlopeStabilityAnalyzer


class TestSlopeStabilityAnalyzer(unittest.TestCase):
    def setUp(self):
        self.analyzer = SlopeStabilityAnalyzer()

    def test_flat_nominal_attitude(self):
        """Level terrain on firm regolith yields NOMINAL_ATTITUDE and high rollover margin."""
        res = self.analyzer.analyze_attitude(
            pitch_deg=0.0,
            roll_deg=0.0,
            terramechanics_data={"expected_wheel_slip_pct": 5.0, "net_drawbar_pull_n": 2000.0},
            commanded_speed_cm_s=4.5,
        )
        self.assertEqual(res["stability_status"], "NOMINAL_ATTITUDE")
        self.assertEqual(res["combined_tilt_deg"], 0.0)
        self.assertEqual(res["rollover_safety_margin_pct"], 100.0)
        self.assertAlmostEqual(res["slope_coupled_slip_pct"], 5.0, places=1)

    def test_excessive_tilt_triggers_tipover_alert(self):
        """Tilts exceeding 30 degrees operational envelope must trigger TIPOVER_ENVELOPE_EXCEEDED."""
        res = self.analyzer.analyze_attitude(
            pitch_deg=24.0,
            roll_deg=22.0,
            terramechanics_data={"expected_wheel_slip_pct": 12.0, "net_drawbar_pull_n": 1500.0},
        )
        self.assertGreaterEqual(res["combined_tilt_deg"], 30.0)
        self.assertEqual(res["stability_status"], "TIPOVER_ENVELOPE_EXCEEDED")
        self.assertLess(res["rollover_safety_margin_pct"], 40.0)

    def test_steep_climb_over_sandy_regolith_triggers_slip_critical(self):
        """Steep climb on loose sand exhausts drawbar pull or reaches critical slip."""
        res = self.analyzer.analyze_attitude(
            pitch_deg=20.0,
            roll_deg=5.0,
            terramechanics_data={"expected_wheel_slip_pct": 55.0, "net_drawbar_pull_n": 600.0},
            commanded_speed_cm_s=1.8,
        )
        self.assertEqual(res["stability_status"], "STEEP_SLOPE_SLIP_CRITICAL")
        self.assertGreater(res["slope_coupled_slip_pct"], 75.0)
        self.assertGreater(res["grade_resistance_n"], 0.0)


if __name__ == "__main__":
    unittest.main()
