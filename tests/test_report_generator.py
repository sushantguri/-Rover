"""
Unit tests for Automated Sol Drive Engineering Field Report Generator.
"""
import unittest
import numpy as np

from src.report_generator import MissionReportGenerator


class TestMissionReportGenerator(unittest.TestCase):
    def setUp(self):
        self.generator = MissionReportGenerator()

    def test_generate_html_report_contains_required_sections(self):
        """Verifies HTML report compiles embedded base64 imagery and all telemetry tables."""
        orig = np.random.randint(40, 220, (64, 64, 3), dtype=np.uint8)
        overlay_multi = np.random.randint(40, 220, (64, 64, 3), dtype=np.uint8)
        overlay_sand = np.random.randint(40, 220, (64, 64, 3), dtype=np.uint8)
        overlay_path = np.random.randint(40, 220, (64, 64, 3), dtype=np.uint8)

        telemetry = {
            "tta_mode": "fast",
            "inference_latency_ms": 42.5,
            "device": "cpu",
            "safety_status": "CRITICAL_HAZARD",
            "traversability_index": 38.5,
            "autonomous_drive_action": "HALT & REPLAN",
            "recommended_speed_cm_s": 0.0,
            "terrain_distribution_pct": {
                "soil": 20.0,
                "bedrock": 30.0,
                "sand": 45.0,
                "big rock": 5.0,
            },
            "terramechanics": {
                "mean_wheel_sinkage_cm": 5.4,
                "expected_wheel_slip_pct": 62.0,
                "actuator_torque_demand_nm": 28.1,
            },
            "power_budget": {
                "energy_cost_wh_per_m": 4.2,
                "traverse_20m_energy_wh": 84.0,
                "battery_impact_pct": 7.0,
            },
            "uncertainty": {
                "confidence_status": "HIGH_CONFIDENCE_CLEAR",
                "mean_prediction_entropy": 0.145,
                "p90_prediction_entropy": 0.320,
                "high_uncertainty_pixel_pct": 3.5,
                "perception_recommendation": "Full autonomous trust level.",
            },
        }
        path_data = {
            "status": "PATH_FOUND",
            "total_path_cost": 148.2,
        }

        html = self.generator.generate_html_report(
            telemetry=telemetry,
            orig_img_np=orig,
            overlay_multiclass=overlay_multi,
            overlay_sand=overlay_sand,
            overlay_path=overlay_path,
            path_data=path_data,
            sol_number=4125,
        )

        self.assertIn("<!DOCTYPE html>", html)
        self.assertIn("Sol 4125 Field Report", html)
        self.assertIn("CRITICAL HAZARD", html)
        self.assertIn("data:image/jpeg;base64,", html)
        self.assertIn("Terramechanics & Power Budget", html)
        self.assertIn("Epistemic Uncertainty Audit", html)
        self.assertIn("38.5 / 100.0", html)
        self.assertIn("84.0 Wh", html)

    def test_generate_html_report_nominal_fallback(self):
        """Verifies report handles nominal status and omitted path overlay cleanly."""
        img = np.zeros((32, 32, 3), dtype=np.uint8)
        telemetry = {
            "safety_status": "NOMINAL_DRIVE",
            "traversability_index": 94.0,
            "terrain_distribution_pct": {"soil": 30.0, "bedrock": 70.0, "sand": 0.0, "big rock": 0.0},
        }

        html = self.generator.generate_html_report(
            telemetry=telemetry,
            orig_img_np=img,
            overlay_multiclass=img,
            overlay_sand=img,
            sol_number=4118,
        )
        self.assertIn("NOMINAL DRIVE", html)
        self.assertIn("Sol 4118", html)


if __name__ == "__main__":
    unittest.main()
