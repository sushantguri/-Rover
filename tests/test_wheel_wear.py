"""
Unit tests for Curiosity/Perseverance Aluminum Wheel Ventifact Wear & Puncture Fatigue Mechanics.
"""
import unittest
import numpy as np

from src.wheel_wear import RoverWheelWearModel, ALUMINUM_YIELD_STRENGTH_MPA


class TestRoverWheelWearModel(unittest.TestCase):
    def setUp(self):
        self.model = RoverWheelWearModel()

    def test_ventifact_roughness_zero_on_pure_soil(self):
        """Terrain with zero bedrock/rocks produces zero ventifact roughness."""
        gray = np.random.randint(50, 200, (64, 64), dtype=np.uint8)
        mask = np.zeros((64, 64), dtype=np.int64)  # 0: Soil
        roughness = self.model.compute_ventifact_roughness(gray, mask)
        self.assertEqual(roughness, 0.0)

    def test_ventifact_roughness_detects_high_frequency_bedrock_relief(self):
        """Checkerboard/high-relief texture on bedrock yields high roughness index."""
        gray = np.zeros((64, 64), dtype=np.uint8)
        gray[::2, ::2] = 240
        gray[1::2, 1::2] = 240
        mask = np.ones((64, 64), dtype=np.int64)  # 1: Bedrock
        roughness = self.model.compute_ventifact_roughness(gray, mask)
        self.assertGreater(roughness, 50.0)

    def test_smooth_regolith_yields_nominal_wear(self):
        """Flat compact soil yields NOMINAL_WHEEL_WEAR and low contact stress."""
        res = self.model.evaluate_wheel_wear(
            terrain_distribution_pct={"soil": 90.0, "bedrock": 10.0, "sand": 0.0, "big rock": 0.0},
            ventifact_roughness_index=5.0,
            drive_distance_m=20.0,
        )
        self.assertEqual(res["wear_status"], "NOMINAL_WHEEL_WEAR")
        self.assertLess(res["peak_contact_stress_mpa"], ALUMINUM_YIELD_STRENGTH_MPA * 0.5)
        self.assertLess(res["puncture_risk_pct"], 5.0)
        self.assertEqual(res["traction_control_mode"], "STANDARD_CRUISE_TORQUE")

    def test_sharp_ventifact_bedrock_triggers_puncture_hazard(self):
        """High-roughness bedrock with wheel slip exceeds yield stress and flags CRITICAL_PUNCTURE_HAZARD."""
        res = self.model.evaluate_wheel_wear(
            terrain_distribution_pct={"bedrock": 85.0, "soil": 5.0, "sand": 0.0, "big rock": 10.0},
            terramechanics_data={"expected_wheel_slip_pct": 35.0},
            ventifact_roughness_index=95.0,
            drive_distance_m=50.0,
        )
        self.assertEqual(res["wear_status"], "CRITICAL_PUNCTURE_HAZARD")
        self.assertGreaterEqual(res["peak_contact_stress_mpa"], ALUMINUM_YIELD_STRENGTH_MPA)
        self.assertGreaterEqual(res["puncture_risk_pct"], 45.0)
        self.assertEqual(res["traction_control_mode"], "TRACTION_CONTROL_ACTIVE_STRICT")
        self.assertGreater(res["leg_skin_loss_um"], 0.01)

    def test_wear_hazard_overlay_preserves_shape(self):
        """Overlay generation returns an RGB uint8 array matching input frame shape."""
        img = np.random.randint(60, 200, (48, 64, 3), dtype=np.uint8)
        mask = np.ones((48, 64), dtype=np.int64)
        overlay = self.model.render_wear_hazard_overlay(img, mask, ventifact_roughness_index=75.0)
        self.assertEqual(overlay.shape, (48, 64, 3))
        self.assertEqual(overlay.dtype, np.uint8)


if __name__ == "__main__":
    unittest.main()
