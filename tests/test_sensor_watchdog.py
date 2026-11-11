"""
Unit tests for Optical Sensor Health & Atmospheric Fault Watchdog.
"""
import unittest
import numpy as np

from src.sensor_watchdog import OpticalSensorWatchdog


class TestOpticalSensorWatchdog(unittest.TestCase):
    def setUp(self):
        self.watchdog = OpticalSensorWatchdog()

    def test_nominal_textured_martian_frame(self):
        """High-contrast textured frame must report NOMINAL_OPTICAL and trust=100."""
        rng = np.random.default_rng(42)
        img = rng.integers(50, 210, (128, 128, 3), dtype=np.uint8)
        res = self.watchdog.inspect_frame(img)

        self.assertEqual(res["sensor_health_status"], "NOMINAL_OPTICAL")
        self.assertEqual(res["optical_trust_score"], 100.0)
        self.assertFalse(res["recommend_tta_boost"])
        self.assertEqual(res["active_fault_flags"], [])

    def test_glare_saturation_warning(self):
        """Heavy specular highlight clipping must trigger GLARE_SATURATION_WARNING and TTA recommendation."""
        img = np.full((100, 100, 3), 120, dtype=np.uint8)
        img[:40, :] = 254  # 40% white specular glare
        res = self.watchdog.inspect_frame(img)

        self.assertEqual(res["sensor_health_status"], "GLARE_SATURATION_WARNING")
        self.assertIn("SOLAR_GLARE_CLIPPING", res["active_fault_flags"])
        self.assertTrue(res["recommend_tta_boost"])

    def test_dark_corrupted_frame_fault(self):
        """Pitch-black / heavily underexposed frame must raise SENSOR_FAULT_DARK_FRAME."""
        img = np.full((100, 100, 3), 4, dtype=np.uint8)
        res = self.watchdog.inspect_frame(img)

        self.assertEqual(res["sensor_health_status"], "SENSOR_FAULT_DARK_FRAME")
        self.assertLess(res["optical_trust_score"], 50.0)
        self.assertIn("DEEP_SHADOW_UNDEREXPOSURE", res["active_fault_flags"])

    def test_dust_storm_low_contrast_haze(self):
        """Uniform flat mid-grey frame simulates dust-storm contrast collapse."""
        img = np.full((100, 100, 3), 145, dtype=np.uint8)
        res = self.watchdog.inspect_frame(img)

        self.assertEqual(res["sensor_health_status"], "ATMOSPHERIC_HAZE_ADVISORY")
        self.assertIn("LOW_CONTRAST_DUST_HAZE", res["active_fault_flags"])
        self.assertTrue(res["recommend_tta_boost"])


if __name__ == "__main__":
    unittest.main()
