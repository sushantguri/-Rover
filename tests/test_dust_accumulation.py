"""
Unit tests for Martian Dust Deposition, Optical Transmission & Convective Cleaning Model.
"""
import unittest
import numpy as np
from src.dust_accumulation import MartianDustAccumulationModel


class TestMartianDustAccumulation(unittest.TestCase):
    def setUp(self):
        self.model = MartianDustAccumulationModel(
            base_settling_rate_per_sol=0.002,
            min_optical_transmittance=0.35,
            critical_cleaning_threshold=0.70,
        )

    def test_clean_optics_immediately_after_cleaning_event(self):
        """Immediately after a cleaning sol (sols_since_clean=0), transmittance is 100%."""
        t = self.model.evaluate_dust_deposition(sol_number=4100, last_cleaning_sol=4100)
        self.assertEqual(t["optical_transmittance_pct"], 100.0)
        self.assertEqual(t["dust_thickness_um"], 0.0)
        self.assertEqual(t["contrast_loss_pct"], 0.0)
        self.assertEqual(t["mitigation_status"], "NOMINAL_OPTICAL_CLARITY")

    def test_progressive_dust_settling_over_hundreds_of_sols(self):
        """Optical transmittance degrades gracefully over 200 sols without vortex events."""
        t = self.model.evaluate_dust_deposition(
            sol_number=4200, last_cleaning_sol=4000, dust_tau=0.8
        )
        self.assertLess(t["optical_transmittance_pct"], 80.0)
        self.assertGreater(t["dust_thickness_um"], 0.5)
        self.assertGreater(t["contrast_loss_pct"], 10.0)

    def test_extreme_dust_storm_clamps_to_minimum_transmittance(self):
        """Long mission life (1000 sols) without cleaning clamps to min_optical_transmittance."""
        t = self.model.evaluate_dust_deposition(
            sol_number=5000, last_cleaning_sol=4000, dust_tau=3.0
        )
        self.assertEqual(t["optical_transmittance_pct"], 35.0)
        self.assertEqual(t["mitigation_status"], "CRITICAL_DUST_OBSCURATION")
        self.assertIn("Perform Mastcam photometric calibration", t["lens_cleaning_advisory"])

    def test_render_dust_overlay_shape_and_dtype(self):
        """Verifies dust overlay preserves input RGB image shape and dtype."""
        img = np.random.randint(50, 200, (64, 64, 3), dtype=np.uint8)
        t = self.model.evaluate_dust_deposition(sol_number=4150, last_cleaning_sol=4000)
        overlay = self.model.render_dust_overlay(img, t)

        self.assertEqual(overlay.shape, (64, 64, 3))
        self.assertEqual(overlay.dtype, np.uint8)


if __name__ == "__main__":
    unittest.main()
