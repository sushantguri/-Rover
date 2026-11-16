"""
Unit tests for MartianAeolianDriftModel (Wind Saltation & Dune Slipface Hazards).
"""
import unittest
import numpy as np

from src.aeolian_drift import MartianAeolianDriftModel


class TestMartianAeolianDriftModel(unittest.TestCase):
    def setUp(self):
        self.model = MartianAeolianDriftModel()

    def test_saltation_threshold_and_flux(self):
        """Verifies low winds produce stable regolith while gale winds initiate active saltation."""
        # Low wind (5 m/s): u* = 5 * 0.045 = 0.225 m/s < 1.45 m/s threshold
        calm = self.model.compute_saltation_flux(wind_speed_m_s=5.0)
        self.assertEqual(calm["saltation_status"], "STABLE_QUIESCENT_REGOLITH")
        self.assertEqual(calm["mass_flux_g_m_s"], 0.0)

        # High wind (40 m/s): u* = 1.80 m/s > 1.45 m/s threshold
        storm = self.model.compute_saltation_flux(wind_speed_m_s=40.0)
        self.assertEqual(storm["saltation_status"], "ACTIVE_SALTATION_DRIFT")
        self.assertGreater(storm["mass_flux_g_m_s"], 0.0)

    def test_downwind_approach_escalates_slipface_risk(self):
        """Verifies downwind dune descent produces higher risk than upwind/crosswind traversal."""
        # Downwind: wind from 270°, rover heading 90° (downwind onto lee slope)
        downwind = self.model.evaluate_dune_slipface_hazard(
            sand_coverage_pct=35.0,
            wind_speed_m_s=15.0,
            wind_azimuth_deg=270.0,
            rover_heading_deg=90.0,
        )

        # Upwind: wind from 270°, rover heading 270° (into wind onto packed stoss slope)
        upwind = self.model.evaluate_dune_slipface_hazard(
            sand_coverage_pct=35.0,
            wind_speed_m_s=15.0,
            wind_azimuth_deg=270.0,
            rover_heading_deg=270.0,
        )

        self.assertGreater(downwind["slipface_risk_index"], upwind["slipface_risk_index"])
        self.assertIn("optimal_crossing_heading_deg", downwind)

    def test_render_drift_overlay_shape_and_dtype(self):
        """Verifies drift overlay renders a valid RGB image matching input dimensions."""
        orig = np.random.randint(40, 200, (64, 64, 3), dtype=np.uint8)
        mask = np.full((64, 64), 2, dtype=np.int64)  # Sand
        hazard = self.model.evaluate_dune_slipface_hazard(sand_coverage_pct=50.0)

        overlay = self.model.render_drift_overlay(
            orig, mask, hazard["wind_azimuth_deg"], hazard["slipface_risk_index"], alpha=0.60
        )
        self.assertEqual(overlay.shape, (64, 64, 3))
        self.assertEqual(overlay.dtype, np.uint8)


if __name__ == "__main__":
    unittest.main()
