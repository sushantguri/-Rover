"""
Unit tests for Martian Diurnal Solar Insolation & Thermal Heater Model.
"""
import unittest

from src.solar_thermal import MartianSolarThermalModel, MARS_SOLAR_CONSTANT_W_M2


class TestMartianSolarThermalModel(unittest.TestCase):
    def setUp(self):
        self.model = MartianSolarThermalModel()

    def test_midday_clear_sky_insolation(self):
        """Midday at high sun angle and low tau yields high solar transmission and warm ground."""
        res = self.model.evaluate_thermal_environment(
            local_sol_hour=13.5,
            solar_elevation_deg=65.0,
            tau=0.4,
            optical_shadow_pct=5.0,
        )
        self.assertEqual(res["thermal_status"], "NOMINAL_DAYLIGHT_DRIVE")
        self.assertGreater(res["total_solar_flux_w_m2"], 250.0)
        self.assertGreater(res["effective_chassis_temp_c"], -35.0)
        self.assertEqual(res["survival_heater_power_w"], 0.0)

    def test_night_cold_triggers_survival_heater(self):
        """Pre-dawn hour (04:00 LTST) drops below -35°C and activates survival heaters."""
        res = self.model.evaluate_thermal_environment(
            local_sol_hour=4.0,
            solar_elevation_deg=0.0,
            tau=0.5,
            optical_shadow_pct=50.0,
        )
        self.assertIn("HEATER_ACTIVE", res["thermal_status"])
        self.assertGreater(res["survival_heater_power_w"], 15.0)
        self.assertLess(res["effective_chassis_temp_c"], -60.0)
        self.assertEqual(res["total_solar_flux_w_m2"], 0.0)

    def test_dust_storm_brownout_condition(self):
        """High atmospheric dust opacity (tau >= 3.0) flags EXTREME_DUST_TAU_BROWNOUT."""
        res = self.model.evaluate_thermal_environment(
            local_sol_hour=12.0,
            solar_elevation_deg=50.0,
            tau=3.8,
        )
        self.assertEqual(res["thermal_status"], "EXTREME_DUST_TAU_BROWNOUT")
        self.assertLess(res["direct_flux_w_m2"], 35.0)
        self.assertIn("dust storm", res["thermal_advisory"].lower())

    def test_solar_elevation_zero_returns_zero_flux(self):
        """Sun below or at horizon produces zero direct and diffuse flux."""
        flux = self.model.compute_solar_insolation(solar_elevation_deg=-5.0, tau=0.5)
        self.assertEqual(flux["total_surface_flux_w_m2"], 0.0)
        self.assertEqual(flux["atmospheric_transmission_pct"], 0.0)


if __name__ == "__main__":
    unittest.main()
