"""
Martian Diurnal Solar Insolation, Atmospheric Dust Optical Depth (Tau) & Thermal Heater Model.
Models solar flux attenuation through Martian atmospheric dust (Beer-Lambert-Bouguer law),
ground surface temperature cycles, and survival heater electrical power draw for rover batteries.
"""
import math
import numpy as np
from typing import Dict, Any

# Mars solar and atmospheric constants
MARS_SOLAR_CONSTANT_W_M2 = 590.0       # Mean solar irradiance at Mars top-of-atmosphere
MARS_SOL_HOURS = 24.66                 # Duration of one Martian Sol in Earth hours
SURVIVAL_HEATER_MAX_W = 35.0           # MSL/M2020 battery survival heater peak draw
BATTERY_TEMP_LOWER_LIMIT_C = -35.0     # Lithium-ion minimum discharge limit before heaters trigger


class MartianSolarThermalModel:
    """
    Computes solar insolation flux, ambient surface temperatures, and heater electrical loads.
    """
    def __init__(
        self,
        nominal_solar_flux_w_m2: float = MARS_SOLAR_CONSTANT_W_M2,
        max_heater_power_w: float = SURVIVAL_HEATER_MAX_W,
    ):
        self.solar_flux = nominal_solar_flux_w_m2
        self.max_heater_w = max_heater_power_w

    def estimate_diurnal_temperature(
        self,
        local_sol_hour: float = 13.5,
        tau: float = 0.5,
    ) -> float:
        """
        Estimates ambient Martian ground temperature (°C) as a function of Local True Solar Time (LTST).
        Peak temperature occurs ~13:30 LTST; minimum occurs at pre-dawn ~05:00 LTST.
        Heavy dust (high tau) dampens peak temperatures by scattering incoming sunlight.
        """
        hour_clamped = local_sol_hour % MARS_SOL_HOURS
        # Peak at hour 13.5
        phase = 2.0 * math.pi * (hour_clamped - 13.5) / MARS_SOL_HOURS

        # Dust damping: high tau cools daytime peak and slightly warms nighttime minimum
        tau_cooling = min(25.0, (tau - 0.4) * 8.0) if tau > 0.4 else 0.0

        mean_temp_c = -48.0
        temp_amplitude_c = 42.0 - (tau_cooling * 0.5)

        surface_temp_c = mean_temp_c + temp_amplitude_c * math.cos(phase) - (tau_cooling if math.cos(phase) > 0 else 0)
        return round(float(surface_temp_c), 1)

    def compute_solar_insolation(
        self,
        solar_elevation_deg: float = 45.0,
        tau: float = 0.5,
    ) -> Dict[str, float]:
        """
        Computes direct and diffuse solar insolation on horizontal Martian surface (W/m²).
        Uses Beer-Lambert extinction: I_dir = S * sin(elev) * exp(-tau / sin(elev)).
        """
        if solar_elevation_deg <= 0.0:
            return {
                "direct_flux_w_m2": 0.0,
                "diffuse_flux_w_m2": 0.0,
                "total_surface_flux_w_m2": 0.0,
                "atmospheric_transmission_pct": 0.0,
            }

        elev_rad = math.radians(solar_elevation_deg)
        sin_elev = max(0.05, math.sin(elev_rad))

        # Optical air mass proxy
        air_mass = 1.0 / sin_elev
        optical_depth_attenuation = math.exp(-tau * air_mass)

        direct = self.solar_flux * sin_elev * optical_depth_attenuation
        # Diffuse forward scattering in dusty Martian CO2 atmosphere
        diffuse = 0.28 * self.solar_flux * sin_elev * (1.0 - optical_depth_attenuation) * (1.0 / (1.0 + 0.3 * tau))

        total = direct + diffuse
        transmission_pct = (total / (self.solar_flux * sin_elev)) * 100.0 if sin_elev > 0 else 0.0

        return {
            "direct_flux_w_m2": round(float(direct), 1),
            "diffuse_flux_w_m2": round(float(diffuse), 1),
            "total_surface_flux_w_m2": round(float(total), 1),
            "atmospheric_transmission_pct": round(float(transmission_pct), 1),
        }

    def evaluate_thermal_environment(
        self,
        local_sol_hour: float = 13.5,
        solar_elevation_deg: float = 45.0,
        tau: float = 0.5,
        optical_shadow_pct: float = 10.0,
    ) -> Dict[str, Any]:
        """
        Full thermal telemetry evaluation: surface temperature, insolation,
        battery survival heater electrical draw, and autonomous operational window.
        """
        ambient_temp_c = self.estimate_diurnal_temperature(local_sol_hour=local_sol_hour, tau=tau)
        solar = self.compute_solar_insolation(solar_elevation_deg=solar_elevation_deg, tau=tau)

        # Shadow chilling: deep terrain shadows are 8-15°C colder than sunlit surfaces
        effective_temp_c = ambient_temp_c - (optical_shadow_pct / 100.0) * 12.0

        # Survival heater demand: activates progressively when effective temp drops below -35°C
        if effective_temp_c < BATTERY_TEMP_LOWER_LIMIT_C:
            temp_deficit = BATTERY_TEMP_LOWER_LIMIT_C - effective_temp_c
            heater_draw_w = float(np.clip(temp_deficit * 1.5, 5.0, self.max_heater_w))
        else:
            heater_draw_w = 0.0

        # Operational thermal classification
        if tau >= 3.0:
            thermal_status = "EXTREME_DUST_TAU_BROWNOUT"
            advisory = (
                f"Severe global dust storm condition (Tau = {tau:.1f}). Over 90% solar flux attenuated. "
                "Auxiliary solar recharge unavailable; rely strictly on MMRTG."
            )
        elif heater_draw_w >= 25.0 or effective_temp_c <= -70.0:
            thermal_status = "SUB_ZERO_HEATER_ACTIVE"
            advisory = (
                f"Extreme Martian cold ({effective_temp_c:.1f}°C). Battery survival heaters drawing {heater_draw_w:.1f} W. "
                "Mobilize with high-torque throttle to overcome cold grease viscosity."
            )
        elif solar["total_surface_flux_w_m2"] < 40.0 and solar_elevation_deg <= 5.0:
            thermal_status = "TWILIGHT_LOW_INSOLATION"
            advisory = "Sun near horizon; optical illumination deteriorating. Conclude autonomous waypoint drive."
        else:
            thermal_status = "NOMINAL_DAYLIGHT_DRIVE"
            advisory = (
                f"Optimal daytime thermal window ({effective_temp_c:.1f}°C, Flux {solar['total_surface_flux_w_m2']:.0f} W/m²). "
                "Survival heaters inactive."
            )

        return {
            "local_sol_hour": round(float(local_sol_hour), 2),
            "solar_elevation_deg": round(float(solar_elevation_deg), 1),
            "dust_optical_depth_tau": round(float(tau), 2),
            "ambient_surface_temp_c": ambient_temp_c,
            "effective_chassis_temp_c": round(effective_temp_c, 1),
            "total_solar_flux_w_m2": solar["total_surface_flux_w_m2"],
            "direct_flux_w_m2": solar["direct_flux_w_m2"],
            "diffuse_flux_w_m2": solar["diffuse_flux_w_m2"],
            "atmospheric_transmission_pct": solar["atmospheric_transmission_pct"],
            "survival_heater_power_w": round(heater_draw_w, 1),
            "thermal_status": thermal_status,
            "thermal_advisory": advisory,
        }
