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
            "geometry_risk": {
                "nearfield_sand_pct": 12.0,
                "nearfield_big_rock_pct": 3.0,
                "boulder_cluster_count": 2,
                "high_centering_risk": "ELEVATED",
            },
            "kinematics": {
                "steering_maneuver": "SLIGHT_LEFT_ARC",
                "initial_heading_deg": -11.4,
                "estimated_distance_m": 24.5,
                "estimated_drive_time_s": 544.4,
            },
            "science_targets": {
                "targets": [
                    {
                        "target_id": "AEGIS-TGT-1",
                        "category": "BEDROCK_OUTCROP",
                        "priority_score": 91.2,
                        "approach_safety": "SAFE_ROVER_APPROACH",
                    }
                ]
            },
            "stereo_geometry": {
                "max_positive_step_cm": 28.4,
                "max_negative_dropoff_cm": 12.1,
                "belly_clearance_cm": 31.6,
                "clearance_margin_pct": 52.0,
                "step_hazard_status": "CRITICAL_STEP_HAZARD",
            },
            "aeolian_drift": {
                "wind_speed_m_s": 14.2,
                "mass_flux_g_m_s": 6.8,
                "slipface_risk_index": 72.5,
                "aeolian_hazard_level": "CRITICAL_SLIPFACE_ENTRAPMENT_RISK",
                "optimal_crossing_heading_deg": 180.0,
            },
            "dsn_telemetry": {
                "link_budget": {
                    "comm_profile": "UHF_MRO_RELAY",
                    "effective_bitrate_kbps": 2048.0,
                    "est_transfer_time_s": 3.4,
                    "passes_needed": 1,
                },
                "pass_schedule": {
                    "next_orbiter": "MRO (Mars Reconnaissance Orbiter)",
                    "hours_until_pass": 1.5,
                },
                "packets": {
                    "total_bytes": 850000,
                    "priority_counts": {"P1_CRITICAL": 1, "P2_SCIENCE": 1, "P3_COSTMAP": 2, "P4_IMAGE": 3},
                },
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
            overlay_zone_radar=orig,
            overlay_science=orig,
            overlay_wheel_wear=orig,
            overlay_fused_costmap=orig,
            overlay_stereo=orig,
            overlay_aeolian=orig,
        )

        self.assertIn("<!DOCTYPE html>", html)
        self.assertIn("Sol 4125 Field Report", html)
        self.assertIn("CRITICAL HAZARD", html)
        self.assertIn("data:image/jpeg;base64,", html)
        self.assertIn("Terramechanics & Power Budget", html)
        self.assertIn("Epistemic Uncertainty Audit", html)
        self.assertIn("Horizon Zones, Steering Kinematics & AEGIS Geological Targeting", html)
        self.assertIn("3D Stereo Step Hazards, Aeolian Sand Transport & DSN Relay Budget", html)
        self.assertIn("CRITICAL_STEP_HAZARD", html)
        self.assertIn("CRITICAL_SLIPFACE_ENTRAPMENT_RISK", html)
        self.assertIn("2048.0 kbps", html)
        self.assertIn("AEGIS-TGT-1", html)
        self.assertIn("SLIGHT_LEFT_ARC", html)
        self.assertIn("38.5 / 100.0", html)
        self.assertIn("84.0 Wh", html)

    def test_generate_html_report_wheel_wear_and_solar_thermal(self):
        """Verifies report renders wheel fatigue and diurnal solar sections."""
        img = np.zeros((32, 32, 3), dtype=np.uint8)
        telemetry = {
            "safety_status": "NOMINAL_DRIVE",
            "traversability_index": 92.0,
            "terrain_distribution_pct": {"soil": 40.0, "bedrock": 60.0, "sand": 0.0, "big rock": 0.0},
            "wheel_wear": {
                "ventifact_roughness_index": 45.0,
                "peak_contact_stress_mpa": 165.2,
                "aluminum_yield_ratio": 0.60,
                "puncture_risk_pct": 4.2,
                "wear_rate_um_per_km": 72.0,
                "leg_skin_loss_um": 1.44,
                "wear_status": "NOMINAL_WHEEL_WEAR",
                "traction_control_mode": "STANDARD_CRUISE_TORQUE",
            },
            "solar_thermal": {
                "local_sol_hour": 13.5,
                "solar_elevation_deg": 45.0,
                "dust_optical_depth_tau": 0.5,
                "ambient_surface_temp_c": -6.0,
                "effective_chassis_temp_c": -7.2,
                "total_solar_flux_w_m2": 242.0,
                "atmospheric_transmission_pct": 58.0,
                "survival_heater_power_w": 0.0,
                "thermal_status": "NOMINAL_DAYLIGHT_DRIVE",
            },
            "costmap_fusion": {
                "mean_navigable_cost": 2.4,
                "lethal_obstacle_pct": 1.2,
            },
        }

        html = self.generator.generate_html_report(
            telemetry=telemetry,
            orig_img_np=img,
            overlay_multiclass=img,
            overlay_sand=img,
            overlay_wheel_wear=img,
            overlay_fused_costmap=img,
            sol_number=4120,
        )
        self.assertIn("Wheel Ventifact Mechanics, Diurnal Solar Insolation & Nav2 Costmap Fusion", html)
        self.assertIn("165.2 MPa", html)
        self.assertIn("STANDARD_CRUISE_TORQUE", html)
        self.assertIn("242.0 W/m²", html)
        self.assertIn("NOMINAL_DAYLIGHT_DRIVE", html)

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

    def test_generate_html_report_vo_dwa_dust(self):
        """Verifies report renders visual odometry, DWA local reactive steering, and dust accumulation."""
        img = np.zeros((32, 32, 3), dtype=np.uint8)
        telemetry = {
            "safety_status": "NOMINAL_DRIVE",
            "traversability_index": 88.0,
            "terrain_distribution_pct": {"soil": 50.0, "bedrock": 50.0, "sand": 0.0, "big rock": 0.0},
            "visual_odometry": {
                "total_keypoints": 142,
                "bedrock_keypoints": 88,
                "soil_keypoints": 54,
                "sand_keypoints": 0,
                "vo_confidence_pct": 86.5,
                "measured_displacement_m": 0.98,
                "wheel_slip_pct": 4.2,
                "vo_status": "HIGH_CONFIDENCE_TRACKING",
            },
            "dwa_planner": {
                "selected_v_cm_s": 4.5,
                "selected_w_deg_s": 0.0,
                "steer_action": "FORWARD_NOMINAL",
                "trajectory_score": 0.92,
                "evaluated_trajectories": 49,
            },
            "dust_accumulation": {
                "optical_transmittance_pct": 91.2,
                "dust_thickness_um": 0.45,
                "contrast_reduction_pct": 8.8,
                "clearing_vortex_prob_pct": 3.2,
                "dust_status": "MODERATE_DUST_NOMINAL",
            },
        }

        html = self.generator.generate_html_report(
            telemetry=telemetry,
            orig_img_np=img,
            overlay_multiclass=img,
            overlay_sand=img,
            overlay_vo=img,
            overlay_dwa=img,
            overlay_dust=img,
            sol_number=4130,
        )
        self.assertIn("Visual Odometry Feature Slip, DWA Local Reactive Steering & Optical Lens Dust", html)
        self.assertIn("142", html)
        self.assertIn("86.5%", html)
        self.assertIn("FORWARD_NOMINAL", html)
        self.assertIn("91.2%", html)
        self.assertIn("0.45 μm", html)


if __name__ == "__main__":
    unittest.main()
