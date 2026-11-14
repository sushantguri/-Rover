"""
Automated NASA Sol Drive Engineering Report Generator.
Compiles neural perception telemetry, Bekker-Wong terramechanics, MMRTG power budgets,
and visual diagnostics into a standalone, self-contained HTML engineering report.
"""
import os
import io
import base64
import datetime
import numpy as np
from PIL import Image
from typing import Dict, Any, Optional


def _img_to_base64(img_np: np.ndarray) -> str:
    pil_img = Image.fromarray(img_np)
    # Resize for clean report embedding if extremely large
    pil_img.thumbnail((640, 640), Image.BILINEAR)
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=88)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


class MissionReportGenerator:
    """
    Generates self-contained HTML Sol Drive Engineering Field Reports.
    """
    def generate_html_report(
        self,
        telemetry: Dict[str, Any],
        orig_img_np: np.ndarray,
        overlay_multiclass: np.ndarray,
        overlay_sand: np.ndarray,
        overlay_path: Optional[np.ndarray] = None,
        path_data: Optional[Dict[str, Any]] = None,
        sol_number: int = 4118,
        overlay_zone_radar: Optional[np.ndarray] = None,
        overlay_science: Optional[np.ndarray] = None,
        overlay_wheel_wear: Optional[np.ndarray] = None,
        overlay_fused_costmap: Optional[np.ndarray] = None
    ) -> str:
        """
        Compiles complete telemetry and imagery into a polished printable HTML report.
        """
        now_utc = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        dist = telemetry.get("terrain_distribution_pct", {})
        mech = telemetry.get("terramechanics", {})
        uncert = telemetry.get("uncertainty", {})
        power = telemetry.get("power_budget", {})
        geom = telemetry.get("geometry_risk", {})
        kin = telemetry.get("kinematics", {})
        sci = telemetry.get("science_targets", {})
        sensor = telemetry.get("sensor_health", {})
        slope = telemetry.get("slope_stability", {})
        cal = telemetry.get("calibration_safety", {})
        wear = telemetry.get("wheel_wear", {})
        solar = telemetry.get("solar_thermal", {})
        cost_fus = telemetry.get("costmap_fusion", {})

        sensor_row_html = ""
        if sensor:
            sensor_row_html = (
                f"<tr><td>Optical Sensor Watchdog</td>"
                f"<td><strong>{sensor.get('sensor_health_status', 'NOMINAL_OPTICAL')}</strong> "
                f"(Trust {sensor.get('optical_trust_score', 100.0)}%, Glare {sensor.get('overexposed_pct', 0.0)}%)</td></tr>"
            )
        slope_row_html = ""
        if slope:
            slope_row_html = (
                f"<tr><td>Attitude &amp; Slope Stability</td>"
                f"<td><strong>{slope.get('stability_status', 'NOMINAL_ATTITUDE')}</strong> "
                f"(Tilt {slope.get('combined_tilt_deg', 0.0)}°, Slip {slope.get('slope_coupled_slip_pct', 0.0)}%)</td></tr>"
            )
        cal_row_html = ""
        if cal:
            cal_row_html = (
                f"<tr><td>Conformal Sand Recall Margin</td>"
                f"<td><strong>{cal.get('conformal_alert', 'CALIBRATED_NOMINAL')}</strong> "
                f"(Conservative {cal.get('conservative_sand_pct', 0.0)}%, Borderline +{cal.get('borderline_sand_pct', 0.0)}%)</td></tr>"
            )

        status = telemetry.get("safety_status", "UNKNOWN")
        if "CRITICAL" in status:
            badge_bg = "#ef4444"
            status_label = "CRITICAL HAZARD — AUTONOMOUS HALT"
        elif "CAUTION" in status:
            badge_bg = "#f59e0b"
            status_label = "CAUTION ADVISORY — SLOW CREEP MODE"
        else:
            badge_bg = "#10b981"
            status_label = "NOMINAL DRIVE — PATH CLEAR"

        b64_orig = _img_to_base64(orig_img_np)
        b64_multi = _img_to_base64(overlay_multiclass)
        b64_sand = _img_to_base64(overlay_sand)
        b64_path = _img_to_base64(overlay_path) if overlay_path is not None else b64_multi

        extra_img_html = ""
        if overlay_zone_radar is not None or overlay_science is not None or overlay_wheel_wear is not None or overlay_fused_costmap is not None:
            if overlay_zone_radar is not None:
                b64_radar = _img_to_base64(overlay_zone_radar)
                extra_img_html += f"""
                <div class="img-box">
                    <img src="data:image/jpeg;base64,{b64_radar}" alt="Near-Field Zone Radar">
                    <div class="img-caption">E. Near-Field Zone & Boulder Risk Radar</div>
                </div>"""
            if overlay_science is not None:
                b64_sci = _img_to_base64(overlay_science)
                extra_img_html += f"""
                <div class="img-box">
                    <img src="data:image/jpeg;base64,{b64_sci}" alt="AEGIS Science Targets">
                    <div class="img-caption">F. AEGIS Autonomous Science Target Reticles</div>
                </div>"""
            if overlay_wheel_wear is not None:
                b64_wear = _img_to_base64(overlay_wheel_wear)
                extra_img_html += f"""
                <div class="img-box">
                    <img src="data:image/jpeg;base64,{b64_wear}" alt="Ventifact Wheel Wear Hazard">
                    <div class="img-caption">G. Ventifact Wheel Wear & Puncture Risk</div>
                </div>"""
            if overlay_fused_costmap is not None:
                b64_fused = _img_to_base64(overlay_fused_costmap)
                extra_img_html += f"""
                <div class="img-box">
                    <img src="data:image/jpeg;base64,{b64_fused}" alt="Fused Nav2 Costmap">
                    <div class="img-caption">H. Multi-Layer Fused Nav2 Costmap</div>
                </div>"""

        advanced_section_html = ""
        if geom or kin or sci:
            targets_rows = "".join(
                f"<tr><td>{t.get('target_id')}</td><td>{t.get('category')}</td>"
                f"<td>{t.get('priority_score')}</td><td>{t.get('approach_safety')}</td></tr>"
                for t in sci.get("targets", [])
            ) or "<tr><td colspan='4'>No high-priority outcrop/rock targets detected.</td></tr>"
            advanced_section_html = f"""
        <h2>4. Horizon Zones, Steering Kinematics & AEGIS Geological Targeting</h2>
        <div class="grid-2">
            <div class="card">
                <strong>Range-Zone Hazards & Steering Kinematics</strong>
                <table>
                    <tr><th>Subsystem Metric</th><th>Estimate</th></tr>
                    <tr><td>Near-Field (0-5m) Sand / Rock</td><td>{geom.get('nearfield_sand_pct', 0.0)}% / {geom.get('nearfield_big_rock_pct', 0.0)}%</td></tr>
                    <tr><td>Boulder Clusters / Centering Risk</td><td>{geom.get('boulder_cluster_count', 0)} clusters ({geom.get('high_centering_risk', 'N/A')})</td></tr>
                    <tr><td>Steering Maneuver</td><td><strong>{kin.get('steering_maneuver', 'N/A')}</strong> ({kin.get('initial_heading_deg', 0.0)}°)</td></tr>
                    <tr><td>Projected Traverse Time</td><td>{kin.get('estimated_distance_m', 0.0)} m in {kin.get('estimated_drive_time_s', 0.0)} s</td></tr>
                </table>
            </div>
            <div class="card">
                <strong>AEGIS Autonomous Science Priority Queue</strong>
                <table>
                    <tr><th>Target ID</th><th>Class</th><th>Score</th><th>Access</th></tr>
                    {targets_rows}
                </table>
            </div>
        </div>"""

        wear_section_html = ""
        if wear or solar or cost_fus:
            wear_section_html = f"""
        <h2>5. Wheel Ventifact Mechanics, Diurnal Solar Insolation & Nav2 Costmap Fusion</h2>
        <div class="grid-2">
            <div class="card">
                <strong>Curiosity/Perseverance Aluminum Wheel Fatigue</strong>
                <table>
                    <tr><th>Mechanics Metric</th><th>Telemetry</th></tr>
                    <tr><td>Ventifact Roughness Index</td><td>{wear.get('ventifact_roughness_index', 0.0)} / 100</td></tr>
                    <tr><td>Peak Contact Stress</td><td><strong>{wear.get('peak_contact_stress_mpa', 0.0)} MPa</strong> (Yield Ratio: {wear.get('aluminum_yield_ratio', 0.0)})</td></tr>
                    <tr><td>Puncture Risk / 100m</td><td><strong>{wear.get('puncture_risk_pct', 0.0)}%</strong> ({wear.get('wear_status', 'NOMINAL')})</td></tr>
                    <tr><td>Abrasive Wear Rate</td><td>{wear.get('wear_rate_um_per_km', 0.0)} μm/km (Leg: {wear.get('leg_skin_loss_um', 0.0)} μm)</td></tr>
                    <tr><td>Traction Control Mode</td><td><strong>{wear.get('traction_control_mode', 'N/A')}</strong></td></tr>
                </table>
            </div>
            <div class="card">
                <strong>Diurnal Solar Insolation & Thermal Survival</strong>
                <table>
                    <tr><th>Environmental Factor</th><th>Measurement</th></tr>
                    <tr><td>Local Sol Time (LTST)</td><td>{solar.get('local_sol_hour', 13.5)} hrs (Elev: {solar.get('solar_elevation_deg', 45.0)}°)</td></tr>
                    <tr><td>Dust Optical Depth (τ)</td><td><strong>{solar.get('dust_optical_depth_tau', 0.5)}</strong> ({solar.get('thermal_status', 'NOMINAL')})</td></tr>
                    <tr><td>Ground / Chassis Temp</td><td>{solar.get('ambient_surface_temp_c', -48.0)}°C / {solar.get('effective_chassis_temp_c', -48.0)}°C</td></tr>
                    <tr><td>Total Solar Insolation</td><td>{solar.get('total_solar_flux_w_m2', 0.0)} W/m² (Trans: {solar.get('atmospheric_transmission_pct', 0.0)}%)</td></tr>
                    <tr><td>Battery Survival Heaters</td><td><strong>{solar.get('survival_heater_power_w', 0.0)} W</strong></td></tr>
                </table>
            </div>
        </div>"""

        path_status = path_data.get("status", "N/A") if path_data else "N/A"
        path_cost = path_data.get("total_path_cost", "N/A") if path_data else "N/A"

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>NASA JPL Sol {sol_number} Drive Engineering Report</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
            background-color: #f8fafc;
            color: #0f172a;
            margin: 0;
            padding: 32px;
            line-height: 1.5;
        }}
        .container {{
            max-width: 1050px;
            margin: 0 auto;
            background: #ffffff;
            border: 1px solid #cbd5e1;
            border-radius: 10px;
            padding: 36px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.06);
        }}
        .header {{
            border-bottom: 3px solid #0f172a;
            padding-bottom: 18px;
            margin-bottom: 24px;
            display: flex;
            justify-content: space-between;
            align-items: flex-end;
        }}
        .header h1 {{
            margin: 0;
            font-size: 24px;
            color: #0f172a;
        }}
        .header .sub {{
            font-size: 13px;
            color: #64748b;
            margin-top: 4px;
        }}
        .status-badge {{
            background: {badge_bg};
            color: white;
            font-weight: 800;
            font-size: 12px;
            padding: 6px 14px;
            border-radius: 6px;
            letter-spacing: 0.5px;
        }}
        h2 {{
            font-size: 16px;
            text-transform: uppercase;
            letter-spacing: 0.8px;
            color: #334155;
            border-bottom: 1px solid #e2e8f0;
            padding-bottom: 6px;
            margin-top: 28px;
        }}
        .grid-2 {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 18px;
            margin-top: 14px;
        }}
        .card {{
            background: #f1f5f9;
            border: 1px solid #e2e8f0;
            border-radius: 8px;
            padding: 16px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 13.5px;
            margin-top: 8px;
        }}
        th, td {{
            text-align: left;
            padding: 8px 10px;
            border-bottom: 1px solid #cbd5e1;
        }}
        th {{
            color: #475569;
            font-weight: 700;
        }}
        .img-grid {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 16px;
            margin-top: 14px;
        }}
        .img-box {{
            border: 1px solid #cbd5e1;
            border-radius: 8px;
            padding: 10px;
            background: #ffffff;
            text-align: center;
        }}
        .img-box img {{
            width: 100%;
            border-radius: 4px;
        }}
        .img-caption {{
            font-size: 12px;
            font-weight: 700;
            color: #475569;
            margin-top: 6px;
        }}
        .footer {{
            margin-top: 32px;
            font-size: 11px;
            color: #94a3b8;
            text-align: center;
            border-top: 1px solid #e2e8f0;
            padding-top: 12px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <div>
                <h1>🔴 NASA JPL Mars Rover AutoNav — Sol {sol_number} Field Report</h1>
                <div class="sub">Generated: {now_utc} • Perception Engine: DeepLabV3+ (ResNet-50) • TTA Mode: {telemetry.get('tta_mode', 'Disabled')}</div>
            </div>
            <div class="status-badge">{status_label}</div>
        </div>

        <h2>1. Visual Perception & Path Diagnostics</h2>
        <div class="img-grid">
            <div class="img-box">
                <img src="data:image/jpeg;base64,{b64_orig}" alt="Original Camera Feed">
                <div class="img-caption">A. Raw Navcam Optical Feed</div>
            </div>
            <div class="img-box">
                <img src="data:image/jpeg;base64,{b64_multi}" alt="Multiclass Segmentation">
                <div class="img-caption">B. DeepLabV3+ Multiclass Terrain Map</div>
            </div>
            <div class="img-box">
                <img src="data:image/jpeg;base64,{b64_sand}" alt="Sand Hazard Radar">
                <div class="img-caption">C. Sand Entrapment Hazard Isolation</div>
            </div>
            <div class="img-box">
                <img src="data:image/jpeg;base64,{b64_path}" alt="A* Trajectory">
                <div class="img-caption">D. Planned A* Drive Trajectory ({path_status})</div>
            </div>{extra_img_html}
        </div>

        <h2>2. Terrain Distribution & Traversability Telemetry</h2>
        <div class="grid-2">
            <div class="card">
                <strong>Semantic Terrain Composition</strong>
                <table>
                    <tr><th>Terrain Class</th><th>Coverage (%)</th><th>Navigability</th></tr>
                    <tr><td>Bedrock (Class 1)</td><td>{dist.get('bedrock', 0.0):.2f}%</td><td>Optimal Traction</td></tr>
                    <tr><td>Soil (Class 0)</td><td>{dist.get('soil', 0.0):.2f}%</td><td>Safe Regolith</td></tr>
                    <tr><td>Sand (Class 2)</td><td><strong>{dist.get('sand', 0.0):.2f}%</strong></td><td>Wheel Sinkage Risk</td></tr>
                    <tr><td>Big Rock (Class 3)</td><td><strong>{dist.get('big rock', 0.0):.2f}%</strong></td><td>Collision Obstacle</td></tr>
                </table>
            </div>
            <div class="card">
                <strong>Autonomous Drive Command Summary</strong>
                <table>
                    <tr><th>Metric</th><th>Value</th></tr>
                    <tr><td>Traversability Index</td><td><strong>{telemetry.get('traversability_index', 0.0):.1f} / 100.0</strong></td></tr>
                    <tr><td>Commanded Action</td><td>{telemetry.get('autonomous_drive_action', 'N/A')}</td></tr>
                    <tr><td>Target Drive Velocity</td><td>{telemetry.get('recommended_speed_cm_s', 0.0)} cm/s</td></tr>
                    <tr><td>A* Trajectory Cost</td><td>{path_cost}</td></tr>
                    <tr><td>Inference Latency</td><td>{telemetry.get('inference_latency_ms', 0.0)} ms ({telemetry.get('device', 'cpu')})</td></tr>
                </table>
            </div>
        </div>

        <h2>3. Bekker-Wong Terramechanics, MMRTG Power & Perception Confidence</h2>
        <div class="grid-2">
            <div class="card">
                <strong>Terramechanics & Power Budget (Curiosity Class)</strong>
                <table>
                    <tr><th>Physical Parameter</th><th>Simulated Estimate</th></tr>
                    <tr><td>Mean Wheel Sinkage Depth</td><td>{mech.get('mean_wheel_sinkage_cm', 0.0)} cm</td></tr>
                    <tr><td>Expected Wheel Slip Ratio</td><td>{mech.get('expected_wheel_slip_pct', 0.0)}%</td></tr>
                    <tr><td>Actuator Torque Demand</td><td>{mech.get('actuator_torque_demand_nm', 0.0)} N·m / wheel</td></tr>
                    <tr><td>Specific Energy Cost</td><td>{power.get('energy_cost_wh_per_m', 0.0)} Wh / meter</td></tr>
                    <tr><td>20m Traverse Battery Draw</td><td>{power.get('traverse_20m_energy_wh', 0.0)} Wh ({power.get('battery_impact_pct', 0.0)}%)</td></tr>
                    {slope_row_html}
                </table>
            </div>
            <div class="card">
                <strong>Epistemic Uncertainty Audit (Shannon Entropy)</strong>
                <table>
                    <tr><th>Uncertainty Metric</th><th>Measurement</th></tr>
                    <tr><td>Confidence Status</td><td><strong>{uncert.get('confidence_status', 'N/A')}</strong></td></tr>
                    <tr><td>Mean Prediction Entropy</td><td>{uncert.get('mean_prediction_entropy', 0.0)} (norm [0-1])</td></tr>
                    <tr><td>P90 Entropy Bound</td><td>{uncert.get('p90_prediction_entropy', 0.0)}</td></tr>
                    <tr><td>Ambiguous Pixel Ratio</td><td>{uncert.get('high_uncertainty_pixel_pct', 0.0)}%</td></tr>
                    <tr><td>Operational Guidance</td><td>{uncert.get('perception_recommendation', 'N/A')}</td></tr>
                    {sensor_row_html}
                    {cal_row_html}
                </table>
            </div>
        </div>
        {advanced_section_html}
        {wear_section_html}

        <div class="footer">
            Automated Robotics Perception Report • NASA AI4Mars Terrain Segmentation Framework • DeepLabV3+ ResNet-50
        </div>
    </div>
</body>
</html>"""
        return html
