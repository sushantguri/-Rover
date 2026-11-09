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
        sol_number: int = 4118
    ) -> str:
        """
        Compiles complete telemetry and imagery into a polished printable HTML report.
        """
        now_utc = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        dist = telemetry.get("terrain_distribution_pct", {})
        mech = telemetry.get("terramechanics", {})
        uncert = telemetry.get("uncertainty", {})
        power = telemetry.get("power_budget", {})

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
            </div>
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
                </table>
            </div>
        </div>

        <div class="footer">
            Automated Robotics Perception Report • NASA AI4Mars Terrain Segmentation Framework • DeepLabV3+ ResNet-50
        </div>
    </div>
</body>
</html>"""
        return html
