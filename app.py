"""
🔴 NASA JPL Mars Rover Terrain Segmentation & Autonomous Hazard Detection
Interactive Mission Control Telemetry Dashboard powered by DeepLabV3+ (ResNet-50).
"""
import os
import glob
import time
import datetime
import tempfile
import numpy as np
from PIL import Image
import torch
import gradio as gr

from src.model import get_deeplabv3plus_model
from src.dataset import CLASSES
from src.visualization import create_overlay
from src.path_planner import MarsPathPlanner
from src.terramechanics import RoverTerramechanicsEngine
from src.uncertainty import UncertaintyEstimator
from src.tta import TestTimeAugmenter
from src.power_budget import RoverPowerBudgetEstimator
from src.report_generator import MissionReportGenerator
from src.geometry_risk import TerrainGeometryAnalyzer
from src.kinematics import RoverKinematicPlanner
from src.science_targets import AEGISTargetSelector
from src.sensor_watchdog import OpticalSensorWatchdog

SAMPLE_DIR = "sample_images"
curated_sample_paths = [
    os.path.join(SAMPLE_DIR, "test_sandy_terrain.jpg"),
    os.path.join(SAMPLE_DIR, "test_rocky_terrain.jpg"),
    os.path.join(SAMPLE_DIR, "test_mixed_terrain.jpg"),
    os.path.join(SAMPLE_DIR, "test_bedrock_dominant.jpg"),
    os.path.join(SAMPLE_DIR, "test1_high_sand_with_rocks_31.jpg"),
    os.path.join(SAMPLE_DIR, "test1_soil_dominant_294.jpg"),
    os.path.join(SAMPLE_DIR, "curiosity_navcam_sample_1.jpg"),
    os.path.join(SAMPLE_DIR, "train_sample_9.jpg"),
]
# Filter to existing files
sample_files = [p for p in curated_sample_paths if os.path.exists(p)]
if not sample_files:
    sample_files = sorted(glob.glob(os.path.join(SAMPLE_DIR, "*.jpg")))

device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
device_name = "Apple Silicon MPS (GPU)" if device == "mps" else ("NVIDIA CUDA (GPU)" if device == "cuda" else "CPU")
_model = None

# Custom Robot Costmap Palette:
# Bedrock: Cyan/Blue (Optimal) | Soil: Green (Safe) | Sand: Orange (High Slip Hazard) | Big Rock: Red (Collision)
COST_PALETTE = np.array([
    [76, 175, 80],     # 0: Soil - Soft Green
    [33, 150, 243],    # 1: Bedrock - Cyan/Blue
    [255, 152, 0],     # 2: Sand - Glowing Orange
    [239, 68, 68],     # 3: Big Rock - Crimson Red
], dtype=np.uint8)


def load_inference_model():
    global _model
    if _model is not None:
        return _model
    
    ckpt_path = "checkpoints/best_deeplabv3plus.pth"
    if not os.path.exists(ckpt_path):
        print(f"Checkpoint not found at {ckpt_path}; loading base DeepLabV3+...")
        model = get_deeplabv3plus_model(num_classes=4).to(device)
    else:
        print(f"Loading best model weights from {ckpt_path} onto {device}...")
        ckpt = torch.load(ckpt_path, map_location=device)
        model = get_deeplabv3plus_model(num_classes=4).to(device)
        model.load_state_dict(ckpt["model_state_dict"])
    
    model.eval()
    _model = model
    return _model


def generate_costmap_overlay(orig_np, pred_np, alpha=0.6):
    """
    Renders an autonomous traversability costmap used by robot path planners (A*/D*).
    """
    colored_cost = COST_PALETTE[np.clip(pred_np, 0, 3)]
    blended = (orig_np * (1.0 - alpha) + colored_cost * alpha).astype(np.uint8)
    return blended


def make_side_by_side_comparison(orig_np, overlay_np):
    """
    Stitches original camera frame and segmented overlay horizontally with a high-contrast divider.
    """
    h, w, _ = orig_np.shape
    divider = np.zeros((h, 6, 3), dtype=np.uint8)
    divider[:, :, 0] = 255  # Red accent line
    divider[:, :, 1] = 75
    divider[:, :, 2] = 62
    return np.hstack([orig_np, divider, overlay_np])


def segment_mars_terrain(input_image, opacity=0.6, img_size_str="512", sand_threshold=25.0, tta_mode="disabled", sol_number=4118):
    """
    High-fidelity inference pipeline with multi-view outputs, TTA ensemble,
    Bekker-Wong terramechanics, MMRTG power budget telemetry, and HTML field reports.
    """
    if input_image is None:
        empty_msg = """
        <div class="empty-state-box">
            <div style="font-size: 32px; margin-bottom: 8px;">🛰️</div>
            <div style="font-weight: 600; font-size: 16px; color: #94a3b8;">NO CAMERA FEED DETECTED</div>
            <div style="font-size: 13px; color: #64748b; margin-top: 4px;">Upload a Mars rover image or select a mission sample to initiate terrain telemetry.</div>
        </div>
        """
        return None, None, None, None, None, None, None, None, empty_msg, "", None

    t_start = time.time()
    model = load_inference_model()
    
    img_size = int(img_size_str)
    orig_img = Image.fromarray(input_image).convert("RGB")
    orig_w, orig_h = orig_img.size
    orig_np = np.array(orig_img)

    # 0. Optical Sensor Health & Atmospheric Fault Diagnostics (Watchdog)
    watchdog = OpticalSensorWatchdog()
    sensor_health = watchdog.inspect_frame(orig_np)
    if tta_mode == "auto (watchdog)":
        run_tta = sensor_health["recommend_tta_boost"]
        effective_tta_mode = "fast" if run_tta else "disabled"
    else:
        run_tta = tta_mode in ("fast", "full")
        effective_tta_mode = tta_mode

    # Preprocessing
    resized_img = orig_img.resize((img_size, img_size), Image.BILINEAR)
    img_arr = np.array(resized_img, dtype=np.float32) / 255.0
    
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)
    normalized = (img_arr - mean) / std
    
    img_tensor = torch.from_numpy(normalized).permute(2, 0, 1).unsqueeze(0).to(device)
    
    # Model Inference (Single-Pass or Multi-View Test-Time Augmentation)
    with torch.no_grad():
        if run_tta:
            tta_engine = TestTimeAugmenter(model, mode=effective_tta_mode)
            logits = tta_engine.predict_pseudo_logits(img_tensor)
        else:
            logits = model(img_tensor)
        pred_map = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
        
    # Resize prediction back to original dimensions using NEAREST interpolation
    pred_full = Image.fromarray(pred_map).resize((orig_w, orig_h), Image.NEAREST)
    pred_np = np.array(pred_full, dtype=np.int64)
    
    t_end = time.time()
    latency_ms = (t_end - t_start) * 1000.0
    
    # 1. Standard Multiclass Terrain Map
    overlay_standard = create_overlay(orig_np, pred_np, alpha=opacity, highlight_sand=False)
    
    # 2. Sand Hazard Isolation Radar (dims everything else, highlights sand)
    sand_hazard_view = create_overlay(orig_np, pred_np, alpha=0.80, highlight_sand=True)
    
    # 3. Robot Traversability Costmap
    costmap_view = generate_costmap_overlay(orig_np, pred_np, alpha=opacity)
    
    # 4. Side-by-Side Dual View
    side_by_side_view = make_side_by_side_comparison(orig_np, overlay_standard)
    
    # 5. Autonomous A* Trajectory Path
    planner = MarsPathPlanner(obstacle_inflation_radius=2)
    path_data = planner.plan_path(pred_np, planning_resolution=128)
    trajectory_view = planner.render_path_overlay(overlay_standard, path_data)

    # 6. Epistemic Uncertainty & Prediction Entropy
    uncertainty_engine = UncertaintyEstimator()
    entropy_map = uncertainty_engine.compute_entropy_map(logits)
    uncertainty_view = uncertainty_engine.generate_uncertainty_heatmap(orig_np, entropy_map, alpha=opacity)
    uncertainty_telemetry = uncertainty_engine.evaluate_uncertainty_telemetry(entropy_map)

    # Metrics Computation
    total_pixels = pred_np.size
    class_pcts = {}
    class_pixel_counts = {}
    for c_id, c_name in enumerate(CLASSES):
        px = int((pred_np == c_id).sum())
        class_pixel_counts[c_name] = px
        class_pcts[c_name] = (px / total_pixels) * 100.0

    # Terramechanics Physics Modeling
    mechanics_engine = RoverTerramechanicsEngine()
    mechanics = mechanics_engine.evaluate_terrain_mechanics(class_pcts)

    sand_pct = class_pcts["sand"]
    rock_pct = class_pcts["big rock"]
    bedrock_pct = class_pcts["bedrock"]
    soil_pct = class_pcts["soil"]
    
    # Traversability Index calculation: Bedrock=100, Soil=85, Sand=15, Big Rock=0
    traversability = (bedrock_pct * 1.0 + soil_pct * 0.85 + sand_pct * 0.15 + rock_pct * 0.0)
    traversability = max(0.0, min(100.0, traversability))
    
    # Autonomous Driving Decision Engine
    if sand_pct >= float(sand_threshold):
        safety_status_code = "CRITICAL_HAZARD"
        drive_action_code = "EMERGENCY_HALT_AND_REPLAN"
        status_title = "CRITICAL WHEEL ENTRAPMENT HAZARD"
        status_desc = f"Dense sand dunes cover <strong>{sand_pct:.1f}%</strong> of forward frame (safety threshold: {sand_threshold}%). High risk of catastrophic wheel sinkage matching Spirit rover entrapment (Troy, 2009). Autonomous traverse halted; route replan triggered."
        status_theme = "critical"
        status_color = "#ef4444"
        badge_text = "🔴 EMERGENCY HALT & REPLAN"
        speed_rec = "0.0 cm/s (HALTED)"
        recommended_speed_val = 0.0
        slip_risk = "HIGH RISK (85% - 95%)"
        steering_mode = "REVERSE & ARC EVASION"
    elif sand_pct > 8.0 or rock_pct > 2.0:
        safety_status_code = "CAUTION_ADVISORY"
        drive_action_code = "SLOW_CREEP_AND_AVOID"
        status_title = "CAUTION: MODERATE TERRAIN HAZARDS"
        status_desc = f"Scattered sand drifts (<strong>{sand_pct:.1f}%</strong>) or boulder obstacles (<strong>{rock_pct:.1f}%</strong>) detected. Autonomous speed throttled. Engage differential-drive slow creep and activate wheel slip feedback sensors."
        status_theme = "caution"
        status_color = "#f59e0b"
        badge_text = "🟡 CAUTION ADVISORY (SLOW CREEP)"
        speed_rec = "1.8 cm/s (THROTTLED)"
        recommended_speed_val = 1.8
        slip_risk = "MODERATE (25% - 40%)"
        steering_mode = "OBSTACLE DETOUR ARC"
    else:
        safety_status_code = "NOMINAL_SAFE"
        drive_action_code = "PROCEED_NOMINAL_SPEED"
        status_title = "TRAVERSE PERMITTED: NOMINAL DRIVE"
        status_desc = f"Firm rock pavement (<strong>{bedrock_pct:.1f}%</strong>) and compact regolith (<strong>{soil_pct:.1f}%</strong>) dominate the forward terrain. Forward path clear for nominal waypoint traverse."
        status_theme = "nominal"
        status_color = "#10b981"
        badge_text = "🟢 DRIVE PERMITTED (NOMINAL)"
        speed_rec = "4.5 cm/s (NOMINAL CRUISE)"
        recommended_speed_val = 4.5
        slip_risk = "LOW (< 5%)"
        steering_mode = "NOMINAL WAYPOINT TRAVERSE"

    # MMRTG Power Budget & Mobility Energy Consumption
    power_estimator = RoverPowerBudgetEstimator()
    power_budget = power_estimator.estimate_traverse_energy(
        terramechanics_data=mechanics,
        commanded_speed_cm_s=recommended_speed_val,
        planned_distance_m=20.0
    )

    # 7. Horizon Range-Zone & Boulder Geometry High-Centering Radar
    geometry_analyzer = TerrainGeometryAnalyzer()
    geometry_risk = geometry_analyzer.analyze(pred_np)
    zone_radar_view = geometry_analyzer.render_zone_radar_overlay(orig_np, pred_np, geometry_risk, alpha=opacity)

    # 8. Steering Kinematics & Trajectory Smoothing Profile
    kin_planner = RoverKinematicPlanner()
    kinematics = kin_planner.profile_trajectory(
        path_data=path_data,
        image_shape=pred_np.shape,
        effective_speed_cm_s=power_budget["effective_ground_speed_cm_s"],
        pred_mask=pred_np
    )

    # 9. AEGIS Autonomous Geological Science Target Prioritizer
    science_selector = AEGISTargetSelector(top_k=3)
    science_targets = science_selector.select_targets(pred_np, entropy_map=entropy_map, top_k=3)
    science_targets_view = science_selector.render_target_reticles(orig_np, science_targets)

    targets_list_html = "".join(
        f"<div style='display:flex;justify-content:space-between;padding:4px 0;border-bottom:1px solid rgba(255,255,255,0.06);font-size:11.5px;'>"
        f"<span style='color:#e2e8f0;font-weight:700;'>#{t['rank']} {t['category'].replace('_', ' ')}</span>"
        f"<span style='color:#38bdf8;font-family:JetBrains Mono,monospace;'>{t['priority_score']} pts • {t['approach_safety'].replace('_', ' ')}</span>"
        f"</div>"
        for t in science_targets.get("targets", [])
    ) or "<div style='font-size:11.5px;color:#64748b;'>No exposed bedrock/rock targets locked in current optical frame.</div>"

    # HTML Telemetry Dashboard
    now_utc = datetime.datetime.now(datetime.timezone.utc).strftime("%H:%M:%S UTC")
    
    telemetry_html = f"""
    <div class="telemetry-dashboard">
        <!-- Safety Alert Banner -->
        <div class="status-banner banner-{status_theme}">
            <div class="banner-header">
                <span class="badge-pill">{badge_text}</span>
                <span class="timestamp-tag">AUTONAV CORE • {now_utc}</span>
            </div>
            <div class="banner-title">{status_title}</div>
            <div class="banner-desc">{status_desc}</div>
        </div>

        <!-- Traversability Score Meter -->
        <div class="telemetry-card score-card">
            <div class="card-title-row">
                <span class="card-label">TRAVERSABILITY CONFIDENCE INDEX</span>
                <span class="score-number" style="color: {status_color};">{traversability:.1f}<span class="score-denom">/100</span></span>
            </div>
            <div class="progress-bar-bg">
                <div class="progress-bar-fill" style="width: {traversability}%; background: linear-gradient(90deg, #ef4444, #f59e0b, #10b981);"></div>
            </div>
            <div class="card-caption">Higher index indicates firmer rock/soil and lower wheel resistance.</div>
        </div>

        <!-- 4 Terrain Breakdown Grid -->
        <div class="terrain-grid">
            <!-- Soil -->
            <div class="terrain-tile">
                <div class="tile-header">
                    <span class="tile-indicator indicator-soil"></span>
                    <span class="tile-name">SOIL (REGOLITH)</span>
                    <span class="tile-pct">{soil_pct:.1f}%</span>
                </div>
                <div class="tile-bar-bg">
                    <div class="tile-bar-fill fill-soil" style="width: {soil_pct}%;"></div>
                </div>
                <div class="tile-sub">Safe compact dirt • Low slip</div>
            </div>

            <!-- Bedrock -->
            <div class="terrain-tile">
                <div class="tile-header">
                    <span class="tile-indicator indicator-bedrock"></span>
                    <span class="tile-name">BEDROCK (PAVEMENT)</span>
                    <span class="tile-pct">{bedrock_pct:.1f}%</span>
                </div>
                <div class="tile-bar-bg">
                    <div class="tile-bar-fill fill-bedrock" style="width: {bedrock_pct}%;"></div>
                </div>
                <div class="tile-sub">Exposed rock • Optimal traction</div>
            </div>

            <!-- Sand -->
            <div class="terrain-tile tile-danger">
                <div class="tile-header">
                    <span class="tile-indicator indicator-sand"></span>
                    <span class="tile-name">SAND (WHEEL SINK)</span>
                    <span class="tile-pct" style="color: #ff9800;">{sand_pct:.1f}%</span>
                </div>
                <div class="tile-bar-bg">
                    <div class="tile-bar-fill fill-sand" style="width: {sand_pct}%;"></div>
                </div>
                <div class="tile-sub" style="color: #f59e0b;">🚨 Critical wheel entrapment risk</div>
            </div>

            <!-- Big Rock -->
            <div class="terrain-tile tile-danger">
                <div class="tile-header">
                    <span class="tile-indicator indicator-rock"></span>
                    <span class="tile-name">BIG ROCK (&gt;30cm)</span>
                    <span class="tile-pct" style="color: #ef4444;">{rock_pct:.1f}%</span>
                </div>
                <div class="tile-bar-bg">
                    <div class="tile-bar-fill fill-rock" style="width: {rock_pct}%;"></div>
                </div>
                <div class="tile-sub" style="color: #ef4444;">⚠️ Chassis collision obstacle</div>
            </div>
        </div>

        <!-- Planetary Terramechanics Physics HUD -->
        <div class="telemetry-card">
            <div class="hud-header">BEKKER-WONG PLANETARY TERRAMECHANICS (CURIOSITY CLASS)</div>
            <div class="hud-grid">
                <div class="hud-item">
                    <div class="hud-label">ESTIMATED WHEEL SINKAGE</div>
                    <div class="hud-val" style="color: {'#ef4444' if mechanics['mean_wheel_sinkage_cm'] > 4.5 else '#f59e0b' if mechanics['mean_wheel_sinkage_cm'] > 2.0 else '#10b981'};">
                        {mechanics['mean_wheel_sinkage_cm']:.1f} cm
                    </div>
                </div>
                <div class="hud-item">
                    <div class="hud-label">WHEEL SLIP ESTIMATE</div>
                    <div class="hud-val" style="color: {'#ef4444' if mechanics['expected_wheel_slip_pct'] > 40.0 else '#f59e0b' if mechanics['expected_wheel_slip_pct'] > 15.0 else '#10b981'};">
                        {mechanics['expected_wheel_slip_pct']:.1f}%
                    </div>
                </div>
                <div class="hud-item">
                    <div class="hud-label">MOTOR TORQUE DEMAND</div>
                    <div class="hud-val">{mechanics['actuator_torque_demand_nm']:.1f} N·m / wheel</div>
                </div>
                <div class="hud-item">
                    <div class="hud-label">PERCEPTION CONFIDENCE</div>
                    <div class="hud-val" style="color: #38bdf8;">{100.0 - uncertainty_telemetry['high_uncertainty_pixel_pct']:.1f}% ({uncertainty_telemetry['confidence_status'].split('_')[0]})</div>
                </div>
            </div>
        </div>

        <!-- Robot Drive Recommendations HUD -->
        <div class="telemetry-card">
            <div class="hud-header">RECOMMENDED AUTONOMOUS DRIVE PARAMETERS</div>
            <div class="hud-grid">
                <div class="hud-item">
                    <div class="hud-label">RECOMMENDED SPEED</div>
                    <div class="hud-val">{speed_rec}</div>
                </div>
                <div class="hud-item">
                    <div class="hud-label">WHEEL SLIP ESTIMATE</div>
                    <div class="hud-val" style="color: {status_color};">{slip_risk}</div>
                </div>
                <div class="hud-item">
                    <div class="hud-label">STEERING ACTION</div>
                    <div class="hud-val">{steering_mode}</div>
                </div>
                <div class="hud-item">
                    <div class="hud-label">INFERENCE LATENCY</div>
                    <div class="hud-val">{latency_ms:.1f} ms ({device_name.split()[0]}{' • TTA:' + effective_tta_mode.upper() if run_tta else ''})</div>
                </div>
            </div>
        </div>

        <!-- MMRTG Power Budget & Mobility Energy Telemetry -->
        <div class="telemetry-card">
            <div class="hud-header">MMRTG POWER BUDGET & MOBILITY ENERGY ESTIMATE (SOL {int(sol_number)})</div>
            <div class="hud-grid">
                <div class="hud-item">
                    <div class="hud-label">SPECIFIC ENERGY COST</div>
                    <div class="hud-val" style="color: {'#ef4444' if power_budget['energy_cost_wh_per_m'] > 3.0 else '#f59e0b' if power_budget['energy_cost_wh_per_m'] > 1.2 else '#10b981'};">
                        {power_budget['energy_cost_wh_per_m']:.2f} Wh/m
                    </div>
                </div>
                <div class="hud-item">
                    <div class="hud-label">20M TRAVERSE DRAW</div>
                    <div class="hud-val">{power_budget['traverse_20m_energy_wh']:.1f} Wh ({power_budget['battery_impact_pct']:.2f}%)</div>
                </div>
                <div class="hud-item">
                    <div class="hud-label">EFFECTIVE GROUND SPEED</div>
                    <div class="hud-val">{power_budget['effective_ground_speed_cm_s']:.2f} cm/s</div>
                </div>
                <div class="hud-item">
                    <div class="hud-label">POWER SUBSYSTEM STATE</div>
                    <div class="hud-val" style="color: {'#ef4444' if power_budget['thermal_motor_warning'] else '#f59e0b' if power_budget['power_status'] == 'NET_BATTERY_DISCHARGE' else '#10b981' if power_budget['power_status'] == 'RTG_SUSTAINABLE_DRIVE' else '#94a3b8'};">
                        {power_budget['power_status'].replace('_', ' ')}
                    </div>
                </div>
            </div>
        </div>

        <!-- Range-Zone Geometry & Autonomous Steering Kinematics -->
        <div class="telemetry-card">
            <div class="hud-header">HORIZON ZONES, BOULDER RISK & STEERING KINEMATICS</div>
            <div class="hud-grid">
                <div class="hud-item">
                    <div class="hud-label">NEAR-FIELD (0-5M) SAND / ROCK</div>
                    <div class="hud-val">{geometry_risk['nearfield_sand_pct']:.1f}% / {geometry_risk['nearfield_big_rock_pct']:.1f}%</div>
                </div>
                <div class="hud-item">
                    <div class="hud-label">BELLY HIGH-CENTERING RISK</div>
                    <div class="hud-val" style="color: {'#ef4444' if geometry_risk['high_centering_risk'] == 'CRITICAL' else '#f59e0b' if geometry_risk['high_centering_risk'] == 'ELEVATED' else '#10b981'};">
                        {geometry_risk['high_centering_risk']} ({geometry_risk['boulder_cluster_count']} clusters)
                    </div>
                </div>
                <div class="hud-item">
                    <div class="hud-label">STEERING ARC MANEUVER</div>
                    <div class="hud-val">{kinematics['steering_maneuver'].replace('_', ' ')} ({kinematics['initial_heading_deg']:+.1f}°)</div>
                </div>
                <div class="hud-item">
                    <div class="hud-label">PROJECTED TRAVERSE ARC</div>
                    <div class="hud-val">{kinematics['estimated_distance_m']:.1f} m / {kinematics['estimated_drive_time_s']:.0f} s</div>
                </div>
            </div>
        </div>

        <!-- AEGIS Autonomous Geological Science Targets -->
        <div class="telemetry-card">
            <div class="hud-header">AEGIS AUTONOMOUS GEOLOGICAL TARGET QUEUE ({science_targets['aegis_status'].replace('_', ' ')})</div>
            {targets_list_html}
        </div>

        <!-- Optical Sensor Health & Watchdog Diagnostics -->
        <div class="telemetry-card">
            <div class="hud-header">OPTICAL SENSOR WATCHDOG & ATMOSPHERIC DIAGNOSTICS</div>
            <div class="hud-grid">
                <div class="hud-item">
                    <div class="hud-label">SENSOR HEALTH STATE</div>
                    <div class="hud-val" style="color: {'#ef4444' if 'FAULT' in sensor_health['sensor_health_status'] else '#f59e0b' if 'WARNING' in sensor_health['sensor_health_status'] or 'ADVISORY' in sensor_health['sensor_health_status'] else '#10b981'};">
                        {sensor_health['sensor_health_status'].replace('_', ' ')}
                    </div>
                </div>
                <div class="hud-item">
                    <div class="hud-label">OPTICAL TRUST / LUMINANCE</div>
                    <div class="hud-val">{sensor_health['optical_trust_score']:.0f}% (Lum {sensor_health['mean_luminance']:.0f})</div>
                </div>
                <div class="hud-item">
                    <div class="hud-label">GLARE / SHADOW CLIPPING</div>
                    <div class="hud-val">{sensor_health['overexposed_pct']:.1f}% / {sensor_health['underexposed_pct']:.1f}%</div>
                </div>
                <div class="hud-item">
                    <div class="hud-label">SHARPNESS GRADIENT</div>
                    <div class="hud-val">{sensor_health['sharpness_gradient']:.1f} (std {sensor_health['contrast_std']:.1f})</div>
                </div>
            </div>
        </div>
    </div>
    """

    # Compile Downloadable Self-Contained Sol Drive Engineering Field Report
    report_telemetry = {
        "tta_mode": effective_tta_mode,
        "inference_latency_ms": round(latency_ms, 2),
        "device": device,
        "sensor_health": sensor_health,
        "safety_status": safety_status_code,
        "traversability_index": round(traversability, 1),
        "autonomous_drive_action": drive_action_code,
        "recommended_speed_cm_s": recommended_speed_val,
        "terrain_distribution_pct": class_pcts,
        "terramechanics": mechanics,
        "power_budget": power_budget,
        "uncertainty": uncertainty_telemetry,
        "geometry_risk": geometry_risk,
        "kinematics": kinematics,
        "science_targets": science_targets,
    }
    report_gen = MissionReportGenerator()
    report_html = report_gen.generate_html_report(
        telemetry=report_telemetry,
        orig_img_np=orig_np,
        overlay_multiclass=overlay_standard,
        overlay_sand=sand_hazard_view,
        overlay_path=trajectory_view,
        path_data=path_data,
        sol_number=int(sol_number),
        overlay_zone_radar=zone_radar_view,
        overlay_science=science_targets_view,
    )
    report_path = os.path.join(tempfile.gettempdir(), f"nasa_sol_{int(sol_number)}_field_report.html")
    with open(report_path, "w", encoding="utf-8") as rf:
        rf.write(report_html)

    # Monospace Robot Flight Log
    watchdog_flags = ",".join(sensor_health["active_fault_flags"]) or "NONE"
    flight_log = f"""[UTC {now_utc}] [AUTONAV-AI] Ingestion: {orig_w}x{orig_h} frame -> Resized to {img_size}x{img_size} (TTA: {effective_tta_mode})
[UTC {now_utc}] [WATCHDOG-OPT] Status={sensor_health['sensor_health_status']} | Trust={sensor_health['optical_trust_score']:.0f}% | Flags={watchdog_flags}
[UTC {now_utc}] [INFERENCE] DeepLabV3+ ResNet-50 executed in {latency_ms:.1f}ms on {device_name}
[UTC {now_utc}] [TELEMETRY] Terrain: Bedrock={bedrock_pct:.1f}%, Soil={soil_pct:.1f}%, Sand={sand_pct:.1f}%, BigRock={rock_pct:.1f}%
[UTC {now_utc}] [GEOMETRY] NearField Sand={geometry_risk['nearfield_sand_pct']:.1f}% | Clusters={geometry_risk['boulder_cluster_count']} | HighCenterRisk={geometry_risk['high_centering_risk']}
[UTC {now_utc}] [KINEMATICS] Maneuver={kinematics['steering_maneuver']} ({kinematics['initial_heading_deg']:+.1f} deg) | Arc={kinematics['estimated_distance_m']:.1f}m
[UTC {now_utc}] [AEGIS-SCI] Status={science_targets['aegis_status']} | Targets Locked={science_targets['target_count']}
[UTC {now_utc}] [POWER-MMRTG] {power_budget['power_status']} | 20m Energy={power_budget['traverse_20m_energy_wh']:.1f}Wh | EffSpeed={power_budget['effective_ground_speed_cm_s']:.2f}cm/s
[UTC {now_utc}] [AUTONAV-PLANNER] Drive Command: {speed_rec} | Mode: {steering_mode} | Sol {int(sol_number)} Report Ready."""

    return overlay_standard, sand_hazard_view, costmap_view, side_by_side_view, trajectory_view, uncertainty_view, zone_radar_view, science_targets_view, telemetry_html, flight_log, report_path


# High-Tech NASA JPL Dark Glassmorphism CSS
CUSTOM_CSS = """
/* Futuristic Space Control Theme */
body, .gradio-container {
    background-color: #080b12 !important;
    color: #e2e8f0 !important;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
}

/* Header Styling */
.mission-header {
    background: linear-gradient(135deg, rgba(15, 23, 42, 0.95), rgba(30, 41, 59, 0.75));
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 14px;
    padding: 24px 28px;
    margin-bottom: 24px;
    box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.6);
    position: relative;
    overflow: hidden;
}
.mission-header::after {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    width: 4px;
    height: 100%;
    background: linear-gradient(180deg, #ef4444, #f97316);
}
.mission-top-badges {
    display: flex;
    gap: 12px;
    align-items: center;
    margin-bottom: 10px;
    flex-wrap: wrap;
}
.mission-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(239, 68, 68, 0.15);
    border: 1px solid rgba(239, 68, 68, 0.4);
    color: #fca5a5;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.8px;
    padding: 4px 10px;
    border-radius: 6px;
    text-transform: uppercase;
}
.pulse-dot {
    width: 7px;
    height: 7px;
    background-color: #10b981;
    border-radius: 50%;
    box-shadow: 0 0 8px #10b981;
    display: inline-block;
}
.mission-title {
    font-size: 26px;
    font-weight: 800;
    letter-spacing: -0.5px;
    margin: 0 0 6px 0;
    background: linear-gradient(90deg, #ffffff, #cbd5e1);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.mission-subtitle {
    font-size: 13.5px;
    color: #94a3b8;
    line-height: 1.5;
    margin: 0;
}

/* Control Panels & Accordion */
.panel-card {
    background: #0f172a !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    border-radius: 12px !important;
    padding: 16px !important;
    margin-bottom: 16px !important;
}

/* Primary Action Button */
.btn-analyze {
    background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%) !important;
    border: 1px solid rgba(255, 255, 255, 0.2) !important;
    color: white !important;
    font-weight: 700 !important;
    letter-spacing: 0.5px !important;
    border-radius: 8px !important;
    padding: 12px 20px !important;
    box-shadow: 0 4px 20px rgba(239, 68, 68, 0.35) !important;
    transition: all 0.2s ease !important;
}
.btn-analyze:hover {
    box-shadow: 0 6px 25px rgba(239, 68, 68, 0.55) !important;
    transform: translateY(-1px) !important;
}

/* Empty State Box */
.empty-state-box {
    text-align: center;
    padding: 40px 20px;
    background: rgba(15, 23, 42, 0.6);
    border: 1px dashed rgba(255, 255, 255, 0.15);
    border-radius: 12px;
}

/* Status Alert Banners */
.status-banner {
    border-radius: 10px;
    padding: 16px 20px;
    margin-bottom: 16px;
    backdrop-filter: blur(8px);
}
.banner-critical {
    background: rgba(239, 68, 68, 0.12);
    border: 1px solid rgba(239, 68, 68, 0.5);
    border-left: 5px solid #ef4444;
}
.banner-caution {
    background: rgba(245, 158, 11, 0.12);
    border: 1px solid rgba(245, 158, 11, 0.5);
    border-left: 5px solid #f59e0b;
}
.banner-nominal {
    background: rgba(16, 185, 129, 0.12);
    border: 1px solid rgba(16, 185, 129, 0.5);
    border-left: 5px solid #10b981;
}
.banner-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 8px;
}
.badge-pill {
    font-size: 11px;
    font-weight: 800;
    letter-spacing: 0.5px;
    padding: 3px 8px;
    border-radius: 4px;
    background: rgba(0, 0, 0, 0.4);
}
.timestamp-tag {
    font-size: 11px;
    font-family: 'JetBrains Mono', monospace;
    color: #94a3b8;
}
.banner-title {
    font-size: 16px;
    font-weight: 800;
    margin-bottom: 6px;
    color: #ffffff;
}
.banner-desc {
    font-size: 13px;
    color: #cbd5e1;
    line-height: 1.5;
}

/* Telemetry Cards */
.telemetry-card {
    background: rgba(15, 23, 42, 0.7);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 10px;
    padding: 16px;
    margin-bottom: 14px;
}
.card-title-row {
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    margin-bottom: 8px;
}
.card-label {
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.8px;
    color: #94a3b8;
    text-transform: uppercase;
}
.score-number {
    font-size: 24px;
    font-weight: 800;
    font-family: 'JetBrains Mono', monospace;
}
.score-denom {
    font-size: 13px;
    color: #64748b;
    font-weight: 500;
}
.progress-bar-bg {
    height: 8px;
    background: #1e293b;
    border-radius: 4px;
    overflow: hidden;
    margin-bottom: 8px;
}
.progress-bar-fill {
    height: 100%;
    border-radius: 4px;
    transition: width 0.4s ease;
}
.card-caption {
    font-size: 11.5px;
    color: #64748b;
}

/* Terrain Grid */
.terrain-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 10px;
    margin-bottom: 14px;
}
.terrain-tile {
    background: rgba(15, 23, 42, 0.8);
    border: 1px solid rgba(255, 255, 255, 0.07);
    border-radius: 8px;
    padding: 12px;
}
.tile-header {
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 6px;
}
.tile-indicator {
    width: 8px;
    height: 8px;
    border-radius: 2px;
}
.indicator-soil { background-color: #f1c40f; }
.indicator-bedrock { background-color: #00b4d8; }
.indicator-sand { background-color: #ff7b00; }
.indicator-rock { background-color: #ff3860; }

.tile-name {
    font-size: 10.5px;
    font-weight: 700;
    color: #cbd5e1;
    flex-grow: 1;
}
.tile-pct {
    font-size: 14px;
    font-weight: 800;
    font-family: 'JetBrains Mono', monospace;
    color: #ffffff;
}
.tile-bar-bg {
    height: 5px;
    background: #1e293b;
    border-radius: 3px;
    overflow: hidden;
    margin-bottom: 6px;
}
.tile-bar-fill {
    height: 100%;
    border-radius: 3px;
}
.fill-soil { background: #f1c40f; }
.fill-bedrock { background: #00b4d8; }
.fill-sand { background: #ff7b00; }
.fill-rock { background: #ff3860; }
.tile-sub {
    font-size: 10.5px;
    color: #64748b;
}

/* HUD Parameters Grid */
.hud-header {
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.8px;
    color: #94a3b8;
    margin-bottom: 12px;
}
.hud-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 12px;
}
.hud-item {
    background: rgba(0, 0, 0, 0.3);
    border-radius: 6px;
    padding: 8px 10px;
}
.hud-label {
    font-size: 9.5px;
    font-weight: 700;
    color: #64748b;
    letter-spacing: 0.5px;
    margin-bottom: 4px;
}
.hud-val {
    font-size: 12.5px;
    font-weight: 700;
    color: #f1f5f9;
    font-family: 'JetBrains Mono', monospace;
}

/* Monospace Flight Terminal Box */
.terminal-card {
    background: #020617 !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    border-radius: 8px !important;
    padding: 12px !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 11.5px !important;
    color: #38bdf8 !important;
}
"""

with gr.Blocks(title="NASA JPL Mars Rover Terrain Segmentation") as demo:
    # Inject Custom Space-Grade Styles
    gr.HTML(f"<style>{CUSTOM_CSS}</style>")
    
    # Mission Control Header
    gr.HTML(f"""
    <div class="mission-header">
        <div class="mission-top-badges">
            <span class="mission-badge"><span class="pulse-dot"></span> AUTONAV SYSTEM ONLINE</span>
            <span class="mission-badge" style="border-color: rgba(56, 189, 248, 0.4); color: #7dd3fc; background: rgba(56, 189, 248, 0.15);">CURIOSITY / PERSEVERANCE FLIGHT TELEMETRY</span>
            <span class="mission-badge" style="border-color: rgba(251, 146, 60, 0.4); color: #fdba74; background: rgba(251, 146, 60, 0.15);">DEEPLABV3+ • RESNET-50</span>
            <span class="mission-badge" style="border-color: rgba(168, 85, 247, 0.4); color: #d8b4fe; background: rgba(168, 85, 247, 0.15);">COMPUTE: {device_name}</span>
        </div>
        <h1 class="mission-title">🔴 NASA JPL Mars Rover Terrain & Hazard Segmentation</h1>
        <p class="mission-subtitle">
            Autonomous pixel-level traversability classifier and sand entrapment detector trained on NASA JPL's <strong>AI4Mars</strong> dataset. 
            Identifies safe bedrock pavement, navigable soil, and triggers emergency replanning for loose sand dunes and collision boulders.
        </p>
    </div>
    """)
    
    with gr.Row():
        # LEFT COLUMN: Controls & Input
        with gr.Column(scale=5):
            input_image = gr.Image(
                type="numpy", 
                label="Rover Camera Optical Feed (Navcam / Hazcam / Mastcam)",
                height=380
            )
            
            with gr.Accordion("⚙️ Sensor & Autonomous Perception Controls", open=True):
                with gr.Row():
                    opacity_slider = gr.Slider(
                        minimum=0.1, 
                        maximum=1.0, 
                        value=0.60, 
                        step=0.05, 
                        label="Overlay Opacity"
                    )
                    img_size_dropdown = gr.Dropdown(
                        choices=["384", "512"], 
                        value="512", 
                        label="Inference Resolution",
                        info="512: High accuracy | 384: Rapid telemetry"
                    )
                
                sand_thresh_slider = gr.Slider(
                    minimum=5.0, 
                    maximum=40.0, 
                    value=25.0, 
                    step=1.0, 
                    label="Sand Entrapment Tripwire Threshold (%)",
                    info="Triggers emergency halt if sand exceeds this ratio (Spirit trap prevention)"
                )

                with gr.Row():
                    tta_mode_dropdown = gr.Dropdown(
                        choices=["disabled", "auto (watchdog)", "fast", "full"],
                        value="disabled",
                        label="Multi-View TTA Ensemble",
                        info="disabled | auto (watchdog-triggered) | fast (2x) | full (4x)"
                    )
                    sol_number_input = gr.Number(
                        value=4118,
                        precision=0,
                        label="Mars Mission Sol #",
                        info="Tagged in telemetry & downloadable field report"
                    )
                
            submit_btn = gr.Button(
                "🛰️ ANALYZE TERRAIN & COMPUTE TRAVERSABILITY", 
                variant="primary", 
                elem_classes=["btn-analyze"],
                size="lg"
            )
            
            if sample_files:
                gr.Markdown("### 📂 Curated Mission Samples (Click to load):")
                gr.Examples(
                    examples=[[f] for f in sample_files[:6]],
                    inputs=[input_image],
                    label="Try Curiosity & Opportunity Mission Images"
                )

        # RIGHT COLUMN: Multi-View Outputs & Telemetry HUD
        with gr.Column(scale=6):
            with gr.Tabs():
                with gr.TabItem("🗺️ Multiclass Terrain Map"):
                    output_overlay = gr.Image(type="numpy", label="DeepLabV3+ Semantic Segmentation", height=420)
                with gr.TabItem("🚨 Sand Hazard Radar (Spirit Trap Alert)"):
                    output_sand = gr.Image(type="numpy", label="Isolated Sand Entrapment Hazard Overlay", height=420)
                with gr.TabItem("🧭 Traversability Costmap"):
                    output_costmap = gr.Image(type="numpy", label="Autonomous Path-Planning Costmap (A*/D*)", height=420)
                with gr.TabItem("↔️ Side-by-Side Dual View"):
                    output_sbs = gr.Image(type="numpy", label="Original Rover Photo vs Segmented Classification", height=420)
                with gr.TabItem("🚀 A* Autonomous Drive Path"):
                    output_path = gr.Image(type="numpy", label="Collision-Free A* Navigation Trajectory", height=420)
                with gr.TabItem("🔮 Prediction Uncertainty Heatmap"):
                    output_uncertainty = gr.Image(type="numpy", label="Epistemic Shannon Entropy Confidence Heatmap", height=420)
                with gr.TabItem("📡 Near-Field Zone Radar"):
                    output_zone_radar = gr.Image(type="numpy", label="0-5m Near-Field Hazard & Boulder Cluster Radar", height=420)
                with gr.TabItem("🎯 AEGIS Science Targets"):
                    output_science = gr.Image(type="numpy", label="Autonomous ChemCam / Mastcam Science Target Reticles", height=420)
                    
            # Telemetry Metrics Dashboard
            stats_box = gr.HTML("""
            <div class="empty-state-box">
                <div style="font-size: 32px; margin-bottom: 8px;">🛰️</div>
                <div style="font-weight: 600; font-size: 16px; color: #94a3b8;">AWAITING CAMERA INGESTION</div>
                <div style="font-size: 13px; color: #64748b; margin-top: 4px;">Select a sample or upload an image and click <strong>ANALYZE TERRAIN</strong>.</div>
            </div>
            """)

            report_download = gr.File(
                label="📄 Download Sol Drive Engineering Field Report (HTML)",
                interactive=False
            )

    # BOTTOM ROW: Monospace Autonomous Telemetry Console
    with gr.Row():
        with gr.Column():
            with gr.Accordion("💻 Autonomous Flight Computer Telemetry Console (Live Log)", open=False):
                flight_terminal = gr.Textbox(
                    label="Flight Computer Console", 
                    lines=6, 
                    max_lines=10, 
                    interactive=False,
                    elem_classes=["terminal-card"],
                    value="[SYSTEM IDLE] AutoNav vision engine awaiting image ingestion..."
                )

    # Legend & Reference Info
    with gr.Row():
        gr.Markdown("""
        <div style="background: rgba(15, 23, 42, 0.6); border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 10px; padding: 14px 20px; margin-top: 10px;">
            <div style="font-size: 12px; font-weight: 700; letter-spacing: 0.8px; color: #94a3b8; margin-bottom: 8px; text-transform: uppercase;">
                Classification Legend & Safety Roles:
            </div>
            <div style="display: flex; gap: 24px; flex-wrap: wrap; font-size: 13px;">
                <span><span style="color: #f1c40f; font-weight: 800;">■ Soil (0):</span> Compact Martian regolith (Safe driving surface)</span>
                <span><span style="color: #00b4d8; font-weight: 800;">■ Bedrock (1):</span> Solid rock pavement (Optimal grip, lowest slip)</span>
                <span><span style="color: #ff7b00; font-weight: 800;">■ Sand (2):</span> Soft ripple dunes (<strong style="color: #ff7b00;">Critical Hazard</strong> — trapped Spirit rover in 2009)</span>
                <span><span style="color: #ff3860; font-weight: 800;">■ Big Rock (3):</span> Boulders &gt; 30 cm height (<strong style="color: #ff3860;">Collision Hazard</strong>)</span>
            </div>
        </div>
        """)

    # Interactive Event Binding
    submit_btn.click(
        fn=segment_mars_terrain,
        inputs=[input_image, opacity_slider, img_size_dropdown, sand_thresh_slider, tta_mode_dropdown, sol_number_input],
        outputs=[output_overlay, output_sand, output_costmap, output_sbs, output_path, output_uncertainty, output_zone_radar, output_science, stats_box, flight_terminal, report_download]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
