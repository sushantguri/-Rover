"""
Standalone Inference and Robotic Telemetry Pipeline.
Provides high-performance inference on single images or directories of Mars rover frames.
Generates multiclass terrain maps, sand hazard alerts, robot path-planning costmaps, and JSON telemetry.
"""
import os
import json
import time
from typing import Dict, Any, Tuple, Optional
import numpy as np
from PIL import Image
import torch

from src.model import get_deeplabv3plus_model
from src.dataset import CLASSES
from src.visualization import create_overlay
from src.terramechanics import RoverTerramechanicsEngine
from src.uncertainty import UncertaintyEstimator
from src.tta import TestTimeAugmenter
from src.power_budget import RoverPowerBudgetEstimator
from src.geometry_risk import TerrainGeometryAnalyzer
from src.science_targets import AEGISTargetSelector
from src.sensor_watchdog import OpticalSensorWatchdog
from src.calibration import ProbabilityCalibrator
from src.slope_stability import SlopeStabilityAnalyzer
from src.wheel_wear import RoverWheelWearModel
from src.solar_thermal import MartianSolarThermalModel
from src.costmap_fusion import AutoNavCostmapFuser

# Costmap palette: Bedrock=Cyan/Blue, Soil=Green, Sand=Orange (Hazard), Big Rock=Red (Obstacle)
COST_PALETTE = np.array([
    [76, 175, 80],     # 0: Soil
    [33, 150, 243],    # 1: Bedrock
    [255, 152, 0],     # 2: Sand
    [239, 68, 68],     # 3: Big Rock
], dtype=np.uint8)


class MarsTerrainSegmenter:
    """
    Production-grade inference engine for Mars rover terrain segmentation.
    """
    def __init__(self, checkpoint_path: str = "checkpoints/best_deeplabv3plus.pth", device: Optional[str] = None):
        if device is None:
            if torch.backends.mps.is_available():
                self.device = torch.device("mps")
            elif torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        self.model = get_deeplabv3plus_model(num_classes=4).to(self.device)
        
        if os.path.exists(checkpoint_path):
            ckpt = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(ckpt["model_state_dict"])
            self.has_weights = True
        else:
            self.has_weights = False

        self.model.eval()
        self.terramechanics_engine = RoverTerramechanicsEngine()
        self.uncertainty_estimator = UncertaintyEstimator()
        self.power_estimator = RoverPowerBudgetEstimator()
        self.geometry_analyzer = TerrainGeometryAnalyzer()
        self.science_selector = AEGISTargetSelector(top_k=3)
        self.sensor_watchdog = OpticalSensorWatchdog()
        self.calibrator = ProbabilityCalibrator()
        self.slope_analyzer = SlopeStabilityAnalyzer()
        self.wheel_wear_model = RoverWheelWearModel()
        self.solar_thermal_model = MartianSolarThermalModel()
        self.costmap_fuser = AutoNavCostmapFuser()
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)

    def predict(
        self, 
        image_input: Image.Image | np.ndarray | str, 
        img_size: int = 512,
        sand_threshold: float = 25.0,
        opacity: float = 0.60,
        use_tta: bool = False,
        tta_mode: str = "fast",
        auto_tta: bool = False,
        pitch_deg: float = 0.0,
        roll_deg: float = 0.0,
        local_sol_hour: float = 13.5,
        dust_tau: float = 0.5,
        solar_elevation_deg: float = 45.0
    ) -> Dict[str, Any]:
        """
        Runs full terrain perception, generates visualizations, and computes drive telemetry.
        """
        # Load image
        if isinstance(image_input, str):
            orig_img = Image.open(image_input).convert("RGB")
        elif isinstance(image_input, np.ndarray):
            orig_img = Image.fromarray(image_input).convert("RGB")
        else:
            orig_img = image_input.convert("RGB")

        orig_w, orig_h = orig_img.size
        orig_np = np.array(orig_img)

        # Optical sensor fault diagnostics (solar glare, dust haze, dark frames)
        sensor_health = self.sensor_watchdog.inspect_frame(orig_np)
        effective_use_tta = bool(use_tta or (auto_tta and sensor_health["recommend_tta_boost"]))

        # Preprocess
        t0 = time.time()
        resized = orig_img.resize((img_size, img_size), Image.BILINEAR)
        arr = np.array(resized, dtype=np.float32) / 255.0
        normalized = (arr - self.mean) / self.std
        tensor = torch.from_numpy(normalized).permute(2, 0, 1).unsqueeze(0).to(self.device)

        # Forward pass (Single Pass or Multi-View TTA Ensemble)
        with torch.no_grad():
            if effective_use_tta:
                tta = TestTimeAugmenter(self.model, mode=tta_mode)
                logits = tta.predict_pseudo_logits(tensor)
            else:
                logits = self.model(tensor)
            pred_map = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

        # Scale prediction back to original dimensions
        pred_full = np.array(Image.fromarray(pred_map).resize((orig_w, orig_h), Image.NEAREST), dtype=np.int64)
        latency_ms = (time.time() - t0) * 1000.0

        # Create Visual Overlays
        overlay_multiclass = create_overlay(orig_np, pred_full, alpha=opacity, highlight_sand=False)
        overlay_sand_hazard = create_overlay(orig_np, pred_full, alpha=0.80, highlight_sand=True)
        
        # Create Traversability Costmap
        cost_colored = COST_PALETTE[np.clip(pred_full, 0, 3)]
        costmap_overlay = (orig_np * (1.0 - opacity) + cost_colored * opacity).astype(np.uint8)

        # Compute Epistemic Uncertainty & Heatmap
        entropy_map = self.uncertainty_estimator.compute_entropy_map(logits)
        overlay_uncertainty = self.uncertainty_estimator.generate_uncertainty_heatmap(orig_np, entropy_map, alpha=opacity)
        uncertainty_telemetry = self.uncertainty_estimator.evaluate_uncertainty_telemetry(entropy_map)

        # Temperature-Scaled Calibration & Conformal Sand Safety Margin
        calibration_safety, borderline_sand_mask = self.calibrator.evaluate_safety_margin(
            logits, pred_full, sand_threshold_pct=sand_threshold
        )
        overlay_safety_margin = self.calibrator.render_safety_margin_overlay(
            orig_np, pred_full, borderline_sand_mask, alpha=opacity
        )

        # Perspective Range-Zone & Boulder Geometry Risk Analysis
        geometry_risk = self.geometry_analyzer.analyze(pred_full)
        overlay_zone_radar = self.geometry_analyzer.render_zone_radar_overlay(
            orig_np, pred_full, geometry_risk, alpha=opacity
        )

        # AEGIS Autonomous Geological Science Target Selection
        science_targets = self.science_selector.select_targets(pred_full, entropy_map=entropy_map, top_k=3)
        overlay_science_targets = self.science_selector.render_target_reticles(orig_np, science_targets)

        # Telemetry Analysis
        total_pixels = pred_full.size
        class_pcts = {}
        class_pixels = {}
        for c_id, c_name in enumerate(CLASSES):
            px = int((pred_full == c_id).sum())
            class_pixels[c_name] = px
            class_pcts[c_name] = float((px / total_pixels) * 100.0)

        sand_pct = class_pcts["sand"]
        rock_pct = class_pcts["big rock"]
        bedrock_pct = class_pcts["bedrock"]
        soil_pct = class_pcts["soil"]

        # Terramechanics Physics Analysis
        mechanics_telemetry = self.terramechanics_engine.evaluate_terrain_mechanics(class_pcts)

        # Traversability Index (0 to 100)
        traversability_index = float(bedrock_pct * 1.0 + soil_pct * 0.85 + sand_pct * 0.15 + rock_pct * 0.0)
        traversability_index = max(0.0, min(100.0, traversability_index))

        # Autonomous Decision Logic
        if sand_pct >= sand_threshold:
            safety_status = "CRITICAL_HAZARD"
            drive_action = "EMERGENCY_HALT_AND_REPLAN"
            recommended_speed_cm_s = 0.0
            slip_probability = "HIGH (> 85%)"
        elif sand_pct > 8.0 or rock_pct > 2.0:
            safety_status = "CAUTION_ADVISORY"
            drive_action = "SLOW_CREEP_MODE"
            recommended_speed_cm_s = 1.8
            slip_probability = "MODERATE (25-40%)"
        else:
            safety_status = "NOMINAL_DRIVE"
            drive_action = "CONTINUE_WAYPOINT_TRAVERSE"
            recommended_speed_cm_s = 4.5
            slip_probability = "LOW (< 5%)"

        # Rover Attitude & Gravity-Coupled Slope Stability
        slope_stability = self.slope_analyzer.analyze_attitude(
            pitch_deg=pitch_deg,
            roll_deg=roll_deg,
            terramechanics_data=mechanics_telemetry,
            commanded_speed_cm_s=recommended_speed_cm_s,
        )
        if slope_stability["stability_status"] == "TIPOVER_ENVELOPE_EXCEEDED":
            safety_status = "CRITICAL_HAZARD"
            drive_action = "EMERGENCY_HALT_ATTITUDE_RECOVERY"
            recommended_speed_cm_s = 0.0
            slip_probability = "EXTREME (ROLLOVER RISK)"

        # MMRTG Power Budget Modeling
        power_telemetry = self.power_estimator.estimate_traverse_energy(
            terramechanics_data=mechanics_telemetry,
            commanded_speed_cm_s=recommended_speed_cm_s,
            planned_distance_m=20.0
        )

        # Martian Ventifact Roughness & Wheel Wear Fatigue Mechanics
        gray_frame = np.mean(orig_np, axis=2).astype(np.uint8) if orig_np.ndim == 3 else orig_np
        ventifact_roughness = self.wheel_wear_model.compute_ventifact_roughness(gray_frame, pred_full)
        wheel_wear_telemetry = self.wheel_wear_model.evaluate_wheel_wear(
            terrain_distribution_pct=class_pcts,
            terramechanics_data=mechanics_telemetry,
            ventifact_roughness_index=ventifact_roughness,
            drive_distance_m=20.0,
        )
        overlay_wheel_wear = self.wheel_wear_model.render_wear_hazard_overlay(
            orig_np, pred_full, ventifact_roughness_index=ventifact_roughness, alpha=opacity
        )

        # Diurnal Solar Insolation, Dust Optical Depth (Tau), & Battery Heater Thermal Model
        optical_shadow = float(sensor_health.get("underexposed_pct", 5.0))
        solar_thermal_telemetry = self.solar_thermal_model.evaluate_thermal_environment(
            local_sol_hour=local_sol_hour,
            solar_elevation_deg=solar_elevation_deg,
            tau=dust_tau,
            optical_shadow_pct=optical_shadow,
        )

        # Multi-Layer AutoNav Costmap Fusion (Semantic, Slope, Uncertainty, Ventifact Protection)
        fused_costmap, costmap_fusion_telemetry = self.costmap_fuser.fuse_layers(
            pred_mask=pred_full,
            entropy_map=entropy_map,
            pitch_deg=pitch_deg,
            roll_deg=roll_deg,
            ventifact_roughness_index=ventifact_roughness,
        )
        overlay_fused_costmap = self.costmap_fuser.render_fused_costmap_overlay(
            orig_np, fused_costmap, alpha=opacity
        )

        telemetry = {
            "image_dimensions": {"width": orig_w, "height": orig_h},
            "inference_resolution": img_size,
            "tta_mode": tta_mode if effective_use_tta else "disabled",
            "auto_tta_triggered": bool(auto_tta and not use_tta and sensor_health["recommend_tta_boost"]),
            "inference_latency_ms": round(latency_ms, 2),
            "device": str(self.device),
            "sensor_health": sensor_health,
            "terrain_distribution_pct": class_pcts,
            "terrain_pixel_counts": class_pixels,
            "traversability_index": round(traversability_index, 2),
            "safety_status": safety_status,
            "autonomous_drive_action": drive_action,
            "recommended_speed_cm_s": recommended_speed_cm_s,
            "wheel_slip_probability": slip_probability,
            "terramechanics": mechanics_telemetry,
            "slope_stability": slope_stability,
            "power_budget": power_telemetry,
            "wheel_wear": wheel_wear_telemetry,
            "solar_thermal": solar_thermal_telemetry,
            "costmap_fusion": costmap_fusion_telemetry,
            "uncertainty": uncertainty_telemetry,
            "calibration_safety": calibration_safety,
            "geometry_risk": geometry_risk,
            "science_targets": science_targets
        }

        return {
            "prediction_mask": pred_full,
            "overlay_multiclass": overlay_multiclass,
            "overlay_sand_hazard": overlay_sand_hazard,
            "overlay_costmap": costmap_overlay,
            "overlay_fused_costmap": overlay_fused_costmap,
            "overlay_uncertainty": overlay_uncertainty,
            "overlay_zone_radar": overlay_zone_radar,
            "overlay_science_targets": overlay_science_targets,
            "overlay_safety_margin": overlay_safety_margin,
            "overlay_wheel_wear": overlay_wheel_wear,
            "telemetry": telemetry
        }
