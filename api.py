"""
Production FastAPI Microservice for Mars Rover Terrain Perception & Path Planning.
Exposes REST endpoints for client applications, ground control stations, and remote rovers.

Run server:
  uvicorn api:app --host 0.0.0.0 --port 8000 --reload
"""
import io
import base64
from typing import Optional
from fastapi import FastAPI, File, UploadFile, Query, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
import numpy as np

from src.inference import MarsTerrainSegmenter
from src.path_planner import MarsPathPlanner
from src.kinematics import RoverKinematicPlanner

app = FastAPI(
    title="NASA JPL Mars Rover AutoNav Perception API",
    description="REST API for DeepLabV3+ Mars Rover Terrain Segmentation, Traversability Scoring, Steering Kinematics, AEGIS Science Targeting, and A* Path Planning.",
    version="2.1.0"
)

# Global model instance
_segmenter: Optional[MarsTerrainSegmenter] = None
_planner: Optional[MarsPathPlanner] = None
_kinematics: Optional[RoverKinematicPlanner] = None


def get_segmenter() -> MarsTerrainSegmenter:
    global _segmenter
    if _segmenter is None:
        _segmenter = MarsTerrainSegmenter()
    return _segmenter


def get_planner() -> MarsPathPlanner:
    global _planner
    if _planner is None:
        _planner = MarsPathPlanner(obstacle_inflation_radius=2)
    return _planner


def get_kinematic_planner() -> RoverKinematicPlanner:
    global _kinematics
    if _kinematics is None:
        _kinematics = RoverKinematicPlanner()
    return _kinematics


def encode_image_base64(img_array: np.ndarray) -> str:
    pil_img = Image.fromarray(img_array)
    buffer = io.BytesIO()
    pil_img.save(buffer, format="JPEG", quality=90)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


@app.get("/")
def root():
    return {
        "service": "NASA JPL Mars Rover Terrain Segmentation API",
        "model": "DeepLabV3+ (ResNet-50)",
        "status": "ONLINE",
        "docs_url": "/docs",
        "endpoints": ["/v1/segment", "/v1/plan", "/v1/science-targets", "/v1/classes"]
    }


@app.get("/v1/classes")
def get_classes():
    return {
        "classes": [
            {"id": 0, "name": "soil", "type": "Safe Regolith", "traction": "Nominal"},
            {"id": 1, "name": "bedrock", "type": "Solid Pavement", "traction": "Optimal"},
            {"id": 2, "name": "sand", "type": "Loose Dune / Ripple", "hazard": "High Wheel Slip Risk"},
            {"id": 3, "name": "big rock", "type": "Boulder > 30cm", "hazard": "Chassis Collision"}
        ],
        "ignored_index": 255
    }


@app.post("/v1/segment")
async def segment_terrain(
    file: UploadFile = File(...),
    resolution: int = Query(512, ge=256, le=1024, description="Inference resolution"),
    sand_threshold: float = Query(25.0, ge=5.0, le=50.0, description="Sand emergency halt threshold (%)"),
    use_tta: bool = Query(False, description="Enable multi-view Test-Time Augmentation ensemble"),
    include_base64_overlays: bool = Query(False, description="Whether to include base64-encoded JPEG overlays")
):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image format: {e}")

    segmenter = get_segmenter()
    result = segmenter.predict(image, img_size=resolution, sand_threshold=sand_threshold, use_tta=use_tta)

    response = {
        "telemetry": result["telemetry"]
    }

    if include_base64_overlays:
        response["overlays"] = {
            "multiclass_terrain": encode_image_base64(result["overlay_multiclass"]),
            "sand_hazard_radar": encode_image_base64(result["overlay_sand_hazard"]),
            "traversability_costmap": encode_image_base64(result["overlay_costmap"]),
            "uncertainty_heatmap": encode_image_base64(result["overlay_uncertainty"]),
            "zone_radar": encode_image_base64(result["overlay_zone_radar"]),
            "science_targets": encode_image_base64(result["overlay_science_targets"])
        }

    return JSONResponse(content=response)


@app.post("/v1/plan")
async def plan_trajectory(
    file: UploadFile = File(...),
    planning_resolution: int = Query(128, ge=64, le=256, description="A* search grid resolution"),
    include_base64_path: bool = Query(False, description="Whether to include base64-encoded trajectory visualization")
):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image format: {e}")

    segmenter = get_segmenter()
    seg_res = segmenter.predict(image, img_size=512)
    pred_mask = seg_res["prediction_mask"]

    planner = get_planner()
    plan_data = planner.plan_path(pred_mask, planning_resolution=planning_resolution)

    kin_planner = get_kinematic_planner()
    eff_speed = seg_res["telemetry"]["power_budget"]["effective_ground_speed_cm_s"]
    kin_data = kin_planner.profile_trajectory(
        plan_data,
        image_shape=pred_mask.shape,
        effective_speed_cm_s=eff_speed,
        pred_mask=pred_mask
    )

    response = {
        "status": plan_data["status"],
        "waypoint_count": plan_data["waypoint_count"],
        "total_path_cost": plan_data["total_path_cost"],
        "start_coordinate": list(plan_data["start"]),
        "goal_coordinate": list(plan_data["goal"]),
        "waypoints": [list(pt) for pt in plan_data["waypoints"]],
        "kinematics": {
            key: ([list(p) for p in val] if key == "smoothed_waypoints" else val)
            for key, val in kin_data.items()
        }
    }

    if include_base64_path:
        path_overlay = planner.render_path_overlay(seg_res["overlay_multiclass"], plan_data)
        response["trajectory_overlay"] = encode_image_base64(path_overlay)

    return JSONResponse(content=response)


@app.post("/v1/science-targets")
async def extract_science_targets(
    file: UploadFile = File(...),
    resolution: int = Query(512, ge=256, le=1024, description="Inference resolution"),
    include_base64_reticle: bool = Query(False, description="Include base64 optical targeting reticle overlay")
):
    """Autonomous NASA AEGIS-style geological science target identification endpoint."""
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image format: {e}")

    segmenter = get_segmenter()
    seg_res = segmenter.predict(image, img_size=resolution)

    response: dict = {
        "aegis_targets": seg_res["telemetry"]["science_targets"],
        "geometry_risk": seg_res["telemetry"]["geometry_risk"],
    }
    if include_base64_reticle:
        response["target_reticle_overlay"] = encode_image_base64(seg_res["overlay_science_targets"])

    return JSONResponse(content=response)

