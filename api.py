"""
Production FastAPI Microservice for Mars Rover Terrain Perception & Path Planning.
Exposes REST endpoints for client applications, ground control stations, and remote rovers.

Run server:
  uvicorn api:app --host 0.0.0.0 --port 8000 --reload
"""
import io
import base64
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, File, UploadFile, Query, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from PIL import Image
import numpy as np

from src.inference import MarsTerrainSegmenter
from src.path_planner import MarsPathPlanner
from src.kinematics import RoverKinematicPlanner
from src.ros_bridge import Ros2NavBridge
from src.sol_simulator import SolMissionSimulator
from src.dsn_telemetry import DsnCommManager

app = FastAPI(
    title="NASA JPL Mars Rover AutoNav Perception API",
    description="REST API for DeepLabV3+ Mars Rover Terrain Segmentation, Traversability Scoring, Steering Kinematics, Slope Stability, Conformal Sand Margins, AEGIS Science Targeting, Sol Mission Simulation, ROS 2 Nav2 Export, and A* Path Planning.",
    version="2.3.0"
)


class SolSimRequest(BaseModel):
    initial_soc_pct: float = Field(90.0, ge=5.0, le=100.0)
    min_reserve_soc_pct: float = Field(30.0, ge=0.0, le=80.0)
    max_consecutive_halts: int = Field(3, ge=1, le=20)
    steps: List[Dict[str, Any]] = Field(..., min_length=1)

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
        "endpoints": [
            "/v1/segment",
            "/v1/plan",
            "/v1/science-targets",
            "/v1/ros2-packet",
            "/v1/simulate-sol",
            "/v1/costmap-fusion",
            "/v1/wheel-wear",
            "/v1/stereo-geometry",
            "/v1/aeolian-drift",
            "/v1/dsn-budget",
            "/v1/visual-odometry",
            "/v1/dwa-plan",
            "/v1/dust-accumulation",
            "/v1/classes"
        ]
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
    resolution: int = Query(512, ge=64, le=1024, description="Inference resolution"),
    sand_threshold: float = Query(25.0, ge=5.0, le=50.0, description="Sand emergency halt threshold (%)"),
    use_tta: bool = Query(False, description="Enable multi-view Test-Time Augmentation ensemble"),
    auto_tta: bool = Query(False, description="Automatically enable TTA when optical watchdog detects glare, haze, or dark frames"),
    pitch_deg: float = Query(0.0, ge=-45.0, le=45.0, description="Rover longitudinal pitch angle in degrees"),
    roll_deg: float = Query(0.0, ge=-45.0, le=45.0, description="Rover lateral roll angle in degrees"),
    local_sol_hour: float = Query(13.5, ge=0.0, le=24.66, description="Martian local true solar time in hours"),
    dust_tau: float = Query(0.5, ge=0.1, le=6.0, description="Atmospheric dust optical depth tau"),
    solar_elevation_deg: float = Query(45.0, ge=-20.0, le=90.0, description="Solar elevation angle in degrees"),
    rover_heading_deg: float = Query(0.0, ge=0.0, le=360.0, description="Rover heading azimuth in degrees"),
    wind_speed_m_s: float = Query(8.0, ge=0.0, le=60.0, description="Martian wind speed in m/s"),
    wind_azimuth_deg: float = Query(270.0, ge=0.0, le=360.0, description="Wind source azimuth in degrees"),
    comm_profile: str = Query("UHF_MRO_RELAY", description="DSN / Orbiter relay profile"),
    commanded_distance_m: float = Query(1.0, ge=0.1, le=20.0, description="Commanded wheel drive distance in meters"),
    target_heading_deg: float = Query(0.0, ge=-180.0, le=180.0, description="Target waypoint heading in degrees"),
    sol_number: int = Query(4125, ge=1, le=10000, description="Mars mission sol number"),
    last_cleaning_sol: int = Query(4000, ge=1, le=10000, description="Last convective vortex cleaning sol"),
    include_base64_overlays: bool = Query(False, description="Whether to include base64-encoded JPEG overlays")
):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image format: {e}")

    segmenter = get_segmenter()
    result = segmenter.predict(
        image,
        img_size=resolution,
        sand_threshold=sand_threshold,
        use_tta=use_tta,
        auto_tta=auto_tta,
        pitch_deg=pitch_deg,
        roll_deg=roll_deg,
        local_sol_hour=local_sol_hour,
        dust_tau=dust_tau,
        solar_elevation_deg=solar_elevation_deg,
        rover_heading_deg=rover_heading_deg,
        wind_speed_m_s=wind_speed_m_s,
        wind_azimuth_deg=wind_azimuth_deg,
        comm_profile=comm_profile,
        commanded_distance_m=commanded_distance_m,
        target_heading_deg=target_heading_deg,
        sol_number=sol_number,
        last_cleaning_sol=last_cleaning_sol,
    )

    response = {
        "telemetry": result["telemetry"]
    }

    if include_base64_overlays:
        response["overlays"] = {
            "multiclass_terrain": encode_image_base64(result["overlay_multiclass"]),
            "sand_hazard_radar": encode_image_base64(result["overlay_sand_hazard"]),
            "traversability_costmap": encode_image_base64(result["overlay_costmap"]),
            "fused_costmap": encode_image_base64(result["overlay_fused_costmap"]),
            "stereo_elevation": encode_image_base64(result["overlay_stereo_elevation"]),
            "aeolian_drift": encode_image_base64(result["overlay_aeolian_drift"]),
            "visual_odometry": encode_image_base64(result["overlay_visual_odometry"]),
            "dwa_trajectories": encode_image_base64(result["overlay_dwa_trajectories"]),
            "dust_deposition": encode_image_base64(result["overlay_dust_deposition"]),
            "uncertainty_heatmap": encode_image_base64(result["overlay_uncertainty"]),
            "zone_radar": encode_image_base64(result["overlay_zone_radar"]),
            "science_targets": encode_image_base64(result["overlay_science_targets"]),
            "safety_margin": encode_image_base64(result["overlay_safety_margin"]),
            "wheel_wear": encode_image_base64(result["overlay_wheel_wear"]),
        }

    return JSONResponse(content=response)


@app.post("/v1/costmap-fusion")
async def fuse_costmap(
    file: UploadFile = File(...),
    resolution: int = Query(512, ge=64, le=1024, description="Inference resolution"),
    pitch_deg: float = Query(0.0, ge=-45.0, le=45.0, description="Rover pitch angle in degrees"),
    roll_deg: float = Query(0.0, ge=-45.0, le=45.0, description="Rover roll angle in degrees"),
    include_base64_overlay: bool = Query(False, description="Whether to include base64-encoded fused costmap overlay")
):
    """Fuses semantic terrain traversability, slope resistance, uncertainty buffers, and ventifact wear into a unified navigation costmap."""
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image format: {e}")

    segmenter = get_segmenter()
    result = segmenter.predict(image, img_size=resolution, pitch_deg=pitch_deg, roll_deg=roll_deg)
    
    response: Dict[str, Any] = {
        "costmap_fusion": result["telemetry"]["costmap_fusion"],
        "slope_stability": result["telemetry"]["slope_stability"],
        "traversability_index": result["telemetry"]["traversability_index"],
        "safety_status": result["telemetry"]["safety_status"],
    }
    if include_base64_overlay:
        response["fused_costmap_overlay"] = encode_image_base64(result["overlay_fused_costmap"])

    return JSONResponse(content=response)


@app.post("/v1/wheel-wear")
async def evaluate_wheel_wear(
    file: UploadFile = File(...),
    resolution: int = Query(512, ge=64, le=1024, description="Inference resolution"),
    include_base64_overlay: bool = Query(False, description="Whether to include ruby-red wheel wear hazard overlay")
):
    """Evaluates Curiosity/Perseverance 0.75 mm aluminum wheel contact stress, ventifact roughness, and puncture risk."""
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image format: {e}")

    segmenter = get_segmenter()
    result = segmenter.predict(image, img_size=resolution)

    response: Dict[str, Any] = {
        "wheel_wear": result["telemetry"]["wheel_wear"],
        "terramechanics": result["telemetry"]["terramechanics"],
        "safety_status": result["telemetry"]["safety_status"],
    }
    if include_base64_overlay:
        response["wheel_wear_overlay"] = encode_image_base64(result["overlay_wheel_wear"])

    return JSONResponse(content=response)


@app.post("/v1/stereo-geometry")
async def evaluate_stereo_geometry(
    file: UploadFile = File(...),
    resolution: int = Query(512, ge=64, le=1024, description="Inference resolution"),
    include_base64_overlay: bool = Query(False, description="Whether to include base64-encoded 3D elevation overlay")
):
    """Evaluates 3D stereo disparity, positive step obstacles, negative trench drop-offs, and ground clearance."""
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image format: {e}")

    segmenter = get_segmenter()
    result = segmenter.predict(image, img_size=resolution)

    response: Dict[str, Any] = {
        "stereo_geometry": result["telemetry"]["stereo_geometry"],
        "geometry_risk": result["telemetry"]["geometry_risk"],
        "safety_status": result["telemetry"]["safety_status"],
    }
    if include_base64_overlay:
        response["stereo_elevation_overlay"] = encode_image_base64(result["overlay_stereo_elevation"])

    return JSONResponse(content=response)


@app.post("/v1/aeolian-drift")
async def evaluate_aeolian_drift(
    file: UploadFile = File(...),
    resolution: int = Query(512, ge=64, le=1024, description="Inference resolution"),
    rover_heading_deg: float = Query(0.0, ge=0.0, le=360.0, description="Rover heading azimuth"),
    wind_speed_m_s: float = Query(8.0, ge=0.0, le=60.0, description="Wind speed in m/s"),
    wind_azimuth_deg: float = Query(270.0, ge=0.0, le=360.0, description="Wind source azimuth"),
    include_base64_overlay: bool = Query(False, description="Whether to include base64-encoded aeolian drift overlay")
):
    """Evaluates atmospheric wind shear, Bagnold sand saltation flux, and dune slipface avalanche entrapment."""
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image format: {e}")

    segmenter = get_segmenter()
    result = segmenter.predict(
        image,
        img_size=resolution,
        rover_heading_deg=rover_heading_deg,
        wind_speed_m_s=wind_speed_m_s,
        wind_azimuth_deg=wind_azimuth_deg,
    )

    response: Dict[str, Any] = {
        "aeolian_drift": result["telemetry"]["aeolian_drift"],
        "sand_distribution_pct": result["telemetry"]["terrain_distribution_pct"]["sand"],
        "safety_status": result["telemetry"]["safety_status"],
    }
    if include_base64_overlay:
        response["aeolian_drift_overlay"] = encode_image_base64(result["overlay_aeolian_drift"])

    return JSONResponse(content=response)


@app.get("/v1/dsn-budget")
def query_dsn_budget(
    comm_profile: str = Query("UHF_MRO_RELAY", description="Orbiter relay / DTE link profile"),
    local_sol_hour: float = Query(13.5, ge=0.0, le=24.66, description="Local solar time in hours"),
    has_overlays: bool = Query(True, description="Whether packet queue includes optical overlays")
):
    """Query NASA DSN & Mars Relay Orbiter link budget and pass schedule directly."""
    mgr = DsnCommManager()
    packets = mgr.packetize_telemetry({"safety_status": "NOMINAL_DRIVE"}, has_overlays=has_overlays)
    budget = mgr.evaluate_link_budget(packets, comm_profile=comm_profile)
    sched = mgr.schedule_next_pass(current_sol_hour=local_sol_hour)
    return JSONResponse(content={
        "packets": packets,
        "link_budget": budget,
        "pass_schedule": sched,
    })


@app.post("/v1/dsn-budget")
async def evaluate_dsn_budget(
    file: UploadFile = File(...),
    resolution: int = Query(512, ge=64, le=1024, description="Inference resolution"),
    comm_profile: str = Query("UHF_MRO_RELAY", description="Orbiter relay / DTE link profile"),
    local_sol_hour: float = Query(13.5, ge=0.0, le=24.66, description="Local solar time in hours")
):
    """Simulates NASA Deep Space Network (DSN) & Mars Relay Orbiter (MRO/TGO) telemetry downlink budgets."""
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image format: {e}")

    segmenter = get_segmenter()
    result = segmenter.predict(
        image,
        img_size=resolution,
        comm_profile=comm_profile,
        local_sol_hour=local_sol_hour,
    )

    response: Dict[str, Any] = {
        "dsn_telemetry": result["telemetry"]["dsn_telemetry"],
        "safety_status": result["telemetry"]["safety_status"],
    }
    return JSONResponse(content=response)


@app.post("/v1/visual-odometry")
async def evaluate_visual_odometry(
    file: UploadFile = File(...),
    resolution: int = Query(512, ge=64, le=1024, description="Inference resolution"),
    commanded_distance_m: float = Query(1.0, ge=0.1, le=20.0, description="Commanded wheel drive distance in meters"),
    include_base64_overlay: bool = Query(False, description="Whether to include base64-encoded VO feature tracking overlay")
):
    """Evaluates visual odometry feature tracking, true ground displacement, and wheel slip ratio."""
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image format: {e}")

    segmenter = get_segmenter()
    result = segmenter.predict(
        image,
        img_size=resolution,
        commanded_distance_m=commanded_distance_m,
    )

    response: Dict[str, Any] = {
        "visual_odometry": result["telemetry"]["visual_odometry"],
        "safety_status": result["telemetry"]["safety_status"],
    }
    if include_base64_overlay:
        response["visual_odometry_overlay"] = encode_image_base64(result["overlay_visual_odometry"])

    return JSONResponse(content=response)


@app.post("/v1/dwa-plan")
async def evaluate_dwa_local_plan(
    file: UploadFile = File(...),
    resolution: int = Query(512, ge=64, le=1024, description="Inference resolution"),
    target_heading_deg: float = Query(0.0, ge=-180.0, le=180.0, description="Target waypoint heading in degrees"),
    include_base64_overlay: bool = Query(False, description="Whether to include base64-encoded DWA trajectories fan overlay")
):
    """Simulates Dynamic Window Approach (DWA) circular arc rollouts for reactive local obstacle avoidance."""
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image format: {e}")

    segmenter = get_segmenter()
    result = segmenter.predict(
        image,
        img_size=resolution,
        target_heading_deg=target_heading_deg,
    )

    dwa_data = result["telemetry"]["dwa_planner"]
    response: Dict[str, Any] = {
        "dwa_planner": {
            "selected_v_cm_s": dwa_data["selected_v_cm_s"],
            "selected_w_deg_s": dwa_data["selected_w_deg_s"],
            "steer_action": dwa_data["steer_action"],
            "dwa_status": dwa_data["dwa_status"],
            "heading_alignment_pct": dwa_data["heading_alignment_pct"],
            "clearance_margin_pct": dwa_data["clearance_margin_pct"],
            "candidate_count": dwa_data["candidate_count"],
        },
        "safety_status": result["telemetry"]["safety_status"],
    }
    if include_base64_overlay:
        response["dwa_trajectories_overlay"] = encode_image_base64(result["overlay_dwa_trajectories"])

    return JSONResponse(content=response)


@app.get("/v1/dust-accumulation")
def query_dust_accumulation(
    sol_number: int = Query(4125, ge=1, le=10000, description="Mars mission sol number"),
    last_cleaning_sol: int = Query(4000, ge=1, le=10000, description="Last convective vortex cleaning sol"),
    dust_tau: float = Query(0.5, ge=0.1, le=6.0, description="Atmospheric dust optical depth tau"),
    wind_speed_m_s: float = Query(8.0, ge=0.0, le=60.0, description="Martian wind speed in m/s")
):
    """Calculates camera optical window dust settling, contrast attenuation, and cleaning vortex probability."""
    from src.dust_accumulation import MartianDustAccumulationModel
    model = MartianDustAccumulationModel()
    return JSONResponse(
        content=model.evaluate_dust_deposition(
            sol_number=sol_number,
            last_cleaning_sol=last_cleaning_sol,
            dust_tau=dust_tau,
            wind_speed_m_s=wind_speed_m_s,
        )
    )


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
    resolution: int = Query(512, ge=64, le=1024, description="Inference resolution"),
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


@app.post("/v1/ros2-packet")
async def export_ros2_packet(
    file: UploadFile = File(...),
    resolution: int = Query(512, ge=64, le=1024, description="Inference resolution"),
    planning_resolution: int = Query(96, ge=32, le=192, description="OccupancyGrid & A* grid resolution")
):
    """Exports serialized ROS 2 Nav2 / rosbridge messages (OccupancyGrid, Path, Twist, MarkerArray)."""
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image format: {e}")

    segmenter = get_segmenter()
    seg_res = segmenter.predict(image, img_size=resolution)
    pred_mask = seg_res["prediction_mask"]
    telemetry = seg_res["telemetry"]

    planner = get_planner()
    plan_data = planner.plan_path(pred_mask, planning_resolution=planning_resolution)

    kin_planner = get_kinematic_planner()
    eff_speed = telemetry["power_budget"]["effective_ground_speed_cm_s"]
    kinematics = kin_planner.profile_trajectory(
        plan_data,
        image_shape=pred_mask.shape,
        effective_speed_cm_s=eff_speed,
        pred_mask=pred_mask,
    )

    small_mask = np.array(
        Image.fromarray(pred_mask.astype(np.uint8)).resize(
            (planning_resolution, planning_resolution), Image.NEAREST
        ),
        dtype=np.int64,
    )
    bridge = Ros2NavBridge(
        map_resolution_meters=round(0.05 * (pred_mask.shape[0] / max(1, planning_resolution)), 4)
    )
    waypoints_for_ros = kinematics["smoothed_waypoints"] or plan_data["waypoints"]

    payload = {
        "topics": {
            "/autonav/costmap": bridge.create_occupancy_grid_msg(small_mask),
            "/autonav/global_plan": bridge.create_nav_path_msg(waypoints_for_ros, pred_mask.shape),
            "/cmd_vel": bridge.create_cmd_vel_msg(kinematics, eff_speed),
            "/aegis/markers": bridge.create_aegis_markers_msg(telemetry["science_targets"], pred_mask.shape),
            "/autonav/telemetry": bridge.create_terrain_telemetry_msg(
                telemetry["terrain_distribution_pct"], telemetry["traversability_index"]
            ),
        },
        "meta": {
            "safety_status": telemetry["safety_status"],
            "steering_maneuver": kinematics["steering_maneuver"],
            "sensor_health_status": telemetry["sensor_health"]["sensor_health_status"],
            "optical_trust_score": telemetry["sensor_health"]["optical_trust_score"],
        },
    }
    return JSONResponse(content=payload)


@app.post("/v1/simulate-sol")
def simulate_sol_mission(request: SolSimRequest):
    """Simulates multi-waypoint Mars Sol MMRTG battery SoC cycles, odometry drift, and safety aborts."""
    sim = SolMissionSimulator(
        initial_soc_pct=request.initial_soc_pct,
        min_reserve_soc_pct=request.min_reserve_soc_pct,
        max_consecutive_halts=request.max_consecutive_halts,
    )
    return JSONResponse(content=sim.simulate_sol_traverse(request.steps))



