"""
Autonomous Rover Steering Kinematics & Trajectory Profiler.
Simplifies jagged grid A* waypoints via Ramer-Douglas-Peucker polyline reduction,
computes Ackermann steering commands, turn curvature metrics, corridor clearance,
and physical drive duration estimates for 6-wheel Mars rovers.
"""
import math
from typing import List, Tuple, Dict, Any
import numpy as np


def _point_line_distance(
    pt: Tuple[int, int],
    start: Tuple[int, int],
    end: Tuple[int, int]
) -> float:
    if start == end:
        return math.hypot(pt[0] - start[0], pt[1] - start[1])
    num = abs((end[1] - start[1]) * pt[0] - (end[0] - start[0]) * pt[1] + end[0] * start[1] - end[1] * start[0])
    den = math.hypot(end[0] - start[0], end[1] - start[1])
    return num / max(den, 1e-8)


def rdp_simplify(waypoints: List[Tuple[int, int]], epsilon: float = 5.0) -> List[Tuple[int, int]]:
    """
    Reduces dense A* grid pixels into clean physical rover drive leg anchors using Ramer-Douglas-Peucker.
    """
    if len(waypoints) <= 2:
        return list(waypoints)

    start, end = waypoints[0], waypoints[-1]
    max_dist = -1.0
    max_idx = 0

    for i in range(1, len(waypoints) - 1):
        d = _point_line_distance(waypoints[i], start, end)
        if d > max_dist:
            max_dist = d
            max_idx = i

    if max_dist > epsilon:
        left = rdp_simplify(waypoints[: max_idx + 1], epsilon)
        right = rdp_simplify(waypoints[max_idx:], epsilon)
        return left[:-1] + right
    return [start, end]


class RoverKinematicPlanner:
    """
    Translates pixel-space A* trajectories into physical rover steering commands and timing.
    """
    def __init__(self, field_depth_m: float = 25.0, rdp_epsilon_ratio: float = 0.015):
        self.field_depth_m = field_depth_m
        self.rdp_epsilon_ratio = rdp_epsilon_ratio

    def profile_trajectory(
        self,
        path_data: Dict[str, Any],
        image_shape: Tuple[int, int],
        effective_speed_cm_s: float = 4.5,
        pred_mask: np.ndarray | None = None
    ) -> Dict[str, Any]:
        """
        Computes smoothed waypoints, initial steering heading, cumulative turn angle,
        drive time, and corridor safety margin.
        """
        h, w = image_shape[:2]
        waypoints = path_data.get("waypoints", [])

        if path_data.get("status") != "PATH_ACQUIRED" or len(waypoints) < 2:
            return {
                "kinematic_status": "NO_VIABLE_ARC",
                "steering_maneuver": "HALT_NO_PATH",
                "initial_heading_deg": 0.0,
                "max_turn_angle_deg": 0.0,
                "cumulative_turn_deg": 0.0,
                "estimated_distance_m": 0.0,
                "estimated_drive_time_s": 0.0,
                "corridor_safety_margin": "BLOCKED",
                "smoothed_waypoints": [],
                "segment_count": 0,
            }

        epsilon = max(3.0, min(h, w) * self.rdp_epsilon_ratio)
        smoothed = rdp_simplify(waypoints, epsilon=epsilon)

        # Estimate physical scale (~field_depth_m over vertical image span)
        meters_per_px = self.field_depth_m / max(1, h)
        total_px_len = 0.0
        segment_headings = []

        for idx in range(len(smoothed) - 1):
            r1, c1 = smoothed[idx]
            r2, c2 = smoothed[idx + 1]
            dr = r1 - r2  # positive moving upward toward horizon (forward rover +X axis)
            dc = c2 - c1  # positive moving right (+Y lateral steering)
            seg_len = math.hypot(dr, dc)
            if seg_len < 1e-5:
                continue
            total_px_len += seg_len
            heading_deg = math.degrees(math.atan2(dc, max(1e-5, abs(dr)) if dr >= 0 else dr))
            segment_headings.append(heading_deg)

        est_dist_m = round(total_px_len * meters_per_px, 2)
        initial_heading = round(segment_headings[0], 1) if segment_headings else 0.0

        turn_deltas = [
            abs(segment_headings[i + 1] - segment_headings[i])
            for i in range(len(segment_headings) - 1)
        ]
        max_turn = round(max(turn_deltas), 1) if turn_deltas else round(abs(initial_heading), 1)
        cumulative_turn = round(sum(turn_deltas) + abs(initial_heading), 1)

        # Classify primary initial maneuver
        if abs(initial_heading) <= 8.0 and max_turn <= 15.0:
            maneuver = "STRAIGHT_CRUISE"
        elif initial_heading < -22.0 or (initial_heading < 0 and max_turn > 35.0):
            maneuver = "HARD_EVASION_LEFT"
        elif initial_heading > 22.0 or (initial_heading > 0 and max_turn > 35.0):
            maneuver = "HARD_EVASION_RIGHT"
        elif initial_heading < 0:
            maneuver = "SLIGHT_LEFT_ARC"
        else:
            maneuver = "SLIGHT_RIGHT_ARC"

        # Traverse duration at slip-adjusted speed
        if effective_speed_cm_s > 0.05:
            speed_m_s = effective_speed_cm_s / 100.0
            drive_time_s = round(est_dist_m / speed_m_s, 1)
        else:
            drive_time_s = 0.0

        # Corridor hazard exposure check along full waypoints if mask provided
        corridor_margin = "NOMINAL_CLEARANCE"
        if pred_mask is not None and len(waypoints) > 0:
            step = max(1, len(waypoints) // 30)
            sampled = waypoints[::step]
            sand_hits = 0
            for r, c in sampled:
                rr = int(np.clip(r, 0, h - 1))
                cc = int(np.clip(c, 0, w - 1))
                if pred_mask[rr, cc] == 2:
                    sand_hits += 1
            sand_frac = sand_hits / len(sampled)
            if sand_frac > 0.25:
                corridor_margin = "SAND_EXPOSURE_WARNING"
            elif max_turn > 45.0:
                corridor_margin = "TIGHT_CHICANE_MANEUVER"

        return {
            "kinematic_status": "TRAJECTORY_FEASIBLE",
            "steering_maneuver": maneuver,
            "initial_heading_deg": initial_heading,
            "max_turn_angle_deg": max_turn,
            "cumulative_turn_deg": cumulative_turn,
            "estimated_distance_m": est_dist_m,
            "estimated_drive_time_s": drive_time_s,
            "corridor_safety_margin": corridor_margin,
            "smoothed_waypoints": smoothed,
            "segment_count": max(0, len(smoothed) - 1),
        }
