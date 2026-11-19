"""
NASA JPL Mars Rover Dynamic Window Approach (DWA) Local Reactive Planner.
Evaluates admissible velocity space (v, w) under Martian rover kinematic acceleration limits,
simulating forward circular arc rollouts across segmented terrain hazards (Sand traps, Big Rocks,
and 3D step obstacles) to select optimal collision-free steering commands.

References:
  - Fox, Burgard, Thrun, "The Dynamic Window Approach to Collision Avoidance", IEEE RAM, 1997.
  - Biesiadecki & Maimone, "The Mars Exploration Rover Surface Mobility Flight Software",
    IEEE Aerospace Conference, 2006.
"""
import math
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
from PIL import Image, ImageDraw


class RoverDWALocalPlanner:
    """
    Dynamic Window Approach (DWA) local planner for Mars rovers.
    Simulates short-horizon circular arc rollouts across candidate lookahead paths
    to steer around obstacles toward the global goal waypoint.
    """

    def __init__(
        self,
        max_speed_cm_s: float = 4.5,
        min_speed_cm_s: float = 0.0,
        max_yaw_rate_deg_s: float = 12.0,
        max_accel_cm_s2: float = 2.0,
        max_yaw_accel_deg_s2: float = 6.0,
        lookahead_m: float = 3.5,
        n_arc_steps: int = 16,
        rover_footprint_radius_m: float = 1.1,
    ):
        self.max_speed_cm_s = max_speed_cm_s
        self.min_speed_cm_s = min_speed_cm_s
        self.max_yaw_rate_deg_s = max_yaw_rate_deg_s
        self.max_accel_cm_s2 = max_accel_cm_s2
        self.max_yaw_accel_deg_s2 = max_yaw_accel_deg_s2
        self.lookahead_m = lookahead_m
        self.n_arc_steps = n_arc_steps
        self.rover_footprint_radius_m = rover_footprint_radius_m

    def compute_dynamic_window(
        self, current_v_cm_s: float, current_w_deg_s: float
    ) -> Tuple[float, float, float, float]:
        """
        Calculates admissible velocity boundaries [vs_min, vs_max, ws_min, ws_max]
        factoring in rover motor physical acceleration capabilities.
        """
        # Linear velocity window
        vs_min = max(self.min_speed_cm_s, current_v_cm_s - self.max_accel_cm_s2 * 1.0)
        vs_max = min(self.max_speed_cm_s, current_v_cm_s + self.max_accel_cm_s2 * 1.0)

        # Angular yaw rate window
        ws_min = max(-self.max_yaw_rate_deg_s, current_w_deg_s - self.max_yaw_accel_deg_s2 * 1.0)
        ws_max = min(self.max_yaw_rate_deg_s, current_w_deg_s + self.max_yaw_accel_deg_s2 * 1.0)

        return vs_min, vs_max, ws_min, ws_max

    def generate_candidate_trajectories(
        self,
        current_v_cm_s: float,
        current_w_deg_s: float,
        n_v_samples: int = 3,
        n_w_samples: int = 11,
    ) -> List[Dict[str, Any]]:
        """
        Discretizes the dynamic window into candidate velocity pairs (v, w)
        and integrates forward circular arc rollouts extending over lookahead_m.
        """
        vs_min, vs_max, ws_min, ws_max = self.compute_dynamic_window(current_v_cm_s, current_w_deg_s)
        v_samples = np.linspace(vs_min, vs_max, n_v_samples)
        w_samples = np.linspace(ws_min, ws_max, n_w_samples)

        candidates = []
        ds = self.lookahead_m / float(self.n_arc_steps)

        for v in v_samples:
            for w in w_samples:
                # Curvature kappa = w / v (in rad/m)
                # When v is very small, use effective scale
                v_eff_m_s = max(0.01, (v / 100.0))
                w_rad_s = math.radians(w)
                curvature = w_rad_s / v_eff_m_s

                traj = []
                x, y, theta = 0.0, 0.0, 0.0
                for _ in range(self.n_arc_steps):
                    x += ds * math.cos(theta)
                    y += ds * math.sin(theta)
                    theta += curvature * ds
                    traj.append((x, y, theta))

                candidates.append(
                    {
                        "v_cm_s": round(float(v), 2),
                        "w_deg_s": round(float(w), 2),
                        "trajectory": traj,
                        "final_pose": (x, y, theta),
                    }
                )

        return candidates

    def plan_local_twist(
        self,
        pred_mask: np.ndarray,
        target_heading_deg: float = 0.0,
        current_v_cm_s: float = 3.0,
        current_w_deg_s: float = 0.0,
    ) -> Dict[str, Any]:
        """
        Evaluates candidate DWA trajectories against segmented terrain classes,
        scoring each by heading alignment, terrain clearance, and forward speed.
        """
        h, w = pred_mask.shape[:2]
        candidates = self.generate_candidate_trajectories(current_v_cm_s, current_w_deg_s)

        target_theta_rad = math.radians(target_heading_deg)
        best_candidate = None
        best_score = -float("inf")
        scored_trajectories = []

        # Terrain cost map: Rock=lethal (1000), Sand=high (40), Bedrock/Soil=low (1)
        for cand in candidates:
            traj = cand["trajectory"]
            x_fin, y_fin, th_fin = cand["final_pose"]

            # 1. Heading score: alignment with target waypoint
            angle_diff = abs(th_fin - target_theta_rad)
            heading_score = max(0.0, 1.0 - (angle_diff / math.pi))

            # 2. Obstacle clearance score & collision detection
            min_dist_to_rock = 999.0
            sand_exposure = 0.0
            is_collision = False

            # Convert (x, y) robot coordinates to normalized image pixel coordinates
            # Rover camera looks forward: x in [0, 4m], y in [-2m, 2m]
            for (px, py, _) in traj:
                if px <= 0.0:
                    continue
                # Mapping: bottom center is rover base (h-1, w//2)
                # Forward x maps to row from (h-1) up to (h * 0.3)
                img_r = int((h - 1) - (px / 4.0) * (h * 0.65))
                img_c = int((w // 2) - (py / 2.0) * (w * 0.45))

                if 0 <= img_r < h and 0 <= img_c < w:
                    cls_id = pred_mask[img_r, img_c]
                    if cls_id == 3:  # Big Rock
                        is_collision = True
                        min_dist_to_rock = 0.0
                        break
                    elif cls_id == 2:  # Sand
                        sand_exposure += 1.0

            if is_collision:
                clearance_score = 0.0
            else:
                clearance_score = max(0.0, 1.0 - (sand_exposure / max(1.0, len(traj))))

            # 3. Speed score: incentivize forward progress
            speed_score = cand["v_cm_s"] / max(0.1, self.max_speed_cm_s)

            # Combined objective function (alpha=0.45, beta=0.40, gamma=0.15)
            if is_collision:
                total_score = -100.0
            else:
                total_score = 0.45 * heading_score + 0.40 * clearance_score + 0.15 * speed_score

            cand_result = {
                "v_cm_s": cand["v_cm_s"],
                "w_deg_s": cand["w_deg_s"],
                "score": round(float(total_score), 3),
                "is_collision": is_collision,
                "heading_score": round(float(heading_score), 3),
                "clearance_score": round(float(clearance_score), 3),
                "trajectory": traj,
            }
            scored_trajectories.append(cand_result)

            if total_score > best_score:
                best_score = total_score
                best_candidate = cand_result

        # Fallback if all paths collide: emergency stop
        if best_candidate is None or best_candidate["score"] < 0.0:
            best_candidate = {
                "v_cm_s": 0.0,
                "w_deg_s": 0.0,
                "score": 0.0,
                "is_collision": True,
                "heading_score": 0.0,
                "clearance_score": 0.0,
                "trajectory": [(0.0, 0.0, 0.0)],
            }
            dwa_status = "DWA_EMERGENCY_STOP"
            steer_action = "HALT_NO_ADMISSIBLE_PATH"
        else:
            w_opt = best_candidate["w_deg_s"]
            if abs(w_opt) < 2.0:
                steer_action = "STRAIGHT_FORWARD"
            elif w_opt > 0.0:
                steer_action = f"ARC_LEFT_{abs(w_opt):.1f}_DEG_S"
            else:
                steer_action = f"ARC_RIGHT_{abs(w_opt):.1f}_DEG_S"
            dwa_status = "DWA_TRAJECTORY_OPTIMAL"

        return {
            "selected_v_cm_s": best_candidate["v_cm_s"],
            "selected_w_deg_s": best_candidate["w_deg_s"],
            "dwa_score": best_candidate["score"],
            "steer_action": steer_action,
            "dwa_status": dwa_status,
            "heading_alignment_pct": round(best_candidate["heading_score"] * 100.0, 1),
            "clearance_margin_pct": round(best_candidate["clearance_score"] * 100.0, 1),
            "candidate_count": len(candidates),
            "best_trajectory": best_candidate["trajectory"],
            "all_trajectories": scored_trajectories,
        }

    def render_dwa_overlay(
        self,
        image_np: np.ndarray,
        pred_mask: np.ndarray,
        dwa_result: Dict[str, Any],
        alpha: float = 0.65,
    ) -> np.ndarray:
        """
        Renders multi-arc DWA candidate trajectories fan overlay onto the rover camera view,
        highlighting the optimal selected arc in Cyan and collided arcs in Red.
        """
        h, w = pred_mask.shape[:2]
        pil_img = Image.fromarray(image_np.copy()).convert("RGBA")
        overlay = Image.new("RGBA", pil_img.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        # Draw all candidate trajectories
        for cand in dwa_result.get("all_trajectories", []):
            traj = cand["trajectory"]
            pts = []
            for (px, py, _) in traj:
                if px <= 0.0:
                    continue
                img_r = int((h - 1) - (px / 4.0) * (h * 0.65))
                img_c = int((w // 2) - (py / 2.0) * (w * 0.45))
                if 0 <= img_r < h and 0 <= img_c < w:
                    pts.append((img_c, img_r))

            if len(pts) >= 2:
                if cand["is_collision"]:
                    line_color = (239, 68, 68, 80)     # Faint red (collision)
                    w_width = 1
                else:
                    line_color = (250, 204, 21, 90)    # Faint yellow (candidate)
                    w_width = 1
                draw.line(pts, fill=line_color, width=w_width)

        # Draw winning optimal trajectory in prominent Cyan
        best_pts = []
        for (px, py, _) in dwa_result.get("best_trajectory", []):
            if px <= 0.0:
                continue
            img_r = int((h - 1) - (px / 4.0) * (h * 0.65))
            img_c = int((w // 2) - (py / 2.0) * (w * 0.45))
            if 0 <= img_r < h and 0 <= img_c < w:
                best_pts.append((img_c, img_r))

        if len(best_pts) >= 2:
            draw.line(best_pts, fill=(6, 182, 212, 240), width=4)  # Cyan
            for pt in best_pts[::2]:
                draw.ellipse([pt[0] - 2, pt[1] - 2, pt[0] + 2, pt[1] + 2], fill=(255, 255, 255, 255))

        blended = Image.alpha_composite(pil_img, overlay)
        return np.array(blended.convert("RGB"))
