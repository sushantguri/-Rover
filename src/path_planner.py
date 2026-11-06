"""
Autonomous Path Planning Engine for Mars Rovers.
Implements an 8-connected A* (A-Star) search algorithm over the neural terrain costmap.
Computes minimum-energy, collision-free waypoint trajectories avoiding sand dunes and rock obstacles.
"""
import heapq
import numpy as np
from PIL import Image, ImageDraw
from typing import List, Tuple, Dict, Any, Optional

# Cost coefficients corresponding to classes:
# 0: Soil, 1: Bedrock, 2: Sand (hazard), 3: Big Rock (obstacle)
TERRAIN_COST_WEIGHTS = {
    0: 2.0,       # Soil: Low cost, nominal friction
    1: 1.0,       # Bedrock: Minimum cost, optimal traction
    2: 18.0,      # Sand: Severe penalty (high wheel slip, risk of entrapment)
    3: 1000.0,    # Big Rock: Lethal obstacle
}


class Node:
    __slots__ = ('r', 'c', 'g', 'h', 'f', 'parent')
    def __init__(self, r: int, c: int, g: float = 0.0, h: float = 0.0, parent=None):
        self.r = r
        self.c = c
        self.g = g
        self.h = h
        self.f = g + h
        self.parent = parent

    def __lt__(self, other):
        return self.f < other.f


class MarsPathPlanner:
    """
    Path planner utilizing terrain cost maps derived from DeepLabV3+ segmentation.
    """
    def __init__(self, obstacle_inflation_radius: int = 2):
        self.inflation_radius = obstacle_inflation_radius
        # 8-connected grid movement: (dr, dc, move_cost_multiplier)
        self.moves = [
            (-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),      # Orthogonal
            (-1, -1, 1.414), (-1, 1, 1.414), (1, -1, 1.414), (1, 1, 1.414) # Diagonal
        ]

    def build_cost_grid(self, pred_mask: np.ndarray) -> np.ndarray:
        """
        Converts class ID prediction mask into a 2D traversal cost matrix with obstacle inflation.
        """
        h, w = pred_mask.shape
        cost_grid = np.zeros((h, w), dtype=np.float32)

        # Base cost assignment
        for class_id, weight in TERRAIN_COST_WEIGHTS.items():
            cost_grid[pred_mask == class_id] = weight

        # Unlabeled / ignored regions (255) treated as untraversable
        cost_grid[pred_mask == 255] = 1000.0

        # Obstacle inflation: Inflate big rocks (class 3) to protect rover wheel chassis
        if self.inflation_radius > 0:
            rock_coords = np.argwhere(pred_mask == 3)
            for r, c in rock_coords:
                r_min = max(0, r - self.inflation_radius)
                r_max = min(h, r + self.inflation_radius + 1)
                c_min = max(0, c - self.inflation_radius)
                c_max = min(w, c + self.inflation_radius + 1)
                cost_grid[r_min:r_max, c_min:c_max] = np.maximum(
                    cost_grid[r_min:r_max, c_min:c_max], 
                    500.0  # Inflated buffer cost
                )

        return cost_grid

    def heuristic(self, r1: int, c1: int, r2: int, c2: int) -> float:
        """Euclidean distance heuristic for A* search."""
        return float(np.sqrt((r1 - r2) ** 2 + (c1 - c2) ** 2))

    def plan_path(
        self, 
        pred_mask: np.ndarray, 
        start: Optional[Tuple[int, int]] = None, 
        goal: Optional[Tuple[int, int]] = None,
        planning_resolution: int = 128
    ) -> Dict[str, Any]:
        """
        Plans optimal trajectory from start to goal.
        Downsamples grid to planning_resolution for sub-second planning, then scales path back.
        """
        orig_h, orig_w = pred_mask.shape

        # Resize mask to fast planning resolution
        pil_mask = Image.fromarray(pred_mask.astype(np.uint8)).resize(
            (planning_resolution, planning_resolution), Image.NEAREST
        )
        plan_mask = np.array(pil_mask, dtype=np.int64)
        cost_grid = self.build_cost_grid(plan_mask)

        gh, gw = plan_mask.shape
        scale_r = orig_h / gh
        scale_c = orig_w / gw

        # Default start: bottom-center (where the rover is positioned)
        if start is None:
            start_r, start_c = gh - 4, gw // 2
        else:
            start_r = int(np.clip(start[0] / scale_r, 0, gh - 1))
            start_c = int(np.clip(start[1] / scale_c, 0, gw - 1))

        # Default goal: top horizon center
        if goal is None:
            goal_r, goal_c = 4, gw // 2
        else:
            goal_r = int(np.clip(goal[0] / scale_r, 0, gh - 1))
            goal_c = int(np.clip(goal[1] / scale_c, 0, gw - 1))

        # Priority queue for A*
        open_set = []
        heapq.heappush(open_set, (0.0, Node(start_r, start_c, 0.0, self.heuristic(start_r, start_c, goal_r, goal_c))))
        
        # Best g-costs
        g_scores = { (start_r, start_c): 0.0 }
        closed_set = set()

        path_found = False
        goal_node = None

        while open_set:
            _, current = heapq.heappop(open_set)
            curr_coord = (current.r, current.c)

            if curr_coord in closed_set:
                continue
            closed_set.add(curr_coord)

            # Check if within tolerance of goal
            if abs(current.r - goal_r) <= 2 and abs(current.c - goal_c) <= 2:
                path_found = True
                goal_node = current
                break

            for dr, dc, mult in self.moves:
                nr, nc = current.r + dr, current.c + dc
                if not (0 <= nr < gh and 0 <= nc < gw):
                    continue

                cell_cost = cost_grid[nr, nc]
                if cell_cost >= 1000.0:  # Absolute obstacle
                    continue

                tentative_g = current.g + (cell_cost * mult)
                neighbor_coord = (nr, nc)

                if neighbor_coord in g_scores and tentative_g >= g_scores[neighbor_coord]:
                    continue

                g_scores[neighbor_coord] = tentative_g
                h = self.heuristic(nr, nc, goal_r, goal_c)
                neighbor_node = Node(nr, nc, tentative_g, h, parent=current)
                heapq.heappush(open_set, (neighbor_node.f, neighbor_node))

        # Reconstruct path
        path_pixels = []
        total_energy_cost = 0.0

        if path_found and goal_node is not None:
            curr = goal_node
            while curr is not None:
                # Scale coordinates back to original resolution
                real_r = int(curr.r * scale_r)
                real_c = int(curr.c * scale_c)
                path_pixels.append((real_r, real_c))
                curr = curr.parent
            path_pixels.reverse()
            total_energy_cost = float(goal_node.g)
            status = "PATH_ACQUIRED"
        else:
            status = "PATH_BLOCKED_BY_OBSTACLES"

        return {
            "status": status,
            "waypoints": path_pixels,
            "waypoint_count": len(path_pixels),
            "total_path_cost": round(total_energy_cost, 2),
            "start": (int(start_r * scale_r), int(start_c * scale_c)),
            "goal": (int(goal_r * scale_r), int(goal_c * scale_c)),
        }

    def render_path_overlay(
        self, 
        base_image: np.ndarray, 
        path_data: Dict[str, Any], 
        path_color: Tuple[int, int, int] = (0, 230, 118) # Neon Green
    ) -> np.ndarray:
        """
        Draws the planned trajectory with waypoint nodes onto the rover camera image.
        """
        img_pil = Image.fromarray(base_image.copy()).convert("RGBA")
        overlay = Image.new("RGBA", img_pil.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        waypoints = path_data.get("waypoints", [])
        if len(waypoints) > 1:
            # Draw smooth trajectory line
            line_coords = [(c, r) for r, c in waypoints]
            draw.line(line_coords, fill=path_color + (230,), width=5)

            # Draw start marker (Rover Position)
            sr, sc = path_data["start"]
            draw.ellipse([(sc - 8, sr - 8), (sc + 8, sr + 8)], fill=(0, 176, 255, 255), outline=(255, 255, 255, 255), width=2)

            # Draw goal marker (Destination)
            gr, gc = path_data["goal"]
            draw.ellipse([(gc - 8, gr - 8), (gc + 8, gr + 8)], fill=(255, 61, 0, 255), outline=(255, 255, 255, 255), width=2)

            # Draw periodic waypoint dots along the path
            for i, (r, c) in enumerate(waypoints[::max(1, len(waypoints) // 8)]):
                if i != 0:
                    draw.ellipse([(c - 4, r - 4), (c + 4, r + 4)], fill=(255, 255, 255, 220))

        combined = Image.alpha_composite(img_pil, overlay).convert("RGB")
        return np.array(combined)
