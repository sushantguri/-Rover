"""
Horizon Range-Zone & Boulder High-Centering Geometric Risk Analyzer.
Partitions rover Navcam imagery into perspective distance bands (0-5m Near-Field,
5-15m Mid-Range, 15-30m Horizon), clusters contiguous Big Rock obstacles via connected
components to assess Curiosity/Perseverance belly high-centering risk, and renders a
zone-annotated tactical radar overlay.
"""
from collections import deque
from typing import Dict, Any, List, Tuple
import numpy as np
from PIL import Image, ImageDraw

from src.dataset import CLASSES

# Downsampled resolution for fast pure-NumPy/Python BFS connected components
_CLUSTER_GRID = 160


class TerrainGeometryAnalyzer:
    """
    Computes perspective range-zone composition, obstacle clusters, and terrain roughness.
    """
    def __init__(
        self,
        near_split: float = 0.65,
        mid_split: float = 0.30,
        min_cluster_area_pct: float = 0.15
    ):
        self.near_split = near_split
        self.mid_split = mid_split
        self.min_cluster_area_pct = min_cluster_area_pct

    def _zone_stats(self, zone_mask: np.ndarray) -> Dict[str, Any]:
        total = max(1, zone_mask.size)
        stats: Dict[str, float] = {}
        for c_id, c_name in enumerate(CLASSES):
            key = f"{c_name.replace(' ', '_')}_pct"
            stats[key] = round(float((zone_mask == c_id).sum() / total * 100.0), 2)

        trav = (
            stats["bedrock_pct"] * 1.0
            + stats["soil_pct"] * 0.85
            + stats["sand_pct"] * 0.15
            + stats["big_rock_pct"] * 0.0
        )
        stats["traversability_index"] = round(float(np.clip(trav, 0.0, 100.0)), 1)
        return stats

    def _detect_boulder_clusters(self, pred_mask: np.ndarray) -> List[Dict[str, Any]]:
        orig_h, orig_w = pred_mask.shape
        grid_h = min(orig_h, _CLUSTER_GRID)
        grid_w = min(orig_w, _CLUSTER_GRID)

        if (grid_h, grid_w) != (orig_h, orig_w):
            pil_small = Image.fromarray(pred_mask.astype(np.uint8)).resize((grid_w, grid_h), Image.NEAREST)
            small_mask = np.array(pil_small, dtype=np.uint8)
        else:
            small_mask = pred_mask

        rock_binary = (small_mask == 3)
        visited = np.zeros_like(rock_binary, dtype=bool)
        min_grid_px = max(2, int((self.min_cluster_area_pct / 100.0) * grid_h * grid_w))

        scale_r = orig_h / grid_h
        scale_c = orig_w / grid_w
        near_grid_row = int(self.near_split * grid_h)
        clusters: List[Dict[str, Any]] = []

        rock_ys, rock_xs = np.where(rock_binary)
        for sy, sx in zip(rock_ys.tolist(), rock_xs.tolist()):
            if visited[sy, sx]:
                continue

            queue = deque([(sy, sx)])
            visited[sy, sx] = True
            min_r = max_r = sy
            min_c = max_c = sx
            area = 0

            while queue:
                r, c = queue.popleft()
                area += 1
                if r < min_r:
                    min_r = r
                if r > max_r:
                    max_r = r
                if c < min_c:
                    min_c = c
                if c > max_c:
                    max_c = c

                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < grid_h and 0 <= nc < grid_w:
                        if rock_binary[nr, nc] and not visited[nr, nc]:
                            visited[nr, nc] = True
                            queue.append((nr, nc))

            if area >= min_grid_px:
                r1 = int(min_r * scale_r)
                c1 = int(min_c * scale_c)
                r2 = min(orig_h - 1, int((max_r + 1) * scale_r))
                c2 = min(orig_w - 1, int((max_c + 1) * scale_c))
                span_px = int(max(r2 - r1, c2 - c1))
                in_nearfield = max_r >= near_grid_row
                clusters.append({
                    "bbox_xyxy": [c1, r1, c2, r2],
                    "area_pct": round(float(area / (grid_h * grid_w) * 100.0), 2),
                    "span_px": span_px,
                    "span_ratio_to_frame": round(float(span_px / max(1, orig_w)), 3),
                    "in_nearfield": bool(in_nearfield),
                })

        clusters.sort(key=lambda x: x["area_pct"], reverse=True)
        return clusters

    def analyze(self, pred_mask: np.ndarray) -> Dict[str, Any]:
        """
        Analyzes perspective range zones, boulder clustering, and terrain boundary roughness.
        """
        h, w = pred_mask.shape
        r_mid = max(1, int(self.mid_split * h))
        r_near = max(r_mid + 1, int(self.near_split * h))

        far_zone = pred_mask[:r_mid, :]
        mid_zone = pred_mask[r_mid:r_near, :]
        near_zone = pred_mask[r_near:, :]

        zones = {
            "far_horizon_15_30m": self._zone_stats(far_zone),
            "mid_range_5_15m": self._zone_stats(mid_zone),
            "near_field_0_5m": self._zone_stats(near_zone),
        }

        # Terrain boundary heterogeneity / roughness index [0, 100]
        h_diff = np.mean(pred_mask[:, 1:] != pred_mask[:, :-1]) if w > 1 else 0.0
        v_diff = np.mean(pred_mask[1:, :] != pred_mask[:-1, :]) if h > 1 else 0.0
        edge_density = 0.5 * (float(h_diff) + float(v_diff))
        # Normalize typical Navcam transition density (~0.15 = rugged boundary terrain) onto [0, 100]
        roughness_index = round(float(np.clip(edge_density / 0.15 * 100.0, 0.0, 100.0)), 1)

        clusters = self._detect_boulder_clusters(pred_mask)
        max_span_ratio = max((c["span_ratio_to_frame"] for c in clusters), default=0.0)
        near_boulders = sum(1 for c in clusters if c["in_nearfield"])
        near_rock_pct = zones["near_field_0_5m"]["big_rock_pct"]
        near_sand_pct = zones["near_field_0_5m"]["sand_pct"]

        # Chassis belly high-centering assessment
        if (near_boulders > 0 and max_span_ratio >= 0.18) or near_rock_pct >= 8.0:
            high_centering_risk = "CRITICAL"
        elif len(clusters) > 0 and (max_span_ratio >= 0.10 or near_rock_pct >= 2.0):
            high_centering_risk = "ELEVATED"
        else:
            high_centering_risk = "LOW"

        # Immediate front-wheel hazard vs distant sand distinction
        if near_sand_pct >= 25.0 or high_centering_risk == "CRITICAL":
            immediate_zone_alert = "IMMEDIATE_NEARFIELD_OBSTRUCTION"
        elif zones["mid_range_5_15m"]["sand_pct"] >= 25.0 or high_centering_risk == "ELEVATED":
            immediate_zone_alert = "MIDRANGE_DETOUR_ADVISORY"
        else:
            immediate_zone_alert = "NEARFIELD_CORRIDOR_CLEAR"

        return {
            "zones": zones,
            "nearfield_sand_pct": near_sand_pct,
            "nearfield_big_rock_pct": near_rock_pct,
            "horizon_sand_pct": zones["far_horizon_15_30m"]["sand_pct"],
            "terrain_roughness_index": roughness_index,
            "boulder_cluster_count": len(clusters),
            "nearfield_boulder_count": near_boulders,
            "max_boulder_span_ratio": round(float(max_span_ratio), 3),
            "high_centering_risk": high_centering_risk,
            "immediate_zone_alert": immediate_zone_alert,
            "boulder_clusters": clusters[:8],
        }

    def render_zone_radar_overlay(
        self,
        orig_np: np.ndarray,
        pred_mask: np.ndarray,
        analysis: Dict[str, Any],
        alpha: float = 0.45
    ) -> np.ndarray:
        """
        Renders perspective range band lines (0-5m, 5-15m, 15-30m), near-field hazard
        highlighting, and detected boulder obstacle bounding boxes.
        """
        h, w, _ = orig_np.shape
        r_mid = max(1, int(self.mid_split * h))
        r_near = max(r_mid + 1, int(self.near_split * h))

        base = orig_np.copy().astype(np.float32)

        # Highlight near-field sand & rock hazards vividly, mute safe areas lightly
        near_mask = np.zeros((h, w), dtype=bool)
        near_mask[r_near:, :] = True
        near_sand = near_mask & (pred_mask == 2)
        near_rock = near_mask & (pred_mask == 3)

        if np.any(near_sand):
            base[near_sand] = (1.0 - alpha) * base[near_sand] + alpha * np.array([255.0, 140.0, 0.0])
        if np.any(near_rock):
            base[near_rock] = (1.0 - alpha) * base[near_rock] + alpha * np.array([239.0, 68.0, 68.0])

        img_pil = Image.fromarray(np.clip(base, 0, 255).astype(np.uint8)).convert("RGBA")
        overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        # Dashed perspective divider lines for 15-30m and 5-15m bands
        dash_len = max(8, w // 32)
        for y, col in ((r_mid, (56, 189, 248, 200)), (r_near, (245, 158, 11, 220))):
            for x in range(0, w, dash_len * 2):
                draw.line([(x, y), (min(w, x + dash_len), y)], fill=col, width=2)

        # Zone HUD tags on left margin
        zones = analysis["zones"]
        tags = [
            (8, 8, f"FAR 15-30m | Sand {zones['far_horizon_15_30m']['sand_pct']:.0f}%", (56, 189, 248, 220)),
            (8, r_mid + 6, f"MID 5-15m | Sand {zones['mid_range_5_15m']['sand_pct']:.0f}%", (16, 185, 129, 220)),
            (8, r_near + 6, f"NEAR 0-5m | Sand {zones['near_field_0_5m']['sand_pct']:.0f}% | Risk: {analysis['high_centering_risk']}", (245, 158, 11, 235)),
        ]
        for tx, ty, label, color in tags:
            pad_w = min(w - 16, max(130, len(label) * 7 + 12))
            draw.rectangle([(tx, ty), (tx + pad_w, ty + 18)], fill=(15, 23, 42, 185), outline=color, width=1)
            draw.text((tx + 6, ty + 3), label, fill=(248, 250, 252, 255))

        # Boulder bounding boxes
        for idx, cluster in enumerate(analysis.get("boulder_clusters", [])[:5], start=1):
            c1, r1, c2, r2 = cluster["bbox_xyxy"]
            box_col = (239, 68, 68, 245) if cluster["in_nearfield"] else (251, 146, 60, 210)
            c2_clamped = max(c1, c2)
            r2_clamped = max(r1, r2)
            draw.rectangle([(c1, r1), (c2_clamped, r2_clamped)], outline=box_col, width=3)
            tag_y = max(2, r1 - 15)
            bx1 = max(0, min(w - 2, c1))
            bx2 = max(bx1 + 2, min(w, bx1 + 78))
            draw.rectangle([(bx1, tag_y), (bx2, tag_y + 14)], fill=(127, 29, 29, 220))
            draw.text((bx1 + 4, tag_y + 2), f"ROCK #{idx}", fill=(255, 255, 255, 255))

        combined = Image.alpha_composite(img_pil, overlay).convert("RGB")
        return np.array(combined)
