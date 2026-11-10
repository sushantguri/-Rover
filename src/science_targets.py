"""
NASA AEGIS-Inspired Autonomous Geological Science Target Prioritizer.
Autonomously detects high-value Astrobiology & Geology targets (exposed Bedrock Outcrops,
Bedrock-Soil Stratigraphic Contacts, and Accessible Standoff Boulders) for Mastcam-Z,
ChemCam, or SuperCam spectroscopic targeting while checking wheel-trap approach safety.
"""
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
from PIL import Image, ImageDraw

_SEARCH_GRID = 96


class AEGISTargetSelector:
    """
    Ranks top-K autonomous geological targets and renders optical targeting reticles.
    """
    def __init__(self, top_k: int = 3, min_separation_ratio: float = 0.18):
        self.top_k = top_k
        self.min_separation_ratio = min_separation_ratio

    def _window_fraction(self, binary_mask: np.ndarray, radius: int = 2) -> np.ndarray:
        """Fast integral-box filter returning local neighborhood fraction in [0, 1]."""
        h, w = binary_mask.shape
        padded = np.pad(binary_mask.astype(np.float32), radius, mode="edge")
        acc = np.zeros((h, w), dtype=np.float32)
        k = 2 * radius + 1
        for dr in range(k):
            for dc in range(k):
                acc += padded[dr : dr + h, dc : dc + w]
        return acc / float(k * k)

    def select_targets(
        self,
        pred_mask: np.ndarray,
        entropy_map: Optional[np.ndarray] = None,
        top_k: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Identifies and ranks candidate science targets with approach safety classification.
        """
        k = top_k if top_k is not None else self.top_k
        orig_h, orig_w = pred_mask.shape

        gh = min(orig_h, _SEARCH_GRID)
        gw = min(orig_w, _SEARCH_GRID)
        small_mask = np.array(
            Image.fromarray(pred_mask.astype(np.uint8)).resize((gw, gh), Image.NEAREST),
            dtype=np.int64,
        )

        if entropy_map is not None:
            ent_u8 = np.clip(entropy_map * 255.0, 0, 255).astype(np.uint8)
            small_ent = np.array(
                Image.fromarray(ent_u8).resize((gw, gh), Image.BILINEAR),
                dtype=np.float32,
            ) / 255.0
            conf_grid = 1.0 - np.clip(small_ent, 0.0, 1.0)
        else:
            conf_grid = np.ones((gh, gw), dtype=np.float32)

        # Exclude far sky/horizon border rows (top 12%) from close-range instrument targeting
        reach_mask = np.ones((gh, gw), dtype=bool)
        reach_mask[: max(1, int(0.12 * gh)), :] = False

        is_soil = (small_mask == 0)
        is_bedrock = (small_mask == 1)
        is_sand = (small_mask == 2)
        is_rock = (small_mask == 3)

        bedrock_density = self._window_fraction(is_bedrock, radius=2)
        soil_density = self._window_fraction(is_soil, radius=2)
        sand_proximity = self._window_fraction(is_sand, radius=3)
        rock_density = self._window_fraction(is_rock, radius=1)

        candidates: List[Dict[str, Any]] = []

        # 1. Bedrock Outcrops (optimal for abrasion/PIXL/SHERLOC contact science)
        outcrop_valid = is_bedrock & reach_mask & (bedrock_density >= 0.55)
        if np.any(outcrop_valid):
            # Penalize proximity to loose sand dunes
            score_map = (bedrock_density * 0.65 + conf_grid * 0.35 - sand_proximity * 0.50) * 100.0
            score_map[~outcrop_valid] = -1e6
            ys, xs = np.where(outcrop_valid)
            # Take best candidates sorted by score
            scores = score_map[ys, xs]
            order = np.argsort(scores)[::-1][:15]
            for idx in order:
                if scores[idx] <= 10.0:
                    continue
                candidates.append({
                    "gr": int(ys[idx]),
                    "gc": int(xs[idx]),
                    "category": "BEDROCK_OUTCROP",
                    "instrument_mode": "CONTACT_ARM / PIXL",
                    "score": float(np.clip(scores[idx], 0.0, 99.9)),
                    "sand_near": float(sand_proximity[ys[idx], xs[idx]]),
                })

        # 2. Stratigraphic Contact Boundaries (bedrock meeting soil layers)
        contact_valid = (is_bedrock | is_soil) & reach_mask & (bedrock_density >= 0.25) & (soil_density >= 0.25)
        if np.any(contact_valid):
            balance = 1.0 - np.abs(bedrock_density - soil_density)
            score_map = (balance * 0.60 + conf_grid * 0.40 - sand_proximity * 0.45) * 95.0
            score_map[~contact_valid] = -1e6
            ys, xs = np.where(contact_valid)
            scores = score_map[ys, xs]
            order = np.argsort(scores)[::-1][:15]
            for idx in order:
                if scores[idx] <= 15.0:
                    continue
                candidates.append({
                    "gr": int(ys[idx]),
                    "gc": int(xs[idx]),
                    "category": "STRATIGRAPHIC_CONTACT",
                    "instrument_mode": "MASTCAM_Z / SUPERCAM",
                    "score": float(np.clip(scores[idx], 0.0, 99.9)),
                    "sand_near": float(sand_proximity[ys[idx], xs[idx]]),
                })

        # 3. Accessible Float Rock / Boulder Specimens (ChemCam / SuperCam LIBS laser)
        rock_valid = is_rock & reach_mask & (rock_density >= 0.35)
        if np.any(rock_valid):
            firm_surround = np.clip(bedrock_density + soil_density, 0.0, 1.0)
            score_map = (rock_density * 0.40 + firm_surround * 0.30 + conf_grid * 0.30 - sand_proximity * 0.40) * 90.0
            score_map[~rock_valid] = -1e6
            ys, xs = np.where(rock_valid)
            scores = score_map[ys, xs]
            order = np.argsort(scores)[::-1][:15]
            for idx in order:
                if scores[idx] <= 15.0:
                    continue
                candidates.append({
                    "gr": int(ys[idx]),
                    "gc": int(xs[idx]),
                    "category": "FLOAT_ROCK_SPECIMEN",
                    "instrument_mode": "STANDOFF_CHEMCAM_LIBS",
                    "score": float(np.clip(scores[idx], 0.0, 99.9)),
                    "sand_near": float(sand_proximity[ys[idx], xs[idx]]),
                })

        # Spatial non-maximum suppression across categories
        candidates.sort(key=lambda c: c["score"], reverse=True)
        min_sep_grid = max(4.0, min(gh, gw) * self.min_separation_ratio)
        scale_r = orig_h / gh
        scale_c = orig_w / gw

        selected: List[Dict[str, Any]] = []
        for cand in candidates:
            too_close = any(
                np.hypot(cand["gr"] - prev["gr"], cand["gc"] - prev["gc"]) < min_sep_grid
                for prev in selected
            )
            if too_close:
                continue

            px_x = int(min(orig_w - 1, round((cand["gc"] + 0.5) * scale_c)))
            px_y = int(min(orig_h - 1, round((cand["gr"] + 0.5) * scale_r)))

            if cand["sand_near"] >= 0.25 or cand["category"] == "FLOAT_ROCK_SPECIMEN":
                approach = "STANDOFF_LASER_ONLY"
            else:
                approach = "SAFE_ROVER_APPROACH"

            selected.append({
                "rank": len(selected) + 1,
                "target_id": f"AEGIS-TGT-{len(selected) + 1}",
                "category": cand["category"],
                "recommended_instrument": cand["instrument_mode"],
                "priority_score": round(cand["score"], 1),
                "approach_safety": approach,
                "pixel_xy": [px_x, px_y],
                "normalized_xy": [round(px_x / max(1, orig_w), 3), round(px_y / max(1, orig_h), 3)],
                "gr": cand["gr"],
                "gc": cand["gc"],
            })
            if len(selected) >= k:
                break

        # Remove internal grid coordinates from public output
        public_targets = [
            {key: val for key, val in tgt.items() if key not in ("gr", "gc")}
            for tgt in selected
        ]

        return {
            "target_count": len(public_targets),
            "targets": public_targets,
            "aegis_status": "TARGETS_LOCKED" if public_targets else "NO_EXPOSED_GEOLOGY_FOUND",
        }

    def render_target_reticles(
        self,
        orig_np: np.ndarray,
        target_data: Dict[str, Any]
    ) -> np.ndarray:
        """
        Draws high-tech JPL AEGIS optical targeting reticles onto the original rover camera image.
        """
        img_pil = Image.fromarray(orig_np.copy()).convert("RGBA")
        w, h = img_pil.size
        overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        palette = {
            "BEDROCK_OUTCROP": (56, 189, 248, 255),       # Cyan
            "STRATIGRAPHIC_CONTACT": (250, 204, 21, 255), # Gold
            "FLOAT_ROCK_SPECIMEN": (244, 114, 182, 255),  # Magenta/Rose
        }

        radius = max(14, int(min(w, h) * 0.04))
        for tgt in target_data.get("targets", []):
            cx, cy = tgt["pixel_xy"]
            col = palette.get(tgt["category"], (56, 189, 248, 255))

            # Outer ring + crosshair ticks
            draw.ellipse([(cx - radius, cy - radius), (cx + radius, cy + radius)], outline=col, width=2)
            draw.ellipse([(cx - 4, cy - 4), (cx + 4, cy + 4)], fill=col)

            tick = max(6, radius // 2)
            draw.line([(cx - radius - tick, cy), (cx - radius // 2, cy)], fill=col, width=2)
            draw.line([(cx + radius // 2, cy), (cx + radius + tick, cy)], fill=col, width=2)
            draw.line([(cx, cy - radius - tick), (cx, cy - radius // 2)], fill=col, width=2)
            draw.line([(cx, cy + radius // 2), (cx, cy + radius + tick)], fill=col, width=2)

            label = f"#{tgt['rank']} {tgt['category'].replace('_', ' ')} ({tgt['priority_score']:.0f})"
            tag_x = min(max(4, cx + radius + 6), max(4, w - 180))
            tag_y = max(4, min(h - 22, cy - 9))
            tag_w = min(w - tag_x - 2, max(120, len(label) * 6 + 10))
            draw.rectangle([(tag_x, tag_y), (tag_x + tag_w, tag_y + 16)], fill=(15, 23, 42, 215), outline=col, width=1)
            draw.text((tag_x + 5, tag_y + 3), label, fill=(248, 250, 252, 255))

        combined = Image.alpha_composite(img_pil, overlay).convert("RGB")
        return np.array(combined)
