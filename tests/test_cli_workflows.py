"""
Smoke tests for CLI utility workflows (latency profiler and drive HUD renderer).
"""
import os
import json
import tempfile
import unittest
import numpy as np

from scripts.benchmark_latency import run_benchmark
from scripts.process_drive_sequence import draw_hud_overlay


class TestCliWorkflows(unittest.TestCase):
    def test_benchmark_latency_custom_resolution_and_output(self):
        """Verifies benchmark_latency writes JSON report safely to a custom directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            out_json = os.path.join(tmpdir, "nested", "bench.json")
            results = run_benchmark(
                device_str="cpu",
                num_iterations=1,
                warmup_iterations=0,
                resolutions=[64],
                output_file=out_json,
            )
            self.assertIn("64x64", results)
            self.assertGreater(results["64x64"]["throughput_fps"], 0.0)
            self.assertTrue(os.path.exists(out_json))
            with open(out_json, "r", encoding="utf-8") as f:
                saved = json.load(f)
            self.assertEqual(saved["iterations"], 1)

    def test_draw_hud_overlay_preserves_dimensions(self):
        """Verifies sequential drive HUD overlay preserves frame shape across alert states."""
        frame = np.random.randint(40, 200, (160, 240, 3), dtype=np.uint8)
        telemetry = {
            "safety_status": "CAUTION_ADVISORY",
            "traversability_index": 72.4,
            "recommended_speed_cm_s": 1.8,
            "terrain_distribution_pct": {
                "bedrock": 35.0,
                "soil": 45.0,
                "sand": 18.0,
                "big rock": 2.0,
            },
        }
        mechanics = {"mean_wheel_sinkage_cm": 2.3, "expected_wheel_slip_pct": 19.5}
        pil_out = draw_hud_overlay(frame, telemetry, mechanics, frame_idx=2, total_frames=6)
        self.assertEqual(pil_out.size, (240, 160))
        self.assertEqual(pil_out.mode, "RGB")


if __name__ == "__main__":
    unittest.main()

