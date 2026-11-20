"""
Unit and integration tests for Visual Odometry (VO) Slip Audit CLI.
"""
import os
import sys
import json
import csv
import tempfile
import unittest
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.inference import MarsTerrainSegmenter
from src.visual_odometry import RoverVisualOdometryModel
from scripts.audit_visual_odometry import audit_single_image, run_vo_audit
from scripts.batch_mission_audit import run_batch_audit


class TestVisualOdometryCli(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.segmenter = MarsTerrainSegmenter(device="cpu")
        cls.vo_model = RoverVisualOdometryModel()

    def test_audit_single_image(self):
        """Verifies audit_single_image computes valid VO metrics and schema."""
        with tempfile.TemporaryDirectory() as tmpdir:
            img_path = os.path.join(tmpdir, "test_rover.jpg")
            arr = np.random.randint(50, 200, (64, 64, 3), dtype=np.uint8)
            Image.fromarray(arr).save(img_path)

            rec = audit_single_image(
                segmenter=self.segmenter,
                vo_model=self.vo_model,
                image_path=img_path,
                commanded_distance_m=1.2,
                resolution=64,
                save_overlay=True,
                output_dir=tmpdir,
            )

            self.assertEqual(rec["image"], "test_rover.jpg")
            self.assertEqual(rec["commanded_distance_m"], 1.2)
            self.assertIn("total_keypoints", rec)
            self.assertIn("firm_anchor_count", rec)
            self.assertIn("vo_confidence_pct", rec)
            self.assertIn("wheel_slip_pct", rec)
            self.assertIn("vo_status", rec)
            self.assertIn("overlay_path", rec)
            self.assertTrue(os.path.exists(rec["overlay_path"]))

    def test_run_vo_audit_batch_mode(self):
        """Verifies run_vo_audit on a directory writes valid JSON and CSV summaries."""
        with tempfile.TemporaryDirectory() as in_dir, tempfile.TemporaryDirectory() as out_dir:
            for i in range(2):
                p = os.path.join(in_dir, f"sol_frame_{i}.jpg")
                Image.fromarray(np.random.randint(40, 220, (64, 64, 3), dtype=np.uint8)).save(p)

            records = run_vo_audit(
                input_path=in_dir,
                output_dir=out_dir,
                commanded_distance_m=1.0,
                resolution=64,
                save_overlays=True,
            )

            self.assertEqual(len(records), 2)
            json_file = os.path.join(out_dir, "vo_audit_summary.json")
            csv_file = os.path.join(out_dir, "vo_audit_summary.csv")

            self.assertTrue(os.path.exists(json_file))
            self.assertTrue(os.path.exists(csv_file))

            with open(json_file, "r", encoding="utf-8") as jf:
                loaded = json.load(jf)
                self.assertEqual(len(loaded), 2)
                self.assertIn("wheel_slip_pct", loaded[0])

            with open(csv_file, "r", encoding="utf-8") as cf:
                reader = csv.DictReader(cf)
                rows = list(reader)
                self.assertEqual(len(rows), 2)
                self.assertIn("vo_confidence_pct", rows[0])

    def test_batch_mission_audit_integration(self):
        """Verifies batch_mission_audit incorporates VO, DWA, and dust fields."""
        with tempfile.TemporaryDirectory() as in_dir, tempfile.TemporaryDirectory() as out_dir:
            p = os.path.join(in_dir, "sample_scene.jpg")
            Image.fromarray(np.random.randint(50, 200, (64, 64, 3), dtype=np.uint8)).save(p)

            records = run_batch_audit(
                input_dir=in_dir,
                output_dir=out_dir,
                img_size=64,
                limit=1,
            )

            self.assertEqual(len(records), 1)
            rec = records[0]
            self.assertIn("vo_confidence_pct", rec)
            self.assertIn("vo_slip_pct", rec)
            self.assertIn("dwa_steer_action", rec)
            self.assertIn("dwa_speed_cm_s", rec)
            self.assertIn("dust_transmittance_pct", rec)
            self.assertIn("dust_thickness_um", rec)


if __name__ == "__main__":
    unittest.main()
