"""
Unit and integration tests for Stereo 3D Elevation Audit CLI and Enriched Batch Mission Audit.
"""
import os
import csv
import json
import tempfile
import unittest
import numpy as np
from PIL import Image

from scripts.audit_stereo_geometry import run_stereo_geometry_audit
from scripts.batch_mission_audit import run_batch_audit


class TestStereoGeometryCli(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.input_dir = os.path.join(self.tmpdir.name, "inputs")
        self.output_dir = os.path.join(self.tmpdir.name, "outputs")
        self.batch_out = os.path.join(self.tmpdir.name, "batch_out")
        os.makedirs(self.input_dir, exist_ok=True)

        # Create two synthetic rover scene frames
        for i in range(2):
            img = Image.fromarray(np.random.randint(60, 200, (64, 64, 3), dtype=np.uint8))
            img.save(os.path.join(self.input_dir, f"scene_{i:02d}.jpg"))

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_run_stereo_geometry_audit_generates_json_csv_and_overlays(self):
        """Verifies audit CLI evaluates 3D step obstacles and writes structured summaries."""
        records = run_stereo_geometry_audit(
            input_dir=self.input_dir,
            output_dir=self.output_dir,
            img_size=64,
            limit=2,
            save_overlays=True,
        )

        self.assertEqual(len(records), 2)
        r0 = records[0]
        self.assertIn("image", r0)
        self.assertIn("max_positive_step_cm", r0)
        self.assertIn("max_negative_dropoff_cm", r0)
        self.assertIn("belly_clearance_cm", r0)
        self.assertIn("clearance_margin_pct", r0)
        self.assertIn("step_hazard_status", r0)
        self.assertIn("traverse_permitted", r0)
        self.assertIn("disparity_min_px", r0)
        self.assertIn("disparity_max_px", r0)

        # Check JSON output
        json_path = os.path.join(self.output_dir, "stereo_geometry_audit_summary.json")
        self.assertTrue(os.path.exists(json_path))
        with open(json_path, "r", encoding="utf-8") as jf:
            loaded_json = json.load(jf)
        self.assertEqual(len(loaded_json), 2)

        # Check CSV output
        csv_path = os.path.join(self.output_dir, "stereo_geometry_audit_summary.csv")
        self.assertTrue(os.path.exists(csv_path))
        with open(csv_path, "r", encoding="utf-8") as cf:
            reader = list(csv.DictReader(cf))
        self.assertEqual(len(reader), 2)

        # Check overlays
        overlay_0 = os.path.join(self.output_dir, "overlays", "scene_00_stereo.jpg")
        self.assertTrue(os.path.exists(overlay_0))

    def test_batch_mission_audit_includes_stereo_and_aeolian_fields(self):
        """Verifies batch_mission_audit records step height, aeolian flux, and DSN rate."""
        records = run_batch_audit(
            input_dir=self.input_dir,
            output_dir=self.batch_out,
            img_size=64,
            limit=2,
        )
        self.assertEqual(len(records), 2)
        r0 = records[0]
        self.assertIn("max_step_height_cm", r0)
        self.assertIn("step_hazard_status", r0)
        self.assertIn("wind_speed_m_s", r0)
        self.assertIn("sand_mass_flux_g_m_s", r0)
        self.assertIn("slipface_risk_index", r0)
        self.assertIn("dsn_rate_kbps", r0)

    def test_run_stereo_geometry_audit_missing_dir_raises(self):
        """Verifies audit CLI raises FileNotFoundError on empty or non-existent directory."""
        empty_dir = os.path.join(self.tmpdir.name, "empty_dir")
        os.makedirs(empty_dir, exist_ok=True)
        with self.assertRaises(FileNotFoundError):
            run_stereo_geometry_audit(input_dir=empty_dir, output_dir=self.output_dir)


if __name__ == "__main__":
    unittest.main()
