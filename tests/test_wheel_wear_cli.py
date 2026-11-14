"""
Unit and integration tests for Fleet Wheel Wear & Ventifact Puncture Audit CLI.
"""
import os
import csv
import json
import tempfile
import unittest
import numpy as np
from PIL import Image

from scripts.audit_wheel_wear import run_wheel_wear_audit


class TestWheelWearCli(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.input_dir = os.path.join(self.tmpdir.name, "inputs")
        self.output_dir = os.path.join(self.tmpdir.name, "outputs")
        os.makedirs(self.input_dir, exist_ok=True)

        # Create two synthetic rover scene frames
        for i in range(2):
            img = Image.fromarray(np.random.randint(60, 200, (64, 64, 3), dtype=np.uint8))
            img.save(os.path.join(self.input_dir, f"frame_{i:02d}.jpg"))

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_run_wheel_wear_audit_generates_json_csv_and_overlays(self):
        """Verifies audit CLI computes fatigue telemetry and outputs structured summaries."""
        records = run_wheel_wear_audit(
            input_dir=self.input_dir,
            output_dir=self.output_dir,
            img_size=64,
            limit=2,
            traverse_leg_m=25.0,
            save_overlays=True,
        )

        self.assertEqual(len(records), 2)
        r0 = records[0]
        self.assertIn("image", r0)
        self.assertIn("ventifact_roughness_index", r0)
        self.assertIn("peak_contact_stress_mpa", r0)
        self.assertIn("aluminum_yield_ratio", r0)
        self.assertIn("puncture_risk_pct", r0)
        self.assertIn("wear_rate_um_per_km", r0)
        self.assertIn("leg_skin_loss_um", r0)
        self.assertIn("cumulative_skin_loss_um", r0)
        self.assertIn("wear_status", r0)
        self.assertIn("traction_control_mode", r0)

        # Check JSON output
        json_path = os.path.join(self.output_dir, "wheel_wear_audit_summary.json")
        self.assertTrue(os.path.exists(json_path))
        with open(json_path, "r", encoding="utf-8") as jf:
            loaded_json = json.load(jf)
        self.assertEqual(len(loaded_json), 2)

        # Check CSV output
        csv_path = os.path.join(self.output_dir, "wheel_wear_audit_summary.csv")
        self.assertTrue(os.path.exists(csv_path))
        with open(csv_path, "r", encoding="utf-8") as cf:
            reader = list(csv.DictReader(cf))
        self.assertEqual(len(reader), 2)

        # Check overlays
        overlay_0 = os.path.join(self.output_dir, "overlays", "frame_00_wear.jpg")
        self.assertTrue(os.path.exists(overlay_0))

    def test_run_wheel_wear_audit_missing_dir_raises(self):
        """Verifies audit CLI raises FileNotFoundError on empty or non-existent directory."""
        empty_dir = os.path.join(self.tmpdir.name, "empty_dir")
        os.makedirs(empty_dir, exist_ok=True)
        with self.assertRaises(FileNotFoundError):
            run_wheel_wear_audit(input_dir=empty_dir, output_dir=self.output_dir)


if __name__ == "__main__":
    unittest.main()
