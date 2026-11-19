"""
Unit tests for NASA Rover Visual Odometry (VO) & Feature Slip Compensation Model.
"""
import unittest
import numpy as np
from src.visual_odometry import RoverVisualOdometryModel


class TestVisualOdometryModel(unittest.TestCase):
    def setUp(self):
        self.vo = RoverVisualOdometryModel(min_features_required=10, critical_slip_threshold=0.40)

    def test_extract_salient_keypoints_on_textured_frame(self):
        """Verifies feature extractor identifies high-contrast gradient peaks."""
        # Create textured checkerboard image
        img = np.zeros((128, 128, 3), dtype=np.uint8)
        img[::16, :] = 200
        img[:, ::16] = 200

        # Bedrock substrate (class 1)
        mask = np.ones((128, 128), dtype=np.int64)

        kps = self.vo.extract_salient_keypoints(img, mask, max_features=30)
        self.assertGreater(len(kps), 0)
        self.assertLessEqual(len(kps), 30)
        first_kp = kps[0]
        self.assertIn("row", first_kp)
        self.assertIn("col", first_kp)
        self.assertIn("terrain_class", first_kp)
        self.assertEqual(first_kp["terrain_class"], 1)

    def test_firm_bedrock_yields_nominal_vo(self):
        """Firm bedrock with low expected slip should produce NOMINAL_VO_TRACKING."""
        mask = np.ones((128, 128), dtype=np.int64)  # bedrock
        img = np.random.randint(60, 220, (128, 128, 3), dtype=np.uint8)

        t = self.vo.evaluate_vo_tracking(
            mask, commanded_distance_m=2.0, expected_wheel_slip_pct=5.0, image_np=img
        )
        self.assertIn(t["vo_status"], ["NOMINAL_VO_TRACKING", "ELEVATED_SLIP_COMPENSATED"])
        self.assertGreater(t["actual_distance_m"], 1.8)
        self.assertLess(t["odometry_error_m"], 0.2)
        self.assertGreaterEqual(t["vo_confidence_pct"], 50.0)

    def test_excessive_slip_flags_critical_wheel_slippage(self):
        """High expected slip (>40%) triggers CRITICAL_WHEEL_SLIPPAGE."""
        mask = np.zeros((128, 128), dtype=np.int64)  # soil
        img = np.random.randint(40, 200, (128, 128, 3), dtype=np.uint8)

        t = self.vo.evaluate_vo_tracking(
            mask, commanded_distance_m=2.0, expected_wheel_slip_pct=55.0, image_np=img
        )
        self.assertEqual(t["vo_status"], "CRITICAL_WHEEL_SLIPPAGE")
        self.assertEqual(t["wheel_slip_pct"], 55.0)
        self.assertAlmostEqual(t["actual_distance_m"], 0.9, delta=0.05)

    def test_feature_starvation_on_blank_frame(self):
        """A featureless blank image must trigger VO_FEATURE_STARVATION."""
        blank_img = np.full((128, 128, 3), 128, dtype=np.uint8)
        mask = np.zeros((128, 128), dtype=np.int64)

        t = self.vo.evaluate_vo_tracking(
            mask, commanded_distance_m=1.0, expected_wheel_slip_pct=10.0, image_np=blank_img
        )
        self.assertEqual(t["vo_status"], "VO_FEATURE_STARVATION")
        self.assertLess(t["total_keypoints"], 10)

    def test_render_tracking_overlay_shape(self):
        """Verifies rendered tracking overlay preserves input RGB dimensions."""
        img = np.random.randint(40, 200, (64, 64, 3), dtype=np.uint8)
        mask = np.ones((64, 64), dtype=np.int64)
        t = self.vo.evaluate_vo_tracking(mask, image_np=img)
        overlay = self.vo.render_tracking_overlay(img, mask, t)

        self.assertEqual(overlay.shape, (64, 64, 3))
        self.assertEqual(overlay.dtype, np.uint8)


if __name__ == "__main__":
    unittest.main()
