"""
Unit tests for DsnCommManager (DSN / UHF Relay Communication Budget & Telemetry Prioritizer).
"""
import unittest

from src.dsn_telemetry import DsnCommManager


class TestDsnCommManager(unittest.TestCase):
    def setUp(self):
        self.comm = DsnCommManager()

    def test_packetize_telemetry_queues(self):
        """Verifies packetizer organizes telemetry into 4 priority queues and calculates sizes."""
        telemetry = {
            "safety_status": "CRITICAL_HAZARD",
            "traversability_index": 35.0,
            "science_targets": {"target_count": 2},
        }
        packets = self.comm.packetize_telemetry(telemetry, has_overlays=True)

        self.assertTrue(packets["is_emergency_priority"])
        self.assertGreater(packets["p1_critical_alert_kb"], 1.0)
        self.assertGreater(packets["p4_imagery_payload_kb"], 100.0)
        self.assertGreater(packets["total_packet_batch_kb"], 100.0)

    def test_evaluate_link_budget_uhf_vs_dte(self):
        """Verifies UHF relay provides orders of magnitude higher throughput than DTE X-band."""
        batch = {"total_packet_batch_kb": 80.0, "p1_critical_alert_kb": 2.0}

        # UHF MRO Relay (1024 kbps nominal)
        uhf_budget = self.comm.evaluate_link_budget(batch, comm_profile="UHF_MRO_RELAY", pass_duration_s=600.0)
        self.assertTrue(uhf_budget["downlink_completed"])
        self.assertLess(uhf_budget["total_downlink_time_s"], 5.0)
        self.assertEqual(uhf_budget["pass_completion_pct"], 100.0)

        # DTE X-band (16 kbps nominal)
        dte_budget = self.comm.evaluate_link_budget(batch, comm_profile="DTE_X_BAND", pass_duration_s=3600.0)
        self.assertGreater(dte_budget["total_downlink_time_s"], uhf_budget["total_downlink_time_s"])
        self.assertLess(dte_budget["effective_bitrate_kbps"], uhf_budget["effective_bitrate_kbps"])

    def test_schedule_next_pass(self):
        """Verifies next orbiter pass schedule calculation given local sol hour."""
        # At hour 13.5, MRO pass at 15.5 is 2.0 hours away
        sched = self.comm.schedule_next_pass(current_sol_hour=13.5)
        self.assertEqual(sched["next_orbiter"], "MRO")
        self.assertAlmostEqual(sched["hours_until_pass"], 2.0, places=1)
        self.assertIn("BUFFER_STAGING_ACTIVE", sched["comm_window_status"])


if __name__ == "__main__":
    unittest.main()
