"""
NASA Deep Space Network (DSN) & Mars Relay Orbiter UHF Telemetry Prioritizer.
Simulates X-band Direct-to-Earth (DTE) and UHF Proximity Link (MRO/TGO/Odyssey) link budgets,
prioritizes multi-queue flight telemetry packets, and manages onboard communication buffers.
"""
from typing import Dict, Any, List
import math
import numpy as np

# Nominal DSN & Mars Relay Proximity Link bitrates (kbps)
LINK_PROFILES = {
    "UHF_MRO_RELAY": {"nominal_kbps": 1024.0, "max_pass_s": 720.0, "carrier": "UHF 401.5 MHz (Electra)"},
    "UHF_TGO_RELAY": {"nominal_kbps": 2048.0, "max_pass_s": 600.0, "carrier": "UHF 401.5 MHz (Electra)"},
    "UHF_ODY_RELAY": {"nominal_kbps": 256.0,  "max_pass_s": 540.0, "carrier": "UHF 401.5 MHz (CE505)"},
    "DTE_X_BAND":    {"nominal_kbps": 16.0,   "max_pass_s": 3600.0, "carrier": "X-Band 8.4 GHz (HGA)"},
}

# Orbiter nominal pass schedule in Local True Solar Time (LTST hours)
PASS_SCHEDULE_LTST = [
    {"orbiter": "Odyssey", "pass_hour": 4.5, "type": "UHF_ODY_RELAY"},
    {"orbiter": "MRO",     "pass_hour": 15.5, "type": "UHF_MRO_RELAY"},
    {"orbiter": "TGO",     "pass_hour": 19.2, "type": "UHF_TGO_RELAY"},
]


class DsnCommManager:
    """
    Manages telemetry packet queues, link budgets, and orbiter transmission windows.
    """
    def __init__(self, buffer_capacity_kb: float = 256000.0):
        self.buffer_capacity_kb = buffer_capacity_kb

    def packetize_telemetry(
        self,
        telemetry_dict: Dict[str, Any],
        has_overlays: bool = False,
    ) -> Dict[str, Any]:
        """
        Organizes terrain telemetry into 4 prioritized queues:
        - P1: Emergency Safety & Health (Immediate halt/tilt/puncture alerts)
        - P2: Nav & Science Metadata (AEGIS targets, attitude, terramechanics)
        - P3: Costmap & Trajectory Profile (Compressed Nav2 grids, waypoints)
        - P4: Heavy Image Payloads (Optical frames, overlays)
        """
        safety_status = telemetry_dict.get("safety_status", "NOMINAL_DRIVE")
        is_emergency = "CRITICAL" in safety_status or "HALT" in safety_status

        # Payload byte size estimations (KB)
        p1_bytes_kb = 1.2 if not is_emergency else 2.5
        p2_bytes_kb = 6.4
        p3_bytes_kb = 24.0
        p4_bytes_kb = 185.0 if has_overlays else 45.0

        total_kb = p1_bytes_kb + p2_bytes_kb + p3_bytes_kb + p4_bytes_kb

        packets = {
            "p1_critical_alert_kb": p1_bytes_kb,
            "p2_science_telemetry_kb": p2_bytes_kb,
            "p3_costmap_plan_kb": p3_bytes_kb,
            "p4_imagery_payload_kb": p4_bytes_kb,
            "total_packet_batch_kb": round(total_kb, 1),
            "is_emergency_priority": is_emergency,
        }
        return packets

    def evaluate_link_budget(
        self,
        packet_batch: Dict[str, Any],
        comm_profile: str = "UHF_MRO_RELAY",
        pass_duration_s: float = 600.0,
        channel_snr_db: float = 14.0,
    ) -> Dict[str, Any]:
        """
        Calculates achievable downlink volume and queue delivery times under active link profile.
        """
        profile = LINK_PROFILES.get(comm_profile, LINK_PROFILES["UHF_MRO_RELAY"])
        nominal_kbps = profile["nominal_kbps"]

        # SNR modulation adjustment: if SNR is degraded, link rate drops gracefully
        snr_factor = min(1.2, max(0.4, channel_snr_db / 14.0))
        effective_kbps = nominal_kbps * snr_factor

        # Total channel capacity in kilobytes: (effective_kbps * pass_duration_s) / 8 bits
        effective_capacity_kb = (effective_kbps * pass_duration_s) / 8.0

        required_kb = packet_batch.get("total_packet_batch_kb", 50.0)

        # Transmission timeline: P1 transmitted in first few seconds
        p1_kb = packet_batch.get("p1_critical_alert_kb", 1.5)
        p1_tx_time_s = (p1_kb * 8.0) / effective_kbps

        total_tx_time_s = (required_kb * 8.0) / effective_kbps

        downlink_completed = effective_capacity_kb >= required_kb
        completion_pct = min(100.0, (effective_capacity_kb / max(1.0, required_kb)) * 100.0)

        # Buffer utilization
        buffer_utilization_pct = (required_kb / self.buffer_capacity_kb) * 100.0

        return {
            "comm_link_profile": comm_profile,
            "comm_profile": comm_profile,
            "carrier_frequency": profile["carrier"],
            "effective_bitrate_kbps": round(effective_kbps, 1),
            "pass_duration_s": round(pass_duration_s, 1),
            "channel_capacity_kb": round(effective_capacity_kb, 1),
            "batch_size_kb": round(required_kb, 1),
            "p1_delivery_latency_s": round(p1_tx_time_s, 2),
            "total_downlink_time_s": round(total_tx_time_s, 2),
            "downlink_completed": downlink_completed,
            "pass_completion_pct": round(completion_pct, 1),
            "buffer_utilization_pct": round(buffer_utilization_pct, 4),
        }

    def schedule_next_pass(self, current_sol_hour: float = 13.5) -> Dict[str, Any]:
        """
        Determines the next upcoming relay orbiter overflight pass relative to rover sol hour.
        """
        curr = current_sol_hour % 24.66
        # Find next pass
        upcoming = None
        min_wait = 999.0
        for p in PASS_SCHEDULE_LTST:
            diff = (p["pass_hour"] - curr) % 24.66
            if diff < min_wait:
                min_wait = diff
                upcoming = p

        hours_to_pass = min_wait
        minutes_to_pass = hours_to_pass * 60.0

        if hours_to_pass < 1.0:
            status = "RELAY_WINDOW_IMMINENT"
            advisory = f"{upcoming['orbiter']} UHF pass commences in {minutes_to_pass:.0f} min. Arm high-speed Electra transceiver."
        elif hours_to_pass < 3.5:
            status = "BUFFER_STAGING_ACTIVE"
            advisory = f"Next pass: {upcoming['orbiter']} in {hours_to_pass:.1f}h. Consolidate high-priority drive packets."
        else:
            status = "DTE_CRUISE_WINDOW"
            advisory = f"Next orbiter overflight ({upcoming['orbiter']}) in {hours_to_pass:.1f}h. Rely on low-speed DTE X-band."

        return {
            "next_orbiter": upcoming["orbiter"],
            "next_pass_type": upcoming["type"],
            "next_pass_sol_hour": upcoming["pass_hour"],
            "hours_until_pass": round(hours_to_pass, 2),
            "minutes_until_pass": round(minutes_to_pass, 0),
            "comm_window_status": status,
            "comm_advisory": advisory,
        }
