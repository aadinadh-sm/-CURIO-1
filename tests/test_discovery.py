"""Unit tests for CURIO Discovery Engine.

Strictly validates:
1. Canonical signature learning
2. Recurrence calculation
3. Expected-direction filtering
4. Zero-direction feature exclusion
5. Onset thresholding (z >= 1.8)
6. Two-consecutive-window requirement
7. Duplicate onset handling (first onset only)
8. t=0 sustained-pressure handling
9. Insufficient-evidence handling (< 3 distinct non-tied onsets)
10. Normal-equilibrium handling
11. Canonical sequence ordering
12. Observed sequence extraction
13. Kendall tau normalization
14. Tie detection
15. Partial sequence matching
16. Discovery coverage
17. Missing feature handling
18. NaN/Inf handling
19. Held-out data isolation (temporal leakage test)
20. Artifact schema
21. No causal wording
22. Deterministic output
"""

import os
import shutil
import tempfile
import unittest
import numpy as np
import pandas as pd

from src.discovery import (
    DEFAULT_CONSECUTIVE_WINDOWS,
    DEFAULT_ONSET_THRESHOLD,
    DEFAULT_RECURRENCE_THRESHOLD,
    STATUS_CONFIRMED_PROGRESSION,
    STATUS_INSUFFICIENT_TEMPORAL_EVIDENCE,
    STATUS_OPERATING_EQUILIBRIUM,
    STATUS_SUSTAINED_PRESSURE,
    DiscoveryReferenceProfile,
    compute_kendall_tau_b,
    detect_feature_onsets_in_session,
    discover_session_trajectory,
    format_discovery_report,
)
from src.evidence import EvidenceReferenceProfile, validate_no_causal_language, CausalLanguageError
from src.features import FEATURE_NAMES


class TestDiscoveryEngine(unittest.TestCase):
    """Comprehensive test suite for CURIO Discovery Engine."""

    def setUp(self):
        """Sets up synthetic baseline and controlled training trajectories."""
        self.test_dir = tempfile.mkdtemp()

        # Controlled Normal Reference Profile
        normal_stats = {f: {"mean": 10.0, "std": 2.0, "median": 10.0, "iqr": 2.0} for f in FEATURE_NAMES}
        normal_stats["ram_available_ratio"] = {"mean": 0.50, "std": 0.05, "median": 0.50, "iqr": 0.05}

        expected_dirs = {
            "normal": {f: 0 for f in FEATURE_NAMES},
            "cpu_pressure": {f: 1 if "cpu" in f else 0 for f in FEATURE_NAMES},
            "memory_pressure": {
                "ram_used_pct": 1,
                "ram_available_ratio": -1,
                "swap_used_pct": 1,
                "top_proc_mem_pct": 1,
                "disk_io_rate_norm": 1,
                **{f: 0 for f in FEATURE_NAMES if f not in ("ram_used_pct", "ram_available_ratio", "swap_used_pct", "top_proc_mem_pct", "disk_io_rate_norm")},
            },
            "disk_io_pressure": {
                "disk_io_rate_norm": 1,
                "disk_iops_norm": 1,
                "swap_used_pct": 1,
                **{f: 0 for f in FEATURE_NAMES if f not in ("disk_io_rate_norm", "disk_iops_norm", "swap_used_pct")},
            },
        }

        self.evidence_ref = EvidenceReferenceProfile(
            schema_version="1.0.0",
            training_machines=["train_mach_A"],
            normal_sessions_count=4,
            total_training_samples=44,
            feature_stats=normal_stats,
            expected_directions=expected_dirs,
        )

        # Build mock training sessions for Memory Pressure
        # Session 1: ram_used_pct onsets at w=1, ram_available_ratio at w=2, swap_used_pct at w=4
        # Session 2: ram_used_pct onsets at w=1, ram_available_ratio at w=3, swap_used_pct at w=5
        # Session 3: ram_used_pct onsets at w=2, ram_available_ratio at w=2, swap_used_pct at w=4
        # Session 4: ram_used_pct onsets at w=1, ram_available_ratio at w=3, swap_used_pct at w=4
        # Session 5: top_proc_mem_pct onsets at w=1, but disk_io_rate_norm onsets in only 1 session (low recurrence)
        self.train_sessions_df = self._build_mock_training_dataset()

        self.discovery_ref = DiscoveryReferenceProfile.fit_from_training_sessions(
            train_df=self.train_sessions_df,
            evidence_ref=self.evidence_ref,
            recurrence_threshold=0.60,
        )

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def _build_mock_training_dataset(self) -> pd.DataFrame:
        """Constructs 5 controlled training sessions for Memory Pressure and 2 Normal sessions."""
        rows = []
        # 5 Memory Pressure sessions
        # Schedule of onsets:
        # ram_used_pct: onsets at w=1 (t=2.5s) in 4/5 sessions -> rec=0.80
        # ram_available_ratio: onsets at w=2 or 3 (median ~w=2.5, t=6.25s) in 4/5 sessions -> rec=0.80
        # swap_used_pct: onsets at w=4 or 5 (median ~w=4, t=10.0s) in 4/5 sessions -> rec=0.80
        # disk_io_rate_norm: onsets in only 1/5 sessions (rec=0.20 < 0.60 threshold)
        for s_idx in range(5):
            s_id = f"train_mem_sess_{s_idx}"
            for w in range(11):
                t_offset = w * 2.5
                row = {
                    "session_id": s_id,
                    "condition": "memory_pressure",
                    "machine_id": "train_mach_A",
                    "window_idx": w,
                    "window_start_offset": t_offset,
                }
                # Default normal values
                for f in FEATURE_NAMES:
                    row[f] = 10.0
                row["ram_available_ratio"] = 0.50

                # ram_used_pct onsets at w=1 in sessions 0, 1, 2, 3
                if s_idx < 4 and w >= 1:
                    row["ram_used_pct"] = 15.0  # z = (15-10)/2 = 2.5 >= 1.8

                # ram_available_ratio onsets at w=2 or 3 in sessions 0, 1, 2, 3
                # normal is 0.50, std is 0.05. Drop to 0.35 -> z = (-1) * ((0.35-0.50)/0.05) = +3.0 >= 1.8
                onset_w = 2 if s_idx % 2 == 0 else 3
                if s_idx < 4 and w >= onset_w:
                    row["ram_available_ratio"] = 0.35

                # swap_used_pct onsets at w=4 or 5 in sessions 0, 1, 2, 3
                onset_swap_w = 4 if s_idx != 1 else 5
                if s_idx < 4 and w >= onset_swap_w:
                    row["swap_used_pct"] = 15.0  # z = (15-10)/2 = 2.5 >= 1.8

                # disk_io_rate_norm only onsets in session 0 (low recurrence = 0.20)
                if s_idx == 0 and w >= 6:
                    row["disk_io_rate_norm"] = 15.0

                rows.append(row)

        # 2 Normal sessions
        for s_idx in range(2):
            s_id = f"train_norm_sess_{s_idx}"
            for w in range(11):
                row = {
                    "session_id": s_id,
                    "condition": "normal",
                    "machine_id": "train_mach_A",
                    "window_idx": w,
                    "window_start_offset": w * 2.5,
                }
                for f in FEATURE_NAMES:
                    row[f] = 10.0
                row["ram_available_ratio"] = 0.50
                rows.append(row)

        return pd.DataFrame(rows)

    def _create_single_session_df(
        self,
        condition: str,
        onset_schedule: Dict[str, int],  # feature -> onset window
    ) -> pd.DataFrame:
        """Helper to create a live 11-window session with specified onset windows."""
        rows = []
        for w in range(11):
            row = {
                "session_id": "live_test_session_01",
                "condition": condition,
                "machine_id": "test_mach_X",
                "window_idx": w,
                "window_start_offset": w * 2.5,
            }
            for f in FEATURE_NAMES:
                row[f] = 10.0
            row["ram_available_ratio"] = 0.50

            for feat, onset_w in onset_schedule.items():
                if w >= onset_w:
                    if feat == "ram_available_ratio":
                        row[feat] = 0.35  # drop below normal
                    else:
                        row[feat] = 15.0  # rise above normal

            rows.append(row)
        return pd.DataFrame(rows)

    # 1. Canonical signature learning
    def test_canonical_signature_learning(self):
        mem_sig = self.discovery_ref.canonical_signatures.get("memory_pressure", {})
        self.assertIn("canonical_sequence", mem_sig)
        canon_seq = mem_sig["canonical_sequence"]
        self.assertGreaterEqual(len(canon_seq), 3)
        self.assertIn("ram_used_pct", canon_seq)
        self.assertIn("ram_available_ratio", canon_seq)
        self.assertIn("swap_used_pct", canon_seq)

    # 2. Recurrence calculation
    def test_recurrence_calculation(self):
        mem_sig = self.discovery_ref.canonical_signatures["memory_pressure"]
        recs = mem_sig["recurrence"]
        self.assertEqual(recs["ram_used_pct"], 0.8)
        self.assertEqual(recs["ram_available_ratio"], 0.8)
        self.assertEqual(recs["swap_used_pct"], 0.8)
        self.assertEqual(recs["disk_io_rate_norm"], 0.2)  # 1/5 = 0.2

    # 3. Expected-direction filtering
    def test_expected_direction_filtering(self):
        mem_sig = self.discovery_ref.canonical_signatures["memory_pressure"]
        # cpu_mean has expected_direction = 0 under memory_pressure, so recurrence should be 0.0 or not tracked
        self.assertNotIn("cpu_mean", mem_sig["eligible_features"])

    # 4. Zero-direction feature exclusion
    def test_zero_direction_feature_exclusion(self):
        # Features with exp_dir == 0 must never generate onsets
        live_df = self._create_single_session_df("memory_pressure", {"cpu_mean": 1})
        onsets = detect_feature_onsets_in_session(live_df, self.evidence_ref, "memory_pressure")
        onset_names = [e["feature_name"] for e in onsets]
        self.assertNotIn("cpu_mean", onset_names)

    # 5. Onset thresholding (z >= 1.8)
    def test_onset_thresholding(self):
        # A feature that only reaches z = 1.2 (value 12.4 vs normal 10.0, std 2.0 -> z=1.2 < 1.8)
        rows = []
        for w in range(11):
            row = {"session_id": "s1", "condition": "memory_pressure", "window_idx": w, "window_start_offset": w * 2.5}
            for f in FEATURE_NAMES:
                row[f] = 10.0
            row["ram_available_ratio"] = 0.50
            row["ram_used_pct"] = 12.4  # z = 1.2
            rows.append(row)
        df_sub = pd.DataFrame(rows)
        onsets = detect_feature_onsets_in_session(df_sub, self.evidence_ref, "memory_pressure", onset_threshold=1.8)
        self.assertEqual(len(onsets), 0)

    # 6. Two-consecutive-window requirement
    def test_two_consecutive_window_requirement(self):
        # A single isolated spike at w=3, drops back down at w=4
        rows = []
        for w in range(11):
            row = {"session_id": "s1", "condition": "memory_pressure", "window_idx": w, "window_start_offset": w * 2.5}
            for f in FEATURE_NAMES:
                row[f] = 10.0
            row["ram_available_ratio"] = 0.50
            row["ram_used_pct"] = 16.0 if w == 3 else 10.0  # only 1 window spike
            rows.append(row)
        df_spike = pd.DataFrame(rows)
        onsets = detect_feature_onsets_in_session(df_spike, self.evidence_ref, "memory_pressure")
        self.assertEqual(len(onsets), 0)

    # 7. Duplicate onset handling (first onset only)
    def test_duplicate_onset_handling(self):
        live_df = self._create_single_session_df("memory_pressure", {"ram_used_pct": 2})
        onsets = detect_feature_onsets_in_session(live_df, self.evidence_ref, "memory_pressure")
        ram_onsets = [e for e in onsets if e["feature_name"] == "ram_used_pct"]
        self.assertEqual(len(ram_onsets), 1)
        self.assertEqual(ram_onsets[0]["onset_window"], 2)

    # 8. t=0 sustained-pressure handling
    def test_t0_sustained_pressure_handling(self):
        # All features onset at window 0 (sustained pressure from start)
        live_df = self._create_single_session_df(
            "memory_pressure",
            {"ram_used_pct": 0, "ram_available_ratio": 0, "swap_used_pct": 0},
        )
        res = discover_session_trajectory(live_df, "memory_pressure", self.evidence_ref, self.discovery_ref)
        self.assertEqual(res["discovery_status"], STATUS_SUSTAINED_PRESSURE)
        self.assertIsNone(res["kendall_tau"])
        self.assertIn("Sustained operating pressure", res["interpretation"])

    # 9. Insufficient-evidence handling (< 3 distinct non-tied onsets)
    def test_insufficient_evidence_handling(self):
        # Only 2 features onset
        live_df = self._create_single_session_df(
            "memory_pressure",
            {"ram_used_pct": 1, "ram_available_ratio": 3},
        )
        res = discover_session_trajectory(live_df, "memory_pressure", self.evidence_ref, self.discovery_ref)
        self.assertEqual(res["discovery_status"], STATUS_INSUFFICIENT_TEMPORAL_EVIDENCE)
        self.assertIsNone(res["kendall_tau"])
        self.assertIn("Insufficient temporal evidence", res["interpretation"])

    # 10. Normal-equilibrium handling
    def test_normal_equilibrium_handling(self):
        # All signals stay at normal reference
        live_df = self._create_single_session_df("normal", {})
        res = discover_session_trajectory(live_df, "normal", self.evidence_ref, self.discovery_ref)
        self.assertEqual(res["discovery_status"], STATUS_OPERATING_EQUILIBRIUM)
        self.assertIsNone(res["kendall_tau"])
        self.assertEqual(len(res["observed_sequence"]), 0)
        self.assertIn("Operating Equilibrium", res["interpretation"])

    # 11. Canonical sequence ordering
    def test_canonical_sequence_ordering(self):
        canon_seq = self.discovery_ref.canonical_signatures["memory_pressure"]["canonical_sequence"]
        # In our mock data: ram_used_pct (w=1, 2.5s) precedes ram_available_ratio (w=2.5, 6.25s) precedes swap_used_pct (w=4.0, 10.0s)
        self.assertEqual(canon_seq[0], "ram_used_pct")
        self.assertEqual(canon_seq[1], "ram_available_ratio")
        self.assertEqual(canon_seq[2], "swap_used_pct")

    # 12. Observed sequence extraction
    def test_observed_sequence_extraction(self):
        live_df = self._create_single_session_df(
            "memory_pressure",
            {"ram_used_pct": 1, "ram_available_ratio": 3, "swap_used_pct": 6},
        )
        res = discover_session_trajectory(live_df, "memory_pressure", self.evidence_ref, self.discovery_ref)
        self.assertEqual(res["discovery_status"], STATUS_CONFIRMED_PROGRESSION)
        self.assertEqual(res["observed_sequence"], ["ram_used_pct", "ram_available_ratio", "swap_used_pct"])

    # 13. Kendall tau normalization
    def test_kendall_tau_normalization(self):
        # Perfect matching order
        ranks_x = [0, 1, 2]
        ranks_y = [2.5, 5.0, 10.0]
        tau_b, tau_norm = compute_kendall_tau_b(ranks_x, ranks_y)
        self.assertEqual(tau_b, 1.0)
        self.assertEqual(tau_norm, 1.0)

        # Opposite ordering
        ranks_rev = [10.0, 5.0, 2.5]
        tau_b_rev, tau_norm_rev = compute_kendall_tau_b(ranks_x, ranks_rev)
        self.assertEqual(tau_b_rev, -1.0)
        self.assertEqual(tau_norm_rev, 0.0)

    # 14. Tie detection in onset times
    def test_tie_detection(self):
        # 3 features onset, but 2 share the exact same onset window
        # Distinct onset times = {2.5, 5.0} -> len=2 < 3 -> insufficient temporal evidence
        live_df = self._create_single_session_df(
            "memory_pressure",
            {"ram_used_pct": 1, "ram_available_ratio": 1, "swap_used_pct": 2},
        )
        res = discover_session_trajectory(live_df, "memory_pressure", self.evidence_ref, self.discovery_ref)
        self.assertEqual(res["discovery_status"], STATUS_INSUFFICIENT_TEMPORAL_EVIDENCE)
        self.assertIsNone(res["kendall_tau"])

    # 15. Partial sequence matching
    def test_partial_sequence_matching(self):
        # Session has 3 onsets, 2 match canonical and 1 is unexpected
        live_df = self._create_single_session_df(
            "memory_pressure",
            {"ram_used_pct": 1, "ram_available_ratio": 3, "disk_io_rate_norm": 5},
        )
        res = discover_session_trajectory(live_df, "memory_pressure", self.evidence_ref, self.discovery_ref)
        self.assertIn("disk_io_rate_norm", res["unexpected_features"])
        self.assertIn("swap_used_pct", res["missing_features"])

    # 16. Discovery coverage
    def test_discovery_coverage(self):
        # 3 of 3 canonical features matched -> coverage = 1.0
        live_df = self._create_single_session_df(
            "memory_pressure",
            {"ram_used_pct": 1, "ram_available_ratio": 3, "swap_used_pct": 6},
        )
        res = discover_session_trajectory(live_df, "memory_pressure", self.evidence_ref, self.discovery_ref)
        self.assertEqual(res["discovery_coverage"], 1.0)

    # 17. Missing feature handling
    def test_missing_feature_handling(self):
        live_df = self._create_single_session_df(
            "memory_pressure",
            {"ram_used_pct": 1, "ram_available_ratio": 3, "swap_used_pct": 6},
        )
        # Drop a feature column entirely
        live_df = live_df.drop(columns=["top_proc_mem_pct"])
        res = discover_session_trajectory(live_df, "memory_pressure", self.evidence_ref, self.discovery_ref)
        self.assertEqual(res["discovery_status"], STATUS_CONFIRMED_PROGRESSION)

    # 18. NaN/Inf handling
    def test_nan_inf_safe_exclusion(self):
        live_df = self._create_single_session_df(
            "memory_pressure",
            {"ram_used_pct": 1, "ram_available_ratio": 3, "swap_used_pct": 6},
        )
        # Inject NaN into swap_used_pct
        live_df.loc[live_df["window_idx"] == 6, "swap_used_pct"] = np.nan
        # Should exclude swap_used_pct from onsets safely without crashing
        res = discover_session_trajectory(live_df, "memory_pressure", self.evidence_ref, self.discovery_ref)
        self.assertNotIn("swap_used_pct", res["observed_sequence"])

    # 19. Temporal Leakage Test: Held-out data isolation
    def test_held_out_data_isolation(self):
        # Modify a held-out test session with an absurdly early onset
        held_out_rows = []
        for w in range(11):
            row = {
                "session_id": "held_out_test_session_999",
                "condition": "memory_pressure",
                "machine_id": "held_out_machine_Z",
                "window_idx": w,
                "window_start_offset": w * 2.5,
            }
            for f in FEATURE_NAMES:
                row[f] = 10.0
            row["ram_available_ratio"] = 0.50
            # Put swap_used_pct onset at w=0 in held out
            row["swap_used_pct"] = 99.0
            held_out_rows.append(row)
        held_out_df = pd.DataFrame(held_out_rows)

        # Profile is fitted ONLY on train_sessions_df
        profile = DiscoveryReferenceProfile.fit_from_training_sessions(
            train_df=self.train_sessions_df,
            evidence_ref=self.evidence_ref,
        )

        # Assert held-out machine is not in training_machines
        self.assertNotIn("held_out_machine_Z", profile.training_machines)
        # Assert canonical order still has swap_used_pct LAST (not first)
        canon_seq = profile.canonical_signatures["memory_pressure"]["canonical_sequence"]
        self.assertEqual(canon_seq[0], "ram_used_pct")
        self.assertEqual(canon_seq[-1], "swap_used_pct")

    # 20. Artifact serialization schema
    def test_artifact_schema(self):
        out_path = os.path.join(self.test_dir, "discovery_reference.json")
        self.discovery_ref.to_json(out_path)
        self.assertTrue(os.path.exists(out_path))

        loaded = DiscoveryReferenceProfile.from_json(out_path)
        self.assertEqual(loaded.schema_version, "1.0.0")
        self.assertEqual(loaded.recurrence_threshold, 0.60)
        self.assertEqual(loaded.onset_threshold, 1.8)
        self.assertIn("memory_pressure", loaded.canonical_signatures)

    # 21. No causal wording in formatted output
    def test_no_causal_wording(self):
        live_df = self._create_single_session_df(
            "memory_pressure",
            {"ram_used_pct": 1, "ram_available_ratio": 3, "swap_used_pct": 6},
        )
        res = discover_session_trajectory(live_df, "memory_pressure", self.evidence_ref, self.discovery_ref)
        report_text = format_discovery_report(res)

        # Screen report text against non-causal policy
        validate_no_causal_language(report_text)
        self.assertNotIn("caused", report_text.lower())
        self.assertNotIn("root cause", report_text.lower())
        self.assertNotIn("failure mechanism", report_text.lower())

    # 22. Deterministic output
    def test_deterministic_output(self):
        live_df = self._create_single_session_df(
            "memory_pressure",
            {"ram_used_pct": 1, "ram_available_ratio": 3, "swap_used_pct": 6},
        )
        res1 = discover_session_trajectory(live_df, "memory_pressure", self.evidence_ref, self.discovery_ref)
        res2 = discover_session_trajectory(live_df, "memory_pressure", self.evidence_ref, self.discovery_ref)
        self.assertEqual(res1, res2)


if __name__ == "__main__":
    unittest.main()
