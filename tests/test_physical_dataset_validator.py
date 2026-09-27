"""Unit Tests for Physical Dataset Validator (Rules A through P).

Verifies that physical telemetry and feature sessions strictly comply with:
- Exactly 61 raw samples (Rule F)
- Exactly 11 feature windows (Rule G)
- Exactly the 12 features (Rule A)
- Zero NaNs, zero Infs (Rules B, C)
- Strictly valid conditions and stress levels (Rules D, K)
- Prohibition of synthetic machine IDs (Rule M)
- Provenance and session isolation (Rules E, H, I, J, L, N, O, P)
"""

import os
import shutil
import tempfile
import unittest
import numpy as np
import pandas as pd

from src.features import FEATURE_NAMES
from src.physical_dataset_validator import (
    EXPECTED_FEATURE_WINDOWS,
    EXPECTED_RAW_SAMPLES,
    PhysicalDataValidationError,
    validate_physical_dataset,
    validate_physical_feature_session,
    validate_physical_raw_session,
)


def _create_mock_raw_df(
    n_samples=61,
    machine_id="physical_machine_A",
    session_id="phys_sess_01",
    condition="normal",
    stress_level="none",
    background_workload="idle",
) -> pd.DataFrame:
    """Helper to create a valid mock raw session DataFrame."""
    timestamps = [1700000000.0 + i * 0.5 for i in range(n_samples)]
    return pd.DataFrame({
        "timestamp": timestamps,
        "cpu_overall": [5.0] * n_samples,
        "cpu_cores": ["[5.0, 5.0]"] * n_samples,
        "vmem_percent": [40.0] * n_samples,
        "vmem_available": [8000000000] * n_samples,
        "vmem_total": [16000000000] * n_samples,
        "swap_percent": [5.0] * n_samples,
        "disk_read_bytes": [1000] * n_samples,
        "disk_write_bytes": [2000] * n_samples,
        "disk_read_count": [10] * n_samples,
        "disk_write_count": [20] * n_samples,
        "machine_id": [machine_id] * n_samples,
        "session_id": [session_id] * n_samples,
        "condition": [condition] * n_samples,
        "stress_level": [stress_level] * n_samples,
        "background_workload": [background_workload] * n_samples,
        "is_ramp_up": [0] * n_samples,
    })


def _create_mock_feat_df(
    n_windows=11,
    machine_id="physical_machine_A",
    session_id="phys_sess_01",
    condition="normal",
    stress_level="none",
    background_workload="idle",
) -> pd.DataFrame:
    """Helper to create a valid mock feature session DataFrame."""
    data = {f: [1.0] * n_windows for f in FEATURE_NAMES}
    data["machine_id"] = [machine_id] * n_windows
    data["session_id"] = [session_id] * n_windows
    data["condition"] = [condition] * n_windows
    data["stress_level"] = [stress_level] * n_windows
    data["background_workload"] = [background_workload] * n_windows
    data["is_ramp_up"] = [0] * n_windows
    data["window_idx"] = list(range(n_windows))
    return pd.DataFrame(data)


class TestPhysicalSessionValidation(unittest.TestCase):
    """Tests single session validation for raw and feature data."""

    def test_valid_raw_session(self):
        raw_df = _create_mock_raw_df()
        res = validate_physical_raw_session(raw_df, "phys_sess_01")
        self.assertEqual(res["sample_count"], EXPECTED_RAW_SAMPLES)
        self.assertEqual(res["condition"], "normal")

    def test_raw_sample_count_mismatch(self):
        """Rule F: Must have exactly 61 samples."""
        raw_df = _create_mock_raw_df(n_samples=50)
        with self.assertRaises(PhysicalDataValidationError) as ctx:
            validate_physical_raw_session(raw_df)
        self.assertIn("Rule F violation", str(ctx.exception))

    def test_raw_synthetic_id_prohibited(self):
        """Rule M: Synthetic machine IDs are forbidden."""
        raw_df = _create_mock_raw_df(machine_id="machine_A_desktop_ryzen_16c")
        with self.assertRaises(PhysicalDataValidationError) as ctx:
            validate_physical_raw_session(raw_df)
        self.assertIn("Rule M violation", str(ctx.exception))

    def test_raw_nan_fails(self):
        """Rule B: Zero NaN in raw data."""
        raw_df = _create_mock_raw_df()
        raw_df.loc[3, "cpu_overall"] = np.nan
        with self.assertRaises(PhysicalDataValidationError) as ctx:
            validate_physical_raw_session(raw_df)
        self.assertIn("Rule B violation", str(ctx.exception))

    def test_valid_feature_session(self):
        feat_df = _create_mock_feat_df()
        res = validate_physical_feature_session(feat_df, "phys_sess_01")
        self.assertEqual(res["window_count"], EXPECTED_FEATURE_WINDOWS)

    def test_feature_window_count_mismatch(self):
        """Rule G: Must have exactly 11 feature windows."""
        feat_df = _create_mock_feat_df(n_windows=10)
        with self.assertRaises(PhysicalDataValidationError) as ctx:
            validate_physical_feature_session(feat_df)
        self.assertIn("Rule G violation", str(ctx.exception))

    def test_feature_missing_column(self):
        """Rule A: Missing feature column fails loudly."""
        feat_df = _create_mock_feat_df()
        feat_df = feat_df.drop(columns=["disk_io_rate_norm"])
        with self.assertRaises(PhysicalDataValidationError) as ctx:
            validate_physical_feature_session(feat_df)
        self.assertIn("Rule A violation", str(ctx.exception))

    def test_feature_inf_fails(self):
        """Rule C: Infinite feature fails."""
        feat_df = _create_mock_feat_df()
        feat_df.loc[2, "ram_used_pct"] = np.inf
        with self.assertRaises(PhysicalDataValidationError) as ctx:
            validate_physical_feature_session(feat_df)
        self.assertIn("Rule C violation", str(ctx.exception))


class TestPhysicalDatasetDirectoryValidation(unittest.TestCase):
    """Tests end-to-end directory validation with mock physical datasets."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.raw_dir = os.path.join(self.temp_dir, "raw")
        self.proc_dir = os.path.join(self.temp_dir, "processed")
        os.makedirs(self.raw_dir)
        os.makedirs(self.proc_dir)

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_complete_valid_physical_dataset(self):
        """Verifies full directory pass when all 4 conditions exist across sessions."""
        conditions = ["normal", "cpu_pressure", "memory_pressure", "disk_io_pressure"]
        for idx, cond in enumerate(conditions):
            s_id = f"physical_machine_A_{cond}_{idx:02d}"
            raw = _create_mock_raw_df(condition=cond, session_id=s_id)
            feat = _create_mock_feat_df(condition=cond, session_id=s_id)
            raw.to_csv(os.path.join(self.raw_dir, f"{s_id}_raw.csv"), index=False)
            feat.to_csv(os.path.join(self.proc_dir, f"{s_id}_features.csv"), index=False)

        audit = validate_physical_dataset(self.raw_dir, self.proc_dir)
        self.assertEqual(audit["status"], "PASSED")
        self.assertEqual(audit["total_sessions"], 4)
        self.assertEqual(audit["total_feature_windows"], 44)
        self.assertEqual(audit["machine_count"], 1)

    def test_missing_feature_pairing_fails(self):
        """Unmatched raw/processed files must fail."""
        s_id = "physical_machine_A_normal_01"
        raw = _create_mock_raw_df(session_id=s_id)
        raw.to_csv(os.path.join(self.raw_dir, f"{s_id}_raw.csv"), index=False)
        # Omit feature CSV
        with self.assertRaises(PhysicalDataValidationError):
            validate_physical_dataset(self.raw_dir, self.proc_dir)


if __name__ == "__main__":
    unittest.main()
