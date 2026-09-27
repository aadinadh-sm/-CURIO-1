"""Integration and Unit Tests for CURIO Live Diagnosis Pipeline (src/diagnosis_pipeline.py).

Verifies the central CurioDiagnosisPipeline controller using stubbed/pre-collected
telemetry without requiring 30-second live delays.
"""

import json
import os
import shutil
import tempfile
import unittest
from unittest.mock import patch, MagicMock

import numpy as np

from src.collector import load_raw_telemetry
from src.diagnosis_pipeline import (
    CurioDiagnosisPipeline,
    DEFAULT_MODEL_DIR,
    EXPECTED_RAW_SAMPLES,
    EXPECTED_FEATURE_WINDOWS,
)
from src.features import FEATURE_NAMES

PHYSICAL_RAW_DIR = "data/physical_raw"


class TestDiagnosisPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Locate sample physical raw files for testing
        cls.normal_file = os.path.join(PHYSICAL_RAW_DIR, "physical_machine_A_normal_none_5743354b_raw.csv")
        cls.cpu_file = os.path.join(PHYSICAL_RAW_DIR, "physical_machine_A_cpu_pressure_high_b4e1568b_raw.csv")
        cls.mem_file = os.path.join(PHYSICAL_RAW_DIR, "physical_machine_A_memory_pressure_high_b8adbb32_raw.csv")
        cls.disk_file = os.path.join(PHYSICAL_RAW_DIR, "physical_machine_A_disk_io_pressure_high_63628538_raw.csv")

        # Load samples
        cls.normal_samples = load_raw_telemetry(cls.normal_file)
        cls.cpu_samples = load_raw_telemetry(cls.cpu_file)
        cls.mem_samples = load_raw_telemetry(cls.mem_file)
        cls.disk_samples = load_raw_telemetry(cls.disk_file)

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.pipeline = CurioDiagnosisPipeline(
            model_dir=DEFAULT_MODEL_DIR,
            history_dir=self.temp_dir,
        )

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_complete_61_sample_session(self):
        """Test 1: Complete 61-sample session execution."""
        self.assertEqual(len(self.normal_samples), EXPECTED_RAW_SAMPLES)
        result = self.pipeline.run_diagnosis(
            raw_samples=self.normal_samples,
            save_history=False,
            session_id="test_normal_run",
        )
        self.assertEqual(result["session_id"], "test_normal_run")
        self.assertEqual(result["raw_samples_count"], 61)
        self.assertEqual(result["feature_windows_count"], 11)
        self.assertEqual(result["diagnosis"]["condition"], "normal")
        self.assertGreater(result["diagnosis"]["confidence"], 0.80)

    def test_incomplete_session_rejected(self):
        """Test 2: Incomplete session (< 61 samples) is rejected safely."""
        short_samples = self.normal_samples[:40]
        with self.assertRaises(ValueError) as ctx:
            self.pipeline.run_diagnosis(raw_samples=short_samples, save_history=False)
        self.assertIn("Collection error", str(ctx.exception))
        self.assertIn("Expected 61", str(ctx.exception))

    def test_11_window_extraction(self):
        """Test 3: Exactly 11 feature windows are evaluated."""
        result = self.pipeline.run_diagnosis(raw_samples=self.cpu_samples, save_history=False)
        self.assertEqual(len(result["diagnosis"]["per_window_predictions"]), EXPECTED_FEATURE_WINDOWS)
        self.assertEqual(len(result["abnormality"]["per_window_scores"]), EXPECTED_FEATURE_WINDOWS)

    def test_model_loading_and_missing_artifact_handling(self):
        """Test 4: Pipeline raises FileNotFoundError if any required artifact is missing."""
        with tempfile.TemporaryDirectory() as empty_dir:
            with self.assertRaises(FileNotFoundError) as ctx:
                CurioDiagnosisPipeline(model_dir=empty_dir)
            self.assertIn("Missing required deployment artifact", str(ctx.exception))

    def test_prediction_aggregation_across_11_windows(self):
        """Test 5: Prediction aggregation computes mean probability across 11 windows."""
        result = self.pipeline.run_diagnosis(raw_samples=self.cpu_samples, save_history=False)
        diag = result["diagnosis"]
        self.assertEqual(diag["condition"], "cpu_pressure")

        # Verify mean probability calculation
        per_win = diag["per_window_predictions"]
        expected_mean_cpu = np.mean([w["class_probabilities"]["cpu_pressure"] for w in per_win])
        self.assertAlmostEqual(diag["class_probabilities"]["cpu_pressure"], expected_mean_cpu, places=4)
        self.assertEqual(diag["confidence"], round(expected_mean_cpu, 4))

    def test_abnormality_aggregation(self):
        """Test 6: Abnormality score and session abnormal flag behavior."""
        # CPU stress should have elevated abnormality and session_abnormal=True
        cpu_res = self.pipeline.run_diagnosis(raw_samples=self.cpu_samples, save_history=False)
        abn_cpu = cpu_res["abnormality"]
        self.assertGreater(abn_cpu["mean_score"], 0.80)
        self.assertGreaterEqual(abn_cpu["abnormal_window_count"], 6)
        self.assertTrue(abn_cpu["session_abnormal"])

        # Normal session should have low abnormality and session_abnormal=False
        norm_res = self.pipeline.run_diagnosis(raw_samples=self.normal_samples, save_history=False)
        abn_norm = norm_res["abnormality"]
        self.assertLess(abn_norm["mean_score"], 0.50)
        self.assertEqual(abn_norm["abnormal_window_count"], 0)
        self.assertFalse(abn_norm["session_abnormal"])

    def test_evidence_integration(self):
        """Test 7: Evidence integration selects representative strongest window."""
        result = self.pipeline.run_diagnosis(raw_samples=self.cpu_samples, save_history=False)
        ev = result["evidence"]
        self.assertEqual(ev["condition"], "cpu_pressure")
        self.assertIn("representative_window_idx", ev)
        self.assertTrue(0 <= ev["representative_window_idx"] < 11)
        self.assertGreater(len(ev["supporting_evidence"]), 0)
        top_feat = ev["supporting_evidence"][0]["feature_name"]
        self.assertIn(top_feat, ["cpu_mean", "cpu_max", "cpu_core_imbalance"])

    def test_discovery_integration(self):
        """Test 8: Discovery integration processes 11-window trajectory."""
        result = self.pipeline.run_diagnosis(raw_samples=self.disk_samples, save_history=False)
        disc = result["discovery"]
        self.assertEqual(disc["predicted_condition"], "disk_io_pressure")
        self.assertIn(disc["discovery_status"], [
            "SUSTAINED_PRESSURE",
            "CONFIRMED_PROGRESSION",
            "INSUFFICIENT_TEMPORAL_EVIDENCE",
            "OPERATING_EQUILIBRIUM",
        ])
        self.assertIn("disk_io_rate_norm", disc["canonical_sequence"])

    def test_report_schema_conformity(self):
        """Test 9: Output schema conforms strictly to specification."""
        result = self.pipeline.run_diagnosis(raw_samples=self.mem_samples, save_history=False)
        required_top_keys = [
            "session_id",
            "capture_started_at",
            "capture_duration_seconds",
            "raw_samples_count",
            "feature_windows_count",
            "diagnosis",
            "abnormality",
            "evidence",
            "discovery",
            "performance",
            "metadata",
        ]
        for k in required_top_keys:
            self.assertIn(k, result)

        self.assertIn("condition", result["diagnosis"])
        self.assertIn("confidence", result["diagnosis"])
        self.assertIn("class_probabilities", result["diagnosis"])
        self.assertIn("mean_score", result["abnormality"])
        self.assertIn("threshold", result["abnormality"])
        self.assertIn("session_abnormal", result["abnormality"])

    def test_history_persistence(self):
        """Test 10: Saved diagnosis record in history directory."""
        test_sess_id = "test_persist_sess_999"
        result = self.pipeline.run_diagnosis(
            raw_samples=self.normal_samples,
            save_history=True,
            session_id=test_sess_id,
            save_raw=True,
        )
        json_file = os.path.join(self.temp_dir, f"{test_sess_id}_diagnosis.json")
        raw_file = os.path.join(self.temp_dir, f"{test_sess_id}_raw.csv")

        self.assertTrue(os.path.exists(json_file))
        self.assertTrue(os.path.exists(raw_file))

        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.assertEqual(data["session_id"], test_sess_id)
        self.assertEqual(data["diagnosis"]["condition"], "normal")

    def test_cancellation_handling(self):
        """Test 11: Ctrl+C / KeyboardInterrupt during live collection is handled safely."""
        with patch("src.collector.TelemetryCollector.collect", side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt):
                self.pipeline.run_diagnosis(raw_samples=None, save_history=False)

    def test_invalid_telemetry_nan_inf_handling(self):
        """Test 12: NaN or Infinite values in raw telemetry are safely rejected."""
        corrupt_samples = [dict(s) for s in self.normal_samples]
        corrupt_samples[10]["cpu_overall"] = float("nan")

        with self.assertRaises(ValueError) as ctx:
            self.pipeline.run_diagnosis(raw_samples=corrupt_samples, save_history=False)
        self.assertIn("NaN or Infinite", str(ctx.exception))

    def test_missing_feature_handling(self):
        """Test 13: Feature extraction failure or missing features raises ValueError."""
        with patch("src.diagnosis_pipeline.extract_feature_dataframe") as mock_extract:
            # Return DataFrame with missing columns
            mock_extract.return_value = {"bad_col": [1, 2, 3]}
            with self.assertRaises(Exception):
                self.pipeline.run_diagnosis(raw_samples=self.normal_samples, save_history=False)

    def test_deterministic_output(self):
        """Test 14: Pipeline execution is completely deterministic on identical inputs."""
        res1 = self.pipeline.run_diagnosis(raw_samples=self.cpu_samples, save_history=False, session_id="fixed_id")
        res2 = self.pipeline.run_diagnosis(raw_samples=self.cpu_samples, save_history=False, session_id="fixed_id")

        self.assertEqual(res1["diagnosis"]["condition"], res2["diagnosis"]["condition"])
        self.assertEqual(res1["diagnosis"]["confidence"], res2["diagnosis"]["confidence"])
        self.assertEqual(res1["abnormality"]["mean_score"], res2["abnormality"]["mean_score"])
        self.assertEqual(res1["abnormality"]["session_abnormal"], res2["abnormality"]["session_abnormal"])
        self.assertEqual(res1["discovery"]["discovery_status"], res2["discovery"]["discovery_status"])


if __name__ == "__main__":
    unittest.main()
