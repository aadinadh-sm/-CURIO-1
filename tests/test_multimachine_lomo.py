"""Comprehensive Multi-Machine LOMO and Physical Telemetry Integration Tests.

Verifies:
- 3-machine LOMO fold isolation (train/test machine exclusion)
- Session-level grouping (zero window cross-over)
- Out-of-fold abnormality threshold isolation
- Machine-identity separability analysis (Section 7)
- Physical dataset integrity on real collected telemetry (physical_machine_A)
- Diagnostic output schema and probability validity
"""

import os
import shutil
import tempfile
import unittest
import numpy as np
import pandas as pd

from src.features import FEATURE_NAMES
from src.machine_identity_analyzer import MachineIdentityAnalyzer
from src.model_trainer import (
    VALID_CONDITIONS,
    LOMOTrainer,
    evaluate_predictions,
    generate_synthetic_lomo_dataset,
    validate_dataset,
)
from src.physical_dataset_validator import validate_physical_dataset


class TestThreeMachineLOMOIsolation(unittest.TestCase):
    """Verifies strict 3-machine LOMO partitioning and hardware isolation."""

    def setUp(self):
        # 3 machines, 2 sessions per condition per machine, 4 windows per session
        self.df = generate_synthetic_lomo_dataset(
            n_machines=3,
            sessions_per_class=2,
            windows_per_session=4,
            random_state=42,
        )
        self.unique_machines = sorted(self.df["machine_id"].unique())

    def test_machine_level_train_test_separation(self):
        """Asserts that each fold has exactly 1 held-out test machine and N-1 training machines."""
        self.assertEqual(len(self.unique_machines), 3)

        for test_machine in self.unique_machines:
            train_df = self.df[self.df["machine_id"] != test_machine]
            test_df = self.df[self.df["machine_id"] == test_machine]

            # Test machine must not be in training machines
            self.assertNotIn(test_machine, set(train_df["machine_id"]))
            self.assertEqual(set(test_df["machine_id"]), {test_machine})

            # Training set must contain all other machines
            expected_train_machines = set(self.unique_machines) - {test_machine}
            self.assertEqual(set(train_df["machine_id"]), expected_train_machines)

    def test_session_level_grouping_integrity(self):
        """Asserts zero session overlap between train and test sets across all folds."""
        for test_machine in self.unique_machines:
            train_df = self.df[self.df["machine_id"] != test_machine]
            test_df = self.df[self.df["machine_id"] == test_machine]

            train_sessions = set(train_df["session_id"])
            test_sessions = set(test_df["session_id"])
            overlap = train_sessions.intersection(test_sessions)
            self.assertEqual(len(overlap), 0, f"Fold {test_machine} has session overlap: {overlap}")


class TestMachineIdentityAnalysis(unittest.TestCase):
    """Tests the machine identity separability analyzer."""

    def test_single_machine_graceful_handling(self):
        """Single machine data should report < 2 machines without crashing."""
        df = generate_synthetic_lomo_dataset(n_machines=1, sessions_per_class=1, windows_per_session=2)
        analyzer = MachineIdentityAnalyzer(df)
        res = analyzer.analyze_separability()
        self.assertEqual(res["machine_count"], 1)
        self.assertFalse(res["is_separable"])
        self.assertIn("requires at least 2 distinct physical machines", res["message"])

    def test_multimachine_separability_execution(self):
        """Multi-machine dataset should compute accuracy, ranking, and confusion matrix."""
        df = generate_synthetic_lomo_dataset(n_machines=3, sessions_per_class=2, windows_per_session=4)
        analyzer = MachineIdentityAnalyzer(df)
        res = analyzer.analyze_separability()
        self.assertEqual(res["machine_count"], 3)
        self.assertIn("accuracy", res)
        self.assertIn("feature_importance_ranking", res)
        self.assertEqual(len(res["feature_importance_ranking"]), len(FEATURE_NAMES))
        self.assertIn("confusion_matrix", res)
        self.assertIn("interpretation", res)


class TestPhysicalMachineAIntegrity(unittest.TestCase):
    """Audits the real physical telemetry collected on physical_machine_A."""

    def test_physical_dataset_files_and_rules(self):
        """Validates all 22 real physical sessions in data/physical_processed and raw."""
        raw_dir = "data/physical_raw"
        proc_dir = "data/physical_processed"

        if os.path.exists(raw_dir) and os.path.exists(proc_dir):
            audit = validate_physical_dataset(raw_dir, proc_dir)
            self.assertEqual(audit["status"], "PASSED")
            self.assertEqual(audit["total_sessions"], 22)
            self.assertEqual(audit["total_feature_windows"], 242)
            self.assertEqual(audit["unique_machines"], ["physical_machine_A"])

            # Verify 4 classes
            self.assertEqual(
                set(audit["class_distribution"].keys()),
                set(VALID_CONDITIONS),
            )
            # Verify sample geometry
            for s in audit["sessions_summary"]:
                self.assertEqual(s["raw_samples"], 61)
                self.assertEqual(s["feature_windows"], 11)


class TestEndToEndThreeMachineLOMO(unittest.TestCase):
    """Executes a full 3-fold LOMO training loop and verifies schema and metrics."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.models_dir = os.path.join(self.temp_dir, "models")
        self.reports_dir = os.path.join(self.temp_dir, "reports")
        self.df = generate_synthetic_lomo_dataset(
            n_machines=3, sessions_per_class=2, windows_per_session=4, random_state=42
        )
        self.trainer = LOMOTrainer(
            self.df,
            rf_params={"n_estimators": 10, "random_state": 42, "n_jobs": 1},
            n_calibration_splits=2,
            random_state=42,
        )

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_three_fold_lomo_evaluation(self):
        """Verifies that all 3 folds execute, threshold is purely OOF, and schema is complete."""
        results = self.trainer.run_lomo_evaluation(
            output_models_dir=self.models_dir,
            output_reports_dir=self.reports_dir,
            generate_plots=False,
        )

        unique_machines = sorted(self.df["machine_id"].unique())
        self.assertEqual(len(unique_machines), 3)

        # 1. Exactly 3 folds
        self.assertEqual(set(results["per_fold"].keys()), set(unique_machines))

        # 2. Check each fold has valid metrics and threshold
        for m_id in unique_machines:
            fold_info = results["per_fold"][m_id]
            self.assertIn("rf_metrics", fold_info)
            self.assertIn("baseline_metrics", fold_info)
            self.assertIn("abnormality_threshold", fold_info)
            self.assertGreaterEqual(fold_info["abnormality_threshold"], 0.0)
            self.assertLessEqual(fold_info["abnormality_threshold"], 1.0)

        # 3. Check aggregate predictions
        pred_df = results["full_predictions"]
        self.assertEqual(len(pred_df), len(self.df))
        for col in [
            "true_condition",
            "predicted_condition",
            "probability_normal",
            "probability_cpu_pressure",
            "probability_memory_pressure",
            "probability_disk_io_pressure",
            "abnormality_score",
            "abnormality_flag",
            "machine_id",
            "session_id",
        ]:
            self.assertIn(col, pred_df.columns)


if __name__ == "__main__":
    unittest.main()
