"""Unit and Integration Tests for CURIO Cross-Machine ML Training and LOMO Evaluation.

Verifies:
- Data validation and strict integrity checks
- LOMO partitioning correctness
- Session grouping and calibration split purity (zero leakage)
- Abnormality threshold calculation
- Held-out prediction output schema
- Metric and Brier score correctness
- End-to-end synthetic 3-machine dry run
"""

import os
import shutil
import tempfile
import unittest
import joblib
import numpy as np
import pandas as pd

from src.features import FEATURE_NAMES
from src.model_trainer import (
    DEFAULT_RF_PARAMS,
    VALID_CONDITIONS,
    LOMOTrainer,
    evaluate_predictions,
    generate_synthetic_lomo_dataset,
    multiclass_brier_score,
    validate_dataset,
)


class TestDatasetValidation(unittest.TestCase):
    """Tests strict dataset validation checks."""

    def setUp(self):
        # Generate a small valid synthetic dataset
        self.df = generate_synthetic_lomo_dataset(
            n_machines=3, sessions_per_class=1, windows_per_session=3, random_state=42
        )

    def test_valid_dataset_passes(self):
        """Valid dataset should pass with no exceptions."""
        try:
            validate_dataset(self.df)
        except Exception as e:
            self.fail(f"validate_dataset raised unexpected exception: {e}")

    def test_missing_feature_column_fails(self):
        """Dropping any required feature must raise ValueError."""
        df_bad = self.df.drop(columns=["cpu_mean"])
        with self.assertRaises(ValueError) as ctx:
            validate_dataset(df_bad)
        self.assertIn("missing required feature columns", str(ctx.exception).lower())

    def test_missing_metadata_column_fails(self):
        """Dropping machine_id or session_id or condition must raise ValueError."""
        df_bad = self.df.drop(columns=["machine_id"])
        with self.assertRaises(ValueError) as ctx:
            validate_dataset(df_bad)
        self.assertIn("missing required metadata columns", str(ctx.exception).lower())

    def test_nan_feature_fails(self):
        """NaN values anywhere in features must raise ValueError."""
        df_bad = self.df.copy()
        df_bad.loc[0, "ram_used_pct"] = np.nan
        with self.assertRaises(ValueError) as ctx:
            validate_dataset(df_bad)
        self.assertIn("contains nan values", str(ctx.exception).lower())

    def test_inf_feature_fails(self):
        """Inf values in features must raise ValueError."""
        df_bad = self.df.copy()
        df_bad.loc[1, "disk_io_rate_norm"] = np.inf
        with self.assertRaises(ValueError) as ctx:
            validate_dataset(df_bad)
        self.assertIn("infinite values", str(ctx.exception).lower())

    def test_invalid_condition_fails(self):
        """Unknown condition labels must raise ValueError."""
        df_bad = self.df.copy()
        df_bad.loc[0, "condition"] = "network_congestion"
        with self.assertRaises(ValueError) as ctx:
            validate_dataset(df_bad)
        self.assertIn("invalid conditions", str(ctx.exception).lower())

    def test_missing_class_fails(self):
        """Dataset lacking any of the 4 conditions must raise ValueError."""
        df_bad = self.df[self.df["condition"] != "disk_io_pressure"].copy()
        with self.assertRaises(ValueError) as ctx:
            validate_dataset(df_bad)
        self.assertIn("missing required condition classes", str(ctx.exception).lower())

    def test_duplicate_session_across_machines_fails(self):
        """Session ID associated with multiple machines must raise ValueError."""
        df_bad = self.df.copy()
        first_session = df_bad["session_id"].iloc[0]
        # Assign this same session_id to a row with a different machine_id
        diff_machine_idx = df_bad[df_bad["machine_id"] != df_bad["machine_id"].iloc[0]].index[0]
        df_bad.loc[diff_machine_idx, "session_id"] = first_session

        with self.assertRaises(ValueError) as ctx:
            validate_dataset(df_bad)
        self.assertIn("associated with multiple machines", str(ctx.exception).lower())


class TestLOMOPartitioningAndPurity(unittest.TestCase):
    """Verifies Leave-One-Machine-Out partitioning and session purity."""

    def setUp(self):
        self.df = generate_synthetic_lomo_dataset(
            n_machines=3, sessions_per_class=2, windows_per_session=5, random_state=123
        )
        self.trainer = LOMOTrainer(self.df, n_calibration_splits=2)

    def test_lomo_isolation_and_session_separation(self):
        """Asserts zero overlap between train and test sessions, and zero test machine in train."""
        unique_machines = sorted(self.df["machine_id"].unique())
        self.assertEqual(len(unique_machines), 3)

        for held_out in unique_machines:
            train_df = self.df[self.df["machine_id"] != held_out]
            test_df = self.df[self.df["machine_id"] == held_out]

            # 1. No held-out machine in train
            self.assertNotIn(held_out, set(train_df["machine_id"]))
            self.assertEqual(set(test_df["machine_id"]), {held_out})

            # 2. No session appears in both train and test
            train_sessions = set(train_df["session_id"])
            test_sessions = set(test_df["session_id"])
            overlap = train_sessions.intersection(test_sessions)
            self.assertEqual(len(overlap), 0, f"Found session overlap in fold {held_out}: {overlap}")

            # 3. All 4 classes represented in training
            self.assertEqual(set(train_df["condition"].unique()), set(VALID_CONDITIONS))

    def test_threshold_out_of_fold_and_test_machine_isolation(self):
        """Verifies that the abnormality threshold uses only training-side out-of-fold predictions.

        Perturbing the held-out test machine's telemetry must have ZERO impact on the fold's
        abnormality threshold.
        """
        held_out = sorted(self.df["machine_id"].unique())[0]

        tmp1 = tempfile.mkdtemp()
        tmp2 = tempfile.mkdtemp()
        try:
            trainer_base = LOMOTrainer(
                self.df, rf_params={"n_estimators": 10, "random_state": 42}, n_calibration_splits=2
            )
            res_base = trainer_base.run_lomo_evaluation(
                output_models_dir=os.path.join(tmp1, "m"),
                output_reports_dir=os.path.join(tmp1, "r"),
                generate_plots=False,
            )
            theta_base = res_base["abnormality_thresholds"][held_out]

            # Heavily perturb the test machine's features
            df_perturbed = self.df.copy()
            test_mask = df_perturbed["machine_id"] == held_out
            df_perturbed.loc[test_mask, "cpu_mean"] = 99.0
            df_perturbed.loc[test_mask, "ram_used_pct"] = 99.0

            trainer_pert = LOMOTrainer(
                df_perturbed, rf_params={"n_estimators": 10, "random_state": 42}, n_calibration_splits=2
            )
            res_pert = trainer_pert.run_lomo_evaluation(
                output_models_dir=os.path.join(tmp2, "m"),
                output_reports_dir=os.path.join(tmp2, "r"),
                generate_plots=False,
            )
            theta_pert = res_pert["abnormality_thresholds"][held_out]

            self.assertAlmostEqual(
                theta_base,
                theta_pert,
                places=6,
                msg="Held-out machine leaked into abnormality threshold calculation!",
            )
        finally:
            shutil.rmtree(tmp1, ignore_errors=True)
            shutil.rmtree(tmp2, ignore_errors=True)


class TestCalibrationGroupSeparation(unittest.TestCase):
    """Verifies that internal calibration folds strictly group by session_id."""

    def test_calibration_session_group_isolation(self):
        """Verifies no session appears in both calibration-train and calibration-val folds."""
        from sklearn.model_selection import StratifiedGroupKFold

        df = generate_synthetic_lomo_dataset(
            n_machines=3, sessions_per_class=2, windows_per_session=5, random_state=42
        )
        train_df = df[df["machine_id"] != df["machine_id"].iloc[0]]

        X_train = train_df[FEATURE_NAMES].values
        y_train = train_df["condition"].values
        groups = train_df["session_id"].values

        sgkf = StratifiedGroupKFold(n_splits=2)
        cv_splits = list(sgkf.split(X_train, y_train, groups=groups))

        for fold_idx, (tr_idx, val_idx) in enumerate(cv_splits):
            tr_sess = set(groups[tr_idx])
            val_sess = set(groups[val_idx])
            overlap = tr_sess.intersection(val_sess)
            self.assertEqual(
                len(overlap), 0,
                f"Calibration fold {fold_idx} has session overlap: {overlap}"
            )


class TestMetricsAndBrierScore(unittest.TestCase):
    """Verifies multiclass Brier score and evaluation metrics."""

    def test_brier_score_perfect(self):
        """Brier score should be 0.0 for perfect probability predictions."""
        classes = ["normal", "cpu_pressure", "memory_pressure", "disk_io_pressure"]
        y_true = ["normal", "cpu_pressure", "memory_pressure", "disk_io_pressure"]
        y_prob = np.eye(4)  # 1.0 for true class, 0.0 for others
        score = multiclass_brier_score(y_true, y_prob, classes)
        self.assertAlmostEqual(score, 0.0, places=6)

    def test_brier_score_worst(self):
        """Brier score should be 2.0 when predicting 1.0 for the wrong class."""
        classes = ["normal", "cpu_pressure"]
        y_true = ["normal", "cpu_pressure"]
        y_prob = np.array([[0.0, 1.0], [1.0, 0.0]])
        score = multiclass_brier_score(y_true, y_prob, classes)
        self.assertAlmostEqual(score, 2.0, places=6)

    def test_evaluate_predictions_structure(self):
        """Verifies evaluate_predictions returns all expected metric fields."""
        classes = sorted(VALID_CONDITIONS)
        y_true = list(classes)
        y_pred = list(classes)
        y_prob = np.eye(4)

        metrics = evaluate_predictions(y_true, y_pred, y_prob, classes)
        self.assertEqual(metrics["accuracy"], 1.0)
        self.assertEqual(metrics["macro_f1"], 1.0)
        self.assertIn("per_class", metrics)
        self.assertIn("confusion_matrix", metrics)
        self.assertIn("brier_score", metrics)
        self.assertIn("log_loss", metrics)


class TestLOMOEndToEndDryRun(unittest.TestCase):
    """Performs an end-to-end synthetic dry run of LOMO training and evaluation."""

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.models_dir = os.path.join(self.test_dir, "models")
        self.reports_dir = os.path.join(self.test_dir, "reports")

        # 3 machines, 2 sessions per class, 4 windows per session
        self.df = generate_synthetic_lomo_dataset(
            n_machines=3, sessions_per_class=2, windows_per_session=4, random_state=42
        )
        self.rf_params = {"n_estimators": 10, "random_state": 42, "n_jobs": 1}
        self.trainer = LOMOTrainer(
            self.df,
            rf_params=self.rf_params,
            n_calibration_splits=2,
            random_state=42,
        )

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_end_to_end_lomo_pipeline(self):
        """Verifies the complete LOMO evaluation loop, artifact generation, and output schema."""
        results = self.trainer.run_lomo_evaluation(
            output_models_dir=self.models_dir,
            output_reports_dir=self.reports_dir,
            generate_plots=True,
        )

        unique_machines = sorted(self.df["machine_id"].unique())

        # 1. Check per-fold results exist for every machine
        self.assertEqual(set(results["per_fold"].keys()), set(unique_machines))

        # 2. Check fold artifacts were saved to disk
        for machine_id in unique_machines:
            fold_dir = os.path.join(self.models_dir, f"fold_{machine_id}")
            self.assertTrue(os.path.isdir(fold_dir), f"Missing fold dir: {fold_dir}")

            model_path = os.path.join(fold_dir, "model.joblib")
            cal_meta_path = os.path.join(fold_dir, "calibration_metadata.json")
            thresh_path = os.path.join(fold_dir, "abnormality_threshold.json")
            feat_meta_path = os.path.join(fold_dir, "feature_metadata.json")

            self.assertTrue(os.path.exists(model_path))
            self.assertTrue(os.path.exists(cal_meta_path))
            self.assertTrue(os.path.exists(thresh_path))
            self.assertTrue(os.path.exists(feat_meta_path))

            # Verify saved model can be loaded and predict
            loaded_model = joblib.load(model_path)
            sample_x = np.random.randn(2, len(FEATURE_NAMES))
            probs = loaded_model.predict_proba(sample_x)
            self.assertEqual(probs.shape, (2, 4))

        # 3. Check held-out prediction schema
        pred_df = results["full_predictions"]
        expected_cols = [
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
        ]
        for col in expected_cols:
            self.assertIn(col, pred_df.columns, f"Prediction DF missing column: {col}")

        self.assertEqual(len(pred_df), len(self.df))

        # Probabilities should sum to ~1.0
        prob_sum = (
            pred_df["probability_normal"]
            + pred_df["probability_cpu_pressure"]
            + pred_df["probability_memory_pressure"]
            + pred_df["probability_disk_io_pressure"]
        )
        np.testing.assert_allclose(prob_sum.values, 1.0, atol=1e-4)

        # 4. Check Abnormality Threshold calculation
        for machine_id, theta in results["abnormality_thresholds"].items():
            self.assertGreaterEqual(theta, 0.0)
            self.assertLessEqual(theta, 1.0)

        # 5. Check aggregate metrics
        agg_rf = results["aggregate_rf"]
        self.assertGreaterEqual(agg_rf["accuracy"], 0.70)
        self.assertGreaterEqual(agg_rf["macro_f1"], 0.70)
        self.assertIn("brier_score", agg_rf)
        self.assertIn("log_loss", agg_rf)

        # 6. Check report and plots generated
        self.assertTrue(os.path.exists(results["report_path"]))
        self.assertTrue(os.path.exists(os.path.join(self.reports_dir, "feature_importance_summary.csv")))
        self.assertTrue(os.path.exists(os.path.join(self.reports_dir, "confusion_matrix_aggregate.png")))
        self.assertTrue(os.path.exists(os.path.join(self.reports_dir, "calibration_aggregate.png")))


if __name__ == "__main__":
    unittest.main()
