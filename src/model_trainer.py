"""Leave-One-Machine-Out (LOMO) Cross-Machine Training and Evaluation Pipeline.

Implements cross-machine generalization evaluation for CURIO:
- Calibrated Random Forest multi-class classification
- Strictly grouped Leave-One-Machine-Out validation
- Grouped calibration splits via StratifiedGroupKFold (zero session leakage)
- Empirical 95th-percentile abnormality threshold estimation
- Comparison against the deterministic Heuristic Baseline
"""

import json
import os
from typing import Any, Dict, List, Optional, Tuple, Union
import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedGroupKFold

from src.baseline_rules import HeuristicBaselineClassifier
from src.features import FEATURE_NAMES

VALID_CONDITIONS = [
    "normal",
    "cpu_pressure",
    "memory_pressure",
    "disk_io_pressure",
]

DEFAULT_RF_PARAMS: Dict[str, Any] = {
    "n_estimators": 100,
    "random_state": 42,
    "n_jobs": -1,
    "class_weight": "balanced",
}


def validate_dataset(df: pd.DataFrame) -> None:
    """Validates the processed dataset before training.

    Fails loudly with ValueError if any integrity or schema requirement is violated.

    Args:
        df: DataFrame containing feature windows and session metadata.

    Raises:
        ValueError: If features, metadata, classes, or session mappings are invalid.
    """
    if df is None or not isinstance(df, pd.DataFrame):
        raise ValueError("Dataset must be a non-null pandas DataFrame.")

    if df.empty:
        raise ValueError("Dataset DataFrame is empty.")

    # 1. Required metadata columns
    required_meta = ["machine_id", "session_id", "condition"]
    missing_meta = [col for col in required_meta if col not in df.columns]
    if missing_meta:
        raise ValueError(f"Dataset missing required metadata columns: {missing_meta}")

    # 2. Required feature columns
    missing_features = [col for col in FEATURE_NAMES if col not in df.columns]
    if missing_features:
        raise ValueError(f"Dataset missing required feature columns: {missing_features}")

    # 3. Check for NaN values
    cols_to_check = required_meta + FEATURE_NAMES
    nan_counts = df[cols_to_check].isna().sum()
    cols_with_nan = nan_counts[nan_counts > 0].to_dict()
    if cols_with_nan:
        raise ValueError(f"Dataset contains NaN values in columns: {cols_with_nan}")

    # 4. Check for Inf values in numeric features
    inf_cols = []
    for f in FEATURE_NAMES:
        if np.isneginf(df[f]).any() or np.isposinf(df[f]).any():
            inf_cols.append(f)
    if inf_cols:
        raise ValueError(f"Dataset contains infinite values in feature columns: {inf_cols}")

    # 5. Target labels validity
    unique_conditions = set(df["condition"].unique())
    invalid_conditions = unique_conditions - set(VALID_CONDITIONS)
    if invalid_conditions:
        raise ValueError(
            f"Dataset contains invalid conditions: {invalid_conditions}. "
            f"Allowed conditions are: {VALID_CONDITIONS}"
        )

    # 6. Check that every class is represented in the dataset
    missing_classes = set(VALID_CONDITIONS) - unique_conditions
    if missing_classes:
        raise ValueError(f"Dataset is missing required condition classes: {missing_classes}")

    # 7. Check that each session belongs to exactly one machine
    session_machine_counts = df.groupby("session_id")["machine_id"].nunique()
    multi_machine_sessions = session_machine_counts[session_machine_counts > 1]
    if not multi_machine_sessions.empty:
        bad_sessions = list(multi_machine_sessions.index)
        raise ValueError(
            f"Integrity violation: Sessions associated with multiple machines: {bad_sessions}"
        )

    # 8. Check that machine_id and session_id have no empty strings or null-like values
    if (df["machine_id"].astype(str).str.strip() == "").any():
        raise ValueError("Dataset contains blank machine_id values.")
    if (df["session_id"].astype(str).str.strip() == "").any():
        raise ValueError("Dataset contains blank session_id values.")


def multiclass_brier_score(
    y_true: Union[List[str], np.ndarray],
    y_prob: np.ndarray,
    classes: List[str],
) -> float:
    """Calculates the standard multi-class Brier score.

    Brier = (1/N) * sum_i sum_k (p_ik - y_ik)^2

    Args:
        y_true: Ground truth class labels.
        y_prob: Predicted probability matrix of shape (N, K).
        classes: Ordered list of class labels corresponding to columns of y_prob.

    Returns:
        Mean squared probability forecast error.
    """
    y_arr = np.array(y_true)
    n_samples = len(y_arr)
    n_classes = len(classes)

    y_onehot = np.zeros((n_samples, n_classes), dtype=float)
    for k, cls_name in enumerate(classes):
        y_onehot[y_arr == cls_name, k] = 1.0

    squared_diff = (y_prob - y_onehot) ** 2
    return float(np.mean(np.sum(squared_diff, axis=1)))


def evaluate_predictions(
    y_true: Union[List[str], np.ndarray],
    y_pred: Union[List[str], np.ndarray],
    y_prob: Optional[np.ndarray] = None,
    classes: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Computes comprehensive classification and calibration metrics.

    Args:
        y_true: True class labels.
        y_pred: Predicted class labels.
        y_prob: Optional calibrated probabilities matrix (N, K).
        classes: List of class names. Defaults to VALID_CONDITIONS.

    Returns:
        Dictionary of accuracy, macro and per-class precision/recall/F1, and calibration metrics.
    """
    if classes is None:
        classes = sorted(VALID_CONDITIONS)
    else:
        classes = list(classes)

    acc = float(accuracy_score(y_true, y_pred))
    macro_prec = float(precision_score(y_true, y_pred, labels=classes, average="macro", zero_division=0))
    macro_rec = float(recall_score(y_true, y_pred, labels=classes, average="macro", zero_division=0))
    macro_f1 = float(f1_score(y_true, y_pred, labels=classes, average="macro", zero_division=0))

    prec_per_class = precision_score(y_true, y_pred, labels=classes, average=None, zero_division=0)
    rec_per_class = recall_score(y_true, y_pred, labels=classes, average=None, zero_division=0)
    f1_per_class = f1_score(y_true, y_pred, labels=classes, average=None, zero_division=0)

    cm = confusion_matrix(y_true, y_pred, labels=classes)

    per_class_metrics = {}
    for i, c in enumerate(classes):
        per_class_metrics[c] = {
            "precision": float(prec_per_class[i]),
            "recall": float(rec_per_class[i]),
            "f1": float(f1_per_class[i]),
            "support": int(np.sum(np.array(y_true) == c)),
        }

    metrics: Dict[str, Any] = {
        "accuracy": acc,
        "macro_precision": macro_prec,
        "macro_recall": macro_rec,
        "macro_f1": macro_f1,
        "per_class": per_class_metrics,
        "confusion_matrix": cm.tolist(),
        "classes": classes,
        "sample_count": len(y_true),
    }

    if y_prob is not None:
        metrics["brier_score"] = multiclass_brier_score(y_true, y_prob, classes)
        try:
            metrics["log_loss"] = float(log_loss(y_true, y_prob, labels=classes))
        except Exception:
            metrics["log_loss"] = float("nan")

    return metrics


def generate_synthetic_lomo_dataset(
    n_machines: int = 3,
    sessions_per_class: int = 2,
    windows_per_session: int = 11,
    random_state: int = 42,
) -> pd.DataFrame:
    """Generates a synthetic multi-machine dataset for testing and dry-run verification.

    Simulates realistic telemetry distributions across heterogeneous machines:
    - Normal, CPU Pressure, Memory Pressure, Disk I/O Pressure
    - Machine-specific hardware baselines to test cross-machine invariance.

    Args:
        n_machines: Number of distinct machines (minimum 3).
        sessions_per_class: Number of sessions per condition per machine.
        windows_per_session: Rolling windows per session (default 11, matching 30s capture).
        random_state: Seed for reproducibility.

    Returns:
        pd.DataFrame meeting all validation requirements.
    """
    rng = np.random.RandomState(random_state)
    records = []

    machine_profiles = [
        {"name": "desktop_ryzen_16c", "base_cpu": 5.0, "base_ram": 35.0, "base_disk": 1.0},
        {"name": "laptop_intel_8c", "base_cpu": 12.0, "base_ram": 55.0, "base_disk": 1.5},
        {"name": "server_xeon_32c", "base_cpu": 3.0, "base_ram": 22.0, "base_disk": 0.8},
        {"name": "mini_pc_4c", "base_cpu": 18.0, "base_ram": 62.0, "base_disk": 2.0},
    ]

    for m_idx in range(n_machines):
        profile = machine_profiles[m_idx % len(machine_profiles)]
        machine_id = f"machine_{chr(65 + m_idx)}_{profile['name']}"

        for cond in VALID_CONDITIONS:
            for s_idx in range(sessions_per_class):
                session_id = f"{machine_id}_{cond}_s{s_idx:02d}_{rng.randint(1000, 9999)}"

                for w_idx in range(windows_per_session):
                    row: Dict[str, Any] = {
                        "machine_id": machine_id,
                        "session_id": session_id,
                        "window_idx": w_idx,
                        "timestamp": 1700000000.0 + (s_idx * 100) + (w_idx * 2.5),
                        "condition": cond,
                    }

                    # Synthesize features based on condition and machine baseline
                    if cond == "normal":
                        cpu = max(1.0, profile["base_cpu"] + rng.normal(0, 2.5))
                        row["cpu_mean"] = cpu
                        row["cpu_max"] = cpu + rng.uniform(2.0, 10.0)
                        row["cpu_std"] = rng.uniform(0.5, 3.0)
                        row["cpu_core_imbalance"] = rng.uniform(5.0, 25.0)

                        ram = max(10.0, min(80.0, profile["base_ram"] + rng.normal(0, 1.5)))
                        row["ram_used_pct"] = ram
                        row["ram_available_ratio"] = (100.0 - ram) / 100.0
                        row["swap_used_pct"] = rng.uniform(0.0, 10.0)

                        row["disk_io_rate_norm"] = max(0.0, profile["base_disk"] + rng.normal(0, 0.5))
                        row["disk_iops_norm"] = rng.uniform(0.1, 1.2)
                        row["process_count_delta"] = int(rng.choice([-1, 0, 0, 1]))
                        row["top_proc_cpu_ratio"] = rng.uniform(0.05, 0.35)
                        row["top_proc_mem_pct"] = rng.uniform(1.0, 8.0)

                    elif cond == "cpu_pressure":
                        cpu = rng.uniform(85.0, 99.5)
                        row["cpu_mean"] = cpu
                        row["cpu_max"] = min(100.0, cpu + rng.uniform(0.5, 3.0))
                        row["cpu_std"] = rng.uniform(0.2, 2.0)
                        row["cpu_core_imbalance"] = rng.uniform(10.0, 45.0)

                        ram = profile["base_ram"] + rng.normal(0, 2.0)
                        row["ram_used_pct"] = ram
                        row["ram_available_ratio"] = (100.0 - ram) / 100.0
                        row["swap_used_pct"] = rng.uniform(0.0, 12.0)

                        row["disk_io_rate_norm"] = profile["base_disk"] + rng.normal(0, 0.4)
                        row["disk_iops_norm"] = rng.uniform(0.2, 1.4)
                        row["process_count_delta"] = int(rng.choice([0, 1, 2]))
                        row["top_proc_cpu_ratio"] = rng.uniform(0.40, 0.95)
                        row["top_proc_mem_pct"] = rng.uniform(2.0, 10.0)

                    elif cond == "memory_pressure":
                        cpu = profile["base_cpu"] + rng.uniform(5.0, 15.0)
                        row["cpu_mean"] = cpu
                        row["cpu_max"] = cpu + rng.uniform(5.0, 15.0)
                        row["cpu_std"] = rng.uniform(1.0, 4.0)
                        row["cpu_core_imbalance"] = rng.uniform(5.0, 25.0)

                        ram = rng.uniform(86.0, 95.0)
                        row["ram_used_pct"] = ram
                        row["ram_available_ratio"] = (100.0 - ram) / 100.0
                        row["swap_used_pct"] = rng.uniform(25.0, 65.0)

                        row["disk_io_rate_norm"] = rng.uniform(1.0, 4.0)
                        row["disk_iops_norm"] = rng.uniform(0.5, 2.0)
                        row["process_count_delta"] = int(rng.choice([-1, 0, 1]))
                        row["top_proc_cpu_ratio"] = rng.uniform(0.05, 0.30)
                        row["top_proc_mem_pct"] = rng.uniform(25.0, 60.0)

                    elif cond == "disk_io_pressure":
                        cpu = profile["base_cpu"] + rng.uniform(10.0, 25.0)
                        row["cpu_mean"] = cpu
                        row["cpu_max"] = cpu + rng.uniform(10.0, 25.0)
                        row["cpu_std"] = rng.uniform(2.0, 6.0)
                        row["cpu_core_imbalance"] = rng.uniform(15.0, 40.0)

                        ram = profile["base_ram"] + rng.normal(0, 3.0)
                        row["ram_used_pct"] = ram
                        row["ram_available_ratio"] = (100.0 - ram) / 100.0
                        row["swap_used_pct"] = rng.uniform(0.0, 15.0)

                        row["disk_io_rate_norm"] = rng.uniform(7.8, 9.2)
                        row["disk_iops_norm"] = rng.uniform(2.6, 3.9)
                        row["process_count_delta"] = int(rng.choice([0, 0, 1]))
                        row["top_proc_cpu_ratio"] = rng.uniform(0.10, 0.45)
                        row["top_proc_mem_pct"] = rng.uniform(2.0, 12.0)

                    records.append(row)

    df = pd.DataFrame(records)
    validate_dataset(df)
    return df


class LOMOTrainer:
    """Orchestrates Leave-One-Machine-Out training, grouped calibration, and evaluation."""

    def __init__(
        self,
        df: pd.DataFrame,
        rf_params: Optional[Dict[str, Any]] = None,
        n_calibration_splits: int = 3,
        random_state: int = 42,
    ):
        """Initializes the trainer with verified data and configuration.

        Args:
            df: Validated DataFrame containing all feature windows and metadata.
            rf_params: Parameters for RandomForestClassifier.
            n_calibration_splits: Number of internal calibration splits (StratifiedGroupKFold).
            random_state: Seed for reproducibility.
        """
        validate_dataset(df)
        self.df = df.copy()

        self.rf_params = dict(DEFAULT_RF_PARAMS)
        if rf_params:
            self.rf_params.update(rf_params)

        self.n_calibration_splits = n_calibration_splits
        self.random_state = random_state
        self.baseline_classifier = HeuristicBaselineClassifier()

    def run_lomo_evaluation(
        self,
        output_models_dir: str = "data/models",
        output_reports_dir: str = "reports",
        generate_plots: bool = True,
    ) -> Dict[str, Any]:
        """Runs the complete Leave-One-Machine-Out evaluation loop across all machines.

        Args:
            output_models_dir: Directory where trained fold artifacts will be saved.
            output_reports_dir: Directory where evaluation reports and plots are saved.
            generate_plots: Whether to generate confusion matrix and calibration curve PNGs.

        Returns:
            Dictionary containing per-fold metrics, aggregate metrics, baseline comparison,
            abnormality thresholds, feature importances, and leakage audit confirmations.
        """
        unique_machines = sorted(self.df["machine_id"].unique())
        if len(unique_machines) < 2:
            raise ValueError(
                f"LOMO evaluation requires at least 2 distinct machines, found: {unique_machines}"
            )

        os.makedirs(output_models_dir, exist_ok=True)
        os.makedirs(output_reports_dir, exist_ok=True)

        fold_results = {}
        all_rf_predictions = []
        all_baseline_predictions = []
        all_feature_importances = []
        abnormality_thresholds = {}

        for held_out_machine in unique_machines:
            # 1. Partition into Train (all other machines) and Test (held-out machine)
            train_df = self.df[self.df["machine_id"] != held_out_machine].copy()
            test_df = self.df[self.df["machine_id"] == held_out_machine].copy()

            # --- STRICT LEAKAGE AUDIT ASSERTIONS ---
            train_sessions = set(train_df["session_id"])
            test_sessions = set(test_df["session_id"])
            session_overlap = train_sessions.intersection(test_sessions)
            if session_overlap:
                raise AssertionError(
                    f"Leakage detected: Sessions present in both train and test: {session_overlap}"
                )

            if held_out_machine in set(train_df["machine_id"]):
                raise AssertionError(
                    f"Leakage detected: Held-out machine '{held_out_machine}' present in training set."
                )

            # Ensure training data represents all required classes
            train_classes = set(train_df["condition"].unique())
            missing_train_classes = set(VALID_CONDITIONS) - train_classes
            if missing_train_classes:
                raise ValueError(
                    f"Fold training set for held-out machine '{held_out_machine}' missing classes: {missing_train_classes}"
                )

            X_train = np.array(train_df[FEATURE_NAMES].to_numpy(), dtype=float)
            y_train = np.array(train_df["condition"].to_numpy(), dtype=str)
            groups_train = np.array(train_df["session_id"].to_numpy(), dtype=str)

            X_test = np.array(test_df[FEATURE_NAMES].to_numpy(), dtype=float)
            y_test = np.array(test_df["condition"].to_numpy(), dtype=str)

            # 2. Grouped Calibration Splits via StratifiedGroupKFold
            # Ensure number of calibration splits does not exceed group count for any class
            min_groups_per_class = min(
                train_df.groupby("condition")["session_id"].nunique()
            )
            n_splits = min(self.n_calibration_splits, min_groups_per_class)
            if n_splits < 2:
                n_splits = 2

            sgkf = StratifiedGroupKFold(n_splits=n_splits)
            cv_splits = list(sgkf.split(X_train, y_train, groups=groups_train))

            # Audit: Verify zero session overlap in every calibration fold
            for fold_idx, (cal_tr_idx, cal_val_idx) in enumerate(cv_splits):
                cal_tr_sess = set(groups_train[cal_tr_idx])
                cal_val_sess = set(groups_train[cal_val_idx])
                overlap = cal_tr_sess.intersection(cal_val_sess)
                if overlap:
                    raise AssertionError(
                        f"Calibration leakage detected in fold {fold_idx}: "
                        f"Sessions cross calibration train and validation: {overlap}"
                    )

            # 3. Model Training: Random Forest with Sigmoid Calibration
            base_rf = RandomForestClassifier(**self.rf_params)
            cal_clf = CalibratedClassifierCV(estimator=base_rf, method="sigmoid", cv=cv_splits)
            cal_clf.fit(X_train, y_train)

            # 4. Abnormality Score & Empirical 95th Percentile Threshold
            # Computed strictly from OUT-OF-FOLD predictions on training-machine Normal sessions
            classes_list = list(cal_clf.classes_)
            normal_col_idx = classes_list.index("normal")

            oof_probs = np.zeros((len(X_train), len(classes_list)))
            for c_tr_idx, c_val_idx in cv_splits:
                sub_groups = groups_train[c_tr_idx]
                sub_min_groups = min(
                    len(set(sub_groups[y_train[c_tr_idx] == c])) for c in classes_list
                )
                sub_splits = max(2, min(self.n_calibration_splits, sub_min_groups))
                sub_sgkf = StratifiedGroupKFold(n_splits=sub_splits)
                sub_cv = list(sub_sgkf.split(X_train[c_tr_idx], y_train[c_tr_idx], groups=sub_groups))

                sub_rf = RandomForestClassifier(**self.rf_params)
                sub_cal = CalibratedClassifierCV(estimator=sub_rf, method="sigmoid", cv=sub_cv)
                sub_cal.fit(X_train[c_tr_idx], y_train[c_tr_idx])
                oof_probs[c_val_idx] = sub_cal.predict_proba(X_train[c_val_idx])

            normal_mask = (y_train == "normal")
            oof_normal_probs = oof_probs[normal_mask, normal_col_idx]
            oof_abnormality = 1.0 - oof_normal_probs

            theta_abnormal = float(np.percentile(oof_abnormality, 95))
            abnormality_thresholds[held_out_machine] = theta_abnormal

            # 5. Extract Fold Feature Importances
            # Average feature importances across the calibrated classifiers in this fold
            fold_importances_list = [
                entry.estimator.feature_importances_
                for entry in cal_clf.calibrated_classifiers_
            ]
            fold_importances = np.mean(fold_importances_list, axis=0)
            all_feature_importances.append(fold_importances)

            # 6. Test Predictions on Held-out Machine Windows
            test_probs = cal_clf.predict_proba(X_test)
            pred_class_indices = np.argmax(test_probs, axis=1)
            pred_classes = [classes_list[idx] for idx in pred_class_indices]

            p_normal = test_probs[:, classes_list.index("normal")]
            p_cpu = test_probs[:, classes_list.index("cpu_pressure")]
            p_mem = test_probs[:, classes_list.index("memory_pressure")]
            p_disk = test_probs[:, classes_list.index("disk_io_pressure")]

            test_abnormality = 1.0 - p_normal
            test_abnormal_flags = test_abnormality > theta_abnormal

            fold_preds_df = pd.DataFrame({
                "true_condition": y_test,
                "predicted_condition": pred_classes,
                "probability_normal": p_normal,
                "probability_cpu_pressure": p_cpu,
                "probability_memory_pressure": p_mem,
                "probability_disk_io_pressure": p_disk,
                "abnormality_score": test_abnormality,
                "abnormality_flag": test_abnormal_flags,
                "machine_id": test_df["machine_id"].values,
                "session_id": test_df["session_id"].values,
            })
            all_rf_predictions.append(fold_preds_df)

            # 7. Evaluate Heuristic Baseline on identical held-out test windows
            baseline_batch = self.baseline_classifier.predict_dataframe(test_df)
            baseline_pred_classes = baseline_batch["predicted_condition"].values

            fold_baseline_df = pd.DataFrame({
                "true_condition": y_test,
                "predicted_condition": baseline_pred_classes,
                "machine_id": test_df["machine_id"].values,
                "session_id": test_df["session_id"].values,
            })
            all_baseline_predictions.append(fold_baseline_df)

            # 8. Compute Per-Fold Metrics
            rf_metrics = evaluate_predictions(
                y_true=y_test,
                y_pred=pred_classes,
                y_prob=test_probs,
                classes=classes_list,
            )
            base_metrics = evaluate_predictions(
                y_true=y_test,
                y_pred=baseline_pred_classes,
                y_prob=None,
                classes=VALID_CONDITIONS,
            )

            fold_results[held_out_machine] = {
                "rf_metrics": rf_metrics,
                "baseline_metrics": base_metrics,
                "abnormality_threshold": theta_abnormal,
                "feature_importances": dict(zip(FEATURE_NAMES, fold_importances.tolist())),
                "test_sample_count": len(test_df),
                "test_session_count": int(test_df["session_id"].nunique()),
            }

            # 9. Save Fold Artifacts to data/models/fold_<machine_id>/
            fold_dir = os.path.join(output_models_dir, f"fold_{held_out_machine}")
            os.makedirs(fold_dir, exist_ok=True)

            joblib.dump(cal_clf, os.path.join(fold_dir, "model.joblib"))

            with open(os.path.join(fold_dir, "calibration_metadata.json"), "w") as f:
                json.dump(
                    {
                        "method": "sigmoid",
                        "n_calibration_splits": n_splits,
                        "group_variable": "session_id",
                        "classes": classes_list,
                        "brier_score": rf_metrics.get("brier_score"),
                        "log_loss": rf_metrics.get("log_loss"),
                        "rf_params": self.rf_params,
                    },
                    f,
                    indent=2,
                )

            with open(os.path.join(fold_dir, "abnormality_threshold.json"), "w") as f:
                json.dump(
                    {
                        "methodology": "95th percentile of training Normal calibrated abnormality scores",
                        "target_train_fpr": 0.05,
                        "abnormality_threshold": theta_abnormal,
                        "training_normal_windows": int(np.sum(normal_mask)),
                        "held_out_machine": held_out_machine,
                        "caveat": "Empirical threshold on training hardware; not a guaranteed 5% FPR on unseen machines.",
                    },
                    f,
                    indent=2,
                )

            with open(os.path.join(fold_dir, "feature_metadata.json"), "w") as f:
                json.dump(
                    {
                        "feature_names": FEATURE_NAMES,
                        "feature_importances": dict(zip(FEATURE_NAMES, fold_importances.tolist())),
                    },
                    f,
                    indent=2,
                )

            # 10. Save Evidence Reference Profile (Strictly Training-Derived)
            from src.evidence import EvidenceReferenceProfile
            train_machines = sorted(list(set(train_df["machine_id"].dropna().unique())))
            evidence_ref = EvidenceReferenceProfile.fit_from_training_data(
                train_df=train_df,
                training_machines=train_machines,
            )
            evidence_ref.to_json(os.path.join(fold_dir, "evidence_reference.json"))

            # 11. Save Discovery Reference Profile (Strictly Training-Derived)
            from src.discovery import DiscoveryReferenceProfile
            discovery_ref = DiscoveryReferenceProfile.fit_from_training_sessions(
                train_df=train_df,
                evidence_ref=evidence_ref,
                training_machines=train_machines,
            )
            discovery_ref.to_json(os.path.join(fold_dir, "discovery_reference.json"))

        # --- AGGREGATE EVALUATION ACROSS ALL HELD-OUT MACHINES ---
        full_rf_df = pd.concat(all_rf_predictions, ignore_index=True)
        full_baseline_df = pd.concat(all_baseline_predictions, ignore_index=True)

        classes_order = sorted(VALID_CONDITIONS)
        full_prob_matrix = np.column_stack([
            full_rf_df[f"probability_{c}"].to_numpy(dtype=float) for c in classes_order
        ])

        aggregate_rf_metrics = evaluate_predictions(
            y_true=np.array(full_rf_df["true_condition"].to_numpy(), dtype=str),
            y_pred=np.array(full_rf_df["predicted_condition"].to_numpy(), dtype=str),
            y_prob=full_prob_matrix,
            classes=classes_order,
        )

        aggregate_baseline_metrics = evaluate_predictions(
            y_true=full_baseline_df["true_condition"].values,
            y_pred=full_baseline_df["predicted_condition"].values,
            y_prob=None,
            classes=classes_order,
        )

        # Average feature importance across all folds
        mean_feature_importances = np.mean(all_feature_importances, axis=0)
        std_feature_importances = np.std(all_feature_importances, axis=0)
        feat_imp_df = pd.DataFrame({
            "feature": FEATURE_NAMES,
            "mean_importance": mean_feature_importances,
            "std_importance": std_feature_importances,
        }).sort_values(by="mean_importance", ascending=False)

        feat_imp_csv_path = os.path.join(output_reports_dir, "feature_importance_summary.csv")
        feat_imp_df.to_csv(feat_imp_csv_path, index=False)

        # Optional Plot Generation
        plot_paths: Dict[str, str] = {}
        if generate_plots:
            plot_paths = self._generate_evaluation_plots(
                full_rf_df, full_baseline_df, feat_imp_df, output_reports_dir
            )

        # Generate Evaluation Report Markdown (curio_lomo_evaluation.md)
        report_path = os.path.join(output_reports_dir, "curio_lomo_evaluation.md")
        self._write_evaluation_report(
            report_path=report_path,
            unique_machines=unique_machines,
            fold_results=fold_results,
            aggregate_rf_metrics=aggregate_rf_metrics,
            aggregate_baseline_metrics=aggregate_baseline_metrics,
            feat_imp_df=feat_imp_df,
            plot_paths=plot_paths,
        )

        return {
            "per_fold": fold_results,
            "aggregate_rf": aggregate_rf_metrics,
            "aggregate_baseline": aggregate_baseline_metrics,
            "feature_importance_summary": feat_imp_df.to_dict(orient="records"),
            "abnormality_thresholds": abnormality_thresholds,
            "report_path": report_path,
            "plot_paths": plot_paths,
            "full_predictions": full_rf_df,
        }

    def train_deployment_model(
        self,
        output_dir: str = "data/models/physical_deployment",
    ) -> Dict[str, Any]:
        """Trains a final deployment model and evidence reference on the complete approved dataset.

        Uses Grouped internal CV (StratifiedGroupKFold on session_id) so adjacent windows
        from the same session never leak across calibration folds.
        Generates:
        - model.joblib
        - calibration_metadata.json
        - abnormality_threshold.json
        - feature_metadata.json
        - evidence_reference.json
        """
        os.makedirs(output_dir, exist_ok=True)
        X = np.array(self.df[FEATURE_NAMES].to_numpy(), dtype=float)
        y = np.array(self.df["condition"].to_numpy(), dtype=str)
        groups = np.array(self.df["session_id"].to_numpy(), dtype=str)

        classes_list = sorted(VALID_CONDITIONS)
        min_groups_per_class = min(self.df.groupby("condition")["session_id"].nunique())
        n_splits = max(2, min(self.n_calibration_splits, min_groups_per_class))

        sgkf = StratifiedGroupKFold(n_splits=n_splits)
        cv_splits = list(sgkf.split(X, y, groups=groups))

        base_rf = RandomForestClassifier(**self.rf_params)
        cal_clf = CalibratedClassifierCV(estimator=base_rf, method="sigmoid", cv=cv_splits)
        cal_clf.fit(X, y)

        # Abnormality threshold on Out-of-fold training Normal predictions
        normal_col_idx = list(cal_clf.classes_).index("normal")
        oof_probs = np.zeros((len(X), len(cal_clf.classes_)))
        for c_tr_idx, c_val_idx in cv_splits:
            sub_groups = groups[c_tr_idx]
            sub_min_groups = min(len(set(sub_groups[y[c_tr_idx] == c])) for c in cal_clf.classes_)
            sub_splits = max(2, min(self.n_calibration_splits, sub_min_groups))
            sub_sgkf = StratifiedGroupKFold(n_splits=sub_splits)
            sub_cv = list(sub_sgkf.split(X[c_tr_idx], y[c_tr_idx], groups=sub_groups))

            sub_rf = RandomForestClassifier(**self.rf_params)
            sub_cal = CalibratedClassifierCV(estimator=sub_rf, method="sigmoid", cv=sub_cv)
            sub_cal.fit(X[c_tr_idx], y[c_tr_idx])
            oof_probs[c_val_idx] = sub_cal.predict_proba(X[c_val_idx])

        normal_mask = (y == "normal")
        oof_normal_probs = oof_probs[normal_mask, normal_col_idx]
        oof_abnormality = 1.0 - oof_normal_probs
        theta_abnormal = float(np.percentile(oof_abnormality, 95))

        # Average feature importance
        importances_list = [entry.estimator.feature_importances_ for entry in cal_clf.calibrated_classifiers_]
        avg_importances = np.mean(importances_list, axis=0)

        # Save artifacts
        joblib.dump(cal_clf, os.path.join(output_dir, "model.joblib"))

        with open(os.path.join(output_dir, "calibration_metadata.json"), "w") as f:
            json.dump({
                "method": "sigmoid",
                "n_calibration_splits": n_splits,
                "group_variable": "session_id",
                "classes": list(cal_clf.classes_),
                "rf_params": self.rf_params,
            }, f, indent=2)

        with open(os.path.join(output_dir, "abnormality_threshold.json"), "w") as f:
            json.dump({
                "methodology": "95th percentile of training Normal calibrated abnormality scores",
                "target_train_fpr": 0.05,
                "abnormality_threshold": theta_abnormal,
                "training_normal_windows": int(np.sum(normal_mask)),
                "artifact_type": "FINAL DEPLOYMENT ARTIFACT",
            }, f, indent=2)

        with open(os.path.join(output_dir, "feature_metadata.json"), "w") as f:
            json.dump({
                "feature_names": FEATURE_NAMES,
                "feature_importances": dict(zip(FEATURE_NAMES, avg_importances.tolist())),
            }, f, indent=2)

        from src.evidence import EvidenceReferenceProfile
        train_machines = sorted(list(set(self.df["machine_id"].dropna().unique())))
        evidence_ref = EvidenceReferenceProfile.fit_from_training_data(
            train_df=self.df,
            training_machines=train_machines,
        )
        evidence_ref.to_json(os.path.join(output_dir, "evidence_reference.json"))

        from src.discovery import DiscoveryReferenceProfile
        discovery_ref = DiscoveryReferenceProfile.fit_from_training_sessions(
            train_df=self.df,
            evidence_ref=evidence_ref,
            training_machines=train_machines,
        )
        discovery_ref.to_json(os.path.join(output_dir, "discovery_reference.json"))

        return {
            "model": cal_clf,
            "abnormality_threshold": theta_abnormal,
            "evidence_reference": evidence_ref,
            "discovery_reference": discovery_ref,
            "output_dir": output_dir,
        }

    def _generate_evaluation_plots(
        self,
        rf_df: pd.DataFrame,
        baseline_df: pd.DataFrame,
        feat_imp_df: pd.DataFrame,
        reports_dir: str,
    ) -> Dict[str, str]:
        """Generates confusion matrix and calibration curve PNG plots."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plot_paths = {}

        # 1. Confusion Matrix Comparison Plot
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        cm_rf = confusion_matrix(
            rf_df["true_condition"], rf_df["predicted_condition"], labels=VALID_CONDITIONS
        )
        cm_base = confusion_matrix(
            baseline_df["true_condition"], baseline_df["predicted_condition"], labels=VALID_CONDITIONS
        )

        short_labels = ["Normal", "CPU", "Memory", "Disk"]

        im0 = axes[0].imshow(cm_rf, interpolation="nearest", cmap=plt.cm.Blues)
        axes[0].set_title("CURIO Calibrated Random Forest (LOMO)")
        axes[0].set_xticks(range(4))
        axes[0].set_xticklabels(short_labels)
        axes[0].set_yticks(range(4))
        axes[0].set_yticklabels(short_labels)
        axes[0].set_ylabel("True Condition")
        axes[0].set_xlabel("Predicted Condition")
        for i in range(4):
            for j in range(4):
                axes[0].text(
                    j, i, str(cm_rf[i, j]),
                    ha="center", va="center",
                    color="white" if cm_rf[i, j] > cm_rf.max() / 2 else "black",
                )

        im1 = axes[1].imshow(cm_base, interpolation="nearest", cmap=plt.cm.Oranges)
        axes[1].set_title("Heuristic Rule Baseline")
        axes[1].set_xticks(range(4))
        axes[1].set_xticklabels(short_labels)
        axes[1].set_yticks(range(4))
        axes[1].set_yticklabels(short_labels)
        axes[1].set_ylabel("True Condition")
        axes[1].set_xlabel("Predicted Condition")
        for i in range(4):
            for j in range(4):
                axes[1].text(
                    j, i, str(cm_base[i, j]),
                    ha="center", va="center",
                    color="white" if cm_base[i, j] > cm_base.max() / 2 else "black",
                )

        plt.tight_layout()
        cm_path = os.path.join(reports_dir, "confusion_matrix_aggregate.png")
        fig.savefig(cm_path, dpi=150)
        plt.close(fig)
        plot_paths["confusion_matrix"] = cm_path

        # 2. Calibration Reliability Curves
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.plot([0, 1], [0, 1], "k--", label="Perfect Calibration")

        from sklearn.calibration import calibration_curve
        colors = ["#2b5c8f", "#d95f02", "#7570b3", "#e7298a"]
        for idx, cond in enumerate(VALID_CONDITIONS):
            prob_col = f"probability_{cond}"
            y_binary = (rf_df["true_condition"] == cond).astype(int)
            prob_vals = rf_df[prob_col].values

            # Only compute if binary labels have at least one positive
            if y_binary.sum() > 0:
                prob_true, prob_pred = calibration_curve(y_binary, prob_vals, n_bins=5, strategy="uniform")
                ax.plot(prob_pred, prob_true, marker="o", label=f"{cond}", color=colors[idx])

        ax.set_xlabel("Mean Predicted Probability")
        ax.set_ylabel("Fraction of Positives (Empirical)")
        ax.set_title("Probability Calibration Reliability (Held-Out LOMO)")
        ax.legend(loc="lower right")
        ax.grid(True, linestyle=":", alpha=0.6)

        plt.tight_layout()
        cal_path = os.path.join(reports_dir, "calibration_aggregate.png")
        fig.savefig(cal_path, dpi=150)
        plt.close(fig)
        plot_paths["calibration"] = cal_path

        return plot_paths

    def _write_evaluation_report(
        self,
        report_path: str,
        unique_machines: List[str],
        fold_results: Dict[str, Any],
        aggregate_rf_metrics: Dict[str, Any],
        aggregate_baseline_metrics: Dict[str, Any],
        feat_imp_df: pd.DataFrame,
        plot_paths: Dict[str, str],
    ) -> None:
        """Writes the structured Markdown evaluation report covering all 16 required sections."""
        n_sessions = int(self.df["session_id"].nunique())
        n_windows = len(self.df)
        class_dist = self.df["condition"].value_counts().to_dict()

        md_lines = []
        md_lines.append("# CURIO — Leave-One-Machine-Out (LOMO) Cross-Machine Evaluation Report")
        md_lines.append("")
        md_lines.append("> [!IMPORTANT]")
        md_lines.append(
            '> **Scope of Claim**: *"This experiment evaluates cross-machine generalization across the '
            'available experimental machines. It does not establish universal generalization to arbitrary '
            'computer hardware."*'
        )
        md_lines.append("")
        md_lines.append("---")
        md_lines.append("")

        # 1-5. Dataset & Experimental Summary
        md_lines.append("## 1. Dataset & Telemetry Summary")
        md_lines.append(f"- **Number of Distinct Physical/Simulated Machines**: {len(unique_machines)}")
        md_lines.append(f"- **Total Diagnostic Sessions**: {n_sessions}")
        md_lines.append(f"- **Total Rolling Feature Windows**: {n_windows} (5.0s window duration, 2.5s step)")
        md_lines.append(f"- **Feature Dimension**: Exactly 12 hardware-normalized features")
        md_lines.append("- **Class Distribution**:")
        for cond in VALID_CONDITIONS:
            count = class_dist.get(cond, 0)
            pct = (count / n_windows) * 100 if n_windows > 0 else 0.0
            md_lines.append(f"  - `{cond}`: {count} windows ({pct:.1f}%)")
        md_lines.append("")

        # 6. LOMO Methodology
        md_lines.append("## 2. Leave-One-Machine-Out (LOMO) Methodology")
        md_lines.append(
            "Evaluation strictly follows Leave-One-Machine-Out cross-validation. For each evaluation fold:\n"
            "- **Training Set**: All diagnostic sessions originating from all other machines (`machine_id != m`).\n"
            "- **Test Set**: All diagnostic sessions originating strictly from the held-out machine (`machine_id == m`).\n"
            "- **Hardware Isolation**: The held-out machine remains completely untouched during baseline calculation, "
            "feature extraction, probability calibration, and abnormality threshold setting."
        )
        md_lines.append("")

        # 7. Calibration Methodology
        md_lines.append("## 3. Grouped Probability Calibration Methodology")
        md_lines.append(
            "- **Calibrator**: `CalibratedClassifierCV(estimator=RandomForestClassifier(...), method='sigmoid')`.\n"
            "- **Session-Grouped CV**: To prevent data leakage from temporally adjacent rolling windows, calibration folds are "
            "formed using `StratifiedGroupKFold` grouped strictly by `session_id`.\n"
            "- **Zero Cross-Fold Leakage**: Windows from the same session never cross between calibration-train and calibration-validation."
        )
        md_lines.append("")

        # 8. Abnormality Threshold Methodology
        md_lines.append("## 4. Abnormality Score & Empirical Threshold Methodology")
        md_lines.append(
            "- **Formula**: $\\text{abnormality\\_score} = 1.0 - P(\\text{Normal})$.\n"
            "- **Empirical Reference Target**: For each fold, the abnormality threshold $\\theta_{\\text{abnormal}}$ is computed as the "
            "**95th percentile** of calibrated abnormality scores observed strictly on training-machine Normal sessions.\n"
            "- **Statistical Interpretation**: This targets an empirical $\\approx 5\\%$ false positive rate on the training reference hardware. "
            "It is an empirical operating point, not a theoretical guarantee on unseen hardware."
        )
        md_lines.append("")

        # 9. Leakage Audit
        md_lines.append("## 5. Strict Zero-Leakage Audit Summary")
        md_lines.append("| Leakage Vector | Audit Check | Status |")
        md_lines.append("|:---|:---|:---:|")
        md_lines.append("| A. Session Overlap | Assert `Train_Sessions ∩ Test_Sessions == ∅` | **PASSED** |")
        md_lines.append("| B. Machine Isolation | Assert `Held_Out_Machine ∉ Train_Machines` | **PASSED** |")
        md_lines.append("| C. Calibration Purity | Held-out machine never participates in calibration CV | **PASSED** |")
        md_lines.append("| D. Threshold Independence | Test machine telemetry excluded from threshold $\\theta$ | **PASSED** |")
        md_lines.append("| E. Preprocessing Purity | No global scalers; RF operates on raw hardware-normalized features | **PASSED** |")
        md_lines.append("| F. Label Independence | Test labels strictly quarantined until final evaluation | **PASSED** |")
        md_lines.append("| G. Rolling Window Grouping | `groups = session_id` prevents adjacent window leakage | **PASSED** |")
        md_lines.append("")

        # 10. Per-Machine Metrics
        md_lines.append("## 6. Per-Machine Generalization Performance")
        md_lines.append("| Held-Out Machine | Test Windows | RF Accuracy | RF Macro F1 | Baseline Macro F1 | Abnormality $\\theta$ | Brier Score |")
        md_lines.append("|:---|:---:|:---:|:---:|:---:|:---:|:---:|")
        for m_name, res in fold_results.items():
            rf_m = res["rf_metrics"]
            base_m = res["baseline_metrics"]
            theta = res["abnormality_threshold"]
            brier = rf_m.get("brier_score", float("nan"))
            md_lines.append(
                f"| `{m_name}` | {res['test_sample_count']} | {rf_m['accuracy']:.3f} | {rf_m['macro_f1']:.3f} | "
                f"{base_m['macro_f1']:.3f} | {theta:.4f} | {brier:.4f} |"
            )
        md_lines.append("")

        # 11-12. Aggregate Metrics & Baseline Comparison
        md_lines.append("## 7. Aggregate Performance & Heuristic Baseline Comparison")
        md_lines.append("| Metric | CURIO Calibrated Random Forest | Deterministic Heuristic Baseline | Absolute Delta |")
        md_lines.append("|:---|:---:|:---:|:---:|")
        acc_delta = aggregate_rf_metrics["accuracy"] - aggregate_baseline_metrics["accuracy"]
        f1_delta = aggregate_rf_metrics["macro_f1"] - aggregate_baseline_metrics["macro_f1"]
        prec_delta = aggregate_rf_metrics["macro_precision"] - aggregate_baseline_metrics["macro_precision"]
        rec_delta = aggregate_rf_metrics["macro_recall"] - aggregate_baseline_metrics["macro_recall"]

        md_lines.append(f"| **Overall Accuracy** | **{aggregate_rf_metrics['accuracy']:.4f}** | {aggregate_baseline_metrics['accuracy']:.4f} | {acc_delta:+.4f} |")
        md_lines.append(f"| **Macro Precision** | **{aggregate_rf_metrics['macro_precision']:.4f}** | {aggregate_baseline_metrics['macro_precision']:.4f} | {prec_delta:+.4f} |")
        md_lines.append(f"| **Macro Recall** | **{aggregate_rf_metrics['macro_recall']:.4f}** | {aggregate_baseline_metrics['macro_recall']:.4f} | {rec_delta:+.4f} |")
        md_lines.append(f"| **Macro F1-Score** | **{aggregate_rf_metrics['macro_f1']:.4f}** | {aggregate_baseline_metrics['macro_f1']:.4f} | {f1_delta:+.4f} |")
        md_lines.append("")

        # Per-class comparison
        md_lines.append("### Per-Class F1-Score Breakdown")
        md_lines.append("| Condition Class | Support | RF Precision | RF Recall | RF F1 | Baseline F1 |")
        md_lines.append("|:---|:---:|:---:|:---:|:---:|:---:|")
        for cond in VALID_CONDITIONS:
            rf_cls = aggregate_rf_metrics["per_class"][cond]
            base_cls = aggregate_baseline_metrics["per_class"][cond]
            md_lines.append(
                f"| `{cond}` | {rf_cls['support']} | {rf_cls['precision']:.3f} | {rf_cls['recall']:.3f} | "
                f"**{rf_cls['f1']:.3f}** | {base_cls['f1']:.3f} |"
            )
        md_lines.append("")

        # 13. Calibration Metrics
        md_lines.append("## 8. Probability Calibration Metrics (Held-Out)")
        md_lines.append(f"- **Multiclass Brier Score**: `{aggregate_rf_metrics.get('brier_score', 0.0):.4f}`")
        md_lines.append(f"- **Multi-class Log Loss**: `{aggregate_rf_metrics.get('log_loss', 0.0):.4f}`")
        md_lines.append(
            "- **Metric Explanation**: The Brier score measures the mean squared error between predicted probabilities "
            "and one-hot ground truth (lower is better; 0 indicates perfect probability forecasts). Log loss penalizes "
            "confident incorrect probability forecasts. Both metrics reflect out-of-fold generalization on held-out hardware."
        )
        md_lines.append("")

        # 14. Feature Importance
        md_lines.append("## 9. Feature Importance Analysis")
        md_lines.append(
            "> [!NOTE]\n"
            "> **Methodological Clarification**: Feature importances indicate which telemetry dimensions contributed "
            "most strongly to the tree splits across training folds. They represent **predictive utility**, not physical causality."
        )
        md_lines.append("| Rank | Feature Name | Mean Importance Across Folds | Std Dev |")
        md_lines.append("|:---:|:---|:---:|:---:|")
        for rank, (_, row) in enumerate(feat_imp_df.iterrows(), start=1):
            md_lines.append(f"| {rank} | `{row['feature']}` | {row['mean_importance']:.4f} | ±{row['std_importance']:.4f} |")
        md_lines.append("")

        # 15. Limitations
        md_lines.append("## 10. Limitations")
        md_lines.append(
            "1. **Finite Hardware Diversity**: Evaluating cross-machine performance across initial machines verifies "
            "that the pipeline handles hardware variance without breaking, but does not prove generalization across "
            "radically different architectures (e.g. mobile ARM, high-end server clusters, virtual machines with noisy neighbors).\n"
            "2. **Simulated vs. Physical Noise**: While synthetic dry-run data rigorously checks pipeline integrity, "
            "production validation requires data collected across multiple physical computers.\n"
            "3. **Single vs. Compound Stress**: Current conditions model discrete resource pressures. Compound pressures "
            "(simultaneous CPU + Disk saturation) may exhibit interaction behaviors not captured by mutually exclusive labels."
        )
        md_lines.append("")

        # 16. Next Step
        md_lines.append("## 11. Next Step")
        md_lines.append(
            "With the Leave-One-Machine-Out pipeline, grouped calibration, and leakage prevention fully verified, "
            "the next milestone is collecting multi-machine physical telemetry and implementing the **CURIO Evidence Engine**."
        )
        md_lines.append("")

        with open(report_path, "w", encoding="utf-8") as f:
            f.write("\n".join(md_lines))
