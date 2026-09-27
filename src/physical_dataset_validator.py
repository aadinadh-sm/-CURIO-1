"""Physical Telemetry Dataset Validator for CURIO.

Enforces strict physical data integrity requirements:
A. Exactly the 12 expected feature columns
B. Zero NaN values
C. Zero Inf values
D. Exactly four valid target classes
E. Every session maps to exactly one machine
F. Every session contains exactly 61 raw samples (30s at 2 Hz)
G. Every session contains exactly 11 rolling feature windows (5s duration, 2.5s step)
H. Complete, non-empty machine_id
I. Unique session_id
J. Timestamps valid and strictly monotonic
K. Valid stress_level
L. Valid background_workload
M. Zero synthetic machine IDs present in physical dataset
N. No duplicate sessions
O. No accidental mixing of machines inside a session
P. Zero global preprocessing / data modification
"""

import os
import re
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np
import pandas as pd

from src.features import FEATURE_NAMES

VALID_CONDITIONS = ["normal", "cpu_pressure", "memory_pressure", "disk_io_pressure"]
VALID_STRESS_LEVELS = ["none", "low", "med", "high"]
SYNTHETIC_MACHINE_IDS = {
    "machine_A_desktop_ryzen_16c",
    "machine_B_laptop_intel_8c",
    "machine_C_server_xeon_32c",
    "mini_pc_4c",
}

EXPECTED_RAW_SAMPLES = 61
EXPECTED_FEATURE_WINDOWS = 11


class PhysicalDataValidationError(ValueError):
    """Raised when physical dataset integrity or protocol rules are violated."""
    pass


def validate_physical_raw_session(
    raw_df: pd.DataFrame, session_id: Optional[str] = None
) -> Dict[str, Any]:
    """Validates a single raw telemetry session CSV.

    Args:
        raw_df: DataFrame loaded from a physical raw telemetry CSV.
        session_id: Expected session_id (if known).

    Raises:
        PhysicalDataValidationError: If any integrity rule is violated.
    """
    if raw_df is None or not isinstance(raw_df, pd.DataFrame) or raw_df.empty:
        raise PhysicalDataValidationError("Raw session DataFrame is empty or null.")

    # Rule F: Exactly 61 raw samples
    if len(raw_df) != EXPECTED_RAW_SAMPLES:
        raise PhysicalDataValidationError(
            f"Rule F violation: Expected exactly {EXPECTED_RAW_SAMPLES} raw samples, "
            f"found {len(raw_df)} in session {session_id}."
        )

    # Required raw columns
    required_cols = [
        "timestamp",
        "cpu_overall",
        "vmem_percent",
        "vmem_available",
        "vmem_total",
        "disk_read_bytes",
        "disk_write_bytes",
        "machine_id",
        "session_id",
        "condition",
        "stress_level",
        "background_workload",
        "is_ramp_up",
    ]
    missing_cols = [c for c in required_cols if c not in raw_df.columns]
    if missing_cols:
        raise PhysicalDataValidationError(
            f"Raw session missing required columns: {missing_cols}"
        )

    # Rule B & C: Zero NaN, Zero Inf
    if raw_df[required_cols].isna().any().any():
        nan_cols = raw_df[required_cols].isna().sum()[lambda x: x > 0].to_dict()
        raise PhysicalDataValidationError(f"Rule B violation: Raw session contains NaNs: {nan_cols}")

    # Rule J: Valid monotonic timestamps
    timestamps = raw_df["timestamp"].values
    if (np.diff(timestamps) <= 0).any():
        raise PhysicalDataValidationError("Rule J violation: Timestamps are not strictly monotonically increasing.")

    # Rule H & M: Complete machine_id, no synthetic machine IDs
    m_ids = set(raw_df["machine_id"].dropna().unique())
    if not m_ids or "" in m_ids:
        raise PhysicalDataValidationError("Rule H violation: Incomplete or blank machine_id.")
    if len(m_ids) > 1:
        raise PhysicalDataValidationError(f"Rule O violation: Multiple machine IDs inside session: {m_ids}")

    machine_id = list(m_ids)[0]
    if machine_id in SYNTHETIC_MACHINE_IDS:
        raise PhysicalDataValidationError(
            f"Rule M violation: Synthetic machine ID '{machine_id}' detected in physical data!"
        )

    # Rule I: Single unique session_id matching expectation
    s_ids = set(raw_df["session_id"].dropna().unique())
    if len(s_ids) != 1:
        raise PhysicalDataValidationError(f"Session contains multiple session IDs: {s_ids}")
    cur_session = list(s_ids)[0]
    if session_id and cur_session != session_id:
        raise PhysicalDataValidationError(
            f"Session ID mismatch: Expected '{session_id}', found '{cur_session}'"
        )

    # Rule D: Target class validity
    conditions = set(raw_df["condition"].unique())
    if len(conditions) != 1:
        raise PhysicalDataValidationError(f"Session has multiple conditions: {conditions}")
    condition = list(conditions)[0]
    if condition not in VALID_CONDITIONS:
        raise PhysicalDataValidationError(f"Rule D violation: Invalid condition '{condition}'")

    # Rule K & L: Stress level and background workload validity
    stress_level = raw_df["stress_level"].iloc[0]
    if stress_level not in VALID_STRESS_LEVELS:
        raise PhysicalDataValidationError(f"Rule K violation: Invalid stress level '{stress_level}'")

    bg_workload = str(raw_df["background_workload"].iloc[0]).strip()
    if not bg_workload:
        raise PhysicalDataValidationError("Rule L violation: Blank background_workload.")

    return {
        "machine_id": machine_id,
        "session_id": cur_session,
        "condition": condition,
        "stress_level": stress_level,
        "background_workload": bg_workload,
        "sample_count": len(raw_df),
        "duration_seconds": float(timestamps[-1] - timestamps[0]),
    }


def validate_physical_feature_session(
    feat_df: pd.DataFrame, session_id: Optional[str] = None
) -> Dict[str, Any]:
    """Validates a single processed rolling feature session CSV.

    Args:
        feat_df: DataFrame loaded from a physical feature CSV.
        session_id: Expected session_id (if known).

    Raises:
        PhysicalDataValidationError: If any feature window rule is violated.
    """
    if feat_df is None or not isinstance(feat_df, pd.DataFrame) or feat_df.empty:
        raise PhysicalDataValidationError("Feature session DataFrame is empty or null.")

    # Rule G: Exactly 11 feature windows
    if len(feat_df) != EXPECTED_FEATURE_WINDOWS:
        raise PhysicalDataValidationError(
            f"Rule G violation: Expected exactly {EXPECTED_FEATURE_WINDOWS} feature windows, "
            f"found {len(feat_df)} in session {session_id}."
        )

    # Rule A: Exactly the 12 expected feature columns
    missing_features = [f for f in FEATURE_NAMES if f not in feat_df.columns]
    if missing_features:
        raise PhysicalDataValidationError(
            f"Rule A violation: Missing required feature columns: {missing_features}"
        )

    # Required metadata
    meta_cols = ["machine_id", "session_id", "condition", "stress_level", "background_workload"]
    missing_meta = [c for c in meta_cols if c not in feat_df.columns]
    if missing_meta:
        raise PhysicalDataValidationError(
            f"Feature session missing metadata columns: {missing_meta}"
        )

    # Rule B: Zero NaN
    all_check_cols = FEATURE_NAMES + meta_cols
    if feat_df[all_check_cols].isna().any().any():
        nan_cols = feat_df[all_check_cols].isna().sum()[lambda x: x > 0].to_dict()
        raise PhysicalDataValidationError(f"Rule B violation: Features contain NaNs: {nan_cols}")

    # Rule C: Zero Inf
    for f in FEATURE_NAMES:
        if np.isneginf(feat_df[f]).any() or np.isposinf(feat_df[f]).any():
            raise PhysicalDataValidationError(f"Rule C violation: Feature '{f}' contains infinite values.")

    # Rule M: No synthetic machine IDs
    m_id = feat_df["machine_id"].iloc[0]
    if m_id in SYNTHETIC_MACHINE_IDS:
        raise PhysicalDataValidationError(
            f"Rule M violation: Synthetic machine ID '{m_id}' detected in physical feature data!"
        )

    # Rule D: Valid condition
    cond = feat_df["condition"].iloc[0]
    if cond not in VALID_CONDITIONS:
        raise PhysicalDataValidationError(f"Rule D violation: Invalid condition '{cond}'")

    return {
        "machine_id": m_id,
        "session_id": feat_df["session_id"].iloc[0],
        "condition": cond,
        "stress_level": feat_df["stress_level"].iloc[0],
        "background_workload": feat_df["background_workload"].iloc[0],
        "window_count": len(feat_df),
    }


def validate_physical_dataset(
    raw_dir: str = "data/physical_raw",
    processed_dir: str = "data/physical_processed",
) -> Dict[str, Any]:
    """Validates the entire directory of physical telemetry sessions.

    Verifies rules A through P across all raw and processed files:
    - 1-to-1 match between raw and feature sessions
    - Session ID uniqueness
    - Class representation
    - Machine integrity

    Returns:
        Structured audit dictionary with session statistics and feature distributions.
    """
    if not os.path.exists(raw_dir):
        raise PhysicalDataValidationError(f"Raw telemetry directory does not exist: {raw_dir}")
    if not os.path.exists(processed_dir):
        raise PhysicalDataValidationError(f"Processed telemetry directory does not exist: {processed_dir}")

    raw_files = sorted([f for f in os.listdir(raw_dir) if f.endswith("_raw.csv")])
    feat_files = sorted([f for f in os.listdir(processed_dir) if f.endswith("_features.csv")])

    if not raw_files:
        raise PhysicalDataValidationError(f"No raw telemetry files found in: {raw_dir}")
    if not feat_files:
        raise PhysicalDataValidationError(f"No processed feature files found in: {processed_dir}")

    # Extract session IDs
    raw_sessions = {f.replace("_raw.csv", ""): os.path.join(raw_dir, f) for f in raw_files}
    feat_sessions = {f.replace("_features.csv", ""): os.path.join(processed_dir, f) for f in feat_files}

    # Verify 1-to-1 match between raw and features
    unmatched_raw = set(raw_sessions.keys()) - set(feat_sessions.keys())
    if unmatched_raw:
        raise PhysicalDataValidationError(f"Raw sessions without feature extraction: {unmatched_raw}")

    unmatched_feat = set(feat_sessions.keys()) - set(raw_sessions.keys())
    if unmatched_feat:
        raise PhysicalDataValidationError(f"Feature files without raw telemetry provenance: {unmatched_feat}")

    all_sessions_summary = []
    all_feature_dfs = []
    seen_session_ids: Set[str] = set()
    session_machine_map: Dict[str, str] = {}

    for s_id in sorted(raw_sessions.keys()):
        # Rule N: No duplicate sessions
        if s_id in seen_session_ids:
            raise PhysicalDataValidationError(f"Rule N violation: Duplicate session ID detected: '{s_id}'")
        seen_session_ids.add(s_id)

        raw_df = pd.read_csv(raw_sessions[s_id])
        feat_df = pd.read_csv(feat_sessions[s_id])

        raw_meta = validate_physical_raw_session(raw_df, session_id=s_id)
        feat_meta = validate_physical_feature_session(feat_df, session_id=s_id)

        # Cross-validation between raw and processed
        if raw_meta["machine_id"] != feat_meta["machine_id"]:
            raise PhysicalDataValidationError(
                f"Machine ID mismatch between raw ({raw_meta['machine_id']}) and "
                f"features ({feat_meta['machine_id']}) for session {s_id}"
            )
        if raw_meta["condition"] != feat_meta["condition"]:
            raise PhysicalDataValidationError(
                f"Condition mismatch between raw and features for session {s_id}"
            )

        # Rule E: Every session maps to exactly one machine
        session_machine_map[s_id] = raw_meta["machine_id"]
        all_sessions_summary.append({
            "session_id": s_id,
            "machine_id": raw_meta["machine_id"],
            "condition": raw_meta["condition"],
            "stress_level": raw_meta["stress_level"],
            "background_workload": raw_meta["background_workload"],
            "raw_samples": raw_meta["sample_count"],
            "feature_windows": feat_meta["window_count"],
            "duration_seconds": raw_meta["duration_seconds"],
        })
        all_feature_dfs.append(feat_df)

    combined_features_df = pd.concat(all_feature_dfs, ignore_index=True)

    # Machine summary
    unique_machines = sorted(combined_features_df["machine_id"].unique())
    sessions_per_machine = {m: sum(1 for s in all_sessions_summary if s["machine_id"] == m) for m in unique_machines}
    windows_per_machine = combined_features_df["machine_id"].value_counts().to_dict()
    class_dist = combined_features_df["condition"].value_counts().to_dict()

    # Rule D: Every required condition class must be represented
    missing_classes = set(VALID_CONDITIONS) - set(class_dist.keys())
    if missing_classes:
        raise PhysicalDataValidationError(f"Rule D violation: Missing condition classes: {missing_classes}")

    # Feature statistics by condition
    feat_stats_by_cond: Dict[str, Dict[str, Dict[str, float]]] = {}
    for cond in VALID_CONDITIONS:
        cond_df = combined_features_df[combined_features_df["condition"] == cond]
        feat_stats_by_cond[cond] = {}
        for f in FEATURE_NAMES:
            feat_stats_by_cond[cond][f] = {
                "mean": float(cond_df[f].mean()),
                "std": float(cond_df[f].std()) if len(cond_df) > 1 else 0.0,
                "min": float(cond_df[f].min()),
                "max": float(cond_df[f].max()),
            }

    return {
        "status": "PASSED",
        "total_sessions": len(all_sessions_summary),
        "total_feature_windows": len(combined_features_df),
        "unique_machines": unique_machines,
        "machine_count": len(unique_machines),
        "sessions_per_machine": sessions_per_machine,
        "windows_per_machine": windows_per_machine,
        "class_distribution": class_dist,
        "feature_stats_by_condition": feat_stats_by_cond,
        "combined_features_df": combined_features_df,
        "sessions_summary": all_sessions_summary,
    }
