"""Evidence Engine for CURIO.

Explains *why* CURIO made a specific diagnosis by evaluating observed telemetry signals
against a training-derived Normal operating reference:
1. What condition was predicted?
2. Which observed telemetry signals support that prediction?
3. How strongly did those signals deviate in the expected direction from learned Normal reference?

Strict Methodological Principles:
- Local vs Global: Evaluates current telemetry deviation, NEVER uses global RF feature importance as per-instance explanation.
- Training-Only Reference: Reference statistics are fitted strictly from training data, never from test sessions.
- Probability != Evidence: Model confidence (calibrated probability) is kept strictly separate from evidence strength.
- Non-Causal Wording: Central wording policy prohibits causal claims ("caused", "root cause", "hardware failure").
- Safe Numerical Handling: Bounded z-scores, division-by-zero protection (sigma_safe = max(std, epsilon)).
"""

import json
import math
import os
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import numpy as np
import pandas as pd

from src.features import FEATURE_NAMES

# ----------------------------------------------------------------------
# Constants & Configuration
# ----------------------------------------------------------------------

VALID_CONDITIONS = ["normal", "cpu_pressure", "memory_pressure", "disk_io_pressure"]

# Evidence Score Bands for UI and Reporting
STRONG_EVIDENCE_THRESHOLD = 2.0
MODERATE_EVIDENCE_THRESHOLD = 1.0
WEAK_EVIDENCE_THRESHOLD = -1.0
CONTRADICTORY_THRESHOLD = -1.0  # Directional score <= -1.0 is considered Contradictory Evidence

# Numerical Safety Parameters
SAFE_EPSILON = 1e-4
SCORE_CAP = 10.0  # Max magnitude for display to avoid single stable feature explosion

# Subsystem Feature Groups
SUBSYSTEM_MAP: Dict[str, str] = {
    "cpu_mean": "cpu",
    "cpu_max": "cpu",
    "cpu_std": "cpu",
    "cpu_core_imbalance": "cpu",
    "top_proc_cpu_ratio": "cpu",
    "process_count_delta": "cpu",
    "ram_used_pct": "memory",
    "ram_available_ratio": "memory",
    "swap_used_pct": "memory",
    "top_proc_mem_pct": "memory",
    "disk_io_rate_norm": "disk",
    "disk_iops_norm": "disk",
}

# Condition-relevant primary features (used to assess evidence consistency)
CONDITION_RELEVANT_FEATURES: Dict[str, List[str]] = {
    "cpu_pressure": [
        "cpu_mean",
        "cpu_max",
        "cpu_std",
        "cpu_core_imbalance",
        "top_proc_cpu_ratio",
        "process_count_delta",
    ],
    "memory_pressure": [
        "ram_used_pct",
        "ram_available_ratio",
        "swap_used_pct",
        "top_proc_mem_pct",
    ],
    "disk_io_pressure": [
        "disk_io_rate_norm",
        "disk_iops_norm",
    ],
    "normal": FEATURE_NAMES,
}

FEATURE_DISPLAY_METADATA: Dict[str, Dict[str, str]] = {
    "cpu_mean": {
        "display_name": "Average CPU Utilization",
        "unit": "%",
        "subsystem": "cpu",
        "increase_phrase": "elevated",
        "decrease_phrase": "reduced",
    },
    "cpu_max": {
        "display_name": "Peak Core CPU Utilization",
        "unit": "%",
        "subsystem": "cpu",
        "increase_phrase": "elevated",
        "decrease_phrase": "reduced",
    },
    "cpu_std": {
        "display_name": "CPU Volatility",
        "unit": "%",
        "subsystem": "cpu",
        "increase_phrase": "increased",
        "decrease_phrase": "smoothed",
    },
    "cpu_core_imbalance": {
        "display_name": "CPU Core Load Imbalance",
        "unit": "%",
        "subsystem": "cpu",
        "increase_phrase": "elevated",
        "decrease_phrase": "balanced",
    },
    "ram_used_pct": {
        "display_name": "Physical RAM Utilization",
        "unit": "%",
        "subsystem": "memory",
        "increase_phrase": "elevated",
        "decrease_phrase": "reduced",
    },
    "ram_available_ratio": {
        "display_name": "Available Memory Ratio",
        "unit": "ratio",
        "subsystem": "memory",
        "increase_phrase": "elevated",
        "decrease_phrase": "substantially reduced",
    },
    "swap_used_pct": {
        "display_name": "Swap / Pagefile Utilization",
        "unit": "%",
        "subsystem": "memory",
        "increase_phrase": "elevated",
        "decrease_phrase": "reduced",
    },
    "disk_io_rate_norm": {
        "display_name": "Disk I/O Throughput Rate",
        "unit": "log10(B/s)",
        "subsystem": "disk",
        "increase_phrase": "elevated",
        "decrease_phrase": "subdued",
    },
    "disk_iops_norm": {
        "display_name": "Disk Operation Frequency (IOPS)",
        "unit": "log10(ops/s)",
        "subsystem": "disk",
        "increase_phrase": "elevated",
        "decrease_phrase": "subdued",
    },
    "process_count_delta": {
        "display_name": "Process Count Variation",
        "unit": "count",
        "subsystem": "process",
        "increase_phrase": "increased",
        "decrease_phrase": "decreased",
    },
    "top_proc_cpu_ratio": {
        "display_name": "Top-Process CPU Dominance",
        "unit": "ratio",
        "subsystem": "cpu",
        "increase_phrase": "concentrated in top process",
        "decrease_phrase": "dispersed across processes",
    },
    "top_proc_mem_pct": {
        "display_name": "Top-Process Memory Share",
        "unit": "%",
        "subsystem": "memory",
        "increase_phrase": "elevated",
        "decrease_phrase": "reduced",
    },
}

# ----------------------------------------------------------------------
# Central Wording Policy & Causal Language Guardrails
# ----------------------------------------------------------------------

FORBIDDEN_CAUSAL_PATTERNS = [
    re.compile(r"\bcaused\b", re.IGNORECASE),
    re.compile(r"\bcauses\b", re.IGNORECASE),
    re.compile(r"\bcause\s+of\b", re.IGNORECASE),
    re.compile(r"\broot\s*cause\b", re.IGNORECASE),
    re.compile(r"\bproved\b", re.IGNORECASE),
    re.compile(r"\bproves\b", re.IGNORECASE),
    re.compile(r"\bresulted\s+in\b", re.IGNORECASE),
    re.compile(r"\bhardware\s+failure\b", re.IGNORECASE),
    re.compile(r"\bbroken\s+hardware\b", re.IGNORECASE),
    re.compile(r"\bfailing\s+hardware\b", re.IGNORECASE),
    re.compile(r"\bchance\s+of\s+damage\b", re.IGNORECASE),
]


class CausalLanguageError(ValueError):
    """Raised when generated explanation text violates the non-causal wording policy."""
    pass


class InvalidTelemetryError(ValueError):
    """Raised when input telemetry contains NaN, Inf, or invalid feature values."""
    pass


def validate_no_causal_language(text: str) -> str:
    """Verifies that the text does not contain forbidden causal or hardware failure claims."""
    for pattern in FORBIDDEN_CAUSAL_PATTERNS:
        match = pattern.search(text)
        if match:
            raise CausalLanguageError(
                f"Causal language violation: Forbidden phrase '{match.group(0)}' detected in explanation text: '{text}'"
            )
    return text


# ----------------------------------------------------------------------
# Training-Only Reference Profile
# ----------------------------------------------------------------------

class EvidenceReferenceProfile:
    """Encapsulates training-derived Normal baseline statistics and condition-specific directions.

    Never fitted on test data or held-out validation machines.
    """

    def __init__(
        self,
        schema_version: str = "1.0.0",
        created_at: Optional[str] = None,
        training_machines: Optional[List[str]] = None,
        normal_sessions_count: int = 0,
        total_training_samples: int = 0,
        feature_stats: Optional[Dict[str, Dict[str, float]]] = None,
        expected_directions: Optional[Dict[str, Dict[str, int]]] = None,
        class_means: Optional[Dict[str, Dict[str, float]]] = None,
        epsilon: float = SAFE_EPSILON,
        score_cap: float = SCORE_CAP,
    ):
        self.schema_version = schema_version
        self.created_at = created_at or datetime.now(timezone.utc).isoformat()
        self.training_machines = training_machines or []
        self.normal_sessions_count = normal_sessions_count
        self.total_training_samples = total_training_samples
        self.feature_stats = feature_stats or {}
        self.expected_directions = expected_directions or {}
        self.class_means = class_means or {}
        self.epsilon = epsilon
        self.score_cap = score_cap

    @classmethod
    def fit_from_training_data(
        cls,
        train_df: pd.DataFrame,
        training_machines: Optional[List[str]] = None,
        min_direction_z: float = 0.5,
    ) -> "EvidenceReferenceProfile":
        """Fits Normal baseline statistics and class-specific expected directions strictly from training data.

        Args:
            train_df: DataFrame containing training sessions (must have 'condition' and all FEATURE_NAMES).
            training_machines: Optional explicit list of training machine IDs.
            min_direction_z: Minimum normalized difference to assign a +1 or -1 expected direction.

        Raises:
            ValueError: If training data is empty, lacks Normal sessions, or missing required features.
        """
        if train_df is None or train_df.empty:
            raise ValueError("Training DataFrame cannot be empty.")

        for f in FEATURE_NAMES:
            if f not in train_df.columns:
                raise ValueError(f"Training data missing required feature: '{f}'")

        if "condition" not in train_df.columns:
            raise ValueError("Training data must contain 'condition' column.")

        normal_mask = train_df["condition"] == "normal"
        normal_df = train_df[normal_mask]
        if normal_df.empty:
            raise ValueError("Training data contains zero Normal sessions. Cannot fit Normal reference profile.")

        normal_sessions_cnt = int(normal_df["session_id"].nunique()) if "session_id" in normal_df.columns else len(normal_df)

        if training_machines is None:
            if "machine_id" in train_df.columns:
                training_machines = sorted(list(set(train_df["machine_id"].dropna().unique())))
            else:
                training_machines = ["unknown_training_hardware"]

        # 1. Compute Normal baseline statistics for each feature
        feature_stats: Dict[str, Dict[str, float]] = {}
        for f in FEATURE_NAMES:
            vals = normal_df[f].to_numpy(dtype=float)
            mean_val = float(np.mean(vals))
            std_val = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
            median_val = float(np.median(vals))
            q25, q75 = np.percentile(vals, [25, 75])
            iqr_val = float(q75 - q25)

            feature_stats[f] = {
                "mean": mean_val,
                "std": std_val,
                "median": median_val,
                "iqr": iqr_val,
            }

        # 2. Compute class means and determine expected directions for abnormal conditions
        class_means: Dict[str, Dict[str, float]] = {}
        expected_directions: Dict[str, Dict[str, int]] = {c: {} for c in VALID_CONDITIONS}

        # Normal condition has 0 expected direction for all features (it is the reference)
        for f in FEATURE_NAMES:
            expected_directions["normal"][f] = 0

        for cond in ["cpu_pressure", "memory_pressure", "disk_io_pressure"]:
            cond_mask = train_df["condition"] == cond
            cond_df = train_df[cond_mask]
            class_means[cond] = {}

            for f in FEATURE_NAMES:
                if not cond_df.empty:
                    c_mean = float(np.mean(cond_df[f].to_numpy(dtype=float)))
                else:
                    c_mean = feature_stats[f]["mean"]
                class_means[cond][f] = c_mean

                # Normalized difference relative to normal std
                norm_mean = feature_stats[f]["mean"]
                norm_std = feature_stats[f]["std"]
                sigma_safe = max(norm_std, SAFE_EPSILON)
                delta_z = (c_mean - norm_mean) / sigma_safe

                # Direction assignment based on empirical training shift
                if delta_z >= min_direction_z:
                    expected_directions[cond][f] = 1
                elif delta_z <= -min_direction_z:
                    expected_directions[cond][f] = -1
                else:
                    expected_directions[cond][f] = 0

            # Domain-anchored consistency checks (ensures foundational physical directions)
            if cond == "cpu_pressure":
                expected_directions[cond]["cpu_mean"] = 1
                expected_directions[cond]["cpu_max"] = 1
            elif cond == "memory_pressure":
                expected_directions[cond]["ram_used_pct"] = 1
                expected_directions[cond]["ram_available_ratio"] = -1
            elif cond == "disk_io_pressure":
                expected_directions[cond]["disk_io_rate_norm"] = 1
                expected_directions[cond]["disk_iops_norm"] = 1

        return cls(
            schema_version="1.0.0",
            created_at=datetime.now(timezone.utc).isoformat(),
            training_machines=training_machines,
            normal_sessions_count=normal_sessions_cnt,
            total_training_samples=len(train_df),
            feature_stats=feature_stats,
            expected_directions=expected_directions,
            class_means=class_means,
            epsilon=SAFE_EPSILON,
            score_cap=SCORE_CAP,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serializes the reference profile to a dictionary."""
        return {
            "schema_version": self.schema_version,
            "created_at": self.created_at,
            "training_machines": self.training_machines,
            "normal_sessions_count": self.normal_sessions_count,
            "total_training_samples": self.total_training_samples,
            "feature_stats": self.feature_stats,
            "expected_directions": self.expected_directions,
            "class_means": self.class_means,
            "epsilon": self.epsilon,
            "score_cap": self.score_cap,
            "thresholds": {
                "strong_evidence": STRONG_EVIDENCE_THRESHOLD,
                "moderate_evidence": MODERATE_EVIDENCE_THRESHOLD,
                "weak_evidence": WEAK_EVIDENCE_THRESHOLD,
                "contradictory": CONTRADICTORY_THRESHOLD,
            },
        }

    def to_json(self, filepath: str) -> None:
        """Writes reference profile to a JSON file."""
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EvidenceReferenceProfile":
        """Deserializes a reference profile from a dictionary."""
        return cls(
            schema_version=data.get("schema_version", "1.0.0"),
            created_at=data.get("created_at"),
            training_machines=data.get("training_machines", []),
            normal_sessions_count=data.get("normal_sessions_count", 0),
            total_training_samples=data.get("total_training_samples", 0),
            feature_stats=data.get("feature_stats", {}),
            expected_directions=data.get("expected_directions", {}),
            class_means=data.get("class_means", {}),
            epsilon=data.get("epsilon", SAFE_EPSILON),
            score_cap=data.get("score_cap", SCORE_CAP),
        )

    @classmethod
    def from_json(cls, filepath: str) -> "EvidenceReferenceProfile":
        """Loads a reference profile from a JSON file."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Reference profile file not found: {filepath}")
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)


# ----------------------------------------------------------------------
# Evidence Generation Core
# ----------------------------------------------------------------------

def _format_value_for_statement(feature_name: str, value: float) -> str:
    """Formats numeric values appropriately for human-readable statements."""
    if feature_name in ("ram_used_pct", "swap_used_pct", "cpu_mean", "cpu_max", "cpu_std", "cpu_core_imbalance", "top_proc_mem_pct"):
        return f"{value:.1f}%"
    elif feature_name in ("ram_available_ratio", "top_proc_cpu_ratio"):
        return f"{value:.2f}"
    elif feature_name in ("disk_io_rate_norm", "disk_iops_norm"):
        return f"{value:.2f}"
    elif feature_name == "process_count_delta":
        sign = "+" if value > 0 else ""
        return f"{sign}{int(round(value))}"
    return f"{value:.2f}"


def build_evidence_statement(
    feature_name: str,
    observed_val: float,
    normal_ref: float,
    directional_score: float,
    direction: str,
    predicted_condition: str,
) -> str:
    """Builds a human-readable, strictly non-causal explanation statement for a feature."""
    meta = FEATURE_DISPLAY_METADATA.get(
        feature_name,
        {"display_name": feature_name, "increase_phrase": "increased", "decrease_phrase": "decreased"},
    )
    disp = meta["display_name"]
    obs_str = _format_value_for_statement(feature_name, observed_val)
    ref_str = _format_value_for_statement(feature_name, normal_ref)

    if predicted_condition == "normal":
        if abs(directional_score) < MODERATE_EVIDENCE_THRESHOLD:
            statement = f"{disp} ({obs_str}) was within the expected learned Normal operating range (ref {ref_str})."
        else:
            statement = f"{disp} ({obs_str}) deviated slightly from Normal reference (ref {ref_str}) but remained below pressure thresholds."
        return validate_no_causal_language(statement)

    # Abnormal conditions
    if directional_score >= STRONG_EVIDENCE_THRESHOLD:
        if direction in ("decreased", "reduced", "substantially reduced"):
            statement = f"{disp} ({obs_str}) was substantially lower than the learned Normal reference ({ref_str})."
        else:
            statement = f"{disp} ({obs_str}) was substantially elevated relative to the learned Normal reference ({ref_str})."
    elif directional_score >= MODERATE_EVIDENCE_THRESHOLD:
        if direction in ("decreased", "reduced"):
            statement = f"{disp} ({obs_str}) showed moderate reduction compared to Normal reference ({ref_str})."
        else:
            statement = f"{disp} ({obs_str}) showed moderate elevation above the learned Normal reference ({ref_str})."
    elif directional_score <= CONTRADICTORY_THRESHOLD:
        statement = (
            f"{disp} ({obs_str}) moved against the expected direction for {predicted_condition.replace('_', ' ').title()} "
            f"(observed {obs_str} vs Normal reference {ref_str})."
        )
    else:
        statement = f"{disp} ({obs_str}) showed weak or neutral deviation from Normal reference ({ref_str})."

    return validate_no_causal_language(statement)


def build_overall_interpretation(
    predicted_condition: str,
    supporting_count: int,
    contradictory_count: int,
    top_subsystems: List[str],
) -> str:
    """Generates the overall summary interpretation without causal claims."""
    if predicted_condition == "normal":
        text = (
            "All observed telemetry signals remain consistent with the learned Normal operating reference. "
            "No significant abnormal directional deviations were detected."
        )
        return validate_no_causal_language(text)

    cond_name = predicted_condition.replace("_", " ").title()
    subsystem_str = " and ".join(top_subsystems) if top_subsystems else "system"

    if supporting_count >= 2:
        text = f"Multiple {subsystem_str}-related telemetry signals moved in the expected direction for {cond_name}."
    elif supporting_count == 1:
        text = f"A primary {subsystem_str} telemetry signal moved in the expected direction for {cond_name}."
    else:
        text = f"Telemetry signals showed limited directional support for {cond_name}."

    if contradictory_count > 0:
        text += f" Note: {contradictory_count} signal(s) moved against expected directions."

    return validate_no_causal_language(text)


def generate_evidence(
    predicted_condition: str,
    calibrated_probabilities: Union[Dict[str, float], np.ndarray, List[float]],
    current_features: Union[Dict[str, float], pd.Series, np.ndarray],
    reference_profile: EvidenceReferenceProfile,
    abnormality_score: Optional[float] = None,
    classes: Optional[List[str]] = None,
    max_supporting: int = 3,
    max_contradictory: int = 2,
    strict_validation: bool = True,
) -> Dict[str, Any]:
    """Generates a structured, evidence-backed explanation for a CURIO diagnosis.

    Answers:
    1. What condition was predicted?
    2. Which observed telemetry signals support that prediction?
    3. How strongly did those signals deviate from the learned Normal operating reference?

    Args:
        predicted_condition: 'normal', 'cpu_pressure', 'memory_pressure', 'disk_io_pressure'.
        calibrated_probabilities: Probability dictionary, array, or list for classes.
        current_features: Feature vector (dict, Series, or 1D array of 12 features).
        reference_profile: Training-derived Normal reference statistics & expected directions.
        abnormality_score: Optional calibrated abnormality score (1 - P(Normal)).
        classes: Optional ordered class list corresponding to probability array.
        max_supporting: Maximum number of top supporting features to highlight (default 3).
        max_contradictory: Maximum number of contradictory features to highlight (default 2).
        strict_validation: If True, raises InvalidTelemetryError on NaN/Inf/missing. If False, excludes safely.

    Returns:
        Structured evidence dictionary conforming to CURIO Evidence Engine schema.
    """
    # 1. Condition Validation
    if predicted_condition not in VALID_CONDITIONS:
        raise ValueError(
            f"Invalid predicted condition '{predicted_condition}'. Allowed: {VALID_CONDITIONS}"
        )

    # 2. Probability Validation & Extraction
    if classes is None:
        classes = sorted(VALID_CONDITIONS)

    prob_dict: Dict[str, float] = {}
    if isinstance(calibrated_probabilities, dict):
        prob_dict = {str(k): float(v) for k, v in calibrated_probabilities.items()}
    elif isinstance(calibrated_probabilities, (list, np.ndarray)):
        prob_arr = np.asarray(calibrated_probabilities, dtype=float).flatten()
        if len(prob_arr) != len(classes):
            raise ValueError(
                f"Probabilities length ({len(prob_arr)}) does not match classes count ({len(classes)})."
            )
        prob_dict = {classes[i]: float(prob_arr[i]) for i in range(len(classes))}
    else:
        raise ValueError(f"Unsupported calibrated_probabilities type: {type(calibrated_probabilities)}")

    confidence = prob_dict.get(predicted_condition, 0.0)
    if not (0.0 <= confidence <= 1.0001):
        raise ValueError(f"Calibrated probability for '{predicted_condition}' must be in [0, 1], got {confidence}")

    if abnormality_score is None and "normal" in prob_dict:
        abnormality_score = float(max(0.0, min(1.0, 1.0 - prob_dict["normal"])))

    # 3. Features Validation & Extraction
    feat_dict: Dict[str, float] = {}
    if isinstance(current_features, dict):
        feat_dict = dict(current_features)
    elif isinstance(current_features, pd.Series):
        feat_dict = current_features.to_dict()
    elif isinstance(current_features, np.ndarray):
        arr = current_features.flatten()
        if len(arr) != len(FEATURE_NAMES):
            raise ValueError(f"Feature array length ({len(arr)}) does not match FEATURE_NAMES ({len(FEATURE_NAMES)}).")
        feat_dict = dict(zip(FEATURE_NAMES, arr.tolist()))
    else:
        raise ValueError(f"Unsupported current_features type: {type(current_features)}")

    # Check for missing, NaN, Inf
    missing_feats = [f for f in FEATURE_NAMES if f not in feat_dict]
    if missing_feats:
        if strict_validation:
            raise InvalidTelemetryError(f"Missing required telemetry features: {missing_feats}")

    invalid_features: List[Dict[str, Any]] = []
    valid_feat_dict: Dict[str, float] = {}

    for f in FEATURE_NAMES:
        if f not in feat_dict:
            invalid_features.append({"feature_name": f, "status": "missing"})
            continue

        raw_val = feat_dict[f]
        try:
            val = float(raw_val)
        except (ValueError, TypeError):
            if strict_validation:
                raise InvalidTelemetryError(f"Feature '{f}' contains non-numeric value: {raw_val}")
            invalid_features.append({"feature_name": f, "status": "non_numeric", "raw_value": str(raw_val)})
            continue

        if math.isnan(val):
            if strict_validation:
                raise InvalidTelemetryError(f"Feature '{f}' contains NaN.")
            invalid_features.append({"feature_name": f, "status": "nan"})
            continue

        if math.isinf(val):
            if strict_validation:
                raise InvalidTelemetryError(f"Feature '{f}' contains infinite value: {val}")
            invalid_features.append({"feature_name": f, "status": "inf"})
            continue

        valid_feat_dict[f] = val

    # 4. Evaluate Telemetry against Training Reference
    all_evaluated: List[Dict[str, Any]] = []

    for f, obs_val in valid_feat_dict.items():
        stats = reference_profile.feature_stats.get(f, {"mean": 0.0, "std": 1.0, "median": 0.0, "iqr": 0.0})
        norm_mean = stats["mean"]
        norm_std = stats["std"]

        # Safe division
        sigma_safe = max(norm_std, reference_profile.epsilon)
        z_raw = (obs_val - norm_mean) / sigma_safe

        # Score cap to prevent single stable feature from dominating absurdly
        z_capped = float(np.clip(z_raw, -reference_profile.score_cap, reference_profile.score_cap))

        expected_dir = reference_profile.expected_directions.get(predicted_condition, {}).get(f, 0)

        # Directional Score calculation:
        # If expected_dir is +1 (expected to increase), positive z -> positive directional score.
        # If expected_dir is -1 (expected to decrease, e.g. available RAM), negative z -> positive directional score.
        # If expected_dir is 0 (no expected directional change), directional score is 0.
        if predicted_condition == "normal":
            # For Normal, directional deviation is evaluated as closeness to 0
            directional_score = -abs(z_capped)  # 0 is best, larger deviation is negative support for Normal
        else:
            directional_score = float(expected_dir * z_capped)

        # Direction string
        if z_raw >= 0.5:
            direction_str = "elevated" if expected_dir >= 0 else "increased"
        elif z_raw <= -0.5:
            direction_str = "reduced" if expected_dir <= 0 else "decreased"
        else:
            direction_str = "stable"

        # Evidence Strength Classification
        if predicted_condition == "normal":
            if abs(z_raw) < 1.0:
                strength_str = "strong"  # firmly within normal
            elif abs(z_raw) < 2.0:
                strength_str = "moderate"
            elif abs(z_raw) < 3.0:
                strength_str = "weak"
            else:
                strength_str = "contradictory"
        else:
            if directional_score >= STRONG_EVIDENCE_THRESHOLD:
                strength_str = "strong"
            elif directional_score >= MODERATE_EVIDENCE_THRESHOLD:
                strength_str = "moderate"
            elif directional_score <= CONTRADICTORY_THRESHOLD:
                strength_str = "contradictory"
            else:
                strength_str = "weak"

        statement = build_evidence_statement(
            feature_name=f,
            observed_val=obs_val,
            normal_ref=norm_mean,
            directional_score=directional_score,
            direction=direction_str,
            predicted_condition=predicted_condition,
        )

        all_evaluated.append({
            "feature_name": f,
            "display_name": FEATURE_DISPLAY_METADATA.get(f, {}).get("display_name", f),
            "observed_value": float(round(obs_val, 4)),
            "normal_reference": float(round(norm_mean, 4)),
            "normal_std": float(round(norm_std, 4)),
            "raw_z": float(round(z_raw, 3)),
            "directional_deviation": float(round(directional_score, 3)),
            "expected_direction": int(expected_dir),
            "directional_score": float(round(directional_score, 3)),
            "direction": direction_str,
            "evidence_strength": strength_str,
            "subsystem": SUBSYSTEM_MAP.get(f, "general"),
            "human_readable_statement": statement,
        })

    # 5. Evidence Ranking & Categorization
    supporting: List[Dict[str, Any]] = []
    contradictory: List[Dict[str, Any]] = []
    neutral: List[Dict[str, Any]] = []

    if predicted_condition == "normal":
        # For Normal, rank by how close the signal is to Normal reference (smallest |raw_z| first)
        sorted_for_normal = sorted(all_evaluated, key=lambda x: abs(x["raw_z"]))
        supporting = sorted_for_normal[:max_supporting]
        # Any signal deviating significantly from normal (|raw_z| >= 2.0) is flagged
        contradictory = [x for x in sorted_for_normal if abs(x["raw_z"]) >= 2.0][:max_contradictory]
        neutral = [x for x in sorted_for_normal if x not in supporting and x not in contradictory]
    else:
        # For Abnormal conditions, rank supporting by directional_score descending
        candidate_supporting = [
            x for x in all_evaluated if x["directional_score"] >= MODERATE_EVIDENCE_THRESHOLD
        ]
        candidate_supporting.sort(key=lambda x: x["directional_score"], reverse=True)
        supporting = candidate_supporting[:max_supporting]

        # Contradictory: features that moved in the opposite direction (directional_score <= CONTRADICTORY_THRESHOLD)
        candidate_contradictory = [
            x for x in all_evaluated if x["directional_score"] <= CONTRADICTORY_THRESHOLD
        ]
        # Most contradictory (most negative) first
        candidate_contradictory.sort(key=lambda x: x["directional_score"])
        contradictory = candidate_contradictory[:max_contradictory]

        # Neutral: features with weak/neutral deviation (-1.0 < score < 1.0)
        used_names = {x["feature_name"] for x in supporting + contradictory}
        neutral = [x for x in all_evaluated if x["feature_name"] not in used_names]
        neutral.sort(key=lambda x: abs(x["directional_score"]), reverse=True)

    # 6. Multi-Feature Evidence Consistency
    relevant_feats = CONDITION_RELEVANT_FEATURES.get(predicted_condition, FEATURE_NAMES)
    relevant_eval = [x for x in all_evaluated if x["feature_name"] in relevant_feats]

    if predicted_condition == "normal":
        # For Normal: fraction of relevant features within normal bounds (|z| < 2.0)
        consistent_count = sum(1 for x in relevant_eval if abs(x["raw_z"]) < 2.0)
        evidence_consistency = float(round(consistent_count / max(1, len(relevant_eval)), 3))
    else:
        # For Abnormal: fraction of condition-relevant features with positive supporting direction (score >= 1.0)
        consistent_count = sum(1 for x in relevant_eval if x["directional_score"] >= MODERATE_EVIDENCE_THRESHOLD)
        evidence_consistency = float(round(consistent_count / max(1, len(relevant_eval)), 3))

    # Top Subsystems
    top_subsystems = []
    for item in supporting:
        sub = item.get("subsystem", "system")
        if sub not in top_subsystems and sub != "general":
            top_subsystems.append(sub)

    # Aggregate Evidence Score (mean of top supporting or max support)
    if supporting:
        overall_evidence_score = float(round(np.mean([x["directional_score"] for x in supporting]), 3))
    else:
        overall_evidence_score = 0.0

    # Overall Interpretation Text
    overall_interpretation = build_overall_interpretation(
        predicted_condition=predicted_condition,
        supporting_count=len(supporting),
        contradictory_count=len(contradictory),
        top_subsystems=top_subsystems,
    )

    result: Dict[str, Any] = {
        "condition": predicted_condition,
        "confidence": float(round(confidence, 4)),
        "abnormality_score": float(round(abnormality_score, 4)) if abnormality_score is not None else None,
        "overall_interpretation": overall_interpretation,
        "supporting_evidence": supporting,
        "contradictory_evidence": contradictory,
        "neutral_features": neutral,
        "evidence_score": overall_evidence_score,
        "evidence_consistency": evidence_consistency,
        "invalid_features": invalid_features,
        "class_probabilities": prob_dict,
        "evaluated_features_count": len(all_evaluated),
    }

    return result
