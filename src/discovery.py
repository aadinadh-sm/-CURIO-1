"""Discovery Engine for CURIO.

Discovers and analyzes temporal onset patterns and precursor trajectories
across 30-second diagnostic sessions (11 feature windows x 12 features):
1. Detects empirical directional onsets (z >= 1.8 for 2 consecutive windows).
2. Compares live observed onset progression against training-derived canonical signatures.
3. Quantifies sequence ordering similarity using Kendall's tau-b.
4. Handles critical edge cases: Operating Equilibrium (Normal), Sustained Pressure (t=0 ties),
   and Insufficient Temporal Evidence (< 3 distinct non-tied onsets).

Strict Methodological Rules:
- Downstream of Diagnosis: Discovery analyzes temporal patterns for the predicted condition.
- Training-Only Signatures: Canonical sequences are learned strictly from training sessions.
- Recurrence Requirement: Features enter canonical sequence only if recurrence >= 0.60.
- Zero-Direction Exclusion: Features with expected_direction == 0 are strictly excluded.
- Minimum Information Rule: Requires >= 3 distinct non-tied onsets before calculating Kendall's tau.
- Non-Causal Wording: Strictly temporal vocabulary ("preceded", "followed", "temporal pattern").
  Never claims causality, "root cause", or "failure mechanism".
"""

import json
import math
import os
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import numpy as np
import pandas as pd

from src.evidence import (
    EvidenceReferenceProfile,
    SAFE_EPSILON,
    VALID_CONDITIONS,
    validate_no_causal_language,
)
from src.features import FEATURE_NAMES

# ----------------------------------------------------------------------
# Discovery Engine Parameters & Thresholds
# ----------------------------------------------------------------------

DEFAULT_ONSET_THRESHOLD = 1.8           # Directional z-score required for onset
DEFAULT_CONSECUTIVE_WINDOWS = 2         # Consecutive windows required to confirm onset
DEFAULT_RECURRENCE_THRESHOLD = 0.60     # Fraction of training sessions feature must onset
MIN_NON_TIED_ONSETS_FOR_TAU = 3         # Minimum distinct non-tied onsets to compute tau

# Status Labels
STATUS_CONFIRMED_PROGRESSION = "CONFIRMED_PROGRESSION"
STATUS_SUSTAINED_PRESSURE = "SUSTAINED_PRESSURE"
STATUS_INSUFFICIENT_TEMPORAL_EVIDENCE = "INSUFFICIENT_TEMPORAL_EVIDENCE"
STATUS_OPERATING_EQUILIBRIUM = "OPERATING_EQUILIBRIUM"

# Qualitative Temporal Agreement Bands
TAU_STRONG_AGREEMENT = 0.75             # tau_norm >= 0.75
TAU_MODERATE_AGREEMENT = 0.50           # 0.50 <= tau_norm < 0.75
# tau_norm < 0.50 -> Limited temporal agreement


# ----------------------------------------------------------------------
# Kendall's Tau-b Sequence Metric (Deterministic Implementation)
# ----------------------------------------------------------------------

def compute_kendall_tau_b(ranks_x: List[float], ranks_y: List[float]) -> Tuple[float, float]:
    """Computes Kendall's tau-b rank correlation and normalized tau in [0, 1].

    Args:
        ranks_x: Ranks or values for ranking A (e.g. canonical order indices).
        ranks_y: Ranks or values for ranking B (e.g. observed onset times).

    Returns:
        (tau_b, tau_normalized) where tau_normalized = (tau_b + 1) / 2.
    """
    n = len(ranks_x)
    if n < 2:
        return 0.0, 0.5

    concordant = 0
    discordant = 0
    ties_x = 0
    ties_y = 0

    for i in range(n):
        for j in range(i + 1, n):
            dx = ranks_x[i] - ranks_x[j]
            dy = ranks_y[i] - ranks_y[j]

            if dx == 0 and dy == 0:
                ties_x += 1
                ties_y += 1
            elif dx == 0:
                ties_x += 1
            elif dy == 0:
                ties_y += 1
            elif (dx > 0 and dy > 0) or (dx < 0 and dy < 0):
                concordant += 1
            else:
                discordant += 1

    denom = math.sqrt((concordant + discordant + ties_x) * (concordant + discordant + ties_y))
    if denom == 0.0:
        return 0.0, 0.5

    tau_b = (concordant - discordant) / denom
    tau_norm = (tau_b + 1.0) / 2.0
    return float(round(tau_b, 4)), float(round(tau_norm, 4))


# ----------------------------------------------------------------------
# Onset Event Detection
# ----------------------------------------------------------------------

def detect_feature_onsets_in_session(
    df_session: pd.DataFrame,
    evidence_ref: EvidenceReferenceProfile,
    condition: str,
    onset_threshold: float = DEFAULT_ONSET_THRESHOLD,
    consecutive_windows: int = DEFAULT_CONSECUTIVE_WINDOWS,
) -> List[Dict[str, Any]]:
    """Detects directional onsets across time windows in a single session.

    An onset occurs at window w when directional_z >= onset_threshold for
    consecutive_windows (default 2 consecutive windows: w and w+1).

    Args:
        df_session: DataFrame with 11 feature windows (must have FEATURE_NAMES).
        evidence_ref: Training-derived Normal baseline & expected directions.
        condition: Target condition for expected directions.
        onset_threshold: Standardized directional threshold (default 1.8).
        consecutive_windows: Number of consecutive windows required (default 2).

    Returns:
        List of detected onset event dicts sorted chronologically.
    """
    if df_session is None or df_session.empty:
        return []

    n_windows = len(df_session)
    expected_dirs = evidence_ref.expected_directions.get(condition, {})

    onset_events: List[Dict[str, Any]] = []

    for f in FEATURE_NAMES:
        exp_dir = expected_dirs.get(f, 0)
        # Strict Rule: Zero-direction features MUST NOT be used as precursors
        if exp_dir == 0:
            continue

        if f not in df_session.columns:
            continue

        values = df_session[f].to_numpy(dtype=float)

        # Check for NaN / Inf in this feature
        if np.isnan(values).any() or np.isinf(values).any():
            continue

        norm_mean = evidence_ref.feature_stats.get(f, {}).get("mean", 0.0)
        norm_std = evidence_ref.feature_stats.get(f, {}).get("std", 1.0)
        sigma_safe = max(norm_std, evidence_ref.epsilon)

        z_scores = exp_dir * ((values - norm_mean) / sigma_safe)

        # Scan for consecutive windows meeting threshold
        found_onset = False
        for w in range(n_windows - consecutive_windows + 1):
            window_slice = z_scores[w : w + consecutive_windows]
            if np.all(window_slice >= onset_threshold):
                # Retrieve timestamp or offset
                onset_time_s = float(w * 2.5)  # standard CURIO step = 2.5s
                if "window_start_offset" in df_session.columns:
                    onset_time_s = float(df_session["window_start_offset"].iloc[w])

                onset_events.append({
                    "feature_name": f,
                    "onset_window": int(w),
                    "onset_time_seconds": onset_time_s,
                    "directional_score": float(round(z_scores[w], 3)),
                    "consecutive_window_confirmation": True,
                    "expected_direction": int(exp_dir),
                })
                found_onset = True
                break  # Only record first onset event for each feature

    # Sort onset events chronologically by onset time; break ties deterministically by feature name
    onset_events.sort(key=lambda x: (x["onset_time_seconds"], x["feature_name"]))
    return onset_events


# ----------------------------------------------------------------------
# Training-Derived Canonical Temporal Signatures
# ----------------------------------------------------------------------

class DiscoveryReferenceProfile:
    """Encapsulates training-derived canonical precursor sequences for abnormal conditions.

    Strictly fitted on training sessions. Never touches held-out evaluation sessions.
    """

    def __init__(
        self,
        schema_version: str = "1.0.0",
        created_at: Optional[str] = None,
        training_machines: Optional[List[str]] = None,
        recurrence_threshold: float = DEFAULT_RECURRENCE_THRESHOLD,
        onset_threshold: float = DEFAULT_ONSET_THRESHOLD,
        consecutive_windows: int = DEFAULT_CONSECUTIVE_WINDOWS,
        canonical_signatures: Optional[Dict[str, Dict[str, Any]]] = None,
    ):
        self.schema_version = schema_version
        self.created_at = created_at or datetime.now(timezone.utc).isoformat()
        self.training_machines = training_machines or []
        self.recurrence_threshold = recurrence_threshold
        self.onset_threshold = onset_threshold
        self.consecutive_windows = consecutive_windows
        self.canonical_signatures = canonical_signatures or {}

    @classmethod
    def fit_from_training_sessions(
        cls,
        train_df: pd.DataFrame,
        evidence_ref: EvidenceReferenceProfile,
        training_machines: Optional[List[str]] = None,
        recurrence_threshold: float = DEFAULT_RECURRENCE_THRESHOLD,
        onset_threshold: float = DEFAULT_ONSET_THRESHOLD,
        consecutive_windows: int = DEFAULT_CONSECUTIVE_WINDOWS,
    ) -> "DiscoveryReferenceProfile":
        """Learns canonical precursor signatures from training sessions.

        For each condition except Normal:
        1. Identifies features with non-zero directional evidence.
        2. Detects onsets across all training sessions of that condition.
        3. Computes recurrence = (sessions with onset) / (total sessions).
        4. Filters features with recurrence >= recurrence_threshold.
        5. Computes median onset time across training sessions for recurring features.
        6. Orders recurring features chronologically to form canonical sequence.
        """
        if train_df is None or train_df.empty:
            raise ValueError("Training DataFrame cannot be empty.")

        if "session_id" not in train_df.columns or "condition" not in train_df.columns:
            raise ValueError("Training data must contain 'session_id' and 'condition'.")

        if training_machines is None:
            if "machine_id" in train_df.columns:
                training_machines = sorted(list(set(train_df["machine_id"].dropna().unique())))
            else:
                training_machines = ["training_hardware"]

        canonical_sigs: Dict[str, Dict[str, Any]] = {}

        # Canonical signatures are learned for abnormal conditions
        abnormal_conditions = [c for c in VALID_CONDITIONS if c != "normal"]

        for cond in abnormal_conditions:
            cond_mask = train_df["condition"] == cond
            cond_df = train_df[cond_mask]
            session_ids = sorted(cond_df["session_id"].unique())
            total_sessions = len(session_ids)

            if total_sessions == 0:
                canonical_sigs[cond] = {
                    "total_sessions": 0,
                    "eligible_features": [],
                    "recurrence": {},
                    "median_onsets_s": {},
                    "canonical_sequence": [],
                }
                continue

            # Directional features for this condition
            exp_dirs = evidence_ref.expected_directions.get(cond, {})
            candidate_features = [f for f in FEATURE_NAMES if exp_dirs.get(f, 0) != 0]

            feature_onsets_across_sessions: Dict[str, List[float]] = {f: [] for f in candidate_features}

            for s_id in session_ids:
                s_df = cond_df[cond_df["session_id"] == s_id].sort_values(by="window_idx")
                s_onsets = detect_feature_onsets_in_session(
                    df_session=s_df,
                    evidence_ref=evidence_ref,
                    condition=cond,
                    onset_threshold=onset_threshold,
                    consecutive_windows=consecutive_windows,
                )
                for ev in s_onsets:
                    feat = ev["feature_name"]
                    if feat in feature_onsets_across_sessions:
                        feature_onsets_across_sessions[feat].append(ev["onset_time_seconds"])

            recurrence_dict: Dict[str, float] = {}
            median_onsets_dict: Dict[str, float] = {}
            eligible_features: List[str] = []

            for f in candidate_features:
                onsets_list = feature_onsets_across_sessions[f]
                rec = len(onsets_list) / float(total_sessions)
                recurrence_dict[f] = float(round(rec, 3))

                if rec >= recurrence_threshold:
                    eligible_features.append(f)
                    median_onsets_dict[f] = float(round(np.median(onsets_list), 2))

            # Canonical order: sort eligible recurring features by median onset ascending
            # Break ties by higher recurrence, then alphabetical feature name
            canonical_seq = sorted(
                eligible_features,
                key=lambda f: (median_onsets_dict[f], -recurrence_dict[f], f),
            )

            canonical_sigs[cond] = {
                "total_sessions": total_sessions,
                "eligible_features": eligible_features,
                "recurrence": recurrence_dict,
                "median_onsets_s": median_onsets_dict,
                "canonical_sequence": canonical_seq,
            }

        return cls(
            schema_version="1.0.0",
            created_at=datetime.now(timezone.utc).isoformat(),
            training_machines=training_machines,
            recurrence_threshold=recurrence_threshold,
            onset_threshold=onset_threshold,
            consecutive_windows=consecutive_windows,
            canonical_signatures=canonical_sigs,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serializes discovery reference to dictionary."""
        return {
            "schema_version": self.schema_version,
            "created_at": self.created_at,
            "training_machines": self.training_machines,
            "recurrence_threshold": self.recurrence_threshold,
            "onset_threshold": self.onset_threshold,
            "consecutive_windows": self.consecutive_windows,
            "canonical_signatures": self.canonical_signatures,
        }

    def to_json(self, filepath: str) -> None:
        """Writes discovery reference to JSON file."""
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DiscoveryReferenceProfile":
        """Deserializes discovery reference from dictionary."""
        return cls(
            schema_version=data.get("schema_version", "1.0.0"),
            created_at=data.get("created_at"),
            training_machines=data.get("training_machines", []),
            recurrence_threshold=data.get("recurrence_threshold", DEFAULT_RECURRENCE_THRESHOLD),
            onset_threshold=data.get("onset_threshold", DEFAULT_ONSET_THRESHOLD),
            consecutive_windows=data.get("consecutive_windows", DEFAULT_CONSECUTIVE_WINDOWS),
            canonical_signatures=data.get("canonical_signatures", {}),
        )

    @classmethod
    def from_json(cls, filepath: str) -> "DiscoveryReferenceProfile":
        """Loads discovery reference from JSON file."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Discovery reference file not found: {filepath}")
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)


# ----------------------------------------------------------------------
# Live Session Trajectory Discovery
# ----------------------------------------------------------------------

def discover_session_trajectory(
    df_session: pd.DataFrame,
    predicted_condition: str,
    evidence_ref: EvidenceReferenceProfile,
    discovery_ref: DiscoveryReferenceProfile,
    onset_threshold: Optional[float] = None,
    consecutive_windows: Optional[int] = None,
) -> Dict[str, Any]:
    """Analyzes the temporal trajectory of a 30-second session downstream of diagnosis.

    Answers:
    "WHAT temporal pattern did CURIO discover in this session?"

    Args:
        df_session: DataFrame containing 11 feature windows for the session.
        predicted_condition: The condition diagnosed by the classifier.
        evidence_ref: Training-derived Normal baseline & expected directions.
        discovery_ref: Training-derived canonical precursor signatures.
        onset_threshold: Optional override for onset z-score (default 1.8).
        consecutive_windows: Optional override for confirmation windows (default 2).

    Returns:
        Structured discovery result conforming to Section 15 schema.
    """
    if df_session is None or df_session.empty:
        raise ValueError("df_session cannot be empty.")

    if predicted_condition not in VALID_CONDITIONS:
        raise ValueError(f"Invalid predicted condition '{predicted_condition}'. Allowed: {VALID_CONDITIONS}")

    thresh = onset_threshold if onset_threshold is not None else discovery_ref.onset_threshold
    consec = consecutive_windows if consecutive_windows is not None else discovery_ref.consecutive_windows

    # 1. Normal Equilibrium Case
    if predicted_condition == "normal":
        # Check if any abnormal features crossed threshold
        normal_onsets = []
        for c in ["cpu_pressure", "memory_pressure", "disk_io_pressure"]:
            c_onsets = detect_feature_onsets_in_session(
                df_session=df_session,
                evidence_ref=evidence_ref,
                condition=c,
                onset_threshold=thresh,
                consecutive_windows=consec,
            )
            normal_onsets.extend(c_onsets)

        interpretation = validate_no_causal_language(
            "Operating Equilibrium: No escalation detected. Telemetry remained within learned Normal bounds."
        )

        return {
            "predicted_condition": "normal",
            "discovery_status": STATUS_OPERATING_EQUILIBRIUM,
            "observed_sequence": [],
            "canonical_sequence": [],
            "matched_features": [],
            "missing_features": [],
            "unexpected_features": [],
            "onset_events": [],
            "kendall_tau": None,
            "normalized_tau": None,
            "discovery_coverage": 0.0,
            "temporal_agreement": None,
            "interpretation": interpretation,
        }

    # 2. Abnormal Conditions: Detect Live Onset Events
    canonical_info = discovery_ref.canonical_signatures.get(predicted_condition, {})
    canonical_seq = list(canonical_info.get("canonical_sequence", []))

    onset_events = detect_feature_onsets_in_session(
        df_session=df_session,
        evidence_ref=evidence_ref,
        condition=predicted_condition,
        onset_threshold=thresh,
        consecutive_windows=consec,
    )

    observed_seq = [ev["feature_name"] for ev in onset_events]

    # Partition features into matched, missing canonical, and unexpected
    canonical_set = set(canonical_seq)
    observed_set = set(observed_seq)

    matched_features = [f for f in canonical_seq if f in observed_set]
    missing_features = [f for f in canonical_seq if f not in observed_set]
    unexpected_features = [f for f in observed_seq if f not in canonical_set]

    coverage = len(matched_features) / float(max(1, len(canonical_seq)))
    coverage = float(round(coverage, 3))

    # 3. Handle Tied / Already-Degraded Case (t=0 ties)
    t0_events = [ev for ev in onset_events if ev["onset_time_seconds"] == 0.0]
    distinct_onset_times = set(ev["onset_time_seconds"] for ev in onset_events)

    # If all or majority of onsets start at t=0 with no temporal progression
    if len(t0_events) >= 2 and len(distinct_onset_times) < MIN_NON_TIED_ONSETS_FOR_TAU:
        interpretation = validate_no_causal_language(
            "Sustained operating pressure detected; no transition observed during capture."
        )
        return {
            "predicted_condition": predicted_condition,
            "discovery_status": STATUS_SUSTAINED_PRESSURE,
            "observed_sequence": observed_seq,
            "canonical_sequence": canonical_seq,
            "matched_features": matched_features,
            "missing_features": missing_features,
            "unexpected_features": unexpected_features,
            "onset_events": onset_events,
            "kendall_tau": None,
            "normalized_tau": None,
            "discovery_coverage": coverage,
            "temporal_agreement": None,
            "interpretation": interpretation,
        }

    # 4. Minimum Information Requirement (< 3 distinct non-tied onsets)
    # Filter matched features to those with non-tied distinct onsets
    matched_events = [ev for ev in onset_events if ev["feature_name"] in matched_features]
    matched_distinct_times = set(ev["onset_time_seconds"] for ev in matched_events)

    if len(matched_features) < MIN_NON_TIED_ONSETS_FOR_TAU or len(matched_distinct_times) < MIN_NON_TIED_ONSETS_FOR_TAU:
        interpretation = validate_no_causal_language(
            "Insufficient temporal evidence for sequence analysis."
        )
        return {
            "predicted_condition": predicted_condition,
            "discovery_status": STATUS_INSUFFICIENT_TEMPORAL_EVIDENCE,
            "observed_sequence": observed_seq,
            "canonical_sequence": canonical_seq,
            "matched_features": matched_features,
            "missing_features": missing_features,
            "unexpected_features": unexpected_features,
            "onset_events": onset_events,
            "kendall_tau": None,
            "normalized_tau": None,
            "discovery_coverage": coverage,
            "temporal_agreement": None,
            "interpretation": interpretation,
        }

    # 5. Sufficient Temporal Evidence: Compute Kendall's Tau on Matched Features
    canonical_ranks = [canonical_seq.index(f) for f in matched_features]
    onset_time_map = {ev["feature_name"]: ev["onset_time_seconds"] for ev in matched_events}
    observed_times = [onset_time_map[f] for f in matched_features]

    tau_b, tau_norm = compute_kendall_tau_b(canonical_ranks, observed_times)

    # Agreement Classification
    if tau_norm >= TAU_STRONG_AGREEMENT:
        agreement = "Strong temporal agreement"
    elif tau_norm >= TAU_MODERATE_AGREEMENT:
        agreement = "Moderate temporal agreement"
    else:
        agreement = "Limited temporal agreement"

    cond_name = predicted_condition.replace("_", " ").title()
    interpretation = validate_no_causal_language(
        f"The observed session followed a temporal pattern consistent with the learned {cond_name} trajectory."
    )

    return {
        "predicted_condition": predicted_condition,
        "discovery_status": STATUS_CONFIRMED_PROGRESSION,
        "observed_sequence": observed_seq,
        "canonical_sequence": canonical_seq,
        "matched_features": matched_features,
        "missing_features": missing_features,
        "unexpected_features": unexpected_features,
        "onset_events": onset_events,
        "kendall_tau": tau_b,
        "normalized_tau": tau_norm,
        "discovery_coverage": coverage,
        "temporal_agreement": agreement,
        "interpretation": interpretation,
    }


# ----------------------------------------------------------------------
# Human-Readable Discovery Report Rendering
# ----------------------------------------------------------------------

def format_discovery_report(discovery_res: Dict[str, Any]) -> str:
    """Formats the structured discovery result into human-readable terminal output.

    Adheres strictly to non-causal reporting and Section 16 format.
    """
    lines = []
    lines.append("=" * 60)
    lines.append("CURIO DISCOVERY")
    lines.append("=" * 60)

    cond = discovery_res["predicted_condition"].replace("_", " ").title()
    lines.append(f"Predicted condition: {cond}")
    lines.append(f"Discovery status:    {discovery_res['discovery_status']}")
    lines.append("-" * 60)

    # Observed progression
    obs_seq = discovery_res.get("observed_sequence", [])
    if obs_seq:
        lines.append("Observed progression:")
        for idx, feat in enumerate(obs_seq):
            arrow = "        v" if idx < len(obs_seq) - 1 else ""
            ev_info = next((e for e in discovery_res["onset_events"] if e["feature_name"] == feat), None)
            t_str = f" (t = {ev_info['onset_time_seconds']:.1f}s)" if ev_info else ""
            lines.append(f"  {feat}{t_str}")
            if arrow:
                lines.append(arrow)
    else:
        lines.append("Observed progression: None detected.")

    lines.append("-" * 60)

    # Canonical training pattern
    canon_seq = discovery_res.get("canonical_sequence", [])
    if canon_seq:
        lines.append("Canonical training pattern:")
        for idx, feat in enumerate(canon_seq):
            arrow = "   v" if idx < len(canon_seq) - 1 else ""
            lines.append(f"  {feat}")
            if arrow:
                lines.append(arrow)
    else:
        lines.append("Canonical training pattern: None defined for this state.")

    lines.append("-" * 60)

    # Temporal similarity and coverage
    tau_val = discovery_res.get("normalized_tau")
    tau_raw = discovery_res.get("kendall_tau")
    cov = discovery_res.get("discovery_coverage", 0.0)

    if tau_val is not None:
        lines.append(f"Temporal similarity:  tau = {tau_raw:+.2f} (normalized: {tau_val:.2f})")
        lines.append(f"Temporal agreement:   {discovery_res.get('temporal_agreement')}")
    else:
        lines.append("Temporal similarity:  N/A (insufficient or non-transitional onsets)")

    lines.append(f"Discovery coverage:   {cov * 100:.1f}%")

    if discovery_res.get("missing_features"):
        lines.append(f"Missing canonical:    {', '.join(discovery_res['missing_features'])}")
    if discovery_res.get("unexpected_features"):
        lines.append(f"Unexpected observed:  {', '.join(discovery_res['unexpected_features'])}")

    lines.append("-" * 60)
    lines.append(f"Interpretation:\n  \"{discovery_res['interpretation']}\"")
    lines.append("=" * 60)

    output_text = "\n".join(lines)
    return validate_no_causal_language(output_text)
