"""Discovery Engine Interactive Demonstration for CURIO.

Loads verified physical telemetry from data/physical_processed/ and the calibrated
deployment model from data/models/physical_deployment/, performs diagnosis,
and runs the Discovery Engine to discover and report temporal trajectory patterns:
1. What condition was diagnosed?
2. What temporal onset progression was observed across the 11 feature windows?
3. How does this compare against the training-derived canonical signature?
4. Sequence similarity via Kendall's tau-b (or honest fallbacks for sustained pressure / equilibrium).
"""

import argparse
import os
import sys
from typing import Dict, Any, Optional
import joblib
import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.discovery import (
    DiscoveryReferenceProfile,
    discover_session_trajectory,
    format_discovery_report,
)
from src.evidence import EvidenceReferenceProfile, generate_evidence
from src.features import FEATURE_NAMES

PROCESSED_DIR = "data/physical_processed"
DEFAULT_MODEL_DIR = "data/models/physical_deployment"


def run_discovery_on_session(
    df_session: pd.DataFrame,
    model: Any,
    evidence_ref: EvidenceReferenceProfile,
    discovery_ref: DiscoveryReferenceProfile,
) -> Dict[str, Any]:
    """Runs diagnosis and subsequent temporal trajectory discovery on a full session."""
    X_session = df_session[FEATURE_NAMES].to_numpy(dtype=float)
    probs_all_windows = model.predict_proba(X_session)
    # Session-level diagnosis = argmax of mean window probabilities
    mean_probs = np.mean(probs_all_windows, axis=0)
    classes = list(model.classes_)
    pred_idx = np.argmax(mean_probs)
    pred_condition = classes[pred_idx]

    # Run Discovery Engine downstream of diagnosis
    discovery_res = discover_session_trajectory(
        df_session=df_session,
        predicted_condition=pred_condition,
        evidence_ref=evidence_ref,
        discovery_ref=discovery_ref,
    )

    return discovery_res, pred_condition, mean_probs[pred_idx]


def run_discovery_demo(
    condition: Optional[str] = None,
    session_id: Optional[str] = None,
    model_dir: str = DEFAULT_MODEL_DIR,
    processed_dir: str = PROCESSED_DIR,
    all_conditions: bool = False,
):
    # 1. Load Model, Evidence Reference, and Discovery Reference
    model_path = os.path.join(model_dir, "model.joblib")
    ev_ref_path = os.path.join(model_dir, "evidence_reference.json")
    disc_ref_path = os.path.join(model_dir, "discovery_reference.json")

    for p in (model_path, ev_ref_path, disc_ref_path):
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"Missing deployment artifact: '{p}'. Run 'scripts/build_deployment_model.py' first."
            )

    model = joblib.load(model_path)
    evidence_ref = EvidenceReferenceProfile.from_json(ev_ref_path)
    discovery_ref = DiscoveryReferenceProfile.from_json(disc_ref_path)

    # 2. Select Sessions to Discover
    proc_files = [f for f in os.listdir(processed_dir) if f.endswith("_features.csv")]
    if not proc_files:
        raise FileNotFoundError(f"No processed feature CSVs found in '{processed_dir}'.")

    target_conditions = (
        ["normal", "cpu_pressure", "memory_pressure", "disk_io_pressure"]
        if all_conditions
        else [condition] if condition else ["memory_pressure"]
    )

    for target_cond in target_conditions:
        matched_file = None
        for pf in proc_files:
            if session_id and session_id in pf:
                matched_file = pf
                break
            if f"_{target_cond}_" in pf:
                matched_file = pf
                break

        if not matched_file:
            print(f"[!] No session found matching condition '{target_cond}'.")
            continue

        df_session = pd.read_csv(os.path.join(processed_dir, matched_file)).sort_values(by="window_idx")
        sess_name = str(df_session["session_id"].iloc[0])
        true_cond = str(df_session["condition"].iloc[0])

        discovery_res, pred_cond, confidence = run_discovery_on_session(
            df_session=df_session,
            model=model,
            evidence_ref=evidence_ref,
            discovery_ref=discovery_ref,
        )

        print("\n" + "=" * 60)
        print(f"SESSION: {sess_name}")
        print(f"Windows: {len(df_session)} | True Condition: {true_cond.replace('_', ' ').title()}")
        print(f"Diagnosis: {pred_cond.replace('_', ' ').title()} ({confidence * 100:.1f}% confidence)")
        print("=" * 60)

        # Print formatted discovery report
        report_str = format_discovery_report(discovery_res)
        print(report_str)


def main():
    parser = argparse.ArgumentParser(description="CURIO Discovery Engine Demonstration")
    parser.add_argument(
        "--condition",
        choices=["normal", "cpu_pressure", "memory_pressure", "disk_io_pressure"],
        help="Condition class to inspect",
    )
    parser.add_argument("--session_id", help="Specific session ID substring to explain")
    parser.add_argument("--all_conditions", action="store_true", help="Demonstrate all 4 conditions")
    parser.add_argument("--model_dir", default=DEFAULT_MODEL_DIR, help="Path to model directory")
    parser.add_argument("--processed_dir", default=PROCESSED_DIR, help="Path to processed features")
    args = parser.parse_args()

    run_discovery_demo(
        condition=args.condition,
        session_id=args.session_id,
        model_dir=args.model_dir,
        processed_dir=args.processed_dir,
        all_conditions=args.all_conditions,
    )


if __name__ == "__main__":
    main()
