"""Evidence Engine Interactive Demonstration for CURIO.

Loads verified physical telemetry from data/physical_processed/ and the calibrated
deployment model from data/models/physical_deployment/, generates live diagnoses,
and outputs explainable evidence reports answering:
1. What condition did CURIO predict?
2. Which observed telemetry signals support that prediction?
3. How strongly did those signals deviate from the learned Normal operating reference?
"""

import argparse
import json
import os
import sys
import joblib
import numpy as np
import pandas as pd

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.evidence import EvidenceReferenceProfile, generate_evidence
from src.features import FEATURE_NAMES

PROCESSED_DIR = "data/physical_processed"
DEFAULT_MODEL_DIR = "data/models/physical_deployment"


def print_single_diagnosis_report(
    session_id: str,
    window_idx: int,
    true_condition: str,
    evidence_res: Dict[str, Any],
    threshold: float,
):
    """Formats and prints the human-readable CURIO diagnosis and evidence breakdown."""
    pred = evidence_res["condition"]
    conf = evidence_res["confidence"]
    abn = evidence_res["abnormality_score"]

    print("=" * 60)
    print("CURIO DIAGNOSIS & EVIDENCE REPORT")
    print("=" * 60)
    print(f"Session ID:         {session_id}")
    print(f"Window Index:       {window_idx}")
    print(f"True Condition:     {true_condition.replace('_', ' ').title()}")
    print("-" * 60)
    print(f"Condition:          {pred.replace('_', ' ').title()}")
    print(f"Model Confidence:   {conf * 100:.1f}%")
    print(f"Abnormality Score:  {abn:.4f}  (Validation Threshold: {threshold:.4f})")
    print(f"Abnormal Status:    {'ABNORMAL' if abn > threshold else 'WITHIN NORMAL OPERATING BOUNDS'}")
    print(f"Evidence Consist.:  {evidence_res['evidence_consistency'] * 100:.1f}%")
    print("-" * 60)

    print("WHY CURIO DIAGNOSED THIS:")
    if evidence_res["supporting_evidence"]:
        for idx, item in enumerate(evidence_res["supporting_evidence"], start=1):
            print(f"  {idx}. {item['human_readable_statement']}")
            print(f"     [Signal: {item['feature_name']}, Dev: {item['directional_score']:+.2f} std ({item['evidence_strength'].upper()})]")
    else:
        print("  No strong directional deviations detected.")

    if evidence_res.get("contradictory_evidence"):
        print("\nCONTRADICTORY EVIDENCE:")
        for idx, item in enumerate(evidence_res["contradictory_evidence"], start=1):
            print(f"  {idx}. {item['human_readable_statement']}")

    print("\nOVERALL INTERPRETATION:")
    print(f"  \"{evidence_res['overall_interpretation']}\"")
    print("=" * 60 + "\n")


def run_evidence_demo(
    condition: Optional[str] = None,
    session_id: Optional[str] = None,
    model_dir: str = DEFAULT_MODEL_DIR,
    processed_dir: str = PROCESSED_DIR,
    all_conditions: bool = False,
):
    # 1. Load Model & Artifacts
    model_path = os.path.join(model_dir, "model.joblib")
    thresh_path = os.path.join(model_dir, "abnormality_threshold.json")
    ref_path = os.path.join(model_dir, "evidence_reference.json")

    if not os.path.exists(model_path) or not os.path.exists(ref_path):
        raise FileNotFoundError(
            f"Deployment artifacts missing in '{model_dir}'. Run 'scripts/build_deployment_model.py' first."
        )

    model = joblib.load(model_path)
    ref_profile = EvidenceReferenceProfile.from_json(ref_path)

    threshold = 0.5
    if os.path.exists(thresh_path):
        with open(thresh_path, "r") as f:
            threshold = float(json.load(f).get("abnormality_threshold", 0.5))

    classes = list(model.classes_)

    # 2. Select Sessions to Explain
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

        df_session = pd.read_csv(os.path.join(processed_dir, matched_file))
        # Choose a stable middle window (e.g. index 5 of 11)
        win_idx = min(5, len(df_session) - 1)
        window_row = df_session.iloc[win_idx]

        feats_vec = window_row[FEATURE_NAMES].to_dict()
        X_vec = window_row[FEATURE_NAMES].to_numpy(dtype=float).reshape(1, -1)

        probs = model.predict_proba(X_vec)[0]
        pred_idx = np.argmax(probs)
        pred_condition = classes[pred_idx]

        evidence_res = generate_evidence(
            predicted_condition=pred_condition,
            calibrated_probabilities=probs,
            current_features=feats_vec,
            reference_profile=ref_profile,
            classes=classes,
        )

        print_single_diagnosis_report(
            session_id=str(window_row["session_id"]),
            window_idx=win_idx,
            true_condition=str(window_row["condition"]),
            evidence_res=evidence_res,
            threshold=threshold,
        )


def main():
    parser = argparse.ArgumentParser(description="CURIO Evidence Engine Demo")
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

    run_evidence_demo(
        condition=args.condition,
        session_id=args.session_id,
        model_dir=args.model_dir,
        processed_dir=args.processed_dir,
        all_conditions=args.all_conditions,
    )


if __name__ == "__main__":
    main()
