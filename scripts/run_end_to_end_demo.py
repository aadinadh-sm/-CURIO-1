"""End-to-End Replay Demonstration for CURIO Diagnosis Pipeline.

Demonstrates the complete end-to-end CURIO diagnosis pipeline using validated
physical telemetry from data/physical_raw/ without requiring a 30-second live capture.

Demonstrates:
    Pre-collected telemetry (61 samples)
                v
    12-Feature Extraction (11 windows)
                v
    Calibrated ML Diagnosis (Random Forest)
                v
    Session Abnormality Assessment
                v
    Evidence Engine Attribution
                v
    Discovery Engine Progression
                v
    Final Diagnostic Report (Standard & Technical)

Usage:
    python scripts/run_end_to_end_demo.py --all_conditions
    python scripts/run_end_to_end_demo.py --condition memory_pressure
    python scripts/run_end_to_end_demo.py --condition cpu_pressure --technical
"""

import argparse
import os
import sys
import time
from typing import Dict, List, Optional

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Add repository root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.collector import load_raw_telemetry
from src.diagnosis_pipeline import (
    CurioDiagnosisPipeline,
    DEFAULT_MODEL_DIR,
    DEFAULT_HISTORY_DIR,
)
from src.reporting import render_diagnosis_report, render_performance_summary

PHYSICAL_RAW_DIR = "data/physical_raw"


def run_replay_demo(
    condition: Optional[str] = None,
    session_id: Optional[str] = None,
    all_conditions: bool = False,
    technical: bool = False,
    model_dir: str = DEFAULT_MODEL_DIR,
    raw_dir: str = PHYSICAL_RAW_DIR,
):
    print("=" * 60)
    print("CURIO END-TO-END PIPELINE DEMONSTRATION")
    print("Mode: Replay mode (using pre-collected physical telemetry)")
    print("=" * 60)
    print("")

    pipeline = CurioDiagnosisPipeline(model_dir=model_dir, history_dir=DEFAULT_HISTORY_DIR)

    raw_files = [f for f in os.listdir(raw_dir) if f.endswith("_raw.csv")]
    if not raw_files:
        raise FileNotFoundError(f"No raw physical telemetry CSVs found in '{raw_dir}'.")

    target_conditions = (
        ["normal", "cpu_pressure", "memory_pressure", "disk_io_pressure"]
        if all_conditions
        else [condition] if condition else ["memory_pressure"]
    )

    for target_cond in target_conditions:
        matched_file = None
        for rf in sorted(raw_files):
            if session_id and session_id in rf:
                matched_file = rf
                break
            if f"_{target_cond}_" in rf:
                matched_file = rf
                break

        if not matched_file:
            print(f"[!] No raw physical telemetry file found matching condition: {target_cond}")
            continue

        raw_path = os.path.join(raw_dir, matched_file)
        raw_samples = load_raw_telemetry(raw_path)

        true_cond = raw_samples[0].get("condition", "unknown")
        sess_name = raw_samples[0].get("session_id", os.path.splitext(matched_file)[0])

        print("\n" + "#" * 60)
        print(f"REPLAY SESSION: {sess_name}")
        print(f"Source file:    {matched_file}")
        print(f"True condition: {true_cond.replace('_', ' ').title()}")
        print(f"Raw samples:    {len(raw_samples)} samples (30.0s @ 2 Hz)")
        print("#" * 60)
        print("")

        # Run diagnosis pipeline on pre-collected physical telemetry
        result = pipeline.run_diagnosis(
            raw_samples=raw_samples,
            save_history=False,
            session_id=sess_name,
        )

        # Render report
        t_rep_start = time.perf_counter()
        report_text = render_diagnosis_report(result, technical=technical)
        reporting_ms = (time.perf_counter() - t_rep_start) * 1000.0
        result["performance"]["reporting_ms"] = round(reporting_ms, 2)

        print(report_text)
        print("\n" + render_performance_summary(result["performance"]))


def main():
    parser = argparse.ArgumentParser(description="CURIO End-to-End Replay Demonstration")
    parser.add_argument(
        "--condition",
        choices=["normal", "cpu_pressure", "memory_pressure", "disk_io_pressure"],
        help="Specific condition to replay",
    )
    parser.add_argument("--session_id", help="Specific session ID substring to replay")
    parser.add_argument(
        "--all_conditions",
        action="store_true",
        help="Replay one physical session for each of the 4 conditions",
    )
    parser.add_argument(
        "--technical",
        action="store_true",
        help="Include full technical telemetry tables and discovery metrics",
    )
    parser.add_argument("--model_dir", default=DEFAULT_MODEL_DIR, help="Path to model directory")
    parser.add_argument("--raw_dir", default=PHYSICAL_RAW_DIR, help="Path to raw telemetry directory")
    args = parser.parse_args()

    run_replay_demo(
        condition=args.condition,
        session_id=args.session_id,
        all_conditions=args.all_conditions,
        technical=args.technical,
        model_dir=args.model_dir,
        raw_dir=args.raw_dir,
    )


if __name__ == "__main__":
    main()
