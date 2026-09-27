"""CURIO Live Diagnosis Command-Line Tool.

Executes a live 30-second system telemetry capture (2 Hz, 61 samples),
extracts 12-feature rolling windows, performs calibrated Random Forest
diagnosis, checks session abnormality, generates evidence attribution,
and discovers temporal onset progression.

Usage:
    python scripts/diagnose_computer.py
    python scripts/diagnose_computer.py --technical
    python scripts/diagnose_computer.py --no_save
"""

import argparse
import os
import sys
import time

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Add repository root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.diagnosis_pipeline import (
    CurioDiagnosisPipeline,
    DEFAULT_MODEL_DIR,
    DEFAULT_HISTORY_DIR,
    EXPECTED_DURATION_SECONDS,
    EXPECTED_RAW_SAMPLES,
)
from src.reporting import render_diagnosis_report, render_performance_summary


def print_progress_bar(elapsed: float, current: int, total: int, bar_length: int = 32) -> None:
    """Renders a smooth terminal progress bar during 30-second capture."""
    fraction = min(1.0, max(0.0, current / max(1, total)))
    filled_len = int(round(bar_length * fraction))
    bar = "=" * filled_len + "-" * (bar_length - filled_len)
    percent = fraction * 100.0
    sys.stdout.write(f"\r  [{bar}] {percent:5.1f}% ({current}/{total} samples, {elapsed:4.1f}s)")
    sys.stdout.flush()


def run_live_cli():
    parser = argparse.ArgumentParser(description="CURIO Live Computer Diagnosis")
    parser.add_argument(
        "--technical",
        action="store_true",
        help="Display detailed technical diagnostics, per-window tables, and telemetry metrics",
    )
    parser.add_argument(
        "--no_save",
        action="store_true",
        help="Do not save diagnosis record to data/diagnosis_history/",
    )
    parser.add_argument(
        "--save_raw",
        action="store_true",
        help="Save raw telemetry CSV alongside diagnosis JSON for debugging",
    )
    parser.add_argument(
        "--model_dir",
        default=DEFAULT_MODEL_DIR,
        help="Path to deployment model directory",
    )
    parser.add_argument(
        "--history_dir",
        default=DEFAULT_HISTORY_DIR,
        help="Path to diagnosis history directory",
    )
    args = parser.parse_args()

    print("=" * 50)
    print("CURIO")
    print("Computer Intelligence & Discovery")
    print("=" * 50)
    print("")
    print("Initializing diagnosis pipeline...")

    try:
        pipeline = CurioDiagnosisPipeline(model_dir=args.model_dir, history_dir=args.history_dir)
    except Exception as e:
        print(f"\n[!] Pipeline Initialization Error: {e}")
        sys.exit(1)

    print("[+] Model & calibration profiles loaded")
    print("[+] Evidence reference profile loaded")
    print("[+] Discovery canonical signatures loaded")
    print("[+] Abnormality gating threshold loaded")
    print("")
    print("Telemetry subsystems initialized:")
    print("  [+] CPU telemetry (overall & per-core)")
    print("  [+] Memory telemetry (physical RAM & swap)")
    print("  [+] Disk telemetry (read/write bytes & IOPS)")
    print("  [+] Process telemetry (active processes & dominance)")
    print("")
    print(f"Collecting 30-second diagnostic session ({EXPECTED_RAW_SAMPLES} samples @ 2 Hz)...")
    print("Press Ctrl+C to cancel at any time.")
    print("")

    try:
        t_diag_start = time.perf_counter()
        result = pipeline.run_diagnosis(
            duration_seconds=EXPECTED_DURATION_SECONDS,
            progress_callback=print_progress_bar,
            save_history=not args.no_save,
            save_raw=args.save_raw,
        )
        print("\n")  # Newline after progress bar
        print("Analyzing...")
        print("  [+] Machine learning diagnosis complete")
        print("  [+] Abnormality assessment complete")
        print("  [+] Evidence attribution complete")
        print("  [+] Temporal discovery complete")
        print("")

        # Render report
        t_rep_start = time.perf_counter()
        report_text = render_diagnosis_report(result, technical=args.technical)
        reporting_ms = (time.perf_counter() - t_rep_start) * 1000.0
        result["performance"]["reporting_ms"] = round(reporting_ms, 2)

        print(report_text)

        if not args.no_save:
            sess_id = result["session_id"]
            save_path = os.path.join(args.history_dir, f"{sess_id}_diagnosis.json")
            print(f"\n[i] Diagnosis record saved to: {save_path}")

        print("\n" + render_performance_summary(result["performance"]))

    except KeyboardInterrupt:
        print("\n")
        print("=" * 50)
        print("DIAGNOSIS CANCELLED")
        print("SYSTEM STATE RESTORED")
        print("=" * 50)
        sys.exit(130)
    except Exception as e:
        print(f"\n\n[!] DIAGNOSIS ERROR: {e}")
        sys.exit(1)


if __name__ == "__main__":
    run_live_cli()
