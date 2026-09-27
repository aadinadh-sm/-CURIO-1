"""Script to generate and analyze a small pilot dataset for CURIO.

Executes 11 pilot sessions covering Normal, CPU Pressure, Memory Pressure,
and Disk I/O Pressure at varied intensities and backgrounds, validating
feature distributions, absence of NaN/inf values, and metadata integrity.
"""

import os
import sys
import time
import numpy as np
import pandas as pd

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath("."))

from src.dataset_builder import DatasetBuilder
from src.features import FEATURE_NAMES


def run_pilot():
    builder = DatasetBuilder(
        raw_dir="data/raw",
        processed_dir="data/processed",
        machine_id="local_dev_laptop",
    )

    # 11 Pilot experimental runs
    experiment_matrix = [
        # Protocol A: Normal
        {"condition": "normal", "stress_level": "none", "background": "idle", "duration": 15.0},
        {"condition": "normal", "stress_level": "none", "background": "active_office", "duration": 15.0},

        # Protocol B: CPU Pressure
        {"condition": "cpu_pressure", "stress_level": "low", "background": "idle", "duration": 15.0},
        {"condition": "cpu_pressure", "stress_level": "med", "background": "active_office", "duration": 15.0},
        {"condition": "cpu_pressure", "stress_level": "high", "background": "idle", "duration": 15.0},

        # Protocol B: Memory Pressure
        {"condition": "memory_pressure", "stress_level": "low", "background": "idle", "duration": 15.0},
        {"condition": "memory_pressure", "stress_level": "med", "background": "active_office", "duration": 15.0},
        {"condition": "memory_pressure", "stress_level": "high", "background": "idle", "duration": 15.0},

        # Protocol B: Disk I/O Pressure
        {"condition": "disk_io_pressure", "stress_level": "low", "background": "idle", "duration": 15.0},
        {"condition": "disk_io_pressure", "stress_level": "med", "background": "active_office", "duration": 15.0},
        {"condition": "disk_io_pressure", "stress_level": "high", "background": "idle", "duration": 15.0},
    ]

    print(f"Starting CURIO Pilot Collection ({len(experiment_matrix)} sessions, ~15s each)...")
    t0 = time.time()
    results = []

    for idx, exp in enumerate(experiment_matrix, 1):
        print(f"\n[{idx}/{len(experiment_matrix)}] Running {exp['condition']} ({exp['stress_level']}) | bg: {exp['background']}...")
        sess_t0 = time.time()
        res = builder.record_session(
            condition=exp["condition"],
            stress_level=exp["stress_level"],
            background_workload=exp["background"],
            duration_seconds=exp["duration"],
            ramp_duration_seconds=3.0,
        )
        elapsed = time.time() - sess_t0
        print(f"  -> Finished in {elapsed:.1f}s | Samples: {res['sample_count']} | Windows: {res['window_count']}")
        results.append(res)
        time.sleep(1.0) # Settle buffer between sessions

    total_time = time.time() - t0
    print(f"\nPilot collection complete in {total_time:.1f}s ({total_time/60.0:.2f} mins).")

    # Load all processed features
    print("\nAggregating pilot features...")
    df_all = builder.load_all_processed_features()
    print(f"Total windowed feature vectors gathered: {len(df_all)}")

    # 1. NaN and Inf check
    nan_counts = df_all[FEATURE_NAMES].isna().sum().sum()
    inf_counts = np.isinf(df_all[FEATURE_NAMES].values).sum()
    print(f"Data Health Check: NaNs = {nan_counts}, Infs = {inf_counts}")
    assert nan_counts == 0, "Found NaNs in feature space!"
    assert inf_counts == 0, "Found Infs in feature space!"

    # 2. Metadata integrity check
    for col in ["machine_id", "session_id", "condition", "stress_level", "background_workload", "is_ramp_up"]:
        assert col in df_all.columns, f"Missing metadata column: {col}"
    print("[OK] Metadata integrity verified across all feature vectors.")

    # 3. Compute condition-wise feature distributions (steady-state only: is_ramp_up == 0)
    df_steady = df_all[df_all["is_ramp_up"] == 0]
    print(f"Steady-state windows for distribution analysis: {len(df_steady)}")

    summary = df_steady.groupby("condition")[FEATURE_NAMES].mean().round(3)
    print("\n=== PILOT FEATURE MEANS BY CONDITION (STEADY-STATE) ===")
    print(summary.to_string())

    # Save summary report
    summary.to_csv("data/pilot_feature_summary.csv")
    print("\nSummary saved to data/pilot_feature_summary.csv")

    return df_all, summary


if __name__ == "__main__":
    run_pilot()
