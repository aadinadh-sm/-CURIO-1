"""30-Second Trajectory Pilot Collection & Validation Script for CURIO.

Captures full 30-second sessions (61 samples at 2 Hz, 11 rolling windows)
across Normal, CPU Low/High, Memory Low/High, and Disk Low/High.
Analyzes temporal trajectories, checks onset distributions, and enforces
strict Discovery Engine fallback rules.
"""

import os
import sys
import time
from typing import Any, Dict, List
import numpy as np
import pandas as pd

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath("."))

from src.dataset_builder import DatasetBuilder
from src.features import FEATURE_NAMES


def run_trajectory_pilot():
    builder = DatasetBuilder(
        raw_dir="data/raw_trajectory",
        processed_dir="data/processed_trajectory",
        machine_id="local_dev_laptop",
    )

    # 7 Trajectory validation runs (30 seconds each)
    trajectory_matrix = [
        {"condition": "normal", "stress_level": "none", "background": "idle", "duration": 30.0},
        {"condition": "cpu_pressure", "stress_level": "low", "background": "idle", "duration": 30.0},
        {"condition": "cpu_pressure", "stress_level": "high", "background": "active_office", "duration": 30.0},
        {"condition": "memory_pressure", "stress_level": "low", "background": "idle", "duration": 30.0},
        {"condition": "memory_pressure", "stress_level": "high", "background": "active_office", "duration": 30.0},
        {"condition": "disk_io_pressure", "stress_level": "low", "background": "idle", "duration": 30.0},
        {"condition": "disk_io_pressure", "stress_level": "high", "background": "active_office", "duration": 30.0},
    ]

    print(f"=== Starting CURIO 30-Second Trajectory Pilot ({len(trajectory_matrix)} sessions) ===")
    t0 = time.time()
    session_results = []

    for idx, exp in enumerate(trajectory_matrix, 1):
        print(f"\n[{idx}/{len(trajectory_matrix)}] Capturing 30s session: {exp['condition']} ({exp['stress_level']}) | bg: {exp['background']}...")
        s_t0 = time.time()
        res = builder.record_session(
            condition=exp["condition"],
            stress_level=exp["stress_level"],
            background_workload=exp["background"],
            duration_seconds=exp["duration"],
            ramp_duration_seconds=5.0,
        )
        elapsed = time.time() - s_t0
        print(f"  -> Finished in {elapsed:.1f}s | Samples: {res['sample_count']} (expected 61) | Windows: {res['window_count']} (expected 11)")
        session_results.append(res)
        time.sleep(1.0) # Settle buffer

    total_time = time.time() - t0
    print(f"\nAll 7 trajectory sessions captured in {total_time:.1f}s ({total_time/60.0:.2f} mins).")

    # Temporal Trajectory Analysis
    print("\n=== ANALYZING 11-WINDOW TEMPORAL TRAJECTORIES ===")
    analysis_records = []

    # Get normal baseline reference from the Normal session
    normal_res = next(r for r in session_results if r["condition"] == "normal")
    df_normal = normal_res["features_df"]
    normal_means = df_normal[FEATURE_NAMES].mean()
    normal_stds = df_normal[FEATURE_NAMES].std().replace(0.0, 1.0)

    for res in session_results:
        cond = res["condition"]
        lvl = res["stress_level"]
        df_feat = res["features_df"]
        sample_count = res["sample_count"]
        window_count = res["window_count"]

        # Health checks
        nans = df_feat[FEATURE_NAMES].isna().sum().sum()
        infs = np.isinf(df_feat[FEATURE_NAMES].values).sum()

        print(f"\n--- Condition: {cond.upper()} (Stress: {lvl}) ---")
        print(f"Samples: {sample_count}, Windows: {window_count}, NaNs: {nans}, Infs: {infs}")

        # Detect onsets across the 11 windows:
        # Standardized deviation z(t) = (x(t) - mu_normal) / sigma_normal
        onset_times = {}
        for f in FEATURE_NAMES:
            z_series = (df_feat[f] - normal_means[f]) / normal_stds[f]
            # Detect first window where |z| >= 1.8
            onset_idx = None
            for w_idx, z_val in enumerate(z_series):
                if abs(z_val) >= 1.8:
                    onset_idx = w_idx
                    break
            onset_times[f] = onset_idx

        # Filter features that showed onset
        active_onsets = {f: idx for f, idx in onset_times.items() if idx is not None}
        distinct_onsets = set(active_onsets.values())

        print(f"Features crossing onset (|z| >= 1.8): {len(active_onsets)} / 12")
        for f, idx in sorted(active_onsets.items(), key=lambda x: x[1]):
            val_at_onset = df_feat[f].iloc[idx]
            normal_val = normal_means[f]
            direction = "INCREASE" if val_at_onset > normal_val else "DECREASE"
            print(f"  • {f} -> Onset Window {idx} (T+{df_feat['window_start_offset'].iloc[idx]:.1f}s), {direction} (val: {val_at_onset:.2f} vs norm {normal_val:.2f})")

        # Evaluate Discovery Engine Fallback Rules
        if cond == "normal":
            trajectory_verdict = "Operating Equilibrium: No escalation detected. Telemetry remained within normal baseline boundaries."
        elif len(active_onsets) > 0 and all(idx == 0 for idx in active_onsets.values()):
            trajectory_verdict = "Sustained operating pressure detected; no transition observed during capture."
        elif len(distinct_onsets) < 3:
            trajectory_verdict = "Insufficient temporal evidence for sequence analysis (fewer than 3 distinct non-tied onset events)."
        else:
            trajectory_verdict = f"Dynamic escalation observed across {len(distinct_onsets)} distinct onset stages."

        print(f"Discovery Engine Finding: \"{trajectory_verdict}\"")

        analysis_records.append({
            "session_id": res["session_id"],
            "condition": cond,
            "stress_level": lvl,
            "sample_count": sample_count,
            "window_count": window_count,
            "nans": nans,
            "infs": infs,
            "active_onsets_count": len(active_onsets),
            "distinct_onsets_count": len(distinct_onsets),
            "discovery_verdict": trajectory_verdict,
        })

    # Save summary dataframe
    df_summary = pd.DataFrame(analysis_records)
    df_summary.to_csv("data/trajectory_pilot_summary.csv", index=False)
    print("\nTrajectory validation summary written to data/trajectory_pilot_summary.csv")


if __name__ == "__main__":
    run_trajectory_pilot()
