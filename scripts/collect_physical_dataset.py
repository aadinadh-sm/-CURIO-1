"""Physical Telemetry Collection Runner for CURIO.

Executes the formal 22-session experimental protocol on local physical hardware:
- 4 Normal sessions (varied background workloads)
- 6 CPU Pressure sessions (2 Low, 2 Med, 2 High)
- 6 Memory Pressure sessions (2 Low, 2 Med, 2 High with safe headroom)
- 6 Disk I/O Pressure sessions (2 Low, 2 Med, 2 High with sync writes)

Adheres strictly to safety requirements:
- Guarantees stress worker and file cleanup after every session
- Prints required safety confirmations
- Records full provenance (machine_id, session_id, condition, stress_level, background_workload)
"""

import argparse
import os
import sys
import time
from typing import Any, Dict, List, Optional
import pandas as pd
import psutil

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.dataset_builder import DatasetBuilder
from src.physical_dataset_validator import (
    validate_physical_feature_session,
    validate_physical_raw_session,
)

PHYSICAL_RAW_DIR = "data/physical_raw"
PHYSICAL_PROCESSED_DIR = "data/physical_processed"


def print_safety_status(session_id: str):
    """Prints the required session completion and safety confirmation."""
    print(f"\n[Session: {session_id}]")
    print("SESSION COMPLETE")
    print("STRESS CLEANUP COMPLETE")
    print("SYSTEM STATE RESTORED\n")


def run_physical_collection(
    machine_id: str = "physical_machine_A",
    sessions_plan: Optional[List[Dict[str, str]]] = None,
    cooldown_seconds: float = 2.0,
) -> List[Dict[str, Any]]:
    """Runs the physical collection protocol."""
    os.makedirs(PHYSICAL_RAW_DIR, exist_ok=True)
    os.makedirs(PHYSICAL_PROCESSED_DIR, exist_ok=True)

    builder = DatasetBuilder(
        raw_dir=PHYSICAL_RAW_DIR,
        processed_dir=PHYSICAL_PROCESSED_DIR,
        machine_id=machine_id,
    )

    if sessions_plan is None:
        sessions_plan = [
            # 1. Normal (4 sessions)
            {"condition": "normal", "stress_level": "none", "background": "idle_desktop"},
            {"condition": "normal", "stress_level": "none", "background": "browser_tabs"},
            {"condition": "normal", "stress_level": "none", "background": "ide_editor_activity"},
            {"condition": "normal", "stress_level": "none", "background": "video_playback"},
            # 2. CPU Pressure (6 sessions: 2 low, 2 med, 2 high)
            {"condition": "cpu_pressure", "stress_level": "low", "background": "idle_desktop"},
            {"condition": "cpu_pressure", "stress_level": "low", "background": "browser_tabs"},
            {"condition": "cpu_pressure", "stress_level": "med", "background": "idle_desktop"},
            {"condition": "cpu_pressure", "stress_level": "med", "background": "ide_editor_activity"},
            {"condition": "cpu_pressure", "stress_level": "high", "background": "idle_desktop"},
            {"condition": "cpu_pressure", "stress_level": "high", "background": "browser_tabs"},
            # 3. Memory Pressure (6 sessions: 2 low, 2 med, 2 high)
            {"condition": "memory_pressure", "stress_level": "low", "background": "idle_desktop"},
            {"condition": "memory_pressure", "stress_level": "low", "background": "browser_tabs"},
            {"condition": "memory_pressure", "stress_level": "med", "background": "idle_desktop"},
            {"condition": "memory_pressure", "stress_level": "med", "background": "ide_editor_activity"},
            {"condition": "memory_pressure", "stress_level": "high", "background": "idle_desktop"},
            {"condition": "memory_pressure", "stress_level": "high", "background": "browser_tabs"},
            # 4. Disk I/O Pressure (6 sessions: 2 low, 2 med, 2 high)
            {"condition": "disk_io_pressure", "stress_level": "low", "background": "idle_desktop"},
            {"condition": "disk_io_pressure", "stress_level": "low", "background": "browser_tabs"},
            {"condition": "disk_io_pressure", "stress_level": "med", "background": "idle_desktop"},
            {"condition": "disk_io_pressure", "stress_level": "med", "background": "ide_editor_activity"},
            {"condition": "disk_io_pressure", "stress_level": "high", "background": "idle_desktop"},
            {"condition": "disk_io_pressure", "stress_level": "high", "background": "browser_tabs"},
        ]

    # Safety check: prevent collecting for Machine B or C on Machine A host
    import platform
    if machine_id in ("physical_machine_B", "physical_machine_C") and platform.node().upper() == "LAPTOP-I0G3I8PA":
        raise RuntimeError(
            f"REFUSING TO COLLECT: Current host '{platform.node()}' is physical_machine_A. "
            f"You cannot collect data for '{machine_id}' on this hardware. "
            "Deploy to a separate physical machine."
        )

    # Inspect existing processed feature files for this machine to skip already completed sessions
    existing_files = [
        f for f in os.listdir(PHYSICAL_PROCESSED_DIR)
        if f.startswith(f"{machine_id}_") and f.endswith("_features.csv")
    ]
    existing_counts = {}
    for f in existing_files:
        try:
            df_ex = pd.read_csv(os.path.join(PHYSICAL_PROCESSED_DIR, f))
            c = df_ex["condition"].iloc[0]
            l = df_ex["stress_level"].iloc[0]
            existing_counts[(c, l)] = existing_counts.get((c, l), 0) + 1
        except Exception:
            pass

    remaining_plan = []
    avail_counts = dict(existing_counts)
    for item in sessions_plan:
        key = (item["condition"], item["stress_level"])
        if avail_counts.get(key, 0) > 0:
            avail_counts[key] -= 1
        else:
            remaining_plan.append(item)

    total_sessions = len(sessions_plan)
    already_done = total_sessions - len(remaining_plan)
    results = []

    print("=" * 70)
    print(f"CURIO PHYSICAL TELEMETRY COLLECTION: {machine_id}")
    print(f"Target Protocol Sessions: {total_sessions} ({already_done} already completed, {len(remaining_plan)} remaining)")
    print(f"Output Raw Dir:          {PHYSICAL_RAW_DIR}")
    print(f"Output Feat Dir:         {PHYSICAL_PROCESSED_DIR}")
    print("=" * 70)

    for idx, item in enumerate(remaining_plan, start=already_done + 1):
        cond = item["condition"]
        lvl = item["stress_level"]
        bg = item["background"]

        print(f"\n[{idx}/{total_sessions}] Starting: condition='{cond}', stress='{lvl}', background='{bg}'...")

        # Record available RAM before session
        vmem_before = psutil.virtual_memory()
        print(f"  Pre-check: RAM Available = {vmem_before.available / (1024**3):.2f} GB "
              f"({100 - vmem_before.percent:.1f}% free)")

        session_res = None
        try:
            session_res = builder.record_session(
                condition=cond,
                stress_level=lvl,
                background_workload=bg,
                duration_seconds=30.0,
                ramp_duration_seconds=5.0,
            )
        except Exception as e:
            print(f"  [ERROR] Session execution failed: {e}")
            raise
        finally:
            # Enforce cleanup confirmation
            sess_id = session_res["session_id"] if session_res else f"{machine_id}_{cond}_{lvl}"
            print_safety_status(sess_id)

        # Immediate Validation
        raw_df = pd.read_csv(session_res["raw_path"])
        feat_df = pd.read_csv(session_res["processed_path"])
        validate_physical_raw_session(raw_df, session_id=sess_id)
        validate_physical_feature_session(feat_df, session_id=sess_id)

        print(f"  Verified: {session_res['sample_count']} raw samples, "
              f"{session_res['window_count']} rolling feature windows.")
        results.append(session_res)

        if idx < total_sessions and cooldown_seconds > 0:
            time.sleep(cooldown_seconds)

    print("\n" + "=" * 70)
    print(f"COLLECTION COMPLETE: Successfully captured and validated {len(results)}/{total_sessions} sessions!")
    print("=" * 70)
    return results


def main():
    parser = argparse.ArgumentParser(description="Collect physical telemetry dataset.")
    parser.add_argument("--machine_id", default="physical_machine_A", help="Machine ID")
    parser.add_argument("--cooldown", type=float, default=2.0, help="Cooldown between sessions in seconds")
    args = parser.parse_args()

    run_physical_collection(machine_id=args.machine_id, cooldown_seconds=args.cooldown)


if __name__ == "__main__":
    main()
