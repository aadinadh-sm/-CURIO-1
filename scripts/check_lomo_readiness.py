"""Physical LOMO Readiness Checker for CURIO.

Evaluates whether the local repository satisfies the strict prerequisites
for running a true cross-machine Leave-One-Machine-Out (LOMO) physical evaluation:

Readiness Requirements:
1. At least 3 physically distinct machines cataloged (e.g. physical_machine_A, B, C)
2. Zero synthetic or virtual machine IDs in physical directories
3. Every machine has exactly 22 validated sessions (4 Normal, 6 CPU, 6 Memory, 6 Disk)
4. Every machine has exactly 242 feature windows
5. Zero NaN / Inf values across the 12 features
6. No session ID collisions across machines

Outputs:
- TRUE PHYSICAL LOMO READY (exit code 0) if all 3 machines are fully verified
- TRUE PHYSICAL LOMO NOT READY (exit code 1) if < 3 machines or incomplete data
"""

import argparse
import os
import sys
from typing import Any, Dict, List, Set
import numpy as np
import pandas as pd

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.features import FEATURE_NAMES
from src.physical_dataset_validator import (
    SYNTHETIC_MACHINE_IDS,
    VALID_CONDITIONS,
    EXPECTED_RAW_SAMPLES,
    EXPECTED_FEATURE_WINDOWS,
)

RAW_DIR = "data/physical_raw"
PROCESSED_DIR = "data/physical_processed"
METADATA_DIR = "data/physical_metadata"

EXPECTED_MACHINES = ["physical_machine_A", "physical_machine_B", "physical_machine_C"]
EXPECTED_SESSIONS_PER_MACHINE = 22
EXPECTED_WINDOWS_PER_MACHINE = 242


def evaluate_lomo_readiness(
    raw_dir: str = RAW_DIR,
    processed_dir: str = PROCESSED_DIR,
    metadata_dir: str = METADATA_DIR,
) -> Dict[str, Any]:
    """Inspects physical telemetry directories and determines LOMO readiness."""
    if not os.path.exists(processed_dir):
        return {
            "ready": False,
            "status": "NOT READY",
            "reason": f"Processed directory '{processed_dir}' does not exist.",
            "present_machines": [],
            "missing_machines": EXPECTED_MACHINES,
            "machine_details": {},
        }

    # Discover processed files
    proc_files = [f for f in os.listdir(processed_dir) if f.endswith("_features.csv")]
    machine_files: Dict[str, List[str]] = {}

    for f in proc_files:
        # Extract machine_id prefix (everything before condition)
        # Standard format: <machine_id>_<condition>_<stress>_<uuid>_features.csv
        parts = f.replace("_features.csv", "").split("_")
        # Match against known prefixes or find machine_id from dataframe
        fpath = os.path.join(processed_dir, f)
        try:
            df = pd.read_csv(fpath)
            m_id = str(df["machine_id"].iloc[0])
            machine_files.setdefault(m_id, []).append(fpath)
        except Exception:
            continue

    machine_details: Dict[str, Any] = {}
    verified_machines: List[str] = []
    all_session_ids: Set[str] = set()

    for m_id, fpaths in machine_files.items():
        # Check synthetic machine violation
        if m_id in SYNTHETIC_MACHINE_IDS:
            return {
                "ready": False,
                "status": "NOT READY",
                "reason": f"Synthetic machine ID '{m_id}' found in physical dataset.",
                "present_machines": list(machine_files.keys()),
                "missing_machines": [m for m in EXPECTED_MACHINES if m not in machine_files],
                "machine_details": {},
            }

        conditions_count: Dict[str, int] = {}
        total_windows = 0
        has_nans = False
        has_infs = False
        m_session_ids: Set[str] = set()

        for fp in fpaths:
            df = pd.read_csv(fp)
            c = str(df["condition"].iloc[0])
            s_id = str(df["session_id"].iloc[0])
            conditions_count[c] = conditions_count.get(c, 0) + 1
            total_windows += len(df)
            m_session_ids.add(s_id)

            # Feature check
            feats = df[FEATURE_NAMES]
            if feats.isna().any().any():
                has_nans = True
            if np.isinf(feats.to_numpy()).any():
                has_infs = True

        # Check collision with other machines
        colliding_sessions = m_session_ids.intersection(all_session_ids)
        all_session_ids.update(m_session_ids)

        is_valid = (
            len(fpaths) == EXPECTED_SESSIONS_PER_MACHINE
            and total_windows == EXPECTED_WINDOWS_PER_MACHINE
            and conditions_count.get("normal", 0) == 4
            and conditions_count.get("cpu_pressure", 0) == 6
            and conditions_count.get("memory_pressure", 0) == 6
            and conditions_count.get("disk_io_pressure", 0) == 6
            and not has_nans
            and not has_infs
            and not colliding_sessions
        )

        machine_details[m_id] = {
            "sessions_count": len(fpaths),
            "total_windows": total_windows,
            "conditions_count": conditions_count,
            "has_nans": has_nans,
            "has_infs": has_infs,
            "collisions": list(colliding_sessions),
            "is_valid": is_valid,
        }

        if is_valid:
            verified_machines.append(m_id)

    missing_expected = [m for m in EXPECTED_MACHINES if m not in verified_machines]
    is_ready = len(verified_machines) >= 3 and len(missing_expected) == 0

    status = "TRUE PHYSICAL LOMO READY" if is_ready else "TRUE PHYSICAL LOMO NOT READY"
    reason = None
    if not is_ready:
        if len(verified_machines) < 3:
            reason = f"Fewer than 3 physical machines available (Found: {len(verified_machines)} / 3)."
        else:
            reason = f"Missing expected machines: {missing_expected}"

    return {
        "ready": is_ready,
        "status": status,
        "reason": reason,
        "present_machines": verified_machines,
        "missing_machines": missing_expected,
        "machine_details": machine_details,
    }


def main():
    parser = argparse.ArgumentParser(description="CURIO Physical LOMO Readiness Checker")
    parser.add_argument("--processed_dir", default=PROCESSED_DIR, help="Path to processed features directory")
    args = parser.parse_args()

    print("=" * 70)
    print("CURIO PHYSICAL LOMO READINESS ASSESSMENT")
    print("=" * 70)

    res = evaluate_lomo_readiness(processed_dir=args.processed_dir)

    print(f"\nStatus: {res['status']}")
    if res["reason"]:
        print(f"Reason: {res['reason']}")

    print(f"\nVerified Machines Present ({len(res['present_machines'])}):")
    for m in res["present_machines"]:
        d = res["machine_details"][m]
        print(f"  [+] {m}: {d['sessions_count']} sessions, {d['total_windows']} windows, conditions: {d['conditions_count']}")

    if res["missing_machines"]:
        print(f"\nMissing Target Machines ({len(res['missing_machines'])}):")
        for m in res["missing_machines"]:
            print(f"  [-] {m}")

    print("\n" + "=" * 70)
    if res["ready"]:
        print("All 3 physical machines (physical_machine_A, physical_machine_B, physical_machine_C)")
        print("have complete, verified datasets ready for Leave-One-Machine-Out evaluation.")
        print("======================================================================")
        sys.exit(0)
    else:
        print("True cross-machine LOMO cannot be executed until datasets for all 3")
        print("physical machines are collected and imported.")
        print("DO NOT fabricate synthetic data or alias existing machines.")
        print("======================================================================")
        sys.exit(1)


if __name__ == "__main__":
    main()
