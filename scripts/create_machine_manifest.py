"""Machine Manifest Generator for CURIO.

Creates a verifiable hardware and dataset manifest for a participating physical machine:
data/physical_metadata/<machine_id>_manifest.json

Includes:
- Schema version & creation timestamp
- Host hardware specs (CPU, cores, RAM, disk partitions, OS)
- Software environment versions (Python, psutil, scikit-learn, numpy, pandas, joblib)
- Protocol parameters (frozen 4 classes, 12 features, 61 samples, 11 windows)
- Dataset summary (session counts, condition breakdown, total feature windows, session IDs)
"""

import argparse
from datetime import datetime, timezone
import json
import os
import platform
import subprocess
import sys
from typing import Any, Dict, List
import numpy as np
import pandas as pd
import psutil

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.features import FEATURE_NAMES

RAW_DIR = "data/physical_raw"
PROCESSED_DIR = "data/physical_processed"
METADATA_DIR = "data/physical_metadata"


def get_cpu_model_name() -> str:
    """Retrieves human-readable CPU brand string."""
    system = platform.system()
    try:
        if system == "Windows":
            cmd = "wmic cpu get name /value"
            output = subprocess.check_output(cmd, shell=True, text=True, stderr=subprocess.DEVNULL)
            for line in output.splitlines():
                if line.strip().startswith("Name="):
                    return line.split("Name=", 1)[1].strip()
            return platform.processor() or "Unknown Windows CPU"
        elif system == "Linux":
            with open("/proc/cpuinfo", "r") as f:
                for line in f:
                    if "model name" in line:
                        return line.split(":", 1)[1].strip()
        elif system == "Darwin":
            cmd = ["sysctl", "-n", "machdep.cpu.brand_string"]
            return subprocess.check_output(cmd, text=True).strip()
    except Exception:
        pass
    return platform.processor() or "Generic CPU"


def get_curio_version() -> str:
    """Retrieves current git commit or version identifier."""
    try:
        git_hash = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        return f"git-{git_hash}"
    except Exception:
        return "v0.9.0-dev"


def create_manifest(
    machine_id: str = "physical_machine_A",
    raw_dir: str = RAW_DIR,
    processed_dir: str = PROCESSED_DIR,
    metadata_dir: str = METADATA_DIR,
) -> Dict[str, Any]:
    """Generates the manifest dictionary and writes it to disk."""
    import sklearn
    import joblib

    mem = psutil.virtual_memory()
    swap = psutil.swap_memory()
    partitions = []
    for p in psutil.disk_partitions(all=False):
        try:
            usage = psutil.disk_usage(p.mountpoint)
            partitions.append({
                "device": p.device,
                "mountpoint": p.mountpoint,
                "fstype": p.fstype,
                "total_gb": round(usage.total / (1024**3), 2),
            })
        except Exception:
            continue

    # Inspect sessions on disk
    raw_files = []
    if os.path.exists(raw_dir):
        raw_files = [
            f for f in os.listdir(raw_dir)
            if f.startswith(f"{machine_id}_") and f.endswith("_raw.csv")
        ]

    processed_files = []
    if os.path.exists(processed_dir):
        processed_files = [
            f for f in os.listdir(processed_dir)
            if f.startswith(f"{machine_id}_") and f.endswith("_features.csv")
        ]

    condition_breakdown: Dict[str, int] = {}
    total_windows = 0
    session_ids: List[str] = []

    for pf in sorted(processed_files):
        pf_path = os.path.join(processed_dir, pf)
        try:
            df = pd.read_csv(pf_path)
            sess_id = str(df["session_id"].iloc[0])
            cond = str(df["condition"].iloc[0])
            session_ids.append(sess_id)
            condition_breakdown[cond] = condition_breakdown.get(cond, 0) + 1
            total_windows += len(df)
        except Exception:
            pass

    manifest = {
        "schema_version": "1.0.0",
        "machine_id": machine_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "hardware": {
            "hostname": platform.node(),
            "operating_system": f"{platform.system()} {platform.release()} ({platform.version()})",
            "cpu_model": get_cpu_model_name(),
            "architecture": platform.machine(),
            "logical_cpu_count": psutil.cpu_count(logical=True),
            "physical_cpu_count": psutil.cpu_count(logical=False),
            "total_ram_gb": round(mem.total / (1024**3), 2),
            "total_swap_gb": round(swap.total / (1024**3), 2),
            "storage_partitions": partitions,
        },
        "environment": {
            "python_version": platform.python_version(),
            "psutil_version": psutil.__version__,
            "sklearn_version": sklearn.__version__,
            "numpy_version": np.__version__,
            "pandas_version": pd.__version__,
            "joblib_version": joblib.__version__,
            "curio_version": get_curio_version(),
        },
        "protocol": {
            "target_classes": ["normal", "cpu_pressure", "memory_pressure", "disk_io_pressure"],
            "features_count": len(FEATURE_NAMES),
            "features_list": FEATURE_NAMES,
            "sample_rate_hz": 2.0,
            "session_duration_s": 30.0,
            "window_duration_s": 5.0,
            "step_duration_s": 2.5,
            "samples_per_session": 61,
            "windows_per_session": 11,
        },
        "dataset_summary": {
            "raw_sessions_count": len(raw_files),
            "processed_sessions_count": len(processed_files),
            "total_feature_windows": total_windows,
            "condition_breakdown": condition_breakdown,
            "session_ids": sorted(list(set(session_ids))),
        },
    }

    os.makedirs(metadata_dir, exist_ok=True)
    manifest_path = os.path.join(metadata_dir, f"{machine_id}_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    return manifest


def main():
    parser = argparse.ArgumentParser(description="CURIO Machine Manifest Generator")
    parser.add_argument(
        "--machine_id",
        default="physical_machine_A",
        help="Target machine identifier (e.g. physical_machine_A, physical_machine_B, physical_machine_C)",
    )
    args = parser.parse_args()

    manifest = create_manifest(machine_id=args.machine_id)
    out_file = os.path.join(METADATA_DIR, f"{args.machine_id}_manifest.json")

    print("=" * 70)
    print(f"CURIO MACHINE MANIFEST GENERATED: {args.machine_id}")
    print("=" * 70)
    print(f"  Manifest File:        {out_file}")
    print(f"  Hostname:             {manifest['hardware']['hostname']}")
    print(f"  CPU Model:            {manifest['hardware']['cpu_model']}")
    print(f"  CPU Cores:            {manifest['hardware']['logical_cpu_count']} logical, {manifest['hardware']['physical_cpu_count']} physical")
    print(f"  RAM:                  {manifest['hardware']['total_ram_gb']} GB")
    print(f"  Raw Sessions:         {manifest['dataset_summary']['raw_sessions_count']}")
    print(f"  Processed Sessions:   {manifest['dataset_summary']['processed_sessions_count']}")
    print(f"  Total Windows:        {manifest['dataset_summary']['total_feature_windows']}")
    print(f"  Condition Breakdown:  {manifest['dataset_summary']['condition_breakdown']}")
    print("=" * 70)


if __name__ == "__main__":
    main()
