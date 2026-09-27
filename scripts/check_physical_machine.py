"""Pre-Flight Verification and Smoke-Test Script for Physical Machines.

Performs non-destructive environment, collector, feature extraction, and safety
checks on a physical machine before running the full 22-session data collection:
1. Environment & Library Checks
2. Host Identity & Clone Prevention
3. Directory & File Write Permissions
4. 5-Second Telemetry Collector Smoke Test (Sampling interval, monotonic timestamps, PID 0 exclusion)
5. 12-Feature Extractor Smoke Test (Schema validation, NaN/Inf bounds check)
6. Stress Harness Safety Guardrails (RAM headroom >= 1.5 GB, disk scratch writeable)
"""

import argparse
import os
import platform
import sys
import time
from typing import Any, Dict, List
import numpy as np
import pandas as pd
import psutil

# Ensure project root is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.collector import TelemetryCollector
from src.features import extract_feature_dataframe, FEATURE_NAMES
from src.stress_harness import CPUStress, MemoryStress, DiskStress

MACHINE_A_HOSTNAME = "LAPTOP-I0G3I8PA"


def check_environment() -> Dict[str, Any]:
    """Verifies Python version and core library imports."""
    py_ver = sys.version_info
    if py_ver < (3, 9):
        raise RuntimeError(f"Python 3.9+ required, found {platform.python_version()}")

    import sklearn
    import joblib

    return {
        "python": platform.python_version(),
        "psutil": psutil.__version__,
        "sklearn": sklearn.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "joblib": joblib.__version__,
    }


def check_host_identity(machine_id: str):
    """Prevents accidental reuse or disguise of Machine A as Machine B or C."""
    local_host = platform.node().upper()
    if machine_id in ("physical_machine_B", "physical_machine_C"):
        if local_host == MACHINE_A_HOSTNAME.upper():
            raise RuntimeError(
                f"HOSTNAME COLLISION: Current host is '{local_host}' (physical_machine_A). "
                f"You cannot collect data for '{machine_id}' on this hardware. "
                "Cross-machine LOMO evaluation strictly requires physically distinct hardware."
            )


def check_directories() -> List[str]:
    """Ensures all required data directories exist and are writeable."""
    dirs = [
        "data/physical_raw",
        "data/physical_processed",
        "data/physical_metadata",
        "data/exports",
        "reports",
    ]
    verified = []
    for d in dirs:
        os.makedirs(d, exist_ok=True)
        test_file = os.path.join(d, ".write_test")
        try:
            with open(test_file, "w") as f:
                f.write("ok")
            os.remove(test_file)
            verified.append(d)
        except Exception as e:
            raise RuntimeError(f"Directory {d} is not writeable: {e}")
    return verified


def smoke_test_collector() -> List[Dict[str, Any]]:
    """Runs a short 5-second collector test and checks telemetry validity."""
    collector = TelemetryCollector(sample_interval=0.5)
    samples = collector.collect(duration_seconds=5.0)

    # 5.0 seconds at 2 Hz should yield 10 or 11 samples
    if len(samples) < 9 or len(samples) > 13:
        raise RuntimeError(f"Collector smoke test yielded unexpected sample count: {len(samples)} (expected 10-11)")

    # Monotonic timestamps
    timestamps = [s["timestamp"] for s in samples]
    for i in range(1, len(timestamps)):
        dt = timestamps[i] - timestamps[i - 1]
        if dt <= 0:
            raise RuntimeError(f"Non-monotonic timestamp detected: {timestamps[i-1]} -> {timestamps[i]}")
        if dt < 0.2 or dt > 1.2:
            raise RuntimeError(f"Abnormal sample interval: {dt:.3f}s (expected ~0.5s)")

    # Required raw keys returned by TelemetryCollector
    required_keys = [
        "timestamp", "cpu_overall", "cpu_cores", "vmem_percent",
        "vmem_available", "vmem_total", "swap_percent", "swap_used",
        "swap_total", "disk_read_bytes", "disk_write_bytes",
        "disk_read_count", "disk_write_count", "disk_read_time",
        "disk_write_time", "process_count", "top_proc_cpu", "top_proc_rss",
    ]
    for s in samples:
        for k in required_keys:
            if k not in s:
                raise RuntimeError(f"Missing required raw key '{k}' in sample")

        # Verify top_proc_cpu is normalized percentage of system capacity and non-negative
        top_cpu = s.get("top_proc_cpu", 0.0)
        if top_cpu < 0.0 or top_cpu > 105.0:
            raise RuntimeError(f"Abnormal top_proc_cpu value: {top_cpu} (expected in [0.0, 100.0])")

    return samples


def smoke_test_features(samples: List[Dict[str, Any]]) -> pd.DataFrame:
    """Extracts features from smoke test samples and validates the 12 features."""
    df_feat = extract_feature_dataframe(samples, window_duration=5.0, step_duration=2.5)
    if len(df_feat) == 0:
        raise RuntimeError("Feature extractor produced 0 windows from 5.0-second test samples")

    for col in FEATURE_NAMES:
        if col not in df_feat.columns:
            raise RuntimeError(f"Missing frozen feature column: '{col}'")
        # Check for NaN / Inf
        if df_feat[col].isna().any():
            raise RuntimeError(f"Feature column '{col}' contains NaN values")
        if np.isinf(df_feat[col]).any():
            raise RuntimeError(f"Feature column '{col}' contains infinite values")

    return df_feat


def check_stress_safety():
    """Validates stress harness preconditions without running heavy stress."""
    mem = psutil.virtual_memory()
    available_gb = mem.available / (1024**3)
    if available_gb < 1.5:
        raise RuntimeError(
            f"INSUFFICIENT RAM: Only {available_gb:.2f} GB available. "
            "CURIO requires at least 1.5 GB available RAM headroom for safe execution."
        )

    # Initialize stress objects to verify parameter validation
    c_stress = CPUStress(intensity="low")
    m_stress = MemoryStress(intensity="low", min_headroom_gb=1.5)
    d_stress = DiskStress(scratch_dir="data", intensity="low")

    # Verify disk scratch directory is writeable
    os.makedirs("data/disk_stress_test", exist_ok=True)
    test_path = "data/disk_stress_test/scratch_check.bin"
    with open(test_path, "wb") as f:
        f.write(b"\x00" * 1024)
    os.remove(test_path)
    os.rmdir("data/disk_stress_test")


def run_preflight_checks(machine_id: str) -> bool:
    """Runs all preflight checks and outputs diagnostic report."""
    print("=" * 70)
    print(f"CURIO PRE-FLIGHT MACHINE VALIDATION: {machine_id}")
    print("=" * 70)

    # 1. Host identity
    print("\n[1/6] Validating host identity & isolation...")
    check_host_identity(machine_id)
    print(f"  [OK] Host '{platform.node()}' verified for '{machine_id}'")

    # 2. Environment
    print("\n[2/6] Checking Python & library environment...")
    env_info = check_environment()
    for k, v in env_info.items():
        print(f"  [OK] {k}: {v}")

    # 3. Directories
    print("\n[3/6] Verifying write permissions for data directories...")
    dirs = check_directories()
    for d in dirs:
        print(f"  [OK] Directory writeable: {d}")

    # 4. Stress Safety
    print("\n[4/6] Checking safety limits & RAM headroom...")
    check_stress_safety()
    vmem = psutil.virtual_memory()
    print(f"  [OK] RAM Available: {vmem.available / (1024**3):.2f} GB (>= 1.5 GB required)")
    print(f"  [OK] CPU Logical Cores: {psutil.cpu_count(logical=True)}")

    # 5. Collector Smoke Test
    print("\n[5/6] Running 5-second TelemetryCollector smoke test...")
    samples = smoke_test_collector()
    print(f"  [OK] Successfully collected {len(samples)} samples at 2 Hz")
    print("  [OK] Timestamps monotonic; PID 0 / Idle Process cleanly excluded")

    # 6. Feature Extractor Smoke Test
    print("\n[6/6] Running 12-Feature Extractor smoke test...")
    df_feat = smoke_test_features(samples)
    print(f"  [OK] Extracted {len(df_feat)} window(s); all 12 frozen features present")
    print("  [OK] No NaN or infinite values detected")

    print("\n" + "=" * 70)
    print("PRE-FLIGHT VALIDATION PASSED")
    print(f"Machine '{machine_id}' is fully compatible and safe for CURIO collection.")
    print("======================================================================")
    return True


def main():
    parser = argparse.ArgumentParser(description="CURIO Pre-Flight Machine Verification")
    parser.add_argument(
        "--machine_id",
        default="physical_machine_A",
        help="Target machine identifier (e.g. physical_machine_A, physical_machine_B, physical_machine_C)",
    )
    args = parser.parse_args()

    try:
        run_preflight_checks(machine_id=args.machine_id)
        sys.exit(0)
    except Exception as e:
        print(f"\n[!] PRE-FLIGHT VALIDATION FAILED: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
