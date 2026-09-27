"""Feature Extraction Engine for CURIO.

Extracts 12 hardware-normalized telemetry features across rolling time windows
to characterize system operational states without machine-specific bias.
"""

import math
from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd

FEATURE_NAMES = [
    "cpu_mean",
    "cpu_max",
    "cpu_std",
    "cpu_core_imbalance",
    "ram_used_pct",
    "ram_available_ratio",
    "swap_used_pct",
    "disk_io_rate_norm",
    "disk_iops_norm",
    "process_count_delta",
    "top_proc_cpu_ratio",
    "top_proc_mem_pct",
]


def extract_features_from_window(window_samples: List[Dict[str, Any]]) -> Dict[str, float]:
    """Computes the 12 hardware-normalized features from a slice of telemetry samples."""
    if not window_samples:
        raise ValueError("Cannot extract features from an empty sample window.")

    n = len(window_samples)
    t_start = window_samples[0]["timestamp"]
    t_end = window_samples[-1]["timestamp"]
    dt = max(1e-3, t_end - t_start)

    # 1-3. CPU Stats
    cpu_vals = [s["cpu_overall"] for s in window_samples]
    cpu_mean = float(np.mean(cpu_vals))
    cpu_max = float(np.max(cpu_vals))
    cpu_std = float(np.std(cpu_vals, ddof=1)) if n > 1 else 0.0

    # 4. CPU Core Imbalance (Max core minus min core within each tick, averaged)
    imbalances = []
    for s in window_samples:
        cores = s.get("cpu_cores", [])
        if cores:
            imbalances.append(float(np.max(cores) - np.min(cores)))
        else:
            imbalances.append(0.0)
    cpu_core_imbalance = float(np.mean(imbalances))

    # 5-6. RAM Stats
    ram_pct_vals = [s["vmem_percent"] for s in window_samples]
    ram_used_pct = float(np.mean(ram_pct_vals))

    avail_ratios = [
        s["vmem_available"] / max(1.0, float(s["vmem_total"])) for s in window_samples
    ]
    ram_available_ratio = float(np.mean(avail_ratios))

    # 7. Swap Stats
    swap_pct_vals = [s["swap_percent"] for s in window_samples]
    swap_used_pct = float(np.mean(swap_pct_vals))

    # 8. Disk I/O Throughput Rate (log-compressed bytes/sec)
    r_bytes_delta = max(0, window_samples[-1]["disk_read_bytes"] - window_samples[0]["disk_read_bytes"])
    w_bytes_delta = max(0, window_samples[-1]["disk_write_bytes"] - window_samples[0]["disk_write_bytes"])
    total_bytes_delta = r_bytes_delta + w_bytes_delta
    bytes_per_sec = total_bytes_delta / dt
    disk_io_rate_norm = float(math.log10(bytes_per_sec + 1.0))

    # 9. Disk IOPS Activity (log-compressed ops/sec)
    r_count_delta = max(0, window_samples[-1].get("disk_read_count", 0) - window_samples[0].get("disk_read_count", 0))
    w_count_delta = max(0, window_samples[-1].get("disk_write_count", 0) - window_samples[0].get("disk_write_count", 0))
    total_ops_delta = r_count_delta + w_count_delta
    ops_per_sec = total_ops_delta / dt
    disk_iops_norm = float(math.log10(ops_per_sec + 1.0))

    # 10. Process Count Delta
    process_count_delta = float(window_samples[-1]["process_count"] - window_samples[0]["process_count"])

    # 11. Top Process CPU Dominance Ratio (relative to current system CPU)
    top_proc_cpu_mean = float(np.mean([s["top_proc_cpu"] for s in window_samples]))
    # Ratio of top process CPU against total active CPU, bounded [0.0, 1.0]
    top_proc_cpu_ratio = float(min(1.0, max(0.0, top_proc_cpu_mean / max(cpu_mean, 1.0))))

    # 12. Top Process Memory Percentage of Total Physical RAM
    top_mem_ratios = [
        (s["top_proc_rss"] / max(1.0, float(s["vmem_total"]))) * 100.0
        for s in window_samples
    ]
    top_proc_mem_pct = float(np.mean(top_mem_ratios))

    return {
        "cpu_mean": cpu_mean,
        "cpu_max": cpu_max,
        "cpu_std": cpu_std,
        "cpu_core_imbalance": cpu_core_imbalance,
        "ram_used_pct": ram_used_pct,
        "ram_available_ratio": ram_available_ratio,
        "swap_used_pct": swap_used_pct,
        "disk_io_rate_norm": disk_io_rate_norm,
        "disk_iops_norm": disk_iops_norm,
        "process_count_delta": process_count_delta,
        "top_proc_cpu_ratio": top_proc_cpu_ratio,
        "top_proc_mem_pct": top_proc_mem_pct,
    }


def extract_windows(
    samples: List[Dict[str, Any]],
    window_duration: float = 5.0,
    step_duration: float = 2.5,
) -> List[Tuple[float, float, List[Dict[str, Any]]]]:
    """Partitions a continuous sample stream into overlapping time slices.

    Returns:
        List of tuples: (window_start_time, window_end_time, window_samples)
    """
    if not samples:
        return []

    t_start = samples[0]["timestamp"]
    t_end = samples[-1]["timestamp"]
    total_duration = t_end - t_start

    # If samples duration is smaller than one window, return single window
    if total_duration < window_duration:
        return [(t_start, t_end, samples)]

    windows = []
    current_start = t_start
    while current_start + window_duration <= t_end + 1e-3:
        current_end = current_start + window_duration
        # Filter samples that fall inside [current_start, current_end]
        w_samples = [s for s in samples if current_start <= s["timestamp"] <= current_end]
        if w_samples:
            windows.append((current_start, current_end, w_samples))
        current_start += step_duration

    return windows


def extract_feature_dataframe(
    samples: List[Dict[str, Any]],
    window_duration: float = 5.0,
    step_duration: float = 2.5,
) -> pd.DataFrame:
    """Extracts features across rolling windows and returns a structured pandas DataFrame."""
    windows = extract_windows(samples, window_duration=window_duration, step_duration=step_duration)
    if not windows:
        return pd.DataFrame(columns=["window_idx", "window_start", "window_end"] + FEATURE_NAMES)

    rows = []
    t_session_start = samples[0]["timestamp"]
    for idx, (w_start, w_end, w_samples) in enumerate(windows):
        feats = extract_features_from_window(w_samples)
        feats["window_idx"] = idx
        feats["window_start_offset"] = w_start - t_session_start
        feats["window_end_offset"] = w_end - t_session_start
        rows.append(feats)

    cols = ["window_idx", "window_start_offset", "window_end_offset"] + FEATURE_NAMES
    return pd.DataFrame(rows)[cols]
