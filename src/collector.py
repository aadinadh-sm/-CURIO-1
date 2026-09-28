"""Telemetry Collection Module for CURIO.

Collects high-frequency system metrics (CPU, Memory, Disk, and Process activity)
at 2 Hz (0.5-second intervals) non-intrusively using psutil on Windows.
"""

import csv
import json
import os
import threading
import time
from typing import Callable, Dict, List, Optional, Any
import psutil


class TelemetryCollector:
    """Samples system telemetry at fixed intervals with background process monitoring."""

    def __init__(self, sample_interval: float = 0.5):
        self.sample_interval = sample_interval
        self._latest_process_stats: Dict[str, Any] = {
            "top_proc_cpu": 0.0,
            "top_proc_cpu_name": "",
            "top_proc_rss": 0.0,
            "top_proc_rss_name": "",
            "proc_count": 0.0,
        }
        self._stop_event = threading.Event()
        self._proc_thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._warmup()

    def _warmup(self) -> None:
        """Initializes psutil internal caches to eliminate first-call OS query lag."""
        try:
            psutil.cpu_percent(interval=None)
            psutil.cpu_percent(percpu=True)
            psutil.virtual_memory()
            psutil.swap_memory()
            psutil.disk_io_counters()
        except Exception:
            pass

    def _process_monitor_loop(self) -> None:
        """Background thread scanning active processes periodically without blocking 2 Hz loop."""
        num_cores = psutil.cpu_count(logical=True) or 1
        while not self._stop_event.is_set():
            max_rss = 0.0
            max_cpu = 0.0
            max_rss_name = ""
            max_cpu_name = ""
            p_count = 0
            try:
                for p in psutil.process_iter(['name', 'memory_info', 'cpu_percent']):
                    try:
                        p_count += 1
                        if p.pid == 0:
                            continue
                        p_name = p.info.get('name') or ''
                        if p_name.lower() in ('system idle process', 'idle'):
                            continue

                        mem = p.info.get('memory_info')
                        if mem and mem.rss > max_rss:
                            max_rss = float(mem.rss)
                            max_rss_name = p_name
                        
                        cpu = p.info.get('cpu_percent')
                        if cpu and cpu > max_cpu:
                            max_cpu = float(cpu)
                            max_cpu_name = p_name
                    except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                        continue
                
                normalized_cpu = max_cpu / float(num_cores)

                with self._lock:
                    self._latest_process_stats = {
                        "top_proc_cpu": normalized_cpu,
                        "top_proc_cpu_name": max_cpu_name,
                        "top_proc_rss": max_rss,
                        "top_proc_rss_name": max_rss_name,
                        "proc_count": float(p_count),
                    }
            except Exception:
                pass

            # Responsive wait using Event instead of blocking time.sleep
            self._stop_event.wait(timeout=1.5)

    def _sample_instant(self, timestamp: float) -> Dict[str, Any]:
        """Captures an instantaneous telemetry snapshot."""
        # Core CPU
        cpu_overall = psutil.cpu_percent(interval=None)
        cpu_cores = psutil.cpu_percent(percpu=True)

        # Virtual Memory
        vmem = psutil.virtual_memory()
        swap = psutil.swap_memory()

        # Disk I/O
        disk = psutil.disk_io_counters()
        read_bytes = getattr(disk, 'read_bytes', 0) if disk else 0
        write_bytes = getattr(disk, 'write_bytes', 0) if disk else 0
        read_count = getattr(disk, 'read_count', 0) if disk else 0
        write_count = getattr(disk, 'write_count', 0) if disk else 0
        read_time = getattr(disk, 'read_time', 0) if disk else 0
        write_time = getattr(disk, 'write_time', 0) if disk else 0

        # Process snapshot from background worker
        with self._lock:
            proc_stats = dict(self._latest_process_stats)

        # Fallback for process count if worker hasn't completed first pass
        if proc_stats["proc_count"] == 0.0:
            try:
                proc_stats["proc_count"] = float(len(psutil.pids()))
            except Exception:
                proc_stats["proc_count"] = 1.0

        return {
            "timestamp": timestamp,
            "cpu_overall": float(cpu_overall),
            "cpu_cores": [float(c) for c in cpu_cores],
            "vmem_percent": float(vmem.percent),
            "vmem_available": int(vmem.available),
            "vmem_total": int(vmem.total),
            "swap_percent": float(swap.percent),
            "swap_used": int(swap.used),
            "swap_total": int(swap.total),
            "disk_read_bytes": int(read_bytes),
            "disk_write_bytes": int(write_bytes),
            "disk_read_count": int(read_count),
            "disk_write_count": int(write_count),
            "disk_read_time": int(read_time),
            "disk_write_time": int(write_time),
            "process_count": int(proc_stats["proc_count"]),
            "top_proc_cpu": float(proc_stats["top_proc_cpu"]),
            "top_proc_rss": float(proc_stats["top_proc_rss"]),
            "top_proc_cpu_name": str(proc_stats.get("top_proc_cpu_name", "")),
            "top_proc_rss_name": str(proc_stats.get("top_proc_rss_name", "")),
        }

    def collect(
        self,
        duration_seconds: float = 30.0,
        progress_callback: Optional[Callable[[float, int, int], None]] = None,
    ) -> List[Dict[str, Any]]:
        """Collects telemetry samples at self.sample_interval for duration_seconds.

        Args:
            duration_seconds: Total duration to capture in seconds.
            progress_callback: Optional callback(elapsed_seconds, samples_collected, total_expected).

        Returns:
            List of telemetry sample dictionaries.
        """
        samples: List[Dict[str, Any]] = []
        expected_samples = max(1, int(round(duration_seconds / self.sample_interval)) + 1)

        # Start process monitor thread
        self._stop_event.clear()
        self._proc_thread = threading.Thread(target=self._process_monitor_loop, daemon=True)
        self._proc_thread.start()

        start_time = time.time()
        next_tick = start_time

        try:
            for sample_idx in range(expected_samples):
                now = time.time()
                sample = self._sample_instant(now)
                samples.append(sample)

                if progress_callback:
                    progress_callback(now - start_time, sample_idx + 1, expected_samples)

                if sample_idx < expected_samples - 1:
                    next_tick += self.sample_interval
                    sleep_duration = next_tick - time.time()
                    if sleep_duration > 0:
                        time.sleep(sleep_duration)
                    else:
                        next_tick = time.time()
        finally:
            self._stop_event.set()
            if self._proc_thread and self._proc_thread.is_alive():
                self._proc_thread.join(timeout=0.5)

        return samples


def save_raw_telemetry(samples: List[Dict[str, Any]], filepath: str) -> None:
    """Serializes raw samples to a CSV file."""
    if not samples:
        return

    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    meta_fields = [
        "machine_id",
        "session_id",
        "condition",
        "stress_level",
        "background_workload",
        "is_ramp_up",
    ]
    core_fields = [
        "timestamp",
        "cpu_overall",
        "cpu_cores_json",
        "vmem_percent",
        "vmem_available",
        "vmem_total",
        "swap_percent",
        "swap_used",
        "swap_total",
        "disk_read_bytes",
        "disk_write_bytes",
        "disk_read_count",
        "disk_write_count",
        "disk_read_time",
        "disk_write_time",
        "process_count",
        "top_proc_cpu",
        "top_proc_rss",
    ]
    fieldnames = meta_fields + core_fields

    with open(filepath, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for s in samples:
            row = dict(s)
            if "cpu_cores" in row:
                row["cpu_cores_json"] = json.dumps(row.pop("cpu_cores"))
            writer.writerow(row)


def load_raw_telemetry(filepath: str) -> List[Dict[str, Any]]:
    """Loads raw telemetry samples from a CSV file."""
    samples: List[Dict[str, Any]] = []
    with open(filepath, mode="r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sample = {
                "timestamp": float(row["timestamp"]),
                "cpu_overall": float(row["cpu_overall"]),
                "cpu_cores": json.loads(row["cpu_cores_json"]) if "cpu_cores_json" in row and row["cpu_cores_json"] else [],
                "vmem_percent": float(row["vmem_percent"]),
                "vmem_available": int(row["vmem_available"]),
                "vmem_total": int(row["vmem_total"]),
                "swap_percent": float(row["swap_percent"]),
                "swap_used": int(row["swap_used"]),
                "swap_total": int(row["swap_total"]),
                "disk_read_bytes": int(row["disk_read_bytes"]),
                "disk_write_bytes": int(row["disk_write_bytes"]),
                "disk_read_count": int(row.get("disk_read_count", 0)),
                "disk_write_count": int(row.get("disk_write_count", 0)),
                "disk_read_time": int(row["disk_read_time"]),
                "disk_write_time": int(row["disk_write_time"]),
                "process_count": int(row["process_count"]),
                "top_proc_cpu": float(row["top_proc_cpu"]),
                "top_proc_rss": float(row["top_proc_rss"]),
            }
            # Optional metadata
            for m in ["machine_id", "session_id", "condition", "stress_level", "background_workload"]:
                if m in row:
                    sample[m] = row[m]
            if row.get("is_ramp_up"):
                try:
                    sample["is_ramp_up"] = int(row["is_ramp_up"])
                except ValueError:
                    pass

            samples.append(sample)
    return samples
