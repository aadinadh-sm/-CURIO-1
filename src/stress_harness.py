"""Synthetic Stress Generation Harness for CURIO.

Provides safe, controlled synthetic workload generators for CPU Pressure,
Memory Pressure, and Disk I/O Pressure on Windows with strict safety bounds
and guaranteed resource cleanup.
"""

import math
import multiprocessing
import os
import sys
import threading
import time
from typing import List, Optional
import psutil


def _cpu_worker(stop_event: multiprocessing.Event) -> None:
    """CPU worker loop with batched iterations to eliminate IPC lock overhead."""
    total = 0.0
    while not stop_event.is_set():
        # Inner loop prevents lock contention while burning user CPU
        for _ in range(50000):
            total += 12345.67 * 9876.54


def _disk_worker(filepath: str, chunk_size_mb: int, stop_event: threading.Event) -> None:
    """Disk I/O worker writing and flushing binary blocks to force physical storage activity."""
    chunk = b"X" * (chunk_size_mb * 1024 * 1024)
    max_file_size = 500 * 1024 * 1024  # Cap scratch file at 500 MB
    try:
        with open(filepath, "wb") as f:
            while not stop_event.is_set():
                f.seek(0)
                written = 0
                while written < max_file_size and not stop_event.is_set():
                    f.write(chunk)
                    f.flush()
                    try:
                        os.fsync(f.fileno())
                    except OSError:
                        pass
                    written += len(chunk)
                    time.sleep(0.005)
    except Exception:
        pass


class CPUStress:
    """Generates controlled multi-core CPU pressure."""

    def __init__(self, intensity: str = "med"):
        """Args:
            intensity: 'low' (50% cores), 'med' (75% cores), 'high' (100% cores).
        """
        self.intensity = intensity
        self._stop_event = multiprocessing.Event()
        self._procs: List[multiprocessing.Process] = []

    def start(self) -> None:
        total_cores = psutil.cpu_count(logical=True) or 2
        if self.intensity == "low":
            num_workers = max(1, int(total_cores * 0.50))
        elif self.intensity == "high":
            num_workers = max(1, total_cores)
        else:  # med
            num_workers = max(1, int(total_cores * 0.75))

        self._stop_event.clear()
        self._procs = [
            multiprocessing.Process(target=_cpu_worker, args=(self._stop_event,), daemon=True)
            for _ in range(num_workers)
        ]
        for p in self._procs:
            p.start()

    def stop(self) -> None:
        self._stop_event.set()
        for p in self._procs:
            p.join(timeout=1.0)
            if p.is_alive():
                p.terminate()
        self._procs = []

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()


class MemoryStress:
    """Generates controlled physical memory pressure with safety headroom retention."""

    def __init__(self, intensity: str = "med", min_headroom_gb: float = 1.5):
        """Args:
            intensity: 'low' (60% RAM target), 'med' (75% RAM target), 'high' (85% RAM target).
            min_headroom_gb: Minimum free RAM to leave untouched for OS stability.
        """
        self.intensity = intensity
        self.min_headroom_bytes = int(min_headroom_gb * (1024**3))
        self._chunks: List[bytearray] = []
        self._running = False

    def start(self) -> None:
        vmem = psutil.virtual_memory()
        target_ratios = {"low": 0.60, "med": 0.75, "high": 0.85}
        target_ratio = target_ratios.get(self.intensity, 0.75)

        # Calculate bytes needed to push total RAM usage to target_ratio
        current_used = vmem.total - vmem.available
        target_used = int(vmem.total * target_ratio)
        bytes_to_alloc = max(0, target_used - current_used)

        # Bound by safe headroom
        max_safe_alloc = max(0, vmem.available - self.min_headroom_bytes)
        alloc_target = min(bytes_to_alloc, max_safe_alloc)

        chunk_size = 32 * 1024 * 1024  # 32 MB chunks
        num_chunks = alloc_target // chunk_size

        self._chunks = []
        try:
            for _ in range(num_chunks):
                # Write non-zero bytes so Windows commits physical memory pages
                self._chunks.append(bytearray(b"M" * chunk_size))
            self._running = True
        except MemoryError:
            pass

    def stop(self) -> None:
        self._chunks.clear()
        self._running = False

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()


class DiskStress:
    """Generates controlled disk I/O write pressure with automatic scratch cleanup."""

    def __init__(self, scratch_dir: str = "data", intensity: str = "med"):
        """Args:
            scratch_dir: Directory where temporary scratch file is created.
            intensity: 'low' (2MB blocks), 'med' (8MB blocks), 'high' (16MB blocks).
        """
        self.scratch_dir = scratch_dir
        self.intensity = intensity
        self.scratch_file = os.path.join(scratch_dir, f"curio_disk_stress_{os.getpid()}.tmp")
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        os.makedirs(self.scratch_dir, exist_ok=True)
        chunk_sizes = {"low": 2, "med": 8, "high": 16}
        chunk_mb = chunk_sizes.get(self.intensity, 8)

        self._stop_event.clear()
        self._thread = threading.Thread(
            target=_disk_worker,
            args=(self.scratch_file, chunk_mb, self._stop_event),
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        self._thread = None

        if os.path.exists(self.scratch_file):
            try:
                os.remove(self.scratch_file)
            except OSError:
                pass

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
