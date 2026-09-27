"""Dataset Builder & Session Orchestration Module for CURIO.

Manages the execution of Protocol A (Normal) and Protocol B (Stress) sessions,
attaching rigorous provenance metadata (machine_id, session_id, condition,
stress_level, background_workload) and generating both raw and windowed feature data.
"""

import os
import platform
import time
import uuid
from typing import Any, Dict, List, Optional
import pandas as pd

from src.collector import TelemetryCollector, save_raw_telemetry, load_raw_telemetry
from src.features import extract_feature_dataframe, FEATURE_NAMES
from src.stress_harness import CPUStress, MemoryStress, DiskStress


def get_default_machine_id() -> str:
    """Returns a standardized local machine identifier."""
    node = platform.node().lower().strip()
    return node if node else "unknown_machine"


class DatasetBuilder:
    """Orchestrates telemetry capture sessions under controlled experimental protocols."""

    def __init__(
        self,
        raw_dir: str = "data/raw",
        processed_dir: str = "data/processed",
        machine_id: Optional[str] = None,
    ):
        self.raw_dir = raw_dir
        self.processed_dir = processed_dir
        self.machine_id = machine_id or get_default_machine_id()
        os.makedirs(self.raw_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)

    def record_session(
        self,
        condition: str,
        stress_level: str = "none",
        background_workload: str = "idle",
        duration_seconds: float = 30.0,
        ramp_duration_seconds: float = 5.0,
        session_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Executes a single controlled telemetry session and stores raw and feature data.

        Args:
            condition: 'normal', 'cpu_pressure', 'memory_pressure', 'disk_io_pressure'.
            stress_level: 'none', 'low', 'med', 'high'.
            background_workload: 'idle', 'browsing', 'coding', 'office', 'streaming'.
            duration_seconds: Total duration to capture.
            ramp_duration_seconds: Initial portion treated as ramp-up.
            session_id: Optional unique session ID (auto-generated if None).

        Returns:
            Dictionary containing session summary and output paths.
        """
        if session_id is None:
            short_uuid = str(uuid.uuid4())[:8]
            session_id = f"{self.machine_id}_{condition}_{stress_level}_{short_uuid}"

        collector = TelemetryCollector(sample_interval=0.5)

        # Select stress context manager if applicable
        stress_context = None
        if condition == "cpu_pressure":
            stress_context = CPUStress(intensity=stress_level)
        elif condition == "memory_pressure":
            stress_context = MemoryStress(intensity=stress_level, min_headroom_gb=1.5)
        elif condition == "disk_io_pressure":
            stress_context = DiskStress(scratch_dir="data", intensity=stress_level)

        start_time = time.time()
        samples: List[Dict[str, Any]] = []

        if stress_context:
            with stress_context:
                # Allow a brief moment for stress worker initialization
                time.sleep(0.5)
                samples = collector.collect(duration_seconds=duration_seconds)
        else:
            samples = collector.collect(duration_seconds=duration_seconds)

        # Tag raw samples with metadata
        for s in samples:
            s["machine_id"] = self.machine_id
            s["session_id"] = session_id
            s["condition"] = condition
            s["stress_level"] = stress_level
            s["background_workload"] = background_workload
            offset = s["timestamp"] - start_time
            s["is_ramp_up"] = 1 if offset < ramp_duration_seconds else 0

        # Save raw telemetry
        raw_path = os.path.join(self.raw_dir, f"{session_id}_raw.csv")
        save_raw_telemetry(samples, raw_path)

        # Extract rolling window features
        df_features = extract_feature_dataframe(samples, window_duration=5.0, step_duration=2.5)

        # Attach metadata to windowed features
        df_features["machine_id"] = self.machine_id
        df_features["session_id"] = session_id
        df_features["condition"] = condition
        df_features["stress_level"] = stress_level
        df_features["background_workload"] = background_workload
        # Mark window as ramp-up if window starts before ramp duration
        df_features["is_ramp_up"] = (df_features["window_start_offset"] < ramp_duration_seconds).astype(int)

        processed_path = os.path.join(self.processed_dir, f"{session_id}_features.csv")
        df_features.to_csv(processed_path, index=False)

        return {
            "session_id": session_id,
            "machine_id": self.machine_id,
            "condition": condition,
            "stress_level": stress_level,
            "background_workload": background_workload,
            "sample_count": len(samples),
            "window_count": len(df_features),
            "raw_path": raw_path,
            "processed_path": processed_path,
            "features_df": df_features,
        }

    def load_all_processed_features(self) -> pd.DataFrame:
        """Loads and concatenates all feature CSVs in self.processed_dir."""
        csv_files = [
            os.path.join(self.processed_dir, f)
            for f in os.listdir(self.processed_dir)
            if f.endswith("_features.csv")
        ]
        if not csv_files:
            return pd.DataFrame()

        dfs = [pd.read_csv(f) for f in csv_files]
        return pd.concat(dfs, ignore_index=True)
