"""CURIO Live Diagnosis Pipeline.

Coordinates high-frequency telemetry collection (30 seconds, 61 samples @ 2 Hz),
12-feature windowing (11 windows), calibrated multi-class diagnosis, session
abnormality assessment, evidence attribution, and temporal discovery.

Adheres strictly to inference-only execution with zero runtime training,
strict non-causal language, and deterministic evaluation.
"""

from datetime import datetime, timezone
import json
import os
import time
from typing import Any, Callable, Dict, List, Optional, Union
import uuid

import joblib
import numpy as np
import pandas as pd

from src.collector import TelemetryCollector, save_raw_telemetry
from src.discovery import (
    DiscoveryReferenceProfile,
    discover_session_trajectory,
)
from src.evidence import (
    EvidenceReferenceProfile,
    generate_evidence,
    validate_no_causal_language,
)
from src.features import (
    FEATURE_NAMES,
    extract_feature_dataframe,
)

DEFAULT_MODEL_DIR = "data/models/physical_deployment"
DEFAULT_HISTORY_DIR = "data/diagnosis_history"
EXPECTED_RAW_SAMPLES = 61
EXPECTED_FEATURE_WINDOWS = 11
EXPECTED_DURATION_SECONDS = 30.0
EXPECTED_SAMPLE_INTERVAL = 0.5


class DiagnosisCancelledError(Exception):
    """Raised when a diagnostic capture/session is cancelled by user request."""
    pass


class CurioDiagnosisPipeline:
    """Central operational pipeline coordinator for CURIO diagnosis."""

    def __init__(
        self,
        model_dir: str = DEFAULT_MODEL_DIR,
        history_dir: str = DEFAULT_HISTORY_DIR,
    ) -> None:
        """Initializes the pipeline and validates deployment artifacts.

        Args:
            model_dir: Directory containing trained model and reference profiles.
            history_dir: Directory where diagnosis run records will be stored.
        """
        self.model_dir = model_dir
        self.history_dir = history_dir
        os.makedirs(self.history_dir, exist_ok=True)

        self._validate_and_load_artifacts()

    def _validate_and_load_artifacts(self) -> None:
        """Verifies existence of all required deployment artifacts and loads them."""
        required_files = {
            "model": os.path.join(self.model_dir, "model.joblib"),
            "calibration_metadata": os.path.join(self.model_dir, "calibration_metadata.json"),
            "abnormality_threshold": os.path.join(self.model_dir, "abnormality_threshold.json"),
            "feature_metadata": os.path.join(self.model_dir, "feature_metadata.json"),
            "evidence_reference": os.path.join(self.model_dir, "evidence_reference.json"),
            "discovery_reference": os.path.join(self.model_dir, "discovery_reference.json"),
        }

        missing = [f"{k} ('{path}')" for k, path in required_files.items() if not os.path.exists(path)]
        if missing:
            raise FileNotFoundError(
                f"Missing required deployment artifact(s): {', '.join(missing)}. "
                f"Please ensure '{self.model_dir}' contains all required deployment files."
            )

        # 1. Load ML Model
        self.model = joblib.load(required_files["model"])
        self.classes = list(self.model.classes_)
        if "normal" not in self.classes:
            raise ValueError(f"Model classes must include 'normal'. Found: {self.classes}")

        # 2. Load Calibration Metadata
        with open(required_files["calibration_metadata"], "r", encoding="utf-8") as f:
            self.calibration_metadata = json.load(f)

        # 3. Load Abnormality Threshold
        with open(required_files["abnormality_threshold"], "r", encoding="utf-8") as f:
            abn_data = json.load(f)
            if "abnormality_threshold" not in abn_data:
                raise KeyError("abnormality_threshold.json must contain 'abnormality_threshold' key.")
            self.abnormality_threshold = float(abn_data["abnormality_threshold"])
            self.abnormality_metadata = abn_data

        # 4. Load Feature Metadata
        with open(required_files["feature_metadata"], "r", encoding="utf-8") as f:
            self.feature_metadata = json.load(f)

        # 5. Load Evidence Reference Profile
        self.evidence_ref = EvidenceReferenceProfile.from_json(required_files["evidence_reference"])

        # 6. Load Discovery Reference Profile
        self.discovery_ref = DiscoveryReferenceProfile.from_json(required_files["discovery_reference"])

    def run_diagnosis(
        self,
        raw_samples: Optional[List[Dict[str, Any]]] = None,
        duration_seconds: float = EXPECTED_DURATION_SECONDS,
        sample_interval: float = EXPECTED_SAMPLE_INTERVAL,
        progress_callback: Optional[Callable[[float, int, int], None]] = None,
        stage_callback: Optional[Callable[[str], None]] = None,
        cancel_event: Optional[Any] = None,
        save_history: bool = True,
        session_id: Optional[str] = None,
        save_raw: bool = False,
    ) -> Dict[str, Any]:
        """Executes a full 30-second live diagnosis session.

        Sequence:
        1. Collects 30 seconds of telemetry @ 2 Hz (or uses injected raw_samples for testing/replay).
        2. Validates raw sample geometry (exactly 61 samples).
        3. Extracts rolling features across 11 windows (5s window, 2.5s step).
        4. Validates feature integrity (no NaN/Inf, all 12 features present).
        5. Computes calibrated probabilities across all 11 windows.
        6. Aggregates session-level condition via mean calibrated probabilities.
        7. Evaluates session abnormality against the deployment threshold.
        8. Generates explanatory evidence for the representative strongest window.
        9. Discovers the temporal onset trajectory across the 11 windows.
        10. Records execution latencies and optionally saves diagnosis history.

        Returns:
            Structured diagnosis result dictionary conforming to Section 9 schema.
        """
        pipeline_start_perf = time.perf_counter()
        started_at_iso = datetime.now(timezone.utc).isoformat()

        if not session_id:
            time_tag = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            session_id = f"curio_session_{time_tag}_{uuid.uuid4().hex[:6]}"

        # --------------------------------------------------------------
        # 1. Telemetry Capture
        # --------------------------------------------------------------
        if stage_callback:
            stage_callback("collecting")

        t_capture_start = time.perf_counter()
        if raw_samples is None:
            collector = TelemetryCollector(sample_interval=sample_interval)

            def internal_progress(el: float, cur: int, tot: int):
                if cancel_event and cancel_event.is_set():
                    raise DiagnosisCancelledError("Diagnosis cancelled by user request.")
                if progress_callback:
                    progress_callback(el, cur, tot)

            try:
                collected_samples = collector.collect(
                    duration_seconds=duration_seconds,
                    progress_callback=internal_progress,
                )
            except (KeyboardInterrupt, DiagnosisCancelledError):
                # Ensure resources are cleanly released
                collector._stop_event.set()
                if collector._proc_thread and collector._proc_thread.is_alive():
                    collector._proc_thread.join(timeout=0.5)
                raise
            capture_duration = time.perf_counter() - t_capture_start
        else:
            if cancel_event and cancel_event.is_set():
                raise DiagnosisCancelledError("Diagnosis cancelled by user request.")
            collected_samples = raw_samples
            capture_duration = duration_seconds

        # Validate Raw Sample Geometry
        expected_samples = max(1, int(round(duration_seconds / sample_interval)) + 1)
        if len(collected_samples) != expected_samples:
            raise ValueError(
                f"Collection error: Expected {expected_samples} raw telemetry samples for "
                f"{duration_seconds}s at {1.0/sample_interval:.1f} Hz, but received {len(collected_samples)}."
            )

        # --------------------------------------------------------------
        # 2. Feature Extraction
        # --------------------------------------------------------------
        if stage_callback:
            stage_callback("extracting_features")

        t_feat_start = time.perf_counter()
        df_features = extract_feature_dataframe(
            samples=collected_samples,
            window_duration=5.0,
            step_duration=2.5,
        )
        feature_extraction_ms = (time.perf_counter() - t_feat_start) * 1000.0

        # Validate Window Count
        if len(df_features) != EXPECTED_FEATURE_WINDOWS:
            raise ValueError(
                f"Feature extraction error: Expected {EXPECTED_FEATURE_WINDOWS} rolling windows, "
                f"but extracted {len(df_features)}."
            )

        # Validate Feature Integrity (No NaNs, Infs, missing columns)
        missing_feats = [f for f in FEATURE_NAMES if f not in df_features.columns]
        if missing_feats:
            raise ValueError(f"Feature extraction error: Missing expected feature column(s): {missing_feats}")

        feat_values = df_features[FEATURE_NAMES].to_numpy(dtype=float)
        if np.isnan(feat_values).any() or np.isinf(feat_values).any():
            raise ValueError("Feature validation error: Extracted features contain NaN or Infinite values.")

        # --------------------------------------------------------------
        # 3. Calibrated ML Diagnosis
        # --------------------------------------------------------------
        if stage_callback:
            stage_callback("ml_diagnosis")

        t_inf_start = time.perf_counter()
        window_probs = self.model.predict_proba(feat_values)
        inference_ms = (time.perf_counter() - t_inf_start) * 1000.0

        if window_probs.shape != (EXPECTED_FEATURE_WINDOWS, len(self.classes)):
            raise ValueError(
                f"Inference error: Expected probability matrix of shape ({EXPECTED_FEATURE_WINDOWS}, {len(self.classes)}), "
                f"got {window_probs.shape}."
            )

        normal_idx = self.classes.index("normal")

        # Per-window predictions
        per_window_predictions: List[Dict[str, Any]] = []
        window_abnormalities: List[float] = []

        for w_idx in range(EXPECTED_FEATURE_WINDOWS):
            w_p = window_probs[w_idx]
            w_best_idx = int(np.argmax(w_p))
            w_abn = float(1.0 - w_p[normal_idx])
            window_abnormalities.append(w_abn)

            per_window_predictions.append({
                "window_idx": w_idx,
                "window_start_seconds": w_idx * 2.5,
                "window_end_seconds": w_idx * 2.5 + 5.0,
                "predicted_condition": self.classes[w_best_idx],
                "confidence": float(w_p[w_best_idx]),
                "probability_normal": float(w_p[normal_idx]),
                "abnormality_score": float(w_abn),
                "class_probabilities": {self.classes[i]: float(w_p[i]) for i in range(len(self.classes))},
            })

        # Session-Level Aggregation: Mean calibrated probability across all 11 windows
        mean_probs = np.mean(window_probs, axis=0)
        session_pred_idx = int(np.argmax(mean_probs))
        session_condition = self.classes[session_pred_idx]
        session_confidence = float(mean_probs[session_pred_idx])

        class_probabilities = {self.classes[i]: float(mean_probs[i]) for i in range(len(self.classes))}

        # --------------------------------------------------------------
        # 4. Session Abnormality Assessment
        # --------------------------------------------------------------
        mean_abnormality = float(np.mean(window_abnormalities))
        max_abnormality = float(np.max(window_abnormalities))
        final_window_abnormality = float(window_abnormalities[-1])

        thresh = self.abnormality_threshold
        abnormal_window_count = int(sum(1 for s in window_abnormalities if s > thresh))
        abnormal_window_ratio = float(abnormal_window_count / len(window_abnormalities))

        # Deterministic Session Abnormal Flag: Majority of diagnostic windows (>= 6 of 11) above threshold
        session_abnormal = bool(abnormal_window_count >= 6)

        abnormality_summary = {
            "mean_score": round(mean_abnormality, 4),
            "max_score": round(max_abnormality, 4),
            "final_window_score": round(final_window_abnormality, 4),
            "abnormal_window_count": abnormal_window_count,
            "abnormal_window_ratio": round(abnormal_window_ratio, 4),
            "threshold": round(thresh, 4),
            "session_abnormal": session_abnormal,
            "per_window_scores": [round(float(s), 4) for s in window_abnormalities],
        }

        # --------------------------------------------------------------
        # 5. Evidence Integration
        # --------------------------------------------------------------
        if stage_callback:
            stage_callback("generating_evidence")

        t_ev_start = time.perf_counter()

        # Deterministic Representative Window Selection:
        # For Normal: Window with lowest abnormality score (highest normal probability)
        # For Abnormal conditions: Window with highest predicted condition probability
        if session_condition == "normal":
            rep_window_idx = int(np.argmin(window_abnormalities))
        else:
            rep_window_idx = int(np.argmax(window_probs[:, session_pred_idx]))

        rep_features = df_features.iloc[rep_window_idx][FEATURE_NAMES].to_dict()
        rep_probs = window_probs[rep_window_idx]
        rep_abnormality = window_abnormalities[rep_window_idx]

        evidence_res = generate_evidence(
            predicted_condition=session_condition,
            calibrated_probabilities=rep_probs,
            current_features=rep_features,
            reference_profile=self.evidence_ref,
            abnormality_score=rep_abnormality,
            classes=self.classes,
        )
        evidence_res["representative_window_idx"] = rep_window_idx
        evidence_res["representative_window_time"] = f"{rep_window_idx * 2.5:.1f}s-{rep_window_idx * 2.5 + 5.0:.1f}s"
        evidence_ms = (time.perf_counter() - t_ev_start) * 1000.0

        # --------------------------------------------------------------
        # 6. Discovery Integration
        # --------------------------------------------------------------
        if stage_callback:
            stage_callback("discovering_patterns")

        t_disc_start = time.perf_counter()
        discovery_res = discover_session_trajectory(
            df_session=df_features,
            predicted_condition=session_condition,
            evidence_ref=self.evidence_ref,
            discovery_ref=self.discovery_ref,
        )
        discovery_ms = (time.perf_counter() - t_disc_start) * 1000.0

        # --------------------------------------------------------------
        # 7. Performance Tracking
        # --------------------------------------------------------------
        total_analysis_ms = feature_extraction_ms + inference_ms + evidence_ms + discovery_ms
        perf_summary = {
            "capture_duration_seconds": round(capture_duration, 2),
            "feature_extraction_ms": round(feature_extraction_ms, 2),
            "inference_ms": round(inference_ms, 2),
            "evidence_ms": round(evidence_ms, 2),
            "discovery_ms": round(discovery_ms, 2),
            "reporting_ms": 0.0,  # Updated upon report generation
            "total_analysis_ms": round(total_analysis_ms, 2),
        }

        # --------------------------------------------------------------
        # 8. Assemble Result Schema
        # --------------------------------------------------------------
        result: Dict[str, Any] = {
            "session_id": session_id,
            "capture_started_at": started_at_iso,
            "capture_duration_seconds": round(capture_duration, 2),
            "raw_samples_count": len(collected_samples),
            "feature_windows_count": len(df_features),
            "diagnosis": {
                "condition": session_condition,
                "confidence": round(session_confidence, 4),
                "class_probabilities": {k: round(v, 4) for k, v in class_probabilities.items()},
                "session_probability_normal": round(class_probabilities.get("normal", 0.0), 4),
                "session_probability_cpu": round(class_probabilities.get("cpu_pressure", 0.0), 4),
                "session_probability_memory": round(class_probabilities.get("memory_pressure", 0.0), 4),
                "session_probability_disk": round(class_probabilities.get("disk_io_pressure", 0.0), 4),
                "per_window_predictions": per_window_predictions,
            },
            "abnormality": abnormality_summary,
            "evidence": evidence_res,
            "discovery": discovery_res,
            "performance": perf_summary,
            "metadata": {
                "model_version": self.calibration_metadata.get("method", "calibrated_rf"),
                "abnormality_threshold": thresh,
                "feature_names": FEATURE_NAMES,
                "pipeline_version": "1.0.0",
            },
        }

        # --------------------------------------------------------------
        # 9. Save Diagnosis History
        # --------------------------------------------------------------
        if save_history:
            history_path = os.path.join(self.history_dir, f"{session_id}_diagnosis.json")
            with open(history_path, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2)

            if save_raw and collected_samples:
                raw_path = os.path.join(self.history_dir, f"{session_id}_raw.csv")
                save_raw_telemetry(collected_samples, raw_path)

        return result
