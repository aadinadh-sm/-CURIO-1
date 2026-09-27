"""FastAPI Local Intelligence Server for CURIO.

Provides a secure, local-only REST API bound strictly to 127.0.0.1, coordinating
asynchronous live telemetry diagnosis, pre-collected replay diagnosis,
explanatory evidence, temporal trajectory discovery, and historical records.

Security & Integrity:
- Bound strictly to 127.0.0.1.
- Restrictive CORS configuration for local frontend dev & production ports.
- Input validation & strict regex identifier sanitization against path traversal.
- Zero external network requests, zero telemetry uploads, zero LLM dependencies.
- Zero runtime model training or threshold alteration.
"""

from datetime import datetime, timezone
import json
import os
import re
import threading
import time
from typing import Any, Dict, List, Optional
import uuid

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.collector import load_raw_telemetry
from src.diagnosis_pipeline import (
    CurioDiagnosisPipeline,
    DEFAULT_MODEL_DIR,
    DEFAULT_HISTORY_DIR,
    EXPECTED_RAW_SAMPLES,
    DiagnosisCancelledError,
)

# ----------------------------------------------------------------------
# Application Setup & Local Security
# ----------------------------------------------------------------------

app = FastAPI(
    title="CURIO Local Intelligence API",
    version="1.0.0",
    description="Local machine intelligence for discovering unusual operating patterns.",
    docs_url="/api/docs",
    redoc_url=None,
    openapi_url="/api/openapi.json",
)

# Restrict CORS to trusted local endpoints only
CORS_ORIGINS = [
    "http://127.0.0.1:3000",
    "http://localhost:3000",
    "http://127.0.0.1:5173",
    "http://localhost:5173",
    "http://127.0.0.1:8000",
    "http://localhost:8000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

PHYSICAL_RAW_DIR = "data/physical_raw"

# Lazy-loaded pipeline singleton
_pipeline: Optional[CurioDiagnosisPipeline] = None
_pipeline_lock = threading.Lock()


def get_pipeline() -> CurioDiagnosisPipeline:
    """Returns the shared CurioDiagnosisPipeline instance."""
    global _pipeline
    with _pipeline_lock:
        if _pipeline is None:
            _pipeline = CurioDiagnosisPipeline(
                model_dir=DEFAULT_MODEL_DIR,
                history_dir=DEFAULT_HISTORY_DIR,
            )
        return _pipeline


def set_pipeline(pipeline: CurioDiagnosisPipeline) -> None:
    """Sets the pipeline instance (primarily used for unit testing)."""
    global _pipeline
    with _pipeline_lock:
        _pipeline = pipeline


# ----------------------------------------------------------------------
# In-Memory Active Diagnosis Tracker
# ----------------------------------------------------------------------

_active_tasks: Dict[str, Dict[str, Any]] = {}
_tasks_lock = threading.Lock()


def sanitize_identifier(identifier: str) -> str:
    """Validates that an identifier contains only safe alphanumeric/dash/underscore characters."""
    if not identifier or not re.match(r"^[a-zA-Z0-9_\-]+$", identifier):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid identifier. Only alphanumeric characters, hyphens, and underscores are allowed.",
        )
    return identifier


# ----------------------------------------------------------------------
# Request & Response Schemas
# ----------------------------------------------------------------------

class StartDiagnosisRequest(BaseModel):
    mode: str = Field(default="live", description="'live' for 30s hardware capture, or 'replay' for instant demo")
    replay_condition: Optional[str] = Field(default=None, description="Target condition for replay mode (e.g. cpu_pressure)")
    session_id: Optional[str] = Field(default=None, description="Optional custom session ID")


class DiagnosisStatusResponse(BaseModel):
    diagnosis_id: str
    state: str  # idle, collecting, analyzing, complete, cancelled, error
    stage: str
    elapsed_seconds: float
    progress: float
    samples_collected: int
    total_samples: int
    started_at: str
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


# ----------------------------------------------------------------------
# API Endpoints
# ----------------------------------------------------------------------

@app.get("/api/health")
def get_health() -> Dict[str, Any]:
    """Health check endpoint confirming local-only operational status."""
    return {
        "status": "ok",
        "service": "CURIO Local Intelligence API",
        "version": "1.0.0",
        "local_only": True,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/api/status")
def get_status() -> Dict[str, Any]:
    """Provides operational readiness, model metadata, and historical counts."""
    try:
        pipeline = get_pipeline()
        history_files = [f for f in os.listdir(pipeline.history_dir) if f.endswith("_diagnosis.json")] if os.path.exists(pipeline.history_dir) else []
        return {
            "ready": True,
            "model_loaded": True,
            "deployment_dir": pipeline.model_dir,
            "classes": pipeline.classes,
            "abnormality_threshold": pipeline.abnormality_threshold,
            "features_count": len(pipeline.feature_metadata.get("feature_names", [])),
            "history_count": len(history_files),
            "local_only": True,
            "active_tasks_count": len(_active_tasks),
        }
    except Exception as e:
        return {
            "ready": False,
            "model_loaded": False,
            "error": str(e),
            "local_only": True,
        }


def _run_diagnosis_worker(
    diagnosis_id: str,
    mode: str,
    replay_condition: Optional[str],
    pipeline: CurioDiagnosisPipeline,
    cancel_event: threading.Event,
) -> None:
    """Background worker executing the diagnostic sequence."""
    try:
        def on_progress(elapsed: float, current: int, total: int):
            with _tasks_lock:
                if diagnosis_id in _active_tasks:
                    task = _active_tasks[diagnosis_id]
                    task["elapsed_seconds"] = round(elapsed, 1)
                    task["samples_collected"] = current
                    task["total_samples"] = total
                    task["progress"] = round(min(1.0, current / max(1, total)), 3)
                    task["stage"] = f"Collecting telemetry... ({current}/{total} samples)"

        def on_stage(stage_name: str):
            with _tasks_lock:
                if diagnosis_id in _active_tasks:
                    task = _active_tasks[diagnosis_id]
                    if stage_name == "collecting":
                        task["state"] = "collecting"
                        task["stage"] = "Collecting 30-second telemetry session..."
                    elif stage_name in ("extracting_features", "ml_diagnosis"):
                        task["state"] = "analyzing"
                        task["stage"] = "Analyzing telemetry & calculating calibrated probabilities..."
                    elif stage_name == "generating_evidence":
                        task["state"] = "analyzing"
                        task["stage"] = "Generating explanatory evidence attribution..."
                    elif stage_name == "discovering_patterns":
                        task["state"] = "analyzing"
                        task["stage"] = "Discovering temporal onset progression..."

        raw_samples = None
        if mode == "replay":
            with _tasks_lock:
                if diagnosis_id in _active_tasks:
                    _active_tasks[diagnosis_id]["stage"] = "Loading physical telemetry for replay..."

            # Find matching physical raw file
            target_cond = replay_condition or "memory_pressure"
            raw_files = [f for f in os.listdir(PHYSICAL_RAW_DIR) if f.endswith("_raw.csv")] if os.path.exists(PHYSICAL_RAW_DIR) else []
            matched = None
            for rf in sorted(raw_files):
                if f"_{target_cond}_" in rf:
                    matched = rf
                    break
            if not matched and raw_files:
                matched = raw_files[0]

            if not matched:
                raise FileNotFoundError(f"No pre-collected physical telemetry found in '{PHYSICAL_RAW_DIR}'.")

            raw_path = os.path.join(PHYSICAL_RAW_DIR, matched)
            raw_samples = load_raw_telemetry(raw_path)

            # Rapid progress animation simulation for replay UX
            for tick in range(1, 62, 10):
                if cancel_event.is_set():
                    raise DiagnosisCancelledError("Replay diagnosis cancelled by user.")
                on_progress(tick * 0.5, tick, 61)
                time.sleep(0.02)

        # Run diagnosis pipeline
        result = pipeline.run_diagnosis(
            raw_samples=raw_samples,
            progress_callback=on_progress,
            stage_callback=on_stage,
            cancel_event=cancel_event,
            save_history=True,
            session_id=diagnosis_id,
        )

        with _tasks_lock:
            if diagnosis_id in _active_tasks:
                task = _active_tasks[diagnosis_id]
                task["state"] = "complete"
                task["progress"] = 1.0
                task["stage"] = "Diagnosis complete."
                task["result"] = result

    except DiagnosisCancelledError:
        with _tasks_lock:
            if diagnosis_id in _active_tasks:
                task = _active_tasks[diagnosis_id]
                task["state"] = "cancelled"
                task["stage"] = "Diagnosis cancelled. System state restored."
    except Exception as e:
        with _tasks_lock:
            if diagnosis_id in _active_tasks:
                task = _active_tasks[diagnosis_id]
                task["state"] = "error"
                task["stage"] = "Error encountered during diagnosis."
                task["error"] = str(e)


@app.post("/api/diagnosis/start", status_code=status.HTTP_202_ACCEPTED)
def start_diagnosis(req: StartDiagnosisRequest = StartDiagnosisRequest()) -> Dict[str, Any]:
    """Initiates an asynchronous 30-second live or instant replay diagnosis."""
    pipeline = get_pipeline()

    if req.session_id:
        diagnosis_id = sanitize_identifier(req.session_id)
    else:
        time_tag = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        diagnosis_id = f"curio_{time_tag}_{uuid.uuid4().hex[:6]}"

    with _tasks_lock:
        # Check if already running with same ID
        if diagnosis_id in _active_tasks and _active_tasks[diagnosis_id]["state"] in ("collecting", "analyzing"):
            raise HTTPException(status_code=409, detail=f"Diagnosis '{diagnosis_id}' is already running.")

        cancel_ev = threading.Event()
        _active_tasks[diagnosis_id] = {
            "diagnosis_id": diagnosis_id,
            "state": "collecting",
            "stage": "Initializing telemetry capture...",
            "elapsed_seconds": 0.0,
            "progress": 0.0,
            "samples_collected": 0,
            "total_samples": EXPECTED_RAW_SAMPLES,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "cancel_event": cancel_ev,
            "result": None,
            "error": None,
            "mode": req.mode,
        }

    thread = threading.Thread(
        target=_run_diagnosis_worker,
        args=(diagnosis_id, req.mode, req.replay_condition, pipeline, cancel_ev),
        daemon=True,
    )
    thread.start()

    return {
        "diagnosis_id": diagnosis_id,
        "state": "collecting",
    }


@app.get("/api/diagnosis/{diagnosis_id}")
def get_diagnosis_status(diagnosis_id: str) -> Dict[str, Any]:
    """Exposes live progress or returns the completed diagnosis result."""
    safe_id = sanitize_identifier(diagnosis_id)

    with _tasks_lock:
        if safe_id in _active_tasks:
            task = _active_tasks[safe_id]
            return {
                "diagnosis_id": safe_id,
                "state": task["state"],
                "stage": task["stage"],
                "elapsed_seconds": task["elapsed_seconds"],
                "progress": task["progress"],
                "samples_collected": task["samples_collected"],
                "total_samples": task["total_samples"],
                "started_at": task["started_at"],
                "result": task.get("result"),
                "error": task.get("error"),
            }

    # If not active in memory, check if it was previously saved to history
    pipeline = get_pipeline()
    history_file = os.path.join(pipeline.history_dir, f"{safe_id}_diagnosis.json")
    if os.path.exists(history_file):
        with open(history_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        return {
            "diagnosis_id": safe_id,
            "state": "complete",
            "stage": "Diagnosis loaded from history.",
            "elapsed_seconds": data.get("capture_duration_seconds", 30.0),
            "progress": 1.0,
            "samples_collected": data.get("raw_samples_count", EXPECTED_RAW_SAMPLES),
            "total_samples": EXPECTED_RAW_SAMPLES,
            "started_at": data.get("capture_started_at", ""),
            "result": data,
            "error": None,
        }

    raise HTTPException(status_code=404, detail=f"Diagnosis session '{safe_id}' not found.")


@app.post("/api/diagnosis/{diagnosis_id}/cancel")
def cancel_diagnosis(diagnosis_id: str) -> Dict[str, Any]:
    """Cancels an ongoing live diagnosis and restores system state."""
    safe_id = sanitize_identifier(diagnosis_id)

    with _tasks_lock:
        if safe_id not in _active_tasks:
            raise HTTPException(status_code=404, detail=f"Active diagnosis '{safe_id}' not found.")

        task = _active_tasks[safe_id]
        if task["state"] in ("collecting", "analyzing"):
            task["cancel_event"].set()
            task["state"] = "cancelled"
            task["stage"] = "Diagnosis cancelled. System state restored."
            return {
                "status": "cancelled",
                "diagnosis_id": safe_id,
                "message": "Diagnosis cancelled. System state restored.",
            }
        else:
            return {
                "status": task["state"],
                "diagnosis_id": safe_id,
                "message": f"Diagnosis cannot be cancelled because it is in '{task['state']}' state.",
            }


@app.get("/api/history")
def get_history() -> List[Dict[str, Any]]:
    """Returns sorted summary records of all historical diagnoses."""
    pipeline = get_pipeline()
    if not os.path.exists(pipeline.history_dir):
        return []

    summaries = []
    files = [f for f in os.listdir(pipeline.history_dir) if f.endswith("_diagnosis.json")]

    for f in files:
        fpath = os.path.join(pipeline.history_dir, f)
        try:
            with open(fpath, "r", encoding="utf-8") as fp:
                data = json.load(fp)
            sess_id = data.get("session_id", os.path.splitext(f)[0].replace("_diagnosis", ""))
            timestamp = data.get("capture_started_at", "")
            diag = data.get("diagnosis", {})
            abn = data.get("abnormality", {})
            disc = data.get("discovery", {})
            summaries.append({
                "session_id": sess_id,
                "timestamp": timestamp,
                "condition": diag.get("condition", "unknown"),
                "confidence": diag.get("confidence", 0.0),
                "session_abnormal": abn.get("session_abnormal", False),
                "abnormality_score": abn.get("mean_score", 0.0),
                "duration_seconds": data.get("capture_duration_seconds", 30.0),
                "discovery_status": disc.get("discovery_status", "UNKNOWN"),
            })
        except Exception:
            continue

    # Sort descending by timestamp
    summaries.sort(key=lambda x: x["timestamp"], reverse=True)
    return summaries


@app.get("/api/history/{session_id}")
def get_history_detail(session_id: str) -> Dict[str, Any]:
    """Retrieves the complete diagnosis object for a historical session."""
    safe_id = sanitize_identifier(session_id)
    pipeline = get_pipeline()

    target_path = os.path.abspath(os.path.join(pipeline.history_dir, f"{safe_id}_diagnosis.json"))
    base_dir = os.path.abspath(pipeline.history_dir)

    # Path traversal protection
    if not target_path.startswith(base_dir):
        raise HTTPException(status_code=400, detail="Path traversal attempt rejected.")

    if not os.path.exists(target_path):
        raise HTTPException(status_code=404, detail=f"Historical diagnosis '{safe_id}' not found.")

    with open(target_path, "r", encoding="utf-8") as f:
        return json.load(f)


# ----------------------------------------------------------------------
# Optional Frontend Static Files Serving
# ----------------------------------------------------------------------

FRONTEND_DIST = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "frontend", "dist"))
if os.path.exists(FRONTEND_DIST):
    app.mount("/", StaticFiles(directory=FRONTEND_DIST, html=True), name="frontend")
