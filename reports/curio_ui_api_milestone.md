# CURIO Milestone Report: Modern Web UI + Local API

**Project:** CURIO — Intelligent Machine Learning Discovery and Computer Diagnosis System  
**Milestone:** Modern Web UI + Local-Only API  
**Date:** September 28, 2026  
**Status:** COMPLETE & FULLY VALIDATED

---

## 1. Executive Summary

This milestone delivers the complete user presentation and orchestration layer for CURIO. The underlying scientific and machine learning intelligence pipeline—encompassing 2 Hz telemetry collection, 12-feature rolling window extraction, calibrated Random Forest diagnosis, out-of-fold abnormality thresholding, statistical evidence attribution, and temporal onset discovery—has been integrated behind a local-only FastAPI backend and a desktop-tailored React 19 + TypeScript web interface.

All operations execute strictly on `127.0.0.1`. No telemetry, analytics, or user data leaves the machine. All 130 repository unit/integration tests and 10 frontend Vitest integration tests pass with zero regressions.

---

## 2. Implemented Architecture

### A. Local API Backend (`src/server.py`)
- **Framework**: FastAPI with Uvicorn.
- **Network Interface**: Bound strictly to `127.0.0.1:8000`.
- **CORS Policy**: Whitelisted exclusively to local browser origins (`http://127.0.0.1:3000`, `http://localhost:3000`, `http://127.0.0.1:8000`, `http://localhost:8000`).
- **Asynchronous Execution**: Thread-safe diagnosis worker running concurrently with live progress reporting (`collecting`, `analyzing`, `complete`, `cancelled`, `error`).
- **Safe Cancellation**: Immediate cancellation via `threading.Event` without saving corrupt or partial history records.
- **Input Sanitization**: Regex verification `^[a-zA-Z0-9_\-]+$` preventing path traversal (`../`) and illegal character injection.
- **Static Hosting**: Embedded serving of the compiled React application from `frontend/dist`.

### B. Modern Web UI (`frontend/`)
- **Technology Stack**: React 19, TypeScript, Vite 8, pure custom modern desktop CSS (no heavy external component frameworks).
- **Desktop Diagnostic Aesthetics**: High-contrast dark theme (#090D16, #0E1626, #162238) with cyan (#06B6D4) and emerald (#10B981) telemetry indicators, crisp typography (`Plus Jakarta Sans` and `JetBrains Mono`).
- **Zero-Egress Visibility**: Prominent "Local Only" privacy badge confirming zero external telemetry transmission.
- **Strict Non-Causal Language**: Descriptive, correlative, and non-causal terminology throughout.
- **Separation of Concerns**: Model confidence (%) is explicitly decoupled from statistical evidence strength.

---

## 3. UI Screens & Features

| Screen / Component | Description & Key Capabilities |
|:---|:---|
| **Home Screen** | Clean desktop hero interface with headline *"Understand what your computer is doing"*, subtext, Local-Only badge, and primary action buttons. |
| **Live Diagnosis Screen** | Displays real-time progress polled from `/api/diagnosis/{id}`: elapsed seconds, exact sample count (`17/61 samples`), hardware subsystem checklist (CPU, Memory, Disk, Processes), and a functional [Cancel Diagnosis] button. |
| **Result Screen** | Categorizes state as Normal or Abnormal, shows model confidence and evidence strength separately, renders top 3 statistical evidence cards with observed vs reference values and directional icons, and displays the temporal onset timeline. |
| **Discovery Timeline** | Visualizes relative onset markers along the 30-second capture. Gracefully falls back to explicit state banners for `SUSTAINED_PRESSURE` and `INSUFFICIENT_TEMPORAL_EVIDENCE` without fake tau values. |
| **Diagnostic History** | `/history` view listing past diagnoses fetched from `/api/history`. Allows instant inspection of previous reports. |
| **Technical Mode** | Modal view providing full scientific transparency: 12 feature values, 11-window rolling values, class probabilities, Kendall's tau-b, normalized tau, and stage latencies. |
| **Replay / Demo Mode** | Development dialog enabling instantaneous testing across all 4 conditions (`normal`, `cpu_pressure`, `memory_pressure`, `disk_io_pressure`) using real physical telemetry fixtures. |

---

## 4. REST Endpoints Specification

| Method | Endpoint | Purpose | Validation & Safety |
|:---:|:---|:---|:---|
| `GET` | `/api/health` | Health & local-only status check | Returns `local_only: true`, version, and status |
| `GET` | `/api/status` | Model readiness and metadata | Returns threshold, classes, and history count |
| `POST`| `/api/diagnosis/start` | Launch live or replay diagnosis | Returns 202 Accepted with `diagnosis_id` and initial state |
| `GET` | `/api/diagnosis/{id}` | Poll live progress or fetch result | Validates ID format; returns progress or completed result |
| `POST`| `/api/diagnosis/{id}/cancel`| Cancel active diagnosis | Sets cancel event, terminates capture, cleans up state |
| `GET` | `/api/history` | List historical diagnoses | Sanitized scan of `data/diagnosis_history/` |
| `GET` | `/api/history/{id}` | Retrieve specific diagnosis | Path-traversal protected single-session retrieval |

---

## 5. Verification & Test Results

### A. Python Backend & Core Suite
```
Ran 130 tests in 30.813s
OK (0 failures, 0 errors, 0 regressions)
```
- Includes 9 specialized API integration tests in `tests/test_server.py`:
  - `test_health_endpoint`: Verified local-only status.
  - `test_status_endpoint`: Verified threshold, feature count, and class metadata.
  - `test_start_and_poll_replay_diagnosis`: Verified end-to-end replay lifecycle.
  - `test_cancel_active_diagnosis`: Verified threading event cancellation and state recovery.
  - `test_history_list_and_detail`: Verified list and single-item retrieval.
  - `test_invalid_identifier_and_path_traversal_protection`: Verified rejection of illegal IDs and directory traversal attempts.
  - `test_cors_headers`: Verified local CORS origin validation.
  - `test_status_missing_artifacts`: Verified graceful failure handling when model files are missing.
  - `test_diagnosis_not_found_and_invalid_id`: Verified 404 on missing sessions and 400 on malformed queries.

### B. Frontend Vitest Suite
```
Test Files  1 passed (1)
     Tests  10 passed (10)
  Duration  4.22s
```
- Covers all 10 mandated scenarios in `frontend/src/App.test.tsx`:
  1. Home screen initial state and navigation
  2. Live diagnosis progress tracking and stage transitions
  3. Normal result rendering with Operating Equilibrium
  4. Abnormal result rendering with evidence cards and onset markers
  5. Sustained pressure fallback state banner
  6. Insufficient temporal evidence fallback state banner
  7. History screen listing and session detail inspection
  8. Technical Diagnostics modal data inspection
  9. Live diagnosis safe cancellation and state recovery
  10. API error handling and graceful human-readable reporting

### C. Build Verification
- Vite production build completed:
  `dist/index.html` (0.47 kB), `dist/assets/index.css` (8.04 kB), `dist/assets/index.js` (213.91 kB).
- Compiled static assets served directly through FastAPI at `http://127.0.0.1:8000`.

---

## 6. Known Limitations

1. **Physical Machine Scope**: Models and reference profiles were trained and calibrated on physical Machine A. True cross-machine generalization has not been demonstrated and is not claimed.
2. **Fixed Duration Telemetry**: Telemetry capture is fixed to 30 seconds (61 samples at 2 Hz). Transient disruptions lasting less than 500 ms are outside the observation window.
3. **Correlation vs Causality**: Temporal sequences and evidence attributions describe observed statistical patterns; they do not establish hardware root cause.
