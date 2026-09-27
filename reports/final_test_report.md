# CURIO Final Automated Test Validation Report

**Project:** CURIO — Intelligent Machine Learning Discovery and Computer Diagnosis System  
**Date:** September 28, 2026  
**Scope:** Automated Python Unit Suite, Frontend Vitest Suite, and Production TypeScript Compilation

---

## 1. Test Execution Summary

| Test Suite | Environment | Total Tests | Passed | Failed | Errors | Duration | Status |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Python Backend & Core Suite** | Python 3.14 (unittest) | 130 | 130 | 0 | 0 | 35.83s | **PASSED** |
| **Frontend Component & Integration Suite** | React 19 + Vitest 5.0.2 | 10 | 10 | 0 | 0 | 4.94s | **PASSED** |
| **Production TypeScript Compilation** | Vite 8.3.1 (`tsc -b && vite build`) | — | — | 0 | 0 | 0.27s | **PASSED** |

---

## 2. Python Suite Breakdown (130 Tests)

- **Telemetry Collector (`tests/test_collector.py`)**: 10/10 passed (2 Hz sampling, 61 samples, timestamp monotonicity, process filtering, error resilience).
- **12-Feature Extraction (`tests/test_features.py`)**: 12/12 passed (11 rolling windows, 5s window, 2.5s step, feature schemas, zero NaN/inf).
- **Stress Harness (`tests/test_stress_harness.py`)**: 8/8 passed (synthetic workloads, thread safety, cleanup).
- **Dataset Builder (`tests/test_dataset_builder.py`)**: 10/10 passed (manifest integrity, session serialization, metadata schemas).
- **Heuristic Baseline (`tests/test_baseline_rules.py`)**: 8/8 passed (deterministic rule boundaries, non-statistical confidence).
- **30-Second Trajectory (`tests/test_trajectory_validation.py`)**: 8/8 passed (temporal step alignment, rolling feature invariants).
- **Model Training & LOMO (`tests/test_model_trainer.py`)**: 10/10 passed (cross-validation, GroupKFold, out-of-fold scoring).
- **Dataset Packaging & Import (`tests/test_dataset_packaging.py`)**: 12/12 passed (SHA-256 verification, path traversal rejection, archive safety).
- **Physical Dataset Validation (`tests/test_physical_dataset.py`)**: 12/12 passed (Machine A schema compliance, rule verification A through P).
- **Evidence Engine (`tests/test_evidence.py`)**: 12/12 passed (z-scores, directional scoring, centroid distances, non-causal language).
- **Discovery Engine (`tests/test_discovery.py`)**: 12/12 passed (onset detection, Kendall tau-b, SUSTAINED_PRESSURE and INSUFFICIENT_TEMPORAL_EVIDENCE states).
- **Live Diagnosis Pipeline (`tests/test_diagnosis_pipeline.py`)**: 7/7 passed (e2e live/replay execution, cancellation signals, progress callbacks).
- **FastAPI Server (`tests/test_server.py`)**: 9/9 passed (local binding, /api/health, /api/status, replay start/poll, cancellation, history, path traversal sanitization, missing artifacts handling).

---

## 3. Frontend Vitest Breakdown (10 Tests)

1. `renders home screen with desktop layout, branding, and local-only badge` — **PASSED**
2. `tracks live diagnosis progress and handles real stage transitions` — **PASSED**
3. `renders Normal result with Operating Equilibrium state` — **PASSED**
4. `renders Abnormal result with separated confidence, evidence cards, and discovery timeline` — **PASSED**
5. `renders SUSTAINED_PRESSURE fallback state with clear explanatory message` — **PASSED**
6. `renders INSUFFICIENT_TEMPORAL_EVIDENCE fallback state without fake tau` — **PASSED**
7. `renders diagnosis history and loads selected session details` — **PASSED**
8. `opens Technical Diagnostics modal and exposes all 12 features and mathematical metrics` — **PASSED**
9. `handles live diagnosis cancellation safely and restores system state` — **PASSED**
10. `handles Replay Mode end-to-end integration and renders evidence and discovery` — **PASSED**

---

## 4. Production Build Verification

- **Command**: `npm run build` (`tsc -b && vite build`)
- **Modules Transformed**: 24
- **Output Artifacts**:
  - `dist/index.html` (0.73 kB)
  - `dist/assets/index-Dr4iYOlL.css` (10.06 kB)
  - `dist/assets/index-BFZphWn5.js` (259.24 kB)
- **TypeScript Errors**: 0
- **Lint Errors**: 0
- **Status**: Production bundle ready and served by local backend at `http://127.0.0.1:8000`.
