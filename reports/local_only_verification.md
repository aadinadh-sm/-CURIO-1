# CURIO Local-Only Security & Privacy Verification Report

**Project:** CURIO — Intelligent Machine Learning Discovery and Computer Diagnosis System  
**Date:** September 28, 2026  
**Auditor:** Antigravity Autonomous Pair Programmer  
**Status:** 100% VERIFIED LOCAL-ONLY (ZERO EGRESS)

---

## 1. Network Boundary Audit

| Checkpoint | Implementation File | Verification Mechanism | Status |
|:---|:---|:---|:---:|
| **Server Interface Binding** | `src/server.py`, `scripts/run_curio.py` | FastAPI binds strictly to `127.0.0.1`. Attempts to bind to `0.0.0.0` or external network adapters are rejected with code 1. | **VERIFIED** |
| **CORS Origins Whitelist** | `src/server.py` | Restricted to `http://127.0.0.1:3000`, `http://localhost:3000`, `http://127.0.0.1:8000`, `http://localhost:8000`. | **VERIFIED** |
| **Frontend Communication** | `frontend/src/api.ts` | All API calls target relative paths `/api/*` on `127.0.0.1`. No external URLs configured. | **VERIFIED** |
| **Outbound Telemetry Upload** | Entire codebase | Zero code paths make HTTP requests to external analytics, crash reporting, or telemetry aggregation endpoints. | **VERIFIED** |
| **Cloud Services Dependency** | Requirements & architecture | Zero dependencies on AWS, Azure, GCP, Firebase, or external databases. | **VERIFIED** |
| **External LLM / AI APIs** | Entire codebase | Zero imports or calls to OpenAI, Anthropic, Google Gemini, Ollama, or third-party AI endpoints. All intelligence runs via local Random Forest + statistical formulas. | **VERIFIED** |

---

## 2. Ingress & Injection Protection

1. **Path Traversal Shield**:
   - `sanitize_identifier(identifier: str)` strictly enforces `^[a-zA-Z0-9_\-]+$`.
   - Any URI parameter containing `..`, `/`, `\`, `%2e%2e`, or control characters triggers an immediate `HTTP 400 Bad Request`.
2. **Filesystem Isolation**:
   - History retrieval operates only within `data/diagnosis_history/`.
   - Replay file retrieval operates only within `data/physical_raw/`.
   - Static asset serving operates only within `frontend/dist/`.

---

## 3. Data Storage Locality

- All raw samples, feature tensors, and diagnostic JSON files remain entirely on the host filesystem:
  - Models: `data/models/physical_deployment/`
  - Historical records: `data/diagnosis_history/`
  - Temporary memory: Task tracking resides only in volatile process RAM and is cleared on server shutdown.
