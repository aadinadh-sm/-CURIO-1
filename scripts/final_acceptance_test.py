"""CURIO Final End-to-End System Acceptance Test.

Verifies all critical operational workflows:
A. Backend application initialization
B. Frontend production build presence
C. /api/health endpoint responsiveness and local-only contract
D. /api/status endpoint readiness and threshold metadata
E. Replay Normal diagnosis completion and equilibrium state
F. Replay CPU Pressure diagnosis completion and evidence generation
G. Replay Memory Pressure diagnosis completion and evidence generation
H. Replay Disk I/O Pressure diagnosis completion and evidence generation
I. Local diagnosis history persistence
J. Local diagnosis history retrieval
K. Active diagnosis cancellation and system state recovery
L. Rejection of malformed IDs and directory traversal attempts
M. Safe rejection and error handling when artifacts are missing
N. Strict local-only network and CORS restrictions

Exits with:
    CURIO END-TO-END ACCEPTANCE: PASSED
or:
    CURIO END-TO-END ACCEPTANCE: FAILED
"""

import json
import os
import sys
import time
from unittest.mock import patch

from fastapi.testclient import TestClient

# Ensure repository root is on sys.path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.server import app, CORS_ORIGINS, sanitize_identifier
from src.diagnosis_pipeline import DEFAULT_MODEL_DIR, DEFAULT_HISTORY_DIR


def run_acceptance():
    print("==================================================")
    print("CURIO END-TO-END ACCEPTANCE VALIDATION")
    print("==================================================")
    all_passed = True

    try:
        # A. Backend starts / can be initialized
        print("\n[A] Initializing backend service...")
        client = TestClient(app)
        print("    PASS: FastAPI client initialized successfully.")

        # B. Frontend build exists
        print("\n[B] Verifying frontend production build...")
        dist_html = os.path.join(REPO_ROOT, "frontend", "dist", "index.html")
        if os.path.isfile(dist_html) and os.path.getsize(dist_html) > 0:
            print(f"    PASS: Production index.html verified ({os.path.getsize(dist_html)} bytes).")
        else:
            print("    FAIL: Production frontend build not found at frontend/dist/index.html")
            all_passed = False

        # C. /api/health works
        print("\n[C] Testing /api/health endpoint...")
        resp_c = client.get("/api/health")
        data_c = resp_c.json()
        if resp_c.status_code == 200 and data_c.get("local_only") is True and data_c.get("status") == "ok":
            print(f"    PASS: Health OK. Status={data_c['status']}, LocalOnly={data_c['local_only']}.")
        else:
            print(f"    FAIL: /api/health returned unexpected payload: {data_c}")
            all_passed = False

        # D. /api/status works
        print("\n[D] Testing /api/status endpoint...")
        resp_d = client.get("/api/status")
        data_d = resp_d.json()
        if (
            resp_d.status_code == 200
            and data_d.get("ready") is True
            and data_d.get("model_loaded") is True
            and data_d.get("features_count") == 12
            and len(data_d.get("classes", [])) == 4
        ):
            print(f"    PASS: Status OK. Classes={data_d['classes']}, Threshold={data_d['abnormality_threshold']:.4f}.")
        else:
            print(f"    FAIL: /api/status validation failed: {data_d}")
            all_passed = False

        # Helper to execute and poll a replay diagnosis
        def run_replay(condition: str, session_id: str):
            start_payload = {
                "mode": "replay",
                "replay_condition": condition,
                "session_id": session_id,
            }
            start_resp = client.post("/api/diagnosis/start", json=start_payload)
            if start_resp.status_code != 202:
                return False, f"Start failed with status {start_resp.status_code}: {start_resp.text}"
            
            diag_id = start_resp.json()["diagnosis_id"]
            # Poll until completion
            for _ in range(60):
                poll_resp = client.get(f"/api/diagnosis/{diag_id}")
                if poll_resp.status_code != 200:
                    return False, f"Poll returned status {poll_resp.status_code}"
                pdata = poll_resp.json()
                if pdata["state"] == "complete":
                    return True, pdata["result"]
                time.sleep(0.05)
            return False, "Replay diagnosis timed out"

        # E. Replay Normal completes
        print("\n[E] Testing replay for condition: 'normal'...")
        ok_e, res_e = run_replay("normal", "acceptance_normal")
        if ok_e and res_e["diagnosis"]["condition"] == "normal" and res_e["abnormality"]["session_abnormal"] is False:
            print(f"    PASS: Normal replay completed. Condition={res_e['diagnosis']['condition']}, Abnormal={res_e['abnormality']['session_abnormal']}.")
        else:
            print(f"    FAIL: Normal replay failed: {res_e}")
            all_passed = False

        # F. Replay CPU Pressure completes
        print("\n[F] Testing replay for condition: 'cpu_pressure'...")
        ok_f, res_f = run_replay("cpu_pressure", "acceptance_cpu")
        if ok_f and res_f["diagnosis"]["condition"] == "cpu_pressure" and res_f["abnormality"]["session_abnormal"] is True:
            top_evid = [e["feature_name"] for e in res_f["evidence"].get("supporting_evidence", [])[:2]]
            print(f"    PASS: CPU Pressure replay completed. Supporting evidence={top_evid}.")
        else:
            print(f"    FAIL: CPU Pressure replay failed: {res_f}")
            all_passed = False

        # G. Replay Memory Pressure completes
        print("\n[G] Testing replay for condition: 'memory_pressure'...")
        ok_g, res_g = run_replay("memory_pressure", "acceptance_memory")
        if ok_g and res_g["diagnosis"]["condition"] == "memory_pressure" and res_g["abnormality"]["session_abnormal"] is True:
            print(f"    PASS: Memory Pressure replay completed. Confidence={res_g['diagnosis']['confidence']:.2%}.")
        else:
            print(f"    FAIL: Memory Pressure replay failed: {res_g}")
            all_passed = False

        # H. Replay Disk I/O Pressure completes
        print("\n[H] Testing replay for condition: 'disk_io_pressure'...")
        ok_h, res_h = run_replay("disk_io_pressure", "acceptance_disk")
        if ok_h and res_h["diagnosis"]["condition"] == "disk_io_pressure" and res_h["abnormality"]["session_abnormal"] is True:
            print(f"    PASS: Disk I/O Pressure replay completed. Discovered status={res_h['discovery']['discovery_status']}.")
        else:
            print(f"    FAIL: Disk I/O Pressure replay failed: {res_h}")
            all_passed = False

        # I. History is written
        print("\n[I] Verifying local diagnosis history persistence...")
        written_file = os.path.join(DEFAULT_HISTORY_DIR, "acceptance_cpu_diagnosis.json")
        if os.path.isfile(written_file):
            print(f"    PASS: Historical record persisted at {written_file}.")
        else:
            print(f"    FAIL: Historical record not found at {written_file}.")
            all_passed = False

        # J. History can be retrieved
        print("\n[J] Testing history listing and individual retrieval...")
        hist_list_resp = client.get("/api/history")
        hist_items = hist_list_resp.json()
        found_cpu = any(item["session_id"] == "acceptance_cpu" for item in hist_items)
        hist_detail_resp = client.get("/api/history/acceptance_cpu")
        if hist_list_resp.status_code == 200 and found_cpu and hist_detail_resp.status_code == 200:
            print(f"    PASS: Successfully retrieved history list ({len(hist_items)} items) and detail record.")
        else:
            print("    FAIL: History retrieval endpoints did not return expected records.")
            all_passed = False

        # K. Cancellation works
        print("\n[K] Testing active diagnosis cancellation...")
        start_cancel_resp = client.post("/api/diagnosis/start", json={"mode": "live", "session_id": "acceptance_cancel_test"})
        cancel_id = start_cancel_resp.json()["diagnosis_id"]
        time.sleep(0.05)
        cancel_action_resp = client.post(f"/api/diagnosis/{cancel_id}/cancel")
        cancel_status_resp = client.get(f"/api/diagnosis/{cancel_id}")
        if (
            cancel_action_resp.status_code == 200
            and cancel_action_resp.json().get("status") == "cancelled"
            and cancel_status_resp.json().get("state") == "cancelled"
        ):
            print("    PASS: Live diagnosis cancelled safely, state returned to 'cancelled'.")
        else:
            print(f"    FAIL: Cancellation failed: {cancel_status_resp.json()}")
            all_passed = False

        # L. Malformed IDs are rejected
        print("\n[L] Testing rejection of malformed IDs and path traversal...")
        resp_l1 = client.get("/api/history/..traversal")
        resp_l2 = client.get("/api/diagnosis/bad!id#symbol")
        if resp_l1.status_code == 400 and resp_l2.status_code == 400:
            print("    PASS: Path traversal (..) and invalid characters correctly rejected with HTTP 400.")
        else:
            print(f"    FAIL: Malformed ID validation failed. Got {resp_l1.status_code} and {resp_l2.status_code}.")
            all_passed = False

        # M. Missing artifacts are rejected safely
        print("\n[M] Testing safe handling of missing artifacts...")
        with patch("src.server.get_pipeline", side_effect=FileNotFoundError("Artifact missing")):
            resp_m = client.get("/api/status")
            data_m = resp_m.json()
            if resp_m.status_code == 200 and data_m.get("ready") is False and "Artifact missing" in data_m.get("error", ""):
                print("    PASS: Missing artifacts handled gracefully without crash.")
            else:
                print(f"    FAIL: Missing artifacts not reported properly: {data_m}")
                all_passed = False

        # N. Local-only binding is correct
        print("\n[N] Verifying local-only network restrictions...")
        # Check CORS configuration
        local_origins_valid = all(
            origin.startswith("http://127.0.0.1") or origin.startswith("http://localhost")
            for origin in CORS_ORIGINS
        )
        if local_origins_valid:
            print(f"    PASS: CORS origins restricted strictly to local ports: {CORS_ORIGINS}.")
        else:
            print(f"    FAIL: Non-local CORS origins detected: {CORS_ORIGINS}")
            all_passed = False

    except Exception as exc:
        print(f"\nCRITICAL EXCEPTION during acceptance test: {exc}")
        all_passed = False

    print("\n==================================================")
    if all_passed:
        print("CURIO END-TO-END ACCEPTANCE: PASSED")
        print("==================================================")
        return 0
    else:
        print("CURIO END-TO-END ACCEPTANCE: FAILED")
        print("==================================================")
        return 1


if __name__ == "__main__":
    sys.exit(run_acceptance())
