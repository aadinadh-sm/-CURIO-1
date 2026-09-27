"""Unit and Integration Tests for CURIO FastAPI Server (src/server.py)."""

import json
import os
import shutil
import tempfile
import time
import unittest
from unittest.mock import patch, MagicMock

from fastapi.testclient import TestClient

from src.diagnosis_pipeline import (
    CurioDiagnosisPipeline,
    DEFAULT_MODEL_DIR,
)
from src.server import app, set_pipeline, sanitize_identifier, _active_tasks


class TestCurioServer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def setUp(self):
        self.temp_history_dir = tempfile.mkdtemp()
        self.test_pipeline = CurioDiagnosisPipeline(
            model_dir=DEFAULT_MODEL_DIR,
            history_dir=self.temp_history_dir,
        )
        set_pipeline(self.test_pipeline)
        _active_tasks.clear()

    def tearDown(self):
        shutil.rmtree(self.temp_history_dir, ignore_errors=True)

    def test_health_endpoint(self):
        """Test GET /api/health confirms local-only status."""
        resp = self.client.get("/api/health")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["status"], "ok")
        self.assertTrue(data["local_only"])
        self.assertEqual(data["version"], "1.0.0")

    def test_status_endpoint(self):
        """Test GET /api/status exposes model metadata and threshold."""
        resp = self.client.get("/api/status")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertTrue(data["ready"])
        self.assertTrue(data["model_loaded"])
        self.assertIn("cpu_pressure", data["classes"])
        self.assertAlmostEqual(data["abnormality_threshold"], 0.8536, places=2)
        self.assertEqual(data["features_count"], 12)

    def test_start_and_poll_replay_diagnosis(self):
        """Test starting an instant replay diagnosis and polling for completion."""
        req_payload = {
            "mode": "replay",
            "replay_condition": "cpu_pressure",
            "session_id": "test_api_cpu_replay",
        }
        start_resp = self.client.post("/api/diagnosis/start", json=req_payload)
        self.assertEqual(start_resp.status_code, 202)
        start_data = start_resp.json()
        diag_id = start_data["diagnosis_id"]
        self.assertEqual(diag_id, "test_api_cpu_replay")
        self.assertEqual(start_data["state"], "collecting")

        # Poll status until complete (replay takes < 1 second)
        max_attempts = 30
        completed = False
        final_data = None
        for _ in range(max_attempts):
            poll_resp = self.client.get(f"/api/diagnosis/{diag_id}")
            self.assertEqual(poll_resp.status_code, 200)
            final_data = poll_resp.json()
            if final_data["state"] == "complete":
                completed = True
                break
            time.sleep(0.05)

        self.assertTrue(completed, "Replay diagnosis did not reach complete state.")
        self.assertIsNotNone(final_data["result"])
        self.assertEqual(final_data["result"]["diagnosis"]["condition"], "cpu_pressure")
        self.assertGreater(final_data["result"]["diagnosis"]["confidence"], 0.80)
        self.assertIn("evidence", final_data["result"])
        self.assertIn("discovery", final_data["result"])

    def test_cancel_active_diagnosis(self):
        """Test cancelling an in-progress diagnosis restores system state."""
        # Start a simulated long diagnosis
        with patch.object(self.test_pipeline, "run_diagnosis") as mock_run:
            def slow_run(*args, **kwargs):
                cancel_event = kwargs.get("cancel_event")
                # Wait for cancel event
                for _ in range(50):
                    if cancel_event and cancel_event.is_set():
                        from src.diagnosis_pipeline import DiagnosisCancelledError
                        raise DiagnosisCancelledError("Cancelled")
                    time.sleep(0.05)
                return {}

            mock_run.side_effect = slow_run

            start_resp = self.client.post("/api/diagnosis/start", json={"mode": "live", "session_id": "to_cancel"})
            self.assertEqual(start_resp.status_code, 202)

            # Issue cancellation
            time.sleep(0.1)
            cancel_resp = self.client.post("/api/diagnosis/to_cancel/cancel")
            self.assertEqual(cancel_resp.status_code, 200)
            self.assertEqual(cancel_resp.json()["status"], "cancelled")

            # Check status confirms cancellation
            status_resp = self.client.get("/api/diagnosis/to_cancel")
            self.assertEqual(status_resp.json()["state"], "cancelled")

    def test_history_list_and_detail(self):
        """Test GET /api/history and GET /api/history/{id}."""
        # Create a mock historical diagnosis JSON in test_history_dir
        hist_id = "test_hist_sess_42"
        mock_data = {
            "session_id": hist_id,
            "capture_started_at": "2026-09-28T00:15:00Z",
            "capture_duration_seconds": 30.0,
            "raw_samples_count": 61,
            "feature_windows_count": 11,
            "diagnosis": {
                "condition": "memory_pressure",
                "confidence": 0.9123,
                "class_probabilities": {"normal": 0.05, "memory_pressure": 0.9123},
            },
            "abnormality": {
                "mean_score": 0.95,
                "session_abnormal": True,
            },
            "discovery": {
                "discovery_status": "SUSTAINED_PRESSURE",
            },
        }
        hist_path = os.path.join(self.temp_history_dir, f"{hist_id}_diagnosis.json")
        with open(hist_path, "w", encoding="utf-8") as f:
            json.dump(mock_data, f)

        # GET /api/history
        list_resp = self.client.get("/api/history")
        self.assertEqual(list_resp.status_code, 200)
        items = list_resp.json()
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0]["session_id"], hist_id)
        self.assertEqual(items[0]["condition"], "memory_pressure")
        self.assertTrue(items[0]["session_abnormal"])

        # GET /api/history/{id}
        detail_resp = self.client.get(f"/api/history/{hist_id}")
        self.assertEqual(detail_resp.status_code, 200)
        detail_data = detail_resp.json()
        self.assertEqual(detail_data["session_id"], hist_id)
        self.assertEqual(detail_data["diagnosis"]["condition"], "memory_pressure")

    def test_invalid_identifier_and_path_traversal_protection(self):
        """Test malicious path traversal and illegal character inputs are rejected."""
        # Traversal attempt with dots in session id
        resp1 = self.client.get("/api/history/..passwd")
        self.assertEqual(resp1.status_code, 400)

        # Non-alphanumeric characters
        resp2 = self.client.get("/api/history/invalid!id@session")
        self.assertEqual(resp2.status_code, 400)

        # Non-existent session
        resp3 = self.client.get("/api/history/non_existent_session_999")
        self.assertEqual(resp3.status_code, 404)

    def test_cors_headers(self):
        """Test CORS headers properly permit local frontend origins."""
        headers = {"Origin": "http://localhost:3000"}
        resp = self.client.get("/api/health", headers=headers)
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.headers.get("access-control-allow-origin"), "http://localhost:3000")

    def test_status_missing_artifacts(self):
        """Test GET /api/status when model artifacts are missing."""
        with patch("src.server.get_pipeline", side_effect=FileNotFoundError("Model artifact missing")):
            resp = self.client.get("/api/status")
            self.assertEqual(resp.status_code, 200)
            data = resp.json()
            self.assertFalse(data["ready"])
            self.assertFalse(data["model_loaded"])
            self.assertIn("Model artifact missing", data["error"])

    def test_diagnosis_not_found_and_invalid_id(self):
        """Test GET /api/diagnosis/{id} with missing and invalid IDs."""
        # Non-existent ID
        resp1 = self.client.get("/api/diagnosis/non_existent_id_123")
        self.assertEqual(resp1.status_code, 404)

        # Invalid ID with special chars
        resp2 = self.client.get("/api/diagnosis/invalid$id!bad")
        self.assertEqual(resp2.status_code, 400)


if __name__ == "__main__":
    unittest.main()
