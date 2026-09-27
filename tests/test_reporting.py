"""Unit tests for CURIO Reporting Module (src/reporting.py)."""

import unittest

from src.reporting import render_diagnosis_report, render_performance_summary


class TestReportingModule(unittest.TestCase):
    def setUp(self):
        self.mock_normal_result = {
            "session_id": "test_sess_normal_001",
            "capture_started_at": "2026-09-28T00:00:00Z",
            "capture_duration_seconds": 30.0,
            "raw_samples_count": 61,
            "feature_windows_count": 11,
            "diagnosis": {
                "condition": "normal",
                "confidence": 0.9543,
                "class_probabilities": {
                    "normal": 0.9543,
                    "cpu_pressure": 0.0150,
                    "memory_pressure": 0.0150,
                    "disk_io_pressure": 0.0157,
                },
                "per_window_predictions": [
                    {"window_idx": i, "predicted_condition": "normal", "probability_normal": 0.95, "abnormality_score": 0.05}
                    for i in range(11)
                ],
            },
            "abnormality": {
                "mean_score": 0.0457,
                "max_score": 0.0510,
                "final_window_score": 0.0480,
                "abnormal_window_count": 0,
                "abnormal_window_ratio": 0.0,
                "threshold": 0.8536,
                "session_abnormal": False,
                "per_window_scores": [0.045] * 11,
            },
            "evidence": {
                "condition": "normal",
                "confidence": 0.9543,
                "abnormality_score": 0.0457,
                "supporting_evidence": [],
                "contradictory_evidence": [],
                "neutral_features": [],
                "overall_interpretation": "All observed telemetry signals remain consistent with the learned Normal operating reference.",
                "representative_window_idx": 0,
            },
            "discovery": {
                "predicted_condition": "normal",
                "discovery_status": "OPERATING_EQUILIBRIUM",
                "observed_sequence": [],
                "canonical_sequence": [],
                "kendall_tau": None,
                "normalized_tau": None,
                "discovery_coverage": 0.0,
                "interpretation": "Operating Equilibrium: No escalation detected. Telemetry remained within learned Normal bounds.",
            },
            "performance": {
                "capture_duration_seconds": 30.0,
                "feature_extraction_ms": 10.5,
                "inference_ms": 12.0,
                "evidence_ms": 2.5,
                "discovery_ms": 1.5,
                "reporting_ms": 0.3,
                "total_analysis_ms": 26.5,
            },
        }

        self.mock_abnormal_result = {
            "session_id": "test_sess_cpu_001",
            "capture_started_at": "2026-09-28T00:00:00Z",
            "capture_duration_seconds": 30.0,
            "raw_samples_count": 61,
            "feature_windows_count": 11,
            "diagnosis": {
                "condition": "cpu_pressure",
                "confidence": 0.9610,
                "class_probabilities": {
                    "normal": 0.0120,
                    "cpu_pressure": 0.9610,
                    "memory_pressure": 0.0128,
                    "disk_io_pressure": 0.0142,
                },
                "per_window_predictions": [
                    {"window_idx": i, "predicted_condition": "cpu_pressure", "probability_normal": 0.012, "abnormality_score": 0.988}
                    for i in range(11)
                ],
            },
            "abnormality": {
                "mean_score": 0.9880,
                "max_score": 0.9880,
                "final_window_score": 0.9880,
                "abnormal_window_count": 11,
                "abnormal_window_ratio": 1.0,
                "threshold": 0.8536,
                "session_abnormal": True,
                "per_window_scores": [0.988] * 11,
            },
            "evidence": {
                "condition": "cpu_pressure",
                "confidence": 0.9610,
                "abnormality_score": 0.9880,
                "supporting_evidence": [
                    {
                        "feature_name": "cpu_mean",
                        "human_readable_statement": "Average CPU Utilization (100.0%) was substantially elevated relative to the learned Normal reference (18.6%).",
                        "directional_score": 10.0,
                        "evidence_strength": "strong",
                    },
                    {
                        "feature_name": "cpu_max",
                        "human_readable_statement": "Peak Core CPU Utilization (100.0%) was substantially elevated relative to the learned Normal reference (31.8%).",
                        "directional_score": 5.79,
                        "evidence_strength": "strong",
                    },
                ],
                "contradictory_evidence": [],
                "neutral_features": [],
                "overall_interpretation": "Multiple CPU-related telemetry signals moved in the expected direction for CPU Pressure.",
                "representative_window_idx": 3,
                "all_evaluated_features": [
                    {
                        "feature_name": "cpu_mean",
                        "observed_value": 100.0,
                        "normal_reference": 18.6,
                        "directional_score": 10.0,
                        "evidence_strength": "strong",
                    },
                    {
                        "feature_name": "cpu_max",
                        "observed_value": 100.0,
                        "normal_reference": 31.8,
                        "directional_score": 5.79,
                        "evidence_strength": "strong",
                    },
                ],
            },
            "discovery": {
                "predicted_condition": "cpu_pressure",
                "discovery_status": "SUSTAINED_PRESSURE",
                "observed_sequence": ["cpu_max", "cpu_mean"],
                "canonical_sequence": ["cpu_max", "cpu_mean", "top_proc_cpu_ratio"],
                "onset_events": [
                    {"feature_name": "cpu_max", "onset_time_seconds": 0.0},
                    {"feature_name": "cpu_mean", "onset_time_seconds": 0.0},
                ],
                "kendall_tau": None,
                "normalized_tau": None,
                "discovery_coverage": 0.667,
                "missing_features": ["top_proc_cpu_ratio"],
                "unexpected_features": [],
                "interpretation": "Sustained operating pressure detected; no transition observed during capture.",
            },
            "performance": {
                "capture_duration_seconds": 30.0,
                "feature_extraction_ms": 11.2,
                "inference_ms": 115.0,
                "evidence_ms": 2.5,
                "discovery_ms": 1.8,
                "reporting_ms": 0.4,
                "total_analysis_ms": 130.5,
            },
        }

    def test_render_normal_standard_report(self):
        report = render_diagnosis_report(self.mock_normal_result, technical=False)
        self.assertIn("CURIO DIAGNOSIS", report)
        self.assertIn("Status:\nNORMAL", report)
        self.assertIn("Condition:\nNormal Operation", report)
        self.assertIn("Model confidence:\n95%", report)
        self.assertIn("Operating Equilibrium:\nNo escalation detected.", report)
        self.assertNotIn("TECHNICAL DIAGNOSTIC DETAILS", report)

    def test_render_abnormal_standard_report(self):
        report = render_diagnosis_report(self.mock_abnormal_result, technical=False)
        self.assertIn("Status:\nABNORMAL", report)
        self.assertIn("Condition:\nCpu Pressure", report)
        self.assertIn("Model confidence:\n96%", report)
        self.assertIn("WHY CURIO THINKS THIS", report)
        self.assertIn("Average CPU Utilization (100.0%) was substantially elevated", report)
        self.assertIn("CURIO DISCOVERY", report)
        self.assertIn("cpu_max (t = 0.0s)", report)
        self.assertIn("Discovery coverage:\n67%", report)
        self.assertNotIn("TECHNICAL DIAGNOSTIC DETAILS", report)

    def test_render_technical_mode(self):
        report = render_diagnosis_report(self.mock_abnormal_result, technical=True)
        self.assertIn("CURIO TECHNICAL DIAGNOSTIC DETAILS", report)
        self.assertIn("Session ID:         test_sess_cpu_001", report)
        self.assertIn("Class Probabilities (Mean across 11 windows):", report)
        self.assertIn("11-Window Diagnostic Trajectory:", report)
        self.assertIn("Evidence Feature Deviations (Window 3):", report)
        self.assertIn("PERFORMANCE SUMMARY", report)

    def test_performance_summary_rendering(self):
        summary = render_performance_summary(self.mock_abnormal_result["performance"])
        self.assertIn("PERFORMANCE SUMMARY", summary)
        self.assertIn("Telemetry collection: 30.0s", summary)
        self.assertIn("Feature extraction:   11.2 ms", summary)
        self.assertIn("ML inference:         115.0 ms", summary)
        self.assertIn("Total analysis time:  130.5 ms", summary)

    def test_non_causal_language_guardrail_in_reporting(self):
        # Corrupt interpretation with causal keyword
        corrupt_result = dict(self.mock_abnormal_result)
        corrupt_result["discovery"] = dict(corrupt_result["discovery"])
        corrupt_result["discovery"]["interpretation"] = "High CPU load caused by the top process."

        with self.assertRaises(ValueError):
            render_diagnosis_report(corrupt_result)


if __name__ == "__main__":
    unittest.main()
