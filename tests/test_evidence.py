"""Unit tests for CURIO Evidence Engine.

Strictly validates:
1. Normal vector produces mostly neutral evidence.
2. Clear CPU vector produces CPU-supporting evidence.
3. Clear Memory vector produces Memory-supporting evidence.
4. Clear Disk vector produces Disk-supporting evidence.
5. Contradictory feature gets lower/negative directional support.
6. Missing feature is handled safely.
7. NaN is rejected or safely excluded according to documented policy.
8. Inf is rejected or safely excluded.
9. Zero standard deviation does not cause division by zero.
10. Evidence is different when the predicted class changes.
11. Probability is not confused with evidence strength.
12. Evidence reference statistics are training-derived.
13. Test/held-out statistics cannot enter the reference artifact.
14. Causal language is not generated.
15. Output schema is stable.
16. Top evidence contains only valid known features.
"""

import math
import unittest
import numpy as np
import pandas as pd

from src.evidence import (
    CONTRADICTORY_THRESHOLD,
    FORBIDDEN_CAUSAL_PATTERNS,
    MODERATE_EVIDENCE_THRESHOLD,
    SAFE_EPSILON,
    SCORE_CAP,
    STRONG_EVIDENCE_THRESHOLD,
    VALID_CONDITIONS,
    CausalLanguageError,
    EvidenceReferenceProfile,
    InvalidTelemetryError,
    build_evidence_statement,
    generate_evidence,
    validate_no_causal_language,
)
from src.features import FEATURE_NAMES


class TestEvidenceEngine(unittest.TestCase):
    """Comprehensive test suite for CURIO Evidence Engine."""

    def setUp(self):
        """Builds a standardized synthetic training reference profile for controlled testing."""
        # Standard synthetic Normal training baseline
        normal_stats = {
            "cpu_mean": {"mean": 15.0, "std": 5.0, "median": 14.5, "iqr": 6.0},
            "cpu_max": {"mean": 25.0, "std": 8.0, "median": 24.0, "iqr": 9.0},
            "cpu_std": {"mean": 3.0, "std": 1.5, "median": 2.8, "iqr": 1.2},
            "cpu_core_imbalance": {"mean": 5.0, "std": 2.0, "median": 4.8, "iqr": 2.2},
            "ram_used_pct": {"mean": 50.0, "std": 4.0, "median": 50.0, "iqr": 5.0},
            "ram_available_ratio": {"mean": 0.50, "std": 0.04, "median": 0.50, "iqr": 0.05},
            "swap_used_pct": {"mean": 10.0, "std": 2.0, "median": 10.0, "iqr": 2.0},
            "disk_io_rate_norm": {"mean": 2.0, "std": 0.5, "median": 1.9, "iqr": 0.6},
            "disk_iops_norm": {"mean": 1.0, "std": 0.3, "median": 1.0, "iqr": 0.4},
            "process_count_delta": {"mean": 0.0, "std": 2.0, "median": 0.0, "iqr": 2.0},
            "top_proc_cpu_ratio": {"mean": 0.20, "std": 0.05, "median": 0.19, "iqr": 0.06},
            "top_proc_mem_pct": {"mean": 5.0, "std": 1.0, "median": 5.0, "iqr": 1.2},
        }

        # Expected directions per class
        expected_dirs = {
            "normal": {f: 0 for f in FEATURE_NAMES},
            "cpu_pressure": {
                "cpu_mean": 1,
                "cpu_max": 1,
                "cpu_std": 1,
                "cpu_core_imbalance": 1,
                "top_proc_cpu_ratio": 1,
                "process_count_delta": 0,
                "ram_used_pct": 0,
                "ram_available_ratio": 0,
                "swap_used_pct": 0,
                "disk_io_rate_norm": 0,
                "disk_iops_norm": 0,
                "top_proc_mem_pct": 0,
            },
            "memory_pressure": {
                "ram_used_pct": 1,
                "ram_available_ratio": -1,  # decreases under memory pressure
                "swap_used_pct": 1,
                "top_proc_mem_pct": 1,
                "cpu_mean": 0,
                "cpu_max": 0,
                "cpu_std": 0,
                "cpu_core_imbalance": 0,
                "top_proc_cpu_ratio": 0,
                "process_count_delta": 0,
                "disk_io_rate_norm": 0,
                "disk_iops_norm": 0,
            },
            "disk_io_pressure": {
                "disk_io_rate_norm": 1,
                "disk_iops_norm": 1,
                "cpu_mean": 0,
                "cpu_max": 0,
                "cpu_std": 0,
                "cpu_core_imbalance": 0,
                "top_proc_cpu_ratio": 0,
                "process_count_delta": 0,
                "ram_used_pct": 0,
                "ram_available_ratio": 0,
                "swap_used_pct": 0,
                "top_proc_mem_pct": 0,
            },
        }

        self.ref_profile = EvidenceReferenceProfile(
            schema_version="1.0.0",
            training_machines=["test_train_machine_1"],
            normal_sessions_count=4,
            total_training_samples=44,
            feature_stats=normal_stats,
            expected_directions=expected_dirs,
        )

        self.normal_vector = {f: normal_stats[f]["mean"] for f in FEATURE_NAMES}

    # 1. Normal vector produces mostly neutral evidence
    def test_normal_vector_neutral_evidence(self):
        probs = {"normal": 0.94, "cpu_pressure": 0.02, "memory_pressure": 0.02, "disk_io_pressure": 0.02}
        res = generate_evidence(
            predicted_condition="normal",
            calibrated_probabilities=probs,
            current_features=self.normal_vector,
            reference_profile=self.ref_profile,
        )

        self.assertEqual(res["condition"], "normal")
        self.assertEqual(res["confidence"], 0.94)
        self.assertEqual(len(res["contradictory_evidence"]), 0)
        self.assertGreater(res["evidence_consistency"], 0.9)
        self.assertIn("learned Normal operating reference", res["overall_interpretation"])
        for item in res["supporting_evidence"]:
            self.assertLess(abs(item["raw_z"]), 1.0)

    # 2. Clear CPU vector produces CPU-supporting evidence
    def test_clear_cpu_vector_supporting_evidence(self):
        cpu_feats = dict(self.normal_vector)
        cpu_feats["cpu_mean"] = 85.0  # +14 std
        cpu_feats["cpu_max"] = 98.0   # +9 std
        cpu_feats["cpu_core_imbalance"] = 25.0  # +10 std

        probs = {"normal": 0.01, "cpu_pressure": 0.96, "memory_pressure": 0.02, "disk_io_pressure": 0.01}
        res = generate_evidence(
            predicted_condition="cpu_pressure",
            calibrated_probabilities=probs,
            current_features=cpu_feats,
            reference_profile=self.ref_profile,
        )

        self.assertEqual(res["condition"], "cpu_pressure")
        self.assertEqual(res["confidence"], 0.96)
        supp_names = [x["feature_name"] for x in res["supporting_evidence"]]
        self.assertIn("cpu_mean", supp_names)
        self.assertIn("cpu_max", supp_names)

        # Check top evidence item properties
        top_item = res["supporting_evidence"][0]
        self.assertEqual(top_item["evidence_strength"], "strong")
        self.assertGreaterEqual(top_item["directional_score"], STRONG_EVIDENCE_THRESHOLD)
        self.assertIn("elevated", top_item["direction"])
        self.assertIn("cpu", res["overall_interpretation"].lower())

    # 3. Clear Memory vector produces Memory-supporting evidence
    def test_clear_memory_vector_supporting_evidence(self):
        mem_feats = dict(self.normal_vector)
        mem_feats["ram_used_pct"] = 88.0  # +9.5 std
        mem_feats["ram_available_ratio"] = 0.12  # (0.12 - 0.50) / 0.04 = -9.5 std (expected dir = -1)
        mem_feats["swap_used_pct"] = 25.0  # +7.5 std

        probs = {"normal": 0.02, "cpu_pressure": 0.01, "memory_pressure": 0.94, "disk_io_pressure": 0.03}
        res = generate_evidence(
            predicted_condition="memory_pressure",
            calibrated_probabilities=probs,
            current_features=mem_feats,
            reference_profile=self.ref_profile,
        )

        self.assertEqual(res["condition"], "memory_pressure")
        supp_names = [x["feature_name"] for x in res["supporting_evidence"]]
        self.assertIn("ram_available_ratio", supp_names)
        self.assertIn("ram_used_pct", supp_names)

        # ram_available_ratio: expected_direction is -1, raw_z is negative, so directional_score is POSITIVE!
        avail_item = next(x for x in res["supporting_evidence"] if x["feature_name"] == "ram_available_ratio")
        self.assertGreaterEqual(avail_item["directional_score"], STRONG_EVIDENCE_THRESHOLD)
        self.assertIn("substantially lower", avail_item["human_readable_statement"].lower())
        self.assertIn("reduced", avail_item["direction"].lower())

    # 4. Clear Disk vector produces Disk-supporting evidence
    def test_clear_disk_vector_supporting_evidence(self):
        disk_feats = dict(self.normal_vector)
        disk_feats["disk_io_rate_norm"] = 7.2  # +10.4 std
        disk_feats["disk_iops_norm"] = 3.8     # +9.3 std

        probs = {"normal": 0.01, "cpu_pressure": 0.01, "memory_pressure": 0.01, "disk_io_pressure": 0.97}
        res = generate_evidence(
            predicted_condition="disk_io_pressure",
            calibrated_probabilities=probs,
            current_features=disk_feats,
            reference_profile=self.ref_profile,
        )

        self.assertEqual(res["condition"], "disk_io_pressure")
        supp_names = [x["feature_name"] for x in res["supporting_evidence"]]
        self.assertIn("disk_io_rate_norm", supp_names)
        self.assertIn("disk_iops_norm", supp_names)
        top_item = res["supporting_evidence"][0]
        self.assertEqual(top_item["evidence_strength"], "strong")

    # 5. Contradictory feature gets lower/negative directional support
    def test_contradictory_feature_support(self):
        # Under CPU pressure, if CPU usage drops to 2.0% (far below normal 15.0%), it is contradictory
        contra_feats = dict(self.normal_vector)
        contra_feats["cpu_mean"] = 2.0  # -2.6 std below normal mean
        contra_feats["cpu_max"] = 5.0   # -2.5 std below normal mean

        probs = {"normal": 0.05, "cpu_pressure": 0.80, "memory_pressure": 0.05, "disk_io_pressure": 0.10}
        res = generate_evidence(
            predicted_condition="cpu_pressure",
            calibrated_probabilities=probs,
            current_features=contra_feats,
            reference_profile=self.ref_profile,
        )

        contra_names = [x["feature_name"] for x in res["contradictory_evidence"]]
        self.assertTrue("cpu_mean" in contra_names or "cpu_max" in contra_names)
        contra_item = next(x for x in res["contradictory_evidence"] if x["feature_name"] in ("cpu_mean", "cpu_max"))
        self.assertLessEqual(contra_item["directional_score"], CONTRADICTORY_THRESHOLD)
        self.assertEqual(contra_item["evidence_strength"], "contradictory")
        self.assertIn("moved against", contra_item["human_readable_statement"].lower())

    # 6. Missing feature is handled safely
    def test_missing_feature_handling(self):
        incomplete_feats = dict(self.normal_vector)
        del incomplete_feats["top_proc_mem_pct"]

        probs = {"normal": 0.90, "cpu_pressure": 0.03, "memory_pressure": 0.03, "disk_io_pressure": 0.04}

        # Strict mode raises error
        with self.assertRaises(InvalidTelemetryError):
            generate_evidence(
                predicted_condition="normal",
                calibrated_probabilities=probs,
                current_features=incomplete_feats,
                reference_profile=self.ref_profile,
                strict_validation=True,
            )

        # Non-strict mode excludes safely
        res = generate_evidence(
            predicted_condition="normal",
            calibrated_probabilities=probs,
            current_features=incomplete_feats,
            reference_profile=self.ref_profile,
            strict_validation=False,
        )
        self.assertEqual(len(res["invalid_features"]), 1)
        self.assertEqual(res["invalid_features"][0]["feature_name"], "top_proc_mem_pct")
        self.assertEqual(res["invalid_features"][0]["status"], "missing")
        self.assertEqual(res["evaluated_features_count"], 11)

    # 7. NaN is rejected or safely excluded according to policy
    def test_nan_handling(self):
        nan_feats = dict(self.normal_vector)
        nan_feats["ram_used_pct"] = np.nan

        probs = {"normal": 0.05, "cpu_pressure": 0.05, "memory_pressure": 0.85, "disk_io_pressure": 0.05}

        # Strict: raises InvalidTelemetryError
        with self.assertRaises(InvalidTelemetryError):
            generate_evidence(
                predicted_condition="memory_pressure",
                calibrated_probabilities=probs,
                current_features=nan_feats,
                reference_profile=self.ref_profile,
                strict_validation=True,
            )

        # Non-strict: safely recorded and excluded
        res = generate_evidence(
            predicted_condition="memory_pressure",
            calibrated_probabilities=probs,
            current_features=nan_feats,
            reference_profile=self.ref_profile,
            strict_validation=False,
        )
        self.assertEqual(len(res["invalid_features"]), 1)
        self.assertEqual(res["invalid_features"][0]["status"], "nan")

    # 8. Inf is rejected or safely excluded
    def test_inf_handling(self):
        inf_feats = dict(self.normal_vector)
        inf_feats["cpu_mean"] = float("inf")

        probs = {"normal": 0.1, "cpu_pressure": 0.8, "memory_pressure": 0.05, "disk_io_pressure": 0.05}

        with self.assertRaises(InvalidTelemetryError):
            generate_evidence(
                predicted_condition="cpu_pressure",
                calibrated_probabilities=probs,
                current_features=inf_feats,
                reference_profile=self.ref_profile,
                strict_validation=True,
            )

        res = generate_evidence(
            predicted_condition="cpu_pressure",
            calibrated_probabilities=probs,
            current_features=inf_feats,
            reference_profile=self.ref_profile,
            strict_validation=False,
        )
        self.assertEqual(len(res["invalid_features"]), 1)
        self.assertEqual(res["invalid_features"][0]["status"], "inf")

    # 9. Zero standard deviation does not cause division by zero
    def test_zero_std_no_division_by_zero(self):
        zero_std_stats = dict(self.ref_profile.feature_stats)
        zero_std_stats["cpu_std"] = {"mean": 0.0, "std": 0.0, "median": 0.0, "iqr": 0.0}

        zero_std_profile = EvidenceReferenceProfile(
            feature_stats=zero_std_stats,
            expected_directions=self.ref_profile.expected_directions,
            epsilon=SAFE_EPSILON,
            score_cap=SCORE_CAP,
        )

        test_feats = dict(self.normal_vector)
        test_feats["cpu_std"] = 1.0  # (1.0 - 0.0) / SAFE_EPSILON would be 10,000 without score cap

        probs = {"normal": 0.05, "cpu_pressure": 0.90, "memory_pressure": 0.02, "disk_io_pressure": 0.03}
        res = generate_evidence(
            predicted_condition="cpu_pressure",
            calibrated_probabilities=probs,
            current_features=test_feats,
            reference_profile=zero_std_profile,
        )

        # Must not crash, score capped at SCORE_CAP
        cpu_std_item = next(x for x in res["supporting_evidence"] if x["feature_name"] == "cpu_std")
        self.assertFalse(math.isnan(cpu_std_item["directional_score"]))
        self.assertFalse(math.isinf(cpu_std_item["directional_score"]))
        self.assertEqual(cpu_std_item["directional_score"], SCORE_CAP)

    # 10. Evidence is different when predicted class changes
    def test_evidence_differs_when_predicted_class_changes(self):
        # A mixed stress vector with elevated CPU and elevated RAM
        mixed_feats = dict(self.normal_vector)
        mixed_feats["cpu_mean"] = 80.0
        mixed_feats["ram_used_pct"] = 85.0
        mixed_feats["ram_available_ratio"] = 0.15

        probs_cpu = {"normal": 0.05, "cpu_pressure": 0.70, "memory_pressure": 0.20, "disk_io_pressure": 0.05}
        probs_mem = {"normal": 0.05, "cpu_pressure": 0.20, "memory_pressure": 0.70, "disk_io_pressure": 0.05}

        res_cpu = generate_evidence(
            predicted_condition="cpu_pressure",
            calibrated_probabilities=probs_cpu,
            current_features=mixed_feats,
            reference_profile=self.ref_profile,
        )

        res_mem = generate_evidence(
            predicted_condition="memory_pressure",
            calibrated_probabilities=probs_mem,
            current_features=mixed_feats,
            reference_profile=self.ref_profile,
        )

        top_cpu_feat = res_cpu["supporting_evidence"][0]["feature_name"]
        top_mem_feat = res_mem["supporting_evidence"][0]["feature_name"]

        self.assertEqual(top_cpu_feat, "cpu_mean")
        self.assertIn(top_mem_feat, ("ram_used_pct", "ram_available_ratio"))
        self.assertNotEqual(res_cpu["overall_interpretation"], res_mem["overall_interpretation"])

    # 11. Probability is not confused with evidence strength
    def test_probability_vs_evidence_strength_separation(self):
        # Case A: High model confidence (0.95), but moderate deviation
        mod_feats = dict(self.normal_vector)
        mod_feats["cpu_mean"] = 22.0  # +1.4 std (moderate)

        probs_high = {"normal": 0.02, "cpu_pressure": 0.95, "memory_pressure": 0.02, "disk_io_pressure": 0.01}
        res_high_conf = generate_evidence(
            predicted_condition="cpu_pressure",
            calibrated_probabilities=probs_high,
            current_features=mod_feats,
            reference_profile=self.ref_profile,
        )

        # Case B: Lower model confidence (0.55), but extreme deviation
        extreme_feats = dict(self.normal_vector)
        extreme_feats["cpu_mean"] = 95.0  # +16 std (strong)

        probs_low = {"normal": 0.15, "cpu_pressure": 0.55, "memory_pressure": 0.15, "disk_io_pressure": 0.15}
        res_low_conf = generate_evidence(
            predicted_condition="cpu_pressure",
            calibrated_probabilities=probs_low,
            current_features=extreme_feats,
            reference_profile=self.ref_profile,
        )

        # Probabilities are preserved as model confidence
        self.assertEqual(res_high_conf["confidence"], 0.95)
        self.assertEqual(res_low_conf["confidence"], 0.55)

        # Evidence scores reflect telemetry deviation independently of model probability
        self.assertLess(res_high_conf["evidence_score"], res_low_conf["evidence_score"])
        self.assertEqual(res_high_conf["supporting_evidence"][0]["evidence_strength"], "moderate")
        self.assertEqual(res_low_conf["supporting_evidence"][0]["evidence_strength"], "strong")

    # 12. Evidence reference statistics are training-derived
    def test_reference_statistics_training_derived(self):
        # Create a mock training dataframe with 20 Normal windows and 20 CPU windows
        rows = []
        for i in range(20):
            row = {f: 10.0 + i for f in FEATURE_NAMES}
            row["condition"] = "normal"
            row["session_id"] = f"train_norm_{i // 5}"
            row["machine_id"] = "train_mach_A"
            rows.append(row)

        for i in range(20):
            row = {f: 50.0 + i for f in FEATURE_NAMES}
            row["condition"] = "cpu_pressure"
            row["session_id"] = f"train_cpu_{i // 5}"
            row["machine_id"] = "train_mach_A"
            rows.append(row)

        df_train = pd.DataFrame(rows)
        profile = EvidenceReferenceProfile.fit_from_training_data(df_train)

        self.assertEqual(profile.training_machines, ["train_mach_A"])
        self.assertEqual(profile.normal_sessions_count, 4)
        self.assertAlmostEqual(profile.feature_stats["cpu_mean"]["mean"], np.mean([10.0 + i for i in range(20)]))
        self.assertEqual(profile.expected_directions["cpu_pressure"]["cpu_mean"], 1)

    # 13. Test/held-out statistics cannot enter the reference artifact
    def test_held_out_statistics_isolation(self):
        # Build training dataframe
        train_rows = [{f: 10.0 for f in FEATURE_NAMES} for _ in range(10)]
        for r in train_rows:
            r["condition"] = "normal"
            r["machine_id"] = "machine_train"
            r["session_id"] = "sess_tr_1"
        df_train = pd.DataFrame(train_rows)

        # Build held-out test dataframe with very different values
        test_rows = [{f: 999.0 for f in FEATURE_NAMES} for _ in range(5)]
        for r in test_rows:
            r["condition"] = "normal"
            r["machine_id"] = "machine_test_held_out"
            r["session_id"] = "sess_test_1"
        df_test = pd.DataFrame(test_rows)

        profile = EvidenceReferenceProfile.fit_from_training_data(df_train)

        # Assert held-out machine is not in profile
        self.assertNotIn("machine_test_held_out", profile.training_machines)
        self.assertAlmostEqual(profile.feature_stats["cpu_mean"]["mean"], 10.0)

    # 14. Causal language is not generated
    def test_no_causal_language_policy(self):
        # Test the validator explicitly
        with self.assertRaises(CausalLanguageError):
            validate_no_causal_language("CPU load caused the system failure.")
        with self.assertRaises(CausalLanguageError):
            validate_no_causal_language("The root cause was memory exhaustion.")
        with self.assertRaises(CausalLanguageError):
            validate_no_causal_language("This resulted in broken hardware.")

        # Test all generated statements across conditions
        conditions_to_test = ["normal", "cpu_pressure", "memory_pressure", "disk_io_pressure"]
        for cond in conditions_to_test:
            probs = {c: 0.25 for c in conditions_to_test}
            probs[cond] = 0.85
            res = generate_evidence(
                predicted_condition=cond,
                calibrated_probabilities=probs,
                current_features=self.normal_vector,
                reference_profile=self.ref_profile,
            )

            validate_no_causal_language(res["overall_interpretation"])
            for item in res["supporting_evidence"] + res["contradictory_evidence"] + res["neutral_features"]:
                validate_no_causal_language(item["human_readable_statement"])
                for pattern in FORBIDDEN_CAUSAL_PATTERNS:
                    self.assertIsNone(pattern.search(item["human_readable_statement"]))

    # 15. Output schema is stable
    def test_output_schema_stability(self):
        probs = {"normal": 0.1, "cpu_pressure": 0.8, "memory_pressure": 0.05, "disk_io_pressure": 0.05}
        res = generate_evidence(
            predicted_condition="cpu_pressure",
            calibrated_probabilities=probs,
            current_features=self.normal_vector,
            reference_profile=self.ref_profile,
        )

        expected_keys = {
            "condition",
            "confidence",
            "abnormality_score",
            "overall_interpretation",
            "supporting_evidence",
            "contradictory_evidence",
            "neutral_features",
            "evidence_score",
            "evidence_consistency",
            "invalid_features",
            "class_probabilities",
            "evaluated_features_count",
        }
        self.assertEqual(set(res.keys()), expected_keys)

        # Verify item schema
        if res["neutral_features"]:
            item = res["neutral_features"][0]
            item_keys = {
                "feature_name",
                "display_name",
                "observed_value",
                "normal_reference",
                "normal_std",
                "raw_z",
                "directional_deviation",
                "expected_direction",
                "directional_score",
                "direction",
                "evidence_strength",
                "subsystem",
                "human_readable_statement",
            }
            self.assertEqual(set(item.keys()), item_keys)

    # 16. Top evidence contains only valid known features
    def test_top_evidence_valid_known_features(self):
        feats = dict(self.normal_vector)
        feats["cpu_mean"] = 90.0
        feats["cpu_max"] = 95.0
        feats["disk_io_rate_norm"] = 8.0

        probs = {"normal": 0.01, "cpu_pressure": 0.95, "memory_pressure": 0.02, "disk_io_pressure": 0.02}
        res = generate_evidence(
            predicted_condition="cpu_pressure",
            calibrated_probabilities=probs,
            current_features=feats,
            reference_profile=self.ref_profile,
        )

        for item in res["supporting_evidence"]:
            self.assertIn(item["feature_name"], FEATURE_NAMES)
            self.assertIsInstance(item["observed_value"], float)
            self.assertIsInstance(item["directional_score"], float)


if __name__ == "__main__":
    unittest.main()
