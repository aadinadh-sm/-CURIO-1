"""Unit tests for HeuristicBaselineClassifier using unittest."""

import unittest
import numpy as np
import pandas as pd

from src.baseline_rules import HeuristicBaselineClassifier
from src.features import FEATURE_NAMES


class TestHeuristicBaselineClassifier(unittest.TestCase):
    def setUp(self):
        self.clf = HeuristicBaselineClassifier()
        # Default normal feature baseline
        self.normal_features = {
            "cpu_mean": 20.0,
            "cpu_max": 35.0,
            "cpu_std": 5.0,
            "cpu_core_imbalance": 40.0,
            "ram_used_pct": 55.0,
            "ram_available_ratio": 0.45,
            "swap_used_pct": 5.0,
            "disk_io_rate_norm": 4.5,
            "disk_iops_norm": 1.2,
            "process_count_delta": 0.0,
            "top_proc_cpu_ratio": 0.25,
            "top_proc_mem_pct": 8.0,
        }

    def test_normal_case(self):
        res = self.clf.predict_window(self.normal_features)
        self.assertEqual(res["predicted_condition"], "normal")
        self.assertFalse(res["is_ambiguous"])
        self.assertEqual(res["rule_scores"]["normal"], 1.0)
        self.assertGreater(len(res["rule_evidence"]), 0)

    def test_clear_cpu_case(self):
        feats = dict(self.normal_features)
        feats["cpu_mean"] = 88.5
        feats["cpu_max"] = 92.0
        res = self.clf.predict_window(feats)
        self.assertEqual(res["predicted_condition"], "cpu_pressure")
        self.assertFalse(res["is_ambiguous"])
        self.assertGreater(res["rule_scores"]["cpu_pressure"], 0.5)

    def test_clear_memory_case(self):
        feats = dict(self.normal_features)
        feats["ram_used_pct"] = 92.0
        feats["ram_available_ratio"] = 0.07
        res = self.clf.predict_window(feats)
        self.assertEqual(res["predicted_condition"], "memory_pressure")
        self.assertFalse(res["is_ambiguous"])
        self.assertGreater(res["rule_scores"]["memory_pressure"], 0.5)

    def test_clear_disk_case(self):
        feats = dict(self.normal_features)
        feats["disk_io_rate_norm"] = 8.3
        feats["disk_iops_norm"] = 2.9
        res = self.clf.predict_window(feats)
        self.assertEqual(res["predicted_condition"], "disk_io_pressure")
        self.assertFalse(res["is_ambiguous"])
        self.assertGreater(res["rule_scores"]["disk_io_pressure"], 0.5)

    def test_ambiguous_case(self):
        # Both CPU and Disk triggers with similar intensity
        feats = dict(self.normal_features)
        feats["cpu_mean"] = 72.0
        feats["disk_io_rate_norm"] = 7.6
        res = self.clf.predict_window(feats)
        self.assertTrue(res["is_ambiguous"])
        self.assertIn(res["predicted_condition"], ["cpu_pressure", "disk_io_pressure"])
        self.assertTrue(any("Ambiguous" in e for e in res["rule_evidence"]))

    def test_missing_features(self):
        feats = dict(self.normal_features)
        del feats["cpu_mean"]
        with self.assertRaises(ValueError):
            self.clf.predict_window(feats)

    def test_invalid_nan_features(self):
        feats = dict(self.normal_features)
        feats["ram_used_pct"] = np.nan
        with self.assertRaises(ValueError):
            self.clf.predict_window(feats)

    def test_invalid_inf_features(self):
        feats = dict(self.normal_features)
        feats["disk_iops_norm"] = np.inf
        with self.assertRaises(ValueError):
            self.clf.predict_window(feats)

    def test_predict_dataframe(self):
        row1 = dict(self.normal_features)
        row2 = dict(self.normal_features)
        row2["cpu_mean"] = 90.0

        df = pd.DataFrame([row1, row2])
        preds_df = self.clf.predict_dataframe(df)

        self.assertEqual(len(preds_df), 2)
        self.assertEqual(preds_df["predicted_condition"].iloc[0], "normal")
        self.assertEqual(preds_df["predicted_condition"].iloc[1], "cpu_pressure")


if __name__ == "__main__":
    unittest.main()
