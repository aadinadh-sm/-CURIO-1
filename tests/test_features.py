"""Unit tests for Feature Extraction Engine using unittest."""

import time
import unittest
import numpy as np
import pandas as pd

from src.features import (
    FEATURE_NAMES,
    extract_features_from_window,
    extract_windows,
    extract_feature_dataframe,
)


class TestFeatures(unittest.TestCase):
    def _create_synthetic_samples(self, count=61, sample_interval=0.5):
        """Generates synthetic samples simulating 30 seconds of telemetry."""
        samples = []
        t0 = 1000.0
        for i in range(count):
            t = t0 + i * sample_interval
            samples.append({
                "timestamp": t,
                "cpu_overall": 25.0 + 5.0 * np.sin(i * 0.2),
                "cpu_cores": [20.0, 30.0, 22.0, 28.0],
                "vmem_percent": 65.0 + i * 0.1,
                "vmem_available": 4000000000 - i * 10000000,
                "vmem_total": 16000000000,
                "swap_percent": 10.0,
                "swap_used": 1000000000,
                "swap_total": 10000000000,
                "disk_read_bytes": 1000000 + i * 50000,
                "disk_write_bytes": 2000000 + i * 100000,
                "disk_read_count": 100 + i * 5,
                "disk_write_count": 200 + i * 10,
                "disk_read_time": 50,
                "disk_write_time": 100,
                "process_count": 250 + (1 if i % 10 == 0 else 0),
                "top_proc_cpu": 15.0,
                "top_proc_rss": 800000000,
            })
        return samples

    def test_feature_names_constant(self):
        self.assertEqual(len(FEATURE_NAMES), 12)
        self.assertIn("cpu_mean", FEATURE_NAMES)
        self.assertIn("disk_iops_norm", FEATURE_NAMES)
        self.assertIn("disk_io_rate_norm", FEATURE_NAMES)
        self.assertIn("top_proc_cpu_ratio", FEATURE_NAMES)

    def test_extract_features_single_window(self):
        samples = self._create_synthetic_samples(count=10, sample_interval=0.5)
        feats = extract_features_from_window(samples)

        self.assertEqual(len(feats), 12)
        for k in FEATURE_NAMES:
            self.assertIn(k, feats)
            self.assertFalse(np.isnan(feats[k]))
            self.assertFalse(np.isinf(feats[k]))

        # Validate ranges
        self.assertTrue(0.0 <= feats["cpu_mean"] <= 100.0)
        self.assertTrue(0.0 <= feats["ram_used_pct"] <= 100.0)
        self.assertTrue(0.0 <= feats["ram_available_ratio"] <= 1.0)
        self.assertTrue(0.0 <= feats["top_proc_cpu_ratio"] <= 1.0)
        self.assertGreaterEqual(feats["disk_io_rate_norm"], 0.0)
        self.assertGreaterEqual(feats["disk_iops_norm"], 0.0)

    def test_window_partitioning_30s(self):
        # 60 samples at 0.5s = 29.5s span
        samples = self._create_synthetic_samples(count=61, sample_interval=0.5)
        windows = extract_windows(samples, window_duration=5.0, step_duration=2.5)

        # In 29.5s:
        # starts: 0.0, 2.5, 5.0, 7.5, 10.0, 12.5, 15.0, 17.5, 20.0, 22.5, 24.5 (up to 24.5+5=29.5)
        # Expected: exactly 11 windows
        self.assertEqual(len(windows), 11)

        for w_start, w_end, w_samples in windows:
            self.assertAlmostEqual(w_end - w_start, 5.0, places=2)
            self.assertGreaterEqual(len(w_samples), 9)

    def test_extract_feature_dataframe(self):
        samples = self._create_synthetic_samples(count=61, sample_interval=0.5)
        df = extract_feature_dataframe(samples, window_duration=5.0, step_duration=2.5)

        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 11)
        self.assertIn("window_idx", df.columns)
        self.assertIn("window_start_offset", df.columns)
        for f in FEATURE_NAMES:
            self.assertIn(f, df.columns)
            self.assertEqual(df[f].isna().sum(), 0)


if __name__ == "__main__":
    unittest.main()
