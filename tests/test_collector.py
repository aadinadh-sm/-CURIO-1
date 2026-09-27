"""Unit tests for TelemetryCollector using Python standard unittest."""

import os
import tempfile
import time
import unittest

from src.collector import TelemetryCollector, save_raw_telemetry, load_raw_telemetry


class TestTelemetryCollector(unittest.TestCase):
    def test_collector_initialization(self):
        collector = TelemetryCollector(sample_interval=0.5)
        self.assertEqual(collector.sample_interval, 0.5)

    def test_collector_short_sampling(self):
        collector = TelemetryCollector(sample_interval=0.5)
        duration = 2.0
        expected_samples = 5

        t0 = time.time()
        samples = collector.collect(duration_seconds=duration)
        elapsed = time.time() - t0

        self.assertEqual(len(samples), expected_samples)
        expected_span = (expected_samples - 1) * collector.sample_interval
        self.assertLess(abs(elapsed - expected_span), 0.35)

        for s in samples:
            self.assertIn("timestamp", s)
            self.assertTrue(0.0 <= s["cpu_overall"] <= 100.0)
            self.assertIsInstance(s["cpu_cores"], list)
            self.assertGreater(len(s["cpu_cores"]), 0)
            self.assertTrue(0.0 <= s["vmem_percent"] <= 100.0)
            self.assertGreater(s["vmem_available"], 0)
            self.assertGreater(s["vmem_total"], 0)
            self.assertTrue(0.0 <= s["swap_percent"] <= 100.0)
            self.assertGreaterEqual(s["process_count"], 1)
            self.assertGreaterEqual(s["top_proc_cpu"], 0.0)
            self.assertGreaterEqual(s["top_proc_rss"], 0.0)

    def test_save_and_load_raw_telemetry(self):
        collector = TelemetryCollector(sample_interval=0.5)
        samples = collector.collect(duration_seconds=1.0)
        self.assertEqual(len(samples), 3)

        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = os.path.join(tmpdir, "telemetry_test.csv")
            save_raw_telemetry(samples, csv_path)
            self.assertTrue(os.path.exists(csv_path))

            loaded = load_raw_telemetry(csv_path)
            self.assertEqual(len(loaded), 3)

            for orig, loaded_sample in zip(samples, loaded):
                self.assertAlmostEqual(orig["timestamp"], loaded_sample["timestamp"], places=4)
                self.assertAlmostEqual(orig["cpu_overall"], loaded_sample["cpu_overall"], places=4)
                self.assertEqual(orig["cpu_cores"], loaded_sample["cpu_cores"])
                self.assertEqual(orig["vmem_available"], loaded_sample["vmem_available"])
                self.assertEqual(orig["process_count"], loaded_sample["process_count"])


if __name__ == "__main__":
    unittest.main()
