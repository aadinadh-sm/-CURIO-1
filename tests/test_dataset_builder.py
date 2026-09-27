"""Unit tests for DatasetBuilder using unittest."""

import os
import tempfile
import unittest

from src.dataset_builder import DatasetBuilder


class TestDatasetBuilder(unittest.TestCase):
    def test_record_short_session(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            raw_dir = os.path.join(tmpdir, "raw")
            proc_dir = os.path.join(tmpdir, "processed")

            builder = DatasetBuilder(raw_dir=raw_dir, processed_dir=proc_dir, machine_id="test_lap_01")
            res = builder.record_session(
                condition="normal",
                stress_level="none",
                background_workload="idle",
                duration_seconds=3.0,
                ramp_duration_seconds=1.0,
                session_id="test_sess_01",
            )

            self.assertEqual(res["session_id"], "test_sess_01")
            self.assertEqual(res["machine_id"], "test_lap_01")
            self.assertEqual(res["condition"], "normal")
            self.assertTrue(os.path.exists(res["raw_path"]))
            self.assertTrue(os.path.exists(res["processed_path"]))

            df = res["features_df"]
            self.assertGreater(len(df), 0)
            self.assertIn("machine_id", df.columns)
            self.assertIn("condition", df.columns)
            self.assertIn("is_ramp_up", df.columns)
            self.assertEqual(df["machine_id"].iloc[0], "test_lap_01")

            # Test loader
            all_df = builder.load_all_processed_features()
            self.assertEqual(len(all_df), len(df))


if __name__ == "__main__":
    unittest.main()
