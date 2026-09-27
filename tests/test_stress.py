"""Unit tests for Stress Generation Harness using unittest."""

import os
import tempfile
import time
import unittest
import psutil

from src.stress_harness import CPUStress, MemoryStress, DiskStress


class TestStressHarness(unittest.TestCase):
    def test_cpu_stress_lifecycle(self):
        stress = CPUStress(intensity="low")
        stress.start()
        time.sleep(1.0)
        self.assertGreater(len(stress._procs), 0)
        for p in stress._procs:
            self.assertTrue(p.is_alive())

        stress.stop()
        time.sleep(0.5)
        for p in stress._procs:
            self.assertFalse(p.is_alive())
        self.assertEqual(len(stress._procs), 0)

    def test_memory_stress_safety_and_cleanup(self):
        vmem_start = psutil.virtual_memory()
        stress = MemoryStress(intensity="low", min_headroom_gb=1.5)
        stress.start()

        # Verify headroom retained
        vmem_during = psutil.virtual_memory()
        self.assertGreaterEqual(vmem_during.available, stress.min_headroom_bytes - (100 * 1024 * 1024))

        stress.stop()
        self.assertEqual(len(stress._chunks), 0)

    def test_disk_stress_scratch_cleanup(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            stress = DiskStress(scratch_dir=tmpdir, intensity="low")
            stress.start()
            time.sleep(1.0)
            self.assertTrue(os.path.exists(stress.scratch_file))

            stress.stop()
            time.sleep(0.2)
            self.assertFalse(os.path.exists(stress.scratch_file))

    def test_context_managers(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with DiskStress(scratch_dir=tmpdir, intensity="low") as d:
                time.sleep(0.5)
                scratch = d.scratch_file
                self.assertTrue(os.path.exists(scratch))
            self.assertFalse(os.path.exists(scratch))


if __name__ == "__main__":
    unittest.main()
