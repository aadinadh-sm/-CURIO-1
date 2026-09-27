"""Unit tests for physical machine deployment, packaging, verification, and LOMO readiness.

Covers:
- Machine inventory and clone detection
- Pre-flight smoke testing and safety checks
- Manifest generation and schema verification
- Dataset packaging and SHA-256 checksum generation
- Dataset importer cryptographic verification, zip safety, and collision guardrails
- Physical LOMO readiness evaluation
"""

import io
import json
import os
import platform
import shutil
import tempfile
import unittest
import zipfile
import numpy as np
import pandas as pd

from scripts.check_lomo_readiness import evaluate_lomo_readiness
from scripts.check_physical_machine import (
    check_environment,
    check_directories,
    check_host_identity,
    smoke_test_features,
)
from scripts.create_machine_manifest import create_manifest
from scripts.import_machine_dataset import (
    import_dataset,
    DatasetImportError,
    compute_bytes_sha256,
)
from scripts.inventory_physical_machines import (
    get_cpu_model_name,
    get_local_machine_inventory,
    inventory_all_machines,
    MACHINE_A_HOSTNAME,
)
from scripts.package_machine_dataset import (
    package_dataset,
    compute_sha256,
)
from src.features import FEATURE_NAMES


class TestPhysicalDeployment(unittest.TestCase):
    """Test suite for physical machine onboarding and dataset packaging tools."""

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.raw_dir = os.path.join(self.test_dir, "data", "physical_raw")
        self.proc_dir = os.path.join(self.test_dir, "data", "physical_processed")
        self.meta_dir = os.path.join(self.test_dir, "data", "physical_metadata")
        self.export_dir = os.path.join(self.test_dir, "data", "exports")
        os.makedirs(self.raw_dir, exist_ok=True)
        os.makedirs(self.proc_dir, exist_ok=True)
        os.makedirs(self.meta_dir, exist_ok=True)
        os.makedirs(self.export_dir, exist_ok=True)

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def _create_mock_session(self, machine_id: str, condition: str, stress_level: str, sess_idx: int):
        """Helper to create a valid 61-sample raw session and 11-window feature session."""
        session_id = f"{machine_id}_{condition}_{stress_level}_sess{sess_idx:02d}"

        # 61 raw samples
        timestamps = [1700000000.0 + i * 0.5 for i in range(61)]
        raw_data = {
            "timestamp": timestamps,
            "cpu_overall": [25.0] * 61,
            "vmem_percent": [50.0] * 61,
            "vmem_available": [8000000000] * 61,
            "vmem_total": [16000000000] * 61,
            "disk_read_bytes": [1000 + i * 100 for i in range(61)],
            "disk_write_bytes": [2000 + i * 200 for i in range(61)],
            "machine_id": [machine_id] * 61,
            "session_id": [session_id] * 61,
            "condition": [condition] * 61,
            "stress_level": [stress_level] * 61,
            "background_workload": ["idle_desktop"] * 61,
            "is_ramp_up": [1 if i < 10 else 0 for i in range(61)],
        }
        raw_df = pd.DataFrame(raw_data)
        raw_path = os.path.join(self.raw_dir, f"{session_id}_raw.csv")
        raw_df.to_csv(raw_path, index=False)

        # 11 feature windows
        feat_rows = []
        for w in range(11):
            row = {
                "window_idx": w,
                "window_start_offset": w * 2.5,
                "window_end_offset": w * 2.5 + 5.0,
                "cpu_mean": 25.0,
                "cpu_max": 25.0,
                "cpu_std": 0.0,
                "cpu_core_imbalance": 2.0,
                "ram_used_pct": 50.0,
                "ram_available_ratio": 0.50,
                "swap_used_pct": 5.0,
                "disk_io_rate_norm": 3.2,
                "disk_iops_norm": 1.1,
                "process_count_delta": 0.0,
                "top_proc_cpu_ratio": 0.4,
                "top_proc_mem_pct": 2.0,
                "machine_id": machine_id,
                "session_id": session_id,
                "condition": condition,
                "stress_level": stress_level,
                "background_workload": "idle_desktop",
                "is_ramp_up": 1 if w == 0 else 0,
            }
            feat_rows.append(row)

        feat_df = pd.DataFrame(feat_rows)
        proc_path = os.path.join(self.proc_dir, f"{session_id}_features.csv")
        feat_df.to_csv(proc_path, index=False)
        return session_id

    # 1. Inventory & Clone Detection Tests
    def test_inventory_hardware_inspection(self):
        cpu = get_cpu_model_name()
        self.assertIsInstance(cpu, str)
        self.assertGreater(len(cpu), 0)

        info = get_local_machine_inventory("physical_machine_A")
        self.assertEqual(info["machine_id"], "physical_machine_A")
        self.assertIn("logical_cpu_count", info)
        self.assertIn("total_ram_gb", info)
        self.assertGreater(info["total_ram_gb"], 0)

    def test_clone_warning_on_machine_a(self):
        if platform.node().upper() == MACHINE_A_HOSTNAME.upper():
            res = inventory_all_machines(target_machine_id="physical_machine_B", metadata_dir=self.meta_dir)
            self.assertTrue(res["is_machine_a_clone"])
            self.assertFalse(res["has_sufficient_diversity"])

    # 2. Pre-Flight Verification Tests
    def test_preflight_environment_check(self):
        env = check_environment()
        self.assertIn("python", env)
        self.assertIn("psutil", env)
        self.assertIn("sklearn", env)

    def test_preflight_host_identity_rejection(self):
        if platform.node().upper() == MACHINE_A_HOSTNAME.upper():
            with self.assertRaises(RuntimeError) as ctx:
                check_host_identity("physical_machine_B")
            self.assertIn("HOSTNAME COLLISION", str(ctx.exception))

    def test_smoke_test_features(self):
        samples = [
            {
                "timestamp": 100.0 + i * 0.5,
                "cpu_overall": 20.0,
                "cpu_cores": [20.0, 20.0],
                "vmem_percent": 45.0,
                "vmem_available": 8000000000,
                "vmem_total": 16000000000,
                "swap_percent": 10.0,
                "swap_used": 1000000000,
                "swap_total": 10000000000,
                "disk_read_bytes": 1000 + i * 10,
                "disk_write_bytes": 2000 + i * 20,
                "disk_read_count": 10 + i,
                "disk_write_count": 20 + i,
                "process_count": 150,
                "top_proc_cpu": 5.0,
                "top_proc_rss": 200000000,
            }
            for i in range(11)
        ]
        df_feat = smoke_test_features(samples)
        for feat in FEATURE_NAMES:
            self.assertIn(feat, df_feat.columns)
        self.assertFalse(df_feat[FEATURE_NAMES].isna().any().any())

    # 3. Manifest Generation Tests
    def test_manifest_creation(self):
        m_id = "test_machine_M"
        self._create_mock_session(m_id, "normal", "none", 1)
        self._create_mock_session(m_id, "cpu_pressure", "high", 2)

        manifest = create_manifest(
            machine_id=m_id,
            raw_dir=self.raw_dir,
            processed_dir=self.proc_dir,
            metadata_dir=self.meta_dir,
        )
        self.assertEqual(manifest["schema_version"], "1.0.0")
        self.assertEqual(manifest["machine_id"], m_id)
        self.assertEqual(manifest["dataset_summary"]["raw_sessions_count"], 2)
        self.assertEqual(manifest["dataset_summary"]["processed_sessions_count"], 2)
        self.assertEqual(manifest["dataset_summary"]["total_feature_windows"], 22)

        manifest_file = os.path.join(self.meta_dir, f"{m_id}_manifest.json")
        self.assertTrue(os.path.exists(manifest_file))

    # 4. Packaging and SHA-256 Checksum Tests
    def test_package_dataset(self):
        m_id = "test_machine_P"
        self._create_mock_session(m_id, "normal", "none", 1)

        # Override module directories temporarily
        import scripts.package_machine_dataset as pmd
        orig_raw = pmd.RAW_DIR
        orig_proc = pmd.PROCESSED_DIR
        orig_meta = pmd.METADATA_DIR
        pmd.RAW_DIR = self.raw_dir
        pmd.PROCESSED_DIR = self.proc_dir
        pmd.METADATA_DIR = self.meta_dir

        zip_path = package_dataset(
            machine_id=m_id,
            output_dir=self.export_dir,
            raw_dir=self.raw_dir,
            processed_dir=self.proc_dir,
            metadata_dir=self.meta_dir,
            skip_validation=False,
        )
        self.assertTrue(os.path.exists(zip_path))

        # Inspect zip contents
        with zipfile.ZipFile(zip_path, "r") as zf:
            names = zf.namelist()
            self.assertIn("checksums.sha256", names)
            self.assertIn(f"data/physical_metadata/{m_id}_manifest.json", names)
            # Verify checksums.sha256 format
            ck_text = zf.read("checksums.sha256").decode("utf-8")
            self.assertIn(f"data/physical_metadata/{m_id}_manifest.json", ck_text)

    # 5. Importer Security and Verification Tests
    def test_import_checksum_tamper_detection(self):
        m_id = "test_machine_T"
        self._create_mock_session(m_id, "normal", "none", 1)

        zip_path = package_dataset(
            machine_id=m_id,
            output_dir=self.export_dir,
            raw_dir=self.raw_dir,
            processed_dir=self.proc_dir,
            metadata_dir=self.meta_dir,
        )

        # Corrupt the archive by modifying a byte in one of the files
        corrupt_zip = os.path.join(self.export_dir, "corrupt.zip")
        with zipfile.ZipFile(zip_path, "r") as zin:
            with zipfile.ZipFile(corrupt_zip, "w") as zout:
                for item in zin.infolist():
                    content = zin.read(item.filename)
                    if item.filename.endswith("_raw.csv"):
                        # Tamper with content without updating checksums.sha256
                        content = content + b"\n#tampered"
                    zout.writestr(item, content)

        with self.assertRaises(DatasetImportError) as ctx:
            import_dataset(corrupt_zip, project_root=self.test_dir, dry_run=True)
        self.assertIn("Cryptographic hash mismatch", str(ctx.exception))

    def test_import_zip_slip_rejection(self):
        slip_zip = os.path.join(self.export_dir, "slip.zip")
        with zipfile.ZipFile(slip_zip, "w") as zf:
            zf.writestr("../evil.txt", "attack")
            zf.writestr("checksums.sha256", "hash  ../evil.txt")

        with self.assertRaises(DatasetImportError) as ctx:
            import_dataset(slip_zip, project_root=self.test_dir)
        self.assertIn("dangerous file path", str(ctx.exception))

    def test_import_session_collision_rejection(self):
        # Create Machine A with session_01
        m_a = "machine_A"
        sess_a = self._create_mock_session(m_a, "normal", "none", 1)

        # Create Machine B with the exact same session_id (collision)
        m_b = "machine_B"
        b_raw_dir = os.path.join(self.test_dir, "b_raw")
        b_proc_dir = os.path.join(self.test_dir, "b_proc")
        b_meta_dir = os.path.join(self.test_dir, "b_meta")
        os.makedirs(b_raw_dir, exist_ok=True)
        os.makedirs(b_proc_dir, exist_ok=True)
        os.makedirs(b_meta_dir, exist_ok=True)

        # Create collision file in B's dirs
        df = pd.read_csv(os.path.join(self.proc_dir, f"{sess_a}_features.csv"))
        df.to_csv(os.path.join(b_proc_dir, f"{m_b}_colliding_features.csv"), index=False)
        raw_df = pd.read_csv(os.path.join(self.raw_dir, f"{sess_a}_raw.csv"))
        raw_df.to_csv(os.path.join(b_raw_dir, f"{m_b}_colliding_raw.csv"), index=False)

        zip_path = package_dataset(
            machine_id=m_b,
            output_dir=self.export_dir,
            raw_dir=b_raw_dir,
            processed_dir=b_proc_dir,
            metadata_dir=b_meta_dir,
        )
        with self.assertRaises(DatasetImportError) as ctx:
            import_dataset(zip_path, project_root=self.test_dir)
        self.assertIn("SESSION COLLISION", str(ctx.exception))

    # 6. LOMO Readiness Evaluation Tests
    def test_lomo_readiness_insufficient_machines(self):
        # Only 1 machine present
        m_id = "physical_machine_A"
        self._create_mock_session(m_id, "normal", "none", 1)

        res = evaluate_lomo_readiness(
            raw_dir=self.raw_dir,
            processed_dir=self.proc_dir,
            metadata_dir=self.meta_dir,
        )
        self.assertFalse(res["ready"])
        self.assertEqual(res["status"], "TRUE PHYSICAL LOMO NOT READY")
        self.assertIn("physical_machine_B", res["missing_machines"])
        self.assertIn("physical_machine_C", res["missing_machines"])

    def test_lomo_readiness_three_complete_machines(self):
        # Build 22 sessions for Machine A, B, and C
        machines = ["physical_machine_A", "physical_machine_B", "physical_machine_C"]
        conditions_spec = (
            [("normal", "none")] * 4
            + [("cpu_pressure", "high")] * 6
            + [("memory_pressure", "high")] * 6
            + [("disk_io_pressure", "high")] * 6
        )

        for m in machines:
            for idx, (cond, lvl) in enumerate(conditions_spec):
                self._create_mock_session(m, cond, lvl, idx)

        res = evaluate_lomo_readiness(
            raw_dir=self.raw_dir,
            processed_dir=self.proc_dir,
            metadata_dir=self.meta_dir,
        )
        self.assertTrue(res["ready"])
        self.assertEqual(res["status"], "TRUE PHYSICAL LOMO READY")
        self.assertEqual(len(res["present_machines"]), 3)
        self.assertEqual(len(res["missing_machines"]), 0)


if __name__ == "__main__":
    unittest.main()
