# CURIO — Milestone Report: Portable Physical-Machine Deployment & Onboarding (B + C)

**Date**: 2026-09-27  
**Project**: CURIO (Undergraduate Capstone)  
**Milestone**: Portable Physical-Machine Deployment for B + C  
**Test Suite Status**: **64 / 64 Tests Passing (100% OK)**  
**LOMO Readiness Status**: **TRUE PHYSICAL LOMO NOT READY (1 / 3 Physical Machines Available)**  

---

## 1. Executive Summary & Objective

In this milestone, CURIO was extended with a production-grade, portable deployment, verification, packaging, and onboarding toolkit. This tooling enables a human operator to easily deploy CURIO to secondary and tertiary physical computers (`physical_machine_B` and `physical_machine_C`), execute pre-flight safety and smoke tests, run the frozen 22-session telemetry collection protocol, package the resulting dataset into a cryptographically verified archive with SHA-256 checksums, and import it into the central analysis repository without manual error or data leakage.

Crucially, **no synthetic data, virtual machines, or aliased hosts were used to bypass the physical diversity requirement**. The readiness engine strictly confirms that true physical Leave-One-Machine-Out (LOMO) cross-validation cannot run until genuine datasets from Machines B and C are collected and imported.

---

## 2. Implemented Components & Tooling

| Component | Path | Purpose |
| :--- | :--- | :--- |
| **Setup Script** | [`setup_physical_machine.ps1`](file:///c:/Users/Aadinadh%20S%20M/Downloads/cuuuu/scripts/setup_physical_machine.ps1) | Automated PowerShell setup for dependencies (`psutil`, `scikit-learn`, `numpy`, `pandas`, `joblib`), directory scaffolding, and environment verification. |
| **Inventory & Clone Guard** | [`inventory_physical_machines.py`](file:///c:/Users/Aadinadh%20S%20M/Downloads/cuuuu/scripts/inventory_physical_machines.py) | Fingerprints CPU, RAM, OS, cores; detects and rejects attempts to register Machine A (`LAPTOP-I0G3I8PA`) as Machine B or C. |
| **Pre-Flight Checker** | [`check_physical_machine.py`](file:///c:/Users/Aadinadh%20S%20M/Downloads/cuuuu/scripts/check_physical_machine.py) | Non-destructive smoke test: validates directories, RAM safety headroom ($\ge 1.5$ GB), 5s 2 Hz collector test, PID 0 exclusion, and 12-feature extractor (0 NaNs/Infs). |
| **Manifest Generator** | [`create_machine_manifest.py`](file:///c:/Users/Aadinadh%20S%20M/Downloads/cuuuu/scripts/create_machine_manifest.py) | Produces signed JSON manifest (`data/physical_metadata/<machine_id>_manifest.json`) cataloging hardware, environment, protocol, and session IDs. |
| **Dataset Packager** | [`package_machine_dataset.py`](file:///c:/Users/Aadinadh%20S%20M/Downloads/cuuuu/scripts/package_machine_dataset.py) | Validates all sessions against Rules A–P, generates SHA-256 checksums, and bundles raw, processed, and manifest files into `<machine_id>_curio_dataset.zip`. |
| **Dataset Importer** | [`import_machine_dataset.py`](file:///c:/Users/Aadinadh%20S%20M/Downloads/cuuuu/scripts/import_machine_dataset.py) | Verifies zip safety, SHA-256 checksums, manifest schema, cross-machine session ID collision guardrails, and full Rules A–P telemetry validation before extraction. |
| **LOMO Readiness Engine** | [`check_lomo_readiness.py`](file:///c:/Users/Aadinadh%20S%20M/Downloads/cuuuu/scripts/check_lomo_readiness.py) | Verifies whether $\ge 3$ distinct physical machines have complete 22-session / 242-window datasets. Emits `TRUE PHYSICAL LOMO READY` (0) or `NOT READY` (1). |
| **Operator Runbook** | [`PHYSICAL_MACHINE_COLLECTION.md`](file:///c:/Users/Aadinadh%20S%20M/Downloads/cuuuu/docs/PHYSICAL_MACHINE_COLLECTION.md) | Comprehensive step-by-step guide for human operators collecting telemetry on Machines B and C. |
| **Unit Test Suite** | [`test_physical_deployment.py`](file:///c:/Users/Aadinadh%20S%20M/Downloads/cuuuu/tests/test_physical_deployment.py) | 12 automated unit tests covering inventory, pre-flight checks, manifests, packaging, tamper detection, zip slips, collision rejection, and readiness logic. |

---

## 3. Test Suite Verification

The full test suite was executed across the entire repository:
```text
Ran 64 tests in 25.919s
OK
```

Breakdown of test coverage (64 total tests):
- `tests/test_collector.py` (3 tests): 2 Hz sampling, non-blocking process worker, timestamp monotonicity.
- `tests/test_features.py` (6 tests): 12 hardware-normalized features, empty window handling, numerical stability.
- `tests/test_stress.py` (3 tests): CPU, memory headroom ($\ge 1.5$ GB), disk write cleanup.
- `tests/test_dataset_builder.py` (2 tests): Provenance tagging, window step indexing.
- `tests/test_baseline_rules.py` (5 tests): Deterministic heuristic classification, explainable rule scores.
- `tests/test_model_trainer.py` (19 tests): Grouped CV, sigmoid probability calibration, out-of-fold 95th percentile abnormality thresholding, leakage guards.
- `tests/test_physical_dataset_validator.py` (8 tests): Rules A through P physical integrity checks.
- `tests/test_multimachine_lomo.py` (6 tests): Synthetic LOMO baseline and multi-machine validation logic.
- `tests/test_physical_deployment.py` (12 tests): Manifest creation, packaging, tamper detection, zip slip safety, session collision guards, LOMO readiness evaluation.

---

## 4. Current Hardware & Dataset Status

### Machine A (`physical_machine_A`)
- **Status**: COMPLETE & FROZEN.
- **Hardware Profile**:
  - Hostname: `LAPTOP-I0G3I8PA`
  - OS: Windows 11 (10.0.26200)
  - CPU: AMD Ryzen 7 (16 logical, 8 physical cores)
  - RAM: 15.82 GB (Swap: 18.0 GB)
- **Dataset**:
  - 22 completed sessions (4 Normal, 6 CPU Pressure, 6 Memory Pressure, 6 Disk I/O Pressure).
  - 242 feature windows.
  - Manifest created: [`physical_machine_A_manifest.json`](file:///c:/Users/Aadinadh%20S%20M/Downloads/cuuuu/data/physical_metadata/physical_machine_A_manifest.json).
  - Export archive created: `data/exports/physical_machine_A_curio_dataset.zip` (0.11 MB, 45 files, verified SHA-256).
- **Single-Machine Benchmark Performance**:
  - Random Forest Accuracy: **95.87%** (232/242 windows)
  - Macro F1: **0.9518** (vs. Heuristic Baseline: 0.6567)
  - Out-of-fold Abnormality Threshold: **0.7816**

### Machine B (`physical_machine_B`)
- **Status**: NOT YET AVAILABLE IN THIS ENVIRONMENT.
- **Onboarding Readiness**: Fully automated via `scripts/setup_physical_machine.ps1` and `docs/PHYSICAL_MACHINE_COLLECTION.md`.
- **Target Protocol**: 22 sessions (4 Normal, 6 CPU, 6 Memory, 6 Disk).

### Machine C (`physical_machine_C`)
- **Status**: NOT YET AVAILABLE IN THIS ENVIRONMENT.
- **Onboarding Readiness**: Fully automated via `scripts/setup_physical_machine.ps1` and `docs/PHYSICAL_MACHINE_COLLECTION.md`.
- **Target Protocol**: 22 sessions (4 Normal, 6 CPU, 6 Memory, 6 Disk).

---

## 5. Physical LOMO Readiness Assessment

Executing `python scripts/check_lomo_readiness.py` produces the following verified output:
```text
======================================================================
CURIO PHYSICAL LOMO READINESS ASSESSMENT
======================================================================

Status: TRUE PHYSICAL LOMO NOT READY
Reason: Fewer than 3 physical machines available (Found: 1 / 3).

Verified Machines Present (1):
  [+] physical_machine_A: 22 sessions, 242 windows, conditions: {'cpu_pressure': 6, 'disk_io_pressure': 6, 'memory_pressure': 6, 'normal': 4}

Missing Target Machines (2):
  [-] physical_machine_B
  [-] physical_machine_C

======================================================================
True cross-machine LOMO cannot be executed until datasets for all 3
physical machines are collected and imported.
DO NOT fabricate synthetic data or alias existing machines.
======================================================================
```

---

## 6. Security, Integrity & Anti-Fabrication Safeguards

1. **Anti-Masquerade / Clone Defense**:
   If an operator attempts to run `inventory_physical_machines.py`, `check_physical_machine.py`, or `collect_physical_dataset.py` with `--machine_id physical_machine_B` on Machine A's hardware (`LAPTOP-I0G3I8PA`), the tools immediately detect the hostname collision, print a warning, and abort.
2. **Cryptographic Tamper Detection**:
   Every dataset export packages a `checksums.sha256` manifest. The importer recalculates the SHA-256 hash of each file in-memory before writing to disk. Modifying even a single byte triggers a `DatasetImportError` and aborts.
3. **Session Collision Guardrails**:
   The importer verifies that none of the incoming session UUIDs collide with any existing session from another machine.
4. **Zero-Leakage Invariance**:
   All 12 telemetry features are strictly hardware-normalized (ratios, percentages, differences, log-compressed activity rates). Out-of-fold thresholding ensures test machines never influence calibration or decision boundaries.

---

## 7. Next Step

Transfer the CURIO repository to the first external physical computer (`physical_machine_B`) and follow the steps in [`PHYSICAL_MACHINE_COLLECTION.md`](file:///c:/Users/Aadinadh%20S%20M/Downloads/cuuuu/docs/PHYSICAL_MACHINE_COLLECTION.md) to record and import its 22 telemetry sessions.
