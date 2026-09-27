# CURIO: Physical Machine Telemetry Collection & Onboarding Runbook

This runbook guides human operators through setting up, verifying, collecting, packaging, and importing physical telemetry from **physical_machine_B** and **physical_machine_C** for the CURIO capstone project.

---

## 1. Experimental Protocol & Frozen Architecture

All participating machines strictly follow the frozen CURIO telemetry protocol:

| Parameter | Specification | Purpose |
| :--- | :--- | :--- |
| **Target Classes (4)** | `normal`, `cpu_pressure`, `memory_pressure`, `disk_io_pressure` | System state classification |
| **Telemetry Features (12)** | `cpu_mean`, `cpu_max`, `cpu_std`, `cpu_core_imbalance`, `ram_used_pct`, `ram_available_ratio`, `swap_used_pct`, `disk_io_rate_norm`, `disk_iops_norm`, `process_count_delta`, `top_proc_cpu_ratio`, `top_proc_mem_pct` | Hardware-normalized representations |
| **Sampling Rate** | 2.0 Hz (0.50 s tick) | Sub-second state dynamics |
| **Session Duration** | 30.0 seconds | Controlled capture window |
| **Raw Samples / Session** | Exactly 61 samples | Complete 30-second series |
| **Feature Window** | 5.0 s duration, 2.5 s step | Moving window representation |
| **Feature Windows / Session**| Exactly 11 windows | Standardized temporal resolution |
| **Sessions per Machine** | Exactly 22 sessions | 4 Normal, 6 CPU, 6 Memory, 6 Disk |
| **Feature Windows / Machine**| Exactly 242 windows | Standardized dataset per machine |

> [!IMPORTANT]
> **PHYSICALLY DISTINCT HARDWARE MANDATE**:
> Leave-One-Machine-Out (LOMO) evaluation requires genuine cross-machine variance. Under no circumstances should virtual machines, emulators, or the same physical host under different aliases be used. Machine A (`LAPTOP-I0G3I8PA`, AMD Ryzen 7 16-core, 16 GB RAM) is already complete and frozen. Machines B and C must be distinct physical computers.

---

## 2. Hardware & Software Prerequisites

Target Machine Requirements:
- **Operating System**: Windows 10 or 11 (64-bit).
- **CPU**: Multi-core processor (x86_64 / AMD64).
- **Memory**: Minimum 8 GB physical RAM, with at least **1.5 GB available RAM headroom** during testing.
- **Storage**: At least 5 GB free disk space on `C:\`.
- **Python**: Python 3.10, 3.11, 3.12, 3.13, or 3.14 (x64).
- **Shell**: PowerShell 5.1+ or PowerShell Core (`pwsh`).

---

## 3. Step-by-Step Operator Walkthrough

### Step 1: Deploy Repository to Target Machine
Copy or clone the CURIO repository to the target physical computer:
```powershell
git clone <repo_url> curio
cd curio
```

### Step 2: Environment Setup
Execute the automated setup script in PowerShell:
```powershell
pwsh ./scripts/setup_physical_machine.ps1
```
This script verifies Python 3.x, installs required dependencies (`psutil`, `scikit-learn`, `numpy`, `pandas`, `joblib`), creates all data directories (`data/physical_raw`, `data/physical_processed`, `data/physical_metadata`, `data/exports`, `reports`), and performs basic import tests.

### Step 3: Hardware Fingerprint & Inventory
Fingerprint the machine to catalog its hardware profile:
```powershell
# For Machine B:
python scripts/inventory_physical_machines.py --machine_id physical_machine_B

# For Machine C:
python scripts/inventory_physical_machines.py --machine_id physical_machine_C
```
*Note: If run on the Machine A host, this script will immediately detect the hostname collision and reject registration.*

### Step 4: Pre-Flight Verification & Smoke Testing
Before launching the full 22-session protocol, run the automated non-destructive pre-flight test:
```powershell
# For Machine B:
python scripts/check_physical_machine.py --machine_id physical_machine_B

# For Machine C:
python scripts/check_physical_machine.py --machine_id physical_machine_C
```
This performs:
1. Host identity & isolation check.
2. Environment and library import check.
3. Directory write permission verification.
4. Stress safety check ($\ge 1.5$ GB available RAM).
5. 5-second `TelemetryCollector` test (checks 2 Hz monotonic ticks, PID 0 exclusion).
6. 12-feature extractor test (verifies 0 NaNs/Infs).

You must see:
```text
======================================================================
PRE-FLIGHT VALIDATION PASSED
Machine 'physical_machine_B' is fully compatible and safe for CURIO collection.
======================================================================
```

### Step 5: Full 22-Session Telemetry Collection
Launch the collection protocol:
```powershell
# For Machine B:
python scripts/collect_physical_dataset.py --machine_id physical_machine_B

# For Machine C:
python scripts/collect_physical_dataset.py --machine_id physical_machine_C
```
The runner will execute:
- **4 Normal sessions**: With varied natural workloads (`idle_desktop`, `browser_tabs`, `ide_editor_activity`, `video_playback`).
- **6 CPU Pressure sessions**: 2 Low, 2 Medium, 2 High.
- **6 Memory Pressure sessions**: 2 Low, 2 Medium, 2 High (enforcing $\ge 1.5$ GB RAM headroom).
- **6 Disk I/O Pressure sessions**: 2 Low, 2 Medium, 2 High (sync direct file writes).

*Resume Support*: If interrupted, re-running the script automatically skips sessions already recorded.

### Step 6: Generate Dataset Manifest
Once all 22 sessions are complete, generate the signed manifest:
```powershell
python scripts/create_machine_manifest.py --machine_id physical_machine_B
```
This produces `data/physical_metadata/physical_machine_B_manifest.json` with full hardware specifications, environment versions, and session catalogs.

### Step 7: Package Dataset for Export
Bundle raw sessions, processed feature windows, the manifest, and cryptographic SHA-256 checksums into an export archive:
```powershell
python scripts/package_machine_dataset.py --machine_id physical_machine_B
```
This runs pre-packaging validation across all sessions and outputs:
`data/exports/physical_machine_B_curio_dataset.zip`

### Step 8: Transfer Archive to Analysis Machine
Copy `physical_machine_B_curio_dataset.zip` (and subsequently `physical_machine_C_curio_dataset.zip`) to the primary analysis computer (where Machine A data resides).

### Step 9: Import & Cryptographically Verify Dataset
On the primary analysis computer, import and verify the dataset:
```powershell
# First run dry-run verification:
python scripts/import_machine_dataset.py data/exports/physical_machine_B_curio_dataset.zip --dry_run

# Execute full verified import:
python scripts/import_machine_dataset.py data/exports/physical_machine_B_curio_dataset.zip
```
The importer performs:
1. Zip slip / path traversal defense.
2. SHA-256 validation of every file against `checksums.sha256`.
3. Manifest schema check.
4. Cross-machine session ID collision check.
5. Rules A through P physical dataset validation.
6. Automatic cataloging into `data/physical_metadata/machine_inventory.json`.

Repeat Steps 1–9 for `physical_machine_C`.

---

## 4. Verifying True Physical LOMO Readiness

After importing Machine B and Machine C onto the analysis workstation, check the readiness status:
```powershell
python scripts/check_lomo_readiness.py
```

### When Not Ready (Current State):
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
```

### When Ready (After Importing B & C):
```text
======================================================================
CURIO PHYSICAL LOMO READINESS ASSESSMENT
======================================================================
Status: TRUE PHYSICAL LOMO READY
All 3 physical machines (physical_machine_A, physical_machine_B, physical_machine_C)
have complete, verified datasets ready for Leave-One-Machine-Out evaluation.
======================================================================
```

---

## 5. Troubleshooting & Safety Guardrails

- **Insufficient RAM Error**: If `check_physical_machine.py` reports available RAM $< 1.5$ GB, close resource-heavy background processes (e.g. game launchers, virtual machines) before proceeding.
- **Disk Space**: Disk stress generates temporary files in `data/disk_stress_scratch/` and removes them immediately upon session completion. Ensure $\ge 3$ GB free disk space.
- **Host Collision Warning**: If you see `CRITICAL WARNING: HOSTNAME HARDWARE COLLISION DETECTED!`, you are running on Machine A's hardware. You must deploy to a separate physical machine.
