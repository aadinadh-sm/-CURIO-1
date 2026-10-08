# CURIO — Intelligent Machine Learning Discovery and Computer Diagnosis System

CURIO is a local machine-learning system that observes computer telemetry, diagnoses probable operating conditions, explains the supporting evidence, and discovers temporal patterns in system behavior.

---

## 1. What CURIO Is
CURIO is an undergraduate capstone engineering project delivering a transparent, local-first computer diagnostic assistant. Unlike black-box system monitors or intrusive cloud-connected telemetry agents, CURIO captures short, high-resolution operating snapshots, applies statistically calibrated machine learning models to identify resource pressure conditions, provides concrete evidence explaining its inference, and models the temporal onset sequence of subsystem degradation.

## 2. Problem Statement
Diagnosing computer slowdowns, resource contention, and anomalous operating behavior is challenging for both everyday users and system administrators:
- Traditional Task Managers display raw, instantaneous resource percentages without context or historical trajectory.
- Enterprise Application Performance Monitoring (APM) tools require cloud synchronization, subscription accounts, and extensive background overhead, raising significant privacy and security concerns.
- Users are left asking: *"Why is my computer slow, what changed first, and what evidence supports this conclusion?"*

## 3. Core Idea
CURIO addresses this challenge through a transparent, privacy-preserving multi-stage pipeline:
1. **Standardized Capture**: Captures exactly 30 seconds of system telemetry at 2 Hz (61 discrete snapshots).
2. **Rolling Temporal Windows**: Computes 12 statistical features across 11 overlapping 5-second windows (2.5-second step).
3. **Calibrated Inference**: Classifies operating states using a Random Forest model calibrated via Platt sigmoid scaling.
4. **Out-of-Fold Abnormality Filtering**: Evaluates session non-normality against an empirical 95th percentile threshold (`0.8536`), preventing false alarms.
5. **Evidence Attribution**: Quantifies *why* the diagnosis was made using z-scores and directional alignment relative to learned Normal reference centroids.
6. **Temporal Discovery**: Discovers *when* and *in what sequence* subsystem signals disrupted by ranking onset events and calculating Kendall's tau-b ($\tau$) rank correlation against canonical failure progressions.

---

## 4. Architecture

```text
                     CURIO WEB UI (React 19 + TypeScript)
                                      ↓
                         LOCAL REST API (FastAPI)
                           (Strictly 127.0.0.1:8000)
                                      ↓
                          CURIO DIAGNOSIS PIPELINE
                                      ↓
                 ┌────────────────────┼────────────────────┐
                 ↓                    ↓                    ↓
          ML DIAGNOSIS         EVIDENCE ENGINE      DISCOVERY ENGINE
      (Calibrated RF Model)   (Z-score Attribution) (Kendall Tau-b Onset)
                 ↓                    ↓                    ↓
                 └────────────────────┼────────────────────┘
                                      ↓
                           FINAL DIAGNOSTIC REPORT
                                      ↓
                        LOCAL HISTORY (JSON Storage)
```

---

## 5. Machine Learning Methodology
- **Target Conditions (4 Frozen Classes)**:
  1. `normal` — Balanced baseline system operation.
  2. `cpu_pressure` — Compute-bound execution, thread saturation, or core imbalance.
  3. `memory_pressure` — Physical RAM exhaustion, working set eviction, or pagefile swapping.
  4. `disk_io_pressure` — Heavy read/write throughput saturation and IOPS queuing.
- **12 Frozen Features**:
  `cpu_mean`, `cpu_std`, `cpu_max`, `cpu_core_imbalance`, `top_proc_cpu_ratio`, `ram_used_pct`, `ram_available_ratio`, `swap_used_pct`, `disk_io_rate_norm`, `disk_iops_norm`, `process_count_delta`, `top_proc_mem_pct`.
- **Model Architecture**: Ensemble of 100 Decision Trees (`RandomForestClassifier`) calibrated with sigmoid probability scaling.
- **Validation**: Leave-One-Machine-Out (LOMO) and Group-K-Fold cross-validation grouped strictly by capture session.

---

## 6. Evidence Engine
The Evidence Engine answers: *"Why did CURIO make this diagnosis?"*
- Evaluates the representative strongest window during the capture.
- Computes standardized distance ($z$-score) against the training-only Normal reference profile ($\mu_{normal}, \sigma_{normal}$).
- Evaluates directional consistency (+1 for expected elevation, -1 for expected depression).
- Formulates structured evidence items:
  - Observed value vs Normal reference baseline.
  - Direction indicator (`↑ Elevated` or `↓ Depressed`).
  - Evidence strength (`Strong`, `Moderate`, `Inconclusive`, `Contradictory`).
- **Core Principle**: Model Confidence (%) is explicitly decoupled from Statistical Evidence Strength.

---

## 7. Discovery Engine
The Discovery Engine answers: *"What was the temporal progression of subsystem disruption?"*
- Evaluates the 11-window temporal trajectory to detect the earliest onset time for each canonical feature ($t_{onset}$ at $1.5\sigma$ deviation).
- Orders observed onsets and computes Kendall's tau-b ($\tau$) rank correlation against canonical benchmark progressions.
- Gracefully handles non-transition states:
  - `OPERATING_EQUILIBRIUM`: Normal operation, no escalation detected.
  - `SUSTAINED_PRESSURE`: Elevated stress was already present at window 0 and sustained throughout; no transition observed during the 30-second capture.
  - `INSUFFICIENT_TEMPORAL_EVIDENCE`: Less than 2 distinct canonical features transitioned during the capture.

---

## 8. Live Diagnosis Flow
1. User clicks **Diagnose My Computer**.
2. Background worker samples host CPU, memory, disk, and process telemetry at 2 Hz for 30.0 seconds (61 raw samples).
3. 11 rolling feature windows are extracted (~6 ms).
4. Calibrated Random Forest infers condition probabilities across all 11 windows (~160 ms).
5. Abnormality detector compares mean non-normal probability against the 0.8536 threshold.
6. Evidence Engine attributes top supporting features (~2 ms).
7. Discovery Engine analyzes onset order and calculates Kendall $\tau_b$ (~2 ms).
8. Diagnostic report is persisted locally to `data/diagnosis_history/` and rendered in the web UI.

## 8a. Diagnose an Existing CSV Capture
From the home screen, choose a CSV or select one of the built-in Normal, CPU pressure, Memory pressure, or Disk I/O examples. CURIO validates the file, then sends its 61 raw samples through the same feature extraction, calibrated inference, evidence attribution, and temporal discovery stages used for live capture. The complete report opens in the normal results view and is added to local diagnosis history.

Uploads must be a UTF-8 CURIO raw telemetry CSV with exactly 61 rows spanning 30 seconds at approximately 0.5-second intervals. Required fields are `timestamp`, `cpu_overall`, `cpu_cores_json`, `vmem_percent`, `vmem_available`, `vmem_total`, `swap_percent`, `swap_used`, `swap_total`, `disk_read_bytes`, `disk_write_bytes`, `disk_read_count`, `disk_write_count`, `disk_read_time`, `disk_write_time`, `process_count`, `top_proc_cpu`, and `top_proc_rss`. CURIO analyzes the upload in memory and does not save the raw CSV; only the resulting diagnosis report is stored locally. Processed feature tables are not accepted as raw captures because they cannot provide the original per-sample timing needed for evidence and trajectory discovery.

---

## 9. Installation & Prerequisites

### Prerequisites
- Python 3.10+ (tested on Python 3.14)
- Node.js 18+ and npm 9+ (tested on Node v22.18.0)

### Setup
```bash
# 1. Clone repository
git clone https://github.com/aadicybersec-glitch/CURIO.git
cd CURIO

# 2. Install Python dependencies
pip install -r requirements.txt

# 3. Install Frontend dependencies & build
cd frontend
npm install
npm run build
cd ..
```

---

## 10. Running CURIO

Launch both the backend API and frontend with a single command:
```bash
python scripts/run_curio.py
```

Endpoints available immediately:
- **Backend API**: `http://127.0.0.1:8000`
- **Frontend UI (Static Build)**: `http://127.0.0.1:8000`
- **Frontend UI (Vite Dev Server)**: `http://127.0.0.1:3000`

---

## 11. Replay Mode
For demonstration and evaluation without waiting for the 30-second physical capture:
1. Click **Replay Mode** in the navigation header or home screen.
2. Select any condition: `Normal`, `CPU Pressure`, `Memory Pressure`, or `Disk I/O Pressure`.
3. CURIO instantly replays verified physical telemetry from `data/physical_raw/` through the live feature extraction, ML, evidence, and discovery engines in under 400 ms.
4. The UI clearly displays a prominent `REPLAY / DEMO MODE` badge to preserve scientific integrity.

---

## 12. Technical Mode
Designed for examiners, viva defense, and engineers:
- Click **Technical Mode** in the navigation bar or results view.
- Exposes:
  - All 12 feature values for the representative window.
  - Complete 11-window rolling values matrix.
  - Calibrated class probability distributions.
  - Abnormality score vs deployment threshold (`0.8536`).
  - Directional scores and z-scores relative to Normal reference centroids.
  - Canonical sequence, observed sequence, and Kendall $\tau_b$.
  - Per-stage execution latencies (capture, extraction, inference, evidence, discovery).

---

## 13. Testing & Verification

Run the complete test suite:
```bash
# Run all 130 Python unit and integration tests
python -m unittest discover tests

# Run all 10 frontend Vitest integration tests
cd frontend && npm test && cd ..

# Run the automated end-to-end acceptance test (checks A through N)
python scripts/final_acceptance_test.py
```

---

## 14. Empirical Results

### Physical Machine Evaluation
Validated on physical computer Machine A across 22 independent capture sessions under diverse background workloads:
- **Total Feature Windows**: 242
- **Random Forest Accuracy**: **95.87%**
- **Random Forest Macro F1**: **0.9518**
- **Macro Precision**: **0.9659**
- **Macro Recall**: **0.9451**

### Comparison with Deterministic Baseline
- **Heuristic Rule-Based Baseline Macro F1**: **0.6567**
- **Random Forest Performance Delta**: **+0.2951 Macro F1 improvement** (+44.9% relative increase).

---

## 15. Known Limitations
1. **Single-Machine Evaluation**: Empirical validation was conducted on physical Machine A. Cross-machine hardware generalization has not been evaluated, and universal generalization across arbitrary hardware architectures is explicitly not claimed.
2. **Fixed 30-Second Window**: Telemetry capture is constrained to a 30-second duration at 2 Hz (61 discrete samples). Sub-second transient spikes (< 500 ms) fall below the Nyquist-Shannon sampling threshold.
3. **Non-Causal Statistical Association**: CURIO identifies empirical statistical correlations and temporal sequences; it does not claim to establish definitive hardware or kernel root causality.
4. **Local-Only Boundary**: CURIO operates strictly on `127.0.0.1`. It does not provide remote fleet management or cloud monitoring.

---

## Product Website and Windows Preview

The repository also contains the CURIO product website in `frontend/`. It explains the local diagnostic workflow and links to the Windows preview bundle. The hosted site is a product/download page; computer diagnosis still runs on the user's own computer.

### Run the Windows preview

1. Download `curio-windows-preview.zip` from the latest GitHub Release and extract it.
2. Double-click `START-CURIO.bat`.
3. The first launch creates a private Python environment and installs dependencies. It needs Python 3.12 or 3.13 and an internet connection once. Administrator access is not required.
4. CURIO opens the diagnostic app at `http://127.0.0.1:8000/app/`. The product page at `/` is optional; it is not a second app to launch. The local backend serves the diagnostic interface and its API together.

This is a setup bundle, not a signed Windows installer.

### Deploy the product website with Vercel

Import this repository and set the Vercel **Root Directory** to `frontend`. Vercel will build the Vite product site and publish `dist/`. The diagnostic interface is included in the local Windows bundle; it needs CURIO's on-device API and is not a hosted computer-monitoring service.

### Publish a Windows preview release

Push a version tag such as `v1.0.2`. The GitHub Actions workflow at `.github/workflows/release.yml` builds the product pages and packages the local app, model, example telemetry, and Windows launcher into `curio-windows-preview.zip`, then attaches it to a GitHub Release. The website download link always points to the latest release asset.
