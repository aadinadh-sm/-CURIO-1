# CURIO Live Diagnosis Validation Report

**Project:** CURIO — Intelligent Machine Learning Discovery and Computer Diagnosis System  
**Session ID:** `curio_session_20260927_191901_935030`  
**Execution Timestamp:** 2026-09-27T19:19:01.460643+00:00  
**Mode:** Physical Live Telemetry Capture (30 seconds @ 2 Hz)

---

## 1. System Integration Verification

This test performs genuine end-to-end physical hardware telemetry capture on the live host computer to verify that all pipeline stages function synchronously in production.

| Metric / Checkpoint | Expected Value | Observed Physical Value | Status |
|:---|:---:|:---:|:---:|
| **Capture Duration** | 30.0s ($\pm 1.0\text{s}$) | 30.6s | **VERIFIED** |
| **Raw Samples Collected** | Exactly 61 | Exactly 61 | **VERIFIED** |
| **Sampling Frequency** | 2.0 Hz (0.5s interval) | 1.99 Hz | **VERIFIED** |
| **Feature Extraction Windows** | Exactly 11 | Exactly 11 | **VERIFIED** |
| **Feature Dimensionality** | Exactly 12 features/window | Exactly 12 features/window | **VERIFIED** |
| **NaN / Inf Invariant** | 0 NaN, 0 Inf | 0 NaN, 0 Inf | **VERIFIED** |
| **ML Inference Latency** | $< 500\text{ ms}$ | 160.1 ms | **VERIFIED** |
| **Abnormality Evaluation** | Evaluated vs 0.8536 | Mean: 0.8233 (Normal) | **VERIFIED** |
| **Evidence Attribution** | Top $\le 3$ items | Generated in 2.7 ms | **VERIFIED** |
| **Discovery Analysis** | Valid fallback / progression | `INSUFFICIENT_TEMPORAL_EVIDENCE` | **VERIFIED** |
| **History Persistence** | Valid JSON in `data/diagnosis_history/` | Persisted cleanly (18.8 KB) | **VERIFIED** |
| **Filesystem Cleanliness** | 0 temporary/partial files | 0 lingering files | **VERIFIED** |

---

## 2. Diagnostic Output Summary

- **Session Abnormality**: `NORMAL` (Mean abnormality score `0.8233` remained below the out-of-fold threshold `0.8536`).
- **Model Confidence**: 74.1% for candidate memory condition (due to standard background desktop RAM utilization), but correctly filtered to Normal Operating State by the out-of-fold abnormality boundary.
- **Top Evidence Items**:
  1. `swap_used_pct`: 5.60% (Normal Reference: 5.87%, $z\text{-dir} = +6.28$, strong evidence).
  2. `ram_available_ratio`: 0.18 (Normal Reference: 0.22, $z\text{-dir} = +1.83$, moderate evidence).
  3. `ram_used_pct`: 82.22% (Normal Reference: 78.15%, $z\text{-dir} = +1.82$, moderate evidence).
- **Discovery Status**: `INSUFFICIENT_TEMPORAL_EVIDENCE` (No temporal escalation observed; signals remained in equilibrium).
- **Interpretation**: *"Insufficient temporal evidence for sequence analysis."* (Strict non-causal fallback behavior).

---

## 3. Performance Breakdown

```
--------------------------------------------------
  Telemetry collection: 30.6s
  Feature extraction:   14.9 ms
  ML inference:         160.1 ms
  Evidence analysis:    2.7 ms
  Discovery analysis:   1.3 ms
  Reporting latency:    1.9 ms
  Total analysis time:  178.9 ms
--------------------------------------------------
```

> **Note**: This physical live run confirms seamless integration of all pipeline stages. As stipulated in the methodological rules, this single live run serves solely as an integration validation checkpoint and is NOT treated as a new statistical generalization benchmark.
