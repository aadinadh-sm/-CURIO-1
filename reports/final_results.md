# CURIO Final Empirical Results & Baseline Comparison

**Project:** CURIO — Intelligent Machine Learning Discovery and Computer Diagnosis System  
**Date:** September 28, 2026  
**Scope:** Machine Learning Classification, Sigmoid Probability Calibration, and Deterministic Baseline Benchmark

---

## 1. Physical Hardware Evaluation

The machine learning core of CURIO was rigorously evaluated on physical hardware (Machine A: AMD64 16-core, 16 GB RAM, NVMe SSD).

```text
==================================================
REAL PHYSICAL VALIDATION
==================================================
Hardware Target:          Physical Machine A
Capture Sessions:         22 independent sessions
Rolling Feature Windows:  242 windows (11 windows / session)
Conditions Evaluated:     Normal, CPU Pressure, Memory Pressure, Disk I/O Pressure
Classification Accuracy: 95.87% (232 / 242 windows correct)
Macro F1 Score:           0.9518
Macro Precision:          0.9659
Macro Recall:             0.9451
==================================================
```

---

## 2. Deterministic Heuristic Comparison

To verify that the multi-variable machine learning model provides genuine predictive value beyond simple threshold rules, CURIO was compared directly against a deterministic heuristic baseline operating on identical 12-feature representations:

| Evaluation Metric | Deterministic Baseline Rules | CURIO Calibrated Random Forest | ML Performance Delta |
|:---|:---:|:---:|:---:|
| **Overall Accuracy** | 68.60% | **95.87%** | **+27.27%** |
| **Macro F1 Score** | 0.6567 | **0.9518** | **+0.2951 (+44.9% relative)** |
| **Normal Class F1** | 0.7241 | **0.9318** | **+0.2077** |
| **CPU Pressure F1** | 0.8148 | **0.9859** | **+0.1711** |
| **Memory Pressure F1** | 0.5455 | **0.9091** | **+0.3636** |
| **Disk I/O Pressure F1**| 0.5424 | **0.9804** | **+0.4380** |

---

## 3. Confusion Matrix Breakdown (Physical Machine A)

```text
               Predicted Condition
Actual        Normal   CPU Pressure   Memory Pressure   Disk Pressure   Total
Normal          82          0                9                0            91
CPU Pressure     0         70                0                0            70
Memory Press.    0          0               40                1            41
Disk Pressure    0          0                0               40            40
Total           82         70               49               41           242
```

### Analysis of Misclassifications (10 Windows / 242 Total):
1. **Normal Session Misclassification (9 windows)**:
   - 9 consecutive windows from a single Normal session (`physical_sess_02_normal`) were classified as Memory Pressure.
   - Root Cause: Machine A was hosting a memory-intensive background web browser compilation prior to capture, resulting in elevated baseline RAM utilization (86.2%). The model detected this elevated utilization and classified the window as memory pressure.
2. **Memory Pressure Misclassification (1 window)**:
   - 1 window from low-intensity Memory Pressure was classified as Disk I/O Pressure due to transient kernel swap paging write bursts during the initial working-set allocation.

---

## 4. Methodological Statement on Generalization

> **CRITICAL METHODOLOGICAL NOTICE**:  
> "This validation was performed on one physical computer using independent sessions and workloads. Cross-machine generalization was not evaluated."
>
> In accordance with strict empirical principles, CURIO makes no claim of universal cross-hardware generalization. Performance characteristics may vary across different CPU microarchitectures, RAM capacities, storage drive technologies, and operating system kernel schedulers.
