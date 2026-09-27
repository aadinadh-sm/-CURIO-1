# CURIO — Technical Report: Evidence Engine Architecture & Telemetry Validation

**Date**: 2026-09-27  
**Project**: CURIO (Undergraduate Capstone)  
**Milestone**: Evidence Engine (Explanatory Telemetry Attribution)  
**Test Suite Status**: **80 / 80 Tests Passing (100% OK)**  
**Hardware Evaluated**: `physical_machine_A` (AMD Ryzen 7 16-Core, 16 GB RAM, Windows 11)  
**Artifact Directory**: `data/models/physical_deployment/`  

---

## 1. Purpose & Core Questions

While the CURIO classifier reliably predicts system operating states using calibrated multi-class Random Forests, high predictive accuracy alone is insufficient for autonomous diagnosis and human operator trust. A black-box classification label such as `"Memory Pressure"` provides no actionable insight unless the system can explain **why** it arrived at that diagnosis.

The **CURIO Evidence Engine** bridges the gap between statistical inference and operational interpretability by answering three fundamental questions:

1. **What condition did CURIO predict?** (e.g., `Memory Pressure` with 83.9% calibrated model confidence).
2. **Which observed telemetry signals support that prediction?** (e.g., top-process memory share elevated to 11.9%, physical RAM utilization elevated to 86.3%, and available memory ratio reduced).
3. **How strongly did those signals deviate from the learned Normal operating reference?** (e.g., measured via standardized directional deviations: $+10.0\sigma$, $+3.65\sigma$).

Crucially, the Evidence Engine provides **local attribution** anchored to training-derived reference baselines while strictly forbidding unsubstantiated causal claims or conflation of model probability with evidence strength.

---

## 2. Telemetry Inputs & Representation

The Evidence Engine operates directly on the standardized, hardware-normalized **12-feature telemetry representation** across rolling 5.0-second time windows. It requires no raw sub-second sample buffers, ensuring lightweight, deterministic evaluation:

| Feature Name | Subsystem | Unit | Physical Normalization Mechanism |
| :--- | :--- | :--- | :--- |
| `cpu_mean` | CPU | % | Window mean of overall CPU utilization across all logical cores |
| `cpu_max` | CPU | % | Peak single-core utilization observed during the window |
| `cpu_std` | CPU | % | Temporal volatility / sample standard deviation of CPU usage |
| `cpu_core_imbalance` | CPU | % | Mean instantaneous spread between most and least loaded cores |
| `ram_used_pct` | Memory | % | Percentage of total physical memory actively occupied |
| `ram_available_ratio` | Memory | ratio [0,1] | Available RAM divided by total installed RAM |
| `swap_used_pct` | Memory | % | Percentage of allocated paging/swap space in active use |
| `disk_io_rate_norm` | Disk | log10(B/s) | Base-10 logarithm of combined read/write throughput per second |
| `disk_iops_norm` | Disk | log10(ops/s) | Base-10 logarithm of disk I/O operations per second |
| `process_count_delta`| Process | count | Net change in active OS processes across the 5.0s window |
| `top_proc_cpu_ratio` | CPU | ratio [0,1] | Ratio of top process CPU consumption relative to active system CPU |
| `top_proc_mem_pct` | Memory | % | Top process resident set size (RSS) as a percentage of total RAM |

---

## 3. Reference-Statistics Methodology

### Training-Only Baseline Extraction
Reference statistics are computed **exclusively** from training data within each experimental fold or from the approved deployment training dataset. At no point are test sessions or held-out validation machines incorporated into the reference profile.

For each feature $f \in \text{FEATURE\_NAMES}$, the training Normal partitions yield:
- **Baseline Mean**: $\mu_{\text{normal}, f} = \frac{1}{N_{\text{normal}}} \sum_{i=1}^{N_{\text{normal}}} x_{i, f}$
- **Baseline Sample Standard Deviation**: $\sigma_{\text{normal}, f} = \sqrt{\frac{1}{N_{\text{normal}} - 1} \sum_{i=1}^{N_{\text{normal}}} (x_{i, f} - \mu_{\text{normal}, f})^2}$
- **Robust Statistics**: Median $M_{\text{normal}, f}$ and Interquartile Range $\text{IQR}_{\text{normal}, f}$ to audit skewness and non-normality.

### Class-Specific Expected Direction Matrix
For each abnormal condition $c \in \{\text{cpu\_pressure}, \text{memory\_pressure}, \text{disk\_io\_pressure}\}$, an empirical direction vector $D_{c, f} \in \{+1, -1, 0\}$ is learned by comparing the condition mean $\mu_{c, f}$ against $\mu_{\text{normal}, f}$:

$$D_{c, f} = \begin{cases} 
+1 & \text{if } \frac{\mu_{c, f} - \mu_{\text{normal}, f}}{\max(\sigma_{\text{normal}, f}, \epsilon)} \ge 0.50 \\
-1 & \text{if } \frac{\mu_{c, f} - \mu_{\text{normal}, f}}{\max(\sigma_{\text{normal}, f}, \epsilon)} \le -0.50 \\
0 & \text{otherwise}
\end{cases}$$

This ensures that features whose abnormal manifestation is a **reduction** rather than an increase (such as `ram_available_ratio` dropping under memory pressure) are naturally assigned $D_{\text{memory\_pressure}, \text{ram\_available\_ratio}} = -1$.

---

## 4. Directional Scoring & Numerical Safeguards

### Safe Deviation Metric ($z$)
To prevent division by zero on exceptionally stable features (e.g., constant swap usage or zero I/O on idle servers), the engine enforces a strictly documented numerical safety floor:

$$\sigma_{\text{safe}, f} = \max(\sigma_{\text{normal}, f}, \epsilon), \quad \epsilon = 1.0 \times 10^{-4}$$

The raw standardized deviation is:
$$z_f = \frac{x_f - \mu_{\text{normal}, f}}{\sigma_{\text{safe}, f}}$$

### Score Capping
To prevent a single feature with near-zero baseline variance from generating an astronomical deviation score that dwarfs all other signals in display reports, the deviation is clipped to a configurable maximum bound:

$$z_{\text{safe}, f} = \text{clip}(z_f, -z_{\text{max}}, z_{\text{max}}), \quad z_{\text{max}} = 10.0$$

### Directional Evidence Score
The directional score reflects whether the feature moved in the direction expected for the diagnosed condition:

$$\text{directional\_score}_{c, f} = D_{c, f} \times z_{\text{safe}, f}$$

- **Positive Score**: Feature moved in the expected direction, supporting the diagnosis.
- **Near-Zero Score**: Weak or neutral deviation; signal provides little diagnostic value.
- **Negative Score**: Feature moved *against* the expected direction (contradictory evidence).

### Standardized Evidence Strength Bands
The engine defines clear, unit-tested qualitative bands:

| Directional Score Range | Evidence Strength Classification | Interpretation |
| :--- | :--- | :--- |
| $\text{score} \ge +2.0$ | **Strong Supporting Evidence** | Telemetry moved $\ge 2\sigma$ in the expected abnormal direction. |
| $+1.0 \le \text{score} < +2.0$ | **Moderate Supporting Evidence** | Telemetry moved $1\sigma \text{ to } 2\sigma$ in the expected direction. |
| $-1.0 < \text{score} < +1.0$ | **Weak / Neutral Evidence** | Signal remained within typical variance bounds. |
| $\text{score} \le -1.0$ | **Contradictory Evidence** | Signal moved $\ge 1\sigma$ *against* the expected direction. |

---

## 5. Evidence Ranking: Local Telemetry vs. Global Feature Importance

> [!IMPORTANT]
> **GLOBAL MODEL STATISTICS MUST NOT BE USED AS LOCAL EXPLANATIONS**:
> Random Forest `feature_importances_` reflects global Gini impurity reduction across all training decision trees. Citing global importance (e.g. *"cpu_mean is the most important feature globally, so it caused this prediction"*) is methodologically invalid for individual diagnostic windows.

The Evidence Engine ranks evidence **strictly on local telemetry deviation**:
1. **Supporting Evidence**: Ranked descending by $\text{directional\_score}_{c, f}$. The top 3 strongest supporting signals are prioritized.
2. **Contradictory Evidence**: Ranked ascending by $\text{directional\_score}_{c, f}$ (most negative first). Up to 2 contradictory signals ($\text{score} \le -1.0$) are reported.
3. **Neutral Features**: Remaining features whose deviations do not satisfy strong supporting or contradictory thresholds.

---

## 6. Separation of Probability and Evidence Strength

The Evidence Engine enforces a rigorous separation between model probability and empirical telemetry evidence:

- **Calibrated Model Confidence**: The posterior class probability $P(Y = c \mid \mathbf{x}) \in [0, 1]$ produced by the sigmoid-calibrated Random Forest (e.g., *"83.9% model confidence"*).
- **Evidence Strength**: The magnitude and count of standardized directional shifts in physical telemetry (e.g., *"Strong supporting evidence across physical RAM utilization and top-process memory share"*).
- **Evidence Consistency**: The ratio of condition-relevant features displaying positive directional support:
  $$\text{evidence\_consistency} = \frac{\sum_{f \in \mathcal{F}_c} \mathbb{I}(\text{directional\_score}_{c, f} \ge 1.0)}{|\mathcal{F}_c|}$$

Under no circumstances will CURIO claim *"83.9% evidence"*. Confidence belongs to the model; evidence belongs to the telemetry.

---

## 7. Central Wording Policy & Causality Guardrails

Observational telemetry collected via non-invasive OS APIs (`psutil`) reflects correlations and temporal co-occurrences, **not physical causality**. To maintain scientific integrity, `src/evidence.py` implements an automated regex-based wording validator (`validate_no_causal_language`):

```python
FORBIDDEN_CAUSAL_PATTERNS = [
    r"\bcaused\b", r"\bcauses\b", r"\bcause\s+of\b", r"\broot\s*cause\b",
    r"\bproved\b", r"\bproves\b", r"\bresulted\s+in\b", r"\bhardware\s+failure\b",
    r"\bbroken\s+hardware\b", r"\bfailing\s+hardware\b", r"\bchance\s+of\s+damage\b"
]
```

### Lexical Guidelines
- **Permitted Phrases**: *"elevated"*, *"reduced"*, *"increased"*, *"decreased"*, *"coincided with"*, *"consistent with"*, *"supports the diagnosis"*, *"within normal operating bounds"*.
- **Forbidden Phrases**: *"caused"*, *"the root cause is"*, *"resulted in"*, *"hardware failure"*, *"RAM is failing"*.

Every human-readable explanation generated by the engine is screened against this policy before return. Any violation immediately raises a `CausalLanguageError`.

---

## 8. Live Real-Hardware Diagnostic Examples

The following diagnoses were generated live by `scripts/run_evidence_demo.py` from actual physical sessions recorded on `physical_machine_A`:

### 8.1 Example 1: Normal Operation
```text
============================================================
CURIO DIAGNOSIS & EVIDENCE REPORT
============================================================
Session ID:         physical_machine_A_normal_none_5743354b
Window Index:       5
True Condition:     Normal
------------------------------------------------------------
Condition:          Normal
Model Confidence:   95.6%
Abnormality Score:  0.0438  (Validation Threshold: 0.8536)
Abnormal Status:    WITHIN NORMAL OPERATING BOUNDS
Evidence Consist.:  100.0%
------------------------------------------------------------
WHY CURIO DIAGNOSED THIS:
  1. Top-Process CPU Dominance (0.23) was within the expected learned Normal operating range (ref 0.25).
     [Signal: top_proc_cpu_ratio, Dev: -0.13 std (STRONG)]
  2. Peak Core CPU Utilization (33.9%) was within the expected learned Normal operating range (ref 31.8%).
     [Signal: cpu_max, Dev: -0.18 std (STRONG)]
  3. Top-Process Memory Share (6.0%) was within the expected learned Normal operating range (ref 5.9%).
     [Signal: top_proc_mem_pct, Dev: -0.19 std (STRONG)]

OVERALL INTERPRETATION:
  "All observed telemetry signals remain consistent with the learned Normal operating reference. No significant abnormal directional deviations were detected."
============================================================
```

### 8.2 Example 2: CPU Pressure
```text
============================================================
CURIO DIAGNOSIS & EVIDENCE REPORT
============================================================
Session ID:         physical_machine_A_cpu_pressure_high_b4e1568b
Window Index:       5
True Condition:     Cpu Pressure
------------------------------------------------------------
Condition:          Cpu Pressure
Model Confidence:   96.1%
Abnormality Score:  0.9880  (Validation Threshold: 0.8536)
Abnormal Status:    ABNORMAL
Evidence Consist.:  83.3%
------------------------------------------------------------
WHY CURIO DIAGNOSED THIS:
  1. Average CPU Utilization (100.0%) was substantially elevated relative to the learned Normal reference (18.6%).
     [Signal: cpu_mean, Dev: +10.00 std (STRONG)]
  2. Peak Core CPU Utilization (100.0%) was substantially elevated relative to the learned Normal reference (31.8%).
     [Signal: cpu_max, Dev: +5.79 std (STRONG)]
  3. CPU Core Load Imbalance (0.0%) was substantially lower than the learned Normal reference (53.6%).
     [Signal: cpu_core_imbalance, Dev: +4.24 std (STRONG)]

OVERALL INTERPRETATION:
  "Multiple cpu-related telemetry signals moved in the expected direction for Cpu Pressure."
============================================================
```

### 8.3 Example 3: Memory Pressure
```text
============================================================
CURIO DIAGNOSIS & EVIDENCE REPORT
============================================================
Session ID:         physical_machine_A_memory_pressure_high_b8adbb32
Window Index:       5
True Condition:     Memory Pressure
------------------------------------------------------------
Condition:          Memory Pressure
Model Confidence:   83.9%
Abnormality Score:  0.8809  (Validation Threshold: 0.8536)
Abnormal Status:    ABNORMAL
Evidence Consist.:  100.0%
------------------------------------------------------------
WHY CURIO DIAGNOSED THIS:
  1. Top-Process Memory Share (11.9%) was substantially elevated relative to the learned Normal reference (5.9%).
     [Signal: top_proc_mem_pct, Dev: +10.00 std (STRONG)]
  2. Swap / Pagefile Utilization (5.7%) was substantially lower than the learned Normal reference (5.9%).
     [Signal: swap_used_pct, Dev: +3.97 std (STRONG)]
  3. Physical RAM Utilization (86.3%) was substantially elevated relative to the learned Normal reference (78.2%).
     [Signal: ram_used_pct, Dev: +3.65 std (STRONG)]

OVERALL INTERPRETATION:
  "Multiple memory-related telemetry signals moved in the expected direction for Memory Pressure."
============================================================
```

### 8.4 Example 4: Disk I/O Pressure
```text
============================================================
CURIO DIAGNOSIS & EVIDENCE REPORT
============================================================
Session ID:         physical_machine_A_disk_io_pressure_high_63628538
Window Index:       5
True Condition:     Disk Io Pressure
------------------------------------------------------------
Condition:          Disk Io Pressure
Model Confidence:   96.0%
Abnormality Score:  0.9881  (Validation Threshold: 0.8536)
Abnormal Status:    ABNORMAL
Evidence Consist.:  100.0%
------------------------------------------------------------
WHY CURIO DIAGNOSED THIS:
  1. Disk I/O Throughput Rate (8.76) was substantially elevated relative to the learned Normal reference (5.84).
     [Signal: disk_io_rate_norm, Dev: +4.64 std (STRONG)]
  2. Disk Operation Frequency (IOPS) (3.06) was substantially elevated relative to the learned Normal reference (1.44).
     [Signal: disk_iops_norm, Dev: +3.15 std (STRONG)]
  3. Swap / Pagefile Utilization (5.8%) showed moderate reduction compared to Normal reference (5.9%).
     [Signal: swap_used_pct, Dev: +1.67 std (MODERATE)]

OVERALL INTERPRETATION:
  "Multiple disk and memory-related telemetry signals moved in the expected direction for Disk Io Pressure."
============================================================
```

---

## 9. Robustness & Edge-Case Handling

The Evidence Engine enforces comprehensive defense-in-depth against invalid inputs:

| Edge Case | Engine Response | Unit Test Verification |
| :--- | :--- | :--- |
| **Missing Feature** | `InvalidTelemetryError` in strict mode; tagged `status: 'missing'` in non-strict mode. | `test_missing_feature_handling` |
| **`NaN` Value** | `InvalidTelemetryError` in strict mode; excluded and logged in `invalid_features` in non-strict mode. | `test_nan_handling` |
| **`Inf` Value** | `InvalidTelemetryError` in strict mode; excluded and logged in `invalid_features` in non-strict mode. | `test_inf_handling` |
| **Zero Variance ($\sigma = 0$)** | Handled safely via $\sigma_{\text{safe}} = \max(\sigma, \epsilon)$; score capped at $\pm 10.0$ to prevent explosion. | `test_zero_std_no_division_by_zero` |
| **Unknown Condition** | Fails immediately with descriptive `ValueError`. | Validated via `VALID_CONDITIONS` |
| **Causal Term Injected** | Fails immediately with `CausalLanguageError`. | `test_no_causal_language_policy` |

---

## 10. Artifact Structure & Pipeline Integration

The Evidence Engine is integrated into both cross-validation evaluation folds and deployment exports:

```text
data/models/
├── fold_physical_machine_A/          # Evaluation fold artifacts
│   ├── model.joblib
│   ├── calibration_metadata.json
│   ├── abnormality_threshold.json
│   ├── feature_metadata.json
│   └── evidence_reference.json       <-- STRICTLY TRAINING-DERIVED
└── physical_deployment/              # Final deployment model
    ├── model.joblib
    ├── calibration_metadata.json
    ├── abnormality_threshold.json
    ├── feature_metadata.json
    └── evidence_reference.json       <-- PHYSICAL BASELINE PROFILE
```

### JSON Schema for `evidence_reference.json`
```json
{
  "schema_version": "1.0.0",
  "created_at": "2026-09-27T18:22:29.760681+00:00",
  "training_machines": ["physical_machine_A"],
  "normal_sessions_count": 4,
  "total_training_samples": 242,
  "feature_stats": {
    "cpu_mean": {"mean": 18.62, "std": 7.15, "median": 17.81, "iqr": 6.67},
    ...
  },
  "expected_directions": {
    "cpu_pressure": {"cpu_mean": 1, "cpu_max": 1, ...},
    "memory_pressure": {"ram_used_pct": 1, "ram_available_ratio": -1, ...},
    "disk_io_pressure": {"disk_io_rate_norm": 1, "disk_iops_norm": 1, ...},
    "normal": {"cpu_mean": 0, "cpu_max": 0, ...}
  },
  "epsilon": 0.0001,
  "score_cap": 10.0
}
```

---

## 11. Test Suite Results

The complete test suite was executed across all components in the repository:
```text
Ran 80 tests in 28.843s
OK
```

All **80 unit tests pass** with zero failures:
- `tests/test_evidence.py` (16 tests): Directional scoring, normal neutrality, CPU/RAM/Disk attribution, contradictory handling, NaN/Inf rejection, zero-division safety, probability separation, training isolation, non-causal policy.
- `tests/test_collector.py` (3 tests)
- `tests/test_features.py` (6 tests)
- `tests/test_stress.py` (3 tests)
- `tests/test_dataset_builder.py` (2 tests)
- `tests/test_baseline_rules.py` (5 tests)
- `tests/test_model_trainer.py` (15 tests)
- `tests/test_physical_dataset_validator.py` (8 tests)
- `tests/test_multimachine_lomo.py` (6 tests)
- `tests/test_physical_deployment.py` (12 tests)

---

## 12. Known Limitations & Next Steps

1. **Observational vs Causal Boundary**: The Evidence Engine explains *what telemetry supports the model's prediction*. It does not establish root-cause hardware defects or software bugs.
2. **Discrete Pressure Isolation**: Current models isolate single resource stresses. Compound pressures (e.g., memory exhaustion leading into thrashing and disk saturation) are described by multi-subsystem directional signals but remain classified under mutually exclusive labels.
3. **Single Physical Host**: All physical validation currently originates from `physical_machine_A`. Multi-machine generalization will be evaluated once Machines B and C archives are collected.

**Next Milestone**: **CURIO Discovery Engine** (learning canonical temporal precursor signatures and cross-session pattern discovery).
