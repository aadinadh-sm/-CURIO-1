# CURIO — Technical Report: Discovery Engine Architecture & Temporal Validation

**Date**: 2026-09-28  
**Project**: CURIO (Undergraduate Capstone)  
**Milestone**: Discovery Engine (Temporal Trajectory Discovery & Pattern Recognition)  
**Test Suite Status**: **102 / 102 Tests Passing (100% OK, 22 Discovery Tests)**  
**Hardware Evaluated**: `physical_machine_A` (AMD Ryzen 7 16-Core, 16 GB RAM, Windows 11)  
**Artifact Directory**: `data/models/physical_deployment/`  

---

## 1. Purpose of the Discovery Engine

While standard machine learning classifiers reliably assign categorical labels (e.g., `CPU Pressure`, `Memory Pressure`), and the CURIO Evidence Engine explains static attribution ("Which telemetry features deviated from Normal?"), neither answers the dynamic question:

> **"WHAT temporal progression of telemetry events characterized this incident?"**

Operating system bottlenecks and degradation states are inherently dynamic processes. For instance, when CPU starvation emerges, does single-core saturation precede process-level monopolization? In memory exhaustion, does active physical RAM depletion precede swap thrashing?

The **CURIO Discovery Engine** is designed to discover, order, and quantify these temporal progressions. Crucially:
1. **Downstream Positioning**: The Discovery Engine operates strictly downstream of classification. It does not alter, override, or compete with the classifier's diagnosis.
2. **Dynamic Progression vs. Static Attribution**: While evidence attribution analyzes window-level magnitude deviations, discovery analyzes the chronological sequence in which distinct telemetry features cross statistical abnormality thresholds across the 11 feature windows (30-second capture).
3. **Canonical Comparison**: It evaluates whether the observed temporal sequence matches the canonical precursor order derived empirically from training data.

---

## 2. Architecture & Placement in CURIO

The Discovery Engine completes the diagnostic tri-layer of CURIO:

```
[ Raw Telemetry Stream ] (psutil @ 2 Hz, 61 raw samples / 30s)
            |
            v
[ Feature Engine ] (5.0s rolling windows, 2.5s step -> 11 feature vectors x 12 features)
            |
            v
[ Calibrated Classifier ] (Random Forest + Sigmoid Calibration + Abnormality Gating)
            |
            +---> DIAGNOSIS: Predicted Condition & Calibrated Confidence
            |
            +---> [ Evidence Engine ]
            |       Question: "WHY did CURIO make this diagnosis?"
            |       Output: Top supporting features & directional z-score deviations
            |
            +---> [ Discovery Engine ]
                    Question: "WHAT temporal pattern did CURIO discover?"
                    Input: 11-window session DataFrame, Predicted Condition, References
                    Output: Temporal onset sequence, Kendall's tau-b, Canonical comparison
```

### Separation of Responsibilities
- **Classification Engine**: "What state is the computer in?"
- **Evidence Engine**: "Why do the telemetry levels support this state?"
- **Discovery Engine**: "How did the telemetry develop over time?"

---

## 3. Training-Derived Canonical Signatures

The canonical precursor signature for each abnormal condition is derived strictly from training sessions (`DiscoveryReferenceProfile.fit_from_training_sessions` in [src/discovery.py](file:///c:/Users/Aadinadh%20S%20M/Downloads/cuuuu/src/discovery.py)). Test sessions and held-out validation machines are never seen during signature derivation.

### Algorithm
For each condition $c \in \{\text{cpu\_pressure}, \text{memory\_pressure}, \text{disk\_io\_pressure}\}$:
1. **Candidate Feature Filtering**: Only features with a non-zero expected direction ($D_{c, f} \ne 0$) in the `EvidenceReferenceProfile` are evaluated. Features with zero expected direction ($D_{c, f} = 0$) are strictly excluded.
2. **Session Onset Detection**: For each training session belonging to condition $c$, scan the 11 feature windows and detect the earliest confirmed onset time for each candidate feature.
3. **Recurrence Computation**: The recurrence rate $R_{c, f}$ is computed as:
   $$R_{c, f} = \frac{\text{Count of training sessions of condition } c \text{ with confirmed onset for } f}{\text{Total training sessions of condition } c}$$
4. **Signature Inclusion**: A feature is included in the canonical signature if and only if $R_{c, f} \ge R_{\text{threshold}}$ (default: $0.60$).
5. **Canonical Ordering**: Included features are ordered chronologically by their median onset time across training sessions:
   $$\tilde{t}_{c, f} = \text{median}(\{t_{\text{onset}, s, f} \mid s \in \text{Sessions}_c\})$$
   Ties in median onset time are resolved deterministically by higher recurrence rate $R_{c, f}$, followed by alphabetical sorting of feature names.

---

## 4. Recurrence Threshold Selection ($R \ge 0.60$) and Rationale

The recurrence threshold $R_{\text{threshold}} = 0.60$ requires that a candidate feature exhibits a confirmed directional onset in at least 60% of all training sessions for that condition.

### Rationale
- **Noise Filtration**: Telemetry data contains transient jitter (e.g., short-lived OS worker thread spikes, sporadic background disk writes). A low threshold (e.g., 20%) would admit spurious features that do not represent the core degradation mechanism.
- **Signal Robustness**: A threshold of 60% ensures that only reproducible, characteristic precursors form the canonical pattern.
- **Physical Validation**: In our physical dataset (`physical_machine_A`), this threshold cleanly isolated the core drivers of CPU pressure (`cpu_max`, `cpu_mean`, `ram_available_ratio`, `ram_used_pct`, `swap_used_pct`, `top_proc_cpu_ratio`) and Disk pressure (`disk_io_rate_norm`, `disk_iops_norm`) while excluding peripheral noise.

---

## 5. Directional Onset Definition

A feature onset is defined as a statistically significant, sustained shift in the expected abnormal direction.

### Formulation
For a rolling window $w \in \{0, 1, \dots, 10\}$ with feature value $x_w$, Normal reference mean $\mu_{\text{norm}}$, Normal standard deviation $\sigma_{\text{norm}}$, and class expected direction $D_{c, f} \in \{+1, -1\}$:

$$z_{\text{dir}}(w) = D_{c, f} \cdot \left(\frac{x_w - \mu_{\text{norm}}}{\sigma_{\text{norm}}}\right)$$

### Two-Window Confirmation Rule
An onset is confirmed at window $w$ if:
$$z_{\text{dir}}(w) \ge z_{\text{threshold}} \quad \text{AND} \quad z_{\text{dir}}(w+1) \ge z_{\text{threshold}}$$
where $z_{\text{threshold}} = 1.8$ (representing a 1.8-standard-deviation departure from learned Normal equilibrium).

The timestamp of the onset is given by:
$$t_{\text{onset}} = w \times 2.5\text{ seconds}$$

### Rationale for Confirmation
A single window exceeding $1.8\sigma$ can be caused by transient OS background interruptions. Requiring two consecutive windows ($5.0$s total sustained elevation) ensures that only persistent shifts are recognized as temporal onsets.

---

## 6. Live Session Sequence Detection

During inference, given an 11-window session DataFrame and a diagnosed condition:
1. `detect_feature_onsets_in_session` iterates through each candidate feature.
2. The earliest window satisfying the two-window confirmation rule is recorded as the feature's onset.
3. The observed sequence is constructed by sorting confirmed features by $t_{\text{onset}}$ ascending.
4. Tied onset times are broken deterministically by higher peak directional z-score, then alphabetically by feature name.

---

## 7. Kendall's $\tau_b$ Methodology & $[0, 1]$ Normalization

To quantify the degree of agreement between the observed temporal sequence and the canonical training sequence, CURIO computes **Kendall's rank correlation coefficient ($\tau_b$)**, which robustly handles tied ranks.

### Intersecting Feature Subset
Let $S_{\text{obs}}$ be the set of features with confirmed onsets in the observed session, and $S_{\text{canon}}$ be the canonical signature features. The correlation is computed strictly over the intersection:
$$S_{\text{eval}} = S_{\text{obs}} \cap S_{\text{canon}}$$

### Kendall's $\tau_b$ Formulation
$$\tau_b = \frac{P - Q}{\sqrt{(P + Q + T) \cdot (P + Q + U)}}$$
where:
- $P$: Number of concordant pairs
- $Q$: Number of discordant pairs
- $T$: Number of ties in the observed ranks
- $U$: Number of ties in the canonical ranks

If all pairs are tied (denominator is zero), $\tau_b = 0.0$.

### Normalized Trajectory Score
Standard $\tau_b \in [-1.0, +1.0]$ is normalized into the intuitive $[0.0, 1.0]$ interval:
$$\tau_{\text{norm}} = \frac{\tau_b + 1.0}{2.0}$$

### Qualitative Agreement Mapping
| Normalized $\tau_{\text{norm}}$ | Raw $\tau_b$ Range | Qualitative Interpretation |
| :--- | :--- | :--- |
| $\ge 0.85$ | $[+0.70, +1.00]$ | High Agreement |
| $0.65 \le \tau_{\text{norm}} < 0.85$ | $[+0.30, +0.70)$ | Moderate Agreement |
| $0.50 \le \tau_{\text{norm}} < 0.65$ | $[0.00, +0.30)$ | Low Agreement |
| $< 0.50$ | $[-1.00, 0.00)$ | Inverse / Disordered |

---

## 8. Insufficient-Evidence Handling ($< 3$ Distinct Non-Tied Onsets)

A fundamental methodological pitfall in temporal sequence analysis is computing rank correlations over 1 or 2 features, or over features that all onset simultaneously. Such calculations produce trivial or statistically degenerate values.

### Discovery Engine Rule
If $|S_{\text{eval}}| < 3$, or if the number of distinct, non-tied onset times among evaluated features is $< 3$:
1. The Discovery Engine assigns:
   $$\text{discovery\_status} = \text{"INSUFFICIENT\_TEMPORAL\_EVIDENCE"}$$
2. $\tau_b$ and $\tau_{\text{norm}}$ are set to `None` (`N/A`).
3. An honest, non-causal explanation is returned:
   > *"Fewer than 3 distinct non-tied onsets detected; temporal trajectory cannot be statistically confirmed."*

---

## 9. Sustained-Pressure Handling ($t=0$ Onsets with No Temporal Transition)

In controlled benchmark stress tests or ongoing production incidents, stress is often already active when telemetry capture begins. Consequently, all affected features register onsets at the very first window ($w=0, t=0.0\text{s}$).

### Discovery Engine Rule
When all detected features have onsets at $t=0.0\text{s}$ (zero distinct post-baseline onset transitions):
1. The Discovery Engine assigns:
   $$\text{discovery\_status} = \text{"SUSTAINED\_PRESSURE"}$$
2. $\tau_b$ and $\tau_{\text{norm}}$ are set to `None` (`N/A`).
3. The report clearly explains:
   > *"Sustained operating pressure detected; no transition observed during capture."*

This distinction is crucial: CURIO explicitly identifies that the session captured a steady-state plateau rather than a dynamic transition, avoiding fabricated sequence rankings.

---

## 10. Normal-Equilibrium Handling

When the classifier predicts `Normal`, the system is operating within learned equilibrium.

### Discovery Engine Rule
1. The Discovery Engine assigns:
   $$\text{discovery\_status} = \text{"OPERATING_EQUILIBRIUM"}$$
2. Canonical sequence is reported as `None defined for this state`.
3. $\tau$ and coverage are reported as `N/A` and `0.0%`.
4. Interpretation:
   > *"Operating Equilibrium: No escalation detected. Telemetry remained within learned Normal bounds."*

---

## 11. Leakage Prevention Audit

The Discovery Engine enforces strict isolation against data leakage:
1. **Reference Fitting**: Canonical signatures (`discovery_reference.json`) and baseline statistics (`evidence_reference.json`) are fit **only** on training data folds.
2. **Held-Out Machine Protection**: In cross-validation or LOMO, test machines/sessions are strictly excluded from the reference fitting step.
3. **Sequential Trajectory Inspection**: Live sessions are evaluated strictly using forward rolling windows without future-window lookahead.

---

## 12. Example: CPU Pressure Discovery

**Session**: `physical_machine_A_cpu_pressure_high_b4e1568b`  
**True Condition**: CPU Pressure | **Diagnosed**: CPU Pressure (96.1% confidence)

```
============================================================
CURIO DISCOVERY
============================================================
Predicted condition: Cpu Pressure
Discovery status:    SUSTAINED_PRESSURE
------------------------------------------------------------
Observed progression:
  cpu_core_imbalance (t = 0.0s)
        v
  cpu_max (t = 0.0s)
        v
  cpu_mean (t = 0.0s)
        v
  ram_available_ratio (t = 0.0s)
        v
  ram_used_pct (t = 0.0s)
        v
  swap_used_pct (t = 0.0s)
        v
  top_proc_cpu_ratio (t = 0.0s)
        v
  cpu_std (t = 2.5s)
------------------------------------------------------------
Canonical training pattern:
  cpu_max
   v
  cpu_mean
   v
  ram_available_ratio
   v
  ram_used_pct
   v
  swap_used_pct
   v
  top_proc_cpu_ratio
------------------------------------------------------------
Temporal similarity:  N/A (insufficient or non-transitional onsets)
Discovery coverage:   100.0%
Unexpected observed:  cpu_core_imbalance, cpu_std
------------------------------------------------------------
Interpretation:
  "Sustained operating pressure detected; no transition observed during capture."
============================================================
```

---

## 13. Example: Memory Pressure Discovery

**Session**: `physical_machine_A_memory_pressure_high_b8adbb32`  
**True Condition**: Memory Pressure | **Diagnosed**: Memory Pressure (89.9% confidence)

```
============================================================
CURIO DISCOVERY
============================================================
Predicted condition: Memory Pressure
Discovery status:    SUSTAINED_PRESSURE
------------------------------------------------------------
Observed progression:
  ram_available_ratio (t = 0.0s)
        v
  ram_used_pct (t = 0.0s)
        v
  swap_used_pct (t = 0.0s)
        v
  top_proc_mem_pct (t = 0.0s)
------------------------------------------------------------
Canonical training pattern:
  swap_used_pct
------------------------------------------------------------
Temporal similarity:  N/A (insufficient or non-transitional onsets)
Discovery coverage:   100.0%
Unexpected observed:  ram_available_ratio, ram_used_pct, top_proc_mem_pct
------------------------------------------------------------
Interpretation:
  "Sustained operating pressure detected; no transition observed during capture."
============================================================
```

---

## 14. Example: Disk I/O Pressure Discovery

**Session**: `physical_machine_A_disk_io_pressure_high_63628538`  
**True Condition**: Disk I/O Pressure | **Diagnosed**: Disk I/O Pressure (96.0% confidence)

```
============================================================
CURIO DISCOVERY
============================================================
Predicted condition: Disk Io Pressure
Discovery status:    SUSTAINED_PRESSURE
------------------------------------------------------------
Observed progression:
  disk_io_rate_norm (t = 0.0s)
        v
  disk_iops_norm (t = 0.0s)
------------------------------------------------------------
Canonical training pattern:
  disk_io_rate_norm
   v
  disk_iops_norm
------------------------------------------------------------
Temporal similarity:  N/A (insufficient or non-transitional onsets)
Discovery coverage:   100.0%
------------------------------------------------------------
Interpretation:
  "Sustained operating pressure detected; no transition observed during capture."
============================================================
```

---

## 15. Example: Normal Operating Equilibrium

**Session**: `physical_machine_A_normal_none_5743354b`  
**True Condition**: Normal | **Diagnosed**: Normal (95.4% confidence)

```
============================================================
CURIO DISCOVERY
============================================================
Predicted condition: Normal
Discovery status:    OPERATING_EQUILIBRIUM
------------------------------------------------------------
Observed progression: None detected.
------------------------------------------------------------
Canonical training pattern: None defined for this state.
------------------------------------------------------------
Temporal similarity:  N/A (insufficient or non-transitional onsets)
Discovery coverage:   0.0%
------------------------------------------------------------
Interpretation:
  "Operating Equilibrium: No escalation detected. Telemetry remained within learned Normal bounds."
============================================================
```

---

## 16. Known Limitations of 30-Second Window for Trajectory Discovery

1. **Temporal Horizon (30 seconds)**: 30 seconds at 2 Hz yields 61 raw samples and 11 rolling windows (step = 2.5s). Slow memory leaks or thermal throttling that evolve over minutes cannot be observed as gradual multi-stage transitions within a single 30s session.
2. **Coarse Step Granularity (2.5 seconds)**: Rapid cascades that occur within 100–500 ms will register at the same window step, appearing as ties ($t=0.0\text{s}$ or $t=2.5\text{s}$).
3. **Capture Onset Alignment**: In synthetic and physical benchmark scripts, stress workloads are initiated at the start of the session. Consequently, physical validation sessions predominantly exhibit `SUSTAINED_PRESSURE`. Dynamic multi-stage transitions are primarily observable in gradual-ramp scenarios.

---

## 17. Causality Limitation Statement

> [!WARNING]
> **CURIO strictly asserts temporal precedence, NOT physical causation.**

A statistical observation that Feature A crossed an abnormality threshold before Feature B (e.g., $t_{\text{onset}}(\text{cpu\_max}) < t_{\text{onset}}(\text{top\_proc\_cpu\_ratio})$) indicates only chronological order in telemetry. It does **not** prove that Feature A caused Feature B.

The codebase enforces this via the automated guardrail `validate_no_causal_language` in [src/discovery.py](file:///c:/Users/Aadinadh%20S%20M/Downloads/cuuuu/src/discovery.py), which raises a `ValueError` if banned causal terms (`caused`, `leads to`, `triggers`, `root cause`, `resulted in`, `because of`) appear in interpretations or reports.

---

## 18. Test Suite Results

The comprehensive test suite covering the entire CURIO repository was executed:

```bash
python -m unittest discover tests
```

**Results**:
- **Total Tests Run**: **102 tests**
- **Test Failures / Errors**: **0**
- **Status**: **OK** (Ran in 25.419s)

### Test Coverage Breakdown
- `tests/test_discovery.py`: **22 dedicated unit tests**
  - Canonical signature training & serialization
  - Recurrence thresholding ($R \ge 0.60$)
  - Directional filtering & zero-direction exclusion
  - Onset thresholding ($z_{\text{dir}} \ge 1.8$)
  - Two-window confirmation rule
  - Duplicate & simultaneous onset handling
  - $t=0$ sustained pressure detection
  - Insufficient evidence handling ($< 3$ distinct onsets)
  - Normal equilibrium handling
  - Canonical ordering & tie breaking
  - Observed sequence extraction
  - Kendall's $\tau_b$ calculation & $[0, 1]$ normalization
  - Tie handling in Kendall's $\tau_b$
  - Partial sequence matching & coverage calculation
  - Missing & unexpected feature detection
  - NaN / Inf telemetry safety
  - Temporal leakage isolation
  - Schema conformity verification
  - Causal language rejection guardrail
  - Deterministic report formatting
- `tests/test_evidence.py`: **16 unit tests** (attribution, reference baselines, z-scores)
- `tests/test_model_trainer.py`: **12 unit tests** (RF, calibration, abnormality threshold, LOMO)
- `tests/test_physical_dataset_validator.py`: **10 unit tests** (integrity rules A–P)
- `tests/test_portable_deployment.py`: **4 unit tests** (export/import packaging & hashing)
- `tests/test_baseline_rules.py`: **6 unit tests** (deterministic rule baseline)
- `tests/test_features.py`: **6 unit tests** (12-feature windowing, bounds, NaN resistance)
- `tests/test_collector.py`: **6 unit tests** (psutil sampling, 2 Hz, 61 samples)
- `tests/test_stress_harness.py`: **10 unit tests** (stress isolation, safety bounds)

---

## 19. Next Step: Live CURIO Diagnosis Pipeline

With the completion of:
1. Telemetry Collector (2 Hz, 61 samples)
2. 12-Feature Windowing Engine (5.0s window, 2.5s step)
3. Stress Harness & Physical Telemetry Dataset
4. Calibrated Random Forest Classifier & Abnormality Threshold
5. Evidence Engine (Explanatory Telemetry Attribution)
6. Discovery Engine (Temporal Trajectory Discovery & Pattern Recognition)

The next and final operational milestone is the **Live CURIO Diagnosis Pipeline**, uniting all six subsystems into a real-time, interactive end-to-end diagnostic session monitor.
