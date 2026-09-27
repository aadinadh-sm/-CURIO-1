# CURIO — Physical Telemetry Evaluation Report

> [!IMPORTANT]
> **Scope of Claim**: *"This experiment evaluates diagnostic classification and abnormality detection across physical telemetry captured on actual hardware. It does not establish universal generalization to arbitrary computer hardware."*

> [!WARNING]
> **Multi-Machine Hardware Constraint**: *"Insufficient physical machine diversity for the planned 3-machine LOMO experiment."*
> Currently, exactly 1 physical test machine (`physical_machine_A`) is available in this local environment.
> As mandated by the protocol, synthetic machines have not been fabricated as physical machines.
> Evaluation on this physical dataset uses session-level grouped cross-validation (StratifiedGroupKFold on `session_id`).

---

## 1. Dataset & Telemetry Summary
- **Physical Machines**: 1 (`physical_machine_A`)
- **Total Physical Sessions Captured**: 22
- **Total Rolling Feature Windows**: 242 (5.0s window, 2.5s step)
- **Feature Dimension**: Exactly 12 hardware-normalized features
- **Classes Evaluated**: `normal`, `cpu_pressure`, `memory_pressure`, `disk_io_pressure`

## 2. Evaluation Methodology
- **Grouping Unit**: `groups = session_id`. Rolling windows from the same session never cross validation folds.
- **Calibrator**: `CalibratedClassifierCV(method='sigmoid')` fitted inside training splits with grouped internal CV.
- **Abnormality Threshold**: $\theta = 1.0 - P(\text{Normal})$, calculated from out-of-fold predictions on Normal sessions (95th percentile target).
- **Calculated Abnormality Threshold $\theta$**: `0.6313`
- **Zero-Leakage Assurance**: Training and test splits strictly maintain zero session intersection.

## 3. Physical Telemetry Performance vs Heuristic Baseline
| Metric | CURIO Calibrated Random Forest | Deterministic Heuristic Baseline | Absolute Delta |
|:---|:---:|:---:|:---:|
| **Overall Accuracy** | **0.9587** | 0.6983 | +0.2603 |
| **Macro Precision** | **0.9659** | 0.8369 | +0.1289 |
| **Macro Recall** | **0.9451** | 0.7216 | +0.2235 |
| **Macro F1-Score** | **0.9518** | 0.6567 | +0.2950 |

### Per-Class F1-Score Breakdown
| Condition Class | Support | RF Precision | RF Recall | RF F1 | Baseline F1 |
|:---|:---:|:---:|:---:|:---:|:---:|
| `normal` | 44 | 1.000 | 0.795 | **0.886** | 0.544 |
| `cpu_pressure` | 66 | 1.000 | 1.000 | **1.000** | 0.881 |
| `memory_pressure` | 66 | 0.878 | 0.985 | **0.929** | 0.216 |
| `disk_io_pressure` | 66 | 0.985 | 1.000 | **0.992** | 0.985 |

## 4. Probability Calibration Metrics (Physical Telemetry)
- **Multiclass Brier Score**: `0.0569`
- **Multiclass Log Loss**: `0.1609`

## 5. Failure Analysis
A total of **10** misclassified window(s) were observed:

| Session ID | True Class | Predicted Class | P(True) | P(Pred) | Stress Level | Workload |
|:---|:---:|:---:|:---:|:---:|:---:|:---|
| `physical_machine_A_memory_pressure_low_17cca6d0` | `memory_pressure` | `disk_io_pressure` | 0.000 | 0.580 | low | browser_tabs |
| `physical_machine_A_normal_none_df3ba8f6` | `normal` | `memory_pressure` | 0.301 | 0.000 | none | idle_desktop |
| `physical_machine_A_normal_none_df3ba8f6` | `normal` | `memory_pressure` | 0.395 | 0.000 | none | idle_desktop |
| `physical_machine_A_normal_none_df3ba8f6` | `normal` | `memory_pressure` | 0.431 | 0.000 | none | idle_desktop |
| `physical_machine_A_normal_none_df3ba8f6` | `normal` | `memory_pressure` | 0.410 | 0.000 | none | idle_desktop |
| `physical_machine_A_normal_none_df3ba8f6` | `normal` | `memory_pressure` | 0.356 | 0.000 | none | idle_desktop |
| `physical_machine_A_normal_none_df3ba8f6` | `normal` | `memory_pressure` | 0.365 | 0.000 | none | idle_desktop |
| `physical_machine_A_normal_none_df3ba8f6` | `normal` | `memory_pressure` | 0.402 | 0.000 | none | idle_desktop |
| `physical_machine_A_normal_none_df3ba8f6` | `normal` | `memory_pressure` | 0.396 | 0.000 | none | idle_desktop |
| `physical_machine_A_normal_none_df3ba8f6` | `normal` | `memory_pressure` | 0.387 | 0.000 | none | idle_desktop |

## 6. Known Limitations
1. **Single Physical Host**: Telemetry is captured on `physical_machine_A` (AMD Ryzen 7 16-core, 16 GB RAM). True cross-machine evaluation requires collecting identical protocols on additional physical machines (`physical_machine_B`, `physical_machine_C`).
2. **Memory Headroom Buffer**: Safe memory stress preserves $\ge 1.5\text{ GB}$ physical headroom to prevent operating system instability, resulting in moderate memory pressure separation compared to uncapped synthetics.
3. **Single vs. Compound Pressures**: The four classes model mutually exclusive operating states; real-world thrashing often exhibits dual pressures.

## 7. Next Step
With physical telemetry validated, out-of-fold thresholding verified, and the ML pipeline confirmed on live hardware, the team should acquire telemetry from secondary physical hardware to complete the full 3-machine LOMO benchmark.
