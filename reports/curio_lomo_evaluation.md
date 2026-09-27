# CURIO — Leave-One-Machine-Out (LOMO) Cross-Machine Evaluation Report

> [!IMPORTANT]
> **Scope of Claim**: *"This experiment evaluates cross-machine generalization across the available experimental machines. It does not establish universal generalization to arbitrary computer hardware."*

---

## 1. Dataset & Telemetry Summary
- **Number of Distinct Physical/Simulated Machines**: 3
- **Total Diagnostic Sessions**: 24
- **Total Rolling Feature Windows**: 264 (5.0s window duration, 2.5s step)
- **Feature Dimension**: Exactly 12 hardware-normalized features
- **Class Distribution**:
  - `normal`: 66 windows (25.0%)
  - `cpu_pressure`: 66 windows (25.0%)
  - `memory_pressure`: 66 windows (25.0%)
  - `disk_io_pressure`: 66 windows (25.0%)

## 2. Leave-One-Machine-Out (LOMO) Methodology
Evaluation strictly follows Leave-One-Machine-Out cross-validation. For each evaluation fold:
- **Training Set**: All diagnostic sessions originating from all other machines (`machine_id != m`).
- **Test Set**: All diagnostic sessions originating strictly from the held-out machine (`machine_id == m`).
- **Hardware Isolation**: The held-out machine remains completely untouched during baseline calculation, feature extraction, probability calibration, and abnormality threshold setting.

## 3. Grouped Probability Calibration Methodology
- **Calibrator**: `CalibratedClassifierCV(estimator=RandomForestClassifier(...), method='sigmoid')`.
- **Session-Grouped CV**: To prevent data leakage from temporally adjacent rolling windows, calibration folds are formed using `StratifiedGroupKFold` grouped strictly by `session_id`.
- **Zero Cross-Fold Leakage**: Windows from the same session never cross between calibration-train and calibration-validation.

## 4. Abnormality Score & Empirical Threshold Methodology
- **Formula**: $\text{abnormality\_score} = 1.0 - P(\text{Normal})$.
- **Empirical Reference Target**: For each fold, the abnormality threshold $\theta_{\text{abnormal}}$ is computed as the **95th percentile** of calibrated abnormality scores observed strictly on training-machine Normal sessions.
- **Statistical Interpretation**: This targets an empirical $\approx 5\%$ false positive rate on the training reference hardware. It is an empirical operating point, not a theoretical guarantee on unseen hardware.

## 5. Strict Zero-Leakage Audit Summary
| Leakage Vector | Audit Check | Status |
|:---|:---|:---:|
| A. Session Overlap | Assert `Train_Sessions ∩ Test_Sessions == ∅` | **PASSED** |
| B. Machine Isolation | Assert `Held_Out_Machine ∉ Train_Machines` | **PASSED** |
| C. Calibration Purity | Held-out machine never participates in calibration CV | **PASSED** |
| D. Threshold Independence | Test machine telemetry excluded from threshold $\theta$ | **PASSED** |
| E. Preprocessing Purity | No global scalers; RF operates on raw hardware-normalized features | **PASSED** |
| F. Label Independence | Test labels strictly quarantined until final evaluation | **PASSED** |
| G. Rolling Window Grouping | `groups = session_id` prevents adjacent window leakage | **PASSED** |

## 6. Per-Machine Generalization Performance
| Held-Out Machine | Test Windows | RF Accuracy | RF Macro F1 | Baseline Macro F1 | Abnormality $\theta$ | Brier Score |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| `machine_A_desktop_ryzen_16c` | 88 | 1.000 | 1.000 | 1.000 | 0.0736 | 0.0058 |
| `machine_B_laptop_intel_8c` | 88 | 1.000 | 1.000 | 1.000 | 0.0736 | 0.0173 |
| `machine_C_server_xeon_32c` | 88 | 1.000 | 1.000 | 1.000 | 0.0701 | 0.0068 |

## 7. Aggregate Performance & Heuristic Baseline Comparison
| Metric | CURIO Calibrated Random Forest | Deterministic Heuristic Baseline | Absolute Delta |
|:---|:---:|:---:|:---:|
| **Overall Accuracy** | **1.0000** | 1.0000 | +0.0000 |
| **Macro Precision** | **1.0000** | 1.0000 | +0.0000 |
| **Macro Recall** | **1.0000** | 1.0000 | +0.0000 |
| **Macro F1-Score** | **1.0000** | 1.0000 | +0.0000 |

### Per-Class F1-Score Breakdown
| Condition Class | Support | RF Precision | RF Recall | RF F1 | Baseline F1 |
|:---|:---:|:---:|:---:|:---:|:---:|
| `normal` | 66 | 1.000 | 1.000 | **1.000** | 1.000 |
| `cpu_pressure` | 66 | 1.000 | 1.000 | **1.000** | 1.000 |
| `memory_pressure` | 66 | 1.000 | 1.000 | **1.000** | 1.000 |
| `disk_io_pressure` | 66 | 1.000 | 1.000 | **1.000** | 1.000 |

## 8. Probability Calibration Metrics (Held-Out)
- **Multiclass Brier Score**: `0.0100`
- **Multi-class Log Loss**: `0.0815`
- **Metric Explanation**: The Brier score measures the mean squared error between predicted probabilities and one-hot ground truth (lower is better; 0 indicates perfect probability forecasts). Log loss penalizes confident incorrect probability forecasts. Both metrics reflect out-of-fold generalization on held-out hardware.

## 9. Feature Importance Analysis
> [!NOTE]
> **Methodological Clarification**: Feature importances indicate which telemetry dimensions contributed most strongly to the tree splits across training folds. They represent **predictive utility**, not physical causality.
| Rank | Feature Name | Mean Importance Across Folds | Std Dev |
|:---:|:---|:---:|:---:|
| 1 | `cpu_max` | 0.1645 | ±0.0132 |
| 2 | `cpu_mean` | 0.1522 | ±0.0175 |
| 3 | `disk_iops_norm` | 0.1421 | ±0.0047 |
| 4 | `disk_io_rate_norm` | 0.1250 | ±0.0048 |
| 5 | `ram_available_ratio` | 0.0862 | ±0.0022 |
| 6 | `ram_used_pct` | 0.0758 | ±0.0008 |
| 7 | `swap_used_pct` | 0.0739 | ±0.0078 |
| 8 | `top_proc_mem_pct` | 0.0713 | ±0.0082 |
| 9 | `top_proc_cpu_ratio` | 0.0628 | ±0.0032 |
| 10 | `cpu_std` | 0.0337 | ±0.0035 |
| 11 | `cpu_core_imbalance` | 0.0082 | ±0.0005 |
| 12 | `process_count_delta` | 0.0043 | ±0.0013 |

## 10. Limitations
1. **Finite Hardware Diversity**: Evaluating cross-machine performance across initial machines verifies that the pipeline handles hardware variance without breaking, but does not prove generalization across radically different architectures (e.g. mobile ARM, high-end server clusters, virtual machines with noisy neighbors).
2. **Simulated vs. Physical Noise**: While synthetic dry-run data rigorously checks pipeline integrity, production validation requires data collected across multiple physical computers.
3. **Single vs. Compound Stress**: Current conditions model discrete resource pressures. Compound pressures (simultaneous CPU + Disk saturation) may exhibit interaction behaviors not captured by mutually exclusive labels.

## 11. Next Step
With the Leave-One-Machine-Out pipeline, grouped calibration, and leakage prevention fully verified, the next milestone is collecting multi-machine physical telemetry and implementing the **CURIO Evidence Engine**.
