"""Deterministic Heuristic Rule-Based Baseline Classifier for CURIO.

Provides an uncalibrated, threshold-based diagnostic engine for comparison
against machine-learning models. Implements explicit, transparent rules
over the 12 hardware-normalized telemetry features.
"""

from typing import Any, Dict, List, Union
import numpy as np
import pandas as pd

from src.features import FEATURE_NAMES


class HeuristicBaselineClassifier:
    """Rule-based classifier evaluating system operational pressure without statistical learning."""

    # Explicit, documented rule thresholds
    CPU_PRESSURE_THRESHOLD = 70.0  # Overall CPU mean >= 70%
    CPU_ROGUE_THRESHOLD = 50.0     # Overall CPU >= 50% with rogue process dominance
    CPU_ROGUE_RATIO = 0.70         # Single process taking >= 70% of current CPU

    RAM_USED_THRESHOLD = 85.0      # Physical RAM usage >= 85%
    RAM_AVAIL_THRESHOLD = 0.12     # Available memory ratio <= 12%
    SWAP_PRESSURE_THRESHOLD = 25.0 # Swap/pagefile usage >= 25%

    DISK_THROUGHPUT_LOG_THRESHOLD = 7.5  # log10(bytes/sec + 1) >= 7.5 (~31.6 MB/s)
    DISK_IOPS_LOG_THRESHOLD = 2.5        # log10(ops/sec + 1) >= 2.5 (~316 ops/sec)

    VALID_CONDITIONS = [
        "normal",
        "cpu_pressure",
        "memory_pressure",
        "disk_io_pressure",
    ]

    def predict_window(
        self, features: Union[Dict[str, Any], pd.Series]
    ) -> Dict[str, Any]:
        """Classifies a single feature window using deterministic rules.

        Args:
            features: Dictionary or pandas Series containing the 12 features.

        Returns:
            Dictionary with predicted_condition, rule_evidence, rule_scores, is_ambiguous.
        """
        # 1. Validate input completeness
        if features is None:
            raise ValueError("Input features cannot be None.")

        # Convert to dictionary if Series
        if isinstance(features, pd.Series):
            feat_dict = features.to_dict()
        elif isinstance(features, dict):
            feat_dict = dict(features)
        else:
            raise TypeError(f"Expected dict or pd.Series, got {type(features).__name__}")

        # Check for missing or NaN features
        missing = [f for f in FEATURE_NAMES if f not in feat_dict]
        if missing:
            raise ValueError(f"Missing required features: {missing}")

        for f in FEATURE_NAMES:
            val = feat_dict[f]
            if val is None or (isinstance(val, float) and (np.isnan(val) or np.isinf(val))):
                raise ValueError(f"Feature '{f}' contains invalid/NaN/Inf value: {val}")

        cpu_mean = float(feat_dict["cpu_mean"])
        top_proc_cpu_ratio = float(feat_dict["top_proc_cpu_ratio"])
        ram_used_pct = float(feat_dict["ram_used_pct"])
        ram_avail_ratio = float(feat_dict["ram_available_ratio"])
        swap_used_pct = float(feat_dict["swap_used_pct"])
        disk_io_rate_norm = float(feat_dict["disk_io_rate_norm"])
        disk_iops_norm = float(feat_dict["disk_iops_norm"])

        # 2. Evaluate Rule Conditions & Evidence
        evidence: List[str] = []
        rule_scores: Dict[str, float] = {
            "normal": 0.1,  # Baseline default score
            "cpu_pressure": 0.0,
            "memory_pressure": 0.0,
            "disk_io_pressure": 0.0,
        }

        # --- CPU Pressure Rules ---
        if cpu_mean >= self.CPU_PRESSURE_THRESHOLD:
            score = min(1.0, 0.5 + (cpu_mean - self.CPU_PRESSURE_THRESHOLD) / 60.0)
            rule_scores["cpu_pressure"] = max(rule_scores["cpu_pressure"], score)
            evidence.append(f"CPU mean load ({cpu_mean:.1f}%) exceeds threshold ({self.CPU_PRESSURE_THRESHOLD}%).")
        elif cpu_mean >= self.CPU_ROGUE_THRESHOLD and top_proc_cpu_ratio >= self.CPU_ROGUE_RATIO:
            score = 0.65 + (top_proc_cpu_ratio - self.CPU_ROGUE_RATIO) * 0.5
            rule_scores["cpu_pressure"] = max(rule_scores["cpu_pressure"], score)
            evidence.append(
                f"Elevated CPU ({cpu_mean:.1f}%) with dominant single process ratio ({top_proc_cpu_ratio:.2f})."
            )

        # --- Memory Pressure Rules ---
        mem_triggered = False
        if ram_used_pct >= self.RAM_USED_THRESHOLD:
            score = min(1.0, 0.5 + (ram_used_pct - self.RAM_USED_THRESHOLD) / 30.0)
            rule_scores["memory_pressure"] = max(rule_scores["memory_pressure"], score)
            evidence.append(f"RAM usage ({ram_used_pct:.1f}%) exceeds threshold ({self.RAM_USED_THRESHOLD}%).")
            mem_triggered = True

        if ram_avail_ratio <= self.RAM_AVAIL_THRESHOLD:
            score = min(1.0, 0.6 + (self.RAM_AVAIL_THRESHOLD - ram_avail_ratio) / 0.15)
            rule_scores["memory_pressure"] = max(rule_scores["memory_pressure"], score)
            evidence.append(
                f"Available RAM ratio ({ram_avail_ratio:.3f}) below safety threshold ({self.RAM_AVAIL_THRESHOLD})."
            )
            mem_triggered = True

        if swap_used_pct >= self.SWAP_PRESSURE_THRESHOLD:
            score = min(1.0, 0.5 + (swap_used_pct - self.SWAP_PRESSURE_THRESHOLD) / 50.0)
            rule_scores["memory_pressure"] = max(rule_scores["memory_pressure"], score)
            evidence.append(f"Pagefile swap usage ({swap_used_pct:.1f}%) is heavily elevated.")
            mem_triggered = True

        # --- Disk I/O Pressure Rules ---
        if disk_io_rate_norm >= self.DISK_THROUGHPUT_LOG_THRESHOLD:
            score = min(1.0, 0.5 + (disk_io_rate_norm - self.DISK_THROUGHPUT_LOG_THRESHOLD) / 3.0)
            rule_scores["disk_io_pressure"] = max(rule_scores["disk_io_pressure"], score)
            evidence.append(
                f"Disk I/O rate norm ({disk_io_rate_norm:.2f}) indicates heavy storage throughput."
            )

        if disk_iops_norm >= self.DISK_IOPS_LOG_THRESHOLD:
            score = min(1.0, 0.5 + (disk_iops_norm - self.DISK_IOPS_LOG_THRESHOLD) / 2.0)
            rule_scores["disk_io_pressure"] = max(rule_scores["disk_io_pressure"], score)
            evidence.append(
                f"Disk IOPS norm ({disk_iops_norm:.2f}) indicates high storage transaction rate."
            )

        # 3. Determine Winning Class & Ambiguity Check
        # Sort pressure conditions by score
        pressure_conditions = ["cpu_pressure", "memory_pressure", "disk_io_pressure"]
        triggered = [c for c in pressure_conditions if rule_scores[c] > 0.0]

        is_ambiguous = False
        if not triggered:
            predicted = "normal"
            rule_scores["normal"] = 1.0
            evidence.append("All resource signals remained within normal operational thresholds.")
        elif len(triggered) == 1:
            predicted = triggered[0]
        else:
            # Multiple conditions triggered: evaluate conflict
            sorted_by_score = sorted(triggered, key=lambda c: rule_scores[c], reverse=True)
            top_1 = sorted_by_score[0]
            top_2 = sorted_by_score[1]
            diff = rule_scores[top_1] - rule_scores[top_2]

            if diff < 0.15:
                is_ambiguous = True
                evidence.append(
                    f"Ambiguous operating state: Both {top_1} (score {rule_scores[top_1]:.2f}) "
                    f"and {top_2} (score {rule_scores[top_2]:.2f}) triggered conflicting rules."
                )

            predicted = top_1

        return {
            "predicted_condition": predicted,
            "rule_scores": rule_scores,
            "rule_evidence": evidence,
            "is_ambiguous": is_ambiguous,
        }

    def predict_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Batch-classifies all rows in a feature DataFrame."""
        results = []
        for _, row in df.iterrows():
            res = self.predict_window(row)
            results.append({
                "predicted_condition": res["predicted_condition"],
                "rule_score_cpu": res["rule_scores"]["cpu_pressure"],
                "rule_score_mem": res["rule_scores"]["memory_pressure"],
                "rule_score_disk": res["rule_scores"]["disk_io_pressure"],
                "rule_score_normal": res["rule_scores"]["normal"],
                "is_ambiguous": res["is_ambiguous"],
            })
        return pd.concat([df.reset_index(drop=True), pd.DataFrame(results)], axis=1)
