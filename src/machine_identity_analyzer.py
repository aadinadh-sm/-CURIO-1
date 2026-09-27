"""Machine-Identity Separability Analyzer for CURIO.

Evaluates whether the 12 hardware-normalized features leak or encode machine identity.
Performs a diagnostic classification task:
    X: 12 features
    y: machine_id

This is purely an exploratory diagnostic tool to inspect whether telemetry signatures
are overly tied to hardware characteristics. It is NOT part of the CURIO diagnostic model
and does not modify feature sets or thresholds.
"""

from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedGroupKFold

from src.features import FEATURE_NAMES


class MachineIdentityAnalyzer:
    """Diagnostic tool to assess machine separability from telemetry features."""

    def __init__(self, df: pd.DataFrame, random_state: int = 42):
        """Initializes with a multi-machine dataset containing 'machine_id' and the 12 features."""
        if "machine_id" not in df.columns:
            raise ValueError("Dataset must contain 'machine_id' column.")
        missing_feats = [f for f in FEATURE_NAMES if f not in df.columns]
        if missing_feats:
            raise ValueError(f"Missing required feature columns: {missing_feats}")

        self.df = df.copy()
        self.random_state = random_state
        self.machines = sorted(self.df["machine_id"].unique())

    def analyze_separability(self) -> Dict[str, Any]:
        """Tests whether a classifier can distinguish machine identities from features alone."""
        n_machines = len(self.machines)
        if n_machines < 2:
            return {
                "machine_count": n_machines,
                "is_separable": False,
                "message": (
                    f"Only {n_machines} machine present ('{self.machines[0]}'). "
                    "Machine identity classification requires at least 2 distinct physical machines."
                ),
                "accuracy": 0.0,
                "feature_importance": {},
            }

        X = np.array(self.df[FEATURE_NAMES].to_numpy(), dtype=float)
        y = np.array(self.df["machine_id"].to_numpy(), dtype=str)
        groups = np.array(self.df["session_id"].to_numpy(), dtype=str)

        # Use StratifiedGroupKFold on session_id so windows from same session don't leak
        min_sessions = min(self.df.groupby("machine_id")["session_id"].nunique())
        n_splits = max(2, min(3, min_sessions))
        sgkf = StratifiedGroupKFold(n_splits=n_splits)

        oof_preds = [""] * len(self.df)
        feature_importances = []

        for tr_idx, val_idx in sgkf.split(X, y, groups=groups):
            clf = RandomForestClassifier(
                n_estimators=50,
                random_state=self.random_state,
                class_weight="balanced",
                n_jobs=-1,
            )
            clf.fit(X[tr_idx], y[tr_idx])
            preds = clf.predict(X[val_idx])
            for i_loc, i_glob in enumerate(val_idx):
                oof_preds[i_glob] = preds[i_loc]
            feature_importances.append(clf.feature_importances_)

        acc = float(accuracy_score(y, oof_preds))
        cm = confusion_matrix(y, oof_preds, labels=self.machines).tolist()

        mean_importances = np.mean(feature_importances, axis=0)
        feat_rank = [
            {"feature": f, "importance": float(imp)}
            for f, imp in sorted(zip(FEATURE_NAMES, mean_importances), key=lambda x: x[1], reverse=True)
        ]

        is_strongly_separable = acc >= 0.80

        return {
            "machine_count": n_machines,
            "machines": self.machines,
            "accuracy": acc,
            "is_separable": is_strongly_separable,
            "confusion_matrix": cm,
            "feature_importance_ranking": feat_rank,
            "top_identifying_features": [item["feature"] for item in feat_rank[:3]],
            "interpretation": (
                "High machine separability detected. The features carry hardware-specific "
                "baselines (e.g. idle RAM or core count differences) that allow identifying the host."
                if is_strongly_separable
                else "Low machine separability. Telemetry features are predominantly condition-driven "
                "rather than host-identifying."
            ),
        }
