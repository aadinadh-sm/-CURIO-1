"""Runner Script: CURIO Leave-One-Machine-Out (LOMO) Cross-Machine Evaluation Dry Run.

Executes the complete LOMO validation pipeline on a 3-machine synthetic benchmark:
1. Generates 3-machine dataset across 4 conditions (24 sessions, 264 rolling windows).
2. Verifies strict dataset integrity (no NaNs, no Infs, session-machine uniqueness).
3. Trains Calibrated RandomForestClassifier across 3 LOMO folds.
4. Performs grouped probability calibration via StratifiedGroupKFold (zero session leakage).
5. Computes fold-specific empirical 95th-percentile abnormality thresholds.
6. Evaluates held-out predictions against the deterministic Heuristic Baseline.
7. Saves models and metadata to data/models/fold_<machine_id>/.
8. Saves confusion matrices, calibration curves, and feature importances to reports/.
9. Writes the comprehensive evaluation report to reports/curio_lomo_evaluation.md.
"""

import sys
import os

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.model_trainer import (
    LOMOTrainer,
    generate_synthetic_lomo_dataset,
    validate_dataset,
)


def main():
    print("=" * 70)
    print("CURIO — LEAVE-ONE-MACHINE-OUT (LOMO) DRY RUN PIPELINE")
    print("=" * 70)

    # 1. Generate Synthetic 3-Machine Dataset
    print("\n[Step 1/5] Generating 3-machine synthetic dataset...")
    df = generate_synthetic_lomo_dataset(
        n_machines=3,
        sessions_per_class=2,
        windows_per_session=11,  # Matches 30s capture (61 samples at 2 Hz -> 11 windows)
        random_state=42,
    )
    print(f" -> Generated {len(df)} feature windows across {df['session_id'].nunique()} sessions.")
    print(f" -> Machines: {sorted(df['machine_id'].unique())}")
    print(f" -> Class distribution:\n{df['condition'].value_counts().to_string()}")

    # 2. Validate Dataset Integrity
    print("\n[Step 2/5] Validating dataset integrity...")
    validate_dataset(df)
    print(" -> Data validation passed cleanly. Zero NaNs, zero Infs, strictly valid schema.")

    # 3. Initialize LOMO Trainer
    print("\n[Step 3/5] Initializing LOMOTrainer with grouped calibration...")
    trainer = LOMOTrainer(
        df=df,
        rf_params={"n_estimators": 100, "random_state": 42, "n_jobs": -1, "class_weight": "balanced"},
        n_calibration_splits=3,
        random_state=42,
    )

    # 4. Run LOMO Evaluation
    print("\n[Step 4/5] Executing Leave-One-Machine-Out cross-validation loop...")
    results = trainer.run_lomo_evaluation(
        output_models_dir="data/models",
        output_reports_dir="reports",
        generate_plots=True,
    )

    # 5. Summary Output
    print("\n[Step 5/5] Evaluation completed successfully!")
    print("=" * 70)
    print("LOMO EVALUATION SUMMARY")
    print("=" * 70)

    print("\n--- Per-Fold Generalization ---")
    for m_id, fold_info in results["per_fold"].items():
        rf_m = fold_info["rf_metrics"]
        base_m = fold_info["baseline_metrics"]
        theta = fold_info["abnormality_threshold"]
        print(f"Held-Out Machine: {m_id}")
        print(f"  Test Windows:       {fold_info['test_sample_count']}")
        print(f"  RF Accuracy:        {rf_m['accuracy']:.4f}")
        print(f"  RF Macro F1:        {rf_m['macro_f1']:.4f}  (Baseline F1: {base_m['macro_f1']:.4f})")
        print(f"  Abnormality Theta:  {theta:.4f}")
        print(f"  Brier Score:        {rf_m.get('brier_score', 0.0):.4f}")
        print(f"  Log Loss:           {rf_m.get('log_loss', 0.0):.4f}")

    agg_rf = results["aggregate_rf"]
    agg_base = results["aggregate_baseline"]
    print("\n--- Aggregate Cross-Machine Performance ---")
    print(f"RF Overall Accuracy:      {agg_rf['accuracy']:.4f}  (Baseline: {agg_base['accuracy']:.4f})")
    print(f"RF Macro Precision:       {agg_rf['macro_precision']:.4f}  (Baseline: {agg_base['macro_precision']:.4f})")
    print(f"RF Macro Recall:          {agg_rf['macro_recall']:.4f}  (Baseline: {agg_base['macro_recall']:.4f})")
    print(f"RF Macro F1:              {agg_rf['macro_f1']:.4f}  (Baseline: {agg_base['macro_f1']:.4f})")
    print(f"Multiclass Brier Score:   {agg_rf.get('brier_score', 0.0):.4f}")
    print(f"Multiclass Log Loss:      {agg_rf.get('log_loss', 0.0):.4f}")

    print("\n--- Artifacts Created ---")
    print(f"Report:                   {results['report_path']}")
    for plot_name, path in results["plot_paths"].items():
        print(f"Plot ({plot_name}):        {path}")
    print("Models directory:         data/models/fold_<machine_id>/")

    print("\nDry run completed with zero data leakage!")


if __name__ == "__main__":
    main()
