"""Build Deployment Model & Evidence Reference Profile for Physical Telemetry.

Loads all verified physical telemetry sessions from data/physical_processed/,
trains a Calibrated Random Forest model with session-grouped internal CV,
computes the out-of-fold empirical 95th-percentile abnormality threshold,
fits the training Evidence Reference Profile, and saves all deployment artifacts to:
data/models/physical_deployment/
  - model.joblib
  - calibration_metadata.json
  - abnormality_threshold.json
  - feature_metadata.json
  - evidence_reference.json
"""

import argparse
import os
import sys
import pandas as pd

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.dataset_builder import DatasetBuilder
from src.model_trainer import LOMOTrainer

PHYSICAL_PROCESSED_DIR = "data/physical_processed"
DEFAULT_OUTPUT_DIR = "data/models/physical_deployment"


def build_deployment_artifacts(
    processed_dir: str = PHYSICAL_PROCESSED_DIR,
    output_dir: str = DEFAULT_OUTPUT_DIR,
) -> None:
    print("=" * 70)
    print("CURIO PHYSICAL DEPLOYMENT MODEL & EVIDENCE PROFILE BUILDER")
    print("=" * 70)

    builder = DatasetBuilder(processed_dir=processed_dir)
    df = builder.load_all_processed_features()

    if df.empty:
        raise ValueError(f"No processed feature sessions found in '{processed_dir}'.")

    # Filter to physical machines only
    unique_machines = sorted(df["machine_id"].unique())
    print(f"Loaded {len(df)} feature windows across {df['session_id'].nunique()} sessions.")
    print(f"Participating Machine(s): {unique_machines}")
    print(f"Class Distribution:\n{df['condition'].value_counts().to_dict()}")

    trainer = LOMOTrainer(df=df)
    results = trainer.train_deployment_model(output_dir=output_dir)

    print("\n" + "=" * 70)
    print(f"DEPLOYMENT ARTIFACTS GENERATED IN: {output_dir}")
    print("=" * 70)
    print(f"  Model:                 {os.path.join(output_dir, 'model.joblib')}")
    print(f"  Calibration Metadata:  {os.path.join(output_dir, 'calibration_metadata.json')}")
    print(f"  Abnormality Threshold: {os.path.join(output_dir, 'abnormality_threshold.json')} (theta = {results['abnormality_threshold']:.4f})")
    print(f"  Feature Metadata:      {os.path.join(output_dir, 'feature_metadata.json')}")
    print(f"  Evidence Reference:    {os.path.join(output_dir, 'evidence_reference.json')}")
    print(f"  Discovery Reference:   {os.path.join(output_dir, 'discovery_reference.json')}")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="CURIO Physical Deployment Artifact Builder")
    parser.add_argument("--processed_dir", default=PHYSICAL_PROCESSED_DIR, help="Path to processed features")
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR, help="Path to output models directory")
    args = parser.parse_args()

    build_deployment_artifacts(processed_dir=args.processed_dir, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
