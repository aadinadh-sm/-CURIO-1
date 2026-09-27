"""Dataset Packaging Utility for CURIO.

Packages a single physical machine's verified telemetry dataset into a portable,
tamper-evident zip archive:
<output_dir>/<machine_id>_curio_dataset.zip

Contains:
- data/physical_raw/<machine_id>_*_raw.csv
- data/physical_processed/<machine_id>_*_features.csv
- data/physical_metadata/<machine_id>_manifest.json
- checksums.sha256 (SHA256 hashes of all bundled files)

Strict Packaging Constraints:
- NEVER packages synthetic data, virtual environments, cache, or model binaries
- NEVER packages files belonging to other machine IDs
- Enforces pre-packaging validation using physical_dataset_validator (Rules A-P)
"""

import argparse
import hashlib
import os
import sys
import zipfile
from typing import Dict, List, Tuple
import pandas as pd

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scripts.create_machine_manifest import create_manifest
from src.physical_dataset_validator import (
    validate_physical_feature_session,
    validate_physical_raw_session,
)

RAW_DIR = "data/physical_raw"
PROCESSED_DIR = "data/physical_processed"
METADATA_DIR = "data/physical_metadata"
DEFAULT_EXPORT_DIR = "data/exports"


def compute_sha256(filepath: str) -> str:
    """Computes SHA-256 hash of a local file."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def find_machine_files(
    machine_id: str,
    raw_dir: str = RAW_DIR,
    processed_dir: str = PROCESSED_DIR,
    metadata_dir: str = METADATA_DIR,
) -> Dict[str, List[Tuple[str, str]]]:
    """Discovers all files belonging to machine_id, mapped to relative archive paths.

    Returns dict mapping category to list of (disk_path, archive_path).
    """
    raw_files = []
    if os.path.exists(raw_dir):
        for f in sorted(os.listdir(raw_dir)):
            if f.startswith(f"{machine_id}_") and f.endswith("_raw.csv"):
                disk_p = os.path.join(raw_dir, f)
                arc_p = f"data/physical_raw/{f}"
                raw_files.append((disk_p, arc_p))

    processed_files = []
    if os.path.exists(processed_dir):
        for f in sorted(os.listdir(processed_dir)):
            if f.startswith(f"{machine_id}_") and f.endswith("_features.csv"):
                disk_p = os.path.join(processed_dir, f)
                arc_p = f"data/physical_processed/{f}"
                processed_files.append((disk_p, arc_p))

    manifest_file = os.path.join(metadata_dir, f"{machine_id}_manifest.json")
    metadata_files = []
    if os.path.exists(manifest_file):
        metadata_files.append((manifest_file, f"data/physical_metadata/{machine_id}_manifest.json"))

    return {
        "raw": raw_files,
        "processed": processed_files,
        "metadata": metadata_files,
    }


def validate_before_packaging(machine_files: Dict[str, List[Tuple[str, str]]], machine_id: str):
    """Runs physical dataset validator on all raw and processed files for machine_id."""
    raw_entries = machine_files["raw"]
    proc_entries = machine_files["processed"]

    if not raw_entries or not proc_entries:
        raise ValueError(
            f"Cannot package {machine_id}: found {len(raw_entries)} raw and "
            f"{len(proc_entries)} processed files. Data must not be empty."
        )

    print(f"Validating {len(raw_entries)} raw sessions for {machine_id}...")
    for disk_p, _ in raw_entries:
        df = pd.read_csv(disk_p)
        validate_physical_raw_session(df)

    print(f"Validating {len(proc_entries)} feature sessions for {machine_id}...")
    for disk_p, _ in proc_entries:
        df = pd.read_csv(disk_p)
        validate_physical_feature_session(df)

    print("Pre-packaging validation PASSED (Rules A through P).")


def package_dataset(
    machine_id: str = "physical_machine_A",
    output_dir: str = DEFAULT_EXPORT_DIR,
    raw_dir: str = RAW_DIR,
    processed_dir: str = PROCESSED_DIR,
    metadata_dir: str = METADATA_DIR,
    skip_validation: bool = False,
) -> str:
    """Packages the verified dataset and manifest into a single zip archive."""
    os.makedirs(output_dir, exist_ok=True)

    # 1. Update / ensure manifest exists
    create_manifest(
        machine_id=machine_id,
        raw_dir=raw_dir,
        processed_dir=processed_dir,
        metadata_dir=metadata_dir,
    )

    # 2. Collect files
    machine_files = find_machine_files(
        machine_id=machine_id,
        raw_dir=raw_dir,
        processed_dir=processed_dir,
        metadata_dir=metadata_dir,
    )
    all_entries = machine_files["raw"] + machine_files["processed"] + machine_files["metadata"]

    if not machine_files["metadata"]:
        raise FileNotFoundError(f"Manifest for {machine_id} was not created.")

    # 3. Validate sessions
    if not skip_validation:
        validate_before_packaging(machine_files, machine_id)

    # 4. Generate Checksums
    checksum_lines = []
    for disk_p, arc_p in all_entries:
        digest = compute_sha256(disk_p)
        checksum_lines.append(f"{digest}  {arc_p}\n")

    # 5. Create Zip Archive
    zip_filename = f"{machine_id}_curio_dataset.zip"
    zip_filepath = os.path.join(output_dir, zip_filename)

    with zipfile.ZipFile(zip_filepath, "w", zipfile.ZIP_DEFLATED) as zf:
        # Write files
        for disk_p, arc_p in all_entries:
            zf.write(disk_p, arcname=arc_p)
        # Write checksum file at root of archive
        zf.writestr("checksums.sha256", "".join(checksum_lines))

    # Compute zip file hash
    archive_hash = compute_sha256(zip_filepath)
    archive_size_mb = os.path.getsize(zip_filepath) / (1024 * 1024)

    print("=" * 70)
    print(f"CURIO DATASET PACKAGED: {machine_id}")
    print("=" * 70)
    print(f"  Archive Path:    {zip_filepath}")
    print(f"  Archive Size:    {archive_size_mb:.2f} MB")
    print(f"  Archive SHA256:  {archive_hash}")
    print(f"  Raw Sessions:    {len(machine_files['raw'])}")
    print(f"  Processed Feat:  {len(machine_files['processed'])}")
    print(f"  Manifest:        {machine_files['metadata'][0][1]}")
    print(f"  Checksums:       checksums.sha256 ({len(checksum_lines)} entries)")
    print("=" * 70)

    return zip_filepath


def main():
    parser = argparse.ArgumentParser(description="CURIO Machine Dataset Packager")
    parser.add_argument(
        "--machine_id",
        default="physical_machine_A",
        help="Machine identifier to package (e.g. physical_machine_A, physical_machine_B, physical_machine_C)",
    )
    parser.add_argument(
        "--output_dir",
        default=DEFAULT_EXPORT_DIR,
        help=f"Directory to save zip archive (default: {DEFAULT_EXPORT_DIR})",
    )
    parser.add_argument(
        "--skip_validation",
        action="store_true",
        help="Skip strict validation before packaging (not recommended)",
    )
    args = parser.parse_args()

    package_dataset(
        machine_id=args.machine_id,
        output_dir=args.output_dir,
        skip_validation=args.skip_validation,
    )


if __name__ == "__main__":
    main()
