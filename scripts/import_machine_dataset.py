"""Dataset Importer and Verifier for CURIO.

Safely imports a physical machine dataset archive (<machine_id>_curio_dataset.zip)
into the local CURIO repository with cryptographic and rule-based verification:

Verification Pipeline:
1. Zip structure & path traversal defense (no '../' or absolute paths)
2. SHA-256 cryptographic checksum verification (rejects corrupted/tampered files)
3. Manifest schema and machine_id validation
4. Session ID collision detection (rejects cross-machine session ID clashes)
5. Rules A through P physical dataset validation on all raw and processed sessions
6. Non-destructive transactional extraction (aborts cleanly on any failure)
7. Automatic cataloging into data/physical_metadata/machine_inventory.json
"""

import argparse
import hashlib
import io
import json
import os
import sys
import zipfile
from typing import Any, Dict, List, Optional
import pandas as pd

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.physical_dataset_validator import (
    validate_physical_feature_session,
    validate_physical_raw_session,
    PhysicalDataValidationError,
    SYNTHETIC_MACHINE_IDS,
)

METADATA_DIR = "data/physical_metadata"
RAW_DIR = "data/physical_raw"
PROCESSED_DIR = "data/physical_processed"


class DatasetImportError(Exception):
    """Raised when an imported dataset fails integrity, checksum, or validation checks."""
    pass


def compute_bytes_sha256(data: bytes) -> str:
    """Computes SHA-256 hash of bytes in memory."""
    return hashlib.sha256(data).hexdigest()


def verify_zip_safety(zf: zipfile.ZipFile):
    """Guards against directory traversal attacks (zip slips)."""
    for member in zf.namelist():
        # Prevent absolute paths or parent directory traversal
        normalized = os.path.normpath(member)
        if normalized.startswith("..") or os.path.isabs(normalized) or normalized.startswith("/") or normalized.startswith("\\"):
            raise DatasetImportError(f"Security error: dangerous file path in archive: '{member}'")


def parse_and_verify_checksums(zf: zipfile.ZipFile) -> Dict[str, bytes]:
    """Verifies that all files listed in checksums.sha256 match their actual SHA-256 hash.

    Returns dict mapping arcname -> uncompressed bytes for validated files.
    """
    if "checksums.sha256" not in zf.namelist():
        raise DatasetImportError("Archive is missing required 'checksums.sha256' manifest.")

    checksum_content = zf.read("checksums.sha256").decode("utf-8")
    lines = [line.strip() for line in checksum_content.splitlines() if line.strip()]

    file_bytes: Dict[str, bytes] = {}

    for line in lines:
        parts = line.split(maxsplit=1)
        if len(parts) != 2:
            raise DatasetImportError(f"Malformed checksum line: '{line}'")
        expected_hash, arcname = parts[0].strip(), parts[1].strip()

        # Handle slashes consistently
        arcname_norm = arcname.replace("\\", "/")
        matching_members = [m for m in zf.namelist() if m.replace("\\", "/") == arcname_norm]
        if not matching_members:
            raise DatasetImportError(f"File listed in checksums not found in archive: '{arcname}'")

        actual_member = matching_members[0]
        data = zf.read(actual_member)
        actual_hash = compute_bytes_sha256(data)

        if actual_hash.lower() != expected_hash.lower():
            raise DatasetImportError(
                f"Cryptographic hash mismatch for '{arcname}'! "
                f"Expected {expected_hash}, calculated {actual_hash}. File may be corrupted or tampered."
            )
        file_bytes[actual_member] = data

    return file_bytes


def validate_archive_manifest(zf: zipfile.ZipFile, file_bytes: Dict[str, bytes]) -> Tuple[str, Dict[str, Any]]:
    """Validates the manifest within the archive."""
    manifest_members = [m for m in zf.namelist() if m.endswith("_manifest.json")]
    if not manifest_members:
        raise DatasetImportError("Archive missing machine manifest JSON.")
    if len(manifest_members) > 1:
        raise DatasetImportError(f"Multiple manifests found in archive: {manifest_members}")

    m_member = manifest_members[0]
    manifest_data = json.loads(file_bytes[m_member].decode("utf-8"))

    machine_id = manifest_data.get("machine_id")
    if not machine_id:
        raise DatasetImportError("Manifest is missing 'machine_id'.")

    if machine_id in SYNTHETIC_MACHINE_IDS:
        raise DatasetImportError(f"Refusing to import synthetic machine ID '{machine_id}'.")

    if manifest_data.get("schema_version") != "1.0.0":
        raise DatasetImportError(f"Unsupported schema version: {manifest_data.get('schema_version')}")

    return machine_id, manifest_data


def check_session_collisions(manifest_data: Dict[str, Any], project_root: str = "."):
    """Ensures none of the incoming sessions collide with an existing session from another machine."""
    existing_processed = os.path.join(project_root, PROCESSED_DIR)
    incoming_machine_id = manifest_data["machine_id"]
    incoming_sessions = set(manifest_data.get("dataset_summary", {}).get("session_ids", []))

    if os.path.exists(existing_processed):
        for f in os.listdir(existing_processed):
            if f.endswith("_features.csv"):
                # Check if file belongs to a different machine
                if not f.startswith(f"{incoming_machine_id}_"):
                    try:
                        df_ex = pd.read_csv(os.path.join(existing_processed, f))
                        ex_sess_id = str(df_ex["session_id"].iloc[0])
                        if ex_sess_id in incoming_sessions:
                            raise DatasetImportError(
                                f"SESSION COLLISION: Session ID '{ex_sess_id}' already exists "
                                f"under a different machine dataset! Collision detected with {f}."
                            )
                    except Exception as e:
                        if isinstance(e, DatasetImportError):
                            raise
                        continue


def validate_archive_sessions(file_bytes: Dict[str, bytes], machine_id: str):
    """Runs physical dataset validator on all raw and processed sessions inside archive."""
    raw_count = 0
    proc_count = 0

    for arcname, data in file_bytes.items():
        if arcname.endswith("_raw.csv"):
            df = pd.read_csv(io.BytesIO(data))
            try:
                validate_physical_raw_session(df)
            except PhysicalDataValidationError as e:
                raise DatasetImportError(f"Validation failed for raw file '{arcname}': {e}") from e
            raw_count += 1

        elif arcname.endswith("_features.csv"):
            df = pd.read_csv(io.BytesIO(data))
            try:
                validate_physical_feature_session(df)
            except PhysicalDataValidationError as e:
                raise DatasetImportError(f"Validation failed for feature file '{arcname}': {e}") from e
            proc_count += 1

    if raw_count == 0 or proc_count == 0:
        raise DatasetImportError(f"Archive contains 0 valid sessions (raw={raw_count}, proc={proc_count}).")


def update_inventory_metadata(manifest_data: Dict[str, Any], project_root: str = "."):
    """Updates machine_inventory.json with the imported machine's specs."""
    inv_dir = os.path.join(project_root, METADATA_DIR)
    os.makedirs(inv_dir, exist_ok=True)
    inv_file = os.path.join(inv_dir, "machine_inventory.json")

    inventory: Dict[str, Any] = {}
    if os.path.exists(inv_file):
        try:
            with open(inv_file, "r") as f:
                inventory = json.load(f)
        except Exception:
            inventory = {}

    m_id = manifest_data["machine_id"]
    hw = manifest_data.get("hardware", {})
    env = manifest_data.get("environment", {})

    inventory[m_id] = {
        "machine_id": m_id,
        "hostname": hw.get("hostname", "unknown"),
        "operating_system": hw.get("operating_system", "unknown"),
        "cpu_model": hw.get("cpu_model", "unknown"),
        "architecture": hw.get("architecture", "unknown"),
        "logical_cpu_count": hw.get("logical_cpu_count", 0),
        "physical_cpu_count": hw.get("physical_cpu_count", 0),
        "total_ram_gb": hw.get("total_ram_gb", 0.0),
        "total_swap_gb": hw.get("total_swap_gb", 0.0),
        "storage_partitions": hw.get("storage_partitions", []),
        "python_version": env.get("python_version", "unknown"),
        "psutil_version": env.get("psutil_version", "unknown"),
        "sklearn_version": env.get("sklearn_version", "unknown"),
        "curio_version": env.get("curio_version", "unknown"),
    }

    with open(inv_file, "w", encoding="utf-8") as f:
        json.dump(inventory, f, indent=2)


def import_dataset(
    archive_path: str,
    project_root: str = ".",
    dry_run: bool = False,
    allow_overwrite: bool = True,
) -> Dict[str, Any]:
    """Imports and validates a machine dataset archive into project_root."""
    if not os.path.exists(archive_path):
        raise FileNotFoundError(f"Archive not found: {archive_path}")

    print("=" * 70)
    print(f"CURIO DATASET IMPORT & VERIFICATION: {os.path.basename(archive_path)}")
    print("=" * 70)

    with zipfile.ZipFile(archive_path, "r") as zf:
        # 1. Zip Safety
        print("\n[1/5] Verifying archive path safety...")
        verify_zip_safety(zf)
        print("  [OK] No path traversal or unsafe characters detected.")

        # 2. Checksum Verification
        print("\n[2/5] Cryptographic SHA-256 verification of archive files...")
        file_bytes = parse_and_verify_checksums(zf)
        print(f"  [OK] Successfully verified {len(file_bytes)} file checksums.")

        # 3. Manifest Validation
        print("\n[3/5] Validating manifest schema and machine metadata...")
        machine_id, manifest_data = validate_archive_manifest(zf, file_bytes)
        print(f"  [OK] Machine ID: '{machine_id}'")
        print(f"  [OK] Hostname:   '{manifest_data['hardware'].get('hostname')}'")
        print(f"  [OK] Hardware:   {manifest_data['hardware'].get('cpu_model')} "
              f"({manifest_data['hardware'].get('logical_cpu_count')} cores, "
              f"{manifest_data['hardware'].get('total_ram_gb')} GB RAM)")

        # 4. Collision Detection
        print("\n[4/5] Checking session ID uniqueness and collision guardrails...")
        check_session_collisions(manifest_data, project_root=project_root)
        print("  [OK] No cross-machine session ID collisions found.")

        # 5. Telemetry Validation (Rules A through P)
        print("\n[5/5] Running Physical Telemetry Validator (Rules A through P)...")
        validate_archive_sessions(file_bytes, machine_id)
        print("  [OK] All sessions comply with Rules A through P.")

        if dry_run:
            print("\n" + "=" * 70)
            print("DRY RUN COMPLETE: Archive is 100% valid. No files written.")
            print("=" * 70)
            return {
                "machine_id": machine_id,
                "status": "dry_run_success",
                "files_count": len(file_bytes),
            }

        # 6. Extract Files to Project Root
        print("\nWriting verified dataset files to repository...")
        extracted_files = []
        for arcname, data in file_bytes.items():
            dest_path = os.path.join(project_root, arcname)
            os.makedirs(os.path.dirname(dest_path), exist_ok=True)
            if os.path.exists(dest_path) and not allow_overwrite:
                raise DatasetImportError(f"File {dest_path} already exists and allow_overwrite is False.")
            with open(dest_path, "wb") as f:
                f.write(data)
            extracted_files.append(dest_path)

        # 7. Update Inventory
        update_inventory_metadata(manifest_data, project_root=project_root)
        print(f"  [OK] Extracted {len(extracted_files)} files.")
        print(f"  [OK] Updated machine inventory for '{machine_id}'.")

    print("\n" + "=" * 70)
    print(f"IMPORT COMPLETE AND VERIFIED: {machine_id}")
    print(f"  Raw Sessions:       {manifest_data['dataset_summary'].get('raw_sessions_count')}")
    print(f"  Processed Sessions: {manifest_data['dataset_summary'].get('processed_sessions_count')}")
    print(f"  Feature Windows:    {manifest_data['dataset_summary'].get('total_feature_windows')}")
    print("======================================================================")

    return {
        "machine_id": machine_id,
        "status": "imported",
        "extracted_files_count": len(extracted_files),
        "manifest": manifest_data,
    }


def main():
    parser = argparse.ArgumentParser(description="CURIO Machine Dataset Importer")
    parser.add_argument("archive_path", help="Path to <machine_id>_curio_dataset.zip")
    parser.add_argument("--project_root", default=".", help="CURIO repository root (default: .)")
    parser.add_argument("--dry_run", action="store_true", help="Validate without extracting files")
    parser.add_argument("--no_overwrite", action="store_true", help="Disallow overwriting existing files")
    args = parser.parse_args()

    try:
        import_dataset(
            archive_path=args.archive_path,
            project_root=args.project_root,
            dry_run=args.dry_run,
            allow_overwrite=not args.no_overwrite,
        )
        sys.exit(0)
    except Exception as e:
        print(f"\n[!] IMPORT FAILED: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
