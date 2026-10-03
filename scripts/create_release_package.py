"""CURIO Release Package Builder.

Packages clean project source code, documentation, reports, deployment models,
and compiled frontend assets into a standalone distribution archive:
    dist/curio_release.zip
and computes its cryptographic checksum:
    dist/SHA256SUMS.txt

Usage:
    python scripts/create_release_package.py
"""

import hashlib
import os
import sys
import zipfile

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DIST_DIR = os.path.join(REPO_ROOT, "dist")
ZIP_PATH = os.path.join(DIST_DIR, "curio_release.zip")
SHA256_PATH = os.path.join(DIST_DIR, "SHA256SUMS.txt")

# Items to include in the release package (relative to REPO_ROOT)
INCLUDE_ITEMS = [
    "src",
    "scripts",
    "tests",
    "docs",
    "reports",
    os.path.join("data", "physical_raw"),
    os.path.join("data", "models", "physical_deployment"),
    os.path.join("data", "physical_metadata"),
    os.path.join("frontend", "src"),
    os.path.join("frontend", "public"),
    os.path.join("frontend", "dist"),
    os.path.join("frontend", "index.html"),
    os.path.join("frontend", "app"),
    os.path.join("frontend", "package.json"),
    os.path.join("frontend", "tsconfig.json"),
    os.path.join("frontend", "tsconfig.app.json"),
    os.path.join("frontend", "tsconfig.node.json"),
    os.path.join("frontend", "vite.config.ts"),
    "README.md",
    "requirements.txt",
    "START-CURIO.bat",
    "START-CURIO-HERE.txt",
    ".gitignore",
    "CURIO_VERSION",
]

# Patterns or directories to strictly exclude
EXCLUDE_SUBSTRINGS = [
    "__pycache__",
    ".pytest_cache",
    "node_modules",
    ".curio-venv",
    ".vite",
    ".git",
    ".pyc",
    ".pyo",
    ".log",
    ".tmp",
    # The downloadable bundle is copied into the site after this archive is built.
    "frontend/public/downloads/",
    "frontend/dist/downloads/",
]


def should_exclude(rel_path: str) -> bool:
    normalized = rel_path.replace("\\", "/")
    return any(ex in normalized for ex in EXCLUDE_SUBSTRINGS)


def create_package():
    print("==================================================")
    print("CURIO RELEASE PACKAGING")
    print("==================================================")
    os.makedirs(DIST_DIR, exist_ok=True)

    # Remove existing release archive if present
    if os.path.exists(ZIP_PATH):
        try:
            os.remove(ZIP_PATH)
        except Exception:
            pass

    file_count = 0
    total_bytes = 0

    print(f"Creating archive: {ZIP_PATH}")
    with zipfile.ZipFile(ZIP_PATH, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zipf:
        for item in INCLUDE_ITEMS:
            item_path = os.path.join(REPO_ROOT, item)
            if not os.path.exists(item_path):
                continue

            if os.path.isfile(item_path):
                rel_path = os.path.relpath(item_path, REPO_ROOT)
                if not should_exclude(rel_path):
                    zipf.write(item_path, rel_path)
                    file_count += 1
                    total_bytes += os.path.getsize(item_path)
            elif os.path.isdir(item_path):
                for root, dirs, files in os.walk(item_path):
                    # Filter subdirs in place
                    dirs[:] = [d for d in dirs if not should_exclude(d)]
                    for f in files:
                        full_fpath = os.path.join(root, f)
                        rel_path = os.path.relpath(full_fpath, REPO_ROOT)
                        if not should_exclude(rel_path):
                            zipf.write(full_fpath, rel_path)
                            file_count += 1
                            total_bytes += os.path.getsize(full_fpath)

    zip_size_mb = os.path.getsize(ZIP_PATH) / (1024.0 * 1024.0)
    print(f"\nArchive created successfully:")
    print(f"  Files packaged: {file_count}")
    print(f"  Source size:    {total_bytes / (1024.0 * 1024.0):.2f} MB")
    print(f"  Compressed:     {zip_size_mb:.2f} MB")

    # Calculate SHA256
    print("\nComputing SHA-256 cryptographic checksum...")
    sha256 = hashlib.sha256()
    with open(ZIP_PATH, "rb") as f:
        while chunk := f.read(65536):
            sha256.update(chunk)
    digest = sha256.hexdigest()

    checksum_line = f"{digest}  curio_release.zip\n"
    with open(SHA256_PATH, "w", encoding="utf-8") as f:
        f.write(checksum_line)

    print(f"  SHA256: {digest}")
    print(f"  Checksum written to: {SHA256_PATH}")
    print("==================================================")
    print("CURIO RELEASE PACKAGE READY")
    print("==================================================")


if __name__ == "__main__":
    create_package()
