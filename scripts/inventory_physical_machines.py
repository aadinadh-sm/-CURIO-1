"""Physical Machine Inventory Script for CURIO.

Inspects, fingerprints, and catalogs participating physical machines:
- machine_id
- operating_system
- CPU model and architecture
- logical & physical CPU core counts
- total RAM & swap
- storage details
- Python & library environment (psutil, scikit-learn, numpy, pandas)
- Hardware diversity check (< 3 physical machines warning)
- Machine A similarity detection (prevents masquerading Machine A as B or C)
"""

import argparse
import json
import os
import platform
import subprocess
import sys
from typing import Any, Dict, List, Optional
import psutil

METADATA_DIR = "data/physical_metadata"

# Known fingerprint for physical_machine_A (to prevent accidental/falsified reuse)
MACHINE_A_HOSTNAME = "LAPTOP-I0G3I8PA"


def get_cpu_model_name() -> str:
    """Retrieves human-readable CPU brand string across platforms."""
    system = platform.system()
    try:
        if system == "Windows":
            cmd = "wmic cpu get name /value"
            output = subprocess.check_output(cmd, shell=True, text=True, stderr=subprocess.DEVNULL)
            for line in output.splitlines():
                if line.strip().startswith("Name="):
                    return line.split("Name=", 1)[1].strip()
            return platform.processor() or "Unknown Windows CPU"
        elif system == "Linux":
            with open("/proc/cpuinfo", "r") as f:
                for line in f:
                    if "model name" in line:
                        return line.split(":", 1)[1].strip()
        elif system == "Darwin":
            cmd = ["sysctl", "-n", "machdep.cpu.brand_string"]
            return subprocess.check_output(cmd, text=True).strip()
    except Exception:
        pass
    return platform.processor() or "Generic CPU"


def get_curio_version() -> str:
    """Retrieves current git commit or version identifier."""
    try:
        git_hash = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        return f"git-{git_hash}"
    except Exception:
        return "v0.9.0-dev"


def get_local_machine_inventory(machine_id: str = "physical_machine_A") -> Dict[str, Any]:
    """Inspects and catalogs the current host hardware specifications."""
    mem = psutil.virtual_memory()
    swap = psutil.swap_memory()
    partitions = []
    for p in psutil.disk_partitions(all=False):
        try:
            usage = psutil.disk_usage(p.mountpoint)
            partitions.append({
                "device": p.device,
                "mountpoint": p.mountpoint,
                "fstype": p.fstype,
                "total_gb": round(usage.total / (1024**3), 2),
            })
        except Exception:
            continue

    sklearn_version = "not_installed"
    try:
        import sklearn
        sklearn_version = sklearn.__version__
    except ImportError:
        pass

    return {
        "machine_id": machine_id,
        "hostname": platform.node(),
        "operating_system": f"{platform.system()} {platform.release()} ({platform.version()})",
        "cpu_model": get_cpu_model_name(),
        "architecture": platform.machine(),
        "logical_cpu_count": psutil.cpu_count(logical=True),
        "physical_cpu_count": psutil.cpu_count(logical=False),
        "total_ram_gb": round(mem.total / (1024**3), 2),
        "total_swap_gb": round(swap.total / (1024**3), 2),
        "storage_partitions": partitions,
        "python_version": platform.python_version(),
        "psutil_version": psutil.__version__,
        "sklearn_version": sklearn_version,
        "curio_version": get_curio_version(),
    }


def inventory_all_machines(
    target_machine_id: Optional[str] = None,
    metadata_dir: str = METADATA_DIR,
) -> Dict[str, Any]:
    """Catalogs all participating physical machines from metadata and local host.

    Enforces the rule: DO NOT fabricate machines. If fewer than 3 genuinely distinct
    physical machines are available, report:
    'Insufficient physical machine diversity for the planned 3-machine LOMO experiment.'
    """
    os.makedirs(metadata_dir, exist_ok=True)
    inventory_file = os.path.join(metadata_dir, "machine_inventory.json")

    existing_inventory: Dict[str, Any] = {}
    if os.path.exists(inventory_file):
        try:
            with open(inventory_file, "r") as f:
                existing_inventory = json.load(f)
        except Exception:
            existing_inventory = {}

    local_hostname = platform.node()
    current_machine_id = target_machine_id or "physical_machine_A"

    # Similarity check: warn if attempting to register Machine A host as B or C
    is_machine_a_clone = False
    if current_machine_id in ("physical_machine_B", "physical_machine_C"):
        if local_hostname.upper() == MACHINE_A_HOSTNAME.upper():
            is_machine_a_clone = True

    local_info = get_local_machine_inventory(current_machine_id)

    # Only record if not an illegal disguise of Machine A
    if not is_machine_a_clone:
        existing_inventory[current_machine_id] = local_info
        with open(inventory_file, "w", encoding="utf-8") as f:
            json.dump(existing_inventory, f, indent=2)

    distinct_machines = list(existing_inventory.keys())
    n_machines = len(distinct_machines)

    has_sufficient_diversity = (n_machines >= 3)
    diversity_warning = None
    if not has_sufficient_diversity:
        diversity_warning = (
            "Insufficient physical machine diversity for the planned 3-machine LOMO experiment."
        )

    return {
        "distinct_machines_count": n_machines,
        "machines": existing_inventory,
        "has_sufficient_diversity": has_sufficient_diversity,
        "diversity_warning": diversity_warning,
        "is_machine_a_clone": is_machine_a_clone,
        "current_machine_id": current_machine_id,
        "local_info": local_info,
    }


def main():
    parser = argparse.ArgumentParser(description="CURIO Physical Machine Inventory & Fingerprinting")
    parser.add_argument(
        "--machine_id",
        default="physical_machine_A",
        help="Identifier for local machine (e.g. physical_machine_A, physical_machine_B, physical_machine_C)",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("CURIO PHYSICAL MACHINE INVENTORY")
    print("=" * 70)

    res = inventory_all_machines(target_machine_id=args.machine_id)

    if res.get("is_machine_a_clone"):
        print("\n" + "!" * 70)
        print("CRITICAL WARNING: HOSTNAME HARDWARE COLLISION DETECTED!")
        print(f"Current host '{platform.node()}' matches physical_machine_A.")
        print(f"Attempting to register this machine as '{args.machine_id}' is rejected.")
        print("Cross-machine LOMO evaluation strictly requires physically distinct hardware.")
        print("Do NOT reuse physical_machine_A under an alias.")
        print("!" * 70)

    n = res["distinct_machines_count"]
    print(f"\nTotal physical machines cataloged in inventory: {n}")

    for m_id, m_data in res["machines"].items():
        print(f"\n[{m_id}]")
        print(f"  Hostname:     {m_data['hostname']}")
        print(f"  OS:           {m_data['operating_system']}")
        print(f"  CPU Model:    {m_data['cpu_model']}")
        print(f"  CPU Cores:    {m_data['logical_cpu_count']} logical, {m_data['physical_cpu_count']} physical")
        print(f"  RAM:          {m_data['total_ram_gb']} GB")
        print(f"  Python:       {m_data['python_version']} (psutil {m_data['psutil_version']}, sklearn {m_data.get('sklearn_version', 'N/A')})")
        print(f"  CURIO:        {m_data['curio_version']}")

    print("\n" + "=" * 70)
    if not res["has_sufficient_diversity"]:
        print("DIVERSITY ASSESSMENT:")
        print(f"  [!] {res['diversity_warning']}")
        print(f"  Only {n} physical machine(s) available in this environment.")
        print("  DO NOT fabricate virtual/synthetic machines as physical machines.")
    else:
        print("DIVERSITY ASSESSMENT: Sufficient physical machines cataloged (>= 3).")
    print("=" * 70)


if __name__ == "__main__":
    main()
