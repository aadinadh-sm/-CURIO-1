"""CURIO Startup and Orchestration Script.

Validates machine learning and diagnostic deployment artifacts, starts the
FastAPI backend service bound strictly to 127.0.0.1:8000, optionally launches
the Vite development frontend on 127.0.0.1:3000, and displays local endpoints.

Usage:
    python scripts/run_curio.py
    python scripts/run_curio.py --no-frontend
    python scripts/run_curio.py --browser
"""

import argparse
import os
import signal
import subprocess
import sys
import time
import webbrowser
from typing import List, Optional

# Ensure repository root is on sys.path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

DEFAULT_MODEL_DIR = os.path.join(REPO_ROOT, "data", "models", "physical_deployment")
REQUIRED_ARTIFACTS = [
    "model.joblib",
    "calibration_metadata.json",
    "abnormality_threshold.json",
    "feature_metadata.json",
    "evidence_reference.json",
    "discovery_reference.json",
]


def validate_artifacts(model_dir: str = DEFAULT_MODEL_DIR) -> bool:
    """Validates that all necessary model and reference artifacts are present."""
    print("==================================================")
    print("1. VALIDATING DEPLOYMENT ARTIFACTS")
    print("==================================================")
    print(f"Target directory: {model_dir}")

    if not os.path.isdir(model_dir):
        print(f"ERROR: Model directory not found: {model_dir}")
        print("Please run: python scripts/build_deployment_model.py")
        return False

    all_found = True
    for artifact in REQUIRED_ARTIFACTS:
        art_path = os.path.join(model_dir, artifact)
        if os.path.isfile(art_path):
            size_kb = os.path.getsize(art_path) / 1024.0
            print(f"  [OK] {artifact} ({size_kb:.1f} KB)")
        else:
            print(f"  [MISSING] {artifact}")
            all_found = False

    if not all_found:
        print("\nERROR: Missing required deployment artifacts.")
        print("Generate deployment artifacts with: python scripts/build_deployment_model.py")
        return False

    print("All required deployment artifacts verified.\n")
    return True


def start_services(
    host: str = "127.0.0.1",
    backend_port: int = 8000,
    frontend_port: int = 3000,
    start_frontend: bool = True,
    open_browser: bool = False,
):
    """Starts the FastAPI backend and optional Vite frontend dev server."""
    # Enforce local-only host
    if host not in ("127.0.0.1", "localhost"):
        print(f"ERROR: Security violation. CURIO must bind ONLY to 127.0.0.1 (got: {host})")
        sys.exit(1)

    # 1. Validate Artifacts
    if not validate_artifacts():
        sys.exit(1)

    print("==================================================")
    print("2. STARTING CURIO LOCAL SERVICES")
    print("==================================================")

    processes: List[subprocess.Popen] = []

    def cleanup(signum=None, frame=None):
        print("\nShutting down CURIO services...")
        for p in processes:
            try:
                p.terminate()
                p.wait(timeout=2)
            except Exception:
                try:
                    p.kill()
                except Exception:
                    pass
        print("All CURIO processes stopped safely.")
        sys.exit(0)

    signal.signal(signal.SIGINT, cleanup)
    signal.signal(signal.SIGTERM, cleanup)

    # Launch Backend via Uvicorn subprocess
    backend_cmd = [
        sys.executable,
        "-m",
        "uvicorn",
        "src.server:app",
        "--host",
        host,
        "--port",
        str(backend_port),
        "--log-level",
        "info",
    ]
    print(f"Starting Backend on http://{host}:{backend_port}...")
    backend_proc = subprocess.Popen(backend_cmd, cwd=REPO_ROOT)
    processes.append(backend_proc)

    # Launch Frontend Dev Server if requested and frontend folder exists
    frontend_dir = os.path.join(REPO_ROOT, "frontend")
    frontend_proc: Optional[subprocess.Popen] = None
    if start_frontend and os.path.isdir(frontend_dir):
        # Check if npm is runnable
        try:
            npm_cmd = "npm.cmd" if sys.platform == "win32" else "npm"
            dev_cmd = [npm_cmd, "run", "dev", "--", "--port", str(frontend_port), "--host", host]
            print(f"Starting Vite Frontend Dev Server on http://{host}:{frontend_port}...")
            frontend_proc = subprocess.Popen(dev_cmd, cwd=frontend_dir)
            processes.append(frontend_proc)
        except Exception as e:
            print(f"Notice: Could not launch frontend dev server ({e}).")
            print(f"Built frontend is accessible via Backend on http://{host}:{backend_port}")

    # Allow servers a brief moment to initialize
    time.sleep(1.5)

    print("\n==================================================")
    print("CURIO READY")
    print("==================================================")
    print(f"\nLocal product page (optional):\nhttp://{host}:{backend_port}/")
    print(f"\nCURIO diagnostic app:\nhttp://{host}:{backend_port}/app/")
    print(f"\nLocal API:\nhttp://{host}:{backend_port}/api")
    if frontend_proc and frontend_proc.poll() is None:
        print(f"\nVite app server:\nhttp://{host}:{frontend_port}/app/")
    else:
        print("The app and API are served together by the local backend.")
    print("==================================================")
    print("Local-only mode active. No data leaves this device.")
    print("Press Ctrl+C to terminate all services.\n")

    if open_browser:
        # The root route is the product/marketing page. The actual diagnostic
        # interface is a separate Vite entry mounted at /app/ in both dev and
        # release builds.
        target_url = (
            f"http://{host}:{frontend_port}/app/"
            if (frontend_proc and frontend_proc.poll() is None)
            else f"http://{host}:{backend_port}/app/"
        )
        webbrowser.open(target_url)

    try:
        while True:
            # Check if backend died unexpectedly
            if backend_proc.poll() is not None:
                print(f"Backend terminated with exit code {backend_proc.returncode}.")
                break
            if frontend_proc and frontend_proc.poll() is not None:
                print(f"Frontend dev server terminated with exit code {frontend_proc.returncode}.")
                break
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        cleanup()


def main():
    parser = argparse.ArgumentParser(description="CURIO Local Intelligence Web Application Runner")
    parser.add_argument("--host", default="127.0.0.1", help="Binding host (strictly 127.0.0.1)")
    parser.add_argument("--backend-port", type=int, default=8000, help="Backend port (default: 8000)")
    parser.add_argument("--frontend-port", type=int, default=3000, help="Frontend port (default: 3000)")
    parser.add_argument(
        "--no-frontend",
        action="store_true",
        help="Do not launch Vite dev server; serve built frontend from backend",
    )
    parser.add_argument("--browser", action="store_true", help="Automatically open browser on launch")

    args = parser.parse_args()
    start_services(
        host=args.host,
        backend_port=args.backend_port,
        frontend_port=args.frontend_port,
        start_frontend=not args.no_frontend,
        open_browser=args.browser,
    )


if __name__ == "__main__":
    main()
