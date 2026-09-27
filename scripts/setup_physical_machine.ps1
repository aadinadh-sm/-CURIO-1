# CURIO Physical Machine Setup Script (PowerShell)
# Sets up environment and directory structure for collecting physical telemetry on Machine B or Machine C.

$ErrorActionPreference = "Stop"

Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host "CURIO: Physical Machine Setup & Environment Preparation" -ForegroundColor Cyan
Write-Host "======================================================================" -ForegroundColor Cyan

# 1. Check Python
Write-Host "`n[1/5] Checking Python installation..." -ForegroundColor Yellow
try {
    $pythonVersion = python --version 2>&1
    Write-Host "  Found: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Error "Python is not installed or not in PATH. Please install Python 3.10+ (x64) and re-run."
    exit 1
}

# 2. Check and Install Pip Requirements
Write-Host "`n[2/5] Checking required dependencies..." -ForegroundColor Yellow
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectRoot = Split-Path -Parent $scriptDir
$reqFile = Join-Path $projectRoot "requirements.txt"

if (Test-Path $reqFile) {
    Write-Host "  Installing/verifying packages from requirements.txt..." -ForegroundColor Gray
    python -m pip install --upgrade pip --quiet
    python -m pip install -r $reqFile --quiet
    Write-Host "  Dependencies verified." -ForegroundColor Green
} else {
    Write-Warning "requirements.txt not found at $reqFile. Installing standard CURIO packages directly..."
    python -m pip install psutil scikit-learn numpy pandas joblib --quiet
}

# 3. Create Required Directory Structure
Write-Host "`n[3/5] Creating directory structure..." -ForegroundColor Yellow
$dirs = @(
    (Join-Path $projectRoot "data/physical_raw"),
    (Join-Path $projectRoot "data/physical_processed"),
    (Join-Path $projectRoot "data/physical_metadata"),
    (Join-Path $projectRoot "data/exports"),
    (Join-Path $projectRoot "reports")
)

foreach ($dir in $dirs) {
    if (-not (Test-Path $dir)) {
        New-Item -ItemType Directory -Path $dir -Force | Out-Null
        Write-Host "  Created: $dir" -ForegroundColor Gray
    } else {
        Write-Host "  Exists:  $dir" -ForegroundColor Gray
    }
}
Write-Host "  Directory structure ready." -ForegroundColor Green

# 4. Verify Python Modules
Write-Host "`n[4/5] Verifying Python library imports..." -ForegroundColor Yellow
$verifyCmd = "import psutil, sklearn, numpy, pandas, joblib; print(f'psutil {psutil.__version__}, sklearn {sklearn.__version__}, numpy {numpy.__version__}, pandas {pandas.__version__}')"
try {
    $modOutput = python -c $verifyCmd 2>&1
    Write-Host "  Modules verified: $modOutput" -ForegroundColor Green
} catch {
    Write-Error "Failed to import required libraries: $_"
    exit 1
}

# 5. Check Host Info
Write-Host "`n[5/5] Running initial hardware inventory..." -ForegroundColor Yellow
$invScript = Join-Path $projectRoot "scripts/inventory_physical_machines.py"
if (Test-Path $invScript) {
    python $invScript
} else {
    Write-Host "  Inventory script not found, skipping." -ForegroundColor Gray
}

Write-Host "`n======================================================================" -ForegroundColor Cyan
Write-Host "SETUP COMPLETE: Machine environment is ready for CURIO collection." -ForegroundColor Cyan
Write-Host "Next step: Run 'python scripts/check_physical_machine.py --machine_id <machine_id>'" -ForegroundColor Cyan
Write-Host "======================================================================" -ForegroundColor Cyan
