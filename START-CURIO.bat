@echo off
setlocal EnableExtensions EnableDelayedExpansion
cd /d "%~dp0"
title CURIO - Local Computer Health Check

echo.
echo  CURIO - Local Computer Health Check
echo  -----------------------------------
echo  CURIO runs on this computer. Your readings stay local.
echo.

if not exist ".curio-venv\Scripts\python.exe" (
  echo  First-time setup: finding a supported Python version...
  set "PYTHON_CMD="
  where py >nul 2>nul
  if not errorlevel 1 (
    py -3.13 --version >nul 2>nul && set "PYTHON_CMD=py -3.13"
    if not defined PYTHON_CMD py -3.12 --version >nul 2>nul && set "PYTHON_CMD=py -3.12"
  )
  if not defined PYTHON_CMD (
    where python >nul 2>nul
    if not errorlevel 1 (
      python -c "import sys; raise SystemExit(0 if (3,12) <= sys.version_info[:2] < (3,14) else 1)" >nul 2>nul && set "PYTHON_CMD=python"
    )
  )
  if not defined PYTHON_CMD (
    echo  Python 3.12 or 3.13 was not found.
    echo  Install Python from https://www.python.org/downloads/ and select "Add Python to PATH".
    echo  Then double-click START-CURIO.bat again.
    echo.
    pause
    exit /b 1
  )
  !PYTHON_CMD! -m venv .curio-venv
  if errorlevel 1 goto setup_failed
  echo.
  echo  Installing CURIO components. This needs an internet connection once.
  .curio-venv\Scripts\python.exe -m pip install --upgrade pip
  if errorlevel 1 goto setup_failed
  .curio-venv\Scripts\python.exe -m pip install -r requirements.txt
  if errorlevel 1 goto setup_failed
)

echo.
echo  Starting CURIO. Keep this window open while you use it.
echo  Close this window to stop CURIO.
echo.
.curio-venv\Scripts\python.exe scripts\run_curio.py --no-frontend --browser
if errorlevel 1 (
  echo.
  echo  CURIO stopped with an error. Copy the message above if you need help.
  pause
)
exit /b

:setup_failed
echo.
echo  Setup did not finish. Check your internet connection and try again.
echo  If the problem continues, copy the message above for support.
pause
exit /b 1
