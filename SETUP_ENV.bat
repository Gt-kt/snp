@echo off
setlocal
title Titan Trade - Environment Setup

cd /d "%~dp0"

echo.
echo TITAN TRADE - CLEAN PYTHON ENVIRONMENT
echo =====================================
echo Creates/updates .venv and installs pinned compatible dependencies.
echo.

if not exist ".venv\Scripts\python.exe" (
  python -m venv .venv
)

call ".venv\Scripts\activate.bat"
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt -c constraints.txt
python -m pip install -e ".[dev]" -c constraints.txt
python -m pip check

echo.
echo Environment ready. Press any key to close...
pause > nul
