@echo off
setlocal
title Titan Trade - Dashboard

cd /d "%~dp0"

echo.
echo TITAN TRADE - MANUAL SWING DASHBOARD
echo ===================================
echo Starts the dashboard with auto-scan enabled.
echo No broker orders are sent by this command.
echo.

if exist ".venv\Scripts\activate.bat" (
  call ".venv\Scripts\activate.bat"
)

where titan-dashboard >nul 2>nul
if %errorlevel% equ 0 (
  titan-dashboard --auto-scan
) else (
  python titan_dashboard.py --auto-scan
)

echo.
echo Dashboard stopped. Press any key to close...
pause > nul
