@echo off
setlocal
title Titan Trade - Pro Swing Scan

cd /d "%~dp0"

echo.
echo TITAN TRADE - PRO SWING SCAN
echo ============================
echo Manual mode only. No broker orders are sent.
echo.

python titan_trade_v3.py --pro

echo.
echo ============================
echo Scan complete. Press any key to close...
pause > nul
