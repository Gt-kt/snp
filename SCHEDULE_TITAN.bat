@echo off
setlocal
title Titan Trade - Schedule Setup

cd /d "%~dp0"

echo.
echo TITAN TRADE - AUTOMATIC SCHEDULING
echo =================================
echo This creates Windows Task Scheduler jobs for manual pro scans.
echo No broker orders are sent by these tasks.
echo.
echo NOTE: Times below are local PC time.
echo During US daylight saving time, the regular US market opens at 22:30 KST.
echo During US standard time, it opens at 23:30 KST.
echo.
echo Press any key to continue or close this window to cancel...
pause > nul

echo.
echo Creating scheduled task for 22:35 local time...
schtasks /create /tn "TitanTrade_MarketOpen" /tr "cmd /c cd /d \"%~dp0\" && python titan_trade_v3.py --pro" /sc daily /st 22:35 /f

echo.
echo Creating scheduled task for 05:55 local time...
schtasks /create /tn "TitanTrade_MarketClose" /tr "cmd /c cd /d \"%~dp0\" && python titan_trade_v3.py --pro" /sc daily /st 05:55 /f

echo.
echo =================================
echo DONE. Titan Trade will run at:
echo   - 22:35 local time
echo   - 05:55 local time
echo.
echo To remove scheduled tasks, run:
echo   schtasks /delete /tn "TitanTrade_MarketOpen" /f
echo   schtasks /delete /tn "TitanTrade_MarketClose" /f
echo =================================
echo.
pause
