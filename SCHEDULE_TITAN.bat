@echo off
title Titan Trade - Schedule Setup
color 0E

echo.
echo  TITAN TRADE - AUTOMATIC SCHEDULING
echo  ===================================
echo.
echo  This will create a Windows Task Scheduler task to run Titan Trade
echo  automatically every day at 9:35 AM and 3:55 PM (market hours).
echo.
echo  Press any key to continue or close this window to cancel...
pause > nul

echo.
echo  Creating scheduled task for 9:35 AM (after market open)...
schtasks /create /tn "TitanTrade_MarketOpen" /tr "cmd /c cd /d \"%~dp0\" && python titan_trade.py" /sc daily /st 09:35 /f

echo.
echo  Creating scheduled task for 3:55 PM (before market close)...
schtasks /create /tn "TitanTrade_MarketClose" /tr "cmd /c cd /d \"%~dp0\" && python titan_trade.py" /sc daily /st 15:55 /f

echo.
echo  ===================================
echo  DONE! Titan Trade will now run automatically at:
echo    - 9:35 AM (after market opens)
echo    - 3:55 PM (before market closes)
echo.
echo  To remove scheduled tasks, run:
echo    schtasks /delete /tn "TitanTrade_MarketOpen" /f
echo    schtasks /delete /tn "TitanTrade_MarketClose" /f
echo  ===================================
echo.
pause
