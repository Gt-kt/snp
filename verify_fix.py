
import os
import shutil
from titan_trade import TitanBrain

# Clean up cache to force re-fetch
CACHE_DIR = "cache_sp500_elite"
SP500_CACHE_FILE = os.path.join(CACHE_DIR, "sp500_tickers.json")

print("[*] Cleaning up cache...")
if os.path.exists(SP500_CACHE_FILE):
    os.remove(SP500_CACHE_FILE)
    print("    - Deleted sp500_tickers.json")
else:
    print("    - Cache file not found (clean)")

print("\n[*] Initializing TitanBrain...")
brain = TitanBrain()

print("[*] Calling get_data() to fetch tickers...")
try:
    tickers, data = brain.get_data()
    print(f"\n[+] Success! Fetched {len(tickers)} tickers.")
    
    if len(tickers) > 400:
        print("[+] SUCCESS: Ticker count is within expected S&P 500 range.")
    else:
        print(f"[-] WARNING: Ticker count {len(tickers)} is low. Check logic.")
        
except Exception as e:
    print(f"[-] FAILED: get_data raised exception: {e}")
