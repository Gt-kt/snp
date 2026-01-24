
import pandas as pd
import requests

print("Attempting to fetch S&P 500 list from Wikipedia...")
try:
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"}
    # The original code used header "U":"M" which might be insufficient now or Wikipedia blocks it.
    # But let's first try EXACTLY what the user has.
    
    print("Trying with original headers...")
    df = pd.read_html(requests.get(url, headers={"U":"M"}).text)[0]
    print(f"Success! Found {len(df)} tickers.")
    print(df.head())
except Exception as e:
    print(f"Original method failed: {e}")
    
    print("\nTrying with proper User-Agent...")
    try:
        df = pd.read_html(requests.get(url, headers=headers).text)[0]
        print(f"Success with proper UA! Found {len(df)} tickers.")
    except Exception as e2:
        print(f"Proper UA method failed: {e2}")
