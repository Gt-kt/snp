import yfinance as yf
import pandas as pd
import numpy as np
import requests
import os
import time
from datetime import datetime, timedelta
from tabulate import tabulate
from dataclasses import dataclass

# Disable warnings
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
pd.options.mode.chained_assignment = None

# Constants
CACHE_DIR = "cache_sp500_elite"
SP500_CACHE_FILE = os.path.join(CACHE_DIR, "sp500_tickers.json")
OHLCV_CACHE_FILE = os.path.join(CACHE_DIR, "sp500_ohlcv_bulk.parquet")
DAYS_TO_FETCH = 400 # ~1.5 years for SMA200 + Trend checks

@dataclass
class EliteSetup:
    ticker: str
    current_price: float
    pivot_price: float
    stop_loss: float
    target_price: float
    sector: str
    score: float
    rs_rating: float
    sector_rs: float
    rr_ratio: float
    eps_growth: any
    rev_growth: any
    roe: any

class EliteBrain:
    def __init__(self):
        if not os.path.exists(CACHE_DIR):
            os.makedirs(CACHE_DIR)
            
    def get_sp500_tickers(self):
        """Fetch S&P 500 tickers from Wikipedia or cache."""
        if os.path.exists(SP500_CACHE_FILE):
            mod_time = os.path.getmtime(SP500_CACHE_FILE)
            if (time.time() - mod_time) < 86400 * 7: # 1 week cache
                print("Using cached S&P 500 tickers...")
                return pd.read_json(SP500_CACHE_FILE, typ='series').tolist()
        
        print("Fetching new S&P 500 list...")
        url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        try:
            r = requests.get(url, headers=headers)
            r.raise_for_status()
            tables = pd.read_html(r.text)
            df = tables[0]
            tickers = df['Symbol'].tolist()
            # Clean tickers (BRK.B -> BRK-B)
            tickers = [t.replace('.', '-') for t in tickers]
            
            pd.Series(tickers).to_json(SP500_CACHE_FILE)
            return tickers
        except Exception as e:
            print(f"Error fetching S&P 500 list: {e}")
            print("Using fallback list (Mega Caps)...")
            return ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "AMD", "NFLX", "JPM", "V", "LLY", "AVGO", "COST"]

    def fetch_bulk_data(self, tickers):
        """Download data for all tickers at once and cache it."""
        print(f"Downloading data for {len(tickers)} tickers (Bulk)...")
        data = yf.download(tickers, period="2y", interval="1d", group_by='ticker', auto_adjust=True, threads=True)
        
        # Save to Parquet (Fast I/O)
        data.to_parquet(OHLCV_CACHE_FILE)
        return data

    def load_data(self, tickers):
        """Load data from cache or download if stale."""
        if os.path.exists(OHLCV_CACHE_FILE):
            mod_time = os.path.getmtime(OHLCV_CACHE_FILE)
            # If cache is less than 12 hours old, use it
            if (time.time() - mod_time) < 43200: 
                print("Using cached market data...")
                return pd.read_parquet(OHLCV_CACHE_FILE)
        
        return self.fetch_bulk_data(tickers)

    def calculate_rs_rating(self, close_series, spy_close_series):
        """
        Calculate Relative Strength Rating (0-99) vs SPY.
        Simple logic: 6-month performance comparison.
        """
        try:
            # 6 Month (126 days) return
            stock_ret = close_series.pct_change(126).iloc[-1]
            spy_ret = spy_close_series.pct_change(126).iloc[-1]
            
            # Simple Scoring: If stock > SPY * 2 -> 90+, etc.
            # We want a relative score.
            # Let's simple return the Excess Return % * 100 (capped at 99)
            excess = (stock_ret - spy_ret) * 100
            score = 50 + excess # Baseline 50
            return max(1, min(99, score))
        except:
            return 50

    def calculate_sector_momentum(self, tickers, data):
        """Identify leading sectors."""
        # This requires sector mapping.
        # For simplicity, we will skip sector grouping in this simplified loop
        # and just focus on Stock RS.
        return {}

    def check_trend_template(self, series_close, series_high, series_low):
        """
        Minervini Trend Template (Stage 2 Uptrend).
        Returns True if passed.
        """
        try:
            c = series_close[-1]
            sma50 = series_close.rolling(50).mean().iloc[-1]
            sma150 = series_close.rolling(150).mean().iloc[-1]
            sma200 = series_close.rolling(200).mean().iloc[-1]
            low52 = series_low.rolling(252).min().iloc[-1]
            high52 = series_high.rolling(252).max().iloc[-1]
            
            # 1. Price > 150 > 200
            if not (c > sma150 and sma150 > sma200): return False
            
            # 2. 50 SMA > 150 SMA
            if not (sma50 > sma150): return False
            
            # 3. Price > 50 SMA
            if not (c > sma50): return False
            
            # 4. Price > 25% of 52-week Low
            if not (c > low52 * 1.25): return False
            
            # 5. Price within 25% of 52-week High
            if not (c > high52 * 0.75): return False
            
            # 6. 200 SMA Rising (at least flat/up from 1 month ago)
            sma200_1m = series_close.rolling(200).mean().iloc[-20]
            if sma200 < sma200_1m: return False
            
            return True
        except:
            return False

    def detect_vcp(self, closes, highs, lows):
        """
        Detect VCP (Volatility Contraction Pattern).
        Looking for:
        1. Tightness (Range < 15%)
        2. Pivot (Price near High)
        3. Volume Dry Up (Handled in main loop)
        """
        try:
            # Last 60 Days
            c = closes[-60:]
            h = highs[-60:]
            l = lows[-60:]
            
            if len(c) < 60: return False, 0.0
            
            # Identify most recent handle (20 days)
            h3 = h[-20:].max()
            l3 = l[-20:].min()
            
            # Calculate Depth
            depth = (h3 - l3) / h3
            
            # Contraction Logic: depth should be small
            # RELAXED VCP Logic for "Money Printer" (Catch explosive WDC types)
            is_tight = depth < 0.15 # 15% Max Handle Depth
            
            # Price Near Pivot (High of Handle)
            curr = c.iloc[-1]
            near_pivot = (h3 - curr) / h3 < 0.06 # Within 6% of breakout
            
            if is_tight and near_pivot:
                return True, h3 + 0.05 # Pivot = High + buffer
            
            return False, 0.0
            
        except:
            return False, 0.0

    def get_fundamental_data(self, ticker):
        """Fetch EPS, Rev, ROE."""
        try:
            t = yf.Ticker(ticker)
            info = t.info
            eps_g = info.get('earningsGrowth', 0)
            rev_g = info.get('revenueGrowth', 0)
            roe = info.get('returnOnEquity', 0)
            
            if eps_g is None: eps_g = 0
            if rev_g is None: rev_g = 0
            if roe is None: roe = 0
             
            return round(eps_g*100, 1), round(rev_g*100, 1), round(roe*100, 1)
        except:
            return 0, 0, 0

    def scan_for_elite_setups(self):
        tickers = self.get_sp500_tickers()
        data = self.load_data(tickers)
        
        elite_setups = []
        
        # Get SPY for RS comparison
        if "SPY" in data.columns.levels[0]:
            spy_data = data["SPY"]["Close"]
        else:
            # Fetch SPY if missing
            spy_data = yf.download("SPY", period="2y")['Close']

        print("\nScanning for Stage 2 Trend + Valid VCP (This requires patience)...")
        
        for t in tickers:
            try:
                df = data[t].dropna()
                if len(df) < 250: continue
                
                # 1. Trend Filter (Fast)
                if not self.check_trend_template(df['Close'], df['High'], df['Low']):
                    continue
                
                # 2. VCP Filter (Pattern)
                is_vcp, pivot = self.detect_vcp(df['Close'], df['High'], df['Low'])
                
                if is_vcp:
                    print(f"    Possible candidate: {t} ... Checking Fundamentals...")
                    
                    # 3. Fundamentals (Slow - Only do on candidates)
                    eps, rev, roe = self.get_fundamental_data(t)
                    
                    # FUNDAMENTAL FILTER: Penalize junk, reward growth
                    fund_score = 0
                    if eps > 10: fund_score += 1
                    if rev > 10: fund_score += 1
                    if roe > 15: fund_score += 1
                    
                    # If fundamentals are awful (negative growth), skip unless technicals are perfect
                    if eps < -10 and rev < -5: continue
                    
                    # 4. Metrics
                    curr_price = df['Close'].iloc[-1]
                    stop_loss = curr_price * 0.93 # 7% Hard Stop
                    
                    # Reward to Risk
                    # Target = 3R (3 * risk) + Entry
                    risk = pivot - stop_loss
                    target = pivot + (risk * 3)
                    rr_ratio = (target - pivot) / risk if risk > 0 else 0
                    
                    # RS Rating
                    rs_rating = self.calculate_rs_rating(df['Close'], spy_data)
                    
                    if rs_rating < 60: continue # Must outperform SPY slightly
                    
                    # Total Score
                    # RS (0-100) + Fund (0-3 * 10) + RR (ratio * 10)
                    total_score = rs_rating + (fund_score * 20) + (rr_ratio * 5)
                    
                    elite_setups.append(EliteSetup(
                        t, curr_price, pivot, stop_loss, target, "General",
                        total_score, rs_rating, 0, rr_ratio, eps, rev, roe
                    ))
                    
            except Exception as e:
                continue
                
        return elite_setups

def main():
    print("--- ULTRA-ELITE TRADING ENGINE v5.0 (Money Printer Edition) ---")
    brain = EliteBrain()
    setups = brain.scan_for_elite_setups()
    
    if not setups:
        print("No Elite setups found. The market might be weak.")
        return

    # Sort by Score
    setups.sort(key=lambda x: x.score, reverse=True)
    
    # FILTER TOP 5 (Focus on the very best)
    top_picks = setups[:5]
    
    table_data = []
    for s in top_picks:
        table_data.append([
            s.ticker, 
            f"${s.current_price:.2f}", 
            f"${s.pivot_price:.2f}", 
            f"${s.stop_loss:.2f}", 
            f"${s.target_price:.2f}", 
            f"{s.rr_ratio:.1f}R",
            int(s.score),
            f"{s.eps_growth}%", 
            f"{s.rev_growth}%", 
            f"{s.roe}%"
        ])
        
    print("\n" + tabulate(table_data, headers=["Ticker", "Price", "PIVOT (Buy)", "Stop", "Target", "R/R", "Score", "EPS%", "Rev%", "ROE"], tablefmt="fancy_grid"))
    
    print("\n" + "="*80)
    print(" MONEY PRINTER EXECUTION PLAN")
    print("="*80)
    
    for s in top_picks:
        print(f"[*] {s.ticker}")
        print(f"    ENTRY:       BUY STOP LIMIT @ ${s.pivot_price:.2f}")
        print(f"    STOP LOSS:   ${s.stop_loss:.2f} (Max 7%)")
        print(f"    TARGET:      ${s.target_price:.2f} (Sell 1/2 @ 20% gain, Trail the rest)")
        print(f"    RATIONALE:   RS Rating {int(s.rs_rating)} | Growth: EPS {s.eps_growth}%")
        print("-" * 40)
        
    print(f"\nTotal Candidates Scanned: 500+")
    print(f"Elite Setups Found: {len(setups)}")
    print(f"Showing Top {len(top_picks)}.")

if __name__ == "__main__":
    main()
