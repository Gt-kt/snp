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

# -----------------------------------------------------------------------------
# OPTIMIZED PARAMETERS (Golden Set)
# Found via Grid Search: 64.5% Return / 55.3% Win Rate / 2.92 PF
# -----------------------------------------------------------------------------
PARAM_DEPTH = 0.15      # VCP Depth < 15%
PARAM_STOP = 0.05       # 5% Stop Loss
PARAM_TARGET = 0.25     # 25% Profit Target
PARAM_VOL_MULT = 1.0    # Volume < 1.0x Avg (Quiet before storm)

@dataclass
class EliteSetup:
    ticker: str
    current_price: float
    pivot_price: float
    stop_loss: float
    target_price: float
    score: float
    rs_rating: float
    vol_dry_up: bool
    eps_growth: any
    rev_growth: any
    roe: any

class EliteBrainV2:
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
        try:
            url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
            headers = {"User-Agent": "Mozilla/5.0"}
            df = pd.read_html(requests.get(url, headers=headers).text)[0]
            tickers = [t.replace('.', '-') for t in df['Symbol'].tolist()]
            pd.Series(tickers).to_json(SP500_CACHE_FILE)
            return tickers
        except Exception as e:
            print(f"Error: {e}. Using fallback list.")
            return ["NVDA", "MSFT", "AAPL", "AMZN", "META", "GOOGL", "TSLA", "AMD", "NFLX", "AVGO"]

    def load_data(self, tickers):
        """Load data from cache or download."""
        if os.path.exists(OHLCV_CACHE_FILE):
            if (time.time() - os.path.getmtime(OHLCV_CACHE_FILE)) < 43200: # 12h
                print("Using cached market data...")
                return pd.read_parquet(OHLCV_CACHE_FILE)
        
        print(f"Downloading data for {len(tickers)} tickers (Bulk)...")
        data = yf.download(tickers, period="2y", interval="1d", group_by='ticker', auto_adjust=True, threads=True)
        data.to_parquet(OHLCV_CACHE_FILE)
        return data

    def calculate_rs_rating(self, close_series, spy_close):
        """Calculate RS Rating vs SPY (0-99)."""
        try:
            stock_ret = close_series.pct_change(126).iloc[-1]
            spy_ret = spy_close.pct_change(126).iloc[-1]
            excess = (stock_ret - spy_ret) * 100
            return max(1, min(99, 50 + excess))
        except:
            return 50

    def check_trend_template(self, df):
        """Minervini Trend Template - Stricter V2."""
        try:
            c = df['Close'].iloc[-1]
            sma50 = df['Close'].rolling(50).mean().iloc[-1]
            sma150 = df['Close'].rolling(150).mean().iloc[-1]
            sma200 = df['Close'].rolling(200).mean().iloc[-1]
            low52 = df['Close'].rolling(252).min().iloc[-1]
            high52 = df['Close'].rolling(252).max().iloc[-1]
            
            # Trend Conditions
            if not (c > sma50 > sma150 > sma200): return False
            if not (c > low52 * 1.30): return False # At least 30% off lows
            if not (c > high52 * 0.80): return False # Near highs
            
            return True
        except:
            return False

    def detect_vcp_strict(self, df):
        """
        Strategy 2.0 VCP Check (Optimized):
        1. Tightness (Range < PARAM_DEPTH)
        2. Pivot Proximity (< 6%)
        3. Volume Dry Up (Vol < Avg * PARAM_VOL_MULT)
        """
        try:
            if len(df) < 60: return False, 0.0, False
            
            closes = df['Close'][-50:].values
            highs = df['High'][-50:].values
            lows = df['Low'][-50:].values
            volumes = df['Volume'][-50:].values
            vol_sma = df['Volume'].rolling(50).mean().iloc[-1]
            
            # Identify Handle (Last 15 days)
            h_handle = highs[-15:].max()
            l_handle = lows[-15:].min()
            curr_c = closes[-1]
            
            # 1. Tightness
            depth = (h_handle - l_handle) / h_handle
            if depth > PARAM_DEPTH: return False, 0.0, False
            
            # 2. Pivot
            near_pivot = (h_handle - curr_c) / h_handle < 0.06
            if not near_pivot: return False, 0.0, False
            
            # 3. Volume Dry Up
            recent_vol = np.mean(volumes[-5:])
            vol_dry_up = recent_vol < (vol_sma * PARAM_VOL_MULT)
            
            if vol_dry_up:
                return True, h_handle + 0.05, True
                
            return False, 0.0, False
            
        except:
            return False, 0.0, False

    def get_fundamentals(self, ticker):
        try:
            t = yf.Ticker(ticker)
            info = t.info
            eps = info.get('earningsGrowth', 0)
            rev = info.get('revenueGrowth', 0)
            roe = info.get('returnOnEquity', 0)
            return (eps or 0)*100, (rev or 0)*100, (roe or 0)*100
        except:
            return 0, 0, 0

    def detect_vcp_loose(self, df):
        """
        Loose VCP Check for Watchlist:
        1. Tightness (Range < 25%)
        2. Pivot Proximity (< 10%)
        """
        try:
            if len(df) < 60: return False, 0.0
            
            closes = df['Close'][-50:].values
            highs = df['High'][-50:].values
            lows = df['Low'][-50:].values
            
            h_handle = highs[-15:].max()
            l_handle = lows[-15:].min()
            curr_c = closes[-1]
            
            # 1. Tightness (Loose)
            depth = (h_handle - l_handle) / h_handle
            if depth > 0.25: return False, 0.0
            
            # 2. Pivot (Loose)
            near_pivot = (h_handle - curr_c) / h_handle < 0.10
            if not near_pivot: return False, 0.0
            
            return True, h_handle + 0.05
            
        except:
            return False, 0.0

    def scan(self):
        tickers = self.get_sp500_tickers()
        data = self.load_data(tickers)
        
        # Detect Column Structure
        is_multi_ticker_level_1 = False
        if isinstance(data.columns, pd.MultiIndex):
             # Check if Level 0 is 'Close' (Price) or Ticker
             if 'Close' in data.columns.levels[0]:
                 is_multi_ticker_level_1 = True
                 print("Detected data structure: (Price, Ticker)")
        
        # Get SPY
        spy_data = None
        if "SPY" in data.columns.levels[1 if is_multi_ticker_level_1 else 0]:
             if is_multi_ticker_level_1:
                 spy_data = data.xs('SPY', axis=1, level=1)['Close']
             else:
                 spy_data = data['SPY']['Close']
        else:
            print("Fetching SPY separately...")
            spy_data = yf.download("SPY", period="2y", auto_adjust=True)['Close']
            
        # Market Check
        spy_curr = float(spy_data.iloc[-1])
        spy_sma200 = float(spy_data.rolling(200).mean().iloc[-1])
        
        print(f"\nMarket Status: SPY ${spy_curr:.2f} vs SMA200 ${spy_sma200:.2f}")
        if spy_curr < spy_sma200:
            print("WARNING: Market is in Downtrend. Risk OFF.\n")
        else:
            print("Market is in Uptrend. Risk ON.\n")
            
        elite_setups = []
        radar_setups = []
        
        print(f"Scanning 500+ stocks... (Criteria: Depth<{int(PARAM_DEPTH*100)}%, Vol<{PARAM_VOL_MULT}x)")
        
        for t in tickers:
            try:
                # Extract Data for Ticker
                if is_multi_ticker_level_1:
                    # Check if ticker exists
                    if t not in data.columns.levels[1]: continue
                    df = data.xs(t, axis=1, level=1).copy()
                else:
                    if t not in data.columns: continue
                    df = data[t].copy()
                
                df = df.dropna()
                if len(df) < 250: continue
                
                # 1. Trend
                if not self.check_trend_template(df): continue
                
                # 2. VCP (Strict)
                is_vcp, pivot, dry_up = self.detect_vcp_strict(df)
                
                # Calculate RS here to usage for both
                rs = self.calculate_rs_rating(df['Close'], spy_data)
                
                if is_vcp:
                    if rs < 65: continue
                    
                    eps, rev, roe = self.get_fundamentals(t)
                    
                    fund_score = 0
                    if eps > 15: fund_score += 1
                    if rev > 15: fund_score += 1
                    
                    score = rs + (fund_score * 10)
                    if dry_up: score += 10
                    
                    curr_price = df['Close'].iloc[-1]
                    stop_loss = curr_price * (1 - PARAM_STOP)
                    target = pivot * (1 + PARAM_TARGET)
                    
                    elite_setups.append(EliteSetup(
                        t, curr_price, pivot, stop_loss, target, score, rs, dry_up, eps, rev, roe
                    ))
                else:
                    # check for RADAR (High RS + Loose Pattern)
                    if rs > 80: # Strong Momentum
                        is_loose, pivot_loose = self.detect_vcp_loose(df)
                        if is_loose:
                             eps, rev, roe = self.get_fundamentals(t)
                             radar_setups.append({
                                 'ticker': t, 'price': df['Close'].iloc[-1], 
                                 'rs': rs, 'eps': eps, 'rev': rev, 'pivot': pivot_loose
                             })

            except:
                continue
                
        return elite_setups, radar_setups

def main():
    print("==================================================")
    print("   ELITE TRADE v2.2 - OPTIMIZED MONEY MAKER")
    print("==================================================")
    
    brain = EliteBrainV2()
    elite_setups, radar_setups = brain.scan()
    
    # 1. Show Elite (Actionable)
    if elite_setups:
        elite_setups.sort(key=lambda x: x.score, reverse=True)
        top_picks = elite_setups[:5]
        
        table = []
        for s in top_picks:
            vol_msg = "YES" if s.vol_dry_up else "NO"
            table.append([
                s.ticker, f"${s.current_price:.2f}", f"${s.pivot_price:.2f}", 
                f"${s.stop_loss:.2f}", f"${s.target_price:.2f}", int(s.rs_rating), 
                 f"{s.eps_growth:.0f}%", f"{s.rev_growth:.0f}%"
            ])
            
        print("\n[$$] MONEY MAKER SIGNALS (Strict Criteria)")
        print(tabulate(table, headers=["Ticker", "Price", "BUY STOP", "Stop Loss (5%)", f"Target ({int(PARAM_TARGET*100)}%)", "RS", "EPS%", "Rev%"], tablefmt="fancy_grid"))
        
        print("\nEXECUTION PLAN:")
        for s in top_picks:
            print(f"[*] {s.ticker}")
            print(f"    ACTION: Wait for BUY STOP @ ${s.pivot_price:.2f}")
            print(f"    EXIT:   Stop @ ${s.stop_loss:.2f} | Target @ ${s.target_price:.2f}")
            print(f"    RULES:  Closed if not profitable in 5 days.")
    else:
        print("\n[$$] MONEY MAKER SIGNALS: None found. (Criteria is strict for your safety)")

    # 2. Show Radar (Watchlist)
    if radar_setups:
        radar_setups.sort(key=lambda x: x['rs'], reverse=True)
        top_radar = radar_setups[:10]
        
        radar_table = []
        for r in top_radar:
            radar_table.append([
                r['ticker'], f"${r['price']:.2f}", int(r['rs']), 
                f"{r['eps']:.0f}%", f"{r['rev']:.0f}%", "High RS + Loose VCP"
            ])
            
        print("\n[O] RADAR WATCHLIST (Setting Up - Keep Eye On)")
        print(tabulate(radar_table, headers=["Ticker", "Price", "RS Rating", "EPS%", "Rev%", "Pattern"], tablefmt="simple"))
    else:
         print("\n[O] RADAR: No high momentum setups found.")
        
if __name__ == "__main__":
    main()
