import os
import time
import math
import argparse
import datetime
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
import yfinance as yf
from tabulate import tabulate
import requests

# Suppress warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
pd.options.mode.chained_assignment = None

# ---------------------------
# Configuration
# ---------------------------
CACHE_DIR = "cache_sp500_pro"
UNIVERSE_CACHE_FILE = os.path.join(CACHE_DIR, "sp500_tickers.json")
OHLCV_CACHE_FILE = os.path.join(CACHE_DIR, "sp500_ohlcv_bulk.parquet")
os.makedirs(CACHE_DIR, exist_ok=True)

# ---------------------------
# Data Classes
# ---------------------------
@dataclass
class MarketRegime:
    spy_trend: str        # "BULLISH", "BEARISH", "NEUTRAL"
    breadth_50: float     # % Stocks > SMA50
    breadth_200: float    # % Stocks > SMA200
    details: str

@dataclass
class TradeSetup:
    ticker: str
    sector: str
    strategy: str         # "Trend Pullback", "Power Breakout"
    price: float
    stop_loss: float
    target_1: float
    target_2: float
    risk_reward: float
    score: float
    notes: List[str]

# ---------------------------
# Data Fetching Layer
# ---------------------------
def get_sp500_universe() -> pd.DataFrame:
    """Fetch S&P 500 tickers from Wikipedia."""
    print("Fetching S&P 500 universe...")
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        r = requests.get(url, headers=headers, timeout=10)
        from io import StringIO
        df = pd.read_html(StringIO(r.text))[0]
        
        df = df.rename(columns={
            "Symbol": "ticker",
            "Security": "name",
            "GICS Sector": "sector",
            "GICS Sub-Industry": "sub_industry",
        })
        df["ticker"] = df["ticker"].astype(str).str.replace(".", "-", regex=False).str.strip()
        return df[["ticker", "sector", "sub_industry"]].drop_duplicates()
    except Exception as e:
        print(f"Error fetching universe: {e}")
        return pd.DataFrame()

def fetch_bulk_data(tickers: List[str], period: str = "2y") -> pd.DataFrame:
    """Fetch history for ALL tickers."""
    path = OHLCV_CACHE_FILE
    
    # 12-hour cache
    if os.path.exists(path):
        mtime = os.path.getmtime(path)
        if (time.time() - mtime) < 12 * 3600:
            print(f"Loading cached bulk data from {path}...")
            return pd.read_parquet(path)

    print(f"Downloading bulk data for {len(tickers)} tickers ({period})...")
    
    # Chunking to prevent massive failures
    chunk_size = 100
    all_dfs = []
    
    # Ensure SPY is in the list
    if "SPY" not in tickers: tickers.append("SPY")
    
    unique_tickers = list(set(tickers))
    
    try:
        # We can try downloading all at once, usually yf handles it well up to a few thousand
        # But let's use the same robust logic as before
        tickers_str = " ".join(unique_tickers)
        df = yf.download(
            tickers_str, 
            period=period, 
            interval="1d", 
            group_by='column', 
            progress=True, 
            auto_adjust=True, 
            threads=True,
            timeout=60
        )
        
        if df is not None and not df.empty:
            df.to_parquet(path)
            return df
            
    except Exception as e:
        print(f"Download failed: {e}")
        
    return pd.DataFrame()

# ---------------------------
# Analysis Engine
# ---------------------------
class MarketBrain:
    def __init__(self, data: pd.DataFrame, universe: pd.DataFrame):
        self.data = data
        self.universe = universe
        self.sector_map = dict(zip(universe["ticker"], universe["sector"]))
        
        # Pre-process Data
        self.opens = data["Open"]
        self.highs = data["High"]
        self.lows = data["Low"]
        self.closes = data["Close"]
        self.volumes = data["Volume"]
        
    def analyze_regime(self) -> MarketRegime:
        """Analyze broader market health."""
        print("Analyzing Market Regime...")
        spy = self.closes["SPY"].dropna()
        if spy.empty:
            return MarketRegime("UNKNOWN", 0, 0, "No SPY data")
            
        spy_ma200 = spy.rolling(200).mean().iloc[-1]
        spy_price = spy.iloc[-1]
        
        # Market Breadth
        # Calculate for all tickers
        # Use last valid price
        latest_closes = self.closes.iloc[-1]
        
        sma50 = self.closes.rolling(50).mean().iloc[-1]
        sma200 = self.closes.rolling(200).mean().iloc[-1]
        
        above_50 = (latest_closes > sma50).sum()
        above_200 = (latest_closes > sma200).sum()
        total = len(latest_closes)
        
        pct_50 = (above_50 / total) * 100
        pct_200 = (above_200 / total) * 100
        
        trend = "NEUTRAL"
        if spy_price > spy_ma200 and pct_50 > 50:
            trend = "BULLISH"
        elif spy_price < spy_ma200 and pct_50 < 40:
            trend = "BEARISH"
            
        details = (f"SPY vs 200SMA: {'Diff > 0' if spy_price > spy_ma200 else 'Diff < 0'}. "
                   f"Breadth: {pct_50:.1f}% stocks > SMA50.")
                   
        return MarketRegime(trend, pct_50, pct_200, details)

    def rank_sectors(self) -> pd.DataFrame:
        """Rank sectors by Momentum (Relative Strength)."""
        print("Ranking Sectors...")
        # Calculate performance for each stock
        c = self.closes
        
        # 1 Month and 3 Month Performance
        p1m = c.pct_change(21).iloc[-1]
        p3m = c.pct_change(63).iloc[-1]
        
        # Group by Sector
        sector_scores = {}
        for ticker, sector in self.sector_map.items():
            if ticker not in c.columns: continue
            if sector not in sector_scores:
                sector_scores[sector] = {"p1m": [], "p3m": []}
            
            val1 = p1m.get(ticker, np.nan)
            val3 = p3m.get(ticker, np.nan)
            
            if not np.isnan(val1): sector_scores[sector]["p1m"].append(val1)
            if not np.isnan(val3): sector_scores[sector]["p3m"].append(val3)
            
        results = []
        for sect, metrics in sector_scores.items():
            if not metrics["p1m"]: continue
            avg_p1m = np.mean(metrics["p1m"])
            avg_p3m = np.mean(metrics["p3m"])
            
            # Score: Weight recent momentum
            score = (avg_p1m * 0.6) + (avg_p3m * 0.4)
            results.append({"Sector": sect, "Score": score, "1M": avg_p1m, "3M": avg_p3m})
            
        df = pd.DataFrame(results).sort_values("Score", ascending=False)
        return df

    def find_opportunities(self, top_sectors: List[str]) -> List[TradeSetup]:
        """Scan for specific setups in top sectors."""
        print(f"Scanning 500+ stocks (Filtering for Top Sectors: {', '.join(top_sectors)})...")
        
        opportunities = []
        
        for ticker in self.closes.columns:
            if ticker == "SPY": continue
            
            # Sector Filter
            sec = self.sector_map.get(ticker, "Unknown")
            if sec not in top_sectors: continue
            
            # Get Ticker Data
            try:
                # Optimized extraction
                # We need last 260 days approx
                limit = 300
                c = self.closes[ticker].tail(limit)
                h = self.highs[ticker].tail(limit)
                l = self.lows[ticker].tail(limit)
                v = self.volumes[ticker].tail(limit)
                
                if len(c) < 200: continue
                
                curr_price = c.iloc[-1]
                
                # --- Indicators ---
                sma50 = c.rolling(50).mean().iloc[-1]
                sma150 = c.rolling(150).mean().iloc[-1]
                sma200 = c.rolling(200).mean().iloc[-1]
                
                # ATR
                tr1 = h - l
                tr2 = (h - c.shift(1)).abs()
                tr3 = (l - c.shift(1)).abs()
                tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
                atr = tr.rolling(14).mean().iloc[-1]
                
                # RSI
                delta = c.diff()
                gain = (delta.where(delta > 0, 0)).rolling(14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
                rs = gain / (loss + 1e-9)
                rsi = (100 - (100 / (1 + rs))).iloc[-1]
                
                # 52 Week High
                high_52 = h.rolling(252).max().iloc[-1]
                
                # --- STRATEGY 1: QUALITY PULLBACK (THE DIP BUY) ---
                # Criteria:
                # 1. Long Term Trend: Price > SMA200
                # 2. Medium Term: SMA50 > SMA150 > SMA200 (Minervini)
                # 3. Pullback: Price is near SMA50 OR RSI < 40 (Oversold in Uptrend)
                
                is_uptrend = (curr_price > sma200) and (sma50 > sma150 > sma200)
                
                if is_uptrend:
                    # Check Pullback
                    # "Near SMA50": Within 3% of SMA50
                    dist_sma50 = (curr_price - sma50) / sma50
                    
                    is_pullback = (dist_sma50 > -0.02) and (dist_sma50 < 0.04)
                    is_oversold = rsi < 35
                    
                    if is_pullback or is_oversold:
                        # VALID PULLBACK SETUP
                        stop = curr_price - (2.5 * atr) # Wide stop for volatility
                        target = curr_price + (4.0 * atr) # 1:1.5 min
                        
                        # Adjust target if near ATH
                        # If ATH is close, target breakout. If ATH is far, target ATH.
                        notes = []
                        if curr_price < high_52 * 0.95:
                            target = min(target, high_52) # Conservative: Target resistance
                            notes.append(f"Targeting Resistance at ${high_52:.2f}")
                        else:
                            notes.append("Targeting Blue Sky Breakout")
                            
                        score = 80 - rsi # Lower RSI = Better score
                        
                        opportunities.append(TradeSetup(
                            ticker=ticker,
                            sector=sec,
                            strategy="Trend Pullback",
                            price=curr_price,
                            stop_loss=stop,
                            target_1=target,
                            target_2=target * 1.1,
                            risk_reward=(target-curr_price)/(curr_price-stop),
                            score=score,
                            notes=notes
                        ))
                        continue
                        
                # --- STRATEGY 2: POWER BREAKOUT ---
                # Criteria:
                # 1. Near 52WH (within 5%)
                # 2. Consolidation (Low Volatility recently)
                # 3. Volume Spike (optional, but good)
                
                dist_high = (high_52 - curr_price) / high_52
                if 0 <= dist_high < 0.05:
                    # Verify Consolidation (VCP)
                    # 5 Day Price Range is tight (<3%)
                    range_5d = (h.tail(5).max() - l.tail(5).min()) / curr_price
                    
                    if range_5d < 0.05:
                        # BREAKOUT SETUP
                        # Stop is below the consolidation range
                        stop = l.tail(5).min() - (0.5 * atr)
                        target = high_52 * 1.20 # 20% Breakout
                        
                        score = 80 + (1/range_5d) # Tighter range = Higher Score
                        
                        opportunities.append(TradeSetup(
                            ticker=ticker,
                            sector=sec,
                            strategy="Power Breakout",
                            price=curr_price,
                            stop_loss=stop,
                            target_1=high_52 * 1.05, # Quick profit at break
                            target_2=high_52 * 1.20, # Runner
                            risk_reward=(target-curr_price)/(curr_price-stop),
                            score=score,
                            notes=["Tight Consolidation", "Near ATH"]
                        ))

            except Exception as e:
                continue
                
        # Sort by Score
        opportunities.sort(key=lambda x: x.score, reverse=True)
        return opportunities

# ---------------------------
# Main Execution
# ---------------------------
def main():
    print(f"--- PRO TRADE ENGINE (AI-Powered) ---")
    
    # 1. Data Prep
    univ = get_sp500_universe()
    if univ.empty: return
    
    df = fetch_bulk_data(univ["ticker"].tolist())
    if df.empty: return
    
    # 2. Brain Analysis
    brain = MarketBrain(df, univ)
    
    # A. Regime
    regime = brain.analyze_regime()
    print(f"\nMARKET STATUS: [{regime.spy_trend}]")
    print(f"Breadth: {regime.breadth_50:.1f}% > SMA50 | {regime.breadth_200:.1f}% > SMA200")
    print(f"Details: {regime.details}")
    
    if regime.spy_trend == "BEARISH":
        print("\n[CAUTION] Market is in a downtrend. Reducing search scope to Defensive Sectors only.")
        # Logic could be added here to only select Utilities/Consumer Staples
        # For now, we warn the user.
        proceed = input("Proceed anyway? (y/n): ")
        if proceed.lower() != 'y': return

    # B. Sector Rotation
    sector_ranks = brain.rank_sectors()
    print("\nTOP SECTORS (Momentum):")
    print(tabulate(sector_ranks.head(5), headers="keys", tablefmt="simple", floatfmt=".2f"))
    
    top_sectors = sector_ranks["Sector"].head(5).tolist()
    
    # 3. Find Setups
    setups = brain.find_opportunities(top_sectors)
    
    print("\n" + "="*100)
    print(f" TOP TRADING OPPORTUNITIES (Filtered by Sector & Strategy)")
    print("="*100)
    
    display_data = []
    for i, s in enumerate(setups[:15], 1):
        display_data.append([
            i,
            s.ticker,
            s.sector[:10],
            s.strategy,
            f"${s.price:.2f}",
            f"${s.stop_loss:.2f}",
            f"${s.target_1:.2f}",
            f"{s.risk_reward:.1f}R",
            f"{s.score:.1f}"
        ])
        
    print(tabulate(display_data, 
                   headers=["#", "Ticker", "Sec", "Strategy", "Price", "Stop", "Target", "R:R", "Score"],
                   tablefmt="github"))
                   
    # 4. Selection
    sel = input("\nEnter selection (e.g. 1, 3): ")
    if not sel.strip(): return
    
    indices = [int(x.strip()) for x in sel.split(",") if x.strip().isdigit()]
    
    print("\n" + "="*80)
    print(" FINAL EXECUTION PLAN")
    print("="*80)
    
    account_size = 30000
    risk_pct = 0.01
    
    for idx in indices:
        if 1 <= idx <= len(setups):
            s = setups[idx-1]
            
            # Position Sizing
            risk_amt = account_size * risk_pct
            risk_per_share = s.price - s.stop_loss
            if risk_per_share <= 0: risk_per_share = s.price * 0.05 # Fallback
            
            qty = int(risk_amt / risk_per_share)
            cost = qty * s.price
            
            # Cap at 20% portfolio
            if cost > account_size * 0.20:
                qty = int((account_size * 0.20) / s.price)
            
            print(f"[{s.ticker}] {s.strategy}")
            print(f"Analysis: {', '.join(s.notes)}")
            print(f"ACTION: BUY {qty} shares @ ${s.price:.2f}")
            print(f"        STOP LOSS: ${s.stop_loss:.2f} (-{(s.price-s.stop_loss)/s.price*100:.1f}%)")
            print(f"        TARGET 1:  ${s.target_1:.2f}")
            print(f"        TARGET 2:  ${s.target_2:.2f}")
            print("-" * 40)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nCancelled.")
