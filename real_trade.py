import os
print("Starting script...")
import time
import math
import argparse
import datetime
import requests
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf
from tabulate import tabulate
import matplotlib.pyplot as plt

# Suppress pandas future warnings for cleaner output
warnings.simplefilter(action='ignore', category=FutureWarning)

# ---------------------------
# Configuration
# ---------------------------
CACHE_DIR = "cache_sp500_v5"
UNIVERSE_CACHE_FILE = os.path.join(CACHE_DIR, "sp500_tickers.json")
OHLCV_CACHE_FILE = os.path.join(CACHE_DIR, "sp500_ohlcv_bulk.parquet")
os.makedirs(CACHE_DIR, exist_ok=True)

# ---------------------------
# Data Structures
# ---------------------------
@dataclass
class StockAnalysis:
    ticker: str
    sector: str
    price: float
    rs_rating: float       # 0-99 Relative Strength vs Market
    rsi_14: float
    atr_14: float
    volatility_20: float   # annualized std dev
    
    # Trend Template Components
    sma_50: float
    sma_150: float
    sma_200: float
    high_52wk: float
    low_52wk: float
    
    # Distance Metrics
    bs_52wk_high_pct: float  # How far slightly below 52wk high (e.g. -5%)
    bs_52wk_low_pct: float   # How far above 52wk low (e.g. +30%)
    
    # Signals
    trend_template_met: bool # Minervini Trend Template
    vcp_pattern: bool        # Volatility Contraction Pattern
    pivot_price: float       # Suggested Buy Point (e.g. recent resistance)
    
    composite_score: float

@dataclass
class TradePlan:
    ticker: str
    entry_price: float
    stop_loss: float
    target_1: float
    target_2: float
    shares: int
    risk_amount: float
    note: str

# ---------------------------
# Universe & Data
# ---------------------------
def get_sp500_universe() -> pd.DataFrame:
    """Fetch S&P 500 tickers from Wikipedia with retry logic."""
    print("Fetching S&P 500 universe from Wikipedia...")
    urls = [
        "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
        "https://web.archive.org/web/20230801000000/https://en.wikipedia.org/wiki/List_of_S%26P_500_companies" # Backup
    ]
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }

    for url in urls:
        try:
            r = requests.get(url, headers=headers, timeout=10)
            r.raise_for_status()
            
            from io import StringIO
            df = pd.read_html(StringIO(r.text))[0]
            
            df = df.rename(columns={
                "Symbol": "ticker",
                "Security": "name",
                "GICS Sector": "sector",
                "GICS Sub-Industry": "sub_industry",
            })
            # Clean tickers (BF.B -> BF-B)
            df["ticker"] = df["ticker"].astype(str).str.replace(".", "-", regex=False).str.strip()
            result = df[["ticker", "name", "sector", "sub_industry"]].drop_duplicates()
            if not result.empty:
                return result
        except Exception as e:
            print(f"Failed to fetch from {url}: {e}")
            continue

    print("Error: Could not fetch universe from any source.")
    return pd.DataFrame()

def fetch_bulk_data(tickers: List[str], period: str = "3y") -> pd.DataFrame:
    """
    Fetch history for ALL tickers.
    Downloaded data: '3y' to ensure valid 200 SMA + Trend calc even after some holidays.
    """
    path = OHLCV_CACHE_FILE
    
    # Check cache (12 hours validity)
    if os.path.exists(path):
        mtime = os.path.getmtime(path)
        if (time.time() - mtime) < 12 * 3600:
            print(f"Loading cached bulk data from {path}...")
            return pd.read_parquet(path)

    print(f"Downloading bulk data for {len(tickers)} tickers (period={period})...")
    tickers_str = " ".join(tickers)
    
    # Retry logic
    for attempt in range(3):
        try:
            # group_by='column' -> MultiIndex columns [Open, High...], top level is Price Type
            df = yf.download(
                tickers_str, 
                period=period, 
                interval="1d", 
                group_by='column', 
                progress=True, 
                auto_adjust=True, 
                threads=True,
                timeout=30
            )
            
            if df is not None and not df.empty:
                # Basic validation: check if we have data for at least 50% of tickers
                # df['Close'] should have shape (rows, n_tickers)
                if 'Close' in df.columns and df['Close'].shape[1] > len(tickers) * 0.5:
                    df.to_parquet(path)
                    print("Download successful.")
                    return df
                else:
                    print(f"Warning: Downloaded data seems incomplete (Attempt {attempt+1})")
        except Exception as e:
            print(f"Bulk download failed (Attempt {attempt+1}): {e}")
            time.sleep(2)
            
    return pd.DataFrame()

# ---------------------------
# Indicators
# ---------------------------
# ---------------------------
# Indicators
# ---------------------------
def calculate_rs_rating(closes: pd.DataFrame) -> pd.Series:
    """
    Calculate IBD-style RS Rating (0-99) for each column in 'closes'.
    Weighted ROC: 40% 12m + 20% 9m + 20% 6m + 20% 3m.
    """
    # We assume 'closes' columns are tickers.
    c = closes
    
    # Safe checks for length
    if len(c) < 260:
        return pd.Series(50, index=c.columns) # Default neutral if not enough data
    
    r12m = c.pct_change(252)
    r9m = c.pct_change(189)
    r6m = c.pct_change(126)
    r3m = c.pct_change(63)
    
    # Weighted score
    raw_score = (0.4 * r12m.iloc[-1]) + (0.2 * r9m.iloc[-1]) + (0.2 * r6m.iloc[-1]) + (0.2 * r3m.iloc[-1])
    
    # Rank 0-100
    raw_score = raw_score.dropna()
    if raw_score.empty:
        return pd.Series()
        
    ranks = raw_score.rank(pct=True) * 99
    return ranks

def prepare_dataframe(df_ticker: pd.DataFrame) -> pd.DataFrame:
    """
    Compute technical indicators for the entire DataFrame (Vectorized).
    Returns df with added columns: SMA_50, SMA_150, SMA_200, RSI, ATR, VCP, etc.
    """
    d = df_ticker.copy()
    if len(d) < 260:
        return pd.DataFrame()
    
    # Fill NAs
    d = d.ffill()
    
    close = d["Close"]
    high = d["High"]
    low = d["Low"]
    volume = d["Volume"]
    
    # --- Moving Averages ---
    d["SMA_50"] = close.rolling(50).mean()
    d["SMA_150"] = close.rolling(150).mean()
    d["SMA_200"] = close.rolling(200).mean()
    
    # --- 52 Week High/Low ---
    d["High_52"] = high.rolling(252).max()
    d["Low_52"] = low.rolling(252).min()
    
    # --- RSI ---
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / (loss + 1e-9)
    d["RSI"] = 100 - (100 / (1 + rs))
    
    # --- ATR ---
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    d["ATR"] = tr.rolling(14).mean()
    
    # --- Volatility ---
    d["Vol_20"] = close.pct_change().rolling(20).std() * math.sqrt(252)
    
    # --- Minervini Trend Template (Vectorized) ---
    # 1. Price > 150 SMA and > 200 SMA
    c1 = (close > d["SMA_150"]) & (close > d["SMA_200"])
    # 2. 150 SMA > 200 SMA
    c2 = d["SMA_150"] > d["SMA_200"]
    # 3. 200 SMA trending up (look back 20 days)
    c3 = d["SMA_200"] > d["SMA_200"].shift(21)
    # 4. 50 SMA > 150 SMA and > 200 SMA
    c4 = (d["SMA_50"] > d["SMA_150"]) & (d["SMA_50"] > d["SMA_200"])
    # 5. Price > 50 SMA
    c5 = close > d["SMA_50"]
    # 6. Price >= 30% above 52-week Low
    c6 = close >= (1.3 * d["Low_52"])
    # 7. Price within 25% of 52-week High
    c7 = close >= (0.75 * d["High_52"])
    
    d["Trend_Template"] = c1 & c2 & c3 & c4 & c5 & c6 & c7

    # --- VCP (Simplified Vectorized) ---
    # 1. Tightness (10d range)
    # Minervini likes < 10-15%. Let's tighten to 12% for quality.
    d["Range_10"] = (high.rolling(10).max() - low.rolling(10).min()) / close
    is_tight = d["Range_10"] < 0.12 
    
    # 2. Volume Dry Up (Vol < 50SMA or Low Volatility)
    vol_sma50 = volume.rolling(50).mean()
    vol_dry = (volume < vol_sma50) | (d["Vol_20"] < 0.25)
    
    d["VCP"] = is_tight & vol_dry & d["Trend_Template"]
    d["Pivot"] = high.rolling(20).max() 
    
    return d

class Backtester:
    def __init__(self, df_bulk: pd.DataFrame, initial_balance: float = 100000):
        self.df_bulk = df_bulk
        self.balance = initial_balance
        self.equity = initial_balance
        self.positions: List[Dict] = [] 
        self.trade_history: List[Dict] = []
        self.equity_curve = []
        self.risk_per_trade_pct = 0.01 # 1% Risk per trade
        
        # Pre-calculate SPY regime
        self.spy = None
        if "SPY" in df_bulk["Close"].columns:
            self.spy = df_bulk["Close"]["SPY"].rolling(200).mean()
            
    def run(self, start_date_str: str = "2023-01-01"):
        print(f"Preparing data for backtest (Start: {start_date_str})...")
        tickers = [c for c in self.df_bulk["Close"].columns if c != "SPY"]
        data_map = {}
        
        # Optimize: Only calc for tickers that have data?
        print(f"Calculating indicators for {len(tickers)} tickers...")
        count = 0
        for t in tickers:
            try:
                # Extract Single Ticker Series for speed
                # Check if we have columns
                if t not in self.df_bulk["Close"].columns: continue
                
                # Create DataFrame manually to avoid MultiIndex slicing issues if any
                sub = pd.DataFrame({
                    "Open": self.df_bulk["Open"][t],
                    "High": self.df_bulk["High"][t],
                    "Low": self.df_bulk["Low"][t],
                    "Close": self.df_bulk["Close"][t],
                    "Volume": self.df_bulk["Volume"][t]
                }).dropna()
                
                if len(sub) < 50: continue
                
                processed = prepare_dataframe(sub)
                if not processed.empty:
                    data_map[t] = processed
            except KeyError:
                pass
            count += 1
            if count % 100 == 0: print(f"  {count}...", end="\r")
            
        print("\nStarting Simulation Loop...")
        
        # Create master timeline
        if "SPY" in self.df_bulk["Close"].columns:
            timeline = self.df_bulk["Close"]["SPY"].index
        else:
            timeline = self.df_bulk.index
            
        timeline = [d for d in timeline if d >= pd.Timestamp(start_date_str)]
        
        for date in timeline:
            self._process_day(date, data_map)
            
        self._generate_report()

    def _process_day(self, date, data_map):
        current_equity = self.balance
        
        # SPY Regime
        regime_bullish = True
        if self.spy is not None:
             try:
                spy_sma = self.spy.loc[date]
                spy_price = self.df_bulk["Close"]["SPY"].loc[date]
                if spy_price < spy_sma:
                    regime_bullish = False
             except:
                 pass
        
        active_positions = []
        for pos in self.positions:
            ticker = pos['ticker']
            if ticker not in data_map or date not in data_map[ticker].index:
                active_positions.append(pos)
                current_equity += (pos['last_price'] * pos['shares']) # Estimate
                continue 
                
            row = data_map[ticker].loc[date]
            curr_price = row["Close"]
            pos['last_price'] = curr_price
            
            # --- Exit Logic ---
            # 1. Stop Loss
            if row["Low"] <= pos['stop_loss']:
                exit_price = min(pos['stop_loss'], row['Open']) 
                if row['Open'] < pos['stop_loss']: exit_price = row['Open'] # Gap Down
                
                pnl = (exit_price - pos['entry']) * pos['shares']
                self.balance += (exit_price * pos['shares'])
                self.trade_history.append({
                    "ticker": ticker,
                    "entry_date": pos['entry_date'],
                    "exit_date": date,
                    "entry_price": pos['entry'],
                    "exit_price": exit_price,
                    "pnl": pnl,
                    "return_pct": (exit_price - pos['entry']) / pos['entry'],
                    "reason": "Stop Loss"
                })
                continue 
            
            # 2. Profit Taking (8% Target)
            profit_pct = (curr_price - pos['entry']) / pos['entry']
            
            if profit_pct >= 0.08:
                pnl = (curr_price - pos['entry']) * pos['shares']
                self.balance += (curr_price * pos['shares'])
                self.trade_history.append({
                    "ticker": ticker,
                    "entry_date": pos['entry_date'],
                    "exit_date": date,
                    "entry_price": pos['entry'],
                    "exit_price": curr_price,
                    "pnl": pnl,
                    "return_pct": profit_pct,
                    "reason": "Target 8%"
                })
                continue
            
            # 3. Trailing Stop (Break Even)
            # If we are up 4%, move stop to entry.
            if profit_pct > 0.04 and pos['stop_loss'] < pos['entry']:
                pos['stop_loss'] = pos['entry'] * 1.01 # Secure BE early



            # 3. Trailing Stop (Break Even)
            if profit_pct > 0.05 and pos['stop_loss'] < pos['entry']:
                pos['stop_loss'] = pos['entry'] * 1.01 # Secure BE early
            
            # Mark to Market
            current_equity += (curr_price * pos['shares'])
            active_positions.append(pos)
            
        self.positions = active_positions
        self.equity = current_equity
        self.equity_curve.append({"date": date, "equity": self.equity})
        
        # --- Entry Logic ---
        if not regime_bullish: return
        
        # Max Risk per trade = 1% of Equity
        # Position Size = Risk_Amount / (Entry - Stop)
        # Max Positions = 10
        if len(self.positions) >= 10: return
        
        cash_available = self.balance
        if cash_available < 2000: return 
        
        candidates = []
        for t, df in data_map.items():
            if date not in df.index: continue
            row = df.loc[date]
            
            # STRATEGY: Balanced Growth (ATR Dynamic)
            # 1. Trend: SMA200 Up
            # 2. Trigger: Oversold (RSI < 30)
            
            if row["Close"] > row["SMA_200"]:
                if row["RSI"] < 30:
                     candidates.append((t, row["RSI"], row["Close"], row["ATR"], row["Pivot"]))
                
        # Sort by Lowest RSI
        candidates.sort(key=lambda x: x[1])
        
        risk_equity = self.equity * self.risk_per_trade_pct
        
        for cand in candidates[:5]: 
            if len(self.positions) >= 10: break
            if self.balance < 2000: break
            if any(p['ticker'] == cand[0] for p in self.positions): continue
            
            t, rsi, price, atr, pivot = cand
            
            # Dynamic ATR Stop (2.0 ATR)
            # Adapts to each stock's volatility
            stop_price = price - (2.0 * atr)
            
            # Safety Cap: Max 10% Risk
            if stop_price < price * 0.90:
                stop_price = price * 0.90
            
            shares = int(risk_equity / (price - stop_price))
            
            max_shares = int((self.equity * 0.15) / price)
            shares = min(shares, max_shares) 

            
            self.balance -= (shares * price)
            self.positions.append({
                "ticker": t,
                "entry_date": date,
                "entry": price,
                "shares": shares,
                "stop_loss": stop_price,
                "last_price": price
            })
            
    def _generate_report(self):
        trades = pd.DataFrame(self.trade_history)
        curve = pd.DataFrame(self.equity_curve).set_index("date")
        
        # Calculate Drawdown
        curve["peak"] = curve["equity"].cummax()
        curve["drawdown"] = (curve["equity"] - curve["peak"]) / curve["peak"]
        max_dd = curve["drawdown"].min()
        
        # Sharpe
        curve["returns"] = curve["equity"].pct_change()
        sharpe = 0
        if curve["returns"].std() > 0:
            sharpe = (curve["returns"].mean() / curve["returns"].std()) * np.sqrt(252)
            
        total_ret_pct = ((self.equity - curve['equity'].iloc[0])/curve['equity'].iloc[0])*100
        
        print("\n" + "="*50)
        print(" BACKTEST RESULTS (Minervini VCP Strategy)")
        print("="*50)
        print(f"Final Equity:   ${self.equity:,.2f}")
        print(f"Total Return:   {total_ret_pct:.2f}%")
        print(f"Max Drawdown:   {max_dd*100:.2f}%")
        print(f"Sharpe Ratio:   {sharpe:.2f}")
        
        if not trades.empty:
            win_rate = len(trades[trades['pnl']>0]) / len(trades) * 100
            
            # Calculate Holding Time
            trades['days_held'] = (trades['exit_date'] - trades['entry_date']).dt.days
            avg_days = trades['days_held'].mean()
            
            print(f"Total Trades:   {len(trades)}")
            print(f"Win Rate:       {win_rate:.1f}%")
            print(f"Avg Win:        {trades[trades['pnl']>0]['return_pct'].mean()*100:.2f}%")
            print(f"Avg Loss:       {trades[trades['pnl']<=0]['return_pct'].mean()*100:.2f}%")
            print(f"Avg Hold Time:  {avg_days:.1f} days")
            
            trades.to_csv("backtest_trades.csv")
        else:
            print("No trades executed.")
            
        curve.to_csv("equity_curve.csv")
        # Ensure matplotlib doesn't popup
        try:
             plt.figure(figsize=(10,6))
             plt.plot(curve.index, curve["equity"], label="VCP Strategy")
             plt.plot(curve.index, curve["peak"], "g--", alpha=0.3)
             plt.title(f"Equity Curve (Net: {total_ret_pct:.1f}%)")
             plt.savefig("backtest_chart.png")
        except:
             pass

def calculate_technical_stats(df_ticker: pd.DataFrame) -> Optional[dict]:
    # Wrapper for old compatibility (Scanner mode)
    # We can just call prepare_dataframe and take the last row
    d = prepare_dataframe(df_ticker)
    if d.empty: return None
    
    row = d.iloc[-1]
    
    # Map back to old dict structure for the scanner display
    return {
        "price": row["Close"],
        "rsi": row["RSI"],
        "atr": row["ATR"],
        "vol": row["Vol_20"],
        "sma50": row["SMA_50"],
        "sma150": row["SMA_150"],
        "sma200": row["SMA_200"],
        "high_52": row["High_52"],
        "low_52": row["Low_52"],
        "pct_off_high": (row["Close"] - row["High_52"]) / row["High_52"],
        "pct_off_low": (row["Close"] - row["Low_52"]) / row["Low_52"],
        "uptrend_template": row["Trend_Template"],
        "vcp": row["VCP"],
        "pivot": row["Pivot"]
    }

# ---------------------------
# Core Logic
# ---------------------------
def run_analysis(account_size: float, risk_pct: float, regime_filter: bool = True) -> None:
    # 1. Get Universe
    univ = get_sp500_universe()
    if univ.empty:
        print("Failed to load universe.")
        return
    
    tickers = univ["ticker"].tolist()
    sector_map = dict(zip(univ["ticker"], univ["sector"]))
    
    # 2. Bulk Data
    # Add SPY for regime check
    if "SPY" not in tickers:
        tickers.append("SPY")
        
    df = fetch_bulk_data(tickers)
    
    if df.empty:
        print("No data returned.")
        return

    # 3. Market Regime Check
    if regime_filter:
        try:
            spy_c = df["Close"]["SPY"].dropna()
            spy_sma200 = spy_c.rolling(200).mean().iloc[-1]
            spy_last = spy_c.iloc[-1]
            if spy_last < spy_sma200:
                print(f"\n[!]  MARKET REGIME WARNING: SPY (${spy_last:.2f}) is below 200 SMA (${spy_sma200:.2f})")
                print("    Historically, long-only strategies fail in this environment.")
                proceed = input("    Do you want to proceed anyway? (y/N): ")
                if proceed.lower() != 'y':
                    return
            else:
                print(f"\n[OK] Market Regime: BULLISH (SPY > 200 SMA)")
        except KeyError:
            print("Could not analyze SPY for regime filter. Skipping.")

    # 4. Calculate Indicators for ALL stocks
    print("Calculating Minervini Trend Template & Technicals...")
    
    # 4.1 Relative Strength Rating (Vectorized)
    closes = df["Close"]
    rs_ratings = calculate_rs_rating(closes)
    
    results = []
    
    valid_tickers = [t for t in tickers if t in closes.columns and t != "SPY"]
    
    total_processed = 0
    passed_trend = 0
    
    for t in valid_tickers:
        total_processed += 1
        try:
            # Reconstruct DataFrame for single ticker
            sub = pd.DataFrame({
                "Open": df["Open"][t],
                "High": df["High"][t],
                "Low": df["Low"][t],
                "Close": df["Close"][t],
                "Volume": df["Volume"][t]
            }).dropna()
            
            stats = calculate_technical_stats(sub)
            if not stats:
                continue
            
            # --- Filters ---
            
            # 1. Trend Template (Must be in Uptrend)
            if not stats["uptrend_template"]:
                continue
            passed_trend += 1
            
            # 2. Strategy: Balanced Growth
            # We want RSI < 30 (Good Value)
            if stats["rsi"] > 30: 
                continue
                
            # 3. Liquidity
            if stats["price"] < 5: 
                continue
            if (stats["price"] * sub["Volume"].iloc[-1]) < 1_000_000: 
                continue
            
            rating = rs_ratings.get(t, 0)
            
            # --- Scoring ---
            # Prioritize: Lowest RSI (Deepest Discount) + High RS Rating
            # Higher Score = Better
            rsi_score = 100 - stats["rsi"]
            rs_score = rating
            final_score = (0.7 * rsi_score) + (0.3 * rs_score)
            
            results.append(StockAnalysis(
                ticker=t,
                sector=sector_map.get(t, "Unknown"),
                price=stats["price"],
                rs_rating=rating,
                rsi_14=stats["rsi"],
                atr_14=stats["atr"],
                volatility_20=stats["vol"],
                sma_50=stats["sma50"],
                sma_150=stats["sma150"],
                sma_200=stats["sma200"],
                high_52wk=stats["high_52"],
                low_52wk=stats["low_52"],
                bs_52wk_high_pct=stats["pct_off_high"],
                bs_52wk_low_pct=stats["pct_off_low"],
                trend_template_met=stats["uptrend_template"],
                vcp_pattern=stats["vcp"],
                pivot_price=stats["pivot"],
                composite_score=final_score
            ))
            
        except KeyError:
            continue
            
    # 5. Rank and Display
    results.sort(key=lambda x: x.composite_score, reverse=True)
    
    top_picks = results[:50]
    
    print(f"\nStats: Processed {total_processed} tickers -> {passed_trend} met Trend Template -> {len(results)} potential Value Buys.")
    print("\n" + "="*95)
    print(f" TOP RECOMMENDATIONS (Balanced Growth)")
    print(f" Strategy: RSI < 30. Target 8%. Stop 2ATR. Move BE @ +4%.")
    print("="*95)
    
    display_rows = []
    for i, r in enumerate(top_picks[:20], 1):
        display_rows.append([
            i,
            r.ticker,
            r.sector[:15],
            f"${r.price:.2f}",
            int(r.rs_rating),
            f"{r.rsi_14:.1f}", # Show RSI
            f"{r.pivot_price:.2f}",
            f"{r.bs_52wk_high_pct*100:.1f}%",
            f"{r.composite_score:.1f}"
        ])
        
    print(tabulate(display_rows, 
                   headers=["#", "Ticker", "Sector", "Price", "RS", "RSI", "Pivot", "Off High", "Score"], 
                   tablefmt="github"))
    
    # 6. Interactive Selection
    selection = input("\nEnter tickers to trade (comma sep) or numbers (e.g. 1, 3, NVDA): ")
    if not selection.strip():
        return
        
    parts = [x.strip().upper() for x in selection.split(",") if x.strip()]
    chosen_objs = []
    
    for p in parts:
        if p.isdigit():
            idx = int(p)
            if 1 <= idx <= len(top_picks):
                chosen_objs.append(top_picks[idx-1])
        else:
            found = next((x for x in results if x.ticker == p), None)
            if found:
                chosen_objs.append(found)
            else:
                print(f"Ticker {p} not found in analyzed results.")
    
    # 7. Position Sizing
    plans = []
    print("\nGenerating Trade Plans...")
    
    for obj in chosen_objs:
        entry = obj.price
        atr = obj.atr_14
        
        # Stop Loss:
        # Dynamic ATR Stop (2.0 ATR) for Smart Risk Management
        stop_price = entry - (2.0 * atr)
            
        # Safety Cap: Max 10% Risk
        if stop_price < entry * 0.90:
             stop_price = entry * 0.90
        
        # Risk Calculation
        risk_per_share = entry - stop_price
        
        risk_amt = account_size * (risk_pct / 100.0)
        shares = int(risk_amt // risk_per_share)
        
        # Max Position Size Cap (15% of equity)
        max_shares = int((account_size * 0.15) / entry)
        shares = min(shares, max_shares)
        
        if shares < 1: shares = 1
            
        # Targets
        # 8% Swing Target
        tp1 = entry * 1.08 
        tp2 = entry * 1.15 
        
        plans.append(TradePlan(
            ticker=obj.ticker,
            entry_price=entry,
            stop_loss=stop_price,
            target_1=tp1,
            target_2=tp2,
            shares=shares,
            risk_amount=risk_amt,
            note="Smart Trade: Move Stop to BE if +4%."
        ))

    # 8. Export & Print
    if not plans:
        return
        
    plan_rows = []
    orders_data = [] 
    
    for p in plans:
        plan_rows.append([
            p.ticker,
            p.shares,
            f"${p.entry_price:.2f}",
            f"${p.stop_loss:.2f} ({(p.entry_price-p.stop_loss)/p.entry_price*100:.1f}%)",
            f"${p.target_1:.2f}",
            f"${p.risk_amount:.0f}"
        ])
        
        orders_data.append({
            "ticker": p.ticker,
            "shares": p.shares,
            "entry_price": round(p.entry_price, 2),
            "stop_loss_price": round(p.stop_loss, 2),
            "take_profit_1": round(p.target_1, 2),
            "take_profit_2": round(p.target_2, 2),
            "note": p.note
        })
        
    print("\n" + "="*60)
    print(" EXECUTION PLAN (Size for 1% Risk)")
    print("="*60)
    print(tabulate(plan_rows, headers=["Ticker", "Qty", "Entry", "Stop Loss", "Target 1", "Risk $"], tablefmt="presto"))
    
    pd.DataFrame(orders_data).to_csv("orders.csv", index=False, encoding='utf-8-sig')
    print("\n[OK] orders.csv saved.")


def main():
    parser = argparse.ArgumentParser(description="Professional S&P 500 Scanner (Minervini Style)")
    parser.add_argument("--account", type=float, default=30000.0, help="Account Equity")
    parser.add_argument("--risk", type=float, default=1.0, help="Risk % per trade")
    parser.add_argument("--no_regime", action="store_true", help="Ignore SPY 200SMA filter")
    parser.add_argument("--backtest", action="store_true", help="Run Backtest Simulation")
    
    args = parser.parse_args()
    
    if args.backtest:
        print("Initializing Backtest Engine...")
        # Load data
        univ = get_sp500_universe()
        tickers = univ["ticker"].tolist()
        if "SPY" not in tickers: tickers.append("SPY")
        
        df = fetch_bulk_data(tickers, period="2y") # 2y for backtest
        if df.empty:
            print("No data for backtest.")
            return
            
        bt = Backtester(df, initial_balance=args.account)
        bt.run(start_date_str="2024-01-01") # Run for last year
        return

    print(f"Starting Engine (Account: ${args.account:,.0f}, Risk: {args.risk}%)...")
    try:
        run_analysis(args.account, args.risk, regime_filter=not args.no_regime)
    except KeyboardInterrupt:
        print("\nAborted.")

if __name__ == "__main__":
    main()
