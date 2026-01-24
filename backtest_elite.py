import os
import time
import pandas as pd
import numpy as np
import yfinance as yf
from dataclasses import dataclass
from typing import List, Optional

# Disable warnings
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
pd.options.mode.chained_assignment = None

CACHE_DIR = "cache_sp500_elite"
OHLCV_CACHE_FILE = os.path.join(CACHE_DIR, "sp500_ohlcv_bulk.parquet")

@dataclass
class TradeRecord:
    ticker: str
    entry_date: pd.Timestamp
    entry_price: float
    exit_date: Optional[pd.Timestamp]
    exit_price: float
    reason: str # "Target", "Stop"
    pnl_pct: float
    r_multiple: float

def load_data():
    if not os.path.exists(OHLCV_CACHE_FILE):
        print("Data cache not found. Please run elite_trade.py first to download data.")
        return None
    print(f"Loading data from {OHLCV_CACHE_FILE}...")
    return pd.read_parquet(OHLCV_CACHE_FILE)

def check_trend_template(row, sma50, sma150, sma200, low52, high52):
    """Vectorized Trend Template Check (Single Row or Series)."""
    close = row
    
    # 1. Price > 50 > 150 > 200
    c1 = (close > sma50) & (sma50 > sma150) & (sma150 > sma200)
    
    # 2. Price > 1.25 * 52w Low
    c2 = close > (low52 * 1.25)
    
    # 3. Price > 0.75 * 52w High
    c3 = close > (high52 * 0.75)
    
    return c1 & c2 & c3

def detect_vcp_hist(close_series, high_series, low_series):
    """
    Historical VCP Check on a specific window of data.
    Looking at the last 60 days from the 'current' point.
    """
    try:
        # We need at least 60 days
        if len(close_series) < 60: return False, 0.0
        
        # Slices relative to the end of the window
        # Window: last 60 days
        h3 = high_series[-20:].max()       # Recent handle (20d)
        l3 = low_series[-20:].min()
        
        h2 = high_series[-40:-20].max()    # Mid wave (20-40d ago)
        l2 = low_series[-40:-20].min()
        
        h1 = high_series[-60:-40].max()    # Old wave (40-60d ago)
        l1 = low_series[-60:-40].min()
        
        # Calculate Ranges
        r3 = (h3 - l3) / h3
        r2 = (h2 - l2) / h2
        r1 = (h1 - l1) / h1
        
        # Logic: Contraction
        # Ideally r1 > r2 > r3
        is_contracting = (r2 < r1) and (r3 < r2)
        is_tight = r3 < 0.10 # Handle must be tight (<10%)
        
        curr_close = close_series[-1]
        near_pivot = (h3 - curr_close) / h3 < 0.05
        
        if is_contracting and is_tight and near_pivot:
            return True, h3 + 0.05 # Pivot is High of Handle
            
    except:
        return False, 0.0
    
    return False, 0.0

def run_backtest(df):
    print("Preparing data for backtest (Pre-calculating indicators)...")
    
    # Pre-calc indicators for ALL tickers at once
    # This assumes df is MultiIndex columns (Ticker, O/H/L/C/V) or Grouped
    
    # If using yfinance 'group_by=column', columns are (Price, Ticker)
    # We want dict of DataFrames or similar
    
    tickers = df.columns.levels[1].tolist()
    
    # Dictionary of "Clean" Dataframes per ticker
    stock_data = {}
    
    for t in tickers:
        try:
            # Extract checks
            cols = df.xs(t, axis=1, level=1).copy()
            if cols['Close'].isna().all(): continue
            
            # Drop NaNs
            cols = cols.dropna()
            if len(cols) < 300: continue
            
            # Calc Indicators
            cols['SMA50'] = cols['Close'].rolling(50).mean()
            cols['SMA150'] = cols['Close'].rolling(150).mean()
            cols['SMA200'] = cols['Close'].rolling(200).mean()
            cols['Low52'] = cols['Close'].rolling(252).min()
            cols['High52'] = cols['Close'].rolling(252).max()
            
            # Trend Template Boolean
            cols['InTrend'] = (
                (cols['Close'] > cols['SMA50']) & 
                (cols['SMA50'] > cols['SMA150']) & 
                (cols['SMA150'] > cols['SMA200']) &
                (cols['Close'] > cols['Low52'] * 1.25) &
                (cols['Close'] > cols['High52'] * 0.75)
            )
            
            stock_data[t] = cols
        except:
            continue
            
    print(f"Data valid for {len(stock_data)} tickers.")
    print("Starting simulation (Walking forward 1 year)...")
    
    trades: List[TradeRecord] = []
    
    # We will simulate the last 250 days (approx 1 year)
    # Finding common index
    sample_ticker = list(stock_data.keys())[0]
    dates = stock_data[sample_ticker].index
    test_dates = dates[-252:] 
    
    # Iterate Days
    total_trends = 0
    total_setups = 0
    
    for i, date in enumerate(test_dates):
        if i % 20 == 0: 
            print(f"Simulating date: {date.date()}... (Trend: {total_trends}, Setups: {total_setups})")
        
        for t, data in stock_data.items():
            if date not in data.index: continue
            
            # Look at "Yesterday" to scan (we scan after close to buy next day)
            curr_idx = data.index.get_loc(date)
            if curr_idx < 100: continue
            
            yesterday_row = data.iloc[curr_idx]
            
            # 1. Must be in Trend Template (Yesterday)
            if not yesterday_row['InTrend']: continue
            total_trends += 1
            
            # 2. Check VCP (Expensive check, only run if Trend is OK)
            # Pass window ending yesterday
            window_closes = data['Close'].iloc[curr_idx-80:curr_idx+1].values
            window_highs = data['High'].iloc[curr_idx-80:curr_idx+1].values
            window_lows = data['Low'].iloc[curr_idx-80:curr_idx+1].values
            
            is_setup, pivot = detect_vcp_hist(window_closes, window_highs, window_lows)
            
            if is_setup:
                total_setups += 1
                # WE HAVE A SETUP AT CLOSE OF 'date' (Today)
                # Place BUY STOP for 'date+1' (Tomorrow)
                
                # Check next day exists
                if curr_idx + 1 >= len(data): continue
                
                tomorrow_row = data.iloc[curr_idx + 1]
                tomorrow_date = data.index[curr_idx + 1]
                
                tom_high = tomorrow_row['High']
                tom_open = tomorrow_row['Open']
                tom_low = tomorrow_row['Low']
                
                # Did we break the Pivot?
                if tom_high > pivot:
                    # TRIGGERED!
                    
                    # Entry Price: Pivot (if intraday break) or Open (if gap up above pivot)
                    entry_price = max(pivot, tom_open)
                    
                    # Stop Loss: 5% trailing or structural
                    stop_loss = entry_price * 0.95
                    
                    # Target: 3R (15% gain approx)
                    target_price = entry_price * 1.15
                    
                    # Fast Forward to find Exit
                    # Look at future data points starting from DAY AFTER Entry?
                    # No, we can exit on the same day if volatility is huge (Day trade), 
                    # but for simplicity let's start checking exit from Tomorrow's Close or Next Day.
                    # Let's simple check: Did we hit Stop or Target INTRA-DAY on Entry Day?
                    
                    exit_price = None
                    exit_reason = None
                    exit_date = None
                    
                    # Check Intraday Exit on Entry Day
                    # Worst case assumption: We hit Stop first if Low < Stop
                    if tom_low < stop_loss:
                         exit_price = stop_loss
                         exit_reason = "Stop (Day 1)"
                         exit_date = tomorrow_date
                    elif tom_high > target_price and (tom_open < target_price): # If we opened below target
                         exit_price = target_price
                         exit_reason = "Target (Day 1)"
                         exit_date = tomorrow_date
                    
                    if exit_price is None:
                        # Scan Future Days
                        future_data = data.iloc[curr_idx+2:]
                        
                        for f_date, f_row in future_data.iterrows():
                            # Check Low for Stop
                            if f_row['Low'] < stop_loss:
                                exit_price = stop_loss # Stopped out
                                exit_reason = "Stop"
                                exit_date = f_date
                                break
                            
                            # Check High for Target
                            if f_row['High'] > target_price:
                                exit_price = target_price # Profit
                                exit_reason = "Target"
                                exit_date = f_date
                                break
                                
                    # If trades ran to end of data
                    if exit_price is None:
                        exit_price = data.iloc[-1]['Close']
                        exit_reason = "Open"
                        exit_date = data.index[-1]
                        
                    # Record Trade
                    pnl = (exit_price - entry_price) / entry_price
                    r_mult = pnl / 0.05 # risking 5%
                    
                    trades.append(TradeRecord(
                        t, tomorrow_date, entry_price, exit_date, exit_price, exit_reason, pnl, r_mult
                    ))
                    
    return trades

def analyze_results(trades: List[TradeRecord]):
    if not trades:
        print("No trades generated.")
        return
        
    df = pd.DataFrame([t.__dict__ for t in trades])
    
    total_trades = len(df)
    winners = df[df['pnl_pct'] > 0]
    losers = df[df['pnl_pct'] <= 0]
    
    win_rate = len(winners) / total_trades * 100
    avg_win = winners['pnl_pct'].mean() * 100
    avg_loss = losers['pnl_pct'].mean() * 100
    
    gross_win = winners['pnl_pct'].sum()
    gross_loss = abs(losers['pnl_pct'].sum())
    profit_factor = gross_win / gross_loss if gross_loss != 0 else 0
    
    print("\n" + "="*60)
    print(" BACKTEST RESULTS (Last 252 Days)")
    print("="*60)
    print(f"Total Trades:      {total_trades}")
    print(f"Win Rate:          {win_rate:.1f}%")
    print(f"Profit Factor:     {profit_factor:.2f}")
    print(f"Avg Win:           {avg_win:.2f}%")
    print(f"Avg Loss:          {avg_loss:.2f}%")
    print(f"Expected Value:    {(avg_win * (win_rate/100)) - (abs(avg_loss) * ((100-win_rate)/100)):.2f}% per trade")
    print("-" * 60)
    print("Sample Trades:")
    print(df[['ticker', 'entry_date', 'reason', 'pnl_pct']].tail(10))

def main():
    print("--- ULTRA-ELITE BACKTESTER ---")
    df = load_data()
    if df is None: return
    
    trades = run_backtest(df)
    analyze_results(trades)

if __name__ == "__main__":
    main()
