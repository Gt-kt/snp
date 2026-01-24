import os
import time
import pandas as pd
import numpy as np
import yfinance as yf
from dataclasses import dataclass, field
from typing import List, Optional, Dict

# Disable warnings
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
pd.options.mode.chained_assignment = None

CACHE_DIR = "cache_sp500_elite"
OHLCV_CACHE_FILE = os.path.join(CACHE_DIR, "sp500_ohlcv_bulk.parquet")

# ---------------------------------------------------------
# Configuration (Strategy 2.0 "Money Maker")
# ---------------------------------------------------------
INITIAL_CAPITAL = 100000
MAX_POSITIONS = 5 # Concentrated bets (Best 5)
RISK_PER_TRADE = 0.02 # 2% Risk (Confidence High)
MAX_ALLOCATION = 0.25 # Max 25% (4 positions max usually)
MIN_RS_SCORE = 70 # Stronger RS requirement

# Time Stops
MAX_HOLD_DAYS_WITHOUT_PROFIT = 5 # If not green in 5 days, KILL IT.

@dataclass
class Position:
    ticker: str
    entry_date: pd.Timestamp
    shares: int
    entry_price: float
    stop_loss: float
    target_price: float
    current_price: float 
    highest_price: float
    days_held: int = 0
    half_sold: bool = False
    
@dataclass
class TradeLog:
    ticker: str
    entry_date: pd.Timestamp
    exit_date: pd.Timestamp
    entry_price: float
    exit_price: float
    pnl: float
    pnl_pct: float
    reason: str
    days_held: int

class Portfolio:
    def __init__(self, cash):
        self.initial_cash = cash
        self.cash = cash
        self.positions: Dict[str, Position] = {}
        self.history: List[TradeLog] = []
        self.equity_curve = []
        
    @property
    def total_equity(self):
        pos_value = sum(p.current_price * p.shares for p in self.positions.values())
        return self.cash + pos_value

# ---------------------------------------------------------
# Logic Layers
# ---------------------------------------------------------
def load_data():
    if not os.path.exists(OHLCV_CACHE_FILE):
        return None
    print(f"Loading data from {OHLCV_CACHE_FILE}...")
    return pd.read_parquet(OHLCV_CACHE_FILE)

def calculate_indicators(df):
    """Pre-calculate vectorized indicators."""
    print("Calculating indicators...")
    tickers = df.columns.levels[1].tolist()
    stock_data = {}
    
    spy_data = None
    if "SPY" in tickers:
        spy_df = df.xs("SPY", axis=1, level=1).copy().dropna()
        spy_df['SMA50'] = spy_df['Close'].rolling(50).mean()
        spy_df['SMA200'] = spy_df['Close'].rolling(200).mean()
        spy_data = spy_df
        
    for t in tickers:
        if t == "SPY": continue
        try:
            cols = df.xs(t, axis=1, level=1).copy().dropna()
            if len(cols) < 260: continue
            
            # Trend
            cols['SMA20'] = cols['Close'].rolling(20).mean()
            cols['SMA50'] = cols['Close'].rolling(50).mean()
            cols['SMA150'] = cols['Close'].rolling(150).mean()
            cols['SMA200'] = cols['Close'].rolling(200).mean()
            cols['Low52'] = cols['Close'].rolling(252).min()
            cols['High52'] = cols['Close'].rolling(252).max()
            cols['VolSMA50'] = cols['Volume'].rolling(50).mean() # Volume MA
            
            # RS Rating (6m momentum)
            cols['RS_Score'] = cols['Close'].pct_change(126) * 100
            
            # Trend Check (Vectorized) - Minervini Trend Template
            cols['InTrend'] = (
                (cols['Close'] > cols['SMA50']) &
                (cols['SMA50'] > cols['SMA150']) &
                (cols['SMA150'] > cols['SMA200']) &
                (cols['Close'] > cols['Low52'] * 1.30) & # Stricter: 30% off lows
                (cols['Close'] > cols['High52'] * 0.80)   # Stricter: Near Highs
            )
            
            stock_data[t] = cols
        except:
            continue
            
    return stock_data, spy_data

def detect_vcp_strict(data, idx):
    """
    STRICT VCP Check (Strategy 2.0).
    1. Volatility Contraction (Price tightness)
    2. Volume Dry Up (Low volume in tight areas)
    """
    if idx < 60: return False, 0.0
    
    # Check Price action (Last 50 days is enough for the final contraction)
    closes = data['Close'].iloc[idx-50:idx+1].values
    highs = data['High'].iloc[idx-50:idx+1].values
    lows = data['Low'].iloc[idx-50:idx+1].values
    volumes = data['Volume'].iloc[idx-50:idx+1].values
    vol_sma = data['VolSMA50'].iloc[idx]
    
    # 1. Identify the Pivot (Handle) - Last 10-15 days
    h_handle = highs[-15:].max()
    l_handle = lows[-15:].min()
    current_close = closes[-1]
    
    # Depth of handle
    depth = (h_handle - l_handle) / h_handle
    
    # Requirement 1: Tightness (< 15% depth for the handle)
    is_tight = depth < 0.15
    
    # Requirement 2: Near Pivot (< 6% from high)
    near_pivot = (h_handle - current_close) / h_handle < 0.06
    
    # Requirement 3: Volume Dry Up
    # Average volume in the last 5 days should be lower than 50-day avg, 
    # OR create a specific day where Vol is very low.
    recent_vol_avg = np.mean(volumes[-5:])
    vol_dry_up = recent_vol_avg < (vol_sma * 1.0) # Volume is at or below average (Normal/Quiet)
    
    if is_tight and near_pivot and vol_dry_up:
        return True, h_handle + 0.02 # Pivot is High + cents
        
    return False, 0.0

def run_simulation(stock_data, spy_data):
    print("Starting Strategy 2.0 Simulation...")
    port = Portfolio(INITIAL_CAPITAL)
    
    sample_t = list(stock_data.keys())[0]
    all_dates = stock_data[sample_t].index
    test_dates = all_dates[-252:] 
    
    for i, date in enumerate(test_dates):
        if i % 20 == 0:
            print(f"Date: {date.date()} | Equity: ${port.total_equity:.0f} | Pos: {len(port.positions)}")
            
        # 1. Update Market Regime (Strict)
        market_healthy = True
        if spy_data is not None and date in spy_data.index:
            spy_row = spy_data.loc[date]
            # Market Filter: SPY > SMA200 AND SPY_SMA50 > SPY_SMA200
            if (spy_row['Close'] < spy_row['SMA200']) or (spy_row['SMA50'] < spy_row['SMA200']):
                market_healthy = False
        
        # 2. Manage Portfolio
        tickers_to_del = []
        
        for ticker in list(port.positions.keys()):
            pos = port.positions[ticker]
            
            if ticker not in stock_data or date not in stock_data[ticker].index:
                continue
            
            daily_row = stock_data[ticker].loc[date]
            
            pos.current_price = daily_row['Close']
            pos.highest_price = max(pos.highest_price, daily_row['High'])
            pos.days_held += 1
            
            # A. Check Stop Loss
            if daily_row['Low'] < pos.stop_loss:
                exit_price = pos.stop_loss
                if daily_row['Open'] < pos.stop_loss: exit_price = daily_row['Open'] # Gap down
                
                pnl = (exit_price - pos.entry_price) * pos.shares
                pnl_pct = (exit_price - pos.entry_price) / pos.entry_price
                
                port.cash += (exit_price * pos.shares)
                port.history.append(TradeLog(ticker, pos.entry_date, date, pos.entry_price, exit_price, pnl, pnl_pct, "Stop Loss", pos.days_held))
                tickers_to_del.append(ticker)
                continue
                
            # B. Time Stop (Kill Dead Money)
            # If after 5 days, profit is < 1% (stagnant), SELL.
            current_pnl_pct = (pos.current_price - pos.entry_price) / pos.entry_price
            if pos.days_held >= MAX_HOLD_DAYS_WITHOUT_PROFIT and current_pnl_pct < 0.01:
                 exit_price = daily_row['Close']
                 pnl = (exit_price - pos.entry_price) * pos.shares
                 pnl_pct = current_pnl_pct
                 
                 port.cash += (exit_price * pos.shares)
                 port.history.append(TradeLog(ticker, pos.entry_date, date, pos.entry_price, exit_price, pnl, pnl_pct, "Time Stop (Stagnant)", pos.days_held))
                 tickers_to_del.append(ticker)
                 continue

            # C. Take Profit (Sell 33% at 15%)
            if not pos.half_sold and daily_row['High'] > (pos.entry_price * 1.15):
                exit_price = pos.entry_price * 1.15
                shares_to_sell = int(pos.shares * 0.33)
                
                if shares_to_sell > 0:
                    pnl = (exit_price - pos.entry_price) * shares_to_sell
                    pnl_pct = 0.15
                    
                    port.cash += (exit_price * shares_to_sell)
                    pos.shares -= shares_to_sell
                    pos.half_sold = True
                    
                    # Move Stop to Breakeven
                    pos.stop_loss = max(pos.stop_loss, pos.entry_price)
                    port.history.append(TradeLog(ticker, pos.entry_date, date, pos.entry_price, exit_price, pnl, pnl_pct, "Target 1 (33%)", pos.days_held))
            
            # D. Trailing Stop
            # If price moves up considerably, tighten stop
            # Trail by 10% from High
            new_stop = pos.highest_price * 0.90
            if new_stop > pos.stop_loss:
                pos.stop_loss = new_stop
        
        for t in tickers_to_del:
            del port.positions[t]
            
        # 3. Scan for New Buys (Only if Market Healthy)
        if market_healthy and len(port.positions) < MAX_POSITIONS:
            candidates = []
            
            for t, data in stock_data.items():
                if date not in data.index: continue
                
                curr_idx = data.index.get_loc(date)
                if curr_idx < 1: continue
                prev_idx = curr_idx - 1
                
                # Filter: Trend, RS > 70
                if not data['InTrend'].iloc[prev_idx]: continue
                if data['RS_Score'].iloc[prev_idx] < MIN_RS_SCORE: continue
                
                # Check VCP Setup (Yesterday)
                is_setup, pivot = detect_vcp_strict(data, prev_idx)
                
                if is_setup:
                    # CHECK TODAY:
                    # 1. Price > Pivot (Breakout)
                    # 2. Volume > Avg (Confirmation) - We can only check if volume is high SO FAR (difficult in daily bars)
                    # Sim: If High > Pivot AND Total Volume > Avg * 1.2
                    
                    daily_vol = data['Volume'].iloc[curr_idx]
                    avg_vol = data['VolSMA50'].iloc[curr_idx]
                    
                    if data['High'].iloc[curr_idx] > pivot:
                        # Volume Confirmation Check
                        # In real-time we'd wait for volume. Here we check end-of-day volume.
                        if daily_vol > (avg_vol * 1.0): # Volume was at least average
                             # Don't buy if gap up is too huge (> 5%)
                             if data['Open'].iloc[curr_idx] < pivot * 1.05:
                                 rs = data['RS_Score'].iloc[prev_idx]
                                 candidates.append({'ticker': t, 'pivot': pivot, 'rs': rs, 'open': data['Open'].iloc[curr_idx]})
            
            # 4. Rank & Buy
            if candidates:
                # Sort by RS Strength
                candidates.sort(key=lambda x: x['rs'], reverse=True)
                
                slots_open = MAX_POSITIONS - len(port.positions)
                for cand in candidates[:slots_open]:
                    t = cand['ticker']
                    pivot = cand['pivot']
                    open_price = cand['open']
                    
                    entry_price = max(pivot, open_price)
                    stop = entry_price * 0.94 # 6% Stop (Tighter)
                    
                    # Risk 2% Equity
                    risk_amt = port.total_equity * RISK_PER_TRADE
                    risk_share = entry_price - stop
                    if risk_share <= 0: continue
                    
                    qty = int(risk_amt / risk_share)
                    
                    # Max allocation check
                    max_cost = port.total_equity * MAX_ALLOCATION
                    if (qty * entry_price) > max_cost:
                        qty = int(max_cost / entry_price)
                        
                    cost = qty * entry_price
                    
                    if port.cash >= cost and qty > 0:
                        port.cash -= cost
                        port.positions[t] = Position(t, date, qty, entry_price, stop, 0, entry_price, entry_price, 0)
                        port.history.append(TradeLog(t, date, date, 0, 0, 0, 0, "ENTRY", 0))
                        
    return port

def analyze_portfolio(port: Portfolio):
    history = [t for t in port.history if t.reason != "ENTRY"]
    
    final_equity = port.total_equity
    return_pct = ((final_equity - port.initial_cash) / port.initial_cash) * 100
    
    print("\n" + "="*60)
    print(" BACKTEST RESULTS V2.0 (STRATEGY 2.0)")
    print("="*60)
    print(f"Final Equity:      ${final_equity:,.2f} ({return_pct:+.1f}%)")
    
    if not history: 
        print("No trades triggered.")
        return
        
    df = pd.DataFrame([t.__dict__ for t in history])
    
    winners = df[df['pnl'] > 0]
    losers = df[df['pnl'] <= 0]
    
    win_rate = len(winners) / len(history) * 100
    profit_factor = winners['pnl'].sum() / abs(losers['pnl'].sum()) if len(losers)>0 and losers['pnl'].sum() != 0 else 999
    
    print(f"Trades:            {len(history)}")
    print(f"Win Rate:          {win_rate:.1f}%")
    print(f"Profit Factor:     {profit_factor:.2f}")
    if len(losers) > 0:
        print(f"Avg Loss:          {losers['pnl_pct'].mean()*100:.2f}%")
    if len(winners) > 0:
        print(f"Avg Win:           {winners['pnl_pct'].mean()*100:.2f}%") 
        
    print(f"Avg Hold (Days):   {df['days_held'].mean():.1f}")
    print("-" * 60)
    print(df.sort_values('pnl', ascending=False)[['ticker', 'pnl_pct', 'reason', 'days_held']].head(5))

def main():
    print("--- ELITE BACKTESTER V2 (MONEY MAKER) ---")
    df = load_data()
    if df is None: return
    stock_data, spy_data = calculate_indicators(df)
    port = run_simulation(stock_data, spy_data)
    analyze_portfolio(port)

if __name__ == "__main__":
    main()
