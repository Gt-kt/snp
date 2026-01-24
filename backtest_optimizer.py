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
# Configuration
# ---------------------------------------------------------
INITIAL_CAPITAL = 100000
MAX_POSITIONS = 8
RISK_PER_TRADE = 0.01 # 1% Risk
MAX_ALLOCATION = 0.20 # Max 20%
MIN_RS_SCORE = 0 # Just outperforming

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
        spy_df['SMA200'] = spy_df['Close'].rolling(200).mean()
        spy_data = spy_df
        
    for t in tickers:
        if t == "SPY": continue
        try:
            cols = df.xs(t, axis=1, level=1).copy().dropna()
            if len(cols) < 260: continue
            
            # Trend
            cols['SMA50'] = cols['Close'].rolling(50).mean()
            cols['SMA150'] = cols['Close'].rolling(150).mean()
            cols['SMA200'] = cols['Close'].rolling(200).mean()
            cols['Low52'] = cols['Close'].rolling(252).min()
            cols['High52'] = cols['Close'].rolling(252).max()
            
            # RS Rating (6m momentum)
            cols['RS_Score'] = cols['Close'].pct_change(126) * 100
            
            # Trend Check (Vectorized)
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
            
    return stock_data, spy_data

def detect_vcp(data, idx):
    """Check VCP at specific index."""
    if idx < 60: return False, 0.0
    
    # Check Price action
    closes = data['Close'].iloc[idx-60:idx+1].values
    highs = data['High'].iloc[idx-60:idx+1].values
    lows = data['Low'].iloc[idx-60:idx+1].values
    
    h3 = highs[-20:].max()
    l3 = lows[-20:].min()
    current_close = closes[-1]
    
    range3 = (h3 - l3) / h3
    near_pivot = (h3 - current_close) / h3 < 0.06
    
    # Relaxed
    if range3 < 0.15 and near_pivot: 
        return True, h3 + 0.05 
        
    return False, 0.0

def run_simulation(stock_data, spy_data):
    print("Starting Portfolio Simulation (Walking Forward)...")
    port = Portfolio(INITIAL_CAPITAL)
    
    sample_t = list(stock_data.keys())[0]
    all_dates = stock_data[sample_t].index
    test_dates = all_dates[-252:] 
    
    for i, date in enumerate(test_dates):
        if i % 20 == 0:
            print(f"Date: {date.date()} | Equity: ${port.total_equity:.0f} | Cash: ${port.cash:.0f} | Pos: {len(port.positions)}")
            
        # 1. Update Market Regime
        market_healthy = True
        if spy_data is not None and date in spy_data.index:
            spy_row = spy_data.loc[date]
            if spy_row['Close'] < spy_row['SMA200']:
                market_healthy = False
        
        # 2. Manage Portfolio
        for ticker in list(port.positions.keys()):
            pos = port.positions[ticker]
            
            if ticker not in stock_data or date not in stock_data[ticker].index:
                continue
            
            daily_row = stock_data[ticker].loc[date]
            
            pos.current_price = daily_row['Close']
            pos.highest_price = max(pos.highest_price, daily_row['High'])
            
            # A. Check Stop Loss
            if daily_row['Low'] < pos.stop_loss:
                exit_price = pos.stop_loss
                if daily_row['Open'] < pos.stop_loss: exit_price = daily_row['Open']
                     
                pnl = (exit_price - pos.entry_price) * pos.shares
                pnl_pct = (exit_price - pos.entry_price) / pos.entry_price
                
                port.cash += (exit_price * pos.shares)
                port.history.append(TradeLog(ticker, pos.entry_date, date, pos.entry_price, exit_price, pnl, pnl_pct, "Stop Loss"))
                del port.positions[ticker]
                continue
                
            # B. Take Profit (Sell 50% at 20%)
            if not pos.half_sold and daily_row['High'] > (pos.entry_price * 1.20):
                exit_price = pos.entry_price * 1.20
                shares_to_sell = pos.shares // 2
                
                pnl = (exit_price - pos.entry_price) * shares_to_sell
                pnl_pct = 0.20
                
                port.cash += (exit_price * shares_to_sell)
                pos.shares -= shares_to_sell
                pos.half_sold = True
                
                # Move Stop to Breakeven on remainder
                pos.stop_loss = max(pos.stop_loss, pos.entry_price)
                
                port.history.append(TradeLog(ticker, pos.entry_date, date, pos.entry_price, exit_price, pnl, pnl_pct, "Target 1 (50%)"))
            
            # C. Trailing Stop (Only after 20% gain)
            gain_from_entry = (pos.highest_price - pos.entry_price) / pos.entry_price
            if gain_from_entry > 0.20:
                # Trail by 10%
                new_stop = pos.highest_price * 0.90
                pos.stop_loss = max(pos.stop_loss, new_stop)
                
        # 3. Scan for New Buys
        if market_healthy and len(port.positions) < MAX_POSITIONS:
            candidates = []
            
            for t, data in stock_data.items():
                if date not in data.index: continue
                
                curr_idx = data.index.get_loc(date)
                if curr_idx < 1: continue
                prev_idx = curr_idx - 1
                
                # Filter: Trend, RS > 0
                if not data['InTrend'].iloc[prev_idx]: continue
                if data['RS_Score'].iloc[prev_idx] < MIN_RS_SCORE: continue
                
                is_setup, pivot = detect_vcp(data, prev_idx)
                
                if is_setup:
                    if data['High'].iloc[curr_idx] > pivot:
                        if data['Open'].iloc[curr_idx] < pivot * 1.03: 
                            rs = data['RS_Score'].iloc[prev_idx]
                            candidates.append({'ticker': t, 'pivot': pivot, 'rs': rs, 'open': data['Open'].iloc[curr_idx]})
            
            # 4. Rank & Buy
            if candidates:
                candidates.sort(key=lambda x: x['rs'], reverse=True)
                
                slots_open = MAX_POSITIONS - len(port.positions)
                for cand in candidates[:slots_open]:
                    t = cand['ticker']
                    pivot = cand['pivot']
                    open_price = cand['open']
                    
                    entry_price = max(pivot, open_price)
                    stop = entry_price * 0.93 # 7% Stop
                    
                    # Risk 1% Equity
                    risk_amt = port.total_equity * RISK_PER_TRADE
                    risk_share = entry_price - stop
                    qty = int(risk_amt / risk_share)
                    
                    # Max 20% Alloc
                    max_cost = port.total_equity * MAX_ALLOCATION
                    if (qty * entry_price) > max_cost:
                        qty = int(max_cost / entry_price)
                        
                    cost = qty * entry_price
                    
                    if port.cash >= cost and qty > 0:
                        port.cash -= cost
                        # Correct Init
                        port.positions[t] = Position(t, date, qty, entry_price, stop, entry_price*1.20, entry_price, entry_price)
                        port.history.append(TradeLog(t, date, date, 0, 0, 0, 0, "ENTRY"))
                        
        port.equity_curve.append({'date': date, 'equity': port.total_equity})
        
    return port

def analyze_portfolio(port: Portfolio):
    history = [t for t in port.history if t.reason != "ENTRY"]
    
    final_equity = port.total_equity
    return_pct = ((final_equity - port.initial_cash) / port.initial_cash) * 100
    
    print("\n" + "="*60)
    print(" BACKTEST RESULTS (OPTIMIZED)")
    print("="*60)
    print(f"Final Equity:      ${final_equity:,.2f} ({return_pct:+.1f}%)")
    
    if not history: return
        
    df = pd.DataFrame([t.__dict__ for t in history])
    df.to_csv("optimizer_trades.csv", index=False)
    
    winners = df[df['pnl'] > 0]
    losers = df[df['pnl'] <= 0]
    
    win_rate = len(winners) / len(history) * 100
    profit_factor = winners['pnl'].sum() / abs(losers['pnl'].sum()) if len(losers)>0 else 999
    
    print(f"Trades:            {len(history)}")
    print(f"Win Rate:          {win_rate:.1f}%")
    print(f"Profit Factor:     {profit_factor:.2f}")
    print(f"Avg Loss:          {losers['pnl_pct'].mean()*100:.2f}%")
    print(f"Avg Win:           {winners['pnl_pct'].mean()*100:.2f}%") 
    print("-" * 60)
    print(df.sort_values('pnl', ascending=False)[['ticker', 'pnl_pct', 'reason']].head(5))

def main():
    print("--- ELITE PORTFOLIO OPTIMIZER ---")
    df = load_data()
    if df is None: return
    stock_data, spy_data = calculate_indicators(df)
    port = run_simulation(stock_data, spy_data)
    analyze_portfolio(port)

if __name__ == "__main__":
    main()
