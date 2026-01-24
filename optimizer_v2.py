import os
import time
import pandas as pd
import numpy as np
import itertools
from dataclasses import dataclass
from typing import List, Dict

# Disable warnings
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
pd.options.mode.chained_assignment = None

CACHE_DIR = "cache_sp500_elite"
OHLCV_CACHE_FILE = os.path.join(CACHE_DIR, "sp500_ohlcv_bulk.parquet")

# ---------------------------------------------------------
# Configuration Ranges (Grid Search)
# ---------------------------------------------------------
PARAM_GRID = {
    'vcp_depth': [0.12, 0.15, 0.18, 0.20],
    'stop_loss': [0.05, 0.07, 0.08],
    'target_pct': [0.15, 0.20, 0.25], 
    'vol_mult': [0.8, 1.0, 1.2]
}

INITIAL_CAPITAL = 100000
MAX_POSITIONS = 5

@dataclass
class SimulationResult:
    params: dict
    final_equity: float
    total_trades: int
    win_rate: float
    profit_factor: float
    return_pct: float
    score: float # Custom scoring metric

# ---------------------------------------------------------
# Core Logic (Optimized for Speed)
# ---------------------------------------------------------
def load_and_prep_data():
    if not os.path.exists(OHLCV_CACHE_FILE):
        return None, None
    
    print(f"Loading data from {OHLCV_CACHE_FILE}...")
    df = pd.read_parquet(OHLCV_CACHE_FILE)
    
    print("Pre-calculating indicators for ALL assets...")
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
            cols['SMA50'] = cols['Close'].rolling(50).mean()
            cols['SMA150'] = cols['Close'].rolling(150).mean()
            cols['SMA200'] = cols['Close'].rolling(200).mean()
            cols['Low52'] = cols['Close'].rolling(252).min()
            cols['High52'] = cols['Close'].rolling(252).max()
            cols['VolSMA50'] = cols['Volume'].rolling(50).mean()
            
            # RS Rating (6m momentum)
            cols['RS_Score'] = cols['Close'].pct_change(126) * 100
            
            # Trend Check (Vectorized)
            cols['InTrend'] = (
                (cols['Close'] > cols['SMA50']) &
                (cols['SMA50'] > cols['SMA150']) &
                (cols['SMA150'] > cols['SMA200']) &
                (cols['Close'] > cols['Low52'] * 1.30) & 
                (cols['Close'] > cols['High52'] * 0.80)
            )
            
            # Pre-calculate VCP Depth/Pivot for speed
            # We can't fully pre-calc dynamic checks, but we can have data ready
            stock_data[t] = cols
        except:
            continue
            
    return stock_data, spy_data

def run_single_sim(stock_data, spy_data, params):
    """Run one simulation with specific parameters."""
    cash = INITIAL_CAPITAL
    positions = {} # {ticker: {shares, entry, stop, ...}}
    history = []
    
    # Extract params
    p_depth = params['vcp_depth']
    p_stop = params['stop_loss']
    p_target = params['target_pct']
    p_vol = params['vol_mult']
    
    sample_t = list(stock_data.keys())[0]
    all_dates = stock_data[sample_t].index
    test_dates = all_dates[-252:] 
    
    for date in test_dates:
        # 1. Update Portfolio Value
        current_equity = cash + sum([positions[t]['shares'] * stock_data[t].loc[date]['Close'] 
                                     for t in positions if date in stock_data[t].index])
        
        # 2. Market Filter
        market_healthy = True
        if spy_data is not None and date in spy_data.index:
            spy_row = spy_data.loc[date]
            if spy_row['Close'] < spy_row['SMA200']: market_healthy = False
            
        # 3. Manage Positions
        tickers_del = []
        for t, pos in positions.items():
            if date not in stock_data[t].index: continue
            daily = stock_data[t].loc[date]
            
            # Stop Loss
            if daily['Low'] < pos['stop']:
                exit_px = pos['stop']
                if daily['Open'] < pos['stop']: exit_px = daily['Open']
                cash += exit_px * pos['shares']
                pnl = (exit_px - pos['entry']) / pos['entry']
                history.append(pnl)
                tickers_del.append(t)
                continue
                
            # Time Stop (5 days)
            days_held = (date - pos['entry_date']).days
            if days_held > 7 and (daily['Close'] - pos['entry']) / pos['entry'] < 0.01:
                 cash += daily['Close'] * pos['shares']
                 pnl = (daily['Close'] - pos['entry']) / pos['entry']
                 history.append(pnl)
                 tickers_del.append(t)
                 continue
                 
            # Take Profit
            if daily['High'] > pos['target']:
                 exit_px = pos['target']
                 cash += exit_px * pos['shares']
                 pnl = (exit_px - pos['entry']) / pos['entry']
                 history.append(pnl)
                 tickers_del.append(t)
                 continue
        
        for t in tickers_del: del positions[t]
        
        # 4. Buy
        if market_healthy and len(positions) < MAX_POSITIONS:
            for t, data in stock_data.items():
                if date not in data.index: continue
                curr_idx = data.index.get_loc(date)
                if curr_idx < 60: continue
                prev_idx = curr_idx - 1
                
                # Check Trend & RS
                if not data['InTrend'].iloc[prev_idx]: continue
                if data['RS_Score'].iloc[prev_idx] < 70: continue
                
                # Dynamic VCP Check
                closes = data['Close'].iloc[prev_idx-50:prev_idx+1].values
                highs = data['High'].iloc[prev_idx-50:prev_idx+1].values
                lows = data['Low'].iloc[prev_idx-50:prev_idx+1].values
                volumes = data['Volume'].iloc[prev_idx-50:prev_idx+1].values
                vol_sma = data['VolSMA50'].iloc[prev_idx]
                
                h_handle = highs[-15:].max()
                l_handle = lows[-15:].min()
                curr_c = closes[-1]
                
                depth = (h_handle - l_handle) / h_handle
                near_pivot = (h_handle - curr_c) / h_handle < 0.06
                
                # PARAM CHECK: Depth
                if depth > p_depth: continue
                if not near_pivot: continue
                
                # PARAM CHECK: Vol
                recent_vol = np.mean(volumes[-5:])
                if recent_vol > (vol_sma * p_vol): continue
                
                # Trigger?
                pivot = h_handle + 0.02
                if data['High'].iloc[curr_idx] > pivot:
                    # Buy
                    entry = max(pivot, data['Open'].iloc[curr_idx])
                    stop = entry * (1 - p_stop)
                    target = entry * (1 + p_target)
                    
                    risk = current_equity * 0.02
                    shares = int(risk / (entry - stop))
                    max_shares = int((current_equity * 0.25) / entry)
                    shares = min(shares, max_shares)
                    
                    if shares > 0 and cash > (shares * entry):
                        cash -= shares * entry
                        positions[t] = {
                            'shares': shares, 'entry': entry, 'stop': stop, 
                            'target': target, 'entry_date': date
                        }
                        
    # Calc Stats
    total_trades = len(history)
    final_equity = cash + sum([positions[t]['shares'] * stock_data[t].iloc[-1]['Close'] 
                                     for t in positions])
    return_pct = (final_equity - INITIAL_CAPITAL) / INITIAL_CAPITAL
    
    wins = [x for x in history if x > 0]
    losses = [x for x in history if x <= 0]
    win_rate = len(wins) / total_trades if total_trades > 0 else 0
    gross_win = sum(wins)
    gross_loss = abs(sum(losses))
    pf = gross_win / gross_loss if gross_loss > 0 else 0
    
    # Score: Weighted mix of Profit and Win Rate
    # We want high profit but punish low win rates (<30%)
    score = return_pct * 100
    if win_rate < 0.40: score *= 0.5 # Penalty for low win rate
    
    return SimulationResult(params, final_equity, total_trades, win_rate, pf, return_pct, score)

def main():
    print("--- ULTRA-ELITE OPTIMIZER ENGINE ---")
    stock_data, spy_data = load_and_prep_data()
    if not stock_data: return
    
    keys, values = zip(*PARAM_GRID.items())
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    print(f"Running {len(combinations)} simulations...")
    
    results = []
    start_time = time.time()
    
    for i, params in enumerate(combinations):
        if i % 10 == 0: print(f"Sim {i}/{len(combinations)}...")
        res = run_single_sim(stock_data, spy_data, params)
        results.append(res)
        
    print(f"Optimization finished in {time.time() - start_time:.2f}s")
    
    # Sort
    results.sort(key=lambda x: x.score, reverse=True)
    
    print("\nTOP 5 CONFIGURATIONS:")
    print(f"{'Rank':<5} {'Score':<8} {'Ret%':<8} {'WR%':<6} {'PF':<6} {'Trades':<6} {'Params'}")
    print("-" * 80)
    
    for i, r in enumerate(results[:5]):
        print(f"{i+1:<5} {r.score:<8.2f} {r.return_pct*100:<8.1f} {r.win_rate*100:<6.1f} {r.profit_factor:<6.2f} {r.total_trades:<6} {r.params}")

if __name__ == "__main__":
    main()
