import os
import time
import pandas as pd
import numpy as np
import itertools
from dataclasses import dataclass

# Disable warnings
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
pd.options.mode.chained_assignment = None

CACHE_DIR = "cache_sp500_elite"
OHLCV_CACHE_FILE = os.path.join(CACHE_DIR, "sp500_ohlcv_bulk.parquet")

INITIAL_CAPITAL = 100000

@dataclass
class Candidate:
    ticker: str
    date: pd.Timestamp
    entry_price: float
    pivot: float
    depth: float     # 0.10, 0.15 etc
    vol_ratio: float # 0.8, 1.2 etc
    rs_score: float
    # Outcome if held 5 days, 10 days, max_outcome etc
    # To speed up, we can pre-calc the "Potential Outcomes" for this candidate?
    # No, exit depends on stop/target params. 
    # But we can store the FUTURE 50 DAYS PRICE ARRAY for this candidate to allow fast sim.
    future_prices: np.ndarray 
    future_dates: np.ndarray

# ---------------------------------------------------------
# FAST OPTIMIZER LOGIC
# ---------------------------------------------------------
def load_and_prep_data():
    if not os.path.exists(OHLCV_CACHE_FILE): return None, None
    print("Loading data...")
    df = pd.read_parquet(OHLCV_CACHE_FILE)
    
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
            
            # Indicators
            cols['SMA50'] = cols['Close'].rolling(50).mean()
            cols['SMA150'] = cols['Close'].rolling(150).mean()
            cols['SMA200'] = cols['Close'].rolling(200).mean()
            cols['Low52'] = cols['Close'].rolling(252).min()
            cols['High52'] = cols['Close'].rolling(252).max()
            cols['VolSMA50'] = cols['Volume'].rolling(50).mean()
            cols['RS_Score'] = cols['Close'].pct_change(126) * 100
            
            # InTrend
            cols['InTrend'] = (
                (cols['Close'] > cols['SMA50']) & (cols['SMA50'] > cols['SMA150']) &
                (cols['SMA150'] > cols['SMA200']) & 
                (cols['Close'] > cols['Low52'] * 1.25) & (cols['Close'] > cols['High52'] * 0.75)
            )
            stock_data[t] = cols
        except: continue
    return stock_data, spy_data

def pre_scan_candidates(stock_data):
    """Scan ONCE for all LOOSE valid setups (Depth < 0.25)."""
    print("Pre-scanning all candidates (This happens once)...")
    candidates = []
    
    # We scan the last 252 days
    # To do this fast, we still loop stocks, but we just check conditions
    
    for t, df in stock_data.items():
        # Convert to numpy for speed
        closes = df['Close'].values
        highs = df['High'].values
        lows = df['Low'].values
        volumes = df['Volume'].values
        vol_smas = df['VolSMA50'].values
        trends = df['InTrend'].values
        rs_scores = df['RS_Score'].values
        dates = df.index
        
        # Test range: last 252 days
        start_idx = len(df) - 252
        if start_idx < 60: start_idx = 60
        
        for i in range(start_idx, len(df)):
            prev = i - 1
            
            if not trends[prev]: continue
            if rs_scores[prev] < 60: continue
            
            # VCP Check (Numpy slicing is fast enough here)
            # Window 50 days
            h_h = highs[prev-15:prev+1].max() # Handle High
            l_h = lows[prev-15:prev+1].min()  # Handle Low
            curr_c = closes[prev]
            
            depth = (h_h - l_h) / h_h
            if depth > 0.25: continue # Too loose
            
            near_pivot = (h_h - curr_c) / h_h < 0.08
            if not near_pivot: continue
            
            # Vol Ratio
            recent_vol = np.mean(volumes[prev-5:prev+1])
            vol_ratio = recent_vol / vol_smas[prev]
            
            # Check Breakout TODAY (i)
            pivot = h_h + 0.02
            if highs[i] > pivot:
                # Valid Candidate!
                # Store future data for sim
                entry = max(pivot, df['Open'].iloc[i])
                
                # Store next 60 days of OHLC data for fast exit check
                # We need High, Low, Close
                # Let's just store the Arrays
                fut_slice = slice(i, min(i+60, len(df)))
                
                cand = Candidate(
                    t, dates[i], entry, pivot, depth, vol_ratio, rs_scores[prev],
                    df.iloc[fut_slice][['High', 'Low', 'Close']].values,
                    dates[fut_slice].values
                )
                candidates.append(cand)
                
    print(f"Found {len(candidates)} total candidates.")
    return candidates

def run_fast_sim(candidates, params):
    # Sort candidates by date then RS
    # Params
    p_depth = params['vcp_depth']
    p_stop = params['stop_loss']
    p_target = params['target_pct']
    p_vol = params['vol_mult']
    
    # Filter candidates first (The beauty of Pre-Calc)
    valid_cands = [
        c for c in candidates 
        if c.depth <= p_depth and c.vol_ratio <= p_vol
    ]
    # Sort by Date (for simulation time integrity)
    valid_cands.sort(key=lambda x: x.date)
    
    # Run Valid candidates
    cash = INITIAL_CAPITAL
    equity = INITIAL_CAPITAL
    positions = {} # t -> {shares, stop, ...}
    history = []
    
    # We need to step through TIME.
    # Get all unique dates from candidates
    # But we also need to manage exits every day...
    # Okay, "Fast Sim" is non-trivial if we need portfolio constraints.
    # Simplified approach:
    # 1. Iterate unique dates in valid_cands range.
    # 2. Daily: Check Exits -> Check Entries.
    
    if not valid_cands: return 0, 0, 0, 0, 0
    
    start_date = valid_cands[0].date
    end_date = valid_cands[-1].date
    
    # Group candidates by date
    cands_by_date = {}
    for c in valid_cands:
        d = c.date
        if d not in cands_by_date: cands_by_date[d] = []
        cands_by_date[d].append(c)
        
    # Main Loop
    # We need a continuous date iterator? 
    # Or just jump to dates where things happen? (Dates with entries OR exits)
    # Correct way: Iterate all market days. 
    # Approx way: Iterate sorted unique dates of entries? No, miss exits.
    # Let's iterate the date range from start to end.
    
    current_date = start_date
    
    # Helper: Active positions map
    # We need to know price of held stocks to update equity, 
    # but for speed we can update equity only on trade close + final.
    
    active_pos = [] # list of dicts
    
    # Map dates to index for performance? keeping it simple.
    # Actually, we can just loop through the candidate list and "hold" them.
    # PROBLEM: We need to know if we have available cash.
    # So we MUST process sequentially.
    
    # Let's just iterate through the sorted list of ALL dates involved (entries + next 60 days)
    # To save time, we assume standard daily steps.
    
    # Re-use the dates from the candidates
    # This is complex to do perfectly fast.
    # COMPROMISE: We run a loop over the sorted unique dates present in the candidate dataset.
    
    all_dates = sorted(list(set([c.date for c in candidates] + [pd.Timestamp(c.future_dates[-1]) for c in candidates])))
    
    # Pointer for entry candidates
    cand_idx = 0
    
    for date in all_dates:
        # 1. Check Exits
        still_active = []
        for pos in active_pos:
            # Find today's price in pos data
            # pos['data'] matches pos['dates']
            
            # Check if date is in pos['dates']
            # We can optimise this using relative index 
            # days_held = (date - pos['entry_date']).days ... but weekends.
            # Fast check:
            
            try:
                # Find row for this date
                # Slow?
                # Optimization: Position stores (current_idx)
                idx = pos['idx']
                if idx >= len(pos['values']): 
                    # End of data - force close
                    exit_px = pos['values'][-1][2] # Close
                    pnl = (exit_px - pos['entry']) / pos['entry']
                    history.append(pnl)
                    cash += exit_px * pos['shares']
                    continue
                    
                row = pos['values'][idx] # [High, Low, Close]
                pos['idx'] += 1 # Increment for next day
                
                # Check Stop
                if row[1] < pos['stop']:
                    exit_px = pos['stop']
                    if idx == 0: # Entry day logic?
                         pass # Assume executed
                    
                    cash += exit_px * pos['shares']
                    history.append((exit_px - pos['entry']) / pos['entry'])
                    continue # Closed
                    
                # Check Target
                target = pos['entry'] * (1 + p_target)
                if row[0] > target:
                    cash += target * pos['shares']
                    history.append(p_target)
                    continue # Closed
                    
                # Time Stop (5 days)
                if idx >= 5 and (row[2] - pos['entry'])/pos['entry'] < 0.01:
                    cash += row[2] * pos['shares']
                    history.append((row[2] - pos['entry']) / pos['entry'])
                    continue
                    
                still_active.append(pos)
                
            except:
                continue
                
        active_pos = still_active
        equity = cash + sum([p['values'][min(p['idx'], len(p['values'])-1)][2] * p['shares'] for p in active_pos])
        
        # 2. Check Entries
        if date in cands_by_date and len(active_pos) < 5:
            avail_cands = cands_by_date[date]
            # Sort by RS?
            avail_cands.sort(key=lambda x: x.rs_score, reverse=True)
            
            for c in avail_cands:
                if len(active_pos) >= 5: break
                
                risk = equity * 0.02
                stop = c.entry_price * (1 - p_stop)
                shares = int(risk / (c.entry_price - stop))
                max_cost = equity * 0.25
                if (shares * c.entry_price) > max_cost:
                    shares = int(max_cost / c.entry_price)
                    
                if shares > 0 and cash >= (shares * c.entry_price):
                    cash -= (shares * c.entry_price)
                    active_pos.append({
                        'shares': shares,
                        'entry': c.entry_price,
                        'stop': stop,
                        'values': c.future_prices, # Numpy array
                        'dates': c.future_dates,
                        'idx': 0 # Points to today (Day 0)
                    })
                    
    # Results
    ret_pct = (equity - INITIAL_CAPITAL) / INITIAL_CAPITAL
    wins = [x for x in history if x > 0]
    wr = len(wins) / len(history) if history else 0
    pf = sum(wins) / abs(sum([x for x in history if x <= 0])) if history and sum([x for x in history if x<=0]) != 0 else 0
    
    score = ret_pct * 100
    if wr < 0.4: score *= 0.5
    
    return score, ret_pct, wr, pf, len(history)

def main():
    stock_data, _ = load_and_prep_data()
    if not stock_data: return
    
    candidates = pre_scan_candidates(stock_data)
    
    # Grid
    # Expanded grid because it's fast now
    grid = {
        'vcp_depth': [0.12, 0.15, 0.18, 0.20, 0.22],
        'stop_loss': [0.04, 0.05, 0.06, 0.07, 0.08],
        'target_pct': [0.15, 0.20, 0.25, 0.30],
        'vol_mult': [0.8, 1.0, 1.2, 1.5]
    }
    
    keys, values = zip(*grid.items())
    combs = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    print(f"Testing {len(combs)} combinations...")
    results = []
    
    t0 = time.time()
    for i, p in enumerate(combs):
        if i % 100 == 0: print(f"Processing {i}...")
        res = run_fast_sim(candidates, p)
        results.append((res, p))
        
    print(f"Done in {time.time()-t0:.2f}s")
    
    # Sort
    results.sort(key=lambda x: x[0][0], reverse=True)
    
    print("\nWINNER PARAMETERS:")
    top = results[0]
    print(f"Params: {top[1]}")
    print(f"Score: {top[0][0]:.2f} | Ret: {top[0][1]*100:.1f}% | WR: {top[0][2]*100:.1f}% | PF: {top[0][3]:.2f}")
    
    # Save best to file
    with open("best_params.txt", "w") as f:
        f.write(str(top[1]))

if __name__ == "__main__":
    main()
