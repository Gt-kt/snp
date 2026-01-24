import yfinance as yf
import pandas as pd
import numpy as np
import requests
import os
import time
import json
import warnings
from datetime import datetime, timedelta
from dataclasses import dataclass
from tabulate import tabulate
import io

# Disable warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
pd.options.mode.chained_assignment = None

# --- CONSTANTS & CONFIG ---
CACHE_DIR = "cache_sp500_elite"
SP500_CACHE_FILE = os.path.join(CACHE_DIR, "sp500_tickers.json")
OHLCV_CACHE_FILE = os.path.join(CACHE_DIR, "sp500_ohlcv_bulk.parquet")
PORTFOLIO_FILE = "portfolio.json"

# Strategy Defaults
RISK_PER_TRADE = 1000  # $1000 Risk per trade
ACCOUNT_SIZE = 100000 


# -----------------------------------------------------------------------------
# 0. HELPERS & TRACKING
# -----------------------------------------------------------------------------
class RejectionTracker:
    def __init__(self):
        self.stats = {
            "Total": 0,
            "No Data": 0,
            "Low Price/Liquidity": 0,
            "Downtrend (Bear)": 0,
            "No Setup (VCP/Dip)": 0,
            "Rejected (Low Win%)": 0,
            "Bad Risk/Reward": 0,
            "Earnings Risk": 0,
            "Passed": 0
        }
    
    def update(self, reason):
        self.stats["Total"] += 1
        if reason in self.stats:
            self.stats[reason] += 1
        else:
            self.stats[reason] = 1

    def summary(self):
        return self.stats

@dataclass
class TitanSetup:
    ticker: str
    strategy: str      # "BREAKOUT" or "DIP"
    price: float
    trigger: float
    stop: float
    target: float
    qty: int
    win_rate: float    # From Reality Check
    profit_factor: float
    kelly: float       # Suggested size multiplier
    score: float       # Total Confidence Score
    sector: str
    earnings_call: str # New field for clarity
    note: str

# -----------------------------------------------------------------------------
# 1. MARKET REGIME (The "Traffic Light")
# -----------------------------------------------------------------------------
class MarketRegime:
    def __init__(self, data):
        self.data = data # Dictionary of DataFrames

    def analyze_spy(self):
        """
        Analyze SPY to determine market status.
        BULL: Price > SMA200 & SMA50 > SMA200
        BEAR: Price < SMA200
        NEUTRAL: Choppy
        """
        if "SPY" not in self.data:
            return "NEUTRAL", 0.5

        spy = self.data["SPY"]
        if isinstance(spy, pd.Series): # Handle single column edge case
             return "NEUTRAL", 0.5
             
        c = spy['Close']
        sma50 = c.rolling(50).mean().iloc[-1]
        sma200 = c.rolling(200).mean().iloc[-1]
        curr = c.iloc[-1]
        
        status = "NEUTRAL"
        score = 0.5
        
        if curr > sma200:
            if sma50 > sma200: 
                status = "BULL"
                score = 1.0 # Full Gas
            else: 
                status = "RECOVERY"
                score = 0.7 # Caution
        else:
            if curr < sma50:
                status = "BEAR"
                score = 0.0 # Stop
            else:
                status = "Correction"
                score = 0.2
                
        # VIX Check (Optional if available, simplified here)
        return status, score

# -----------------------------------------------------------------------------
# 2. VALIDATOR ENGINE (The "Reality Check")
# -----------------------------------------------------------------------------
class StrategyValidator:
    """
    Backtests a specific strategy logic on a specific single stock logic 
    over the last 1-2 years to see if it actually works.
    """
    def __init__(self, df):
        self.df = df
        
    def _simulate_trade(self, entry, stop, target, ohlc_data, start_idx):
        """Helper for fast simulation of a single trade outcome."""
        closes, highs, lows = ohlc_data
        for j in range(start_idx, min(start_idx+15, len(closes))): # Max hold 15 days for swing
            if lows[j] < stop: return -0.05 # Hit Stop (Approx)
            if highs[j] > target: return 0.15 # Hit Target (Approx)
            # Mark to Market if we run out of time
            if j == min(start_idx+15, len(closes)) - 1:
                return (closes[j] - entry) / entry
        return 0

    def backtest_breakout(self, days=300, depth=0.15, vol_mult=1.5, target_mult=3.5):
        """
        Fast simulation of VCP Breakouts on this specific stock.
        """
        df = self.df.iloc[-days:].copy()
        if len(df) < 100: return {'win_rate': 0, 'pf': 0, 'trades': 0}
        
        closes = df['Close'].values
        highs = df['High'].values
        lows = df['Low'].values
        opens = df['Open'].values
        volumes = df['Volume'].values
        
        # Pre-calc Indicators
        sma50 = df['Close'].rolling(50).mean().values
        sma200 = df['Close'].rolling(200).mean().values
        # ATR Pre-calc (Simplified 14-day)
        tr = np.maximum(df['High'] - df['Low'], np.abs(df['High'] - df['Close'].shift(1)))
        atr = tr.rolling(14).mean().values
        vol_sma = df['Volume'].rolling(50).mean().values
        
        trades = []
        
        # Loop (Simulation)
        # Scan from index 60 to end-1
        for i in range(60, len(df)-1):
            # 1. Trend Filter (Yesterday)
            if not (closes[i] > sma50[i] > sma200[i]): continue
            
            # 2. Pattern (VCP)
            # Handle: 15 days window ending at i
            h_handle = np.max(highs[i-15:i+1])
            l_handle = np.min(lows[i-15:i+1])
            curr_c = closes[i]
            
            # Depth
            d = (h_handle - l_handle) / h_handle
            if d > depth: continue # Too loose
            
            # Near Pivot
            if (h_handle - curr_c) / h_handle > 0.06: continue
            
            # Vol Check
            if volumes[i] > (vol_sma[i] * vol_mult): 
                continue # Don't buy on huge down/churn days? (Simplified)
            
            # SETUP FOUND at end of day 'i'.
            # CHECK NEXT DAY (i+1) for Trigger
            pivot = h_handle + 0.02
            
            next_h = highs[i+1]
            next_l = lows[i+1]
            next_o = opens[i+1]
            next_c = closes[i+1]
            
            if next_h > pivot:
                # Triggered!
                buy_price = max(pivot, next_o)
                
                atr_val = atr[i] if i < len(atr) and not np.isnan(atr[i]) else (buy_price*0.02)
                
                # Stop: Dynamic 2 ATR
                stop_loss = buy_price - (atr_val * 2.0)
                # Target: Variable (Optimized)
                target = buy_price + (atr_val * target_mult)
                
                # Check outcome (Day trade or Swing)
                
                # Did we hit stop same day?
                outcome_pct = 0
                
                # Check Day 1 Stop
                if next_l < stop_loss:
                    outcome_pct = -0.05
                elif next_h > target:
                    outcome_pct = (target - buy_price) / buy_price
                else:
                    # Held to close (Swing) -> Check next 10 days
                    exit_price = next_c
                    held = True
                    for j in range(i+2, min(i+12, len(df))):
                        if lows[j] < stop_loss:
                            exit_price = stop_loss
                            held = False
                            break
                        if highs[j] > target:
                            exit_price = target
                            held = False
                            break
                        exit_price = closes[j] # Mark to market
                    
                    outcome_pct = (exit_price - buy_price) / buy_price
                
                trades.append(outcome_pct)
        
        # Stats
        if not trades: return {'win_rate': 0, 'pf': 0, 'trades': 0}
        
        wins = [t for t in trades if t > 0]
        losses = [t for t in trades if t <= 0]
        
        win_rate = float(len(wins) / len(trades) * 100)
        gross_win = float(sum(wins))
        gross_loss = float(abs(sum(losses)))
        pf = float(gross_win / gross_loss if gross_loss > 0 else (100.0 if gross_win > 0 else 0))
        
        return {'win_rate': win_rate, 'pf': pf, 'trades': len(trades)}

    def backtest_dip(self, days=300):
        """
        Fast simulation of Dip Buys (SMA50 support).
        """
        df = self.df.iloc[-days:].copy()
        if len(df) < 100: return {'win_rate': 0, 'pf': 0, 'trades': 0}
        
        sma50 = df['Close'].rolling(50).mean()
        closes = df['Close']
        lows = df['Low']
        highs = df['High']
        
        trades = []
        
        for i in range(50, len(df)-5):
            # Logic: Trend Up, Pullback to SMA50
            if closes.iloc[i] > sma50.iloc[i]:
                # Check for "Touch" of SMA50 in recent days
                dist = (lows.iloc[i] - sma50.iloc[i]) / sma50.iloc[i]
                
                if -0.02 < dist < 0.02: # Touched/Near
                    # Buy
                    buy_price = closes.iloc[i]
                    stop = buy_price * 0.93
                    target = buy_price * 1.10
                    
                    # Forward Check (next 10 days)
                    exit_price = buy_price
                    for j in range(i+1, min(i+10, len(df))):
                        if lows.iloc[j] < stop:
                            exit_price = stop
                            break
                        if highs.iloc[j] > target:
                            exit_price = target
                            break
                        exit_price = closes.iloc[j]
                        
                    trades.append((exit_price - buy_price)/buy_price)
                    
        if not trades: return {'win_rate': 0, 'pf': 0, 'trades': 0}
        wins = [t for t in trades if t > 0]
        losses = [t for t in trades if t <= 0]
        win_rate = len(wins) / len(trades) * 100
        pf = sum(wins)/abs(sum(losses)) if sum(losses) != 0 else (100.0 if sum(wins) > 0 else 0)
        
        return {'win_rate': win_rate, 'pf': pf, 'trades': len(trades)}

# -----------------------------------------------------------------------------
# 3. OPTIMIZER (Auto-Tuning)
# -----------------------------------------------------------------------------
class Optimizer:
    def __init__(self, validator):
        self.validator = validator
        
    def tune_breakout(self):
        """
        Try different parameters including TARGET MULTIPLIER to find the best fit.
        """
        best_res = {'win_rate': 0, 'pf': 0, 'score': 0}
        best_params = {'depth': 0.15, 'target_mult': 3.5}
        
        # Grid Search: Depth vs Target
        # Depth: 0.15 (Tight) to 0.25 (Loose)
        # Target: 2.5 ATR (Quick) to 6.0 ATR (Runner)
        for d in [0.15, 0.20, 0.25]:
            for t_mult in [2.5, 3.5, 5.0, 6.0]:
                res = self.validator.backtest_breakout(depth=d, target_mult=t_mult)
                # Score: PF * WR
                score = res['pf'] * res['win_rate']
                
                if score > best_res['score'] and res['trades'] >= 2:
                    best_res = res
                    best_res['score'] = score
                    best_params = {'depth': d, 'target_mult': t_mult}
                
        return best_res, best_params

# -----------------------------------------------------------------------------
# 4. MAIN BRAIN
# -----------------------------------------------------------------------------
class TitanBrain:
    def __init__(self):
        if not os.path.exists(CACHE_DIR): os.makedirs(CACHE_DIR)

    def calculate_rsi(self, series, period=14):
        delta = series.diff()
        gain = delta.where(delta > 0, 0).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        rs = gain / (loss + 1e-9)
        return 100 - (100 / (1 + rs))

    def get_data(self):
        # Tickers
        if os.path.exists(SP500_CACHE_FILE) and (time.time() - os.path.getmtime(SP500_CACHE_FILE) < 604800):
            tickers = pd.read_json(SP500_CACHE_FILE, typ='series').tolist()
        else:
            print("Fetching S&P List...")
            try:
                headers = {"User-Agent": "Mozilla/5.0"}
                df = pd.read_html(io.StringIO(requests.get("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies", headers=headers).text))[0]
                tickers = [t.replace('.', '-') for t in df['Symbol'].tolist()]
                pd.Series(tickers).to_json(SP500_CACHE_FILE)
            except:
                tickers = ["NVDA", "MSFT", "AAPL", "AMD", "TSLA", "SPY"]

        # OHLCV
        if os.path.exists(OHLCV_CACHE_FILE) and (time.time() - os.path.getmtime(OHLCV_CACHE_FILE) < 43200):
            print("Loading Market Cache...")
            data = pd.read_parquet(OHLCV_CACHE_FILE)
        else:
            print("Downloading Market Data (This may take 1-2 minutes)...")
            tickers_plus = tickers + ["SPY"]
            
            # Chunking the download to avoid Rate Limits
            chunk_size = 100
            data_frames = []
            
            for i in range(0, len(tickers_plus), chunk_size):
                chunk = tickers_plus[i:i+chunk_size]
                print(f"  > Batch {i//chunk_size + 1}: {chunk[:3]}...", end='\r')
                try:
                    # Threading False acts nicer to APIs sometimes
                    d = yf.download(chunk, period="2y", auto_adjust=True, group_by='ticker', threads=True, progress=False)
                    data_frames.append(d)
                except Exception as e:
                    print(f"\nExample Batch Failed: {e}")
            
            print("\nMerging Data...")
            if data_frames:
                data = pd.concat(data_frames, axis=1)
                data.to_parquet(OHLCV_CACHE_FILE)
            else:
                raise ValueError("Failed to download any data.")
            
        return tickers, data
        
    def calculate_atr(self, df):
        h, l, c = df['High'], df['Low'], df['Close'].shift(1)
        tr = pd.concat([h-l, (h-c).abs(), (l-c).abs()], axis=1).max(axis=1)
        return float(tr.rolling(14).mean().iloc[-1])

    def process_ticker(self, t, data, mkt_status, spy_close):
        """Analyze a single ticker. Returns (TitanSetup, RejectionReason)."""
        try:
            # Extract DF
            if isinstance(data.columns, pd.MultiIndex):
                if t in data.columns.levels[0]: 
                    df = data[t].copy().dropna()
                else: return None, "No Data"
            else: return None, "No Data"
            
            if len(df) < 250: return None, "No Data"
            
            # --- A. INITIAL FILTER (Fast) ---
            c = float(df['Close'].iloc[-1])
            if c < 5.0: return None, "Low Price/Liquidity" # Penny stock filter
            
            close = df['Close']
            high = df['High']
            low = df['Low']
            volume = df['Volume']

            sma50 = float(close.rolling(50).mean().iloc[-1])
            sma200 = float(close.rolling(200).mean().iloc[-1])
            sma50_prev = float(close.rolling(50).mean().iloc[-21])
            sma200_prev = float(close.rolling(200).mean().iloc[-21])
            vol_avg20 = float(volume.rolling(20).mean().iloc[-1])
            dollar_vol_avg20 = vol_avg20 * c
            if dollar_vol_avg20 < 5_000_000: 
                return None, "Low Price/Liquidity"

            atr = self.calculate_atr(df)
            if atr <= 0 or atr / c > 0.12:
                return None, "Low Price/Liquidity"
            
            # Relative strength vs SPY (3M)
            rs_3m = close.pct_change(63).iloc[-1]
            if spy_close is not None:
                spy_rs = spy_close.pct_change(63).iloc[-1]
                rs_diff = rs_3m - spy_rs
            else:
                rs_diff = rs_3m
            if np.isnan(rs_diff):
                rs_diff = 0.0
            ret_6m = close.pct_change(126).iloc[-1]
            rsi14 = self.calculate_rsi(close).iloc[-1]
            sma_uptrend = sma50 > sma50_prev and sma200 > sma200_prev

            # Setup Flags
            is_breakout_setup = False
            is_dip_setup = False
            
            # Breakout Candidate? Trend Up + VCP
            if c > sma50 > sma200 and rs_diff > 0 and ret_6m > 0 and sma_uptrend:
                h_h = float(high[-15:].max())
                l_h = float(low[-15:].min())
                depth = (h_h - l_h) / h_h
                vol_spike = volume.iloc[-1] > (vol_avg20 * 1.2)
                if depth < 0.25 and (h_h - c)/h_h < 0.08 and vol_spike: # Demand + tight range
                    is_breakout_setup = True
                    
            # Dip Candidate? Trend Up + Near SMA50
            if c > sma200 and rs_diff > -0.02 and ret_6m > -0.02 and sma_uptrend:
                dist = (c - sma50) / sma50
                vol_ok = volume.iloc[-1] <= (vol_avg20 * 1.2)
                if -0.03 < dist < 0.04 and vol_ok and rsi14 > 40:
                    is_dip_setup = True
                    
            if not (is_breakout_setup or is_dip_setup): 
                if c < sma200: return None, "Downtrend (Bear)"
                return None, "No Setup (VCP/Dip)"
            
            # --- B. REALITY CHECK (Validation) ---
            val = StrategyValidator(df)
            opt = Optimizer(val)
            
            final_res = None
            strategy_name = ""
            trigger = 0
            stop = 0
            target = 0
            
            if is_breakout_setup:
                # 1. Run Backtest
                res = val.backtest_breakout()
                params = {} # Default empty params
                
                # 2. Optimized if needed
                if res['win_rate'] < 50:
                    res, params = opt.tune_breakout()
                    
                if res['win_rate'] >= 50 and res['pf'] >= 1.2:
                    strategy_name = "BREAKOUT"
                    final_res = res
                    trigger = float(df['High'][-15:].max()) + 0.02
                    stop = trigger - (atr * 2)
                    
                    # Apply Optimized Target
                    tgt_mult = params.get('target_mult', 3.5)
                    target = trigger + (atr * tgt_mult)
                    
                    if tgt_mult > 4.5: strategy_name += "+" # Runner Mode
            
            elif is_dip_setup:
                res = val.backtest_dip()
                if res['win_rate'] >= 55:
                    strategy_name = "DIP BUY"
                    final_res = res
                    trigger = c
                    stop = c - (atr * 2.0)
                    target = c + (atr * 3.0)
            
            if not final_res:
                return None, "Rejected (Low Win%)"

            # Score Calculation
            score = final_res['win_rate'] + (final_res['pf'] * 10)
            if mkt_status == "BULL": score += 10
            
            W = final_res['win_rate'] / 100
            R = 2.0 
            kelly = (W - (1-W)/R) * 0.5
            if kelly < 0: kelly = 0
            
            risk_amt = min(RISK_PER_TRADE, ACCOUNT_SIZE * 0.01)
            shares = int(risk_amt / (trigger - stop)) if (trigger - stop) > 0 else 0
            
            # --- SUPER POWER 1: INFO CHECK ---
            earnings_call = "Unknown"
            sector = "Unknown"
            
            # Only fetch info for valid setups (Optimizes speed)
            try:
                # Use partial fetch or cache if possible, but for now specific call is ok for <10 items
                t_info = yf.Ticker(t).info
                sector = t_info.get('sector', 'Unknown')
                
                # Basic Valuation Check
                fwd_pe = t_info.get('forwardPE', 0)
                if fwd_pe and fwd_pe < 40: score += 5 # Value boost
                
            except: pass
            
            # --- SUPER POWER 2: GOLDEN RATIO FILTER (R:R) ---
            risk_per_share = trigger - stop
            reward_per_share = target - trigger
            
            if risk_per_share <= 0: return None, "Data Error"
            rr_ratio = reward_per_share / risk_per_share
            
            min_rr = 1.4 if strategy_name == "DIP BUY" else 1.8
            
            if rr_ratio < min_rr:
                return None, "Bad Risk/Reward"
            
            # Final Note Construction
            note_str = f"Hist: {final_res['trades']} trades"
            if rr_ratio >= 3.0: note_str += " | 💎 3R+ GEM"
            
            return TitanSetup(
                t, strategy_name, c, trigger, stop, target, shares,
                final_res['win_rate'], final_res['pf'], kelly*100, score, sector, 
                earnings_call, note_str
            ), "Passed"

        except Exception as e:
            return None, f"Error: {str(e)}"


    def scan(self):
        tickers, data = self.get_data()
        spy_close = None
        if isinstance(data.columns, pd.MultiIndex) and "SPY" in data.columns.levels[0]:
            spy_df = data["SPY"].dropna()
            if "Close" in spy_df:
                spy_close = spy_df["Close"]
        
        # 1. Check Market
        regime = MarketRegime(data)
        mkt_status, mkt_score = regime.analyze_spy()
        print(f"\n=== MARKET STATUS: {mkt_status} (Score: {mkt_score}) ===")
        if mkt_score == 0:
            print("!!! MARKET IS IN BEAR TREND. CAUTION ADVISED. !!!")
        
        print("\nScanning & Validating (v5 Reality Check)...")
        print(f"Analyzing {len(tickers)} stocks in parallel (Max 10 workers)...")
        
        results = []
        tracker = RejectionTracker()
        
        import concurrent.futures
        
        # Determine number of threads
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            # Submit all
            future_to_ticker = {
                executor.submit(self.process_ticker, t, data, mkt_status, spy_close): t
                for t in tickers
            }
            
            completed = 0
            for future in concurrent.futures.as_completed(future_to_ticker):
                completed += 1
                if completed % 25 == 0:
                    print(f"Progress: {completed}/{len(tickers)}", end='\r')
                    
                res, reason = future.result()
                tracker.update(reason)
                
                if res:
                    results.append(res)
                    print(f" > FOUND: {res.ticker} ({res.strategy}) WR:{res.win_rate:.0f}% PF:{res.profit_factor:.2f}")

        # Show Rejection Stats
        print("\n" + "-"*40)
        print(" SCAN FILTER REPORT")
        print("-" * 40)
        for k, v in tracker.summary().items():
            print(f"  {k.ljust(20)}: {v}")
        print("-" * 40)
        
        return results, tracker.summary()

def addToPortfolio(setups):
    """Interactive loop to add trades to portfolio."""
    import json
    from datetime import datetime
    import os
    
    # Auto-save logic
    while True:
        print("\n" + "="*50)
        print(" PORTFOLIO MANAGER")
        print(" Type Ticker to ADD to portfolio.json")
        print(" Press ENTER to Finish/Exit")
        print("-" * 50)
        
        choice = input(" [?] Add Ticker > ").strip().upper()
        if not choice: break
        
        # Find the setup
        target_setup = next((s for s in setups if s.ticker == choice), None)
        
        if not target_setup:
            print(f" ❌ '{choice}' not found in the Result List above.")
            continue
            
        # Load Portfolio
        port_file = "portfolio.json"
        if os.path.exists(port_file):
            try:
                with open(port_file, "r") as f:
                    port = json.load(f)
            except: port = {}
        else: port = {}
        
        if choice in port:
            print(f" ⚠️ {choice} is already in your portfolio!")
            if input("    Overwrite? (y/n) > ").lower() != 'y':
                continue
        
        # Create Entry
        entry = {
            "entry_date": datetime.now().strftime("%Y-%m-%d"),
            "entry_price": round(target_setup.trigger, 2),
            "shares": target_setup.qty,
            "stop_loss": round(target_setup.stop, 2),
            "target": round(target_setup.target, 2),
            "highest_price": round(target_setup.trigger, 2),
            "strategy": target_setup.strategy,
            "note": target_setup.note
        }
        
        port[choice] = entry
        
        # Save
        with open(port_file, "w") as f:
            json.dump(port, f, indent=4)
            
        print(f" ✅ ADDED {choice}: Entry ${entry['entry_price']} | Stop ${entry['stop_loss']} | Target ${entry['target']}")
        print(f"    (Size: {entry['shares']} shares)")


def main():
    print("\n$$ TITAN TRADE v6.0 (GUARDIAN EDITION) $$")
    print("----------------------------------------------")
    
    brain = TitanBrain()
    try:
        setups, stats = brain.scan()
    except KeyboardInterrupt:
        print("\nScan Cancelled.")
        return

    if setups:
        setups.sort(key=lambda x: x.score, reverse=True)
        
        # UI CLEAR & HEADER
        print("\n" * 3)
        print("="*60)
        print(f"  TITAN GUARDIAN v6.0 - RESULT SUMMARY")
        print("="*60)
        print(f"    * 'BUY NOW' = Price has triggered (Active Breakout)")
        print(f"    * 'PENDING' = Place Buy Stop Order at Trigger Price")
        print("-" * 60)
        
        table = []
        for s in setups[:15]: # Show Top 15 now
            # Determine Status
            dist = (s.trigger - s.price) / s.price
            if s.price >= s.trigger:
                status = "🚀 BUY NOW"
            elif dist < 0.01:
                status = "⚠️ NEAR"
            else:
                status = "⏳ PENDING"

            table.append([
                s.ticker, 
                s.strategy[:4], 
                f"${s.price:.2f}",
                f"${s.trigger:.2f}", 
                status,
                f"{s.win_rate:.0f}%", 
                f"{s.profit_factor:.2f}",
                f"{s.kelly:.1f}%",
                f"${s.stop:.2f}",
                f"${s.target:.2f}",
                s.note
            ])
            
        print(tabulate(table, headers=["Ticker", "Type", "Price", "Trigger", "Status", "Win%", "PF", "Kelly", "Stop", "Target", "Note"], tablefmt="fancy_grid"))
        
        print("\nRecommendation: Only take trades with Win% > 60% and PF > 1.5")
        
        # Interactive Portfolio Add
        with open("scan_results.txt", "w", encoding="utf-8") as f:
            f.write("TITAN TRADE SCAN RESULTS\n")
            f.write(f"Date: {datetime.now()}\n")
            f.write("="*60 + "\n")
            f.write(tabulate(table, headers=["Ticker", "Type", "Price", "Trigger", "Status", "Win%", "PF", "Kelly", "Stop", "Target", "Note"], tablefmt="grid"))
            f.write("\n\nSCAN FILTER REPORT\n")
            f.write("-" * 40 + "\n")
            for k, v in stats.items():
                f.write(f"{k.ljust(20)}: {v}\n")
        
        print(f"\nReport saved to scan_results.txt")
        addToPortfolio(setups)
        
    else:
        print("\nNo valid setups found that passed the Reality Check.")

if __name__ == "__main__":
    main()
