import yfinance as yf
import pandas as pd
import numpy as np
import requests
import os
import time
import json
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
SECTOR_CACHE_FILE = os.path.join(CACHE_DIR, "sector_ohlcv.parquet")
PORTFOLIO_FILE = "portfolio.json"

# Strategy Parameters (v2.1 Optimized)
PARAM_DEPTH = 0.15      
PARAM_STOP = 0.05       
PARAM_TARGET = 0.25     
PARAM_VOL_MULT = 1.0    
ACCOUNT_SIZE = 100000 
RISK_PER_TRADE = 1000

SECTOR_ETFS = {
    "Technology": "XLK",
    "Financials": "XLF",
    "Healthcare": "XLV",
    "Consumer Discretionary": "XLY",
    "Consumer Staples": "XLP",
    "Energy": "XLE",
    "Industrials": "XLI",
    "Materials": "XLB",
    "Real Estate": "XLRE",
    "Utilities": "XLU",
    "Communication Services": "XLC"
}

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
    sector: str
    sector_rank: int
    atr: float
    smart_size: int 

class PortfolioManager:
    def __init__(self):
        self.filepath = PORTFOLIO_FILE
        if not os.path.exists(self.filepath):
            with open(self.filepath, 'w') as f:
                json.dump({}, f)
                
    def load(self):
        with open(self.filepath, 'r') as f:
            return json.load(f)
            
    def save(self, data):
        with open(self.filepath, 'w') as f:
            json.dump(data, f, indent=4)
            
    def add_position(self, ticker, entry_price, shares, stop_loss, target):
        data = self.load()
        data[ticker] = {
            "entry_date": datetime.now().strftime("%Y-%m-%d"),
            "entry_price": float(entry_price),
            "shares": int(shares),
            "stop_loss": float(stop_loss),
            "target": float(target),
            "highest_price": float(entry_price) # For trailing stop
        }
        self.save(data)
        print(f"[+] Added {ticker} to Portfolio.")

    def review(self):
        data = self.load()
        if not data:
            print("\n[i] Portfolio is empty. Use 'Add Position' to track trades.")
            return

        print("\n==================================================")
        print("   PORTFOLIO MANAGER (Your Positions)")
        print("==================================================")
        
        tickers = list(data.keys())
        if not tickers: return
        
        # Get live data
        prices = yf.download(tickers, period="5d", interval="1d", group_by='ticker', auto_adjust=True, threads=True)
        
        table = []
        for t in tickers:
            try:
                # Handle Structure (Single vs Multi Ticker)
                df = None
                if isinstance(prices.columns, pd.MultiIndex):
                     if t in prices.columns.levels[0]:
                         df = prices[t]
                else:
                    # Flat DF (e.g. single ticker without group_by effect or old yf version)
                    df = prices
                    
                if df is None: 
                    # Try direct access if flat DF has columns like 'Close'
                    if 'Close' in prices.columns and len(tickers) == 1:
                        df = prices
                    else:
                        continue 

                curr_price = float(df['Close'].iloc[-1])
                high_price = float(df['High'].iloc[-1])
                
                pos = data[t]
                entry = pos['entry_price']
                shares = pos['shares']
                
                # Update High for Trail
                if high_price > pos.get('highest_price', 0):
                    pos['highest_price'] = high_price
                    self.save(data)
                
                # Check Rules
                days_held = (datetime.now() - datetime.strptime(pos['entry_date'], "%Y-%m-%d")).days
                pnl_pct = (curr_price - entry) / entry
                
                status = "HOLD"
                action = ""
                
                # 1. Stop Loss
                if curr_price < pos['stop_loss']:
                    status = "EXIT (STOP)"
                    action = "SELL IMMEDIATELY"
                    
                # 2. Target
                elif curr_price >= pos['target']:
                    status = "EXIT (WIN)"
                    action = "TAKE PROFIT"
                    
                # 3. Time Stop (5 Days, <1% profit)
                elif days_held >= 5 and pnl_pct < 0.01:
                    status = "EXIT (TIME)"
                    action = "CUT DEAD MONEY"
                    
                table.append([
                    t, f"${entry:.2f}", f"${curr_price:.2f}", f"{pnl_pct*100:.1f}%", 
                    days_held, status, action
                ])
            except Exception as e:
                print(f"Error reviewing {t}: {e}")
                
        print(tabulate(table, headers=["Ticker", "Entry", "Current", "PnL%", "Days", "Status", "Action"], tablefmt="fancy_grid"))

class EliteBrainV3(PortfolioManager): # Inherits Portfolio features
    def __init__(self):
        super().__init__() # Init portfolio
        if not os.path.exists(CACHE_DIR):
            os.makedirs(CACHE_DIR)
            
    # --- Data & Scanning Logic (Same as v2.5) ---
    def get_sp500_tickers(self):
        if os.path.exists(SP500_CACHE_FILE):
             if (time.time() - os.path.getmtime(SP500_CACHE_FILE)) < 86400 * 7:
                return pd.read_json(SP500_CACHE_FILE, typ='series').tolist()
        try:
            url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
            df = pd.read_html(requests.get(url, headers={"User-Agent": "Mozilla/5.0"}).text)[0]
            tickers = [t.replace('.', '-') for t in df['Symbol'].tolist()]
            pd.Series(tickers).to_json(SP500_CACHE_FILE)
            return tickers
        except: return ["NVDA", "MSFT"]

    def load_data(self, tickers):
        if os.path.exists(OHLCV_CACHE_FILE):
            if (time.time() - os.path.getmtime(OHLCV_CACHE_FILE)) < 43200:
                return pd.read_parquet(OHLCV_CACHE_FILE)
        print("Downloading market data...")
        data = yf.download(tickers, period="2y", auto_adjust=True, threads=True, group_by='ticker')
        data.to_parquet(OHLCV_CACHE_FILE)
        return data
        
    def load_sector_data(self):
        tickers = list(SECTOR_ETFS.values())
        if os.path.exists(SECTOR_CACHE_FILE):
             if (time.time() - os.path.getmtime(SECTOR_CACHE_FILE)) < 43200:
                return pd.read_parquet(SECTOR_CACHE_FILE)
        data = yf.download(tickers, period="1y", auto_adjust=True, group_by='ticker')
        data.to_parquet(SECTOR_CACHE_FILE)
        return data

    def calculate_atr(self, df, period=14):
        high = df['High']
        low = df['Low']
        close = df['Close'].shift(1)
        tr = pd.concat([high-low, (high-close).abs(), (low-close).abs()], axis=1).max(axis=1)
        return tr.rolling(period).mean().iloc[-1]

    def scan(self):
        tickers = self.get_sp500_tickers()
        data = self.load_data(tickers)
        sector_data = self.load_sector_data()
        
        # Sector Rankings
        sector_ranks = {}
        # Simple sector RS logic
        spy = yf.download("SPY", period="2y", auto_adjust=True)['Close']
        # Handle duplicates in SPY
        if isinstance(spy, pd.DataFrame): spy = spy.iloc[:, 0]
        
        sector_list = []
        
        for name, ticker in SECTOR_ETFS.items():
            try:
                # Handle Structure
                s_df = pd.DataFrame()
                if isinstance(sector_data.columns, pd.MultiIndex):
                    if ticker in sector_data.columns.levels[0]:
                        s_df = sector_data[ticker]
                    elif ticker in sector_data.columns.levels[1]:
                         s_df = sector_data.xs(ticker, axis=1, level=1)
                else:
                    if ticker in sector_data: s_df = sector_data[ticker]
                    
                if s_df.empty: continue
                
                # Check for Close
                if 'Close' not in s_df.columns: continue
                s_close = s_df['Close']
                
                # Dedupe
                if isinstance(s_close, pd.DataFrame): s_close = s_close.iloc[:, 0]
                
                sec_ret = float(s_close.pct_change(126).iloc[-1])
                spy_ret = float(spy.pct_change(126).iloc[-1])
                
                rs = (sec_ret - spy_ret) * 100 + 50
                sector_list.append({'name': name, 'rs': rs})
            except Exception as e: 
                pass
        
        sector_list.sort(key=lambda x: x['rs'], reverse=True)
        sector_ranks = {s['name']: i+1 for i, s in enumerate(sector_list)}
        
        print("\nScanning Market (v3.0)...")
        results = []
        
        is_multi = isinstance(data.columns, pd.MultiIndex) and 'Close' in data.columns.levels[0]
        
        for t in tickers:
            try:
                if is_multi:
                    if t not in data.columns.levels[1]: continue
                    df = data.xs(t, axis=1, level=1).dropna()
                else:
                    if t not in data.columns: continue
                    df = data[t].dropna()
                
                # Deduplicate columns to prevent Series errors
                df = df.loc[:, ~df.columns.duplicated()]
                
                if len(df) < 250: continue
                
                # Trend
                c = float(df['Close'].iloc[-1])
                sma50 = float(df['Close'].rolling(50).mean().iloc[-1])
                sma200 = float(df['Close'].rolling(200).mean().iloc[-1])
                if not (c > sma50 > sma200): continue
                
                # VCP
                h_h = float(df['High'][-15:].max())
                l_h = float(df['Low'][-15:].min())
                if (h_h - l_h)/h_h > PARAM_DEPTH: continue # Too deep
                if (h_h - c)/h_h > 0.06: continue # Not near pivot
                
                # Setup Found
                atr = float(self.calculate_atr(df))
                
                # Volatility Sizing
                risk_dist = max(atr * 2, h_h * 0.05)
                shares = int(RISK_PER_TRADE / risk_dist)
                
                # Sector
                info = yf.Ticker(t).info
                sec = info.get('sector', 'Unknown')
                rank = sector_ranks.get(sec, 10)
                
                # RS
                rs = float((df['Close'].pct_change(126).iloc[-1] - spy.pct_change(126).iloc[-1]) * 100 + 50)
                if rs < 70: continue
                
                score = rs + (10 if rank <=3 else 0)
                
                results.append(EliteSetup(
                    t, c, h_h + 0.05, h_h - risk_dist, h_h + (risk_dist*3), score, rs,
                    True, 0, 0, 0, sec, rank, atr, shares
                ))
            except: continue
            
        return results, sector_list

def main():
    print("\n$$ ELITE TRADE v3.0 (MANAGER MODE) $$")
    brain = EliteBrainV3()
    
    # 1. Review Open Positions
    brain.review()
    
    # 2. Market Scan
    print("\n------------------------------------------------")
    input("Press Enter to Run Market Scan...")
    setups, sectors = brain.scan()
    
    if setups:
        setups.sort(key=lambda x: x.score, reverse=True)
        top = setups[:5]
        
        table = []
        for s in top:
            table.append([
                s.ticker, f"${s.pivot_price:.2f}", f"${s.target_price:.2f}", 
                s.smart_size, f"${s.stop_loss:.2f}", s.sector[:10]
            ])
        print("\n[$$] NEW SIGNALS")
        print(tabulate(table, headers=["Ticker", "Buy Stop", "TARGET", "Qty", "Stop Loss", "Sector"], tablefmt="fancy_grid"))
        
        print("\n[?] STRATEGY KEY:")
        print("    * Buy Stop:  ONLY Buy if price CROSSES UP through this level (Don't buy if below!).")
        print("    * Qty:       Recommended share count to limit risk to $1,000.")
        print("    * Status:    HOLD = Keep it. EXIT = Sell it now.")
        
        # 3. Add Position Prompt
        print("\n------------------------------------------------")
        choice = input("Did you take a trade? Enter Ticker to Track (or Enter to skip): ").strip().upper()
        if choice:
            for s in top:
                if s.ticker == choice:
                    brain.add_position(s.ticker, s.pivot_price, s.smart_size, s.stop_loss, s.target_price)
                    break
            else:
                # Custom add
                print("Ticker not in Top 5. Manual Entry:")
                p = input("Entry Price: ")
                q = input("Shares: ")
                sl = input("Stop Loss: ")
                tp = input("Target: ")
                brain.add_position(choice, p, q, sl, tp)
                
    else:
        print("No setups found.")

if __name__ == "__main__":
    main()
