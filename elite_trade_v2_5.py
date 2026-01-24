import yfinance as yf
import pandas as pd
import numpy as np
import requests
import os
import time
from datetime import datetime
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

# -----------------------------------------------------------------------------
# OPTIMIZED PARAMETERS (v2.1 Base)
# -----------------------------------------------------------------------------
PARAM_DEPTH = 0.15      
PARAM_STOP = 0.05       
PARAM_TARGET = 0.25     
PARAM_VOL_MULT = 1.0    

# -----------------------------------------------------------------------------
# v2.5: SECTOR MAPPING & RISK CONFIG
# -----------------------------------------------------------------------------
ACCOUNT_SIZE = 100000 
RISK_PER_TRADE = 1000 # 1% Risk
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
    smart_size: int # Quantity to buy

class EliteBrainV2_5:
    def __init__(self):
        if not os.path.exists(CACHE_DIR):
            os.makedirs(CACHE_DIR)
            
    def get_sp500_tickers(self):
        if os.path.exists(SP500_CACHE_FILE):
            mod_time = os.path.getmtime(SP500_CACHE_FILE)
            if (time.time() - mod_time) < 86400 * 7: 
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
            return ["NVDA", "MSFT", "AAPL"] # Fallback

    def load_data(self, tickers):
        if os.path.exists(OHLCV_CACHE_FILE):
            if (time.time() - os.path.getmtime(OHLCV_CACHE_FILE)) < 43200: 
                print("Using cached market data...")
                return pd.read_parquet(OHLCV_CACHE_FILE)
        
        print(f"Downloading data for {len(tickers)} tickers...")
        data = yf.download(tickers, period="2y", interval="1d", group_by='ticker', auto_adjust=True, threads=True)
        data.to_parquet(OHLCV_CACHE_FILE)
        return data

    def load_sector_data(self):
        """Fetch/Load Sector ETF Data."""
        tickers = list(SECTOR_ETFS.values())
        if os.path.exists(SECTOR_CACHE_FILE):
            if (time.time() - os.path.getmtime(SECTOR_CACHE_FILE)) < 43200:
                return pd.read_parquet(SECTOR_CACHE_FILE)
                
        print("Downloading Sector Data...")
        data = yf.download(tickers, period="1y", interval="1d", group_by='ticker', auto_adjust=True)
        data.to_parquet(SECTOR_CACHE_FILE)
        return data

    def analyze_sectors(self, sector_data, spy_data):
        """Rank sectors by RS Rating."""
        ranks = []
        for name, ticker in SECTOR_ETFS.items():
            try:
                if ticker not in sector_data.columns.levels[0]: continue
                df = sector_data[ticker]
                rs = self.calculate_rs_rating(df['Close'], spy_data)
                ranks.append({'name': name, 'ticker': ticker, 'rs': rs})
            except:
                continue
                
        ranks.sort(key=lambda x: x['rs'], reverse=True)
        # return dict mapping sector name -> rank (1-based)
        rank_map = {r['name']: i+1 for i, r in enumerate(ranks)}
        return rank_map, ranks

    def calculate_atr(self, df, period=14):
        """Calculate Average True Range."""
        high = df['High']
        low = df['Low']
        close = df['Close'].shift(1)
        
        tr1 = high - low
        tr2 = (high - close).abs()
        tr3 = (low - close).abs()
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(period).mean().iloc[-1]
        return atr

    def calculate_rs_rating(self, close_series, spy_close):
        try:
            stock_ret = close_series.pct_change(126).iloc[-1]
            spy_ret = spy_close.pct_change(126).iloc[-1]
            excess = (stock_ret - spy_ret) * 100
            return max(1, min(99, 50 + excess))
        except:
            return 50

    def check_trend_template(self, df):
        try:
            c = df['Close'].iloc[-1]
            sma50 = df['Close'].rolling(50).mean().iloc[-1]
            sma150 = df['Close'].rolling(150).mean().iloc[-1]
            sma200 = df['Close'].rolling(200).mean().iloc[-1]
            low52 = df['Close'].rolling(252).min().iloc[-1]
            high52 = df['Close'].rolling(252).max().iloc[-1]
            
            if not (c > sma50 > sma150 > sma200): return False
            if not (c > low52 * 1.30): return False 
            if not (c > high52 * 0.80): return False 
            return True
        except:
            return False

    def detect_vcp_strict(self, df):
        try:
            if len(df) < 60: return False, 0.0, False
            closes = df['Close'][-50:].values
            highs = df['High'][-50:].values
            lows = df['Low'][-50:].values
            volumes = df['Volume'][-50:].values
            vol_sma = df['Volume'].rolling(50).mean().iloc[-1]
            
            h_handle = highs[-15:].max()
            l_handle = lows[-15:].min()
            curr_c = closes[-1]
            
            depth = (h_handle - l_handle) / h_handle
            if depth > PARAM_DEPTH: return False, 0.0, False
            
            near_pivot = (h_handle - curr_c) / h_handle < 0.06
            if not near_pivot: return False, 0.0, False
            
            recent_vol = np.mean(volumes[-5:])
            vol_dry_up = recent_vol < (vol_sma * PARAM_VOL_MULT)
            
            if vol_dry_up: return True, h_handle + 0.05, True
            return False, 0.0, False
        except:
            return False, 0.0, False

    def detect_vcp_loose(self, df):
        try:
            if len(df) < 60: return False, 0.0
            closes = df['Close'][-50:].values
            highs = df['High'][-50:].values
            lows = df['Low'][-50:].values
            h_h = highs[-15:].max()
            l_h = lows[-15:].min()
            depth = (h_h - l_h) / h_h
            if depth > 0.25: return False, 0.0
            if (h_h - closes[-1]) / h_h < 0.10: return True, h_h + 0.05
            return False, 0.0
        except: return False, 0.0

    def get_fundamentals_and_sector(self, ticker):
        try:
            t = yf.Ticker(ticker)
            info = t.info
            eps = (info.get('earningsGrowth', 0) or 0) * 100
            rev = (info.get('revenueGrowth', 0) or 0) * 100
            roe = (info.get('returnOnEquity', 0) or 0) * 100
            sector = info.get('sector', 'Unknown')
            return eps, rev, roe, sector
        except:
            return 0, 0, 0, 'Unknown'

    def scan(self):
        tickers = self.get_sp500_tickers()
        data = self.load_data(tickers)
        sector_data = self.load_sector_data()
        
        # Structure Check
        is_multi_ticker_level_1 = False
        if isinstance(data.columns, pd.MultiIndex):
             if 'Close' in data.columns.levels[0]: is_multi_ticker_level_1 = True
        
        # SPY
        spy_data = None
        if "SPY" in data.columns.levels[1 if is_multi_ticker_level_1 else 0]:
             if is_multi_ticker_level_1: spy_data = data.xs('SPY', axis=1, level=1)['Close']
             else: spy_data = data['SPY']['Close']
        else:
            spy_data = yf.download("SPY", period="2y", auto_adjust=True)['Close']
            
        # Sector Rankings
        print("\nAnalyzing Sectors...")
        sector_ranks, sector_list = self.analyze_sectors(sector_data, spy_data)
        print(f"Top Sector: {sector_list[0]['name']} (RS: {int(sector_list[0]['rs'])})")
        
        elite_setups = []
        radar_setups = []
        
        print(f"\nScanning 500+ stocks (v2.5)...")
        
        for t in tickers:
            try:
                if is_multi_ticker_level_1:
                    if t not in data.columns.levels[1]: continue
                    df = data.xs(t, axis=1, level=1).copy()
                else:
                    if t not in data.columns: continue
                    df = data[t].copy()
                
                df = df.dropna()
                if len(df) < 250: continue
                
                if not self.check_trend_template(df): continue
                
                is_vcp, pivot, dry_up = self.detect_vcp_strict(df)
                rs = self.calculate_rs_rating(df['Close'], spy_data)
                
                if is_vcp:
                    if rs < 65: continue
                    eps, rev, roe, sector = self.get_fundamentals_and_sector(t)
                    
                    # Sector Bonus
                    sec_rank = sector_ranks.get(sector, 99)
                    score = rs + (10 if sec_rank <= 3 else 0)
                    if dry_up: score += 10
                    
                    # Smart Sizing
                    atr = self.calculate_atr(df)
                    stop_dist = max(atr * 2, pivot * PARAM_STOP) # Use max of 2ATR or 5%
                    risk_per_share = stop_dist
                    shares = int(RISK_PER_TRADE / risk_per_share)
                    
                    stop_loss = pivot - stop_dist
                    target = pivot + (risk_per_share * 3) # 3R Target
                    
                    elite_setups.append(EliteSetup(
                        t, df['Close'].iloc[-1], pivot, stop_loss, target, score, rs, dry_up, 
                        eps, rev, roe, sector, sec_rank, atr, shares
                    ))
                else:
                     if rs > 80:
                        is_loose, pivot_loose = self.detect_vcp_loose(df)
                        if is_loose:
                             eps, rev, roe, sector = self.get_fundamentals_and_sector(t)
                             radar_setups.append({
                                 'ticker': t, 'price': df['Close'].iloc[-1], 
                                 'rs': rs, 'sector': sector, 'pivot': pivot_loose
                             })
            except: continue
                
        return elite_setups, radar_setups, sector_list

def main():
    print("==================================================")
    print("   ELITE TRADE v2.5 - SECTOR ALPHA EDITOR")
    print("==================================================")
    
    brain = EliteBrainV2_5()
    elite_setups, radar_setups, sector_list = brain.scan()
    
    # 0. Show Sectors
    print("\n[#] SECTOR LEADERBOARD")
    s_table = [[i+1, s['name'], int(s['rs'])] for i, s in enumerate(sector_list[:5])]
    print(tabulate(s_table, headers=["Rank", "Sector", "RS Score"], tablefmt="simple"))
    
    # 1. Elite Signals
    if elite_setups:
        elite_setups.sort(key=lambda x: x.score, reverse=True)
        top_picks = elite_setups[:5]
        
        table = []
        for s in top_picks:
            table.append([
                s.ticker, f"${s.pivot_price:.2f}", f"{s.smart_size}", f"${s.stop_loss:.2f}", 
                s.sector[:10], int(s.rs_rating), f"{s.eps_growth:.0f}%"
            ])
            
        print("\n[$$] MONEY MAKER SIGNALS (Smart Sized)")
        print(tabulate(table, headers=["Ticker", "Buy Stop", "Size (Qty)", "Stop Loss", "Sector", "RS", "EPS%"], tablefmt="fancy_grid"))
        
        print("\nEXECUTION PLAN (Professional):")
        for s in top_picks:
            print(f"[*] {s.ticker} ({s.sector})")
            print(f"    ORDER:   Buy {s.smart_size} shares STOP LIMIT @ ${s.pivot_price:.2f}")
            print(f"    RISK:    ${RISK_PER_TRADE} (Based on {s.smart_size} shares * ${(s.pivot_price - s.stop_loss):.2f} risk)")
    else:
        print("\n[$$] SIGNALS: None (Market is tough).")
        
    # 2. Radar
    if radar_setups:
        radar_setups.sort(key=lambda x: x['rs'], reverse=True)
        print(f"\n[O] RADAR WATCHLIST ({len(radar_setups)} Candidates)")
        r_table = [[r['ticker'], f"${r['price']:.2f}", r['sector'][:15], int(r['rs'])] for r in radar_setups[:10]]
        print(tabulate(r_table, headers=["Ticker", "Price", "Sector", "RS"], tablefmt="simple"))

if __name__ == "__main__":
    main()
