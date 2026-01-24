import yfinance as yf
import pandas as pd
import numpy as np
from backtesting import Backtest, Strategy
from backtesting.lib import crossover

import warnings
warnings.filterwarnings("ignore")

def SMA(values, n):
    """
    Return simple moving average of `values`, at
    each step taking into account `n` previous values.
    """
    return pd.Series(values).rolling(n).mean()

def RSI(array, n):
    """Relative Strength Index"""
    # Approximate; good enough
    gain = pd.Series(array).diff()
    loss = gain.copy()
    gain[gain < 0] = 0
    loss[loss > 0] = 0
    rs = gain.ewm(span=n).mean() / loss.abs().ewm(span=n).mean()
    return 100 - 100 / (1 + rs)

class Snp500Strategy(Strategy):
    n_sma = 200
    n_rsi = 4
    rsi_lower = 25
    rsi_upper = 75
    
    stop_loss_atr = 2.0  # ATR multiplier for SL

    def init(self):
        # Compute indicators
        self.sma = self.I(SMA, self.data.Close, self.n_sma)
        self.rsi = self.I(RSI, self.data.Close, self.n_rsi)

    def next(self):
        price = self.data.Close[-1]
        
        # Check if we are in an uptrend (Price > SMA200)
        if len(self.sma) > self.n_sma and price > self.sma[-1]:
            # Entry Condition: RSI < Lower Threshold (Dip)
            if self.rsi[-1] < self.rsi_lower and not self.position:
                self.buy()
        
        # Exit Condition
        if self.position:
            # Sell if RSI is high (Mean Reversion)
            if self.rsi[-1] > self.rsi_upper:
                self.position.close()
            # Stop Loss (Fixed % or simple logic for now, Backtesting.py handles SL/TP in buy() but let's keep it manual or simple first)
            # basic safety net: if price drops 5% below entry, close
            # if price < self.position.pl... (Logic needs to be careful)
            pass

def get_data(ticker="^GSPC", start="2015-01-01", end="2024-01-01"):
    print(f"Downloading data for {ticker}...")
    data = yf.download(ticker, start=start, end=end, progress=False)
    
    # Fix MultiIndex columns (common in new yfinance)
    if isinstance(data.columns, pd.MultiIndex):
        try:
            data.columns = data.columns.droplevel(1) 
        except IndexError:
            pass
            
    # Fallback/Cleanup for weird yfinance formats
    if 'Close' not in data.columns and 'Adj Close' in data.columns:
         data['Close'] = data['Adj Close']

    # Final check for MultiIndex
    if isinstance(data.columns, pd.MultiIndex):
         data.columns = [col[0] for col in data.columns]
    
    # Ensure required columns exist for Backtesting
    required = ['Open', 'High', 'Low', 'Close', 'Volume']
    missing = [c for c in required if c not in data.columns]
    if missing:
        print(f"WARNING: Missing columns: {missing}")
    
    return data

def run_backtest():
    data = get_data()
    
    if data.empty:
        print("No data found. Check internet or ticker.")
        return

    bt = Backtest(data, Snp500Strategy, cash=100000, commission=.002)
    stats = bt.run()
    
    print("\n--- Backtest Results ---")
    print(stats)
    print("\n--- Strategy Parameters ---")
    print(f"SMA: {Snp500Strategy.n_sma}, RSI: {Snp500Strategy.n_rsi} ({Snp500Strategy.rsi_lower}/{Snp500Strategy.rsi_upper})")
    
    # Check simple condition
    win_rate = stats['Win Rate [%]']
    print(f"\nWin Rate: {win_rate:.2f}%")
    
    if win_rate > 50:
        print("SUCCESS: Win Rate > 50%")
    else:
        print("WARNING: Win Rate <= 50%. Optimization needed.")
        
    # Optional: Optimize
    # stats = bt.optimize(n_rsi=range(2, 15, 2), rsi_lower=range(10, 40, 5), maximize='Win Rate [%]')
    # print(stats)

if __name__ == "__main__":
    run_backtest()
