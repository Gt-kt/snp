import yfinance as yf
import pandas as pd
import os

CACHE_DIR = "cache_sp500_elite"
OHLCV_CACHE_FILE = os.path.join(CACHE_DIR, "sp500_ohlcv_bulk.parquet")

def check_trend_template_debug(df, ticker):
    try:
        c = df['Close'].iloc[-1]
        sma50 = df['Close'].rolling(50).mean().iloc[-1]
        sma150 = df['Close'].rolling(150).mean().iloc[-1]
        sma200 = df['Close'].rolling(200).mean().iloc[-1]
        low52 = df['Close'].rolling(252).min().iloc[-1]
        high52 = df['Close'].rolling(252).max().iloc[-1]
        
        print(f"\n--- {ticker} Trend Check ---")
        print(f"Price: {c:.2f}")
        print(f"SMA 50/150/200: {sma50:.2f} / {sma150:.2f} / {sma200:.2f}")
        print(f"52W Low/High:   {low52:.2f} / {high52:.2f}")
        
        c1 = c > sma50 > sma150 > sma200
        print(f"1. Order (P>50>150>200): {c1}")
        
        # 30% off lows
        c2 = c > low52 * 1.30
        print(f"2. > 30% Off Lows ({low52*1.3:.2f}): {c2}")
        
        c3 = c > high52 * 0.80
        print(f"3. Near Highs (<20%): {c3}")
        
        return c1 and c2 and c3
    except Exception as e:
        print(e)
        return False

def main():
    if not os.path.exists(OHLCV_CACHE_FILE):
        print("No cache.")
        return
        
    df = pd.read_parquet(OHLCV_CACHE_FILE)
    
def main():
    if not os.path.exists(OHLCV_CACHE_FILE):
        print("No cache.")
        return
        
    df = pd.read_parquet(OHLCV_CACHE_FILE)
    
    targets = ["NVDA", "TSLA", "MSFT", "PLTR", "CAT"]
    
    if "SPY" in df.columns.levels[0]:
        spy = df["SPY"]["Close"].dropna()
    else:
        print("SPY not in cache, fetching...")
        spy = yf.download("SPY", period="2y", auto_adjust=True)['Close']
    
    with open("debug_log.txt", "w") as f:
        f.write(f"DF Columns: {df.columns}\n")
        if isinstance(df.columns, pd.MultiIndex):
            f.write(f"Levels: {df.columns.levels}\n")
            
        for t in targets:
            if t in df.columns.levels[0]:
                sub = df[t].dropna()
                
                # Manual Trend Check
                c = sub['Close'].iloc[-1]
                sma50 = sub['Close'].rolling(50).mean().iloc[-1]
                sma150 = sub['Close'].rolling(150).mean().iloc[-1]
                sma200 = sub['Close'].rolling(200).mean().iloc[-1]
                low52 = sub['Close'].rolling(252).min().iloc[-1]
                high52 = sub['Close'].rolling(252).max().iloc[-1]
                
                f.write(f"\n--- {t} Trend Check ---\n")
                f.write(f"Price: {c:.2f}\n")
                f.write(f"SMA 50/150/200: {sma50:.2f} / {sma150:.2f} / {sma200:.2f}\n")
                f.write(f"52W Low/High:   {low52:.2f} / {high52:.2f}\n")
                
                c1 = c > sma50 > sma150 > sma200
                f.write(f"1. Order (P>50>150>200): {c1}\n")
                
                c2 = c > low52 * 1.30
                f.write(f"2. > 30% Off Lows ({low52*1.3:.2f}): {c2}\n")
                
                c3 = c > high52 * 0.80
                f.write(f"3. Near Highs (<20%): {c3}\n")
                
                # RS Check
                try:
                    stock_ret = sub['Close'].pct_change(126).iloc[-1]
                    spy_ret = spy.pct_change(126).iloc[-1]
                    excess = (stock_ret - spy_ret) * 100
                    rs = 50 + excess
                    f.write(f"RS Rating: {rs:.2f}\n")
                except Exception as e:
                    f.write(f"RS Calc Error: {e}\n")

if __name__ == "__main__":
    main()
