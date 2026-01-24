
import sys
import os
import pandas as pd
import yfinance as yf
# Import Titan Trade classes
from titan_trade import StrategyValidator, Optimizer, TitanBrain

def verify_upgrade_snp():
    print("=== S&P 500 TITAN UPGRADE VERIFICATION (Top 10 Stocks) ===")
    
    tickers = ["NVDA", "AAPL", "MSFT", "AMD", "TSLA", "GOOGL", "AMZN", "META", "NFLX", "JPM"]
    
    # Summary Stats
    wins_old = 0
    wins_new = 0
    score_improvement = 0.0
    
    print(f"{'Ticker':<8} | {'Old Score':<10} | {'New Score':<10} | {'Imp':<6} | {'Best Target'}")
    print("-" * 60)

    for t in tickers:
        try:
            # 1. Fetch Data
            df = yf.download(t, period="2y", auto_adjust=True, progress=False)
            if df.empty or len(df) < 200: continue
            
            # 2. RUN OLD LOGIC (Fixed Target = 3.5 ATR)
            val = StrategyValidator(df)
            res_base = val.backtest_breakout(target_mult=3.5)
            score_base = res_base['win_rate'] * res_base['pf']
            
            # 3. RUN NEW LOGIC (Auto-Tuned Target)
            opt = Optimizer(val)
            best_res, best_params = opt.tune_breakout()
            score_new = best_res['score']
            
            # Compare
            imp = score_new - score_base
            score_improvement += imp
            
            if score_base > 0: wins_old += 1
            if score_new > 0: wins_new += 1
            
            tgt_str = f"{best_params.get('target_mult')}x"
            
            print(f"{t:<8} | {score_base:<10.2f} | {score_new:<10.2f} | {imp:<+6.2f} | {tgt_str}")
            
        except Exception as e:
            print(f"{t:<8} | ERROR: {str(e)}")

    print("-" * 60)
    print(f"Total Score Improvement: {score_improvement:.2f}")
    
    if wins_new >= wins_old:
         print(f"✅ VERDICT: New Logic is BETTER or EQUAL. (Profitable setups found in {wins_new} vs {wins_old} stocks)")
    else:
         print(f"❌ VERDICT: New Logic performed worse.")

if __name__ == "__main__":
    verify_upgrade_snp()
