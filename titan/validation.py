"""
Titan Trade Validation Module
=============================
Strategy backtesting and validation logic.
"""

import numpy as np
import pandas as pd

from .utils import atr_series, calculate_rsi
from .config import MAX_GAP_PCT


class StrategyValidator:
    """Backtests strategy logic on a specific stock."""
    
    def __init__(self, df):
        self.df = df
    
    def check_gap_risk(self, max_gap_pct=MAX_GAP_PCT, lookback=60):
        """Filter out stocks with history of large overnight gaps."""
        df = self.df
        if len(df) < lookback + 1:
            return True
        
        opens = df['Open'].iloc[-lookback:]
        prev_close = df['Close'].shift(1).iloc[-lookback:]
        gaps = abs(opens - prev_close) / (prev_close + 1e-9)
        large_gap_count = (gaps > max_gap_pct).sum()
        return large_gap_count <= 3
    
    def relative_strength_vs_spy(self, spy_df, lookback=60):
        """Calculate Relative Strength vs SPY."""
        if len(self.df) < lookback or len(spy_df) < lookback:
            return 50.0
        
        stock_ret = (self.df['Close'].iloc[-1] / self.df['Close'].iloc[-lookback] - 1) * 100
        spy_ret = (spy_df['Close'].iloc[-1] / spy_df['Close'].iloc[-lookback] - 1) * 100
        rs_diff = stock_ret - spy_ret
        rs_score = max(0, min(100, 50 + (rs_diff * 5)))
        return rs_score
    
    def is_blue_sky_breakout(self, lookback=252):
        """Check if price is within 5% of 52-week high."""
        if len(self.df) < lookback:
            return False
        high_52w = self.df['High'].iloc[-lookback:].max()
        current = self.df['Close'].iloc[-1]
        return current >= high_52w * 0.95
    
    def _simulate_trade(self, entry, stop, target, ohlc_data, start_idx, 
                       max_hold=10, trail_risk=None, trail_r1=1.0, 
                       trail_r2=2.0, trail_r3=3.0, slippage_pct=0.003):
        """Simulate a trade with realistic execution model."""
        closes, highs, lows, opens = ohlc_data
        stop_curr = stop
        highest_since_entry = entry
        end_idx = min(start_idx + max_hold, len(closes))
        actual_entry = entry * (1 + slippage_pct)

        for j in range(start_idx, end_idx):
            day_open = opens[j] if j < len(opens) else closes[j-1] if j > 0 else entry
            day_high = highs[j]
            day_low = lows[j]
            day_close = closes[j]
            
            highest_since_entry = max(highest_since_entry, day_high)
            
            # Trailing stop logic
            if trail_risk is not None and trail_risk > 0:
                if highest_since_entry >= actual_entry + (trail_risk * trail_r1):
                    stop_curr = max(stop_curr, actual_entry)
                if highest_since_entry >= actual_entry + (trail_risk * trail_r2):
                    stop_curr = max(stop_curr, actual_entry + trail_risk)
                if highest_since_entry >= actual_entry + (trail_risk * trail_r3):
                    stop_curr = max(stop_curr, actual_entry + trail_risk * 2)
            
            # Gap through stop
            if day_open <= stop_curr:
                fill_price = day_open * (1 - slippage_pct)
                return (fill_price - actual_entry) / actual_entry
            
            # Intraday stop
            if day_low <= stop_curr:
                fill_price = stop_curr * (1 - slippage_pct * 0.5)
                return (fill_price - actual_entry) / actual_entry
            
            # Gap through target
            if day_open >= target:
                return (day_open - actual_entry) / actual_entry
            
            # Intraday target
            if day_high >= target:
                fill_price = target * (1 - slippage_pct * 0.3)
                return (fill_price - actual_entry) / actual_entry
            
            # End of holding period
            if j == end_idx - 1:
                fill_price = day_close * (1 - slippage_pct)
                return (fill_price - actual_entry) / actual_entry

        return 0.0

    def backtest_breakout(self, days=500, depth=0.20, vol_mult=1.2, 
                         target_mult=2.5, stop_mult=2.0, return_trades=False):
        """Fast simulation of VCP Breakouts."""
        df = self.df.iloc[-days:].copy()
        if len(df) < 100:
            base = {'win_rate': 0, 'pf': 0, 'trades': 0}
            if return_trades:
                base['trades_list'] = []
            return base
        
        closes = df['Close'].values
        highs = df['High'].values
        lows = df['Low'].values
        opens = df['Open'].values
        volumes = df['Volume'].values
        
        sma50 = df['Close'].rolling(50).mean().values
        sma200 = df['Close'].rolling(200).mean().values
        atr = atr_series(df).values
        vol_sma = df['Volume'].rolling(50).mean().values
        
        trades = []
        
        for i in range(60, len(df)-1):
            if not (closes[i] > sma50[i] > sma200[i]):
                continue
            if i < 70:
                continue
            if not (sma50[i] > sma50[i-10] and sma200[i] > sma200[i-10]):
                continue
            if atr[i] and closes[i] > 0 and (atr[i] / closes[i]) > 0.12:
                continue
            
            h_handle = np.max(highs[i-15:i+1])
            l_handle = np.min(lows[i-15:i+1])
            curr_c = closes[i]
            
            d = (h_handle - l_handle) / h_handle
            if d > depth:
                continue

            range_5 = (np.max(highs[i-5:i+1]) - np.min(lows[i-5:i+1])) / max(curr_c, 1e-9)
            if range_5 > 0.08:
                continue
            
            if (h_handle - curr_c) / h_handle > 0.06:
                continue
            
            if not np.isnan(vol_sma[i]):
                base_vol = np.mean(volumes[i-15:i+1])
                if base_vol > (vol_sma[i] * 1.15):
                    continue
            
            atr_val = atr[i] if i < len(atr) and not np.isnan(atr[i]) else (curr_c * 0.02)
            pivot = h_handle + (atr_val * 0.05)
            
            next_h = highs[i+1]
            next_l = lows[i+1]
            next_o = opens[i+1]
            next_c = closes[i+1]
            next_v = volumes[i+1]
            next_vol_sma = vol_sma[i+1] if i+1 < len(vol_sma) else np.nan
            
            if next_h > pivot:
                if not np.isnan(next_vol_sma) and next_v < (next_vol_sma * vol_mult):
                    continue
                if next_c < (pivot * 0.995):
                    continue
                day_range = max(next_h - next_l, 1e-9)
                close_pos = (next_c - next_l) / day_range
                if close_pos < 0.5:
                    continue
                
                buy_price = max(pivot, next_o)
                atr_val = atr[i] if i < len(atr) and not np.isnan(atr[i]) else (buy_price * 0.02)
                stop_loss = buy_price - (atr_val * stop_mult)
                target = buy_price + (atr_val * target_mult)
                risk = atr_val * stop_mult

                outcome_pct = self._simulate_trade(
                    buy_price, stop_loss, target,
                    (closes, highs, lows, opens),
                    i + 2, max_hold=10, trail_risk=risk, slippage_pct=0.003
                )
                trades.append(outcome_pct)
        
        if not trades:
            base = {'win_rate': 0, 'pf': 0, 'trades': 0}
            if return_trades:
                base['trades_list'] = []
            return base
        
        wins = [t for t in trades if t > 0]
        losses = [t for t in trades if t <= 0]
        
        win_rate = float(len(wins) / len(trades) * 100)
        gross_win = float(sum(wins))
        gross_loss = float(abs(sum(losses)))
        pf = float(gross_win / gross_loss if gross_loss > 0 else (100.0 if gross_win > 0 else 0))
        
        res = {'win_rate': win_rate, 'pf': pf, 'trades': len(trades)}
        if return_trades:
            res['trades_list'] = trades
        return res

    def backtest_dip(self, days=500, stop_mult=2.0, target_mult=3.5, return_trades=False):
        """Fast simulation of Dip Buys (SMA50 support)."""
        df = self.df.iloc[-days:].copy()
        if len(df) < 100:
            base = {'win_rate': 0, 'pf': 0, 'trades': 0}
            if return_trades:
                base['trades_list'] = []
            return base
        
        sma50 = df['Close'].rolling(50).mean()
        sma200 = df['Close'].rolling(200).mean()
        rsi14 = calculate_rsi(df['Close'])
        closes = df['Close']
        opens = df['Open']
        lows = df['Low']
        highs = df['High']
        volumes = df['Volume']
        vol_sma = df['Volume'].rolling(20).mean()
        atr = atr_series(df)
        
        trades = []
        
        for i in range(50, len(df)-5):
            if closes.iloc[i] > sma50.iloc[i] and sma50.iloc[i] > sma200.iloc[i]:
                if sma50.iloc[i] <= sma50.iloc[i-10]:
                    continue
                
                dist = (lows.iloc[i] - sma50.iloc[i]) / sma50.iloc[i]
                
                if -0.02 < dist < 0.02:
                    if rsi14.iloc[i] < 45:
                        continue
                    if not np.isnan(vol_sma.iloc[i]) and volumes.iloc[i] > vol_sma.iloc[i] * 1.3:
                        continue
                    if closes.iloc[i] <= opens.iloc[i]:
                        continue
                    if i + 1 >= len(df):
                        continue
                    
                    buy_price = opens.iloc[i + 1]
                    if buy_price <= 0:
                        continue
                    if abs(buy_price - closes.iloc[i]) / closes.iloc[i] > 0.05:
                        continue
                    
                    atr_val = atr.iloc[i]
                    if np.isnan(atr_val) or atr_val <= 0:
                        atr_val = buy_price * 0.02
                    if (atr_val / buy_price) > 0.07:
                        continue
                    
                    stop = buy_price - (atr_val * stop_mult)
                    target = buy_price + (atr_val * target_mult)

                    outcome_pct = self._simulate_trade(
                        buy_price, stop, target,
                        (closes.values, highs.values, lows.values, opens.values),
                        i + 2, max_hold=10, trail_risk=None, slippage_pct=0.001
                    )
                    trades.append(outcome_pct)
                    
        if not trades:
            base = {'win_rate': 0, 'pf': 0, 'trades': 0}
            if return_trades:
                base['trades_list'] = []
            return base
        
        wins = [t for t in trades if t > 0]
        losses = [t for t in trades if t <= 0]
        win_rate = len(wins) / len(trades) * 100
        pf = sum(wins)/abs(sum(losses)) if sum(losses) != 0 else (100.0 if sum(wins) > 0 else 0)
        
        res = {'win_rate': win_rate, 'pf': pf, 'trades': len(trades)}
        if return_trades:
            res['trades_list'] = trades
        return res


class TrendQualityAnalyzer:
    """Analyzes trend quality using technical factors."""
    
    @staticmethod
    def analyze(df, backtest_res=None):
        if len(df) < 200:
            return {"trend_score": 0, "trend_grade": "F", "factors": {}}
        
        c = df['Close']
        h = df['High']
        v = df['Volume']
        
        factors = {}
        score = 0
        
        # MA Alignment
        sma20 = c.rolling(20).mean().iloc[-1]
        sma50 = c.rolling(50).mean().iloc[-1]
        sma200 = c.rolling(200).mean().iloc[-1]
        curr = c.iloc[-1]
        
        if curr > sma20 > sma50 > sma200:
            ma_alignment = 20
        elif curr > sma50 > sma200:
            ma_alignment = 15
        elif curr > sma200:
            ma_alignment = 10
        elif curr > sma50:
            ma_alignment = 5
        else:
            ma_alignment = 0
        
        factors["ma_alignment"] = ma_alignment
        score += ma_alignment
        
        # Distance from 52-week high
        high_52w = h.iloc[-252:].max() if len(df) >= 252 else h.max()
        pct_from_high = (curr / high_52w - 1) * 100
        
        if pct_from_high > -3:
            proximity_score = 15
        elif pct_from_high > -7:
            proximity_score = 12
        elif pct_from_high > -15:
            proximity_score = 8
        elif pct_from_high > -25:
            proximity_score = 4
        else:
            proximity_score = 0
        
        factors["proximity_to_high"] = proximity_score
        score += proximity_score
        
        # Volume trend
        vol_20d = v.iloc[-20:].mean()
        vol_50d = v.iloc[-50:].mean()
        vol_ratio = vol_20d / (vol_50d + 1e-9)
        
        if vol_ratio > 1.3:
            vol_score = 10
        elif vol_ratio > 1.0:
            vol_score = 7
        elif vol_ratio > 0.7:
            vol_score = 4
        else:
            vol_score = 0
        
        factors["volume_trend"] = vol_score
        score += vol_score
        
        # Momentum
        ret_20d = (curr / c.iloc[-21] - 1) * 100 if len(df) >= 21 else 0
        ret_60d = (curr / c.iloc[-61] - 1) * 100 if len(df) >= 61 else 0
        
        if ret_20d > 5 and ret_60d > 10:
            momentum_score = 15
        elif ret_20d > 2 and ret_60d > 5:
            momentum_score = 12
        elif ret_20d > 0 and ret_60d > 0:
            momentum_score = 8
        elif ret_20d > -5 and ret_60d > -10:
            momentum_score = 4
        else:
            momentum_score = 0
        
        factors["momentum"] = momentum_score
        score += momentum_score
        
        # Grade
        if score >= 50:
            grade = "A"
        elif score >= 40:
            grade = "B"
        elif score >= 30:
            grade = "C"
        elif score >= 20:
            grade = "D"
        else:
            grade = "F"
        
        return {"trend_score": score, "trend_grade": grade, "factors": factors}


class Optimizer:
    """Parameter optimization for strategies."""
    
    def __init__(self, validator):
        self.validator = validator
        
    def tune_breakout(self):
        """Find best breakout parameters."""
        best_res = {'win_rate': 0, 'pf': 0, 'score': 0}
        best_params = {'depth': 0.18, 'target_mult': 3.0}
        
        for d in [0.15, 0.18, 0.22]:
            for t_mult in [2.5, 3.0, 3.5, 4.0]:
                res = self.validator.backtest_breakout(depth=d, target_mult=t_mult)
                if res['trades'] < 15:
                    continue
                score = res['pf'] * res['win_rate']
                if score > best_res['score']:
                    best_res = res
                    best_res['score'] = score
                    best_params = {'depth': d, 'target_mult': t_mult}
                
        return best_res, best_params
