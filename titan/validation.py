"""
Titan Trade Validation Module
=============================
Strategy backtesting and validation logic.
"""

import numpy as np
import pandas as pd

from .utils import atr_series, calculate_rsi, calculate_obv, calculate_ema, multi_timeframe_momentum
from .config import (
    MAX_GAP_PCT,
    DEFAULT_BREAKOUT_STOP_ATR_MULT,
    DEFAULT_BREAKOUT_TARGET_ATR_MULT,
    DEFAULT_LEADER_BREAKOUT_STOP_ATR_MULT,
    DEFAULT_LEADER_BREAKOUT_TARGET_ATR_MULT,
    DEFAULT_DIP_STOP_ATR_MULT,
    DEFAULT_DIP_TARGET_ATR_MULT,
)


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
    
    def volume_accumulation_score(self, lookback=60):
        """Detect institutional accumulation via OBV trend and volume patterns.

        Returns 0-100 score where higher means stronger accumulation.
        Combines:
        - OBV trend direction (is smart money buying?)
        - Volume on up-days vs down-days ratio
        - Recent volume surge relative to baseline
        """
        df = self.df
        if len(df) < lookback + 20:
            return 50.0

        recent = df.iloc[-lookback:]
        close = recent['Close']
        volume = recent['Volume']

        score = 0.0

        # 1. OBV trend (0-40 points): Is OBV making higher highs?
        obv = calculate_obv(close, volume)
        obv_sma20 = obv.rolling(20).mean()
        if len(obv_sma20.dropna()) >= 5:
            obv_current = obv.iloc[-1]
            obv_ma = obv_sma20.iloc[-1]
            obv_ma_prev = obv_sma20.iloc[-20] if len(obv_sma20) >= 20 else obv_sma20.dropna().iloc[0]
            # OBV above its moving average = accumulation
            if obv_current > obv_ma:
                score += 20
            # OBV moving average rising = sustained accumulation
            if obv_ma > obv_ma_prev:
                score += 20

        # 2. Up-volume vs down-volume ratio (0-35 points)
        price_change = close.diff()
        up_vol = volume[price_change > 0].sum()
        down_vol = volume[price_change < 0].sum()
        if down_vol > 0:
            vol_ratio = up_vol / down_vol
            if vol_ratio >= 2.0:
                score += 35
            elif vol_ratio >= 1.5:
                score += 28
            elif vol_ratio >= 1.2:
                score += 20
            elif vol_ratio >= 1.0:
                score += 10
        elif up_vol > 0:
            # All volume on up days, no down days — strong accumulation
            score += 35

        # 3. Recent volume expansion (0-25 points): Are big players stepping in?
        vol_10d = volume.iloc[-10:].mean()
        vol_50d = volume.iloc[-50:].mean() if len(volume) >= 50 else volume.mean()
        if vol_50d > 0:
            expansion = vol_10d / vol_50d
            if expansion >= 1.5:
                score += 25
            elif expansion >= 1.2:
                score += 18
            elif expansion >= 1.0:
                score += 10
            elif expansion >= 0.8:
                score += 5

        return min(100.0, score)

    def momentum_composite_score(self):
        """Calculate multi-timeframe momentum composite.

        Stocks with strong momentum across multiple timeframes
        are more likely to continue trending up. Returns 0-100.
        """
        df = self.df
        if len(df) < 130:
            return 50.0

        close = df['Close']

        # Multi-timeframe momentum (weighted ROC)
        mtf = multi_timeframe_momentum(close)

        # Convert to 0-100 score
        # mtf typically ranges from -30 to +50 for S&P stocks
        score = max(0, min(100, 50 + mtf * 2))

        # Bonus: acceleration check - is momentum increasing?
        if len(close) >= 42:
            mom_recent = (close.iloc[-1] / close.iloc[-21] - 1) * 100
            mom_prior = (close.iloc[-21] / close.iloc[-42] - 1) * 100
            if mom_recent > mom_prior > 0:
                # Accelerating upward momentum
                score = min(100, score + 10)

        return float(score)

    def rs_percentile_rank(self, spy_df, all_returns=None, lookback=63):
        """Calculate relative strength percentile rank vs the market.

        When all_returns (dict of ticker->return) is provided, this
        returns the percentile rank among all stocks. Otherwise falls
        back to a score vs SPY alone. Top-quartile stocks (>75th pct)
        are the strongest candidates.
        """
        if len(self.df) < lookback:
            return 50.0

        stock_ret = (self.df['Close'].iloc[-1] / self.df['Close'].iloc[-lookback] - 1) * 100

        if all_returns is not None and len(all_returns) > 10:
            returns_list = sorted(all_returns.values())
            rank = sum(1 for r in returns_list if r <= stock_ret)
            percentile = (rank / len(returns_list)) * 100
            return float(percentile)

        # Fallback: score vs SPY
        return self.relative_strength_vs_spy(spy_df, lookback)

    def calculate_anchored_vwap(self, lookback=126):
        """Calculate Anchored VWAP from the highest volume day in the lookback period."""
        df = self.df
        if len(df) < lookback:
            return None
            
        recent_df = df.iloc[-lookback:]
        max_vol_idx = recent_df['Volume'].idxmax()
        
        post_anchor_df = df.loc[max_vol_idx:].copy()
        if len(post_anchor_df) == 0:
            return None
            
        typical_price = (post_anchor_df['High'] + post_anchor_df['Low'] + post_anchor_df['Close']) / 3
        tp_v = typical_price * post_anchor_df['Volume']
        
        cum_tp_v = tp_v.cumsum()
        cum_vol = post_anchor_df['Volume'].cumsum()
        
        if cum_vol.iloc[-1] == 0:
            return None
            
        avwap = cum_tp_v / cum_vol
        return float(avwap.iloc[-1])

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

            # Gap through stop
            if day_open <= stop_curr:
                fill_price = day_open * (1 - slippage_pct)
                return (fill_price - actual_entry) / actual_entry

            # Gap through target
            if day_open >= target:
                return (day_open - actual_entry) / actual_entry

            highest_since_entry = max(highest_since_entry, day_high)

            # Trailing stop logic
            if trail_risk is not None and trail_risk > 0:
                if highest_since_entry >= actual_entry + (trail_risk * trail_r1):
                    stop_curr = max(stop_curr, actual_entry)
                if highest_since_entry >= actual_entry + (trail_risk * trail_r2):
                    stop_curr = max(stop_curr, actual_entry + trail_risk)
                if highest_since_entry >= actual_entry + (trail_risk * trail_r3):
                    stop_curr = max(stop_curr, actual_entry + trail_risk * 2)

            # Intraday stop
            if day_low <= stop_curr:
                fill_price = stop_curr * (1 - slippage_pct * 0.5)
                return (fill_price - actual_entry) / actual_entry

            # Intraday target
            if day_high >= target:
                fill_price = target * (1 - slippage_pct * 0.3)
                return (fill_price - actual_entry) / actual_entry

            # End of holding period
            if j == end_idx - 1:
                fill_price = day_close * (1 - slippage_pct)
                return (fill_price - actual_entry) / actual_entry

        return 0.0

    def backtest_breakout(self, days=750, depth=0.22, vol_mult=1.5,
                         target_mult=DEFAULT_BREAKOUT_TARGET_ATR_MULT,
                         stop_mult=DEFAULT_BREAKOUT_STOP_ATR_MULT, return_trades=False,
                         min_entry_idx=None):
        """Fast simulation of VCP Breakouts.

        No lookahead bias: day i = base detected, day i+1 = breakout
        confirmed at close, day i+2 = entry at open. All confirmation
        uses data available before the entry decision.

        Args:
            min_entry_idx: If set, only consider entries at or after this
                index into the sliced dataframe. Used for OOS validation.
        """
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
        vol_sma20 = df['Volume'].rolling(20).mean().values

        trades = []

        entry_start = max(60, min_entry_idx) if min_entry_idx is not None else 60

        # Need i+2 for entry, so stop 2 before end
        for i in range(entry_start, len(df)-2):
            if not (closes[i] > sma50[i] > sma200[i]):
                continue
            if i < 70:
                continue
            if not (sma50[i] > sma50[i-10]):
                continue
            if sma200[i] <= sma200[i-20]:
                continue
            if atr[i] and closes[i] > 0 and (atr[i] / closes[i]) > 0.12:
                continue
            high_252 = np.max(highs[max(0, i-251):i+1])
            if high_252 <= 0 or closes[i] < high_252 * 0.92:
                continue

            h_handle = np.max(highs[i-15:i+1])
            l_handle = np.min(lows[i-15:i+1])
            curr_c = closes[i]

            d = (h_handle - l_handle) / h_handle
            if d > depth:
                continue

            # Volatility contraction: last 5 days range must be tighter
            # than the full 15-day handle range (VCP signature)
            range_5 = (np.max(highs[i-5:i+1]) - np.min(lows[i-5:i+1])) / max(curr_c, 1e-9)
            range_15 = (h_handle - l_handle) / max(curr_c, 1e-9)
            if range_5 > 0.10:
                continue
            if range_15 > 0 and range_5 > range_15 * 0.85:
                continue

            # Close near handle high
            if (h_handle - curr_c) / h_handle > 0.08:
                continue

            # Volume should be drying up in the base (contraction)
            if not np.isnan(vol_sma[i]):
                base_vol = np.mean(volumes[i-15:i+1])
                if base_vol > (vol_sma[i] * 1.3):
                    continue

            atr_val = atr[i] if i < len(atr) and not np.isnan(atr[i]) else (curr_c * 0.02)
            pivot = h_handle + (atr_val * 0.05)

            # --- Day i+1: breakout confirmation day ---
            # All checks below use end-of-day i+1 data. This is the
            # CONFIRMATION day — we observe the breakout after close,
            # then enter the next morning at day i+2 open.
            bo_h = highs[i+1]
            bo_l = lows[i+1]
            bo_c = closes[i+1]
            bo_v = volumes[i+1]
            bo_vol_sma = vol_sma[i+1] if i+1 < len(vol_sma) else np.nan
            bo_vol_sma20 = vol_sma20[i+1] if i+1 < len(vol_sma20) else np.nan

            if bo_h > pivot:
                # Volume surge confirmation
                vol_ref = bo_vol_sma20 if not np.isnan(bo_vol_sma20) else bo_vol_sma
                if not np.isnan(vol_ref) and bo_v < (vol_ref * max(vol_mult, 1.5)):
                    continue
                # Close above pivot (confirmed at end of day)
                if bo_c < (pivot * 0.99):
                    continue
                day_range = max(bo_h - bo_l, 1e-9)
                close_pos = (bo_c - bo_l) / day_range
                # Close in upper half of the range with some authority.
                if close_pos < 0.55:
                    continue

                # --- Day i+2: ENTRY at open (no lookahead) ---
                buy_price = opens[i+2]
                # Reject if gap-up too large (>3% above pivot = chasing)
                if buy_price > pivot * 1.03:
                    continue
                # Reject if gap-down below pivot (breakout failed overnight)
                if buy_price < pivot * 0.97:
                    continue

                atr_val = atr[i] if i < len(atr) and not np.isnan(atr[i]) else (buy_price * 0.02)
                stop_loss = buy_price - (atr_val * stop_mult)
                target = buy_price + (atr_val * target_mult)
                risk = atr_val * stop_mult

                outcome_pct = self._simulate_trade(
                    buy_price, stop_loss, target,
                    (closes, highs, lows, opens),
                    i + 2, max_hold=7, trail_risk=risk, slippage_pct=0.003
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

    def backtest_leader_breakout(
        self,
        days=750,
        stop_mult=DEFAULT_LEADER_BREAKOUT_STOP_ATR_MULT,
        target_mult=DEFAULT_LEADER_BREAKOUT_TARGET_ATR_MULT,
        return_trades=False,
        min_entry_idx=None,
    ):
        """Validate trend-continuation breakouts for leaders near highs.

        This is broader than the classic VCP breakout test. It is intended
        for persistent leaders that keep offering tight continuation pivots
        but do not generate many textbook VCP handles.
        """
        df = self.df.iloc[-days:].copy()
        if len(df) < 140:
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
        ema21 = calculate_ema(df['Close'], 21).values
        atr = atr_series(df).values
        vol_sma20 = df['Volume'].rolling(20).mean().values

        trades = []
        entry_start = max(70, min_entry_idx) if min_entry_idx is not None else 70

        for i in range(entry_start, len(df) - 2):
            if not (closes[i] > sma50[i] > sma200[i]):
                continue
            if sma50[i] <= sma50[i - 10]:
                continue
            if sma200[i] <= sma200[i - 20]:
                continue
            if closes[i] < ema21[i] * 0.985:
                continue

            atr_val = atr[i] if i < len(atr) and not np.isnan(atr[i]) else (closes[i] * 0.02)
            if atr_val <= 0 or (atr_val / max(closes[i], 1e-9)) > 0.12:
                continue

            high_63 = np.max(highs[max(0, i - 62):i + 1])
            if high_63 <= 0 or (high_63 - closes[i]) / high_63 > 0.12:
                continue

            range_10 = (np.max(highs[i - 9:i + 1]) - np.min(lows[i - 9:i + 1])) / max(closes[i], 1e-9)
            range_30 = (np.max(highs[i - 29:i + 1]) - np.min(lows[i - 29:i + 1])) / max(closes[i], 1e-9)
            if range_10 > 0.12:
                continue
            if range_30 > 0 and range_10 > range_30 * 0.95:
                continue

            if not np.isnan(vol_sma20[i]) and volumes[i] > vol_sma20[i] * 2.0:
                continue

            pivot = np.max(highs[i - 10:i + 1]) + (atr_val * 0.03)
            bo_h = highs[i + 1]
            bo_l = lows[i + 1]
            bo_c = closes[i + 1]
            bo_v = volumes[i + 1]
            vol_ref = vol_sma20[i + 1] if i + 1 < len(vol_sma20) else np.nan

            if bo_h <= pivot:
                continue
            if not np.isnan(vol_ref) and bo_v < (vol_ref * 0.90):
                continue
            if bo_c < (pivot * 0.992):
                continue

            day_range = max(bo_h - bo_l, 1e-9)
            close_pos = (bo_c - bo_l) / day_range
            if close_pos < 0.40:
                continue

            buy_price = opens[i + 2]
            if buy_price <= 0:
                continue
            if buy_price > pivot * 1.04:
                continue
            if buy_price < pivot * 0.965:
                continue

            stop_loss = buy_price - (atr_val * stop_mult)
            target = buy_price + (atr_val * target_mult)
            risk = buy_price - stop_loss
            if risk <= 0:
                continue

            outcome_pct = self._simulate_trade(
                buy_price, stop_loss, target,
                (closes, highs, lows, opens),
                i + 2,
                max_hold=7,
                trail_risk=risk,
                trail_r1=1.0,
                trail_r2=1.8,
                trail_r3=2.6,
                slippage_pct=0.002,
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

    def backtest_dip(self, days=750, stop_mult=DEFAULT_DIP_STOP_ATR_MULT,
                     target_mult=DEFAULT_DIP_TARGET_ATR_MULT, return_trades=False,
                     min_entry_idx=None):
        """Fast simulation of Dip Buys (SMA50 support).

        Fixed issues from audit:
        - Require price ABOVE SMA200 (not 3% below — that catches falling knives)
        - Require SMA200 rising (filters out bear market rallies)
        - Added trailing stop to lock in profits on winning dip trades
        - Added bounce candle quality check (close in upper half of range)

        Args:
            min_entry_idx: If set, only consider entries at or after this
                index into the sliced dataframe. Used for OOS validation.
        """
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

        entry_start = max(50, min_entry_idx) if min_entry_idx is not None else 50

        for i in range(entry_start, len(df)-5):
            # FIX: Price must be ABOVE 200 SMA (no 3% below allowance)
            if closes.iloc[i] < sma200.iloc[i]:
                continue

            # 50 SMA must be rising
            if sma50.iloc[i] <= sma50.iloc[i-10]:
                continue

            # FIX: 200 SMA must be at least flat (not falling)
            # This filters out bear market rallies where SMA50 bounces
            # temporarily while the long-term trend is still down
            if i >= 20 and sma200.iloc[i] < sma200.iloc[i-20]:
                continue

            # Distance from 50 SMA
            dist = (lows.iloc[i] - sma50.iloc[i]) / sma50.iloc[i]

            if -0.03 < dist < 0.03:
                # RSI filter
                if rsi14.iloc[i] < 40:
                    continue
                # Volume spike filter (reject panic selling)
                if not np.isnan(vol_sma.iloc[i]) and volumes.iloc[i] > vol_sma.iloc[i] * 1.5:
                    continue

                # FIX: Bounce candle quality — close should be in upper
                # half of the day range (shows buyers stepped in)
                day_range = highs.iloc[i] - lows.iloc[i]
                if day_range > 0:
                    close_position = (closes.iloc[i] - lows.iloc[i]) / day_range
                    if close_position < 0.40:
                        continue
                # Still reject big red days
                daily_change = (closes.iloc[i] - opens.iloc[i]) / opens.iloc[i]
                if daily_change < -0.015:
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
                if (atr_val / buy_price) > 0.08:
                    continue

                stop = buy_price - (atr_val * stop_mult)
                target = buy_price + (atr_val * target_mult)
                # FIX: Add trailing stop to dip buys — the biggest
                # profitability leak was letting winners run back to
                # breakeven. Trail at 1R = breakeven, 2R = lock +1R.
                risk = atr_val * stop_mult

                outcome_pct = self._simulate_trade(
                    buy_price, stop, target,
                    (closes.values, highs.values, lows.values, opens.values),
                    i + 1, max_hold=7, trail_risk=risk, slippage_pct=0.001
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
    """Analyzes trend quality using technical factors.

    Enhanced with multi-timeframe momentum, volume accumulation,
    and relative strength for better stock selection.
    """

    @staticmethod
    def analyze(df, backtest_res=None, accumulation_score=None,
                momentum_score=None, rs_percentile=None):
        if len(df) < 200:
            return {"trend_score": 0, "trend_grade": "F", "factors": {}}

        c = df['Close']
        h = df['High']
        v = df['Volume']

        factors = {}
        score = 0

        # MA Alignment (0-20 pts)
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

        # Distance from 52-week high (0-15 pts)
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

        # Volume trend (0-10 pts)
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

        # Multi-timeframe momentum (0-15 pts) - enhanced
        mtf_raw = multi_timeframe_momentum(c)
        if mtf_raw > 15:
            mtf_score = 15
        elif mtf_raw > 8:
            mtf_score = 12
        elif mtf_raw > 3:
            mtf_score = 8
        elif mtf_raw > 0:
            mtf_score = 4
        else:
            mtf_score = 0

        factors["momentum"] = mtf_score
        score += mtf_score

        # Volume accumulation (0-15 pts) - NEW
        if accumulation_score is not None:
            if accumulation_score >= 75:
                accum_pts = 15
            elif accumulation_score >= 55:
                accum_pts = 12
            elif accumulation_score >= 40:
                accum_pts = 8
            elif accumulation_score >= 25:
                accum_pts = 4
            else:
                accum_pts = 0
            factors["accumulation"] = accum_pts
            score += accum_pts

        # Relative strength rank (0-15 pts) - NEW
        if rs_percentile is not None:
            if rs_percentile >= 85:
                rs_pts = 15
            elif rs_percentile >= 70:
                rs_pts = 12
            elif rs_percentile >= 55:
                rs_pts = 8
            elif rs_percentile >= 40:
                rs_pts = 4
            else:
                rs_pts = 0
            factors["relative_strength"] = rs_pts
            score += rs_pts

        # Momentum acceleration bonus (0-5 pts) - NEW
        if momentum_score is not None and momentum_score >= 70:
            factors["momentum_accel"] = 5
            score += 5

        # Grade - recalibrated for expanded scoring range (max ~100)
        if score >= 65:
            grade = "A"
        elif score >= 50:
            grade = "B"
        elif score >= 35:
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
