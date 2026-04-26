"""
Titan Signal Detector — Multi-Signal Detection for S&P 500
===========================================================
Ported from the proven KOSPI architecture. Detects 11 signal types
that catch stocks at different stages of their move cycle.

Signal priority (checked in this order):
  PRE_BREAK  — tight range near highs, volatility contracting
  VCP        — true volatility contraction pattern
  ACCUM      — volume building while price stays flat
  EARLY_REV  — RSI turning up from oversold
  PULLBACK   — healthy pullback to SMA20 in an uptrend
  REL_STR    — relative strength leader near highs
  MOM_RANK   — top momentum + RS vs SPY
  MOMENTUM   — steady uptrend with volume
  BREAKOUT   — price at 20-day high with volume
  DIP_BUY    — oversold bounce above SMA50
  VOL_SPIKE  — unusual volume without wild price swing
"""

import math
import pandas as pd
import numpy as np

# ---------------------------------------------------------------------------
# Early signal types — these qualify for stalking / pre-move radar
# ---------------------------------------------------------------------------
EARLY_SIGNAL_TYPES = frozenset({
    "PRE_BREAK", "VCP", "ACCUM", "EARLY_REV", "PULLBACK", "REL_STR", "MOM_RANK",
})

BREAKOUT_SIGNAL_TYPES = frozenset({"BREAKOUT", "VOL_SPIKE"})

ALL_SIGNAL_TYPES = (
    "PRE_BREAK", "VCP", "ACCUM", "EARLY_REV", "PULLBACK",
    "REL_STR", "MOM_RANK", "MOMENTUM", "BREAKOUT", "DIP_BUY", "VOL_SPIKE",
)

# Signal freshness — only alert on setups younger than this many bars
SIGNAL_FRESHNESS_BARS = 2

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe(val, default=0.0):
    if val is None:
        return default
    try:
        f = float(val)
        return default if math.isnan(f) else f
    except (TypeError, ValueError):
        return default


def _compute_technicals(df: pd.DataFrame) -> dict:
    """Extract all the technical values we need from the last bar."""
    if df is None or df.empty or len(df) < 60:
        return {}

    last = df.iloc[-1]
    prev = df.iloc[-2] if len(df) >= 2 else last

    close = _safe(last.get("Close"))
    if close <= 0:
        return {}

    open_price = _safe(last.get("Open"), close)
    prev_close = _safe(prev.get("Close"), close)
    volume = _safe(last.get("Volume"))

    # Use previous bar's volume average to avoid incomplete intraday bar
    # (e.g., pre-market or mid-day scan where today's volume is partial)
    avg_volume = _safe(df["Volume"].iloc[:-1].rolling(20).mean().iloc[-1], volume) or 1.0

    # If today's volume is suspiciously low (<30% of avg), the bar is likely
    # incomplete. Use the previous complete bar's volume for ratio calculation.
    if avg_volume > 0 and volume < avg_volume * 0.30 and len(df) >= 3:
        volume_for_ratio = _safe(df["Volume"].iloc[-2])
        volume_ratio = volume_for_ratio / avg_volume if avg_volume > 0 else 1.0
    else:
        volume_ratio = volume / avg_volume if avg_volume > 0 else 1.0

    # Moving averages
    sma20 = _safe(df["Close"].rolling(20).mean().iloc[-1], close)
    sma50 = _safe(df["Close"].rolling(50).mean().iloc[-1], close)
    sma200 = _safe(df["Close"].rolling(200).mean().iloc[-1], close) if len(df) >= 200 else sma50

    # RSI (simple 14-period)
    delta = df["Close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi_series = 100 - (100 / (1 + rs))
    rsi = _safe(rsi_series.iloc[-1], 50.0)
    rsi_prev = _safe(rsi_series.iloc[-2], rsi) if len(rsi_series) >= 2 else rsi
    rsi_5d_ago = _safe(rsi_series.iloc[-6], rsi) if len(rsi_series) >= 6 else rsi
    rsi_turning_up = rsi > rsi_prev and rsi_prev < 45

    # Price changes (with division-by-zero guards)
    change_1d = ((close - prev_close) / prev_close * 100) if prev_close > 0 else 0.0
    _c5 = _safe(df["Close"].iloc[-6]) if len(df) >= 6 else 0.0
    change_5d = ((close - _c5) / _c5 * 100) if _c5 > 0 else 0.0
    _c20 = _safe(df["Close"].iloc[-21]) if len(df) >= 21 else 0.0
    change_20d = ((close - _c20) / _c20 * 100) if _c20 > 0 else 0.0

    # ATR
    high = df["High"]
    low = df["Low"]
    tr = pd.concat([
        high - low,
        (high - df["Close"].shift(1)).abs(),
        (low - df["Close"].shift(1)).abs(),
    ], axis=1).max(axis=1)
    atr = _safe(tr.rolling(14).mean().iloc[-1], close * 0.03) or close * 0.03

    # Volatility contraction (5d range / 20d range)
    range_5 = _safe(high.iloc[-5:].max() - low.iloc[-5:].min(), atr * 5) if len(df) >= 5 else atr * 5
    range_20 = _safe(high.iloc[-20:].max() - low.iloc[-20:].min(), atr * 20) if len(df) >= 20 else atr * 20
    vol_contraction = range_5 / range_20 if range_20 > 0 else 1.0

    # Range as pct
    range_5d_pct = (range_5 / close * 100) if close > 0 else 10.0

    # Distance from 20-day high
    high_20 = _safe(high.rolling(20).max().iloc[-1], close)
    dist_from_high = ((high_20 - close) / close * 100) if close > 0 else 0.0

    # Volume trend (5d avg / prev 5d avg)
    vol_5d = _safe(df["Volume"].iloc[-5:].mean(), volume)
    vol_10d = _safe(df["Volume"].iloc[-10:-5].mean(), volume) or 1.0
    vol_trend = vol_5d / vol_10d if vol_10d > 0 else 1.0

    # OBV slope (normalized by average volume over 10 bars)
    obv = (np.sign(df["Close"].diff()) * df["Volume"]).cumsum()
    obv_val = _safe(obv.iloc[-1])
    obv_sma = _safe(obv.rolling(20).mean().iloc[-1])
    # Compute OBV slope: change over last 10 bars, normalized by avg volume
    obv_10_ago = _safe(obv.iloc[-11]) if len(obv) >= 11 else obv_val
    obv_slope = (obv_val - obv_10_ago) / (avg_volume * 10) if avg_volume > 0 else 0.0

    # MACD histogram
    ema12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df["Close"].ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    macd_signal = macd_line.ewm(span=9, adjust=False).mean()
    macd_hist = macd_line - macd_signal
    macd_hist_val = _safe(macd_hist.iloc[-1])
    macd_hist_prev = _safe(macd_hist.iloc[-2]) if len(macd_hist) >= 2 else macd_hist_val

    # EMA21
    ema21 = _safe(df["Close"].ewm(span=21, adjust=False).mean().iloc[-1], close)

    # SMA trends (for direction bias and safety checks)
    sma50_2 = _safe(df["Close"].rolling(50).mean().iloc[-2], sma50) if len(df) >= 52 else sma50
    sma50_trend = (sma50 - sma50_2) / sma50_2 * 100 if sma50_2 > 0 else 0.0
    sma20_3 = _safe(df["Close"].rolling(20).mean().iloc[-3], sma20) if len(df) >= 23 else sma20
    sma20_trend = (sma20 - sma20_3) / sma20_3 * 100 if sma20_3 > 0 else 0.0

    return {
        "close": close,
        "open": open_price,
        "prev_close": prev_close,
        "volume": volume,
        "avg_volume": avg_volume,
        "volume_ratio": volume_ratio,
        "sma20": sma20,
        "sma50": sma50,
        "sma200": sma200,
        "ema21": ema21,
        "rsi": rsi,
        "rsi_prev": rsi_prev,
        "rsi_5d_ago": rsi_5d_ago,
        "rsi_turning_up": rsi_turning_up,
        "change_1d": change_1d,
        "change_5d": change_5d,
        "change_20d": change_20d,
        "atr": atr,
        "vol_contraction": vol_contraction,
        "range_5d_pct": range_5d_pct,
        "high_20": high_20,
        "dist_from_high": dist_from_high,
        "vol_trend": vol_trend,
        "obv": obv_val,
        "obv_sma": obv_sma,
        "obv_slope": obv_slope,
        "macd_hist": macd_hist_val,
        "macd_hist_prev": macd_hist_prev,
        "sma50_trend": sma50_trend,
        "sma20_trend": sma20_trend,
    }


# ---------------------------------------------------------------------------
# Relative strength vs SPY
# ---------------------------------------------------------------------------

def relative_strength(stock_df: pd.DataFrame, spy_df: pd.DataFrame, window: int = 60) -> float | None:
    """Return relative strength of stock vs SPY over `window` bars.
    Positive = outperforming SPY.  Aligns on DateTimeIndex so halted
    stocks or mismatched row counts don't cause stale comparisons."""
    if stock_df is None or spy_df is None or stock_df.empty or spy_df.empty:
        return None
    try:
        # Align on date index, drop any rows where either side is missing
        aligned = pd.DataFrame({
            "stock": stock_df["Close"],
            "spy": spy_df["Close"],
        }).dropna()
        if len(aligned) < window + 1:
            return None
        stock_return = (aligned["stock"].iloc[-1] / aligned["stock"].iloc[-window - 1]) - 1
        spy_return = (aligned["spy"].iloc[-1] / aligned["spy"].iloc[-window - 1]) - 1
        return float(stock_return - spy_return)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Individual signal type detectors
# ---------------------------------------------------------------------------

def _is_pre_break(t: dict) -> bool:
    return (
        1.0 < t["dist_from_high"] < 12.0
        and t["range_5d_pct"] < 7.0
        and t["close"] > t["sma20"] * 0.98
        and 35.0 < t["rsi"] < 70.0
        and t["vol_contraction"] < 0.85
    )


def _is_vcp(t: dict) -> bool:
    return (
        t["vol_contraction"] < 0.50
        and t["close"] > t["sma20"] > t["sma50"]
        and t["dist_from_high"] < 15.0
        and 40.0 < t["rsi"] < 65.0
    )


def _is_accumulation(t: dict) -> bool:
    return (
        t["vol_trend"] > 1.08
        and abs(t["change_5d"]) < 5.0
        and t["close"] > t["sma50"] * 0.98
        and 35.0 < t["rsi"] < 65.0
        and t["volume_ratio"] < 2.5
    )


def _is_early_reversal(t: dict) -> bool:
    return (
        t["rsi_turning_up"]
        and 28.0 < t["rsi"] < 50.0
        and t["rsi_5d_ago"] < 42.0
        and t["close"] > t["sma50"] * 0.92
    )


def _is_pullback(t: dict) -> bool:
    return (
        t["close"] >= t["sma20"] * 0.975
        and t["close"] <= t["sma20"] * 1.04
        and t["sma20"] > t["sma50"]
        and t["change_20d"] >= 2.0
        and -4.0 <= t["change_5d"] <= 4.0
        and t["change_1d"] > -3.0
        and 38.0 <= t["rsi"] <= 66.0
        and t["volume_ratio"] <= 2.2
        and t["range_5d_pct"] < 8.0
        and t["dist_from_high"] < 15.0
        and t["vol_contraction"] <= 1.3
    )


def _is_relative_strength(t: dict) -> bool:
    return (
        t["dist_from_high"] < 10.0
        and t["change_1d"] > -2.0
        and t["change_20d"] > 1.5
        and t["change_5d"] > max(-2.0, t["change_20d"] * 0.10)
        and 38.0 < t["rsi"] < 75.0
        and t["close"] > t["sma20"] * 0.98
    )


def _is_momentum_rank(t: dict, rs_score: float | None) -> bool:
    if not (
        t["close"] > t["sma20"] > t["sma50"]
        and t["change_20d"] > 3.0
        and t["dist_from_high"] < 12.0
        and 38.0 < t["rsi"] < 76.0
        and t["volume_ratio"] > 0.7
    ):
        return False
    if rs_score is None:
        return False  # No RS data = can't confirm momentum rank, skip
    return rs_score > 0.02


def _is_momentum(t: dict) -> bool:
    return (
        t["close"] > t["sma20"] > t["sma50"]
        and 0.5 < t["change_5d"] < 12.0
        and t["volume_ratio"] > 0.8
        and 35.0 <= t["rsi"] <= 72.0
    )


def _is_breakout(t: dict) -> bool:
    return (
        t["close"] >= t["high_20"] * 0.99
        and t["volume_ratio"] > 1.3
        and t["change_1d"] > -0.5
        and t["rsi"] < 78.0
    )


def _is_dip_buy(t: dict) -> bool:
    return (
        t["rsi"] < 45.0
        and t["close"] > t["sma50"] * 0.95
        and t["change_1d"] > -0.5
        and t["rsi"] > t["rsi_prev"]
    )


def _is_volume_spike(t: dict) -> bool:
    return (
        t["volume_ratio"] > 2.0
        and abs(t["change_1d"]) < 6.0
        and t["close"] > t["sma20"] * 0.98
    )


# ---------------------------------------------------------------------------
# Signal freshness — has this signal type been true for the last N bars?
# ---------------------------------------------------------------------------

def signal_age_bars(df: pd.DataFrame, signal_type: str, spy_df: pd.DataFrame = None,
                    max_lookback: int = SIGNAL_FRESHNESS_BARS) -> int:
    """Return how many bars ago this signal first appeared (0 = brand new today).
    Returns -1 if the signal is not currently active."""
    if df is None or df.empty or len(df) < 62:
        return -1

    # Check if signal is active on the latest bar (among ALL active, not just top-ranked)
    t_now = _compute_technicals(df)
    if not t_now:
        return -1
    rs = relative_strength(df, spy_df) if spy_df is not None else None
    active_now = _get_all_active_signals(t_now, rs)
    if signal_type not in active_now:
        return -1

    # Walk backwards to find when it first appeared
    age = 0
    for lookback in range(1, max_lookback + 1):
        sub = df.iloc[:-lookback]
        if len(sub) < 60:
            break
        t_prev = _compute_technicals(sub)
        if not t_prev:
            break
        rs_prev = relative_strength(sub, spy_df) if spy_df is not None else None
        active_prev = _get_all_active_signals(t_prev, rs_prev)
        if signal_type in active_prev:
            age = lookback
        else:
            break

    return age


# ---------------------------------------------------------------------------
# Main detection entry point
# ---------------------------------------------------------------------------

def _classify_signal(t: dict, rs_score: float | None) -> tuple[str | None, float, list[str]]:
    """Check ALL signal types and return the strongest match."""
    if not t:
        return None, 0.0, []

    candidates: list[tuple[str, float, list[str]]] = []

    if _is_vcp(t):
        s = 7.0 + max(0.0, (0.5 - t["vol_contraction"]) * 3)
        candidates.append(("VCP", s, [f"Contraction {t['vol_contraction']:.2f}", f"{t['dist_from_high']:.1f}% from high"]))

    if _is_pre_break(t):
        s = 6.5 + max(0.0, (8 - t["dist_from_high"]) / 5)
        candidates.append(("PRE_BREAK", s, [f"Only {t['dist_from_high']:.1f}% from high", f"Tight range {t['range_5d_pct']:.1f}%"]))

    if _is_pullback(t):
        pb = abs((t["close"] - t["sma20"]) / t["sma20"]) * 100 if t["sma20"] > 0 else 99
        s = 6.5 + max(0.0, 1.5 - (pb / 2.5)) + min(1.5, t["change_20d"] / 12)
        candidates.append(("PULLBACK", s, [f"Trend intact +{t['change_20d']:.1f}% over 20D", f"Near SMA20 ({pb:.1f}%)"]))

    if _is_accumulation(t):
        s = 6.0 + min(2.0, max(0.0, (t["vol_trend"] - 1) * 3))
        candidates.append(("ACCUM", s, [f"Vol building {t['vol_trend']:.1f}x", f"Price flat {t['change_5d']:+.1f}%"]))

    if _is_relative_strength(t):
        s = 6.0 + max(0.0, (5 - t["dist_from_high"]) / 2)
        candidates.append(("REL_STR", s, [f"Only {t['dist_from_high']:.1f}% from high", "Relative strength"]))

    if _is_momentum_rank(t, rs_score):
        s = 6.0 + min(3.0, max(0.0, (rs_score or 0.0) * 20))
        candidates.append(("MOM_RANK", s, [f"RS vs SPY +{(rs_score or 0.0):.1%}", f"20D +{t['change_20d']:.1f}%"]))

    # EARLY_REV removed — backtest shows negative expectancy (-0.61% at 3D, 41.6% WR)
    # Kept detector for reference but no longer added to candidates
    # if _is_early_reversal(t):
    #     s = 5.5 + max(0.0, (45 - t["rsi"]) / 5)
    #     candidates.append(("EARLY_REV", s, [...]))

    if _is_momentum(t):
        s = 5.0 + min(2.0, t["change_5d"] / 3)
        candidates.append(("MOMENTUM", s, [f"Trend UP +{t['change_5d']:.1f}%", f"Vol {t['volume_ratio']:.1f}x"]))

    if _is_breakout(t):
        s = 5.0 + min(1.5, max(0.0, t["volume_ratio"] - 1))
        candidates.append(("BREAKOUT", s, ["At 20D high", f"Vol {t['volume_ratio']:.1f}x"]))

    # DIP_BUY removed — backtest shows negative expectancy (-0.18% at 3D, 46.8% WR)
    # Raising base score made it worse (more weak signals passed). Killed entirely.
    # if _is_dip_buy(t):
    #     s = 5.5 + max(0.0, (40 - t["rsi"]) / 10)
    #     candidates.append(("DIP_BUY", s, [...]))

    if _is_volume_spike(t):
        s = 4.5 + min(2.0, max(0.0, (t["volume_ratio"] - 2) / 2))
        candidates.append(("VOL_SPIKE", s, [f"Vol spike {t['volume_ratio']:.1f}x", f"Price {t['change_1d']:+.1f}%"]))

    if not candidates:
        return None, 0.0, []

    # Return the strongest signal
    candidates.sort(key=lambda c: -c[1])
    return candidates[0]


def _get_all_active_signals(t: dict, rs_score: float | None) -> set[str]:
    """Return the set of ALL active signal type names (regardless of rank)."""
    if not t:
        return set()
    active = set()
    if _is_vcp(t):
        active.add("VCP")
    if _is_pre_break(t):
        active.add("PRE_BREAK")
    if _is_pullback(t):
        active.add("PULLBACK")
    if _is_accumulation(t):
        active.add("ACCUM")
    if _is_relative_strength(t):
        active.add("REL_STR")
    if _is_momentum_rank(t, rs_score):
        active.add("MOM_RANK")
    if _is_momentum(t):
        active.add("MOMENTUM")
    if _is_breakout(t):
        active.add("BREAKOUT")
    if _is_volume_spike(t):
        active.add("VOL_SPIKE")
    if _is_early_reversal(t):
        active.add("EARLY_REV")
    if _is_dip_buy(t):
        active.add("DIP_BUY")
    return active


def detect_signal(df: pd.DataFrame, spy_df: pd.DataFrame = None) -> tuple[str | None, float, list[str], dict]:
    """Detect the primary signal type for a stock.

    Returns:
        (signal_type, signal_strength, reasons, technicals)
    """
    t = _compute_technicals(df)
    if not t:
        return None, 0.0, [], {}

    rs_score = relative_strength(df, spy_df) if spy_df is not None else None
    signal_type, strength, reasons = _classify_signal(t, rs_score)

    if signal_type is None:
        return None, 0.0, [], t

    # Strength bonuses (same as KOSPI system)
    # MACD histogram crossover
    if t["macd_hist"] > 0 and t["macd_hist_prev"] <= 0:
        strength += 1.5
        reasons.append("MACD crossover")
    elif t["macd_hist"] > t["macd_hist_prev"] > 0:
        strength += 0.5

    # OBV confirmation — slope-based (normalized by avg volume over 10 bars)
    if t["obv_slope"] > 0.05:
        strength += 0.2

    # RS bonus
    if rs_score is not None and rs_score > 0.05:
        strength += min(1.5, rs_score * 10)
        reasons.append(f"RS vs SPY +{rs_score:.1%}")

    return signal_type, strength, reasons, t
