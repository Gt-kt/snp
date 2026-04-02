"""
Titan Signal Backtester — Historical Validation (Production-Grade)
==================================================================
Walk through historical data day-by-day, run signal detection exactly
as the live scanner would, and measure what actually happened.

KEY DIFFERENCE from naive backtesting:
  - Entry at NEXT-DAY OPEN + slippage (not signal-day close)
  - Commission on entry AND exit
  - Forward returns measured from actual entry, not signal date
  - Drawdown, Sharpe, profit factor, equity curve
  - Sample-size warnings for statistical significance

Usage:
    python -m titan.backtester              # Use all available data
    python -m titan.backtester --days 120   # Custom lookback
    python -m titan.backtester --csv out.csv # Export raw signals to CSV
"""

import argparse
import time
import pandas as pd
import numpy as np
from datetime import datetime

from titan.config import (
    OHLCV_CACHE_FILE, DEFAULT_MAX_WORKERS,
    DEFAULT_SLIPPAGE_BREAKOUT_BPS, DEFAULT_SLIPPAGE_DIP_BPS, DEFAULT_COMMISSION_BPS,
)
from titan.signal_detector import (
    detect_signal, _compute_technicals, relative_strength,
    EARLY_SIGNAL_TYPES, BREAKOUT_SIGNAL_TYPES, ALL_SIGNAL_TYPES, _safe,
)
from titan.pro_scanner import (
    _passes_quality_gate, _grade_signal, _detect_market_regime,
    _check_weekly_trend, _has_recent_distribution, _compute_historical_gap_risk,
    _classify_tier, _validated_min_strength, _watchlist_min_strength,
    _score_tradeability, _calculate_position,
    MIN_PRICE, ADX_CHOPPY_THRESHOLD, ADX_STRONG_TREND_THRESHOLD,
    QUALITY_HARD_REASONS, MIN_AVG_DOLLAR_VOLUME,
)
from titan.execution_plan import build_execution_plan, fresh_entry_is_buyable, is_entry_buyable

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
WARMUP_BARS = 200       # Need 200 for SMA200
FORWARD_WINDOWS = [1, 3, 5]
MAX_FORWARD = max(FORWARD_WINDOWS)
MIN_STATISTICAL_SAMPLES = 30  # Below this, results are unreliable


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_cached_data() -> pd.DataFrame:
    """Load cached OHLCV parquet."""
    data = pd.read_parquet(OHLCV_CACHE_FILE)
    if not isinstance(data.columns, pd.MultiIndex):
        raise ValueError("Expected MultiIndex columns in parquet")
    return data


# ---------------------------------------------------------------------------
# Realistic entry model — next-day open + slippage
# ---------------------------------------------------------------------------

def _compute_realistic_entry(
    full_ticker_df: pd.DataFrame,
    signal_date_idx: int,
    signal_type: str,
    max_buy_price: float,
) -> tuple[float, int]:
    """Compute realistic entry: next-day open + slippage + commission.

    Returns (entry_price, entry_date_idx).
    Returns (0, -1) if entry not possible (no data, gapped above max buy).
    """
    next_idx = signal_date_idx + 1
    if next_idx >= len(full_ticker_df):
        return 0.0, -1

    next_open = _safe(full_ticker_df["Open"].iloc[next_idx])
    if next_open <= 0:
        return 0.0, -1

    # Anti-chase: if next open gaps above max buy price, it's a miss
    if max_buy_price > 0 and next_open > max_buy_price:
        return 0.0, -1

    # Slippage: breakout types pay more (chasing momentum)
    if signal_type in BREAKOUT_SIGNAL_TYPES:
        slippage_bps = DEFAULT_SLIPPAGE_BREAKOUT_BPS
    else:
        slippage_bps = DEFAULT_SLIPPAGE_DIP_BPS

    total_cost_bps = slippage_bps + DEFAULT_COMMISSION_BPS
    entry_price = next_open * (1 + total_cost_bps / 10000)

    return round(entry_price, 2), next_idx


# ---------------------------------------------------------------------------
# Forward return computation (from actual entry, not signal date)
# ---------------------------------------------------------------------------

def _compute_forward_returns(
    full_ticker_df: pd.DataFrame,
    entry_date_idx: int,
    entry_price: float,
    stop: float,
    target: float,
) -> dict:
    """Compute forward 1D/3D/5D returns from actual entry date.

    Deducts exit commission from all return calculations.
    """
    result = {}
    total_bars = len(full_ticker_df)
    exit_cost_factor = 1 - DEFAULT_COMMISSION_BPS / 10000  # Commission on exit

    for days in FORWARD_WINDOWS:
        fwd_idx = entry_date_idx + days
        if fwd_idx >= total_bars:
            result[f"return_{days}d"] = None
            result[f"win_{days}d"] = None
        else:
            fwd_close = _safe(full_ticker_df["Close"].iloc[fwd_idx]) * exit_cost_factor
            ret = (fwd_close - entry_price) / entry_price * 100 if entry_price > 0 else 0
            result[f"return_{days}d"] = round(ret, 3)
            result[f"win_{days}d"] = ret > 0

    # Check stop/target hit within 5D from entry
    end_idx = min(entry_date_idx + MAX_FORWARD + 1, total_bars)
    fwd_slice = full_ticker_df.iloc[entry_date_idx + 1:end_idx]

    if len(fwd_slice) > 0:
        max_high = _safe(fwd_slice["High"].max())
        min_low = _safe(fwd_slice["Low"].min())
        result["hit_target_5d"] = max_high >= target if target > 0 else False
        result["hit_stop_5d"] = min_low <= stop if stop > 0 else False
        result["max_gain_5d"] = round((max_high - entry_price) / entry_price * 100, 2) if entry_price > 0 else 0
        result["max_dd_5d"] = round((min_low - entry_price) / entry_price * 100, 2) if entry_price > 0 else 0
    else:
        result["hit_target_5d"] = None
        result["hit_stop_5d"] = None
        result["max_gain_5d"] = None
        result["max_dd_5d"] = None

    return result


# ---------------------------------------------------------------------------
# Single ticker evaluation at a given date
# ---------------------------------------------------------------------------

def _evaluate_ticker(
    ticker: str,
    ticker_df: pd.DataFrame,
    full_ticker_df: pd.DataFrame,
    spy_df: pd.DataFrame,
    regime_info: dict,
    market_score: float,
    signal_date_idx_in_full: int,
    date_str: str,
) -> dict | None:
    """Evaluate one ticker at one historical date. Returns signal record or None."""
    if ticker_df is None or len(ticker_df) < 60:
        return None

    close = _safe(ticker_df["Close"].iloc[-1])
    if close < MIN_PRICE:
        return None

    # Liquidity
    avg_vol = _safe(ticker_df["Volume"].rolling(20).mean().iloc[-1])
    if avg_vol * close < MIN_AVG_DOLLAR_VOLUME:
        return None

    t = _compute_technicals(ticker_df)
    if not t:
        return None

    sma50_trend = t.get("sma50_trend", 0.0)

    # Safety checks
    if sma50_trend < -0.3 and close < t.get("sma50", close) * 0.95:
        return None

    severe_dist, mild_dist = _has_recent_distribution(ticker_df)
    if severe_dist:
        return None

    # Signal detection
    signal_type, strength, reasons, t = detect_signal(ticker_df, spy_df)
    if not signal_type:
        return None

    # Regime adjustments
    regime = regime_info.get("regime", "SIDEWAYS")
    adx = regime_info.get("adx", 20)
    is_choppy = adx < ADX_CHOPPY_THRESHOLD
    is_strong_trend = adx > ADX_STRONG_TREND_THRESHOLD

    if is_choppy:
        if signal_type in ("BREAKOUT", "MOMENTUM", "VOL_SPIKE", "PRE_BREAK"):
            strength -= 2.0
        elif signal_type in ("ACCUM", "VCP", "PULLBACK"):
            strength += 2.0

    if is_strong_trend and signal_type in ("BREAKOUT", "MOMENTUM", "MOM_RANK", "PULLBACK"):
        strength += 2.0

    # Weekly trend
    weekly = _check_weekly_trend(ticker_df)
    weekly_trend = weekly.get("trend", "N/A")
    if weekly.get("valid") and not weekly.get("aligned", True):
        if weekly.get("trend") == "STRONG_DOWN":
            return None
        else:
            strength -= 1.0

    if weekly_trend == "STRONG_UP":
        strength += 1.0

    # Safety/gap penalties
    _safety_penalty = 0.0
    if sma50_trend < -0.1 and close < t.get("sma50", close):
        _safety_penalty -= 1.5
    if mild_dist:
        _safety_penalty -= 1.0

    hist_gap_risk, hist_avg_gap_pct, hist_gap_above_1pct_prob = _compute_historical_gap_risk(ticker_df)
    _gap_penalty = -1.5 if hist_gap_risk == "HIGH" else (-0.5 if hist_gap_risk == "MED" and hist_avg_gap_pct > 0.5 else 0.0)

    strength += _gap_penalty + _safety_penalty

    # Today's gap
    prev_close = _safe(ticker_df["Close"].iloc[-2])
    today_gap = abs(close - prev_close) / prev_close if prev_close > 0 else 0
    if today_gap > 0.05:
        return None

    # Quality gate
    passes_qg, qg_reasons = _passes_quality_gate(
        signal_type, t.get("change_1d", 0), t.get("change_20d", 0), t.get("rsi", 50),
    )
    if not passes_qg and any(r in QUALITY_HARD_REASONS for r in qg_reasons):
        return None

    # Strength floor
    min_str = _watchlist_min_strength(market_score, signal_type)
    if strength < min_str:
        return None

    # Execution plan (2.0 stop / 2.5 target — matches live scanner)
    atr = t.get("atr", close * 0.03)
    stop = round(close - 2.0 * atr, 2)
    target = round(close + 2.5 * atr, 2)

    plan = build_execution_plan(
        signal_type=signal_type, price=close, stop=stop, target=target, atr=atr,
        avg_open_gap_pct=hist_avg_gap_pct, gap_above_1pct_prob=hist_gap_above_1pct_prob,
    )

    # Tier
    is_fresh = True
    age = 0

    tier = _classify_tier(
        signal_type=signal_type, strength=strength, age=age, is_fresh=is_fresh,
        close=close, plan=plan, weekly_trend=weekly_trend, regime=regime,
        market_score=market_score,
    )
    if tier is None:
        return None

    # Grade
    grade = _grade_signal(
        strength, weekly_trend, t.get("vol_contraction", 1),
        t.get("rsi", 50), t.get("dist_from_high", 50), regime,
    )

    # --- REALISTIC ENTRY: next-day open + slippage ---
    max_buy = plan.get("max_buy_price", close * 1.02)
    entry_price, entry_idx = _compute_realistic_entry(
        full_ticker_df, signal_date_idx_in_full, signal_type, max_buy,
    )

    if entry_price <= 0 or entry_idx < 0:
        return None  # Missed — gapped above buy zone or no next-day data

    # Recompute stop/target relative to actual entry price
    # (stop distance stays same ATR multiple, just from real entry)
    stop = round(entry_price - 2.0 * atr, 2)
    target = round(entry_price + 2.5 * atr, 2)

    # Forward returns from ACTUAL ENTRY DATE (not signal date)
    fwd = _compute_forward_returns(full_ticker_df, entry_idx, entry_price, stop, target)

    return {
        "date": date_str,
        "ticker": ticker,
        "signal_type": signal_type,
        "tier": tier,
        "grade": grade,
        "strength": round(strength, 1),
        "signal_price": close,
        "entry_price": entry_price,
        "entry_gap_pct": round((entry_price - close) / close * 100, 2),
        "stop": stop,
        "target": target,
        "rsi": round(t.get("rsi", 50), 1),
        "vol_contraction": round(t.get("vol_contraction", 1), 2),
        "change_1d": round(t.get("change_1d", 0), 1),
        "change_20d": round(t.get("change_20d", 0), 1),
        "weekly_trend": weekly_trend,
        "regime": regime,
        "gap_risk": hist_gap_risk,
        **fwd,
    }


# ---------------------------------------------------------------------------
# Main backtest loop
# ---------------------------------------------------------------------------

def run_backtest(data: pd.DataFrame, lookback_days: int = 9999) -> list[dict]:
    """Walk through historical dates and collect signal records."""
    all_dates = data.index
    total_dates = len(all_dates)

    # Need WARMUP_BARS before first test date + MAX_FORWARD+1 after last
    # (+1 because we enter next-day open, then measure forward from there)
    start_idx = max(WARMUP_BARS, total_dates - lookback_days - MAX_FORWARD - 1)
    end_idx = total_dates - MAX_FORWARD - 1  # Extra day for next-day entry

    if start_idx >= end_idx:
        print("  Not enough data for backtest.")
        return []

    test_dates = all_dates[start_idx:end_idx]
    print(f"  Backtest period: {test_dates[0].date()} to {test_dates[-1].date()} ({len(test_dates)} trading days)")

    if len(test_dates) < 60:
        print(f"  [!] WARNING: Only {len(test_dates)} trading days — results may not be statistically reliable.")

    # Get tickers (exclude SPY, VIX, sector ETFs)
    from titan.config import SECTOR_ETFS
    skip_tickers = {"SPY", "^VIX"} | {etf.upper() for etf in SECTOR_ETFS.values()}
    tickers = [t for t in data.columns.levels[0] if t not in skip_tickers]
    print(f"  Universe: {len(tickers)} stocks")
    print(f"  Slippage model: entry at next-day open + {DEFAULT_SLIPPAGE_DIP_BPS:.0f}-{DEFAULT_SLIPPAGE_BREAKOUT_BPS:.0f} bps + {DEFAULT_COMMISSION_BPS:.0f} bps commission")

    # Pre-extract full ticker DataFrames
    full_frames = {}
    for ticker in tickers:
        try:
            tdf = data[ticker].dropna()
            if len(tdf) >= WARMUP_BARS:
                full_frames[ticker] = tdf
        except Exception:
            pass
    print(f"  Valid tickers with enough history: {len(full_frames)}")

    spy_full = data["SPY"].dropna()

    all_records = []
    missed_entries = 0
    start_time = time.time()

    for i, date in enumerate(test_dates):
        date_str = str(date.date())
        date_iloc = all_dates.get_loc(date)

        # SPY slice for regime
        spy_slice = spy_full.iloc[:date_iloc + 1]
        if len(spy_slice) < 200:
            continue

        regime_info = _detect_market_regime(spy_slice)
        regime = regime_info.get("regime", "SIDEWAYS")

        if regime == "STRONG_BEAR":
            continue

        # Quick breadth calc
        bullish = 0
        checked = 0
        sample_tickers = list(full_frames.keys())[:50]
        for st in sample_tickers:
            tdf = full_frames[st]
            idx = tdf.index.get_indexer([date], method="ffill")[0]
            if idx < 20 or idx < 0:
                continue
            sma20_val = _safe(tdf["Close"].iloc[max(0, idx - 19):idx + 1].mean())
            close_val = _safe(tdf["Close"].iloc[idx])
            if close_val > sma20_val:
                bullish += 1
            checked += 1
        market_score = (bullish / checked * 100) if checked > 0 else 50

        # Process each ticker
        date_records = []
        for ticker, full_df in full_frames.items():
            if date not in full_df.index:
                continue
            tidx = full_df.index.get_loc(date)
            if tidx < 60:
                continue

            ticker_slice = full_df.iloc[:tidx + 1]

            try:
                rec = _evaluate_ticker(
                    ticker=ticker,
                    ticker_df=ticker_slice,
                    full_ticker_df=full_df,
                    spy_df=spy_slice,
                    regime_info=regime_info,
                    market_score=market_score,
                    signal_date_idx_in_full=tidx,
                    date_str=date_str,
                )
                if rec is not None:
                    date_records.append(rec)
            except Exception:
                pass

        all_records.extend(date_records)

        if (i + 1) % 5 == 0 or i == len(test_dates) - 1:
            elapsed = time.time() - start_time
            pct = (i + 1) / len(test_dates) * 100
            print(f"    [{pct:5.1f}%] Day {i+1}/{len(test_dates)} ({date_str}) "
                  f"| {len(date_records)} signals | Total: {len(all_records)} | "
                  f"{elapsed:.0f}s", end="\r")

    print()
    print(f"  Backtest complete: {len(all_records)} total signals in {time.time() - start_time:.0f}s")
    return all_records


# ---------------------------------------------------------------------------
# Risk metrics — the real test of a system
# ---------------------------------------------------------------------------

def compute_risk_metrics(records: list[dict]) -> dict:
    """Compute portfolio-level risk metrics from backtest results.

    These are the numbers that matter for real money:
    - Profit Factor (need > 1.3 minimum, > 1.5 preferred)
    - Sharpe Ratio (need > 0.5 minimum, > 1.0 preferred)
    - Max Drawdown (acceptable depends on account size)
    - Worst Losing Streak (can you stomach it?)
    - Expectancy per trade (must be positive after costs)
    """
    if not records:
        return {}

    df = pd.DataFrame(records).sort_values("date")

    # Use 3D returns as primary holding period
    returns = df["return_3d"].dropna() / 100  # pct to decimal

    if len(returns) == 0:
        return {}

    # --- Profit Factor ---
    gross_wins = returns[returns > 0].sum()
    gross_losses = abs(returns[returns < 0].sum())
    profit_factor = gross_wins / gross_losses if gross_losses > 0 else float('inf')

    # --- Win/Loss Ratio ---
    avg_win = returns[returns > 0].mean() if (returns > 0).any() else 0
    avg_loss = abs(returns[returns < 0].mean()) if (returns < 0).any() else 0
    win_loss_ratio = avg_win / avg_loss if avg_loss > 0 else float('inf')

    # --- Expectancy ---
    win_rate = (returns > 0).mean()
    expectancy = win_rate * avg_win - (1 - win_rate) * avg_loss

    # --- Sharpe Ratio (annualized) ---
    # Assume ~3-day hold, so ~84 trades/year
    trades_per_year = 252 / 3
    mean_ret = returns.mean()
    std_ret = returns.std()
    risk_free_daily = 0.05 / trades_per_year  # ~5% annual risk-free
    sharpe = ((mean_ret - risk_free_daily) * np.sqrt(trades_per_year)) / std_ret if std_ret > 0 else 0

    # --- Equity Curve + Max Drawdown ---
    # Group by date, average all signals that day (represents capital allocation)
    daily_returns = df.groupby("date")["return_3d"].mean().dropna() / 100
    daily_returns = daily_returns.sort_index()
    equity = (1 + daily_returns).cumprod()
    rolling_max = equity.cummax()
    drawdowns = (equity - rolling_max) / rolling_max
    max_drawdown = drawdowns.min() * 100

    # Max drawdown duration (in trading days)
    in_dd = drawdowns < 0
    dd_start = None
    max_dd_duration = 0
    current_dd_duration = 0
    for dt, is_in_dd in in_dd.items():
        if is_in_dd:
            current_dd_duration += 1
            max_dd_duration = max(max_dd_duration, current_dd_duration)
        else:
            current_dd_duration = 0

    # --- Worst Losing Streak ---
    wins = (returns > 0).astype(int)
    streak = 0
    worst_streak = 0
    for w in wins:
        if w == 0:
            streak += 1
            worst_streak = max(worst_streak, streak)
        else:
            streak = 0

    # --- Best/Worst Single Trade ---
    best_trade = returns.max() * 100
    worst_trade = returns.min() * 100

    # --- Monthly breakdown ---
    df_dated = df.copy()
    df_dated["month"] = pd.to_datetime(df_dated["date"]).dt.to_period("M")
    monthly = df_dated.groupby("month").agg(
        trades=("return_3d", "count"),
        avg_ret=("return_3d", "mean"),
        wr=("win_3d", "mean"),
    )
    monthly["wr"] = monthly["wr"] * 100
    profitable_months = (monthly["avg_ret"] > 0).sum()
    total_months = len(monthly)

    return {
        "profit_factor": round(profit_factor, 2),
        "sharpe_ratio": round(sharpe, 2),
        "max_drawdown_pct": round(max_drawdown, 2),
        "max_dd_duration_days": max_dd_duration,
        "worst_losing_streak": worst_streak,
        "win_loss_ratio": round(win_loss_ratio, 2),
        "expectancy_pct": round(expectancy * 100, 3),
        "avg_win_pct": round(avg_win * 100, 3),
        "avg_loss_pct": round(avg_loss * 100, 3),
        "best_trade_pct": round(best_trade, 2),
        "worst_trade_pct": round(worst_trade, 2),
        "total_trades": len(returns),
        "profitable_months": profitable_months,
        "total_months": total_months,
        "win_rate_pct": round(win_rate * 100, 1),
        "equity_curve": {str(k): round(v, 4) for k, v in equity.to_dict().items()},
        "monthly": monthly.to_dict("index"),
    }


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def compute_statistics(records: list[dict]) -> dict:
    """Aggregate backtest results into statistics."""
    if not records:
        return {"total": 0}

    df = pd.DataFrame(records)
    stats = {"total": len(df)}

    def _calc_group(group_df: pd.DataFrame) -> dict:
        n = len(group_df)
        result = {"count": n, "low_sample": n < MIN_STATISTICAL_SAMPLES}
        for d in FORWARD_WINDOWS:
            col = f"return_{d}d"
            win_col = f"win_{d}d"
            valid = group_df[group_df[col].notna()]
            if len(valid) > 0:
                result[f"wr_{d}d"] = round(valid[win_col].sum() / len(valid) * 100, 1)
                result[f"avg_{d}d"] = round(valid[col].mean(), 3)
                result[f"med_{d}d"] = round(valid[col].median(), 3)
            else:
                result[f"wr_{d}d"] = None
                result[f"avg_{d}d"] = None
                result[f"med_{d}d"] = None

        valid_5d = group_df[group_df["hit_target_5d"].notna()]
        if len(valid_5d) > 0:
            result["target_hit_5d"] = round(valid_5d["hit_target_5d"].sum() / len(valid_5d) * 100, 1)
            result["stop_hit_5d"] = round(valid_5d["hit_stop_5d"].sum() / len(valid_5d) * 100, 1)
        else:
            result["target_hit_5d"] = None
            result["stop_hit_5d"] = None

        return result

    # Overall
    stats["overall"] = _calc_group(df)

    # By signal type
    stats["by_signal_type"] = {}
    for sig in df["signal_type"].unique():
        sub = df[df["signal_type"] == sig]
        stats["by_signal_type"][sig] = _calc_group(sub)

    # By grade
    stats["by_grade"] = {}
    for g in ["A", "B", "C"]:
        sub = df[df["grade"] == g]
        if len(sub) > 0:
            stats["by_grade"][g] = _calc_group(sub)

    # By tier
    stats["by_tier"] = {}
    for tier in ["VALIDATED", "ACTIVE", "OPPORTUNITY", "WATCHLIST"]:
        sub = df[df["tier"] == tier]
        if len(sub) > 0:
            stats["by_tier"][tier] = _calc_group(sub)

    # By regime
    stats["by_regime"] = {}
    for r in df["regime"].unique():
        sub = df[df["regime"] == r]
        stats["by_regime"][r] = _calc_group(sub)

    return stats


# ---------------------------------------------------------------------------
# Threshold recommendations
# ---------------------------------------------------------------------------

def threshold_recommendations(records: list[dict]) -> list[str]:
    """Analyze if current thresholds are good or need adjustment."""
    if not records:
        return ["No data to analyze."]

    df = pd.DataFrame(records)
    recs = []

    for sig in sorted(df["signal_type"].unique()):
        sub = df[(df["signal_type"] == sig) & df["return_3d"].notna()]
        if len(sub) < 5:
            continue

        wr_3d = sub["win_3d"].sum() / len(sub) * 100
        avg_3d = sub["return_3d"].mean()
        n = len(sub)

        cutoff_stats = []
        for cutoff in [4.0, 5.0, 6.0, 7.0, 8.0, 10.0]:
            above = sub[sub["strength"] >= cutoff]
            if len(above) >= 3:
                wr = above["win_3d"].sum() / len(above) * 100
                cutoff_stats.append((cutoff, len(above), round(wr, 1)))

        current_floor = _validated_min_strength(50.0, sig)
        status = "OK" if wr_3d >= 50 else ("WEAK" if wr_3d >= 40 else "BAD")
        sample_note = " *" if n < MIN_STATISTICAL_SAMPLES else ""

        line = f"  {sig:<12} n={n:>4}{sample_note}  WR-3D={wr_3d:5.1f}%  Avg={avg_3d:+6.3f}%  Floor={current_floor:.1f}  [{status}]"

        if cutoff_stats:
            best = max(cutoff_stats, key=lambda x: x[2])
            if best[2] > wr_3d + 3 and best[1] >= 5:
                line += f"  -> Raise to {best[0]:.0f} ({best[2]:.0f}% WR on {best[1]} signals)"
            elif wr_3d < 45:
                line += "  -> Consider removing or tightening"

        recs.append(line)

    recs.append("")
    recs.append("  GRADE VALIDATION:")
    for g in ["A", "B", "C"]:
        sub = df[(df["grade"] == g) & df["return_3d"].notna()]
        if len(sub) >= 3:
            wr = sub["win_3d"].sum() / len(sub) * 100
            avg = sub["return_3d"].mean()
            valid = "GOOD" if (g == "A" and wr > 55) or (g == "B" and wr > 48) or (g == "C" and wr > 40) else "REVIEW"
            recs.append(f"    Grade {g}: n={len(sub):>4}  WR-3D={wr:5.1f}%  Avg={avg:+6.3f}%  [{valid}]")

    grade_wrs = {}
    for g in ["A", "B", "C"]:
        sub = df[(df["grade"] == g) & df["return_3d"].notna()]
        if len(sub) >= 3:
            grade_wrs[g] = sub["win_3d"].sum() / len(sub) * 100

    if "A" in grade_wrs and "C" in grade_wrs:
        if grade_wrs["A"] <= grade_wrs["C"]:
            recs.append("    [!] Grade A does NOT outperform Grade C -- grading formula needs work")
        else:
            diff = grade_wrs["A"] - grade_wrs["C"]
            recs.append(f"    Grade spread: A is +{diff:.1f}pp above C -- {'good' if diff > 5 else 'narrow'}")

    recs.append("")
    recs.append("  (* = fewer than 30 samples, treat with caution)")

    return recs


# ---------------------------------------------------------------------------
# Equity curve ASCII visualization
# ---------------------------------------------------------------------------

def print_equity_curve(equity_data: dict, width: int = 50):
    """Print ASCII equity curve to terminal."""
    if not equity_data:
        return

    dates = sorted(equity_data.keys())
    values = [equity_data[d] for d in dates]

    min_val = min(values)
    max_val = max(values)
    spread = max_val - min_val or 0.01

    print(f"\n  EQUITY CURVE ({len(dates)} trading days)")
    print(f"  {'=' * (width + 22)}")

    # Sample ~20 rows for display
    step = max(1, len(dates) // 20)
    for i in range(0, len(dates), step):
        d = dates[i]
        v = values[i]
        bar_len = max(0, int((v - min_val) / spread * width))
        ret_pct = (v - 1) * 100
        bar = "#" * bar_len
        print(f"  {d[:10]}  {ret_pct:+7.2f}%  |{bar}")

    # Final value
    final_ret = (values[-1] - 1) * 100
    print(f"  {'=' * (width + 22)}")
    print(f"  Final cumulative: {final_ret:+.2f}%")


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def print_report(stats: dict, recs: list[str], risk: dict = None):
    """Print formatted backtest report."""
    print()
    print("=" * 78)
    print("  TITAN SIGNAL BACKTESTER — Production-Grade Validation")
    print("  Entry: next-day open + slippage + commission")
    print("=" * 78)

    if stats["total"] == 0:
        print("  No signals found in backtest period.")
        return

    o = stats["overall"]
    print(f"\n  Total signals: {o['count']}")
    if o.get("low_sample"):
        print(f"  [!] WARNING: Sample size below {MIN_STATISTICAL_SAMPLES} — results unreliable!")

    print(f"\n  OVERALL PERFORMANCE (after slippage + commission)")
    print(f"  {'':>12} {'Win Rate':>10} {'Avg Ret':>10} {'Med Ret':>10}")
    print(f"  {'-' * 45}")
    for d in FORWARD_WINDOWS:
        wr = o.get(f"wr_{d}d")
        avg = o.get(f"avg_{d}d")
        med = o.get(f"med_{d}d")
        wr_s = f"{wr:.1f}%" if wr is not None else "n/a"
        avg_s = f"{avg:+.3f}%" if avg is not None else "n/a"
        med_s = f"{med:+.3f}%" if med is not None else "n/a"
        print(f"  {'Return ' + str(d) + 'D':>12} {wr_s:>10} {avg_s:>10} {med_s:>10}")

    tgt = o.get("target_hit_5d")
    stp = o.get("stop_hit_5d")
    if tgt is not None:
        print(f"\n  Target hit (5D): {tgt:.1f}%  |  Stop hit (5D): {stp:.1f}%")

    # --- RISK METRICS (the real test) ---
    if risk:
        print(f"\n  {'=' * 60}")
        print(f"  RISK METRICS — Would you trust this with real money?")
        print(f"  {'=' * 60}")

        pf = risk["profit_factor"]
        sr = risk["sharpe_ratio"]
        mdd = risk["max_drawdown_pct"]
        ws = risk["worst_losing_streak"]
        exp = risk["expectancy_pct"]

        # Traffic light system
        pf_grade = "PASS" if pf > 1.3 else ("MARGINAL" if pf > 1.1 else "FAIL")
        sr_grade = "PASS" if sr > 0.5 else ("MARGINAL" if sr > 0.2 else "FAIL")
        mdd_grade = "PASS" if mdd > -10 else ("MARGINAL" if mdd > -20 else "FAIL")
        exp_grade = "PASS" if exp > 0.05 else ("MARGINAL" if exp > 0 else "FAIL")

        print(f"\n  Profit Factor:     {pf:>8.2f}   [{pf_grade}]  (need > 1.3)")
        print(f"  Sharpe Ratio:      {sr:>8.2f}   [{sr_grade}]  (need > 0.5)")
        print(f"  Max Drawdown:      {mdd:>7.2f}%   [{mdd_grade}]  (< -10% is concerning)")
        print(f"  Worst Lose Streak: {ws:>8d}   trades in a row")
        print(f"  Expectancy/trade:  {exp:>7.3f}%   [{exp_grade}]  (must be > 0)")
        print(f"  Avg Win:           {risk['avg_win_pct']:>7.3f}%")
        print(f"  Avg Loss:          {risk['avg_loss_pct']:>7.3f}%")
        print(f"  Win/Loss Ratio:    {risk['win_loss_ratio']:>8.2f}")
        print(f"  Best Trade:        {risk['best_trade_pct']:>+7.2f}%")
        print(f"  Worst Trade:       {risk['worst_trade_pct']:>+7.2f}%")
        print(f"  DD Duration:       {risk['max_dd_duration_days']:>8d}   trading days")
        print(f"  Profitable Months: {risk['profitable_months']}/{risk['total_months']}")

        # Overall verdict
        passes = sum(1 for g in [pf_grade, sr_grade, mdd_grade, exp_grade] if g == "PASS")
        fails = sum(1 for g in [pf_grade, sr_grade, mdd_grade, exp_grade] if g == "FAIL")

        print(f"\n  {'─' * 60}")
        if fails > 0:
            print(f"  VERDICT: NOT READY for real money ({fails} FAIL)")
        elif passes == 4:
            print(f"  VERDICT: PASSES all checks — paper trade for 30 days to confirm")
        else:
            print(f"  VERDICT: MARGINAL — proceed with SMALL size only")

        # Equity curve
        print_equity_curve(risk.get("equity_curve", {}))

        # Monthly breakdown
        if risk.get("monthly"):
            print(f"\n  MONTHLY BREAKDOWN")
            print(f"  {'Month':<12} {'Trades':>7} {'WR':>7} {'Avg Ret':>9}")
            print(f"  {'-' * 38}")
            for month, data in sorted(risk["monthly"].items(), key=lambda x: str(x[0])):
                wr = data.get("wr", 0)
                avg = data.get("avg_ret", 0)
                n = data.get("trades", 0)
                print(f"  {str(month):<12} {n:>7} {wr:>6.1f}% {avg:>+8.3f}%")

    # By signal type
    print(f"\n  BY SIGNAL TYPE")
    print(f"  {'Signal':<12} {'Count':>6} {'WR-1D':>7} {'WR-3D':>7} {'WR-5D':>7} {'Avg-3D':>8} {'Tgt%':>6} {'Stp%':>6}")
    print(f"  {'-' * 65}")
    for sig, s in sorted(stats["by_signal_type"].items(), key=lambda x: -(x[1].get("wr_3d") or 0)):
        star = " *" if s.get("low_sample") else ""
        wr1 = f"{s['wr_1d']:.1f}%" if s.get("wr_1d") is not None else "n/a"
        wr3 = f"{s['wr_3d']:.1f}%" if s.get("wr_3d") is not None else "n/a"
        wr5 = f"{s['wr_5d']:.1f}%" if s.get("wr_5d") is not None else "n/a"
        avg3 = f"{s['avg_3d']:+.3f}%" if s.get("avg_3d") is not None else "n/a"
        tgt = f"{s['target_hit_5d']:.0f}%" if s.get("target_hit_5d") is not None else "n/a"
        stp = f"{s['stop_hit_5d']:.0f}%" if s.get("stop_hit_5d") is not None else "n/a"
        print(f"  {sig + star:<12} {s['count']:>6} {wr1:>7} {wr3:>7} {wr5:>7} {avg3:>8} {tgt:>6} {stp:>6}")

    # By grade
    print(f"\n  BY GRADE")
    print(f"  {'Grade':<8} {'Count':>6} {'WR-1D':>7} {'WR-3D':>7} {'WR-5D':>7} {'Avg-3D':>8} {'Avg-5D':>8}")
    print(f"  {'-' * 55}")
    for g in ["A", "B", "C"]:
        s = stats["by_grade"].get(g)
        if not s:
            continue
        wr1 = f"{s['wr_1d']:.1f}%" if s.get("wr_1d") is not None else "n/a"
        wr3 = f"{s['wr_3d']:.1f}%" if s.get("wr_3d") is not None else "n/a"
        wr5 = f"{s['wr_5d']:.1f}%" if s.get("wr_5d") is not None else "n/a"
        avg3 = f"{s['avg_3d']:+.3f}%" if s.get("avg_3d") is not None else "n/a"
        avg5 = f"{s['avg_5d']:+.3f}%" if s.get("avg_5d") is not None else "n/a"
        print(f"  {g:<8} {s['count']:>6} {wr1:>7} {wr3:>7} {wr5:>7} {avg3:>8} {avg5:>8}")

    # By tier
    print(f"\n  BY TIER")
    print(f"  {'Tier':<14} {'Count':>6} {'WR-1D':>7} {'WR-3D':>7} {'WR-5D':>7} {'Avg-3D':>8} {'Avg-5D':>8}")
    print(f"  {'-' * 60}")
    for tier in ["VALIDATED", "ACTIVE", "OPPORTUNITY", "WATCHLIST"]:
        s = stats["by_tier"].get(tier)
        if not s:
            continue
        star = " *" if s.get("low_sample") else ""
        wr1 = f"{s['wr_1d']:.1f}%" if s.get("wr_1d") is not None else "n/a"
        wr3 = f"{s['wr_3d']:.1f}%" if s.get("wr_3d") is not None else "n/a"
        wr5 = f"{s['wr_5d']:.1f}%" if s.get("wr_5d") is not None else "n/a"
        avg3 = f"{s['avg_3d']:+.3f}%" if s.get("avg_3d") is not None else "n/a"
        avg5 = f"{s['avg_5d']:+.3f}%" if s.get("avg_5d") is not None else "n/a"
        print(f"  {tier + star:<14} {s['count']:>6} {wr1:>7} {wr3:>7} {wr5:>7} {avg3:>8} {avg5:>8}")

    # By regime
    print(f"\n  BY REGIME")
    print(f"  {'Regime':<14} {'Count':>6} {'WR-3D':>7} {'Avg-3D':>8}")
    print(f"  {'-' * 40}")
    for r, s in sorted(stats["by_regime"].items(), key=lambda x: -x[1]["count"]):
        wr3 = f"{s['wr_3d']:.1f}%" if s.get("wr_3d") is not None else "n/a"
        avg3 = f"{s['avg_3d']:+.3f}%" if s.get("avg_3d") is not None else "n/a"
        print(f"  {r:<14} {s['count']:>6} {wr3:>7} {avg3:>8}")

    # Recommendations
    print(f"\n  THRESHOLD ANALYSIS")
    print(f"  {'-' * 75}")
    for line in recs:
        print(line)

    print()
    print("=" * 78)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Titan Signal Backtester")
    parser.add_argument("--days", type=int, default=9999, help="Trading days to backtest (default: all available)")
    parser.add_argument("--csv", type=str, default=None, help="Export raw signals to CSV")
    args = parser.parse_args()

    print("\n" + "=" * 78)
    print("  TITAN SIGNAL BACKTESTER — Production Grade")
    print("=" * 78)

    print("  Loading data...")
    data = load_cached_data()
    print(f"  Data: {len(data.index)} days, {len(data.columns.levels[0])} tickers")
    print(f"  Range: {data.index[0].date()} to {data.index[-1].date()}")

    records = run_backtest(data, lookback_days=args.days)
    stats = compute_statistics(records)
    risk = compute_risk_metrics(records)
    recs = threshold_recommendations(records)
    print_report(stats, recs, risk)

    if args.csv:
        pd.DataFrame(records).to_csv(args.csv, index=False)
        print(f"  Exported {len(records)} signals to {args.csv}")
