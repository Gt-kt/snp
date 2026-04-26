"""
Short-horizon edge profiles and family prior system.
Ported from KOSPI architecture — adapted for US market (S&P 500).
"""

import math
import pandas as pd
import numpy as np

from titan.config import (
    DEFAULT_COMMISSION_BPS,
    DEFAULT_SLIPPAGE_BREAKOUT_BPS,
    DEFAULT_SLIPPAGE_DIP_BPS,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SHORT_EDGE_HORIZONS = (1, 3, 5)
SHORT_EDGE_LOOKBACK_DAYS = 504          # ~2 years of trading days
SHORT_EDGE_MIN_ANALOGS = 4              # Minimum analog matches to trust
SHORT_EDGE_TOP_MATCHES = 12             # Keep closest 12 analogs
SHORT_EDGE_MIN_EXPECTED_RETURN = 0.10   # 0.10% minimum expected return
SHORT_EDGE_MIN_UP_PROB = 52.0           # 52% win probability minimum

FAMILY_PRIOR_MIN_TRADES = 200
FAMILY_PRIOR_MIN_OOS_TRADES = 60
FAMILY_PRIOR_MIN_EXPECTED_RETURN = 0.10
FAMILY_PRIOR_MIN_UP_PROB = 48.0

PRO_MIN_BT_TRADES = 100

# Round-trip cost for US stocks: slippage in + commission in + commission out + slippage out
_ROUND_TRIP_COST_PCT = (
    (DEFAULT_SLIPPAGE_DIP_BPS + DEFAULT_COMMISSION_BPS) * 2
) / 100  # ~0.03%

DEFAULT_WARMUP = 60
DEFAULT_TRAIN_SPLIT = 0.70

# Features for analog distance measurement
_DISTANCE_FEATURES = {
    "change_1d": 2.5,
    "change_5d": 5.0,
    "change_20d": 12.0,
    "rsi": 15.0,
    "volume_ratio": 2.5,
    "dist_from_high": 6.0,
    "vol_contraction": 0.60,
}

# Horizon weights: 1D=55%, 3D=30%, 5D=15%
_HORIZON_WEIGHTS = {1: 0.55, 3: 0.30, 5: 0.15}

# Signal type sets (imported lazily to avoid circular imports)
EARLY_SIGNAL_TYPES = {
    "VCP", "PRE_BREAK", "ACCUM", "PULLBACK", "REL_STR", "MOM_RANK",
}
BREAKOUT_SIGNAL_TYPES = {"BREAKOUT", "MOMENTUM"}


# ---------------------------------------------------------------------------
# Signal state distance (analog matching)
# ---------------------------------------------------------------------------

def signal_state_distance(current: dict, historical: dict) -> float:
    """Measure how similar a historical setup is to the current state.

    Uses 7 normalized features.  Returns average normalized absolute
    difference (0 = identical, 1 = maximally different).
    """
    total = 0.0
    count = 0
    for feat, scale in _DISTANCE_FEATURES.items():
        cur_val = current.get(feat)
        hist_val = historical.get(feat)
        if cur_val is None or hist_val is None:
            continue
        try:
            cur_f = float(cur_val)
            hist_f = float(hist_val)
        except (TypeError, ValueError):
            continue
        if math.isnan(cur_f) or math.isnan(hist_f):
            continue
        total += abs(cur_f - hist_f) / scale
        count += 1
    return total / count if count > 0 else 1.0


# ---------------------------------------------------------------------------
# Short-horizon trade summarization
# ---------------------------------------------------------------------------

def summarize_short_horizon_trades(
    trades: list[dict],
    weight_field: str | None = None,
) -> dict:
    """Aggregate a list of trades into weighted per-horizon statistics.

    Each trade dict must have keys like ``return_1d``, ``return_3d``,
    ``return_5d``, and optionally ``open_gap_1d``.
    """
    result: dict = {"sample": len(trades)}
    if not trades:
        for h in SHORT_EDGE_HORIZONS:
            result[f"avg_return_{h}d"] = 0.0
            result[f"up_prob_{h}d"] = 0.0
        result.update(
            avg_open_gap_1d=0.0, open_above_close_prob=0.0,
            gap_above_1pct_prob=0.0, weighted_return=0.0,
            weighted_up_prob=0.0, best_horizon=1,
        )
        return result

    # Per-horizon metrics ---------------------------------------------------
    available_horizons = []
    for h in SHORT_EDGE_HORIZONS:
        key = f"return_{h}d"
        vals, weights = [], []
        for t in trades:
            v = t.get(key)
            if v is None:
                continue
            w = 1.0
            if weight_field:
                w = max(0.01, float(t.get(weight_field, 1.0)))
            vals.append((float(v), w))
            weights.append(w)
        if not vals:
            result[f"avg_return_{h}d"] = 0.0
            result[f"up_prob_{h}d"] = 0.0
            continue
        total_w = sum(w for _, w in vals)
        avg_ret = sum(v * w for v, w in vals) / total_w
        up_prob = 100.0 * sum(w for v, w in vals if v > 0) / total_w
        result[f"avg_return_{h}d"] = round(avg_ret, 4)
        result[f"up_prob_{h}d"] = round(up_prob, 2)
        available_horizons.append(h)

    # Gap analysis ----------------------------------------------------------
    gap_vals = []
    for t in trades:
        g = t.get("open_gap_1d")
        if g is not None:
            gap_vals.append(float(g))
    if gap_vals:
        result["avg_open_gap_1d"] = round(sum(gap_vals) / len(gap_vals), 4)
        result["open_above_close_prob"] = round(
            100.0 * sum(1 for g in gap_vals if g > 0) / len(gap_vals), 2
        )
        result["gap_above_1pct_prob"] = round(
            100.0 * sum(1 for g in gap_vals if g > 1.0) / len(gap_vals), 2
        )
    else:
        result["avg_open_gap_1d"] = 0.0
        result["open_above_close_prob"] = 0.0
        result["gap_above_1pct_prob"] = 0.0

    # Weighted composite ----------------------------------------------------
    if not available_horizons:
        result["weighted_return"] = 0.0
        result["weighted_up_prob"] = 0.0
        result["best_horizon"] = 1
        return result

    w_ret = 0.0
    w_up = 0.0
    w_total = 0.0
    for h in available_horizons:
        hw = _HORIZON_WEIGHTS.get(h, 0.0)
        w_ret += result[f"avg_return_{h}d"] * hw
        w_up += result[f"up_prob_{h}d"] * hw
        w_total += hw
    if w_total > 0:
        w_ret /= w_total
        w_up /= w_total

    result["weighted_return"] = round(w_ret, 4)
    result["weighted_up_prob"] = round(w_up, 2)

    # Best horizon: maximize (return * 4) + ((up_prob - 50) / 5)
    result["best_horizon"] = max(
        available_horizons,
        key=lambda h: (result[f"avg_return_{h}d"] * 4.0)
        + ((result[f"up_prob_{h}d"] - 50.0) / 5.0),
    )
    return result


# ---------------------------------------------------------------------------
# Profile scoring
# ---------------------------------------------------------------------------

def score_short_horizon_profile(profile: dict) -> float:
    """Score a short-horizon profile: higher = better edge."""
    wr = profile.get("weighted_return", 0.0)
    up = profile.get("weighted_up_prob", 50.0)
    sample = profile.get("sample", 0)
    bh = profile.get("best_horizon", 3)

    score = ((up - 50.0) / 4.0) + (wr * 4.0) + min(2.5, sample / 6.0)
    if bh == 1 and wr > 0:
        score += 0.5
    return round(score, 2)


# ---------------------------------------------------------------------------
# Per-signal mini-backtest
# ---------------------------------------------------------------------------

def backtest_signal(
    df: pd.DataFrame,
    signal_type: str,
    technicals_current: dict | None = None,
    lookback_days: int = SHORT_EDGE_LOOKBACK_DAYS,
) -> dict:
    """Run a lightweight mini-backtest using analog matching.

    Instead of re-running detect_signal() at every historical bar (very slow),
    this finds historical bars whose technical fingerprint is similar to the
    current setup (using RSI, volume_ratio, vol_contraction, changes, etc.)
    and measures their forward returns at 1/3/5-day horizons.

    Returns a dict with trade-level stats, OOS stats, and a profile section.
    """
    empty = _empty_backtest_result()
    if df is None or df.empty or technicals_current is None:
        return empty

    min_bars = max(90, DEFAULT_WARMUP + 5 + max(SHORT_EDGE_HORIZONS))
    if len(df) < min_bars:
        return empty

    close = df["Close"]
    open_ = df["Open"] if "Open" in df.columns else close
    high = df["High"] if "High" in df.columns else close
    volume = df["Volume"] if "Volume" in df.columns else pd.Series(0, index=df.index)

    n = len(df)
    oos_start = int(n * DEFAULT_TRAIN_SPLIT)
    start_idx = max(DEFAULT_WARMUP, n - max(lookback_days, SHORT_EDGE_LOOKBACK_DAYS) - max(SHORT_EDGE_HORIZONS))

    # Pre-compute rolling indicators (vectorized, fast)
    sma20 = close.rolling(20).mean()
    high_20 = high.rolling(20).max()
    avg_vol = volume.rolling(20).mean()
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi_series = 100 - (100 / (1 + rs))
    atr5 = (high - df["Low"]).rolling(5).mean() if "Low" in df.columns else close * 0
    atr20 = (high - df["Low"]).rolling(20).mean() if "Low" in df.columns else close * 0

    trades: list[dict] = []

    for i in range(start_idx, n - max(SHORT_EDGE_HORIZONS)):
        c = float(close.iloc[i])
        if c <= 0:
            continue

        # Build technical snapshot for this bar
        hist_t = {}
        try:
            c_prev1 = float(close.iloc[i - 1]) if i >= 1 else c
            c_prev5 = float(close.iloc[i - 5]) if i >= 5 else c
            c_prev20 = float(close.iloc[i - 20]) if i >= 20 else c
            hist_t["change_1d"] = (c - c_prev1) / c_prev1 * 100 if c_prev1 > 0 else 0
            hist_t["change_5d"] = (c - c_prev5) / c_prev5 * 100 if c_prev5 > 0 else 0
            hist_t["change_20d"] = (c - c_prev20) / c_prev20 * 100 if c_prev20 > 0 else 0
            hist_t["rsi"] = float(rsi_series.iloc[i]) if not pd.isna(rsi_series.iloc[i]) else 50
            h20 = float(high_20.iloc[i]) if not pd.isna(high_20.iloc[i]) else c
            hist_t["dist_from_high"] = (h20 - c) / h20 * 100 if h20 > 0 else 0
            av = float(avg_vol.iloc[i]) if not pd.isna(avg_vol.iloc[i]) else 1
            hist_t["volume_ratio"] = float(volume.iloc[i]) / av if av > 0 else 1
            a5 = float(atr5.iloc[i]) if not pd.isna(atr5.iloc[i]) else 0
            a20 = float(atr20.iloc[i]) if not pd.isna(atr20.iloc[i]) else 1
            hist_t["vol_contraction"] = a5 / a20 if a20 > 0 else 1
        except (IndexError, ValueError):
            continue

        # Compute distance to current setup
        dist = signal_state_distance(technicals_current, hist_t)
        if dist > 0.8:
            continue  # Too different, skip

        entry_price = c

        trade: dict = {
            "bar_idx": i,
            "entry": entry_price,
            "is_oos": i >= oos_start,
            "distance": dist,
            "similarity_weight": 1.0 / (1.0 + dist),
        }

        # Open gap
        if i + 1 < n:
            next_open = float(open_.iloc[i + 1])
            trade["open_gap_1d"] = round(
                (next_open - entry_price) / entry_price * 100, 4
            ) if next_open > 0 else 0.0
        else:
            trade["open_gap_1d"] = 0.0

        # Forward returns
        if signal_type in ("BREAKOUT", "VOL_SPIKE"):
            trade_cost = (DEFAULT_SLIPPAGE_BREAKOUT_BPS + DEFAULT_COMMISSION_BPS) * 2 / 100
        else:
            trade_cost = _ROUND_TRIP_COST_PCT

        for h in SHORT_EDGE_HORIZONS:
            future_idx = i + h
            if future_idx >= n:
                continue
            future_close = float(close.iloc[future_idx])
            if future_close <= 0:
                trade[f"return_{h}d"] = 0.0
                continue
            raw_return = (future_close - entry_price) / entry_price * 100
            trade[f"return_{h}d"] = round(raw_return - trade_cost, 4)

        # Weighted return from horizons that actually have data
        w_ret_sum = 0.0
        w_ret_total = 0.0
        for h, hw in _HORIZON_WEIGHTS.items():
            key = f"return_{h}d"
            if key in trade:
                w_ret_sum += trade[key] * hw
                w_ret_total += hw
        trade["return"] = round(w_ret_sum / w_ret_total, 4) if w_ret_total > 0 else 0.0
        trade["win"] = trade["return"] > 0

        trades.append(trade)

    if len(trades) < 3:
        return empty

    # Overall stats
    overall = summarize_short_horizon_trades(trades)

    # OOS stats
    oos_trades = [t for t in trades if t.get("is_oos")]
    oos = summarize_short_horizon_trades(oos_trades) if oos_trades else {}

    # Profile: closest analogs (weighted by similarity)
    sorted_by_dist = sorted(trades, key=lambda t: t.get("distance", 1.0))
    analog_trades = sorted_by_dist[:SHORT_EDGE_TOP_MATCHES]
    profile = summarize_short_horizon_trades(
        analog_trades, weight_field="similarity_weight"
    )
    profile["analog_samples"] = len(analog_trades)
    profile["score"] = score_short_horizon_profile(profile)

    # Validation flags
    if len(oos_trades) >= 3:
        validated = (
            oos.get("weighted_up_prob", 0) >= 50
            and oos.get("weighted_return", 0) > 0
        )
    else:
        validated = (
            overall.get("weighted_up_prob", 0) >= 55
            and overall.get("weighted_return", 0) > 0.3
        )

    profile_validated = (
        profile["analog_samples"] >= SHORT_EDGE_MIN_ANALOGS
        and profile.get("weighted_return", 0) >= SHORT_EDGE_MIN_EXPECTED_RETURN
        and profile.get("weighted_up_prob", 0) >= SHORT_EDGE_MIN_UP_PROB
    )

    return {
        "trades": len(trades),
        "win_rate": overall.get("weighted_up_prob", 0),
        "avg_return": overall.get("weighted_return", 0),
        "validated": validated or profile_validated,
        "avg_return_1d": overall.get("avg_return_1d", 0),
        "avg_return_3d": overall.get("avg_return_3d", 0),
        "avg_return_5d": overall.get("avg_return_5d", 0),
        "up_prob_1d": overall.get("up_prob_1d", 0),
        "up_prob_3d": overall.get("up_prob_3d", 0),
        "up_prob_5d": overall.get("up_prob_5d", 0),
        "best_horizon": overall.get("best_horizon", 1),
        "oos_trades": len(oos_trades),
        "oos_win_rate": oos.get("weighted_up_prob", 0),
        "oos_avg_return": oos.get("weighted_return", 0),
        "avg_open_gap_1d": overall.get("avg_open_gap_1d", 0),
        "gap_above_1pct_prob": overall.get("gap_above_1pct_prob", 0),
        "profile_analog_samples": profile.get("analog_samples", 0),
        "profile_weighted_return": profile.get("weighted_return", 0),
        "profile_weighted_up_prob": profile.get("weighted_up_prob", 0),
        "profile_best_horizon": profile.get("best_horizon", 1),
        "profile_score": profile.get("score", 0),
        "profile_validated": profile_validated,
    }


def _empty_backtest_result() -> dict:
    return {
        "trades": 0, "win_rate": 0, "avg_return": 0, "validated": False,
        "avg_return_1d": 0, "avg_return_3d": 0, "avg_return_5d": 0,
        "up_prob_1d": 0, "up_prob_3d": 0, "up_prob_5d": 0,
        "best_horizon": 1,
        "oos_trades": 0, "oos_win_rate": 0, "oos_avg_return": 0,
        "avg_open_gap_1d": 0, "gap_above_1pct_prob": 0,
        "profile_analog_samples": 0, "profile_weighted_return": 0,
        "profile_weighted_up_prob": 0, "profile_best_horizon": 1,
        "profile_score": 0, "profile_validated": False,
    }


# ---------------------------------------------------------------------------
# Family prior (cross-sectional evidence)
# ---------------------------------------------------------------------------

def empty_signal_family_prior(signal_type: str) -> dict:
    """Return a blank family prior template."""
    return {
        "signal_type": signal_type,
        "stocks": 0,
        "total_trades": 0,
        "weighted_up_prob": 0.0,
        "weighted_return": 0.0,
        "oos_trades": 0,
        "oos_up_prob": 0.0,
        "oos_return": 0.0,
        "best_horizon": 1,
        "avg_open_gap_1d": 0.0,
        "gap_above_1pct_prob": 0.0,
        "reference_up_prob": 0.0,
        "reference_return": 0.0,
        "confidence": "insufficient",
        "supports_thin_history": False,
    }


def summarize_signal_family_prior(
    signal_type: str,
    backtest_results: list[dict],
) -> dict:
    """Aggregate per-stock backtest results into a cross-sectional family prior."""
    prior = empty_signal_family_prior(signal_type)

    stock_count = 0
    total_trades = 0
    total_oos_trades = 0
    w_return = 0.0
    w_up_prob = 0.0
    w_gap = 0.0
    w_gap_1pct = 0.0
    w_oos_return = 0.0
    w_oos_up_prob = 0.0
    horizon_weights: dict[int, int] = {}

    for bt in backtest_results:
        trades = int(bt.get("trades", 0) or 0)
        if trades <= 0:
            continue
        stock_count += 1
        total_trades += trades
        w_return += bt.get("avg_return", 0) * trades
        w_up_prob += bt.get("win_rate", 0) * trades
        w_gap += bt.get("avg_open_gap_1d", 0) * trades
        w_gap_1pct += bt.get("gap_above_1pct_prob", 0) * trades

        bh = bt.get("best_horizon", 1)
        horizon_weights[bh] = horizon_weights.get(bh, 0) + trades

        oos_t = int(bt.get("oos_trades", 0) or 0)
        if oos_t > 0:
            total_oos_trades += oos_t
            w_oos_return += bt.get("oos_avg_return", 0) * oos_t
            w_oos_up_prob += bt.get("oos_win_rate", 0) * oos_t

    if total_trades == 0:
        return prior

    prior["stocks"] = stock_count
    prior["total_trades"] = total_trades
    prior["weighted_up_prob"] = round(w_up_prob / total_trades, 2)
    prior["weighted_return"] = round(w_return / total_trades, 4)
    prior["avg_open_gap_1d"] = round(w_gap / total_trades, 4)
    prior["gap_above_1pct_prob"] = round(w_gap_1pct / total_trades, 2)
    prior["best_horizon"] = max(horizon_weights, key=horizon_weights.get) if horizon_weights else 1
    prior["oos_trades"] = total_oos_trades

    if total_oos_trades > 0:
        prior["oos_up_prob"] = round(w_oos_up_prob / total_oos_trades, 2)
        prior["oos_return"] = round(w_oos_return / total_oos_trades, 4)

    # Reference: prefer OOS if enough data
    use_oos = total_oos_trades >= FAMILY_PRIOR_MIN_OOS_TRADES
    prior["reference_up_prob"] = prior["oos_up_prob"] if use_oos else prior["weighted_up_prob"]
    prior["reference_return"] = prior["oos_return"] if use_oos else prior["weighted_return"]

    # Confidence classification
    supports_thin = (
        total_trades >= FAMILY_PRIOR_MIN_TRADES
        and prior["weighted_return"] >= FAMILY_PRIOR_MIN_EXPECTED_RETURN
        and prior["reference_return"] >= FAMILY_PRIOR_MIN_EXPECTED_RETURN
        and prior["reference_up_prob"] >= FAMILY_PRIOR_MIN_UP_PROB
    )
    clearly_negative = (
        total_trades >= 100
        and prior["weighted_return"] < 0
        and prior["reference_return"] < 0
    )

    if supports_thin:
        prior["confidence"] = "positive"
        prior["supports_thin_history"] = True
    elif clearly_negative:
        prior["confidence"] = "negative"
    else:
        prior["confidence"] = "neutral"

    return prior


def build_signal_family_prior(
    stock_frames: dict[str, pd.DataFrame],
    signal_type: str,
    lookback_days: int = 252,
    max_stocks: int = 30,
) -> dict:
    """Build a family prior by running mini-backtests across a sample of stocks.

    Limited to max_stocks to keep runtime practical (~30 stocks × analog scan).
    """
    results: list[dict] = []
    # Use a deterministic sample of the most liquid stocks
    frames_list = list(stock_frames.items())[:max_stocks]
    for ticker, df in frames_list:
        try:
            # For family prior, we use a generic technical state (median values)
            generic_t = {
                "change_1d": 0, "change_5d": 0, "change_20d": 5,
                "rsi": 50, "volume_ratio": 1.0, "dist_from_high": 5,
                "vol_contraction": 0.7,
            }
            bt = backtest_signal(df, signal_type, technicals_current=generic_t,
                                 lookback_days=lookback_days)
        except Exception:
            continue
        if int(bt.get("trades", 0) or 0) > 0:
            results.append(bt)
    return summarize_signal_family_prior(signal_type, results)


def family_prior_supports_thin_history(family_prior: dict | None) -> bool:
    """Does the family prior have enough positive evidence?"""
    return bool(family_prior) and bool(family_prior.get("supports_thin_history"))


def family_prior_is_negative(family_prior: dict | None) -> bool:
    """Does the family prior have negative confidence?"""
    return (
        bool(family_prior)
        and str(family_prior.get("confidence", "")).lower() == "negative"
    )


# ---------------------------------------------------------------------------
# Move profile resolution (priority cascade)
# ---------------------------------------------------------------------------

def resolve_signal_move_profile(
    backtest_result: dict,
    family_prior: dict | None = None,
    allow_family: bool = False,
) -> dict:
    """Resolve the best available move expectancy source.

    Priority: profile → stock OOS → family → stock overall → empty.
    """
    bt = backtest_result or {}

    # 1. Profile (analog-matched)
    p_samples = int(bt.get("profile_analog_samples", 0) or 0)
    if p_samples >= SHORT_EDGE_MIN_ANALOGS:
        return {
            "source": "profile",
            "samples": p_samples,
            "up_prob": bt.get("profile_weighted_up_prob", 0),
            "expected_return": bt.get("profile_weighted_return", 0),
            "best_horizon": bt.get("profile_best_horizon", 1),
            "avg_open_gap_1d": bt.get("avg_open_gap_1d", 0),
            "gap_above_1pct_prob": bt.get("gap_above_1pct_prob", 0),
        }

    # 2. OOS (out-of-sample stock-level)
    oos_t = int(bt.get("oos_trades", 0) or 0)
    if oos_t >= 3 and bt.get("oos_avg_return", 0) > 0 and bt.get("oos_win_rate", 0) >= 50:
        return {
            "source": "stock_oos",
            "samples": oos_t,
            "up_prob": bt.get("oos_win_rate", 0),
            "expected_return": bt.get("oos_avg_return", 0),
            "best_horizon": bt.get("best_horizon", 1),
            "avg_open_gap_1d": bt.get("avg_open_gap_1d", 0),
            "gap_above_1pct_prob": bt.get("gap_above_1pct_prob", 0),
        }

    # 3. Family prior (cross-sectional)
    if allow_family and family_prior_supports_thin_history(family_prior):
        return {
            "source": "family",
            "samples": family_prior.get("stocks", 0),
            "up_prob": family_prior.get("reference_up_prob", 0),
            "expected_return": family_prior.get("reference_return", 0),
            "best_horizon": family_prior.get("best_horizon", 1),
            "avg_open_gap_1d": family_prior.get("avg_open_gap_1d", 0),
            "gap_above_1pct_prob": family_prior.get("gap_above_1pct_prob", 0),
        }

    # 4. Stock overall (fallback)
    total = int(bt.get("trades", 0) or 0)
    if total > 0:
        return {
            "source": "stock",
            "samples": total,
            "up_prob": bt.get("win_rate", 0),
            "expected_return": bt.get("avg_return", 0),
            "best_horizon": bt.get("best_horizon", 1),
            "avg_open_gap_1d": bt.get("avg_open_gap_1d", 0),
            "gap_above_1pct_prob": bt.get("gap_above_1pct_prob", 0),
        }

    return {}
