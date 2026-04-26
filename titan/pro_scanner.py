"""
Titan Pro Scanner — KOSPI-Architecture S&P 500 Scanner
======================================================
Multi-signal detection with 4-tier classification, execution plans,
anti-chase protection, and forward tracking.

This replaces the old single-signal breakout scanner with the proven
architecture from the KOSPI trading system.

Tiers:
  VALIDATED   — fresh signal, backtest OK, price inside buy zone
  ACTIVE      — 1-2 bars old, still inside plan, edge still positive
  OPPORTUNITY — slightly extended but statistical edge remains
  WATCHLIST   — interesting pattern, not actionable yet
"""

import time
import sys
import threading
import traceback
import concurrent.futures
import pandas as pd
import numpy as np
from datetime import datetime

from titan.config import (
    SECTOR_ETFS, ACCOUNT_SIZE, RISK_PER_TRADE, MAX_POSITIONS,
    VIX_PANIC_THRESHOLD, DEFAULT_OHLCV_TTL_HOURS, DEFAULT_SP500_TTL_DAYS,
    DEFAULT_DATA_PERIOD, DEFAULT_DATA_INTERVAL, DEFAULT_MAX_WORKERS,
    MIN_AVG_DOLLAR_VOLUME, PORTFOLIO_HEAT_MAX, MAX_SECTOR_EXPOSURE,
)
from titan.signal_detector import (
    detect_signal, relative_strength, signal_age_bars,
    EARLY_SIGNAL_TYPES, BREAKOUT_SIGNAL_TYPES, SIGNAL_FRESHNESS_BARS,
    _safe, _compute_technicals,
)
from titan.execution_plan import (
    build_execution_plan, is_entry_buyable, fresh_entry_is_buyable,
    VALIDATED_MAX_CHASE_PCT, ACTIVE_MAX_CHASE_PCT, OPPORTUNITY_MAX_CHASE_PCT,
)
from titan.signal_tracker import SignalTracker
from titan.profiles import (
    backtest_signal, resolve_signal_move_profile,
    build_signal_family_prior, family_prior_supports_thin_history,
    family_prior_is_negative, SHORT_EDGE_MIN_ANALOGS, PRO_MIN_BT_TRADES,
)
from titan.smart_money import get_smart_money_flow, smart_money_bonus, smart_money_grade_factors
from titan.optimizer import should_run_weekly_optimization, run_weekly_optimization, load_optimized_params

# ---------------------------------------------------------------------------
# Portfolio-level risk filter
# ---------------------------------------------------------------------------

class PortfolioFilter:
    """Enforce portfolio constraints AFTER signal detection.

    Prevents:
    - Exceeding max portfolio heat (total risk $ across positions)
    - Stacking too many positions in one sector
    - Exceeding max total positions
    """

    def __init__(
        self,
        account_size: float = ACCOUNT_SIZE,
        max_heat_pct: float = PORTFOLIO_HEAT_MAX,
        max_per_sector: int = MAX_SECTOR_EXPOSURE,
        max_total: int = MAX_POSITIONS,
        open_positions: dict = None,
    ):
        self.account_size = account_size
        self.max_heat_pct = max_heat_pct
        self.max_per_sector = max_per_sector
        self.max_total = max_total

        # Pre-compute current portfolio state
        open_positions = open_positions or {}
        self._current_heat = 0.0
        self._sector_counts: dict[str, int] = {}
        self._open_count = len(open_positions)
        self._open_symbols = {str(symbol).upper() for symbol in open_positions.keys()}

        for symbol, pos in open_positions.items():
            ticker = str(pos.get("ticker") or symbol).upper()
            if ticker:
                self._open_symbols.add(ticker)
            entry = pos.get("entry_price", 0)
            stop = pos.get("stop_loss", pos.get("stop", 0))
            shares = pos.get("shares", 0)
            if entry > stop > 0:
                self._current_heat += (entry - stop) * shares / self.account_size * 100
            sector = pos.get("sector", "Unknown")
            self._sector_counts[sector] = self._sector_counts.get(sector, 0) + 1

    def filter(self, signals: list[dict]) -> tuple[list[dict], list[dict]]:
        """Filter signals by portfolio constraints.

        Args:
            signals: sorted by trade_score descending

        Returns:
            (accepted, rejected) — rejected includes reason in '_rejected' key
        """
        accepted = []
        rejected = []
        running_heat = self._current_heat
        running_sectors = dict(self._sector_counts)
        running_count = self._open_count

        for sig in signals:
            ticker = str(sig.get("ticker") or "").upper()
            if ticker and ticker in self._open_symbols:
                sig["_rejected"] = "already_held"
                rejected.append(sig)
                continue

            # Max positions
            if running_count + len(accepted) >= self.max_total:
                sig["_rejected"] = "max_positions"
                rejected.append(sig)
                continue

            # Portfolio heat
            risk_dollars = sig.get("risk_dollars", 0)
            if self.account_size > 0 and risk_dollars > 0:
                proposed_heat = running_heat + (risk_dollars / self.account_size * 100)
                if proposed_heat > self.max_heat_pct:
                    sig["_rejected"] = f"portfolio_heat_{proposed_heat:.1f}pct"
                    rejected.append(sig)
                    continue
            else:
                proposed_heat = running_heat

            # Sector concentration
            sector = sig.get("sector", "Unknown")
            if sector and sector != "Unknown":
                sector_count = running_sectors.get(sector, 0)
                if sector_count >= self.max_per_sector:
                    sig["_rejected"] = f"sector_limit_{sector}"
                    rejected.append(sig)
                    continue
                running_sectors[sector] = sector_count + 1

            accepted.append(sig)
            running_heat = proposed_heat

        return accepted, rejected


# ---------------------------------------------------------------------------
# Scanner constants
# ---------------------------------------------------------------------------
MIN_PRICE = 5.0                    # Skip penny stocks
MIN_STRENGTH_VALIDATED = 5.0       # Minimum strength for validated tier
MIN_STRENGTH_WATCHLIST = 4.0       # Minimum strength for watchlist
ADX_CHOPPY_THRESHOLD = 20
ADX_STRONG_TREND_THRESHOLD = 30

# Quality gate thresholds (ported from KOSPI v5)
PRO_MAX_CHASE_1D_PCT = 4.5        # 1-day change above this = chasing
PRO_MAX_MOMENTUM_20D_PCT = 25.0   # 20-day change above this = overextended
EARLY_SETUP_OVERHEAT_1D_PCT = 1.5 # Early setup + hot RSI + this 1D change = overheated

# Predictive scoring minimum
PREDICTIVE_MIN_SCORE = 5.0


# ---------------------------------------------------------------------------
# Historical gap risk classification (computed from OHLCV, not external data)
# ---------------------------------------------------------------------------

def _classify_gap_risk(avg_gap_pct: float, gap_above_1pct_prob: float) -> str:
    """Classify gap risk from historical open-vs-prev-close data."""
    prob_pct = (
        float(gap_above_1pct_prob) * 100.0
        if 0.0 <= float(gap_above_1pct_prob) <= 1.0
        else float(gap_above_1pct_prob)
    )
    if avg_gap_pct > 1.5 or prob_pct > 30.0:
        return "HIGH"
    if avg_gap_pct < 0.5 and prob_pct < 10.0:
        return "LOW"
    return "MED"


def _compute_historical_gap_risk(df: pd.DataFrame) -> tuple[str, float, float]:
    """Compute gap risk directly from OHLCV data (last 20 sessions).
    Returns (risk_level, avg_gap_pct, pct_of_days_gap_above_1pct)."""
    try:
        if df is None or len(df) < 22:
            return "MED", 0.0, 0.0
        gap_window = df.iloc[-22:-1]
        gap_opens = gap_window["Open"].values
        gap_prev_closes = df["Close"].iloc[-23:-2].values
        if len(gap_opens) != len(gap_prev_closes):
            return "MED", 0.0, 0.0
        gap_changes = [
            (o - c) / c * 100
            for o, c in zip(gap_opens, gap_prev_closes)
            if c > 0
        ]
        if not gap_changes:
            return "MED", 0.0, 0.0
        avg_gap_pct = sum(abs(g) for g in gap_changes) / len(gap_changes)
        gap_above_1pct_prob = sum(1 for g in gap_changes if abs(g) > 1.0) / len(gap_changes) * 100
        risk = _classify_gap_risk(avg_gap_pct, gap_above_1pct_prob)
        return risk, round(avg_gap_pct, 2), round(gap_above_1pct_prob, 1)
    except Exception:
        return "MED", 0.0, 0.0


# ---------------------------------------------------------------------------
# Weekly trend check
# ---------------------------------------------------------------------------

def _check_weekly_trend(df: pd.DataFrame) -> dict:
    """Check if the weekly trend supports the signal."""
    if df is None or len(df) < 50:
        return {"valid": False, "trend": "N/A", "aligned": True}

    close = df["Close"]
    sma10w = close.rolling(50).mean()  # ~10 weeks
    sma30w = close.rolling(150).mean()  # ~30 weeks

    if pd.isna(sma10w.iloc[-1]) or pd.isna(sma30w.iloc[-1]):
        return {"valid": False, "trend": "N/A", "aligned": True}

    last_close = _safe(close.iloc[-1])
    sma10 = _safe(sma10w.iloc[-1])
    sma30 = _safe(sma30w.iloc[-1])

    if sma10 <= 0 or sma30 <= 0:
        return {"valid": False, "trend": "N/A", "aligned": True}

    sma10_prev = _safe(sma10w.iloc[-6], sma10)
    slope = (sma10 - sma10_prev) / sma10_prev if sma10_prev > 0 else 0

    if last_close > sma10 > sma30 and slope > 0.002:
        trend = "STRONG_UP"
    elif last_close > sma10 > sma30:
        trend = "UP"
    elif last_close < sma10 < sma30 and slope < -0.002:
        trend = "STRONG_DOWN"
    elif last_close < sma10 < sma30:
        trend = "DOWN"
    else:
        trend = "SIDEWAYS"

    aligned = trend not in ("DOWN", "STRONG_DOWN")
    return {"valid": True, "trend": trend, "aligned": aligned}


# ---------------------------------------------------------------------------
# Distribution detection — big red candle + high volume = skip
# ---------------------------------------------------------------------------

def _has_recent_distribution(df: pd.DataFrame) -> tuple[bool, bool]:
    """Check for recent distribution days.
    Returns (severe_distribution, mild_distribution).
    Severe: -6% candle on 3x volume (hard skip).
    Mild: -4% candle on 2x volume (penalty only)."""
    if df is None or len(df) < 6:
        return False, False
    severe = False
    mild = False
    recent = df.iloc[-6:-1]
    for _, candle in recent.iterrows():
        c = _safe(candle.get("Close"))
        o = _safe(candle.get("Open"))
        if o <= 0:
            continue
        vol = _safe(candle.get("Volume"))
        avg_vol = _safe(df["Volume"].loc[:candle.name].tail(20).mean())
        if avg_vol <= 0:
            continue
        vol_ratio = vol / avg_vol
        drop_pct = (c - o) / o
        if drop_pct < -0.06 and vol_ratio > 3.0:
            severe = True
        elif drop_pct < -0.04 and vol_ratio > 2.0:
            mild = True
    return severe, mild


# ---------------------------------------------------------------------------
# Quality gate — reject overextended/overbought/overheated setups
# ---------------------------------------------------------------------------

QUALITY_HARD_REASONS = frozenset({
    "overextended_20d",
    "too_extended_today",
    "overbought_breakout",
    "early_setup_overheated",
})

QUALITY_SOFT_REASONS = frozenset({
    "mild_distribution",
    "moderate_sma50_decline",
})


def _passes_quality_gate(
    signal_type: str,
    change_1d: float,
    change_20d: float,
    rsi: float,
) -> tuple[bool, list[str]]:
    """Reject weak, stale, or overextended setups before ranking."""
    reasons: list[str] = []

    if signal_type in ("BREAKOUT", "MOMENTUM", "MOM_RANK", "REL_STR"):
        if change_20d >= PRO_MAX_MOMENTUM_20D_PCT:
            reasons.append("overextended_20d")
        if change_1d >= PRO_MAX_CHASE_1D_PCT:
            reasons.append("too_extended_today")
        if signal_type in ("BREAKOUT", "MOMENTUM", "MOM_RANK") and rsi >= 68:
            reasons.append("overbought_breakout")

    if signal_type in EARLY_SIGNAL_TYPES and rsi > 65 and change_20d >= 12.0 and change_1d >= EARLY_SETUP_OVERHEAT_1D_PCT:
        reasons.append("early_setup_overheated")

    return (len(reasons) == 0), reasons


def _is_watchlist_quality_candidate(
    signal_type: str,
    qg_reasons: list[str],
    strength: float,
    weekly_trend: str,
) -> bool:
    """Allow strong early setups with only soft quality misses onto watchlist."""
    if not qg_reasons:
        return False
    if signal_type not in EARLY_SIGNAL_TYPES:
        return False
    # Any hard reason = no watchlist
    if any(r in QUALITY_HARD_REASONS for r in qg_reasons):
        return False
    # All reasons must be soft
    if any(r not in QUALITY_SOFT_REASONS for r in qg_reasons):
        return False
    if strength < 4.0:
        return False
    if weekly_trend in ("DOWN", "STRONG_DOWN"):
        return False
    return True


# ---------------------------------------------------------------------------
# Dynamic strength floor (adapts to market_score, not just regime name)
# ---------------------------------------------------------------------------

def _validated_min_strength(market_score: float, signal_type: str) -> float:
    """Dynamic strength floor based on market health."""
    base = 5.0
    if market_score < 30:
        base = 6.5
    elif market_score < 50:
        base = 5.5
    if signal_type in EARLY_SIGNAL_TYPES:
        base -= 0.5
    return base


def _watchlist_min_strength(market_score: float, signal_type: str) -> float:
    val_floor = _validated_min_strength(market_score, signal_type)
    return val_floor - 1.0 if signal_type in EARLY_SIGNAL_TYPES else val_floor - 0.5


# ---------------------------------------------------------------------------
# Signal grading (A/B/C confidence) — ported from KOSPI v5
# ---------------------------------------------------------------------------

def _grade_signal(
    strength: float,
    weekly_trend: str,
    vol_contraction: float,
    rsi: float,
    dist_from_high: float,
    regime: str,
) -> str:
    """Grade signal confidence: A (high conviction), B (good), C (speculative)."""
    factors = 0.0
    if strength >= 10:
        factors += 2
    elif strength >= 7:
        factors += 1
    if weekly_trend in ("STRONG_UP", "UP"):
        factors += 1
    if vol_contraction < 0.55:
        factors += 1
    if dist_from_high < 5:
        factors += 1
    if 40 <= rsi <= 60:
        factors += 0.5
    if regime in ("STRONG_BULL", "BULL"):
        factors += 1

    if factors >= 5:
        return "A"
    elif factors >= 3:
        return "B"
    return "C"


# ---------------------------------------------------------------------------
# Predictive scoring engine — 12-factor coiled spring detector
# Ported from KOSPI v5: "The best time to buy is when a stock is BORING."
# ---------------------------------------------------------------------------

def _score_predictive(
    t: dict, weekly_trend: str, sector: str, top_sectors: list,
    safety_penalty: float = 0.0, gap_penalty: float = 0.0,
) -> tuple[float, list[str], str]:
    """Score a stock by pre-move potential. Returns (score, reasons, direction)."""
    score = 0.0
    reasons = []

    close = t.get("close", 0)
    sma20 = t.get("sma20", close)
    sma50 = t.get("sma50", close)
    sma200 = t.get("sma200", sma50)
    rsi = t.get("rsi", 50)
    rsi_turning_up = t.get("rsi_turning_up", False)
    vol_contraction = t.get("vol_contraction", 1.0)
    change_1d = t.get("change_1d", 0)
    change_5d = t.get("change_5d", 0)
    change_20d = t.get("change_20d", 0)
    volume_ratio = t.get("volume_ratio", 1.0)
    vol_trend = t.get("vol_trend", 1.0)
    dist_from_high = t.get("dist_from_high", 50)
    range_5d_pct = t.get("range_5d_pct", 10)
    sma50_trend = t.get("sma50_trend", 0.0)
    pullback_to_sma20 = abs((close - sma20) / sma20 * 100) if sma20 > 0 else 99

    # Factor 1: VOLATILITY CONTRACTION (0-5 pts) — #1 predictor
    if vol_contraction < 0.25:
        score += 5.0; reasons.append(f"Extreme coil ({vol_contraction:.2f})")
    elif vol_contraction < 0.4:
        score += 4.0; reasons.append(f"Tight coil ({vol_contraction:.2f})")
    elif vol_contraction < 0.55:
        score += 3.0; reasons.append(f"Coiling ({vol_contraction:.2f})")
    elif vol_contraction < 0.7:
        score += 1.5
    elif vol_contraction > 0.9:
        score -= 1.0

    # Factor 2: QUIET TODAY (0-4 pts)
    if -1.0 <= change_1d <= 1.0:
        score += 4.0; reasons.append(f"Dead quiet today ({change_1d:+.1f}%)")
    elif -2.0 <= change_1d <= 2.0:
        score += 3.0; reasons.append(f"Quiet today ({change_1d:+.1f}%)")
    elif -3.0 <= change_1d <= 3.0:
        score += 1.5
    elif change_1d > 8.0:
        score -= 4.0; reasons.append(f"Up +{change_1d:.1f}% today -- too late")
    elif change_1d > 5.0:
        score -= 2.0; reasons.append(f"Already moved +{change_1d:.1f}% -- chasing")
    elif change_1d < -5.0:
        score -= 1.5

    # Factor 3: TREND FOUNDATION (0-3 pts)
    if close > sma20 > sma50:
        score += 3.0; reasons.append("Uptrend intact (Price>SMA20>SMA50)")
    elif close > sma50 and close > sma20:
        score += 2.5
    elif close > sma50:
        score += 1.0
    elif close < sma50 * 0.92:
        score -= 1.5

    # Factor 4: TIGHT CONSOLIDATION (0-3 pts)
    if range_5d_pct < 3.0 and pullback_to_sma20 < 2:
        score += 3.0; reasons.append(f"Ultra-tight base at SMA20 ({range_5d_pct:.1f}% range)")
    elif range_5d_pct < 4.0:
        score += 2.5
    elif range_5d_pct < 6.0:
        score += 1.5
    elif range_5d_pct > 12:
        score -= 1.0

    # Factor 5: BREAKOUT PROXIMITY (0-2.5 pts)
    if 2.0 <= dist_from_high <= 6.0:
        score += 2.5; reasons.append(f"Winding up {dist_from_high:.1f}% below resistance")
    elif 1.0 <= dist_from_high < 2.0:
        score += 2.0
    elif 6.0 < dist_from_high <= 10.0:
        score += 1.0
    elif dist_from_high < 0.5:
        score += 0.5
    elif dist_from_high > 20:
        score -= 1.0

    # Factor 6: RSI NEUTRAL ZONE (0-2 pts)
    if 45 <= rsi <= 55:
        score += 2.0; reasons.append(f"RSI neutral ({rsi:.0f}) -- room to run")
    elif rsi_turning_up and rsi < 45:
        score += 2.0; reasons.append(f"RSI turning up from {rsi:.0f}")
    elif 40 <= rsi <= 60:
        score += 1.5
    elif 35 <= rsi <= 65:
        score += 0.5
    elif rsi > 70:
        score -= 2.0; reasons.append(f"Overbought RSI ({rsi:.0f})")
    elif rsi < 25:
        score -= 1.0

    # Factor 7: QUIET WEEK (0-2 pts)
    if -2.0 <= change_5d <= 3.0:
        score += 2.0; reasons.append(f"Quiet week ({change_5d:+.1f}%) -- still loading")
    elif -4.0 <= change_5d <= 5.0:
        score += 1.0
    elif change_5d > 10.0:
        score -= 1.5
    elif change_5d < -8.0:
        score -= 1.0

    # Factor 8: VOLUME DRYING UP then TICKING (0-2.5 pts)
    if 1.1 < vol_trend <= 1.4 and volume_ratio < 1.2:
        score += 2.5; reasons.append(f"Vol quietly building ({vol_trend:.1f}x)")
    elif 0.7 <= vol_trend <= 1.1 and volume_ratio < 0.8:
        score += 2.0; reasons.append(f"Dry volume ({volume_ratio:.1f}x) -- nobody watching")
    elif vol_trend > 1.5 and volume_ratio > 2.0:
        score -= 1.0
    elif vol_trend > 1.2 and change_1d < 2:
        score += 1.5

    # Factor 9: MACRO SUPPORT (0-2 pts)
    if sma200 > 0:
        if close > sma200 and sma50 > sma200:
            score += 1.5; reasons.append("Macro uptrend (above SMA200)")
        elif close > sma200:
            score += 0.5
        elif close < sma200 * 0.9:
            score -= 1.0
    if sma50_trend > 0.1:
        score += 0.5; reasons.append("SMA50 rising")
    elif sma50_trend < -0.2:
        score -= 1.0

    # Factor 10: SAFE ENTRY ZONE (0-1.5 pts)
    if pullback_to_sma20 < 1.5 and close >= sma20 * 0.99:
        score += 1.5; reasons.append(f"At SMA20 support ({pullback_to_sma20:.1f}%)")
    elif pullback_to_sma20 < 3.0:
        score += 1.0
    elif pullback_to_sma20 > 8.0 and close > sma20:
        score -= 0.5

    # Factor 11: GENTLE 20D DRIFT (0-1.5 pts)
    if 3.0 <= change_20d <= 12.0:
        score += 1.5; reasons.append(f"Healthy 20D drift (+{change_20d:.1f}%)")
    elif 0 < change_20d <= 20:
        score += 0.5
    elif change_20d > 25:
        score -= 1.5
    elif change_20d < -15:
        score -= 1.5

    # Factor 12: SECTOR MOMENTUM (0-2 pts) — replaces KOSPI's smart money flow
    # (US market doesn't have foreign/institution flow data like Korea)
    if sector and sector in top_sectors:
        score += 2.0; reasons.append(f"Hot sector: {sector}")

    # Safety/gap penalties at half-weight (research candidates, not buys)
    if safety_penalty < 0:
        score += safety_penalty * 0.3
    if gap_penalty < 0:
        score += gap_penalty * 0.3

    # DIRECTION BIAS — 12 micro-factors (expanded with MA trends)
    up = 0
    down = 0
    # 1-3. Price vs MAs
    if close > sma20: up += 1
    else: down += 1
    if close > sma50: up += 1
    else: down += 1
    if sma200 > 0 and close > sma200: up += 1
    elif sma200 > 0: down += 1
    # 4-5. MA direction
    if sma50_trend > 0.05: up += 1
    elif sma50_trend < -0.05: down += 1
    sma20_trend = t.get("sma20_trend", 0.0)
    if sma20_trend > 0.1: up += 1
    elif sma20_trend < -0.1: down += 1
    # 6. RSI position
    if rsi > 50: up += 1
    elif rsi < 40: down += 1
    # 7. Short-term momentum
    if change_5d > 0: up += 1
    elif change_5d < -3: down += 1
    # 8. Medium-term momentum
    if change_20d > 0: up += 1
    elif change_20d < -5: down += 1
    # 9. Volume trend (smart money direction)
    if vol_trend > 1.1 and change_1d >= 0: up += 1
    elif vol_trend > 1.3 and change_1d < -2: down += 1
    # 10. OBV as smart money proxy (US doesn't have foreign/inst flow)
    if t.get("obv", 0) > t.get("obv_sma", 0): up += 1
    else: down += 1
    # 11. Weekly trend
    if weekly_trend in ("STRONG_UP", "UP"): up += 1
    elif weekly_trend in ("DOWN", "STRONG_DOWN"): down += 1

    net = up - down
    if net >= 6:
        direction = "UP"; score += 3.0; reasons.append(f"Direction: STRONG UP ({up} up vs {down} down)")
    elif net >= 4:
        direction = "UP"; score += 2.0; reasons.append(f"Direction: UP ({up} up vs {down} down)")
    elif net >= 2:
        direction = "LEAN_UP"; score += 1.0
    elif net <= -4:
        direction = "DOWN"; score -= 3.0
    elif net <= -2:
        direction = "LEAN_DOWN"; score -= 1.5
    else:
        direction = "NEUTRAL"

    return score, reasons, direction


# ---------------------------------------------------------------------------
# Market regime (ADX-enhanced)
# ---------------------------------------------------------------------------

def _detect_market_regime(spy_df: pd.DataFrame) -> dict:
    """Detect market regime from SPY with ADX."""
    if spy_df is None or spy_df.empty or len(spy_df) < 200:
        return {"regime": "UNKNOWN", "score": 50, "adx": 20}

    close = spy_df["Close"]
    last_close = _safe(close.iloc[-1])
    sma20 = _safe(close.rolling(20).mean().iloc[-1], last_close)
    sma50 = _safe(close.rolling(50).mean().iloc[-1], last_close)
    sma200 = _safe(close.rolling(200).mean().iloc[-1], last_close)

    # ADX calculation
    high = spy_df["High"]
    low = spy_df["Low"]
    plus_dm = high.diff().clip(lower=0)
    minus_dm = (-low.diff()).clip(lower=0)
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs(),
    ], axis=1).max(axis=1)
    atr14 = tr.rolling(14).mean()
    plus_di = 100 * (plus_dm.rolling(14).mean() / atr14)
    minus_di = 100 * (minus_dm.rolling(14).mean() / atr14)
    dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di + 1e-10))
    adx = _safe(dx.rolling(14).mean().iloc[-1], 20)

    sma20_5d_ago = _safe(close.rolling(20).mean().iloc[-6], sma20)
    trend_slope = (sma20 - sma20_5d_ago) / sma20_5d_ago if sma20_5d_ago > 0 else 0

    # RSI
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = _safe((100 - (100 / (1 + rs))).iloc[-1], 50)

    # VIX (if available as ^VIX column — handled separately)

    score = 0
    if last_close > sma20 > sma50 > sma200:
        score += 40
    elif last_close > sma20 > sma50:
        score += 30
    elif last_close > sma20:
        score += 20
    elif last_close > sma50:
        score += 10
    elif last_close < sma20 < sma50 < sma200:
        score -= 40
    elif last_close < sma20 < sma50:
        score -= 30
    elif last_close < sma20:
        score -= 20

    if adx > 30:
        score += 30 if trend_slope > 0 else -30
    elif adx > 25:
        score += 20 if trend_slope > 0 else -20
    elif adx > 20:
        score += 10 if trend_slope > 0 else -10

    if rsi > 60:
        score += 15
    elif rsi > 50:
        score += 10
    elif rsi < 40:
        score -= 15
    elif rsi < 50:
        score -= 10

    if score >= 40:
        regime = "STRONG_BULL"
    elif score >= 20:
        regime = "BULL"
    elif score <= -40:
        regime = "STRONG_BEAR"
    elif score <= -20:
        regime = "BEAR"
    else:
        regime = "SIDEWAYS"

    return {"regime": regime, "score": score, "adx": adx}


# ---------------------------------------------------------------------------
# Pre-move radar scoring (from KOSPI)
# ---------------------------------------------------------------------------

def _score_tradeability(
    base_score: float,
    signal_type: str,
    t: dict,
    plan: dict,
    signal_age: int = 0,
    weekly_trend: str = "N/A",
) -> float:
    """Score a candidate for manual tradeability (higher = better)."""
    score = base_score

    # Freshness bonus
    if signal_age == 0:
        score += 0.8
    elif signal_age == 1:
        score += 0.25

    # Weekly trend
    if weekly_trend == "STRONG_UP":
        score += 0.7
    elif weekly_trend == "UP":
        score += 0.35

    # RSI positioning
    rsi = t.get("rsi", 50)
    if 46.0 <= rsi <= 62.0:
        score += 0.6
    elif 38.0 <= rsi < 68.0:
        score += 0.25
    elif rsi > 70.0:
        score -= 0.8

    # Volatility contraction = coiled spring
    vc = t.get("vol_contraction", 1.0)
    if vc <= 0.55:
        score += 0.7
    elif vc <= 0.75:
        score += 0.25

    # Distance from breakout
    dfh = t.get("dist_from_high", 50)
    if dfh <= 3.0:
        score += 0.8
    elif dfh <= 6.0:
        score += 0.35
    elif dfh > 12.0:
        score -= 0.5

    # Change-based modifiers
    c1d = t.get("change_1d", 0)
    c5d = t.get("change_5d", 0)
    c20d = t.get("change_20d", 0)

    if 0.0 <= c1d <= 2.5:
        score += 0.45
    elif c1d > 4.0:
        score -= 0.8

    if -1.0 <= c5d <= 4.0:
        score += 0.25
    elif c5d > 7.0:
        score -= 0.45

    if 0.0 <= c20d <= 14.0:
        score += 0.35
    elif c20d > 18.0:
        score -= 0.55

    # Volume
    vr = t.get("volume_ratio", 1.0)
    if vr >= 1.2:
        score += 0.25

    # Gap risk
    if plan.get("gap_risk") == "LOW":
        score += 0.25

    # Entry buffer intact
    close = t.get("close", 0)
    max_buy = _safe(plan.get("max_buy_price"), close)
    if close > 0 and max_buy > close:
        buffer_pct = (max_buy / close - 1.0) * 100
        if 0.2 <= buffer_pct <= 1.2:
            score += 0.25

    return score


# ---------------------------------------------------------------------------
# Position sizing
# ---------------------------------------------------------------------------

def _calculate_position(price, stop, account_size=ACCOUNT_SIZE, risk_per_trade=RISK_PER_TRADE):
    risk_per_share = abs(price - stop)
    if risk_per_share <= 0:
        return 0, 0
    qty = int(risk_per_trade / risk_per_share)
    cost = qty * price
    max_cost = account_size * 0.12  # Max 12% per position
    if cost > max_cost:
        qty = int(max_cost / price)
    return max(0, qty), round(risk_per_share * qty, 2)


# ---------------------------------------------------------------------------
# Process a single ticker
# ---------------------------------------------------------------------------

def _process_ticker(
    ticker: str,
    df: pd.DataFrame,
    spy_df: pd.DataFrame,
    regime_info: dict,
    market_score: float,
    strength_floor: float,
    top_sectors: list | None = None,
    family_priors: dict | None = None,
    stock_frames: dict | None = None,
    family_lock: threading.Lock | None = None,
) -> tuple[dict | None, dict | None]:
    """Analyze one ticker.

    Returns:
        (signal_result or None, predictive_result or None)
    """
    if df is None or df.empty or len(df) < 60:
        return None, None

    close = _safe(df["Close"].iloc[-1])
    if close < MIN_PRICE:
        return None, None

    # Liquidity check
    avg_vol = _safe(df["Volume"].rolling(20).mean().iloc[-1])
    if avg_vol * close < MIN_AVG_DOLLAR_VOLUME:
        return None, None

    # Compute technicals first (needed for both signal and predictive)
    t = _compute_technicals(df)
    if not t:
        return None, None

    sma50_trend = t.get("sma50_trend", 0.0)

    # --- Safety filters (penalties, not hard kills) ---
    _safety_penalty = 0.0

    # SMA50 safety: only hard-skip truly toxic trends (steep + far below)
    if sma50_trend < -0.3 and close < t.get("sma50", close) * 0.95:
        return None, None  # Toxic trend, hard skip
    elif sma50_trend < -0.1 and close < t.get("sma50", close):
        _safety_penalty -= 1.5  # Moderate decline: penalize

    # Distribution check (now returns severity)
    severe_dist, mild_dist = _has_recent_distribution(df)
    if severe_dist:
        _safety_penalty -= 2.0
    elif mild_dist:
        _safety_penalty -= 1.0

    # Historical gap risk from OHLCV
    hist_gap_risk, hist_avg_gap_pct, hist_gap_above_1pct_prob = _compute_historical_gap_risk(df)
    _gap_penalty = 0.0
    if hist_gap_risk == "HIGH":
        _gap_penalty = -1.5
    elif hist_gap_risk == "MED" and hist_avg_gap_pct > 0.5:
        _gap_penalty = -0.5

    # Weekly trend (needed for both paths)
    weekly = _check_weekly_trend(df)
    weekly_trend = weekly.get("trend", "N/A")
    weekly_aligned = weekly.get("aligned", True)

    # === PREDICTIVE SCORING (runs on ALL stocks, not just signal matches) ===
    pred_result = None
    pred_score, pred_reasons, pred_direction = _score_predictive(
        t, weekly_trend, "",  # sector enriched later
        top_sectors or [],
        safety_penalty=_safety_penalty,
        gap_penalty=_gap_penalty,
    )
    if pred_score >= PREDICTIVE_MIN_SCORE:
        atr_pred = t.get("atr", close * 0.03)
        trigger_2d = _safe(df["High"].iloc[-2:].max()) if len(df) >= 2 else round(close * 1.01, 2)
        stop_2d = max(
            _safe(df["Low"].iloc[-2:].min()) if len(df) >= 2 else round(close * 0.97, 2),
            round(trigger_2d - atr_pred, 2),
        )
        risk_pct = round((trigger_2d - stop_2d) / trigger_2d * 100, 1) if trigger_2d > 0 else 5.0
        # Risk-on-trigger check: if risk > 5%, use 1.2 ATR stop
        if risk_pct > 5.0:
            stop_2d = round(trigger_2d - atr_pred * 1.2, 2)
            risk_pct = round((trigger_2d - stop_2d) / trigger_2d * 100, 1) if trigger_2d > 0 else 5.0
        target_pred = round(trigger_2d + (trigger_2d - stop_2d) * 1.5, 2)
        pred_result = {
            "ticker": ticker,
            "pred_score": round(pred_score, 1),
            "direction": pred_direction,
            "pred_reasons": pred_reasons[:5],
            "trigger": round(trigger_2d, 2),
            "stop": round(stop_2d, 2),
            "target": round(target_pred, 2),
            "risk_pct": risk_pct,
            "price": close,
            "rsi": round(t.get("rsi", 50), 1),
            "vol_contraction": round(t.get("vol_contraction", 1), 2),
            "change_1d": round(t.get("change_1d", 0), 1),
            "weekly_trend": weekly_trend,
            "sector": "",
        }

    # === SIGNAL DETECTION PATH ===
    # Severe distribution = hard skip for signal path (predictive still runs above)
    if severe_dist:
        return None, pred_result

    # Detect signal
    signal_type, strength, reasons, t = detect_signal(df, spy_df)
    if not signal_type:
        return None, pred_result

    # Market regime adjustments
    regime = regime_info.get("regime", "SIDEWAYS")
    adx = regime_info.get("adx", 20)
    is_choppy = adx < ADX_CHOPPY_THRESHOLD
    is_strong_trend = adx > ADX_STRONG_TREND_THRESHOLD

    # Strong downtrend penalty: ADX > 40 + BEAR/STRONG_BEAR = very dangerous
    # Only allow relative strength leaders with weekly uptrend
    if adx > 40 and regime in ("STRONG_BEAR", "BEAR"):
        if signal_type not in ("REL_STR", "MOM_RANK"):
            strength -= 3.0
            reasons.append(f"Strong downtrend penalty (ADX {adx:.0f}, {regime})")
        elif weekly_trend not in ("STRONG_UP", "UP"):
            strength -= 2.0
            reasons.append(f"Bear trend, no weekly uptrend")

    # Choppy market: penalize breakout types, bonus safe types (KOSPI v5 style)
    if is_choppy:
        if signal_type in ("BREAKOUT", "MOMENTUM", "VOL_SPIKE", "PRE_BREAK"):
            strength -= 2.0
            reasons.append(f"Choppy market penalty (ADX {adx:.0f})")
        elif signal_type in ("ACCUM", "VCP", "PULLBACK"):
            strength += 2.0
            reasons.append("Choppy market bonus (safe setup)")

    # Strong trend bonus
    if is_strong_trend and signal_type in ("BREAKOUT", "MOMENTUM", "MOM_RANK", "PULLBACK"):
        strength += 2.0
        reasons.append(f"Strong trend bonus (ADX {adx:.0f})")

    # Weekly trend — soft filter (penalty for DOWN, hard skip only STRONG_DOWN)
    if weekly.get("valid") and not weekly_aligned:
        wk_trend = weekly.get("trend", "DOWN")
        if wk_trend == "STRONG_DOWN":
            return None, pred_result  # Hard skip only truly bearish
        else:
            strength -= 1.0
            reasons.append("Weekly DOWN penalty")

    # Weekly recheck for unknown/N/A
    if weekly_trend in ("N/A",) and not weekly.get("valid", False) and len(df) >= 60:
        weekly_recheck = _check_weekly_trend(df)
        if weekly_recheck.get("valid"):
            weekly = weekly_recheck
            weekly_trend = weekly_recheck.get("trend", weekly_trend)
            if not weekly_recheck.get("aligned", False) and weekly_trend == "STRONG_DOWN":
                return None, pred_result

    if weekly_trend == "STRONG_UP":
        strength += 1.0

    # Signal age + staleness penalty (KOSPI v5)
    age = signal_age_bars(df, signal_type, spy_df)
    is_fresh = age == 0
    if age >= 3:
        return None, pred_result  # Expired signal
    elif age == 2:
        strength -= 1.5
        reasons.append(f"Stale signal ({age} bars old)")

    # Apply accumulated safety/gap penalties to signal strength
    strength += _gap_penalty + _safety_penalty
    if _gap_penalty < 0:
        reasons.append(f"Gap penalty {_gap_penalty:+.1f}")
    if _safety_penalty < 0:
        reasons.append(f"Safety penalty {_safety_penalty:+.1f}")

    # Emit historical gap risk into reasons
    if hist_gap_risk == "HIGH":
        reasons.append(
            f"HIGH GAP RISK: avg gap {hist_avg_gap_pct:+.1f}%, "
            f"{hist_gap_above_1pct_prob:.0f}% days >1% -- wait for pullback"
        )
    elif hist_gap_risk == "MED" and hist_avg_gap_pct > 0.3:
        reasons.append(f"Medium gap risk (avg {hist_avg_gap_pct:+.1f}%)")

    # Today's gap check (still hard skip for extreme gaps)
    prev_close = _safe(df["Close"].iloc[-2])
    today_gap_pct = abs(close - prev_close) / prev_close if prev_close > 0 else 0
    if today_gap_pct > 0.05:
        return None, pred_result

    # Quality gate — hard rejections
    passes_qg, qg_reasons = _passes_quality_gate(
        signal_type, t.get("change_1d", 0), t.get("change_20d", 0), t.get("rsi", 50),
    )
    if not passes_qg:
        # Hard reasons = completely block
        if any(r in QUALITY_HARD_REASONS for r in qg_reasons):
            # But check if it qualifies for watchlist via soft rejection path
            if _is_watchlist_quality_candidate(signal_type, qg_reasons, strength, weekly_trend):
                pass  # Let it through to watchlist classification below
            else:
                return None, pred_result

    # Strength check (dynamic floor based on market_score)
    min_str = _watchlist_min_strength(market_score, signal_type)
    if strength < min_str:
        return None, pred_result

    # Build execution plan
    # Backtest evidence: Keep 2.0 ATR stop (wider = stable buy zones, less whipsaw).
    # Reduced target from 3.0 to 2.5 ATR (more targets hit: 17.8% vs 11.3%).
    atr = t.get("atr", close * 0.03)
    # In bear markets, use tighter targets (1.8 ATR instead of 2.5)
    target_mult = 1.8 if regime in ("STRONG_BEAR", "BEAR") else 2.5
    stop = round(close - (2.0 * atr), 2)
    target = round(close + (target_mult * atr), 2)

    # ── Per-signal mini-backtest (run BEFORE execution plan to get best_horizon) ──
    bt_result = {}
    move_profile = {}
    family_prior = None
    try:
        bt_result = backtest_signal(df, signal_type, technicals_current=t)
        bt_trades = int(bt_result.get("trades", 0) or 0)
        profile_samples = int(bt_result.get("profile_analog_samples", 0) or 0)

        # Thin history check — get family prior if needed
        thin_stock_history = (
            signal_type in EARLY_SIGNAL_TYPES
            and bt_trades < PRO_MIN_BT_TRADES
            and profile_samples < SHORT_EDGE_MIN_ANALOGS
        )
        if thin_stock_history and family_priors is not None:
            _lock = family_lock or threading.Lock()
            with _lock:
                if signal_type not in family_priors and stock_frames:
                    # Build family prior on first encounter of this signal type
                    family_priors[signal_type] = build_signal_family_prior(
                        stock_frames, signal_type, lookback_days=252
                    )
                family_prior = family_priors.get(signal_type)

            # Reject if family evidence is negative
            if family_prior_is_negative(family_prior):
                reasons.append("Family prior negative")
                strength -= 2.0

        # Reject signal if backtest shows clearly negative edge
        if bt_trades >= 10 and bt_result.get("avg_return", 0) < -0.2:
            if not bt_result.get("profile_validated", False):
                reasons.append(f"Negative backtest edge ({bt_result['avg_return']:+.2f}%)")
                return None, pred_result

        # Apply profile-based strength adjustment
        move_profile = resolve_signal_move_profile(
            bt_result, family_prior=family_prior,
            allow_family=thin_stock_history if family_prior else False,
        )
        if move_profile:
            mp_return = move_profile.get("expected_return", 0)
            mp_prob = move_profile.get("up_prob", 50)
            if mp_return > 0.3 and mp_prob >= 55:
                strength += 1.0
                reasons.append(f"Edge profile +{mp_return:.1f}% ({mp_prob:.0f}% WR)")
            elif mp_return < -0.1 and mp_prob < 48:
                strength -= 1.0
                reasons.append(f"Weak edge {mp_return:+.1f}%")
    except Exception:
        pass  # Mini-backtest is optional, don't break scanner

    # ── Build execution plan (AFTER backtest so we have best_horizon) ─────
    plan = build_execution_plan(
        signal_type=signal_type,
        price=close,
        stop=stop,
        target=target,
        atr=atr,
        best_horizon=move_profile.get("best_horizon", 3),
        move_up_prob=move_profile.get("up_prob", 0),
        move_expected_return=move_profile.get("expected_return", 0),
        avg_open_gap_pct=hist_avg_gap_pct,
        gap_above_1pct_prob=hist_gap_above_1pct_prob,
    )

    # Classify into tiers (before smart money — avoid unnecessary HTTP calls)
    tier = _classify_tier(
        signal_type=signal_type,
        strength=strength,
        age=age,
        is_fresh=is_fresh,
        close=close,
        plan=plan,
        weekly_trend=weekly_trend,
        regime=regime,
        market_score=market_score,
    )

    if tier is None:
        return None, pred_result

    # ── Smart money flow (only for signals that passed tier classification) ──
    sm_flow = {}
    if tier in ("VALIDATED", "ACTIVE", "OPPORTUNITY"):
        try:
            sm_flow = get_smart_money_flow(ticker)
            sm_bonus = smart_money_bonus(sm_flow)
            if sm_bonus != 0:
                strength += sm_bonus
                if sm_bonus > 0:
                    reasons.append(f"Smart money +{sm_bonus:.0f}")
                else:
                    reasons.append(f"Smart money {sm_bonus:+.0f}")
        except Exception:
            pass

    # Signal grading (A/B/C)
    grade = _grade_signal(
        strength, weekly_trend, t.get("vol_contraction", 1),
        t.get("rsi", 50), t.get("dist_from_high", 50), regime,
    )

    # Calculate tradeability score
    trade_score = _score_tradeability(
        base_score=strength,
        signal_type=signal_type,
        t=t,
        plan=plan,
        signal_age=age,
        weekly_trend=weekly_trend,
    )

    # Position sizing
    qty, risk_dollars = _calculate_position(close, stop)
    rr_ratio = (target - close) / (close - stop) if (close - stop) > 0 else 0

    sector = ""  # Enriched later

    signal_result = {
        "ticker": ticker,
        "signal_type": signal_type,
        "tier": tier,
        "grade": grade,
        "signal_strength": round(strength, 1),
        "trade_score": round(trade_score, 1),
        "signal_age": age,
        "price": close,
        "stop": stop,
        "target": target,
        "buy_zone_low": plan["buy_zone_low"],
        "buy_zone_high": plan["buy_zone_high"],
        "max_buy_price": plan["max_buy_price"],
        "cancel_above_pct": plan["cancel_above_pct"],
        "time_stop_days": plan["time_stop_days"],
        "entry_style": plan["entry_style"],
        "entry_note": plan["entry_note"],
        "gap_risk": plan["gap_risk"],
        "qty": qty,
        "risk_dollars": risk_dollars,
        "rr_ratio": round(rr_ratio, 1),
        "rsi": round(t.get("rsi", 50), 1),
        "volume_ratio": round(t.get("volume_ratio", 1), 1),
        "vol_contraction": round(t.get("vol_contraction", 1), 2),
        "dist_from_high": round(t.get("dist_from_high", 0), 1),
        "change_1d": round(t.get("change_1d", 0), 1),
        "change_5d": round(t.get("change_5d", 0), 1),
        "change_20d": round(t.get("change_20d", 0), 1),
        "weekly_trend": weekly_trend,
        "reasons": reasons[:5],
        "sector": sector,
        "date": datetime.now().strftime("%Y-%m-%d"),
        # New: backtest evidence
        "bt_trades": bt_result.get("trades", 0),
        "bt_win_rate": round(bt_result.get("win_rate", 0), 1),
        "bt_avg_return": round(bt_result.get("avg_return", 0), 2),
        "bt_validated": bt_result.get("validated", False),
        "profile_samples": bt_result.get("profile_analog_samples", 0),
        "profile_score": round(bt_result.get("profile_score", 0), 1),
        "move_source": move_profile.get("source", ""),
        "move_up_prob": round(move_profile.get("up_prob", 0), 1),
        "move_expected_return": round(move_profile.get("expected_return", 0), 2),
        # New: smart money
        "smart_money_score": sm_flow.get("smart_money_score", 0),
        "inst_ownership_pct": sm_flow.get("inst_ownership_pct", 0),
        "short_pct_float": sm_flow.get("short_pct_float", 0),
        "put_call_ratio": sm_flow.get("put_call_ratio", 1.0),
    }

    return signal_result, pred_result


# ---------------------------------------------------------------------------
# Tier classification
# ---------------------------------------------------------------------------

def _classify_tier(
    signal_type: str,
    strength: float,
    age: int,
    is_fresh: bool,
    close: float,
    plan: dict,
    weekly_trend: str,
    regime: str,
    market_score: float,
) -> str | None:
    """Classify a signal into VALIDATED / ACTIVE / OPPORTUNITY / WATCHLIST.
    Returns None if the signal should be dropped entirely."""

    # Hard blocks
    if plan.get("gap_risk") == "HIGH" and signal_type in BREAKOUT_SIGNAL_TYPES:
        return None

    is_bear = regime in ("STRONG_BEAR", "BEAR")
    bear_boost = 2.0 if regime == "STRONG_BEAR" else (1.0 if regime == "BEAR" else 0.0)

    # Dynamic floors based on market health
    val_floor = _validated_min_strength(market_score, signal_type) + bear_boost
    wl_floor = _watchlist_min_strength(market_score, signal_type) + bear_boost

    # Validated: fresh signal, inside buy zone (0% chase), strong enough
    if is_fresh and strength >= val_floor:
        if fresh_entry_is_buyable(close, plan):
            if is_bear:
                if signal_type in EARLY_SIGNAL_TYPES and weekly_trend in ("STRONG_UP", "UP"):
                    return "VALIDATED"
            else:
                if signal_type in EARLY_SIGNAL_TYPES:
                    return "VALIDATED"
                if strength >= val_floor + 1:
                    return "VALIDATED"

    # Active: 0-2 bars old, still inside buy zone with small chase tolerance.
    # NOTE: age == 0 signals that barely missed VALIDATED (price slightly above
    # max_buy_price) should still be ACTIVE, not OPPORTUNITY.
    if age <= SIGNAL_FRESHNESS_BARS:
        if is_entry_buyable(close, plan, ACTIVE_MAX_CHASE_PCT):
            if is_bear:
                if signal_type in EARLY_SIGNAL_TYPES and strength >= val_floor and weekly_trend in ("STRONG_UP", "UP"):
                    return "ACTIVE"
            else:
                if signal_type in EARLY_SIGNAL_TYPES and strength >= val_floor:
                    return "ACTIVE"
                if strength >= val_floor + 1:
                    return "ACTIVE"

    # Opportunity: slightly extended or outside buy zone but edge may remain
    if age <= SIGNAL_FRESHNESS_BARS and strength >= val_floor:
        if is_entry_buyable(close, plan, OPPORTUNITY_MAX_CHASE_PCT):
            return "OPPORTUNITY"

    # Watchlist: interesting but not actionable
    if strength >= wl_floor:
        if signal_type in EARLY_SIGNAL_TYPES:
            return "WATCHLIST"
        if strength >= wl_floor + 1:
            return "WATCHLIST"

    return None


# ---------------------------------------------------------------------------
# Main pro_scan entry point
# ---------------------------------------------------------------------------

def pro_scan(settings=None, market_data_bundle=None) -> dict:
    """Run the pro scanner — KOSPI-style multi-signal detection.

    Args:
        settings: dict of runtime settings (cache_ttl, force_refresh, etc.)
        market_data_bundle: (tickers, data) tuple if pre-loaded

    Returns dict with:
        regime, vix, market_score,
        validated, active, opportunity, watchlist (lists of result dicts),
        stalk_orders (ready-to-place bracket orders),
        scan_time, stats
    """
    # Import here to avoid circular import — titan_trade_v3 has the data loader
    import titan_trade_v3 as v3
    from titan.market import MarketRegime, SectorAnalyzer

    if settings is None:
        settings = {}

    start_time = time.time()
    print("\n" + "=" * 70)
    print("  TITAN PRO SCANNER -- Multi-Signal Detection")
    print("=" * 70)

    # ── Load data ──────────────────────────────────────────────────────────
    if market_data_bundle is not None:
        tickers, data = market_data_bundle
    else:
        tickers, data = v3.get_market_data(
            cache_ttl_hours=settings.get("cache_ttl_hours", DEFAULT_OHLCV_TTL_HOURS),
            sp500_ttl_days=settings.get("sp500_ttl_days", DEFAULT_SP500_TTL_DAYS),
            force_refresh=settings.get("force_refresh_cache", False),
            data_period=settings.get("data_period", DEFAULT_DATA_PERIOD),
            data_interval=settings.get("data_interval", DEFAULT_DATA_INTERVAL),
        )

    # ── Validate ticker count ─────────────────────────────────────────────
    if len(tickers) < 20:
        print(f"  [!] WARNING: Only {len(tickers)} tickers loaded (expected ~500).")
        print(f"  [!] S&P 500 list may have failed to download. Results will be incomplete!")

    # ── SPY + VIX ──────────────────────────────────────────────────────────
    spy_df = None
    vix_level = None
    if isinstance(data.columns, pd.MultiIndex):
        if "SPY" in data.columns.levels[0]:
            spy_df = data["SPY"].dropna()
        if "^VIX" in data.columns.levels[0]:
            vix_df = data["^VIX"].dropna()
            if "Close" in vix_df:
                vix_level = _safe(vix_df["Close"].iloc[-1])

    if spy_df is not None and len(spy_df) < 200:
        print(f"  [!] WARNING: SPY data has only {len(spy_df)} bars (need 200+). Regime detection may be inaccurate.")

    # ── Regime ─────────────────────────────────────────────────────────────
    regime_info = _detect_market_regime(spy_df) if spy_df is not None else {"regime": "UNKNOWN", "score": 50, "adx": 20}
    regime = regime_info["regime"]
    adx = regime_info["adx"]

    print(f"  Regime: {regime}  |  ADX: {adx:.0f}  |  VIX: {vix_level:.1f}" if vix_level else f"  Regime: {regime}  |  ADX: {adx:.0f}")

    # VIX panic — warn but continue scanning with caution
    if vix_level and vix_level > VIX_PANIC_THRESHOLD:
        print(f"  [!] VIX PANIC ({vix_level:.1f}) — scanning with EXTREME caution.")
        print("      Volatility is very high. Only the strongest setups will appear.")

    if regime == "STRONG_BEAR":
        # Backtest evidence: 36.5% WR in STRONG_BEAR, avg -1.05% at 3D.
        # Still scan but with maximum caution — bear_boost already raises thresholds by 2.0.
        print("  [!] STRONG BEAR — scanning with MAXIMUM caution (thresholds raised).")
        print("      Position sizes should be MINIMAL. Consider staying in cash.")

    # ── Top sectors ────────────────────────────────────────────────────────
    top_sectors = []
    sector_map = {}
    sector_frames = {}
    if isinstance(data.columns, pd.MultiIndex):
        for sector_name, etf_ticker in SECTOR_ETFS.items():
            if etf_ticker in data.columns.levels[0]:
                sdf = data[etf_ticker].dropna()
                if isinstance(sdf, pd.DataFrame) and not sdf.empty and "Close" in sdf:
                    sector_frames[etf_ticker] = sdf
        if sector_frames:
            top_sectors = SectorAnalyzer(sector_frames).get_top_sectors(top_n=3, lookback_days=20)

    # Build sector map from SectorMapper if available
    try:
        from titan.market import SectorMapper
        sm = SectorMapper()
        for t in tickers:
            sector_map[t] = sm.get_sector(t)
    except Exception:
        pass

    if top_sectors:
        print(f"  Top Sectors: {', '.join(top_sectors)}")

    # ── Breadth ────────────────────────────────────────────────────────────
    bullish = 0
    checked = 0
    for t in tickers:
        try:
            if isinstance(data.columns, pd.MultiIndex) and t in data.columns.levels[0]:
                tdf = data[t]["Close"].dropna()
                if len(tdf) >= 20:
                    sma = tdf.rolling(20).mean().iloc[-1]
                    if _safe(tdf.iloc[-1]) > _safe(sma):
                        bullish += 1
                    checked += 1
        except Exception:
            pass
    market_score = (bullish / checked * 100) if checked > 0 else 50
    print(f"  Breadth: {bullish}/{checked} above SMA20 ({market_score:.0f}%)")

    # ── Dynamic strength floor (may be overridden by optimizer) ─────────
    strength_floor = _watchlist_min_strength(market_score, "GENERIC")

    # ── Weekly auto-optimization ──────────────────────────────────────────
    try:
        if should_run_weekly_optimization():
            # Build stock frames for optimization
            _opt_frames = {}
            if isinstance(data.columns, pd.MultiIndex):
                for t in tickers[:30]:
                    if t in data.columns.levels[0]:
                        tdf = data[t].dropna()
                        if len(tdf) >= 200:
                            _opt_frames[t] = tdf
            if len(_opt_frames) >= 5:
                run_weekly_optimization(_opt_frames, spy_df)
        opt_params = load_optimized_params()
    except Exception:
        opt_params = {}

    # Apply optimized strength floor if available
    if opt_params:
        if regime in ("STRONG_BEAR", "BEAR") and "strength_floor_bear" in opt_params:
            strength_floor = max(strength_floor, opt_params["strength_floor_bear"] - 1.0)
        elif regime in ("STRONG_BULL", "BULL") and "strength_floor_bull" in opt_params:
            strength_floor = max(strength_floor, opt_params["strength_floor_bull"] - 1.0)
        elif "strength_floor_sideways" in opt_params:
            strength_floor = max(strength_floor, opt_params["strength_floor_sideways"] - 1.0)

    # ── Build stock frames dict + family prior cache ──────────────────────
    stock_frames = {}
    if isinstance(data.columns, pd.MultiIndex):
        for t in tickers:
            if t in data.columns.levels[0]:
                tdf = data[t].dropna()
                if len(tdf) >= 60:
                    stock_frames[t] = tdf
    family_priors: dict = {}
    family_lock = threading.Lock()

    # ── Scan all tickers ───────────────────────────────────────────────────
    print(f"  Scanning {len(tickers)} stocks...")

    results = []
    predictive_candidates = []

    def _analyze(ticker):
        try:
            tdf = stock_frames.get(ticker)
            if tdf is not None and not tdf.empty:
                return _process_ticker(
                    ticker, tdf, spy_df, regime_info, market_score,
                    strength_floor, top_sectors,
                    family_priors=family_priors,
                    stock_frames=stock_frames,
                    family_lock=family_lock,
                )
        except Exception:
            traceback.print_exc()
            print(f"  [ERROR] _analyze failed for {ticker}", file=sys.stderr)
        return None, None

    max_workers = int(settings.get("max_workers", DEFAULT_MAX_WORKERS))
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_analyze, t): t for t in tickers}
        done = 0
        for future in concurrent.futures.as_completed(futures):
            done += 1
            if done % 50 == 0:
                print(f"    {done}/{len(tickers)}...", end="\r")
            sig_result, pred_result = future.result()
            if sig_result is not None:
                sig_result["sector"] = sector_map.get(sig_result["ticker"], "")
                results.append(sig_result)
            if pred_result is not None:
                pred_result["sector"] = sector_map.get(pred_result["ticker"], "")
                predictive_candidates.append(pred_result)

    print(f"  Scan complete: {len(results)} signals, {len(predictive_candidates)} predictive from {len(tickers)} stocks")

    # ── Sort and classify into tiers ───────────────────────────────────────
    validated = sorted([r for r in results if r["tier"] == "VALIDATED"], key=lambda r: -r["trade_score"])
    active = sorted([r for r in results if r["tier"] == "ACTIVE"], key=lambda r: -r["trade_score"])
    opportunity = sorted([r for r in results if r["tier"] == "OPPORTUNITY"], key=lambda r: -r["trade_score"])
    watchlist = sorted([r for r in results if r["tier"] == "WATCHLIST"], key=lambda r: -r["trade_score"])

    # ── Predictive radar — top coiled springs ──────────────────────────────
    # Sort by direction first (UP > others), then by score
    dir_order = {"UP": 0, "LEAN_UP": 1, "NEUTRAL": 2, "LEAN_DOWN": 3, "DOWN": 4}
    predictive_radar = sorted(
        predictive_candidates,
        key=lambda r: (dir_order.get(r["direction"], 2), -r["pred_score"]),
    )[:20]

    # ── Portfolio risk filter ─────────────────────────────────────────────
    open_positions = settings.get("open_positions", {})
    pf = PortfolioFilter(open_positions=open_positions)
    actionable = sorted(validated + active, key=lambda r: -r["trade_score"])
    accepted, portfolio_rejected = pf.filter(actionable)

    if portfolio_rejected:
        rej_reasons = {}
        for r in portfolio_rejected:
            reason = r.get("_rejected", "unknown")
            rej_reasons[reason] = rej_reasons.get(reason, 0) + 1
        print(f"  Portfolio filter: {len(portfolio_rejected)} signals blocked — {rej_reasons}")

    # ── Build stalk orders ─────────────────────────────────────────────────
    # Include VALIDATED (fresh) + ACTIVE (1-2 bars old, still in buy zone).
    # ACTIVE signals are real setups — just not brand-new. Excluding them
    # means the scanner almost never produces actionable orders.
    # Sort deterministically: tier priority (VALIDATED first), then trade_score desc.
    tier_priority = {"VALIDATED": 0, "ACTIVE": 1}
    stalk_candidates = sorted(
        [r for r in accepted if r["tier"] in ("VALIDATED", "ACTIVE")],
        key=lambda r: (tier_priority.get(r["tier"], 9), -r["trade_score"], r["ticker"]),
    )
    stalk_orders = []
    for r in stalk_candidates:
        if len(stalk_orders) >= MAX_POSITIONS:
            break
        if r["qty"] <= 0:
            continue
        # Stability: skip grade C (no statistical edge)
        if r.get("grade") == "C":
            continue
        # Stability: require strength >= dynamic floor + 1.0 buffer
        # (relaxed from +2.0 — too strict was producing 0 orders)
        sig_floor = _validated_min_strength(market_score, r["signal_type"])
        if r["signal_strength"] < sig_floor + 1.0:
            continue
        # Extra check for ACTIVE: require positive backtest evidence
        if r["tier"] == "ACTIVE" and r.get("bt_trades", 0) >= 10:
            if r.get("bt_win_rate", 0) < 45:
                continue
        # Profile evidence is noisy, but when it has enough analogs and both
        # expected return and up-probability are negative, it is not a swing buy.
        if int(r.get("profile_samples", 0) or 0) >= SHORT_EDGE_MIN_ANALOGS:
            if float(r.get("move_expected_return", 0) or 0) <= 0 and float(r.get("move_up_prob", 0) or 0) < 50:
                continue
        stalk_orders.append({
            "ticker": r["ticker"],
            "signal_type": r["signal_type"],
            "grade": r.get("grade", "C"),
            "tier": r["tier"],
            "sector": r.get("sector", ""),
            "price": r["price"],
            "qty": r["qty"],
            "limit_price": r["max_buy_price"],
            "buy_zone_low": r["buy_zone_low"],
            "buy_zone_high": r["buy_zone_high"],
            "max_buy_price": r["max_buy_price"],
            "stop_price": r["stop"],
            "target_price": r["target"],
            "risk_dollars": r["risk_dollars"],
            "buy_zone": f"${r['buy_zone_low']:.2f} - ${r['buy_zone_high']:.2f}",
            "time_stop_days": r["time_stop_days"],
            "entry_note": r["entry_note"],
            "trade_score": r["trade_score"],
            "rr_ratio": r.get("rr_ratio", 0),
            "rsi": r.get("rsi", 0),
            "volume_ratio": r.get("volume_ratio", 0),
            "change_5d": r.get("change_5d", 0),
            "vol_contraction": r.get("vol_contraction", 0),
            "bt_win_rate": r.get("bt_win_rate", 0),
            "bt_trades": r.get("bt_trades", 0),
            "move_up_prob": r.get("move_up_prob", 0),
            "move_expected_return": r.get("move_expected_return", 0),
            "profile_samples": r.get("profile_samples", 0),
        })

    # ── Log signals + paper trade tracking ─────────────────────────────────
    tracker = SignalTracker()
    for r in validated + active:
        tracker.log_signal(r)

    # Paper trade: log actionable signals for forward verification
    try:
        from titan.paper_trader import PaperTrader
        pt = PaperTrader()
        # Update existing trades with fresh market data
        # Include all tickers with open paper trades, not just current signals
        ticker_data = {}
        if isinstance(data.columns, pd.MultiIndex):
            # Start with tickers from current signals
            update_tickers = set(sig["ticker"] for sig in validated + active)
            # Add tickers with open paper trades so they get price updates too
            if hasattr(pt, 'get_open_tickers'):
                open_trade_tickers = set(pt.get_open_tickers())
            else:
                # Fallback: read from trades list directly
                open_trade_tickers = set(
                    t["ticker"] for t in getattr(pt, "trades", [])
                    if t.get("status") in ("PENDING_FILL", "FILLED")
                )
            update_tickers |= open_trade_tickers
            for t in update_tickers:
                if t in data.columns.levels[0]:
                    ticker_data[t] = data[t].dropna()
        pt.update(ticker_data)
        # Log new signals — only top VALIDATED + top ACTIVE by trade_score.
        # Cap at 10 to avoid polluting paper trade tracking with 100+ signals.
        paper_candidates = sorted(validated + active, key=lambda r: -r["trade_score"])[:10]
        logged = 0
        for r in paper_candidates:
            if pt.log_signal(r):
                logged += 1
        if logged > 0:
            print(f"  Paper trader: {logged} new signals logged")
        # Show summary if we have closed trades
        summary = pt.get_summary()
        if summary.get("closed", 0) > 0:
            print(f"  Paper trader: {summary['closed']} closed trades | WR: {summary.get('win_rate', 0):.1f}% | Avg: {summary.get('avg_return', 0):+.2f}%")
    except Exception as e:
        pass  # Paper trader is optional, don't break scanner

    scan_time = time.time() - start_time

    # ── Print summary ──────────────────────────────────────────────────────
    _print_summary(regime, vix_level, market_score, adx, top_sectors,
                   validated, active, opportunity, watchlist, stalk_orders,
                   predictive_radar, scan_time)

    return {
        "regime": regime,
        "regime_score": regime_info["score"],
        "adx": adx,
        "vix": vix_level,
        "market_score": market_score,
        "top_sectors": top_sectors,
        "validated": validated,
        "active": active,
        "opportunity": opportunity,
        "watchlist": watchlist,
        "predictive_radar": predictive_radar,
        "stalk_orders": stalk_orders,
        "all_results": results,
        "portfolio_rejected": len(portfolio_rejected),
        "scan_time": round(scan_time, 1),
        "total_scanned": len(tickers),
        "total_signals": len(results),
        "total_predictive": len(predictive_candidates),
        "scan_timestamp": datetime.now().isoformat(),
    }


def _empty_result(regime, vix, scan_time):
    return {
        "regime": regime, "regime_score": 0, "adx": 0, "vix": vix,
        "market_score": 0, "top_sectors": [],
        "validated": [], "active": [], "opportunity": [], "watchlist": [],
        "predictive_radar": [], "stalk_orders": [], "all_results": [],
        "scan_time": round(scan_time, 1), "total_scanned": 0,
        "total_signals": 0, "total_predictive": 0,
        "scan_timestamp": datetime.now().isoformat(),
    }


# ---------------------------------------------------------------------------
# Pretty print
# ---------------------------------------------------------------------------

def _print_summary(regime, vix, market_score, adx, top_sectors,
                   validated, active, opportunity, watchlist, stalk_orders,
                   predictive_radar, scan_time):
    print(f"\n{'=' * 70}")
    print(f"  SCAN RESULTS  ({scan_time:.1f}s)")
    print(f"{'=' * 70}")
    v = f"VIX {vix:.1f}" if vix else "VIX --"
    print(f"  {regime}  |  {v}  |  ADX {adx:.0f}  |  Breadth {market_score:.0f}%")
    if top_sectors:
        print(f"  Hot Sectors: {', '.join(top_sectors)}")
    print(f"\n  VALIDATED: {len(validated)}  |  ACTIVE: {len(active)}  |  "
          f"OPPORTUNITY: {len(opportunity)}  |  WATCHLIST: {len(watchlist)}")

    if stalk_orders:
        print(f"\n{'-' * 70}")
        print("  >> STALK ORDERS -- Place these and go to sleep")
        print(f"{'-' * 70}")
        for o in stalk_orders:
            print(f"  [{o.get('grade','C')}] {o['ticker']:<6} {o['signal_type']:<10}  "
                  f"LIMIT BUY ${o['limit_price']:.2f}  |  "
                  f"STOP ${o['stop_price']:.2f}  |  "
                  f"TARGET ${o['target_price']:.2f}  |  "
                  f"{o['qty']} shares  |  "
                  f"Risk ${o['risk_dollars']:.0f}  |  "
                  f"Score {o['trade_score']:.1f}")
            print(f"         Buy Zone: {o['buy_zone']}  |  "
                  f"Time Stop: {o['time_stop_days']}D  |  {o['entry_note'][:60]}")
    else:
        print("\n  No stalk orders tonight. Wait for setups.")

    if validated:
        print(f"\n{'-' * 70}")
        print("  VALIDATED -- Fresh signals inside buy zone")
        print(f"{'-' * 70}")
        _print_tier(validated)

    if active:
        print(f"\n{'-' * 70}")
        active_shown = min(20, len(active))
        print(f"  ACTIVE -- Still inside entry plan (showing top {active_shown} of {len(active)})")
        print(f"{'-' * 70}")
        _print_tier(active[:20])

    if opportunity:
        print(f"\n{'-' * 70}")
        print("  OPPORTUNITY -- Extended or weaker conviction (monitor for pullback entry)")
        print(f"{'-' * 70}")
        _print_tier(opportunity[:10])

    if watchlist:
        print(f"\n{'-' * 70}")
        print("  WATCHLIST -- Not yet actionable (watch for setup improvement)")
        print(f"{'-' * 70}")
        _print_tier(watchlist[:15])

    if predictive_radar:
        print(f"\n{'-' * 70}")
        print("  PREDICTIVE RADAR -- Coiled springs (before they move)")
        print(f"{'-' * 70}")
        print(f"  {'Ticker':<7} {'Dir':<8} {'Score':>5} {'Price':>8} {'Trigger':>8} {'Stop':>8} "
              f"{'Target':>8} {'Risk%':>5} {'RSI':>4} {'VC':>5} {'Wk':>6} {'Why'}")
        print(f"  {'-' * 110}")
        for r in predictive_radar[:15]:
            why = " | ".join(r.get("pred_reasons", [])[:2])
            print(f"  {r['ticker']:<7} {r['direction']:<8} {r['pred_score']:>5.1f} "
                  f"${r['price']:>7.2f} ${r['trigger']:>7.2f} ${r['stop']:>7.2f} "
                  f"${r['target']:>7.2f} {r['risk_pct']:>4.1f}% {r['rsi']:>4.0f} "
                  f"{r['vol_contraction']:>5.2f} {r['weekly_trend']:>6} {why}")


def _print_tier(items):
    print(f"  {'':>3} {'Ticker':<7} {'Type':<10} {'Score':>5} {'Price':>8} {'Stop':>8} {'Target':>8} "
          f"{'Buy Zone':<20} {'BT':>7} {'TS':>3} {'Wk':>6} {'Reasons'}")
    print(f"  {'-' * 115}")
    for r in items:
        bz = f"${r['buy_zone_low']:.2f}-${r['buy_zone_high']:.2f}"
        reasons_str = " | ".join(r.get("reasons", [])[:2])
        g = r.get("grade", "C")
        bt_str = f"{r.get('bt_win_rate',0):.0f}%/{r.get('bt_trades',0)}" if r.get('bt_trades', 0) > 0 else "  n/a"
        ts = r.get("time_stop_days", 0)
        print(f"  [{g}] {r['ticker']:<7} {r['signal_type']:<10} {r['trade_score']:>5.1f} "
              f"${r['price']:>7.2f} ${r['stop']:>7.2f} ${r['target']:>7.2f} "
              f"{bz:<20} {bt_str:>7} {ts:>3}d {r['weekly_trend']:>6} {reasons_str}")
