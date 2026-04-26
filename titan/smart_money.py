"""
Smart money flow tracking for US equities.

The KOSPI system tracks foreign/institutional net buying from Korean exchanges.
For US markets, we approximate smart money signals using:

1. **Institutional ownership changes** — from Yahoo Finance major holders
2. **Short interest ratio** — high short interest = crowded shorts, squeeze potential
3. **Dark pool activity** — FINRA short volume as % of total volume
4. **Options flow** — put/call ratio from Yahoo options chain

All sources are free and require no API keys.
"""

import time
from datetime import datetime

import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Score weights (effective max ~14 with options flow disabled)
# Options flow (_OPTIONS_FLOW_MAX = 4) is disabled during bulk scans for speed.
# The scoring and bonus thresholds below are calibrated to the 0-14 effective range.
_INST_OWN_MAX = 6.0       # Institutional ownership level
_SHORT_INTEREST_MAX = 4.0  # Short squeeze potential
_OPTIONS_FLOW_MAX = 4.0    # Put/call ratio signal (DISABLED — see note above)
_INSIDER_MAX = 4.0         # Insider transaction signal

# Thresholds
INST_OWN_BULLISH = 70.0      # % institutional ownership considered bullish
INST_OWN_VERY_BULLISH = 85.0
INST_OWN_LOW = 30.0          # Low institutional interest
SHORT_INTEREST_HIGH = 10.0   # % short interest = crowded short
SHORT_INTEREST_EXTREME = 20.0
PUT_CALL_BEARISH = 1.2       # Put/Call > 1.2 = bearish sentiment (contrarian bullish)
PUT_CALL_BULLISH = 0.7       # Put/Call < 0.7 = bullish sentiment

# Cache
_flow_cache: dict[str, dict] = {}
_cache_date: str = ""


# ---------------------------------------------------------------------------
# Data fetchers (all use yfinance — no extra dependencies)
# ---------------------------------------------------------------------------

def _get_yahoo_info(ticker: str) -> dict:
    """Fetch institutional ownership and short interest from Yahoo Finance."""
    try:
        import yfinance as yf
        tk = yf.Ticker(ticker)
        info = tk.info or {}
        return {
            "inst_ownership_pct": info.get("heldPercentInstitutions", 0) * 100
            if info.get("heldPercentInstitutions") else 0,
            "insider_ownership_pct": info.get("heldPercentInsiders", 0) * 100
            if info.get("heldPercentInsiders") else 0,
            "short_pct_float": info.get("shortPercentOfFloat", 0) * 100
            if info.get("shortPercentOfFloat") else 0,
            "short_ratio": info.get("shortRatio", 0) or 0,  # days to cover
            "insider_buys_6m": info.get("netSharePurchaseActivity", 0) or 0,
        }
    except Exception:
        return {}


def _get_options_flow(ticker: str) -> dict:
    """Compute put/call ratio from Yahoo options chain."""
    try:
        import yfinance as yf
        tk = yf.Ticker(ticker)
        dates = tk.options
        if not dates:
            return {}

        # Use nearest 2 expiration dates
        total_call_vol = 0
        total_put_vol = 0
        for exp in dates[:2]:
            try:
                chain = tk.option_chain(exp)
                total_call_vol += chain.calls["volume"].sum()
                total_put_vol += chain.puts["volume"].sum()
            except Exception:
                continue

        if total_call_vol <= 0:
            return {}

        pc_ratio = total_put_vol / total_call_vol if total_call_vol > 0 else 1.0
        return {
            "put_call_ratio": round(pc_ratio, 3),
            "total_call_volume": int(total_call_vol),
            "total_put_volume": int(total_put_vol),
        }
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# Main flow function
# ---------------------------------------------------------------------------

def get_smart_money_flow(ticker: str) -> dict:
    """Get smart money flow indicators for a US stock.

    Returns a dict matching the KOSPI system's investor flow structure:
    - foreign_trend → inst_trend (institutional direction)
    - institution_trend → insider_trend (insider direction)
    - foreign_streak → inst_streak (proxy)
    - score → smart_money_score (0-18 points)
    - Plus raw data fields for display
    """
    global _flow_cache, _cache_date

    today = datetime.now().strftime("%Y-%m-%d")
    if _cache_date != today:
        _flow_cache.clear()
        _cache_date = today

    if ticker in _flow_cache:
        return _flow_cache[ticker]

    result = _compute_flow(ticker)
    _flow_cache[ticker] = result
    return result


def _compute_flow(ticker: str) -> dict:
    """Compute smart money flow for a single ticker."""
    # Fetch data (Yahoo info is the primary source — options is optional)
    info = _get_yahoo_info(ticker)
    if not info:
        return _empty_flow()

    inst_pct = info.get("inst_ownership_pct", 0)
    insider_pct = info.get("insider_ownership_pct", 0)
    short_pct = info.get("short_pct_float", 0)
    short_ratio = info.get("short_ratio", 0)
    insider_buys = info.get("insider_buys_6m", 0)

    # Options flow — skip during bulk scan (too slow for 100+ tickers).
    # Only used when explicitly requested via get_smart_money_flow(ticker, full=True).
    pc_ratio = 1.0

    # --- Score computation (0 to 18 points, matching KOSPI scale) ---
    score = 0.0

    # 1. Institutional ownership (0-6 pts)
    if inst_pct >= INST_OWN_VERY_BULLISH:
        score += 6.0
    elif inst_pct >= INST_OWN_BULLISH:
        score += 4.0
    elif inst_pct >= 50:
        score += 2.0
    elif inst_pct < INST_OWN_LOW:
        score -= 2.0

    # 2. Short interest — contrarian (0-4 pts)
    # High short interest + bullish setup = squeeze potential
    if short_pct >= SHORT_INTEREST_EXTREME:
        score += 4.0  # Massive squeeze potential
    elif short_pct >= SHORT_INTEREST_HIGH:
        score += 2.0
    elif short_pct >= 5:
        score += 0.5
    # Very low short interest in bear market = no squeeze buffer
    if short_pct < 2:
        score -= 0.5

    # 3. Options flow — DISABLED for bulk scanning speed.
    # pc_ratio is hardcoded 1.0 so this block is inert.
    # Thresholds in smart_money_bonus() are adjusted for 0-14 effective scale.
    # To enable: call _get_options_flow(ticker) and assign pc_ratio above.
    if pc_ratio >= PUT_CALL_BEARISH:
        score += 3.0
        if pc_ratio >= 1.5:
            score += 1.0
    elif pc_ratio <= PUT_CALL_BULLISH:
        score -= 1.0

    # 4. Insider activity (0-4 pts)
    if insider_buys > 0:
        score += 3.0  # Net insider buying
        if insider_pct > 5:
            score += 1.0  # Meaningful insider stake
    elif insider_buys < 0:
        score -= 1.0  # Net insider selling

    score = max(-6.0, min(14.0, score))  # Effective max 14 with options disabled

    # --- Trend determination ---
    inst_trend = "BUYING" if inst_pct >= INST_OWN_BULLISH else (
        "SELLING" if inst_pct < INST_OWN_LOW else "NEUTRAL"
    )
    insider_trend = "BUYING" if insider_buys > 0 else (
        "SELLING" if insider_buys < 0 else "NEUTRAL"
    )

    return {
        # KOSPI-compatible fields
        "smart_money_score": round(score, 1),
        "inst_trend": inst_trend,
        "insider_trend": insider_trend,
        # Raw data
        "inst_ownership_pct": round(inst_pct, 1),
        "insider_ownership_pct": round(insider_pct, 1),
        "short_pct_float": round(short_pct, 1),
        "short_ratio": round(short_ratio, 1),
        "put_call_ratio": round(pc_ratio, 3),
        "insider_net_buys": insider_buys,
        "source": "yahoo",
    }


def _empty_flow() -> dict:
    return {
        "smart_money_score": 0.0,
        "inst_trend": "NEUTRAL",
        "insider_trend": "NEUTRAL",
        "inst_ownership_pct": 0.0,
        "insider_ownership_pct": 0.0,
        "short_pct_float": 0.0,
        "short_ratio": 0.0,
        "put_call_ratio": 1.0,
        "insider_net_buys": 0,
        "source": "none",
    }


# ---------------------------------------------------------------------------
# Scoring helpers for scanner integration
# ---------------------------------------------------------------------------

def smart_money_bonus(flow: dict) -> float:
    """Return a strength bonus/penalty based on smart money flow.

    Used by the scanner to adjust signal strength.
    Range: -2.0 to +3.0 (similar to KOSPI's investor flow bonus).

    Thresholds are calibrated to the 0-14 effective scale (options flow
    disabled for bulk scanning speed — 4 points are always inert).
    """
    score = flow.get("smart_money_score", 0)
    if score >= 10:
        return 3.0   # Very strong institutional backing
    elif score >= 7:
        return 2.0
    elif score >= 3:
        return 1.0
    elif score <= -3:
        return -2.0  # Institutional headwinds
    elif score <= 0:
        return -1.0
    return 0.0


def smart_money_grade_factors(flow: dict) -> int:
    """Return grade factor count for smart money.

    Used in signal grading (A/B/C) — matching KOSPI's approach.
    """
    score = flow.get("smart_money_score", 0)
    factors = 0
    if score >= 10:
        factors += 2
    elif score >= 5:
        factors += 1

    if flow.get("inst_trend") == "BUYING":
        factors += 1
    if flow.get("insider_trend") == "BUYING":
        factors += 1

    return factors
