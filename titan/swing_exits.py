"""
Swing-trade exit signals — partial profit ladder, time-stop runway, quick-profit.
=================================================================================
The user holds 2–4 business days. The #1 thing that kills a swing book is
staying in stale positions too long; the #2 thing is letting a +6% Day-2
pop round-trip back to flat. This module produces small, opt-in *hints*
(not hard SELL alerts) that the position evaluator can surface to the UI:

  * partial_ladder_signal — "+1R? take 1/3. +2R? take another 1/3."
  * quick_profit_signal   — "+5% in 1 day — consider locking some in."
  * time_stop_approaching — "1 business day until forced exit."

None of these override the hard exits (stop / target / trailing / timestop).
They all run in addition, as non-blocking advisories.
"""
from __future__ import annotations

from typing import Any, Dict, Optional


# Ladder thresholds (R-multiples of risk taken at entry).
LADDER_1R = 1.0
LADDER_2R = 2.0

# Quick-profit heuristic (catches Day-1 pops before they round-trip).
QUICK_PROFIT_PCT = 5.0
QUICK_PROFIT_MAX_DAYS = 2

# Time-stop warning runway (business days).
TIME_STOP_WARN_DAYS = 1


def _r_unit(entry: float, stop: Optional[float]) -> Optional[float]:
    """Dollar-per-share at 1R of risk. Falls back to 5% of entry if stop is
    missing or invalid (above entry). Returns None for degenerate inputs."""
    if not entry or entry <= 0:
        return None
    if stop and stop > 0 and stop < entry:
        return float(entry - stop)
    return float(entry) * 0.05


def partial_ladder_signal(
    entry: float,
    stop: Optional[float],
    current: float,
    shares: float,
) -> Optional[Dict[str, Any]]:
    """Recommend a partial exit based on R-multiple progress.

    Returns the *highest rung currently earned*:
      * r >= 2.0 → "sell another 1/3, trail the rest" (2R rung)
      * 1.0 <= r < 2.0 → "sell 1/3, move stop to breakeven" (1R rung)
      * otherwise → None

    This is *stateless* — it doesn't know whether the user has already acted.
    The UI should always display the highest earned rung; the user ignores it
    after acting. This keeps the backend simple and the advice consistent.
    """
    if not current or current <= 0 or not shares or shares <= 0:
        return None
    r_unit = _r_unit(entry, stop)
    if r_unit is None or r_unit <= 0:
        return None
    r_multiple = (float(current) - float(entry)) / r_unit
    if r_multiple >= LADDER_2R:
        sell_shares = max(1, int(float(shares) * 0.34))
        return {
            "rung": "2R",
            "r_multiple": round(r_multiple, 2),
            "sell_shares": sell_shares,
            "action": "Take another 1/3 off. Trail the rest with the stop.",
        }
    if r_multiple >= LADDER_1R:
        sell_shares = max(1, int(float(shares) * 0.33))
        return {
            "rung": "1R",
            "r_multiple": round(r_multiple, 2),
            "sell_shares": sell_shares,
            "action": "Take 1/3 off. Raise stop to breakeven.",
        }
    return None


def quick_profit_signal(
    entry: float,
    current: float,
    days_held: int,
    pct_threshold: float = QUICK_PROFIT_PCT,
    max_days: int = QUICK_PROFIT_MAX_DAYS,
) -> Optional[Dict[str, Any]]:
    """Fast-pop warning for a 2–4 day hold: if we're up a lot in <= 2 days,
    the odds of giving it back are high. Nudges the user to book partial.

    Returns None unless days_held is in [0, max_days] and gain >= pct_threshold.
    """
    if days_held is None or days_held < 0 or days_held > max_days:
        return None
    if not entry or entry <= 0 or not current or current <= 0:
        return None
    gain_pct = (float(current) - float(entry)) / float(entry) * 100.0
    if gain_pct < pct_threshold:
        return None
    day_label = "today" if days_held == 0 else (
        f"{days_held} business day{'s' if days_held != 1 else ''}"
    )
    return {
        "gain_pct": round(gain_pct, 2),
        "days_held": int(days_held),
        "action": (
            f"+{gain_pct:.1f}% in {day_label} — consider locking in partial "
            f"profit before a fade."
        ),
    }


def time_stop_approaching(
    days_held: Optional[int],
    time_stop_days: Optional[int],
    threshold_days: int = TIME_STOP_WARN_DAYS,
) -> bool:
    """True when we're within `threshold_days` of the hard time-stop but not
    yet past it. Use this to surface a WARN_TIMESTOP state so the user can
    plan their exit before the forced-exit alert fires at market open."""
    if time_stop_days is None or days_held is None:
        return False
    ts = int(time_stop_days)
    dh = int(days_held)
    if ts <= 0:
        return False
    if dh >= ts:
        return False  # already past — let SELL_TIMESTOP handle it
    return (ts - dh) <= max(0, int(threshold_days))
