"""
Position-level risk validation and trailing-stop logic.

Two jobs:
  1. Before accepting a new manual position, check that the dollar risk
     (shares × (entry − stop)) is sane relative to ACCOUNT_SIZE. Hard-cap at
     1% of account, soft-warn at 0.5%.
  2. For open positions, compute an effective stop = max(hard_stop, trailing).
     Trailing kicks in once the position is in profit, locking gains as the
     high-water mark rises.
"""
from __future__ import annotations

from typing import Optional

from titan.config import ACCOUNT_SIZE, MAX_RISK_PCT_PER_TRADE

# Risk thresholds (as percent of ACCOUNT_SIZE)
HARD_CAP_PCT = max(MAX_RISK_PCT_PER_TRADE, 1.0)  # never risk more than this
SOFT_WARN_PCT = HARD_CAP_PCT * 0.5               # prompt but allow

# Sanity: refuse entry prices that diverge > 10% from live price
SANITY_PRICE_DIVERGENCE = 0.10

# Trailing stop defaults
DEFAULT_TRAIL_PCT = 0.06              # 6% below highest-since-entry
DEFAULT_TRAIL_ATR_MULT = 2.5          # or 2.5 × ATR, whichever is tighter
DEFAULT_TRAIL_ACTIVATION_R = 1.0      # start trailing once up by 1R from entry


def compute_risk_dollars(
    entry_price: float, stop: Optional[float], shares: float
) -> float:
    """Dollar risk = shares × (entry − stop). If stop is missing or above
    entry (invalid), assume a 5% worst-case loss."""
    entry_price = float(entry_price or 0)
    shares = float(shares or 0)
    if entry_price <= 0 or shares <= 0:
        return 0.0
    if stop and stop > 0 and stop < entry_price:
        return round((entry_price - float(stop)) * shares, 2)
    return round(entry_price * 0.05 * shares, 2)


def validate_position_risk(
    entry_price: float,
    stop: Optional[float],
    shares: float,
    account_size: float = ACCOUNT_SIZE,
) -> dict:
    """Check if the dollar risk on a new position is acceptable.

    Returns {
        'ok': bool,                 # True if under hard cap
        'level': 'OK'|'SOFT'|'HARD',# severity
        'risk_dollars': float,
        'risk_pct': float,          # risk as % of account
        'msg': str | None,
    }
    """
    risk = compute_risk_dollars(entry_price, stop, shares)
    if account_size <= 0:
        return {
            "ok": False,
            "level": "HARD",
            "risk_dollars": risk,
            "risk_pct": 0.0,
            "msg": "ACCOUNT_SIZE is zero or negative — cannot compute risk.",
        }
    pct = risk / account_size * 100.0
    if pct > HARD_CAP_PCT:
        return {
            "ok": False,
            "level": "HARD",
            "risk_dollars": risk,
            "risk_pct": round(pct, 3),
            "msg": (
                f"Risk ${risk:,.2f} ({pct:.2f}% of account) exceeds hard cap "
                f"of {HARD_CAP_PCT:.1f}%. Reduce shares or tighten stop."
            ),
        }
    if pct > SOFT_WARN_PCT:
        return {
            "ok": True,
            "level": "SOFT",
            "risk_dollars": risk,
            "risk_pct": round(pct, 3),
            "msg": (
                f"Risk ${risk:,.2f} ({pct:.2f}% of account) is above the soft "
                f"warn of {SOFT_WARN_PCT:.2f}%. Still under hard cap of {HARD_CAP_PCT:.1f}%."
            ),
        }
    return {
        "ok": True,
        "level": "OK",
        "risk_dollars": risk,
        "risk_pct": round(pct, 3),
        "msg": None,
    }


def validate_entry_price(
    entry_price: float, live_price: Optional[float],
    max_divergence: float = SANITY_PRICE_DIVERGENCE,
) -> dict:
    """Check if user-entered entry_price is close enough to the live price.
    Catches fat-finger typos (e.g. $500 instead of $50).

    If live_price is unknown (market closed, fetch failed), returns OK — we
    don't want to block the user when we have no reference.
    """
    if live_price is None or live_price <= 0 or entry_price <= 0:
        return {"ok": True, "level": "OK", "msg": None, "divergence_pct": None}
    div = abs(entry_price - live_price) / live_price
    if div > max_divergence:
        return {
            "ok": False,
            "level": "HARD",
            "divergence_pct": round(div * 100, 2),
            "msg": (
                f"Entry ${entry_price:.2f} diverges {div*100:.1f}% from live "
                f"price ${live_price:.2f}. Typo? Add force=true to override."
            ),
        }
    return {
        "ok": True,
        "level": "OK",
        "divergence_pct": round(div * 100, 2),
        "msg": None,
    }


def compute_trailing_stop(
    entry_price: float,
    hard_stop: Optional[float],
    highest_since_entry: Optional[float],
    atr: Optional[float] = None,
    trail_pct: float = DEFAULT_TRAIL_PCT,
    trail_atr_mult: float = DEFAULT_TRAIL_ATR_MULT,
    activation_r: float = DEFAULT_TRAIL_ACTIVATION_R,
) -> Optional[float]:
    """Compute the trailing stop level.

    Logic:
      - Trailing only activates once the position is up by `activation_r` R
        from entry (locks in 0R minimum, i.e. breakeven).
      - Raw trail = highest_since_entry × (1 - trail_pct), OR
                    highest_since_entry − (trail_atr_mult × atr) if ATR given.
      - Trail is never lower than entry × 0.999 (breakeven floor once activated).
      - Returns None if no trailing stop applies.
    """
    entry_price = float(entry_price or 0)
    if entry_price <= 0:
        return None
    if not highest_since_entry or highest_since_entry <= entry_price:
        return None

    # Activation: need enough upside to justify trailing
    if hard_stop and hard_stop > 0 and hard_stop < entry_price:
        r_move = (highest_since_entry - entry_price) / (entry_price - hard_stop)
        if r_move < activation_r:
            return None
    else:
        # No hard stop: activate once up > 3%
        if (highest_since_entry - entry_price) / entry_price < 0.03:
            return None

    if atr and atr > 0:
        trailing = highest_since_entry - (trail_atr_mult * atr)
    else:
        trailing = highest_since_entry * (1.0 - trail_pct)

    # Breakeven floor: never trail below entry
    trailing = max(trailing, entry_price * 0.999)
    return round(trailing, 4)


def effective_stop(
    entry_price: float,
    hard_stop: Optional[float],
    highest_since_entry: Optional[float],
    atr: Optional[float] = None,
) -> Optional[float]:
    """Return the larger of (hard_stop, trailing_stop). This is what actually
    protects the position."""
    trail = compute_trailing_stop(
        entry_price, hard_stop, highest_since_entry, atr=atr
    )
    candidates = [s for s in (hard_stop, trail) if s is not None and s > 0]
    if not candidates:
        return None
    return max(candidates)
