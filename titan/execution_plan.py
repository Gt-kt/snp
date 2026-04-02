"""
Titan Execution Plan — Anti-Chase Entry Logic
==============================================
Ported from the KOSPI architecture. Builds concrete buy zones,
max buy prices, and time stops for every signal.

The key insight: never chase. If the stock already moved past
your buy zone, skip it. There will be another setup.
"""

from titan.signal_detector import EARLY_SIGNAL_TYPES, BREAKOUT_SIGNAL_TYPES, _safe

# ---------------------------------------------------------------------------
# Gap risk classification
# ---------------------------------------------------------------------------

def classify_gap_risk(avg_open_gap_pct: float = 0.0, gap_above_1pct_prob: float = 0.0) -> str:
    if avg_open_gap_pct > 1.5 or gap_above_1pct_prob > 0.30:
        return "HIGH"
    if avg_open_gap_pct < 0.5 and gap_above_1pct_prob < 0.10:
        return "LOW"
    return "MED"


# ---------------------------------------------------------------------------
# Execution plan builder
# ---------------------------------------------------------------------------

def build_execution_plan(
    signal_type: str,
    price: float,
    stop: float,
    target: float,
    atr: float,
    best_horizon: int = 3,
    move_up_prob: float = 0.0,
    move_expected_return: float = 0.0,
    avg_open_gap_pct: float = 0.0,
    gap_above_1pct_prob: float = 0.0,
) -> dict:
    """Build a manual entry plan that limits chasing and defines a time stop.

    Returns dict with:
      buy_zone_low, buy_zone_high, max_buy_price,
      cancel_above_pct, time_stop_days, entry_style, entry_note, gap_risk
    """
    if price <= 0 or stop <= 0 or stop >= price or target <= price:
        return {
            "buy_zone_low": price,
            "buy_zone_high": price,
            "max_buy_price": price,
            "cancel_above_pct": 0.0,
            "time_stop_days": 3,
            "entry_style": "invalid",
            "entry_note": "Trade plan unavailable.",
            "gap_risk": "HIGH",
        }

    best_horizon = max(1, int(best_horizon or 1))
    atr = max(atr, price * 0.01)
    gap_risk = classify_gap_risk(avg_open_gap_pct, gap_above_1pct_prob)

    # Tuning per signal type
    min_rr = 1.4 if signal_type in EARLY_SIGNAL_TYPES else 1.2
    atr_buffer = 0.45 if signal_type in EARLY_SIGNAL_TYPES else 0.30
    pct_cap = 0.018 if signal_type in EARLY_SIGNAL_TYPES else 0.012
    pullback_buffer = 0.30 if signal_type in EARLY_SIGNAL_TYPES else 0.12
    entry_style = "pullback"
    entry_note = "Prefer entries near the buy zone, not after a fast opening spike."

    if signal_type in BREAKOUT_SIGNAL_TYPES:
        entry_style = "tight_breakout"
        pullback_buffer = 0.05
        entry_note = "Buy only on a controlled entry. Skip if it opens above max buy price."
    elif signal_type == "PULLBACK":
        entry_style = "trend_pullback"
        atr_buffer = 0.35
        pct_cap = 0.014
        pullback_buffer = 0.35
        entry_note = "Stalk entries near 20-day support. Skip if pullback loses that level."
    elif signal_type == "DIP_BUY":
        entry_style = "bounce_confirmation"
        atr_buffer = 0.25
        pct_cap = 0.010
        pullback_buffer = 0.25
        entry_note = "Buy only if the bounce holds. Exit fast if no follow-through."

    # Adjust for gap risk
    if gap_risk == "HIGH":
        pct_cap *= 0.85
        atr_buffer *= 0.90
        if signal_type in BREAKOUT_SIGNAL_TYPES:
            entry_style = "breakout_retest"
        entry_note = f"{entry_note} Gap risk is high — do NOT chase the open."
    elif gap_risk == "LOW" and signal_type in EARLY_SIGNAL_TYPES:
        entry_note = f"{entry_note} Gap risk is low — safe to stalk inside buy zone."

    # Calculate buy zone
    rr_cap = (target + (min_rr * stop)) / (1.0 + min_rr) if target > stop else price
    max_buy_price = min(price + atr * atr_buffer, price * (1 + pct_cap), rr_cap)
    buy_zone_low = max(stop * 1.02, price - atr * pullback_buffer)

    if signal_type in BREAKOUT_SIGNAL_TYPES:
        buy_zone_low = min(price, max_buy_price)

    buy_zone_low = min(buy_zone_low, max_buy_price)

    # Time stop
    time_stop_days = min(7, best_horizon + (1 if signal_type in EARLY_SIGNAL_TYPES else 0))
    if move_up_prob >= 60 and move_expected_return >= 0.8:
        time_stop_days = min(8, time_stop_days + 1)
    if gap_risk == "HIGH":
        time_stop_days = max(1, time_stop_days - 1)

    return {
        "buy_zone_low": round(buy_zone_low, 2),
        "buy_zone_high": round(max_buy_price, 2),
        "max_buy_price": round(max_buy_price, 2),
        "cancel_above_pct": round(max(0.0, (max_buy_price / price - 1.0) * 100), 2),
        "time_stop_days": max(1, time_stop_days),
        "entry_style": entry_style,
        "entry_note": entry_note,
        "gap_risk": gap_risk,
    }


# ---------------------------------------------------------------------------
# Anti-chase checks
# ---------------------------------------------------------------------------

def is_entry_buyable(price: float, plan: dict, allowed_chase_pct: float = 0.0) -> bool:
    """Check if current price is still within the execution plan."""
    price = _safe(price)
    max_buy = _safe(plan.get("max_buy_price"), price)
    if price <= 0 or max_buy <= 0:
        return False
    return price <= max_buy * (1.0 + allowed_chase_pct / 100.0)


def fresh_entry_is_buyable(price: float, plan: dict) -> bool:
    """Fresh validated names must sit inside their own entry plan — zero chase tolerance."""
    return is_entry_buyable(price, plan, allowed_chase_pct=0.0)


# Chase tolerance by tier
VALIDATED_MAX_CHASE_PCT = 0.0    # Fresh validated: must be inside plan
ACTIVE_MAX_CHASE_PCT = 0.20      # Active setups: tiny chase OK
OPPORTUNITY_MAX_CHASE_PCT = 1.0   # Opportunity: slight chase if plan allows pullback
