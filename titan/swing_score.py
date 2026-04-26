"""
Swing-trade setup scoring — tuned for 2–4 day holds.
=====================================================
The user is a swing trader (buys at US open, sells 2–4 days later). This module
scores each scanner setup on how well it matches a short-hold profile and
produces:
  - a 0–100 composite score
  - a verdict label (TOP_PICK / STRONG / DECENT / WEAK)
  - a breakdown dict (so the UI can tell the user *why* it's a top pick)

Why each factor matters for a 2–4 day hold:
  * Momentum (change_5d):   no uptrend = no swing. Too hot = top is near.
  * Volume surge:           institutions need to be buying for us to ride.
  * R/R ratio:              a 2–4 day hold at 1:1 is a terrible use of capital.
  * RSI fit:                >75 = exhaustion, <30 = falling knife.
  * Historical edge:        bt_win_rate + sample size filter survivorship bias.
  * Proximity:              inside buy-zone = fresh. Past max-buy = chasing.
  * Grade:                  engine's A/B/C already encodes broad setup quality.
  * Base tightness:         tight bases launch cleaner 2–4 day moves.

The maximum possible score is 100. TOP_PICK threshold is 75 — deliberately high
so the user only sees 0–3 top picks per scan, not 20.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

# Public thresholds — import these in UI / tests, don't duplicate magic numbers.
TOP_PICK_MIN = 75
STRONG_MIN = 60
DECENT_MIN = 45


def _get(row: Dict[str, Any], key: str, default: Any = None) -> Any:
    """Dict-or-object getter that tolerates None rows."""
    if row is None:
        return default
    if isinstance(row, dict):
        return row.get(key, default)
    return getattr(row, key, default)


def _score_momentum(change_5d: Optional[float]) -> int:
    """0–20. Sweet spot for a 2–4 day hold is +2% to +8% over the past week —
    uptrending but not yet extended. Deep-negative 5d change is a falling knife;
    very hot (>15%) is overextended and prone to mean-revert."""
    if change_5d is None:
        return 0
    c = float(change_5d)
    if 2.0 <= c <= 8.0:
        return 20
    if 8.0 < c <= 15.0:
        return 14
    if 0.0 <= c < 2.0:
        return 10
    if c > 15.0:
        return 6
    if -2.0 <= c < 0.0:
        return 2
    return 0  # negative beyond -2%: falling knife


def _score_volume(volume_ratio: Optional[float]) -> int:
    """0–15. volume_ratio is today vs 20-day avg. >2x is clear institutional
    footprint; <1x means the setup is quiet and prone to fade."""
    if volume_ratio is None:
        return 0
    v = float(volume_ratio)
    if v >= 2.0:
        return 15
    if v >= 1.5:
        return 12
    if v >= 1.2:
        return 8
    if v >= 1.0:
        return 4
    return 0


def _score_rr(rr_ratio: Optional[float]) -> int:
    """0–15. For a 2–4 day hold you need at least 2:1 reward/risk to be worth
    the capital lock-up and opportunity cost. 3:1+ is ideal."""
    if rr_ratio is None:
        return 0
    r = float(rr_ratio)
    if r >= 3.0:
        return 15
    if r >= 2.0:
        return 12
    if r >= 1.5:
        return 7
    return 0


def _score_rsi(rsi: Optional[float]) -> int:
    """0–10. 40–70 is the sweet spot — trending but not exhausted. >75 means
    you're buying the top; <30 means you're catching a falling knife."""
    if rsi is None:
        return 5  # unknown → neutral (don't penalize)
    r = float(rsi)
    if 40.0 <= r <= 70.0:
        return 10
    if 30.0 <= r < 40.0 or 70.0 < r <= 75.0:
        return 6
    if r > 75.0:
        return 2
    return 0


def _score_history(bt_trades: Optional[int], bt_win_rate: Optional[float]) -> int:
    """0–15. Historical win rate is only meaningful with enough samples.
    <10 trades or <50% WR = no statistical edge."""
    n = int(bt_trades or 0)
    wr = float(bt_win_rate or 0)
    if n >= 20 and wr >= 60:
        return 15
    if n >= 10 and wr >= 55:
        return 10
    if n >= 10 and wr >= 50:
        return 5
    if n >= 5 and wr >= 50:
        return 3
    return 0


def _score_proximity(
    live_price: Optional[float],
    buy_zone_low: Optional[float],
    max_buy: Optional[float],
) -> int:
    """0–10. Inside the zone near the low = freshest, best expected R/R.
    Past max-buy = chasing (worse R/R because stop is further below entry).
    Below zone = setup hasn't reset yet."""
    if live_price is None or live_price <= 0:
        return 5  # unknown → neutral
    if not max_buy or not buy_zone_low or max_buy <= buy_zone_low:
        return 5
    lp = float(live_price)
    mb = float(max_buy)
    bzl = float(buy_zone_low)
    if bzl <= lp <= mb:
        # Closer to low = better. Max 10 at bzl, 5 at mb.
        frac = (lp - bzl) / (mb - bzl)
        return max(5, round(10 - 5 * frac))
    if lp > mb:
        over = (lp - mb) / mb
        if over < 0.01:  # <1% over = barely chasing
            return 4
        return 0
    # below zone
    return 3


def _score_grade(grade: Optional[str]) -> int:
    """0–10. Engine's own quality tier."""
    if not grade:
        return 0
    return {"A": 10, "B": 6, "C": 2}.get(str(grade).upper(), 0)


def _score_base(vol_contraction: Any) -> int:
    """0–5. `vol_contraction` from the scanner is a ratio: current volatility
    divided by the longer-term average. <0.6 = tight base (good); >=1.0 =
    expanded range (bad fuel). The UI uses the same 0.6 threshold to color
    the cell green. Booleans are also accepted for back-compat with tests."""
    if vol_contraction is None:
        return 0
    if isinstance(vol_contraction, bool):
        return 5 if vol_contraction else 0
    try:
        v = float(vol_contraction)
    except (TypeError, ValueError):
        return 0
    if v <= 0:
        return 0
    if v < 0.6:
        return 5
    if v < 0.8:
        return 2
    return 0


def score_swing_setup(
    row: Dict[str, Any], live_price: Optional[float] = None
) -> Dict[str, Any]:
    """Score a scanner setup row 0–100 for a 2–4 day swing hold.

    `row` can be the full scan row (with rsi, volume_ratio, change_5d, …) or
    a partial dict — missing fields score 0 except RSI/proximity, which are
    neutralized when unknown so honest data gaps aren't treated as defects.

    `live_price` is optional; if passed, it's used for proximity scoring
    (otherwise we fall back to row['price']).
    """
    lp = live_price if live_price is not None else _get(row, "price")

    breakdown = {
        "momentum": _score_momentum(_get(row, "change_5d")),
        "volume": _score_volume(_get(row, "volume_ratio")),
        "rr": _score_rr(_get(row, "rr_ratio")),
        "rsi": _score_rsi(_get(row, "rsi")),
        "history": _score_history(
            _get(row, "bt_trades"), _get(row, "bt_win_rate")
        ),
        "proximity": _score_proximity(
            lp, _get(row, "buy_zone_low"), _get(row, "max_buy_price")
        ),
        "grade": _score_grade(_get(row, "grade")),
        "base": _score_base(_get(row, "vol_contraction")),
    }
    score = sum(breakdown.values())
    if score >= TOP_PICK_MIN:
        verdict = "TOP_PICK"
    elif score >= STRONG_MIN:
        verdict = "STRONG"
    elif score >= DECENT_MIN:
        verdict = "DECENT"
    else:
        verdict = "WEAK"

    # Concrete reasons (for UI "why is this a top pick?" tooltips).
    reasons: List[str] = []
    c5 = _get(row, "change_5d")
    if c5 is not None and 2 <= float(c5) <= 8:
        reasons.append(f"5-day trend +{float(c5):.1f}% (sweet spot)")
    vr = _get(row, "volume_ratio")
    if vr is not None and float(vr) >= 1.5:
        reasons.append(f"Volume {float(vr):.1f}x avg")
    rr = _get(row, "rr_ratio")
    if rr is not None and float(rr) >= 2.0:
        reasons.append(f"R/R {float(rr):.1f}:1")
    bt_n = int(_get(row, "bt_trades") or 0)
    bt_wr = float(_get(row, "bt_win_rate") or 0)
    if bt_n >= 10 and bt_wr >= 55:
        reasons.append(f"Backtest {bt_wr:.0f}% win ({bt_n} trades)")
    vc = _get(row, "vol_contraction")
    if vc is not None:
        try:
            if isinstance(vc, bool):
                if vc:
                    reasons.append("Tight base")
            elif float(vc) > 0 and float(vc) < 0.6:
                reasons.append(f"Tight base ({float(vc):.2f})")
        except (TypeError, ValueError):
            pass
    grade = _get(row, "grade")
    if grade == "A":
        reasons.append("Grade A setup")

    return {
        "score": int(score),
        "verdict": verdict,
        "breakdown": breakdown,
        "reasons": reasons,
    }


def rank_stalk_orders(
    orders: List[Dict[str, Any]],
    row_lookup: Optional[Any] = None,
    live_prices: Optional[Dict[str, float]] = None,
) -> List[Dict[str, Any]]:
    """Enrich each stalk order with a swing_score dict and sort best-first.

    `row_lookup` is a callable `ticker -> full_scan_row | None` used to fetch
    fields (rsi, volume_ratio, …) the stalk-order dict doesn't carry itself.
    If None, we score using whatever is on the order dict directly.

    Mutates and returns the input list. Stable: orders with the same score
    keep their relative position, which preserves the scanner's own tie-break.
    """
    if not orders:
        return orders
    live_prices = live_prices or {}
    for o in orders:
        ticker = o.get("ticker")
        full = None
        if row_lookup is not None and ticker:
            try:
                full = row_lookup(ticker)
            except Exception:
                full = None
        # Merge: prefer full-row fields when present, fall back to stalk order.
        merged: Dict[str, Any] = {}
        if full:
            merged.update(full)
        for k, v in o.items():
            if k not in merged or merged.get(k) in (None, 0, ""):
                merged[k] = v
        # stalk orders don't carry max_buy_price/buy_zone_low by default —
        # pull them from the full row if possible, otherwise derive from
        # limit_price (== scanner's max_buy).
        if "max_buy_price" not in merged and o.get("limit_price"):
            merged["max_buy_price"] = o["limit_price"]
        lp = live_prices.get(ticker) if ticker else None
        o["swing_score"] = score_swing_setup(merged, live_price=lp)

    orders.sort(
        key=lambda x: (
            -int((x.get("swing_score") or {}).get("score") or 0),
            -float(x.get("trade_score") or 0),
            x.get("ticker") or "",
        )
    )
    return orders


def top_picks(
    orders: List[Dict[str, Any]], limit: int = 5, min_score: int = TOP_PICK_MIN
) -> List[Dict[str, Any]]:
    """Return up to `limit` stalk orders whose swing score >= `min_score`.

    Assumes `orders` has already been enriched by `rank_stalk_orders`.
    """
    picks = [
        o for o in orders
        if int((o.get("swing_score") or {}).get("score") or 0) >= min_score
    ]
    return picks[:limit]
