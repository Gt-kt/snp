"""
Final trade decision layer for the manual swing workflow.

The scanner finds candidates. This module answers the operational question:
"Should I buy one of these tonight, or wait?"

It is deliberately conservative. A BUY requires a high swing score, a live-valid
entry zone, no existing position in the same ticker, and no account-level block.
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

from titan.config import MAX_POSITIONS, RISK_PER_TRADE, VIX_PANIC_THRESHOLD
from titan.swing_score import TOP_PICK_MIN


BUY = "BUY"
WAIT = "WAIT"
SKIP = "SKIP"


def _num(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _swing_score(order: Dict[str, Any]) -> int:
    return int(_num((order.get("swing_score") or {}).get("score"), 0))


def _order_blockers(order: Dict[str, Any], risk_budget: float) -> List[str]:
    blockers: List[str] = []
    if order.get("already_held"):
        blockers.append("already held")
    if order.get("live_status") in ("BROKEN", "PAST_MAX_BUY"):
        blockers.append(str(order.get("live_status")).lower())
    if order.get("sector_concentration"):
        blockers.append("sector concentration")
    if order.get("event_risk", {}).get("status") in ("HIGH", "UNKNOWN"):
        blockers.append(f"event risk {order.get('event_risk', {}).get('status')}")
    intraday_status = order.get("intraday_status")
    if intraday_status in (
        "NO_INTRADAY_CONFIRMATION",
        "STALE_INTRADAY",
        "GAP_UP_CHASE",
        "GAP_DOWN_BROKEN",
        "WAIT_INTRADAY",
        "PREMARKET_UNCONFIRMED",
    ):
        blockers.append(intraday_status.lower())
    if _swing_score(order) < TOP_PICK_MIN:
        blockers.append(f"swing score {_swing_score(order)} < {TOP_PICK_MIN}")
    if str(order.get("grade") or "C").upper() not in ("A", "B"):
        blockers.append("grade below B")
    if int(_num(order.get("bt_trades"), 0)) < 20:
        blockers.append("thin backtest sample")
    if int(_num(order.get("bt_trades"), 0)) >= 20 and _num(order.get("bt_win_rate"), 0) < 50:
        blockers.append("backtest win rate below 50%")
    if int(_num(order.get("profile_samples"), 0)) >= 4:
        if _num(order.get("move_expected_return"), 0) <= 0 and _num(order.get("move_up_prob"), 0) < 50:
            blockers.append("negative recent analog profile")
    if risk_budget > 0 and _num(order.get("risk_dollars"), 0) > risk_budget * 1.05:
        blockers.append("risk over budget")
    return blockers


def _clean_order(order: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "ticker": order.get("ticker"),
        "signal_type": order.get("signal_type"),
        "grade": order.get("grade"),
        "tier": order.get("tier"),
        "qty": order.get("qty"),
        "limit_price": order.get("limit_price"),
        "stop_price": order.get("stop_price"),
        "target_price": order.get("target_price"),
        "risk_dollars": order.get("risk_dollars"),
        "time_stop_days": order.get("time_stop_days"),
        "live_status": order.get("live_status"),
        "live_price": order.get("live_price"),
        "intraday_status": order.get("intraday_status"),
        "intraday": order.get("intraday"),
        "event_risk": order.get("event_risk"),
        "bt_win_rate": order.get("bt_win_rate"),
        "bt_trades": order.get("bt_trades"),
        "move_up_prob": order.get("move_up_prob"),
        "move_expected_return": order.get("move_expected_return"),
        "profile_samples": order.get("profile_samples"),
        "swing_score": order.get("swing_score"),
        "entry_note": order.get("entry_note"),
    }


def _rank_orders(orders: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    def live_bucket(order: Dict[str, Any]) -> int:
        status = order.get("live_status")
        if order.get("already_held"):
            return 4
        if status == "BROKEN":
            return 3
        if status == "PAST_MAX_BUY":
            return 2
        if status == "BELOW_ZONE":
            return 1
        return 0

    return sorted(
        list(orders),
        key=lambda o: (
            live_bucket(o),
            -_swing_score(o),
            -_num(o.get("trade_score"), 0),
            str(o.get("ticker") or ""),
        ),
    )


def build_trade_decision(
    scan: Dict[str, Any],
    open_positions: Optional[List[Dict[str, Any]]] = None,
    settings: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Return a single BUY/WAIT/SKIP decision for the latest scan."""
    settings = settings or {}
    open_positions = open_positions or []
    open_count = len([p for p in open_positions if p.get("status", "OPEN") == "OPEN"])
    max_positions = int(settings.get("max_positions", settings.get("max_live_positions", MAX_POSITIONS)) or MAX_POSITIONS)
    risk_budget = _num(settings.get("risk_per_trade"), RISK_PER_TRADE)

    account_blockers: List[str] = []
    regime = str(scan.get("regime") or scan.get("mkt_status") or "UNKNOWN").upper()
    vix = _num(scan.get("vix"), 0)

    if scan.get("daily_loss_cap_hit"):
        account_blockers.append("daily loss cap hit")
    if open_count >= max_positions:
        account_blockers.append(f"position cap full ({open_count}/{max_positions})")
    if regime in ("BEAR", "STRONG_BEAR", "CRISIS", "RISK_OFF"):
        account_blockers.append(f"risk-off regime: {regime}")
    if vix >= _num(settings.get("vix_panic_threshold"), VIX_PANIC_THRESHOLD):
        account_blockers.append(f"VIX panic: {vix:.1f}")

    orders = _rank_orders(scan.get("stalk_orders") or [])
    if not orders:
        return {
            "action": SKIP,
            "ticker": None,
            "confidence": "NONE",
            "headline": "No trade tonight",
            "reason": "No valid stalk orders were produced.",
            "blockers": account_blockers,
            "candidate": None,
            "alternatives": [],
        }

    evaluated = []
    for order in orders:
        blockers = account_blockers + _order_blockers(order, risk_budget)
        evaluated.append((order, blockers))
        if not blockers:
            return {
                "action": BUY,
                "ticker": order.get("ticker"),
                "confidence": "HIGH" if _swing_score(order) >= 85 else "MEDIUM",
                "headline": f"Buy {order.get('ticker')} only inside the plan",
                "reason": "Top setup passed swing score, live-entry, duplicate-position, and risk checks.",
                "blockers": [],
                "candidate": _clean_order(order),
                "alternatives": [_clean_order(o) for o, b in evaluated[1:4] if not b],
            }

    best, blockers = evaluated[0]
    return {
        "action": WAIT,
        "ticker": best.get("ticker"),
        "confidence": "LOW",
        "headline": "Wait tonight",
        "reason": f"Best candidate {best.get('ticker')} is blocked: {', '.join(blockers)}.",
        "blockers": blockers,
        "candidate": _clean_order(best),
        "alternatives": [
            {**_clean_order(o), "blockers": b}
            for o, b in evaluated[1:4]
        ],
    }
