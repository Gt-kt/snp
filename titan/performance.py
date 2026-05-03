"""Performance analytics for manually logged swing trades."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from titan.market_time import bdays_between_et


MIN_REVIEW_TRADES = 20


def _closed_trades(positions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        p for p in positions
        if p.get("status") == "CLOSED" and p.get("pnl_dollars") is not None
    ]


def _trade_hold_days(trade: dict[str, Any]) -> int | None:
    entry = trade.get("entry_date")
    exit_ = trade.get("exit_date")
    if not entry or not exit_:
        return None
    try:
        return bdays_between_et(entry, exit_)
    except Exception:
        return None


def summarize_trades(trades: list[dict[str, Any]]) -> dict[str, Any]:
    count = len(trades)
    gross_profit = round(sum(max(float(t.get("pnl_dollars") or 0.0), 0.0) for t in trades), 2)
    gross_loss = round(abs(sum(min(float(t.get("pnl_dollars") or 0.0), 0.0) for t in trades)), 2)
    wins = [t for t in trades if float(t.get("pnl_dollars") or 0.0) > 0]
    losses = [t for t in trades if float(t.get("pnl_dollars") or 0.0) < 0]
    pnl_values = [float(t.get("pnl_dollars") or 0.0) for t in trades]
    pct_values = [float(t.get("pnl_pct") or 0.0) for t in trades]
    hold_days = [d for d in (_trade_hold_days(t) for t in trades) if d is not None]
    profit_factor = None
    if gross_loss > 0:
        profit_factor = round(gross_profit / gross_loss, 2)
    elif gross_profit > 0:
        profit_factor = 999.0

    return {
        "trades": count,
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": round(len(wins) / count * 100.0, 1) if count else 0.0,
        "net_pnl_dollars": round(sum(pnl_values), 2),
        "gross_profit": gross_profit,
        "gross_loss": gross_loss,
        "profit_factor": profit_factor,
        "expectancy_dollars": round(sum(pnl_values) / count, 2) if count else 0.0,
        "avg_return_pct": round(sum(pct_values) / count, 2) if count else 0.0,
        "avg_win_dollars": round(sum(float(t.get("pnl_dollars") or 0.0) for t in wins) / len(wins), 2) if wins else 0.0,
        "avg_loss_dollars": round(sum(float(t.get("pnl_dollars") or 0.0) for t in losses) / len(losses), 2) if losses else 0.0,
        "avg_hold_days": round(sum(hold_days) / len(hold_days), 1) if hold_days else None,
    }


def _group_summary(trades: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for trade in trades:
        label = str(trade.get(key) or "UNKNOWN").strip() or "UNKNOWN"
        grouped[label].append(trade)
    rows = []
    for label, rows_for_label in grouped.items():
        summary = summarize_trades(rows_for_label)
        summary["name"] = label
        rows.append(summary)
    rows.sort(key=lambda r: (-r["trades"], -r["expectancy_dollars"], r["name"]))
    return rows


def build_performance_report(positions: list[dict[str, Any]]) -> dict[str, Any]:
    closed = _closed_trades(positions)
    overall = summarize_trades(closed)
    return {
        "status": "success",
        "sample_warning": len(closed) < MIN_REVIEW_TRADES,
        "min_review_trades": MIN_REVIEW_TRADES,
        "overall": overall,
        "by_signal_type": _group_summary(closed, "signal_type"),
        "by_sector": _group_summary(closed, "sector"),
        "by_exit_reason": _group_summary(closed, "exit_reason"),
    }
