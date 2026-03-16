"""
Titan Trade Opportunity Reporting
=================================
Shared helpers for exporting scan results and measuring opportunity flow.
"""

from typing import Iterable


def _get_setup_value(setup, field, default=None):
    """Read a field from either a setup object or serialized dict."""
    if isinstance(setup, dict):
        return setup.get(field, default)
    return getattr(setup, field, default)


def build_scan_opportunity_summary(setups, watchlist, market_status, vix_level=None):
    """Summarize whether the current scan produced actionable or research ideas."""
    setups = list(setups or [])
    actionable_count = len(setups)
    research_count = len(watchlist or [])
    total_ideas = actionable_count + research_count
    market_status = str(market_status or "UNKNOWN")
    market_upper = market_status.upper()
    vix_value = float(vix_level or 0.0)
    pilot_count = sum(
        1 for setup in setups
        if str(_get_setup_value(setup, "opportunity_tier", "VALIDATED")).upper() == "PILOT"
    )
    validated_count = max(actionable_count - pilot_count, 0)

    if actionable_count > 0:
        state = "ACTIONABLE"
        if pilot_count and not validated_count:
            headline = f"{pilot_count} pilot setup{'s' if pilot_count != 1 else ''} ready for manual review."
            detail = "Reduced-size breakouts cleared the pilot bar. Auto execution stays disabled."
        elif pilot_count:
            headline = (
                f"{validated_count} validated setup{'s' if validated_count != 1 else ''} plus "
                f"{pilot_count} pilot setup{'s' if pilot_count != 1 else ''} ready for review."
            )
            detail = (
                f"Research ladder also has {research_count} additional stalk name"
                f"{'s' if research_count != 1 else ''}. Pilot ideas are reduced-size and manual-only."
                if research_count
                else "Pilot ideas are reduced-size and manual-only."
            )
        else:
            headline = f"{actionable_count} actionable setup{'s' if actionable_count != 1 else ''} ready for review."
            detail = (
                f"Research ladder also has {research_count} additional stalk name"
                f"{'s' if research_count != 1 else ''}."
                if research_count
                else "Focus on the actionable list first."
            )
    elif research_count > 0:
        research_noun = "name is" if research_count == 1 else "names are"
        state = "RESEARCH"
        headline = (
            f"No action-ready setups today, but {research_count} research {research_noun} worth stalking."
        )
        detail = "The engine stayed selective, but the watchlist still has momentum names."
    elif vix_value >= 30.0 or "PANIC" in market_upper or "FEAR" in market_upper:
        state = "DEFENSIVE"
        headline = "Volatility is elevated and long-side opportunity is scarce."
        detail = "Protect capital first. This is a defensive session, not a forcing session."
    elif "BEAR" in market_upper:
        state = "DEFENSIVE"
        headline = "The market regime is bearish and the long book is intentionally quiet."
        detail = "A blank long list is expected here; this is discipline, not a miss."
    else:
        state = "QUIET"
        headline = "No action-ready setups or research names cleared the bar."
        detail = "This is a genuinely quiet scan, not just an empty action bucket."

    return {
        "state": state,
        "headline": headline,
        "detail": detail,
        "market_status": market_status,
        "vix_level": round(vix_value, 2) if vix_level is not None else None,
        "actionable_count": actionable_count,
        "validated_count": validated_count,
        "pilot_count": pilot_count,
        "research_count": research_count,
        "total_ideas": total_ideas,
        "has_actionable": actionable_count > 0,
        "has_research": research_count > 0,
    }


def _serialize_setup(setup, action_labeler=None):
    """Normalize a TitanSetup into a JSON-safe payload."""
    action = action_labeler(setup) if action_labeler else getattr(setup, "confirmation_status", "WATCH")
    return {
        "ticker": setup.ticker,
        "strategy": setup.strategy,
        "price": round(setup.price, 2),
        "trigger": round(setup.trigger, 2),
        "target": round(setup.target, 2),
        "stop": round(setup.stop, 2),
        "confidence_grade": setup.confidence_grade,
        "action": action,
        "sector": setup.sector,
        "sector_aligned": bool(getattr(setup, "sector_aligned", False)),
        "win_rate": round(setup.win_rate, 1),
        "profit_factor": round(setup.profit_factor, 2),
        "momentum_score": round(setup.momentum_score, 1),
        "rs_percentile": round(setup.rs_percentile, 1),
        "pre_breakout_score": round(getattr(setup, "pre_breakout_score", 0.0), 1),
        "robustness_score": round(getattr(setup, "robustness_score", 0.0), 1),
        "walk_forward_pass_rate": round(getattr(setup, "walk_forward_pass_rate", 0.0), 3),
        "walk_forward_pf": round(getattr(setup, "walk_forward_pf", 0.0), 2),
        "walk_forward_trades": int(getattr(setup, "walk_forward_trades", 0)),
        "regime_score": round(getattr(setup, "regime_score", 0.0), 3),
        "oos_pf": round(getattr(setup, "oos_pf", 0.0), 2),
        "oos_trades": int(getattr(setup, "oos_trades", 0)),
        "net_expectancy": round(getattr(setup, "net_expectancy", 0.0), 5),
        "distance_from_high_pct": round(getattr(setup, "distance_from_high_pct", 0.0), 2),
        "distance_to_entry_pct": round(getattr(setup, "distance_to_entry_pct", 0.0), 2),
        "confirmation_status": getattr(setup, "confirmation_status", "WATCH"),
        "entry_ready_score": round(getattr(setup, "entry_ready_score", 0.0), 1),
        "distance_to_pivot_pct": round(getattr(setup, "distance_to_pivot_pct", 0.0), 2),
        "starter_trigger": round(getattr(setup, "starter_trigger", setup.trigger), 2),
        "add_on_trigger": round(getattr(setup, "add_on_trigger", 0.0), 2),
        "partial_target": round(getattr(setup, "partial_target", 0.0), 2),
        "starter_qty": int(setup.qty),
        "planned_total_qty": int(getattr(setup, "planned_total_qty", setup.qty)),
        "add_on_qty": int(getattr(setup, "add_on_qty", 0)),
        "opportunity_tier": getattr(setup, "opportunity_tier", "VALIDATED"),
        "execution_eligible": bool(getattr(setup, "execution_eligible", True)),
        "position_size_scalar": round(getattr(setup, "position_size_scalar", 1.0), 2),
    }


def _serialize_watchlist_item(item):
    """Normalize a research watchlist candidate into a stable payload."""
    return {
        "ticker": item.get("ticker", ""),
        "theme": item.get("theme", ""),
        "status": item.get("status", "RESEARCH"),
        "price": round(float(item.get("price", 0.0) or 0.0), 2),
        "trigger": round(float(item.get("trigger", 0.0) or 0.0), 2),
        "distance_to_entry_pct": round(float(item.get("distance_to_entry_pct", 0.0) or 0.0), 2),
        "sector": item.get("sector", "Unknown"),
        "sector_aligned": bool(item.get("sector_aligned", False)),
        "rs_percentile": round(float(item.get("rs_percentile", 0.0) or 0.0), 1),
        "momentum_score": round(float(item.get("momentum_score", 0.0) or 0.0), 1),
        "accumulation_score": round(float(item.get("accumulation_score", 0.0) or 0.0), 1),
        "earnings_call": item.get("earnings_call", "Unknown"),
        "score": round(float(item.get("score", 0.0) or 0.0), 1),
        "why": item.get("why", ""),
        "pre_breakout_score": round(float(item.get("pre_breakout_score", 0.0) or 0.0), 1),
        "research_only": True,
    }


def build_scan_export_data(setups, stats, market_data, vix_level, timestamp, action_labeler=None):
    """Create the dashboard/export payload from scan results."""
    setups = list(setups or [])
    stats = stats or {}
    market_data = market_data or {}
    watchlist = list(market_data.get("watchlist") or [])
    top_sectors = list(market_data.get("top_sectors") or [])
    opportunity = build_scan_opportunity_summary(
        setups,
        watchlist,
        market_data.get("mkt_status", "Unknown"),
        vix_level=vix_level,
    )

    return {
        "timestamp": timestamp,
        "market_status": market_data.get("mkt_status", "Unknown"),
        "top_sectors": top_sectors,
        "vix_level": round(vix_level, 2) if vix_level is not None else None,
        "passed_count": stats.get("Passed", 0),
        "actionable_count": len(setups),
        "validated_count": opportunity["validated_count"],
        "pilot_count": opportunity["pilot_count"],
        "watchlist_bucket": "RESEARCH_ONLY",
        "watchlist_count": len(watchlist),
        "research_watchlist_count": len(watchlist),
        "watchlist": [_serialize_watchlist_item(item) for item in watchlist],
        "research_watchlist": [_serialize_watchlist_item(item) for item in watchlist],
        "total_scanned": stats.get("Total", 0),
        "opportunity_state": opportunity["state"],
        "opportunity_headline": opportunity["headline"],
        "opportunity_detail": opportunity["detail"],
        "opportunity": opportunity,
        "setups": [_serialize_setup(s, action_labeler=action_labeler) for s in setups],
    }


def normalize_scan_payload(payload):
    """Backfill older scan exports to the current dashboard contract."""
    if not isinstance(payload, dict):
        payload = {}

    setups = list(payload.get("setups") or [])
    watchlist = list(payload.get("research_watchlist") or payload.get("watchlist") or [])
    top_sectors = list(payload.get("top_sectors") or [])
    vix_level = payload.get("vix_level")
    if isinstance(vix_level, str):
        try:
            vix_level = float(vix_level)
        except ValueError:
            vix_level = None

    opportunity = build_scan_opportunity_summary(
        setups,
        watchlist,
        payload.get("market_status", "Unknown"),
        vix_level=vix_level,
    )

    normalized = dict(payload)
    normalized["setups"] = setups
    normalized["watchlist"] = watchlist
    normalized["research_watchlist"] = watchlist
    normalized["top_sectors"] = top_sectors
    normalized["passed_count"] = int(
        normalized.get("passed_count", normalized.get("actionable_count", len(setups))) or 0
    )
    normalized["actionable_count"] = int(
        normalized.get("actionable_count", normalized.get("passed_count", len(setups))) or 0
    )
    normalized["pilot_count"] = int(
        normalized.get(
            "pilot_count",
            sum(
                1 for setup in setups
                if str(_get_setup_value(setup, "opportunity_tier", "VALIDATED")).upper() == "PILOT"
            ),
        ) or 0
    )
    normalized["validated_count"] = int(
        normalized.get("validated_count", max(normalized["actionable_count"] - normalized["pilot_count"], 0)) or 0
    )
    normalized["watchlist_count"] = int(
        normalized.get("watchlist_count", normalized.get("research_watchlist_count", len(watchlist))) or 0
    )
    normalized["research_watchlist_count"] = int(
        normalized.get("research_watchlist_count", normalized.get("watchlist_count", len(watchlist))) or 0
    )
    normalized["opportunity"] = normalized.get("opportunity") or opportunity
    normalized["opportunity_state"] = normalized.get("opportunity_state") or opportunity["state"]
    normalized["opportunity_headline"] = normalized.get("opportunity_headline") or opportunity["headline"]
    normalized["opportunity_detail"] = normalized.get("opportunity_detail") or opportunity["detail"]
    return normalized


def summarize_replay_opportunity(scan_rows: Iterable[dict]):
    """Measure how often the replay produced actionable or research ideas."""
    rows = list(scan_rows or [])
    actionable_days = 0
    research_days = 0
    quiet_days = 0
    max_quiet_streak = 0
    quiet_streak = 0

    for row in rows:
        actionable = int(row.get("setups_found", 0) or 0)
        research = int(row.get("watchlist_count", 0) or 0)
        if actionable > 0:
            actionable_days += 1
            quiet_streak = 0
        elif research > 0:
            research_days += 1
            quiet_streak = 0
        else:
            quiet_days += 1
            quiet_streak += 1
            max_quiet_streak = max(max_quiet_streak, quiet_streak)

    scan_count = len(rows)
    idea_days = actionable_days + research_days
    coverage_pct = (idea_days / scan_count * 100.0) if scan_count else 0.0
    return {
        "Actionable_Days": actionable_days,
        "Research_Days": research_days,
        "Quiet_Days": quiet_days,
        "Idea_Days": idea_days,
        "Idea_Coverage_Pct": coverage_pct,
        "Max_Quiet_Streak": max_quiet_streak,
    }
