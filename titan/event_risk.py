"""
Best-effort event/news risk checks for short-hold swing trades.

This does not try to be a professional news terminal. It makes hidden event risk
explicit and gives the decision layer a conservative blocker when earnings or
headline risk is visible through the configured public data source.
"""
from __future__ import annotations

from datetime import datetime, timezone
import re
from typing import Any, Dict, List, Optional


HIGH_KEYWORDS = (
    "guidance",
    "downgrade",
    "cuts rating",
    "cut rating",
    "sec investigation",
    "sec probe",
    "sec charges",
    "investigation",
    "lawsuit",
    "probe",
    "fraud",
    "recall",
    "bankruptcy",
)

MED_KEYWORDS = (
    "earnings",
    "earnings preview",
    "earnings expected",
    "earnings call",
    "analyst",
    "price target",
    "upgrade",
    "initiates",
    "merger",
    "acquisition",
    "layoffs",
)


def _contains_phrase(text: str, phrase: str) -> bool:
    escaped = re.escape(phrase.lower()).replace(r"\ ", r"\s+")
    return re.search(rf"(?<![a-z0-9]){escaped}(?![a-z0-9])", text.lower()) is not None


def _headline(item: Dict[str, Any]) -> str:
    content = item.get("content") if isinstance(item.get("content"), dict) else {}
    return str(
        item.get("title")
        or item.get("headline")
        or content.get("title")
        or ""
    )


def _published_ts(item: Dict[str, Any]) -> Optional[float]:
    content = item.get("content") if isinstance(item.get("content"), dict) else {}
    raw = (
        item.get("providerPublishTime")
        or item.get("pubDate")
        or item.get("published_at")
        or content.get("pubDate")
        or content.get("displayTime")
    )
    if raw is None:
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        pass
    try:
        return datetime.fromisoformat(str(raw).replace("Z", "+00:00")).timestamp()
    except (TypeError, ValueError):
        return None


def _recent_news_items(news: Any, max_age_hours: float) -> List[Dict[str, Any]]:
    if not isinstance(news, list):
        return []
    now_ts = datetime.now(timezone.utc).timestamp()
    recent: List[Dict[str, Any]] = []
    for item in news:
        if not isinstance(item, dict):
            continue
        ts = _published_ts(item)
        if ts is None or (now_ts - ts) <= max_age_hours * 3600:
            recent.append(item)
    return recent


def assess_event_risk(
    ticker: str,
    earnings_days: Optional[int] = None,
    blackout_days: int = 7,
    post_earnings_days: int = 1,
    max_news_age_hours: float = 72.0,
) -> Dict[str, Any]:
    """Return LOW/MED/HIGH/UNKNOWN event risk for a ticker."""
    reasons: List[str] = []

    if earnings_days is not None:
        try:
            d = int(earnings_days)
            if -post_earnings_days <= d <= blackout_days:
                return {
                    "status": "HIGH",
                    "reasons": [f"earnings window ({d:+d}d)"],
                    "headlines": [],
                }
            if blackout_days < d <= blackout_days + 7:
                reasons.append(f"earnings soon ({d:+d}d)")
        except (TypeError, ValueError):
            pass

    headlines: List[str] = []
    try:
        import yfinance as yf

        news = getattr(yf.Ticker(ticker), "news", [])
        for item in _recent_news_items(news, max_news_age_hours):
            title = _headline(item).strip()
            if not title:
                continue
            lower = title.lower()
            headlines.append(title[:180])
            if any(_contains_phrase(lower, k) for k in HIGH_KEYWORDS):
                reasons.append(f"headline risk: {title[:120]}")
            elif any(_contains_phrase(lower, k) for k in MED_KEYWORDS):
                reasons.append(f"news watch: {title[:120]}")
    except Exception as exc:
        return {
            "status": "UNKNOWN",
            "reasons": [f"news check failed: {exc}"],
            "headlines": [],
        }

    high_reasons = [r for r in reasons if r.startswith("headline risk") or r.startswith("earnings")]
    if high_reasons:
        return {"status": "HIGH", "reasons": high_reasons[:3], "headlines": headlines[:5]}
    if reasons:
        return {"status": "MED", "reasons": reasons[:3], "headlines": headlines[:5]}
    return {"status": "LOW", "reasons": [], "headlines": headlines[:5]}
