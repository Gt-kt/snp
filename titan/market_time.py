"""
Market-time helpers — anchor everything to America/New_York.

The user lives in Korea (KST = UTC+9). The US market runs in ET (UTC-5/-4).
If we use `datetime.now()` everywhere, business-day counts drift by one depending
on which side of midnight KST we're on, and "today" in the dashboard becomes
ambiguous. All date/time logic that matters for trading should use these
helpers instead of `datetime.now()` / `date.today()`.
"""
from __future__ import annotations

from datetime import datetime, date, timedelta, time as dt_time
from typing import Optional

import numpy as np

try:
    from zoneinfo import ZoneInfo
    ET = ZoneInfo("America/New_York")
except Exception:  # pragma: no cover -- pre-3.9 fallback
    import pytz  # type: ignore
    ET = pytz.timezone("America/New_York")


# Market session boundaries (in ET)
_PRE_MARKET_OPEN = dt_time(4, 0)
_REGULAR_OPEN = dt_time(9, 30)
_REGULAR_CLOSE = dt_time(16, 0)
_AFTER_HOURS_CLOSE = dt_time(20, 0)


def now_et() -> datetime:
    """Current wall-clock time in ET."""
    return datetime.now(ET)


def today_et() -> date:
    """Current calendar date in ET."""
    return now_et().date()


def today_et_str() -> str:
    """Current ET date as 'YYYY-MM-DD'."""
    return today_et().strftime("%Y-%m-%d")


def last_trading_day_et(reference: Optional[date] = None) -> date:
    """Most recent weekday at or before the reference (ignores holidays).

    If today is Saturday, returns Friday. Sunday → Friday. Monday → Monday.
    """
    ref = reference or today_et()
    # np.busday_offset with '-0' gives the same day if it's a business day,
    # else rolls back to the previous one.
    return np.busday_offset(ref, 0, roll="preceding").astype(date)


def bdays_between_et(start_iso: str, end_iso: Optional[str] = None) -> int:
    """Business days between two YYYY-MM-DD strings (end defaults to today in ET).

    Returns 0 on parse error or if end <= start.
    """
    try:
        start = datetime.strptime(start_iso, "%Y-%m-%d").date()
        end = (
            datetime.strptime(end_iso, "%Y-%m-%d").date()
            if end_iso
            else today_et()
        )
        if end <= start:
            return 0
        return int(np.busday_count(start, end))
    except Exception:
        return 0


def market_session_et(ref: Optional[datetime] = None) -> str:
    """Return CLOSED / PRE_MARKET / REGULAR / AFTER_HOURS for a given ET time.

    Ignores US holidays — a closed holiday returns the time-based classification
    (e.g. 10 AM on Christmas reports REGULAR). Good enough for dashboard UX.
    """
    n = ref or now_et()
    if n.tzinfo is None:
        n = n.replace(tzinfo=ET)
    else:
        n = n.astimezone(ET)

    if n.weekday() >= 5:  # Sat/Sun
        return "CLOSED"
    t = n.time()
    if _PRE_MARKET_OPEN <= t < _REGULAR_OPEN:
        return "PRE_MARKET"
    if _REGULAR_OPEN <= t < _REGULAR_CLOSE:
        return "REGULAR"
    if _REGULAR_CLOSE <= t < _AFTER_HOURS_CLOSE:
        return "AFTER_HOURS"
    return "CLOSED"


def market_is_open_et() -> bool:
    """Regular session only (9:30–16:00 ET, weekdays, ignores holidays)."""
    return market_session_et() == "REGULAR"
