"""Unit tests for titan.market_time — ET-anchored time helpers."""

import os
import sys
from datetime import datetime, date, timedelta

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from titan.market_time import (
    ET,
    now_et,
    today_et,
    today_et_str,
    last_trading_day_et,
    bdays_between_et,
    market_session_et,
    market_is_open_et,
)


def test_now_et_is_tz_aware():
    n = now_et()
    assert n.tzinfo is not None
    # America/New_York offset ranges from -5 (EST) to -4 (EDT)
    off = n.utcoffset()
    assert off in (timedelta(hours=-5), timedelta(hours=-4))


def test_today_et_returns_date():
    d = today_et()
    assert isinstance(d, date)


def test_today_et_str_format():
    s = today_et_str()
    # YYYY-MM-DD
    assert len(s) == 10
    assert s[4] == "-" and s[7] == "-"
    datetime.strptime(s, "%Y-%m-%d")  # must parse


def test_last_trading_day_sat_to_fri():
    # 2026-04-18 is a Saturday
    sat = date(2026, 4, 18)
    assert last_trading_day_et(sat) == date(2026, 4, 17)  # Friday


def test_last_trading_day_sun_to_fri():
    sun = date(2026, 4, 19)
    assert last_trading_day_et(sun) == date(2026, 4, 17)


def test_last_trading_day_weekday_is_itself():
    wed = date(2026, 4, 15)
    assert last_trading_day_et(wed) == wed


def test_bdays_between_basic():
    # Mon 2026-04-13 → Fri 2026-04-17 = 4 business days
    assert bdays_between_et("2026-04-13", "2026-04-17") == 4


def test_bdays_between_skips_weekend():
    # Fri → Mon should be 1 business day
    assert bdays_between_et("2026-04-10", "2026-04-13") == 1


def test_bdays_between_zero_on_equal():
    assert bdays_between_et("2026-04-15", "2026-04-15") == 0


def test_bdays_between_zero_on_inverted():
    assert bdays_between_et("2026-04-20", "2026-04-15") == 0


def test_bdays_between_handles_bad_input():
    assert bdays_between_et("not-a-date") == 0
    assert bdays_between_et("2026-04-15", "also-bad") == 0


def test_market_session_weekend():
    # 2026-04-18 Saturday noon ET — always CLOSED
    sat_noon = datetime(2026, 4, 18, 12, 0, tzinfo=ET)
    assert market_session_et(sat_noon) == "CLOSED"


def test_market_session_pre_market():
    # 2026-04-15 Wednesday 05:30 ET — PRE_MARKET
    pre = datetime(2026, 4, 15, 5, 30, tzinfo=ET)
    assert market_session_et(pre) == "PRE_MARKET"


def test_market_session_regular():
    # 2026-04-15 Wednesday 10:00 ET — REGULAR
    reg = datetime(2026, 4, 15, 10, 0, tzinfo=ET)
    assert market_session_et(reg) == "REGULAR"


def test_market_session_after_hours():
    # 2026-04-15 Wednesday 17:00 ET — AFTER_HOURS
    ah = datetime(2026, 4, 15, 17, 0, tzinfo=ET)
    assert market_session_et(ah) == "AFTER_HOURS"


def test_market_session_overnight():
    # Wednesday 03:00 ET — CLOSED (before pre-market)
    night = datetime(2026, 4, 15, 3, 0, tzinfo=ET)
    assert market_session_et(night) == "CLOSED"


def test_market_session_naive_datetime():
    # Naive datetime should be treated as ET, not crash
    naive = datetime(2026, 4, 15, 10, 0)
    assert market_session_et(naive) == "REGULAR"


def test_market_session_utc_converted():
    # 14:00 UTC on 2026-04-15 = 10:00 ET (EDT) → REGULAR
    from datetime import timezone as tz
    utc = datetime(2026, 4, 15, 14, 0, tzinfo=tz.utc)
    assert market_session_et(utc) == "REGULAR"


def test_market_is_open_returns_bool():
    result = market_is_open_et()
    assert isinstance(result, bool)


if __name__ == "__main__":
    tests = [v for k, v in globals().items() if k.startswith("test_") and callable(v)]
    passed = 0
    failed = []
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
            passed += 1
        except Exception as e:
            import traceback
            print(f"  FAIL  {t.__name__}: {e}")
            traceback.print_exc()
            failed.append(t.__name__)
    print(f"\n{passed}/{len(tests)} passed")
    if failed:
        print(f"Failed: {failed}")
        sys.exit(1)
