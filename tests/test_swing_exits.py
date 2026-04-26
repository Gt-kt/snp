"""Unit tests for titan.swing_exits — partial ladder, quick-profit, time-stop warnings."""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from titan.swing_exits import (
    partial_ladder_signal,
    quick_profit_signal,
    time_stop_approaching,
    LADDER_1R,
    LADDER_2R,
    QUICK_PROFIT_PCT,
    QUICK_PROFIT_MAX_DAYS,
)


# ---------------------------------------------------------------- partial_ladder

def test_ladder_no_signal_before_1r():
    # Entry 100, stop 95 → 1R = $5. Current 103 → 0.6R < 1R threshold.
    assert partial_ladder_signal(entry=100, stop=95, current=103, shares=100) is None


def test_ladder_1r_signal_fires():
    # Current 105 = exactly 1R
    r = partial_ladder_signal(entry=100, stop=95, current=105, shares=90)
    assert r is not None
    assert r["rung"] == "1R"
    assert r["r_multiple"] == 1.0
    assert r["sell_shares"] == int(90 * 0.33)
    assert "breakeven" in r["action"].lower()


def test_ladder_2r_upgrades_past_1r():
    # Current 110 = 2R. Must upgrade to 2R (not hand us 1R).
    r = partial_ladder_signal(entry=100, stop=95, current=110, shares=100)
    assert r is not None
    assert r["rung"] == "2R"
    assert r["r_multiple"] == 2.0
    # Takes another 1/3
    assert r["sell_shares"] == 34


def test_ladder_without_stop_uses_5pct_default_r():
    # No stop → 1R = 5% of 100 = $5. Current 105 → 1R triggers.
    r = partial_ladder_signal(entry=100, stop=None, current=105, shares=50)
    assert r is not None
    assert r["rung"] == "1R"


def test_ladder_ignores_bad_stop_above_entry():
    # Bogus stop (above entry) → fallback to 5% default
    r = partial_ladder_signal(entry=100, stop=110, current=105, shares=50)
    assert r is not None
    assert r["rung"] == "1R"


def test_ladder_at_loss_is_none():
    assert partial_ladder_signal(entry=100, stop=95, current=98, shares=100) is None


def test_ladder_zero_shares_none():
    assert partial_ladder_signal(entry=100, stop=95, current=108, shares=0) is None


def test_ladder_zero_entry_none():
    assert partial_ladder_signal(entry=0, stop=95, current=108, shares=100) is None


def test_ladder_sell_shares_always_at_least_1():
    # Tiny position — floor at 1 share, not 0
    r = partial_ladder_signal(entry=100, stop=95, current=105, shares=2)
    assert r is not None
    assert r["sell_shares"] >= 1


# ---------------------------------------------------------------- quick_profit

def test_quick_profit_fires_above_threshold_in_window():
    # +6% in 1 day → fires (threshold 5%, max_days 2)
    r = quick_profit_signal(entry=100, current=106, days_held=1)
    assert r is not None
    assert r["gain_pct"] == 6.0
    assert r["days_held"] == 1
    assert "+6.0%" in r["action"] or "+6%" in r["action"]


def test_quick_profit_day_zero_uses_today_label():
    r = quick_profit_signal(entry=100, current=106, days_held=0)
    assert r is not None
    assert "today" in r["action"].lower()


def test_quick_profit_below_threshold_silent():
    # +4% is below 5% → silent
    assert quick_profit_signal(entry=100, current=104, days_held=1) is None


def test_quick_profit_past_window_silent():
    # Day 3 is past the 2-day window → let partial ladder + trailing handle it
    assert quick_profit_signal(entry=100, current=120, days_held=3) is None


def test_quick_profit_negative_days_silent():
    assert quick_profit_signal(entry=100, current=110, days_held=-1) is None


def test_quick_profit_flat_or_loss_silent():
    assert quick_profit_signal(entry=100, current=100, days_held=1) is None
    assert quick_profit_signal(entry=100, current=95, days_held=1) is None


def test_quick_profit_zero_entry_silent():
    assert quick_profit_signal(entry=0, current=105, days_held=1) is None


def test_quick_profit_custom_threshold():
    # Caller overrides to 10% — +6% won't fire
    assert quick_profit_signal(entry=100, current=106, days_held=1, pct_threshold=10) is None


# ---------------------------------------------------------------- time_stop_approaching

def test_time_stop_warn_one_day_before():
    # Held 4/5 → 1 day left → warn
    assert time_stop_approaching(days_held=4, time_stop_days=5) is True


def test_time_stop_warn_silent_with_runway():
    # Held 2/5 → 3 days left → silent
    assert time_stop_approaching(days_held=2, time_stop_days=5) is False


def test_time_stop_already_past_silent():
    # Already at/past the limit → SELL_TIMESTOP handles it, this returns False
    assert time_stop_approaching(days_held=5, time_stop_days=5) is False
    assert time_stop_approaching(days_held=7, time_stop_days=5) is False


def test_time_stop_custom_threshold():
    # threshold_days=2 — 3/5 held → 2 days left → fires
    assert time_stop_approaching(days_held=3, time_stop_days=5, threshold_days=2) is True


def test_time_stop_none_inputs_false():
    assert time_stop_approaching(None, 5) is False
    assert time_stop_approaching(3, None) is False
    assert time_stop_approaching(3, 0) is False


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
