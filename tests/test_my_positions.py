"""Unit tests for titan.my_positions — the manual position tracker."""

import json
import os
import sys
import tempfile
from datetime import datetime, timedelta

import numpy as np

# Allow import when running from repo root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from titan.my_positions import (
    MyPositions,
    ALERT_HOLD, ALERT_SELL_STOP, ALERT_SELL_TARGET, ALERT_SELL_TIMESTOP,
    ALERT_SELL_TRAIL, ALERT_WARN_STOP, ALERT_WARN_TARGET, ALERT_WARN_TIMESTOP,
    DEFAULT_TIME_STOP_DAYS,
)
from titan.market_time import today_et_str


def _tmp():
    fd, path = tempfile.mkstemp(suffix=".json")
    os.close(fd)
    os.unlink(path)
    return path


def test_add_and_list():
    path = _tmp()
    try:
        mp = MyPositions(path)
        assert mp.list_all() == []
        pos = mp.add("ROST", entry_price=223.56, shares=10,
                     stop=218.0, target=237.0, time_stop_days=8)
        assert pos["ticker"] == "ROST"
        assert pos["status"] == "OPEN"
        assert pos["stop"] == 218.0
        assert pos["target"] == 237.0
        assert len(mp.list_open()) == 1

        # Persists across instances
        mp2 = MyPositions(path)
        assert len(mp2.list_all()) == 1
        assert mp2.list_all()[0]["ticker"] == "ROST"
    finally:
        if os.path.exists(path):
            os.unlink(path)


def test_invalid_stop_or_target_is_dropped():
    path = _tmp()
    try:
        mp = MyPositions(path)
        # stop above entry is invalid → stored as None
        pos = mp.add("AAA", entry_price=100.0, shares=1, stop=105.0, target=110.0)
        assert pos["stop"] is None
        assert pos["target"] == 110.0
        # target below entry is invalid → stored as None
        pos2 = mp.add("BBB", entry_price=100.0, shares=1, stop=95.0, target=99.0)
        assert pos2["stop"] == 95.0
        assert pos2["target"] is None
    finally:
        if os.path.exists(path):
            os.unlink(path)


def test_add_rejects_bad_input():
    path = _tmp()
    try:
        mp = MyPositions(path)
        for bad in [("", 100, 1), ("X", 0, 1), ("X", 100, 0), ("X", -1, 1), ("X", 100, -1)]:
            try:
                mp.add(*bad)
                assert False, f"expected ValueError for {bad}"
            except ValueError:
                pass
        try:
            mp.add("X", 100, 1, entry_date="2026/01/01")
            assert False, "expected ValueError for bad date"
        except ValueError:
            pass
    finally:
        if os.path.exists(path):
            os.unlink(path)


def test_evaluate_hold_and_sell_states():
    path = _tmp()
    try:
        mp = MyPositions(path)
        today = datetime.now().strftime("%Y-%m-%d")
        pos = mp.add("X", entry_price=100.0, shares=10,
                     stop=95.0, target=110.0, time_stop_days=10,
                     entry_date=today)

        # HOLD: price in the middle
        e = mp.evaluate(pos, current_price=103.0, today=today)
        assert e["alert"] == ALERT_HOLD
        assert e["pnl_pct_live"] == 3.0
        assert e["pnl_dollars_live"] == 30.0

        # SELL_STOP: price hits stop
        e = mp.evaluate(pos, current_price=94.5, today=today)
        assert e["alert"] == ALERT_SELL_STOP

        # SELL_TARGET: price hits target
        e = mp.evaluate(pos, current_price=110.5, today=today)
        assert e["alert"] == ALERT_SELL_TARGET

        # WARN_STOP: within 1% above stop
        e = mp.evaluate(pos, current_price=95.5, today=today)
        assert e["alert"] == ALERT_WARN_STOP, f"got {e['alert']}"

        # WARN_TARGET: within 1% below target
        e = mp.evaluate(pos, current_price=109.5, today=today)
        assert e["alert"] == ALERT_WARN_TARGET, f"got {e['alert']}"
    finally:
        if os.path.exists(path):
            os.unlink(path)


def test_time_stop_alert():
    path = _tmp()
    try:
        mp = MyPositions(path)
        # Entry date 15 business days ago -> time stop 5
        today = today_et_str()
        entry_dt = _bday_iso_offset(15)
        pos = mp.add("X", entry_price=100.0, shares=10,
                     stop=95.0, target=110.0, time_stop_days=5,
                     entry_date=entry_dt)

        # Price neutral, but we're way past time stop
        e = mp.evaluate(pos, current_price=102.0, today=today)
        assert e["alert"] == ALERT_SELL_TIMESTOP
        assert e["days_held"] >= 5
    finally:
        if os.path.exists(path):
            os.unlink(path)


def test_close_computes_pnl():
    path = _tmp()
    try:
        mp = MyPositions(path)
        pos = mp.add("X", entry_price=100.0, shares=10, stop=95.0, target=110.0)
        closed = mp.close(pos["id"], exit_price=108.0, reason="MANUAL")
        assert closed["status"] == "CLOSED"
        assert closed["pnl_pct"] == 8.0
        assert closed["pnl_dollars"] == 80.0
        assert closed["exit_reason"] == "MANUAL"

        # Closing again is a no-op (keeps original exit values)
        again = mp.close(pos["id"], exit_price=999.0, reason="IGNORE")
        assert again["exit_price"] == 108.0
    finally:
        if os.path.exists(path):
            os.unlink(path)


def test_delete_and_update():
    path = _tmp()
    try:
        mp = MyPositions(path)
        pos = mp.add("X", entry_price=100.0, shares=10, stop=95.0, target=110.0)
        updated = mp.update(pos["id"], stop=97.0, target=115.0, time_stop_days=7, notes="raised stop")
        assert updated["stop"] == 97.0
        assert updated["target"] == 115.0
        assert updated["time_stop_days"] == 7
        assert updated["notes"] == "raised stop"
        assert mp.delete(pos["id"]) is True
        assert mp.list_all() == []
        assert mp.delete(pos["id"]) is False
    finally:
        if os.path.exists(path):
            os.unlink(path)


def test_atomic_write_survives_bad_file():
    # Corrupt file → load resets positions, backup created
    path = _tmp()
    try:
        with open(path, "w") as f:
            f.write("{not valid json")
        mp = MyPositions(path)
        assert mp.list_all() == []
        assert os.path.exists(path + ".corrupt.bak")
        os.unlink(path + ".corrupt.bak")
    finally:
        if os.path.exists(path):
            os.unlink(path)


def test_summary():
    path = _tmp()
    try:
        mp = MyPositions(path)
        a = mp.add("A", 100, 10, stop=95, target=110)
        b = mp.add("B", 50, 20, stop=45, target=60)
        c = mp.add("C", 200, 5, stop=190, target=220)
        mp.close(a["id"], exit_price=108)   # +8% +$80
        mp.close(b["id"], exit_price=44)    # -12% -$120
        s = mp.summary()
        assert s["total"] == 3
        assert s["open"] == 1
        assert s["closed"] == 2
        assert s["win_rate"] == 50.0
        assert s["total_pnl_dollars"] == -40.0
    finally:
        if os.path.exists(path):
            os.unlink(path)


def test_evaluate_closed_position():
    path = _tmp()
    try:
        mp = MyPositions(path)
        pos = mp.add("X", 100, 10, stop=95, target=110)
        closed = mp.close(pos["id"], exit_price=108, reason="TARGET")
        # Evaluate a closed position: reports CLOSED, no live price needed
        e = mp.evaluate(closed, current_price=None)
        assert e["alert"] == "CLOSED"
        assert e["alert_reason"] == "TARGET"
    finally:
        if os.path.exists(path):
            os.unlink(path)


# ================================================================
# Trailing-stop behavior (priority-3 feature)
# ================================================================

def test_seeded_high_water_mark_on_add():
    path = _tmp()
    try:
        mp = MyPositions(path)
        pos = mp.add("X", 100.0, 10, stop=95, target=120)
        # Seeded so trailing math never sees None
        assert pos["highest_since_entry"] == 100.0
    finally:
        if os.path.exists(path):
            os.unlink(path)


def test_record_high_water_monotonic():
    path = _tmp()
    try:
        mp = MyPositions(path)
        pos = mp.add("X", 100.0, 10, stop=95, target=120)

        # Lower price → no change
        changed = mp.record_high_water({"X": 98.0})
        assert changed is False
        after = mp.list_all()[0]
        assert after["highest_since_entry"] == 100.0

        # Higher price → updates
        changed = mp.record_high_water({"X": 107.5})
        assert changed is True
        after = mp.list_all()[0]
        assert after["highest_since_entry"] == 107.5

        # Still-higher price updates again
        mp.record_high_water({"X": 112.25})
        assert mp.list_all()[0]["highest_since_entry"] == 112.25

        # A drop afterward doesn't lower it (ratchet)
        mp.record_high_water({"X": 105.0})
        assert mp.list_all()[0]["highest_since_entry"] == 112.25
    finally:
        if os.path.exists(path):
            os.unlink(path)


def test_record_high_water_ignores_closed():
    path = _tmp()
    try:
        mp = MyPositions(path)
        pos = mp.add("X", 100.0, 10, stop=95, target=120)
        mp.close(pos["id"], exit_price=118)
        # Post-close updates should be ignored
        changed = mp.record_high_water({"X": 500.0})
        assert changed is False
        assert mp.list_all()[0]["highest_since_entry"] == 100.0
    finally:
        if os.path.exists(path):
            os.unlink(path)


def test_record_high_water_empty_prices():
    path = _tmp()
    try:
        mp = MyPositions(path)
        mp.add("X", 100.0, 10, stop=95, target=120)
        assert mp.record_high_water({}) is False
        assert mp.record_high_water(None) is False
    finally:
        if os.path.exists(path):
            os.unlink(path)


def test_trailing_stop_emitted_once_activated():
    path = _tmp()
    try:
        mp = MyPositions(path)
        today = today_et_str()
        # Entry 100, stop 95 → 1R = $5. Push highest to 110 (2R up).
        pos = mp.add("X", 100.0, 10, stop=95.0, target=130.0, entry_date=today)
        mp.record_high_water({"X": 110.0})

        # Fetch fresh copy (record_high_water persists the high-water mark)
        p = mp.list_all()[0]
        e = mp.evaluate(p, current_price=108.0, today=today)

        assert e["trailing_stop"] is not None
        assert e["effective_stop"] is not None
        # Effective stop is the higher of hard(95) and trail(~103.4) → trail wins
        assert e["effective_stop"] > 95.0
    finally:
        if os.path.exists(path):
            os.unlink(path)


def test_sell_trail_alert_fires_below_trailing_stop():
    path = _tmp()
    try:
        mp = MyPositions(path)
        today = today_et_str()
        pos = mp.add("X", 100.0, 10, stop=95.0, target=130.0, entry_date=today)
        mp.record_high_water({"X": 115.0})  # 3R up
        p = mp.list_all()[0]

        # Price drops to 105 — above hard stop (95), but below 6%-trail from 115 = 108.1
        e = mp.evaluate(p, current_price=105.0, today=today)
        assert e["alert"] == ALERT_SELL_TRAIL, f"got {e['alert']} / reason {e['alert_reason']}"
        # Still SELL_STOP takes priority only if price <= hard stop, which it isn't here
        assert "trail" in (e["alert_reason"] or "").lower()
    finally:
        if os.path.exists(path):
            os.unlink(path)


def test_hard_stop_takes_priority_over_trail():
    path = _tmp()
    try:
        mp = MyPositions(path)
        today = today_et_str()
        pos = mp.add("X", 100.0, 10, stop=95.0, target=130.0, entry_date=today)
        mp.record_high_water({"X": 115.0})
        p = mp.list_all()[0]

        # Price at 94 — below hard stop AND below any trail → SELL_STOP wins
        e = mp.evaluate(p, current_price=94.0, today=today)
        assert e["alert"] == ALERT_SELL_STOP
    finally:
        if os.path.exists(path):
            os.unlink(path)


def test_target_takes_priority_over_trail():
    path = _tmp()
    try:
        mp = MyPositions(path)
        today = today_et_str()
        pos = mp.add("X", 100.0, 10, stop=95.0, target=120.0, entry_date=today)
        mp.record_high_water({"X": 122.0})
        p = mp.list_all()[0]

        # Price at target — target wins
        e = mp.evaluate(p, current_price=121.0, today=today)
        assert e["alert"] == ALERT_SELL_TARGET
    finally:
        if os.path.exists(path):
            os.unlink(path)


# ================================================================
# Earnings-warning behavior (priority-1 feature)
# ================================================================

def test_earnings_warning_upcoming_days():
    path = _tmp()
    try:
        mp = MyPositions(path)
        today = today_et_str()
        pos = mp.add("XYZ", 100.0, 10, stop=95.0, target=110.0, entry_date=today)

        def checker(ticker):
            return 3 if ticker == "XYZ" else None

        e = mp.evaluate(pos, current_price=102.0, today=today, earnings_checker=checker)
        assert e["earnings_warning"] is not None
        assert "3d" in e["earnings_warning"].lower() or "3 d" in e["earnings_warning"].lower()
        # Primary alert still HOLD — earnings is a separate axis
        assert e["alert"] == ALERT_HOLD
    finally:
        if os.path.exists(path):
            os.unlink(path)


def test_earnings_warning_today():
    path = _tmp()
    try:
        mp = MyPositions(path)
        today = today_et_str()
        pos = mp.add("XYZ", 100.0, 10, stop=95.0, target=110.0, entry_date=today)
        e = mp.evaluate(
            pos, current_price=102.0, today=today,
            earnings_checker=lambda t: 0,
        )
        assert e["earnings_warning"] is not None
        assert "today" in e["earnings_warning"].lower()
    finally:
        if os.path.exists(path):
            os.unlink(path)


def test_earnings_warning_recently_past():
    path = _tmp()
    try:
        mp = MyPositions(path)
        today = today_et_str()
        pos = mp.add("XYZ", 100.0, 10, stop=95.0, target=110.0, entry_date=today)
        e = mp.evaluate(
            pos, current_price=102.0, today=today,
            earnings_checker=lambda t: -1,
        )
        assert e["earnings_warning"] is not None
        assert "ago" in e["earnings_warning"].lower()
    finally:
        if os.path.exists(path):
            os.unlink(path)


def test_earnings_warning_far_out_is_silent():
    path = _tmp()
    try:
        mp = MyPositions(path)
        today = today_et_str()
        pos = mp.add("XYZ", 100.0, 10, stop=95.0, target=110.0, entry_date=today)
        # 20 days out → outside the -1..7 window → no warning
        e = mp.evaluate(
            pos, current_price=102.0, today=today,
            earnings_checker=lambda t: 20,
        )
        assert e["earnings_warning"] is None
    finally:
        if os.path.exists(path):
            os.unlink(path)


def test_earnings_checker_errors_dont_crash_eval():
    path = _tmp()
    try:
        mp = MyPositions(path)
        today = today_et_str()
        pos = mp.add("XYZ", 100.0, 10, stop=95.0, target=110.0, entry_date=today)

        def angry(ticker):
            raise RuntimeError("network down")

        e = mp.evaluate(pos, current_price=102.0, today=today, earnings_checker=angry)
        assert e["earnings_warning"] is None
        assert e["alert"] == ALERT_HOLD
    finally:
        if os.path.exists(path):
            os.unlink(path)


# ================================================================
# ET time + business-day counting (priority-4 feature)
# ================================================================

def test_days_held_is_business_days():
    # Friday entry + Monday today = 1 business day, not 3 calendar days
    path = _tmp()
    try:
        mp = MyPositions(path)
        pos = mp.add("X", 100.0, 10, stop=95, target=110,
                     entry_date="2026-04-10", time_stop_days=5)
        e = mp.evaluate(pos, current_price=102.0, today="2026-04-13")
        assert e["days_held"] == 1, f"expected 1 business day, got {e['days_held']}"
    finally:
        if os.path.exists(path):
            os.unlink(path)


# ================================================================
# Summary: realized-today tracking (priority-7 feature)
# ================================================================

def test_summary_realized_today():
    path = _tmp()
    try:
        mp = MyPositions(path)
        today = today_et_str()
        a = mp.add("A", 100, 10, stop=95, target=120, entry_date=today)
        b = mp.add("B", 100, 10, stop=95, target=120, entry_date=today)

        # Close one today with a loss, one today with a gain
        mp.close(a["id"], exit_price=92, exit_date=today)   # -$80
        mp.close(b["id"], exit_price=118, exit_date=today)  # +$180

        s = mp.summary()
        assert s["closed_today"] == 2
        assert s["realized_today_dollars"] == 100.0

        # Now close one backdated to yesterday — shouldn't count in today's
        c = mp.add("C", 100, 10, stop=95, target=120, entry_date="2026-04-10")
        mp.close(c["id"], exit_price=90, exit_date="2026-04-11")
        s2 = mp.summary()
        assert s2["closed_today"] == 2  # unchanged
        assert s2["realized_today_dollars"] == 100.0
    finally:
        if os.path.exists(path):
            os.unlink(path)


# ============================================================================
# Swing-trader integration — partial ladder, quick-profit, warn-timestop
# ============================================================================

def _bday_iso_offset(n):
    """Return today-n business days as YYYY-MM-DD string."""
    return str(np.busday_offset(np.datetime64(today_et_str(), "D"), -int(n), roll="following"))


def test_default_time_stop_is_swing_horizon():
    # Regression: swing trader default is 5 business days, not 10
    path = _tmp()
    try:
        mp = MyPositions(path)
        pos = mp.add("AAA", entry_price=100, shares=10, stop=95, target=110)
        assert pos["time_stop_days"] == DEFAULT_TIME_STOP_DAYS == 5
    finally:
        if os.path.exists(path):
            os.unlink(path)


def test_partial_exit_field_attached_at_1r():
    path = _tmp()
    try:
        mp = MyPositions(path)
        pos = mp.add("AAA", entry_price=100, shares=90, stop=95, target=120,
                     entry_date=_bday_iso_offset(1))
        out = mp.evaluate(pos, current_price=105.0)
        assert out["partial_exit"] is not None
        assert out["partial_exit"]["rung"] == "1R"
        # Evaluate doesn't mutate the alert to a hold-override
        assert out["alert"] == ALERT_HOLD
    finally:
        if os.path.exists(path):
            os.unlink(path)


def test_partial_exit_upgrades_to_2r():
    path = _tmp()
    try:
        mp = MyPositions(path)
        pos = mp.add("AAA", entry_price=100, shares=100, stop=95, target=120,
                     entry_date=_bday_iso_offset(1))
        out = mp.evaluate(pos, current_price=110.0)
        assert out["partial_exit"]["rung"] == "2R"
    finally:
        if os.path.exists(path):
            os.unlink(path)


def test_partial_exit_none_at_loss():
    path = _tmp()
    try:
        mp = MyPositions(path)
        pos = mp.add("AAA", entry_price=100, shares=100, stop=95, target=120,
                     entry_date=_bday_iso_offset(1))
        out = mp.evaluate(pos, current_price=98.0)
        assert out["partial_exit"] is None
    finally:
        if os.path.exists(path):
            os.unlink(path)


def test_quick_profit_fires_day_1_big_pop():
    path = _tmp()
    try:
        mp = MyPositions(path)
        pos = mp.add("AAA", entry_price=100, shares=100, stop=95, target=120,
                     entry_date=_bday_iso_offset(1))
        out = mp.evaluate(pos, current_price=106.5)  # +6.5%
        assert out["quick_profit"] is not None
        assert out["quick_profit"]["gain_pct"] == 6.5
        assert out["quick_profit"]["days_held"] == 1
    finally:
        if os.path.exists(path):
            os.unlink(path)


def test_quick_profit_silent_past_day_2():
    path = _tmp()
    try:
        mp = MyPositions(path)
        # Entered 3 business days ago → outside quick-profit window
        pos = mp.add("AAA", entry_price=100, shares=100, stop=95, target=120,
                     entry_date=_bday_iso_offset(3))
        out = mp.evaluate(pos, current_price=106.5)
        assert out["quick_profit"] is None
    finally:
        if os.path.exists(path):
            os.unlink(path)


def test_warn_timestop_fires_with_1_day_left():
    path = _tmp()
    try:
        mp = MyPositions(path)
        # Held 4 of 5 days → 1 day left → WARN_TIMESTOP
        pos = mp.add("AAA", entry_price=100, shares=10, stop=95, target=110,
                     time_stop_days=5, entry_date=_bday_iso_offset(4))
        out = mp.evaluate(pos, current_price=101.0)
        assert out["time_stop_warn"] is True
        assert out["alert"] == ALERT_WARN_TIMESTOP
        assert "1 business day" in out["alert_reason"]
    finally:
        if os.path.exists(path):
            os.unlink(path)


def test_warn_timestop_silent_with_runway():
    path = _tmp()
    try:
        mp = MyPositions(path)
        pos = mp.add("AAA", entry_price=100, shares=10, stop=95, target=110,
                     time_stop_days=5, entry_date=_bday_iso_offset(1))
        out = mp.evaluate(pos, current_price=101.0)
        assert out["time_stop_warn"] is False
        assert out["alert"] == ALERT_HOLD
    finally:
        if os.path.exists(path):
            os.unlink(path)


def test_warn_timestop_emitted_without_live_price():
    # Swing trader opens the dashboard pre-market (no live price) — still
    # needs to see "1 day left".
    path = _tmp()
    try:
        mp = MyPositions(path)
        pos = mp.add("AAA", entry_price=100, shares=10, stop=95, target=110,
                     time_stop_days=5, entry_date=_bday_iso_offset(4))
        out = mp.evaluate(pos, current_price=None)
        assert out["time_stop_warn"] is True
        assert out["alert"] == ALERT_WARN_TIMESTOP
    finally:
        if os.path.exists(path):
            os.unlink(path)


def test_sell_stop_beats_warn_timestop():
    # Hard exits always win, even on the last day.
    path = _tmp()
    try:
        mp = MyPositions(path)
        pos = mp.add("AAA", entry_price=100, shares=10, stop=95, target=110,
                     time_stop_days=5, entry_date=_bday_iso_offset(4))
        out = mp.evaluate(pos, current_price=94.0)
        assert out["alert"] == ALERT_SELL_STOP


    finally:
        if os.path.exists(path):
            os.unlink(path)


def test_sell_timestop_takes_over_after_limit():
    path = _tmp()
    try:
        mp = MyPositions(path)
        # 6 days held with limit 5 → past limit → SELL_TIMESTOP
        pos = mp.add("AAA", entry_price=100, shares=10, stop=95, target=110,
                     time_stop_days=5, entry_date=_bday_iso_offset(6))
        out = mp.evaluate(pos, current_price=101.0)
        assert out["alert"] == ALERT_SELL_TIMESTOP
        assert out["time_stop_warn"] is False  # past the warn window
    finally:
        if os.path.exists(path):
            os.unlink(path)


def test_closed_positions_have_no_swing_advisories():
    path = _tmp()
    try:
        mp = MyPositions(path)
        pos = mp.add("AAA", entry_price=100, shares=10, stop=95, target=110)
        closed = mp.close(pos["id"], exit_price=108)
        out = mp.evaluate(closed, current_price=108.0)
        assert out["partial_exit"] is None
        assert out["quick_profit"] is None
        assert out["time_stop_warn"] is False
    finally:
        if os.path.exists(path):
            os.unlink(path)


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
