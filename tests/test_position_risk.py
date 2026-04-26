"""Unit tests for titan.position_risk — risk validation + trailing stops."""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from titan.position_risk import (
    compute_risk_dollars,
    validate_position_risk,
    validate_entry_price,
    compute_trailing_stop,
    effective_stop,
    HARD_CAP_PCT,
    SOFT_WARN_PCT,
    SANITY_PRICE_DIVERGENCE,
)


# ------------------------------ compute_risk_dollars ------------------------

def test_risk_dollars_with_valid_stop():
    # 10 shares, entry 100, stop 95 → $50 risk
    assert compute_risk_dollars(100.0, 95.0, 10) == 50.0


def test_risk_dollars_missing_stop_uses_5pct():
    # Missing stop → assume 5% loss: 100 * 0.05 * 10 = $50
    assert compute_risk_dollars(100.0, None, 10) == 50.0


def test_risk_dollars_invalid_stop_above_entry():
    # Stop above entry is garbage → fallback to 5% worst-case
    assert compute_risk_dollars(100.0, 110.0, 10) == 50.0


def test_risk_dollars_zero_shares_is_zero():
    assert compute_risk_dollars(100.0, 95.0, 0) == 0.0


def test_risk_dollars_handles_none():
    assert compute_risk_dollars(None, None, None) == 0.0


# ------------------------------ validate_position_risk ----------------------

def test_validate_risk_ok_small_position():
    # 1 share at $100, 5% stop → $5 risk on $100k account = 0.005%
    r = validate_position_risk(100.0, 95.0, 1, account_size=100_000.0)
    assert r["ok"] is True
    assert r["level"] == "OK"
    assert r["risk_dollars"] == 5.0
    assert r["msg"] is None


def test_validate_risk_soft_warn():
    # Risk > SOFT_WARN_PCT but <= HARD_CAP_PCT
    # Target: risk_pct in (SOFT, HARD]. For $100k: (0.5%, 1.0%] = ($500, $1000]
    r = validate_position_risk(100.0, 95.0, 150, account_size=100_000.0)
    # risk = 5 * 150 = $750 → 0.75% of 100k
    assert r["ok"] is True
    assert r["level"] == "SOFT"
    assert r["risk_dollars"] == 750.0
    assert "soft" in r["msg"].lower()


def test_validate_risk_hard_cap():
    # $15 * 100 = $1500 on $100k = 1.5% → HARD
    r = validate_position_risk(100.0, 85.0, 100, account_size=100_000.0)
    assert r["ok"] is False
    assert r["level"] == "HARD"
    assert r["risk_dollars"] == 1500.0
    assert "hard cap" in r["msg"].lower()


def test_validate_risk_rejects_zero_account():
    r = validate_position_risk(100.0, 95.0, 10, account_size=0)
    assert r["ok"] is False
    assert r["level"] == "HARD"
    assert "account_size" in r["msg"].lower()


# ------------------------------ validate_entry_price ------------------------

def test_entry_price_ok_close_to_live():
    # entry=101, live=100 → div = 1/100 = exactly 1.0%
    r = validate_entry_price(101.0, 100.0)
    assert r["ok"] is True
    assert r["level"] == "OK"
    assert r["divergence_pct"] == 1.0


def test_entry_price_rejected_if_diverges_too_far():
    # 50% above live → HARD reject (fat-finger)
    r = validate_entry_price(150.0, 100.0)
    assert r["ok"] is False
    assert r["level"] == "HARD"
    assert r["divergence_pct"] == 50.0
    assert "diverg" in r["msg"].lower()


def test_entry_price_ok_when_live_unknown():
    # No live price → don't block
    r = validate_entry_price(100.0, None)
    assert r["ok"] is True


def test_entry_price_ok_just_under_threshold():
    # 9.9% divergence — well within the 10% max
    live = 100.0
    entry = live * (1 + SANITY_PRICE_DIVERGENCE * 0.99)
    r = validate_entry_price(entry, live)
    assert r["ok"] is True


def test_entry_price_blocked_just_over_threshold():
    # 11% divergence — past the 10% max
    live = 100.0
    entry = live * (1 + SANITY_PRICE_DIVERGENCE * 1.1)
    r = validate_entry_price(entry, live)
    assert r["ok"] is False


def test_entry_price_handles_zero_live():
    r = validate_entry_price(100.0, 0.0)
    assert r["ok"] is True


# ------------------------------ compute_trailing_stop -----------------------

def test_trailing_stop_not_active_without_profit():
    # Highest equals entry → no trailing
    assert compute_trailing_stop(100.0, 95.0, 100.0) is None


def test_trailing_stop_not_active_below_1R():
    # Need up > 1R. Entry 100, stop 95 → 1R = $5.
    # Highest 103 = 0.6R → not activated.
    assert compute_trailing_stop(100.0, 95.0, 103.0) is None


def test_trailing_stop_activates_above_1R():
    # Up > 1R: entry 100, stop 95, highest 108 (1.6R)
    # 6% trail from 108 = 101.52 → locks in some profit
    ts = compute_trailing_stop(100.0, 95.0, 108.0, trail_pct=0.06)
    assert ts is not None
    assert abs(ts - 101.52) < 0.01


def test_trailing_stop_breakeven_floor():
    # Trail falls below entry → clamped to entry * 0.999
    # 5% trail from 106 = 100.7 → above entry, no clamp
    # Use aggressive trail: 20% from 106 = 84.8 → should clamp to ~99.9
    ts = compute_trailing_stop(100.0, 95.0, 106.0, trail_pct=0.20)
    # Before clamp trail would be 84.8, which is below entry * 0.999 = 99.9
    assert ts is not None
    assert abs(ts - 99.9) < 0.01


def test_trailing_stop_atr_mode():
    # ATR-based: highest 108, ATR 2 → 108 - 2.5 * 2 = 103
    ts = compute_trailing_stop(100.0, 95.0, 108.0, atr=2.0)
    assert ts is not None
    assert abs(ts - 103.0) < 0.01


def test_trailing_stop_no_hard_stop_activates_above_3pct():
    # No hard stop → activates only after > 3% upside
    # 2% up = not active
    assert compute_trailing_stop(100.0, None, 102.0) is None
    # 5% up = active
    ts = compute_trailing_stop(100.0, None, 105.0)
    assert ts is not None


def test_trailing_stop_zero_entry():
    assert compute_trailing_stop(0, 95, 108) is None


# ------------------------------ effective_stop ------------------------------

def test_effective_stop_prefers_trailing_when_higher():
    # After a big run: hard 95, trailing ~101 → effective is trailing
    es = effective_stop(100.0, 95.0, 108.0)
    assert es is not None
    assert es > 95.0  # trailing is higher


def test_effective_stop_uses_hard_when_no_trail():
    # Not profitable enough → no trailing → return hard stop
    es = effective_stop(100.0, 95.0, 99.0)
    assert es == 95.0


def test_effective_stop_with_no_inputs_returns_none():
    assert effective_stop(100.0, None, None) is None


def test_effective_stop_negative_hard_stop_ignored():
    # Bad hard stop → trailing only (no trailing here → None)
    assert effective_stop(100.0, -5.0, 99.0) is None


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
