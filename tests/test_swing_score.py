"""Unit tests for titan.swing_score — setup quality scoring for 2-4 day holds."""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from titan.swing_score import (
    score_swing_setup,
    rank_stalk_orders,
    top_picks,
    TOP_PICK_MIN,
    STRONG_MIN,
    DECENT_MIN,
)


def _ideal_row(**overrides):
    """Baseline row that scores perfect — mutate one field to test impact."""
    r = {
        "grade": "A",
        "rsi": 55,
        "volume_ratio": 2.1,
        "rr_ratio": 3.2,
        "change_5d": 5.0,
        "bt_trades": 25,
        "bt_win_rate": 62,
        "vol_contraction": True,
        "price": 102,
        "max_buy_price": 104,
        "buy_zone_low": 100,
    }
    r.update(overrides)
    return r


# --------------------------------------------------------------- score_swing_setup

def test_score_ideal_setup_is_top_pick():
    s = score_swing_setup(_ideal_row())
    assert s["verdict"] == "TOP_PICK"
    assert s["score"] >= TOP_PICK_MIN
    # All eight components contributed
    assert s["breakdown"]["momentum"] == 20
    assert s["breakdown"]["volume"] == 15
    assert s["breakdown"]["rr"] == 15
    assert s["breakdown"]["history"] == 15
    assert s["breakdown"]["base"] == 5
    assert s["breakdown"]["grade"] == 10


def test_score_all_weak_setup_is_weak():
    weak = {
        "grade": "C", "rsi": 82, "volume_ratio": 0.6, "rr_ratio": 1.0,
        "change_5d": -4, "bt_trades": 3, "bt_win_rate": 45,
        "vol_contraction": False,
    }
    s = score_swing_setup(weak)
    assert s["verdict"] == "WEAK"
    assert s["score"] < DECENT_MIN


def test_score_negative_momentum_scores_zero():
    s = score_swing_setup(_ideal_row(change_5d=-5))
    assert s["breakdown"]["momentum"] == 0


def test_score_overextended_momentum_penalized():
    # +20% in 5d = overextended (top-picking risk)
    s = score_swing_setup(_ideal_row(change_5d=20))
    assert s["breakdown"]["momentum"] == 6
    # Still should land below a fresh +5% setup
    fresh = score_swing_setup(_ideal_row(change_5d=5))
    assert s["score"] < fresh["score"]


def test_score_rsi_overbought_heavily_penalized():
    s = score_swing_setup(_ideal_row(rsi=80))
    assert s["breakdown"]["rsi"] == 2


def test_score_rsi_falling_knife_zero():
    s = score_swing_setup(_ideal_row(rsi=25))
    assert s["breakdown"]["rsi"] == 0


def test_score_missing_rsi_is_neutral():
    # Unknown RSI shouldn't punish — we have honest data gaps
    s = score_swing_setup(_ideal_row(rsi=None))
    assert s["breakdown"]["rsi"] == 5


def test_score_low_volume_scores_zero():
    s = score_swing_setup(_ideal_row(volume_ratio=0.8))
    assert s["breakdown"]["volume"] == 0


def test_score_rr_below_1_5_zero():
    s = score_swing_setup(_ideal_row(rr_ratio=1.2))
    assert s["breakdown"]["rr"] == 0


def test_score_history_needs_sample_size():
    # 80% WR but only 3 trades → zero (not statistically meaningful)
    s = score_swing_setup(_ideal_row(bt_trades=3, bt_win_rate=80))
    assert s["breakdown"]["history"] == 0


def test_score_proximity_chasing_past_max_buy():
    # Live price 5% above max_buy — chasing
    s = score_swing_setup(_ideal_row(price=109.2), live_price=109.2)
    assert s["breakdown"]["proximity"] == 0


def test_score_proximity_near_zone_low_is_best():
    # Inside zone, near the low
    s = score_swing_setup(_ideal_row(price=100.3), live_price=100.3)
    near_low = s["breakdown"]["proximity"]
    # Inside zone, near the top
    s2 = score_swing_setup(_ideal_row(price=103.8), live_price=103.8)
    near_top = s2["breakdown"]["proximity"]
    assert near_low > near_top


def test_score_proximity_unknown_is_neutral():
    row = _ideal_row()
    row.pop("max_buy_price")
    row.pop("buy_zone_low")
    s = score_swing_setup(row)
    assert s["breakdown"]["proximity"] == 5


def test_score_grade_a_outscores_grade_b():
    a = score_swing_setup(_ideal_row(grade="A"))
    b = score_swing_setup(_ideal_row(grade="B"))
    c = score_swing_setup(_ideal_row(grade="C"))
    assert a["score"] > b["score"] > c["score"]


def test_score_reasons_generated_for_qualifying_setup():
    s = score_swing_setup(_ideal_row())
    assert len(s["reasons"]) >= 5
    # Specific mentions
    text = " | ".join(s["reasons"])
    assert "5-day trend" in text
    assert "Volume" in text
    assert "R/R" in text


def test_score_reasons_empty_for_weak_setup():
    weak = {
        "grade": "C", "rsi": 82, "volume_ratio": 0.6, "rr_ratio": 1.0,
        "change_5d": -4, "bt_trades": 3, "bt_win_rate": 45,
        "vol_contraction": False,
    }
    s = score_swing_setup(weak)
    assert len(s["reasons"]) == 0


def test_score_none_row_doesnt_crash():
    s = score_swing_setup({})
    assert s["verdict"] == "WEAK"
    assert s["score"] >= 0


def test_score_verdict_thresholds_are_monotonic():
    # TOP_PICK > STRONG > DECENT > WEAK via progressive degradation
    tp = score_swing_setup(_ideal_row())
    assert tp["verdict"] == "TOP_PICK"
    # Mid-tier: drop a couple of strong factors
    strong = score_swing_setup(_ideal_row(
        grade="B", bt_win_rate=50, bt_trades=5, vol_contraction=False, rr_ratio=2.0,
    ))
    assert strong["verdict"] in ("STRONG", "DECENT")
    assert strong["score"] < tp["score"]
    # Weakest: collapse everything
    weak = score_swing_setup({
        "grade": "C", "rsi": 82, "volume_ratio": 0.6, "rr_ratio": 1.0,
        "change_5d": -4, "bt_trades": 3, "bt_win_rate": 45,
    })
    assert weak["verdict"] == "WEAK"
    assert weak["score"] < strong["score"]


# --------------------------------------------------------------- rank_stalk_orders

def test_rank_sorts_by_swing_score_desc():
    orders = [
        {"ticker": "WEAK", "grade": "C", "rsi": 80, "volume_ratio": 0.5,
         "rr_ratio": 1.0, "change_5d": -2, "bt_trades": 2, "bt_win_rate": 40,
         "trade_score": 1.0, "limit_price": 100},
        {"ticker": "STRONG", "grade": "A", "rsi": 55, "volume_ratio": 2.0,
         "rr_ratio": 2.8, "change_5d": 5, "bt_trades": 20, "bt_win_rate": 60,
         "trade_score": 1.0, "vol_contraction": True, "limit_price": 100,
         "max_buy_price": 102, "buy_zone_low": 100, "price": 101},
    ]
    ranked = rank_stalk_orders(orders)
    assert ranked[0]["ticker"] == "STRONG"
    assert ranked[1]["ticker"] == "WEAK"
    assert ranked[0]["swing_score"]["score"] > ranked[1]["swing_score"]["score"]


def test_rank_uses_row_lookup_for_missing_fields():
    # Stalk orders don't carry rsi/volume_ratio by default — row_lookup provides them
    orders = [{"ticker": "ABC", "grade": "A", "limit_price": 100, "trade_score": 5.0}]
    lookup_rows = {
        "ABC": {
            "rsi": 50, "volume_ratio": 1.8, "rr_ratio": 2.5, "change_5d": 4,
            "bt_trades": 15, "bt_win_rate": 58, "vol_contraction": True,
            "price": 101, "max_buy_price": 102, "buy_zone_low": 99,
        }
    }
    rank_stalk_orders(orders, row_lookup=lambda t: lookup_rows.get(t))
    # Should now have a real score, not the degenerate 0 from missing fields
    assert orders[0]["swing_score"]["score"] >= STRONG_MIN


def test_rank_tiebreaks_by_trade_score_then_ticker():
    # Two setups with identical swing score → the higher trade_score comes first
    base = {
        "grade": "A", "rsi": 55, "volume_ratio": 1.8, "rr_ratio": 2.5,
        "change_5d": 5, "bt_trades": 15, "bt_win_rate": 58,
        "vol_contraction": True, "price": 101, "max_buy_price": 102,
        "buy_zone_low": 99,
    }
    orders = [
        dict(base, ticker="B", trade_score=1.0, limit_price=102),
        dict(base, ticker="A", trade_score=5.0, limit_price=102),
    ]
    rank_stalk_orders(orders)
    assert orders[0]["ticker"] == "A"


def test_rank_empty_list_is_noop():
    assert rank_stalk_orders([]) == []


def test_rank_handles_row_lookup_exception():
    orders = [{"ticker": "X", "limit_price": 100, "trade_score": 1.0, "grade": "B"}]
    def bad_lookup(t):
        raise RuntimeError("db down")
    # Must not crash — falls back to whatever is on the order dict
    ranked = rank_stalk_orders(orders, row_lookup=bad_lookup)
    assert ranked[0]["swing_score"]["score"] >= 0


# --------------------------------------------------------------- top_picks

def test_top_picks_filters_below_min_score():
    orders = [
        {"ticker": "A", "swing_score": {"score": 90}},
        {"ticker": "B", "swing_score": {"score": 74}},
        {"ticker": "C", "swing_score": {"score": 80}},
    ]
    picks = top_picks(orders, limit=5, min_score=75)
    assert [p["ticker"] for p in picks] == ["A", "C"]


def test_top_picks_respects_limit():
    orders = [
        {"ticker": f"T{i}", "swing_score": {"score": 90}}
        for i in range(10)
    ]
    picks = top_picks(orders, limit=3, min_score=75)
    assert len(picks) == 3


def test_top_picks_empty_when_nothing_qualifies():
    orders = [{"ticker": "A", "swing_score": {"score": 50}}]
    assert top_picks(orders, min_score=75) == []


def test_top_picks_handles_missing_swing_score():
    orders = [{"ticker": "A"}]  # no swing_score at all
    assert top_picks(orders, min_score=75) == []


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
