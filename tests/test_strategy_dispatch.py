import titan_trade_v3 as mod


class DummyValidator:
    def backtest_breakout(self, return_trades=True, min_entry_idx=None, stop_mult=None, target_mult=None):
        return {
            "kind": "breakout",
            "min_entry_idx": min_entry_idx,
            "stop_mult": stop_mult,
            "target_mult": target_mult,
        }

    def backtest_leader_breakout(self, return_trades=True, min_entry_idx=None, stop_mult=None, target_mult=None):
        return {
            "kind": "leader",
            "min_entry_idx": min_entry_idx,
            "stop_mult": stop_mult,
            "target_mult": target_mult,
        }

    def backtest_dip(self, return_trades=True, min_entry_idx=None, stop_mult=None, target_mult=None):
        return {
            "kind": "dip",
            "min_entry_idx": min_entry_idx,
            "stop_mult": stop_mult,
            "target_mult": target_mult,
        }


def test_run_strategy_backtest_dispatches_leader_variant():
    validator = DummyValidator()

    result = mod.run_strategy_backtest(
        validator,
        True,
        return_trades=True,
        min_entry_idx=42,
        strategy_variant="leader",
        settings={
            "leader_breakout_stop_atr_mult": 1.8,
            "leader_breakout_target_atr_mult": 3.0,
        },
    )

    assert result["kind"] == "leader"
    assert result["min_entry_idx"] == 42
    assert result["stop_mult"] == 1.8
    assert result["target_mult"] == 3.0


def test_run_strategy_backtest_uses_strategy_specific_trade_plan():
    validator = DummyValidator()
    settings = {
        "breakout_stop_atr_mult": 2.0,
        "breakout_target_atr_mult": 2.5,
        "dip_stop_atr_mult": 2.0,
        "dip_target_atr_mult": 3.0,
    }

    breakout = mod.run_strategy_backtest(
        validator,
        True,
        return_trades=True,
        settings=settings,
    )
    dip = mod.run_strategy_backtest(
        validator,
        False,
        return_trades=True,
        settings=settings,
    )

    assert breakout["kind"] == "breakout"
    assert breakout["stop_mult"] == 2.0
    assert breakout["target_mult"] == 2.5
    assert dip["kind"] == "dip"
    assert dip["stop_mult"] == 2.0
    assert dip["target_mult"] == 3.0


def test_precision_filter_rejects_non_strong_bull_leader_breakout():
    rejection = mod.get_precision_filter_rejection(
        "LEADER BO",
        "BULL+CAUTION",
        {"distance_metric_pct": 0.4},
        robustness_score=70.0,
        oos_pf=2.5,
        oos_trades=8,
        settings={
            "high_precision_mode": True,
            "leader_breakout_precision_strong_bull_only": True,
            "leader_breakout_max_wait_pct": 1.0,
            "leader_breakout_precision_min_oos_trades": 5,
            "leader_breakout_precision_min_oos_pf": 2.0,
        },
    )

    assert rejection == ("Precision Filter", "Leader breakouts reserved for STRONG_BULL")
