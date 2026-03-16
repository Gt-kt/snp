import titan_trade_v3 as mod


def _settings(**overrides):
    settings = {
        "require_recent_validation_for_actionable": True,
        "actionable_dip_min_rs": 45.0,
        "actionable_dip_min_momentum": 55.0,
        "actionable_dip_min_accumulation": 30.0,
        "recent_validation_min_oos_trades_breakout": 5,
        "recent_validation_min_oos_trades_dip": 5,
        "recent_validation_min_oos_pf_breakout": 1.0,
        "recent_validation_min_oos_pf_dip": 1.05,
        "recent_validation_min_oos_expectancy_breakout": 0.0,
        "recent_validation_min_oos_expectancy_dip": 0.0,
        "recent_validation_min_wf_trades_breakout": 5,
        "recent_validation_min_wf_trades_dip": 20,
        "recent_validation_min_wf_pf_breakout": 1.0,
        "recent_validation_min_wf_pf_dip": 1.0,
        "recent_validation_min_wf_passrate_breakout": 0.25,
        "recent_validation_min_wf_passrate_dip": 0.34,
        "recent_validation_min_wf_expectancy_breakout": 0.0,
        "recent_validation_min_wf_expectancy_dip": 0.0,
    }
    settings.update(overrides)
    return settings


def test_actionable_quality_rejection_demotes_weak_dip():
    rejection = mod.get_actionable_quality_rejection(
        "DIP BUY",
        rs_pct=31.0,
        mom_score=58.0,
        accum_score=25.0,
        oos_stats={"pf": 2.16, "trades": 4, "expectancy": 0.005},
        wf_stats={"pf": 1.66, "trades": 75, "pass_rate": 0.0, "eligible_folds": 2, "expectancy": 0.001},
        settings=_settings(),
    )

    assert rejection == ("Research Only", "Dip RS 31 < 45")


def test_actionable_quality_rejection_allows_sponsored_dip_with_recent_support():
    rejection = mod.get_actionable_quality_rejection(
        "DIP BUY",
        rs_pct=62.0,
        mom_score=66.0,
        accum_score=44.0,
        oos_stats={"pf": 1.42, "trades": 12, "expectancy": 0.003},
        wf_stats={"pf": 1.18, "trades": 28, "pass_rate": 0.5, "eligible_folds": 2, "expectancy": 0.001},
        settings=_settings(),
    )

    assert rejection is None


def test_actionable_quality_rejection_rejects_breakout_when_recent_validation_fails():
    rejection = mod.get_actionable_quality_rejection(
        "BREAKOUT",
        rs_pct=88.0,
        mom_score=79.0,
        accum_score=72.0,
        oos_stats={"pf": 0.92, "trades": 8, "expectancy": -0.001},
        wf_stats={"pf": 0.95, "trades": 9, "pass_rate": 0.0, "eligible_folds": 2, "expectancy": -0.0005},
        settings=_settings(),
    )

    assert rejection[0] == "Research Only"
    assert "WF pass" in rejection[1]
