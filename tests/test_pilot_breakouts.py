import titan_trade_v3 as mod
from titan.models import TitanSetup


def _pilot_settings(**overrides):
    settings = {
        "enable_pilot_breakouts": True,
        "pilot_breakout_min_prebreakout_score": 90.0,
        "pilot_breakout_min_rs": 90.0,
        "pilot_breakout_min_momentum": 85.0,
        "pilot_breakout_min_accumulation": 65.0,
        "pilot_breakout_max_high_distance_pct": 6.0,
        "pilot_breakout_min_confidence_grade": "B",
        "pilot_breakout_size_scalar": 0.5,
        "pilot_breakout_min_trades": 2,
        "pilot_breakout_min_winrate": 50.0,
        "pilot_breakout_min_pf": 2.0,
        "pilot_breakout_min_expectancy": 0.01,
        "pilot_breakout_skip_regime_floor": True,
        "pilot_breakout_min_robustness": 50.0,
        "pilot_leader_min_oos_trades": 5,
        "pilot_leader_min_oos_pf": 1.4,
    }
    settings.update(overrides)
    return settings


def _setup(**overrides):
    setup = TitanSetup(
        ticker="XOM",
        strategy="BREAKOUT",
        price=110.0,
        trigger=111.0,
        stop=106.0,
        target=121.0,
        qty=10,
        win_rate=60.0,
        profit_factor=2.1,
        kelly=12.0,
        score=120.0,
        sector="Energy",
        earnings_call="Unknown",
        note="demo",
        confidence_grade="B",
    )
    for key, value in overrides.items():
        setattr(setup, key, value)
    return setup


def test_meets_minimum_allows_rounding_noise_at_threshold():
    assert mod.meets_minimum(1.2499999995, 1.25)


def test_get_pilot_breakout_profile_accepts_exceptional_low_sample_breakout():
    profile = mod.get_pilot_breakout_profile(
        "BREAKOUT",
        "BULL",
        True,
        rs_pct=94.0,
        mom_score=90.0,
        accum_score=71.0,
        pre_breakout_score=95.0,
        distance_from_high_pct=2.5,
        backtest_stats={"trades": 3, "win_rate": 66.7, "pf": 2.4, "expectancy": 0.018},
        confidence={"grade": "B"},
        settings=_pilot_settings(),
    )

    assert profile["tier"] == "PILOT"
    assert profile["sample_trades_floor"] == 2
    assert profile["size_scalar"] == 0.5


def test_should_allow_pilot_precision_override_only_for_leader_precision_rules():
    profile = mod.get_pilot_breakout_profile(
        "LEADER BO",
        "BULL",
        True,
        rs_pct=95.0,
        mom_score=91.0,
        accum_score=72.0,
        pre_breakout_score=96.0,
        distance_from_high_pct=1.5,
        backtest_stats={"trades": 26, "win_rate": 46.0, "pf": 1.43, "expectancy": 0.004},
        confidence={"grade": "B"},
        settings=_pilot_settings(),
    )

    assert mod.should_allow_pilot_precision_override(
        profile,
        ("Precision Filter", "Leader breakouts reserved for STRONG_BULL"),
        oos_pf=1.5,
        oos_trades=7,
        robustness_score=58.0,
    )
    assert not mod.should_allow_pilot_precision_override(
        profile,
        ("Precision Filter", "Leader wait 3.8% > 2.5%"),
        oos_pf=1.5,
        oos_trades=7,
        robustness_score=58.0,
    )


def test_manual_action_label_marks_pilot_setup():
    setup = _setup(opportunity_tier="PILOT", confirmation_status="READY")

    assert mod.manual_action_label(setup) == "BUY PILOT"


def test_filter_execution_candidates_skips_manual_only_setups():
    setup = _setup(execution_eligible=False)

    selected, skipped = mod.filter_execution_candidates(
        [setup],
        {
            "min_confidence_grade": "C",
            "min_momentum_score": 0.0,
            "min_accumulation_score": 0.0,
            "min_rs_percentile": 0.0,
        },
    )

    assert selected == []
    assert skipped == [("XOM", "manual-only")]
