from types import SimpleNamespace

import pandas as pd

import backtest_scanner as replay


def _make_df(rows):
    index = pd.date_range("2025-01-01", periods=len(rows), freq="B")
    return pd.DataFrame(rows, index=index)


def test_try_fill_breakout_on_intraday_trigger():
    df = _make_df(
        [
            {"Open": 9.8, "High": 9.9, "Low": 9.7, "Close": 9.8, "Volume": 1000},
            {"Open": 9.9, "High": 10.2, "Low": 9.8, "Close": 10.1, "Volume": 1200},
            {"Open": 10.1, "High": 10.3, "Low": 9.9, "Close": 10.2, "Volume": 1100},
        ]
    )
    setup = SimpleNamespace(strategy="BREAKOUT", trigger=10.0, stop=9.4)

    fill, status = replay.try_fill_setup(setup, df, scan_idx=0, entry_window_days=2)

    assert status == "filled"
    assert fill["entry_idx"] == 1
    assert fill["fill_price"] == 10.0


def test_simulate_trade_exit_hits_gap_target():
    df = _make_df(
        [
            {"Open": 9.8, "High": 10.2, "Low": 9.8, "Close": 10.0, "Volume": 1000},
            {"Open": 10.0, "High": 10.4, "Low": 9.9, "Close": 10.3, "Volume": 1200},
            {"Open": 11.2, "High": 11.4, "Low": 11.0, "Close": 11.1, "Volume": 1300},
        ]
    )
    setup = SimpleNamespace(
        strategy="BREAKOUT",
        stop=9.3,
        target=11.0,
        breakeven_trigger=0.0,
        trailing_stop=0.0,
    )

    trade = replay.simulate_trade_exit(setup, df, entry_idx=1, fill_price=10.0)

    assert trade["exit_reason"] == "gap_target"
    assert trade["exit_idx"] == 2
    assert trade["return_pct"] > 0


def test_simulate_trade_exit_does_not_lookahead_gap_stop():
    df = _make_df(
        [
            {"Open": 9.8, "High": 10.2, "Low": 9.8, "Close": 10.0, "Volume": 1000},
            {"Open": 10.0, "High": 11.6, "Low": 11.1, "Close": 11.4, "Volume": 1200},
            {"Open": 11.2, "High": 11.4, "Low": 10.9, "Close": 11.0, "Volume": 1300},
        ]
    )
    setup = SimpleNamespace(
        strategy="BREAKOUT",
        stop=9.3,
        target=13.0,
        breakeven_trigger=11.0,
        trailing_stop=0.2,
    )

    trade = replay.simulate_trade_exit(setup, df, entry_idx=1, fill_price=10.0)

    assert trade["exit_reason"] != "gap_stop"


def test_build_replay_settings_keeps_research_watchlist_enabled_by_default():
    settings = replay.build_replay_settings()

    assert settings["build_watchlist"] is True
    assert settings["always_build_watchlist"] is True
