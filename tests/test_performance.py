from titan.performance import build_performance_report, summarize_trades


def test_summarize_trades_computes_core_metrics():
    trades = [
        {"pnl_dollars": 100, "pnl_pct": 5, "entry_date": "2026-01-05", "exit_date": "2026-01-07"},
        {"pnl_dollars": -50, "pnl_pct": -2.5, "entry_date": "2026-01-05", "exit_date": "2026-01-06"},
        {"pnl_dollars": 25, "pnl_pct": 1, "entry_date": "2026-01-05", "exit_date": "2026-01-08"},
    ]

    summary = summarize_trades(trades)

    assert summary["trades"] == 3
    assert summary["wins"] == 2
    assert summary["win_rate"] == 66.7
    assert summary["net_pnl_dollars"] == 75
    assert summary["profit_factor"] == 2.5
    assert summary["expectancy_dollars"] == 25


def test_performance_report_groups_by_signal_and_warns_on_small_sample():
    positions = [
        {"status": "CLOSED", "pnl_dollars": 100, "pnl_pct": 5, "signal_type": "BREAKOUT", "sector": "TECH", "exit_reason": "TARGET"},
        {"status": "CLOSED", "pnl_dollars": -50, "pnl_pct": -2.5, "signal_type": "BREAKOUT", "sector": "TECH", "exit_reason": "STOP"},
        {"status": "OPEN", "pnl_dollars": None, "signal_type": "PULLBACK"},
    ]

    report = build_performance_report(positions)

    assert report["sample_warning"] is True
    assert report["overall"]["trades"] == 2
    assert report["by_signal_type"][0]["name"] == "BREAKOUT"
    assert report["by_signal_type"][0]["trades"] == 2
