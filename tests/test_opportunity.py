from titan.models import TitanSetup
from titan.opportunity import (
    build_scan_export_data,
    build_scan_opportunity_summary,
    normalize_scan_payload,
    summarize_replay_opportunity,
)


def _make_setup():
    setup = TitanSetup(
        ticker="AAPL",
        strategy="BREAKOUT",
        price=200.0,
        trigger=201.0,
        stop=193.0,
        target=219.0,
        qty=10,
        win_rate=57.4,
        profit_factor=1.8,
        kelly=12.0,
        score=140.0,
        sector="Technology",
        earnings_call="Unknown",
        note="demo",
        confidence_grade="A",
        momentum_score=88.0,
        rs_percentile=91.0,
        robustness_score=74.0,
        walk_forward_pass_rate=0.5,
        walk_forward_pf=1.3,
        walk_forward_trades=22,
        regime_score=0.67,
        oos_pf=1.4,
        oos_trades=8,
        net_expectancy=0.005,
        distance_to_entry_pct=0.6,
        confirmation_status="READY",
        entry_ready_score=21.0,
    )
    setup.sector_aligned = True
    setup.distance_from_high_pct = 1.2
    setup.distance_to_pivot_pct = 0.4
    return setup


def test_opportunity_summary_marks_actionable_when_setups_exist():
    summary = build_scan_opportunity_summary(
        [_make_setup()],
        [{"ticker": "MSFT"}],
        "STRONG_BULL",
        vix_level=18.5,
    )

    assert summary["state"] == "ACTIONABLE"
    assert summary["actionable_count"] == 1
    assert summary["research_count"] == 1


def test_opportunity_summary_marks_research_when_only_watchlist_exists():
    summary = build_scan_opportunity_summary(
        [],
        [{"ticker": "MSFT"}, {"ticker": "NVDA"}],
        "BULL+CAUTION",
        vix_level=24.0,
    )

    assert summary["state"] == "RESEARCH"
    assert "research name" in summary["headline"].lower()


def test_build_scan_export_data_includes_opportunity_fields():
    export = build_scan_export_data(
        [_make_setup()],
        {"Passed": 1, "Total": 503},
        {
            "mkt_status": "BULL",
            "top_sectors": ["Technology"],
            "watchlist": [{"ticker": "MSFT", "theme": "LEADER", "score": 88.2, "status": "STALK"}],
        },
        vix_level=19.4,
        timestamp="2026-03-16T00:00:00",
        action_labeler=lambda setup: "BUY NOW",
    )

    assert export["actionable_count"] == 1
    assert export["research_watchlist_count"] == 1
    assert export["opportunity_state"] == "ACTIONABLE"
    assert export["setups"][0]["action"] == "BUY NOW"
    assert export["research_watchlist"][0]["ticker"] == "MSFT"


def test_opportunity_summary_distinguishes_pilot_only_sessions():
    setup = _make_setup()
    setup.opportunity_tier = "PILOT"
    setup.execution_eligible = False

    summary = build_scan_opportunity_summary(
        [setup],
        [{"ticker": "MSFT"}],
        "BULL",
        vix_level=19.4,
    )

    assert summary["state"] == "ACTIONABLE"
    assert summary["pilot_count"] == 1
    assert summary["validated_count"] == 0
    assert "pilot setup" in summary["headline"].lower()


def test_summarize_replay_opportunity_counts_actionable_research_and_quiet_days():
    summary = summarize_replay_opportunity(
        [
            {"setups_found": 1, "watchlist_count": 3},
            {"setups_found": 0, "watchlist_count": 5},
            {"setups_found": 0, "watchlist_count": 0},
            {"setups_found": 0, "watchlist_count": 0},
            {"setups_found": 2, "watchlist_count": 4},
        ]
    )

    assert summary["Actionable_Days"] == 2
    assert summary["Research_Days"] == 1
    assert summary["Quiet_Days"] == 2
    assert summary["Idea_Days"] == 3
    assert summary["Max_Quiet_Streak"] == 2


def test_normalize_scan_payload_backfills_legacy_fields():
    normalized = normalize_scan_payload(
        {
            "timestamp": "2026-03-17T00:00:00",
            "market_status": "BULL+CAUTION",
            "watchlist": [{"ticker": "CVX", "theme": "BREAKOUT"}],
            "setups": [],
        }
    )

    assert normalized["research_watchlist"][0]["ticker"] == "CVX"
    assert normalized["watchlist_count"] == 1
    assert normalized["research_watchlist_count"] == 1
    assert normalized["opportunity_state"] == "RESEARCH"


def test_build_scan_export_data_serializes_pilot_metadata():
    setup = _make_setup()
    setup.opportunity_tier = "PILOT"
    setup.execution_eligible = False
    setup.position_size_scalar = 0.5

    export = build_scan_export_data(
        [setup],
        {"Passed": 1, "Total": 100},
        {"mkt_status": "BULL", "top_sectors": ["Technology"], "watchlist": []},
        vix_level=17.2,
        timestamp="2026-03-17T00:00:00",
    )

    assert export["pilot_count"] == 1
    assert export["validated_count"] == 0
    assert export["setups"][0]["opportunity_tier"] == "PILOT"
    assert export["setups"][0]["execution_eligible"] is False
