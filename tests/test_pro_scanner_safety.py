from titan.execution_plan import classify_gap_risk
from titan.pro_scanner import PortfolioFilter, _classify_gap_risk


def test_gap_risk_accepts_probability_percent_units():
    assert classify_gap_risk(0.4, 8.0) == "LOW"
    assert classify_gap_risk(0.4, 0.08) == "LOW"
    assert _classify_gap_risk(0.4, 8.0) == "LOW"
    assert _classify_gap_risk(0.4, 0.08) == "LOW"


def test_gap_risk_marks_high_only_above_threshold():
    assert classify_gap_risk(0.4, 31.0) == "HIGH"
    assert classify_gap_risk(0.4, 0.31) == "HIGH"
    assert classify_gap_risk(1.6, 5.0) == "HIGH"


def test_portfolio_filter_rejects_already_held_symbols():
    pf = PortfolioFilter(
        open_positions={
            "AAPL": {
                "entry_price": 100,
                "stop": 95,
                "shares": 10,
                "sector": "Technology",
            }
        },
        max_total=5,
        max_per_sector=5,
    )

    accepted, rejected = pf.filter(
        [
            {"ticker": "AAPL", "risk_dollars": 100, "sector": "Technology"},
            {"ticker": "MSFT", "risk_dollars": 100, "sector": "Technology"},
        ]
    )

    assert [s["ticker"] for s in accepted] == ["MSFT"]
    assert rejected[0]["_rejected"] == "already_held"
