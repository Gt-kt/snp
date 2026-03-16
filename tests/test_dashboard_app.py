import asyncio

import dashboard_app


def test_scan_results_uses_absolute_file_path(monkeypatch, tmp_path):
    scan_file = tmp_path / "data" / "latest_scan.json"
    scan_file.parent.mkdir(parents=True)
    scan_file.write_text('{"status": "ok"}', encoding="utf-8")
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()

    monkeypatch.setattr(dashboard_app, "SCAN_RESULTS_FILE", scan_file)
    monkeypatch.chdir(elsewhere)

    result = asyncio.run(dashboard_app.get_scan_results())

    assert dashboard_app.STATIC_DIR.is_absolute()
    assert result["status"] == "success"
    assert result["data"]["status"] == "ok"
    assert result["data"]["opportunity_state"] == "QUIET"


def test_portfolio_endpoint_uses_executor_facade(monkeypatch):
    class DummyExecutor:
        def is_connected(self):
            return True

        def get_account_summary(self):
            return {
                "buying_power": 1000.0,
                "cash": 500.0,
                "portfolio_value": 1500.0,
                "day_trade_count": 0,
                "account_status": "ACTIVE",
                "account_mode": "paper",
            }

        def get_positions_summary(self):
            return [
                {
                    "symbol": "AAPL",
                    "qty": 2.0,
                    "market_value": 400.0,
                    "unrealized_pl": 10.0,
                    "unrealized_plpc": 2.5,
                    "current_price": 200.0,
                    "avg_entry_price": 195.0,
                }
            ]

    monkeypatch.setattr(dashboard_app, "AlpacaExecutor", DummyExecutor)

    result = asyncio.run(dashboard_app.get_portfolio())

    assert result["status"] == "success"
    assert result["account"]["account_mode"] == "paper"
    assert result["positions"][0]["symbol"] == "AAPL"


def test_scan_results_normalizes_legacy_payload(monkeypatch, tmp_path):
    scan_file = tmp_path / "data" / "latest_scan.json"
    scan_file.parent.mkdir(parents=True)
    scan_file.write_text(
        '{"market_status":"BULL+CAUTION","watchlist":[{"ticker":"CVX","theme":"BREAKOUT"}],"setups":[]}',
        encoding="utf-8",
    )

    monkeypatch.setattr(dashboard_app, "SCAN_RESULTS_FILE", scan_file)

    result = asyncio.run(dashboard_app.get_scan_results())

    assert result["status"] == "success"
    assert result["data"]["research_watchlist_count"] == 1
    assert result["data"]["opportunity_state"] == "RESEARCH"
