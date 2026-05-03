from titan.risk import PortfolioRiskManager


def test_corrupt_risk_state_blocks_new_trades(tmp_path):
    risk_file = tmp_path / "risk_log.json"
    risk_file.write_text("{not-json", encoding="utf-8")

    manager = PortfolioRiskManager(risk_log_file=str(risk_file))
    can_trade, reason = manager.can_take_new_trade({})

    assert can_trade is False
    assert "RISK STATE LOAD FAILED" in reason
    assert (tmp_path / "risk_log.json.corrupt.bak").exists()
