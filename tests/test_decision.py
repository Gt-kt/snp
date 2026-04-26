from titan.decision import BUY, SKIP, WAIT, build_trade_decision


def _order(**overrides):
    row = {
        "ticker": "AAA",
        "grade": "A",
        "tier": "ACTIVE",
        "qty": 10,
        "limit_price": 100.0,
        "stop_price": 95.0,
        "target_price": 112.0,
        "risk_dollars": 50.0,
        "time_stop_days": 5,
        "live_status": "IN_ZONE",
        "intraday_status": "CONFIRMED",
        "trade_score": 10.0,
        "bt_trades": 60,
        "bt_win_rate": 55.0,
        "move_up_prob": 55.0,
        "move_expected_return": 0.4,
        "profile_samples": 8,
        "event_risk": {"status": "LOW", "reasons": []},
        "swing_score": {"score": 82, "verdict": "TOP_PICK"},
    }
    row.update(overrides)
    return row


def test_buy_when_top_order_passes_all_gates():
    decision = build_trade_decision(
        {"regime": "STRONG_BULL", "vix": 18.0, "stalk_orders": [_order()]},
        open_positions=[],
        settings={"risk_per_trade": 500, "max_positions": 6},
    )

    assert decision["action"] == BUY
    assert decision["ticker"] == "AAA"
    assert decision["candidate"]["limit_price"] == 100.0


def test_skip_when_no_stalk_orders():
    decision = build_trade_decision({"stalk_orders": []})

    assert decision["action"] == SKIP
    assert decision["candidate"] is None


def test_wait_when_best_order_is_already_held():
    decision = build_trade_decision(
        {"regime": "BULL", "vix": 18.0, "stalk_orders": [_order(already_held=True)]},
        open_positions=[{"ticker": "AAA", "status": "OPEN"}],
        settings={"risk_per_trade": 500, "max_positions": 6},
    )

    assert decision["action"] == WAIT
    assert "already held" in decision["blockers"]


def test_wait_when_position_cap_full():
    decision = build_trade_decision(
        {"regime": "BULL", "vix": 18.0, "stalk_orders": [_order()]},
        open_positions=[
            {"ticker": f"T{i}", "status": "OPEN"}
            for i in range(6)
        ],
        settings={"risk_per_trade": 500, "max_positions": 6},
    )

    assert decision["action"] == WAIT
    assert any("position cap full" in b for b in decision["blockers"])


def test_wait_when_price_is_past_max_buy():
    decision = build_trade_decision(
        {"regime": "BULL", "vix": 18.0, "stalk_orders": [_order(live_status="PAST_MAX_BUY")]},
        settings={"risk_per_trade": 500, "max_positions": 6},
    )

    assert decision["action"] == WAIT
    assert "past_max_buy" in decision["reason"]


def test_wait_when_swing_score_is_too_low():
    decision = build_trade_decision(
        {"regime": "BULL", "vix": 18.0, "stalk_orders": [_order(swing_score={"score": 60})]},
        settings={"risk_per_trade": 500, "max_positions": 6},
    )

    assert decision["action"] == WAIT
    assert "swing score 60 < 75" in decision["blockers"]


def test_wait_when_intraday_is_not_confirmed():
    decision = build_trade_decision(
        {"regime": "BULL", "vix": 18.0, "stalk_orders": [_order(intraday_status="WAIT_INTRADAY")]},
        settings={"risk_per_trade": 500, "max_positions": 6},
    )

    assert decision["action"] == WAIT
    assert "wait_intraday" in decision["blockers"]


def test_wait_when_event_risk_is_high():
    decision = build_trade_decision(
        {
            "regime": "BULL",
            "vix": 18.0,
            "stalk_orders": [_order(event_risk={"status": "HIGH", "reasons": ["earnings tomorrow"]})],
        },
        settings={"risk_per_trade": 500, "max_positions": 6},
    )

    assert decision["action"] == WAIT
    assert "event risk HIGH" in decision["blockers"]


def test_wait_when_backtest_evidence_is_weak():
    decision = build_trade_decision(
        {"regime": "BULL", "vix": 18.0, "stalk_orders": [_order(bt_win_rate=47.0)]},
        settings={"risk_per_trade": 500, "max_positions": 6},
    )

    assert decision["action"] == WAIT
    assert "backtest win rate below 50%" in decision["blockers"]
