import pytest
import pandas as pd
from types import SimpleNamespace

import titan_trade_v3 as mod


def test_validate_cli_args_rejects_live_without_execute():
    parser = mod.build_arg_parser()
    args = parser.parse_args(["--trust-mode", "--live-orders"])

    with pytest.raises(SystemExit):
        mod.validate_cli_args(parser, args)


def test_validate_cli_args_rejects_live_with_trust_paper():
    parser = mod.build_arg_parser()
    args = parser.parse_args(["--trust-paper", "--execute-orders", "--live-orders"])

    with pytest.raises(SystemExit):
        mod.validate_cli_args(parser, args)


def test_validate_cli_args_allows_explicit_live_mode():
    parser = mod.build_arg_parser()
    args = parser.parse_args(["--trust-mode", "--execute-orders", "--live-orders"])

    mod.validate_cli_args(parser, args)


def test_build_broker_executor_uses_explicit_mode(monkeypatch):
    seen = []

    class DummyExecutor:
        def __init__(self, use_paper=None):
            seen.append(use_paper)

    monkeypatch.setattr(mod, "AlpacaExecutor", DummyExecutor)

    assert mod.build_broker_executor(False, live_orders=False) is None
    mod.build_broker_executor(True, live_orders=False)
    mod.build_broker_executor(True, live_orders=True)

    assert seen == [True, False]


def test_submit_setup_order_always_uses_bracket_order():
    calls = []

    class DummyExecutor:
        def is_connected(self):
            return True

        def submit_bracket_order(self, **kwargs):
            calls.append(kwargs)
            return True

    setup = SimpleNamespace(ticker="AAPL", trigger=100.0, target=110.0, stop=95.0)

    assert mod.submit_setup_order(DummyExecutor(), setup, 3, is_managed=True) is True
    assert calls == [
        {
            "symbol": "AAPL",
            "qty": 3,
            "entry_price": 100.0,
            "target_price": 110.0,
            "stop_price": 95.0,
        }
    ]


def test_execution_skip_reason_blocks_duplicate_exposure():
    setup = SimpleNamespace(ticker="AAPL")

    assert mod.execution_skip_reason(setup, {"AAPL": {}}, set(), {}) == "already in open positions"
    assert mod.execution_skip_reason(setup, {}, {"AAPL"}, {}) == "already has an open order"
    assert mod.execution_skip_reason(setup, {}, set(), {"AAPL": {}}) == "already managed in portfolio"
    assert mod.execution_skip_reason(setup, {}, set(), {}) is None


def test_record_submitted_setup_updates_exposure_guards():
    setup = SimpleNamespace(ticker="AAPL", trigger=100.0, stop=95.0)
    live_positions = {}
    open_orders = set()

    mod.record_submitted_setup(setup, 4, live_positions, open_orders)

    assert live_positions == {"AAPL": {"entry_price": 100.0, "stop_loss": 95.0, "shares": 4}}
    assert open_orders == {"AAPL"}


def test_manage_open_positions_does_not_assume_partial_fill(monkeypatch):
    portfolio = {
        "AAPL": {
            "ticker": "AAPL",
            "status": "OPEN",
            "entry_price": 100.0,
            "shares": 10,
            "stop_loss": 95.0,
            "target": 120.0,
            "partial_target": 105.0,
            "partial_taken": False,
        }
    }
    saved = {}

    class DummyExecutor:
        def __init__(self):
            self.cancelled = []
            self.market_orders = []
            self.stop_orders = []

        def is_connected(self):
            return True

        def get_open_positions_snapshot(self):
            return {"AAPL": {"shares": 10, "entry_price": 100.0}}

        def get_open_orders(self, *args, **kwargs):
            if kwargs.get("symbol") == "AAPL" and kwargs.get("side") == "sell":
                return [{"symbol": "AAPL", "side": "sell", "type": "stop", "qty": 10, "stop_price": 95.0}]
            return []

        def cancel_orders_for_symbol(self, *args, **kwargs):
            self.cancelled.append((args, kwargs))

        def submit_market_order(self, *args, **kwargs):
            self.market_orders.append((args, kwargs))
            return True

        def submit_stop_order(self, *args, **kwargs):
            self.stop_orders.append((args, kwargs))
            return True

    class DummyMarketRegime:
        def __init__(self, data):
            pass

        def analyze_spy(self):
            return "BULL", 1.0, None

    executor = DummyExecutor()
    monkeypatch.setattr(mod, "load_managed_portfolio", lambda logger=None: portfolio.copy())
    monkeypatch.setattr(mod, "save_managed_portfolio", lambda p, logger=None: saved.update(p))
    monkeypatch.setattr(mod, "build_price_lookup", lambda data, symbols: {"AAPL": 106.0})
    monkeypatch.setattr(mod, "MarketRegime", DummyMarketRegime)
    monkeypatch.setattr(mod.MarketHours, "is_market_open", staticmethod(lambda: True))

    data = pd.DataFrame({"Close": [100.0]})
    updated, actions = mod.manage_open_positions(executor, data, {}, logger=None)

    assert executor.cancelled == []
    assert executor.market_orders == []
    assert updated["AAPL"]["partial_taken"] is False
    assert updated["AAPL"]["shares"] == 10
    assert any("auto partial exit skipped" in action for action in actions)
