import pytest

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
