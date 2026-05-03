import logging
from types import SimpleNamespace

import titan_trade_v3 as mod


def test_manual_mode_honors_disabled_oos_and_walkforward(monkeypatch):
    monkeypatch.setattr(
        mod,
        "load_json_file",
        lambda path, logger=None, **kwargs: {
            "require_oos": False,
            "require_walkforward": False,
        },
    )

    args = SimpleNamespace(
        trust_mode=False,
        trust_paper=False,
        account_size=None,
        risk_per_trade=None,
    )
    settings = mod.build_runtime_settings(args, None, logging.getLogger("test"))

    assert settings["require_oos"] is False
    assert settings["require_walkforward"] is False


def test_should_build_watchlist_when_prime_list_is_thin():
    settings = {
        "build_watchlist": True,
        "always_build_watchlist": False,
        "watchlist_if_fewer_than": 3,
    }

    assert mod.should_build_watchlist([1], settings) is True
    assert mod.should_build_watchlist([1, 2, 3], settings) is False
    assert mod.should_build_watchlist([], {**settings, "build_watchlist": False}) is False


def test_build_runtime_settings_populates_pilot_breakout_defaults(monkeypatch):
    monkeypatch.setattr(mod, "load_json_file", lambda path, logger=None, **kwargs: {})

    args = SimpleNamespace(
        trust_mode=False,
        trust_paper=False,
        account_size=None,
        risk_per_trade=None,
    )
    settings = mod.build_runtime_settings(args, None, logging.getLogger("test"))

    assert settings["enable_pilot_breakouts"] is True
    assert settings["pilot_breakout_min_trades"] == 2
    assert settings["pilot_breakout_size_scalar"] == 0.5
    assert settings["pilot_breakout_min_rr"] == 1.1


def test_load_json_file_strict_rejects_bad_config(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text("{not-json", encoding="utf-8")

    try:
        mod.load_json_file(str(bad), strict=True)
    except ValueError as exc:
        assert "Failed to read" in str(exc)
    else:
        raise AssertionError("Expected strict config load to fail")
