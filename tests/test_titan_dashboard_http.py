"""HTTP-level e2e tests for titan_dashboard FastAPI app.

Covers the new risk / sanity / daily-loss / held-position / sector gates
introduced alongside the trailing-stop and earnings-blackout work.

All tests use a fresh tempfile-backed MyPositions tracker and stub the
yfinance live-price fetch so they run offline and deterministic.
"""

import argparse
import asyncio
import os
import sys
import tempfile
from datetime import timezone

from fastapi.testclient import TestClient

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import titan_dashboard as td
from titan.my_positions import MyPositions


# ---------------------------------------------------------------- fixtures
# The dashboard startup handler kicks off a real S&P 500 scan by default.
# For HTTP tests that's too slow and nondeterministic — we disable it by
# replacing `scanner_loop` with a no-op before constructing TestClient, and
# share a single client across all tests.

async def _noop_scan_loop():
    return


td.scanner_loop = _noop_scan_loop  # prevent startup from scanning S&P 500
td._journal = lambda *args, **kwargs: None

# Module-level shared client: one startup / shutdown pair for the whole file.
_CLIENT = TestClient(td.app)
_CLIENT.__enter__()

import atexit
atexit.register(lambda: _CLIENT.__exit__(None, None, None))


def _fresh_tracker():
    """Replace the module-level MyPositions with a temp-backed one. Returns
    the path so the test can clean up.
    """
    fd, path = tempfile.mkstemp(suffix=".json")
    os.close(fd)
    os.unlink(path)
    td.my_positions = MyPositions(path)
    td.state.last_scan = None
    td.state.last_scan_time = td.datetime.now(timezone.utc).isoformat()
    return path


def _stub_live_price(monkeypatch, price):
    """Make _fetch_live_prices return {ticker: price} for any request."""
    def fake(tickers, prefer_live=False):
        return {t: price for t in tickers}
    monkeypatch.setattr(td, "_fetch_live_prices", fake)
    monkeypatch.setattr(
        td,
        "_fetch_intraday_snapshots",
        lambda tickers: {
            t: {
                "price": price,
                "bars": 10,
                "age_sec": 60,
                "day_open": price,
                "day_high": price,
                "day_low": price,
            }
            for t in tickers
        },
    )
    monkeypatch.setattr(td, "assess_event_risk", lambda *a, **k: {"status": "LOW", "reasons": []})


def _no_live_price(monkeypatch):
    monkeypatch.setattr(td, "_fetch_live_prices", lambda tickers, prefer_live=False: {})
    monkeypatch.setattr(td, "_fetch_intraday_snapshots", lambda tickers: {})
    monkeypatch.setattr(td, "assess_event_risk", lambda *a, **k: {"status": "LOW", "reasons": []})


class _ClientBorrow:
    """Context wrapper so existing `with _mk_client() as c:` call sites work
    against the long-lived shared TestClient without re-entering startup."""
    def __enter__(self):
        return _CLIENT
    def __exit__(self, *a):
        return False


def _mk_client():
    """Borrow the shared, already-started client."""
    return _ClientBorrow()


# ================================================================
# /api/my-positions/validate — preflight check
# ================================================================

def test_dashboard_mutations_require_token_when_configured(monkeypatch):
    path = _fresh_tracker()
    try:
        _stub_live_price(monkeypatch, 100.0)
        monkeypatch.setattr(td, "DASHBOARD_API_TOKEN", "secret-token")
        with _mk_client() as c:
            r = c.post("/api/my-positions", json={
                "ticker": "AAPL",
                "entry_price": 100,
                "shares": 1,
                "stop": 95,
                "force": True,
            })
            assert r.status_code == 401
            r2 = c.post(
                "/api/my-positions",
                headers={"Authorization": "Bearer secret-token"},
                json={
                    "ticker": "AAPL",
                    "entry_price": 100,
                    "shares": 1,
                    "stop": 95,
                    "force": True,
                },
            )
            assert r2.status_code == 200
    finally:
        monkeypatch.setattr(td, "DASHBOARD_API_TOKEN", "")
        if os.path.exists(path):
            os.unlink(path)


def test_dashboard_portfolio_requires_token_when_configured(monkeypatch):
    monkeypatch.setattr(td, "DASHBOARD_API_TOKEN", "secret-token")
    class DummyExecutor:
        def is_connected(self):
            return False

    monkeypatch.setattr(td, "AlpacaExecutor", DummyExecutor)
    try:
        with _mk_client() as c:
            r = c.get("/api/portfolio")
            assert r.status_code == 401
            r2 = c.get("/api/portfolio", headers={"X-Titan-Token": "secret-token"})
            assert r2.status_code == 200
            assert r2.json()["status"] == "disconnected"
    finally:
        monkeypatch.setattr(td, "DASHBOARD_API_TOKEN", "")


def test_dashboard_rejects_bind_all_without_auth(monkeypatch):
    monkeypatch.setattr(td, "DASHBOARD_API_TOKEN", "")
    parser = argparse.ArgumentParser()
    args = argparse.Namespace(host="0.0.0.0", allow_unsafe_no_auth=False)

    try:
        td.validate_dashboard_args(parser, args)
    except SystemExit:
        pass
    else:
        raise AssertionError("Expected parser.error for unauthenticated bind-all")


def test_dashboard_allows_bind_all_with_token(monkeypatch):
    monkeypatch.setattr(td, "DASHBOARD_API_TOKEN", "secret-token")
    parser = argparse.ArgumentParser()
    args = argparse.Namespace(host="0.0.0.0", allow_unsafe_no_auth=False)

    td.validate_dashboard_args(parser, args)
    monkeypatch.setattr(td, "DASHBOARD_API_TOKEN", "")


def test_dashboard_health_reports_non_secret_runtime_state(monkeypatch):
    path = _fresh_tracker()
    try:
        monkeypatch.setenv("APCA_API_KEY_ID", "key")
        monkeypatch.setenv("APCA_API_SECRET_KEY", "secret")
        monkeypatch.setenv("TITAN_ALPACA_USE_PAPER", "true")
        td.state.last_scan_time = "2026-01-01T00:00:00+00:00"

        with _mk_client() as c:
            r = c.get("/api/health")
            assert r.status_code == 200
            body = r.json()
            assert body["status"] == "success"
            assert body["alpaca_keys_configured"] is True
            assert body["broker_mode"] == "paper"
            assert "secret" not in str(body).lower()
            assert "data_dir" in body
    finally:
        td.state.last_scan_time = None
        if os.path.exists(path):
            os.unlink(path)


def test_position_size_calculates_shares_from_risk():
    with _mk_client() as c:
        r = c.get("/api/position-size", params={
            "entry_price": 100,
            "stop": 95,
            "risk_dollars": 500,
        })
        assert r.status_code == 200
        body = r.json()
        assert body["shares"] == 100
        assert body["estimated_risk"] == 500


def test_position_size_rejects_invalid_stop():
    with _mk_client() as c:
        r = c.get("/api/position-size", params={
            "entry_price": 100,
            "stop": 105,
            "risk_dollars": 500,
        })
        assert r.status_code == 400
        assert r.json()["detail"]["kind"] == "SIZE_INPUT_INVALID"


def test_run_scan_lock_prevents_overlap(monkeypatch):
    calls = []

    def fake_scan(settings):
        import time
        calls.append(1)
        time.sleep(0.05)
        return {
            "timestamp": "2026-01-01T00:00:00+00:00",
            "validated": [],
            "active": [],
            "opportunity": [],
            "watchlist": [],
            "stalk_orders": [],
        }

    async def exercise():
        monkeypatch.setattr(td, "_run_scan_sync", fake_scan)
        first = asyncio.create_task(td.run_scan())
        await asyncio.sleep(0)
        second = asyncio.create_task(td.run_scan())
        return await asyncio.gather(first, second)

    result = asyncio.run(exercise())
    assert result == [True, False]
    assert len(calls) == 1


def test_validate_ok_small_position(monkeypatch):
    path = _fresh_tracker()
    try:
        _stub_live_price(monkeypatch, 100.0)
        with _mk_client() as c:
            r = c.get("/api/my-positions/validate",
                      params={"ticker": "X", "entry_price": 100, "shares": 1, "stop": 95})
            assert r.status_code == 200
            body = r.json()
            assert body["status"] == "success"
            assert body["risk"]["level"] == "OK"
            assert body["sanity"]["ok"] is True
            assert body["live_price"] == 100.0
    finally:
        if os.path.exists(path):
            os.unlink(path)


def test_add_position_requires_stop_without_force(monkeypatch):
    path = _fresh_tracker()
    try:
        _stub_live_price(monkeypatch, 100.0)
        with _mk_client() as c:
            r = c.post("/api/my-positions", json={
                "ticker": "AAPL",
                "entry_price": 100,
                "shares": 1,
            })
            assert r.status_code == 400
            assert r.json()["detail"]["kind"] == "PLAN_INVALID"
            r2 = c.post("/api/my-positions", json={
                "ticker": "AAPL",
                "entry_price": 100,
                "shares": 1,
                "force": True,
            })
            assert r2.status_code == 200
    finally:
        if os.path.exists(path):
            os.unlink(path)


def test_add_position_blocks_stale_scan_without_force(monkeypatch):
    path = _fresh_tracker()
    try:
        _stub_live_price(monkeypatch, 100.0)
        td.state.last_scan_time = "2026-01-01T00:00:00+00:00"
        with _mk_client() as c:
            r = c.post("/api/my-positions", json={
                "ticker": "AAPL",
                "entry_price": 100,
                "shares": 1,
                "stop": 95,
            })
            assert r.status_code == 400
            assert r.json()["detail"]["kind"] == "STALE_SCAN"

            r2 = c.post("/api/my-positions", json={
                "ticker": "AAPL",
                "entry_price": 100,
                "shares": 1,
                "stop": 95,
                "force": True,
                "override_reason": "fresh broker fill from manual review",
            })
            assert r2.status_code == 200
    finally:
        if os.path.exists(path):
            os.unlink(path)


def test_add_position_rejects_long_time_stop_without_force(monkeypatch):
    path = _fresh_tracker()
    try:
        _stub_live_price(monkeypatch, 100.0)
        with _mk_client() as c:
            r = c.post("/api/my-positions", json={
                "ticker": "AAPL",
                "entry_price": 100,
                "shares": 1,
                "stop": 95,
                "time_stop_days": 15,
            })
            assert r.status_code == 400
            assert r.json()["detail"]["kind"] == "TIME_STOP_OUT_OF_RANGE"
    finally:
        if os.path.exists(path):
            os.unlink(path)


def test_validate_flags_soft_warn(monkeypatch):
    # Position sized to land between SOFT and HARD
    path = _fresh_tracker()
    try:
        _stub_live_price(monkeypatch, 100.0)
        # account_size is imported at dashboard import time; compute real cap
        from titan.position_risk import HARD_CAP_PCT, SOFT_WARN_PCT
        from titan.config import ACCOUNT_SIZE
        # Target: risk ~= (SOFT+HARD)/2 pct of account
        target_pct = (SOFT_WARN_PCT + HARD_CAP_PCT) / 2.0
        target_risk = ACCOUNT_SIZE * target_pct / 100.0
        shares = max(1, int(target_risk / 5.0))  # stop is $5 below entry

        with _mk_client() as c:
            r = c.get("/api/my-positions/validate",
                      params={"ticker": "X", "entry_price": 100,
                              "shares": shares, "stop": 95})
            body = r.json()
            assert body["risk"]["level"] == "SOFT", f"got {body['risk']['level']}"
    finally:
        if os.path.exists(path):
            os.unlink(path)


def test_validate_flags_hard_cap(monkeypatch):
    path = _fresh_tracker()
    try:
        _stub_live_price(monkeypatch, 100.0)
        from titan.position_risk import HARD_CAP_PCT
        from titan.config import ACCOUNT_SIZE
        # 2x over hard cap
        target_risk = ACCOUNT_SIZE * HARD_CAP_PCT / 100.0 * 2.0
        shares = max(1, int(target_risk / 5.0))

        with _mk_client() as c:
            r = c.get("/api/my-positions/validate",
                      params={"ticker": "X", "entry_price": 100,
                              "shares": shares, "stop": 95})
            body = r.json()
            assert body["risk"]["level"] == "HARD"
            assert body["risk"]["ok"] is False
    finally:
        if os.path.exists(path):
            os.unlink(path)


def test_validate_catches_fat_finger_entry(monkeypatch):
    path = _fresh_tracker()
    try:
        _stub_live_price(monkeypatch, 50.0)  # real price is $50
        with _mk_client() as c:
            r = c.get("/api/my-positions/validate",
                      params={"ticker": "X", "entry_price": 500,  # typo: $500 not $50
                              "shares": 1, "stop": 475})
            body = r.json()
            assert body["sanity"]["ok"] is False
            assert body["sanity"]["level"] == "HARD"
    finally:
        if os.path.exists(path):
            os.unlink(path)


# ================================================================
# POST /api/my-positions — the four gates
# ================================================================

def test_add_position_rejects_hard_cap(monkeypatch):
    path = _fresh_tracker()
    try:
        _no_live_price(monkeypatch)  # no sanity reference
        from titan.position_risk import HARD_CAP_PCT
        from titan.config import ACCOUNT_SIZE
        target_risk = ACCOUNT_SIZE * HARD_CAP_PCT / 100.0 * 3.0
        shares = max(1, int(target_risk / 5.0))

        with _mk_client() as c:
            # Even with force=true, hard cap is refused
            r = c.post("/api/my-positions", json={
                "ticker": "X", "entry_price": 100.0,
                "shares": shares, "stop": 95.0,
                "force": True,
            })
            assert r.status_code == 400
            detail = r.json()["detail"]
            assert detail["kind"] == "RISK_TOO_HIGH"
    finally:
        if os.path.exists(path):
            os.unlink(path)


def test_add_position_soft_warn_blocks_without_force(monkeypatch):
    path = _fresh_tracker()
    try:
        _no_live_price(monkeypatch)
        from titan.position_risk import HARD_CAP_PCT, SOFT_WARN_PCT
        from titan.config import ACCOUNT_SIZE
        target_pct = (SOFT_WARN_PCT + HARD_CAP_PCT) / 2.0
        target_risk = ACCOUNT_SIZE * target_pct / 100.0
        shares = max(1, int(target_risk / 5.0))

        with _mk_client() as c:
            r = c.post("/api/my-positions", json={
                "ticker": "X", "entry_price": 100.0,
                "shares": shares, "stop": 95.0,
            })
            assert r.status_code == 400
            assert r.json()["detail"]["kind"] == "RISK_SOFT_WARN"

            # Force=true lets it through
            r2 = c.post("/api/my-positions", json={
                "ticker": "X", "entry_price": 100.0,
                "shares": shares, "stop": 95.0,
                "force": True,
            })
            assert r2.status_code == 200
            assert r2.json()["status"] == "success"
    finally:
        if os.path.exists(path):
            os.unlink(path)


def test_add_position_rejects_fat_finger_without_force(monkeypatch):
    path = _fresh_tracker()
    try:
        _stub_live_price(monkeypatch, 50.0)
        with _mk_client() as c:
            r = c.post("/api/my-positions", json={
                "ticker": "X", "entry_price": 500.0,  # way off live
                "shares": 1, "stop": 475.0,
            })
            assert r.status_code == 400
            assert r.json()["detail"]["kind"] == "ENTRY_PRICE_DIVERGENT"

            # With force=true the position goes in
            r2 = c.post("/api/my-positions", json={
                "ticker": "X", "entry_price": 500.0,
                "shares": 1, "stop": 475.0,
                "force": True,
            })
            assert r2.status_code == 200
    finally:
        if os.path.exists(path):
            os.unlink(path)


def test_add_position_ok_when_within_all_gates(monkeypatch):
    path = _fresh_tracker()
    try:
        _stub_live_price(monkeypatch, 100.0)
        with _mk_client() as c:
            r = c.post("/api/my-positions", json={
                "ticker": "AAPL", "entry_price": 100.0,
                "shares": 1, "stop": 95.0, "target": 110.0,
            })
            assert r.status_code == 200
            body = r.json()
            assert body["status"] == "success"
            assert body["position"]["ticker"] == "AAPL"
            assert body["risk"]["level"] == "OK"
            assert body["sanity"]["ok"] is True
    finally:
        if os.path.exists(path):
            os.unlink(path)


def test_update_position_rejects_invalid_plan(monkeypatch):
    path = _fresh_tracker()
    try:
        _stub_live_price(monkeypatch, 100.0)
        with _mk_client() as c:
            r = c.post("/api/my-positions", json={
                "ticker": "AAPL", "entry_price": 100.0,
                "shares": 1, "stop": 95.0, "target": 110.0,
            })
            assert r.status_code == 200
            position_id = r.json()["position"]["id"]

            bad_stop = c.patch(f"/api/my-positions/{position_id}", json={"stop": 101.0})
            assert bad_stop.status_code == 400
            assert bad_stop.json()["detail"]["kind"] == "PLAN_INVALID"

            bad_time = c.patch(f"/api/my-positions/{position_id}", json={"time_stop_days": 15})
            assert bad_time.status_code == 400
            assert bad_time.json()["detail"]["kind"] == "TIME_STOP_OUT_OF_RANGE"
    finally:
        if os.path.exists(path):
            os.unlink(path)


def test_daily_loss_cap_blocks_new_entry(monkeypatch):
    """After a large loss today, new buys need force=true."""
    path = _fresh_tracker()
    try:
        _stub_live_price(monkeypatch, 100.0)
        from titan.config import ACCOUNT_SIZE, MAX_DAILY_LOSS_PCT
        from titan.market_time import today_et_str
        today = today_et_str()
        # Seed a closed-today loss large enough to trip the cap
        loss_dollars = ACCOUNT_SIZE * (MAX_DAILY_LOSS_PCT + 1) / 100.0
        # buy 1 share of something at $loss_dollars + 1, sell at $1 (loss = -loss_dollars)
        entry_price = loss_dollars + 10.0
        pos = td.my_positions.add(
            ticker="LOSER", entry_price=entry_price, shares=1,
            stop=entry_price * 0.95, target=entry_price * 1.1,
            entry_date=today,
        )
        td.my_positions.close(pos["id"], exit_price=10.0, exit_date=today)

        with _mk_client() as c:
            r = c.post("/api/my-positions", json={
                "ticker": "X", "entry_price": 100.0,
                "shares": 1, "stop": 95.0,
            })
            assert r.status_code == 400
            assert r.json()["detail"]["kind"] == "DAILY_LOSS_CAP_HIT"

            # Force override works
            r2 = c.post("/api/my-positions", json={
                "ticker": "X", "entry_price": 100.0,
                "shares": 1, "stop": 95.0,
                "force": True,
            })
            assert r2.status_code == 200
    finally:
        if os.path.exists(path):
            os.unlink(path)


def test_add_position_blocks_when_not_latest_buy_decision(monkeypatch):
    path = _fresh_tracker()
    try:
        _stub_live_price(monkeypatch, 100.0)
        td.state.last_scan = {
            "trade_decision": {
                "action": "WAIT",
                "ticker": "AAA",
                "reason": "Best candidate blocked.",
            }
        }
        with _mk_client() as c:
            r = c.post("/api/my-positions", json={
                "ticker": "AAA", "entry_price": 100.0,
                "shares": 1, "stop": 95.0,
            })
            assert r.status_code == 400
            assert r.json()["detail"]["kind"] == "NOT_CURRENT_BUY"

            r2 = c.post("/api/my-positions", json={
                "ticker": "AAA", "entry_price": 100.0,
                "shares": 1, "stop": 95.0,
                "force": True,
            })
            assert r2.status_code == 200
    finally:
        td.state.last_scan = None
        if os.path.exists(path):
            os.unlink(path)


def test_add_position_allows_latest_buy_ticker(monkeypatch):
    path = _fresh_tracker()
    try:
        _stub_live_price(monkeypatch, 100.0)
        td.state.last_scan = {
            "trade_decision": {
                "action": "BUY",
                "ticker": "AAA",
                "reason": "Passed gates.",
            }
        }
        with _mk_client() as c:
            r = c.post("/api/my-positions", json={
                "ticker": "AAA", "entry_price": 100.0,
                "shares": 1, "stop": 95.0,
            })
            assert r.status_code == 200
    finally:
        td.state.last_scan = None
        if os.path.exists(path):
            os.unlink(path)


# ================================================================
# /api/my-positions — listing + summary
# ================================================================

def test_list_positions_includes_daily_loss_fields(monkeypatch):
    path = _fresh_tracker()
    try:
        _stub_live_price(monkeypatch, 100.0)
        with _mk_client() as c:
            c.post("/api/my-positions", json={
                "ticker": "X", "entry_price": 100.0,
                "shares": 1, "stop": 95.0, "target": 110.0,
            })
            r = c.get("/api/my-positions")
            assert r.status_code == 200
            body = r.json()
            # New fields
            assert "daily_realized_dollars" in body
            assert "daily_realized_pct" in body
            assert "daily_loss_cap_pct" in body
            assert "daily_loss_cap_hit" in body
            assert "account_size" in body
            assert body["daily_loss_cap_hit"] is False  # nothing closed yet
            assert len(body["positions"]) == 1
    finally:
        if os.path.exists(path):
            os.unlink(path)


def test_enriched_positions_have_trailing_and_effective_stop(monkeypatch):
    path = _fresh_tracker()
    try:
        _stub_live_price(monkeypatch, 110.0)
        with _mk_client() as c:
            c.post("/api/my-positions", json={
                "ticker": "X", "entry_price": 100.0,
                "shares": 10, "stop": 95.0, "target": 130.0,
            })
            # Simulate the scanner's record_high_water firing at a high print
            # The list endpoint calls _fetch_live_prices (which returns 110)
            # and record_high_water will update highest_since_entry to 110.
            r = c.get("/api/my-positions")
            body = r.json()
            p = body["positions"][0]
            # With current price $110 and entry $100, we have +2R upside →
            # trailing stop should be populated.
            assert "trailing_stop" in p
            assert "effective_stop" in p
            assert p["highest_since_entry"] >= 110.0 - 0.001
    finally:
        if os.path.exists(path):
            os.unlink(path)


# ================================================================
# /api/my-positions/plan/{ticker} — auto-fill helper
# ================================================================

def test_plan_lookup_from_scan():
    path = _fresh_tracker()
    try:
        td.state.last_scan = {
            "validated": [{"ticker": "AAPL", "price": 180.0, "stop": 175.0,
                           "target": 195.0, "time_stop_days": 8,
                           "signal_type": "PULLBACK", "entry_note": "buy dip",
                           "tier": "VALIDATED", "sector": "TECH", "atr": 2.5}],
            "active": [], "opportunity": [], "watchlist": [],
        }
        with _mk_client() as c:
            r = c.get("/api/my-positions/plan/AAPL")
            assert r.status_code == 200
            body = r.json()
            assert body["status"] == "success"
            assert body["plan"]["stop"] == 175.0
            assert body["plan"]["target"] == 195.0

            r2 = c.get("/api/my-positions/plan/ZZZZ")
            assert r2.json()["status"] == "not_found"
    finally:
        td.state.last_scan = None
        if os.path.exists(path):
            os.unlink(path)


# ================================================================
# _augment_with_position_awareness — stalk live validation
# ================================================================

def test_augment_flags_already_held_and_sector(monkeypatch):
    path = _fresh_tracker()
    try:
        _stub_live_price(monkeypatch, 100.0)

        # Plant one open position so it's "held"
        td.my_positions.add("HELD1", 100.0, 1, stop=95.0, target=110.0, sector="TECH")

        export = {
            "validated": [{"ticker": "HELD1", "price": 101.0, "sector": "TECH"}],
            "active": [], "opportunity": [], "watchlist": [],
            "stalk_orders": [
                {"ticker": "HELD1", "limit_price": 101.0, "max_buy_price": 102.0,
                 "buy_zone_low": 99.0, "stop_price": 95.0, "sector": "TECH"},
                {"ticker": "NEWONE", "limit_price": 50.0, "max_buy_price": 51.0,
                 "buy_zone_low": 49.0, "stop_price": 47.0, "sector": "TECH"},  # overlap
                {"ticker": "NEWTWO", "limit_price": 20.0, "max_buy_price": 20.5,
                 "buy_zone_low": 19.5, "stop_price": 18.0, "sector": "HEALTH"},  # clean
            ],
        }
        td._augment_with_position_awareness(export)

        # Validated row is tagged held
        assert export["validated"][0].get("already_held") is True

        # Held-ticker stalk order is flagged
        held = [o for o in export["stalk_orders"] if o["ticker"] == "HELD1"][0]
        assert held.get("already_held") is True

        # Not held but sector overlaps
        newone = [o for o in export["stalk_orders"] if o["ticker"] == "NEWONE"][0]
        assert newone.get("sector_concentration") is True
        assert newone.get("sector_concentration_count", 0) >= 1

        # Different sector, no flag
        newtwo = [o for o in export["stalk_orders"] if o["ticker"] == "NEWTWO"][0]
        assert newtwo.get("sector_concentration") is not True

        # Daily-loss fields populated
        assert "daily_realized_dollars" in export
        assert "daily_loss_cap_hit" in export
        assert export["held_tickers"] == ["HELD1"]
    finally:
        if os.path.exists(path):
            os.unlink(path)


def test_stalk_live_status_in_zone(monkeypatch):
    path = _fresh_tracker()
    try:
        # Live at 100, max_buy 102, zone_low 99 → IN_ZONE
        _stub_live_price(monkeypatch, 100.0)
        export = {
            "validated": [], "active": [], "opportunity": [], "watchlist": [],
            "stalk_orders": [
                {"ticker": "X", "limit_price": 100.0, "max_buy_price": 102.0,
                 "buy_zone_low": 99.0, "stop_price": 95.0, "sector": "TECH"},
            ],
        }
        td._augment_with_position_awareness(export)
        order = export["stalk_orders"][0]
        assert order["live_status"] == "IN_ZONE"
        assert order["live_price"] == 100.0
    finally:
        if os.path.exists(path):
            os.unlink(path)


def test_stalk_live_status_past_max_buy(monkeypatch):
    path = _fresh_tracker()
    try:
        _stub_live_price(monkeypatch, 108.0)  # way above max
        export = {
            "validated": [], "active": [], "opportunity": [], "watchlist": [],
            "stalk_orders": [
                {"ticker": "X", "limit_price": 100.0, "max_buy_price": 102.0,
                 "buy_zone_low": 99.0, "stop_price": 95.0, "sector": "TECH"},
            ],
        }
        td._augment_with_position_awareness(export)
        assert export["stalk_orders"][0]["live_status"] == "PAST_MAX_BUY"


    finally:
        if os.path.exists(path):
            os.unlink(path)


def test_stalk_live_status_broken(monkeypatch):
    path = _fresh_tracker()
    try:
        _stub_live_price(monkeypatch, 94.0)  # below stop
        export = {
            "validated": [], "active": [], "opportunity": [], "watchlist": [],
            "stalk_orders": [
                {"ticker": "X", "limit_price": 100.0, "max_buy_price": 102.0,
                 "buy_zone_low": 99.0, "stop_price": 95.0, "sector": "TECH"},
            ],
        }
        td._augment_with_position_awareness(export)
        assert export["stalk_orders"][0]["live_status"] == "BROKEN"
    finally:
        if os.path.exists(path):
            os.unlink(path)


def test_stalk_live_status_below_zone(monkeypatch):
    path = _fresh_tracker()
    try:
        # Live 96 = below zone_low * 0.97 = 96.03. Tight — use 95.5 instead.
        _stub_live_price(monkeypatch, 95.5)  # below zone_low * 0.97 (=96.03) AND above stop (=95)
        export = {
            "validated": [], "active": [], "opportunity": [], "watchlist": [],
            "stalk_orders": [
                {"ticker": "X", "limit_price": 100.0, "max_buy_price": 102.0,
                 "buy_zone_low": 99.0, "stop_price": 94.0, "sector": "TECH"},
            ],
        }
        td._augment_with_position_awareness(export)
        assert export["stalk_orders"][0]["live_status"] == "BELOW_ZONE"
    finally:
        if os.path.exists(path):
            os.unlink(path)


def test_stalk_live_status_unknown_when_no_live(monkeypatch):
    path = _fresh_tracker()
    try:
        _no_live_price(monkeypatch)
        export = {
            "validated": [], "active": [], "opportunity": [], "watchlist": [],
            "stalk_orders": [
                {"ticker": "X", "limit_price": 100.0, "max_buy_price": 102.0,
                 "buy_zone_low": 99.0, "stop_price": 95.0, "sector": "TECH"},
            ],
        }
        td._augment_with_position_awareness(export)
        assert export["stalk_orders"][0]["live_status"] == "UNKNOWN"
    finally:
        if os.path.exists(path):
            os.unlink(path)


if __name__ == "__main__":
    import inspect
    tests = [v for k, v in globals().items()
             if k.startswith("test_") and callable(v)]

    # Minimal pytest.monkeypatch stub so __main__ mode works without pytest
    class _MP:
        def __init__(self):
            self._undo = []
        def setattr(self, target, name, value=None):
            # Support monkeypatch.setattr(obj, name, value) or (target_str, value)
            if value is None:
                # Single-arg form not used in our tests
                raise NotImplementedError
            old = getattr(target, name)
            self._undo.append((target, name, old))
            setattr(target, name, value)
        def undo(self):
            for target, name, old in reversed(self._undo):
                setattr(target, name, old)

    passed = 0
    failed = []
    for t in tests:
        mp = _MP()
        try:
            sig = inspect.signature(t)
            kwargs = {}
            if "monkeypatch" in sig.parameters:
                kwargs["monkeypatch"] = mp
            t(**kwargs)
            print(f"  PASS  {t.__name__}")
            passed += 1
        except Exception as e:
            import traceback
            print(f"  FAIL  {t.__name__}: {e}")
            traceback.print_exc()
            failed.append(t.__name__)
        finally:
            mp.undo()
    print(f"\n{passed}/{len(tests)} passed")
    if failed:
        print(f"Failed: {failed}")
        sys.exit(1)
