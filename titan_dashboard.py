"""
Titan Pro Scanner -- Live Dashboard
====================================
Real-time S&P 500 scanner dashboard powered by the Pro Scanner
(KOSPI-architecture multi-signal detection).

Usage:
    python titan_dashboard.py                    # default 15-min scan interval
    python titan_dashboard.py --interval 5       # 5-min interval
    python titan_dashboard.py --port 8080        # custom port
    python titan_dashboard.py --host 0.0.0.0     # bind to all interfaces
"""

import asyncio
import argparse
import json
import logging
import os
import sys
import tempfile
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Import pro scanner + v3 utilities
# ---------------------------------------------------------------------------
from titan.pro_scanner import pro_scan
from titan.alpaca_executor import AlpacaExecutor
from titan.my_positions import MyPositions, DEFAULT_TIME_STOP_DAYS
from titan.market_time import market_session_et, today_et_str
from titan.position_risk import (
    validate_position_risk,
    validate_entry_price,
    SANITY_PRICE_DIVERGENCE,
    HARD_CAP_PCT,
    SOFT_WARN_PCT,
)
from titan.swing_score import rank_stalk_orders, top_picks, TOP_PICK_MIN
from titan.decision import build_trade_decision
from titan.event_risk import assess_event_risk
from titan.config import ACCOUNT_SIZE, MAX_DAILY_LOSS_PCT

# Earnings checker — wrap yfinance behind a safe shim (the dashboard degrades
# gracefully if the calendar module is unavailable).
_earnings_import_error: Optional[str] = None
try:
    from titan.market import EarningsCalendar
    def _earnings_days_until(ticker: str) -> Optional[int]:
        try:
            _, days = EarningsCalendar.get_earnings_date(ticker)
            return days
        except Exception:
            return None
except Exception as exc:  # pragma: no cover
    _earnings_import_error = str(exc)
    def _earnings_days_until(ticker: str) -> Optional[int]:
        return None

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("titan-dashboard")

if _earnings_import_error:
    logger.warning(f"EarningsCalendar unavailable: {_earnings_import_error}")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
APP_DIR = Path(__file__).resolve().parent
STATIC_DIR = APP_DIR / "static"
SCAN_FILE = APP_DIR / "data" / "latest_scan.json"

# ---------------------------------------------------------------------------
# App state
# ---------------------------------------------------------------------------
app = FastAPI(title="Titan Trade Live Dashboard")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


class DashboardState:
    """Mutable singleton holding the latest scan + metadata."""

    def __init__(self):
        self.last_scan: Optional[dict] = None
        self.last_scan_time: Optional[str] = None
        self.scan_count: int = 0
        self.is_scanning: bool = False
        self.scan_error: Optional[str] = None
        self.scan_interval_min: int = 15
        self.next_scan_time: Optional[str] = None
        self.scan_history: List[dict] = []      # last N scan summaries
        self.connected_clients: List[WebSocket] = []
        self.settings: dict = {}

    def snapshot(self) -> dict:
        """JSON-safe state summary pushed to every client."""
        return {
            "type": "state",
            "is_scanning": self.is_scanning,
            "last_scan_time": self.last_scan_time,
            "next_scan_time": self.next_scan_time,
            "scan_count": self.scan_count,
            "scan_error": self.scan_error,
            "scan_interval_min": self.scan_interval_min,
            "connected_clients": len(self.connected_clients),
        }


state = DashboardState()

# ---------------------------------------------------------------------------
# Manual position tracker — user trades manually, this tracks what they hold
# ---------------------------------------------------------------------------
my_positions = MyPositions()

# Live-price cache: {ticker: (price, timestamp)} — 60-second TTL to avoid
# hammering yfinance when the user refreshes the positions panel. Locked because
# concurrent HTTP handlers + the scan loop can all invalidate/read simultaneously.
_price_cache: Dict[str, tuple] = {}
_price_cache_lock = __import__("threading").Lock()
_PRICE_TTL_SEC = 60.0


def _price_cache_clear() -> None:
    with _price_cache_lock:
        _price_cache.clear()


def _lookup_price_in_scan(ticker: str) -> Optional[float]:
    """Pull the ticker's last close from the latest scan, if present."""
    scan = state.last_scan
    if not scan:
        return None
    for key in ("validated", "active", "opportunity", "watchlist"):
        for row in scan.get(key) or []:
            if row.get("ticker") == ticker:
                p = row.get("price")
                try:
                    p = float(p)
                    if p > 0:
                        return p
                except (TypeError, ValueError):
                    pass
    return None


def _lookup_row_in_scan(ticker: str) -> Optional[dict]:
    """Return the full scan row for a ticker (across all tiers), if any."""
    scan = state.last_scan
    if not scan:
        return None
    for key in ("validated", "active", "opportunity", "watchlist"):
        for row in scan.get(key) or []:
            if row.get("ticker") == ticker:
                return row
    return None


def _lookup_plan_in_scan(ticker: str) -> Optional[dict]:
    """Return the scanner's stop/target/time-stop for a ticker, if any, from the
    latest scan. Lets the UI auto-fill the plan when the user adds a position.
    """
    scan = state.last_scan
    if not scan:
        return None
    for key in ("validated", "active", "opportunity", "watchlist"):
        for row in scan.get(key) or []:
            if row.get("ticker") == ticker:
                return {
                    "price": row.get("price"),
                    "stop": row.get("stop"),
                    "target": row.get("target"),
                    "time_stop_days": row.get("time_stop_days"),
                    "signal_type": row.get("signal_type"),
                    "entry_note": row.get("entry_note"),
                    "tier": row.get("tier"),
                }
    return None


def _fetch_live_prices(
    tickers: List[str], prefer_live: bool = False
) -> Dict[str, float]:
    """Fetch last prices for a list of tickers, respecting the 60s cache.

    Order of resolution per ticker:
      1. Cache (< 60s old).
      2. If `prefer_live` is False: most recent close from the latest scan.
      3. yfinance 1-minute intraday fetch.

    `prefer_live=True` forces a fresh yfinance fetch — use it when validating
    a brand-new position or a stalk order against the opening print, where
    yesterday's close is not enough.
    """
    if not tickers:
        return {}
    now = time.time()
    prices: Dict[str, float] = {}
    to_fetch: List[str] = []

    with _price_cache_lock:
        for t in tickers:
            cached = _price_cache.get(t)
            if cached and (now - cached[1]) < _PRICE_TTL_SEC:
                prices[t] = cached[0]
                continue
            if not prefer_live:
                scan_price = _lookup_price_in_scan(t)
                if scan_price is not None:
                    prices[t] = scan_price
                    _price_cache[t] = (scan_price, now)
                    continue
            to_fetch.append(t)

    if to_fetch:
        try:
            import yfinance as yf
            data = yf.download(
                " ".join(to_fetch), period="1d", interval="1m",
                progress=False, threads=True, auto_adjust=False,
            )
            if data is not None and not data.empty:
                fresh: Dict[str, float] = {}
                if len(to_fetch) == 1:
                    t = to_fetch[0]
                    close = data.get("Close")
                    if close is not None and not close.empty:
                        last_series = close.dropna() if hasattr(close, "dropna") else None
                        if last_series is not None and not last_series.empty:
                            last = float(last_series.iloc[-1])
                            if last > 0:
                                fresh[t] = last
                else:
                    close_block = data.get("Close")
                    if close_block is not None:
                        for t in to_fetch:
                            if t in close_block.columns:
                                series = close_block[t].dropna()
                                if not series.empty:
                                    last = float(series.iloc[-1])
                                    if last > 0:
                                        fresh[t] = last
                if fresh:
                    with _price_cache_lock:
                        for t, last in fresh.items():
                            prices[t] = last
                            _price_cache[t] = (last, now)
        except Exception as exc:
            logger.warning(f"yfinance live-price fetch failed: {exc}")

    # For tickers that still have no price and prefer_live=True, fall back to
    # the scanner's close so the UI doesn't show blank.
    if prefer_live:
        missing = [t for t in tickers if t not in prices]
        for t in missing:
            scan_price = _lookup_price_in_scan(t)
            if scan_price is not None:
                prices[t] = scan_price
    return prices


def _fetch_intraday_snapshots(tickers: List[str]) -> Dict[str, dict]:
    """Fetch 1-minute intraday context for entry validation.

    This is intentionally separate from `_fetch_live_prices`: price can be a
    fallback, but intraday confirmation must come from actual 1-minute bars.
    """
    if not tickers:
        return {}
    try:
        import yfinance as yf

        data = yf.download(
            " ".join(tickers),
            period="1d",
            interval="1m",
            progress=False,
            threads=True,
            auto_adjust=False,
        )
    except Exception as exc:
        logger.warning(f"yfinance intraday fetch failed: {exc}")
        return {}

    if data is None or data.empty:
        return {}

    def close_block_for(ticker: str):
        if len(tickers) == 1:
            return data
        try:
            if hasattr(data.columns, "nlevels") and data.columns.nlevels > 1:
                if ticker in data.columns.get_level_values(1):
                    return data.xs(ticker, level=1, axis=1)
                if ticker in data.columns.get_level_values(0):
                    return data[ticker]
        except Exception:
            return None
        return None

    now_ts = datetime.now(timezone.utc).timestamp()
    snapshots: Dict[str, dict] = {}
    for ticker in tickers:
        block = close_block_for(ticker)
        if block is None or block.empty or "Close" not in block:
            continue
        close = block["Close"].dropna()
        if close.empty:
            continue
        try:
            last_price = float(close.iloc[-1])
            if last_price <= 0:
                continue
            idx = close.index[-1]
            ts = idx.to_pydatetime().timestamp() if hasattr(idx, "to_pydatetime") else now_ts
            if ts > 10_000_000_000:  # milliseconds defensive guard
                ts = ts / 1000.0
            day_open = float(block["Open"].dropna().iloc[0]) if "Open" in block and not block["Open"].dropna().empty else last_price
            day_high = float(block["High"].dropna().max()) if "High" in block and not block["High"].dropna().empty else last_price
            day_low = float(block["Low"].dropna().min()) if "Low" in block and not block["Low"].dropna().empty else last_price
            snapshots[ticker] = {
                "price": round(last_price, 4),
                "bars": int(len(close)),
                "asof": datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                "age_sec": max(0, int(now_ts - ts)),
                "day_open": round(day_open, 4),
                "day_high": round(day_high, 4),
                "day_low": round(day_low, 4),
            }
        except Exception:
            continue
    return snapshots


def _intraday_status(order: dict, snapshot: Optional[dict], session: str) -> str:
    if session == "PRE_MARKET":
        return "PREMARKET_UNCONFIRMED"
    if session != "REGULAR":
        return "NOT_REQUIRED"
    if not snapshot:
        return "NO_INTRADAY_CONFIRMATION"
    if int(snapshot.get("bars") or 0) < 3:
        return "NO_INTRADAY_CONFIRMATION"
    if int(snapshot.get("age_sec") or 999999) > 900:
        return "STALE_INTRADAY"

    price = float(snapshot.get("price") or 0)
    day_open = float(snapshot.get("day_open") or price)
    max_buy = float(order.get("max_buy_price") or order.get("limit_price") or 0)
    buy_zone_low = float(order.get("buy_zone_low") or 0)
    stop_price = float(order.get("stop_price") or order.get("stop") or 0)

    if stop_price > 0 and day_open <= stop_price:
        return "GAP_DOWN_BROKEN"
    if max_buy > 0 and day_open > max_buy * 1.005:
        return "GAP_UP_CHASE"
    if max_buy > 0 and price > max_buy * 1.005:
        return "GAP_UP_CHASE"
    if buy_zone_low > 0 and price < buy_zone_low * 0.97:
        return "WAIT_INTRADAY"
    if max_buy > 0 and (buy_zone_low <= 0 or price >= buy_zone_low) and price <= max_buy * 1.005:
        return "CONFIRMED"
    return "WAIT_INTRADAY"


# ---------------------------------------------------------------------------
# WebSocket manager
# ---------------------------------------------------------------------------
async def broadcast(message: dict):
    """Send a JSON message to every connected WebSocket client."""
    dead = []
    payload = json.dumps(message, default=str)
    for ws in state.connected_clients:
        try:
            await ws.send_text(payload)
        except Exception:
            dead.append(ws)
    for ws in dead:
        state.connected_clients.remove(ws)


# ---------------------------------------------------------------------------
# Scanner – runs pro_scan() in a thread so it doesn't block the event loop
# ---------------------------------------------------------------------------
def _run_scan_sync(settings: dict) -> dict:
    """Execute a pro scan (blocking). Returns dashboard-ready dict."""
    result = pro_scan(settings=settings)

    # Flatten for dashboard consumption
    export = {
        "mode": "pro",
        "regime": result["regime"],
        "regime_score": result["regime_score"],
        "vix": result["vix"],
        "adx": result["adx"],
        "market_score": result["market_score"],
        "top_sectors": result["top_sectors"],
        "validated": result["validated"],
        "active": result["active"],
        "opportunity": result["opportunity"],
        "watchlist": result["watchlist"],
        "predictive_radar": result.get("predictive_radar", []),
        "stalk_orders": result["stalk_orders"],
        "scan_time": result["scan_time"],
        "total_scanned": result["total_scanned"],
        "total_signals": result["total_signals"],
        "total_predictive": result.get("total_predictive", 0),
        "timestamp": result["scan_timestamp"],
        "market_session": market_session_et(),
    }

    try:
        _augment_with_position_awareness(export)
    except Exception as exc:
        logger.error(f"Position-aware augmentation failed: {exc}\n{traceback.format_exc()}")

    # Persist to disk (atomic write)
    os.makedirs("data", exist_ok=True)
    tmp_fd, tmp_path = tempfile.mkstemp(dir=str(APP_DIR / "data"), suffix=".tmp")
    try:
        with os.fdopen(tmp_fd, "w") as f:
            json.dump(export, f, indent=2, default=str)
        os.replace(tmp_path, SCAN_FILE)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise
    return export


def _augment_with_position_awareness(export: dict) -> None:
    """Tag scan rows with held-position context, live-price validate stalk
    orders, and attach daily-loss / sector concentration metadata.

    Mutates `export` in place. Any failure here must not corrupt the scan.
    """
    # 1. Index held positions and their sectors
    open_positions = my_positions.list_open()
    held_tickers = {p["ticker"] for p in open_positions}
    held_by_ticker = {p["ticker"]: p for p in open_positions}

    # Pull sector for held tickers from the scan if available (when we added a
    # position via the dashboard we stored sector, but older positions may not
    # have it). Fall back to None.
    def _sector_for_ticker(t: str) -> Optional[str]:
        p = held_by_ticker.get(t)
        if p and p.get("sector"):
            return p["sector"]
        for key in ("validated", "active", "opportunity", "watchlist"):
            for row in export.get(key) or []:
                if row.get("ticker") == t:
                    return row.get("sector")
        return None

    held_sectors: Dict[str, int] = {}
    for t in held_tickers:
        sec = _sector_for_ticker(t)
        if sec:
            held_sectors[sec] = held_sectors.get(sec, 0) + 1

    # 2. Tag every signal row
    for key in ("validated", "active", "opportunity", "watchlist"):
        for row in export.get(key) or []:
            tk = row.get("ticker")
            if tk and tk in held_tickers:
                row["already_held"] = True

    # 3. Live-price validate stalk orders
    stalk_orders = export.get("stalk_orders") or []
    if stalk_orders:
        stalk_tickers = sorted({o.get("ticker") for o in stalk_orders if o.get("ticker")})
        # Prefer live prices during regular session (open vs last close matters most then)
        session = market_session_et()
        prefer_live = session in ("PRE_MARKET", "REGULAR")
        live = _fetch_live_prices(stalk_tickers, prefer_live=prefer_live) if stalk_tickers else {}
        intraday = (
            _fetch_intraday_snapshots(stalk_tickers)
            if session in ("PRE_MARKET", "REGULAR") and stalk_tickers
            else {}
        )

        def _row_lookup_for_event(tk: str) -> Optional[dict]:
            for key in ("validated", "active", "opportunity", "watchlist"):
                for row in export.get(key) or []:
                    if row.get("ticker") == tk:
                        return row
            return None

        def _earnings_days_from_row(row: Optional[dict]) -> Optional[int]:
            call = str((row or {}).get("earnings_call") or "")
            if not call or call == "Unknown":
                return None
            try:
                return int(call.replace("d", ""))
            except ValueError:
                return None
            return None

        for order in stalk_orders:
            tk = order.get("ticker")
            if not tk:
                continue

            # Held flag
            if tk in held_tickers:
                order["already_held"] = True

            # Sector overlap (warning, not blocker)
            sec = order.get("sector")
            if sec and held_sectors.get(sec, 0) >= 1 and tk not in held_tickers:
                order["sector_concentration"] = True
                order["sector_concentration_count"] = held_sectors[sec]

            row = _row_lookup_for_event(tk)
            order["event_risk"] = assess_event_risk(
                tk,
                earnings_days=_earnings_days_from_row(row),
            )

            lp = live.get(tk)
            if lp is None or lp <= 0:
                order["live_status"] = "UNKNOWN"
                order["live_note"] = "Live price unavailable."
            else:
                order["live_price"] = round(float(lp), 2)
                max_buy = float(order.get("max_buy_price") or order.get("limit_price") or 0)
                buy_zone_low = float(order.get("buy_zone_low") or 0)
                stop_price = float(order.get("stop_price") or order.get("stop") or 0)

                if stop_price > 0 and lp <= stop_price:
                    order["live_status"] = "BROKEN"
                    order["live_note"] = f"Live ${lp:.2f} already below stop ${stop_price:.2f} — setup invalidated"
                elif max_buy > 0 and lp > max_buy * 1.005:
                    order["live_status"] = "PAST_MAX_BUY"
                    order["live_note"] = (
                        f"Live ${lp:.2f} above max buy ${max_buy:.2f} — do NOT chase"
                    )
                elif buy_zone_low > 0 and lp < buy_zone_low * 0.97:
                    order["live_status"] = "BELOW_ZONE"
                    order["live_note"] = (
                        f"Live ${lp:.2f} below buy zone ${buy_zone_low:.2f} — wait for reclaim"
                    )
                else:
                    order["live_status"] = "IN_ZONE"
                    order["live_note"] = f"Live ${lp:.2f} inside plan."

            snap = intraday.get(tk)
            if snap:
                order["intraday"] = snap
            order["intraday_status"] = _intraday_status(order, snap, session)

    # 4. Swing-trade scoring on stalk orders → rank + surface top picks.
    # Rank AFTER live-status tagging so UI badges and ordering are consistent.
    if stalk_orders:
        def _row_lookup(tk: str):
            for key in ("validated", "active", "opportunity", "watchlist"):
                for row in export.get(key) or []:
                    if row.get("ticker") == tk:
                        return row
            return None
        live_price_map = {
            o["ticker"]: o["live_price"]
            for o in stalk_orders
            if o.get("ticker") and o.get("live_price") is not None
        }
        try:
            rank_stalk_orders(stalk_orders, row_lookup=_row_lookup, live_prices=live_price_map)
            # Penalize BROKEN / PAST_MAX_BUY in the final ordering so users see
            # the best *actionable* setups first. We do this as a post-sort
            # bucketing pass rather than baking it into the score so the raw
            # swing score still reflects the setup's intrinsic quality.
            def _live_bucket(o):
                if o.get("already_held"):
                    return 4
                st = o.get("live_status")
                if st == "BROKEN":
                    return 3
                if st == "PAST_MAX_BUY":
                    return 2
                if st == "BELOW_ZONE":
                    return 1
                return 0
            stalk_orders.sort(
                key=lambda x: (
                    _live_bucket(x),
                    -int((x.get("swing_score") or {}).get("score") or 0),
                    -float(x.get("trade_score") or 0),
                    x.get("ticker") or "",
                )
            )
            export["stalk_orders"] = stalk_orders
            picks = top_picks(stalk_orders, limit=5, min_score=TOP_PICK_MIN)
            # Only surface as TOP PICK if it's also actionable (IN_ZONE / UNKNOWN).
            picks = [
                p for p in picks
                if not p.get("already_held")
                and p.get("live_status") not in ("BROKEN", "PAST_MAX_BUY")
            ]
            export["top_picks"] = picks
            for p in picks:
                p["is_top_pick"] = True
        except Exception as exc:
            logger.error(f"Swing-scoring failed: {exc}\n{traceback.format_exc()}")
            export.setdefault("top_picks", [])
    else:
        export["top_picks"] = []

    # 5. Daily realized P&L (from closed positions exited today)
    summary = my_positions.summary()
    realized_today = summary.get("realized_today_dollars", 0.0) or 0.0
    daily_pnl_pct = (realized_today / ACCOUNT_SIZE * 100.0) if ACCOUNT_SIZE > 0 else 0.0
    export["daily_realized_dollars"] = round(realized_today, 2)
    export["daily_realized_pct"] = round(daily_pnl_pct, 3)
    export["daily_loss_cap_pct"] = MAX_DAILY_LOSS_PCT
    export["daily_loss_cap_hit"] = daily_pnl_pct <= -MAX_DAILY_LOSS_PCT
    export["held_tickers"] = sorted(held_tickers)
    export["held_sector_counts"] = held_sectors
    export["trade_decision"] = build_trade_decision(
        export,
        open_positions=open_positions,
        settings=state.settings,
    )


async def run_scan():
    """Async wrapper – runs the blocking scan in a thread pool."""
    state.is_scanning = True
    state.scan_error = None
    await broadcast(state.snapshot())

    try:
        loop = asyncio.get_event_loop()
        settings = state.settings or {}
        result = await loop.run_in_executor(None, _run_scan_sync, settings)

        state.last_scan = result
        state.last_scan_time = result.get("timestamp", datetime.now(timezone.utc).isoformat())
        state.scan_count += 1

        # Keep last 50 scan summaries
        n_validated = len(result.get("validated", []))
        n_active = len(result.get("active", []))
        n_opportunity = len(result.get("opportunity", []))
        n_watchlist = len(result.get("watchlist", []))
        state.scan_history.append({
            "time": state.last_scan_time,
            "validated": n_validated,
            "active": n_active,
            "opportunity": n_opportunity,
            "watchlist": n_watchlist,
            "regime": result.get("regime", "UNKNOWN"),
            "vix": result.get("vix"),
        })
        if len(state.scan_history) > 50:
            state.scan_history = state.scan_history[-50:]

        logger.info(
            f"Scan #{state.scan_count} complete: "
            f"{n_validated} validated, {n_active} active, "
            f"{n_opportunity} opportunity, {n_watchlist} watchlist"
        )

        # Push full results to all clients
        await broadcast({"type": "scan_result", "data": result})

        # Fresh prices landed — refresh My Positions alerts on the UI
        # (invalidate the price cache so the next /api/my-positions call is fresh)
        _price_cache_clear()
        await broadcast({"type": "my_positions_changed", "reason": "scan"})

    except Exception as exc:
        state.scan_error = str(exc)
        logger.error(f"Scan failed: {exc}\n{traceback.format_exc()}")
        await broadcast({"type": "scan_error", "error": str(exc)})

    finally:
        state.is_scanning = False
        _update_next_scan_time()
        await broadcast(state.snapshot())


def _update_next_scan_time():
    now = datetime.now(timezone.utc)
    delta = state.scan_interval_min * 60
    state.next_scan_time = datetime.fromtimestamp(
        now.timestamp() + delta, tz=timezone.utc
    ).isoformat()


# ---------------------------------------------------------------------------
# Background scanner loop
# ---------------------------------------------------------------------------
async def scanner_loop():
    """Runs scans on a fixed interval forever."""
    # Initial scan on startup
    await run_scan()

    while True:
        await asyncio.sleep(state.scan_interval_min * 60)
        await run_scan()


# ---------------------------------------------------------------------------
# Startup / shutdown
# ---------------------------------------------------------------------------
@app.on_event("startup")
async def on_startup():
    # Pick up interval from env (set by main() before uvicorn.run)
    env_interval = os.environ.get("TITAN_SCAN_INTERVAL_MIN")
    if env_interval:
        state.scan_interval_min = max(1, int(env_interval))

    state.settings = {}
    _update_next_scan_time()

    # Load cached results if available
    if SCAN_FILE.exists():
        try:
            with open(SCAN_FILE) as f:
                state.last_scan = json.load(f)
                if "trade_decision" not in state.last_scan:
                    _augment_with_position_awareness(state.last_scan)
                state.last_scan_time = state.last_scan.get("timestamp")
        except Exception:
            pass

    asyncio.create_task(scanner_loop())
    logger.info(f"Dashboard started – scanning every {state.scan_interval_min} min")


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/")
async def serve_dashboard():
    return FileResponse(str(STATIC_DIR / "live.html"))


from fastapi.responses import Response

@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return Response(content=b"", media_type="image/x-icon")


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    state.connected_clients.append(ws)
    logger.info(f"Client connected ({len(state.connected_clients)} total)")

    # Send current state + last scan immediately
    try:
        await ws.send_text(json.dumps(state.snapshot(), default=str))
        if state.last_scan:
            await ws.send_text(json.dumps({
                "type": "scan_result",
                "data": state.last_scan,
            }, default=str))
    except Exception:
        pass

    try:
        while True:
            data = await ws.receive_text()
            msg = json.loads(data)

            if msg.get("action") == "scan_now" and not state.is_scanning:
                asyncio.create_task(run_scan())
            elif msg.get("action") == "ping":
                await ws.send_text(json.dumps({"type": "pong"}))

    except WebSocketDisconnect:
        pass
    except Exception:
        pass
    finally:
        if ws in state.connected_clients:
            state.connected_clients.remove(ws)
        logger.info(f"Client disconnected ({len(state.connected_clients)} total)")


@app.get("/api/scan-results")
async def get_scan_results():
    if state.last_scan:
        return {"status": "success", "data": state.last_scan}
    if SCAN_FILE.exists():
        with open(SCAN_FILE) as f:
            data = json.load(f)
        if "trade_decision" not in data:
            _augment_with_position_awareness(data)
        return {"status": "success", "data": data}
    return {"status": "pending", "message": "First scan in progress..."}


@app.post("/api/scan-now")
async def trigger_scan():
    if state.is_scanning:
        return {"status": "busy", "message": "Scan already running"}
    asyncio.create_task(run_scan())
    return {"status": "started"}


@app.get("/api/status")
async def get_status():
    return state.snapshot()


@app.get("/api/scan-history")
async def get_scan_history():
    return {"history": state.scan_history}


@app.get("/api/portfolio")
async def get_portfolio():
    try:
        executor = AlpacaExecutor()
        if not executor.is_connected():
            return {"status": "disconnected", "message": "Alpaca not connected"}
        return {
            "status": "success",
            "account": executor.get_account_summary(),
            "positions": executor.get_positions_summary(),
        }
    except Exception as exc:
        return {"status": "error", "message": str(exc)}


# ---------------------------------------------------------------------------
# My Positions — manual trade tracker
# ---------------------------------------------------------------------------
class AddPositionRequest(BaseModel):
    ticker: str
    entry_price: float
    shares: float
    entry_date: Optional[str] = None
    stop: Optional[float] = None
    target: Optional[float] = None
    time_stop_days: Optional[int] = DEFAULT_TIME_STOP_DAYS
    signal_type: Optional[str] = None
    entry_note: Optional[str] = None
    notes: Optional[str] = None
    sector: Optional[str] = None
    atr: Optional[float] = None
    force: bool = False  # bypass soft risk / sanity warnings
    override_reason: Optional[str] = None


class UpdatePositionRequest(BaseModel):
    stop: Optional[float] = None
    target: Optional[float] = None
    time_stop_days: Optional[int] = None
    notes: Optional[str] = None
    entry_note: Optional[str] = None
    atr: Optional[float] = None


class ClosePositionRequest(BaseModel):
    exit_price: float
    exit_date: Optional[str] = None
    reason: Optional[str] = "MANUAL"


def _enrich_positions(positions: List[dict]) -> List[dict]:
    """Attach live price, P&L, days-held, alert, earnings warning,
    and trailing stop to each position."""
    open_positions = [p for p in positions if p.get("status") == "OPEN"]
    tickers = sorted({p["ticker"] for p in open_positions})
    prices = _fetch_live_prices(tickers) if tickers else {}

    # Update high-water marks for open positions using today's live prices.
    # Persisted — trailing stops depend on seeing the highest print since entry.
    if prices:
        my_positions.record_high_water(prices)
        # Re-read after the ratchet so callers see the freshest high-water mark.
        positions = my_positions.list_all()

    enriched: List[dict] = []
    for p in positions:
        cp = prices.get(p.get("ticker")) if p.get("status") == "OPEN" else None
        enriched.append(
            my_positions.evaluate(
                p, current_price=cp, earnings_checker=_earnings_days_until,
            )
        )
    return enriched


@app.get("/api/my-positions")
async def list_my_positions():
    # Reload from disk each request — record_high_water may have written
    # concurrently from the scan loop.
    positions = my_positions.list_all()
    enriched = _enrich_positions(positions)
    summary = my_positions.summary()
    realized_today = summary.get("realized_today_dollars", 0.0) or 0.0
    daily_pnl_pct = (realized_today / ACCOUNT_SIZE * 100.0) if ACCOUNT_SIZE > 0 else 0.0
    return {
        "status": "success",
        "positions": enriched,
        "summary": summary,
        "daily_realized_dollars": round(realized_today, 2),
        "daily_realized_pct": round(daily_pnl_pct, 3),
        "daily_loss_cap_pct": MAX_DAILY_LOSS_PCT,
        "daily_loss_cap_hit": daily_pnl_pct <= -MAX_DAILY_LOSS_PCT,
        "account_size": ACCOUNT_SIZE,
    }


@app.get("/api/my-positions/validate")
async def validate_new_position(ticker: str, entry_price: float, shares: float,
                                 stop: Optional[float] = None):
    """Pre-flight validation: risk %, live price, sanity check. UI calls this
    as the user types the form, so they see warnings before clicking ADD.
    """
    ticker = (ticker or "").upper().strip()
    live = _fetch_live_prices([ticker], prefer_live=True).get(ticker) if ticker else None
    risk = validate_position_risk(entry_price, stop, shares)
    sanity = validate_entry_price(entry_price, live)
    return {
        "status": "success",
        "ticker": ticker,
        "live_price": round(live, 2) if live else None,
        "risk": risk,
        "sanity": sanity,
        "account_size": ACCOUNT_SIZE,
        "hard_cap_pct": HARD_CAP_PCT,
        "soft_warn_pct": SOFT_WARN_PCT,
    }


@app.get("/api/my-positions/plan/{ticker}")
async def lookup_plan(ticker: str):
    """Auto-fill helper: return the scanner's stop/target/time-stop for a ticker
    if it's in the latest scan. UI calls this when the user types a ticker.
    """
    plan = _lookup_plan_in_scan(ticker.upper().strip())
    if plan is None:
        return {"status": "not_found"}
    return {"status": "success", "plan": plan}


@app.post("/api/my-positions")
async def add_my_position(req: AddPositionRequest):
    ticker = (req.ticker or "").upper().strip()

    # -------- Gate 1: Hard risk cap -----------------------------------
    risk = validate_position_risk(req.entry_price, req.stop, req.shares)
    if not risk["ok"]:
        # Hard cap: refuse even with force=true (protects user from themselves)
        raise HTTPException(
            status_code=400,
            detail={"kind": "RISK_TOO_HIGH", **risk},
        )

    # -------- Gate 2: Entry price sanity ------------------------------
    live = _fetch_live_prices([ticker], prefer_live=True).get(ticker) if ticker else None
    sanity = validate_entry_price(req.entry_price, live)
    if not sanity["ok"] and not req.force:
        raise HTTPException(
            status_code=400,
            detail={"kind": "ENTRY_PRICE_DIVERGENT", "live_price": live, **sanity},
        )

    # -------- Gate 3: Soft risk warning (needs force=true to bypass) --
    if risk.get("level") == "SOFT" and not req.force:
        raise HTTPException(
            status_code=400,
            detail={"kind": "RISK_SOFT_WARN", **risk},
        )

    # -------- Gate 4: Daily loss circuit breaker ----------------------
    summary = my_positions.summary()
    realized_today = summary.get("realized_today_dollars", 0.0) or 0.0
    daily_pnl_pct = (realized_today / ACCOUNT_SIZE * 100.0) if ACCOUNT_SIZE > 0 else 0.0
    if daily_pnl_pct <= -MAX_DAILY_LOSS_PCT and not req.force:
        raise HTTPException(
            status_code=400,
            detail={
                "kind": "DAILY_LOSS_CAP_HIT",
                "msg": (
                    f"Daily loss cap hit (realized {daily_pnl_pct:.2f}% <= "
                    f"-{MAX_DAILY_LOSS_PCT}%). Stop trading today or force=true."
                ),
                "realized_today_dollars": realized_today,
                "daily_pnl_pct": daily_pnl_pct,
            },
        )

    # -------- Gate 5: Final decision discipline -----------------------
    # If the latest scan produced a final decision, only the BUY ticker can be
    # added without FORCE. This prevents accidental buys from the Active table.
    decision = (state.last_scan or {}).get("trade_decision") if state.last_scan else None
    if decision:
        allowed_ticker = decision.get("ticker") if decision.get("action") == "BUY" else None
        if ticker != allowed_ticker and not req.force:
            raise HTTPException(
                status_code=400,
                detail={
                    "kind": "NOT_CURRENT_BUY",
                    "msg": (
                        f"Latest decision is {decision.get('action')} "
                        f"{decision.get('ticker') or ''}. Tick FORCE only if you are "
                        "intentionally overriding the system."
                    ),
                    "decision": decision,
                },
            )

    # -------- Auto-fill sector/ATR from latest scan if missing --------
    plan = _lookup_plan_in_scan(ticker) if ticker else None
    plan_full = _lookup_row_in_scan(ticker) if ticker else None
    sector = req.sector or (plan_full.get("sector") if plan_full else None)
    atr = req.atr
    if atr is None and plan_full:
        atr = plan_full.get("atr")

    try:
        pos = my_positions.add(
            ticker=ticker,
            entry_price=req.entry_price,
            shares=req.shares,
            entry_date=req.entry_date,
            stop=req.stop,
            target=req.target,
            time_stop_days=req.time_stop_days or DEFAULT_TIME_STOP_DAYS,
            signal_type=req.signal_type,
            entry_note=req.entry_note,
            notes=req.notes,
            sector=sector,
            atr=atr,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    # Evaluate immediately so the client sees alert state without a round-trip
    cp = live if live else _fetch_live_prices([pos["ticker"]]).get(pos["ticker"])
    enriched = my_positions.evaluate(
        pos, current_price=cp, earnings_checker=_earnings_days_until
    )
    await broadcast({"type": "my_positions_changed", "reason": "added"})
    return {
        "status": "success",
        "position": enriched,
        "risk": risk,
        "sanity": sanity,
    }


@app.patch("/api/my-positions/{position_id}")
async def update_my_position(position_id: str, req: UpdatePositionRequest):
    updated = my_positions.update(
        position_id,
        **{k: v for k, v in req.model_dump().items() if v is not None},
    )
    if updated is None:
        raise HTTPException(status_code=404, detail="Position not found")
    await broadcast({"type": "my_positions_changed", "reason": "updated"})
    return {"status": "success", "position": updated}


@app.post("/api/my-positions/{position_id}/close")
async def close_my_position(position_id: str, req: ClosePositionRequest):
    try:
        closed = my_positions.close(
            position_id,
            exit_price=req.exit_price,
            exit_date=req.exit_date,
            reason=req.reason or "MANUAL",
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    if closed is None:
        raise HTTPException(status_code=404, detail="Position not found")
    await broadcast({"type": "my_positions_changed", "reason": "closed"})
    return {"status": "success", "position": closed}


@app.delete("/api/my-positions/{position_id}")
async def delete_my_position(position_id: str):
    ok = my_positions.delete(position_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Position not found")
    await broadcast({"type": "my_positions_changed", "reason": "deleted"})
    return {"status": "success"}


# ---------------------------------------------------------------------------
# CLI entry
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Titan Trade Live Dashboard")
    parser.add_argument("--port", type=int, default=8000, help="Server port (default 8000)")
    parser.add_argument("--interval", type=int, default=15, help="Scan interval in minutes (default 15)")
    parser.add_argument("--host", default="127.0.0.1", help="Bind address")
    args = parser.parse_args()

    interval = max(1, args.interval)
    os.environ["TITAN_SCAN_INTERVAL_MIN"] = str(interval)

    import uvicorn
    print(f"\n{'='*60}")
    print(f"  TITAN PRO SCANNER — LIVE DASHBOARD")
    print(f"  http://{args.host}:{args.port}")
    print(f"  Scan interval: {interval} minutes")
    print(f"  Mode: Multi-Signal Detection (KOSPI architecture)")
    print(f"{'='*60}\n")

    uvicorn.run(
        "titan_dashboard:app",
        host=args.host,
        port=args.port,
        reload=False,
        log_level="info",
    )


if __name__ == "__main__":
    main()
