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
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Import pro scanner + v3 utilities
# ---------------------------------------------------------------------------
from titan.pro_scanner import pro_scan
from titan.alpaca_executor import AlpacaExecutor

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("titan-dashboard")

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
    }

    # Persist to disk
    os.makedirs("data", exist_ok=True)
    with open(SCAN_FILE, "w") as f:
        json.dump(export, f, indent=2, default=str)
    return export


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
            return {"status": "success", "data": json.load(f)}
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
