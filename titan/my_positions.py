"""
Titan My Positions — Manual Trade Tracker
==========================================
The user trades manually: they see a signal on the dashboard the night
before, buy at market open the next day, and need clear SELL alerts when
their position hits target, stop, or time-stop.

This module is that tracker. It does NOT auto-execute anything. It records
what the user actually bought, evaluates each position against fresh prices
on every refresh, and tells them what to do:

  HOLD / WARN_STOP / WARN_TARGET / SELL_STOP / SELL_TARGET / SELL_TRAIL /
  SELL_TIMESTOP / CLOSED

Plus an `earnings_warning` field (separate from the primary alert) so the
user sees "earnings in 3 days" even when the main alert is HOLD.

Flow:
  1. User sees signal on dashboard (e.g., ROST @ $223.56, stop $218, target $237)
  2. User buys at market open next day — actual fill, say $224.10
  3. User clicks "+ Add Position". Dashboard auto-pulls stop/target/time-stop
     from the scanner's latest plan. Risk is validated against ACCOUNT_SIZE.
  4. On every scan + every 60s, fresh prices update each position:
     - `highest_since_entry` is tracked (persisted) → trailing stop locks gains
     - Alert priority: hard-stop > target > trailing > time-stop > warnings > hold
  5. When alert fires, user sells and clicks CLOSE to log the exit.

Storage: my_positions.json, atomic write (tempfile + os.replace), corrupt-file
backup. All dates anchored to America/New_York.
"""
from __future__ import annotations

import json
import os
import shutil
import tempfile
import threading
import uuid
from datetime import datetime
from typing import Callable, Optional

from titan.market_time import today_et_str, bdays_between_et
from titan.position_risk import compute_trailing_stop, effective_stop
from titan.swing_exits import (
    partial_ladder_signal,
    quick_profit_signal,
    time_stop_approaching,
)

DEFAULT_POSITIONS_FILE = "my_positions.json"

# Swing-trader default horizon: 2–4 business days active + 1 day of slack.
# Was 10 (position-trader default); too long for the user's 2–4 day hold style.
DEFAULT_TIME_STOP_DAYS = 5

# Alert states ----------------------------------------------------------------
ALERT_SELL_TARGET = "SELL_TARGET"       # current >= target  -> exit now
ALERT_SELL_STOP = "SELL_STOP"           # current <= hard stop
ALERT_SELL_TRAIL = "SELL_TRAIL"         # current <= trailing stop (locks gains)
ALERT_SELL_TIMESTOP = "SELL_TIMESTOP"   # days_held >= time_stop_days
ALERT_WARN_STOP = "WARN_STOP"           # within 1% above effective stop
ALERT_WARN_TARGET = "WARN_TARGET"       # within 1% below target
ALERT_WARN_TIMESTOP = "WARN_TIMESTOP"   # 1 business day before forced exit
ALERT_HOLD = "HOLD"
ALERT_CLOSED = "CLOSED"

_SELL_ALERTS = {ALERT_SELL_TARGET, ALERT_SELL_STOP, ALERT_SELL_TRAIL, ALERT_SELL_TIMESTOP}

# EarningsChecker signature: (ticker: str) -> Optional[int]  # days until earnings
EarningsChecker = Optional[Callable[[str], Optional[int]]]


class MyPositions:
    """Thread-safe JSON-backed tracker for positions the user actually bought."""

    def __init__(self, log_file: str = DEFAULT_POSITIONS_FILE):
        self.log_file = log_file
        self.positions: list[dict] = []
        self._lock = threading.RLock()
        self._load()

    # ------------------------------------------------------------------ I/O
    def _load(self) -> None:
        if not os.path.exists(self.log_file):
            self.positions = []
            return
        try:
            with open(self.log_file, "r") as f:
                data = json.load(f)
            if isinstance(data, list):
                self.positions = data
            else:
                self.positions = []
        except (json.JSONDecodeError, Exception):
            bak = self.log_file + ".corrupt.bak"
            try:
                shutil.copy2(self.log_file, bak)
                print(f"  WARNING: {self.log_file} corrupted. Backup -> {bak}")
            except Exception:
                print(f"  WARNING: {self.log_file} corrupted and backup failed.")
            self.positions = []

    def _save(self) -> None:
        dir_name = os.path.dirname(self.log_file) or "."
        tmp_fd, tmp_path = tempfile.mkstemp(dir=dir_name, suffix=".tmp")
        try:
            with os.fdopen(tmp_fd, "w") as f:
                json.dump(self.positions, f, indent=2, default=str)
            os.replace(tmp_path, self.log_file)
        except Exception:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

    # ---------------------------------------------------------------- CRUD
    def add(
        self,
        ticker: str,
        entry_price: float,
        shares: float,
        entry_date: Optional[str] = None,
        stop: Optional[float] = None,
        target: Optional[float] = None,
        time_stop_days: int = DEFAULT_TIME_STOP_DAYS,
        signal_type: Optional[str] = None,
        entry_note: Optional[str] = None,
        notes: Optional[str] = None,
        sector: Optional[str] = None,
        atr: Optional[float] = None,
    ) -> dict:
        """Add a new open position. Raises ValueError on bad input."""
        ticker = (ticker or "").strip().upper()
        if not ticker:
            raise ValueError("ticker is required")
        entry_price = float(entry_price)
        shares = float(shares)
        if entry_price <= 0:
            raise ValueError("entry_price must be > 0")
        if shares <= 0:
            raise ValueError("shares must be > 0")

        entry_date = entry_date or today_et_str()
        try:
            datetime.strptime(entry_date, "%Y-%m-%d")
        except ValueError:
            raise ValueError(f"entry_date must be YYYY-MM-DD, got {entry_date!r}")

        # Sanity-check stop/target; invalid → None so UI shows "no plan"
        stop_f = float(stop) if stop not in (None, "", 0) else None
        target_f = float(target) if target not in (None, "", 0) else None
        if stop_f is not None and stop_f >= entry_price:
            stop_f = None
        if target_f is not None and target_f <= entry_price:
            target_f = None

        pos = {
            "id": uuid.uuid4().hex[:12],
            "ticker": ticker,
            "entry_price": round(entry_price, 4),
            "shares": shares,
            "entry_date": entry_date,
            "stop": round(stop_f, 4) if stop_f is not None else None,
            "target": round(target_f, 4) if target_f is not None else None,
            "time_stop_days": max(1, int(time_stop_days or DEFAULT_TIME_STOP_DAYS)),
            "signal_type": signal_type,
            "entry_note": entry_note,
            "notes": notes,
            "sector": sector,
            "atr": round(float(atr), 4) if atr and atr > 0 else None,
            "highest_since_entry": round(entry_price, 4),  # seed with entry
            "status": "OPEN",
            "exit_price": None,
            "exit_date": None,
            "exit_reason": None,
            "pnl_pct": None,
            "pnl_dollars": None,
            "created_at": datetime.now().isoformat(),
        }
        with self._lock:
            self.positions.append(pos)
            self._save()
        return pos

    def update(self, position_id: str, **fields) -> Optional[dict]:
        """Patch editable fields (stop, target, time_stop_days, notes, atr, sector)."""
        allowed = {"stop", "target", "time_stop_days", "notes", "entry_note", "atr", "sector"}
        with self._lock:
            pos = self._find(position_id)
            if not pos:
                return None
            for k, v in fields.items():
                if k not in allowed:
                    continue
                if k in ("stop", "target", "atr"):
                    pos[k] = float(v) if v not in (None, "", 0) else None
                elif k == "time_stop_days":
                    pos[k] = max(1, int(v or 1))
                else:
                    pos[k] = v
            self._save()
            return dict(pos)

    def close(
        self,
        position_id: str,
        exit_price: float,
        exit_date: Optional[str] = None,
        reason: str = "MANUAL",
    ) -> Optional[dict]:
        """Mark a position CLOSED and record P&L."""
        exit_date = exit_date or today_et_str()
        try:
            datetime.strptime(exit_date, "%Y-%m-%d")
        except ValueError:
            raise ValueError(f"exit_date must be YYYY-MM-DD, got {exit_date!r}")
        exit_price = float(exit_price)
        if exit_price <= 0:
            raise ValueError("exit_price must be > 0")

        with self._lock:
            pos = self._find(position_id)
            if not pos:
                return None
            if pos["status"] != "OPEN":
                return dict(pos)
            entry = float(pos["entry_price"])
            shares = float(pos["shares"])
            pnl_dollars = round((exit_price - entry) * shares, 2)
            pnl_pct = round((exit_price - entry) / entry * 100.0, 2)
            pos["status"] = "CLOSED"
            pos["exit_price"] = round(exit_price, 4)
            pos["exit_date"] = exit_date
            pos["exit_reason"] = reason
            pos["pnl_dollars"] = pnl_dollars
            pos["pnl_pct"] = pnl_pct
            self._save()
            return dict(pos)

    def delete(self, position_id: str) -> bool:
        with self._lock:
            before = len(self.positions)
            self.positions = [p for p in self.positions if p.get("id") != position_id]
            if len(self.positions) != before:
                self._save()
                return True
            return False

    def get(self, position_id: str) -> Optional[dict]:
        with self._lock:
            pos = self._find(position_id)
            return dict(pos) if pos else None

    def list_all(self) -> list[dict]:
        with self._lock:
            return [dict(p) for p in self.positions]

    def list_open(self) -> list[dict]:
        with self._lock:
            return [dict(p) for p in self.positions if p.get("status") == "OPEN"]

    def _find(self, position_id: str) -> Optional[dict]:
        for p in self.positions:
            if p.get("id") == position_id:
                return p
        return None

    # --------------------------------------------------- Live-price ingest
    def record_high_water(self, prices: dict) -> bool:
        """Update `highest_since_entry` for each open position given current
        prices {ticker: price}. Persists changes. Returns True if any row changed.
        """
        if not prices:
            return False
        dirty = False
        with self._lock:
            for pos in self.positions:
                if pos.get("status") != "OPEN":
                    continue
                cp = prices.get(pos.get("ticker"))
                if cp is None or cp <= 0:
                    continue
                prev_high = pos.get("highest_since_entry") or pos.get("entry_price") or 0
                if cp > prev_high:
                    pos["highest_since_entry"] = round(float(cp), 4)
                    dirty = True
            if dirty:
                self._save()
        return dirty

    # --------------------------------------------------------------- Eval
    def evaluate(
        self,
        position: dict,
        current_price: Optional[float] = None,
        today: Optional[str] = None,
        earnings_checker: EarningsChecker = None,
    ) -> dict:
        """Enrich a position with live view: P&L, days-held, effective stop,
        alert state, earnings warning.

        Non-destructive — caller is responsible for persisting via record_high_water.
        """
        today = today or today_et_str()
        entry = float(position.get("entry_price") or 0)
        shares = float(position.get("shares") or 0)
        hard_stop = position.get("stop")
        target = position.get("target")
        entry_date = position.get("entry_date") or today
        time_stop_days = int(position.get("time_stop_days") or DEFAULT_TIME_STOP_DAYS)
        highest = position.get("highest_since_entry") or entry
        atr = position.get("atr")

        days_held = bdays_between_et(entry_date, today)
        days_left = max(0, time_stop_days - days_held)

        # Effective stop = max(hard_stop, trailing_stop)
        eff_stop = effective_stop(entry, hard_stop, highest, atr=atr)
        trail_stop = compute_trailing_stop(entry, hard_stop, highest, atr=atr)

        out = dict(position)
        out["days_held"] = days_held
        out["days_left"] = days_left
        out["effective_stop"] = round(eff_stop, 4) if eff_stop else None
        out["trailing_stop"] = round(trail_stop, 4) if trail_stop else None
        out["current_price"] = None
        out["pnl_pct_live"] = None
        out["pnl_dollars_live"] = None
        out["dist_to_stop_pct"] = None
        out["dist_to_target_pct"] = None
        out["alert"] = ALERT_HOLD
        out["alert_reason"] = None
        out["earnings_warning"] = None
        # Swing-trader advisories (always None for closed / no-live-price cases)
        out["partial_exit"] = None
        out["quick_profit"] = None
        out["time_stop_warn"] = False

        # Already closed → final state, no further eval
        if position.get("status") == "CLOSED":
            out["alert"] = ALERT_CLOSED
            out["alert_reason"] = position.get("exit_reason")
            return out

        # Earnings check (non-blocking warning — runs even if no live price)
        if earnings_checker is not None:
            try:
                days_until = earnings_checker(position.get("ticker") or "")
                if days_until is not None and -1 <= days_until <= 7:
                    if days_until < 0:
                        out["earnings_warning"] = (
                            f"Earnings {abs(days_until)}d ago — results priced in"
                        )
                    elif days_until == 0:
                        out["earnings_warning"] = "EARNINGS TODAY — expect volatility"
                    else:
                        out["earnings_warning"] = f"Earnings in {days_until}d"
            except Exception:
                pass  # earnings check is best-effort

        # Without a live price, we can still surface the time-stop runway —
        # the dashboard user reads this before market open to plan exits.
        out["time_stop_warn"] = time_stop_approaching(days_held, time_stop_days)

        if current_price is None or current_price <= 0 or entry <= 0:
            # No live price → only time-based alerts are meaningful
            if days_held >= time_stop_days:
                out["alert"] = ALERT_SELL_TIMESTOP
                out["alert_reason"] = (
                    f"Held {days_held} business days (limit {time_stop_days})"
                )
            elif out["time_stop_warn"]:
                out["alert"] = ALERT_WARN_TIMESTOP
                out["alert_reason"] = (
                    f"{days_left} business day{'s' if days_left != 1 else ''} "
                    f"left until forced exit (held {days_held}/{time_stop_days})"
                )
            return out

        cp = float(current_price)
        out["current_price"] = round(cp, 4)
        out["pnl_pct_live"] = round((cp - entry) / entry * 100.0, 2)
        out["pnl_dollars_live"] = round((cp - entry) * shares, 2)

        if eff_stop:
            out["dist_to_stop_pct"] = round((cp - eff_stop) / cp * 100.0, 2)
        if target:
            out["dist_to_target_pct"] = round((float(target) - cp) / cp * 100.0, 2)

        # Swing advisories — additive fields, don't override SELL alerts.
        out["partial_exit"] = partial_ladder_signal(entry, hard_stop, cp, shares)
        out["quick_profit"] = quick_profit_signal(entry, cp, days_held)
        out["time_stop_warn"] = time_stop_approaching(days_held, time_stop_days)

        # Alert priority (first match wins):
        #  1. Hard stop hit
        #  2. Target hit
        #  3. Trailing stop hit (but hard stop not yet)
        #  4. Time stop reached
        #  5. Within 1% of effective stop
        #  6. Within 1% of target
        #  7. WARN_TIMESTOP — 1 business day until forced exit
        #  8. HOLD

        if hard_stop and cp <= float(hard_stop):
            out["alert"] = ALERT_SELL_STOP
            out["alert_reason"] = f"Price ${cp:.2f} hit hard stop ${float(hard_stop):.2f}"
            return out

        if target and cp >= float(target):
            out["alert"] = ALERT_SELL_TARGET
            out["alert_reason"] = f"Price ${cp:.2f} reached target ${float(target):.2f}"
            return out

        if trail_stop and cp <= trail_stop:
            peak = float(highest)
            locked_pct = (trail_stop - entry) / entry * 100.0
            out["alert"] = ALERT_SELL_TRAIL
            out["alert_reason"] = (
                f"Trail stop ${trail_stop:.2f} hit (peak ${peak:.2f}, "
                f"locks +{locked_pct:.2f}%)"
            )
            return out

        if days_held >= time_stop_days:
            out["alert"] = ALERT_SELL_TIMESTOP
            out["alert_reason"] = f"Held {days_held} business days (limit {time_stop_days})"
            return out

        # WARN_STOP uses the *effective* stop so trailing warnings also fire
        if (
            eff_stop
            and out["dist_to_stop_pct"] is not None
            and 0 < out["dist_to_stop_pct"] <= 1.0
        ):
            out["alert"] = ALERT_WARN_STOP
            out["alert_reason"] = f"Within {out['dist_to_stop_pct']:.2f}% of stop"
            return out

        if (
            target
            and out["dist_to_target_pct"] is not None
            and 0 < out["dist_to_target_pct"] <= 1.0
        ):
            out["alert"] = ALERT_WARN_TARGET
            out["alert_reason"] = f"Within {out['dist_to_target_pct']:.2f}% of target"
            return out

        if out["time_stop_warn"]:
            out["alert"] = ALERT_WARN_TIMESTOP
            out["alert_reason"] = (
                f"{days_left} business day{'s' if days_left != 1 else ''} "
                f"left until forced exit (held {days_held}/{time_stop_days})"
            )
            return out

        return out

    # ---------------------------------------------------------- Summary
    def summary(self) -> dict:
        """Aggregate stats across closed trades + today's realized P&L."""
        with self._lock:
            positions = [dict(p) for p in self.positions]
        closed = [
            p for p in positions
            if p.get("status") == "CLOSED" and p.get("pnl_pct") is not None
        ]
        open_ = [p for p in positions if p.get("status") == "OPEN"]
        today = today_et_str()
        closed_today = [p for p in closed if p.get("exit_date") == today]

        s = {
            "total": len(positions),
            "open": len(open_),
            "closed": len(closed),
            "closed_today": len(closed_today),
            "realized_today_dollars": round(
                sum(p.get("pnl_dollars") or 0.0 for p in closed_today), 2
            ),
        }
        if closed:
            wins = [p for p in closed if (p.get("pnl_pct") or 0) > 0]
            s["win_rate"] = round(len(wins) / len(closed) * 100.0, 1)
            s["avg_return_pct"] = round(
                sum(p["pnl_pct"] for p in closed) / len(closed), 2
            )
            s["total_pnl_dollars"] = round(
                sum(p.get("pnl_dollars") or 0.0 for p in closed), 2
            )
        return s
