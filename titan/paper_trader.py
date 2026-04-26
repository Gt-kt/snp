"""
Titan Paper Trader — Simulated Forward Execution
=================================================
Track signals in real-time to build ground-truth performance data
BEFORE committing real money.

Flow:
  1. Scanner emits signal -> log_signal() records it as PENDING_FILL
  2. Next scan -> update() checks if today's open filled the order
     - Open within buy zone -> FILLED at open + slippage
     - Gapped above max_buy -> MISSED
  3. Each subsequent scan -> check stop/target/time-stop
  4. On exit -> record P&L and exit reason

After 30+ closed trades, get_summary() gives you the real numbers:
  - Did the fills actually happen?
  - What % of signals were missed (gapped away)?
  - Real win rate AFTER slippage?
  - How often did stops/targets hit?

Usage:
    from titan.paper_trader import PaperTrader
    pt = PaperTrader()
    pt.log_signal(signal_dict)    # From scanner
    pt.update(market_data_dict)   # Each scan
    print(pt.get_summary())       # Check progress
"""

import json
import os
import tempfile
from datetime import datetime
import numpy as np

from titan.config import (
    PAPER_TRADE_FILE,
    DEFAULT_SLIPPAGE_BREAKOUT_BPS, DEFAULT_SLIPPAGE_DIP_BPS,
    DEFAULT_COMMISSION_BPS,
)
from titan.signal_detector import _safe, BREAKOUT_SIGNAL_TYPES


class PaperTrader:
    """Simulated execution tracker for validating signals before real money."""

    def __init__(self, log_file: str = PAPER_TRADE_FILE):
        self.log_file = log_file
        self.trades: list[dict] = []
        self._load()

    def _load(self):
        if os.path.exists(self.log_file):
            try:
                with open(self.log_file, "r") as f:
                    self.trades = json.load(f)
            except (json.JSONDecodeError, Exception):
                # Backup corrupted file before discarding data
                bak_path = self.log_file + ".corrupt.bak"
                try:
                    import shutil
                    shutil.copy2(self.log_file, bak_path)
                    print(f"  WARNING: {self.log_file} is corrupted. Backup saved to {bak_path}")
                except Exception:
                    print(f"  WARNING: {self.log_file} is corrupted and backup failed.")
                self.trades = []

    def _save(self):
        dir_name = os.path.dirname(self.log_file) or "."
        tmp_fd, tmp_path = tempfile.mkstemp(dir=dir_name, suffix='.tmp')
        try:
            with os.fdopen(tmp_fd, 'w') as f:
                json.dump(self.trades, f, indent=2, default=str)
            os.replace(tmp_path, self.log_file)  # atomic on Windows and Unix
        except Exception:
            # Clean up temp file on failure
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

    def log_signal(self, signal: dict) -> bool:
        """Log a new signal as PENDING_FILL. Called by scanner after each scan.

        Returns True if logged, False if duplicate.
        """
        ticker = signal["ticker"]
        date = signal.get("date", datetime.now().strftime("%Y-%m-%d"))
        sig_type = signal["signal_type"]

        # Dedup: don't log same ticker+date+type twice
        for t in self.trades:
            if t["ticker"] == ticker and t["signal_date"] == date and t["signal_type"] == sig_type:
                return False

        trade = {
            "ticker": ticker,
            "signal_date": date,
            "signal_type": sig_type,
            "tier": signal.get("tier", "UNKNOWN"),
            "grade": signal.get("grade", "C"),
            "signal_price": signal["price"],
            "stop": signal["stop"],
            "target": signal["target"],
            "max_buy_price": signal.get("max_buy_price", round(signal["price"] * 1.02, 2)),
            "buy_zone_low": signal.get("buy_zone_low", round(signal["stop"] * 1.02, 2)),
            "time_stop_days": signal.get("time_stop_days", 5),
            "signal_strength": signal.get("signal_strength", 0),
            "status": "PENDING_FILL",
            "entry_price": None,
            "entry_date": None,
            "exit_price": None,
            "exit_date": None,
            "exit_reason": None,
            "pnl_pct": None,
            "days_held": 0,
            "logged_at": datetime.now().isoformat(),
        }
        self.trades.append(trade)
        self._save()
        return True

    def update(self, market_data: dict):
        """Called each scan. Checks fills, stops, targets, time stops.

        Args:
            market_data: {ticker: DataFrame} with at least today's OHLCV
        """
        today = datetime.now().strftime("%Y-%m-%d")
        updated = False

        for trade in self.trades:
            if trade["status"] == "PENDING_FILL":
                if self._check_fill(trade, market_data, today):
                    updated = True
            elif trade["status"] == "FILLED":
                if self._check_exit(trade, market_data, today):
                    updated = True

        # Expire old PENDING_FILL trades (> 2 days old)
        for trade in self.trades:
            if trade["status"] == "PENDING_FILL":
                signal_date = trade.get("signal_date", "")
                if signal_date and signal_date < today:
                    # Give 1 day to fill, then expire
                    try:
                        sig_dt = datetime.strptime(signal_date, "%Y-%m-%d").date()
                        today_dt = datetime.strptime(today, "%Y-%m-%d").date()
                        bdays = int(np.busday_count(sig_dt, today_dt))
                        if bdays > 2:
                            trade["status"] = "EXPIRED"
                            trade["exit_reason"] = "No fill within 2 business days"
                            updated = True
                    except Exception:
                        pass

        if updated:
            self._save()

    def _check_fill(self, trade: dict, market_data: dict, today: str) -> bool:
        """Check if today's open fills the pending order."""
        ticker = trade["ticker"]
        df = market_data.get(ticker)
        if df is None or not hasattr(df, "iloc") or df.empty:
            return False

        today_open = _safe(df["Open"].iloc[-1])
        if today_open <= 0:
            return False

        # Gapped above max buy = missed
        if today_open > trade["max_buy_price"]:
            trade["status"] = "MISSED"
            trade["exit_reason"] = f"Gapped above max buy ${trade['max_buy_price']:.2f} (open: ${today_open:.2f})"
            trade["exit_date"] = today
            return True

        # Opened below buy zone low = breakdown, miss
        if today_open < trade["buy_zone_low"] * 0.97:
            trade["status"] = "MISSED"
            trade["exit_reason"] = f"Opened below buy zone (${today_open:.2f} < ${trade['buy_zone_low']:.2f})"
            trade["exit_date"] = today
            return True

        # Fill at open + slippage + commission
        if trade["signal_type"] in BREAKOUT_SIGNAL_TYPES:
            slippage_bps = DEFAULT_SLIPPAGE_BREAKOUT_BPS
        else:
            slippage_bps = DEFAULT_SLIPPAGE_DIP_BPS

        entry = today_open * (1 + (slippage_bps + DEFAULT_COMMISSION_BPS) / 10000)
        trade["entry_price"] = round(entry, 2)
        trade["entry_date"] = today
        trade["status"] = "FILLED"
        return True

    def _check_exit(self, trade: dict, market_data: dict, today: str) -> bool:
        """Check stop, target, time stop for filled trades."""
        ticker = trade["ticker"]
        df = market_data.get(ticker)
        if df is None or not hasattr(df, "iloc") or df.empty:
            return False

        today_low = _safe(df["Low"].iloc[-1])
        today_high = _safe(df["High"].iloc[-1])
        today_close = _safe(df["Close"].iloc[-1])

        if today_low <= 0 or today_high <= 0 or today_close <= 0:
            return False

        # Compute days_held from entry_date using business days
        try:
            entry_dt = datetime.strptime(trade["entry_date"], "%Y-%m-%d").date()
            today_dt = datetime.strptime(today, "%Y-%m-%d").date()
            trade["days_held"] = int(np.busday_count(entry_dt, today_dt))
        except Exception:
            trade["days_held"] += 1

        # Check stop first (worst case)
        if today_low <= trade["stop"]:
            self._close_trade(trade, trade["stop"], "STOPPED", today)
            return True

        # Check target
        if today_high >= trade["target"]:
            self._close_trade(trade, trade["target"], "TARGET_HIT", today)
            return True

        # Time stop
        if trade["days_held"] >= trade["time_stop_days"]:
            self._close_trade(trade, today_close, "TIME_STOP", today)
            return True

        return False

    def _close_trade(self, trade: dict, exit_price: float, reason: str, today: str):
        """Close a trade with exit commission."""
        exit_after_commission = exit_price * (1 - DEFAULT_COMMISSION_BPS / 10000)
        trade["exit_price"] = round(exit_after_commission, 2)
        trade["exit_date"] = today
        trade["exit_reason"] = reason
        trade["status"] = "CLOSED"

        if trade["entry_price"] and trade["entry_price"] > 0:
            trade["pnl_pct"] = round(
                (exit_after_commission - trade["entry_price"]) / trade["entry_price"] * 100, 2
            )

    def get_summary(self) -> dict:
        """Return paper trading performance summary."""
        if not self.trades:
            return {"total_signals": 0, "message": "No signals tracked yet. Run scanner to start."}

        pending = [t for t in self.trades if t["status"] == "PENDING_FILL"]
        filled = [t for t in self.trades if t["status"] == "FILLED"]
        missed = [t for t in self.trades if t["status"] == "MISSED"]
        expired = [t for t in self.trades if t["status"] == "EXPIRED"]
        closed = [t for t in self.trades if t["status"] == "CLOSED"]

        summary = {
            "total_signals": len(self.trades),
            "pending_fill": len(pending),
            "currently_open": len(filled),
            "missed": len(missed),
            "expired": len(expired),
            "closed": len(closed),
        }

        if not closed:
            summary["message"] = "No closed trades yet. Need more time for results."
            return summary

        wins = [t for t in closed if t["pnl_pct"] > 0]
        losses = [t for t in closed if t["pnl_pct"] <= 0]

        total_pnl = sum(t["pnl_pct"] for t in closed)
        avg_pnl = total_pnl / len(closed)
        win_rate = len(wins) / len(closed) * 100

        summary.update({
            "win_rate": round(win_rate, 1),
            "avg_return": round(avg_pnl, 2),
            "total_pnl_pct": round(total_pnl, 2),
            "avg_win": round(sum(t["pnl_pct"] for t in wins) / len(wins), 2) if wins else 0,
            "avg_loss": round(sum(t["pnl_pct"] for t in losses) / len(losses), 2) if losses else 0,
            "best_trade": max(t["pnl_pct"] for t in closed),
            "worst_trade": min(t["pnl_pct"] for t in closed),
            "fill_rate": round(len(closed) / (len(closed) + len(missed)) * 100, 1) if (closed or missed) else 0,
            "by_exit_reason": {
                reason: len([t for t in closed if t["exit_reason"] == reason])
                for reason in set(t["exit_reason"] for t in closed if t["exit_reason"])
            },
            "by_signal_type": {},
        })

        # Breakdown by signal type
        for sig_type in set(t["signal_type"] for t in closed):
            sig_trades = [t for t in closed if t["signal_type"] == sig_type]
            sig_wins = [t for t in sig_trades if t["pnl_pct"] > 0]
            summary["by_signal_type"][sig_type] = {
                "count": len(sig_trades),
                "win_rate": round(len(sig_wins) / len(sig_trades) * 100, 1),
                "avg_return": round(sum(t["pnl_pct"] for t in sig_trades) / len(sig_trades), 2),
            }

        return summary

    def print_summary(self):
        """Print formatted paper trade summary."""
        s = self.get_summary()

        print(f"\n  {'=' * 60}")
        print(f"  PAPER TRADE TRACKER — Real Execution Simulation")
        print(f"  {'=' * 60}")

        print(f"\n  Total Signals: {s['total_signals']}")
        print(f"  Pending Fill:  {s.get('pending_fill', 0)}")
        print(f"  Open:          {s.get('currently_open', 0)}")
        print(f"  Missed:        {s.get('missed', 0)}")
        print(f"  Closed:        {s.get('closed', 0)}")

        if s.get("message"):
            print(f"\n  {s['message']}")
            return

        print(f"\n  RESULTS (after slippage + commission)")
        print(f"  {'─' * 40}")
        print(f"  Win Rate:     {s['win_rate']:.1f}%")
        print(f"  Avg Return:   {s['avg_return']:+.2f}%")
        print(f"  Total P&L:    {s['total_pnl_pct']:+.2f}%")
        print(f"  Fill Rate:    {s['fill_rate']:.1f}%")
        print(f"  Avg Win:      {s['avg_win']:+.2f}%")
        print(f"  Avg Loss:     {s['avg_loss']:+.2f}%")
        print(f"  Best Trade:   {s['best_trade']:+.2f}%")
        print(f"  Worst Trade:  {s['worst_trade']:+.2f}%")

        if s.get("by_exit_reason"):
            print(f"\n  Exit Reasons:")
            for reason, count in sorted(s["by_exit_reason"].items(), key=lambda x: -x[1]):
                print(f"    {reason:<20} {count:>4}")

        if s.get("by_signal_type"):
            print(f"\n  By Signal Type:")
            for sig, data in sorted(s["by_signal_type"].items(), key=lambda x: -x[1]["count"]):
                print(f"    {sig:<12} n={data['count']:>3}  WR={data['win_rate']:5.1f}%  Avg={data['avg_return']:+.2f}%")

    def reset(self):
        """Clear all paper trades. Use with caution."""
        self.trades = []
        self._save()
