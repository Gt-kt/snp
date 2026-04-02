"""
Titan Signal Tracker — Forward Performance Logging
===================================================
Logs every signal the scanner produces and tracks what actually
happened. This is how you know if the system works in real life,
not just in backtests.

Signal lifecycle:
  1. Scanner detects signal → log_signal()
  2. Next N days → update_results() fills in forward returns
  3. Report → get_report() shows win rate, avg return by signal type
"""

import json
import os
import time
from datetime import datetime
from titan.signal_detector import _safe

SIGNAL_LOG_FILE = "signal_log.json"


class SignalTracker:
    """Persistent signal logger with forward-tracking."""

    def __init__(self, log_file: str = SIGNAL_LOG_FILE):
        self.log_file = log_file
        self._signals: list[dict] = []
        self._load()

    def _load(self):
        if os.path.exists(self.log_file):
            try:
                with open(self.log_file, "r") as f:
                    self._signals = json.load(f)
            except (json.JSONDecodeError, IOError):
                self._signals = []

    def _save(self):
        try:
            with open(self.log_file, "w") as f:
                json.dump(self._signals, f, indent=2, default=str)
        except IOError:
            pass

    def log_signal(self, signal: dict) -> bool:
        """Log a new signal. Returns False if duplicate (same ticker+date+type)."""
        key = (signal.get("ticker"), signal.get("date"), signal.get("signal_type"))
        for existing in self._signals:
            if (existing.get("ticker"), existing.get("date"), existing.get("signal_type")) == key:
                return False

        entry = {
            "ticker": signal.get("ticker"),
            "date": signal.get("date", datetime.now().strftime("%Y-%m-%d")),
            "time": signal.get("time", datetime.now().strftime("%H:%M")),
            "signal_type": signal.get("signal_type"),
            "tier": signal.get("tier", "unknown"),
            "signal_strength": signal.get("signal_strength", 0),
            "price": signal.get("price", 0),
            "stop": signal.get("stop", 0),
            "target": signal.get("target", 0),
            "buy_zone_low": signal.get("buy_zone_low", 0),
            "max_buy_price": signal.get("max_buy_price", 0),
            "entry_style": signal.get("entry_style", ""),
            "sector": signal.get("sector", ""),
            "reasons": signal.get("reasons", []),
            "status": "OPEN",
            "pnl_pct": None,
            "exit_reason": None,
            "logged_at": time.time(),
        }
        self._signals.append(entry)
        self._save()
        return True

    def update_results(self, market_data: dict, hold_days: int = 5):
        """Update open signals with forward returns from market data.

        market_data: dict of {ticker: DataFrame} with OHLCV data
        """
        import pandas as pd

        updated = False
        for signal in self._signals:
            if signal["status"] != "OPEN":
                continue

            ticker = signal["ticker"]
            df = market_data.get(ticker)
            if df is None or df.empty:
                continue

            try:
                signal_date = pd.Timestamp(signal["date"])
                if signal_date not in df.index:
                    # Try to find nearest date
                    mask = df.index >= signal_date
                    if not mask.any():
                        continue
                    signal_idx = df.index.get_loc(df.index[mask][0])
                else:
                    signal_idx = df.index.get_loc(signal_date)

                if isinstance(signal_idx, slice):
                    signal_idx = signal_idx.stop - 1

                exit_idx = signal_idx + hold_days
                if exit_idx >= len(df):
                    continue  # Not enough forward data yet

                entry_price = signal["price"]
                stop_price = signal["stop"]
                target_price = signal["target"]

                # Simulate: check if stop or target was hit during hold period
                exit_reason = "TIME_STOP"
                exit_price = _safe(df.iloc[exit_idx]["Close"])

                for day_offset in range(1, hold_days + 1):
                    idx = signal_idx + day_offset
                    if idx >= len(df):
                        break
                    day_low = _safe(df.iloc[idx]["Low"])
                    day_high = _safe(df.iloc[idx]["High"])

                    if day_low <= stop_price:
                        exit_price = stop_price
                        exit_reason = "STOPPED"
                        break
                    if day_high >= target_price:
                        exit_price = target_price
                        exit_reason = "TARGET_HIT"
                        break

                if entry_price > 0 and exit_price > 0:
                    pnl_pct = ((exit_price - entry_price) / entry_price) * 100
                    signal["pnl_pct"] = round(pnl_pct, 2)
                    signal["exit_reason"] = exit_reason
                    signal["status"] = "WIN" if pnl_pct > 0 else "LOSS"
                    updated = True

            except Exception:
                continue

        if updated:
            self._save()

    def get_report(self, days: int = 60) -> dict:
        """Generate a performance report for the last N calendar days."""
        import pandas as pd

        cutoff = pd.Timestamp.now().normalize() - pd.Timedelta(days=days)
        recent = []
        for s in self._signals:
            try:
                if pd.Timestamp(s["date"]) >= cutoff:
                    recent.append(s)
            except Exception:
                continue

        closed = [s for s in recent if s["status"] in ("WIN", "LOSS")]
        open_signals = [s for s in recent if s["status"] == "OPEN"]

        # By signal type
        by_type: dict[str, dict] = {}
        for s in closed:
            stype = s.get("signal_type", "UNKNOWN")
            bucket = by_type.setdefault(stype, {
                "trades": 0, "wins": 0, "losses": 0,
                "total_return": 0.0, "target_hits": 0, "stopped": 0,
            })
            bucket["trades"] += 1
            if s["status"] == "WIN":
                bucket["wins"] += 1
            else:
                bucket["losses"] += 1
            bucket["total_return"] += _safe(s.get("pnl_pct"))
            if s.get("exit_reason") == "TARGET_HIT":
                bucket["target_hits"] += 1
            if s.get("exit_reason") == "STOPPED":
                bucket["stopped"] += 1

        by_type_rows = []
        for stype, b in by_type.items():
            n = b["trades"]
            by_type_rows.append({
                "signal_type": stype,
                "trades": n,
                "win_rate": (b["wins"] / n * 100) if n else 0,
                "avg_return": (b["total_return"] / n) if n else 0,
                "total_return": b["total_return"],
                "target_hits": b["target_hits"],
                "stopped": b["stopped"],
            })
        by_type_rows.sort(key=lambda r: (r["trades"], r["avg_return"]), reverse=True)

        total_closed = len(closed)
        total_return = sum(_safe(s.get("pnl_pct")) for s in closed)

        return {
            "days": days,
            "total_signals": len(recent),
            "closed": total_closed,
            "open": len(open_signals),
            "win_rate": (sum(1 for s in closed if s["status"] == "WIN") / total_closed * 100) if total_closed else 0,
            "avg_return": (total_return / total_closed) if total_closed else 0,
            "total_return": total_return,
            "by_type": by_type_rows,
            "recent": sorted(recent, key=lambda s: s.get("date", ""), reverse=True)[:15],
        }

    def print_report(self, days: int = 60):
        """Pretty-print the forward performance report."""
        r = self.get_report(days)
        print(f"\n{'=' * 80}")
        print(f"SIGNAL TRACKER -- LAST {r['days']} DAYS")
        print(f"{'=' * 80}")
        print(f"Signals: {r['total_signals']}  |  Closed: {r['closed']}  |  Open: {r['open']}")
        if r["closed"]:
            print(f"Win Rate: {r['win_rate']:.1f}%  |  Avg Return: {r['avg_return']:+.2f}%  |  Total: {r['total_return']:+.2f}%")
        else:
            print("No closed trades yet.")

        if r["by_type"]:
            print(f"\n{'Type':<12} {'Trades':>6} {'Win%':>6} {'Avg%':>7} {'Total%':>8} {'Target':>7} {'Stop':>6}")
            print("-" * 60)
            for row in r["by_type"]:
                print(f"{row['signal_type']:<12} {row['trades']:>6} {row['win_rate']:>5.0f}% "
                      f"{row['avg_return']:>+6.2f}% {row['total_return']:>+7.2f}% "
                      f"{row['target_hits']:>7} {row['stopped']:>6}")

        if r["recent"]:
            print(f"\n{'Date':<12} {'Type':<12} {'Ticker':<8} {'Status':<6} {'PnL%':>7} {'Tier':<10}")
            print("-" * 60)
            for s in r["recent"][:10]:
                pnl = s.get("pnl_pct")
                pnl_str = f"{pnl:+.2f}%" if pnl is not None else "  -"
                print(f"{s.get('date', ''):<12} {s.get('signal_type', ''):<12} "
                      f"{s.get('ticker', ''):<8} {s.get('status', ''):<6} "
                      f"{pnl_str:>7} {s.get('tier', ''):<10}")
