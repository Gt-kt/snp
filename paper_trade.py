import argparse
import json
import os
from datetime import datetime

import pandas as pd
import yfinance as yf

from titan_trade import _ensure_multiindex, _load_config, _parse_tickers

PAPER_FILE = "paper_trades.json"
PAPER_HISTORY = "paper_trades_closed.csv"


def _load_paper(path):
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _save_paper(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def _load_scan(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing scan file: {path}")
    if path.lower().endswith(".json"):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return pd.DataFrame(data)
    return pd.read_csv(path)


def _fetch_latest_bars(tickers):
    if not tickers:
        return pd.DataFrame()
    data = yf.download(
        tickers,
        period="5d",
        interval="1d",
        auto_adjust=True,
        group_by="ticker",
        threads=False,
        progress=False,
    )
    if data is None or data.empty:
        return pd.DataFrame()
    return _ensure_multiindex(data, tickers)


def _calc_shares(entry, stop, risk_per_trade, account_size):
    risk_amt = min(float(risk_per_trade), float(account_size) * 0.01)
    risk_per_share = entry - stop
    if risk_per_share <= 0:
        return 0
    return max(int(risk_amt / risk_per_share), 1)


def open_from_scan(scan_path, paper_path, config_path=None, mode="buy_now", tickers=None, max_positions=0):
    scan = _load_scan(scan_path)
    tickers = _parse_tickers(tickers)
    if tickers:
        scan = scan[scan["Ticker"].isin(tickers)]

    if "Status" not in scan.columns:
        raise ValueError("scan file missing Status column. Use scan_results.csv from titan_trade.py")

    if mode == "buy_now":
        scan = scan[scan["Status"] == "BUY NOW"]
    elif mode == "pending":
        scan = scan[scan["Status"] == "PENDING"]

    if scan.empty:
        print("No matching setups found in scan file.")
        return

    cfg = _load_config(config_path)
    risk_per_trade = float(cfg.get("risk_per_trade", 50.0))
    account_size = float(cfg.get("account_size", 10000.0))

    paper = _load_paper(paper_path)
    added = 0
    for _, row in scan.iterrows():
        if max_positions and added >= max_positions:
            break
        ticker = row["Ticker"]
        if ticker in paper:
            continue
        status = row.get("Status", "PENDING")
        entry_price = float(row["Price"]) if status == "BUY NOW" else None
        trigger = float(row["Trigger"])
        stop = float(row["Stop"])
        target = float(row["Target"])
        shares = _calc_shares(entry_price or trigger, stop, risk_per_trade, account_size)

        paper[ticker] = {
            "ticker": ticker,
            "strategy": row.get("Strategy", ""),
            "status": "OPEN" if status == "BUY NOW" else "PENDING",
            "entry_date": datetime.now().strftime("%Y-%m-%d") if status == "BUY NOW" else None,
            "entry_price": entry_price,
            "trigger": trigger,
            "stop_loss": stop,
            "target": target,
            "shares": shares,
            "opened_from": os.path.basename(scan_path),
            "last_update": None,
        }
        added += 1

    _save_paper(paper_path, paper)
    print(f"Added {added} paper positions to {paper_path}")


def _append_history(path, rows):
    if not rows:
        return
    df = pd.DataFrame(rows)
    if os.path.exists(path):
        df.to_csv(path, mode="a", header=False, index=False)
    else:
        df.to_csv(path, index=False)


def update_positions(paper_path, history_path):
    paper = _load_paper(paper_path)
    if not paper:
        print("No paper trades found.")
        return

    tickers = list(paper.keys())
    data = _fetch_latest_bars(tickers)
    if data.empty:
        print("No market data returned.")
        return

    closed_rows = []
    for ticker in tickers:
        if ticker not in data.columns.levels[0]:
            continue
        df = data[ticker].dropna()
        if df.empty:
            continue
        last = df.iloc[-1]
        last_date = str(df.index[-1].date())
        last_close = float(last["Close"])
        last_high = float(last["High"])
        last_low = float(last["Low"])

        pos = paper[ticker]
        status = pos.get("status")
        trigger = float(pos.get("trigger", 0))
        stop = float(pos.get("stop_loss", 0))
        target = float(pos.get("target", 0))

        # Fill pending orders
        if status == "PENDING":
            strat = pos.get("strategy", "")
            filled = False
            if "BREAKOUT" in strat and last_high >= trigger:
                filled = True
            if "DIP" in strat and last_low <= trigger:
                filled = True
            if filled:
                pos["status"] = "OPEN"
                pos["entry_date"] = last_date
                pos["entry_price"] = trigger
                status = "OPEN"

        if status != "OPEN":
            pos["last_update"] = last_date
            continue

        entry_price = float(pos.get("entry_price") or trigger)
        exit_price = None
        exit_reason = None

        hit_stop = last_low <= stop
        hit_target = last_high >= target

        if hit_stop and hit_target:
            exit_price = stop  # conservative
            exit_reason = "STOP_AND_TARGET_SAME_DAY"
        elif hit_stop:
            exit_price = stop
            exit_reason = "STOP"
        elif hit_target:
            exit_price = target
            exit_reason = "TARGET"

        if exit_price is not None:
            shares = int(pos.get("shares", 0))
            pnl = (exit_price - entry_price) * shares
            pnl_pct = (exit_price - entry_price) / entry_price if entry_price else 0.0
            closed_rows.append(
                {
                    "ticker": ticker,
                    "strategy": pos.get("strategy", ""),
                    "entry_date": pos.get("entry_date"),
                    "exit_date": last_date,
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "shares": shares,
                    "pnl": pnl,
                    "pnl_pct": pnl_pct,
                    "reason": exit_reason,
                }
            )
            del paper[ticker]
        else:
            pos["last_update"] = last_date

    _append_history(history_path, closed_rows)
    _save_paper(paper_path, paper)
    print(f"Update complete. Closed trades: {len(closed_rows)}")


def report(paper_path, history_path):
    paper = _load_paper(paper_path)
    open_count = len(paper)

    if os.path.exists(history_path):
        hist = pd.read_csv(history_path)
    else:
        hist = pd.DataFrame()

    closed = len(hist)
    wins = int((hist["pnl"] > 0).sum()) if closed else 0
    win_rate = (wins / closed * 100) if closed else 0.0
    total_pnl = float(hist["pnl"].sum()) if closed else 0.0
    avg_pnl = float(hist["pnl"].mean()) if closed else 0.0

    print("=== Paper Trade Report ===")
    print(f"Open positions: {open_count}")
    print(f"Closed positions: {closed}")
    print(f"Win rate: {win_rate:.1f}%")
    print(f"Total PnL: {total_pnl:.2f}")
    print(f"Avg PnL: {avg_pnl:.2f}")


def main():
    parser = argparse.ArgumentParser(description="Paper trading utilities for Titan Trade.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    open_p = sub.add_parser("open", help="Open paper trades from scan file.")
    open_p.add_argument("--from-scan", required=True, help="Scan results csv/json file.")
    open_p.add_argument("--paper-file", default=PAPER_FILE, help="Paper trade file.")
    open_p.add_argument("--config", default=None, help="Config file for sizing.")
    open_p.add_argument("--mode", choices=["buy_now", "pending", "all"], default="buy_now")
    open_p.add_argument("--tickers", default="", help="Comma-separated tickers to open.")
    open_p.add_argument("--max-positions", type=int, default=0)

    upd_p = sub.add_parser("update", help="Update open paper trades with latest prices.")
    upd_p.add_argument("--paper-file", default=PAPER_FILE, help="Paper trade file.")
    upd_p.add_argument("--history-file", default=PAPER_HISTORY, help="Closed trade history csv.")

    rep_p = sub.add_parser("report", help="Summarize paper trade performance.")
    rep_p.add_argument("--paper-file", default=PAPER_FILE, help="Paper trade file.")
    rep_p.add_argument("--history-file", default=PAPER_HISTORY, help="Closed trade history csv.")

    args = parser.parse_args()
    if args.cmd == "open":
        open_from_scan(
            args.from_scan,
            args.paper_file,
            config_path=args.config,
            mode=args.mode,
            tickers=args.tickers,
            max_positions=args.max_positions,
        )
    elif args.cmd == "update":
        update_positions(args.paper_file, args.history_file)
    elif args.cmd == "report":
        report(args.paper_file, args.history_file)


if __name__ == "__main__":
    main()
