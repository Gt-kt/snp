import argparse
import contextlib
import io
import logging
import os
from types import SimpleNamespace

import pandas as pd

import titan_trade_v3 as titan_app
from titan.config import OHLCV_CACHE_FILE, SECTOR_ETFS
from titan.market import EarningsCalendar, SectorMapper
from titan.opportunity import build_scan_opportunity_summary, summarize_replay_opportunity


BREAKOUT_STRATEGIES = {"BREAKOUT", "EARLY BO", "LEADER BO"}


def load_cached_data():
    if not os.path.exists(OHLCV_CACHE_FILE):
        raise FileNotFoundError(f"Missing cache file: {OHLCV_CACHE_FILE}")
    return pd.read_parquet(OHLCV_CACHE_FILE)


def build_replay_settings(account_size=None, risk_per_trade=None, include_watchlist=True):
    args = SimpleNamespace(
        trust_mode=False,
        trust_paper=False,
        account_size=account_size,
        risk_per_trade=risk_per_trade,
    )
    settings = titan_app.build_runtime_settings(args, None, logging.getLogger("scanner-replay"))
    settings["max_workers"] = max(1, min(int(settings.get("max_workers", 4)), 4))
    settings["build_watchlist"] = bool(include_watchlist)
    settings["always_build_watchlist"] = bool(include_watchlist)
    return settings


def get_universe_symbols(data, max_tickers):
    excluded = set(SECTOR_ETFS.values()) | {"SPY", "^VIX", "VIX", "VIXY"}
    tickers = [t for t in data.columns.levels[0] if t not in excluded]
    return tickers[:max_tickers]


def build_snapshot_bundle(data, tickers, end_ts):
    required = list(dict.fromkeys(tickers + list(SECTOR_ETFS.values()) + ["SPY", "^VIX"]))
    mask = data.columns.get_level_values(0).isin(required)
    snapshot = data.loc[:end_ts, mask].copy()
    return tickers, snapshot


def resolve_scan_dates(spy_index, lookback_bars, scan_step, max_scans=None, start_date=None, end_date=None):
    dates = pd.Index(spy_index)
    if start_date:
        dates = dates[dates >= pd.Timestamp(start_date)]
    if end_date:
        dates = dates[dates <= pd.Timestamp(end_date)]
    if len(dates) <= lookback_bars + 1:
        return []

    eligible = list(dates[lookback_bars:-1:scan_step])
    if max_scans is not None and max_scans > 0:
        eligible = eligible[-max_scans:]
    return eligible


@contextlib.contextmanager
def replay_scan_context():
    original_get_earnings_date = EarningsCalendar.get_earnings_date
    original_is_in_blackout = EarningsCalendar.is_in_blackout
    original_get_sector = SectorMapper.get_sector

    SectorMapper._load_cache()
    sector_cache = dict(SectorMapper._cache or {})

    def _no_earnings_date(cls, ticker):
        return None, None

    def _no_blackout(cls, ticker, blackout_days=None, post_days=None):
        return False, "Replay ignores live earnings lookups"

    def _cached_sector(cls, ticker):
        return sector_cache.get(ticker, "Unknown")

    EarningsCalendar.get_earnings_date = classmethod(_no_earnings_date)
    EarningsCalendar.is_in_blackout = classmethod(_no_blackout)
    SectorMapper.get_sector = classmethod(_cached_sector)
    try:
        yield
    finally:
        EarningsCalendar.get_earnings_date = original_get_earnings_date
        EarningsCalendar.is_in_blackout = original_is_in_blackout
        SectorMapper.get_sector = original_get_sector


def run_scan_snapshot(tickers, snapshot_data, settings, quiet=True):
    bundle = (tickers, snapshot_data)
    with replay_scan_context():
        if quiet:
            sink = io.StringIO()
            with contextlib.redirect_stdout(sink):
                return titan_app.scan(settings=settings, market_data_bundle=bundle)
        return titan_app.scan(settings=settings, market_data_bundle=bundle)


def strategy_slippage_pct(strategy):
    if strategy == "DIP BUY":
        return 0.001
    if strategy == "LEADER BO":
        return 0.002
    return 0.003


def strategy_hold_days(strategy):
    return 10 if strategy == "DIP BUY" else 12


def try_fill_setup(setup, ticker_df, scan_idx, entry_window_days=3):
    if scan_idx + 1 >= len(ticker_df):
        return None, "no_future_data"

    strategy = getattr(setup, "strategy", "BREAKOUT")
    trigger = float(getattr(setup, "trigger", 0.0) or 0.0)
    stop = float(getattr(setup, "stop", 0.0) or 0.0)
    if trigger <= 0 or stop <= 0:
        return None, "bad_levels"

    last_entry_idx = min(len(ticker_df) - 1, scan_idx + max(1, entry_window_days))
    for entry_idx in range(scan_idx + 1, last_entry_idx + 1):
        bar = ticker_df.iloc[entry_idx]
        day_open = float(bar["Open"])
        day_high = float(bar["High"])

        if strategy in BREAKOUT_STRATEGIES:
            if day_open >= trigger:
                if day_open > trigger * 1.03:
                    return None, "gap_too_far"
                return {
                    "entry_idx": entry_idx,
                    "fill_price": day_open,
                    "fill_type": "gap_open",
                }, "filled"
            if day_high >= trigger:
                return {
                    "entry_idx": entry_idx,
                    "fill_price": trigger,
                    "fill_type": "intraday_trigger",
                }, "filled"
            continue

        if day_open > trigger * 1.03:
            return None, "gap_too_far"
        return {
            "entry_idx": entry_idx,
            "fill_price": day_open,
            "fill_type": "next_open",
        }, "filled"

    return None, "not_triggered"


def simulate_trade_exit(setup, ticker_df, entry_idx, fill_price):
    strategy = getattr(setup, "strategy", "BREAKOUT")
    slippage_pct = strategy_slippage_pct(strategy)
    actual_entry = float(fill_price) * (1 + slippage_pct)
    target = float(getattr(setup, "target", 0.0) or 0.0)
    stop_curr = float(getattr(setup, "stop", 0.0) or 0.0)
    breakeven_trigger = float(getattr(setup, "breakeven_trigger", 0.0) or 0.0)
    trailing_stop = float(getattr(setup, "trailing_stop", 0.0) or 0.0)
    max_hold = strategy_hold_days(strategy)
    highest_since_entry = actual_entry

    last_idx = min(len(ticker_df) - 1, entry_idx + max_hold - 1)
    exit_idx = last_idx
    exit_reason = "time_exit"
    exit_price = float(ticker_df.iloc[last_idx]["Close"]) * (1 - slippage_pct)

    for idx in range(entry_idx, last_idx + 1):
        bar = ticker_df.iloc[idx]
        day_open = float(bar["Open"])
        day_high = float(bar["High"])
        day_low = float(bar["Low"])
        day_close = float(bar["Close"])

        if day_open <= stop_curr:
            exit_idx = idx
            exit_reason = "gap_stop"
            exit_price = day_open * (1 - slippage_pct)
            break
        if target > 0 and day_open >= target:
            exit_idx = idx
            exit_reason = "gap_target"
            exit_price = day_open
            break

        highest_since_entry = max(highest_since_entry, day_high)
        if breakeven_trigger > 0 and highest_since_entry >= breakeven_trigger:
            stop_curr = max(stop_curr, actual_entry)
        if trailing_stop > 0 and highest_since_entry > actual_entry:
            if breakeven_trigger <= 0 or highest_since_entry >= breakeven_trigger:
                stop_curr = max(stop_curr, highest_since_entry - trailing_stop)

        if day_low <= stop_curr:
            exit_idx = idx
            exit_reason = "stop"
            exit_price = stop_curr * (1 - slippage_pct * 0.5)
            break
        if target > 0 and day_high >= target:
            exit_idx = idx
            exit_reason = "target"
            exit_price = target * (1 - slippage_pct * 0.3)
            break
        if idx == last_idx:
            exit_idx = idx
            exit_reason = "time_exit"
            exit_price = day_close * (1 - slippage_pct)

    hold_days = max(1, exit_idx - entry_idx + 1)
    return_pct = (exit_price - actual_entry) / max(actual_entry, 1e-9)
    return {
        "entry_price": actual_entry,
        "exit_price": exit_price,
        "exit_idx": exit_idx,
        "exit_reason": exit_reason,
        "return_pct": return_pct,
        "hold_days": hold_days,
    }


def summarize_trades(trades, account_size):
    if not trades:
        return {
            "Trades": 0,
            "Win_Rate_Pct": 0.0,
            "Profit_Factor": 0.0,
            "Avg_Return_Pct": 0.0,
            "Median_Return_Pct": 0.0,
            "Total_PnL": 0.0,
            "Avg_PnL": 0.0,
            "Avg_Hold_Days": 0.0,
            "Approx_Account_Return_Pct": 0.0,
        }

    pnl_values = [float(t["pnl"]) for t in trades]
    return_values = [float(t["return_pct"]) for t in trades]
    wins = [p for p in pnl_values if p > 0]
    losses = [p for p in pnl_values if p <= 0]
    gross_profit = float(sum(wins))
    gross_loss = float(abs(sum(losses)))
    pf = gross_profit / gross_loss if gross_loss > 0 else (100.0 if gross_profit > 0 else 0.0)

    return {
        "Trades": len(trades),
        "Win_Rate_Pct": float(sum(1 for p in pnl_values if p > 0) / len(trades) * 100),
        "Profit_Factor": float(pf),
        "Avg_Return_Pct": float(pd.Series(return_values).mean() * 100),
        "Median_Return_Pct": float(pd.Series(return_values).median() * 100),
        "Total_PnL": float(sum(pnl_values)),
        "Avg_PnL": float(pd.Series(pnl_values).mean()),
        "Avg_Hold_Days": float(pd.Series([t["hold_days"] for t in trades]).mean()),
        "Approx_Account_Return_Pct": float(sum(pnl_values) / max(account_size, 1.0) * 100),
    }


def replay_scanner(
    data,
    settings,
    max_tickers=100,
    scan_step=5,
    max_scans=52,
    top_n=3,
    max_positions=4,
    lookback_bars=260,
    entry_window_days=3,
    start_date=None,
    end_date=None,
    quiet=True,
    include_watchlist=True,
):
    tickers = get_universe_symbols(data, max_tickers)
    spy_index = data["SPY"].dropna().index if "SPY" in data.columns.levels[0] else data.index
    scan_dates = resolve_scan_dates(
        spy_index,
        lookback_bars=lookback_bars,
        scan_step=scan_step,
        max_scans=max_scans,
        start_date=start_date,
        end_date=end_date,
    )

    open_positions = []
    trades = []
    scan_rows = []
    skip_counts = {"gap_too_far": 0, "not_triggered": 0, "sizing": 0, "duplicate": 0, "capacity": 0}
    max_open_seen = 0

    for scan_date in scan_dates:
        scan_idx = spy_index.get_loc(scan_date)
        active_positions = [p for p in open_positions if p["exit_idx"] > scan_idx]
        current_heat_pct = sum(float(p["heat_pct"]) for p in active_positions)
        buying_power = float(settings["account_size"]) - sum(float(p["reserved_notional"]) for p in active_positions)
        reserved_orders = []
        filled_today = []

        snapshot_tickers, snapshot_data = build_snapshot_bundle(data, tickers, scan_date)
        setups, _, market_data, vix_level = run_scan_snapshot(snapshot_tickers, snapshot_data, settings, quiet=quiet)
        watchlist = market_data.get("watchlist", []) if include_watchlist else []
        opportunity = build_scan_opportunity_summary(
            setups,
            watchlist,
            market_data.get("mkt_status", "Unknown"),
            vix_level=vix_level,
        )

        candidates = list(setups[:max(1, top_n)])
        for setup in candidates:
            if len(active_positions) + len(reserved_orders) >= max_positions:
                skip_counts["capacity"] += 1
                break

            if any(p["ticker"] == setup.ticker for p in active_positions) or any(p["ticker"] == setup.ticker for p in reserved_orders):
                skip_counts["duplicate"] += 1
                continue

            reserved_heat_pct = sum(float(p["heat_pct"]) for p in reserved_orders)
            reserved_notional = sum(float(p["reserved_notional"]) for p in reserved_orders)
            qty, _ = titan_app.calculate_execution_qty(
                setup,
                settings,
                buying_power=max(0.0, buying_power - reserved_notional),
                current_heat_pct=current_heat_pct + reserved_heat_pct,
                vix_level=vix_level,
            )
            if qty < 1:
                skip_counts["sizing"] += 1
                continue

            ticker_df = data[setup.ticker][["Open", "High", "Low", "Close", "Volume"]].dropna()
            fill, fill_status = try_fill_setup(setup, ticker_df, scan_idx, entry_window_days=entry_window_days)
            if not fill:
                if fill_status in skip_counts:
                    skip_counts[fill_status] += 1
                continue

            trade_result = simulate_trade_exit(setup, ticker_df, fill["entry_idx"], fill["fill_price"])
            pnl = (trade_result["exit_price"] - trade_result["entry_price"]) * qty
            risk_per_share = max(float(getattr(setup, "trigger", 0.0) - getattr(setup, "stop", 0.0)), 0.0)
            reserved_notional_value = trade_result["entry_price"] * qty
            heat_pct = ((risk_per_share * qty) / max(float(settings["account_size"]), 1.0)) * 100 if risk_per_share > 0 else 0.0

            trade_row = {
                "scan_date": pd.Timestamp(scan_date),
                "entry_date": ticker_df.index[fill["entry_idx"]],
                "exit_date": ticker_df.index[trade_result["exit_idx"]],
                "ticker": setup.ticker,
                "strategy": setup.strategy,
                "fill_type": fill["fill_type"],
                "entry_price": trade_result["entry_price"],
                "exit_price": trade_result["exit_price"],
                "stop": float(getattr(setup, "stop", 0.0)),
                "target": float(getattr(setup, "target", 0.0)),
                "qty": int(qty),
                "return_pct": float(trade_result["return_pct"]),
                "pnl": float(pnl),
                "hold_days": int(trade_result["hold_days"]),
                "exit_reason": trade_result["exit_reason"],
                "score": float(getattr(setup, "score", 0.0)),
                "robustness_score": float(getattr(setup, "robustness_score", 0.0)),
                "confidence_grade": getattr(setup, "confidence_grade", "F"),
                "oos_pf": float(getattr(setup, "oos_pf", 0.0)),
                "oos_trades": int(getattr(setup, "oos_trades", 0)),
                "entry_ready_score": float(getattr(setup, "entry_ready_score", 0.0)),
                "market_status": market_data.get("mkt_status", "Unknown"),
                "vix_level": float(vix_level or 0.0),
                "distance_to_entry_pct": float(getattr(setup, "distance_to_entry_pct", 0.0)),
                "sector": getattr(setup, "sector", "Unknown"),
                "sector_aligned": bool(getattr(setup, "sector_aligned", False)),
                "heat_pct": float(heat_pct),
                "reserved_notional": float(reserved_notional_value),
                "entry_idx": int(fill["entry_idx"]),
                "exit_idx": int(trade_result["exit_idx"]),
            }
            reserved_orders.append(trade_row)
            filled_today.append(setup.ticker)

        open_positions.extend(reserved_orders)
        trades.extend(reserved_orders)
        max_open_seen = max(max_open_seen, len(active_positions) + len(reserved_orders))

        scan_rows.append(
            {
                "scan_date": pd.Timestamp(scan_date),
                "market_status": market_data.get("mkt_status", "Unknown"),
                "vix_level": float(vix_level or 0.0),
                "setups_found": len(setups),
                "watchlist_count": len(watchlist),
                "selected": len(candidates),
                "filled": len(filled_today),
                "filled_tickers": ",".join(filled_today),
                "top_tickers": ",".join(s.ticker for s in candidates),
                "watchlist_tickers": ",".join(item.get("ticker", "") for item in watchlist[:top_n]),
                "opportunity_state": opportunity["state"],
            }
        )

    summary = {
        "Scan_Count": len(scan_rows),
        "Signals_Selected": int(sum(row["selected"] for row in scan_rows)),
        "Trades_Filled": len(trades),
        "Skipped_Gap": skip_counts["gap_too_far"],
        "Skipped_NotTriggered": skip_counts["not_triggered"],
        "Skipped_Sizing": skip_counts["sizing"],
        "Skipped_Duplicate": skip_counts["duplicate"],
        "Skipped_Capacity": skip_counts["capacity"],
        "Max_Open_Positions": int(max_open_seen),
        "Unique_Tickers": int(len({t["ticker"] for t in trades})),
    }
    summary.update(summarize_trades(trades, float(settings["account_size"])))
    summary.update(summarize_replay_opportunity(scan_rows))

    trades_df = pd.DataFrame(trades)
    scan_df = pd.DataFrame(scan_rows)
    return trades_df, scan_df, summary


def print_summary(summary):
    print("=== Scanner Replay Summary ===")
    for key, value in summary.items():
        if isinstance(value, float):
            print(f"{key}: {value:.2f}")
        else:
            print(f"{key}: {value}")


def main():
    parser = argparse.ArgumentParser(description="Replay the live Titan scanner over historical snapshots.")
    parser.add_argument("--max-tickers", type=int, default=80, help="Number of S&P tickers to replay.")
    parser.add_argument("--scan-step", type=int, default=5, help="Trading-day step between scans.")
    parser.add_argument("--max-scans", type=int, default=52, help="Maximum historical scans to replay.")
    parser.add_argument("--top-n", type=int, default=3, help="Max setups to attempt per scan.")
    parser.add_argument("--max-positions", type=int, default=4, help="Max concurrent positions.")
    parser.add_argument("--lookback-bars", type=int, default=260, help="Warm-up bars before starting scans.")
    parser.add_argument("--entry-window-days", type=int, default=3, help="How many trading days to wait for a trigger.")
    parser.add_argument("--start-date", default=None, help="Optional scan start date (YYYY-MM-DD).")
    parser.add_argument("--end-date", default=None, help="Optional scan end date (YYYY-MM-DD).")
    parser.add_argument("--account-size", type=float, default=None, help="Override account size.")
    parser.add_argument("--risk-per-trade", type=float, default=None, help="Override risk per trade.")
    parser.add_argument("--output", default="scanner_replay_trades.csv", help="Trade CSV output path.")
    parser.add_argument("--scan-output", default="scanner_replay_scans.csv", help="Per-scan CSV output path.")
    parser.add_argument("--quiet", action="store_true", help="Suppress per-scan scanner output.")
    parser.add_argument("--no-watchlist", action="store_true", help="Disable research-watchlist tracking in replay.")
    args = parser.parse_args()

    data = load_cached_data()
    settings = build_replay_settings(
        account_size=args.account_size,
        risk_per_trade=args.risk_per_trade,
        include_watchlist=not args.no_watchlist,
    )
    trades_df, scan_df, summary = replay_scanner(
        data,
        settings,
        max_tickers=args.max_tickers,
        scan_step=args.scan_step,
        max_scans=args.max_scans,
        top_n=args.top_n,
        max_positions=args.max_positions,
        lookback_bars=args.lookback_bars,
        entry_window_days=args.entry_window_days,
        start_date=args.start_date,
        end_date=args.end_date,
        quiet=args.quiet,
        include_watchlist=not args.no_watchlist,
    )

    if not trades_df.empty:
        trades_df.to_csv(args.output, index=False)
    if not scan_df.empty:
        scan_df.to_csv(args.scan_output, index=False)

    print_summary(summary)
    if not trades_df.empty:
        print(f"Saved trades to {args.output}")
    if not scan_df.empty:
        print(f"Saved scan log to {args.scan_output}")


if __name__ == "__main__":
    main()
