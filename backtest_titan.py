import os
import argparse
import pandas as pd
from titan_trade import StrategyValidator

CACHE_DIR = "cache_sp500_elite"
OHLCV_CACHE_FILE = os.path.join(CACHE_DIR, "sp500_ohlcv_bulk.parquet")


def load_data():
    if not os.path.exists(OHLCV_CACHE_FILE):
        raise FileNotFoundError(f"Missing cache file: {OHLCV_CACHE_FILE}")
    return pd.read_parquet(OHLCV_CACHE_FILE)


def run_backtest(data, max_tickers=200, min_bars=250):
    tickers = data.columns.levels[0].tolist()
    tickers = [t for t in tickers if t != "SPY"][:max_tickers]
    results = []

    for t in tickers:
        try:
            df = data[t].dropna()
        except Exception:
            continue

        if len(df) < min_bars:
            continue

        val = StrategyValidator(df)
        breakout = val.backtest_breakout()
        dip = val.backtest_dip()

        results.append(
            {
                "Ticker": t,
                "Breakout_WR": breakout["win_rate"],
                "Breakout_PF": breakout["pf"],
                "Breakout_Trades": breakout["trades"],
                "Dip_WR": dip["win_rate"],
                "Dip_PF": dip["pf"],
                "Dip_Trades": dip["trades"],
            }
        )

    return pd.DataFrame(results)


def summarize(df):
    if df.empty:
        return {}

    summary = {
        "Breakout_WR_mean": df["Breakout_WR"].mean(),
        "Breakout_PF_mean": df["Breakout_PF"].mean(),
        "Breakout_Trades_mean": df["Breakout_Trades"].mean(),
        "Dip_WR_mean": df["Dip_WR"].mean(),
        "Dip_PF_mean": df["Dip_PF"].mean(),
        "Dip_Trades_mean": df["Dip_Trades"].mean(),
    }
    return summary


def main():
    parser = argparse.ArgumentParser(description="Backtest Titan strategies over cached data.")
    parser.add_argument("--max-tickers", type=int, default=200, help="Max tickers to evaluate.")
    parser.add_argument("--min-bars", type=int, default=250, help="Minimum bars required.")
    parser.add_argument("--output", default="backtest_titan_results.csv", help="CSV output file.")
    args = parser.parse_args()

    data = load_data()
    results = run_backtest(data, max_tickers=args.max_tickers, min_bars=args.min_bars)
    summary = summarize(results)

    if not results.empty:
        results.to_csv(args.output, index=False)

    print("=== Titan Backtest Summary ===")
    for k, v in summary.items():
        print(f"{k}: {v:.2f}")

    if results.empty:
        print("No results produced. Check cache data or parameters.")
    else:
        print(f"Saved results to {args.output}")


if __name__ == "__main__":
    main()
