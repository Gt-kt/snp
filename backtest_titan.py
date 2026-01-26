import os
import argparse
import pandas as pd
from titan_trade import StrategyValidator, Optimizer
import numpy as np

CACHE_DIR = "cache_sp500_elite"
OHLCV_CACHE_FILE = os.path.join(CACHE_DIR, "sp500_ohlcv_bulk.parquet")
WF_MIN_TRADES = 5
WF_MIN_PF = 1.2
WF_MIN_EXPECTANCY = 0.0
WF_MIN_FOLDS = 2
REGIME_MIN_TRADES = 5


def load_data():
    if not os.path.exists(OHLCV_CACHE_FILE):
        raise FileNotFoundError(f"Missing cache file: {OHLCV_CACHE_FILE}")
    return pd.read_parquet(OHLCV_CACHE_FILE)


def _apply_costs(trades, cost_bps, slippage_bps=0.0):
    if not trades:
        return trades
    cost = (cost_bps + slippage_bps) / 10000.0
    return [t - cost for t in trades]


def _compute_trade_stats(trades):
    if not trades:
        return {
            "win_rate": 0.0,
            "pf": 0.0,
            "trades": 0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
            "expectancy": 0.0,
            "median": 0.0,
            "std": 0.0,
            "gross_win": 0.0,
            "gross_loss": 0.0,
            "sum_pnl": 0.0,
        }

    wins = [t for t in trades if t > 0]
    losses = [t for t in trades if t <= 0]

    win_rate = float(len(wins) / len(trades) * 100)
    gross_win = float(sum(wins))
    gross_loss = float(abs(sum(losses)))
    pf = float(gross_win / gross_loss if gross_loss > 0 else (100.0 if gross_win > 0 else 0))

    avg_win = float(np.mean(wins)) if wins else 0.0
    avg_loss = float(np.mean(losses)) if losses else 0.0
    expectancy = float(np.mean(trades)) if trades else 0.0
    median = float(np.median(trades)) if trades else 0.0
    std = float(np.std(trades)) if trades else 0.0

    return {
        "win_rate": win_rate,
        "pf": pf,
        "trades": len(trades),
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "expectancy": expectancy,
        "median": median,
        "std": std,
        "gross_win": gross_win,
        "gross_loss": gross_loss,
        "sum_pnl": float(sum(trades)),
    }


def _regime_label(spy_df):
    if spy_df is None or spy_df.empty:
        return "NEUTRAL"
    c = spy_df["Close"]
    if len(c) < 200:
        return "NEUTRAL"
    sma50 = c.rolling(50).mean().iloc[-1]
    sma200 = c.rolling(200).mean().iloc[-1]
    curr = c.iloc[-1]
    if curr > sma200 and sma50 > sma200:
        return "BULL"
    if curr < sma200:
        return "BEAR"
    return "NEUTRAL"


def _regime_segments(df, segments=3):
    n = len(df)
    if n < 120 * segments:
        return []
    seg_len = max(120, n // segments)
    total_len = seg_len * segments
    start = n - total_len
    slices = []
    for i in range(segments):
        s = start + i * seg_len
        e = s + seg_len
        slices.append((s, e))
    return slices


def _regime_stability(ticker_df, spy_df, cost_bps, slippage_bps, segments=3):
    segs = _regime_segments(ticker_df, segments=segments)
    if not segs:
        return {}

    passed_b = 0
    passed_d = 0
    regimes = []

    for s, e in segs:
        td = ticker_df.iloc[s:e]
        sd = spy_df.iloc[s:e] if spy_df is not None else None
        regimes.append(_regime_label(sd))

        val = StrategyValidator(td)
        b = val.backtest_breakout(return_trades=True)
        d = val.backtest_dip(return_trades=True)

        b_trades = _apply_costs(b.get("trades_list", []), cost_bps, slippage_bps)
        d_trades = _apply_costs(d.get("trades_list", []), cost_bps, slippage_bps)
        b_stats = _compute_trade_stats(b_trades)
        d_stats = _compute_trade_stats(d_trades)

        if b_stats["trades"] >= REGIME_MIN_TRADES and b_stats["pf"] >= WF_MIN_PF and b_stats["expectancy"] > WF_MIN_EXPECTANCY:
            passed_b += 1
        if d_stats["trades"] >= REGIME_MIN_TRADES and d_stats["pf"] >= WF_MIN_PF and d_stats["expectancy"] > WF_MIN_EXPECTANCY:
            passed_d += 1

    total = len(segs)
    return {
        "Breakout_RegimeScore": float(passed_b / total) if total > 0 else 0.0,
        "Dip_RegimeScore": float(passed_d / total) if total > 0 else 0.0,
        "Regime_Count": total,
        "Regime_Labels": ",".join(regimes),
    }


def _split_train_test(df, train_ratio):
    if df is None or df.empty:
        return None, None
    split_idx = int(len(df) * train_ratio)
    if split_idx < 120 or (len(df) - split_idx) < 120:
        return None, None
    return df.iloc[:split_idx].copy(), df.iloc[split_idx:].copy()


def _split_oos(df, train_ratio, min_test_bars):
    if df is None or df.empty:
        return None, None
    split_idx = int(len(df) * train_ratio)
    if split_idx < 120 or (len(df) - split_idx) < min_test_bars:
        return None, None
    return df.iloc[:split_idx].copy(), df.iloc[split_idx:].copy()


def _walk_forward_stats(df, cost_bps, slippage_bps, train_ratio):
    train_df, test_df = _split_train_test(df, train_ratio)
    if train_df is None or test_df is None:
        return {}

    train_val = StrategyValidator(train_df)
    test_val = StrategyValidator(test_df)

    train_b = train_val.backtest_breakout(return_trades=True)
    test_b = test_val.backtest_breakout(return_trades=True)
    train_d = train_val.backtest_dip(return_trades=True)
    test_d = test_val.backtest_dip(return_trades=True)

    train_b_trades = _apply_costs(train_b.get("trades_list", []), cost_bps, slippage_bps)
    test_b_trades = _apply_costs(test_b.get("trades_list", []), cost_bps, slippage_bps)
    train_d_trades = _apply_costs(train_d.get("trades_list", []), cost_bps, slippage_bps)
    test_d_trades = _apply_costs(test_d.get("trades_list", []), cost_bps, slippage_bps)

    train_b_stats = _compute_trade_stats(train_b_trades)
    test_b_stats = _compute_trade_stats(test_b_trades)
    train_d_stats = _compute_trade_stats(train_d_trades)
    test_d_stats = _compute_trade_stats(test_d_trades)

    return {
        "Breakout_WF_Train_Trades": train_b_stats["trades"],
        "Breakout_WF_Train_PF": train_b_stats["pf"],
        "Breakout_WF_Train_Expectancy": train_b_stats["expectancy"],
        "Breakout_WF_Train_WR": train_b_stats["win_rate"],
        "Breakout_WF_Train_GrossWin": train_b_stats["gross_win"],
        "Breakout_WF_Train_GrossLoss": train_b_stats["gross_loss"],
        "Breakout_WF_Train_SumPnL": train_b_stats["sum_pnl"],
        "Breakout_WF_Test_Trades": test_b_stats["trades"],
        "Breakout_WF_Test_PF": test_b_stats["pf"],
        "Breakout_WF_Test_Expectancy": test_b_stats["expectancy"],
        "Breakout_WF_Test_WR": test_b_stats["win_rate"],
        "Breakout_WF_Test_GrossWin": test_b_stats["gross_win"],
        "Breakout_WF_Test_GrossLoss": test_b_stats["gross_loss"],
        "Breakout_WF_Test_SumPnL": test_b_stats["sum_pnl"],
        "Dip_WF_Train_Trades": train_d_stats["trades"],
        "Dip_WF_Train_PF": train_d_stats["pf"],
        "Dip_WF_Train_Expectancy": train_d_stats["expectancy"],
        "Dip_WF_Train_WR": train_d_stats["win_rate"],
        "Dip_WF_Train_GrossWin": train_d_stats["gross_win"],
        "Dip_WF_Train_GrossLoss": train_d_stats["gross_loss"],
        "Dip_WF_Train_SumPnL": train_d_stats["sum_pnl"],
        "Dip_WF_Test_Trades": test_d_stats["trades"],
        "Dip_WF_Test_PF": test_d_stats["pf"],
        "Dip_WF_Test_Expectancy": test_d_stats["expectancy"],
        "Dip_WF_Test_WR": test_d_stats["win_rate"],
        "Dip_WF_Test_GrossWin": test_d_stats["gross_win"],
        "Dip_WF_Test_GrossLoss": test_d_stats["gross_loss"],
        "Dip_WF_Test_SumPnL": test_d_stats["sum_pnl"],
    }


def _walk_forward_multi(df, cost_bps, slippage_bps, folds, test_ratio, min_trades):
    if df is None or df.empty or folds < 1:
        return {}

    n = len(df)
    test_len = int(n * test_ratio)
    if test_len < 120:
        test_len = 120
    if n < test_len * folds + 120:
        return {}

    b_fold_stats = []
    d_fold_stats = []
    b_all_trades = []
    d_all_trades = []

    for i in range(folds):
        test_start = n - (folds - i) * test_len
        test_end = test_start + test_len
        train_df = df.iloc[:test_start]
        test_df = df.iloc[test_start:test_end]

        if len(train_df) < 120 or len(test_df) < 120:
            continue

        train_val = StrategyValidator(train_df)
        test_val = StrategyValidator(test_df)

        test_b = test_val.backtest_breakout(return_trades=True)
        test_d = test_val.backtest_dip(return_trades=True)

        test_b_trades = _apply_costs(test_b.get("trades_list", []), cost_bps, slippage_bps)
        test_d_trades = _apply_costs(test_d.get("trades_list", []), cost_bps, slippage_bps)

        b_stats = _compute_trade_stats(test_b_trades)
        d_stats = _compute_trade_stats(test_d_trades)

        b_fold_stats.append(b_stats)
        d_fold_stats.append(d_stats)
        b_all_trades.extend(test_b_trades)
        d_all_trades.extend(test_d_trades)

    if not b_fold_stats and not d_fold_stats:
        return {}

    def fold_mean(stats_list, key):
        vals = [s[key] for s in stats_list if s["trades"] > 0]
        return float(np.mean(vals)) if vals else 0.0

    def pass_rate(stats_list):
        eligible = [s for s in stats_list if s["trades"] >= min_trades]
        if not eligible:
            return 0.0
        passed = [
            s
            for s in eligible
            if s["pf"] >= WF_MIN_PF and s["expectancy"] > WF_MIN_EXPECTANCY
        ]
        return float(len(passed) / len(eligible))

    b_agg = _compute_trade_stats(b_all_trades)
    d_agg = _compute_trade_stats(d_all_trades)

    return {
        "Breakout_WF_Folds": len(b_fold_stats),
        "Breakout_WF_PassRate": pass_rate(b_fold_stats),
        "Breakout_WF_Test_Trades_total": b_agg["trades"],
        "Breakout_WF_Test_PF_trade_weighted": b_agg["pf"],
        "Breakout_WF_Test_Expectancy_trade_weighted": b_agg["expectancy"],
        "Breakout_WF_Test_PF_mean": fold_mean(b_fold_stats, "pf"),
        "Breakout_WF_Test_Expectancy_mean": fold_mean(b_fold_stats, "expectancy"),
        "Dip_WF_Folds": len(d_fold_stats),
        "Dip_WF_PassRate": pass_rate(d_fold_stats),
        "Dip_WF_Test_Trades_total": d_agg["trades"],
        "Dip_WF_Test_PF_trade_weighted": d_agg["pf"],
        "Dip_WF_Test_Expectancy_trade_weighted": d_agg["expectancy"],
        "Dip_WF_Test_PF_mean": fold_mean(d_fold_stats, "pf"),
        "Dip_WF_Test_Expectancy_mean": fold_mean(d_fold_stats, "expectancy"),
        # Backward compatible keys
        "Breakout_WF_Test_Trades": b_agg["trades"],
        "Breakout_WF_Test_PF": b_agg["pf"],
        "Breakout_WF_Test_Expectancy": b_agg["expectancy"],
        "Dip_WF_Test_Trades": d_agg["trades"],
        "Dip_WF_Test_PF": d_agg["pf"],
        "Dip_WF_Test_Expectancy": d_agg["expectancy"],
    }


def run_backtest(
    data,
    max_tickers=200,
    min_bars=250,
    min_trades=5,
    cost_bps=0.0,
    slippage_bps=0.0,
    walk_forward=False,
    train_ratio=0.5,
    wf_folds=4,
    wf_test_ratio=0.2,
    oos=False,
    oos_train_ratio=0.7,
    oos_min_test_bars=252,
):
    tickers = data.columns.levels[0].tolist()
    tickers = [t for t in tickers if t != "SPY"][:max_tickers]
    results = []
    spy_df = None
    try:
        if "SPY" in data.columns.levels[0]:
            spy_df = data["SPY"].dropna()
    except Exception:
        spy_df = None

    for t in tickers:
        try:
            df = data[t].dropna()
        except Exception:
            continue

        if len(df) < min_bars:
            continue

        val = StrategyValidator(df)
        breakout = val.backtest_breakout(return_trades=True)
        dip = val.backtest_dip(return_trades=True)

        breakout_trades = _apply_costs(breakout.get("trades_list", []), cost_bps, slippage_bps)
        dip_trades = _apply_costs(dip.get("trades_list", []), cost_bps, slippage_bps)

        breakout_stats = _compute_trade_stats(breakout_trades)
        dip_stats = _compute_trade_stats(dip_trades)
        wf_stats = {}
        if walk_forward:
            wf_stats = _walk_forward_multi(df, cost_bps, slippage_bps, wf_folds, wf_test_ratio, min_trades)
            if not wf_stats:
                wf_stats = _walk_forward_stats(df, cost_bps, slippage_bps, train_ratio)
        regime_stats = {}
        try:
            if spy_df is not None:
                idx = df.index.intersection(spy_df.index)
                df_regime = df.loc[idx]
                spy_regime = spy_df.loc[idx]
            else:
                df_regime = df
                spy_regime = None
            regime_stats = _regime_stability(df_regime, spy_regime, cost_bps, slippage_bps, segments=3)
        except Exception:
            regime_stats = {}

        oos_stats = {}
        if oos:
            train_df, test_df = _split_oos(df, oos_train_ratio, oos_min_test_bars)
            if train_df is not None and test_df is not None:
                train_val = StrategyValidator(train_df)
                opt = Optimizer(train_val)
                _, params = opt.tune_breakout()
                test_val = StrategyValidator(test_df)
                b_test = test_val.backtest_breakout(
                    depth=params.get("depth", 0.20),
                    target_mult=params.get("target_mult", 3.5),
                    return_trades=True,
                )
                d_test = test_val.backtest_dip(return_trades=True)

                b_trades = _apply_costs(b_test.get("trades_list", []), cost_bps, slippage_bps)
                d_trades = _apply_costs(d_test.get("trades_list", []), cost_bps, slippage_bps)
                b_stats = _compute_trade_stats(b_trades)
                d_stats = _compute_trade_stats(d_trades)
                oos_stats = {
                    "Breakout_OOS_WR": b_stats["win_rate"],
                    "Breakout_OOS_PF": b_stats["pf"],
                    "Breakout_OOS_Trades": b_stats["trades"],
                    "Breakout_OOS_Expectancy": b_stats["expectancy"],
                    "Dip_OOS_WR": d_stats["win_rate"],
                    "Dip_OOS_PF": d_stats["pf"],
                    "Dip_OOS_Trades": d_stats["trades"],
                    "Dip_OOS_Expectancy": d_stats["expectancy"],
                }

        results.append(
            {
                "Ticker": t,
                "Breakout_WR": breakout_stats["win_rate"],
                "Breakout_PF": breakout_stats["pf"],
                "Breakout_Trades": breakout_stats["trades"],
                "Breakout_AvgWin": breakout_stats["avg_win"],
                "Breakout_AvgLoss": breakout_stats["avg_loss"],
                "Breakout_Expectancy": breakout_stats["expectancy"],
                "Breakout_Median": breakout_stats["median"],
                "Breakout_STD": breakout_stats["std"],
                "Breakout_GrossWin": breakout_stats["gross_win"],
                "Breakout_GrossLoss": breakout_stats["gross_loss"],
                "Breakout_SumPnL": breakout_stats["sum_pnl"],
                "Breakout_Eligible": breakout_stats["trades"] >= min_trades,
                "Dip_WR": dip_stats["win_rate"],
                "Dip_PF": dip_stats["pf"],
                "Dip_Trades": dip_stats["trades"],
                "Dip_AvgWin": dip_stats["avg_win"],
                "Dip_AvgLoss": dip_stats["avg_loss"],
                "Dip_Expectancy": dip_stats["expectancy"],
                "Dip_Median": dip_stats["median"],
                "Dip_STD": dip_stats["std"],
                "Dip_GrossWin": dip_stats["gross_win"],
                "Dip_GrossLoss": dip_stats["gross_loss"],
                "Dip_SumPnL": dip_stats["sum_pnl"],
                "Dip_Eligible": dip_stats["trades"] >= min_trades,
                **wf_stats,
                **regime_stats,
                **oos_stats,
            }
        )

    return pd.DataFrame(results)


def summarize(df):
    if df.empty:
        return {}

    breakout_df = df[df["Breakout_Eligible"] == True]
    dip_df = df[df["Dip_Eligible"] == True]

    breakout_trades_all = breakout_df["Breakout_Trades"].sum()
    dip_trades_all = dip_df["Dip_Trades"].sum()
    breakout_gross_win = breakout_df["Breakout_GrossWin"].sum()
    breakout_gross_loss = breakout_df["Breakout_GrossLoss"].sum()
    dip_gross_win = dip_df["Dip_GrossWin"].sum()
    dip_gross_loss = dip_df["Dip_GrossLoss"].sum()
    breakout_sum_pnl = breakout_df["Breakout_SumPnL"].sum()
    dip_sum_pnl = dip_df["Dip_SumPnL"].sum()

    summary = {
        "Breakout_Tickers_Eligible": int(breakout_df.shape[0]),
        "Breakout_WR_mean": breakout_df["Breakout_WR"].mean(),
        "Breakout_PF_mean": breakout_df["Breakout_PF"].mean(),
        "Breakout_Trades_mean": breakout_df["Breakout_Trades"].mean(),
        "Breakout_Trades_total": int(breakout_trades_all),
        "Breakout_PF_trade_weighted": float(
            breakout_gross_win / breakout_gross_loss if breakout_gross_loss > 0 else (100.0 if breakout_gross_win > 0 else 0.0)
        ),
        "Breakout_Expectancy_trade_weighted": float(
            breakout_sum_pnl / breakout_trades_all if breakout_trades_all > 0 else 0.0
        ),
        "Breakout_WR_trade_weighted": float(
            (breakout_df["Breakout_WR"] * breakout_df["Breakout_Trades"]).sum() / breakout_trades_all
        )
        if breakout_trades_all > 0
        else 0.0,
        "Dip_Tickers_Eligible": int(dip_df.shape[0]),
        "Dip_WR_mean": dip_df["Dip_WR"].mean(),
        "Dip_PF_mean": dip_df["Dip_PF"].mean(),
        "Dip_Trades_mean": dip_df["Dip_Trades"].mean(),
        "Dip_Trades_total": int(dip_trades_all),
        "Dip_PF_trade_weighted": float(
            dip_gross_win / dip_gross_loss if dip_gross_loss > 0 else (100.0 if dip_gross_win > 0 else 0.0)
        ),
        "Dip_Expectancy_trade_weighted": float(
            dip_sum_pnl / dip_trades_all if dip_trades_all > 0 else 0.0
        ),
        "Dip_WR_trade_weighted": float(
            (dip_df["Dip_WR"] * dip_df["Dip_Trades"]).sum() / dip_trades_all
        )
        if dip_trades_all > 0
        else 0.0,
    }

    if "Breakout_WF_Test_Trades" in df.columns:
        wf_breakout = df[df["Breakout_WF_Test_Trades"] >= WF_MIN_TRADES]
        wf_dip = df[df["Dip_WF_Test_Trades"] >= WF_MIN_TRADES]
        wf_breakout_trades = wf_breakout["Breakout_WF_Test_Trades"].sum()
        wf_dip_trades = wf_dip["Dip_WF_Test_Trades"].sum()

        if "Breakout_WF_Test_GrossWin" in wf_breakout.columns:
            wf_breakout_gw = wf_breakout["Breakout_WF_Test_GrossWin"].sum()
            wf_breakout_gl = wf_breakout["Breakout_WF_Test_GrossLoss"].sum()
            wf_dip_gw = wf_dip["Dip_WF_Test_GrossWin"].sum()
            wf_dip_gl = wf_dip["Dip_WF_Test_GrossLoss"].sum()
            wf_breakout_sum = wf_breakout["Breakout_WF_Test_SumPnL"].sum()
            wf_dip_sum = wf_dip["Dip_WF_Test_SumPnL"].sum()
            wf_breakout_pf_tw = float(
                wf_breakout_gw / wf_breakout_gl if wf_breakout_gl > 0 else (100.0 if wf_breakout_gw > 0 else 0.0)
            )
            wf_dip_pf_tw = float(
                wf_dip_gw / wf_dip_gl if wf_dip_gl > 0 else (100.0 if wf_dip_gw > 0 else 0.0)
            )
            wf_breakout_exp_tw = float(
                wf_breakout_sum / wf_breakout_trades if wf_breakout_trades > 0 else 0.0
            )
            wf_dip_exp_tw = float(
                wf_dip_sum / wf_dip_trades if wf_dip_trades > 0 else 0.0
            )
        else:
            wf_breakout_pf_col = "Breakout_WF_Test_PF_trade_weighted"
            wf_dip_pf_col = "Dip_WF_Test_PF_trade_weighted"
            wf_breakout_exp_col = "Breakout_WF_Test_Expectancy_trade_weighted"
            wf_dip_exp_col = "Dip_WF_Test_Expectancy_trade_weighted"
            wf_breakout_pf_tw = float(
                (wf_breakout[wf_breakout_pf_col] * wf_breakout["Breakout_WF_Test_Trades"]).sum() / wf_breakout_trades
            ) if wf_breakout_trades > 0 else 0.0
            wf_dip_pf_tw = float(
                (wf_dip[wf_dip_pf_col] * wf_dip["Dip_WF_Test_Trades"]).sum() / wf_dip_trades
            ) if wf_dip_trades > 0 else 0.0
            wf_breakout_exp_tw = float(
                (wf_breakout[wf_breakout_exp_col] * wf_breakout["Breakout_WF_Test_Trades"]).sum() / wf_breakout_trades
            ) if wf_breakout_trades > 0 else 0.0
            wf_dip_exp_tw = float(
                (wf_dip[wf_dip_exp_col] * wf_dip["Dip_WF_Test_Trades"]).sum() / wf_dip_trades
            ) if wf_dip_trades > 0 else 0.0
        summary.update(
            {
                "Breakout_WF_Tickers": int(wf_breakout.shape[0]),
                "Breakout_WF_Test_PF_trade_weighted": wf_breakout_pf_tw,
                "Breakout_WF_Test_Expectancy_trade_weighted": wf_breakout_exp_tw,
                "Dip_WF_Tickers": int(wf_dip.shape[0]),
                "Dip_WF_Test_PF_trade_weighted": wf_dip_pf_tw,
                "Dip_WF_Test_Expectancy_trade_weighted": wf_dip_exp_tw,
            }
        )
    if "Breakout_OOS_Trades" in df.columns:
        oos_b = df[df["Breakout_OOS_Trades"] >= WF_MIN_TRADES]
        oos_d = df[df["Dip_OOS_Trades"] >= WF_MIN_TRADES]
        if not oos_b.empty:
            summary["Breakout_OOS_WR_mean"] = oos_b["Breakout_OOS_WR"].mean()
            summary["Breakout_OOS_PF_mean"] = oos_b["Breakout_OOS_PF"].mean()
            summary["Breakout_OOS_Trades_mean"] = oos_b["Breakout_OOS_Trades"].mean()
        if not oos_d.empty:
            summary["Dip_OOS_WR_mean"] = oos_d["Dip_OOS_WR"].mean()
            summary["Dip_OOS_PF_mean"] = oos_d["Dip_OOS_PF"].mean()
            summary["Dip_OOS_Trades_mean"] = oos_d["Dip_OOS_Trades"].mean()
    return summary


def main():
    parser = argparse.ArgumentParser(description="Backtest Titan strategies over cached data.")
    parser.add_argument("--max-tickers", type=int, default=200, help="Max tickers to evaluate.")
    parser.add_argument("--min-bars", type=int, default=250, help="Minimum bars required.")
    parser.add_argument("--min-trades", type=int, default=5, help="Minimum trades required to include in summary.")
    parser.add_argument("--cost-bps", type=float, default=10.0, help="Round-trip cost in bps per trade.")
    parser.add_argument("--slippage-bps", type=float, default=5.0, help="Additional slippage bps per trade.")
    parser.add_argument("--walk-forward", action="store_true", help="Include walk-forward stats.")
    parser.add_argument("--train-ratio", type=float, default=0.5, help="Train split ratio for single walk-forward.")
    parser.add_argument("--wf-folds", type=int, default=4, help="Number of walk-forward folds (rolling).")
    parser.add_argument("--wf-test-ratio", type=float, default=0.2, help="Test size ratio per fold.")
    parser.add_argument("--oos", action="store_true", help="Include out-of-sample stats.")
    parser.add_argument("--oos-train-ratio", type=float, default=0.7, help="OOS train split ratio.")
    parser.add_argument("--oos-min-test-bars", type=int, default=252, help="OOS minimum test bars.")
    parser.add_argument("--output", default="backtest_titan_results.csv", help="CSV output file.")
    args = parser.parse_args()

    data = load_data()
    results = run_backtest(
        data,
        max_tickers=args.max_tickers,
        min_bars=args.min_bars,
        min_trades=args.min_trades,
        cost_bps=args.cost_bps,
        slippage_bps=args.slippage_bps,
        walk_forward=args.walk_forward,
        train_ratio=args.train_ratio,
        wf_folds=args.wf_folds,
        wf_test_ratio=args.wf_test_ratio,
        oos=args.oos,
        oos_train_ratio=args.oos_train_ratio,
        oos_min_test_bars=args.oos_min_test_bars,
    )
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
