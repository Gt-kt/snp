#!/usr/bin/env python3
"""
Titan Trade v9.0 - Institutional Edition (Alpaca Integration)
=============================================================
Manual-first S&P 500 scanner with optional Trust Mode execution when explicitly enabled.

Usage:
    python titan_trade_v3.py              # Manual scan mode
    python titan_trade_v3.py --help       # Show all options
"""

import sys
import os
import time
import argparse
import logging
import json
import io
import warnings
from datetime import datetime
import concurrent.futures

import pandas as pd
import numpy as np
import yfinance as yf
import requests
from tabulate import tabulate
from dotenv import load_dotenv

# Load .env variables for APCA_API_KEY_ID and APCA_API_SECRET_KEY
load_dotenv()

# Suppress warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
pd.options.mode.chained_assignment = None

# Import from titan package
from titan import (
    # Config
    CACHE_DIR, SP500_CACHE_FILE, OHLCV_CACHE_FILE, PORTFOLIO_FILE,
    ACCOUNT_SIZE, RISK_PER_TRADE, MAX_POSITIONS, MAX_DRAWDOWN_PCT,
    DEFAULT_OHLCV_TTL_HOURS, DEFAULT_SP500_TTL_DAYS, DEFAULT_DATA_PERIOD,
    DEFAULT_DATA_INTERVAL, DEFAULT_MAX_WORKERS, PORTFOLIO_HEAT_MAX,
    VIX_PANIC_THRESHOLD, GAP_PROTECTION, AUTO_MODE_ENABLED,
    MIN_AVG_DOLLAR_VOLUME, MIN_AVG_VOLUME, MAX_RISK_PCT_PER_TRADE,
    MAX_POSITION_PCT_OF_VOLUME, AUTO_TRACK_TOP_N, MAX_SECTOR_EXPOSURE, SECTOR_ETFS, TOP_SECTORS_TO_TRADE,
    DEFAULT_MIN_WIN_RATE_BREAKOUT, DEFAULT_MIN_WIN_RATE_DIP,
    DEFAULT_MIN_PF_BREAKOUT, DEFAULT_MIN_PF_DIP,
    DEFAULT_MIN_TRADES_BREAKOUT, DEFAULT_MIN_TRADES_DIP,
    DEFAULT_MIN_EXPECTANCY_BREAKOUT, DEFAULT_MIN_EXPECTANCY_DIP,
    DEFAULT_MIN_RR_BREAKOUT, DEFAULT_MIN_RR_DIP,
    DEFAULT_REGIME_FACTORS, TRUST_MODE_SETTINGS, SAFE_MODE_SETTINGS,
    V2_OOS_SPLIT_PCT, V2_OOS_MIN_TRADES, V2_OOS_MIN_PF,
    V2_OOS_DECAY_THRESHOLD,
    TRUST_MODE_MAX_POSITIONS, TRUST_MODE_VIX_CAUTION,
    TRUST_MODE_MAX_RISK_PER_TRADE_PCT,
    
    # Classes
    MarketHours, MarketRegime, SectorMapper, EarningsCalendar,
    PortfolioRiskManager, DataValidator, StatisticalConfidenceScorer,
    SignalTracker, TrustModeManager, AutoModeManager,
    StrategyValidator, TrendQualityAnalyzer, Optimizer,
    TitanSetup, RejectionTracker,
    print_trust_mode_header, print_simple_verdict,
    
    # Utils
    atr_series, expectancy, parse_tickers, resolve_output_paths, ensure_multiindex
)
from titan.alpaca_executor import AlpacaExecutor
from titan.market import SectorAnalyzer


def setup_logging(level="INFO"):
    """Configure logging."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s | %(levelname)s | %(message)s"
    )
    return logging.getLogger("titan")

def load_json_file(path, logger=None):
    """Load a JSON dict from disk, returning an empty dict on failure."""
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8-sig") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception as exc:
        if logger:
            logger.warning(f"Failed to read {path}: {exc}")
        return {}


def grade_to_rank(grade):
    """Map letter grades to an integer rank for easy comparison."""
    return {"F": 0, "D": 1, "C": 2, "B": 3, "A": 4}.get(str(grade).upper(), -1)


def get_base_market_status(mkt_status):
    """Normalize a market status like BULL+CAUTION to just BULL."""
    return (mkt_status or "NEUTRAL").split("+", 1)[0]


def build_runtime_settings(args, auto_manager, logger):
    """Merge file config, auto config, CLI, and hard safety overrides."""
    file_config = load_json_file("titan_config.json", logger)
    auto_config = auto_manager.get_config() if auto_manager else {}

    settings = {}
    if args.trust_mode or args.trust_paper:
        settings.update(TRUST_MODE_SETTINGS)
    settings.update(file_config)

    if settings.get("safe_mode"):
        settings.update(SAFE_MODE_SETTINGS)

    configured_account = (
        args.account_size if args.account_size is not None
        else file_config.get("account_size", auto_config.get("account_size", ACCOUNT_SIZE))
    )
    configured_risk = (
        args.risk_per_trade if args.risk_per_trade is not None
        else file_config.get("risk_per_trade", auto_config.get("risk_per_trade", RISK_PER_TRADE))
    )

    account_size = max(float(configured_account or ACCOUNT_SIZE), 1.0)
    max_risk_pct = (
        TRUST_MODE_MAX_RISK_PER_TRADE_PCT
        if (args.trust_mode or args.trust_paper)
        else float(settings.get("max_risk_pct_per_trade", MAX_RISK_PCT_PER_TRADE))
    )
    max_risk_pct = max(0.05, float(max_risk_pct))
    risk_cap = account_size * (max_risk_pct / 100.0)
    runtime_risk = min(max(float(configured_risk or RISK_PER_TRADE), 1.0), risk_cap)

    settings["account_size"] = account_size
    settings["risk_per_trade"] = runtime_risk
    settings["max_risk_pct_per_trade"] = max_risk_pct
    settings["cache_ttl_hours"] = float(settings.get("cache_ttl_hours", DEFAULT_OHLCV_TTL_HOURS))
    settings["sp500_ttl_days"] = float(settings.get("sp500_ttl_days", DEFAULT_SP500_TTL_DAYS))
    settings["force_refresh_cache"] = bool(settings.get("force_refresh_cache", False))
    settings["max_workers"] = int(settings.get("max_workers", DEFAULT_MAX_WORKERS))
    settings["regime_factors"] = settings.get("regime_factors") or DEFAULT_REGIME_FACTORS

    if args.trust_mode or args.trust_paper:
        settings["require_oos"] = True
        settings["min_confidence_grade"] = "B"
        settings["min_momentum_score"] = max(float(settings.get("min_momentum_score", 0)), 40.0)
        settings["min_accumulation_score"] = max(float(settings.get("min_accumulation_score", 0)), 30.0)
        settings["min_rs_percentile"] = max(float(settings.get("min_rs_percentile", 0)), 55.0)
        settings["max_alloc_pct_per_position"] = min(
            float(settings.get("max_alloc_pct_per_position", 12.0)), 12.0
        )
        settings["max_live_positions"] = min(
            int(settings.get("max_live_positions", TRUST_MODE_MAX_POSITIONS)),
            TRUST_MODE_MAX_POSITIONS,
            4,
        )
        settings["max_new_orders_per_run"] = min(
            int(settings.get("max_new_orders_per_run", AUTO_TRACK_TOP_N)),
            AUTO_TRACK_TOP_N,
            3,
        )
    else:
        settings["min_confidence_grade"] = settings.get("min_confidence_grade", "C")
        settings["min_momentum_score"] = float(settings.get("min_momentum_score", 0))
        settings["min_accumulation_score"] = float(settings.get("min_accumulation_score", 0))
        settings["min_rs_percentile"] = float(settings.get("min_rs_percentile", 0))
        settings["max_alloc_pct_per_position"] = min(
            float(settings.get("max_alloc_pct_per_position", 15.0)), 15.0
        )
        settings["max_live_positions"] = min(
            int(settings.get("max_live_positions", MAX_POSITIONS)),
            MAX_POSITIONS,
        )
        settings["max_new_orders_per_run"] = min(
            int(settings.get("max_new_orders_per_run", MAX_POSITIONS)),
            MAX_POSITIONS,
        )

    settings["fallback_stop_pct"] = float(settings.get("fallback_stop_pct", 7.0))
    settings["vix_caution_size_scalar"] = float(settings.get("vix_caution_size_scalar", 0.5))
    settings["execution_buying_power_buffer_pct"] = float(
        settings.get("execution_buying_power_buffer_pct", 2.0)
    )
    settings["max_portfolio_heat_pct"] = float(
        settings.get("max_portfolio_heat_pct", PORTFOLIO_HEAT_MAX)
    )
    settings["allow_early_entries"] = bool(settings.get("allow_early_entries", True))
    settings["disable_dip_in_weak_regimes"] = bool(
        settings.get("disable_dip_in_weak_regimes", True)
    )
    settings["prebreakout_max_distance_pct"] = float(
        settings.get("prebreakout_max_distance_pct", 4.0)
    )
    settings["min_prebreakout_score"] = float(
        settings.get(
            "min_prebreakout_score",
            60.0 if (args.trust_mode or args.trust_paper) else 52.0,
        )
    )
    settings["starter_entry_size_pct"] = min(
        max(float(settings.get("starter_entry_size_pct", 0.35)), 0.20), 0.50
    )
    settings["partial_exit_multiple"] = float(settings.get("partial_exit_multiple", 1.5))
    settings["final_target_multiple"] = float(settings.get("final_target_multiple", 3.2))
    settings["require_oos"] = bool(settings.get("require_oos", True) or not (args.trust_mode or args.trust_paper))
    settings["top_sectors_to_trade"] = max(
        1, int(settings.get("top_sectors_to_trade", TOP_SECTORS_TO_TRADE))
    )
    settings["sector_lookback_days"] = max(
        5, int(settings.get("sector_lookback_days", 20))
    )
    settings["require_top_sector_alignment"] = bool(
        settings.get("require_top_sector_alignment", True)
    )
    settings["breakout_high_proximity_pct"] = max(
        3.0, float(settings.get("breakout_high_proximity_pct", 12.0))
    )
    settings["manual_top_n"] = max(5, int(settings.get("manual_top_n", 12)))
    settings["validation_cost_bps"] = max(0.0, float(settings.get("validation_cost_bps", 10.0)))
    settings["validation_slippage_bps"] = max(0.0, float(settings.get("validation_slippage_bps", 5.0)))
    settings["require_walkforward"] = bool(settings.get("require_walkforward", True) or not (args.trust_mode or args.trust_paper))
    settings["wf_folds"] = max(2, int(settings.get("wf_folds", 3)))
    settings["wf_test_ratio"] = min(max(float(settings.get("wf_test_ratio", 0.2)), 0.10), 0.30)
    settings["wf_min_trades_breakout"] = max(3, int(settings.get("wf_min_trades_breakout", 5)))
    settings["wf_min_trades_dip"] = max(5, int(settings.get("wf_min_trades_dip", 20)))
    settings["wf_min_pf_breakout"] = max(0.8, float(settings.get("wf_min_pf_breakout", 1.0)))
    settings["wf_min_pf_dip"] = max(0.8, float(settings.get("wf_min_pf_dip", 1.0)))
    settings["wf_min_expectancy_breakout"] = float(settings.get("wf_min_expectancy_breakout", 0.0))
    settings["wf_min_expectancy_dip"] = float(settings.get("wf_min_expectancy_dip", 0.0))
    settings["wf_min_passrate_breakout"] = min(max(float(settings.get("wf_min_passrate_breakout", 0.34)), 0.0), 1.0)
    settings["wf_min_passrate_dip"] = min(max(float(settings.get("wf_min_passrate_dip", 0.50)), 0.0), 1.0)
    settings["regime_min_score_breakout"] = min(max(float(settings.get("regime_min_score_breakout", 0.33)), 0.0), 1.0)
    settings["regime_min_score_dip"] = min(max(float(settings.get("regime_min_score_dip", 0.34)), 0.0), 1.0)
    settings["min_robustness_score"] = min(max(float(settings.get("min_robustness_score", 55.0)), 0.0), 100.0)
    settings["min_early_entry_robustness"] = min(max(float(settings.get("min_early_entry_robustness", 62.0)), 0.0), 100.0)

    return settings


def print_runtime_summary(settings, trust_mode=False):
    """Print the live risk profile so the user can see what will be enforced."""
    mode_label = "TRUST" if trust_mode else "SCAN"
    print(f"\n  Runtime ({mode_label}) Settings:")
    print(
        f"    Account: ${settings['account_size']:,.0f} | "
        f"Risk/Trade: ${settings['risk_per_trade']:,.0f} "
        f"({settings['max_risk_pct_per_trade']:.2f}% cap)"
    )
    print(
        f"    OOS Required: {'YES' if settings.get('require_oos') else 'NO'} | "
        f"Min Grade: {settings.get('min_confidence_grade', 'C')} | "
        f"Max Alloc: {settings.get('max_alloc_pct_per_position', 0):.1f}%"
    )
    print(
        f"    Early Entries: {'ON' if settings.get('allow_early_entries') else 'OFF'} | "
        f"Dip Safety: {'ON' if settings.get('disable_dip_in_weak_regimes') else 'OFF'}"
    )
    print(
        f"    Sector Filter: Top {settings.get('top_sectors_to_trade', TOP_SECTORS_TO_TRADE)} | "
        f"Near High <= {settings.get('breakout_high_proximity_pct', 12.0):.0f}%"
    )
    print(
        f"    WF Required: {'YES' if settings.get('require_walkforward') else 'NO'} | "
        f"Robustness >= {settings.get('min_robustness_score', 55.0):.0f} | "
        f"Validation Costs: {settings.get('validation_cost_bps', 10.0) + settings.get('validation_slippage_bps', 5.0):.0f}bps"
    )


def manual_strategy_priority(setup):
    """Rank earlier, leadership-style entries ahead of reactive setups."""
    strategy = getattr(setup, 'strategy', '')
    if strategy == 'EARLY BO':
        return 3
    if strategy == 'BREAKOUT':
        return 2
    return 1


def manual_action_label(setup):
    """Create a plain-English action label for discretionary trading."""
    if getattr(setup, 'strategy', '') == 'EARLY BO':
        return 'BUY STARTER'
    if getattr(setup, 'strategy', '') == 'BREAKOUT':
        return 'BUY BO'
    return 'BUY SUPPORT'


def print_manual_trade_board(setups, settings, top_sectors=None):
    """Print a manual-first trade board with robustness evidence."""
    if not setups:
        return

    top_n = max(int(settings.get('manual_top_n', 12)), 1)
    top_sectors = top_sectors or []

    print("\n  Manual Trade Board:\n")
    table = []
    for s in setups[:top_n]:
        add_on = '-'
        if getattr(s, 'add_on_trigger', 0.0) > 0 and getattr(s, 'add_on_qty', 0) > 0:
            add_on = f"${s.add_on_trigger:.2f}"
        sector_tag = 'TOP' if top_sectors and getattr(s, 'sector', '') in top_sectors else '-'
        targets = f"${getattr(s, 'partial_target', s.target):.2f}/${s.target:.2f}"
        oos_tag = '-'
        if getattr(s, 'oos_trades', 0) > 0:
            oos_tag = f"{getattr(s, 'oos_pf', 0.0):.2f}/{int(getattr(s, 'oos_trades', 0))}"
        wf_tag = '-'
        if getattr(s, 'walk_forward_trades', 0) > 0:
            wf_tag = f"{getattr(s, 'walk_forward_pf', 0.0):.2f}/{getattr(s, 'walk_forward_pass_rate', 0.0) * 100:.0f}%"
        table.append([
            s.ticker,
            manual_action_label(s),
            s.confidence_grade,
            sector_tag,
            f"${s.trigger:.2f}",
            add_on,
            f"${s.stop:.2f}",
            targets,
            int(getattr(s, 'planned_total_qty', s.qty) or s.qty),
            f"{s.rs_percentile:.0f}",
            oos_tag,
            wf_tag,
            f"{getattr(s, 'regime_score', 0.0) * 100:.0f}%",
            f"{getattr(s, 'robustness_score', 0.0):.0f}",
        ])

    print(tabulate(
        table,
        headers=[
            'Ticker', 'Action', 'Grade', 'Top?', 'Entry', 'Add',
            'Stop', 'PT1/TGT', 'Qty', 'RS', 'OOS', 'WF', 'Reg', 'Rob'
        ],
        tablefmt='grid'
    ))
    print("\n  OOS = net PF / trades, WF = net PF / pass rate. Skip anything that gaps or extends materially above the planned entry.")

def apply_validation_costs(trades, settings):
    """Apply a flat round-trip cost model to historical trades."""
    if not trades:
        return []
    total_cost = (
        float(settings.get('validation_cost_bps', 10.0)) +
        float(settings.get('validation_slippage_bps', 5.0))
    ) / 10000.0
    return [float(t) - total_cost for t in trades]


def compute_trade_stats(trades):
    """Compute net trade statistics from a list of percentage returns."""
    if not trades:
        return {
            'win_rate': 0.0,
            'pf': 0.0,
            'trades': 0,
            'avg_win': 0.0,
            'avg_loss': 0.0,
            'expectancy': 0.0,
            'median': 0.0,
            'std': 0.0,
            'gross_win': 0.0,
            'gross_loss': 0.0,
            'sum_pnl': 0.0,
            'trades_list': [],
        }

    wins = [t for t in trades if t > 0]
    losses = [t for t in trades if t <= 0]
    gross_win = float(sum(wins))
    gross_loss = float(abs(sum(losses)))
    return {
        'win_rate': float(len(wins) / len(trades) * 100),
        'pf': float(gross_win / gross_loss if gross_loss > 0 else (100.0 if gross_win > 0 else 0.0)),
        'trades': len(trades),
        'avg_win': float(np.mean(wins)) if wins else 0.0,
        'avg_loss': float(np.mean(losses)) if losses else 0.0,
        'expectancy': float(np.mean(trades)) if trades else 0.0,
        'median': float(np.median(trades)) if trades else 0.0,
        'std': float(np.std(trades)) if trades else 0.0,
        'gross_win': gross_win,
        'gross_loss': gross_loss,
        'sum_pnl': float(sum(trades)),
        'trades_list': list(trades),
    }


def get_adjusted_backtest_stats(backtest_res, settings):
    """Convert a raw backtest result into cost-adjusted trade statistics."""
    return compute_trade_stats(apply_validation_costs(backtest_res.get('trades_list', []), settings))


def get_regime_label(spy_df):
    """Classify a historical SPY window into a simple regime label."""
    if spy_df is None or spy_df.empty or 'Close' not in spy_df:
        return 'NEUTRAL'
    closes = spy_df['Close']
    if len(closes) < 120:
        return 'NEUTRAL'
    sma50 = closes.rolling(50).mean().iloc[-1]
    sma200 = closes.rolling(200).mean().iloc[-1] if len(closes) >= 200 else np.nan
    current_price = closes.iloc[-1]
    if pd.notna(sma200) and current_price > sma200 and sma50 > sma200:
        return 'BULL'
    if pd.notna(sma200) and current_price < sma200:
        return 'BEAR'
    return 'NEUTRAL'


def get_regime_segments(df, segments=3):
    """Split the recent history into coarse regime buckets."""
    n = len(df)
    if n < 120 * segments:
        return []
    seg_len = max(120, n // segments)
    total_len = seg_len * segments
    start = max(n - total_len, 0)
    return [(start + i * seg_len, start + (i + 1) * seg_len) for i in range(segments)]


def evaluate_regime_stability(df, spy_df, settings, is_breakout):
    """Check whether a strategy stays profitable across broad historical regimes."""
    segments = get_regime_segments(df, segments=3)
    if not segments:
        return {'score': 0.0, 'count': 0, 'labels': ''}

    min_trades = int(settings.get('wf_min_trades_breakout' if is_breakout else 'wf_min_trades_dip', 5))
    min_pf = float(settings.get('wf_min_pf_breakout' if is_breakout else 'wf_min_pf_dip', 1.0))
    min_exp = float(settings.get('wf_min_expectancy_breakout' if is_breakout else 'wf_min_expectancy_dip', 0.0))
    passed = 0
    labels = []

    for start, end in segments:
        seg_df = df.iloc[start:end]
        seg_spy = spy_df.iloc[start:end] if spy_df is not None else None
        labels.append(get_regime_label(seg_spy))
        validator = StrategyValidator(seg_df)
        seg_res = validator.backtest_breakout(return_trades=True) if is_breakout else validator.backtest_dip(return_trades=True)
        seg_stats = get_adjusted_backtest_stats(seg_res, settings)
        if seg_stats['trades'] >= min_trades and seg_stats['pf'] >= min_pf and seg_stats['expectancy'] > min_exp:
            passed += 1

    return {
        'score': float(passed / len(segments)) if segments else 0.0,
        'count': len(segments),
        'labels': ','.join(labels),
    }


def evaluate_walk_forward_robustness(df, settings, is_breakout):
    """Measure whether the edge survives repeated forward windows."""
    folds = int(settings.get('wf_folds', 3))
    test_ratio = float(settings.get('wf_test_ratio', 0.2))
    min_trades = int(settings.get('wf_min_trades_breakout' if is_breakout else 'wf_min_trades_dip', 5))
    min_pf = float(settings.get('wf_min_pf_breakout' if is_breakout else 'wf_min_pf_dip', 1.0))
    min_exp = float(settings.get('wf_min_expectancy_breakout' if is_breakout else 'wf_min_expectancy_dip', 0.0))

    n = len(df)
    test_len = max(120, int(n * test_ratio))
    if n < test_len * folds + 120:
        return {'folds': 0, 'pass_rate': 0.0, 'trades': 0, 'pf': 0.0, 'expectancy': 0.0}

    fold_stats = []
    all_trades = []
    for i in range(folds):
        test_start = n - (folds - i) * test_len
        test_end = min(test_start + test_len, n)
        train_df = df.iloc[:test_start]
        test_df = df.iloc[test_start:test_end]
        if len(train_df) < 120 or len(test_df) < 120:
            continue
        validator = StrategyValidator(test_df)
        test_res = validator.backtest_breakout(return_trades=True) if is_breakout else validator.backtest_dip(return_trades=True)
        stats = get_adjusted_backtest_stats(test_res, settings)
        fold_stats.append(stats)
        all_trades.extend(stats['trades_list'])

    if not fold_stats:
        return {'folds': 0, 'pass_rate': 0.0, 'trades': 0, 'pf': 0.0, 'expectancy': 0.0}

    eligible = [stats for stats in fold_stats if stats['trades'] >= min_trades]
    passed = [
        stats for stats in eligible
        if stats['pf'] >= min_pf and stats['expectancy'] > min_exp
    ]
    aggregate = compute_trade_stats(all_trades)
    return {
        'folds': len(fold_stats),
        'pass_rate': float(len(passed) / len(eligible)) if eligible else 0.0,
        'trades': int(aggregate['trades']),
        'pf': float(aggregate['pf']),
        'expectancy': float(aggregate['expectancy']),
    }


def calculate_robustness_score(net_stats, oos_stats, wf_stats, regime_stats, is_breakout):
    """Blend net, OOS, walk-forward, and regime evidence into a 0-100 score."""
    score = 0.0
    trades = int(net_stats.get('trades', 0))
    if is_breakout:
        if trades >= 20:
            score += 15
        elif trades >= 10:
            score += 10
        elif trades >= 5:
            score += 6
    else:
        if trades >= 80:
            score += 15
        elif trades >= 40:
            score += 10
        elif trades >= 20:
            score += 6

    pf = float(net_stats.get('pf', 0.0))
    if pf >= 1.8:
        score += 20
    elif pf >= 1.4:
        score += 14
    elif pf >= 1.2:
        score += 8

    expectancy_value = float(net_stats.get('expectancy', 0.0))
    if expectancy_value >= 0.010:
        score += 15
    elif expectancy_value >= 0.005:
        score += 10
    elif expectancy_value > 0:
        score += 5

    oos_pf = float(oos_stats.get('pf', 0.0))
    oos_trades = int(oos_stats.get('trades', 0))
    if oos_pf >= 1.5:
        score += 15
    elif oos_pf >= 1.2:
        score += 10
    elif oos_pf >= 1.0:
        score += 5
    if oos_trades >= 10:
        score += 10
    elif oos_trades >= 5:
        score += 6
    elif oos_trades >= 3:
        score += 3

    wf_pass_rate = float(wf_stats.get('pass_rate', 0.0))
    wf_pf = float(wf_stats.get('pf', 0.0))
    if wf_pass_rate >= 0.75:
        score += 15
    elif wf_pass_rate >= 0.50:
        score += 10
    elif wf_pass_rate > 0:
        score += 4
    if wf_pf >= 1.5:
        score += 10
    elif wf_pf >= 1.2:
        score += 6
    elif wf_pf >= 1.0:
        score += 3

    regime_score = float(regime_stats.get('score', 0.0))
    if regime_score >= 0.67:
        score += 10
    elif regime_score >= 0.34:
        score += 6
    elif regime_score > 0:
        score += 3

    return min(100.0, score)

def load_open_positions_snapshot(executor=None, settings=None):
    """Build a position snapshot from Alpaca plus local portfolio state."""
    snapshot = {}
    settings = settings or {}

    if executor and executor.is_connected():
        snapshot.update(executor.get_open_positions_snapshot())

    saved_portfolio = load_json_file(PORTFOLIO_FILE)
    for symbol, pos in saved_portfolio.items():
        existing = snapshot.setdefault(symbol, {})
        existing["entry_price"] = float(pos.get("entry_price", existing.get("entry_price", 0.0)) or 0.0)
        existing["shares"] = int(pos.get("shares", existing.get("shares", 0)) or 0)
        stop_loss = float(pos.get("stop_loss", existing.get("stop_loss", 0.0)) or 0.0)
        if stop_loss > 0:
            existing["stop_loss"] = stop_loss
        target = float(pos.get("target", existing.get("target", 0.0)) or 0.0)
        if target > 0:
            existing["target"] = target

    fallback_stop_pct = max(float(settings.get("fallback_stop_pct", 7.0)), 0.5) / 100.0
    for symbol, pos in snapshot.items():
        entry_price = float(pos.get("entry_price", 0.0) or 0.0)
        shares = int(pos.get("shares", 0) or 0)
        if shares <= 0 or entry_price <= 0:
            continue
        if float(pos.get("stop_loss", 0.0) or 0.0) <= 0:
            pos["stop_loss"] = round(entry_price * (1.0 - fallback_stop_pct), 2)

    return snapshot


def sync_trust_positions(trust_manager, position_count):
    """Keep trust state aligned with actual open positions."""
    if trust_manager.state.get("current_positions") != position_count:
        trust_manager.state["current_positions"] = position_count
        trust_manager._save_state()


def filter_execution_candidates(setups, settings):
    """Apply stricter live-trading filters than the scan display uses."""
    min_grade = grade_to_rank(settings.get("min_confidence_grade", "C"))
    min_mom = float(settings.get("min_momentum_score", 0))
    min_acc = float(settings.get("min_accumulation_score", 0))
    min_rs = float(settings.get("min_rs_percentile", 0))

    selected = []
    skipped = []
    for s in setups:
        reasons = []
        if grade_to_rank(getattr(s, "confidence_grade", "F")) < min_grade:
            reasons.append(f"grade<{settings.get('min_confidence_grade', 'C')}")
        if getattr(s, "momentum_score", 0.0) < min_mom:
            reasons.append(f"mom<{min_mom:.0f}")
        if getattr(s, "accumulation_score", 0.0) < min_acc:
            reasons.append(f"acc<{min_acc:.0f}")
        if getattr(s, "rs_percentile", 0.0) < min_rs:
            reasons.append(f"rs<{min_rs:.0f}")
        if reasons:
            skipped.append((s.ticker, ", ".join(reasons)))
            continue
        selected.append(s)

    return selected, skipped


def calculate_execution_qty(setup, settings, buying_power, current_heat_pct, vix_level=None):
    """Clamp a setup quantity to what can safely be executed right now."""
    qty = int(getattr(setup, "qty", 0) or 0)
    if qty <= 0:
        return 0, "no shares"

    if vix_level and vix_level > TRUST_MODE_VIX_CAUTION:
        qty = max(1, int(qty * float(settings.get("vix_caution_size_scalar", 0.5))))

    max_alloc_pct = float(settings.get("max_alloc_pct_per_position", 0))
    if max_alloc_pct > 0 and getattr(setup, "trigger", 0) > 0:
        alloc_cap = int((settings["account_size"] * (max_alloc_pct / 100.0)) / setup.trigger)
        qty = min(qty, alloc_cap)

    power_buffer = max(float(settings.get("execution_buying_power_buffer_pct", 2.0)), 0.0) / 100.0
    if buying_power > 0 and getattr(setup, "trigger", 0) > 0:
        max_bp_shares = int((buying_power * (1.0 - power_buffer)) / setup.trigger)
        qty = min(qty, max_bp_shares)

    risk_per_share = float(getattr(setup, "trigger", 0) - getattr(setup, "stop", 0))
    if risk_per_share <= 0:
        return 0, "invalid stop"

    remaining_heat_pct = float(settings.get("max_portfolio_heat_pct", PORTFOLIO_HEAT_MAX)) - current_heat_pct
    if remaining_heat_pct <= 0:
        return 0, "portfolio heat exhausted"

    max_heat_shares = int((settings["account_size"] * (remaining_heat_pct / 100.0)) / risk_per_share)
    qty = min(qty, max_heat_shares)

    return max(qty, 0), ("ok" if qty > 0 else "sizing clamp")


def analyze_pre_breakout(df, validator, current_price, atr_value, sma50, sma200,
                         accum_score, mom_score, rs_pct, settings):
    """Score tight pre-breakout structures for earlier starter entries."""
    ema10 = float(df['Close'].ewm(span=10, adjust=False).mean().iloc[-1])
    ema21 = float(df['Close'].ewm(span=21, adjust=False).mean().iloc[-1])
    recent_high_15 = float(df['High'].iloc[-16:-1].max())
    recent_high_10 = float(df['High'].iloc[-10:].max())
    recent_low_10 = float(df['Low'].iloc[-10:].min())
    recent_high_30 = float(df['High'].iloc[-30:].max())
    recent_low_30 = float(df['Low'].iloc[-30:].min())
    pivot_trigger = recent_high_15 + 0.02
    dist_to_pivot_pct = max(pivot_trigger - current_price, 0.0) / max(current_price, 1e-9) * 100
    range_10 = (recent_high_10 - recent_low_10) / max(current_price, 1e-9)
    range_30 = (recent_high_30 - recent_low_30) / max(current_price, 1e-9)
    compression_ratio = range_10 / max(range_30, 1e-9)
    avwap = validator.calculate_anchored_vwap(126)
    avwap_distance = ((current_price - avwap) / avwap) if avwap else 0.0

    score = 0.0
    if dist_to_pivot_pct <= settings.get('prebreakout_max_distance_pct', 4.0):
        score += 15
    elif dist_to_pivot_pct <= settings.get('prebreakout_max_distance_pct', 4.0) + 1.5:
        score += 8

    if range_10 <= 0.08:
        score += 15
    elif range_10 <= 0.10:
        score += 8

    if compression_ratio <= 0.65:
        score += 15
    elif compression_ratio <= 0.80:
        score += 8

    if current_price >= ema21:
        score += 8
    if avwap is not None:
        if current_price >= avwap * 0.99:
            score += 10
        elif current_price >= avwap * 0.97:
            score += 5

    if rs_pct >= 80:
        score += 15
    elif rs_pct >= 70:
        score += 10
    elif rs_pct >= 60:
        score += 5

    if accum_score >= 65:
        score += 12
    elif accum_score >= 50:
        score += 8
    elif accum_score >= 40:
        score += 4

    if mom_score >= 65:
        score += 10
    elif mom_score >= 50:
        score += 6
    elif mom_score >= 40:
        score += 3

    if recent_high_30 > 0 and current_price >= recent_high_30 * 0.95:
        score += 5

    early_entry_ready = all([
        settings.get('allow_early_entries', True),
        current_price > sma50 > sma200,
        current_price >= ema21 * 0.99,
        dist_to_pivot_pct <= settings.get('prebreakout_max_distance_pct', 4.0),
        range_10 <= 0.09,
        compression_ratio <= 0.80,
        rs_pct >= max(60.0, float(settings.get('min_rs_percentile', 0))),
        accum_score >= max(45.0, float(settings.get('min_accumulation_score', 0))),
        score >= float(settings.get('min_prebreakout_score', 55.0)),
    ])

    support_floor = min(recent_low_10 - (atr_value * 0.25), current_price - (atr_value * 1.75))
    if avwap is not None:
        support_floor = min(support_floor, avwap - (atr_value * 0.60))
    support_floor = min(support_floor, ema21 - (atr_value * 0.75))
    if support_floor >= current_price:
        support_floor = current_price - (atr_value * 1.5)

    return {
        'score': min(100.0, score),
        'pivot_trigger': pivot_trigger,
        'starter_trigger': round(current_price, 2),
        'starter_stop': round(support_floor, 2),
        'ema10': ema10,
        'ema21': ema21,
        'range_10': range_10,
        'compression_ratio': compression_ratio,
        'avwap': avwap,
        'avwap_distance': avwap_distance,
        'dist_to_pivot_pct': dist_to_pivot_pct,
        'early_entry_ready': early_entry_ready,
    }

def load_managed_portfolio(logger=None):
    """Load managed positions from disk and normalize legacy entries."""
    raw = load_json_file(PORTFOLIO_FILE, logger)
    portfolio = {}
    for symbol, payload in raw.items():
        portfolio[symbol] = normalize_managed_position(symbol, payload)
    return portfolio


def normalize_managed_position(symbol, payload):
    """Normalize legacy portfolio rows into the managed-position schema."""
    payload = dict(payload or {})
    entry_price = float(payload.get('entry_price', 0.0) or 0.0)
    shares = int(payload.get('shares', 0) or 0)
    starter_qty = int(payload.get('starter_qty', shares or payload.get('planned_total_qty', 0) or 0))
    planned_total_qty = int(payload.get('planned_total_qty', shares or starter_qty or 0))
    if shares > planned_total_qty:
        planned_total_qty = shares
    add_on_qty = int(payload.get('add_on_qty', max(planned_total_qty - starter_qty, 0)) or 0)
    if starter_qty < 1 and shares > 0:
        starter_qty = shares

    normalized = {
        'ticker': symbol,
        'status': payload.get('status') or ('OPEN' if shares > 0 else 'PENDING_ENTRY'),
        'created_at': payload.get('created_at') or payload.get('entry_date') or datetime.now().isoformat(),
        'updated_at': payload.get('updated_at') or datetime.now().isoformat(),
        'strategy': payload.get('strategy', ''),
        'entry_price': entry_price,
        'starter_entry_price': float(payload.get('starter_entry_price', payload.get('starter_trigger', entry_price)) or 0.0),
        'shares': shares,
        'starter_qty': starter_qty,
        'planned_total_qty': planned_total_qty,
        'add_on_qty': add_on_qty,
        'add_on_trigger': float(payload.get('add_on_trigger', 0.0) or 0.0),
        'add_on_filled': bool(payload.get('add_on_filled', False)),
        'add_on_order_pending': bool(payload.get('add_on_order_pending', False)),
        'partial_target': float(payload.get('partial_target', 0.0) or 0.0),
        'partial_taken': bool(payload.get('partial_taken', False)),
        'partial_order_pending': bool(payload.get('partial_order_pending', False)),
        'target': float(payload.get('target', 0.0) or 0.0),
        'stop_loss': float(payload.get('stop_loss', 0.0) or 0.0),
        'initial_stop': float(payload.get('initial_stop', payload.get('stop_loss', 0.0)) or 0.0),
        'breakeven_trigger': float(payload.get('breakeven_trigger', 0.0) or 0.0),
        'trailing_stop': float(payload.get('trailing_stop', 0.0) or 0.0),
        'highest_price': float(payload.get('highest_price', entry_price or payload.get('starter_entry_price', 0.0)) or 0.0),
        'entry_order_pending': bool(payload.get('entry_order_pending', shares == 0)),
        'entry_order_price': float(payload.get('entry_order_price', payload.get('starter_entry_price', 0.0)) or 0.0),
        'note': payload.get('note', ''),
        'confidence_grade': payload.get('confidence_grade', 'F'),
        'avwap_distance': float(payload.get('avwap_distance', 0.0) or 0.0),
        'realized_partial_qty': int(payload.get('realized_partial_qty', 0) or 0),
    }
    if normalized['starter_entry_price'] <= 0:
        normalized['starter_entry_price'] = entry_price
    if normalized['stop_loss'] <= 0 and entry_price > 0:
        normalized['stop_loss'] = round(entry_price * 0.93, 2)
    if normalized['initial_stop'] <= 0:
        normalized['initial_stop'] = normalized['stop_loss']
    if normalized['partial_target'] <= 0:
        normalized['partial_target'] = 0.0
    if normalized['target'] <= 0:
        normalized['target'] = float(payload.get('target', 0.0) or 0.0)
    if normalized['shares'] >= normalized['planned_total_qty'] and normalized['planned_total_qty'] > 0:
        normalized['add_on_filled'] = True
    return normalized


def save_managed_portfolio(portfolio, logger=None):
    """Persist managed portfolio state to disk."""
    out = {}
    for symbol, payload in portfolio.items():
        row = dict(payload)
        row['updated_at'] = datetime.now().isoformat()
        out[symbol] = row
    try:
        with open(PORTFOLIO_FILE, 'w') as f:
            json.dump(out, f, indent=4)
    except Exception as exc:
        if logger:
            logger.warning(f"Failed to write {PORTFOLIO_FILE}: {exc}")


def build_price_lookup(data, symbols):
    """Return latest close price for each symbol found in the market data bundle."""
    prices = {}
    if not isinstance(data.columns, pd.MultiIndex):
        return prices
    for symbol in symbols:
        try:
            if symbol in data.columns.levels[0]:
                close = data[symbol]['Close'].dropna()
                if len(close) > 0:
                    prices[symbol] = float(close.iloc[-1])
        except Exception:
            pass
    return prices


def compute_scaled_plan_from_qty(setup, starter_qty):
    """Scale the planned total and add-on size to the actually submitted starter size."""
    starter_qty = int(starter_qty)
    starter_pct = float(getattr(setup, 'starter_size_pct', 1.0) or 1.0)
    starter_pct = min(max(starter_pct, 0.05), 1.0)
    if starter_pct >= 0.999:
        planned_total = starter_qty
    else:
        planned_total = max(starter_qty, int(round(starter_qty / starter_pct)))
    add_on_qty = max(planned_total - starter_qty, 0)
    return planned_total, add_on_qty


def create_managed_position_from_setup(setup, starter_qty):
    """Create the managed portfolio row for a staged entry."""
    planned_total, add_on_qty = compute_scaled_plan_from_qty(setup, starter_qty)
    return normalize_managed_position(setup.ticker, {
        'status': 'PENDING_ENTRY',
        'created_at': datetime.now().isoformat(),
        'strategy': setup.strategy,
        'entry_price': 0.0,
        'starter_entry_price': round(setup.trigger, 2),
        'shares': 0,
        'starter_qty': int(starter_qty),
        'planned_total_qty': int(planned_total),
        'add_on_qty': int(add_on_qty),
        'add_on_trigger': round(getattr(setup, 'add_on_trigger', 0.0), 2),
        'partial_target': round(getattr(setup, 'partial_target', 0.0), 2),
        'target': round(setup.target, 2),
        'stop_loss': round(setup.stop, 2),
        'initial_stop': round(setup.stop, 2),
        'breakeven_trigger': round(getattr(setup, 'breakeven_trigger', 0.0), 2),
        'trailing_stop': round(getattr(setup, 'trailing_stop', 0.0), 2),
        'highest_price': round(setup.price, 2),
        'entry_order_pending': True,
        'entry_order_price': round(setup.trigger, 2),
        'note': setup.note,
        'confidence_grade': getattr(setup, 'confidence_grade', 'F'),
        'avwap_distance': float(getattr(setup, 'avwap_distance', 0.0)),
    })


def choose_exit_order_type():
    """Use market orders during regular hours and limit orders otherwise."""
    return 'market' if MarketHours.is_market_open() else 'limit'


def submit_exit_order(executor, symbol, qty, reference_price):
    """Submit an exit order using market hours-aware behavior."""
    if qty <= 0 or not executor or not executor.is_connected():
        return False
    if choose_exit_order_type() == 'market':
        return executor.submit_market_order(symbol, qty, side='sell')
    limit_price = max(reference_price * 0.995, 0.01)
    return executor.submit_limit_order(symbol, qty, side='sell', limit_price=limit_price)


def sync_protective_stop(symbol, entry, shares, executor):
    """Ensure exactly one protective stop exists for a managed open position."""
    if not executor or not executor.is_connected() or shares <= 0:
        return False
    desired_stop = round(float(entry.get('stop_loss', 0.0) or 0.0), 2)
    if desired_stop <= 0:
        return False
    stop_orders = [
        o for o in executor.get_open_orders(symbol=symbol, side='sell')
        if o.get('type') == 'stop'
    ]
    for order in stop_orders:
        if order.get('qty') == shares and abs((order.get('stop_price') or 0.0) - desired_stop) < 0.02:
            return False
    executor.cancel_orders_for_symbol(symbol, side='sell', order_type='stop')
    return executor.submit_stop_order(symbol, shares, desired_stop)


def manage_open_positions(executor, data, settings, logger=None):
    """Manage staged entries and exits for open portfolio positions."""
    portfolio = load_managed_portfolio(logger)
    if not portfolio:
        return portfolio, []

    prices = build_price_lookup(data, list(portfolio.keys()))
    if executor and executor.is_connected():
        live_positions = executor.get_open_positions_snapshot()
    else:
        live_positions = {
            symbol: {
                'shares': int(entry.get('shares', 0) or 0),
                'entry_price': float(entry.get('entry_price', 0.0) or 0.0),
            }
            for symbol, entry in portfolio.items()
            if int(entry.get('shares', 0) or 0) > 0
        }
    market_status = get_base_market_status(MarketRegime(data).analyze_spy()[0])
    open_orders_by_symbol = {}
    if executor and executor.is_connected():
        for order in executor.get_open_orders():
            open_orders_by_symbol.setdefault(order['symbol'], []).append(order)

    actions = []
    removals = []

    for symbol, entry in portfolio.items():
        current_price = prices.get(symbol, float(entry.get('highest_price', entry.get('starter_entry_price', 0.0)) or 0.0))
        broker_pos = live_positions.get(symbol, {})
        shares = int(broker_pos.get('shares', entry.get('shares', 0)) or 0)
        if current_price > 0:
            entry['highest_price'] = max(float(entry.get('highest_price', 0.0) or 0.0), current_price)

        if shares > 0:
            entry['status'] = 'OPEN'
            entry['shares'] = shares
            if broker_pos.get('entry_price', 0) > 0:
                entry['entry_price'] = round(float(broker_pos['entry_price']), 2)
            entry['entry_order_pending'] = False
            if shares > int(entry.get('starter_qty', 0) or 0):
                entry['add_on_filled'] = True
                entry['add_on_order_pending'] = False

            breakeven_trigger = float(entry.get('breakeven_trigger', 0.0) or 0.0)
            if breakeven_trigger > 0 and current_price >= breakeven_trigger and entry.get('entry_price', 0.0) > 0:
                entry['stop_loss'] = max(float(entry.get('stop_loss', 0.0) or 0.0), float(entry['entry_price']))

            if entry.get('partial_taken') and current_price > 0:
                trail_amount = float(entry.get('trailing_stop', 0.0) or 0.0)
                if trail_amount > 0:
                    trail_stop = current_price - trail_amount
                    if entry.get('entry_price', 0.0) > 0:
                        trail_stop = max(trail_stop, float(entry['entry_price']))
                    entry['stop_loss'] = max(float(entry.get('stop_loss', 0.0) or 0.0), round(trail_stop, 2))

            if (not entry.get('add_on_filled')) and int(entry.get('add_on_qty', 0) or 0) > 0:
                add_on_trigger = float(entry.get('add_on_trigger', 0.0) or 0.0)
                existing_buys = [o for o in open_orders_by_symbol.get(symbol, []) if o.get('side') == 'buy']
                if current_price >= add_on_trigger > 0 and market_status in ('STRONG_BULL', 'BULL', 'RECOVERY'):
                    if MarketHours.is_market_open() and current_price <= add_on_trigger * 1.02 and not existing_buys and executor and executor.is_connected():
                        buy_limit = min(round(current_price * 1.002, 2), round(add_on_trigger * 1.01, 2))
                        if executor.submit_limit_order(symbol, int(entry['add_on_qty']), side='buy', limit_price=buy_limit):
                            entry['add_on_order_pending'] = True
                            actions.append(f"{symbol}: submitted add-on for {entry['add_on_qty']} @ ${buy_limit:.2f}")
                    elif current_price > add_on_trigger * 1.02:
                        entry['add_on_order_pending'] = False
                        entry['add_on_filled'] = False
                        actions.append(f"{symbol}: skipped add-on, price too extended above trigger")
                    elif not MarketHours.is_market_open():
                        actions.append(f"{symbol}: add-on armed for next market-open run")

            if (not entry.get('partial_taken')) and float(entry.get('partial_target', 0.0) or 0.0) > 0 and current_price >= float(entry['partial_target']):
                qty_to_sell = max(1, shares // 2)
                if qty_to_sell < shares and executor and executor.is_connected() and MarketHours.is_market_open():
                    executor.cancel_orders_for_symbol(symbol, side='sell', order_type='stop')
                    if submit_exit_order(executor, symbol, qty_to_sell, current_price):
                        entry['partial_taken'] = True
                        entry['partial_order_pending'] = False
                        entry['realized_partial_qty'] = int(entry.get('realized_partial_qty', 0) or 0) + qty_to_sell
                        entry['shares'] = max(shares - qty_to_sell, 0)
                        if entry.get('entry_price', 0.0) > 0:
                            entry['stop_loss'] = max(float(entry.get('stop_loss', 0.0) or 0.0), float(entry['entry_price']))
                        actions.append(f"{symbol}: took partial profit on {qty_to_sell} shares")
                        shares = entry['shares']
                elif qty_to_sell < shares and not MarketHours.is_market_open():
                    actions.append(f"{symbol}: partial target hit, waiting for market-open execution")

            if float(entry.get('target', 0.0) or 0.0) > 0 and current_price >= float(entry['target']) and shares > 0:
                if executor and executor.is_connected() and MarketHours.is_market_open():
                    executor.cancel_orders_for_symbol(symbol, side='sell')
                    if submit_exit_order(executor, symbol, shares, current_price):
                        removals.append(symbol)
                        actions.append(f"{symbol}: exited final target")
                        continue
                elif not MarketHours.is_market_open():
                    actions.append(f"{symbol}: final target reached, waiting for market-open exit")

            sync_protective_stop(symbol, entry, int(entry.get('shares', shares) or shares), executor)
        else:
            age_days = 0
            try:
                age_days = (datetime.now() - datetime.fromisoformat(entry.get('created_at'))).days
            except Exception:
                pass
            has_open_orders = bool(open_orders_by_symbol.get(symbol))
            if entry.get('status') == 'PENDING_ENTRY' and not has_open_orders and age_days >= 5:
                removals.append(symbol)
                actions.append(f"{symbol}: removed stale pending entry")
                continue
            if entry.get('status') == 'OPEN' and not has_open_orders:
                removals.append(symbol)
                actions.append(f"{symbol}: removed closed position from portfolio")
                continue

        portfolio[symbol] = normalize_managed_position(symbol, entry)

    for symbol in removals:
        portfolio.pop(symbol, None)

    save_managed_portfolio(portfolio, logger)
    return portfolio, actions

def get_market_data(tickers_override=None, cache_ttl_hours=DEFAULT_OHLCV_TTL_HOURS,

                   sp500_ttl_days=DEFAULT_SP500_TTL_DAYS, force_refresh=False):
    """Download and cache market data."""
    
    # Ensure cache directory exists
    if not os.path.exists(CACHE_DIR):
        os.makedirs(CACHE_DIR)
    
    # Get tickers
    tickers = tickers_override[:] if tickers_override else None
    
    if not tickers:
        sp500_ttl_sec = sp500_ttl_days * 86400
        if os.path.exists(SP500_CACHE_FILE) and sp500_ttl_sec > 0:
            if time.time() - os.path.getmtime(SP500_CACHE_FILE) < sp500_ttl_sec:
                try:
                    tickers = pd.read_json(SP500_CACHE_FILE, typ='series').tolist()
                except:
                    tickers = None
        
        if not tickers:
            print("  Fetching S&P 500 list...")
            try:
                headers = {"User-Agent": "Mozilla/5.0"}
                resp = requests.get(
                    "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
                    headers=headers, timeout=15
                )
                df = pd.read_html(io.StringIO(resp.text))[0]
                tickers = [t.replace('.', '-') for t in df['Symbol'].tolist()]
                pd.Series(tickers).to_json(SP500_CACHE_FILE)
            except:
                tickers = ["NVDA", "MSFT", "AAPL", "AMD", "TSLA"]
    
    tickers = list(dict.fromkeys([t.strip().upper() for t in tickers if t]))
    sector_tickers = [ticker.upper() for ticker in SECTOR_ETFS.values()]
    required_symbols = set(tickers + sector_tickers + ["SPY", "^VIX"])
    
    # Check cache
    cache_ttl_sec = cache_ttl_hours * 3600
    cache_valid = (
        os.path.exists(OHLCV_CACHE_FILE) and cache_ttl_sec > 0 and
        (time.time() - os.path.getmtime(OHLCV_CACHE_FILE) < cache_ttl_sec)
    )
    
    # Smart refresh during market hours
    if MarketHours.should_auto_refresh(OHLCV_CACHE_FILE, cache_ttl_hours) and not force_refresh:
        print(f"  Smart refresh: Market is {MarketHours.get_market_status_string()}")
        cache_valid = False
    
    data = None
    if cache_valid and not force_refresh:
        print("  Loading cached market data...")
        try:
            data = pd.read_parquet(OHLCV_CACHE_FILE)
            if not isinstance(data.columns, pd.MultiIndex):
                data = None
            else:
                available_symbols = {str(symbol).upper() for symbol in data.columns.levels[0]}
                if not required_symbols.issubset(available_symbols):
                    data = None
        except:
            data = None
    
    if data is None:
        print("  Downloading market data (1-2 minutes)...")
        tickers_plus = list(dict.fromkeys(tickers + sector_tickers + ["SPY", "^VIX"]))
        
        chunk_size = 100
        data_frames = []
        
        for i in range(0, len(tickers_plus), chunk_size):
            chunk = tickers_plus[i:i+chunk_size]
            print(f"    Batch {i//chunk_size + 1}...", end='\r')
            try:
                d = yf.download(
                    chunk, period=DEFAULT_DATA_PERIOD, interval=DEFAULT_DATA_INTERVAL,
                    auto_adjust=True, group_by='ticker', threads=False, progress=False
                )
                if d is not None and not d.empty:
                    d = ensure_multiindex(d, chunk)
                    data_frames.append(d)
            except Exception as e:
                print(f"\n    Warning: Batch failed - {e}")
        
        print()
        if data_frames:
            data = pd.concat(data_frames, axis=1)
            data.to_parquet(OHLCV_CACHE_FILE)
        else:
            raise ValueError("Failed to download any data")
    
    return tickers, data


def _conviction_bonus(mom_score, accum_score, rs_pct):
    """Calculate conviction bonus (0-10) from forward-looking signals.

    This feeds into StatisticalConfidenceScorer via the consistency_score
    parameter, adding bonus points for stocks with strong momentum,
    institutional accumulation, and relative strength leadership.
    """
    bonus = 0.0
    # Each signal contributes up to ~3.3 points
    if mom_score >= 65:
        bonus += 3.5
    elif mom_score >= 50:
        bonus += 2.0
    elif mom_score >= 35:
        bonus += 1.0
    if accum_score >= 65:
        bonus += 3.5
    elif accum_score >= 45:
        bonus += 2.0
    elif accum_score >= 30:
        bonus += 1.0
    if rs_pct >= 75:
        bonus += 3.0
    elif rs_pct >= 60:
        bonus += 2.0
    elif rs_pct >= 45:
        bonus += 1.0
    return min(10.0, bonus)


def process_ticker(ticker, data, mkt_status, spy_close, settings,
                   spy_df=None, all_stock_returns=None):
    """Process a single ticker and return setup if valid.

    Enhanced with momentum scoring, volume accumulation detection,
    relative strength ranking, and blue sky breakout bonus.
    """
    try:
        # Extract dataframe
        if isinstance(data.columns, pd.MultiIndex):
            if ticker not in data.columns.levels[0]:
                return None, "No Data", None
            df = data[ticker].copy()
        else:
            return None, "No Data", None

        required_cols = ["Open", "High", "Low", "Close", "Volume"]
        if any(col not in df.columns for col in required_cols):
            return None, "No Data", None
        df = df[required_cols].dropna()

        if len(df) < 250:
            return None, "No Data", None

        # Basic filters
        c = float(df['Close'].iloc[-1])
        if c < 5.0:
            return None, "Low Price/Liquidity", None

        vol_avg = df['Volume'].rolling(20).mean().iloc[-1]
        dollar_vol = vol_avg * c

        if dollar_vol < MIN_AVG_DOLLAR_VOLUME or vol_avg < MIN_AVG_VOLUME:
            return None, "Low Price/Liquidity", None

        # ATR check
        atr = atr_series(df).iloc[-1]
        if pd.isna(atr) or atr <= 0 or atr / c > 0.10:
            return None, "Low Price/Liquidity", None

        # Trend check
        sma50 = df['Close'].rolling(50).mean().iloc[-1]
        sma200 = df['Close'].rolling(200).mean().iloc[-1]
        base_mkt_status = get_base_market_status(mkt_status)

        if c < sma200:
            return None, "Downtrend (Bear)", None

        # Setup detection
        trailing_high = float(df['High'].iloc[-252:].max())
        distance_from_high_pct = (
            max(trailing_high - c, 0.0) / max(trailing_high, 1e-9) * 100.0
            if trailing_high > 0 else 100.0
        )
        is_breakout = c > sma50 > sma200
        is_dip = -0.03 < (c - sma50) / sma50 < 0.04 and c > sma200

        if not (is_breakout or is_dip):
            return None, "No Setup (VCP/Dip)", None

        if is_breakout and distance_from_high_pct > float(settings.get('breakout_high_proximity_pct', 12.0)):
            return None, "Not Near High", f"{distance_from_high_pct:.1f}% below 52-week high"

        # Gap risk check
        validator = StrategyValidator(df)
        if GAP_PROTECTION and not validator.check_gap_risk():
            return None, "Gap Risk", None

        # Calculate enhanced selection signals
        accum_score = validator.volume_accumulation_score()
        mom_score = validator.momentum_composite_score()

        # Relative strength percentile (vs all scanned stocks)
        rs_pct = 50.0
        if spy_df is not None and len(spy_df) > 0:
            rs_pct = validator.rs_percentile_rank(
                spy_df, all_returns=all_stock_returns, lookback=63
            )

        prebreakout_plan = {
            'score': 0.0,
            'pivot_trigger': float(df['High'].iloc[-16:-1].max()) + 0.02,
            'starter_trigger': round(c, 2),
            'starter_stop': round(c - (atr * 2), 2),
            'ema10': c,
            'ema21': c,
            'avwap': None,
            'avwap_distance': 0.0,
            'dist_to_pivot_pct': 0.0,
            'early_entry_ready': False,
        }
        if is_breakout:
            prebreakout_plan = analyze_pre_breakout(
                df, validator, c, atr, sma50, sma200, accum_score, mom_score, rs_pct, settings
            )

        if is_dip and settings.get('disable_dip_in_weak_regimes', True):
            if base_mkt_status in ('Correction', 'BEAR', 'STRONG_BEAR'):
                return None, "Regime Filter", f"Dip disabled in {base_mkt_status}"

        # FIX: Use dedicated rejection reason instead of hiding behind "No Setup"
        if mom_score < 30 and accum_score < 25:
            return None, "Rejected (Low Win%)", "Weak momentum + accumulation"

        # Backtest
        if is_breakout:
            res = validator.backtest_breakout(return_trades=True)
            strategy_name = "BREAKOUT"
            min_wr = settings.get('min_winrate_breakout', DEFAULT_MIN_WIN_RATE_BREAKOUT)
            min_pf = settings.get('min_pf_breakout', DEFAULT_MIN_PF_BREAKOUT)
            min_trades = settings.get('min_trades_breakout', DEFAULT_MIN_TRADES_BREAKOUT)
        else:
            res = validator.backtest_dip(return_trades=True)
            strategy_name = "DIP BUY"
            min_wr = settings.get('min_winrate_dip', DEFAULT_MIN_WIN_RATE_DIP)
            min_pf = settings.get('min_pf_dip', DEFAULT_MIN_PF_DIP)
            min_trades = settings.get('min_trades_dip', DEFAULT_MIN_TRADES_DIP)

        net_stats = get_adjusted_backtest_stats(res, settings)
        res.update({
            'trades': net_stats['trades'],
            'win_rate': net_stats['win_rate'],
            'pf': net_stats['pf'],
            'expectancy': net_stats['expectancy'],
            'trades_list': net_stats['trades_list'],
        })

        # Quality filter - Need profitable backtest results
        if res['trades'] < min_trades:
            return None, "Rejected (Quality)", f"Trades: {res['trades']} < {min_trades}"
        if res['win_rate'] < min_wr:
            return None, "Rejected (Quality)", f"WR: {res['win_rate']:.0f}% < {min_wr:.0f}%"
        if res['pf'] < min_pf:
            return None, "Rejected (Quality)", f"PF: {res['pf']:.2f} < {min_pf:.2f}"

        # Expectancy filter - reject if average trade is near zero or negative
        min_exp = (settings.get('min_expectancy_breakout', DEFAULT_MIN_EXPECTANCY_BREAKOUT)
                   if is_breakout else
                   settings.get('min_expectancy_dip', DEFAULT_MIN_EXPECTANCY_DIP))
        if res['expectancy'] < min_exp:
            return None, "Rejected (Quality)", f"Exp: {res['expectancy']:.4f} < {min_exp:.4f}"

        # --- Out-of-sample validation ---
        # Run the backtest again but only on the last 25% of data.
        # Indicators (SMA, ATR) still use the full history for accuracy,
        # but entries are restricted to the OOS window. This reveals
        # whether the edge holds in recent data or is overfit to the past.
        backtest_days = 750
        oos_start = int(backtest_days * (1 - V2_OOS_SPLIT_PCT))
        oos_pf = None
        oos_wr = None
        oos_trades = 0
        oos_penalty = 0

        if is_breakout:
            oos_res = validator.backtest_breakout(
                return_trades=True, min_entry_idx=oos_start)
        else:
            oos_res = validator.backtest_dip(
                return_trades=True, min_entry_idx=oos_start)

        oos_net_stats = get_adjusted_backtest_stats(oos_res, settings)
        oos_trades = oos_net_stats.get('trades', 0)
        oos_expectancy = oos_net_stats.get('expectancy', 0.0)
        if oos_trades >= V2_OOS_MIN_TRADES:
            oos_pf = oos_net_stats.get('pf', 0.0)
            oos_wr = oos_net_stats.get('win_rate', 0.0)

            # Hard reject: OOS is clearly unprofitable
            if oos_pf < 0.8 and oos_trades >= 5:
                return None, "Rejected (OOS)", (
                    f"OOS PF: {oos_pf:.2f} (N={oos_trades})")

            # Confidence penalty: OOS below breakeven threshold
            if oos_pf < V2_OOS_MIN_PF:
                oos_penalty += 10

            # Confidence penalty: win rate decayed significantly
            wr_drop = (res['win_rate'] - oos_wr) / 100.0
            if wr_drop > V2_OOS_DECAY_THRESHOLD:
                oos_penalty += 8

        if settings.get('require_oos'):
            oos_min_trades = int(settings.get('oos_min_trades', V2_OOS_MIN_TRADES))
            min_oos_wr = (
                settings.get('oos_min_winrate_breakout', settings.get('min_winrate_breakout', DEFAULT_MIN_WIN_RATE_BREAKOUT))
                if is_breakout else
                settings.get('oos_min_winrate_dip', settings.get('min_winrate_dip', DEFAULT_MIN_WIN_RATE_DIP))
            )
            min_oos_pf = (
                settings.get('oos_min_pf_breakout', V2_OOS_MIN_PF)
                if is_breakout else
                settings.get('oos_min_pf_dip', V2_OOS_MIN_PF)
            )
            min_oos_exp = (
                settings.get('oos_min_expectancy_breakout', settings.get('min_expectancy_breakout', DEFAULT_MIN_EXPECTANCY_BREAKOUT))
                if is_breakout else
                settings.get('oos_min_expectancy_dip', settings.get('min_expectancy_dip', DEFAULT_MIN_EXPECTANCY_DIP))
            )

            if oos_trades < oos_min_trades:
                return None, "Rejected (OOS)", f"OOS trades: {oos_trades} < {oos_min_trades}"
            if oos_pf is None or oos_wr is None:
                return None, "Rejected (OOS)", "OOS metrics unavailable"
            if oos_wr < min_oos_wr:
                return None, "Rejected (OOS)", f"OOS WR: {oos_wr:.0f}% < {min_oos_wr:.0f}%"
            if oos_pf < min_oos_pf:
                return None, "Rejected (OOS)", f"OOS PF: {oos_pf:.2f} < {min_oos_pf:.2f}"
            if oos_expectancy < min_oos_exp:
                return None, "Rejected (OOS)", f"OOS Exp: {oos_expectancy:.4f} < {min_oos_exp:.4f}"

        oos_stats = {
            'pf': float(oos_pf or 0.0),
            'trades': int(oos_trades),
            'win_rate': float(oos_wr or 0.0),
            'expectancy': float(oos_expectancy or 0.0),
        }
        aligned_spy = None
        if spy_df is not None and len(spy_df) > 0:
            try:
                aligned_index = df.index.intersection(spy_df.index)
                if len(aligned_index) > 0:
                    aligned_spy = spy_df.loc[aligned_index]
            except Exception:
                aligned_spy = None

        wf_stats = evaluate_walk_forward_robustness(df, settings, is_breakout)
        regime_stats = evaluate_regime_stability(df, aligned_spy, settings, is_breakout)
        wf_min_trades = int(settings.get('wf_min_trades_breakout' if is_breakout else 'wf_min_trades_dip', 5))
        wf_min_pf = float(settings.get('wf_min_pf_breakout' if is_breakout else 'wf_min_pf_dip', 1.0))
        wf_min_expectancy = float(settings.get('wf_min_expectancy_breakout' if is_breakout else 'wf_min_expectancy_dip', 0.0))
        wf_min_passrate = float(settings.get('wf_min_passrate_breakout' if is_breakout else 'wf_min_passrate_dip', 0.5))
        regime_floor = float(settings.get('regime_min_score_breakout' if is_breakout else 'regime_min_score_dip', 0.33))

        if settings.get('require_walkforward'):
            if wf_stats.get('trades', 0) < wf_min_trades:
                return None, "Rejected (WF)", f"WF trades: {wf_stats.get('trades', 0)} < {wf_min_trades}"
            if wf_stats.get('pf', 0.0) < wf_min_pf:
                return None, "Rejected (WF)", f"WF PF: {wf_stats.get('pf', 0.0):.2f} < {wf_min_pf:.2f}"
            if wf_stats.get('pass_rate', 0.0) < wf_min_passrate:
                return None, "Rejected (WF)", f"WF pass: {wf_stats.get('pass_rate', 0.0) * 100:.0f}% < {wf_min_passrate * 100:.0f}%"
            if wf_stats.get('expectancy', 0.0) < wf_min_expectancy:
                return None, "Rejected (WF)", f"WF Exp: {wf_stats.get('expectancy', 0.0):.4f} < {wf_min_expectancy:.4f}"

        if regime_stats.get('count', 0) > 0 and regime_stats.get('score', 0.0) < regime_floor:
            return None, "Rejected (Regime)", f"Regime: {regime_stats.get('score', 0.0) * 100:.0f}% < {regime_floor * 100:.0f}%"

        robustness_score = calculate_robustness_score(net_stats, oos_stats, wf_stats, regime_stats, is_breakout)
        if robustness_score < float(settings.get('min_robustness_score', 55.0)):
            return None, "Rejected (WF)", f"Rob: {robustness_score:.0f} < {settings.get('min_robustness_score', 55.0):.0f}"

        # Calculate setup
        breakout_pivot = float(prebreakout_plan.get('pivot_trigger', float(df['High'].iloc[-16:-1].max()) + 0.02))
        early_entry_candidate = (
            is_breakout and
            prebreakout_plan.get('early_entry_ready', False) and
            base_mkt_status in ('STRONG_BULL', 'BULL', 'RECOVERY')
        )
        if early_entry_candidate and robustness_score < float(settings.get('min_early_entry_robustness', 62.0)):
            early_entry_candidate = False

        trigger = breakout_pivot if is_breakout else c
        stop = trigger - (atr * 2)
        target = trigger + (atr * 3.5)
        starter_trigger = trigger
        add_on_trigger = 0.0
        partial_target = 0.0
        breakeven_trigger = 0.0
        trailing_stop = 0.0
        starter_size_pct = 1.0
        planned_total_qty = 0
        add_on_qty = 0

        if early_entry_candidate:
            strategy_name = "EARLY BO"
            starter_trigger = float(prebreakout_plan.get('starter_trigger', c))
            trigger = starter_trigger
            stop = min(
                float(prebreakout_plan.get('starter_stop', c - (atr * 2))),
                trigger - (atr * 0.75)
            )
            if stop >= trigger:
                stop = trigger - (atr * 1.5)
            add_on_trigger = breakout_pivot
            starter_size_pct = float(settings.get('starter_entry_size_pct', 0.35))
            target = max(
                add_on_trigger + (atr * 3.0),
                trigger + ((trigger - stop) * float(settings.get('final_target_multiple', 3.2)))
            )

        risk_per_share = trigger - stop
        if risk_per_share <= 0:
            return None, "Bad Risk/Reward", None

        min_rr = (
            float(settings.get('min_rr_breakout', DEFAULT_MIN_RR_BREAKOUT))
            if is_breakout else
            float(settings.get('min_rr_dip', DEFAULT_MIN_RR_DIP))
        )
        rr_ratio = (target - trigger) / risk_per_share
        if rr_ratio < min_rr:
            return None, "Bad Risk/Reward", f"RR: {rr_ratio:.2f} < {min_rr:.2f}"

        # Earnings check
        is_blackout, reason = EarningsCalendar.is_in_blackout(ticker)
        if is_blackout:
            return None, "Earnings Risk", None

        # Position sizing
        account_size = max(float(settings.get('account_size', ACCOUNT_SIZE)), 1.0)
        max_risk_pct = max(float(settings.get('max_risk_pct_per_trade', MAX_RISK_PCT_PER_TRADE)), 0.05)
        base_risk_amt = min(
            float(settings.get('risk_per_trade', RISK_PER_TRADE)),
            account_size * (max_risk_pct / 100.0)
        )
        regime_factor = float((settings.get('regime_factors') or DEFAULT_REGIME_FACTORS).get(
            base_mkt_status, 1.0
        ))
        regime_scalar = 1.0 / max(1.0, regime_factor)
        position_size_scalar = max(float(settings.get('position_size_scalar', 1.0)), 0.1)
        risk_amt = base_risk_amt * regime_scalar * position_size_scalar

        if risk_amt <= 0:
            return None, "Sizing Constraint", "No risk budget"

        max_shares = DataValidator.max_position_size(vol_avg, c, MAX_POSITION_PCT_OF_VOLUME)
        max_alloc_pct = float(settings.get('max_alloc_pct_per_position', 15.0))
        alloc_ref_price = max(add_on_trigger, trigger)
        alloc_cap_shares = int((account_size * (max_alloc_pct / 100.0)) / max(alloc_ref_price, 1e-9))

        if early_entry_candidate:
            add_on_risk_per_share = max(add_on_trigger - stop, risk_per_share)
            blended_risk_per_share = (
                risk_per_share * starter_size_pct +
                add_on_risk_per_share * (1.0 - starter_size_pct)
            )
            planned_total_qty = int(risk_amt / max(blended_risk_per_share, 1e-9))
            if max_shares > 0:
                planned_total_qty = min(planned_total_qty, max_shares)
            if alloc_cap_shares > 0:
                planned_total_qty = min(planned_total_qty, alloc_cap_shares)
            if planned_total_qty < 1:
                return None, "Sizing Constraint", "Risk budget too small"
            shares = max(1, int(np.ceil(planned_total_qty * starter_size_pct)))
            shares = min(shares, planned_total_qty)
            add_on_qty = max(planned_total_qty - shares, 0)
            starter_size_pct = shares / max(planned_total_qty, 1)
        else:
            planned_total_qty = int(risk_amt / risk_per_share)
            if max_shares > 0:
                planned_total_qty = min(planned_total_qty, max_shares)
            if alloc_cap_shares > 0:
                planned_total_qty = min(planned_total_qty, alloc_cap_shares)
            shares = planned_total_qty

        if shares < 1:
            return None, "Sizing Constraint", "Risk budget too small"

        est_slippage = DataValidator.calculate_realistic_slippage(
            shares, vol_avg, is_breakout=is_breakout
        )
        effective_entry = trigger * (1 + est_slippage)
        effective_stop = stop * (1 - est_slippage * 0.5)
        effective_target = target * (1 - est_slippage * 0.3)
        effective_risk = effective_entry - effective_stop
        if effective_risk <= 0:
            return None, "Bad Risk/Reward", "Invalid effective risk"
        effective_rr = (effective_target - effective_entry) / effective_risk
        if effective_rr < min_rr:
            return None, "Bad Risk/Reward", f"Net RR: {effective_rr:.2f} < {min_rr:.2f}"

        partial_mult = float(settings.get('partial_exit_multiple', 1.5))
        partial_target = trigger + (risk_per_share * partial_mult)
        if add_on_trigger > 0:
            partial_target = max(partial_target, add_on_trigger + (atr * 0.5))
        partial_target = round(partial_target, 2)
        breakeven_trigger = round(add_on_trigger if add_on_trigger > 0 else (trigger + risk_per_share), 2)
        trailing_stop = round(max(atr, risk_per_share * 0.8), 2)
        target = round(max(target, partial_target + risk_per_share), 2)
        stop = round(stop, 2)
        trigger = round(trigger, 2)
        if add_on_trigger > 0:
            add_on_trigger = round(add_on_trigger, 2)

        # Statistical confidence — pure backtest quality (no forward signals)
        # Forward-looking signals (momentum/accumulation/RS) are only used
        # in the composite score for ranking, NOT in the confidence grade.
        # This prevents double-counting and keeps the grade honest.
        # OOS profit factor is passed in when available to reward strategies
        # whose edge holds up in recent unseen data.
        trades_list = res.get('trades_list', [])
        stat_conf = StatisticalConfidenceScorer.calculate_confidence(
            trades=res['trades'],
            win_rate=res['win_rate'],
            profit_factor=res['pf'],
            expectancy=res.get('expectancy', 0),
            oos_pf=oos_pf,
        )

        # Apply OOS penalty — downgrade confidence when recent data
        # shows the edge is weaker than the full backtest suggests
        stat_conf['score'] = max(0, stat_conf['score'] - oos_penalty)
        # Recalculate grade after penalty
        adj_score = stat_conf['score']
        if adj_score >= 70:
            stat_conf['grade'] = 'A'
        elif adj_score >= 55:
            stat_conf['grade'] = 'B'
        elif adj_score >= 40:
            stat_conf['grade'] = 'C'
        elif adj_score >= 25:
            stat_conf['grade'] = 'D'
        else:
            stat_conf['grade'] = 'F'

        t_stat = StatisticalConfidenceScorer.calculate_t_statistic(trades_list)

        # Trend analysis - enhanced with new signals
        trend = TrendQualityAnalyzer.analyze(
            df, res,
            accumulation_score=accum_score,
            momentum_score=mom_score,
            rs_percentile=rs_pct
        )

        # --- Enhanced composite score ---
        # Base: backtest quality
        score = res['win_rate'] + (res['pf'] * 10)

        # Momentum bonus (0-15 pts): stocks going up tend to keep going up
        if mom_score >= 70:
            score += 15
        elif mom_score >= 55:
            score += 10
        elif mom_score >= 40:
            score += 5

        # Accumulation bonus (0-12 pts): institutional buying = smart money
        if accum_score >= 70:
            score += 12
        elif accum_score >= 50:
            score += 8
        elif accum_score >= 35:
            score += 4

        # Relative strength bonus (0-10 pts): leaders outperform
        if rs_pct >= 80:
            score += 10
        elif rs_pct >= 65:
            score += 7
        elif rs_pct >= 50:
            score += 3

        # FIX: Blue sky breakout bonus (0-8 pts)
        # Stocks near 52-week highs have no overhead resistance.
        # Breakouts into blue sky have highest follow-through rate.
        if is_breakout and validator.is_blue_sky_breakout():
            score += 8
        if is_breakout:
            if distance_from_high_pct <= 4:
                score += 6
            elif distance_from_high_pct <= 8:
                score += 3

        pre_breakout_score = float(prebreakout_plan.get('score', 0.0))
        if pre_breakout_score > 0:
            score += min(12, pre_breakout_score * 0.12)
        if abs(prebreakout_plan.get('avwap_distance', 0.0)) <= 0.03:
            score += 4
        if early_entry_candidate:
            score += 10

        # Robustness bonus - reward ideas that survive harder validation
        if robustness_score >= 75:
            score += 12
        elif robustness_score >= 60:
            score += 8
        elif robustness_score >= 50:
            score += 4
        if wf_stats.get('pass_rate', 0.0) >= 0.75:
            score += 6
        elif wf_stats.get('pass_rate', 0.0) >= 0.50:
            score += 3
        if regime_stats.get('score', 0.0) >= 0.67:
            score += 4
        elif regime_stats.get('score', 0.0) >= 0.34:
            score += 2

        # Market regime bonus
        if base_mkt_status in ("STRONG_BULL", "BULL"):
            score += 10
        elif base_mkt_status == "RECOVERY":
            score += 4

        # Statistical significance bonus
        if t_stat >= 2.0:
            score += 10
        elif t_stat >= 1.5:
            score += 5

        # Kelly
        W = res['win_rate'] / 100
        wins = [t for t in trades_list if t > 0]
        losses = [t for t in trades_list if t <= 0]
        avg_win = abs(np.mean(wins)) if wins else 0.02
        avg_loss = abs(np.mean(losses)) if losses else 0.01
        R = avg_win / (avg_loss + 1e-9)
        kelly = max(0, (W * R - (1 - W)) / R) * 0.25 * 100

        # Get sector and earnings info
        sector = SectorMapper.get_sector(ticker)
        earnings_date, days_to = EarningsCalendar.get_earnings_date(ticker)
        earnings_call = f"{days_to:+d}d" if days_to else "Unknown"

        oos_tag = ""
        if oos_pf is not None:
            oos_tag = f" | OOS:{oos_wr:.0f}%/{oos_pf:.2f}(N={oos_trades})"
        elif oos_trades > 0:
            oos_tag = f" | OOS:low-N({oos_trades})"
        else:
            oos_tag = " | OOS:no-data"

        wf_tag = ""
        if wf_stats.get('trades', 0) > 0:
            wf_tag = (
                f" | WF:{wf_stats.get('pf', 0.0):.2f}/"
                f"{wf_stats.get('pass_rate', 0.0) * 100:.0f}%"
                f"(N={wf_stats.get('trades', 0)})"
            )
        regime_tag = f" | Reg:{regime_stats.get('score', 0.0) * 100:.0f}%"
        robust_tag = f" | Rob:{robustness_score:.0f}"

        plan_tag = ""
        if add_on_trigger > 0 and add_on_qty > 0:
            plan_tag = f" | Plan:{shares}+{add_on_qty}@{add_on_trigger:.2f}"

        note = (
            f"N={res['trades']} | Net:{res['win_rate']:.0f}%/{res['pf']:.2f} | {stat_conf['grade']} | {trend['trend_grade']}"
            f" | Mom:{mom_score:.0f} Acc:{accum_score:.0f} RS:{rs_pct:.0f}"
            f" | PBS:{pre_breakout_score:.0f}{oos_tag}{wf_tag}{regime_tag}{robust_tag}{plan_tag}"
        )

        setup = TitanSetup(
            ticker=ticker,
            strategy=strategy_name,
            price=c,
            trigger=trigger,
            stop=stop,
            target=target,
            qty=shares,
            win_rate=res['win_rate'],
            profit_factor=res['pf'],
            kelly=kelly,
            score=score,
            sector=sector,
            earnings_call=earnings_call,
            note=note,
            confidence_score=stat_conf['score'],
            confidence_grade=stat_conf['grade'],
            trend_grade=trend['trend_grade'],
            t_statistic=t_stat,
            momentum_score=mom_score,
            accumulation_score=accum_score,
            rs_percentile=rs_pct,
            pre_breakout_score=pre_breakout_score,
            breakeven_trigger=breakeven_trigger,
            trailing_stop=trailing_stop,
            avwap_distance=prebreakout_plan.get('avwap_distance', 0.0),
            starter_trigger=starter_trigger,
            add_on_trigger=add_on_trigger,
            partial_target=partial_target,
            planned_total_qty=planned_total_qty,
            add_on_qty=add_on_qty,
            starter_size_pct=starter_size_pct,
            robustness_score=robustness_score,
            walk_forward_pass_rate=wf_stats.get('pass_rate', 0.0),
            walk_forward_pf=wf_stats.get('pf', 0.0),
            walk_forward_trades=wf_stats.get('trades', 0),
            regime_score=regime_stats.get('score', 0.0),
            oos_pf=float(oos_pf or 0.0),
            oos_trades=int(oos_trades),
            net_expectancy=res.get('expectancy', 0.0),
        )
        setup.distance_from_high_pct = distance_from_high_pct
        setup.distance_to_pivot_pct = float(prebreakout_plan.get('dist_to_pivot_pct', 0.0))
        setup.sector_aligned = False

        return setup, "Passed", None

    except Exception as e:
        logging.getLogger("titan").debug(f"Error processing {ticker}: {e}")
        return None, "Error", None


def scan(tickers_override=None, settings=None, max_workers=DEFAULT_MAX_WORKERS, market_data_bundle=None):
    """Main scanning function.

    Enhanced with pre-computed relative strength rankings so every
    stock is compared against the full universe for better selection.
    """

    if settings is None:
        settings = {}

    # Get data
    max_workers = int(settings.get('max_workers', max_workers))
    if market_data_bundle is not None:
        tickers, data = market_data_bundle
    else:
        tickers, data = get_market_data(
            tickers_override,
            cache_ttl_hours=settings.get('cache_ttl_hours', DEFAULT_OHLCV_TTL_HOURS),
            sp500_ttl_days=settings.get('sp500_ttl_days', DEFAULT_SP500_TTL_DAYS),
            force_refresh=settings.get('force_refresh_cache', False),
        )

    # Get SPY data
    spy_close = None
    spy_df = None
    if isinstance(data.columns, pd.MultiIndex) and "SPY" in data.columns.levels[0]:
        spy_df = data["SPY"].dropna()
        if "Close" in spy_df:
            spy_close = spy_df["Close"]

    # Analyze market regime
    regime = MarketRegime(data)
    mkt_status, mkt_score, vix_level = regime.analyze_spy()

    print(f"\n  Market Status: {mkt_status} (Score: {mkt_score:.2f})")
    if vix_level:
        print(f"  VIX Level: {vix_level:.1f}")

    top_sectors = []
    sector_frames = {}
    if isinstance(data.columns, pd.MultiIndex):
        for sector_name, etf_ticker in SECTOR_ETFS.items():
            if etf_ticker in data.columns.levels[0]:
                sector_df = data[etf_ticker].dropna()
                if isinstance(sector_df, pd.DataFrame) and not sector_df.empty and 'Close' in sector_df:
                    sector_frames[etf_ticker] = sector_df
    if sector_frames:
        top_sectors = SectorAnalyzer(sector_frames).get_top_sectors(
            top_n=int(settings.get('top_sectors_to_trade', TOP_SECTORS_TO_TRADE)),
            lookback_days=int(settings.get('sector_lookback_days', 20)),
        )
        if top_sectors:
            print(f"  Top Sectors: {', '.join(top_sectors)}")

    # Check VIX panic
    if vix_level and vix_level > VIX_PANIC_THRESHOLD:
        print(f"\n  VIX PANIC ({vix_level:.1f}) - No trading allowed!")
        return [], {}, {"mkt_status": mkt_status, "mkt_score": mkt_score, "top_sectors": top_sectors}, vix_level

    if mkt_score == 0:
        print("\n  BEAR MARKET - No new long positions recommended!")

    # Pre-compute 63-day returns for all stocks (for RS percentile ranking)
    print("  Computing relative strength rankings...")
    all_stock_returns = {}
    lookback = 63
    for t in tickers:
        try:
            if isinstance(data.columns, pd.MultiIndex) and t in data.columns.levels[0]:
                t_close = data[t]['Close'].dropna()
                if len(t_close) > lookback:
                    ret = (t_close.iloc[-1] / t_close.iloc[-lookback] - 1) * 100
                    all_stock_returns[t] = float(ret)
        except Exception:
            pass

    # Scan
    print(f"  Scanning {len(tickers)} stocks...")

    results = []
    tracker = RejectionTracker()

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                process_ticker, t, data, mkt_status, spy_close, settings,
                spy_df=spy_df, all_stock_returns=all_stock_returns
            ): t
            for t in tickers
        }

        completed = 0
        for future in concurrent.futures.as_completed(futures):
            completed += 1
            if completed % 50 == 0:
                print(f"    Progress: {completed}/{len(tickers)}", end='\r')

            try:
                setup, reason, _ = future.result()
                tracker.update(reason)
                if setup:
                    results.append(setup)
                    print(f"    Found: {setup.ticker} ({setup.strategy}) "
                          f"WR:{setup.win_rate:.0f}% Mom:{setup.momentum_score:.0f} "
                          f"RS:{setup.rs_percentile:.0f}")
            except Exception:
                tracker.update("Error")

    print()

    # Apply sector leadership and sector exposure limits
    if results:
        preferred_sectors = set(top_sectors)
        aligned = []
        non_aligned = []
        removed_for_sector = 0
        for s in results:
            s.sector_aligned = bool(preferred_sectors) and (s.sector in preferred_sectors)
            if s.sector_aligned:
                s.score += 6
                aligned.append(s)
            else:
                s.score -= 4
                if preferred_sectors and settings.get('require_top_sector_alignment', True):
                    removed_for_sector += 1
                    continue
                non_aligned.append(s)

        if removed_for_sector:
            tracker.stats['Sector Filter'] = tracker.stats.get('Sector Filter', 0) + removed_for_sector

        results = aligned + non_aligned
        results.sort(
            key=lambda x: (
                1 if getattr(x, 'sector_aligned', False) else 0,
                grade_to_rank(getattr(x, 'confidence_grade', 'F')),
                getattr(x, 'robustness_score', 0.0),
                manual_strategy_priority(x),
                x.score,
                getattr(x, 'pre_breakout_score', 0.0),
                getattr(x, 'rs_percentile', 0.0),
            ),
            reverse=True,
        )
        sector_count = {}
        filtered = []
        for s in results:
            if len(filtered) >= MAX_POSITIONS:
                break
            sector = s.sector or 'Unknown'
            if sector_count.get(sector, 0) >= MAX_SECTOR_EXPOSURE:
                continue
            sector_count[sector] = sector_count.get(sector, 0) + 1
            filtered.append(s)
        results = filtered

    return results, tracker.summary(), {"mkt_status": mkt_status, "mkt_score": mkt_score, "top_sectors": top_sectors}, vix_level


def main():
    """Main entry point."""
    import time
    
    # Plain runs default to manual scanning. Trust mode and order routing must be explicit.
    no_args = len(sys.argv) == 1
    
    # Parse arguments
    parser = argparse.ArgumentParser(description="Titan Trade v9.0 - Manual-First Stock Scanner")
    parser.add_argument("--trust-mode", action="store_true", help="Enable Trust Mode")
    parser.add_argument("--trust-status", action="store_true", help="Show Trust Mode status")
    parser.add_argument("--trust-paper", action="store_true", help="Start paper trading")
    parser.add_argument("--trust-paper-win", action="store_true", help="Record paper win")
    parser.add_argument("--trust-paper-loss", action="store_true", help="Record paper loss")
    parser.add_argument("--trust-bypass", action="store_true", help="Bypass paper validation")
    parser.add_argument("--execute-orders", action="store_true", help="Actually route Alpaca orders in trust mode")
    parser.add_argument("--tickers", default="", help="Custom ticker list")
    parser.add_argument("--account-size", type=float, default=None)
    parser.add_argument("--risk-per-trade", type=float, default=None)
    
    args = parser.parse_args()
    
    # Initialize
    logger = setup_logging()
    auto_manager = AutoModeManager()
    settings = build_runtime_settings(args, auto_manager, logger)
    trust_manager = TrustModeManager(account_size=settings['account_size'])
    risk_manager = PortfolioRiskManager(account_size=settings['account_size'])
    signal_tracker = SignalTracker()

    print_runtime_summary(settings, trust_mode=(args.trust_mode or args.trust_paper))
    
    # Check for paper trading bypass from auto config
    if auto_manager.config.get("paper_trading_bypassed", False):
        trust_manager.state["paper_validated"] = True
        trust_manager._save_state()
    
    # Handle Trust Mode commands
    if args.trust_status:
        print_trust_mode_header()
        status = trust_manager.get_status_report()
        print(f"\n  Paper Validated: {'YES' if status['paper_validated'] else 'NO'}")
        print(f"  Trades Today: {status['trades_today']}/{status['max_daily']}")
        print(f"  Win Rate: {status['win_rate']:.1f}%")
        return
    
    if args.trust_paper_win:
        trust_manager.record_paper_trade(won=True)
        print("  Recorded paper WIN!")
        return
    
    if args.trust_paper_loss:
        trust_manager.record_paper_trade(won=False)
        print("  Recorded paper LOSS.")
        return
    
    if args.trust_bypass:
        print("\n  Type 'I ACCEPT THE RISK' to bypass:")
        if trust_manager.bypass_paper_validation(input("  > ").strip()):
            print("  Bypassed!")
        return
    
    # Trust Mode header
    if args.trust_mode or args.trust_paper:
        print_trust_mode_header()
        
        if not args.trust_paper:
            validated, msg = trust_manager.is_paper_trading_validated()
            if not validated:
                print(f"\n  {msg}")
                print("  Run with --trust-paper to start validation.")
                return
    else:
        print("\n" + "=" * 60)
        print("  TITAN TRADE v9.0 - MANUAL SCAN")
        print("=" * 60)
        print("  Manual mode only. No broker orders will be sent unless --execute-orders is used with trust mode.")
    
    # Parse custom tickers
    tickers = parse_tickers(args.tickers) if args.tickers else None
    managed_portfolio = {}
    data_universe = tickers[:] if tickers else None
    execution_enabled = bool(args.execute_orders and (args.trust_mode or args.trust_paper))
    if execution_enabled:
        managed_portfolio = load_managed_portfolio(logger)
        if data_universe is not None and managed_portfolio:
            data_universe = list(dict.fromkeys(data_universe + list(managed_portfolio.keys())))

    executor = AlpacaExecutor() if execution_enabled else None
    market_data_bundle = None
    management_actions = []

    if execution_enabled:
        market_data_bundle = get_market_data(
            data_universe,
            cache_ttl_hours=settings.get('cache_ttl_hours', DEFAULT_OHLCV_TTL_HOURS),
            sp500_ttl_days=settings.get('sp500_ttl_days', DEFAULT_SP500_TTL_DAYS),
            force_refresh=settings.get('force_refresh_cache', False),
        )
        managed_portfolio, management_actions = manage_open_positions(
            executor, market_data_bundle[1], settings, logger
        )
        if management_actions:
            print("\n  Position Manager:")
            for action in management_actions[:12]:
                print(f"    {action}")

    if executor and executor.is_connected():
        live_equity = executor.get_account_equity()
        if live_equity > 0:
            risk_manager.update_equity(live_equity)

    # Check risk status only when trust/execution state matters
    if args.trust_mode or args.trust_paper or execution_enabled:
        local_positions = load_open_positions_snapshot(executor, settings)
        sync_trust_positions(trust_manager, len(local_positions))
        risk_status = risk_manager.get_risk_status(local_positions)
        if not risk_status['can_trade']:
            print(f"\n  TRADING BLOCKED: {risk_status['reason']}")
            return
    
    # Run scan
    try:
        setups, stats, mkt_data, vix_level = scan(
            tickers, settings, market_data_bundle=market_data_bundle
        )
    except KeyboardInterrupt:
        print("\n  Cancelled.")
        return
    
    # Display results
    if args.trust_mode or args.trust_paper:
        trusted = print_simple_verdict(setups, trust_manager, vix_level)
        
        # Auto-track and Execute via Alpaca
        if trusted:
            if not args.execute_orders:
                print("\n  Manual trust mode only. No Alpaca orders were sent.")
                print("  Use --execute-orders only after you have reviewed the levels and want broker routing.")
            else:
                trusted, skipped = filter_execution_candidates(trusted, settings)
                if skipped:
                    print("\n  Execution safety filters removed:")
                    for ticker, reason in skipped[:10]:
                        print(f"    {ticker}: {reason}")

                if not trusted:
                    print("\n  No setups survived live execution safety filters.")
                else:
                    portfolio_state = load_managed_portfolio(logger)
                    if executor and executor.is_connected():
                        live_positions = load_open_positions_snapshot(executor, settings)
                        sync_trust_positions(trust_manager, len(live_positions))

                        live_risk_status = risk_manager.get_risk_status(live_positions)
                        if not live_risk_status['can_trade']:
                            print(f"\n  [ALPACA] Trading blocked: {live_risk_status['reason']}")
                        else:
                            buying_power = executor.get_buying_power()
                            open_order_symbols = set(executor.get_open_order_symbols())
                            print(f"\n  [ALPACA] Connected. Buying Power: ${buying_power:,.2f}")

                            submitted = 0
                            for s in trusted[:AUTO_TRACK_TOP_N]:
                                if submitted >= int(settings.get('max_new_orders_per_run', AUTO_TRACK_TOP_N)):
                                    break
                                if len(live_positions) >= int(settings.get('max_live_positions', MAX_POSITIONS)):
                                    print("  [ALPACA] Position cap reached. No more orders will be sent.")
                                    break
                                if s.ticker in live_positions:
                                    print(f"  [ALPACA] Skipping {s.ticker}: already in open positions.")
                                    continue
                                if s.ticker in open_order_symbols:
                                    print(f"  [ALPACA] Skipping {s.ticker}: already has an open order.")
                                    continue
                                if s.ticker in portfolio_state:
                                    print(f"  [ALPACA] Skipping {s.ticker}: already managed in portfolio.")
                                    continue

                                current_heat = risk_manager.calculate_portfolio_heat(live_positions)
                                exec_qty, qty_reason = calculate_execution_qty(
                                    s, settings, buying_power, current_heat, vix_level=vix_level
                                )
                                if exec_qty <= 0:
                                    print(f"  [ALPACA] Skipping {s.ticker}: {qty_reason}.")
                                    continue

                                can_trade, reason = risk_manager.can_take_new_trade(live_positions)
                                if not can_trade:
                                    print(f"  [ALPACA] Stopping order flow: {reason}")
                                    break

                                est_cost = exec_qty * s.trigger
                                est_risk = max((s.trigger - s.stop) * exec_qty, 0.0)
                                signal_tracker.add_signal(s, s.price)
                                is_managed = int(getattr(s, 'add_on_qty', 0) or 0) > 0
                                plan_suffix = ""
                                if is_managed:
                                    planned_total, planned_add = compute_scaled_plan_from_qty(s, exec_qty)
                                    plan_suffix = f", add_later={planned_add}@${s.add_on_trigger:.2f}" if planned_add > 0 else ""
                                print(
                                    f"  [ALPACA] Routing order for {s.ticker}: "
                                    f"qty={exec_qty}, cost=${est_cost:,.2f}, risk=${est_risk:,.2f}{plan_suffix}"
                                )

                                placed = False
                                if is_managed:
                                    placed = executor.submit_limit_order(
                                        symbol=s.ticker,
                                        qty=exec_qty,
                                        side='buy',
                                        limit_price=s.trigger,
                                    )
                                    if placed:
                                        portfolio_state[s.ticker] = create_managed_position_from_setup(s, exec_qty)
                                        save_managed_portfolio(portfolio_state, logger)
                                        live_positions[s.ticker] = {
                                            'entry_price': s.trigger,
                                            'stop_loss': s.stop,
                                            'shares': exec_qty,
                                        }
                                else:
                                    placed = executor.submit_bracket_order(
                                        symbol=s.ticker,
                                        qty=exec_qty,
                                        entry_price=s.trigger,
                                        target_price=s.target,
                                        stop_price=s.stop
                                    )
                                    if placed:
                                        live_positions[s.ticker] = {
                                            'entry_price': s.trigger,
                                            'stop_loss': s.stop,
                                            'shares': exec_qty,
                                        }

                                if placed:
                                    trust_manager.record_trade(s.ticker)
                                    buying_power = max(0.0, buying_power - est_cost)
                                    open_order_symbols.add(s.ticker)
                                    submitted += 1

                            if submitted == 0:
                                print("  [ALPACA] No orders submitted after execution safety checks.")
                    else:
                        print("\n  [ALPACA] Not connected. Tracking signals only.")
                        for s in trusted[:int(settings.get('max_new_orders_per_run', AUTO_TRACK_TOP_N))]:
                            signal_tracker.add_signal(s, s.price)
    else:
        if setups:
            print(f"\n  Found {len(setups)} setups:")
            if mkt_data.get('top_sectors'):
                print(f"  Preferred sectors: {', '.join(mkt_data['top_sectors'])}")
            print_manual_trade_board(setups, settings, top_sectors=mkt_data.get('top_sectors'))
        else:
            print("\n  No valid setups found.")
    
    # Show filter stats
    print("\n  Filter Summary:")
    for k, v in stats.items():
        if v > 0:
            print(f"    {k}: {v}")

    # Export to JSON for Web Dashboard
    print("\n  Exporting results for Web Dashboard...")
    try:
        os.makedirs("data", exist_ok=True)
        export_data = {
            "timestamp": datetime.now().isoformat(),
            "market_status": mkt_data.get("mkt_status", "Unknown"),
            "top_sectors": mkt_data.get("top_sectors", []),
            "vix_level": round(vix_level, 2) if vix_level else None,
            "passed_count": stats.get('passed', 0),
            "total_scanned": stats.get('total', 0),
            "setups": [
                {
                    "ticker": s.ticker,
                    "strategy": s.strategy,
                    "price": round(s.price, 2),
                    "trigger": round(s.trigger, 2),
                    "target": round(s.target, 2),
                    "stop": round(s.stop, 2),
                    "confidence_grade": s.confidence_grade,
                    "action": manual_action_label(s),
                    "sector": s.sector,
                    "sector_aligned": bool(getattr(s, 'sector_aligned', False)),
                    "win_rate": round(s.win_rate, 1),
                    "profit_factor": round(s.profit_factor, 2),
                    "momentum_score": round(s.momentum_score, 1),
                    "rs_percentile": round(s.rs_percentile, 1),
                    "pre_breakout_score": round(getattr(s, 'pre_breakout_score', 0.0), 1),
                    "robustness_score": round(getattr(s, 'robustness_score', 0.0), 1),
                    "walk_forward_pass_rate": round(getattr(s, 'walk_forward_pass_rate', 0.0), 3),
                    "walk_forward_pf": round(getattr(s, 'walk_forward_pf', 0.0), 2),
                    "walk_forward_trades": int(getattr(s, 'walk_forward_trades', 0)),
                    "regime_score": round(getattr(s, 'regime_score', 0.0), 3),
                    "oos_pf": round(getattr(s, 'oos_pf', 0.0), 2),
                    "oos_trades": int(getattr(s, 'oos_trades', 0)),
                    "net_expectancy": round(getattr(s, 'net_expectancy', 0.0), 5),
                    "distance_from_high_pct": round(getattr(s, 'distance_from_high_pct', 0.0), 2),
                    "distance_to_pivot_pct": round(getattr(s, 'distance_to_pivot_pct', 0.0), 2),
                    "starter_trigger": round(getattr(s, 'starter_trigger', s.trigger), 2),
                    "add_on_trigger": round(getattr(s, 'add_on_trigger', 0.0), 2),
                    "partial_target": round(getattr(s, 'partial_target', 0.0), 2),
                    "starter_qty": int(s.qty),
                    "planned_total_qty": int(getattr(s, 'planned_total_qty', s.qty)),
                    "add_on_qty": int(getattr(s, 'add_on_qty', 0))
                } for s in setups
            ]
        }
        with open("data/latest_scan.json", "w") as f:
            json.dump(export_data, f, indent=4)
        print("    Saved to data/latest_scan.json")
    except Exception as e:
        print(f"    Failed to export JSON: {e}")

    # Survivorship bias warning — always shown so users don't over-trust
    print("\n  " + "-" * 56)
    print("  NOTE: Backtests use CURRENT S&P 500 members only.")
    print("  Stocks removed from the index (often after declines)")
    print("  are excluded, which can inflate historical win rates.")
    print("  OOS column in notes shows recent out-of-sample check.")
    print("  " + "-" * 56)


if __name__ == "__main__":
    main()
