#!/usr/bin/env python3
"""
Titan Trade v8.0 - Clean Modular Version
========================================
Just run this file. Everything is automatic.

Usage:
    python titan_trade_v2.py              # Auto Mode (Trust Mode enabled)
    python titan_trade_v2.py --help       # Show all options
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
    MAX_POSITION_PCT_OF_VOLUME, AUTO_TRACK_TOP_N, MAX_SECTOR_EXPOSURE,
    DEFAULT_MIN_WIN_RATE_BREAKOUT, DEFAULT_MIN_WIN_RATE_DIP,
    DEFAULT_MIN_PF_BREAKOUT, DEFAULT_MIN_PF_DIP,
    DEFAULT_MIN_TRADES_BREAKOUT, DEFAULT_MIN_TRADES_DIP,
    DEFAULT_REGIME_FACTORS, TRUST_MODE_SETTINGS,
    
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


def setup_logging(level="INFO"):
    """Configure logging."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s | %(levelname)s | %(message)s"
    )
    return logging.getLogger("titan")


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
        except:
            data = None
    
    if data is None:
        print("  Downloading market data (1-2 minutes)...")
        tickers_plus = list(dict.fromkeys(tickers + ["SPY", "^VIX"]))
        
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


def process_ticker(ticker, data, mkt_status, spy_close, settings):
    """Process a single ticker and return setup if valid."""
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
        
        if c < sma200:
            return None, "Downtrend (Bear)", None
        
        # Setup detection
        is_breakout = c > sma50 > sma200
        is_dip = -0.03 < (c - sma50) / sma50 < 0.04 and c > sma200
        
        if not (is_breakout or is_dip):
            return None, "No Setup (VCP/Dip)", None
        
        # Gap risk check
        validator = StrategyValidator(df)
        if GAP_PROTECTION and not validator.check_gap_risk():
            return None, "Gap Risk", None
        
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
        
        res['expectancy'] = expectancy(res.get('trades_list', []))
        
        # Quality filter
        if res['win_rate'] < min_wr or res['pf'] < min_pf or res['trades'] < min_trades:
            return None, "Rejected (Quality)", None
        
        # Calculate setup
        trigger = float(df['High'].iloc[-16:-1].max()) + 0.02 if is_breakout else c
        stop = trigger - (atr * 2)
        target = trigger + (atr * 3.5)
        
        risk_per_share = trigger - stop
        if risk_per_share <= 0:
            return None, "Bad Risk/Reward", None
        
        rr_ratio = (target - trigger) / risk_per_share
        if rr_ratio < 1.5:
            return None, "Bad Risk/Reward", None
        
        # Earnings check
        is_blackout, reason = EarningsCalendar.is_in_blackout(ticker)
        if is_blackout:
            return None, "Earnings Risk", None
        
        # Position sizing
        risk_amt = min(RISK_PER_TRADE, ACCOUNT_SIZE * MAX_RISK_PCT_PER_TRADE / 100)
        shares = max(1, int(risk_amt / risk_per_share))
        max_shares = DataValidator.max_position_size(vol_avg, c, MAX_POSITION_PCT_OF_VOLUME)
        shares = min(shares, max_shares) if max_shares > 0 else shares
        
        # Statistical confidence
        trades_list = res.get('trades_list', [])
        stat_conf = StatisticalConfidenceScorer.calculate_confidence(
            trades=res['trades'],
            win_rate=res['win_rate'],
            profit_factor=res['pf'],
            expectancy=res.get('expectancy', 0)
        )
        t_stat = StatisticalConfidenceScorer.calculate_t_statistic(trades_list)
        
        # Trend analysis
        trend = TrendQualityAnalyzer.analyze(df, res)
        
        # Score
        score = res['win_rate'] + (res['pf'] * 10)
        if mkt_status == "BULL":
            score += 10
        if t_stat >= 2.0:
            score += 10
        
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
        
        note = f"N={res['trades']} | {stat_conf['grade']} | {trend['trend_grade']}"
        
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
            t_statistic=t_stat
        )
        
        return setup, "Passed", None
        
    except Exception as e:
        return None, "Error", None


def scan(tickers_override=None, settings=None, max_workers=DEFAULT_MAX_WORKERS):
    """Main scanning function."""
    
    if settings is None:
        settings = {}
    
    # Get data
    tickers, data = get_market_data(tickers_override)
    
    # Get SPY data
    spy_close = None
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
    
    # Check VIX panic
    if vix_level and vix_level > VIX_PANIC_THRESHOLD:
        print(f"\n  VIX PANIC ({vix_level:.1f}) - No trading allowed!")
        return [], {}, [], vix_level
    
    if mkt_score == 0:
        print("\n  BEAR MARKET - No new long positions recommended!")
    
    # Scan
    print(f"\n  Scanning {len(tickers)} stocks...")
    
    results = []
    tracker = RejectionTracker()
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(process_ticker, t, data, mkt_status, spy_close, settings): t
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
                    print(f"    Found: {setup.ticker} ({setup.strategy}) WR:{setup.win_rate:.0f}%")
            except Exception:
                tracker.update("Error")
    
    print()
    
    # Apply sector limits
    if results:
        results.sort(key=lambda x: x.score, reverse=True)
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
    
    return results, tracker.summary(), [], vix_level


def main():
    """Main entry point."""
    import time
    
    # Auto mode detection
    no_args = len(sys.argv) == 1
    
    if no_args and AUTO_MODE_ENABLED:
        auto_manager = AutoModeManager()
        
        if auto_manager.is_first_run():
            auto_config = auto_manager.run_first_time_setup()
        else:
            auto_config = auto_manager.get_config()
        
        # Enable trust mode
        sys.argv.extend(["--trust-mode"])
        
        # Handle paper trading bypass
        if auto_config.get("paper_trading_bypassed", False):
            trust_mgr = TrustModeManager()
            trust_mgr.state["paper_validated"] = True
            trust_mgr._save_state()
    
    # Parse arguments
    parser = argparse.ArgumentParser(description="Titan Trade v8.0 - Stock Scanner")
    parser.add_argument("--trust-mode", action="store_true", help="Enable Trust Mode")
    parser.add_argument("--trust-status", action="store_true", help="Show Trust Mode status")
    parser.add_argument("--trust-paper", action="store_true", help="Start paper trading")
    parser.add_argument("--trust-paper-win", action="store_true", help="Record paper win")
    parser.add_argument("--trust-paper-loss", action="store_true", help="Record paper loss")
    parser.add_argument("--trust-bypass", action="store_true", help="Bypass paper validation")
    parser.add_argument("--tickers", default="", help="Custom ticker list")
    parser.add_argument("--account-size", type=float, default=ACCOUNT_SIZE)
    parser.add_argument("--risk-per-trade", type=float, default=RISK_PER_TRADE)
    
    args = parser.parse_args()
    
    # Initialize
    logger = setup_logging()
    trust_manager = TrustModeManager(account_size=args.account_size)
    risk_manager = PortfolioRiskManager(account_size=args.account_size)
    signal_tracker = SignalTracker()
    
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
        print("  TITAN TRADE v8.0")
        print("=" * 60)
    
    # Check risk status
    risk_status = risk_manager.get_risk_status()
    if not risk_status['can_trade']:
        print(f"\n  TRADING BLOCKED: {risk_status['reason']}")
        return
    
    # Parse custom tickers
    tickers = parse_tickers(args.tickers) if args.tickers else None
    
    # Use trust mode settings
    settings = dict(TRUST_MODE_SETTINGS) if args.trust_mode else {}
    
    # Run scan
    try:
        setups, stats, _, vix_level = scan(tickers, settings)
    except KeyboardInterrupt:
        print("\n  Cancelled.")
        return
    
    # Display results
    if args.trust_mode or args.trust_paper:
        trusted = print_simple_verdict(setups, trust_manager, vix_level)
        
        # Auto-track
        if trusted:
            for s in trusted[:AUTO_TRACK_TOP_N]:
                signal_tracker.add_signal(s, s.price)
    else:
        if setups:
            print(f"\n  Found {len(setups)} setups:\n")
            table = []
            for s in setups[:10]:
                table.append([
                    s.ticker, s.strategy[:4], f"${s.price:.2f}",
                    f"${s.trigger:.2f}", s.confidence_grade,
                    f"{s.win_rate:.0f}%", f"{s.profit_factor:.2f}",
                    f"${s.stop:.2f}", f"${s.target:.2f}", s.qty
                ])
            print(tabulate(table, headers=[
                "Ticker", "Type", "Price", "Trigger", "Grade",
                "Win%", "PF", "Stop", "Target", "Shares"
            ], tablefmt="grid"))
        else:
            print("\n  No valid setups found.")
    
    # Show filter stats
    print("\n  Filter Summary:")
    for k, v in stats.items():
        if v > 0:
            print(f"    {k}: {v}")


if __name__ == "__main__":
    main()
