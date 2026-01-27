import yfinance as yf
import pandas as pd
import numpy as np
import requests
import os
import time
import json
import warnings
import argparse
import subprocess
import sys
import logging
import math
from datetime import datetime, timedelta, date
from dataclasses import dataclass
from tabulate import tabulate
import io

# Disable warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
pd.options.mode.chained_assignment = None

# --- CONSTANTS & CONFIG ---
CACHE_DIR = "cache_sp500_elite"
SP500_CACHE_FILE = os.path.join(CACHE_DIR, "sp500_tickers.json")
OHLCV_CACHE_FILE = os.path.join(CACHE_DIR, "sp500_ohlcv_bulk.parquet")
PORTFOLIO_FILE = "portfolio.json"
DEFAULT_SP500_TTL_DAYS = 7
DEFAULT_OHLCV_TTL_HOURS = 12
DEFAULT_DATA_PERIOD = "5y"
DEFAULT_DATA_INTERVAL = "1d"
DEFAULT_OUTPUT_DIR = "."
DEFAULT_LOG_LEVEL = "INFO"
VALID_LOG_LEVELS = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
DEFAULT_MAX_WORKERS = max(2, min(12, (os.cpu_count() or 4)))

DEFAULT_NEAR_MISS_REPORT = True
DEFAULT_NEAR_MISS_TOP = 50

DEFAULT_CONFIRM_DAYS_BREAKOUT = 2
DEFAULT_CONFIRM_DAYS_DIP = 2
DEFAULT_REQUIRE_CONFIRMED_SETUP = True
DEFAULT_REGIME_FACTORS = {
    "BULL": 1.0,
    "RECOVERY": 1.05,
    "NEUTRAL": 1.1,
    "Correction": 1.15,
    "BEAR": 1.25,
}

DEFAULT_REQUIRE_OOS = False
DEFAULT_OOS_MIN_TRADES = 5
DEFAULT_OOS_MIN_WINRATE_BREAKOUT = 55.0
DEFAULT_OOS_MIN_WINRATE_DIP = 52.0
DEFAULT_OOS_MIN_PF_BREAKOUT = 1.2
DEFAULT_OOS_MIN_PF_DIP = 1.1
DEFAULT_OOS_MIN_EXPECTANCY_BREAKOUT = 0.0
DEFAULT_OOS_MIN_EXPECTANCY_DIP = 0.0

SAFE_MODE_DEFAULT = True
SAFE_MODE_SETTINGS = {
    "require_walkforward": True,
    "require_oos": True,
    "confirm_days_breakout": 3,
    "confirm_days_dip": 3,
    "require_confirmed_setup": True,
    "min_winrate_breakout": 60.0,
    "min_winrate_dip": 55.0,
    "min_pf_breakout": 1.8,
    "min_pf_dip": 1.3,
    "min_trades_breakout": 5,  # Reduced from 8 for more signals
    "min_trades_dip": 6,  # Reduced from 10 for more signals
    "min_expectancy_breakout": 0.001,
    "min_expectancy_dip": 0.0005,
    "min_rr_breakout": 1.8,  # Reduced from 2.2 for more signals
    "min_rr_dip": 1.4,  # Reduced from 1.6 for more signals
    "wf_min_trades": 3,
    "wf_min_pf": 1.2,
    "wf_min_expectancy": 0.0,
    "wf_min_passrate": 0.0,
    "oos_min_trades": 5,
    "oos_min_winrate_breakout": 55.0,
    "oos_min_winrate_dip": 52.0,
    "oos_min_pf_breakout": 1.3,
    "oos_min_pf_dip": 1.2,
    "oos_min_expectancy_breakout": 0.0,
    "oos_min_expectancy_dip": 0.0,
}

# Strategy Defaults
RISK_PER_TRADE = 1000.0  # $1000 Risk per trade
ACCOUNT_SIZE = 100000.0

# Pro Trader Settings (New)
MAX_POSITIONS = 8  # Maximum concurrent positions
MAX_SECTOR_EXPOSURE = 3  # Max stocks per sector
VIX_HIGH_THRESHOLD = 25  # Reduce size when VIX > this
VIX_EXTREME_THRESHOLD = 30  # Cut size 50% when VIX > this
GAP_PROTECTION = True  # Filter stocks with large overnight gaps
MAX_GAP_PCT = 0.05  # 5% max gap history allowed

# Quality Filters (defaults preserve current behavior)
DEFAULT_MIN_WIN_RATE_BREAKOUT = 50.0
DEFAULT_MIN_WIN_RATE_DIP = 55.0
DEFAULT_MIN_PF_BREAKOUT = 1.2
DEFAULT_MIN_PF_DIP = 0.0
DEFAULT_MIN_TRADES_BREAKOUT = 0
DEFAULT_MIN_TRADES_DIP = 0
DEFAULT_MIN_EXPECTANCY_BREAKOUT = None
DEFAULT_MIN_EXPECTANCY_DIP = None
DEFAULT_MIN_RR_BREAKOUT = 1.8
DEFAULT_MIN_RR_DIP = 1.4

# Walk-forward filters (backtest_titan_results.csv)
WF_MIN_TRADES = 5
WF_MIN_PF = 1.2
WF_MIN_EXPECTANCY = 0.0
WF_MIN_PASSRATE = 0.6
REGIME_MIN_SCORE = 0.6
EARNINGS_BLACKOUT_DAYS = 7
EARNINGS_POST_DAYS = 1


def _true_range(high, low, prev_close):
    return pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)


def _atr_series(df, period=14):
    tr = _true_range(df["High"], df["Low"], df["Close"].shift(1))
    return tr.rolling(period).mean()


def _ensure_multiindex(data, tickers):
    if isinstance(data.columns, pd.MultiIndex):
        return data
    if len(tickers) == 1:
        fixed = data.copy()
        fixed.columns = pd.MultiIndex.from_product([tickers, data.columns])
        return fixed
    raise ValueError("Downloaded data is missing ticker-level columns.")


def _expectancy(trades):
    if not trades:
        return 0.0
    return float(np.mean(trades))


def _ensure_dir(path):
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def _load_config(path):
    if not path:
        return {}
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _parse_tickers(raw):
    if not raw:
        return []
    if isinstance(raw, str):
        parts = [p.strip().upper() for p in raw.split(",")]
    elif isinstance(raw, (list, tuple, set)):
        parts = [str(p).strip().upper() for p in raw]
    else:
        return []
    return [p for p in parts if p]


def _load_tickers_from_file(path):
    if not path or not os.path.exists(path):
        return []
    try:
        if path.lower().endswith(".json"):
            data = _load_config(path)
            if isinstance(data, dict):
                data = data.get("tickers", [])
            if isinstance(data, list):
                return _parse_tickers(data)
            return []
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        raw = [p for p in text.replace("\n", ",").split(",") if p.strip()]
        return _parse_tickers(raw)
    except Exception:
        return []


def _parse_regime_factors(value):
    if not value:
        return {}
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            data = json.loads(value)
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}
    return {}


def _apply_safe_mode(args):
    if not getattr(args, "safe_mode", False):
        return args
    for key, value in SAFE_MODE_SETTINGS.items():
        setattr(args, key, value)
    args.auto_oos = True
    args.near_miss_report = True
    args.save_json = True
    return args


def _resolve_output_paths(output_dir, save_history=True):
    _ensure_dir(output_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    paths = {
        "scan_csv_latest": os.path.join(output_dir, "scan_results.csv"),
        "scan_txt_latest": os.path.join(output_dir, "scan_results.txt"),
        "scan_json_latest": os.path.join(output_dir, "scan_results.json"),
        "near_miss_csv_latest": os.path.join(output_dir, "near_miss.csv"),
        "near_miss_json_latest": os.path.join(output_dir, "near_miss.json"),
        "config_json_latest": os.path.join(output_dir, "run_config.json"),
        "log_file": os.path.join(output_dir, f"titan_trade_{timestamp}.log") if save_history else os.path.join(output_dir, "titan_trade.log"),
    }
    if save_history:
        paths.update({
            "scan_csv": os.path.join(output_dir, f"scan_results_{timestamp}.csv"),
            "scan_txt": os.path.join(output_dir, f"scan_results_{timestamp}.txt"),
            "scan_json": os.path.join(output_dir, f"scan_results_{timestamp}.json"),
            "near_miss_csv": os.path.join(output_dir, f"near_miss_{timestamp}.csv"),
            "near_miss_json": os.path.join(output_dir, f"near_miss_{timestamp}.json"),
            "config_json": os.path.join(output_dir, f"run_config_{timestamp}.json"),
        })
    else:
        # Point to latest files (no history)
        paths.update({
            "scan_csv": paths["scan_csv_latest"],
            "scan_txt": paths["scan_txt_latest"],
            "scan_json": paths["scan_json_latest"],
            "near_miss_csv": paths["near_miss_csv_latest"],
            "near_miss_json": paths["near_miss_json_latest"],
            "config_json": paths["config_json_latest"],
        })
    return paths


def setup_logging(level=DEFAULT_LOG_LEVEL, log_file=None):
    level = (level or DEFAULT_LOG_LEVEL).upper()
    if level not in VALID_LOG_LEVELS:
        level = DEFAULT_LOG_LEVEL
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=handlers,
    )
    logging.captureWarnings(True)
    return logging.getLogger("titan")


LOGGER = logging.getLogger("titan")


def load_wf_results(path):
    if not path or not os.path.exists(path):
        return {}
    try:
        df = pd.read_csv(path)
    except Exception:
        return {}
    if "Ticker" not in df.columns:
        return {}
    df = df.set_index("Ticker")
    return df.to_dict(orient="index")


def _needs_backtest(wf_path, extra_paths=None):
    if not wf_path:
        return False
    if not os.path.exists(wf_path):
        return True
    wf_mtime = os.path.getmtime(wf_path)
    paths = [OHLCV_CACHE_FILE, __file__, os.path.join(os.getcwd(), "backtest_titan.py")]
    if extra_paths:
        paths.extend(extra_paths)
    for p in paths:
        if p and os.path.exists(p) and os.path.getmtime(p) > wf_mtime:
            return True
    return False


def _run_backtest(args):
    cmd = [
        sys.executable,
        "backtest_titan.py",
        "--max-tickers",
        str(args.auto_max_tickers),
        "--min-trades",
        str(args.auto_min_trades),
        "--walk-forward",
        "--wf-folds",
        str(args.auto_wf_folds),
        "--wf-test-ratio",
        str(args.auto_wf_test_ratio),
        "--cost-bps",
        str(args.auto_cost_bps),
        "--slippage-bps",
        str(args.auto_slippage_bps),
    ]
    if args.auto_oos:
        cmd.extend([
            "--oos",
            "--oos-train-ratio",
            str(args.auto_oos_train_ratio),
            "--oos-min-test-bars",
            str(args.auto_oos_min_test_bars),
        ])
    print("Auto-running backtest: " + " ".join(cmd))
    result = subprocess.run(cmd)
    return result.returncode == 0


def _parse_earnings_date(value):
    if value is None:
        return None
    if isinstance(value, (list, tuple, np.ndarray, pd.Series)):
        for item in value:
            parsed = _parse_earnings_date(item)
            if parsed:
                return parsed
        return None
    if isinstance(value, pd.Timestamp):
        return value.to_pydatetime().date()
    if isinstance(value, np.datetime64):
        try:
            return pd.to_datetime(value).date()
        except Exception:
            return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        try:
            return pd.to_datetime(value).date()
        except Exception:
            return None
    return None


def _extract_earnings_date(ticker_obj, info):
    if info:
        for key in ("earningsDate", "nextEarningsDate", "earningsTimestamp"):
            if key in info:
                parsed = _parse_earnings_date(info.get(key))
                if parsed:
                    return parsed
    try:
        cal = ticker_obj.calendar
        if isinstance(cal, pd.DataFrame) and not cal.empty:
            if "Earnings Date" in cal.index:
                return _parse_earnings_date(cal.loc["Earnings Date"].values)
            if "Earnings Date" in cal.columns:
                return _parse_earnings_date(cal["Earnings Date"].values)
    except Exception:
        return None
    return None


def _wf_metrics_for_strategy(wf_row, strategy_name):
    if not wf_row:
        return {}
    prefix = "Breakout" if strategy_name.startswith("BREAKOUT") else "Dip"
    return {
        "trades": wf_row.get(f"{prefix}_WF_Test_Trades_total", wf_row.get(f"{prefix}_WF_Test_Trades", 0)),
        "pf": wf_row.get(f"{prefix}_WF_Test_PF_trade_weighted", wf_row.get(f"{prefix}_WF_Test_PF", 0)),
        "expectancy": wf_row.get(
            f"{prefix}_WF_Test_Expectancy_trade_weighted", wf_row.get(f"{prefix}_WF_Test_Expectancy", -1)
        ),
        "pass_rate": wf_row.get(f"{prefix}_WF_PassRate", 0),
        "regime_score": wf_row.get(f"{prefix}_RegimeScore", None),
    }


def _oos_metrics_for_strategy(wf_row, strategy_name):
    if not wf_row:
        return {}
    prefix = "Breakout" if strategy_name.startswith("BREAKOUT") else "Dip"
    return {
        "trades": wf_row.get(f"{prefix}_OOS_Trades", None),
        "pf": wf_row.get(f"{prefix}_OOS_PF", None),
        "expectancy": wf_row.get(f"{prefix}_OOS_Expectancy", None),
        "win_rate": wf_row.get(f"{prefix}_OOS_WR", None),
    }


def _compute_badge(
    strategy_name,
    wf_row,
    min_regime_score,
    wf_min_trades=WF_MIN_TRADES,
    wf_min_pf=WF_MIN_PF,
    wf_min_expectancy=WF_MIN_EXPECTANCY,
    wf_min_passrate=WF_MIN_PASSRATE,
):
    if not wf_row:
        return "PASS"
    metrics = _wf_metrics_for_strategy(wf_row, strategy_name)
    if not metrics:
        return "PASS"
    try:
        trades = float(metrics.get("trades", 0))
        pf = float(metrics.get("pf", 0))
        exp = float(metrics.get("expectancy", -1))
        pass_rate = float(metrics.get("pass_rate", 0))
    except Exception:
        return "FAIL"
    reg = metrics.get("regime_score", None)
    try:
        reg = float(reg) if reg is not None else None
    except Exception:
        reg = None
    failed = (
        trades < wf_min_trades
        or pf < wf_min_pf
        or exp <= wf_min_expectancy
        or pass_rate < wf_min_passrate
    )
    if reg is not None and reg < min_regime_score:
        failed = True
    return "FAIL" if failed else "PASS"


# -----------------------------------------------------------------------------
# 0. HELPERS & TRACKING
# -----------------------------------------------------------------------------
class RejectionTracker:
    def __init__(self):
        self.stats = {
            "Total": 0,
            "No Data": 0,
            "Low Price/Liquidity": 0,
            "Downtrend (Bear)": 0,
            "No Setup (VCP/Dip)": 0,
            "Rejected (Low Win%)": 0,
            "Rejected (Quality)": 0,
            "Bad Risk/Reward": 0,
            "Earnings Risk": 0,
            "Gap Risk": 0,  # NEW: Filter for gappy stocks
            "WF Filter": 0,
            "OOS Filter": 0,
            "Regime Filter": 0,
            "Error": 0,
            "Passed": 0
        }
    
    def update(self, reason):
        self.stats["Total"] += 1
        if reason in self.stats:
            self.stats[reason] += 1
        else:
            self.stats[reason] = 1

    def summary(self):
        return self.stats

@dataclass
class TitanSetup:
    ticker: str
    strategy: str      # "BREAKOUT" or "DIP"
    price: float
    trigger: float
    stop: float
    target: float
    qty: int
    win_rate: float    # From Reality Check
    profit_factor: float
    kelly: float       # Suggested size multiplier
    score: float       # Total Confidence Score
    sector: str
    earnings_call: str # New field for clarity
    note: str
    # Advanced AI Fields
    ml_score: float = 0.0        # ML-based confidence (0-100)
    options_flow: str = "N/A"    # BULLISH/BEARISH/NEUTRAL
    sentiment: str = "N/A"       # BULLISH/BEARISH/NEUTRAL
    ai_boost: float = 0.0        # Total AI score modifier

# -----------------------------------------------------------------------------
# AI ENHANCEMENT MODULE (Advanced Features)
# -----------------------------------------------------------------------------
class AIEnhancer:
    """
    Advanced AI features for signal enhancement:
    1. ML Signal Filtering (Random Forest-like scoring)
    2. Options Flow Detection (put/call ratio analysis)
    3. News Sentiment Analysis (RSS/headline scanning)
    """
    
    def __init__(self):
        self.sentiment_cache = {}  # Cache sentiment results
        self._session = None
    
    @property
    def session(self):
        if self._session is None:
            self._session = requests.Session()
            self._session.headers.update({
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            })
        return self._session
    
    # -------------------------------------------------------------------------
    # 1. ML SIGNAL FILTER (Feature-based scoring)
    # -------------------------------------------------------------------------
    def ml_score(self, df, backtest_res):
        """
        ML-style signal scoring using handcrafted features.
        Returns: Score 0-100 (higher = better trade probability)
        
        Features used:
        - Backtest win rate
        - Profit factor
        - Trend strength (SMA alignment)
        - Volume confirmation
        - Volatility regime
        - Momentum indicators
        """
        if len(df) < 60:
            return 50.0
        
        score = 0
        
        # Feature 1: Backtest Quality (0-25 points)
        wr = backtest_res.get('win_rate', 50)
        pf = backtest_res.get('pf', 1.0)
        score += min(15, (wr - 50) * 0.5)  # +15 for 80% WR
        score += min(10, (pf - 1.0) * 5)    # +10 for PF 3.0
        
        c = df['Close']
        h = df['High']
        v = df['Volume']
        
        # Feature 2: Trend Strength (0-20 points)
        sma20 = c.rolling(20).mean().iloc[-1]
        sma50 = c.rolling(50).mean().iloc[-1]
        sma200 = c.rolling(200).mean().iloc[-1] if len(df) >= 200 else sma50
        
        if c.iloc[-1] > sma20 > sma50 > sma200:
            score += 20  # Perfect trend alignment
        elif c.iloc[-1] > sma50 > sma200:
            score += 12
        elif c.iloc[-1] > sma200:
            score += 5
        
        # Feature 3: Volume Confirmation (0-15 points)
        vol_avg = v.rolling(20).mean().iloc[-1]
        vol_ratio = v.iloc[-1] / (vol_avg + 1e-9)
        if vol_ratio > 2.0:
            score += 15  # Strong volume surge
        elif vol_ratio > 1.5:
            score += 10
        elif vol_ratio > 1.0:
            score += 5
        
        # Feature 4: Near High (0-15 points) - Blue Sky territory
        high_52w = h.iloc[-252:].max() if len(df) >= 252 else h.max()
        pct_from_high = (c.iloc[-1] / high_52w - 1) * 100
        if pct_from_high > -2:
            score += 15  # At new highs
        elif pct_from_high > -5:
            score += 10
        elif pct_from_high > -10:
            score += 5
        
        # Feature 5: Momentum (0-15 points)
        ret_5d = (c.iloc[-1] / c.iloc[-6] - 1) * 100 if len(df) >= 6 else 0
        ret_20d = (c.iloc[-1] / c.iloc[-21] - 1) * 100 if len(df) >= 21 else 0
        
        if ret_5d > 0 and ret_20d > 0:
            score += 10
        if ret_5d > 3:
            score += 5  # Strong recent momentum
        
        # Feature 6: Volatility Squeeze (0-10 points)
        atr = _atr_series(df).iloc[-1] if len(df) >= 14 else 0
        atr_pct = atr / c.iloc[-1] if c.iloc[-1] > 0 else 0
        if 0.01 < atr_pct < 0.03:
            score += 10  # Goldilocks volatility
        elif atr_pct < 0.05:
            score += 5
        
        return min(100, max(0, score))
    
    # -------------------------------------------------------------------------
    # 2. OPTIONS FLOW DETECTION
    # -------------------------------------------------------------------------
    def get_options_flow(self, ticker):
        """
        Analyze put/call ratio and unusual options activity.
        Returns: "BULLISH", "BEARISH", or "NEUTRAL"
        
        Uses Yahoo Finance options data (free).
        """
        try:
            t = yf.Ticker(ticker)
            
            # Get nearest expiration
            expirations = t.options
            if not expirations:
                return "NEUTRAL", 0
            
            # Use nearest expiration
            nearest_exp = expirations[0]
            chain = t.option_chain(nearest_exp)
            
            calls = chain.calls
            puts = chain.puts
            
            if calls.empty or puts.empty:
                return "NEUTRAL", 0
            
            # Calculate put/call ratio by volume
            call_vol = calls['volume'].sum() if 'volume' in calls.columns else 0
            put_vol = puts['volume'].sum() if 'volume' in puts.columns else 0
            
            if call_vol == 0 and put_vol == 0:
                return "NEUTRAL", 0
            
            pc_ratio = put_vol / (call_vol + 1e-9)
            
            # Calculate unusual activity score
            call_oi = calls['openInterest'].sum() if 'openInterest' in calls.columns else 1
            put_oi = puts['openInterest'].sum() if 'openInterest' in puts.columns else 1
            
            # Volume to OI ratio indicates unusual activity
            call_voi = call_vol / (call_oi + 1e-9)
            put_voi = put_vol / (put_oi + 1e-9)
            
            # Interpret
            if pc_ratio < 0.5 and call_voi > 0.5:
                return "BULLISH", min(20, call_voi * 10)
            elif pc_ratio > 2.0 and put_voi > 0.5:
                return "BEARISH", -min(20, put_voi * 10)
            else:
                return "NEUTRAL", 0
                
        except Exception:
            return "NEUTRAL", 0
    
    # -------------------------------------------------------------------------
    # 3. NEWS SENTIMENT ANALYSIS
    # -------------------------------------------------------------------------
    def get_news_sentiment(self, ticker):
        """
        Analyze recent news headlines for sentiment.
        Returns: "BULLISH", "BEARISH", or "NEUTRAL" and score modifier
        
        Uses Yahoo Finance news feed (free).
        """
        # Check cache first (avoid repeated calls)
        cache_key = f"{ticker}_{datetime.now().strftime('%Y%m%d%H')}"
        if cache_key in self.sentiment_cache:
            return self.sentiment_cache[cache_key]
        
        try:
            t = yf.Ticker(ticker)
            news = t.news if hasattr(t, 'news') else []
            
            if not news:
                result = ("NEUTRAL", 0)
                self.sentiment_cache[cache_key] = result
                return result
            
            # Bullish keywords
            bullish_words = [
                'surge', 'soar', 'jump', 'bull', 'upgrade', 'beat', 'strong',
                'record', 'growth', 'rally', 'breakout', 'buy', 'outperform',
                'profit', 'gain', 'rise', 'up', 'positive', 'boom', 'success'
            ]
            
            # Bearish keywords
            bearish_words = [
                'crash', 'fall', 'drop', 'bear', 'downgrade', 'miss', 'weak',
                'decline', 'loss', 'sell', 'underperform', 'warning', 'concern',
                'down', 'negative', 'fail', 'risk', 'cut', 'layoff', 'lawsuit'
            ]
            
            bullish_count = 0
            bearish_count = 0
            
            for article in news[:10]:  # Check last 10 articles
                title = article.get('title', '').lower()
                
                for word in bullish_words:
                    if word in title:
                        bullish_count += 1
                        
                for word in bearish_words:
                    if word in title:
                        bearish_count += 1
            
            # Determine sentiment
            net_sentiment = bullish_count - bearish_count
            
            if net_sentiment >= 3:
                result = ("BULLISH", min(15, net_sentiment * 3))
            elif net_sentiment <= -3:
                result = ("BEARISH", max(-15, net_sentiment * 3))
            else:
                result = ("NEUTRAL", net_sentiment * 2)
            
            self.sentiment_cache[cache_key] = result
            return result
            
        except Exception:
            return ("NEUTRAL", 0)
    
    # -------------------------------------------------------------------------
    # COMBINED AI ANALYSIS
    # -------------------------------------------------------------------------
    def analyze(self, ticker, df, backtest_res):
        """
        Run all AI analyses and return combined results.
        """
        results = {
            'ml_score': self.ml_score(df, backtest_res),
            'options_flow': 'N/A',
            'options_boost': 0,
            'sentiment': 'N/A', 
            'sentiment_boost': 0,
            'ai_boost': 0
        }
        
        # Options Flow (can be slow, only for promising stocks)
        if results['ml_score'] >= 60:
            flow, boost = self.get_options_flow(ticker)
            results['options_flow'] = flow
            results['options_boost'] = boost
        
        # News Sentiment
        sentiment, s_boost = self.get_news_sentiment(ticker)
        results['sentiment'] = sentiment
        results['sentiment_boost'] = s_boost
        
        # Calculate total AI boost
        results['ai_boost'] = (
            (results['ml_score'] - 50) * 0.2 +  # ML contribution
            results['options_boost'] +           # Options contribution
            results['sentiment_boost']           # Sentiment contribution
        )
        
        return results

# Global AI Enhancer instance
AI_ENHANCER = AIEnhancer()

# -----------------------------------------------------------------------------
# 1. MARKET REGIME (The "Traffic Light")
# -----------------------------------------------------------------------------
class MarketRegime:
    def __init__(self, data):
        self.data = data # Dictionary of DataFrames

    def analyze_spy(self):
        """
        Analyze SPY to determine market status with VIX integration.
        BULL: Price > SMA200 & SMA50 > SMA200
        BEAR: Price < SMA200
        NEUTRAL: Choppy
        """
        if "SPY" not in self.data:
            return "NEUTRAL", 0.5

        spy = self.data["SPY"]
        if isinstance(spy, pd.Series): # Handle single column edge case
             return "NEUTRAL", 0.5
             
        c = spy['Close']
        sma50 = c.rolling(50).mean().iloc[-1]
        sma200 = c.rolling(200).mean().iloc[-1]
        curr = c.iloc[-1]
        
        status = "NEUTRAL"
        score = 0.5
        
        if curr > sma200:
            if sma50 > sma200: 
                status = "BULL"
                score = 1.0 # Full Gas
            else: 
                status = "RECOVERY"
                score = 0.7 # Caution
        else:
            if curr < sma50:
                status = "BEAR"
                score = 0.0 # Stop
            else:
                status = "Correction"
                score = 0.2
        
        # --- VIX INTEGRATION (Pro Trader Secret) ---
        # High VIX = high fear = reduce exposure
        vix_scalar = 1.0
        for vix_key in ["^VIX", "VIX", "VIXY"]:
            if vix_key in self.data:
                try:
                    vix_df = self.data[vix_key]
                    if isinstance(vix_df, pd.DataFrame) and 'Close' in vix_df.columns:
                        vix = float(vix_df['Close'].iloc[-1])
                        if vix > VIX_EXTREME_THRESHOLD:
                            vix_scalar = 0.3  # Extreme fear - cut exposure 70%
                            status = f"{status}+FEAR"
                        elif vix > VIX_HIGH_THRESHOLD:
                            vix_scalar = 0.6  # High fear - cut exposure 40%
                            status = f"{status}+CAUTION"
                        break
                except Exception:
                    pass
                
        return status, score * vix_scalar

# -----------------------------------------------------------------------------
# 2. VALIDATOR ENGINE (The "Reality Check")
# -----------------------------------------------------------------------------
class StrategyValidator:
    """
    Backtests a specific strategy logic on a specific single stock logic 
    over the last 1-2 years to see if it actually works.
    """
    def __init__(self, df):
        self.df = df
    
    def volatility_squeeze(self, bb_period=20, kc_period=20, kc_mult=1.5):
        """
        Detect volatility squeeze (Bollinger bands inside Keltner channels).
        This is the #1 secret of professional breakout traders.
        Returns: Series of True where squeeze is active.
        """
        df = self.df
        close = df['Close']
        
        # Bollinger Bands
        bb_mid = close.rolling(bb_period).mean()
        bb_std = close.rolling(bb_period).std()
        bb_upper = bb_mid + 2 * bb_std
        bb_lower = bb_mid - 2 * bb_std
        
        # Keltner Channels
        atr = _atr_series(df, kc_period)
        kc_mid = close.rolling(kc_period).mean()
        kc_upper = kc_mid + kc_mult * atr
        kc_lower = kc_mid - kc_mult * atr
        
        # Squeeze = BB inside KC (volatility is compressed)
        squeeze = (bb_lower > kc_lower) & (bb_upper < kc_upper)
        return squeeze
    
    def check_gap_risk(self, max_gap_pct=MAX_GAP_PCT, lookback=60):
        """
        Filter out stocks with history of large overnight gaps.
        These will blow past your stop loss.
        Returns: True if stock is SAFE (few large gaps), False if risky.
        """
        df = self.df
        if len(df) < lookback + 1:
            return True  # Not enough data, assume safe
        
        opens = df['Open'].iloc[-lookback:]
        prev_close = df['Close'].shift(1).iloc[-lookback:]
        gaps = abs(opens - prev_close) / (prev_close + 1e-9)
        
        # If stock has had >3 large gaps in last 60 days, it's risky
        large_gap_count = (gaps > max_gap_pct).sum()
        return large_gap_count <= 3
    
    def relative_strength_vs_spy(self, spy_df, lookback=60):
        """
        Calculate Relative Strength vs SPY (Pro Trader Secret).
        Only trade stocks that are OUTPERFORMING the market.
        Returns: RS Score (0-100), >50 means stock is outperforming SPY.
        """
        if len(self.df) < lookback or len(spy_df) < lookback:
            return 50.0  # Neutral if not enough data
        
        stock_ret = (self.df['Close'].iloc[-1] / self.df['Close'].iloc[-lookback] - 1) * 100
        spy_ret = (spy_df['Close'].iloc[-1] / spy_df['Close'].iloc[-lookback] - 1) * 100
        
        # RS = outperformance vs SPY
        rs_diff = stock_ret - spy_ret
        
        # Convert to 0-100 scale (centered at 50)
        # +10% outperformance = 100, -10% underperformance = 0
        rs_score = max(0, min(100, 50 + (rs_diff * 5)))
        return rs_score
    
    def is_blue_sky_breakout(self, lookback=252):
        """
        Detect Blue Sky Breakout (Pro Trader Secret).
        Stock near all-time high = less resistance overhead = easier upside.
        Returns: True if price is within 5% of 52-week high.
        """
        if len(self.df) < lookback:
            return False
        
        high_52w = self.df['High'].iloc[-lookback:].max()
        current = self.df['Close'].iloc[-1]
        
        # Within 5% of 52-week high = blue sky
        return current >= high_52w * 0.95
    
    def momentum_score(self, spy_df=None):
        """
        Calculate Momentum Score (0-100) combining multiple factors.
        Higher = stronger momentum = better trade probability.
        """
        score = 50  # Start neutral
        
        if len(self.df) < 60:
            return score
        
        c = self.df['Close']
        
        # Factor 1: Price vs SMA20 (+10)
        sma20 = c.rolling(20).mean().iloc[-1]
        if c.iloc[-1] > sma20:
            score += 10
        
        # Factor 2: Price vs SMA50 (+10)
        sma50 = c.rolling(50).mean().iloc[-1]
        if c.iloc[-1] > sma50:
            score += 10
        
        # Factor 3: SMA20 > SMA50 (trend confirmation) (+10)
        if sma20 > sma50:
            score += 10
        
        # Factor 4: Volume surge (current vol > 1.5x avg) (+10)
        vol_avg = self.df['Volume'].rolling(20).mean().iloc[-1]
        if self.df['Volume'].iloc[-1] > vol_avg * 1.5:
            score += 10
        
        # Factor 5: Blue sky breakout (+10)
        if self.is_blue_sky_breakout():
            score += 10
        
        # Factor 6: RS vs SPY (if available) (+0 to +10)
        if spy_df is not None:
            rs = self.relative_strength_vs_spy(spy_df)
            score += (rs - 50) / 5  # +10 for RS=100, -10 for RS=0
        
        return min(100, max(0, score))
    
    def _calculate_rsi(self, series, period=14):
        delta = series.diff()
        gain = delta.where(delta > 0, 0).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        rs = gain / (loss + 1e-9)
        return 100 - (100 / (1 + rs))

    def _simulate_trade(self, entry, stop, target, ohlc_data, start_idx, max_hold=8, trail_risk=None, trail_r1=1.0, trail_r2=2.0, trail_r3=3.0):
        """Simulate a trade with 3-tier trailing stop (Pro Trader Secret)."""
        closes, highs, lows = ohlc_data
        stop_curr = stop
        highest_since_entry = entry  # Track highest price for trailing
        end_idx = min(start_idx + max_hold, len(closes))

        for j in range(start_idx, end_idx):
            # Update highest price since entry
            highest_since_entry = max(highest_since_entry, highs[j])
            
            # 3-Tier Trailing Stop Logic (Pro Trader Secret)
            if trail_risk is not None and trail_risk > 0:
                # Tier 1: Move to breakeven at 1R
                if highest_since_entry >= entry + (trail_risk * trail_r1):
                    stop_curr = max(stop_curr, entry)  # breakeven
                # Tier 2: Lock 1R at 2R profit
                if highest_since_entry >= entry + (trail_risk * trail_r2):
                    stop_curr = max(stop_curr, entry + trail_risk)  # lock 1R
                # Tier 3: Lock 2R at 3R profit (NEW!)
                if highest_since_entry >= entry + (trail_risk * trail_r3):
                    stop_curr = max(stop_curr, entry + trail_risk * 2)  # lock 2R

            # Conservative: check stop first
            if lows[j] <= stop_curr:
                return (stop_curr - entry) / entry
            if highs[j] >= target:
                return (target - entry) / entry

            # Mark to market on last day
            if j == end_idx - 1:
                return (closes[j] - entry) / entry

        return 0.0

    def backtest_breakout(self, days=500, depth=0.20, vol_mult=1.2, target_mult=2.5, stop_mult=2.0, return_trades=False):
        """
        Fast simulation of VCP Breakouts on this specific stock.
        """
        df = self.df.iloc[-days:].copy()
        if len(df) < 100:
            base = {'win_rate': 0, 'pf': 0, 'trades': 0}
            if return_trades:
                base['trades_list'] = []
            return base
        
        closes = df['Close'].values
        highs = df['High'].values
        lows = df['Low'].values
        opens = df['Open'].values
        volumes = df['Volume'].values
        
        # Pre-calc Indicators
        sma50 = df['Close'].rolling(50).mean().values
        sma200 = df['Close'].rolling(200).mean().values
        # ATR Pre-calc (14-day)
        atr = _atr_series(df).values
        vol_sma = df['Volume'].rolling(50).mean().values
        
        trades = []
        
        # Loop (Simulation)
        # Scan from index 60 to end-1
        for i in range(60, len(df)-1):
            # 1. Trend Filter
            if not (closes[i] > sma50[i] > sma200[i]):
                continue
            # Rising trend confirmation
            if i < 70:
                continue
            if not (sma50[i] > sma50[i-10] and sma200[i] > sma200[i-10]):
                continue
            # Volatility filter (avoid extreme high-vol regimes)
            if atr[i] and closes[i] > 0 and (atr[i] / closes[i]) > 0.12:
                continue
            
            # 2. Pattern (VCP)
            # Handle: 15 days window ending at i
            h_handle = np.max(highs[i-15:i+1])
            l_handle = np.min(lows[i-15:i+1])
            curr_c = closes[i]
            
            # Depth
            d = (h_handle - l_handle) / h_handle
            if d > depth: continue # Too loose

            # Tighten near breakout (last 5 days range)
            range_5 = (np.max(highs[i-5:i+1]) - np.min(lows[i-5:i+1])) / max(curr_c, 1e-9)
            if range_5 > 0.08:
                continue
            
            # Near Pivot
            if (h_handle - curr_c) / h_handle > 0.06: continue
            
            # Volume contraction during base
            if not np.isnan(vol_sma[i]):
                base_vol = np.mean(volumes[i-15:i+1])
                if base_vol > (vol_sma[i] * 1.15):
                    continue
            
            # SETUP FOUND at end of day 'i'.
            # CHECK NEXT DAY (i+1) for Trigger
            atr_val = atr[i] if i < len(atr) and not np.isnan(atr[i]) else (curr_c * 0.02)
            pivot = h_handle + (atr_val * 0.05)
            
            next_h = highs[i+1]
            next_l = lows[i+1]
            next_o = opens[i+1]
            next_c = closes[i+1]
            next_v = volumes[i+1]
            next_vol_sma = vol_sma[i+1] if i+1 < len(vol_sma) else np.nan
            
            if next_h > pivot:
                # Breakout volume confirmation (if available)
                if not np.isnan(next_vol_sma) and next_v < (next_vol_sma * vol_mult):
                    continue
                # Require close near/above pivot and decent close
                if next_c < (pivot * 0.995):
                    continue
                day_range = max(next_h - next_l, 1e-9)
                close_pos = (next_c - next_l) / day_range
                if close_pos < 0.5:
                    continue
                # Triggered!
                buy_price = max(pivot, next_o)
                
                atr_val = atr[i] if i < len(atr) and not np.isnan(atr[i]) else (buy_price * 0.02)
                # Stop: Dynamic ATR
                stop_loss = buy_price - (atr_val * stop_mult)
                # Target: Variable (Optimized)
                target = buy_price + (atr_val * target_mult)
                risk = atr_val * stop_mult

                outcome_pct = self._simulate_trade(
                    buy_price,
                    stop_loss,
                    target,
                    (closes, highs, lows),
                    i + 1,
                    max_hold=8,
                    trail_risk=risk,
                )
                trades.append(outcome_pct)
        
        # Stats
        if not trades:
            base = {'win_rate': 0, 'pf': 0, 'trades': 0}
            if return_trades:
                base['trades_list'] = []
            return base
        
        wins = [t for t in trades if t > 0]
        losses = [t for t in trades if t <= 0]
        
        win_rate = float(len(wins) / len(trades) * 100)
        gross_win = float(sum(wins))
        gross_loss = float(abs(sum(losses)))
        pf = float(gross_win / gross_loss if gross_loss > 0 else (100.0 if gross_win > 0 else 0))
        
        res = {'win_rate': win_rate, 'pf': pf, 'trades': len(trades)}
        if return_trades:
            res['trades_list'] = trades
        return res

    def backtest_dip(self, days=500, stop_mult=2.0, target_mult=3.5, return_trades=False):
        """
        Fast simulation of Dip Buys (SMA50 support).
        """
        df = self.df.iloc[-days:].copy()
        if len(df) < 100:
            base = {'win_rate': 0, 'pf': 0, 'trades': 0}
            if return_trades:
                base['trades_list'] = []
            return base
        
        sma50 = df['Close'].rolling(50).mean()
        sma200 = df['Close'].rolling(200).mean()
        rsi14 = self._calculate_rsi(df['Close'])
        closes = df['Close']
        opens = df['Open']
        lows = df['Low']
        highs = df['High']
        volumes = df['Volume']
        vol_sma = df['Volume'].rolling(20).mean()
        # ATR
        atr = _atr_series(df)
        
        trades = []
        
        for i in range(50, len(df)-5):
            # Logic: Trend Up, Pullback to SMA50
            if closes.iloc[i] > sma50.iloc[i] and sma50.iloc[i] > sma200.iloc[i]:
                if sma50.iloc[i] <= sma50.iloc[i-10]:
                    continue
                # Check for "Touch" of SMA50 in recent days
                dist = (lows.iloc[i] - sma50.iloc[i]) / sma50.iloc[i]
                
                if -0.02 < dist < 0.02: # Touched/Near
                    # RSI and volume sanity
                    if rsi14.iloc[i] < 45:
                        continue
                    if not np.isnan(vol_sma.iloc[i]) and volumes.iloc[i] > vol_sma.iloc[i] * 1.3:
                        continue
                    # Require a positive candle (bounce)
                    if closes.iloc[i] <= opens.iloc[i]:
                        continue
                    # Buy at next day open to avoid lookahead bias
                    if i + 1 >= len(df):
                        continue
                    buy_price = opens.iloc[i + 1]
                    if buy_price <= 0:
                        continue
                    # Skip extreme gaps (risk control)
                    if abs(buy_price - closes.iloc[i]) / closes.iloc[i] > 0.05:
                        continue
                    atr_val = atr.iloc[i]
                    if np.isnan(atr_val) or atr_val <= 0:
                        atr_val = buy_price * 0.02
                    # Avoid very high volatility dips
                    if (atr_val / buy_price) > 0.07:
                        continue
                    stop = buy_price - (atr_val * stop_mult)
                    target = buy_price + (atr_val * target_mult)

                    outcome_pct = self._simulate_trade(
                        buy_price,
                        stop,
                        target,
                        (closes.values, highs.values, lows.values),
                        i + 1,
                        max_hold=8,
                        trail_risk=None,
                    )
                    trades.append(outcome_pct)
                    
        if not trades:
            base = {'win_rate': 0, 'pf': 0, 'trades': 0}
            if return_trades:
                base['trades_list'] = []
            return base
        wins = [t for t in trades if t > 0]
        losses = [t for t in trades if t <= 0]
        win_rate = len(wins) / len(trades) * 100
        pf = sum(wins)/abs(sum(losses)) if sum(losses) != 0 else (100.0 if sum(wins) > 0 else 0)
        
        res = {'win_rate': win_rate, 'pf': pf, 'trades': len(trades)}
        if return_trades:
            res['trades_list'] = trades
        return res

# -----------------------------------------------------------------------------
# 3. OPTIMIZER (Auto-Tuning)
# -----------------------------------------------------------------------------
class Optimizer:
    def __init__(self, validator):
        self.validator = validator
        
    def tune_breakout(self):
        """
        Try different parameters including TARGET MULTIPLIER to find the best fit.
        """
        best_res = {'win_rate': 0, 'pf': 0, 'score': 0}
        best_params = {'depth': 0.15, 'target_mult': 3.5}
        
        # Grid Search: Depth vs Target
        # Depth: 0.15 (Tight) to 0.25 (Loose)
        # Target: 2.5 ATR (Quick) to 6.0 ATR (Runner)
        for d in [0.15, 0.20, 0.25]:
            for t_mult in [2.5, 3.5, 5.0, 6.0]:
                res = self.validator.backtest_breakout(depth=d, target_mult=t_mult)
                # Score: PF * WR
                score = res['pf'] * res['win_rate']
                
                if score > best_res['score'] and res['trades'] >= 2:
                    best_res = res
                    best_res['score'] = score
                    best_params = {'depth': d, 'target_mult': t_mult}
                
        return best_res, best_params

# -----------------------------------------------------------------------------
# 4. MAIN BRAIN
# -----------------------------------------------------------------------------
class TitanBrain:
    def __init__(
        self,
        wf_path=None,
        min_regime_score=REGIME_MIN_SCORE,
        earnings_blackout_days=EARNINGS_BLACKOUT_DAYS,
        earnings_post_days=EARNINGS_POST_DAYS,
        risk_per_trade=RISK_PER_TRADE,
        account_size=ACCOUNT_SIZE,
        tickers_override=None,
        cache_ttl_hours=DEFAULT_OHLCV_TTL_HOURS,
        sp500_ttl_days=DEFAULT_SP500_TTL_DAYS,
        data_period=DEFAULT_DATA_PERIOD,
        data_interval=DEFAULT_DATA_INTERVAL,
        force_refresh=False,
        min_win_rate_breakout=DEFAULT_MIN_WIN_RATE_BREAKOUT,
        min_win_rate_dip=DEFAULT_MIN_WIN_RATE_DIP,
        min_pf_breakout=DEFAULT_MIN_PF_BREAKOUT,
        min_pf_dip=DEFAULT_MIN_PF_DIP,
        min_trades_breakout=DEFAULT_MIN_TRADES_BREAKOUT,
        min_trades_dip=DEFAULT_MIN_TRADES_DIP,
        min_expectancy_breakout=DEFAULT_MIN_EXPECTANCY_BREAKOUT,
        min_expectancy_dip=DEFAULT_MIN_EXPECTANCY_DIP,
        min_rr_breakout=DEFAULT_MIN_RR_BREAKOUT,
        min_rr_dip=DEFAULT_MIN_RR_DIP,
        wf_min_trades=WF_MIN_TRADES,
        wf_min_pf=WF_MIN_PF,
        wf_min_expectancy=WF_MIN_EXPECTANCY,
        wf_min_passrate=WF_MIN_PASSRATE,
        require_walkforward=False,
        confirm_days_breakout=DEFAULT_CONFIRM_DAYS_BREAKOUT,
        confirm_days_dip=DEFAULT_CONFIRM_DAYS_DIP,
        require_confirmed_setup=DEFAULT_REQUIRE_CONFIRMED_SETUP,
        regime_factors=None,
        require_oos=DEFAULT_REQUIRE_OOS,
        oos_min_trades=DEFAULT_OOS_MIN_TRADES,
        oos_min_winrate_breakout=DEFAULT_OOS_MIN_WINRATE_BREAKOUT,
        oos_min_winrate_dip=DEFAULT_OOS_MIN_WINRATE_DIP,
        oos_min_pf_breakout=DEFAULT_OOS_MIN_PF_BREAKOUT,
        oos_min_pf_dip=DEFAULT_OOS_MIN_PF_DIP,
        oos_min_expectancy_breakout=DEFAULT_OOS_MIN_EXPECTANCY_BREAKOUT,
        oos_min_expectancy_dip=DEFAULT_OOS_MIN_EXPECTANCY_DIP,
    ):
        if not os.path.exists(CACHE_DIR): os.makedirs(CACHE_DIR)
        self.wf_results = load_wf_results(wf_path) if wf_path else {}
        self.min_regime_score = min_regime_score
        self.earnings_blackout_days = earnings_blackout_days
        self.earnings_post_days = earnings_post_days
        self.risk_per_trade = max(float(risk_per_trade), 0.0)
        self.account_size = max(float(account_size), 0.0)
        self.tickers_override = _parse_tickers(tickers_override)
        self.cache_ttl_hours = max(float(cache_ttl_hours), 0.0)
        self.sp500_ttl_days = max(float(sp500_ttl_days), 0.0)
        self.data_period = data_period or DEFAULT_DATA_PERIOD
        self.data_interval = data_interval or DEFAULT_DATA_INTERVAL
        self.force_refresh = bool(force_refresh)
        self.min_win_rate_breakout = float(min_win_rate_breakout)
        self.min_win_rate_dip = float(min_win_rate_dip)
        self.min_pf_breakout = float(min_pf_breakout)
        self.min_pf_dip = float(min_pf_dip)
        self.min_trades_breakout = int(min_trades_breakout)
        self.min_trades_dip = int(min_trades_dip)
        self.min_expectancy_breakout = (
            float(min_expectancy_breakout) if min_expectancy_breakout is not None else None
        )
        self.min_expectancy_dip = (
            float(min_expectancy_dip) if min_expectancy_dip is not None else None
        )
        self.min_rr_breakout = float(min_rr_breakout)
        self.min_rr_dip = float(min_rr_dip)
        self.wf_min_trades = int(wf_min_trades)
        self.wf_min_pf = float(wf_min_pf)
        self.wf_min_expectancy = float(wf_min_expectancy)
        self.wf_min_passrate = float(wf_min_passrate)
        self.require_walkforward = bool(require_walkforward)
        self.confirm_days_breakout = max(1, int(confirm_days_breakout))
        self.confirm_days_dip = max(1, int(confirm_days_dip))
        self.require_confirmed_setup = bool(require_confirmed_setup)
        merged_factors = dict(DEFAULT_REGIME_FACTORS)
        merged_factors.update(_parse_regime_factors(regime_factors))
        self.regime_factors = merged_factors
        self.require_oos = bool(require_oos)
        self.oos_min_trades = int(oos_min_trades)
        self.oos_min_winrate_breakout = float(oos_min_winrate_breakout)
        self.oos_min_winrate_dip = float(oos_min_winrate_dip)
        self.oos_min_pf_breakout = float(oos_min_pf_breakout)
        self.oos_min_pf_dip = float(oos_min_pf_dip)
        self.oos_min_expectancy_breakout = float(oos_min_expectancy_breakout)
        self.oos_min_expectancy_dip = float(oos_min_expectancy_dip)

    def _regime_factor(self, status):
        return float(self.regime_factors.get(status, 1.0))

    def _candidate_snapshot(
        self,
        ticker,
        strategy,
        price,
        trigger,
        stop,
        target,
        res,
        rr_ratio,
        regime_score=None,
        wf_metrics=None,
        reason="",
        min_win_rate=None,
        min_pf=None,
        min_trades=None,
        min_expectancy=None,
        min_rr=None,
        wf_min_trades=None,
        wf_min_pf=None,
        wf_min_expectancy=None,
        wf_min_passrate=None,
        oos_metrics=None,
        oos_min_trades=None,
        oos_min_winrate=None,
        oos_min_pf=None,
        oos_min_expectancy=None,
    ):
        wf_metrics = wf_metrics or {}
        oos_metrics = oos_metrics or {}
        win_rate = float(res.get("win_rate", 0)) if res else 0.0
        pf = float(res.get("pf", 0)) if res else 0.0
        trades = int(res.get("trades", 0)) if res else 0
        expectancy = float(res.get("expectancy", 0.0)) if res else 0.0
        if min_win_rate is None:
            min_win_rate = self.min_win_rate_breakout if strategy.startswith("BREAKOUT") else self.min_win_rate_dip
        if min_pf is None:
            min_pf = self.min_pf_breakout if strategy.startswith("BREAKOUT") else self.min_pf_dip
        if min_trades is None:
            min_trades = self.min_trades_breakout if strategy.startswith("BREAKOUT") else self.min_trades_dip
        if min_expectancy is None:
            min_expectancy = (
                self.min_expectancy_breakout if strategy.startswith("BREAKOUT") else self.min_expectancy_dip
            )
        if min_rr is None:
            min_rr = self.min_rr_breakout if strategy.startswith("BREAKOUT") else self.min_rr_dip
        if wf_min_trades is None:
            wf_min_trades = self.wf_min_trades
        if wf_min_pf is None:
            wf_min_pf = self.wf_min_pf
        if wf_min_expectancy is None:
            wf_min_expectancy = self.wf_min_expectancy
        if wf_min_passrate is None:
            wf_min_passrate = self.wf_min_passrate
        if oos_min_trades is None:
            oos_min_trades = self.oos_min_trades
        if oos_min_winrate is None:
            oos_min_winrate = self.oos_min_winrate_breakout if strategy.startswith("BREAKOUT") else self.oos_min_winrate_dip
        if oos_min_pf is None:
            oos_min_pf = self.oos_min_pf_breakout if strategy.startswith("BREAKOUT") else self.oos_min_pf_dip
        if oos_min_expectancy is None:
            oos_min_expectancy = (
                self.oos_min_expectancy_breakout
                if strategy.startswith("BREAKOUT")
                else self.oos_min_expectancy_dip
            )
        return {
            "Ticker": ticker,
            "Strategy": strategy,
            "Reason": reason,
            "Price": float(price),
            "Trigger": float(trigger) if trigger else 0.0,
            "Stop": float(stop) if stop else 0.0,
            "Target": float(target) if target else 0.0,
            "RR": float(rr_ratio) if rr_ratio is not None else 0.0,
            "WinRate": win_rate,
            "ProfitFactor": pf,
            "Trades": trades,
            "Expectancy": expectancy,
            "WF_Trades": wf_metrics.get("trades"),
            "WF_PF": wf_metrics.get("pf"),
            "WF_Expectancy": wf_metrics.get("expectancy"),
            "WF_PassRate": wf_metrics.get("pass_rate"),
            "RegimeScore": regime_score,
            "MinWinRate": min_win_rate,
            "MinPF": min_pf,
            "MinTrades": min_trades,
            "MinExpectancy": min_expectancy,
            "MinRR": min_rr,
            "WF_MinTrades": wf_min_trades,
            "WF_MinPF": wf_min_pf,
            "WF_MinExpectancy": wf_min_expectancy,
            "WF_MinPassRate": wf_min_passrate,
            "OOS_WR": oos_metrics.get("win_rate"),
            "OOS_PF": oos_metrics.get("pf"),
            "OOS_Trades": oos_metrics.get("trades"),
            "OOS_Expectancy": oos_metrics.get("expectancy"),
            "OOS_MinTrades": oos_min_trades,
            "OOS_MinWinRate": oos_min_winrate,
            "OOS_MinPF": oos_min_pf,
            "OOS_MinExpectancy": oos_min_expectancy,
        }

    def calculate_rsi(self, series, period=14):
        delta = series.diff()
        gain = delta.where(delta > 0, 0).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        rs = gain / (loss + 1e-9)
        return 100 - (100 / (1 + rs))

    def get_data(self):
        # Tickers
        tickers = self.tickers_override[:]
        if not tickers:
            tickers = None
            sp500_ttl_sec = self.sp500_ttl_days * 86400
            if (
                os.path.exists(SP500_CACHE_FILE)
                and sp500_ttl_sec > 0
                and (time.time() - os.path.getmtime(SP500_CACHE_FILE) < sp500_ttl_sec)
            ):
                try:
                    tickers = pd.read_json(SP500_CACHE_FILE, typ='series').tolist()
                except Exception:
                    tickers = None
            if not tickers:
                print("Fetching S&P List...")
                try:
                    headers = {"User-Agent": "Mozilla/5.0"}
                    resp = requests.get(
                        "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
                        headers=headers,
                        timeout=15,
                    )
                    resp.raise_for_status()
                    df = pd.read_html(io.StringIO(resp.text))[0]
                    tickers = [t.replace('.', '-') for t in df['Symbol'].tolist()]
                    pd.Series(tickers).to_json(SP500_CACHE_FILE)
                except Exception:
                    tickers = ["NVDA", "MSFT", "AAPL", "AMD", "TSLA"]
        tickers = [t.strip().upper() for t in tickers if t]
        tickers = list(dict.fromkeys(tickers))

        # OHLCV
        cache_ttl_sec = self.cache_ttl_hours * 3600
        cache_valid = (
            os.path.exists(OHLCV_CACHE_FILE)
            and cache_ttl_sec > 0
            and (time.time() - os.path.getmtime(OHLCV_CACHE_FILE) < cache_ttl_sec)
        )
        if cache_valid and not self.force_refresh:
            print("Loading Market Cache...")
            try:
                data = pd.read_parquet(OHLCV_CACHE_FILE)
            except Exception:
                print("WARNING: Failed to read cache. Re-downloading data.")
                data = None
        else:
            data = None
        if data is not None and not isinstance(data.columns, pd.MultiIndex):
            print("WARNING: Cache format invalid. Re-downloading data.")
            data = None

        if data is None:
            print("Downloading Market Data (This may take 1-2 minutes)...")
            tickers_plus = list(dict.fromkeys(tickers + ["SPY"]))
            
            # Chunking the download to avoid Rate Limits
            chunk_size = 100
            data_frames = []
            
            for i in range(0, len(tickers_plus), chunk_size):
                chunk = tickers_plus[i:i+chunk_size]
                print(f"  > Batch {i//chunk_size + 1}: {chunk[:3]}...", end='\r')
                try:
                    # threads=False reduces rate-limit issues on some systems
                    d = yf.download(
                        chunk,
                        period=self.data_period,
                        interval=self.data_interval,
                        auto_adjust=True,
                        group_by='ticker',
                        threads=False,
                        progress=False,
                    )
                    if d is None or d.empty:
                        continue
                    d = _ensure_multiindex(d, chunk)
                    data_frames.append(d)
                except Exception as e:
                    print(f"\nExample Batch Failed: {e}")
            
            print("\nMerging Data...")
            if data_frames:
                data = pd.concat(data_frames, axis=1)
                data.to_parquet(OHLCV_CACHE_FILE)
            else:
                raise ValueError("Failed to download any data.")
            
        return tickers, data
        
    def calculate_atr(self, df):
        atr = _atr_series(df)
        last = atr.iloc[-1]
        return float(last) if not np.isnan(last) else 0.0

    def process_ticker(self, t, data, mkt_status, spy_close):
        """Analyze a single ticker. Returns (TitanSetup, RejectionReason, candidate_dict)."""
        try:
            regime_factor = self._regime_factor(mkt_status)
            min_win_rate_breakout = min(self.min_win_rate_breakout * regime_factor, 100.0)
            min_win_rate_dip = min(self.min_win_rate_dip * regime_factor, 100.0)
            min_pf_breakout = self.min_pf_breakout * regime_factor
            min_pf_dip = self.min_pf_dip * regime_factor
            min_rr_breakout = self.min_rr_breakout * regime_factor
            min_rr_dip = self.min_rr_dip * regime_factor
            min_expectancy_breakout = (
                self.min_expectancy_breakout * regime_factor
                if self.min_expectancy_breakout is not None
                else None
            )
            min_expectancy_dip = (
                self.min_expectancy_dip * regime_factor
                if self.min_expectancy_dip is not None
                else None
            )
            min_trades_breakout = int(math.ceil(self.min_trades_breakout * regime_factor))
            min_trades_dip = int(math.ceil(self.min_trades_dip * regime_factor))

            # Extract DF
            if isinstance(data.columns, pd.MultiIndex):
                if t in data.columns.levels[0]:
                    df = data[t].copy()
                else:
                    return None, "No Data", None
            else:
                return None, "No Data", None

            required_cols = ["Open", "High", "Low", "Close", "Volume"]
            if any(col not in df.columns for col in required_cols):
                return None, "No Data", None
            df = df[required_cols].dropna()
            
            if len(df) < 250:
                return None, "No Data", None
            
            # --- A. INITIAL FILTER (Fast) ---
            c = float(df['Close'].iloc[-1])
            if c < 5.0:
                return None, "Low Price/Liquidity", None # Penny stock filter
            
            close = df['Close']
            high = df['High']
            low = df['Low']
            volume = df['Volume']

            sma50 = float(close.rolling(50).mean().iloc[-1])
            sma200 = float(close.rolling(200).mean().iloc[-1])
            sma50_prev = float(close.rolling(50).mean().iloc[-21])
            sma200_prev = float(close.rolling(200).mean().iloc[-21])
            vol_avg20_series = volume.rolling(20).mean()
            vol_avg20 = float(vol_avg20_series.iloc[-1])
            dollar_vol_avg20 = vol_avg20 * c
            if dollar_vol_avg20 < 5_000_000: 
                return None, "Low Price/Liquidity", None

            atr = self.calculate_atr(df)
            if atr <= 0 or atr / c > 0.12:
                return None, "Low Price/Liquidity", None
            
            # Relative strength vs SPY (3M)
            rs_3m_series = close.pct_change(63)
            if spy_close is not None:
                spy_rs_series = spy_close.reindex(close.index).pct_change(63)
                rs_diff_series = rs_3m_series - spy_rs_series
            else:
                rs_diff_series = rs_3m_series
            ret_6m_series = close.pct_change(126)
            rsi14_series = self.calculate_rsi(close)
            sma50_series = close.rolling(50).mean()
            sma200_series = close.rolling(200).mean()
            sma50_prev_series = sma50_series.shift(21)
            sma200_prev_series = sma200_series.shift(21)
            sma_uptrend_series = (sma50_series > sma50_prev_series) & (sma200_series > sma200_prev_series)

            # Setup Flags
            is_breakout_setup = False
            is_dip_setup = False
            candidate = None

            def _breakout_candidate(i):
                if i < 70:
                    return False
                c_i = close.iloc[i]
                s50 = sma50_series.iloc[i]
                s200 = sma200_series.iloc[i]
                if np.isnan(s50) or np.isnan(s200):
                    return False
                rs_diff = rs_diff_series.iloc[i]
                ret_6m = ret_6m_series.iloc[i]
                if np.isnan(rs_diff) or np.isnan(ret_6m):
                    return False
                if not (c_i > s50 > s200 and rs_diff > 0 and ret_6m > 0 and sma_uptrend_series.iloc[i]):
                    return False
                if i < 15:
                    return False
                h_h = float(high.iloc[i-15:i+1].max())
                l_h = float(low.iloc[i-15:i+1].min())
                depth = (h_h - l_h) / h_h if h_h else 1.0
                vol_avg = vol_avg20_series.iloc[i]
                vol_spike = volume.iloc[i] > (vol_avg * 1.2) if not np.isnan(vol_avg) else False
                if depth < 0.25 and (h_h - c_i)/h_h < 0.08 and vol_spike:
                    return True
                return False

            def _dip_candidate(i):
                if i < 70:
                    return False
                c_i = close.iloc[i]
                s50 = sma50_series.iloc[i]
                s200 = sma200_series.iloc[i]
                if np.isnan(s50) or np.isnan(s200):
                    return False
                rs_diff = rs_diff_series.iloc[i]
                ret_6m = ret_6m_series.iloc[i]
                if np.isnan(rs_diff) or np.isnan(ret_6m):
                    return False
                if not (c_i > s200 and rs_diff > -0.02 and ret_6m > -0.02 and sma_uptrend_series.iloc[i]):
                    return False
                dist = (c_i - s50) / s50 if s50 else 0.0
                vol_avg = vol_avg20_series.iloc[i]
                vol_ok = volume.iloc[i] <= (vol_avg * 1.2) if not np.isnan(vol_avg) else True
                rsi14 = rsi14_series.iloc[i]
                if -0.03 < dist < 0.04 and vol_ok and rsi14 > 40:
                    return True
                return False

            idx_end = len(df) - 1
            if self.require_confirmed_setup:
                cb = max(1, self.confirm_days_breakout)
                cd = max(1, self.confirm_days_dip)
                is_breakout_setup = all(_breakout_candidate(i) for i in range(idx_end - cb + 1, idx_end + 1))
                is_dip_setup = all(_dip_candidate(i) for i in range(idx_end - cd + 1, idx_end + 1))
            else:
                is_breakout_setup = _breakout_candidate(idx_end)
                is_dip_setup = _dip_candidate(idx_end)
            
            if not (is_breakout_setup or is_dip_setup): 
                if c < sma200:
                    return None, "Downtrend (Bear)", None
                return None, "No Setup (VCP/Dip)", None
            
            # --- B. REALITY CHECK (Validation) ---
            val = StrategyValidator(df)
            opt = Optimizer(val)
            
            # --- GAP PROTECTION FILTER (Pro Trader Secret) ---
            if GAP_PROTECTION and not val.check_gap_risk():
                return None, "Gap Risk", None
            
            final_res = None
            strategy_name = ""
            trigger = 0
            stop = 0
            target = 0
            candidate_res = None
            candidate_strategy = ""
            
            if is_breakout_setup:
                # 1. Run Backtest
                res = val.backtest_breakout(return_trades=True)
                res["expectancy"] = _expectancy(res.get("trades_list", []))
                params = {} # Default empty params

                needs_opt = (
                    res["win_rate"] < min_win_rate_breakout
                    or res["pf"] < min_pf_breakout
                    or res["trades"] < min_trades_breakout
                    or (min_expectancy_breakout is not None and res["expectancy"] < min_expectancy_breakout)
                )

                # 2. Optimized if needed
                if needs_opt:
                    _, params = opt.tune_breakout()
                    res = val.backtest_breakout(
                        depth=params.get("depth", 0.20),
                        target_mult=params.get("target_mult", 3.5),
                        return_trades=True,
                    )
                    res["expectancy"] = _expectancy(res.get("trades_list", []))

                tgt_mult = params.get('target_mult', 3.5)
                # FIX LOOKAHEAD BIAS: Exclude today's high from pivot calculation
                trigger = float(df['High'].iloc[-16:-1].max()) + 0.02
                stop = trigger - (atr * 2)
                target = trigger + (atr * tgt_mult)
                candidate_strategy = "BREAKOUT"
                if tgt_mult > 4.5:
                    candidate_strategy += "+"
                candidate_res = res

                if (
                    res["win_rate"] >= min_win_rate_breakout
                    and res["pf"] >= min_pf_breakout
                    and res["trades"] >= min_trades_breakout
                    and (
                        min_expectancy_breakout is None
                        or res["expectancy"] >= min_expectancy_breakout
                    )
                ):
                    strategy_name = "BREAKOUT"
                    final_res = res
                    if tgt_mult > 4.5:
                        strategy_name += "+" # Runner Mode
            
            elif is_dip_setup:
                res = val.backtest_dip(return_trades=True)
                res["expectancy"] = _expectancy(res.get("trades_list", []))
                trigger = c
                stop = c - (atr * 2.0)
                target = c + (atr * 3.0)
                candidate_strategy = "DIP BUY"
                candidate_res = res
                if (
                    res["win_rate"] >= min_win_rate_dip
                    and res["pf"] >= min_pf_dip
                    and res["trades"] >= min_trades_dip
                    and (
                        min_expectancy_dip is None
                        or res["expectancy"] >= min_expectancy_dip
                    )
                ):
                    strategy_name = "DIP BUY"
                    final_res = res
            
            if not final_res:
                # Build candidate snapshot for near-miss report
                risk_per_share = trigger - stop
                rr_ratio = (target - trigger) / risk_per_share if risk_per_share > 0 else 0.0
                candidate = self._candidate_snapshot(
                    t,
                    candidate_strategy or strategy_name,
                    c,
                    trigger,
                    stop,
                    target,
                    candidate_res,
                    rr_ratio,
                    reason="Rejected (Quality)",
                    min_win_rate=min_win_rate_breakout if candidate_strategy.startswith("BREAKOUT") else min_win_rate_dip,
                    min_pf=min_pf_breakout if candidate_strategy.startswith("BREAKOUT") else min_pf_dip,
                    min_trades=min_trades_breakout if candidate_strategy.startswith("BREAKOUT") else min_trades_dip,
                    min_expectancy=min_expectancy_breakout if candidate_strategy.startswith("BREAKOUT") else min_expectancy_dip,
                    min_rr=min_rr_breakout if candidate_strategy.startswith("BREAKOUT") else min_rr_dip,
                )
                return None, "Rejected (Quality)", candidate

            # --- WALK-FORWARD VALIDATION FILTER (Optional) ---
            risk_per_share = trigger - stop
            rr_ratio = (target - trigger) / risk_per_share if risk_per_share > 0 else 0.0
            regime_score = None
            oos_metrics = {}
            if self.wf_results:
                wf = self.wf_results.get(t, {})
                if not wf:
                    candidate = self._candidate_snapshot(
                        t,
                        strategy_name,
                        c,
                        trigger,
                        stop,
                        target,
                        final_res,
                        rr_ratio,
                        reason="WF Filter",
                        min_win_rate=min_win_rate_breakout if strategy_name.startswith("BREAKOUT") else min_win_rate_dip,
                        min_pf=min_pf_breakout if strategy_name.startswith("BREAKOUT") else min_pf_dip,
                        min_trades=min_trades_breakout if strategy_name.startswith("BREAKOUT") else min_trades_dip,
                        min_expectancy=min_expectancy_breakout if strategy_name.startswith("BREAKOUT") else min_expectancy_dip,
                        min_rr=min_rr_breakout if strategy_name.startswith("BREAKOUT") else min_rr_dip,
                        oos_metrics=oos_metrics,
                    )
                    return None, "WF Filter", candidate
                wf_metrics = _wf_metrics_for_strategy(wf, strategy_name)
                oos_metrics = _oos_metrics_for_strategy(wf, strategy_name)
                try:
                    wf_trades = float(wf_metrics.get("trades", 0))
                    wf_pf = float(wf_metrics.get("pf", 0))
                    wf_exp = float(wf_metrics.get("expectancy", -1))
                    wf_pass = float(wf_metrics.get("pass_rate", 0))
                except Exception:
                    candidate = self._candidate_snapshot(
                        t,
                        strategy_name,
                        c,
                        trigger,
                        stop,
                        target,
                        final_res,
                        rr_ratio,
                        reason="WF Filter",
                        min_win_rate=min_win_rate_breakout if strategy_name.startswith("BREAKOUT") else min_win_rate_dip,
                        min_pf=min_pf_breakout if strategy_name.startswith("BREAKOUT") else min_pf_dip,
                        min_trades=min_trades_breakout if strategy_name.startswith("BREAKOUT") else min_trades_dip,
                        min_expectancy=min_expectancy_breakout if strategy_name.startswith("BREAKOUT") else min_expectancy_dip,
                        min_rr=min_rr_breakout if strategy_name.startswith("BREAKOUT") else min_rr_dip,
                        oos_metrics=oos_metrics,
                    )
                    return None, "WF Filter", candidate
                try:
                    regime_score = wf_metrics.get("regime_score", None)
                    regime_score = float(regime_score) if regime_score is not None else None
                except Exception:
                    regime_score = None

                if (
                    wf_trades < self.wf_min_trades
                    or wf_pf < self.wf_min_pf
                    or wf_exp <= self.wf_min_expectancy
                    or wf_pass < self.wf_min_passrate
                ):
                    candidate = self._candidate_snapshot(
                        t,
                        strategy_name,
                        c,
                        trigger,
                        stop,
                        target,
                        final_res,
                        rr_ratio,
                        regime_score=regime_score,
                        wf_metrics=wf_metrics,
                        reason="WF Filter",
                        min_win_rate=min_win_rate_breakout if strategy_name.startswith("BREAKOUT") else min_win_rate_dip,
                        min_pf=min_pf_breakout if strategy_name.startswith("BREAKOUT") else min_pf_dip,
                        min_trades=min_trades_breakout if strategy_name.startswith("BREAKOUT") else min_trades_dip,
                        min_expectancy=min_expectancy_breakout if strategy_name.startswith("BREAKOUT") else min_expectancy_dip,
                        min_rr=min_rr_breakout if strategy_name.startswith("BREAKOUT") else min_rr_dip,
                        oos_metrics=oos_metrics,
                    )
                    return None, "WF Filter", candidate
                if regime_score is not None and regime_score < self.min_regime_score:
                    candidate = self._candidate_snapshot(
                        t,
                        strategy_name,
                        c,
                        trigger,
                        stop,
                        target,
                        final_res,
                        rr_ratio,
                        regime_score=regime_score,
                        wf_metrics=wf_metrics,
                        reason="Regime Filter",
                        min_win_rate=min_win_rate_breakout if strategy_name.startswith("BREAKOUT") else min_win_rate_dip,
                        min_pf=min_pf_breakout if strategy_name.startswith("BREAKOUT") else min_pf_dip,
                        min_trades=min_trades_breakout if strategy_name.startswith("BREAKOUT") else min_trades_dip,
                        min_expectancy=min_expectancy_breakout if strategy_name.startswith("BREAKOUT") else min_expectancy_dip,
                        min_rr=min_rr_breakout if strategy_name.startswith("BREAKOUT") else min_rr_dip,
                        oos_metrics=oos_metrics,
                    )
                    return None, "Regime Filter", candidate

                if self.require_oos:
                    try:
                        oos_trades = float(oos_metrics.get("trades", 0))
                        oos_pf = float(oos_metrics.get("pf", 0))
                        oos_exp = float(oos_metrics.get("expectancy", 0))
                        oos_wr = float(oos_metrics.get("win_rate", 0))
                    except Exception:
                        oos_trades = 0
                        oos_pf = 0
                        oos_exp = 0
                        oos_wr = 0

                    min_oos_wr = self.oos_min_winrate_breakout if strategy_name.startswith("BREAKOUT") else self.oos_min_winrate_dip
                    min_oos_pf = self.oos_min_pf_breakout if strategy_name.startswith("BREAKOUT") else self.oos_min_pf_dip
                    min_oos_exp = (
                        self.oos_min_expectancy_breakout
                        if strategy_name.startswith("BREAKOUT")
                        else self.oos_min_expectancy_dip
                    )
                    if (
                        oos_trades < self.oos_min_trades
                        or oos_pf < min_oos_pf
                        or oos_exp < min_oos_exp
                        or oos_wr < min_oos_wr
                    ):
                        candidate = self._candidate_snapshot(
                            t,
                            strategy_name,
                            c,
                            trigger,
                            stop,
                            target,
                            final_res,
                            rr_ratio,
                            regime_score=regime_score,
                            wf_metrics=wf_metrics,
                            reason="OOS Filter",
                            min_win_rate=min_win_rate_breakout if strategy_name.startswith("BREAKOUT") else min_win_rate_dip,
                            min_pf=min_pf_breakout if strategy_name.startswith("BREAKOUT") else min_pf_dip,
                            min_trades=min_trades_breakout if strategy_name.startswith("BREAKOUT") else min_trades_dip,
                            min_expectancy=min_expectancy_breakout if strategy_name.startswith("BREAKOUT") else min_expectancy_dip,
                            min_rr=min_rr_breakout if strategy_name.startswith("BREAKOUT") else min_rr_dip,
                            oos_metrics=oos_metrics,
                            oos_min_trades=self.oos_min_trades,
                            oos_min_winrate=min_oos_wr,
                            oos_min_pf=min_oos_pf,
                            oos_min_expectancy=min_oos_exp,
                        )
                        return None, "OOS Filter", candidate

            # Score Calculation
            score = final_res['win_rate'] + (final_res['pf'] * 10)
            if mkt_status == "BULL": score += 10
            
            # --- KELLY CRITERION FIX (Pro Trader Secret) ---
            # Use actual win/loss ratio from backtest, not hardcoded R=2.0
            W = final_res['win_rate'] / 100
            trades_list = final_res.get('trades_list', [])
            if trades_list:
                wins = [t for t in trades_list if t > 0]
                losses = [t for t in trades_list if t <= 0]
                avg_win = abs(np.mean(wins)) if wins else 0.02
                avg_loss = abs(np.mean(losses)) if losses else 0.01
            else:
                avg_win = 0.02
                avg_loss = 0.01
            R = avg_win / (avg_loss + 1e-9)  # Actual win/loss ratio
            kelly = max(0, (W * R - (1 - W)) / R) * 0.25  # Quarter-Kelly (safer)
            
            # --- VOLATILITY-ADJUSTED POSITION SIZING (Pro Trader Secret) ---
            # Reduce position size for volatile stocks
            atr_pct = atr / c if c > 0 else 0.02
            vol_scalar = min(1.0, 0.025 / (atr_pct + 1e-9))  # Reduce size for volatile stocks
            risk_amt = min(self.risk_per_trade * vol_scalar, self.account_size * 0.01)
            shares = max(1, int(risk_amt / (trigger - stop))) if (trigger - stop) > 0 else 0
            
            # --- SUPER POWER 1: INFO CHECK ---
            earnings_call = "Unknown"
            sector = "Unknown"
            
            # Only fetch info for valid setups (Optimizes speed)
            try:
                # Use partial fetch or cache if possible, but for now specific call is ok for <10 items
                ticker_obj = yf.Ticker(t)
                t_info = ticker_obj.info or {}
                sector = t_info.get('sector', 'Unknown')
                
                # Basic Valuation Check
                fwd_pe = t_info.get('forwardPE', 0)
                if fwd_pe and fwd_pe < 40: score += 5 # Value boost

                earnings_date = _extract_earnings_date(ticker_obj, t_info)
                if earnings_date:
                    today = datetime.now().date()
                    days_to = (earnings_date - today).days
                    earnings_call = f"{earnings_date.isoformat()} ({days_to:+d}d)"
                    if -self.earnings_post_days <= days_to <= self.earnings_blackout_days:
                        candidate = self._candidate_snapshot(
                            t,
                            strategy_name,
                            c,
                            trigger,
                            stop,
                            target,
                            final_res,
                            rr_ratio,
                            regime_score=regime_score,
                            wf_metrics=wf_metrics if self.wf_results else None,
                            reason="Earnings Risk",
                            min_win_rate=min_win_rate_breakout if strategy_name.startswith("BREAKOUT") else min_win_rate_dip,
                            min_pf=min_pf_breakout if strategy_name.startswith("BREAKOUT") else min_pf_dip,
                            min_trades=min_trades_breakout if strategy_name.startswith("BREAKOUT") else min_trades_dip,
                            min_expectancy=min_expectancy_breakout if strategy_name.startswith("BREAKOUT") else min_expectancy_dip,
                            min_rr=min_rr_breakout if strategy_name.startswith("BREAKOUT") else min_rr_dip,
                            oos_metrics=oos_metrics,
                        )
                        return None, "Earnings Risk", candidate
                
            except: pass
            
            # --- SUPER POWER 2: GOLDEN RATIO FILTER (R:R) ---
            reward_per_share = target - trigger
            
            if risk_per_share <= 0:
                candidate = self._candidate_snapshot(
                    t,
                    strategy_name,
                    c,
                    trigger,
                    stop,
                    target,
                    final_res,
                    rr_ratio,
                    regime_score=regime_score,
                    wf_metrics=wf_metrics if self.wf_results else None,
                    reason="Data Error",
                    min_win_rate=min_win_rate_breakout if strategy_name.startswith("BREAKOUT") else min_win_rate_dip,
                    min_pf=min_pf_breakout if strategy_name.startswith("BREAKOUT") else min_pf_dip,
                    min_trades=min_trades_breakout if strategy_name.startswith("BREAKOUT") else min_trades_dip,
                    min_expectancy=min_expectancy_breakout if strategy_name.startswith("BREAKOUT") else min_expectancy_dip,
                    min_rr=min_rr_breakout if strategy_name.startswith("BREAKOUT") else min_rr_dip,
                    oos_metrics=oos_metrics,
                )
                return None, "Data Error", candidate
            rr_ratio = reward_per_share / risk_per_share
            
            min_rr = min_rr_dip if strategy_name == "DIP BUY" else min_rr_breakout
            
            if rr_ratio < min_rr:
                candidate = self._candidate_snapshot(
                    t,
                    strategy_name,
                    c,
                    trigger,
                    stop,
                    target,
                    final_res,
                    rr_ratio,
                    regime_score=regime_score,
                    wf_metrics=wf_metrics if self.wf_results else None,
                    reason="Bad Risk/Reward",
                    min_win_rate=min_win_rate_breakout if strategy_name.startswith("BREAKOUT") else min_win_rate_dip,
                    min_pf=min_pf_breakout if strategy_name.startswith("BREAKOUT") else min_pf_dip,
                    min_trades=min_trades_breakout if strategy_name.startswith("BREAKOUT") else min_trades_dip,
                    min_expectancy=min_expectancy_breakout if strategy_name.startswith("BREAKOUT") else min_expectancy_dip,
                    min_rr=min_rr_breakout if strategy_name.startswith("BREAKOUT") else min_rr_dip,
                    oos_metrics=oos_metrics,
                )
                return None, "Bad Risk/Reward", candidate
            
            # Final Note Construction
            note_str = f"Hist: {final_res['trades']} trades"
            if regime_score is not None:
                note_str += f" | Reg:{regime_score:.2f}"
            if rr_ratio >= 3.0: note_str += " | 3R+ GEM"
            
            # --- AI ENHANCEMENT (ML, Options Flow, Sentiment) ---
            ai_results = AI_ENHANCER.analyze(t, df, final_res)
            ml_score = ai_results['ml_score']
            options_flow = ai_results['options_flow']
            sentiment = ai_results['sentiment']
            ai_boost = ai_results['ai_boost']
            
            # Boost score with AI results
            score += ai_boost
            
            # Add AI info to note
            if ml_score >= 70:
                note_str += f" | ML:{ml_score:.0f}🔥"
            elif ml_score >= 60:
                note_str += f" | ML:{ml_score:.0f}"
            
            if options_flow == "BULLISH":
                note_str += " | Opts:📈"
            elif options_flow == "BEARISH":
                note_str += " | Opts:📉"
            
            if sentiment == "BULLISH":
                note_str += " | News:📰+"
            elif sentiment == "BEARISH":
                note_str += " | News:📰-"
            
            return TitanSetup(
                t, strategy_name, c, trigger, stop, target, shares,
                final_res['win_rate'], final_res['pf'], kelly*100, score, sector, 
                earnings_call, note_str, ml_score, options_flow, sentiment, ai_boost
            ), "Passed", None

        except Exception:
            LOGGER.exception("Ticker processing failed: %s", t)
            return None, "Error", None


    def scan(self, max_workers=10, near_miss_report=False, near_miss_top=DEFAULT_NEAR_MISS_TOP):
        tickers, data = self.get_data()
        spy_close = None
        if isinstance(data.columns, pd.MultiIndex) and "SPY" in data.columns.levels[0]:
            spy_df = data["SPY"].dropna()
            if "Close" in spy_df:
                spy_close = spy_df["Close"]
        
        # 1. Check Market
        regime = MarketRegime(data)
        mkt_status, mkt_score = regime.analyze_spy()
        print(f"\n=== MARKET STATUS: {mkt_status} (Score: {mkt_score}) ===")
        if mkt_score == 0:
            print("!!! MARKET IS IN BEAR TREND. CAUTION ADVISED. !!!")
        
        print("\nScanning & Validating (v5 Reality Check)...")
        print(f"Analyzing {len(tickers)} stocks in parallel (Max {max_workers} workers)...")
        
        results = []
        near_misses = []
        tracker = RejectionTracker()
        
        import concurrent.futures
        
        # Determine number of threads
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all
            future_to_ticker = {
                executor.submit(self.process_ticker, t, data, mkt_status, spy_close): t
                for t in tickers
            }
            
            completed = 0
            for future in concurrent.futures.as_completed(future_to_ticker):
                completed += 1
                if completed % 25 == 0:
                    print(f"Progress: {completed}/{len(tickers)}", end='\r')
                    
                try:
                    res, reason, candidate = future.result()
                except Exception:
                    tracker.update("Error")
                    continue
                tracker.update(reason)

                if candidate and near_miss_report:
                    near_misses.append(candidate)

                if res:
                    results.append(res)
                    print(f" > FOUND: {res.ticker} ({res.strategy}) WR:{res.win_rate:.0f}% PF:{res.profit_factor:.2f}")

        # Show Rejection Stats
        print("\n" + "-"*40)
        print(" SCAN FILTER REPORT")
        print("-" * 40)
        for k, v in tracker.summary().items():
            print(f"  {k.ljust(20)}: {v}")
        print("-" * 40)
        
        if near_miss_report and near_misses:
            def _miss_score(row):
                fails = 0
                if row["WinRate"] < row["MinWinRate"]:
                    fails += 1
                if row["ProfitFactor"] < row["MinPF"]:
                    fails += 1
                if row["Trades"] < row["MinTrades"]:
                    fails += 1
                if row["MinExpectancy"] is not None and row["Expectancy"] < row["MinExpectancy"]:
                    fails += 1
                if row["RR"] < row["MinRR"]:
                    fails += 1
                wf_trades = row.get("WF_Trades")
                wf_pf = row.get("WF_PF")
                wf_exp = row.get("WF_Expectancy")
                wf_pass = row.get("WF_PassRate")
                if wf_trades is not None and wf_trades < row["WF_MinTrades"]:
                    fails += 1
                if wf_pf is not None and wf_pf < row["WF_MinPF"]:
                    fails += 1
                if wf_exp is not None and wf_exp <= row["WF_MinExpectancy"]:
                    fails += 1
                if wf_pass is not None and wf_pass < row["WF_MinPassRate"]:
                    fails += 1
                oos_trades = row.get("OOS_Trades")
                oos_pf = row.get("OOS_PF")
                oos_exp = row.get("OOS_Expectancy")
                oos_wr = row.get("OOS_WR")
                if oos_trades is not None and oos_trades < row["OOS_MinTrades"]:
                    fails += 1
                if oos_pf is not None and oos_pf < row["OOS_MinPF"]:
                    fails += 1
                if oos_exp is not None and oos_exp < row["OOS_MinExpectancy"]:
                    fails += 1
                if oos_wr is not None and oos_wr < row["OOS_MinWinRate"]:
                    fails += 1
                return (fails, -row["ProfitFactor"], -row["WinRate"], -row["RR"])

            near_misses.sort(key=_miss_score)
            near_misses = near_misses[: max(1, int(near_miss_top))]

        # --- SECTOR CORRELATION LIMIT (Pro Trader Secret) ---
        # Never bet the farm on one sector
        if results:
            results = self._apply_sector_limits(results)
            
        return results, tracker.summary(), near_misses
    
    def _apply_sector_limits(self, setups, max_per_sector=MAX_SECTOR_EXPOSURE, max_total=MAX_POSITIONS):
        """Limit same-sector exposure and cap total positions."""
        sector_count = {}
        filtered = []
        
        # Sort by score first (take best setups)
        setups_sorted = sorted(setups, key=lambda x: x.score, reverse=True)
        
        for s in setups_sorted:
            if len(filtered) >= max_total:
                break  # Already have enough positions
            
            sector = s.sector or 'Unknown'
            if sector_count.get(sector, 0) >= max_per_sector:
                continue  # Skip, already have max in this sector
            
            sector_count[sector] = sector_count.get(sector, 0) + 1
            filtered.append(s)
        
        return filtered

def addToPortfolio(setups):
    """Interactive loop to add trades to portfolio."""
    import json
    from datetime import datetime
    import os
    
    # Auto-save logic
    while True:
        print("\n" + "="*50)
        print(" PORTFOLIO MANAGER")
        print(f" Type Ticker to ADD to {PORTFOLIO_FILE}")
        print(" Press ENTER to Finish/Exit")
        print("-" * 50)
        
        choice = input(" [ADD] Add Ticker > ").strip().upper()
        if not choice: break
        
        # Find the setup
        target_setup = next((s for s in setups if s.ticker == choice), None)
        
        if not target_setup:
            print(f" ERROR: '{choice}' not found in the Result List above.")
            continue
            
        # Load Portfolio
        port_file = PORTFOLIO_FILE
        if os.path.exists(port_file):
            try:
                with open(port_file, "r") as f:
                    port = json.load(f)
            except: port = {}
        else: port = {}
        
        if choice in port:
            print(f" WARN: {choice} is already in your portfolio!")
            if input("    Overwrite? (y/n) > ").lower() != 'y':
                continue
        
        # Create Entry
        entry = {
            "entry_date": datetime.now().strftime("%Y-%m-%d"),
            "entry_price": round(target_setup.trigger, 2),
            "shares": target_setup.qty,
            "stop_loss": round(target_setup.stop, 2),
            "target": round(target_setup.target, 2),
            "highest_price": round(target_setup.trigger, 2),
            "strategy": target_setup.strategy,
            "note": target_setup.note
        }
        
        port[choice] = entry
        
        # Save
        with open(port_file, "w") as f:
            json.dump(port, f, indent=4)
            
        print(f" ADDED {choice}: Entry ${entry['entry_price']} | Stop ${entry['stop_loss']} | Target ${entry['target']}")
        print(f"    (Size: {entry['shares']} shares)")


def main():
    # Default config file - no need for --config flag anymore
    DEFAULT_CONFIG = "titan_config.json"
    
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", default=DEFAULT_CONFIG)
    pre_args, remaining = pre_parser.parse_known_args()

    parser = argparse.ArgumentParser(description="Run Titan Trade scan.")
    parser.add_argument("--config", default=pre_args.config, help="Path to JSON config file.")
    parser.add_argument(
        "--safe-mode",
        "--no-safe-mode",
        dest="safe_mode",
        action=argparse.BooleanOptionalAction,
        default=SAFE_MODE_DEFAULT,
        help="Enable conservative safety-first defaults.",
    )
    parser.add_argument(
        "--use-walkforward",
        "--no-use-walkforward",
        dest="use_walkforward",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Filter setups using walk-forward test results (default: on).",
    )
    parser.add_argument(
        "--wf-file",
        default="backtest_titan_results.csv",
        help="Walk-forward results file (default: backtest_titan_results.csv).",
    )
    parser.add_argument(
        "--portfolio",
        "--no-portfolio",
        dest="portfolio",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable interactive portfolio prompt (default: off).",
    )
    parser.add_argument(
        "--auto-backtest",
        "--no-auto-backtest",
        dest="auto_backtest",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Auto-run backtest if walk-forward file is missing/stale (default: on).",
    )
    parser.add_argument("--auto-max-tickers", type=int, default=200, help="Auto-backtest max tickers.")
    parser.add_argument("--auto-min-trades", type=int, default=5, help="Auto-backtest minimum trades.")
    parser.add_argument("--auto-wf-folds", type=int, default=4, help="Auto-backtest walk-forward folds.")
    parser.add_argument("--auto-wf-test-ratio", type=float, default=0.2, help="Auto-backtest test ratio per fold.")
    parser.add_argument("--auto-cost-bps", type=float, default=10.0, help="Auto-backtest cost bps.")
    parser.add_argument("--auto-slippage-bps", type=float, default=5.0, help="Auto-backtest slippage bps.")
    parser.add_argument("--auto-oos", action=argparse.BooleanOptionalAction, default=True, help="Auto-backtest include OOS stats.")
    parser.add_argument("--auto-oos-train-ratio", type=float, default=0.75, help="Auto-backtest OOS train ratio.")
    parser.add_argument("--auto-oos-min-test-bars", type=int, default=252, help="Auto-backtest OOS minimum test bars.")
    parser.add_argument("--min-regime-score", type=float, default=REGIME_MIN_SCORE, help="Minimum regime stability score.")
    parser.add_argument("--earnings-blackout-days", type=int, default=EARNINGS_BLACKOUT_DAYS, help="Days before earnings to block trades.")
    parser.add_argument("--earnings-post-days", type=int, default=EARNINGS_POST_DAYS, help="Days after earnings to block trades.")
    parser.add_argument("--risk-per-trade", type=float, default=RISK_PER_TRADE, help="Max $ risk per trade.")
    parser.add_argument("--account-size", type=float, default=ACCOUNT_SIZE, help="Account size for position sizing.")
    parser.add_argument("--max-workers", type=int, default=DEFAULT_MAX_WORKERS, help="Max parallel workers for scanning.")
    parser.add_argument("--cache-ttl-hours", type=float, default=DEFAULT_OHLCV_TTL_HOURS, help="OHLCV cache TTL in hours.")
    parser.add_argument("--sp500-ttl-days", type=float, default=DEFAULT_SP500_TTL_DAYS, help="S&P tickers cache TTL in days.")
    parser.add_argument("--data-period", default=DEFAULT_DATA_PERIOD, help="yfinance period (e.g., 5y, 2y).")
    parser.add_argument("--data-interval", default=DEFAULT_DATA_INTERVAL, help="yfinance interval (e.g., 1d, 1h).")
    parser.add_argument("--min-winrate-breakout", type=float, default=DEFAULT_MIN_WIN_RATE_BREAKOUT, help="Min breakout win rate.")
    parser.add_argument("--min-winrate-dip", type=float, default=DEFAULT_MIN_WIN_RATE_DIP, help="Min dip win rate.")
    parser.add_argument("--min-pf-breakout", type=float, default=DEFAULT_MIN_PF_BREAKOUT, help="Min breakout profit factor.")
    parser.add_argument("--min-pf-dip", type=float, default=DEFAULT_MIN_PF_DIP, help="Min dip profit factor.")
    parser.add_argument("--min-trades-breakout", type=int, default=DEFAULT_MIN_TRADES_BREAKOUT, help="Min breakout trades.")
    parser.add_argument("--min-trades-dip", type=int, default=DEFAULT_MIN_TRADES_DIP, help="Min dip trades.")
    parser.add_argument("--min-expectancy-breakout", type=float, default=DEFAULT_MIN_EXPECTANCY_BREAKOUT, help="Min breakout expectancy.")
    parser.add_argument("--min-expectancy-dip", type=float, default=DEFAULT_MIN_EXPECTANCY_DIP, help="Min dip expectancy.")
    parser.add_argument("--min-rr-breakout", type=float, default=DEFAULT_MIN_RR_BREAKOUT, help="Min breakout risk/reward.")
    parser.add_argument("--min-rr-dip", type=float, default=DEFAULT_MIN_RR_DIP, help="Min dip risk/reward.")
    parser.add_argument("--wf-min-trades", type=int, default=WF_MIN_TRADES, help="WF min trades.")
    parser.add_argument("--wf-min-pf", type=float, default=WF_MIN_PF, help="WF min profit factor.")
    parser.add_argument("--wf-min-expectancy", type=float, default=WF_MIN_EXPECTANCY, help="WF min expectancy.")
    parser.add_argument("--wf-min-passrate", type=float, default=WF_MIN_PASSRATE, help="WF min pass rate.")
    parser.add_argument(
        "--require-oos",
        dest="require_oos",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_REQUIRE_OOS,
        help="Require out-of-sample results; reject if missing/weak.",
    )
    parser.add_argument("--oos-min-trades", type=int, default=DEFAULT_OOS_MIN_TRADES, help="OOS min trades.")
    parser.add_argument("--oos-min-winrate-breakout", type=float, default=DEFAULT_OOS_MIN_WINRATE_BREAKOUT, help="OOS min breakout win rate.")
    parser.add_argument("--oos-min-winrate-dip", type=float, default=DEFAULT_OOS_MIN_WINRATE_DIP, help="OOS min dip win rate.")
    parser.add_argument("--oos-min-pf-breakout", type=float, default=DEFAULT_OOS_MIN_PF_BREAKOUT, help="OOS min breakout PF.")
    parser.add_argument("--oos-min-pf-dip", type=float, default=DEFAULT_OOS_MIN_PF_DIP, help="OOS min dip PF.")
    parser.add_argument("--oos-min-expectancy-breakout", type=float, default=DEFAULT_OOS_MIN_EXPECTANCY_BREAKOUT, help="OOS min breakout expectancy.")
    parser.add_argument("--oos-min-expectancy-dip", type=float, default=DEFAULT_OOS_MIN_EXPECTANCY_DIP, help="OOS min dip expectancy.")
    parser.add_argument(
        "--require-walkforward",
        dest="require_walkforward",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Require walk-forward results; abort if missing.",
    )
    parser.add_argument("--confirm-days-breakout", type=int, default=DEFAULT_CONFIRM_DAYS_BREAKOUT, help="Breakout setup confirmation days.")
    parser.add_argument("--confirm-days-dip", type=int, default=DEFAULT_CONFIRM_DAYS_DIP, help="Dip setup confirmation days.")
    parser.add_argument(
        "--require-confirmed-setup",
        "--no-require-confirmed-setup",
        dest="require_confirmed_setup",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_REQUIRE_CONFIRMED_SETUP,
        help="Require setup to persist across confirmation days.",
    )
    parser.add_argument("--regime-factors", default="", help="JSON dict mapping regime->factor.")
    parser.add_argument(
        "--force-refresh-cache",
        dest="force_refresh_cache",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Force refresh of OHLCV cache (default: off).",
    )
    parser.add_argument("--tickers", default="", help="Comma-separated ticker list to scan.")
    parser.add_argument("--tickers-file", default="", help="Path to ticker list file (txt or json).")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Directory for reports/logs.")
    parser.add_argument("--log-level", default=DEFAULT_LOG_LEVEL, help="Log level (DEBUG, INFO, WARNING, ERROR).")
    parser.add_argument("--log-file", default=None, help="Explicit log file path (optional).")
    parser.add_argument("--no-log-file", action="store_true", help="Disable log file output.")
    parser.add_argument("--report-rows", type=int, default=15, help="Rows to show in console summary.")
    parser.add_argument("--save-json", action="store_true", help="Also write scan_results.json.")
    parser.add_argument(
        "--near-miss-report",
        "--no-near-miss-report",
        dest="near_miss_report",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_NEAR_MISS_REPORT,
        help="Write near-miss report.",
    )
    parser.add_argument("--near-miss-top", type=int, default=DEFAULT_NEAR_MISS_TOP, help="Top N near-miss rows to save.")
    parser.add_argument(
        "--no-history",
        dest="no_history",
        action="store_true",
        help="Disable timestamped output files (keeps only latest versions).",
    )

    config = _load_config(pre_args.config)
    if config:
        valid_keys = {a.dest for a in parser._actions}
        parser.set_defaults(**{k: v for k, v in config.items() if k in valid_keys})
    args = parser.parse_args(remaining)
    args = _apply_safe_mode(args)

    output_paths = _resolve_output_paths(args.output_dir, save_history=not getattr(args, 'no_history', False))
    log_file = None if args.no_log_file else (args.log_file or output_paths["log_file"])
    global LOGGER
    LOGGER = setup_logging(args.log_level, log_file=log_file)

    try:
        args_dump = dict(vars(args))
        args_dump["run_timestamp"] = datetime.now().isoformat()
        args_dump["output_dir"] = os.path.abspath(args.output_dir)
        with open(output_paths["config_json_latest"], "w", encoding="utf-8") as f:
            json.dump(args_dump, f, indent=2)
        with open(output_paths["config_json"], "w", encoding="utf-8") as f:
            json.dump(args_dump, f, indent=2)
    except Exception:
        LOGGER.warning("Failed to write run_config.json")

    print("\n$$ TITAN TRADE v6.0 (GUARDIAN EDITION) $$")
    print("----------------------------------------------")
    LOGGER.info("Run started")
    LOGGER.info("Output dir: %s", args.output_dir)
    if args.safe_mode:
        LOGGER.info("Safe mode: enabled (conservative filters applied)")

    tickers_override = _parse_tickers(args.tickers)
    tickers_override += _load_tickers_from_file(args.tickers_file)
    tickers_override = list(dict.fromkeys(tickers_override))
    if tickers_override:
        LOGGER.info("Custom tickers loaded: %d", len(tickers_override))

    wf_path = None
    if args.use_walkforward:
        if args.auto_backtest:
            if not os.path.exists(OHLCV_CACHE_FILE):
                print("WARNING: cache file not found; auto-backtest skipped. Run titan_trade.py once to build cache.")
            else:
                if _needs_backtest(args.wf_file):
                    ok = _run_backtest(args)
                    if not ok:
                        print("WARNING: auto-backtest failed; proceeding without WF filter.")
        if os.path.exists(args.wf_file):
            wf_path = args.wf_file
            print(f"Using walk-forward filter: {wf_path}")
            if os.path.exists(OHLCV_CACHE_FILE):
                wf_mtime = os.path.getmtime(wf_path)
                cache_mtime = os.path.getmtime(OHLCV_CACHE_FILE)
                if wf_mtime < cache_mtime:
                    print("WARNING: walk-forward file is older than cache data; consider re-running backtest_titan.py.")
        else:
            print(f"Walk-forward file not found ({args.wf_file}); proceeding without WF filter.")
    if args.require_walkforward and not wf_path:
        print("ERROR: Walk-forward results required but not found. Aborting scan.")
        LOGGER.error("Walk-forward required but missing.")
        return
    if args.require_oos and not wf_path:
        print("ERROR: OOS results required but walk-forward file not found. Aborting scan.")
        LOGGER.error("OOS required but missing WF file.")
        return

    brain = TitanBrain(
        wf_path=wf_path,
        min_regime_score=args.min_regime_score,
        earnings_blackout_days=args.earnings_blackout_days,
        earnings_post_days=args.earnings_post_days,
        risk_per_trade=args.risk_per_trade,
        account_size=args.account_size,
        tickers_override=tickers_override,
        cache_ttl_hours=args.cache_ttl_hours,
        sp500_ttl_days=args.sp500_ttl_days,
        data_period=args.data_period,
        data_interval=args.data_interval,
        force_refresh=args.force_refresh_cache,
        min_win_rate_breakout=args.min_winrate_breakout,
        min_win_rate_dip=args.min_winrate_dip,
        min_pf_breakout=args.min_pf_breakout,
        min_pf_dip=args.min_pf_dip,
        min_trades_breakout=args.min_trades_breakout,
        min_trades_dip=args.min_trades_dip,
        min_expectancy_breakout=args.min_expectancy_breakout,
        min_expectancy_dip=args.min_expectancy_dip,
        min_rr_breakout=args.min_rr_breakout,
        min_rr_dip=args.min_rr_dip,
        wf_min_trades=args.wf_min_trades,
        wf_min_pf=args.wf_min_pf,
        wf_min_expectancy=args.wf_min_expectancy,
        wf_min_passrate=args.wf_min_passrate,
        require_walkforward=args.require_walkforward,
        confirm_days_breakout=args.confirm_days_breakout,
        confirm_days_dip=args.confirm_days_dip,
        require_confirmed_setup=args.require_confirmed_setup,
        regime_factors=args.regime_factors,
        require_oos=args.require_oos,
        oos_min_trades=args.oos_min_trades,
        oos_min_winrate_breakout=args.oos_min_winrate_breakout,
        oos_min_winrate_dip=args.oos_min_winrate_dip,
        oos_min_pf_breakout=args.oos_min_pf_breakout,
        oos_min_pf_dip=args.oos_min_pf_dip,
        oos_min_expectancy_breakout=args.oos_min_expectancy_breakout,
        oos_min_expectancy_dip=args.oos_min_expectancy_dip,
    )
    try:
        setups, stats, near_misses = brain.scan(
            max_workers=args.max_workers,
            near_miss_report=args.near_miss_report,
            near_miss_top=args.near_miss_top,
        )
    except KeyboardInterrupt:
        print("\nScan Cancelled.")
        LOGGER.info("Run cancelled by user.")
        return

    if setups:
        setups.sort(key=lambda x: x.score, reverse=True)
        
        # UI CLEAR & HEADER
        print("\n" * 3)
        print("="*60)
        print(f"  TITAN GUARDIAN v6.0 - RESULT SUMMARY")
        print("="*60)
        print(f"    * 'BUY NOW' = Price has triggered (Active Breakout)")
        print(f"    * 'PENDING' = Place Buy Stop Order at Trigger Price")
        print("-" * 60)
        
        table = []
        report_rows = max(1, int(args.report_rows))
        for s in setups[:report_rows]: # Show Top N now
            # Determine Status
            dist = (s.trigger - s.price) / s.price
            if s.price >= s.trigger:
                status = "BUY NOW"
            elif dist < 0.01:
                status = "WARN: NEAR"
            else:
                status = "PENDING"
            wf_row = brain.wf_results.get(s.ticker, {}) if brain.wf_results else {}
            badge = _compute_badge(
                s.strategy,
                wf_row,
                brain.min_regime_score,
                wf_min_trades=brain.wf_min_trades,
                wf_min_pf=brain.wf_min_pf,
                wf_min_expectancy=brain.wf_min_expectancy,
                wf_min_passrate=brain.wf_min_passrate,
            )

            table.append([
                s.ticker, 
                s.strategy[:4], 
                f"${s.price:.2f}",
                f"${s.trigger:.2f}", 
                status,
                badge,
                f"{s.win_rate:.0f}%", 
                f"{s.profit_factor:.2f}",
                f"{s.kelly:.1f}%",
                f"${s.stop:.2f}",
                f"${s.target:.2f}",
                s.note
            ])
            
        print(tabulate(table, headers=["Ticker", "Type", "Price", "Trigger", "Status", "Badge", "Win%", "PF", "Kelly", "Stop", "Target", "Note"], tablefmt="grid"))
        
        print(
            "\nRecommendation:"
            f" Breakout Win% >= {brain.min_win_rate_breakout:.0f}% PF >= {brain.min_pf_breakout:.2f},"
            f" Dip Win% >= {brain.min_win_rate_dip:.0f}% PF >= {brain.min_pf_dip:.2f}"
        )
        
        # Interactive Portfolio Add
        # Save CSV results for easy review
        rows = []
        for s in setups:
            row = {
                "Ticker": s.ticker,
                "Strategy": s.strategy,
                "Price": round(s.price, 4),
                "Trigger": round(s.trigger, 4),
                "Stop": round(s.stop, 4),
                "Target": round(s.target, 4),
                "WinRate": round(s.win_rate, 2),
                "ProfitFactor": round(s.profit_factor, 2),
                "Kelly": round(s.kelly, 2),
                "Score": round(s.score, 2),
                "Sector": s.sector,
                "EarningsCall": s.earnings_call,
                "Note": s.note,
            }
            wf = brain.wf_results.get(s.ticker, {}) if brain.wf_results else {}
            row["Badge"] = _compute_badge(
                s.strategy,
                wf,
                brain.min_regime_score,
                wf_min_trades=brain.wf_min_trades,
                wf_min_pf=brain.wf_min_pf,
                wf_min_expectancy=brain.wf_min_expectancy,
                wf_min_passrate=brain.wf_min_passrate,
            )
            if wf:
                if s.strategy.startswith("BREAKOUT"):
                    row["WF_Test_Trades"] = wf.get("Breakout_WF_Test_Trades_total", wf.get("Breakout_WF_Test_Trades", 0))
                    row["WF_Test_PF"] = wf.get("Breakout_WF_Test_PF_trade_weighted", wf.get("Breakout_WF_Test_PF", 0))
                    row["WF_Test_Expectancy"] = wf.get("Breakout_WF_Test_Expectancy_trade_weighted", wf.get("Breakout_WF_Test_Expectancy", 0))
                    row["WF_PassRate"] = wf.get("Breakout_WF_PassRate", 0)
                else:
                    row["WF_Test_Trades"] = wf.get("Dip_WF_Test_Trades_total", wf.get("Dip_WF_Test_Trades", 0))
                    row["WF_Test_PF"] = wf.get("Dip_WF_Test_PF_trade_weighted", wf.get("Dip_WF_Test_PF", 0))
                    row["WF_Test_Expectancy"] = wf.get("Dip_WF_Test_Expectancy_trade_weighted", wf.get("Dip_WF_Test_Expectancy", 0))
                    row["WF_PassRate"] = wf.get("Dip_WF_PassRate", 0)
                metrics = _wf_metrics_for_strategy(wf, s.strategy)
                reg = metrics.get("regime_score", None)
                try:
                    reg = float(reg) if reg is not None else None
                except Exception:
                    reg = None
                if reg is not None:
                    row["RegimeScore"] = round(reg, 3)
            rows.append(row)

        if rows:
            df_out = pd.DataFrame(rows)
            df_out.to_csv(output_paths["scan_csv_latest"], index=False)
            df_out.to_csv(output_paths["scan_csv"], index=False)
            if args.save_json:
                with open(output_paths["scan_json_latest"], "w", encoding="utf-8") as f:
                    json.dump(rows, f, indent=2)
                with open(output_paths["scan_json"], "w", encoding="utf-8") as f:
                    json.dump(rows, f, indent=2)

        with open(output_paths["scan_txt_latest"], "w", encoding="utf-8") as f:
            f.write("TITAN TRADE SCAN RESULTS\n")
            f.write(f"Date: {datetime.now()}\n")
            f.write("="*60 + "\n")
            f.write(tabulate(table, headers=["Ticker", "Type", "Price", "Trigger", "Status", "Badge", "Win%", "PF", "Kelly", "Stop", "Target", "Note"], tablefmt="grid"))
            f.write("\n\nSCAN FILTER REPORT\n")
            f.write("-" * 40 + "\n")
            for k, v in stats.items():
                f.write(f"{k.ljust(20)}: {v}\n")
        with open(output_paths["scan_txt"], "w", encoding="utf-8") as f:
            f.write("TITAN TRADE SCAN RESULTS\n")
            f.write(f"Date: {datetime.now()}\n")
            f.write("="*60 + "\n")
            f.write(tabulate(table, headers=["Ticker", "Type", "Price", "Trigger", "Status", "Badge", "Win%", "PF", "Kelly", "Stop", "Target", "Note"], tablefmt="grid"))
            f.write("\n\nSCAN FILTER REPORT\n")
            f.write("-" * 40 + "\n")
            for k, v in stats.items():
                f.write(f"{k.ljust(20)}: {v}\n")

        print(f"\nReport saved to {output_paths['scan_txt_latest']} and {output_paths['scan_csv_latest']}")
        LOGGER.info("Reports written to %s", args.output_dir)
        if args.portfolio:
            addToPortfolio(setups)
        
    else:
        print("\nNo valid setups found that passed the Reality Check.")

    if args.near_miss_report and near_misses:
        try:
            nm_df = pd.DataFrame(near_misses)
            nm_df.to_csv(output_paths["near_miss_csv_latest"], index=False)
            nm_df.to_csv(output_paths["near_miss_csv"], index=False)
            with open(output_paths["near_miss_json_latest"], "w", encoding="utf-8") as f:
                json.dump(near_misses, f, indent=2)
            with open(output_paths["near_miss_json"], "w", encoding="utf-8") as f:
                json.dump(near_misses, f, indent=2)
            print(f"\nNear-miss report saved to {output_paths['near_miss_csv_latest']}")
        except Exception:
            LOGGER.warning("Failed to write near-miss report.")

        if not setups:
            try:
                watch_rows = []
                max_rows = max(1, int(args.report_rows))
                for row in near_misses[:max_rows]:
                    watch_rows.append(
                        [
                            row.get("Ticker"),
                            row.get("Strategy"),
                            row.get("Reason"),
                            f"{row.get('WinRate', 0):.1f}%",
                            f"{row.get('ProfitFactor', 0):.2f}",
                            int(row.get("Trades", 0)),
                            f"{row.get('RR', 0):.2f}",
                        ]
                    )
                if watch_rows:
                    print("\nNear-Miss Watchlist (NOT trade signals)")
                    print(tabulate(watch_rows, headers=["Ticker", "Strat", "Reason", "Win%", "PF", "Trades", "RR"], tablefmt="grid"))
            except Exception:
                LOGGER.warning("Failed to display near-miss watchlist.")

if __name__ == "__main__":
    main()
