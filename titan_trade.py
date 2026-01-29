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
import threading
from datetime import datetime, timedelta, date
from dataclasses import dataclass
from tabulate import tabulate
import io

# Optional imports for enhanced features
try:
    import pytz
    HAS_PYTZ = True
except ImportError:
    HAS_PYTZ = False
    
try:
    from plyer import notification
    HAS_PLYER = True
except ImportError:
    HAS_PLYER = False

try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

# Disable warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
pd.options.mode.chained_assignment = None

# --- CONSTANTS & CONFIG ---
CACHE_DIR = "cache_sp500_elite"
SP500_CACHE_FILE = os.path.join(CACHE_DIR, "sp500_tickers.json")
OHLCV_CACHE_FILE = os.path.join(CACHE_DIR, "sp500_ohlcv_bulk.parquet")
PORTFOLIO_FILE = "portfolio.json"
TRADE_LOG_FILE = "trade_log.json"
RISK_LOG_FILE = "risk_log.json"
PRE_TRADE_CHECKLIST_FILE = "pre_trade_checklist.json"
SIGNAL_TRACKER_FILE = "signal_tracker.json"
PAPER_TRADE_FILE = "paper_trades.json"
DEFAULT_SP500_TTL_DAYS = 7

# =============================================================================
# DATA FRESHNESS - Critical for real trading
# =============================================================================
# For REAL trading, data must be fresh. yfinance has ~15min delay.
# Set this to 0.5 (30 minutes) for real trading, or use a real-time data source
DEFAULT_OHLCV_TTL_HOURS = 0.5  # 30 minutes for production (was 12 hours)
MAX_DATA_AGE_MINUTES = 30  # Warn if data older than this

# =============================================================================
# LIQUIDITY REQUIREMENTS - Don't trade illiquid stocks
# =============================================================================
MIN_AVG_DOLLAR_VOLUME = 10_000_000  # $10M daily volume minimum
MIN_AVG_VOLUME = 500_000  # 500K shares minimum
MAX_POSITION_PCT_OF_VOLUME = 1.0  # Never be more than 1% of daily volume

# =============================================================================
# REALISTIC SLIPPAGE MODEL - Based on order size vs volume
# =============================================================================
# Slippage increases as your order becomes larger relative to volume
# Base slippage + (position_size / avg_volume) * volume_impact_factor
BASE_SLIPPAGE_BREAKOUT_BPS = 20.0  # Base 0.20%
BASE_SLIPPAGE_DIP_BPS = 10.0  # Base 0.10%
VOLUME_IMPACT_FACTOR = 50.0  # Additional bps per 1% of volume
DEFAULT_DATA_PERIOD = "5y"
DEFAULT_DATA_INTERVAL = "1d"
DEFAULT_OUTPUT_DIR = "."
DEFAULT_LOG_LEVEL = "INFO"
VALID_LOG_LEVELS = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
DEFAULT_MAX_WORKERS = max(2, min(12, (os.cpu_count() or 4)))

DEFAULT_NEAR_MISS_REPORT = True
DEFAULT_NEAR_MISS_TOP = 50

DEFAULT_CONFIRM_DAYS_BREAKOUT = 3
DEFAULT_CONFIRM_DAYS_DIP = 3
DEFAULT_REQUIRE_CONFIRMED_SETUP = True
DEFAULT_REGIME_FACTORS = {
    "BULL": 1.0,
    "RECOVERY": 1.1,
    "NEUTRAL": 1.2,
    "Correction": 1.4,
    "BEAR": 2.0,  # Double requirements in bear market
}

DEFAULT_REQUIRE_OOS = True  # Always require OOS validation
DEFAULT_OOS_MIN_TRADES = 10
DEFAULT_OOS_MIN_WINRATE_BREAKOUT = 55.0
DEFAULT_OOS_MIN_WINRATE_DIP = 52.0
DEFAULT_OOS_MIN_PF_BREAKOUT = 1.3
DEFAULT_OOS_MIN_PF_DIP = 1.2
DEFAULT_OOS_MIN_EXPECTANCY_BREAKOUT = 0.002
DEFAULT_OOS_MIN_EXPECTANCY_DIP = 0.001

# =============================================================================
# SAFE MODE DEFAULTS - Balanced between safety and practicality
# =============================================================================
SAFE_MODE_DEFAULT = False  # Start with balanced mode, not overly strict
SAFE_MODE_SETTINGS = {
    "require_walkforward": True,
    "require_oos": False,  # OOS is nice but not required
    "confirm_days_breakout": 2,
    "confirm_days_dip": 2,
    "require_confirmed_setup": True,
    # REALISTIC: 10-15 trades is acceptable for individual stock analysis
    # Portfolio diversification provides additional statistical confidence
    "min_trades_breakout": 12,
    "min_trades_dip": 12,
    # Win rate requirements (realistic)
    "min_winrate_breakout": 50.0,
    "min_winrate_dip": 48.0,
    # Profit factor (must be profitable after costs)
    "min_pf_breakout": 1.3,
    "min_pf_dip": 1.2,
    # Expectancy per trade
    "min_expectancy_breakout": 0.002,
    "min_expectancy_dip": 0.001,
    # Risk/reward
    "min_rr_breakout": 1.5,
    "min_rr_dip": 1.3,
    # Walk-forward validation
    "wf_min_trades": 8,
    "wf_min_pf": 1.1,
    "wf_min_expectancy": 0.0,
    "wf_min_passrate": 0.25,  # Must pass 25% of folds (at least 1 of 4)
    # OOS validation (when enabled)
    "oos_min_trades": 5,
    "oos_min_winrate_breakout": 48.0,
    "oos_min_winrate_dip": 45.0,
    "oos_min_pf_breakout": 1.1,
    "oos_min_pf_dip": 1.0,
    "oos_min_expectancy_breakout": 0.0,
    "oos_min_expectancy_dip": 0.0,
}

# =============================================================================
# TRUST MODE - FOR USERS WHO WANT TO RUN AND TRUST THE RESULTS
# =============================================================================
# This mode is EXTREMELY strict. Only the highest confidence setups pass.
# If it says TRADE, you can trust it. If it says DON'T TRADE, respect it.
#
# QUICK START GUIDE FOR TRUST MODE:
# =================================
# STEP 1: Start paper trading validation (required first!)
#         python titan_trade.py --trust-paper
#
# STEP 2: Each time you complete a paper trade, record the result:
#         python titan_trade.py --trust-paper-win   (if you would have profited)
#         python titan_trade.py --trust-paper-loss  (if you would have lost)
#
# STEP 3: After 30 days and 10+ successful paper trades, you're validated!
#         python titan_trade.py --trust-mode
#
# OTHER COMMANDS:
#         python titan_trade.py --trust-status  (check your progress)
#         python titan_trade.py --trust-reset   (start over)
#         python titan_trade.py --trust-bypass  (skip validation - NOT recommended)
#
TRUST_MODE_DEFAULT = False
TRUST_MODE_SETTINGS = {
    # MANDATORY: Walk-forward + Out-of-sample validation
    "require_walkforward": True,
    "require_oos": True,
    # Setup must persist for 3 days (not a flash in the pan)
    "confirm_days_breakout": 3,
    "confirm_days_dip": 3,
    "require_confirmed_setup": True,
    # STRICT: Need 30+ trades for statistical significance (t-test requirement)
    "min_trades_breakout": 30,
    "min_trades_dip": 30,
    # Win rate requirements (higher bar)
    "min_winrate_breakout": 55.0,
    "min_winrate_dip": 52.0,
    # Profit factor (must be clearly profitable)
    "min_pf_breakout": 1.5,
    "min_pf_dip": 1.4,
    # Expectancy per trade (meaningful edge)
    "min_expectancy_breakout": 0.005,  # 0.5% per trade minimum
    "min_expectancy_dip": 0.003,
    # Risk/reward (at least 2:1)
    "min_rr_breakout": 2.0,
    "min_rr_dip": 1.8,
    # Walk-forward validation (stricter)
    "wf_min_trades": 15,
    "wf_min_pf": 1.3,
    "wf_min_expectancy": 0.002,
    "wf_min_passrate": 0.50,  # Must pass 50% of folds (2 of 4)
    # OOS validation (mandatory in trust mode)
    "oos_min_trades": 10,
    "oos_min_winrate_breakout": 52.0,
    "oos_min_winrate_dip": 50.0,
    "oos_min_pf_breakout": 1.3,
    "oos_min_pf_dip": 1.2,
    "oos_min_expectancy_breakout": 0.002,
    "oos_min_expectancy_dip": 0.001,
}

# =============================================================================
# AUTO MODE - JUST RUN AND GO (No flags needed!)
# =============================================================================
# When True, running `python titan_trade.py` with no arguments activates Trust Mode
AUTO_MODE_ENABLED = True
AUTO_MODE_CONFIG_FILE = "titan_auto_config.json"

# =============================================================================
# TRUST MODE - ADDITIONAL SAFETY PARAMETERS
# =============================================================================
# Only show Grade A or B signals (C/D/F are hidden)
TRUST_MODE_MIN_GRADE = "B"
# Require statistical significance (t-stat >= 2.0)
TRUST_MODE_REQUIRE_SIGNIFICANCE = True
# Maximum trades per day (prevent overtrading)
TRUST_MODE_MAX_TRADES_PER_DAY = 2
# Maximum trades per week
TRUST_MODE_MAX_TRADES_PER_WEEK = 5
# Cooling off after consecutive losses
TRUST_MODE_LOSS_STREAK_COOLOFF = 3  # Pause after 3 consecutive losses
TRUST_MODE_COOLOFF_DAYS = 2  # Pause trading for 2 days
# Mandatory paper trading validation period (days)
TRUST_MODE_PAPER_VALIDATION_DAYS = 30
TRUST_MODE_PAPER_MIN_TRADES = 10
TRUST_MODE_PAPER_MIN_WINRATE = 45.0
# Position size reduction
TRUST_MODE_MAX_RISK_PER_TRADE_PCT = 0.3  # Only 0.3% risk per trade (ultra safe)
TRUST_MODE_MAX_POSITIONS = 4  # Max 4 positions at once
# Data freshness (stricter)
TRUST_MODE_MAX_DATA_AGE_HOURS = 1  # Data must be less than 1 hour old during market
# Disable trading on high volatility days
TRUST_MODE_VIX_CAUTION = 18  # Reduce size when VIX > 18
TRUST_MODE_VIX_HALT = 25  # Stop trading when VIX > 25

# =============================================================================
# POSITION SIZING & RISK MANAGEMENT
# =============================================================================
RISK_PER_TRADE = 500.0  # Conservative: $500 max risk per trade
ACCOUNT_SIZE = 100000.0
MAX_RISK_PCT_PER_TRADE = 0.5  # Never risk more than 0.5% per trade

# Portfolio-level risk controls (CRITICAL for survival)
MAX_POSITIONS = 6  # Maximum concurrent positions
MAX_SECTOR_EXPOSURE = 3  # Max stocks per sector (increased for tracking)
MAX_DAILY_LOSS_PCT = 2.0  # Stop trading if down 2% in a day
MAX_WEEKLY_LOSS_PCT = 5.0  # Stop trading if down 5% in a week
MAX_DRAWDOWN_PCT = 15.0  # Pause all trading if drawdown exceeds 15%
PORTFOLIO_HEAT_MAX = 6.0  # Max total portfolio risk (% of account)

# VIX-based position sizing
VIX_HIGH_THRESHOLD = 22  # Reduce size when VIX > this
VIX_EXTREME_THRESHOLD = 28  # Cut size 50% when VIX > this
VIX_PANIC_THRESHOLD = 35  # No new positions when VIX > this

# Gap and execution protection
GAP_PROTECTION = True
MAX_GAP_PCT = 0.04  # 4% max gap history allowed (stricter)

# =============================================================================
# REALISTIC SLIPPAGE MODEL
# =============================================================================
# Breakouts have high slippage due to momentum chasers
DEFAULT_SLIPPAGE_BREAKOUT_BPS = 30.0  # 0.30% slippage for breakouts
DEFAULT_SLIPPAGE_DIP_BPS = 10.0  # 0.10% slippage for dips (less crowded)
DEFAULT_COMMISSION_BPS = 5.0  # $5 per $10k traded

# Quality Filters (balanced defaults)
DEFAULT_MIN_WIN_RATE_BREAKOUT = 50.0
DEFAULT_MIN_WIN_RATE_DIP = 48.0
DEFAULT_MIN_PF_BREAKOUT = 1.3
DEFAULT_MIN_PF_DIP = 1.2
DEFAULT_MIN_TRADES_BREAKOUT = 10  # Realistic for individual stocks
DEFAULT_MIN_TRADES_DIP = 10  # Realistic for individual stocks
DEFAULT_MIN_EXPECTANCY_BREAKOUT = 0.002
DEFAULT_MIN_EXPECTANCY_DIP = 0.001
DEFAULT_MIN_RR_BREAKOUT = 1.5
DEFAULT_MIN_RR_DIP = 1.3

# Walk-forward filters (backtest_titan_results.csv)
WF_MIN_TRADES = 8
WF_MIN_PF = 1.1
WF_MIN_EXPECTANCY = 0.0
WF_MIN_PASSRATE = 0.25  # Pass at least 1 of 4 folds
REGIME_MIN_SCORE = 0.5  # Must work in half of regimes
EARNINGS_BLACKOUT_DAYS = 7
EARNINGS_POST_DAYS = 1

# =============================================================================
# MARKET HOURS & AUTO-REFRESH
# =============================================================================
MARKET_OPEN_HOUR = 9
MARKET_OPEN_MINUTE = 30
MARKET_CLOSE_HOUR = 16
MARKET_CLOSE_MINUTE = 0
MARKET_TIMEZONE = "America/New_York"
AUTO_REFRESH_DURING_MARKET_HOURS = True  # Smart refresh when market is open

# =============================================================================
# NOTIFICATIONS
# =============================================================================
NOTIFICATIONS_ENABLED = True
NOTIFICATION_ON_NEW_SIGNAL = True
NOTIFICATION_ON_STOP_HIT = True
NOTIFICATION_ON_TARGET_HIT = True

# =============================================================================
# SECTOR MAPPING (GICS Sectors)
# =============================================================================
SECTOR_CACHE_FILE = os.path.join(CACHE_DIR, "sector_cache.json")
SECTOR_CACHE_TTL_DAYS = 30  # Refresh sector info monthly

# =============================================================================
# SCHEDULED SCANNING
# =============================================================================
SCHEDULE_MARKET_OPEN_DELAY_MINUTES = 5  # Run 5 min after market open
SCHEDULE_MARKET_CLOSE_BEFORE_MINUTES = 5  # Run 5 min before market close

# =============================================================================
# PERFORMANCE DASHBOARD
# =============================================================================
DASHBOARD_FILE = "performance_dashboard.png"
DASHBOARD_HISTORY_DAYS = 90  # Show last 90 days of performance


# =============================================================================
# MARKET HOURS UTILITIES
# =============================================================================
class MarketHours:
    """Utilities for checking market hours and smart refresh."""
    
    @staticmethod
    def get_eastern_time():
        """Get current time in Eastern timezone."""
        if HAS_PYTZ:
            eastern = pytz.timezone(MARKET_TIMEZONE)
            return datetime.now(eastern)
        else:
            # Fallback: assume running in US Eastern or close to it
            # This is a rough approximation
            return datetime.now()
    
    @staticmethod
    def is_market_open():
        """Check if US stock market is currently open."""
        now = MarketHours.get_eastern_time()
        
        # Check if weekend
        if now.weekday() >= 5:  # Saturday=5, Sunday=6
            return False
        
        # Check market hours (9:30 AM - 4:00 PM ET)
        market_open = now.replace(hour=MARKET_OPEN_HOUR, minute=MARKET_OPEN_MINUTE, second=0, microsecond=0)
        market_close = now.replace(hour=MARKET_CLOSE_HOUR, minute=MARKET_CLOSE_MINUTE, second=0, microsecond=0)
        
        return market_open <= now <= market_close
    
    @staticmethod
    def is_pre_market():
        """Check if in pre-market hours (4:00 AM - 9:30 AM ET)."""
        now = MarketHours.get_eastern_time()
        if now.weekday() >= 5:
            return False
        pre_market_start = now.replace(hour=4, minute=0, second=0, microsecond=0)
        market_open = now.replace(hour=MARKET_OPEN_HOUR, minute=MARKET_OPEN_MINUTE, second=0, microsecond=0)
        return pre_market_start <= now < market_open
    
    @staticmethod
    def is_after_hours():
        """Check if in after-hours (4:00 PM - 8:00 PM ET)."""
        now = MarketHours.get_eastern_time()
        if now.weekday() >= 5:
            return False
        market_close = now.replace(hour=MARKET_CLOSE_HOUR, minute=MARKET_CLOSE_MINUTE, second=0, microsecond=0)
        after_hours_end = now.replace(hour=20, minute=0, second=0, microsecond=0)
        return market_close < now <= after_hours_end
    
    @staticmethod
    def should_auto_refresh(cache_file, ttl_hours=0.5):
        """
        Determine if data should be auto-refreshed.
        During market hours, refresh if data is older than ttl.
        Outside market hours, use cached data.
        """
        if not AUTO_REFRESH_DURING_MARKET_HOURS:
            return False
        
        if not os.path.exists(cache_file):
            return True
        
        # During market hours, be more aggressive about refreshing
        if MarketHours.is_market_open():
            cache_age_seconds = time.time() - os.path.getmtime(cache_file)
            cache_age_hours = cache_age_seconds / 3600
            return cache_age_hours >= ttl_hours
        
        return False
    
    @staticmethod
    def get_market_status_string():
        """Get human-readable market status."""
        if MarketHours.is_market_open():
            return "OPEN"
        elif MarketHours.is_pre_market():
            return "PRE-MARKET"
        elif MarketHours.is_after_hours():
            return "AFTER-HOURS"
        else:
            return "CLOSED"
    
    @staticmethod
    def time_until_market_open():
        """Get timedelta until next market open."""
        now = MarketHours.get_eastern_time()
        
        # Find next market open
        next_open = now.replace(hour=MARKET_OPEN_HOUR, minute=MARKET_OPEN_MINUTE, second=0, microsecond=0)
        
        if now >= next_open:
            # Already past open time today, try tomorrow
            next_open += timedelta(days=1)
        
        # Skip weekends
        while next_open.weekday() >= 5:
            next_open += timedelta(days=1)
        
        return next_open - now


# =============================================================================
# NOTIFICATION SYSTEM
# =============================================================================
class NotificationManager:
    """Cross-platform desktop notifications."""
    
    @staticmethod
    def send(title, message, timeout=10):
        """Send a desktop notification."""
        if not NOTIFICATIONS_ENABLED:
            return False
        
        try:
            if HAS_PLYER:
                notification.notify(
                    title=title,
                    message=message,
                    app_name="Titan Trade",
                    timeout=timeout
                )
                return True
            else:
                # Fallback: Windows toast notification via PowerShell
                if sys.platform == 'win32':
                    try:
                        ps_script = f'''
                        [Windows.UI.Notifications.ToastNotificationManager, Windows.UI.Notifications, ContentType = WindowsRuntime] | Out-Null
                        $template = [Windows.UI.Notifications.ToastNotificationManager]::GetTemplateContent([Windows.UI.Notifications.ToastTemplateType]::ToastText02)
                        $textNodes = $template.GetElementsByTagName("text")
                        $textNodes.Item(0).AppendChild($template.CreateTextNode("{title}")) | Out-Null
                        $textNodes.Item(1).AppendChild($template.CreateTextNode("{message}")) | Out-Null
                        $toast = [Windows.UI.Notifications.ToastNotification]::new($template)
                        [Windows.UI.Notifications.ToastNotificationManager]::CreateToastNotifier("Titan Trade").Show($toast)
                        '''
                        subprocess.run(['powershell', '-Command', ps_script], 
                                      capture_output=True, timeout=5)
                        return True
                    except Exception:
                        pass
                
                # Fallback: just print to console
                print(f"\n🔔 {title}: {message}\n")
                return True
        except Exception as e:
            print(f"Notification failed: {e}")
            return False
    
    @staticmethod
    def notify_new_signal(ticker, strategy, win_rate, profit_factor):
        """Notify about a new trading signal."""
        if NOTIFICATION_ON_NEW_SIGNAL:
            NotificationManager.send(
                f"New Signal: {ticker}",
                f"{strategy} | WR: {win_rate:.0f}% | PF: {profit_factor:.2f}"
            )
    
    @staticmethod
    def notify_stop_hit(ticker, entry_price, exit_price, pnl_pct):
        """Notify when a stop is hit."""
        if NOTIFICATION_ON_STOP_HIT:
            NotificationManager.send(
                f"STOP HIT: {ticker}",
                f"Entry: ${entry_price:.2f} → Exit: ${exit_price:.2f} ({pnl_pct:+.1f}%)"
            )
    
    @staticmethod
    def notify_target_hit(ticker, entry_price, exit_price, pnl_pct):
        """Notify when target is hit."""
        if NOTIFICATION_ON_TARGET_HIT:
            NotificationManager.send(
                f"TARGET HIT: {ticker}",
                f"Entry: ${entry_price:.2f} → Exit: ${exit_price:.2f} ({pnl_pct:+.1f}%)"
            )


# =============================================================================
# SECTOR DETECTION
# =============================================================================
class SectorMapper:
    """Get and cache sector information for tickers."""
    
    _cache = None
    _cache_loaded = False
    
    @classmethod
    def _load_cache(cls):
        """Load sector cache from file."""
        if cls._cache_loaded:
            return
        
        cls._cache = {}
        if os.path.exists(SECTOR_CACHE_FILE):
            try:
                cache_age_days = (time.time() - os.path.getmtime(SECTOR_CACHE_FILE)) / 86400
                if cache_age_days < SECTOR_CACHE_TTL_DAYS:
                    with open(SECTOR_CACHE_FILE, 'r') as f:
                        cls._cache = json.load(f)
            except Exception:
                pass
        cls._cache_loaded = True
    
    @classmethod
    def _save_cache(cls):
        """Save sector cache to file."""
        try:
            os.makedirs(os.path.dirname(SECTOR_CACHE_FILE), exist_ok=True)
            with open(SECTOR_CACHE_FILE, 'w') as f:
                json.dump(cls._cache, f)
        except Exception:
            pass
    
    @classmethod
    def get_sector(cls, ticker):
        """Get sector for a single ticker."""
        cls._load_cache()
        
        if ticker in cls._cache:
            return cls._cache[ticker]
        
        # Fetch from yfinance
        try:
            info = yf.Ticker(ticker).info
            sector = info.get('sector', 'Unknown')
            if sector:
                cls._cache[ticker] = sector
                cls._save_cache()
                return sector
        except Exception:
            pass
        
        return 'Unknown'
    
    @classmethod
    def get_sectors_batch(cls, tickers, max_fetch=50):
        """Get sectors for multiple tickers (with rate limiting)."""
        cls._load_cache()
        
        result = {}
        to_fetch = []
        
        # Check cache first
        for ticker in tickers:
            if ticker in cls._cache:
                result[ticker] = cls._cache[ticker]
            else:
                to_fetch.append(ticker)
        
        # Fetch missing (limit to avoid rate limits)
        for ticker in to_fetch[:max_fetch]:
            try:
                info = yf.Ticker(ticker).info
                sector = info.get('sector', 'Unknown')
                cls._cache[ticker] = sector
                result[ticker] = sector
                time.sleep(0.1)  # Rate limit
            except Exception:
                result[ticker] = 'Unknown'
        
        cls._save_cache()
        return result


# =============================================================================
# EARNINGS CALENDAR
# =============================================================================
class EarningsCalendar:
    """Check earnings dates for stocks."""
    
    _cache = {}
    
    @classmethod
    def get_earnings_date(cls, ticker):
        """
        Get next earnings date for a ticker.
        Returns: (earnings_date, days_until) or (None, None) if not found
        """
        if ticker in cls._cache:
            cached = cls._cache[ticker]
            # Check if cache is still valid (less than 1 day old)
            if cached.get('fetched') and (datetime.now() - cached['fetched']).days < 1:
                return cached.get('date'), cached.get('days_until')
        
        try:
            stock = yf.Ticker(ticker)
            
            # Try to get earnings dates
            try:
                calendar = stock.calendar
                if calendar is not None and not calendar.empty:
                    if 'Earnings Date' in calendar.index:
                        earnings_date = calendar.loc['Earnings Date']
                        if isinstance(earnings_date, pd.Series):
                            earnings_date = earnings_date.iloc[0]
                        if pd.notna(earnings_date):
                            if isinstance(earnings_date, str):
                                earnings_date = pd.to_datetime(earnings_date)
                            days_until = (earnings_date.date() - date.today()).days
                            cls._cache[ticker] = {
                                'date': earnings_date,
                                'days_until': days_until,
                                'fetched': datetime.now()
                            }
                            return earnings_date, days_until
            except Exception:
                pass
            
            # Fallback: try earnings_dates
            try:
                earnings_dates = stock.earnings_dates
                if earnings_dates is not None and not earnings_dates.empty:
                    future_dates = earnings_dates[earnings_dates.index >= pd.Timestamp.now()]
                    if not future_dates.empty:
                        next_date = future_dates.index[0]
                        days_until = (next_date.date() - date.today()).days
                        cls._cache[ticker] = {
                            'date': next_date,
                            'days_until': days_until,
                            'fetched': datetime.now()
                        }
                        return next_date, days_until
            except Exception:
                pass
                
        except Exception:
            pass
        
        cls._cache[ticker] = {'date': None, 'days_until': None, 'fetched': datetime.now()}
        return None, None
    
    @classmethod
    def is_in_blackout(cls, ticker, blackout_days=EARNINGS_BLACKOUT_DAYS, post_days=EARNINGS_POST_DAYS):
        """
        Check if ticker is in earnings blackout period.
        Returns: (is_blackout, reason_string)
        """
        earnings_date, days_until = cls.get_earnings_date(ticker)
        
        if earnings_date is None:
            return False, "Earnings date unknown"
        
        if days_until is not None:
            if 0 <= days_until <= blackout_days:
                return True, f"Earnings in {days_until} days"
            elif -post_days <= days_until < 0:
                return True, f"Earnings {abs(days_until)} days ago"
        
        return False, f"Earnings in {days_until} days" if days_until else "OK"


# =============================================================================
# PERFORMANCE DASHBOARD
# =============================================================================
class PerformanceDashboard:
    """Generate performance charts and statistics."""
    
    @staticmethod
    def generate(tracker_file=SIGNAL_TRACKER_FILE, output_file=DASHBOARD_FILE):
        """Generate a performance dashboard image."""
        if not HAS_MATPLOTLIB:
            print("  Dashboard requires matplotlib. Install with: pip install matplotlib")
            return False
        
        # Load tracker data
        if not os.path.exists(tracker_file):
            print("  No tracking data found yet.")
            return False
        
        try:
            with open(tracker_file, 'r') as f:
                data = json.load(f)
        except Exception as e:
            print(f"  Failed to load tracker data: {e}")
            return False
        
        completed = data.get('completed_signals', [])
        active = data.get('active_signals', {})
        stats = data.get('stats', {})
        
        if not completed and not active:
            print("  No signals to display yet.")
            return False
        
        # Create dashboard
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Titan Trade Performance Dashboard', fontsize=16, fontweight='bold')
        
        # 1. Cumulative P&L Chart
        ax1 = axes[0, 0]
        if completed:
            dates = []
            cum_pnl = []
            running_pnl = 0
            for sig in completed:
                if sig.get('exit_date') and sig.get('pnl_pct') is not None:
                    try:
                        exit_date = datetime.fromisoformat(sig['exit_date'].replace('Z', '+00:00'))
                        dates.append(exit_date)
                        running_pnl += sig['pnl_pct']
                        cum_pnl.append(running_pnl)
                    except Exception:
                        pass
            
            if dates:
                ax1.plot(dates, cum_pnl, 'b-', linewidth=2, marker='o', markersize=4)
                ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
                ax1.fill_between(dates, cum_pnl, 0, 
                               where=[p >= 0 for p in cum_pnl], 
                               color='green', alpha=0.3)
                ax1.fill_between(dates, cum_pnl, 0, 
                               where=[p < 0 for p in cum_pnl], 
                               color='red', alpha=0.3)
                ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
                ax1.set_title('Cumulative P&L (%)')
                ax1.set_ylabel('P&L %')
                ax1.grid(True, alpha=0.3)
            else:
                ax1.text(0.5, 0.5, 'No completed trades', ha='center', va='center', transform=ax1.transAxes)
                ax1.set_title('Cumulative P&L (%)')
        else:
            ax1.text(0.5, 0.5, 'No completed trades', ha='center', va='center', transform=ax1.transAxes)
            ax1.set_title('Cumulative P&L (%)')
        
        # 2. Win/Loss Distribution
        ax2 = axes[0, 1]
        wins = stats.get('wins', 0)
        losses = stats.get('losses', 0)
        if wins + losses > 0:
            colors = ['#2ecc71', '#e74c3c']
            ax2.pie([wins, losses], labels=[f'Wins ({wins})', f'Losses ({losses})'], 
                   colors=colors, autopct='%1.1f%%', startangle=90)
            ax2.set_title(f'Win Rate: {wins/(wins+losses)*100:.1f}%')
        else:
            ax2.text(0.5, 0.5, 'No completed trades', ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title('Win/Loss Distribution')
        
        # 3. P&L Distribution Histogram
        ax3 = axes[1, 0]
        if completed:
            pnls = [s.get('pnl_pct', 0) for s in completed if s.get('pnl_pct') is not None]
            if pnls:
                colors = ['green' if p >= 0 else 'red' for p in pnls]
                ax3.hist(pnls, bins=20, color='steelblue', edgecolor='black', alpha=0.7)
                ax3.axvline(x=0, color='black', linestyle='--', linewidth=2)
                ax3.axvline(x=np.mean(pnls), color='orange', linestyle='-', linewidth=2, label=f'Avg: {np.mean(pnls):.2f}%')
                ax3.set_title('P&L Distribution')
                ax3.set_xlabel('P&L %')
                ax3.set_ylabel('Frequency')
                ax3.legend()
                ax3.grid(True, alpha=0.3)
            else:
                ax3.text(0.5, 0.5, 'No P&L data', ha='center', va='center', transform=ax3.transAxes)
                ax3.set_title('P&L Distribution')
        else:
            ax3.text(0.5, 0.5, 'No completed trades', ha='center', va='center', transform=ax3.transAxes)
            ax3.set_title('P&L Distribution')
        
        # 4. Summary Statistics
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        total_signals = stats.get('total_signals', 0)
        total_pnl = stats.get('total_pnl_pct', 0)
        win_rate = (wins / total_signals * 100) if total_signals > 0 else 0
        avg_pnl = total_pnl / total_signals if total_signals > 0 else 0
        
        summary_text = f"""
        PERFORMANCE SUMMARY
        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        
        Total Trades:     {total_signals}
        Active Signals:   {len(active)}
        
        Wins:             {wins}
        Losses:           {losses}
        Win Rate:         {win_rate:.1f}%
        
        Total P&L:        {total_pnl:+.2f}%
        Average P&L:      {avg_pnl:+.2f}%
        
        Started:          {stats.get('started', 'N/A')[:10] if stats.get('started') else 'N/A'}
        """
        
        ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes, fontsize=12,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        
        # Save dashboard
        try:
            plt.savefig(output_file, dpi=150, bbox_inches='tight')
            plt.close(fig)
            print(f"  Dashboard saved to: {output_file}")
            return True
        except Exception as e:
            print(f"  Failed to save dashboard: {e}")
            plt.close(fig)
            return False
    
    @staticmethod
    def print_stats(tracker_file=SIGNAL_TRACKER_FILE):
        """Print performance statistics to console."""
        if not os.path.exists(tracker_file):
            print("  No tracking data found.")
            return
        
        try:
            with open(tracker_file, 'r') as f:
                data = json.load(f)
        except Exception:
            print("  Failed to load tracker data.")
            return
        
        stats = data.get('stats', {})
        active = data.get('active_signals', {})
        completed = data.get('completed_signals', [])
        
        total = stats.get('total_signals', 0)
        wins = stats.get('wins', 0)
        losses = stats.get('losses', 0)
        total_pnl = stats.get('total_pnl_pct', 0)
        
        print("\n" + "="*60)
        print("  PERFORMANCE STATISTICS")
        print("="*60)
        print(f"  Started:        {stats.get('started', 'N/A')[:10] if stats.get('started') else 'N/A'}")
        print(f"  Active Signals: {len(active)}")
        print(f"  Completed:      {total}")
        print("-"*60)
        
        if total > 0:
            win_rate = wins / total * 100
            avg_pnl = total_pnl / total
            print(f"  Win Rate:       {win_rate:.1f}% ({wins}W / {losses}L)")
            print(f"  Total P&L:      {total_pnl:+.2f}%")
            print(f"  Average P&L:    {avg_pnl:+.2f}%")
            
            # Calculate profit factor from completed trades
            if completed:
                gross_profit = sum(s.get('pnl_pct', 0) for s in completed if s.get('pnl_pct', 0) > 0)
                gross_loss = abs(sum(s.get('pnl_pct', 0) for s in completed if s.get('pnl_pct', 0) < 0))
                pf = gross_profit / gross_loss if gross_loss > 0 else float('inf')
                print(f"  Profit Factor:  {pf:.2f}")
        else:
            print("  No completed trades yet.")
        
        print("="*60)


# =============================================================================
# SCHEDULED SCANNING
# =============================================================================
class ScheduledScanner:
    """Run scans at scheduled times."""
    
    _stop_event = None
    _thread = None
    
    @classmethod
    def start(cls, scan_func, run_immediately=True):
        """
        Start scheduled scanning.
        scan_func: The function to call for scanning (main scan logic)
        """
        if cls._thread is not None and cls._thread.is_alive():
            print("  Scheduler already running.")
            return
        
        cls._stop_event = threading.Event()
        
        def scheduler_loop():
            print("\n" + "="*60)
            print("  SCHEDULED SCANNING MODE")
            print("="*60)
            print(f"  Market Status: {MarketHours.get_market_status_string()}")
            print(f"  Will scan at:")
            print(f"    - {SCHEDULE_MARKET_OPEN_DELAY_MINUTES} min after market open (9:35 AM ET)")
            print(f"    - {SCHEDULE_MARKET_CLOSE_BEFORE_MINUTES} min before market close (3:55 PM ET)")
            print("  Press Ctrl+C to stop.")
            print("="*60)
            
            last_scan_date = None
            scanned_open = False
            scanned_close = False
            
            while not cls._stop_event.is_set():
                now = MarketHours.get_eastern_time()
                today = now.date() if hasattr(now, 'date') else date.today()
                
                # Reset flags on new day
                if last_scan_date != today:
                    last_scan_date = today
                    scanned_open = False
                    scanned_close = False
                
                # Skip weekends
                if now.weekday() >= 5:
                    cls._stop_event.wait(60)
                    continue
                
                hour = now.hour
                minute = now.minute
                
                # Market open scan (9:35 AM)
                target_open_hour = MARKET_OPEN_HOUR
                target_open_minute = MARKET_OPEN_MINUTE + SCHEDULE_MARKET_OPEN_DELAY_MINUTES
                if target_open_minute >= 60:
                    target_open_hour += 1
                    target_open_minute -= 60
                
                if not scanned_open and hour == target_open_hour and minute >= target_open_minute and minute < target_open_minute + 5:
                    print(f"\n[{now.strftime('%H:%M')}] Running scheduled MARKET OPEN scan...")
                    try:
                        scan_func()
                    except Exception as e:
                        print(f"  Scan error: {e}")
                    scanned_open = True
                
                # Market close scan (3:55 PM)
                target_close_hour = MARKET_CLOSE_HOUR
                target_close_minute = MARKET_CLOSE_MINUTE - SCHEDULE_MARKET_CLOSE_BEFORE_MINUTES
                if target_close_minute < 0:
                    target_close_hour -= 1
                    target_close_minute += 60
                
                if not scanned_close and hour == target_close_hour and minute >= target_close_minute and minute < target_close_minute + 5:
                    print(f"\n[{now.strftime('%H:%M')}] Running scheduled MARKET CLOSE scan...")
                    try:
                        scan_func()
                    except Exception as e:
                        print(f"  Scan error: {e}")
                    scanned_close = True
                
                # Wait before checking again
                cls._stop_event.wait(30)  # Check every 30 seconds
        
        cls._thread = threading.Thread(target=scheduler_loop, daemon=True)
        cls._thread.start()
        
        if run_immediately:
            print("  Running initial scan...")
            scan_func()
        
        # Keep main thread alive
        try:
            while cls._thread.is_alive():
                cls._thread.join(timeout=1)
        except KeyboardInterrupt:
            print("\n  Stopping scheduler...")
            cls._stop_event.set()
            cls._thread.join(timeout=5)
            print("  Scheduler stopped.")
    
    @classmethod
    def stop(cls):
        """Stop the scheduler."""
        if cls._stop_event:
            cls._stop_event.set()


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


def _apply_trust_mode(args):
    """
    Apply TRUST MODE settings - for users who want to run and trust results.
    This mode is EXTREMELY strict. Only the highest confidence setups pass.
    """
    if not getattr(args, "trust_mode", False):
        return args
    
    # Apply all trust mode settings
    for key, value in TRUST_MODE_SETTINGS.items():
        setattr(args, key, value)
    
    # Force additional safety settings
    args.auto_oos = True
    args.near_miss_report = True
    args.save_json = True
    args.require_walkforward = True
    args.require_oos = True
    
    # Override risk settings to be ultra-conservative
    args.risk_per_trade = min(args.risk_per_trade, args.account_size * TRUST_MODE_MAX_RISK_PER_TRADE_PCT / 100)
    
    return args


# =============================================================================
# TRUST MODE MANAGER - Tracks trading activity and enforces limits
# =============================================================================
class TrustModeManager:
    """
    Manages Trust Mode protections:
    - Daily/weekly trade limits
    - Loss streak cooloff
    - Paper trading validation
    - Position limits
    """
    
    TRUST_STATE_FILE = "trust_mode_state.json"
    
    def __init__(self, account_size=ACCOUNT_SIZE):
        self.account_size = account_size
        self._load_state()
    
    def _load_state(self):
        """Load trust mode tracking state."""
        self.state = {
            "trades_today": [],
            "trades_this_week": [],
            "last_trade_date": None,
            "last_week_start": None,
            "consecutive_losses": 0,
            "cooloff_until": None,
            "paper_trading_started": None,
            "paper_trades_count": 0,
            "paper_wins": 0,
            "paper_losses": 0,
            "paper_validated": False,
            "total_trades": 0,
            "total_wins": 0,
            "total_losses": 0,
            "current_positions": 0,
        }
        
        if os.path.exists(self.TRUST_STATE_FILE):
            try:
                with open(self.TRUST_STATE_FILE, "r") as f:
                    saved = json.load(f)
                    self.state.update(saved)
            except Exception:
                pass
        
        # Reset daily/weekly counters if needed
        self._reset_counters_if_needed()
    
    def _save_state(self):
        """Save trust mode state."""
        try:
            with open(self.TRUST_STATE_FILE, "w") as f:
                json.dump(self.state, f, indent=2, default=str)
        except Exception:
            pass
    
    def _reset_counters_if_needed(self):
        """Reset daily/weekly counters on new day/week."""
        today = date.today().isoformat()
        week_start = (date.today() - timedelta(days=date.today().weekday())).isoformat()
        
        if self.state["last_trade_date"] != today:
            self.state["trades_today"] = []
            self.state["last_trade_date"] = today
        
        if self.state["last_week_start"] != week_start:
            self.state["trades_this_week"] = []
            self.state["last_week_start"] = week_start
        
        # Check cooloff expiry
        if self.state["cooloff_until"]:
            try:
                cooloff_date = datetime.fromisoformat(self.state["cooloff_until"])
                if datetime.now() > cooloff_date:
                    self.state["cooloff_until"] = None
                    self.state["consecutive_losses"] = 0
            except Exception:
                pass
    
    def is_paper_trading_validated(self):
        """Check if paper trading period is complete and validated."""
        if self.state["paper_validated"]:
            return True, "Paper trading validated"
        
        if not self.state["paper_trading_started"]:
            return False, f"Paper trading not started. Run with --trust-paper for {TRUST_MODE_PAPER_VALIDATION_DAYS} days first."
        
        # Check if enough time has passed
        started = datetime.fromisoformat(self.state["paper_trading_started"])
        days_elapsed = (datetime.now() - started).days
        
        if days_elapsed < TRUST_MODE_PAPER_VALIDATION_DAYS:
            return False, f"Paper trading day {days_elapsed}/{TRUST_MODE_PAPER_VALIDATION_DAYS}. Keep paper trading."
        
        # Check if enough trades
        if self.state["paper_trades_count"] < TRUST_MODE_PAPER_MIN_TRADES:
            return False, f"Need {TRUST_MODE_PAPER_MIN_TRADES} paper trades, have {self.state['paper_trades_count']}."
        
        # Check win rate
        total = self.state["paper_wins"] + self.state["paper_losses"]
        if total > 0:
            win_rate = (self.state["paper_wins"] / total) * 100
            if win_rate < TRUST_MODE_PAPER_MIN_WINRATE:
                return False, f"Paper win rate {win_rate:.0f}% < {TRUST_MODE_PAPER_MIN_WINRATE}%. Keep practicing."
        
        # Validation passed!
        self.state["paper_validated"] = True
        self._save_state()
        return True, "Paper trading validated! You may now use live trading."
    
    def start_paper_trading(self):
        """Start the paper trading validation period."""
        if not self.state["paper_trading_started"]:
            self.state["paper_trading_started"] = datetime.now().isoformat()
            self._save_state()
        return self.state["paper_trading_started"]
    
    def record_paper_trade(self, won: bool):
        """Record a paper trade result."""
        self.state["paper_trades_count"] += 1
        if won:
            self.state["paper_wins"] += 1
        else:
            self.state["paper_losses"] += 1
        self._save_state()
    
    def can_trade_today(self):
        """Check if we can take more trades today."""
        self._reset_counters_if_needed()
        
        # Check cooloff period
        if self.state["cooloff_until"]:
            try:
                cooloff_date = datetime.fromisoformat(self.state["cooloff_until"])
                if datetime.now() < cooloff_date:
                    return False, f"Cooloff period active until {cooloff_date.strftime('%Y-%m-%d')}. {self.state['consecutive_losses']} consecutive losses."
            except Exception:
                pass
        
        # Check daily limit
        if len(self.state["trades_today"]) >= TRUST_MODE_MAX_TRADES_PER_DAY:
            return False, f"Daily limit reached ({TRUST_MODE_MAX_TRADES_PER_DAY} trades). Try again tomorrow."
        
        # Check weekly limit
        if len(self.state["trades_this_week"]) >= TRUST_MODE_MAX_TRADES_PER_WEEK:
            return False, f"Weekly limit reached ({TRUST_MODE_MAX_TRADES_PER_WEEK} trades). Wait for next week."
        
        # Check position limit
        if self.state["current_positions"] >= TRUST_MODE_MAX_POSITIONS:
            return False, f"Position limit reached ({TRUST_MODE_MAX_POSITIONS} positions). Close some first."
        
        return True, "OK"
    
    def record_trade(self, ticker: str, won: bool = None):
        """Record a new trade."""
        self._reset_counters_if_needed()
        
        trade = {
            "ticker": ticker,
            "time": datetime.now().isoformat(),
        }
        
        self.state["trades_today"].append(trade)
        self.state["trades_this_week"].append(trade)
        self.state["total_trades"] += 1
        self.state["current_positions"] += 1
        
        if won is not None:
            self.record_trade_result(won)
        
        self._save_state()
    
    def record_trade_result(self, won: bool):
        """Record a trade result (win/loss)."""
        if won:
            self.state["total_wins"] += 1
            self.state["consecutive_losses"] = 0
        else:
            self.state["total_losses"] += 1
            self.state["consecutive_losses"] += 1
            
            # Check for cooloff trigger
            if self.state["consecutive_losses"] >= TRUST_MODE_LOSS_STREAK_COOLOFF:
                cooloff_date = datetime.now() + timedelta(days=TRUST_MODE_COOLOFF_DAYS)
                self.state["cooloff_until"] = cooloff_date.isoformat()
        
        self._save_state()
    
    def close_position(self):
        """Record a position closure."""
        if self.state["current_positions"] > 0:
            self.state["current_positions"] -= 1
            self._save_state()
    
    def get_status_report(self):
        """Get a status report for display."""
        self._reset_counters_if_needed()
        
        report = {
            "trades_today": len(self.state["trades_today"]),
            "max_daily": TRUST_MODE_MAX_TRADES_PER_DAY,
            "trades_this_week": len(self.state["trades_this_week"]),
            "max_weekly": TRUST_MODE_MAX_TRADES_PER_WEEK,
            "current_positions": self.state["current_positions"],
            "max_positions": TRUST_MODE_MAX_POSITIONS,
            "consecutive_losses": self.state["consecutive_losses"],
            "cooloff_until": self.state["cooloff_until"],
            "paper_validated": self.state["paper_validated"],
            "total_trades": self.state["total_trades"],
            "total_wins": self.state["total_wins"],
            "total_losses": self.state["total_losses"],
        }
        
        if self.state["total_trades"] > 0:
            report["win_rate"] = (self.state["total_wins"] / self.state["total_trades"]) * 100
        else:
            report["win_rate"] = 0
        
        return report
    
    def reset_paper_trading(self):
        """Reset paper trading state (for re-validation)."""
        self.state["paper_trading_started"] = None
        self.state["paper_trades_count"] = 0
        self.state["paper_wins"] = 0
        self.state["paper_losses"] = 0
        self.state["paper_validated"] = False
        self._save_state()
    
    def bypass_paper_validation(self, confirm_code: str):
        """
        Bypass paper validation (for experienced traders).
        Requires typing 'I ACCEPT THE RISK' to proceed.
        """
        if confirm_code == "I ACCEPT THE RISK":
            self.state["paper_validated"] = True
            self._save_state()
            return True
        return False


# =============================================================================
# AUTO MODE MANAGER - Just run the file, everything works
# =============================================================================
class AutoModeManager:
    """
    Manages auto mode configuration and first-time setup.
    Makes the program truly 'run and forget'.
    """
    
    CONFIG_FILE = AUTO_MODE_CONFIG_FILE
    
    def __init__(self):
        self.config = self._load_config()
    
    def _load_config(self):
        """Load auto mode configuration."""
        default_config = {
            "first_run_complete": False,
            "paper_trading_bypassed": False,
            "account_size": ACCOUNT_SIZE,
            "risk_per_trade": RISK_PER_TRADE,
            "notifications_enabled": True,
            "auto_schedule_enabled": False,
            "created": None,
        }
        
        if os.path.exists(self.CONFIG_FILE):
            try:
                with open(self.CONFIG_FILE, "r") as f:
                    saved = json.load(f)
                    default_config.update(saved)
            except Exception:
                pass
        
        return default_config
    
    def _save_config(self):
        """Save auto mode configuration."""
        try:
            with open(self.CONFIG_FILE, "w") as f:
                json.dump(self.config, f, indent=2, default=str)
        except Exception:
            pass
    
    def is_first_run(self):
        """Check if this is the first run."""
        return not self.config.get("first_run_complete", False)
    
    def run_first_time_setup(self):
        """Run first-time setup wizard."""
        print("\n" + "=" * 70)
        print("  " + "█" * 66)
        print("  █" + " " * 64 + "█")
        print("  █" + "     WELCOME TO TITAN TRADE - AUTO MODE     ".center(64) + "█")
        print("  █" + "     First-Time Setup (One time only)       ".center(64) + "█")
        print("  █" + " " * 64 + "█")
        print("  " + "█" * 66)
        print("=" * 70)
        
        print("\n  This setup will configure Titan Trade for automatic operation.")
        print("  After this, just run 'python titan_trade.py' and trust the results.\n")
        
        # Account size
        print("  1. What is your trading account size? (default: $100,000)")
        try:
            acc_input = input("     $ ").strip()
            if acc_input:
                self.config["account_size"] = float(acc_input.replace(",", "").replace("$", ""))
            else:
                self.config["account_size"] = ACCOUNT_SIZE
        except:
            self.config["account_size"] = ACCOUNT_SIZE
        
        # Risk per trade
        print(f"\n  2. Max risk per trade? (default: ${RISK_PER_TRADE:.0f})")
        print("     (This is how much you're willing to lose if stopped out)")
        try:
            risk_input = input("     $ ").strip()
            if risk_input:
                self.config["risk_per_trade"] = float(risk_input.replace(",", "").replace("$", ""))
            else:
                self.config["risk_per_trade"] = RISK_PER_TRADE
        except:
            self.config["risk_per_trade"] = RISK_PER_TRADE
        
        # Paper trading bypass
        print("\n  3. Paper trading validation is RECOMMENDED for 30 days.")
        print("     Do you want to skip paper trading? (NOT recommended)")
        print("     Type 'SKIP' to bypass, or press ENTER to paper trade first:")
        skip = input("     > ").strip()
        
        if skip.upper() == "SKIP":
            print("\n     ⚠️  WARNING: You chose to skip paper trading.")
            print("     Type 'I ACCEPT THE RISK' to confirm:")
            confirm = input("     > ").strip()
            if confirm == "I ACCEPT THE RISK":
                self.config["paper_trading_bypassed"] = True
                print("     Paper trading bypassed. Be careful with real money!")
            else:
                self.config["paper_trading_bypassed"] = False
                print("     Good choice! Paper trading will be required.")
        else:
            self.config["paper_trading_bypassed"] = False
            print("     Great! You'll paper trade first for safety.")
        
        # Complete setup
        self.config["first_run_complete"] = True
        self.config["created"] = datetime.now().isoformat()
        self._save_config()
        
        print("\n" + "=" * 70)
        print("  SETUP COMPLETE!")
        print("=" * 70)
        print(f"  Account Size: ${self.config['account_size']:,.0f}")
        print(f"  Risk Per Trade: ${self.config['risk_per_trade']:,.0f}")
        print(f"  Paper Trading: {'Bypassed' if self.config['paper_trading_bypassed'] else 'Required'}")
        print("=" * 70)
        print("\n  From now on, just run: python titan_trade.py")
        print("  The system will automatically scan and show you TRADE or DON'T TRADE.\n")
        
        input("  Press ENTER to continue to your first scan...")
        
        return self.config
    
    def get_config(self):
        """Get current configuration."""
        return self.config


def print_trust_mode_header():
    """Print the Trust Mode header."""
    print("\n" + "=" * 70)
    print("  " + "█" * 66)
    print("  █" + " " * 64 + "█")
    print("  █" + "     TITAN TRADE - TRUST MODE ACTIVATED     ".center(64) + "█")
    print("  █" + "     If it says TRADE, you can trust it.    ".center(64) + "█")
    print("  █" + "     If it says DON'T, respect it.          ".center(64) + "█")
    print("  █" + " " * 64 + "█")
    print("  " + "█" * 66)
    print("=" * 70)


def print_simple_verdict(setups, trust_manager, vix_level=None):
    """
    Print a simple, clear verdict for Trust Mode users.
    No ambiguity - just TRADE or DON'T TRADE.
    """
    print("\n" + "=" * 70)
    
    # Check if we can even trade
    can_trade, reason = trust_manager.can_trade_today()
    
    if not can_trade:
        print("  ╔══════════════════════════════════════════════════════════════╗")
        print("  ║                                                              ║")
        print("  ║     ██████   ██████  ███    ██ ████████                     ║")
        print("  ║     ██   ██ ██    ██ ████   ██    ██                        ║")
        print("  ║     ██   ██ ██    ██ ██ ██  ██    ██                        ║")
        print("  ║     ██   ██ ██    ██ ██  ██ ██    ██                        ║")
        print("  ║     ██████   ██████  ██   ████    ██     TRADE TODAY       ║")
        print("  ║                                                              ║")
        print("  ╚══════════════════════════════════════════════════════════════╝")
        print(f"\n  REASON: {reason}")
        print("\n  This is a PROTECTIVE LIMIT. Respect it.")
        print("=" * 70)
        return None
    
    # Check VIX
    if vix_level is not None:
        if vix_level > TRUST_MODE_VIX_HALT:
            print("  ╔══════════════════════════════════════════════════════════════╗")
            print("  ║                                                              ║")
            print("  ║              ⚠️  HIGH VOLATILITY WARNING  ⚠️                  ║")
            print("  ║                                                              ║")
            print("  ║                  DON'T TRADE TODAY                          ║")
            print("  ║                                                              ║")
            print("  ╚══════════════════════════════════════════════════════════════╝")
            print(f"\n  VIX is at {vix_level:.1f} (HALT threshold: {TRUST_MODE_VIX_HALT})")
            print("  Market is too volatile. Wait for calmer conditions.")
            print("=" * 70)
            return None
        elif vix_level > TRUST_MODE_VIX_CAUTION:
            print(f"\n  ⚠️  CAUTION: VIX at {vix_level:.1f} - Position sizes reduced 50%")
    
    # Filter for only Grade A or B signals
    trusted_setups = []
    for s in setups:
        grade = getattr(s, 'confidence_grade', 'F')
        t_stat = getattr(s, 't_statistic', 0)
        
        # Must be Grade A or B
        if grade not in ['A', 'B']:
            continue
        
        # Must be statistically significant
        if TRUST_MODE_REQUIRE_SIGNIFICANCE and t_stat < 2.0:
            continue
        
        trusted_setups.append(s)
    
    if not trusted_setups:
        print("  ╔══════════════════════════════════════════════════════════════╗")
        print("  ║                                                              ║")
        print("  ║              NO HIGH-CONFIDENCE TRADES TODAY                ║")
        print("  ║                                                              ║")
        print("  ║         Wait for Grade A/B signals with t-stat ≥ 2.0        ║")
        print("  ║                                                              ║")
        print("  ╚══════════════════════════════════════════════════════════════╝")
        print(f"\n  Scanned all stocks. Found {len(setups)} setups but none meet TRUST criteria.")
        print("  This is GOOD - we're protecting you from marginal trades.")
        print("=" * 70)
        return None
    
    # We have trusted setups!
    print("  ╔══════════════════════════════════════════════════════════════╗")
    print("  ║                                                              ║")
    print("  ║     ████████ ██████   █████  ██████  ███████                ║")
    print("  ║        ██    ██   ██ ██   ██ ██   ██ ██                     ║")
    print("  ║        ██    ██████  ███████ ██   ██ █████                  ║")
    print("  ║        ██    ██   ██ ██   ██ ██   ██ ██                     ║")
    print("  ║        ██    ██   ██ ██   ██ ██████  ███████   SIGNAL!     ║")
    print("  ║                                                              ║")
    print("  ╚══════════════════════════════════════════════════════════════╝")
    
    print(f"\n  Found {len(trusted_setups)} HIGH-CONFIDENCE trade(s):\n")
    
    for i, s in enumerate(trusted_setups[:TRUST_MODE_MAX_TRADES_PER_DAY], 1):
        # Calculate position size adjusted for VIX
        shares = s.qty
        if vix_level and vix_level > TRUST_MODE_VIX_CAUTION:
            shares = max(1, shares // 2)  # Cut size in half
        
        risk_per_share = s.trigger - s.stop
        total_risk = risk_per_share * shares
        potential_profit = (s.target - s.trigger) * shares
        rr_ratio = (s.target - s.trigger) / risk_per_share if risk_per_share > 0 else 0
        
        print(f"  ┌─────────────────────────────────────────────────────────────┐")
        print(f"  │  #{i}  {s.ticker:<6}  {s.strategy:<12}  GRADE: {s.confidence_grade}  t-stat: {s.t_statistic:.1f}  │")
        print(f"  ├─────────────────────────────────────────────────────────────┤")
        print(f"  │  BUY:     {shares:>6} shares @ ${s.trigger:>8.2f}                     │")
        print(f"  │  STOP:    Exit if price falls to ${s.stop:>8.2f}                │")
        print(f"  │  TARGET:  Take profit at ${s.target:>8.2f}                       │")
        print(f"  ├─────────────────────────────────────────────────────────────┤")
        print(f"  │  RISK:    ${total_risk:>8.2f}  |  REWARD: ${potential_profit:>8.2f}  |  R:R {rr_ratio:.1f}:1  │")
        print(f"  │  Win Rate: {s.win_rate:.0f}%  |  Profit Factor: {s.profit_factor:.2f}              │")
        print(f"  └─────────────────────────────────────────────────────────────┘")
        print()
    
    print("  ─────────────────────────────────────────────────────────────────")
    print("  INSTRUCTIONS:")
    print("    1. Open your broker")
    print("    2. Place a BUY STOP order at the TRIGGER price")
    print("    3. Set your STOP LOSS order immediately after fill")
    print("    4. Set your TAKE PROFIT order at TARGET price")
    print("    5. DO NOT move your stop loss lower!")
    print("  ─────────────────────────────────────────────────────────────────")
    print("=" * 70)
    
    return trusted_setups


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


# =============================================================================
# PORTFOLIO RISK MANAGER - Critical for survival
# =============================================================================
class PortfolioRiskManager:
    """
    Tracks portfolio-level risk and enforces hard limits.
    This is what separates surviving traders from blown accounts.
    """
    
    def __init__(
        self,
        account_size=ACCOUNT_SIZE,
        max_daily_loss_pct=MAX_DAILY_LOSS_PCT,
        max_weekly_loss_pct=MAX_WEEKLY_LOSS_PCT,
        max_drawdown_pct=MAX_DRAWDOWN_PCT,
        max_portfolio_heat=PORTFOLIO_HEAT_MAX,
        risk_log_file=RISK_LOG_FILE,
    ):
        self.account_size = account_size
        self.max_daily_loss_pct = max_daily_loss_pct
        self.max_weekly_loss_pct = max_weekly_loss_pct
        self.max_drawdown_pct = max_drawdown_pct
        self.max_portfolio_heat = max_portfolio_heat
        self.risk_log_file = risk_log_file
        
        # Load existing risk state
        self._load_state()
    
    def _load_state(self):
        """Load risk tracking state from file."""
        self.state = {
            "peak_equity": self.account_size,
            "current_equity": self.account_size,
            "daily_pnl": 0.0,
            "weekly_pnl": 0.0,
            "last_trade_date": None,
            "last_week_start": None,
            "open_positions": {},
            "trade_history": [],
        }
        
        if os.path.exists(self.risk_log_file):
            try:
                with open(self.risk_log_file, "r") as f:
                    saved = json.load(f)
                    self.state.update(saved)
            except Exception:
                pass
    
    def _save_state(self):
        """Persist risk state to file."""
        try:
            with open(self.risk_log_file, "w") as f:
                json.dump(self.state, f, indent=2, default=str)
        except Exception:
            pass
    
    def _reset_daily_if_needed(self):
        """Reset daily P&L if it's a new trading day."""
        today = date.today().isoformat()
        if self.state["last_trade_date"] != today:
            self.state["daily_pnl"] = 0.0
            self.state["last_trade_date"] = today
    
    def _reset_weekly_if_needed(self):
        """Reset weekly P&L if it's a new trading week."""
        today = date.today()
        # Get Monday of current week
        week_start = (today - timedelta(days=today.weekday())).isoformat()
        if self.state["last_week_start"] != week_start:
            self.state["weekly_pnl"] = 0.0
            self.state["last_week_start"] = week_start
    
    def update_equity(self, new_equity):
        """Update current equity and track peak for drawdown."""
        self.state["current_equity"] = new_equity
        if new_equity > self.state["peak_equity"]:
            self.state["peak_equity"] = new_equity
        self._save_state()
    
    def record_trade_result(self, pnl_dollars):
        """Record a closed trade result."""
        self._reset_daily_if_needed()
        self._reset_weekly_if_needed()
        
        self.state["daily_pnl"] += pnl_dollars
        self.state["weekly_pnl"] += pnl_dollars
        self.state["current_equity"] += pnl_dollars
        
        if self.state["current_equity"] > self.state["peak_equity"]:
            self.state["peak_equity"] = self.state["current_equity"]
        
        self.state["trade_history"].append({
            "date": datetime.now().isoformat(),
            "pnl": pnl_dollars,
        })
        
        # Keep only last 100 trades in history
        if len(self.state["trade_history"]) > 100:
            self.state["trade_history"] = self.state["trade_history"][-100:]
        
        self._save_state()
    
    def get_current_drawdown_pct(self):
        """Calculate current drawdown from peak."""
        peak = self.state["peak_equity"]
        current = self.state["current_equity"]
        if peak <= 0:
            return 0.0
        return (peak - current) / peak * 100
    
    def get_daily_loss_pct(self):
        """Get today's P&L as percentage of account."""
        self._reset_daily_if_needed()
        return -self.state["daily_pnl"] / self.account_size * 100
    
    def get_weekly_loss_pct(self):
        """Get this week's P&L as percentage of account."""
        self._reset_weekly_if_needed()
        return -self.state["weekly_pnl"] / self.account_size * 100
    
    def calculate_portfolio_heat(self, open_positions):
        """
        Calculate total portfolio risk (heat).
        Heat = sum of (risk per share * shares) for all positions.
        """
        total_risk = 0.0
        for ticker, pos in open_positions.items():
            entry = pos.get("entry_price", 0)
            stop = pos.get("stop_loss", 0)
            shares = pos.get("shares", 0)
            risk_per_share = entry - stop
            if risk_per_share > 0:
                total_risk += risk_per_share * shares
        
        return total_risk / self.account_size * 100 if self.account_size > 0 else 0.0
    
    def can_take_new_trade(self, open_positions=None):
        """
        Check if we're allowed to take a new trade based on risk limits.
        Returns: (allowed: bool, reason: str)
        """
        self._reset_daily_if_needed()
        self._reset_weekly_if_needed()
        
        # Check drawdown
        dd = self.get_current_drawdown_pct()
        if dd >= self.max_drawdown_pct:
            return False, f"MAX DRAWDOWN EXCEEDED: {dd:.1f}% (limit: {self.max_drawdown_pct}%)"
        
        # Check daily loss
        daily_loss = self.get_daily_loss_pct()
        if daily_loss >= self.max_daily_loss_pct:
            return False, f"DAILY LOSS LIMIT: {daily_loss:.1f}% (limit: {self.max_daily_loss_pct}%)"
        
        # Check weekly loss
        weekly_loss = self.get_weekly_loss_pct()
        if weekly_loss >= self.max_weekly_loss_pct:
            return False, f"WEEKLY LOSS LIMIT: {weekly_loss:.1f}% (limit: {self.max_weekly_loss_pct}%)"
        
        # Check portfolio heat
        if open_positions:
            heat = self.calculate_portfolio_heat(open_positions)
            if heat >= self.max_portfolio_heat:
                return False, f"PORTFOLIO HEAT LIMIT: {heat:.1f}% (limit: {self.max_portfolio_heat}%)"
        
        return True, "OK"
    
    def get_position_size_scalar(self):
        """
        Returns a scalar (0.0 to 1.0) to reduce position size based on risk state.
        As we approach limits, we reduce size progressively.
        """
        # Start at 100%
        scalar = 1.0
        
        # Reduce as drawdown increases
        dd = self.get_current_drawdown_pct()
        if dd > self.max_drawdown_pct * 0.5:  # Past 50% of max DD
            dd_scalar = 1.0 - (dd - self.max_drawdown_pct * 0.5) / (self.max_drawdown_pct * 0.5)
            scalar = min(scalar, max(0.25, dd_scalar))
        
        # Reduce as daily loss increases
        daily_loss = self.get_daily_loss_pct()
        if daily_loss > self.max_daily_loss_pct * 0.5:
            daily_scalar = 1.0 - (daily_loss - self.max_daily_loss_pct * 0.5) / (self.max_daily_loss_pct * 0.5)
            scalar = min(scalar, max(0.25, daily_scalar))
        
        return scalar
    
    def get_risk_status(self, open_positions=None):
        """Get current risk status summary."""
        dd = self.get_current_drawdown_pct()
        daily = self.get_daily_loss_pct()
        weekly = self.get_weekly_loss_pct()
        heat = self.calculate_portfolio_heat(open_positions or {})
        
        can_trade, reason = self.can_take_new_trade(open_positions)
        
        return {
            "can_trade": can_trade,
            "reason": reason,
            "drawdown_pct": dd,
            "daily_loss_pct": daily,
            "weekly_loss_pct": weekly,
            "portfolio_heat_pct": heat,
            "position_size_scalar": self.get_position_size_scalar(),
            "current_equity": self.state["current_equity"],
            "peak_equity": self.state["peak_equity"],
        }


# =============================================================================
# STATISTICAL CONFIDENCE SCORER - Replaces fake AI module
# =============================================================================
class StatisticalConfidenceScorer:
    """
    Calculates confidence score based on statistical robustness.
    No magic - just math that professionals trust.
    """
    
    @staticmethod
    def calculate_confidence(
        trades: int,
        win_rate: float,
        profit_factor: float,
        expectancy: float,
        wf_pass_rate: float = None,
        oos_pf: float = None,
        consistency_score: float = None,
    ) -> dict:
        """
        Calculate overall confidence score (0-100) based on statistical factors.
        
        Returns dict with:
        - score: Overall confidence (0-100)
        - grade: Letter grade (A/B/C/D/F)
        - factors: Breakdown of scoring factors
        """
        score = 0
        factors = {}
        
        # Factor 1: Sample Size (0-25 points)
        # Need 30+ trades for basic confidence, 50+ for high confidence
        if trades >= 100:
            sample_score = 25
        elif trades >= 50:
            sample_score = 20
        elif trades >= 30:
            sample_score = 15
        elif trades >= 15:
            sample_score = 8
        else:
            sample_score = 0
        score += sample_score
        factors["sample_size"] = {"value": trades, "score": sample_score, "max": 25}
        
        # Factor 2: Win Rate Consistency (0-20 points)
        # Higher win rate = more consistent edge
        if win_rate >= 60:
            wr_score = 20
        elif win_rate >= 55:
            wr_score = 15
        elif win_rate >= 50:
            wr_score = 10
        elif win_rate >= 45:
            wr_score = 5
        else:
            wr_score = 0
        score += wr_score
        factors["win_rate"] = {"value": win_rate, "score": wr_score, "max": 20}
        
        # Factor 3: Profit Factor (0-20 points)
        # PF > 1.5 is good, > 2.0 is excellent
        if profit_factor >= 2.5:
            pf_score = 20
        elif profit_factor >= 2.0:
            pf_score = 16
        elif profit_factor >= 1.5:
            pf_score = 12
        elif profit_factor >= 1.2:
            pf_score = 8
        elif profit_factor >= 1.0:
            pf_score = 4
        else:
            pf_score = 0
        score += pf_score
        factors["profit_factor"] = {"value": profit_factor, "score": pf_score, "max": 20}
        
        # Factor 4: Expectancy (0-15 points)
        # Must be meaningfully positive after all costs
        if expectancy >= 0.02:  # 2%+ per trade
            exp_score = 15
        elif expectancy >= 0.01:  # 1%+ per trade
            exp_score = 12
        elif expectancy >= 0.005:  # 0.5%+ per trade
            exp_score = 8
        elif expectancy >= 0.002:  # 0.2%+ per trade
            exp_score = 4
        else:
            exp_score = 0
        score += exp_score
        factors["expectancy"] = {"value": expectancy, "score": exp_score, "max": 15}
        
        # Factor 5: Walk-Forward Validation (0-10 points)
        if wf_pass_rate is not None:
            if wf_pass_rate >= 0.75:
                wf_score = 10
            elif wf_pass_rate >= 0.50:
                wf_score = 7
            elif wf_pass_rate >= 0.25:
                wf_score = 3
            else:
                wf_score = 0
            score += wf_score
            factors["wf_pass_rate"] = {"value": wf_pass_rate, "score": wf_score, "max": 10}
        
        # Factor 6: Out-of-Sample Performance (0-10 points)
        if oos_pf is not None:
            if oos_pf >= 1.5:
                oos_score = 10
            elif oos_pf >= 1.2:
                oos_score = 7
            elif oos_pf >= 1.0:
                oos_score = 4
            else:
                oos_score = 0
            score += oos_score
            factors["oos_profit_factor"] = {"value": oos_pf, "score": oos_score, "max": 10}
        
        # Calculate grade
        if score >= 80:
            grade = "A"
        elif score >= 65:
            grade = "B"
        elif score >= 50:
            grade = "C"
        elif score >= 35:
            grade = "D"
        else:
            grade = "F"
        
        return {
            "score": score,
            "grade": grade,
            "factors": factors,
            "tradeable": score >= 50 and trades >= 30,  # Minimum bar for real trading
        }
    
    @staticmethod
    def calculate_t_statistic(trades: list) -> float:
        """
        Calculate t-statistic to test if returns are significantly > 0.
        t > 2.0 suggests edge is statistically significant at 95% confidence.
        """
        if not trades or len(trades) < 10:
            return 0.0
        
        mean_ret = np.mean(trades)
        std_ret = np.std(trades, ddof=1)
        n = len(trades)
        
        if std_ret == 0:
            return 0.0
        
        t_stat = mean_ret / (std_ret / np.sqrt(n))
        return float(t_stat)
    
    @staticmethod
    def calculate_sharpe_ratio(trades: list, periods_per_year: int = 252) -> float:
        """
        Calculate annualized Sharpe ratio from trade returns.
        Sharpe > 1.0 is good, > 2.0 is excellent.
        """
        if not trades or len(trades) < 10:
            return 0.0
        
        mean_ret = np.mean(trades)
        std_ret = np.std(trades, ddof=1)
        
        if std_ret == 0:
            return 0.0
        
        # Annualize (assuming average holding period of ~5 days)
        trades_per_year = periods_per_year / 5
        sharpe = (mean_ret * trades_per_year) / (std_ret * np.sqrt(trades_per_year))
        return float(sharpe)


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
    earnings_call: str
    note: str
    # Statistical Confidence Metrics (real, not fake AI)
    confidence_score: float = 0.0    # Statistical confidence (0-100)
    confidence_grade: str = "F"      # Letter grade (A/B/C/D/F)
    trend_grade: str = "F"           # Trend quality grade
    t_statistic: float = 0.0         # Statistical significance

# =============================================================================
# TREND QUALITY ANALYZER - Simple, robust, no magic
# =============================================================================
class TrendQualityAnalyzer:
    """
    Analyzes trend quality using proven technical factors.
    No fake AI - just straightforward technical analysis.
    """
    
    @staticmethod
    def analyze(df, backtest_res):
        """
        Analyze trend quality and return objective metrics.
        Returns dict with clear, interpretable factors.
        """
        if len(df) < 200:
            return {
                "trend_score": 0,
                "trend_grade": "F",
                "factors": {},
            }
        
        c = df['Close']
        h = df['High']
        v = df['Volume']
        
        factors = {}
        score = 0
        
        # Factor 1: Moving Average Alignment (0-20)
        sma20 = c.rolling(20).mean().iloc[-1]
        sma50 = c.rolling(50).mean().iloc[-1]
        sma200 = c.rolling(200).mean().iloc[-1]
        curr = c.iloc[-1]
        
        ma_alignment = 0
        if curr > sma20 > sma50 > sma200:
            ma_alignment = 20  # Perfect bull alignment
        elif curr > sma50 > sma200:
            ma_alignment = 15
        elif curr > sma200:
            ma_alignment = 10
        elif curr > sma50:
            ma_alignment = 5
        
        factors["ma_alignment"] = ma_alignment
        score += ma_alignment
        
        # Factor 2: Distance from 52-week high (0-15)
        high_52w = h.iloc[-252:].max() if len(df) >= 252 else h.max()
        pct_from_high = (curr / high_52w - 1) * 100
        
        if pct_from_high > -3:
            proximity_score = 15
        elif pct_from_high > -7:
            proximity_score = 12
        elif pct_from_high > -15:
            proximity_score = 8
        elif pct_from_high > -25:
            proximity_score = 4
        else:
            proximity_score = 0
        
        factors["proximity_to_high"] = proximity_score
        factors["pct_from_52w_high"] = round(pct_from_high, 2)
        score += proximity_score
        
        # Factor 3: Volume trend (0-10)
        vol_20d = v.iloc[-20:].mean()
        vol_50d = v.iloc[-50:].mean()
        vol_ratio = vol_20d / (vol_50d + 1e-9)
        
        if vol_ratio > 1.3:
            vol_score = 10  # Volume expanding
        elif vol_ratio > 1.0:
            vol_score = 7
        elif vol_ratio > 0.7:
            vol_score = 4
        else:
            vol_score = 0  # Volume contracting badly
        
        factors["volume_trend"] = vol_score
        score += vol_score
        
        # Factor 4: Trend momentum (0-15)
        ret_20d = (curr / c.iloc[-21] - 1) * 100 if len(df) >= 21 else 0
        ret_60d = (curr / c.iloc[-61] - 1) * 100 if len(df) >= 61 else 0
        
        if ret_20d > 5 and ret_60d > 10:
            momentum_score = 15
        elif ret_20d > 2 and ret_60d > 5:
            momentum_score = 12
        elif ret_20d > 0 and ret_60d > 0:
            momentum_score = 8
        elif ret_20d > -5 and ret_60d > -10:
            momentum_score = 4
        else:
            momentum_score = 0
        
        factors["momentum"] = momentum_score
        factors["return_20d"] = round(ret_20d, 2)
        factors["return_60d"] = round(ret_60d, 2)
        score += momentum_score
        
        # Determine grade
        if score >= 50:
            grade = "A"
        elif score >= 40:
            grade = "B"
        elif score >= 30:
            grade = "C"
        elif score >= 20:
            grade = "D"
        else:
            grade = "F"
        
        return {
            "trend_score": score,
            "trend_grade": grade,
            "factors": factors,
        }


# Global analyzer instance
TREND_ANALYZER = TrendQualityAnalyzer()

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
        
        # --- VIX INTEGRATION ---
        # High VIX = high fear = reduce exposure or halt trading
        vix_scalar = 1.0
        vix_level = None
        for vix_key in ["^VIX", "VIX", "VIXY"]:
            if vix_key in self.data:
                try:
                    vix_df = self.data[vix_key]
                    if isinstance(vix_df, pd.DataFrame) and 'Close' in vix_df.columns:
                        vix_level = float(vix_df['Close'].iloc[-1])
                        if vix_level > VIX_PANIC_THRESHOLD:
                            vix_scalar = 0.0  # PANIC - NO NEW POSITIONS
                            status = f"{status}+PANIC"
                        elif vix_level > VIX_EXTREME_THRESHOLD:
                            vix_scalar = 0.25  # Extreme fear - cut exposure 75%
                            status = f"{status}+FEAR"
                        elif vix_level > VIX_HIGH_THRESHOLD:
                            vix_scalar = 0.5  # High fear - cut exposure 50%
                            status = f"{status}+CAUTION"
                        break
                except Exception:
                    pass
                
        return status, score * vix_scalar, vix_level

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

    def _simulate_trade(
        self, 
        entry, 
        stop, 
        target, 
        ohlc_data, 
        start_idx, 
        max_hold=10,  # Extended from 8 for more realistic holding
        trail_risk=None, 
        trail_r1=1.0, 
        trail_r2=2.0, 
        trail_r3=3.0,
        slippage_pct=0.003,  # 0.3% default slippage
    ):
        """
        Simulate a trade with REALISTIC execution model.
        
        Key improvements:
        1. Gap-through stops: If open gaps below stop, fill at open (not stop)
        2. Slippage on entry and exit
        3. More conservative stop checking (open first, then intraday)
        """
        closes, highs, lows, opens = ohlc_data
        stop_curr = stop
        highest_since_entry = entry
        end_idx = min(start_idx + max_hold, len(closes))
        
        # Apply entry slippage (always pay the spread)
        actual_entry = entry * (1 + slippage_pct)

        for j in range(start_idx, end_idx):
            day_open = opens[j] if j < len(opens) else closes[j-1] if j > 0 else entry
            day_high = highs[j]
            day_low = lows[j]
            day_close = closes[j]
            
            # Update highest price since entry (for trailing)
            highest_since_entry = max(highest_since_entry, day_high)
            
            # 3-Tier Trailing Stop Logic
            if trail_risk is not None and trail_risk > 0:
                # Tier 1: Move to breakeven at 1R
                if highest_since_entry >= actual_entry + (trail_risk * trail_r1):
                    stop_curr = max(stop_curr, actual_entry)
                # Tier 2: Lock 1R at 2R profit
                if highest_since_entry >= actual_entry + (trail_risk * trail_r2):
                    stop_curr = max(stop_curr, actual_entry + trail_risk)
                # Tier 3: Lock 2R at 3R profit
                if highest_since_entry >= actual_entry + (trail_risk * trail_r3):
                    stop_curr = max(stop_curr, actual_entry + trail_risk * 2)
            
            # =================================================================
            # REALISTIC STOP EXECUTION
            # =================================================================
            # Check 1: Gap down through stop on open (WORST CASE - fill at open)
            if day_open <= stop_curr:
                # Stop is gapped through - fill at open, not stop price
                fill_price = day_open * (1 - slippage_pct)  # Extra slippage on panic exit
                return (fill_price - actual_entry) / actual_entry
            
            # Check 2: Intraday stop hit (fill at stop with slippage)
            if day_low <= stop_curr:
                fill_price = stop_curr * (1 - slippage_pct * 0.5)  # Some slippage
                return (fill_price - actual_entry) / actual_entry
            
            # =================================================================
            # TARGET CHECK
            # =================================================================
            # Check for gap up through target (fill at open, lucky)
            if day_open >= target:
                fill_price = day_open  # Lucky - gap through target
                return (fill_price - actual_entry) / actual_entry
            
            # Check for intraday target hit
            if day_high >= target:
                fill_price = target * (1 - slippage_pct * 0.3)  # Some slippage
                return (fill_price - actual_entry) / actual_entry
            
            # Mark to market on last day (forced exit)
            if j == end_idx - 1:
                fill_price = day_close * (1 - slippage_pct)
                return (fill_price - actual_entry) / actual_entry

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
                    (closes, highs, lows, opens),
                    i + 2,  # Start from day after trigger (i+1 is trigger day)
                    max_hold=10,
                    trail_risk=risk,
                    slippage_pct=0.003,  # 0.3% slippage for breakouts (crowded)
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
                        (closes.values, highs.values, lows.values, opens.values),
                        i + 2,  # Start from day after entry
                        max_hold=10,
                        trail_risk=None,
                        slippage_pct=0.001,  # 0.1% slippage for dips (less crowded)
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
# 3. OPTIMIZER (Conservative Auto-Tuning)
# -----------------------------------------------------------------------------
class Optimizer:
    def __init__(self, validator):
        self.validator = validator
        
    def tune_breakout(self):
        """
        Try different parameters to find the best fit.
        REQUIRES minimum 15 trades in optimization to avoid overfitting.
        """
        best_res = {'win_rate': 0, 'pf': 0, 'score': 0}
        best_params = {'depth': 0.18, 'target_mult': 3.0}  # Conservative defaults
        
        # Grid Search with fewer parameters (avoid overfitting)
        # Depth: 0.15 (Tight) to 0.22 (Moderate)
        # Target: 2.5 ATR to 4.0 ATR (conservative range)
        for d in [0.15, 0.18, 0.22]:
            for t_mult in [2.5, 3.0, 3.5, 4.0]:
                res = self.validator.backtest_breakout(depth=d, target_mult=t_mult)
                # Score: PF * WR (only consider if enough trades)
                if res['trades'] < 15:  # Require 15+ trades for optimization
                    continue
                    
                score = res['pf'] * res['win_rate']
                
                if score > best_res['score']:
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

        # OHLCV with Smart Auto-Refresh
        cache_ttl_sec = self.cache_ttl_hours * 3600
        cache_valid = (
            os.path.exists(OHLCV_CACHE_FILE)
            and cache_ttl_sec > 0
            and (time.time() - os.path.getmtime(OHLCV_CACHE_FILE) < cache_ttl_sec)
        )
        
        # Smart auto-refresh during market hours
        smart_refresh = MarketHours.should_auto_refresh(OHLCV_CACHE_FILE, self.cache_ttl_hours)
        if smart_refresh and not self.force_refresh:
            market_status = MarketHours.get_market_status_string()
            print(f"Smart Auto-Refresh: Market is {market_status}, refreshing data...")
            cache_valid = False
        
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
            
            # Strict liquidity check - must meet minimum requirements
            if dollar_vol_avg20 < MIN_AVG_DOLLAR_VOLUME: 
                return None, "Low Price/Liquidity", None
            if vol_avg20 < MIN_AVG_VOLUME:
                return None, "Low Price/Liquidity", None

            atr = self.calculate_atr(df)
            if atr <= 0 or atr / c > 0.10:  # Stricter volatility filter
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
            
            # --- POSITION SIZING: Multiple constraints ---
            # 1. Volatility adjustment
            atr_pct = atr / c if c > 0 else 0.02
            vol_scalar = min(1.0, 0.025 / (atr_pct + 1e-9))
            
            # 2. Risk-based sizing
            risk_amt = min(self.risk_per_trade * vol_scalar, self.account_size * MAX_RISK_PCT_PER_TRADE / 100)
            risk_per_share = trigger - stop
            shares_by_risk = max(1, int(risk_amt / risk_per_share)) if risk_per_share > 0 else 0
            
            # 3. Liquidity-based limit (never more than 1% of daily volume)
            max_shares_by_liquidity = DataValidator.max_position_size(vol_avg20, c, MAX_POSITION_PCT_OF_VOLUME)
            
            # Take the minimum of risk-based and liquidity-based
            shares = min(shares_by_risk, max_shares_by_liquidity) if max_shares_by_liquidity > 0 else shares_by_risk
            
            # 4. Calculate realistic slippage for this position
            is_breakout = strategy_name.startswith("BREAKOUT")
            expected_slippage = DataValidator.calculate_realistic_slippage(shares, vol_avg20, is_breakout)
            
            # If slippage is too high (>1%), reduce position or warn
            if expected_slippage > 0.01:
                # Reduce position to keep slippage under 1%
                target_pct_of_volume = 0.5  # Aim for 0.5% of volume to reduce slippage
                shares = min(shares, int(vol_avg20 * target_pct_of_volume / 100))
            
            # --- SUPER POWER 1: SECTOR & EARNINGS CHECK (with caching) ---
            # Use cached sector mapping for speed
            sector = SectorMapper.get_sector(t)
            
            # Use cached earnings calendar
            earnings_call = "Unknown"
            is_blackout, blackout_reason = EarningsCalendar.is_in_blackout(
                t, self.earnings_blackout_days, self.earnings_post_days
            )
            
            earnings_date_obj, days_to = EarningsCalendar.get_earnings_date(t)
            if earnings_date_obj and days_to is not None:
                earnings_call = f"{earnings_date_obj.date().isoformat() if hasattr(earnings_date_obj, 'date') else str(earnings_date_obj)[:10]} ({days_to:+d}d)"
            
            if is_blackout:
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
            
            # Valuation check (optional - only if we already have info)
            try:
                ticker_obj = yf.Ticker(t)
                t_info = ticker_obj.info or {}
                fwd_pe = t_info.get('forwardPE', 0)
                if fwd_pe and fwd_pe < 40: 
                    score += 5  # Value boost
            except: 
                pass
            
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
            
            # === STATISTICAL CONFIDENCE ANALYSIS ===
            # Get walk-forward pass rate for confidence calculation
            wf_pass_rate = None
            oos_pf_val = None
            if self.wf_results:
                wf = self.wf_results.get(t, {})
                wf_m = _wf_metrics_for_strategy(wf, strategy_name)
                wf_pass_rate = wf_m.get("pass_rate")
                oos_m = _oos_metrics_for_strategy(wf, strategy_name)
                oos_pf_val = oos_m.get("pf")
            
            # Calculate statistical confidence
            trades_list = final_res.get('trades_list', [])
            stat_conf = StatisticalConfidenceScorer.calculate_confidence(
                trades=final_res['trades'],
                win_rate=final_res['win_rate'],
                profit_factor=final_res['pf'],
                expectancy=final_res.get('expectancy', 0),
                wf_pass_rate=wf_pass_rate,
                oos_pf=oos_pf_val,
            )
            confidence_score = stat_conf['score']
            confidence_grade = stat_conf['grade']
            
            # Calculate t-statistic for statistical significance
            t_stat = StatisticalConfidenceScorer.calculate_t_statistic(trades_list)
            
            # Analyze trend quality
            trend_analysis = TREND_ANALYZER.analyze(df, final_res)
            trend_grade = trend_analysis['trend_grade']
            
            # Build note string
            note_str = f"N={final_res['trades']}"
            if regime_score is not None:
                note_str += f" | Reg:{regime_score:.2f}"
            note_str += f" | Conf:{confidence_grade}"
            note_str += f" | Trend:{trend_grade}"
            if t_stat >= 2.0:
                note_str += " | SIG"  # Statistically significant
            if rr_ratio >= 3.0:
                note_str += " | 3R+"
            if not stat_conf['tradeable']:
                note_str += " | WEAK"  # Not meeting minimum statistical bar
            
            # Adjust score based on statistical confidence
            score = final_res['win_rate'] + (final_res['pf'] * 10)
            if mkt_status == "BULL":
                score += 10
            # Bonus for statistical significance
            if t_stat >= 2.0:
                score += 10
            if confidence_grade in ['A', 'B']:
                score += 5
            
            return TitanSetup(
                t, strategy_name, c, trigger, stop, target, shares,
                final_res['win_rate'], final_res['pf'], kelly*100, score, sector, 
                earnings_call, note_str, confidence_score, confidence_grade, 
                trend_grade, t_stat
            ), "Passed", None

        except Exception:
            LOGGER.exception("Ticker processing failed: %s", t)
            return None, "Error", None


    def scan(self, max_workers=10, near_miss_report=False, near_miss_top=DEFAULT_NEAR_MISS_TOP):
        tickers, data = self.get_data()
        spy_close = None
        spy_df = None
        if isinstance(data.columns, pd.MultiIndex) and "SPY" in data.columns.levels[0]:
            spy_df = data["SPY"].dropna()
            if "Close" in spy_df:
                spy_close = spy_df["Close"]
        
        # =====================================================================
        # DATA FRESHNESS CHECK - Critical for real trading
        # =====================================================================
        if spy_df is not None:
            is_fresh, age, freshness_msg = DataValidator.check_data_freshness(spy_df)
            
            print(f"\n{'='*60}")
            print(f"  DATA STATUS")
            print(f"{'='*60}")
            
            if is_fresh:
                print(f"  [OK] {freshness_msg}")
            else:
                print(f"  [WARNING] {freshness_msg}")
                print(f"  ")
                print(f"  !!! DATA IS STALE - NOT SUITABLE FOR LIVE TRADING !!!")
                print(f"  Last data: {spy_df.index[-1]}")
                print(f"  ")
                print(f"  Options:")
                print(f"    1. Run with --force-refresh-cache to download fresh data")
                print(f"    2. For real trading, use a real-time data source")
                print(f"  ")
                
                # For production, you might want to abort here
                # return [], {"Stale Data": 1}, []
        
        # 1. Check Market
        regime = MarketRegime(data)
        mkt_status, mkt_score, vix_level = regime.analyze_spy()
        
        print(f"\n{'='*60}")
        print(f"  MARKET STATUS: {mkt_status}")
        print(f"  Market Score: {mkt_score:.2f}")
        if vix_level is not None:
            print(f"  VIX Level: {vix_level:.1f}")
        print(f"{'='*60}")
        
        # VIX PANIC CHECK - NO NEW POSITIONS
        if vix_level is not None and vix_level > VIX_PANIC_THRESHOLD:
            print(f"\n!!! VIX PANIC ({vix_level:.1f} > {VIX_PANIC_THRESHOLD}) - NO NEW POSITIONS ALLOWED !!!")
            print("Wait for VIX to settle below panic threshold before taking new trades.")
            return [], {"VIX Panic": 1}, []
        
        if mkt_score == 0:
            print("\n!!! MARKET IS IN BEAR TREND. NO NEW LONG POSITIONS RECOMMENDED. !!!")
        
        print("\nScanning & Validating (Production Mode)...")
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


# =============================================================================
# PAPER TRADE TRACKER - Track performance before going live
# =============================================================================
AUTO_TRACK_TOP_N = 3  # Automatically track top N signals


# =============================================================================
# AUTOMATIC SIGNAL TRACKER - Tracks signals without manual intervention
# =============================================================================
class SignalTracker:
    """
    Automatically tracks signals and their outcomes.
    No manual intervention needed - just run the scanner.
    """
    
    def __init__(self, file_path=SIGNAL_TRACKER_FILE):
        self.file_path = file_path
        self._load()
    
    def _load(self):
        """Load tracking data."""
        self.data = {
            "active_signals": {},
            "completed_signals": [],
            "stats": {
                "total_signals": 0,
                "wins": 0,
                "losses": 0,
                "total_pnl_pct": 0.0,
                "started": None,
            }
        }
        if os.path.exists(self.file_path):
            try:
                with open(self.file_path, "r") as f:
                    self.data = json.load(f)
            except Exception:
                pass
    
    def _save(self):
        """Save tracking data."""
        try:
            with open(self.file_path, "w") as f:
                json.dump(self.data, f, indent=2, default=str)
        except Exception as e:
            print(f"Warning: Could not save signal tracker: {e}")
    
    def add_signal(self, setup, current_price):
        """Add a new signal to track."""
        if self.data["stats"]["started"] is None:
            self.data["stats"]["started"] = datetime.now().isoformat()
        
        ticker = setup.ticker
        
        if ticker in self.data["active_signals"]:
            existing = self.data["active_signals"][ticker]
            existing["last_seen"] = datetime.now().isoformat()
            existing["current_price"] = current_price
            first_seen = datetime.fromisoformat(existing["first_seen"])
            existing["days_tracked"] = (datetime.now() - first_seen).days
            self._save()
            return "updated"
        
        signal = {
            "ticker": ticker,
            "strategy": setup.strategy,
            "first_seen": datetime.now().isoformat(),
            "last_seen": datetime.now().isoformat(),
            "entry_price": setup.trigger,
            "current_price": current_price,
            "stop": setup.stop,
            "target": setup.target,
            "win_rate": setup.win_rate,
            "profit_factor": setup.profit_factor,
            "t_statistic": setup.t_statistic,
            "confidence_grade": setup.confidence_grade,
            "status": "WATCHING",
            "triggered_price": None,
            "triggered_date": None,
            "days_tracked": 0,
        }
        
        self.data["active_signals"][ticker] = signal
        self._save()
        return "added"
    
    def update_prices(self, price_dict):
        """Update current prices and check for exits."""
        now = datetime.now()
        
        for ticker, signal in list(self.data["active_signals"].items()):
            if ticker not in price_dict:
                continue
            
            current_price = price_dict[ticker]
            signal["current_price"] = current_price
            signal["last_seen"] = now.isoformat()
            
            if signal["status"] == "WATCHING":
                if current_price >= signal["entry_price"] * 0.995:
                    signal["status"] = "TRIGGERED"
                    signal["triggered_price"] = current_price
                    signal["triggered_date"] = now.isoformat()
            
            if signal["status"] == "TRIGGERED":
                entry = signal["triggered_price"]
                
                if current_price <= signal["stop"]:
                    pnl_pct = (current_price - entry) / entry * 100
                    self._close_signal(ticker, current_price, "STOP", pnl_pct)
                elif current_price >= signal["target"]:
                    pnl_pct = (current_price - entry) / entry * 100
                    self._close_signal(ticker, current_price, "TARGET", pnl_pct)
            
            first_seen = datetime.fromisoformat(signal["first_seen"])
            days_watching = (now - first_seen).days
            signal["days_tracked"] = days_watching
            
            if signal["status"] == "WATCHING" and days_watching > 10:
                self._close_signal(ticker, current_price, "EXPIRED", 0)
        
        self._save()
    
    def _close_signal(self, ticker, exit_price, reason, pnl_pct):
        """Close a signal and record result."""
        if ticker not in self.data["active_signals"]:
            return
        
        signal = self.data["active_signals"].pop(ticker)
        
        # Send notification
        entry = signal.get("triggered_price") or signal.get("entry_price")
        if reason == "STOP":
            NotificationManager.notify_stop_hit(ticker, entry, exit_price, pnl_pct)
        elif reason == "TARGET":
            NotificationManager.notify_target_hit(ticker, entry, exit_price, pnl_pct)
        signal["exit_price"] = exit_price
        signal["exit_date"] = datetime.now().isoformat()
        signal["exit_reason"] = reason
        signal["pnl_pct"] = pnl_pct
        signal["status"] = "CLOSED"
        
        self.data["completed_signals"].append(signal)
        
        if reason in ["STOP", "TARGET"]:
            self.data["stats"]["total_signals"] += 1
            self.data["stats"]["total_pnl_pct"] += pnl_pct
            if pnl_pct > 0:
                self.data["stats"]["wins"] += 1
            else:
                self.data["stats"]["losses"] += 1
        
        if len(self.data["completed_signals"]) > 100:
            self.data["completed_signals"] = self.data["completed_signals"][-100:]
        
        self._save()
    
    def get_active_signals(self):
        """Get list of active signals."""
        return self.data["active_signals"]
    
    def print_status(self):
        """Print tracking status."""
        active = self.data["active_signals"]
        stats = self.data["stats"]
        
        print("\n" + "="*70)
        print("  AUTO-TRACKED SIGNALS")
        print("="*70)
        
        if not active:
            print("  No signals currently being tracked.")
            print("  (Top 3 signals are auto-tracked each run)")
        else:
            print(f"  {'Ticker':<8} {'Status':<10} {'Entry':>10} {'Current':>10} {'P&L':>8} {'Days':>5} {'Grade'}")
            print("-"*70)
            
            for ticker, sig in active.items():
                entry = sig.get("triggered_price") or sig["entry_price"]
                current = sig.get("current_price", entry)
                if sig["status"] == "TRIGGERED":
                    pnl = (current - entry) / entry * 100
                    pnl_str = f"{pnl:+.1f}%"
                else:
                    pnl_str = "-"
                
                grade = sig.get("confidence_grade", "?")
                print(f"  {ticker:<8} {sig['status']:<10} ${entry:>9.2f} ${current:>9.2f} {pnl_str:>8} {sig['days_tracked']:>5}d   {grade}")
        
        print("-"*70)
        
        total = stats["total_signals"]
        if total > 0:
            win_rate = stats["wins"] / total * 100
            avg_pnl = stats["total_pnl_pct"] / total
            print(f"  Completed: {total} trades | {win_rate:.0f}% win rate | {avg_pnl:+.2f}% avg")
        
        print("="*70)


# =============================================================================
# DATA VALIDATOR - Ensure data quality before trading
# =============================================================================
class DataValidator:
    """
    Validates data quality before making trading decisions.
    Bad data = bad trades = lost money.
    """
    
    @staticmethod
    def check_data_freshness(df, max_age_minutes=MAX_DATA_AGE_MINUTES):
        """
        Check if data is fresh enough for trading.
        Returns: (is_fresh: bool, age_minutes: float, warning: str)
        """
        if df is None or df.empty:
            return False, float('inf'), "No data available"
        
        try:
            last_date = df.index[-1]
            if isinstance(last_date, str):
                last_date = pd.to_datetime(last_date)
            
            now = datetime.now()
            
            # For daily data, check if it's from today (or yesterday if before market open)
            if hasattr(last_date, 'date'):
                data_date = last_date.date() if hasattr(last_date, 'date') else last_date
            else:
                data_date = pd.to_datetime(last_date).date()
            
            today = now.date()
            yesterday = (now - timedelta(days=1)).date()
            
            # Market hours check (9:30 AM - 4:00 PM ET)
            market_open = now.replace(hour=9, minute=30, second=0)
            
            if data_date == today:
                return True, 0, "Data is from today"
            elif data_date == yesterday:
                # Acceptable if it's before market open
                if now < market_open:
                    return True, 0, "Data is from yesterday (pre-market)"
                else:
                    age = (now - datetime.combine(yesterday, datetime.min.time())).total_seconds() / 60
                    return age < max_age_minutes * 60, age, f"Data is from yesterday"
            else:
                days_old = (today - data_date).days
                return False, days_old * 24 * 60, f"Data is {days_old} days old - TOO STALE"
        except Exception as e:
            return False, float('inf'), f"Could not verify data freshness: {e}"
    
    @staticmethod
    def check_for_splits_dividends(df, lookback=5):
        """
        Check for potential stock splits or large dividends that could affect analysis.
        Large overnight gaps (>20%) often indicate corporate actions.
        """
        warnings = []
        
        if df is None or len(df) < lookback + 1:
            return warnings
        
        try:
            opens = df['Open'].iloc[-lookback:]
            prev_closes = df['Close'].shift(1).iloc[-lookback:]
            
            overnight_changes = (opens - prev_closes) / prev_closes
            
            for i, change in enumerate(overnight_changes):
                if abs(change) > 0.20:  # 20%+ overnight change
                    date_idx = df.index[-lookback + i]
                    warnings.append(f"Large overnight move ({change*100:.1f}%) on {date_idx} - possible split/dividend")
        except Exception:
            pass
        
        return warnings
    
    @staticmethod
    def check_liquidity(df, min_dollar_vol=MIN_AVG_DOLLAR_VOLUME, min_volume=MIN_AVG_VOLUME):
        """
        Check if stock has sufficient liquidity for trading.
        Returns: (is_liquid: bool, avg_dollar_vol: float, warning: str)
        """
        if df is None or len(df) < 20:
            return False, 0, "Insufficient data for liquidity check"
        
        try:
            avg_volume = df['Volume'].iloc[-20:].mean()
            avg_price = df['Close'].iloc[-20:].mean()
            avg_dollar_vol = avg_volume * avg_price
            
            if avg_dollar_vol < min_dollar_vol:
                return False, avg_dollar_vol, f"Avg dollar volume ${avg_dollar_vol/1e6:.1f}M < ${min_dollar_vol/1e6:.1f}M required"
            
            if avg_volume < min_volume:
                return False, avg_dollar_vol, f"Avg volume {avg_volume/1000:.0f}K < {min_volume/1000:.0f}K required"
            
            return True, avg_dollar_vol, "Liquidity OK"
        except Exception as e:
            return False, 0, f"Liquidity check failed: {e}"
    
    @staticmethod
    def calculate_realistic_slippage(shares, avg_volume, is_breakout=True):
        """
        Calculate realistic slippage based on order size relative to volume.
        Larger orders relative to volume = more slippage.
        """
        if avg_volume <= 0:
            return 0.01  # 1% default for unknown liquidity
        
        # What percentage of daily volume is our order?
        pct_of_volume = (shares / avg_volume) * 100
        
        # Base slippage
        base_bps = BASE_SLIPPAGE_BREAKOUT_BPS if is_breakout else BASE_SLIPPAGE_DIP_BPS
        
        # Volume impact (additional slippage for larger orders)
        volume_impact_bps = pct_of_volume * VOLUME_IMPACT_FACTOR
        
        total_slippage_bps = base_bps + volume_impact_bps
        
        # Cap at 2% (beyond this, the trade is probably not worth it)
        return min(total_slippage_bps / 10000, 0.02)
    
    @staticmethod
    def max_position_size(avg_volume, price, max_pct_of_volume=MAX_POSITION_PCT_OF_VOLUME):
        """
        Calculate maximum position size based on liquidity.
        Never be more than X% of daily volume.
        """
        if avg_volume <= 0 or price <= 0:
            return 0
        
        max_shares = int(avg_volume * (max_pct_of_volume / 100))
        return max_shares


# =============================================================================
# PRE-TRADE CHECKLIST - Professional traders use these
# =============================================================================
class PreTradeChecklist:
    """
    Systematic checklist before taking any trade.
    Professionals never skip this.
    """
    
    def __init__(self, setup, df, risk_manager, data_validator=DataValidator):
        self.setup = setup
        self.df = df
        self.risk_mgr = risk_manager
        self.validator = data_validator
        self.checks = []
        self.passed = True
        self.warnings = []
    
    def run_all_checks(self):
        """Run complete pre-trade checklist."""
        self.checks = []
        self.warnings = []
        self.passed = True
        
        # 1. Data Quality Checks
        self._check_data_freshness()
        self._check_data_quality()
        
        # 2. Liquidity Checks
        self._check_liquidity()
        self._check_position_size_vs_volume()
        
        # 3. Risk Checks
        self._check_portfolio_risk()
        self._check_position_risk()
        
        # 4. Setup Quality Checks
        self._check_statistical_significance()
        self._check_risk_reward()
        
        # 5. Market Condition Checks
        self._check_earnings_proximity()
        self._check_market_hours()
        
        return self.passed, self.checks, self.warnings
    
    def _add_check(self, name, passed, detail):
        status = "PASS" if passed else "FAIL"
        self.checks.append({"name": name, "status": status, "detail": detail})
        if not passed:
            self.passed = False
    
    def _add_warning(self, warning):
        self.warnings.append(warning)
    
    def _check_data_freshness(self):
        is_fresh, age, msg = self.validator.check_data_freshness(self.df)
        self._add_check("Data Freshness", is_fresh, msg)
        if not is_fresh:
            self._add_warning(f"STALE DATA: {msg}")
    
    def _check_data_quality(self):
        split_warnings = self.validator.check_for_splits_dividends(self.df)
        passed = len(split_warnings) == 0
        detail = "No corporate actions detected" if passed else f"{len(split_warnings)} potential issues"
        self._add_check("Data Quality", passed, detail)
        for w in split_warnings:
            self._add_warning(w)
    
    def _check_liquidity(self):
        is_liquid, dollar_vol, msg = self.validator.check_liquidity(self.df)
        self._add_check("Liquidity", is_liquid, f"${dollar_vol/1e6:.1f}M avg daily volume")
        if not is_liquid:
            self._add_warning(f"LOW LIQUIDITY: {msg}")
    
    def _check_position_size_vs_volume(self):
        try:
            avg_vol = self.df['Volume'].iloc[-20:].mean()
            pct_of_vol = (self.setup.qty / avg_vol) * 100 if avg_vol > 0 else 100
            passed = pct_of_vol <= MAX_POSITION_PCT_OF_VOLUME
            self._add_check("Size vs Volume", passed, f"{pct_of_vol:.2f}% of daily volume")
            if not passed:
                self._add_warning(f"Position is {pct_of_vol:.1f}% of daily volume - reduce size")
        except Exception:
            self._add_check("Size vs Volume", False, "Could not calculate")
    
    def _check_portfolio_risk(self):
        can_trade, reason = self.risk_mgr.can_take_new_trade()
        self._add_check("Portfolio Risk", can_trade, reason)
        if not can_trade:
            self._add_warning(f"RISK LIMIT: {reason}")
    
    def _check_position_risk(self):
        try:
            risk_amt = (self.setup.trigger - self.setup.stop) * self.setup.qty
            risk_pct = (risk_amt / self.risk_mgr.account_size) * 100
            passed = risk_pct <= MAX_RISK_PCT_PER_TRADE
            self._add_check("Position Risk", passed, f"{risk_pct:.2f}% of account")
            if not passed:
                self._add_warning(f"Risk {risk_pct:.1f}% exceeds {MAX_RISK_PCT_PER_TRADE}% limit")
        except Exception:
            self._add_check("Position Risk", False, "Could not calculate")
    
    def _check_statistical_significance(self):
        passed = self.setup.t_statistic >= 2.0
        self._add_check("Statistical Significance", passed, f"t-stat: {self.setup.t_statistic:.2f}")
        if not passed:
            self._add_warning(f"Edge not statistically significant (t={self.setup.t_statistic:.2f} < 2.0)")
    
    def _check_risk_reward(self):
        try:
            rr = (self.setup.target - self.setup.trigger) / (self.setup.trigger - self.setup.stop)
            passed = rr >= 1.5
            self._add_check("Risk/Reward", passed, f"{rr:.2f}:1")
        except Exception:
            self._add_check("Risk/Reward", False, "Could not calculate")
    
    def _check_earnings_proximity(self):
        # Use EarningsCalendar API
        is_blackout, reason = EarningsCalendar.is_in_blackout(self.setup.ticker)
        earnings_date, days_until = EarningsCalendar.get_earnings_date(self.setup.ticker)
        
        if is_blackout:
            self._add_check("Earnings Check", False, reason)
            self._add_warning(f"EARNINGS BLACKOUT: {reason}")
        elif earnings_date:
            self._add_check("Earnings Check", True, f"Earnings: {days_until}d away")
        else:
            self._add_check("Earnings Check", True, "Earnings date not found - verify manually")
            self._add_warning("Could not fetch earnings date - verify before trading")
    
    def _check_market_hours(self):
        # Use MarketHours utility
        market_status = MarketHours.get_market_status_string()
        is_open = MarketHours.is_market_open()
        
        if is_open:
            self._add_check("Market Hours", True, f"Market is {market_status}")
        else:
            self._add_check("Market Hours", True, f"Market is {market_status}")
            if market_status == "CLOSED":
                time_until = MarketHours.time_until_market_open()
                hours_until = time_until.total_seconds() / 3600
                self._add_warning(f"Market closed - opens in {hours_until:.1f} hours")
    
    def print_checklist(self):
        """Print formatted checklist."""
        print("\n" + "="*60)
        print("  PRE-TRADE CHECKLIST")
        print(f"  {self.setup.ticker} - {self.setup.strategy}")
        print("="*60)
        
        for check in self.checks:
            status_icon = "[OK]" if check["status"] == "PASS" else "[X]"
            print(f"  {status_icon} {check['name']}: {check['detail']}")
        
        if self.warnings:
            print("\n  WARNINGS:")
            for w in self.warnings:
                print(f"    ! {w}")
        
        print("-"*60)
        if self.passed:
            print("  RESULT: ALL CHECKS PASSED - Trade may proceed")
        else:
            print("  RESULT: CHECKS FAILED - DO NOT TRADE")
        print("="*60)


class PaperTradeTracker:
    """
    Track paper trades to validate system before live trading.
    Recommended: 3-6 months of paper trading before using real money.
    """
    
    def __init__(self, file_path=PAPER_TRADE_FILE):
        self.file_path = file_path
        self._load()
    
    def _load(self):
        """Load paper trade history."""
        self.data = {
            "open_trades": {},
            "closed_trades": [],
            "stats": {
                "total_trades": 0,
                "wins": 0,
                "losses": 0,
                "total_pnl": 0.0,
                "started": None,
            }
        }
        if os.path.exists(self.file_path):
            try:
                with open(self.file_path, "r") as f:
                    self.data = json.load(f)
            except Exception:
                pass
    
    def _save(self):
        """Save paper trade history."""
        try:
            with open(self.file_path, "w") as f:
                json.dump(self.data, f, indent=2, default=str)
        except Exception as e:
            print(f"Warning: Could not save paper trades: {e}")
    
    def add_trade(self, setup: TitanSetup):
        """Add a new paper trade."""
        if self.data["stats"]["started"] is None:
            self.data["stats"]["started"] = datetime.now().isoformat()
        
        trade = {
            "ticker": setup.ticker,
            "strategy": setup.strategy,
            "entry_date": datetime.now().isoformat(),
            "entry_price": setup.trigger,
            "shares": setup.qty,
            "stop": setup.stop,
            "target": setup.target,
            "confidence_grade": setup.confidence_grade,
            "t_statistic": setup.t_statistic,
            "status": "OPEN",
        }
        
        self.data["open_trades"][setup.ticker] = trade
        self._save()
        print(f"  Paper trade added: {setup.ticker}")
    
    def close_trade(self, ticker: str, exit_price: float, reason: str = "manual"):
        """Close a paper trade and record result."""
        if ticker not in self.data["open_trades"]:
            print(f"  Error: {ticker} not in open paper trades")
            return
        
        trade = self.data["open_trades"].pop(ticker)
        trade["exit_date"] = datetime.now().isoformat()
        trade["exit_price"] = exit_price
        trade["exit_reason"] = reason
        trade["status"] = "CLOSED"
        
        # Calculate P&L
        pnl_per_share = exit_price - trade["entry_price"]
        pnl_total = pnl_per_share * trade["shares"]
        pnl_pct = (pnl_per_share / trade["entry_price"]) * 100
        
        trade["pnl_total"] = round(pnl_total, 2)
        trade["pnl_pct"] = round(pnl_pct, 2)
        
        # Update stats
        self.data["stats"]["total_trades"] += 1
        self.data["stats"]["total_pnl"] += pnl_total
        if pnl_total > 0:
            self.data["stats"]["wins"] += 1
        else:
            self.data["stats"]["losses"] += 1
        
        self.data["closed_trades"].append(trade)
        self._save()
        
        result = "WIN" if pnl_total > 0 else "LOSS"
        print(f"  Closed {ticker}: {result} ${pnl_total:.2f} ({pnl_pct:.1f}%)")
    
    def get_stats(self) -> dict:
        """Get paper trading statistics."""
        stats = self.data["stats"]
        total = stats["total_trades"]
        
        if total == 0:
            return {
                "total_trades": 0,
                "win_rate": 0,
                "total_pnl": 0,
                "avg_pnl": 0,
                "open_trades": len(self.data["open_trades"]),
            }
        
        return {
            "total_trades": total,
            "wins": stats["wins"],
            "losses": stats["losses"],
            "win_rate": (stats["wins"] / total) * 100 if total > 0 else 0,
            "total_pnl": stats["total_pnl"],
            "avg_pnl": stats["total_pnl"] / total if total > 0 else 0,
            "open_trades": len(self.data["open_trades"]),
            "started": stats["started"],
        }
    
    def print_summary(self):
        """Print paper trading summary."""
        stats = self.get_stats()
        
        print("\n" + "="*50)
        print("  PAPER TRADE SUMMARY")
        print("="*50)
        
        if stats["total_trades"] == 0:
            print("  No closed paper trades yet.")
            print(f"  Open paper trades: {stats['open_trades']}")
        else:
            print(f"  Total Trades: {stats['total_trades']}")
            print(f"  Win Rate: {stats['win_rate']:.1f}%")
            print(f"  Total P&L: ${stats['total_pnl']:.2f}")
            print(f"  Avg P&L per Trade: ${stats['avg_pnl']:.2f}")
            print(f"  Open Trades: {stats['open_trades']}")
            if stats.get('started'):
                print(f"  Tracking Since: {stats['started'][:10]}")
        
        print("="*50)
        
        # Check if ready for live trading
        if stats['total_trades'] >= 30 and stats['win_rate'] >= 50:
            print("  STATUS: May be ready for small live positions")
        elif stats['total_trades'] >= 20:
            print("  STATUS: Continue paper trading, need 30+ trades")
        else:
            print("  STATUS: Keep paper trading (need 30+ trades minimum)")


def main():
    # ==========================================================================
    # AUTO MODE DETECTION - Just run the file, everything works!
    # ==========================================================================
    # If no arguments provided, enable AUTO MODE (Trust Mode + auto settings)
    import sys
    no_args_provided = len(sys.argv) == 1
    
    if no_args_provided and AUTO_MODE_ENABLED:
        # Initialize auto mode manager
        auto_manager = AutoModeManager()
        
        # First-time setup wizard
        if auto_manager.is_first_run():
            auto_config = auto_manager.run_first_time_setup()
        else:
            auto_config = auto_manager.get_config()
        
        # Inject auto mode settings into sys.argv
        sys.argv.extend([
            "--trust-mode",
            "--account-size", str(auto_config.get("account_size", ACCOUNT_SIZE)),
            "--risk-per-trade", str(auto_config.get("risk_per_trade", RISK_PER_TRADE)),
        ])
        
        # Handle paper trading bypass
        if auto_config.get("paper_trading_bypassed", False):
            # Create trust manager and bypass
            temp_trust_mgr = TrustModeManager(account_size=auto_config.get("account_size", ACCOUNT_SIZE))
            temp_trust_mgr.state["paper_validated"] = True
            temp_trust_mgr._save_state()
    
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
        "--trust-mode",
        "--no-trust-mode",
        dest="trust_mode",
        action=argparse.BooleanOptionalAction,
        default=TRUST_MODE_DEFAULT,
        help="TRUST MODE: Extremely strict filters. Only Grade A/B signals. Simple TRADE/DON'T output.",
    )
    parser.add_argument(
        "--trust-paper",
        dest="trust_paper",
        action="store_true",
        help="Start/continue Trust Mode paper trading validation period.",
    )
    parser.add_argument(
        "--trust-bypass",
        dest="trust_bypass",
        action="store_true",
        help="Bypass paper trading requirement (for experienced traders only).",
    )
    parser.add_argument(
        "--trust-status",
        dest="trust_status",
        action="store_true",
        help="Show Trust Mode status and exit.",
    )
    parser.add_argument(
        "--trust-reset",
        dest="trust_reset",
        action="store_true",
        help="Reset Trust Mode paper trading (start over).",
    )
    parser.add_argument(
        "--trust-paper-win",
        dest="trust_paper_win",
        action="store_true",
        help="Record a paper trade WIN for Trust Mode validation.",
    )
    parser.add_argument(
        "--trust-paper-loss",
        dest="trust_paper_loss",
        action="store_true",
        help="Record a paper trade LOSS for Trust Mode validation.",
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
    parser.add_argument(
        "--paper-trade",
        dest="paper_trade",
        action="store_true",
        help="Enable paper trade mode (tracks simulated trades).",
    )
    parser.add_argument(
        "--paper-stats",
        dest="paper_stats",
        action="store_true",
        help="Show paper trading statistics and exit.",
    )
    parser.add_argument(
        "--schedule",
        dest="schedule_mode",
        action="store_true",
        help="Run in scheduled mode (scans at market open and close).",
    )
    parser.add_argument(
        "--dashboard",
        dest="show_dashboard",
        action="store_true",
        help="Generate and display performance dashboard.",
    )
    parser.add_argument(
        "--stats",
        dest="show_stats",
        action="store_true",
        help="Show performance statistics and exit.",
    )

    config = _load_config(pre_args.config)
    if config:
        valid_keys = {a.dest for a in parser._actions}
        parser.set_defaults(**{k: v for k, v in config.items() if k in valid_keys})
    args = parser.parse_args(remaining)
    args = _apply_safe_mode(args)
    args = _apply_trust_mode(args)
    
    # Initialize Trust Mode Manager
    trust_manager = TrustModeManager(account_size=args.account_size)
    
    # Handle Trust Mode special commands
    if getattr(args, 'trust_status', False):
        print_trust_mode_header()
        status = trust_manager.get_status_report()
        print("\n  TRUST MODE STATUS")
        print("  " + "=" * 50)
        print(f"  Paper Trading Validated: {'YES' if status['paper_validated'] else 'NO'}")
        print(f"  Trades Today: {status['trades_today']}/{status['max_daily']}")
        print(f"  Trades This Week: {status['trades_this_week']}/{status['max_weekly']}")
        print(f"  Current Positions: {status['current_positions']}/{status['max_positions']}")
        print(f"  Consecutive Losses: {status['consecutive_losses']}")
        if status['cooloff_until']:
            print(f"  Cooloff Until: {status['cooloff_until']}")
        print(f"  Total Trades: {status['total_trades']}")
        print(f"  Win Rate: {status['win_rate']:.1f}%")
        print("  " + "=" * 50)
        return
    
    if getattr(args, 'trust_reset', False):
        trust_manager.reset_paper_trading()
        print("  Trust Mode paper trading has been reset.")
        print("  Run with --trust-paper to start validation period.")
        return
    
    if getattr(args, 'trust_bypass', False):
        print("\n  WARNING: You are bypassing paper trading validation.")
        print("  This means you will trade with REAL money without proving")
        print("  that you can follow the system successfully.")
        print("\n  Type exactly: I ACCEPT THE RISK")
        confirm = input("  > ").strip()
        if trust_manager.bypass_paper_validation(confirm):
            print("  Paper validation bypassed. You may now use Trust Mode.")
        else:
            print("  Bypass cancelled. Run with --trust-paper to validate properly.")
        return
    
    if getattr(args, 'trust_paper_win', False):
        trust_manager.record_paper_trade(won=True)
        print("  Recorded paper trade WIN!")
        validated, msg = trust_manager.is_paper_trading_validated()
        print(f"  Status: {msg}")
        return
    
    if getattr(args, 'trust_paper_loss', False):
        trust_manager.record_paper_trade(won=False)
        print("  Recorded paper trade LOSS.")
        validated, msg = trust_manager.is_paper_trading_validated()
        print(f"  Status: {msg}")
        return

    output_paths = _resolve_output_paths(args.output_dir, save_history=not getattr(args, 'no_history', False))
    log_file = None if args.no_log_file else (args.log_file or output_paths["log_file"])
    global LOGGER
    LOGGER = setup_logging(args.log_level, log_file=log_file)
    
    # Paper trade stats mode
    if getattr(args, 'paper_stats', False):
        tracker = PaperTradeTracker()
        tracker.print_summary()
        return
    
    # Performance stats mode
    if getattr(args, 'show_stats', False):
        PerformanceDashboard.print_stats()
        return
    
    # Dashboard mode
    if getattr(args, 'show_dashboard', False):
        print("\n  Generating Performance Dashboard...")
        PerformanceDashboard.print_stats()
        if PerformanceDashboard.generate():
            # Try to open the dashboard image
            if sys.platform == 'win32':
                try:
                    os.startfile(DASHBOARD_FILE)
                except Exception:
                    pass
        return
    
    # Scheduled scanning mode
    if getattr(args, 'schedule_mode', False):
        def run_scan():
            # Import main without schedule flag to avoid recursion
            import sys
            original_argv = sys.argv.copy()
            # Remove --schedule from args
            sys.argv = [a for a in sys.argv if a != '--schedule']
            try:
                main()
            except Exception as e:
                print(f"Scan error: {e}")
            finally:
                sys.argv = original_argv
        
        print("\n" + "="*60)
        print("  SCHEDULED SCANNING MODE")
        print("="*60)
        print(f"  Market Status: {MarketHours.get_market_status_string()}")
        print(f"  Scans will run at:")
        print(f"    - 9:35 AM ET (after market open)")
        print(f"    - 3:55 PM ET (before market close)")
        print("  Press Ctrl+C to stop.")
        print("="*60)
        
        # Run initial scan
        print("\n  Running initial scan...")
        run_scan()
        
        # Enter scheduler loop
        last_scan_date = None
        scanned_open = False
        scanned_close = False
        
        try:
            while True:
                now = MarketHours.get_eastern_time()
                today = now.date() if hasattr(now, 'date') else date.today()
                
                # Reset flags on new day
                if last_scan_date != today:
                    last_scan_date = today
                    scanned_open = False
                    scanned_close = False
                
                # Skip weekends
                if now.weekday() >= 5:
                    time.sleep(60)
                    continue
                
                hour = now.hour
                minute = now.minute
                
                # Market open scan (9:35 AM)
                if not scanned_open and hour == 9 and 35 <= minute < 40:
                    print(f"\n[{now.strftime('%H:%M')}] Running MARKET OPEN scan...")
                    run_scan()
                    scanned_open = True
                
                # Market close scan (3:55 PM)
                if not scanned_close and hour == 15 and 55 <= minute < 60:
                    print(f"\n[{now.strftime('%H:%M')}] Running MARKET CLOSE scan...")
                    run_scan()
                    scanned_close = True
                
                time.sleep(30)  # Check every 30 seconds
                
        except KeyboardInterrupt:
            print("\n  Scheduler stopped.")
        return

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

    # Show market status
    market_status = MarketHours.get_market_status_string()
    
    # TRUST MODE HEADER
    if getattr(args, 'trust_mode', False):
        print_trust_mode_header()
        
        # Check paper trading validation (unless running paper mode)
        if not getattr(args, 'trust_paper', False):
            validated, msg = trust_manager.is_paper_trading_validated()
            if not validated:
                print(f"\n  ⚠️  PAPER TRADING NOT VALIDATED")
                print(f"  {msg}")
                print("\n  Options:")
                print("    --trust-paper  : Start/continue paper trading validation")
                print("    --trust-bypass : Bypass validation (NOT RECOMMENDED)")
                print("    --trust-status : Check your validation progress")
                return
        
        # Trust mode specific info
        print(f"\n  Market Status: {market_status}")
        print(f"  Mode: TRUST MODE (Strictest Filters)")
        print(f"  Min Trades Required: {args.min_trades_breakout} (for statistical significance)")
        print(f"  Only Grade A/B signals shown")
        print(f"  Daily Trade Limit: {TRUST_MODE_MAX_TRADES_PER_DAY}")
        
        # Check if we can trade
        can_trade_check, reason = trust_manager.can_trade_today()
        if not can_trade_check:
            print(f"\n  ⛔ TRADING BLOCKED: {reason}")
            return
        
        status = trust_manager.get_status_report()
        print(f"\n  Trades Today: {status['trades_today']}/{status['max_daily']}")
        print(f"  Positions: {status['current_positions']}/{status['max_positions']}")
    else:
        print("\n" + "="*60)
        print("  TITAN TRADE v7.0 - PRODUCTION MODE")
        print("  Statistical Validation + Risk Management")
        print("="*60)
        print(f"  Market Status: {market_status}")
        print(f"  Min Trades Required: {args.min_trades_breakout} (statistical significance)")
        print(f"  Risk Per Trade: ${args.risk_per_trade:.0f}")
        print(f"  Max Positions: {MAX_POSITIONS}")
        print(f"  Max Drawdown Limit: {MAX_DRAWDOWN_PCT}%")
        print("="*60)
    
    # Paper trading mode for Trust Mode validation
    if getattr(args, 'trust_paper', False):
        trust_manager.start_paper_trading()
        print("\n  📝 PAPER TRADING MODE (Trust Mode Validation)")
        print(f"  Validation requires {TRUST_MODE_PAPER_VALIDATION_DAYS} days and {TRUST_MODE_PAPER_MIN_TRADES} trades")
        validated, msg = trust_manager.is_paper_trading_validated()
        print(f"  Status: {msg}")
    
    # Initialize signal tracker and show current status
    signal_tracker = SignalTracker()
    if signal_tracker.get_active_signals():
        signal_tracker.print_status()
    
    LOGGER.info("Run started - Production Mode")
    LOGGER.info("Output dir: %s", args.output_dir)
    if args.safe_mode:
        LOGGER.info("Safe mode: ENABLED (strict statistical filters)")
    
    # Initialize and check risk manager
    risk_mgr = PortfolioRiskManager(
        account_size=args.account_size,
    )
    
    # Load current portfolio for risk check
    open_positions = {}
    if os.path.exists(PORTFOLIO_FILE):
        try:
            with open(PORTFOLIO_FILE, "r") as f:
                open_positions = json.load(f)
        except Exception:
            pass
    
    risk_status = risk_mgr.get_risk_status(open_positions)
    
    print(f"\n  RISK STATUS:")
    print(f"    Current Drawdown: {risk_status['drawdown_pct']:.1f}% (max: {MAX_DRAWDOWN_PCT}%)")
    print(f"    Daily P&L: {-risk_status['daily_loss_pct']:.1f}%")
    print(f"    Portfolio Heat: {risk_status['portfolio_heat_pct']:.1f}% (max: {PORTFOLIO_HEAT_MAX}%)")
    print(f"    Position Size Scalar: {risk_status['position_size_scalar']:.0%}")
    
    if not risk_status['can_trade']:
        print(f"\n  !!! TRADING BLOCKED: {risk_status['reason']} !!!")
        print("  Wait for conditions to improve before taking new trades.")
        LOGGER.warning("Trading blocked: %s", risk_status['reason'])
        return

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

    # Get VIX level for trust mode
    vix_level_for_trust = None
    try:
        tickers_temp, data_temp = brain.get_data()
        if isinstance(data_temp.columns, pd.MultiIndex):
            for vix_key in ["^VIX", "VIX", "VIXY"]:
                if vix_key in data_temp.columns.levels[0]:
                    vix_df = data_temp[vix_key].dropna()
                    if not vix_df.empty and 'Close' in vix_df.columns:
                        vix_level_for_trust = float(vix_df['Close'].iloc[-1])
                        break
    except Exception:
        pass
    
    if setups:
        setups.sort(key=lambda x: x.score, reverse=True)
        
        # =====================================================================
        # TRUST MODE - SIMPLE OUTPUT
        # =====================================================================
        if getattr(args, 'trust_mode', False) or getattr(args, 'trust_paper', False):
            trusted_setups = print_simple_verdict(setups, trust_manager, vix_level_for_trust)
            
            # Handle paper trading mode
            if getattr(args, 'trust_paper', False) and trusted_setups:
                print("\n  📝 PAPER TRADING: Track this trade on paper")
                print("  When you would exit (stop/target hit), run:")
                print("    --trust-paper-win   (if profitable)")
                print("    --trust-paper-loss  (if stopped out)")
            
            # Save the results regardless
            rows = []
            for s in setups:
                row = {
                    "Ticker": s.ticker,
                    "Strategy": s.strategy,
                    "Price": round(s.price, 4),
                    "Trigger": round(s.trigger, 4),
                    "Stop": round(s.stop, 4),
                    "Target": round(s.target, 4),
                    "Shares": s.qty,
                    "WinRate": round(s.win_rate, 2),
                    "ProfitFactor": round(s.profit_factor, 2),
                    "ConfidenceGrade": s.confidence_grade,
                    "TrendGrade": s.trend_grade,
                    "T_Statistic": round(s.t_statistic, 2),
                    "TrustWorthy": "YES" if s.confidence_grade in ['A', 'B'] and s.t_statistic >= 2.0 else "NO",
                }
                rows.append(row)
            
            if rows:
                df_out = pd.DataFrame(rows)
                df_out.to_csv(output_paths["scan_csv_latest"], index=False)
                df_out.to_csv(output_paths["scan_csv"], index=False)
            
            # Skip the detailed output in trust mode
            # (The simple verdict is all they need)
            pass  # Trust mode output already handled above
            
        else:
            # Normal detailed output for non-trust mode
            # UI CLEAR & HEADER
            print("\n" * 3)
            print("="*70)
            print(f"  TITAN TRADE v7.0 - PRODUCTION MODE")
            print(f"  Statistical Confidence + Risk Management")
            print("="*70)
            print(f"    * 'BUY NOW' = Price has triggered")
            print(f"    * 'PENDING' = Place Buy Stop Order at Trigger Price")
            print(f"    * Conf = Statistical Confidence Grade (A/B/C/D/F)")
            print(f"    * t-stat >= 2.0 = Statistically significant edge")
            print("-" * 70)
        
            table = []
            report_rows = max(1, int(args.report_rows))
            for s in setups[:report_rows]:
                # Determine Status
                dist = (s.trigger - s.price) / s.price
                if s.price >= s.trigger:
                    status = "BUY NOW"
                elif dist < 0.01:
                    status = "NEAR"
                else:
                    status = "PENDING"
                
                # t-statistic indicator
                sig = "Y" if s.t_statistic >= 2.0 else "N"

                table.append([
                    s.ticker, 
                    s.strategy[:4], 
                    f"${s.price:.2f}",
                    f"${s.trigger:.2f}", 
                    status,
                    s.confidence_grade,
                    s.trend_grade,
                    f"{s.win_rate:.0f}%", 
                    f"{s.profit_factor:.2f}",
                    sig,
                    f"${s.stop:.2f}",
                    f"${s.target:.2f}",
                    s.qty
                ])
            
            print(tabulate(table, headers=["Ticker", "Type", "Price", "Trigger", "Status", "Conf", "Trend", "Win%", "PF", "Sig", "Stop", "Target", "Shares"], tablefmt="grid"))
            
            print(
                "\nRecommendation:"
                f" Breakout Win% >= {brain.min_win_rate_breakout:.0f}% PF >= {brain.min_pf_breakout:.2f},"
                f" Dip Win% >= {brain.min_win_rate_dip:.0f}% PF >= {brain.min_pf_dip:.2f}"
            )
            
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
                    "Shares": s.qty,
                    "WinRate": round(s.win_rate, 2),
                    "ProfitFactor": round(s.profit_factor, 2),
                    "Kelly": round(s.kelly, 2),
                    "Score": round(s.score, 2),
                    "ConfidenceScore": round(s.confidence_score, 1),
                    "ConfidenceGrade": s.confidence_grade,
                    "TrendGrade": s.trend_grade,
                    "T_Statistic": round(s.t_statistic, 2),
                    "Significant": "Y" if s.t_statistic >= 2.0 else "N",
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
            
            # =================================================================
            # PRE-TRADE CHECKLIST - Run for top signals
            # =================================================================
            print("\n" + "="*60)
            print("  PRE-TRADE CHECKLIST FOR TOP SIGNALS")
            print("="*60)
            
            # Get the cached data for checklist validation
            tickers_data, all_data = brain.get_data()
            
            for s in setups[:3]:  # Check top 3 signals
                try:
                    if isinstance(all_data.columns, pd.MultiIndex) and s.ticker in all_data.columns.levels[0]:
                        ticker_df = all_data[s.ticker].dropna()
                        checklist = PreTradeChecklist(s, ticker_df, risk_mgr)
                        passed, checks, warnings = checklist.run_all_checks()
                        checklist.print_checklist()
                except Exception as e:
                    print(f"  Could not run checklist for {s.ticker}: {e}")
            
            # =================================================================
            # AUTO-TRACK TOP SIGNALS
            # =================================================================
            print("\n" + "="*60)
            print("  AUTO-TRACKING TOP SIGNALS")
            print("="*60)
            
            for s in setups[:AUTO_TRACK_TOP_N]:
                result = signal_tracker.add_signal(s, s.price)
                if result == "added":
                    print(f"  [+] Added {s.ticker} to tracker (Entry: ${s.trigger:.2f})")
                else:
                    print(f"  [~] Updated {s.ticker} in tracker")
            
            # Update prices for all tracked signals using current data
            tickers_data_for_tracker, all_data_for_tracker = brain.get_data()
            if all_data_for_tracker is not None and not all_data_for_tracker.empty:
                price_dict = {}
                for ticker in signal_tracker.get_active_signals().keys():
                    try:
                        if isinstance(all_data_for_tracker.columns, pd.MultiIndex):
                            if ticker in all_data_for_tracker.columns.levels[0]:
                                ticker_df = all_data_for_tracker[ticker].dropna()
                                if not ticker_df.empty:
                                    price_dict[ticker] = ticker_df['Close'].iloc[-1]
                    except Exception:
                        pass
                if price_dict:
                    signal_tracker.update_prices(price_dict)
            
            # Show updated tracker status
            signal_tracker.print_status()
            
            # Paper trade mode - track without real money
            if getattr(args, 'paper_trade', False) and setups:
                paper_tracker = PaperTradeTracker()
                print("\n" + "="*50)
                print("  PAPER TRADE MODE")
                print("  Add signals to paper portfolio for tracking")
                print("="*50)
                
                for s in setups[:5]:  # Show top 5
                    print(f"  [{setups.index(s)+1}] {s.ticker} ({s.strategy}) - ${s.trigger:.2f}")
                
                print("\n  Enter number to add, or press ENTER to skip:")
                choice = input("  > ").strip()
                
                if choice.isdigit():
                    idx = int(choice) - 1
                    if 0 <= idx < len(setups):
                        paper_tracker.add_trade(setups[idx])
                
                paper_tracker.print_summary()
            
            elif args.portfolio:
                addToPortfolio(setups)
        
    else:
        # No setups found
        if getattr(args, 'trust_mode', False) or getattr(args, 'trust_paper', False):
            # Trust mode simple output
            print_simple_verdict([], trust_manager, vix_level_for_trust)
        else:
            print("\nNo valid setups found that passed statistical filters.")
            print("This is EXPECTED with strict production settings.")
            print("Review near-miss report for stocks close to qualification.")
        
        # Still update tracked signals prices even if no new setups
        if signal_tracker.get_active_signals():
            tickers_data_for_tracker, all_data_for_tracker = brain.get_data()
            if all_data_for_tracker is not None and not all_data_for_tracker.empty:
                price_dict = {}
                for ticker in signal_tracker.get_active_signals().keys():
                    try:
                        if isinstance(all_data_for_tracker.columns, pd.MultiIndex):
                            if ticker in all_data_for_tracker.columns.levels[0]:
                                ticker_df = all_data_for_tracker[ticker].dropna()
                                if not ticker_df.empty:
                                    price_dict[ticker] = ticker_df['Close'].iloc[-1]
                    except Exception:
                        pass
                if price_dict:
                    signal_tracker.update_prices(price_dict)
            
            signal_tracker.print_status()

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
