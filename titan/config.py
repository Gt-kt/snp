"""
Titan Trade Configuration
=========================
All constants and configuration settings in one place.
"""

import os

# =============================================================================
# FILE PATHS
# =============================================================================
CACHE_DIR = "cache_sp500_elite"
SP500_CACHE_FILE = os.path.join(CACHE_DIR, "sp500_tickers.json")
OHLCV_CACHE_FILE = os.path.join(CACHE_DIR, "sp500_ohlcv_bulk.parquet")
PORTFOLIO_FILE = "portfolio.json"
TRADE_LOG_FILE = "trade_log.json"
RISK_LOG_FILE = "risk_log.json"
PRE_TRADE_CHECKLIST_FILE = "pre_trade_checklist.json"
SIGNAL_TRACKER_FILE = "signal_tracker.json"
PAPER_TRADE_FILE = "paper_trades.json"
SECTOR_CACHE_FILE = os.path.join(CACHE_DIR, "sector_cache.json")
DASHBOARD_FILE = "performance_dashboard.png"
AUTO_MODE_CONFIG_FILE = "titan_auto_config.json"

# =============================================================================
# DATA FRESHNESS
# =============================================================================
DEFAULT_SP500_TTL_DAYS = 7
DEFAULT_OHLCV_TTL_HOURS = 0.5  # 30 minutes for production
MAX_DATA_AGE_MINUTES = 30
SECTOR_CACHE_TTL_DAYS = 30
DASHBOARD_HISTORY_DAYS = 90

# =============================================================================
# DATA DOWNLOAD SETTINGS
# =============================================================================
DEFAULT_DATA_PERIOD = "5y"
DEFAULT_DATA_INTERVAL = "1d"
DEFAULT_MAX_WORKERS = max(2, min(12, (os.cpu_count() or 4)))

# =============================================================================
# LIQUIDITY REQUIREMENTS
# =============================================================================
MIN_AVG_DOLLAR_VOLUME = 10_000_000  # $10M daily volume minimum
MIN_AVG_VOLUME = 500_000  # 500K shares minimum
MAX_POSITION_PCT_OF_VOLUME = 1.0  # Never be more than 1% of daily volume

# =============================================================================
# SLIPPAGE MODEL
# =============================================================================
BASE_SLIPPAGE_BREAKOUT_BPS = 20.0  # Base 0.20%
BASE_SLIPPAGE_DIP_BPS = 10.0  # Base 0.10%
VOLUME_IMPACT_FACTOR = 50.0
DEFAULT_SLIPPAGE_BREAKOUT_BPS = 30.0
DEFAULT_SLIPPAGE_DIP_BPS = 10.0
DEFAULT_COMMISSION_BPS = 5.0

# =============================================================================
# POSITION SIZING & RISK MANAGEMENT
# =============================================================================
RISK_PER_TRADE = 500.0
ACCOUNT_SIZE = 100000.0
MAX_RISK_PCT_PER_TRADE = 0.5
MAX_POSITIONS = 6
MAX_SECTOR_EXPOSURE = 3
MAX_DAILY_LOSS_PCT = 2.0
MAX_WEEKLY_LOSS_PCT = 5.0
MAX_DRAWDOWN_PCT = 15.0
PORTFOLIO_HEAT_MAX = 6.0

# =============================================================================
# VIX THRESHOLDS
# =============================================================================
VIX_HIGH_THRESHOLD = 22
VIX_EXTREME_THRESHOLD = 28
VIX_PANIC_THRESHOLD = 35

# =============================================================================
# GAP PROTECTION
# =============================================================================
GAP_PROTECTION = True
MAX_GAP_PCT = 0.04  # 4% max gap history allowed

# =============================================================================
# QUALITY FILTERS (Default)
# =============================================================================
DEFAULT_MIN_WIN_RATE_BREAKOUT = 50.0
DEFAULT_MIN_WIN_RATE_DIP = 48.0
DEFAULT_MIN_PF_BREAKOUT = 1.3
DEFAULT_MIN_PF_DIP = 1.2
DEFAULT_MIN_TRADES_BREAKOUT = 10
DEFAULT_MIN_TRADES_DIP = 10
DEFAULT_MIN_EXPECTANCY_BREAKOUT = 0.002
DEFAULT_MIN_EXPECTANCY_DIP = 0.001
DEFAULT_MIN_RR_BREAKOUT = 1.5
DEFAULT_MIN_RR_DIP = 1.3

# =============================================================================
# WALK-FORWARD FILTERS
# =============================================================================
WF_MIN_TRADES = 8
WF_MIN_PF = 1.1
WF_MIN_EXPECTANCY = 0.0
WF_MIN_PASSRATE = 0.25
REGIME_MIN_SCORE = 0.5

# =============================================================================
# EARNINGS PROTECTION
# =============================================================================
EARNINGS_BLACKOUT_DAYS = 7
EARNINGS_POST_DAYS = 1

# =============================================================================
# MARKET HOURS
# =============================================================================
MARKET_OPEN_HOUR = 9
MARKET_OPEN_MINUTE = 30
MARKET_CLOSE_HOUR = 16
MARKET_CLOSE_MINUTE = 0
MARKET_TIMEZONE = "America/New_York"
AUTO_REFRESH_DURING_MARKET_HOURS = True

# =============================================================================
# NOTIFICATIONS
# =============================================================================
NOTIFICATIONS_ENABLED = True
NOTIFICATION_ON_NEW_SIGNAL = True
NOTIFICATION_ON_STOP_HIT = True
NOTIFICATION_ON_TARGET_HIT = True

# =============================================================================
# SCHEDULING
# =============================================================================
SCHEDULE_MARKET_OPEN_DELAY_MINUTES = 5
SCHEDULE_MARKET_CLOSE_BEFORE_MINUTES = 5

# =============================================================================
# SETUP CONFIRMATION
# =============================================================================
DEFAULT_CONFIRM_DAYS_BREAKOUT = 3
DEFAULT_CONFIRM_DAYS_DIP = 3
DEFAULT_REQUIRE_CONFIRMED_SETUP = True

# =============================================================================
# REGIME FACTORS
# =============================================================================
DEFAULT_REGIME_FACTORS = {
    "BULL": 1.0,
    "RECOVERY": 1.1,
    "NEUTRAL": 1.2,
    "Correction": 1.4,
    "BEAR": 2.0,
}

# =============================================================================
# OOS VALIDATION
# =============================================================================
DEFAULT_REQUIRE_OOS = True
DEFAULT_OOS_MIN_TRADES = 10
DEFAULT_OOS_MIN_WINRATE_BREAKOUT = 55.0
DEFAULT_OOS_MIN_WINRATE_DIP = 52.0
DEFAULT_OOS_MIN_PF_BREAKOUT = 1.3
DEFAULT_OOS_MIN_PF_DIP = 1.2
DEFAULT_OOS_MIN_EXPECTANCY_BREAKOUT = 0.002
DEFAULT_OOS_MIN_EXPECTANCY_DIP = 0.001

# =============================================================================
# SAFE MODE
# =============================================================================
SAFE_MODE_DEFAULT = False
SAFE_MODE_SETTINGS = {
    "require_walkforward": True,
    "require_oos": False,
    "confirm_days_breakout": 2,
    "confirm_days_dip": 2,
    "require_confirmed_setup": True,
    "min_trades_breakout": 12,
    "min_trades_dip": 12,
    "min_winrate_breakout": 50.0,
    "min_winrate_dip": 48.0,
    "min_pf_breakout": 1.3,
    "min_pf_dip": 1.2,
    "min_expectancy_breakout": 0.002,
    "min_expectancy_dip": 0.001,
    "min_rr_breakout": 1.5,
    "min_rr_dip": 1.3,
    "wf_min_trades": 8,
    "wf_min_pf": 1.1,
    "wf_min_expectancy": 0.0,
    "wf_min_passrate": 0.25,
    "oos_min_trades": 5,
    "oos_min_winrate_breakout": 48.0,
    "oos_min_winrate_dip": 45.0,
    "oos_min_pf_breakout": 1.1,
    "oos_min_pf_dip": 1.0,
    "oos_min_expectancy_breakout": 0.0,
    "oos_min_expectancy_dip": 0.0,
}

# =============================================================================
# TRUST MODE
# =============================================================================
TRUST_MODE_DEFAULT = False
AUTO_MODE_ENABLED = True

TRUST_MODE_SETTINGS = {
    "require_walkforward": True,
    "require_oos": True,
    "confirm_days_breakout": 3,
    "confirm_days_dip": 3,
    "require_confirmed_setup": True,
    "min_trades_breakout": 30,
    "min_trades_dip": 30,
    "min_winrate_breakout": 55.0,
    "min_winrate_dip": 52.0,
    "min_pf_breakout": 1.5,
    "min_pf_dip": 1.4,
    "min_expectancy_breakout": 0.005,
    "min_expectancy_dip": 0.003,
    "min_rr_breakout": 2.0,
    "min_rr_dip": 1.8,
    "wf_min_trades": 15,
    "wf_min_pf": 1.3,
    "wf_min_expectancy": 0.002,
    "wf_min_passrate": 0.50,
    "oos_min_trades": 10,
    "oos_min_winrate_breakout": 52.0,
    "oos_min_winrate_dip": 50.0,
    "oos_min_pf_breakout": 1.3,
    "oos_min_pf_dip": 1.2,
    "oos_min_expectancy_breakout": 0.002,
    "oos_min_expectancy_dip": 0.001,
}

TRUST_MODE_MIN_GRADE = "B"
TRUST_MODE_REQUIRE_SIGNIFICANCE = True
TRUST_MODE_MAX_TRADES_PER_DAY = 2
TRUST_MODE_MAX_TRADES_PER_WEEK = 5
TRUST_MODE_LOSS_STREAK_COOLOFF = 3
TRUST_MODE_COOLOFF_DAYS = 2
TRUST_MODE_PAPER_VALIDATION_DAYS = 30
TRUST_MODE_PAPER_MIN_TRADES = 10
TRUST_MODE_PAPER_MIN_WINRATE = 45.0
TRUST_MODE_MAX_RISK_PER_TRADE_PCT = 0.3
TRUST_MODE_MAX_POSITIONS = 4
TRUST_MODE_MAX_DATA_AGE_HOURS = 1
TRUST_MODE_VIX_CAUTION = 18
TRUST_MODE_VIX_HALT = 25

# =============================================================================
# MISC
# =============================================================================
DEFAULT_OUTPUT_DIR = "."
DEFAULT_LOG_LEVEL = "INFO"
VALID_LOG_LEVELS = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
DEFAULT_NEAR_MISS_REPORT = True
DEFAULT_NEAR_MISS_TOP = 50
AUTO_TRACK_TOP_N = 3
