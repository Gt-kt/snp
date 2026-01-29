"""
Titan Trade - Modular Stock Trading Scanner
============================================
A professional-grade S&P 500 scanner with Trust Mode for hands-off trading.

Modules:
- config: All constants and configuration settings
- utils: Helper functions (ATR, parsing, etc.)
- market: Market hours, regime detection, VIX
- validation: Strategy backtesting and validation
- risk: Risk management and position sizing
- signals: Signal tracking and notifications
- trust: Trust mode and auto mode management
- models: Data classes (TitanSetup, etc.)
"""

# Core config
from .config import *

# Utilities
from .utils import (
    atr_series, calculate_rsi, expectancy, ensure_multiindex,
    parse_tickers, load_tickers_from_file, resolve_output_paths
)

# Market analysis
from .market import MarketHours, MarketRegime, SectorMapper, EarningsCalendar

# Risk management
from .risk import PortfolioRiskManager, DataValidator, StatisticalConfidenceScorer

# Signals and notifications
from .signals import SignalTracker, NotificationManager

# Trust Mode
from .trust import TrustModeManager, AutoModeManager, print_trust_mode_header, print_simple_verdict

# Validation
from .validation import StrategyValidator, TrendQualityAnalyzer, Optimizer

# Models
from .models import TitanSetup, RejectionTracker

__version__ = "8.0.0"
__author__ = "Titan Trade"
