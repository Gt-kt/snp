"""
Titan Trade Market Module
=========================
Market hours utilities and regime detection.
"""

import os
import time
import json
import calendar
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta, date

from .config import (
    MARKET_OPEN_HOUR, MARKET_OPEN_MINUTE, MARKET_CLOSE_HOUR, MARKET_CLOSE_MINUTE,
    MARKET_TIMEZONE, AUTO_REFRESH_DURING_MARKET_HOURS,
    VIX_HIGH_THRESHOLD, VIX_EXTREME_THRESHOLD, VIX_PANIC_THRESHOLD,
    SECTOR_CACHE_FILE, SECTOR_CACHE_TTL_DAYS,
    EARNINGS_BLACKOUT_DAYS, EARNINGS_POST_DAYS
)

# Try to import pytz
try:
    import pytz
    HAS_PYTZ = True
except ImportError:
    HAS_PYTZ = False



def _nth_weekday_of_month(year, month, weekday, occurrence):
    first_day = date(year, month, 1)
    offset = (weekday - first_day.weekday()) % 7
    return first_day + timedelta(days=offset + (occurrence - 1) * 7)


def _last_weekday_of_month(year, month, weekday):
    last_day = date(year, month, calendar.monthrange(year, month)[1])
    while last_day.weekday() != weekday:
        last_day -= timedelta(days=1)
    return last_day


def _observed_holiday(day):
    if day.weekday() == 5:
        return day - timedelta(days=1)
    if day.weekday() == 6:
        return day + timedelta(days=1)
    return day


def _easter_sunday(year):
    a = year % 19
    b = year // 100
    c = year % 100
    d = b // 4
    e = b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i = c // 4
    k = c % 4
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month = (h + l - 7 * m + 114) // 31
    day = ((h + l - 7 * m + 114) % 31) + 1
    return date(year, month, day)


def _nyse_holidays(year):
    return {
        _observed_holiday(date(year, 1, 1)),
        _nth_weekday_of_month(year, 1, 0, 3),
        _nth_weekday_of_month(year, 2, 0, 3),
        _easter_sunday(year) - timedelta(days=2),
        _last_weekday_of_month(year, 5, 0),
        _observed_holiday(date(year, 6, 19)),
        _observed_holiday(date(year, 7, 4)),
        _nth_weekday_of_month(year, 9, 0, 1),
        _nth_weekday_of_month(year, 11, 3, 4),
        _observed_holiday(date(year, 12, 25)),
    }


def _nyse_early_closes(year):
    holidays = _nyse_holidays(year)
    early_closes = set()

    black_friday = _nth_weekday_of_month(year, 11, 3, 4) + timedelta(days=1)
    if black_friday.weekday() < 5 and black_friday not in holidays:
        early_closes.add(black_friday)

    july_3 = date(year, 7, 3)
    if july_3.weekday() < 5 and july_3 not in holidays:
        early_closes.add(july_3)

    christmas_eve = date(year, 12, 24)
    if christmas_eve.weekday() < 5 and christmas_eve not in holidays:
        early_closes.add(christmas_eve)

    return early_closes


class MarketHours:
    """Utilities for checking market hours and smart refresh."""
    
    @staticmethod
    def get_eastern_time():
        """Get current time in Eastern timezone."""
        if HAS_PYTZ:
            eastern = pytz.timezone(MARKET_TIMEZONE)
            return datetime.now(eastern)
        return datetime.now()

    @staticmethod
    def is_trading_day(day):
        """Return True when NYSE is expected to be open for the session."""
        return day.weekday() < 5 and day not in _nyse_holidays(day.year)

    @staticmethod
    def get_session_close(day):
        """Return the regular or early-close timestamp for a trading day."""
        close_hour = MARKET_CLOSE_HOUR
        close_minute = MARKET_CLOSE_MINUTE
        if day in _nyse_early_closes(day.year):
            close_hour = 13
            close_minute = 0
        return close_hour, close_minute

    @staticmethod
    def get_session_bounds(now=None):
        """Return today's market-open and market-close timestamps, if tradable."""
        now = now or MarketHours.get_eastern_time()
        trading_day = now.date()
        if not MarketHours.is_trading_day(trading_day):
            return None, None
        market_open = now.replace(hour=MARKET_OPEN_HOUR, minute=MARKET_OPEN_MINUTE, second=0, microsecond=0)
        close_hour, close_minute = MarketHours.get_session_close(trading_day)
        market_close = now.replace(hour=close_hour, minute=close_minute, second=0, microsecond=0)
        return market_open, market_close
    
    @staticmethod
    def is_market_open():
        """Check if US stock market is currently open."""
        now = MarketHours.get_eastern_time()
        market_open, market_close = MarketHours.get_session_bounds(now)
        if market_open is None or market_close is None:
            return False
        return market_open <= now <= market_close
    
    @staticmethod
    def is_pre_market():
        """Check if in pre-market hours (4:00 AM - 9:30 AM ET)."""
        now = MarketHours.get_eastern_time()
        market_open, _ = MarketHours.get_session_bounds(now)
        if market_open is None:
            return False
        pre_market_start = now.replace(hour=4, minute=0, second=0, microsecond=0)
        return pre_market_start <= now < market_open
    
    @staticmethod
    def is_after_hours():
        """Check if in after-hours (session close - 8:00 PM ET)."""
        now = MarketHours.get_eastern_time()
        _, market_close = MarketHours.get_session_bounds(now)
        if market_close is None:
            return False
        after_hours_end = now.replace(hour=20, minute=0, second=0, microsecond=0)
        return market_close < now <= after_hours_end
    
    @staticmethod
    def should_auto_refresh(cache_file, ttl_hours=0.5):
        """Determine if data should be auto-refreshed based on market hours and cache age."""
        if not AUTO_REFRESH_DURING_MARKET_HOURS:
            return False
        if not os.path.exists(cache_file):
            return True
            
        mtime = os.path.getmtime(cache_file)
        
        if MarketHours.is_market_open():
            cache_time_est = datetime.fromtimestamp(mtime, tz=MarketHours.get_eastern_time().tzinfo)
            market_open_today, _ = MarketHours.get_session_bounds(cache_time_est)
            
            now_est = MarketHours.get_eastern_time()
            if market_open_today and cache_time_est < market_open_today <= now_est:
                return True
                
            cache_age_seconds = time.time() - mtime
            cache_age_hours = cache_age_seconds / 3600
            if cache_age_hours >= ttl_hours:
                return True
                
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
        return "CLOSED"
    
    @staticmethod
    def time_until_market_open():
        """Get timedelta until next market open."""
        now = MarketHours.get_eastern_time()
        candidate = now
        
        for _ in range(10):
            trading_day = candidate.date()
            if MarketHours.is_trading_day(trading_day):
                market_open = candidate.replace(hour=MARKET_OPEN_HOUR, minute=MARKET_OPEN_MINUTE, second=0, microsecond=0)
                if trading_day != now.date() or now < market_open:
                    return market_open - now
            candidate = (candidate + timedelta(days=1)).replace(hour=MARKET_OPEN_HOUR, minute=MARKET_OPEN_MINUTE, second=0, microsecond=0)
        
        return timedelta(days=1)


class MarketRegime:
    """Analyze market regime based on SPY and VIX."""
    
    def __init__(self, data):
        self.data = data

    def analyze_spy(self):
        """Analyze SPY to determine market status with VIX integration."""
        if "SPY" not in self.data:
            return "NEUTRAL", 0.5, None

        spy = self.data["SPY"]
        if isinstance(spy, pd.Series):
            return "NEUTRAL", 0.5, None
             
        c = spy['Close']
        sma50 = c.rolling(50).mean().iloc[-1]
        sma200 = c.rolling(200).mean().iloc[-1]
        curr = c.iloc[-1]
        
        status = "NEUTRAL"
        score = 0.5
        
        if curr > sma200:
            if sma50 > sma200: 
                status = "BULL"
                score = 1.0
                if c.rolling(20).mean().iloc[-1] > sma50 and curr > sma50:
                    status = "STRONG_BULL"
            else: 
                status = "RECOVERY"
                score = 0.7
        else:
            if curr < sma50:
                status = "BEAR"
                score = 0.0
                if sma50 < sma200 and c.rolling(20).mean().iloc[-1] < sma50 and curr < sma50:
                    status = "STRONG_BEAR"
            else:
                status = "Correction"
                score = 0.2
        
        # VIX Integration
        vix_scalar = 1.0
        vix_level = None
        for vix_key in ["^VIX", "VIX", "VIXY"]:
            if vix_key in self.data:
                try:
                    vix_df = self.data[vix_key]
                    if isinstance(vix_df, pd.DataFrame) and 'Close' in vix_df.columns:
                        vix_level = float(vix_df['Close'].iloc[-1])
                        if vix_level > VIX_PANIC_THRESHOLD:
                            vix_scalar = 0.0
                            status = f"{status}+PANIC"
                        elif vix_level > VIX_EXTREME_THRESHOLD:
                            vix_scalar = 0.25
                            status = f"{status}+FEAR"
                        elif vix_level > VIX_HIGH_THRESHOLD:
                            vix_scalar = 0.5
                            status = f"{status}+CAUTION"
                        break
                except Exception:
                    pass
                
        return status, score * vix_scalar, vix_level


class SectorMapper:
    """Get and cache sector information for tickers."""
    
    _cache = None
    _cache_loaded = False
    
    @classmethod
    def _load_cache(cls):
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
        try:
            os.makedirs(os.path.dirname(SECTOR_CACHE_FILE), exist_ok=True)
            with open(SECTOR_CACHE_FILE, 'w') as f:
                json.dump(cls._cache, f)
        except Exception:
            pass
    
    @classmethod
    def get_sector(cls, ticker):
        cls._load_cache()
        if ticker in cls._cache:
            return cls._cache[ticker]
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


class SectorAnalyzer:
    """Analyze and rank sector performance."""
    
    def __init__(self, data_dict):
        """Initialize with a dictionary of ticker -> DataFrame containing historical data."""
        self.data_dict = data_dict
        
    def get_top_sectors(self, top_n=3, lookback_days=20):
        """Rank sectors by relative momentum over the lookback period."""
        from .config import SECTOR_ETFS
        
        sector_performance = []
        for sector_name, etf_ticker in SECTOR_ETFS.items():
            if etf_ticker in self.data_dict:
                df = self.data_dict[etf_ticker]
                if isinstance(df, pd.DataFrame) and len(df) >= lookback_days:
                    # Calculate simple rate of return over the lookback
                    current_price = df['Close'].iloc[-1]
                    past_price = df['Close'].iloc[-lookback_days]
                    roc = ((current_price - past_price) / past_price) * 100
                    sector_performance.append((sector_name, roc))
        
        # Sort descending by return
        sector_performance.sort(key=lambda x: x[1], reverse=True)
        
        # Return only the top N sectors
        return [s[0] for s in sector_performance[:top_n]]


class EarningsCalendar:
    """Check earnings dates for stocks."""
    
    _cache = {}
    
    @classmethod
    def get_earnings_date(cls, ticker):
        if ticker in cls._cache:
            cached = cls._cache[ticker]
            if cached.get('fetched') and (datetime.now() - cached['fetched']).days < 1:
                return cached.get('date'), cached.get('days_until')
        
        try:
            stock = yf.Ticker(ticker)
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
        except Exception:
            pass
        
        cls._cache[ticker] = {'date': None, 'days_until': None, 'fetched': datetime.now()}
        return None, None
    
    @classmethod
    def is_in_blackout(cls, ticker, blackout_days=EARNINGS_BLACKOUT_DAYS, post_days=EARNINGS_POST_DAYS):
        earnings_date, days_until = cls.get_earnings_date(ticker)
        if earnings_date is None:
            return False, "Earnings date unknown"
        if days_until is not None:
            if 0 <= days_until <= blackout_days:
                return True, f"Earnings in {days_until} days"
            elif -post_days <= days_until < 0:
                return True, f"Earnings {abs(days_until)} days ago"
        return False, f"Earnings in {days_until} days" if days_until else "OK"

