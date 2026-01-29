"""
Titan Trade Market Module
=========================
Market hours utilities and regime detection.
"""

import os
import time
import json
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
    def is_market_open():
        """Check if US stock market is currently open."""
        now = MarketHours.get_eastern_time()
        if now.weekday() >= 5:
            return False
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
        """Determine if data should be auto-refreshed."""
        if not AUTO_REFRESH_DURING_MARKET_HOURS:
            return False
        if not os.path.exists(cache_file):
            return True
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
        return "CLOSED"
    
    @staticmethod
    def time_until_market_open():
        """Get timedelta until next market open."""
        now = MarketHours.get_eastern_time()
        next_open = now.replace(hour=MARKET_OPEN_HOUR, minute=MARKET_OPEN_MINUTE, second=0, microsecond=0)
        if now >= next_open:
            next_open += timedelta(days=1)
        while next_open.weekday() >= 5:
            next_open += timedelta(days=1)
        return next_open - now


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
            else: 
                status = "RECOVERY"
                score = 0.7
        else:
            if curr < sma50:
                status = "BEAR"
                score = 0.0
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
