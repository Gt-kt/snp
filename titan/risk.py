"""
Titan Trade Risk Management
===========================
Portfolio risk management and position sizing.
"""

import os
import json
import shutil
import numpy as np
from datetime import datetime, timedelta, date

from .config import (
    ACCOUNT_SIZE, MAX_DAILY_LOSS_PCT, MAX_WEEKLY_LOSS_PCT,
    MAX_DRAWDOWN_PCT, PORTFOLIO_HEAT_MAX, RISK_LOG_FILE,
    MAX_RISK_PCT_PER_TRADE, MIN_AVG_DOLLAR_VOLUME, MIN_AVG_VOLUME,
    MAX_POSITION_PCT_OF_VOLUME, BASE_SLIPPAGE_BREAKOUT_BPS,
    BASE_SLIPPAGE_DIP_BPS, VOLUME_IMPACT_FACTOR, MAX_DATA_AGE_MINUTES,
    USE_VOLATILITY_PARITY, TARGET_VOLATILITY_PCT
)
from .storage import write_json_atomic


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
        self._load_state()
    
    def _load_state(self):
        self.state = {
            "peak_equity": self.account_size,
            "current_equity": self.account_size,
            "day_start_equity": self.account_size,
            "week_start_equity": self.account_size,
            "daily_pnl": 0.0,
            "weekly_pnl": 0.0,
            "last_equity_date": None,
            "last_trade_date": None,
            "last_week_start": None,
            "open_positions": {},
            "trade_history": [],
            "state_load_error": None,
        }
        if os.path.exists(self.risk_log_file):
            try:
                with open(self.risk_log_file, "r") as f:
                    saved = json.load(f)
                    self.state.update(saved)
                    self.state["state_load_error"] = None
            except Exception as exc:
                self.state["state_load_error"] = str(exc)
                try:
                    shutil.copy2(self.risk_log_file, self.risk_log_file + ".corrupt.bak")
                except Exception:
                    pass
    
    def _save_state(self):
        try:
            write_json_atomic(self.risk_log_file, self.state, indent=2)
        except Exception as exc:
            raise RuntimeError(f"Failed to save risk state {self.risk_log_file}: {exc}") from exc
    
    def _reset_daily_if_needed(self, reference_equity=None):
        today = date.today().isoformat()
        current_equity = float(reference_equity or self.state.get("current_equity", self.account_size))
        if self.state.get("last_equity_date") != today:
            self.state["day_start_equity"] = current_equity
            self.state["daily_pnl"] = 0.0
            self.state["last_equity_date"] = today
            self.state["last_trade_date"] = today
    
    def _reset_weekly_if_needed(self, reference_equity=None):
        today = date.today()
        week_start = (today - timedelta(days=today.weekday())).isoformat()
        current_equity = float(reference_equity or self.state.get("current_equity", self.account_size))
        if self.state.get("last_week_start") != week_start:
            self.state["week_start_equity"] = current_equity
            self.state["weekly_pnl"] = 0.0
            self.state["last_week_start"] = week_start
    
    def update_equity(self, new_equity, day_start_equity=None):
        new_equity = float(new_equity)
        today = date.today().isoformat()
        if day_start_equity is not None:
            self.state["day_start_equity"] = float(day_start_equity)
            self.state["last_equity_date"] = today
            self.state["last_trade_date"] = today
        else:
            self._reset_daily_if_needed(reference_equity=new_equity)
        self._reset_weekly_if_needed(reference_equity=new_equity)
        self.state["current_equity"] = new_equity
        self.state["daily_pnl"] = new_equity - float(
            self.state.get("day_start_equity", new_equity)
        )
        self.state["weekly_pnl"] = new_equity - float(
            self.state.get("week_start_equity", new_equity)
        )
        if new_equity > self.state["peak_equity"]:
            self.state["peak_equity"] = new_equity
        self._save_state()
    
    def record_trade_result(self, pnl_dollars):
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
        if len(self.state["trade_history"]) > 100:
            self.state["trade_history"] = self.state["trade_history"][-100:]
        self._save_state()
    
    def get_current_drawdown_pct(self):
        peak = self.state["peak_equity"]
        current = self.state["current_equity"]
        if peak <= 0:
            return 0.0
        return (peak - current) / peak * 100
    
    def get_daily_loss_pct(self):
        self._reset_daily_if_needed()
        return -self.state["daily_pnl"] / self.account_size * 100
    
    def get_weekly_loss_pct(self):
        self._reset_weekly_if_needed()
        return -self.state["weekly_pnl"] / self.account_size * 100
    
    def calculate_portfolio_heat(self, open_positions):
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
        self._reset_daily_if_needed()
        self._reset_weekly_if_needed()

        if self.state.get("state_load_error"):
            return False, f"RISK STATE LOAD FAILED: {self.state['state_load_error']}"
        
        dd = self.get_current_drawdown_pct()
        if dd >= self.max_drawdown_pct:
            return False, f"MAX DRAWDOWN EXCEEDED: {dd:.1f}% (limit: {self.max_drawdown_pct}%)"
        
        daily_loss = self.get_daily_loss_pct()
        if daily_loss >= self.max_daily_loss_pct:
            return False, f"DAILY LOSS LIMIT: {daily_loss:.1f}% (limit: {self.max_daily_loss_pct}%)"
        
        weekly_loss = self.get_weekly_loss_pct()
        if weekly_loss >= self.max_weekly_loss_pct:
            return False, f"WEEKLY LOSS LIMIT: {weekly_loss:.1f}% (limit: {self.max_weekly_loss_pct}%)"
        
        if open_positions:
            heat = self.calculate_portfolio_heat(open_positions)
            if heat >= self.max_portfolio_heat:
                return False, f"PORTFOLIO HEAT LIMIT: {heat:.1f}% (limit: {self.max_portfolio_heat}%)"
        
        return True, "OK"
    
    def get_position_size_scalar(self, market_regime="NEUTRAL"):
        scalar = 1.0
        
        # Drawdown Penalty
        dd = self.get_current_drawdown_pct()
        if dd > self.max_drawdown_pct * 0.5:
            dd_scalar = 1.0 - (dd - self.max_drawdown_pct * 0.5) / (self.max_drawdown_pct * 0.5)
            scalar = min(scalar, max(0.25, dd_scalar))
            
        # Daily Loss Penalty
        daily_loss = self.get_daily_loss_pct()
        if daily_loss > self.max_daily_loss_pct * 0.5:
            daily_scalar = 1.0 - (daily_loss - self.max_daily_loss_pct * 0.5) / (self.max_daily_loss_pct * 0.5)
            scalar = min(scalar, max(0.25, daily_scalar))
            
        # Regime Penalty
        if market_regime == "STRONG_BEAR":
            scalar *= 0.25
        elif market_regime == "BEAR":
            scalar *= 0.5
        elif market_regime == "Correction":
            scalar *= 0.75
        elif market_regime == "STRONG_BULL":
            scalar *= 1.25 # Allow slight leverage/expansion in pure bull markets
            
        return scalar
        
    def calculate_position_shares(self, price, stop_loss, atr, market_regime="NEUTRAL", max_volume_shares=None):
        """
        Calculate the number of shares to buy using Volatility Parity if enabled,
        otherwise fall back to standard risk-per-trade fraction.
        """
        scalar = self.get_position_size_scalar(market_regime)
        
        if USE_VOLATILITY_PARITY and atr > 0:
            # Volatility Parity: 
            # We want 1 ATR move to equal TARGET_VOLATILITY_PCT (0.5%) of the account
            target_dollar_volatility = self.account_size * (TARGET_VOLATILITY_PCT / 100.0) * scalar
            shares = target_dollar_volatility / atr
        else:
            # Traditional Fixed Fractional Risk
            risk_per_share = price - stop_loss
            if risk_per_share <= 0:
                return 0
            
            target_risk_dollars = self.account_size * (MAX_RISK_PCT_PER_TRADE / 100.0) * scalar
            shares = target_risk_dollars / risk_per_share
            
        # Apply liquidity limits
        if max_volume_shares is not None:
            shares = min(shares, max_volume_shares)
            
        return int(shares)
    
    def get_risk_status(self, open_positions=None):
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


class DataValidator:
    """Validates data quality before trading."""
    
    @staticmethod
    def check_data_freshness(df, max_age_minutes=MAX_DATA_AGE_MINUTES):
        if df is None or df.empty:
            return False, float('inf'), "No data available"
        try:
            last_date = df.index[-1]
            if isinstance(last_date, str):
                import pandas as pd
                last_date = pd.to_datetime(last_date)
            now = datetime.now()
            if hasattr(last_date, 'date'):
                data_date = last_date.date()
            else:
                import pandas as pd
                data_date = pd.to_datetime(last_date).date()
            today = now.date()
            yesterday = (now - timedelta(days=1)).date()
            market_open = now.replace(hour=9, minute=30, second=0)
            if data_date == today:
                return True, 0, "Data is from today"
            elif data_date == yesterday:
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
    def check_liquidity(df, min_dollar_vol=MIN_AVG_DOLLAR_VOLUME, min_volume=MIN_AVG_VOLUME):
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
        if avg_volume <= 0:
            return 0.01
        pct_of_volume = (shares / avg_volume) * 100
        base_bps = BASE_SLIPPAGE_BREAKOUT_BPS if is_breakout else BASE_SLIPPAGE_DIP_BPS
        volume_impact_bps = pct_of_volume * VOLUME_IMPACT_FACTOR
        total_slippage_bps = base_bps + volume_impact_bps
        return min(total_slippage_bps / 10000, 0.02)
    
    @staticmethod
    def max_position_size(avg_volume, price, max_pct_of_volume=MAX_POSITION_PCT_OF_VOLUME):
        if avg_volume <= 0 or price <= 0:
            return 0
        return int(avg_volume * (max_pct_of_volume / 100))


class StatisticalConfidenceScorer:
    """Calculates confidence score based on statistical robustness.
    
    Scoring is calibrated for realistic profitable trading:
    - Grade A (70+): Excellent setup, high confidence
    - Grade B (55+): Good setup, trade with normal size  
    - Grade C (40+): Acceptable setup, consider reduced size
    - Grade D (25+): Marginal, only in strong bull market
    - Grade F (<25): Avoid
    """
    
    @staticmethod
    def calculate_confidence(trades, win_rate, profit_factor, expectancy,
                           wf_pass_rate=None, oos_pf=None, consistency_score=None):
        score = 0
        factors = {}

        # Sample Size (0-25 points) - Realistic for 500-day backtest
        if trades >= 50:
            sample_score = 25
        elif trades >= 30:
            sample_score = 20
        elif trades >= 20:
            sample_score = 16
        elif trades >= 15:
            sample_score = 12
        elif trades >= 10:
            sample_score = 8
        elif trades >= 5:
            sample_score = 4
        else:
            sample_score = 0
        score += sample_score
        factors["sample_size"] = {"value": trades, "score": sample_score, "max": 25}

        # Win Rate (0-25 points) - Core profitability metric
        if win_rate >= 65:
            wr_score = 25
        elif win_rate >= 58:
            wr_score = 20
        elif win_rate >= 52:
            wr_score = 16
        elif win_rate >= 48:
            wr_score = 12
        elif win_rate >= 45:
            wr_score = 8
        elif win_rate >= 40:
            wr_score = 4
        else:
            wr_score = 0
        score += wr_score
        factors["win_rate"] = {"value": win_rate, "score": wr_score, "max": 25}

        # Profit Factor (0-25 points) - Key for real profitability
        if profit_factor >= 2.0:
            pf_score = 25
        elif profit_factor >= 1.7:
            pf_score = 20
        elif profit_factor >= 1.5:
            pf_score = 16
        elif profit_factor >= 1.3:
            pf_score = 12
        elif profit_factor >= 1.15:
            pf_score = 8
        elif profit_factor >= 1.0:
            pf_score = 4
        else:
            pf_score = 0
        score += pf_score
        factors["profit_factor"] = {"value": profit_factor, "score": pf_score, "max": 25}

        # Expectancy (0-15 points) - Average profit per trade
        if expectancy >= 0.015:
            exp_score = 15
        elif expectancy >= 0.01:
            exp_score = 12
        elif expectancy >= 0.005:
            exp_score = 9
        elif expectancy >= 0.003:
            exp_score = 6
        elif expectancy >= 0.001:
            exp_score = 3
        else:
            exp_score = 0
        score += exp_score
        factors["expectancy"] = {"value": expectancy, "score": exp_score, "max": 15}

        # Walk-Forward bonus (0-5 points)
        if wf_pass_rate is not None:
            if wf_pass_rate >= 0.60:
                wf_score = 5
            elif wf_pass_rate >= 0.40:
                wf_score = 3
            elif wf_pass_rate >= 0.25:
                wf_score = 1
            else:
                wf_score = 0
            score += wf_score
            factors["wf_pass_rate"] = {"value": wf_pass_rate, "score": wf_score, "max": 5}

        # OOS bonus (0-5 points)
        if oos_pf is not None:
            if oos_pf >= 1.3:
                oos_score = 5
            elif oos_pf >= 1.1:
                oos_score = 3
            elif oos_pf >= 1.0:
                oos_score = 1
            else:
                oos_score = 0
            score += oos_score
            factors["oos_profit_factor"] = {"value": oos_pf, "score": oos_score, "max": 5}

        # Conviction bonus (0-10 points) — forward-looking signal quality
        # Rewards stocks with strong momentum, institutional accumulation,
        # and relative strength leadership beyond just backtest metrics.
        if consistency_score is not None and consistency_score > 0:
            conv_score = int(min(10, consistency_score))
            score += conv_score
            factors["conviction"] = {"value": consistency_score, "score": conv_score, "max": 10}

        # Grade - Calibrated for practical profitable trading
        if score >= 70:
            grade = "A"
        elif score >= 55:
            grade = "B"
        elif score >= 40:
            grade = "C"
        elif score >= 25:
            grade = "D"
        else:
            grade = "F"

        return {
            "score": score,
            "grade": grade,
            "factors": factors,
            "tradeable": score >= 40 and trades >= 10 and profit_factor >= 1.15,
        }
    
    @staticmethod
    def calculate_t_statistic(trades_list):
        if not trades_list or len(trades_list) < 10:
            return 0.0
        mean_ret = np.mean(trades_list)
        std_ret = np.std(trades_list, ddof=1)
        n = len(trades_list)
        if std_ret == 0:
            return 0.0
        return float(mean_ret / (std_ret / np.sqrt(n)))




