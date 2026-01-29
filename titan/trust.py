"""
Titan Trade Trust Mode
======================
Trust Mode and Auto Mode management for hands-off trading.
"""

import os
import json
from datetime import datetime, timedelta, date

from .config import (
    ACCOUNT_SIZE, RISK_PER_TRADE, AUTO_MODE_CONFIG_FILE,
    TRUST_MODE_MAX_TRADES_PER_DAY, TRUST_MODE_MAX_TRADES_PER_WEEK,
    TRUST_MODE_LOSS_STREAK_COOLOFF, TRUST_MODE_COOLOFF_DAYS,
    TRUST_MODE_PAPER_VALIDATION_DAYS, TRUST_MODE_PAPER_MIN_TRADES,
    TRUST_MODE_PAPER_MIN_WINRATE, TRUST_MODE_MAX_POSITIONS,
    TRUST_MODE_MIN_GRADE, TRUST_MODE_REQUIRE_SIGNIFICANCE,
    TRUST_MODE_VIX_CAUTION, TRUST_MODE_VIX_HALT
)


class TrustModeManager:
    """Manages Trust Mode protections and limits."""
    
    TRUST_STATE_FILE = "trust_mode_state.json"
    
    def __init__(self, account_size=ACCOUNT_SIZE):
        self.account_size = account_size
        self._load_state()
    
    def _load_state(self):
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
        
        self._reset_counters_if_needed()
    
    def _save_state(self):
        try:
            with open(self.TRUST_STATE_FILE, "w") as f:
                json.dump(self.state, f, indent=2, default=str)
        except Exception:
            pass
    
    def _reset_counters_if_needed(self):
        today = date.today().isoformat()
        week_start = (date.today() - timedelta(days=date.today().weekday())).isoformat()
        
        if self.state["last_trade_date"] != today:
            self.state["trades_today"] = []
            self.state["last_trade_date"] = today
        
        if self.state["last_week_start"] != week_start:
            self.state["trades_this_week"] = []
            self.state["last_week_start"] = week_start
        
        if self.state["cooloff_until"]:
            try:
                cooloff_date = datetime.fromisoformat(self.state["cooloff_until"])
                if datetime.now() > cooloff_date:
                    self.state["cooloff_until"] = None
                    self.state["consecutive_losses"] = 0
            except Exception:
                pass
    
    def is_paper_trading_validated(self):
        if self.state["paper_validated"]:
            return True, "Paper trading validated"
        
        if not self.state["paper_trading_started"]:
            return False, f"Paper trading not started. Run with --trust-paper for {TRUST_MODE_PAPER_VALIDATION_DAYS} days first."
        
        started = datetime.fromisoformat(self.state["paper_trading_started"])
        days_elapsed = (datetime.now() - started).days
        
        if days_elapsed < TRUST_MODE_PAPER_VALIDATION_DAYS:
            return False, f"Paper trading day {days_elapsed}/{TRUST_MODE_PAPER_VALIDATION_DAYS}. Keep paper trading."
        
        if self.state["paper_trades_count"] < TRUST_MODE_PAPER_MIN_TRADES:
            return False, f"Need {TRUST_MODE_PAPER_MIN_TRADES} paper trades, have {self.state['paper_trades_count']}."
        
        total = self.state["paper_wins"] + self.state["paper_losses"]
        if total > 0:
            win_rate = (self.state["paper_wins"] / total) * 100
            if win_rate < TRUST_MODE_PAPER_MIN_WINRATE:
                return False, f"Paper win rate {win_rate:.0f}% < {TRUST_MODE_PAPER_MIN_WINRATE}%. Keep practicing."
        
        self.state["paper_validated"] = True
        self._save_state()
        return True, "Paper trading validated! You may now use live trading."
    
    def start_paper_trading(self):
        if not self.state["paper_trading_started"]:
            self.state["paper_trading_started"] = datetime.now().isoformat()
            self._save_state()
        return self.state["paper_trading_started"]
    
    def record_paper_trade(self, won):
        self.state["paper_trades_count"] += 1
        if won:
            self.state["paper_wins"] += 1
        else:
            self.state["paper_losses"] += 1
        self._save_state()
    
    def can_trade_today(self):
        self._reset_counters_if_needed()
        
        if self.state["cooloff_until"]:
            try:
                cooloff_date = datetime.fromisoformat(self.state["cooloff_until"])
                if datetime.now() < cooloff_date:
                    return False, f"Cooloff period active until {cooloff_date.strftime('%Y-%m-%d')}. {self.state['consecutive_losses']} consecutive losses."
            except Exception:
                pass
        
        if len(self.state["trades_today"]) >= TRUST_MODE_MAX_TRADES_PER_DAY:
            return False, f"Daily limit reached ({TRUST_MODE_MAX_TRADES_PER_DAY} trades). Try again tomorrow."
        
        if len(self.state["trades_this_week"]) >= TRUST_MODE_MAX_TRADES_PER_WEEK:
            return False, f"Weekly limit reached ({TRUST_MODE_MAX_TRADES_PER_WEEK} trades). Wait for next week."
        
        if self.state["current_positions"] >= TRUST_MODE_MAX_POSITIONS:
            return False, f"Position limit reached ({TRUST_MODE_MAX_POSITIONS} positions). Close some first."
        
        return True, "OK"
    
    def record_trade(self, ticker, won=None):
        self._reset_counters_if_needed()
        trade = {"ticker": ticker, "time": datetime.now().isoformat()}
        self.state["trades_today"].append(trade)
        self.state["trades_this_week"].append(trade)
        self.state["total_trades"] += 1
        self.state["current_positions"] += 1
        if won is not None:
            self.record_trade_result(won)
        self._save_state()
    
    def record_trade_result(self, won):
        if won:
            self.state["total_wins"] += 1
            self.state["consecutive_losses"] = 0
        else:
            self.state["total_losses"] += 1
            self.state["consecutive_losses"] += 1
            if self.state["consecutive_losses"] >= TRUST_MODE_LOSS_STREAK_COOLOFF:
                cooloff_date = datetime.now() + timedelta(days=TRUST_MODE_COOLOFF_DAYS)
                self.state["cooloff_until"] = cooloff_date.isoformat()
        self._save_state()
    
    def close_position(self):
        if self.state["current_positions"] > 0:
            self.state["current_positions"] -= 1
            self._save_state()
    
    def get_status_report(self):
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
        self.state["paper_trading_started"] = None
        self.state["paper_trades_count"] = 0
        self.state["paper_wins"] = 0
        self.state["paper_losses"] = 0
        self.state["paper_validated"] = False
        self._save_state()
    
    def bypass_paper_validation(self, confirm_code):
        if confirm_code == "I ACCEPT THE RISK":
            self.state["paper_validated"] = True
            self._save_state()
            return True
        return False


class AutoModeManager:
    """Manages auto mode configuration and first-time setup."""
    
    CONFIG_FILE = AUTO_MODE_CONFIG_FILE
    
    def __init__(self):
        self.config = self._load_config()
    
    def _load_config(self):
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
        try:
            with open(self.CONFIG_FILE, "w") as f:
                json.dump(self.config, f, indent=2, default=str)
        except Exception:
            pass
    
    def is_first_run(self):
        return not self.config.get("first_run_complete", False)
    
    def run_first_time_setup(self):
        print("\n" + "=" * 70)
        print("  WELCOME TO TITAN TRADE - AUTO MODE")
        print("  First-Time Setup (One time only)")
        print("=" * 70)
        
        print("\n  This will configure Titan Trade for automatic operation.\n")
        
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
        print("     Type 'SKIP' to bypass, or press ENTER to paper trade first:")
        skip = input("     > ").strip()
        
        if skip.upper() == "SKIP":
            print("\n     Type 'I ACCEPT THE RISK' to confirm:")
            confirm = input("     > ").strip()
            self.config["paper_trading_bypassed"] = (confirm == "I ACCEPT THE RISK")
        else:
            self.config["paper_trading_bypassed"] = False
        
        self.config["first_run_complete"] = True
        self.config["created"] = datetime.now().isoformat()
        self._save_config()
        
        print("\n" + "=" * 70)
        print("  SETUP COMPLETE!")
        print("=" * 70)
        print(f"  Account Size: ${self.config['account_size']:,.0f}")
        print(f"  Risk Per Trade: ${self.config['risk_per_trade']:,.0f}")
        print("=" * 70)
        
        input("\n  Press ENTER to continue...")
        return self.config
    
    def get_config(self):
        return self.config


def print_trust_mode_header():
    """Print the Trust Mode header."""
    print("\n" + "=" * 70)
    print("  █" * 33)
    print("  █" + "     TITAN TRADE - TRUST MODE     ".center(64) + "█")
    print("  █" + "  If it says TRADE, trust it.     ".center(64) + "█")
    print("  █" * 33)
    print("=" * 70)


def print_simple_verdict(setups, trust_manager, vix_level=None):
    """Print simple TRADE / DON'T TRADE verdict for Trust Mode."""
    print("\n" + "=" * 70)
    
    can_trade, reason = trust_manager.can_trade_today()
    
    if not can_trade:
        print("  ╔══════════════════════════════════════════════════════════════╗")
        print("  ║           DON'T TRADE TODAY - LIMIT REACHED                 ║")
        print("  ╚══════════════════════════════════════════════════════════════╝")
        print(f"\n  REASON: {reason}")
        print("=" * 70)
        return None
    
    if vix_level is not None and vix_level > TRUST_MODE_VIX_HALT:
        print("  ╔══════════════════════════════════════════════════════════════╗")
        print("  ║           DON'T TRADE - HIGH VOLATILITY                     ║")
        print("  ╚══════════════════════════════════════════════════════════════╝")
        print(f"\n  VIX is at {vix_level:.1f} (HALT threshold: {TRUST_MODE_VIX_HALT})")
        print("=" * 70)
        return None
    
    # Filter for Grade A/B and statistical significance
    trusted_setups = []
    for s in setups:
        grade = getattr(s, 'confidence_grade', 'F')
        t_stat = getattr(s, 't_statistic', 0)
        if grade not in ['A', 'B']:
            continue
        if TRUST_MODE_REQUIRE_SIGNIFICANCE and t_stat < 2.0:
            continue
        trusted_setups.append(s)
    
    if not trusted_setups:
        print("  ╔══════════════════════════════════════════════════════════════╗")
        print("  ║           NO HIGH-CONFIDENCE TRADES TODAY                   ║")
        print("  ╚══════════════════════════════════════════════════════════════╝")
        print(f"\n  Scanned stocks. Found {len(setups)} setups but none meet TRUST criteria.")
        print("=" * 70)
        return None
    
    # Show trade signals
    print("  ╔══════════════════════════════════════════════════════════════╗")
    print("  ║                    TRADE SIGNAL!                             ║")
    print("  ╚══════════════════════════════════════════════════════════════╝")
    print(f"\n  Found {len(trusted_setups)} HIGH-CONFIDENCE trade(s):\n")
    
    for i, s in enumerate(trusted_setups[:TRUST_MODE_MAX_TRADES_PER_DAY], 1):
        shares = s.qty
        if vix_level and vix_level > TRUST_MODE_VIX_CAUTION:
            shares = max(1, shares // 2)
        
        risk_per_share = s.trigger - s.stop
        total_risk = risk_per_share * shares
        potential_profit = (s.target - s.trigger) * shares
        rr_ratio = (s.target - s.trigger) / risk_per_share if risk_per_share > 0 else 0
        
        print(f"  ┌─────────────────────────────────────────────────────────────┐")
        print(f"  │  #{i}  {s.ticker:<6}  {s.strategy:<12}  GRADE: {s.confidence_grade}                  │")
        print(f"  ├─────────────────────────────────────────────────────────────┤")
        print(f"  │  BUY:     {shares:>6} shares @ ${s.trigger:>8.2f}                     │")
        print(f"  │  STOP:    ${s.stop:>8.2f}                                       │")
        print(f"  │  TARGET:  ${s.target:>8.2f}                                       │")
        print(f"  │  RISK: ${total_risk:>8.2f}  REWARD: ${potential_profit:>8.2f}  R:R {rr_ratio:.1f}:1     │")
        print(f"  └─────────────────────────────────────────────────────────────┘")
        print()
    
    print("=" * 70)
    return trusted_setups
