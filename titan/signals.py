"""
Titan Trade Signals Module
==========================
Signal tracking and notification management.
"""

import os
import sys
import json
import subprocess
from datetime import datetime

from .config import (
    SIGNAL_TRACKER_FILE, NOTIFICATIONS_ENABLED,
    NOTIFICATION_ON_NEW_SIGNAL, NOTIFICATION_ON_STOP_HIT, NOTIFICATION_ON_TARGET_HIT
)

# Try to import plyer
try:
    from plyer import notification
    HAS_PLYER = True
except ImportError:
    HAS_PLYER = False


class NotificationManager:
    """Cross-platform desktop notifications."""
    
    @staticmethod
    def send(title, message, timeout=10):
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
                print(f"\n🔔 {title}: {message}\n")
                return True
        except Exception as e:
            print(f"Notification failed: {e}")
            return False
    
    @staticmethod
    def notify_new_signal(ticker, strategy, win_rate, profit_factor):
        if NOTIFICATION_ON_NEW_SIGNAL:
            NotificationManager.send(
                f"New Signal: {ticker}",
                f"{strategy} | WR: {win_rate:.0f}% | PF: {profit_factor:.2f}"
            )
    
    @staticmethod
    def notify_stop_hit(ticker, entry_price, exit_price, pnl_pct):
        if NOTIFICATION_ON_STOP_HIT:
            NotificationManager.send(
                f"STOP HIT: {ticker}",
                f"Entry: ${entry_price:.2f} → Exit: ${exit_price:.2f} ({pnl_pct:+.1f}%)"
            )
    
    @staticmethod
    def notify_target_hit(ticker, entry_price, exit_price, pnl_pct):
        if NOTIFICATION_ON_TARGET_HIT:
            NotificationManager.send(
                f"TARGET HIT: {ticker}",
                f"Entry: ${entry_price:.2f} → Exit: ${exit_price:.2f} ({pnl_pct:+.1f}%)"
            )


class SignalTracker:
    """Automatically tracks signals and their outcomes."""
    
    def __init__(self, file_path=SIGNAL_TRACKER_FILE):
        self.file_path = file_path
        self._load()
    
    def _load(self):
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
        try:
            with open(self.file_path, "w") as f:
                json.dump(self.data, f, indent=2, default=str)
        except Exception:
            pass
    
    def add_signal(self, setup, current_price):
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
            "t_statistic": getattr(setup, 't_statistic', 0),
            "confidence_grade": getattr(setup, 'confidence_grade', 'F'),
            "status": "WATCHING",
            "triggered_price": None,
            "triggered_date": None,
            "days_tracked": 0,
        }
        
        self.data["active_signals"][ticker] = signal
        self._save()
        return "added"
    
    def update_prices(self, price_dict):
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
        if ticker not in self.data["active_signals"]:
            return
        
        signal = self.data["active_signals"].pop(ticker)
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
        return self.data["active_signals"]
    
    def print_status(self):
        active = self.data["active_signals"]
        stats = self.data["stats"]
        
        print("\n" + "="*70)
        print("  AUTO-TRACKED SIGNALS")
        print("="*70)
        
        if not active:
            print("  No signals currently being tracked.")
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
