"""
Titan Trade Data Models
=======================
Data classes and type definitions.
"""

from dataclasses import dataclass


@dataclass
class TitanSetup:
    """Represents a trading setup/signal."""
    ticker: str
    strategy: str
    price: float
    trigger: float
    stop: float
    target: float
    qty: int
    win_rate: float
    profit_factor: float
    kelly: float
    score: float
    sector: str
    earnings_call: str
    note: str
    confidence_score: float = 0.0
    confidence_grade: str = "F"
    trend_grade: str = "F"
    t_statistic: float = 0.0


class RejectionTracker:
    """Tracks rejection reasons during scanning."""
    
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
            "Gap Risk": 0,
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
