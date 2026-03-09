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
    momentum_score: float = 0.0
    accumulation_score: float = 0.0
    rs_percentile: float = 0.0
    pre_breakout_score: float = 0.0
    breakeven_trigger: float = 0.0
    trailing_stop: float = 0.0
    avwap_distance: float = 0.0
    starter_trigger: float = 0.0
    add_on_trigger: float = 0.0
    partial_target: float = 0.0
    planned_total_qty: int = 0
    add_on_qty: int = 0
    starter_size_pct: float = 0.0
    robustness_score: float = 0.0
    walk_forward_pass_rate: float = 0.0
    walk_forward_pf: float = 0.0
    walk_forward_trades: int = 0
    regime_score: float = 0.0
    oos_pf: float = 0.0
    oos_trades: int = 0
    net_expectancy: float = 0.0


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
            "Sizing Constraint": 0,
            "Earnings Risk": 0,
            "Gap Risk": 0,
            "Rejected (OOS)": 0,
            "Rejected (WF)": 0,
            "Rejected (Regime)": 0,
            "Regime Filter": 0,
            "WF Filter": 0,
            "OOS Filter": 0,
            "Not Near High": 0,
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
