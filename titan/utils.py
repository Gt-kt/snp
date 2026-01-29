"""
Titan Trade Utilities
=====================
Helper functions for data processing and calculations.
"""

import os
import json
import pandas as pd
import numpy as np
from datetime import datetime


def true_range(high, low, prev_close):
    """Calculate true range."""
    return pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)


def atr_series(df, period=14):
    """Calculate Average True Range series."""
    tr = true_range(df["High"], df["Low"], df["Close"].shift(1))
    return tr.rolling(period).mean()


def calculate_rsi(series, period=14):
    """Calculate Relative Strength Index."""
    delta = series.diff()
    gain = delta.where(delta > 0, 0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs = gain / (loss + 1e-9)
    return 100 - (100 / (1 + rs))


def ensure_multiindex(data, tickers):
    """Ensure data has MultiIndex columns."""
    if isinstance(data.columns, pd.MultiIndex):
        return data
    if len(tickers) == 1:
        fixed = data.copy()
        fixed.columns = pd.MultiIndex.from_product([tickers, data.columns])
        return fixed
    raise ValueError("Downloaded data is missing ticker-level columns.")


def expectancy(trades):
    """Calculate expectancy from list of trade returns."""
    if not trades:
        return 0.0
    return float(np.mean(trades))


def ensure_multiindex(data, tickers):
    """Ensure data has MultiIndex columns."""
    if isinstance(data.columns, pd.MultiIndex):
        return data
    if len(tickers) == 1:
        fixed = data.copy()
        fixed.columns = pd.MultiIndex.from_product([tickers, data.columns])
        return fixed
    raise ValueError("Downloaded data is missing ticker-level columns.")


def ensure_dir(path):
    """Ensure directory exists."""
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def load_json_config(path):
    """Load JSON configuration file."""
    if not path or not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def save_json(path, data):
    """Save data to JSON file."""
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)
        return True
    except Exception:
        return False


def parse_tickers(raw):
    """Parse ticker string or list into clean list."""
    if not raw:
        return []
    if isinstance(raw, str):
        parts = [p.strip().upper() for p in raw.split(",")]
    elif isinstance(raw, (list, tuple, set)):
        parts = [str(p).strip().upper() for p in raw]
    else:
        return []
    return [p for p in parts if p]


def load_tickers_from_file(path):
    """Load tickers from file (txt or json)."""
    if not path or not os.path.exists(path):
        return []
    try:
        if path.lower().endswith(".json"):
            data = load_json_config(path)
            if isinstance(data, dict):
                data = data.get("tickers", [])
            if isinstance(data, list):
                return parse_tickers(data)
            return []
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        raw = [p for p in text.replace("\n", ",").split(",") if p.strip()]
        return parse_tickers(raw)
    except Exception:
        return []


def parse_regime_factors(value):
    """Parse regime factors from string or dict."""
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


def format_currency(value, decimals=2):
    """Format number as currency string."""
    return f"${value:,.{decimals}f}"


def format_percent(value, decimals=1):
    """Format number as percentage string."""
    return f"{value:.{decimals}f}%"


def resolve_output_paths(output_dir, save_history=True):
    """Generate output file paths."""
    ensure_dir(output_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    paths = {
        "scan_csv_latest": os.path.join(output_dir, "scan_results.csv"),
        "scan_txt_latest": os.path.join(output_dir, "scan_results.txt"),
        "scan_json_latest": os.path.join(output_dir, "scan_results.json"),
        "near_miss_csv_latest": os.path.join(output_dir, "near_miss.csv"),
        "near_miss_json_latest": os.path.join(output_dir, "near_miss.json"),
        "config_json_latest": os.path.join(output_dir, "run_config.json"),
        "log_file": os.path.join(output_dir, f"titan_trade_{timestamp}.log") if save_history 
                    else os.path.join(output_dir, "titan_trade.log"),
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
        paths.update({
            "scan_csv": paths["scan_csv_latest"],
            "scan_txt": paths["scan_txt_latest"],
            "scan_json": paths["scan_json_latest"],
            "near_miss_csv": paths["near_miss_csv_latest"],
            "near_miss_json": paths["near_miss_json_latest"],
            "config_json": paths["config_json_latest"],
        })
    
    return paths
