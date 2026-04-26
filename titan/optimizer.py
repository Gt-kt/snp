"""
Weekly auto-optimization for signal detection parameters.

Ported from KOSPI architecture. Pre-computes signals on a basket of liquid
stocks, then optimizes strength-floor thresholds by evaluating which floor
maximizes risk-adjusted forward returns.  Fast: signal detection runs once,
threshold grid search is just filtering.
"""

import json
import os
import time
from datetime import datetime

import numpy as np
import pandas as pd

from titan.config import CACHE_DIR

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
WEEKLY_OPT_PATH = os.path.join(CACHE_DIR, "weekly_optimization.json")
SIGNAL_PARAMS_PATH = os.path.join(CACHE_DIR, "optimized_signal_params.json")

# ---------------------------------------------------------------------------
# Default signal detection parameters (tunable)
# ---------------------------------------------------------------------------
DEFAULT_SIGNAL_PARAMS = {
    "strength_floor_bull": 6.0,
    "strength_floor_bear": 8.0,
    "strength_floor_sideways": 7.0,
}

# Grid search range — single-dimensional.
# We optimize one base floor and derive bear/sideways from it:
#   bear = base + 1.0, sideways = base + 0.5
# This is honest: without per-signal regime tracking, a 3D grid is theater.
_BASE_FLOOR_GRID = [5.0, 5.5, 6.0, 6.5, 7.0]


# ---------------------------------------------------------------------------
# Weekly check
# ---------------------------------------------------------------------------

def should_run_weekly_optimization() -> bool:
    """Check if 7+ days have passed since last optimization run."""
    if not os.path.exists(WEEKLY_OPT_PATH):
        return True
    try:
        with open(WEEKLY_OPT_PATH, "r") as f:
            data = json.load(f)
        last_run = datetime.fromisoformat(data.get("last_run", "2020-01-01"))
        return (datetime.now() - last_run).days >= 7
    except Exception:
        return True


def load_optimized_params() -> dict:
    """Load optimized signal params or return defaults."""
    if os.path.exists(SIGNAL_PARAMS_PATH):
        try:
            with open(SIGNAL_PARAMS_PATH, "r") as f:
                params = json.load(f)
            merged = DEFAULT_SIGNAL_PARAMS.copy()
            merged.update(params)
            return merged
        except Exception:
            pass
    return DEFAULT_SIGNAL_PARAMS.copy()


# ---------------------------------------------------------------------------
# Optimization engine
# ---------------------------------------------------------------------------

def run_weekly_optimization(
    stock_frames: dict[str, pd.DataFrame],
    spy_df: pd.DataFrame | None = None,
) -> dict:
    """Optimize strength-floor thresholds using pre-computed signal data.

    Strategy:
    1. Run detect_signal() ONCE on each of 10 liquid stocks over last 252 bars
    2. Record (signal_type, strength, 3-day forward return) for each hit
    3. Grid-search over floor thresholds to find which floor maximizes
       median(win_rate * (1 + avg_return))

    This is fast because signal detection happens once, then it's just filtering.
    """
    from titan.signal_detector import detect_signal

    print("  [OPT] Running weekly parameter optimization...")
    start = time.time()

    # Pick top 10 stocks by volume (not 20 — speed)
    vol_rank = []
    for ticker, df in stock_frames.items():
        if df is None or df.empty or len(df) < 200:
            continue
        avg_vol = float(df["Volume"].iloc[-20:].mean()) if len(df) >= 20 else 0
        vol_rank.append((ticker, avg_vol, df))
    vol_rank.sort(key=lambda x: x[1], reverse=True)
    test_stocks = vol_rank[:10]

    if len(test_stocks) < 5:
        print("  [OPT] Not enough stocks. Using defaults.")
        _save_optimization_results(DEFAULT_SIGNAL_PARAMS.copy(), 0)
        return DEFAULT_SIGNAL_PARAMS.copy()

    # Step 1: Pre-compute all signals with forward returns
    all_signals: list[dict] = []
    for ticker, _, df in test_stocks:
        signals = _precompute_signals(df, spy_df)
        all_signals.extend(signals)

    if len(all_signals) < 20:
        print(f"  [OPT] Only {len(all_signals)} signals found. Using defaults.")
        _save_optimization_results(DEFAULT_SIGNAL_PARAMS.copy(), 0)
        return DEFAULT_SIGNAL_PARAMS.copy()

    print(f"  [OPT] Pre-computed {len(all_signals)} signals across {len(test_stocks)} stocks")

    # Step 2: Grid search over base floor threshold (1-D)
    # Bear and sideways floors are derived from the base (bull) floor.
    best_score = -999.0
    best_params = DEFAULT_SIGNAL_PARAMS.copy()

    for base_floor in _BASE_FLOOR_GRID:
        params = {
            "strength_floor_bull": base_floor,
            "strength_floor_bear": base_floor + 1.0,
            "strength_floor_sideways": base_floor + 0.5,
        }

        # Filter signals by the base floor
        filtered = [s for s in all_signals if s["strength"] >= base_floor]
        if len(filtered) < 10:
            continue

        wins = sum(1 for s in filtered if s["return_3d"] > 0)
        avg_ret = sum(s["return_3d"] for s in filtered) / len(filtered)
        wr = wins / len(filtered) * 100

        score = (wr / 100.0) * (1 + avg_ret / 100.0)
        # Bonus for having enough trades
        if len(filtered) >= 30:
            score += 0.01

        if score > best_score:
            best_score = score
            best_params = params.copy()

    elapsed = time.time() - start
    print(f"  [OPT] Complete in {elapsed:.1f}s. Best score: {best_score:.4f}")
    print(f"  [OPT] Optimal floors: bull={best_params['strength_floor_bull']}, "
          f"bear={best_params['strength_floor_bear']}, "
          f"sideways={best_params['strength_floor_sideways']}")

    _save_optimization_results(best_params, best_score)
    return best_params


def _precompute_signals(
    df: pd.DataFrame,
    spy_df: pd.DataFrame | None,
) -> list[dict]:
    """Detect signals over last 252 bars and record forward returns."""
    from titan.signal_detector import detect_signal

    close = df["Close"]
    results = []
    min_idx = max(60, len(df) - 252)
    last_sig_idx = -999

    for i in range(min_idx, len(df) - 5):
        if i - last_sig_idx < 5:
            continue

        slice_df = df.iloc[: i + 1]
        try:
            sig_type, strength, _, _ = detect_signal(slice_df, spy_df)
        except Exception:
            continue

        if sig_type is None:
            continue

        last_sig_idx = i
        entry = float(close.iloc[i])
        if entry <= 0:
            continue

        exit_3d = float(close.iloc[min(i + 3, len(df) - 1)])
        ret_3d = (exit_3d - entry) / entry * 100

        results.append({
            "signal_type": sig_type,
            "strength": strength,
            "return_3d": ret_3d,
        })

    return results


def _save_optimization_results(params: dict, score: float) -> None:
    """Save optimized params and metadata to disk."""
    os.makedirs(CACHE_DIR, exist_ok=True)

    with open(SIGNAL_PARAMS_PATH, "w") as f:
        json.dump(params, f, indent=2)

    meta = {
        "last_run": datetime.now().isoformat(),
        "best_score": score,
        "params": params,
    }
    with open(WEEKLY_OPT_PATH, "w") as f:
        json.dump(meta, f, indent=2)
