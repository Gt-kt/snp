# Titan Trade v10.0 — Next-Generation Architecture
## "Stalk, Don't Chase"

### Author Context
Solo trader in Korea. Scans at 11:30 PM KST (US market open).
Places orders and goes to sleep. Trades once every 3-4 days.
$100K account. Needs pre-breakout signals, not post-breakout chasing.

---

## 1. ARCHITECTURE OVERVIEW

### Current v9 Flow (Chase Model)
```
Scan 500 stocks → Find confirmed breakouts → Enter next morning → Hope it keeps going
                  ^^^^^^^^^^^^^^^^^^^^^^^^
                  PROBLEM: The move already happened
```

### New v10 Flow (Stalk Model)
```
Phase A: Nightly Universe Ranking (runs every scan)
  500 stocks → Feature extraction → ML ranker → Top 30 "coiling" candidates

Phase B: Stalking Pipeline (the core innovation)
  Top 30 → Pre-breakout scoring → Compression analysis → Trigger proximity
        → Volume dry-up detection → Institutional footprint → Sector wind
        → Output: 3-8 "place this limit order tonight" signals

Phase C: Order Management (fire and sleep)
  Signals → GTC limit orders at pivot → Bracket: entry + stop + target
         → Auto-manage via Alpaca → Morning review via dashboard

Phase D: Position Lifecycle (hands-off)
  Filled → Trail stop to breakeven at 1R → Partial exit at 1.5R
        → Trail remainder → Full exit at target or 5-day timeout
```

### Key Principle: Separate RANKING from ENTRY TIMING
- ML handles ranking (which stocks are most likely to break out soon)
- Rules handle timing (where exactly to place the limit order)
- This avoids the classic ML overfitting trap of trying to predict exact prices

---

## 2. SIGNAL PIPELINE (What Changes, What Stays)

### KEEP (These Work)
- OOS validation (last 25% holdout)
- Walk-forward robustness (3-fold, but upgrade to 5-fold)
- Regime stability scoring
- Cost-adjusted backtesting (15bps round-trip)
- Gap protection filter
- Sector rotation (top N sectors)
- Statistical confidence grading (A-F)
- Earnings blackout filter
- Portfolio heat management

### REPLACE

#### 2a. Entry Model: Confirmed Breakout → Pre-Breakout Stalking

OLD: Wait for breakout confirmation (volume surge + close above pivot)
NEW: Score the SETUP QUALITY before breakout, place limit at pivot

```
Pre-Breakout Stalking Score (0-100):

Compression Signals (0-30 pts):
  - 10-day range / 30-day range ratio (compression_ratio)
    ≤ 0.50: 15 pts (extremely tight — spring is loaded)
    ≤ 0.65: 10 pts
    ≤ 0.80:  5 pts
  - 5-day range / 10-day range (nested compression)
    ≤ 0.60: 15 pts (tightening within the tight range)
    ≤ 0.75: 10 pts
    ≤ 0.90:  5 pts

Proximity Signals (0-20 pts):
  - Distance to pivot (15-day high + 0.02)
    ≤ 1.0%: 12 pts (about to touch the trigger)
    ≤ 2.0%:  8 pts
    ≤ 3.5%:  5 pts
  - Price vs AVWAP (anchored from highest volume day)
    Above AVWAP:  8 pts (institutions are in profit, won't dump)
    Within 1%:    4 pts

Institutional Footprint (0-25 pts):
  - Volume dry-up in base (low volume = supply absorbed)
    Recent 5-day vol < 60% of 50-day avg: 10 pts
    Recent 5-day vol < 80% of 50-day avg:  5 pts
  - Up-volume ratio (from accumulation score)
    ≥ 1.8: 10 pts
    ≥ 1.3:  6 pts
  - OBV trend (rising OBV with flat price = stealth accumulation)
    OBV 10d > OBV 20d SMA:  5 pts

Momentum Quality (0-25 pts):
  - RS percentile (vs full S&P 500 universe)
    ≥ 85: 12 pts
    ≥ 75:  8 pts
    ≥ 65:  4 pts
  - Multi-timeframe momentum alignment
    21d, 63d, 126d all positive: 8 pts
    Two of three positive:       4 pts
  - EMA stack (10 > 21 > 50 > 200)
    Full stack:  5 pts
    Partial:     2 pts
```

#### 2b. Ranking Model: Rule-Based Score → Gradient Boosted Ranker

Purpose: Given 30-50 candidates that pass basic filters, RANK them by
probability of breaking out within 5 trading days.

Model: LightGBM ranker (not classifier — we rank, not predict)

Features (all point-in-time, no lookahead):
```
Technical:
  - compression_ratio_10_30     (range contraction)
  - compression_ratio_5_10      (nested contraction)
  - distance_to_pivot_pct       (how close to breakout)
  - atr_percentile_60d          (volatility regime of the stock)
  - price_vs_ema21_pct          (trend tightness)
  - price_vs_avwap_pct          (institutional profit/loss)
  - volume_dry_up_ratio         (5d vol / 50d vol)
  - obv_slope_10d               (stealth accumulation)

Fundamental/Flow:
  - rs_percentile_63d           (relative strength rank)
  - momentum_composite          (multi-timeframe momentum)
  - accumulation_score          (up-volume ratio)
  - sector_rank_20d             (sector momentum)
  - days_since_last_pivot_test  (freshness of the pattern)
  - earnings_days_away          (distance to earnings event)

Market Context:
  - spy_roc_5d                  (short-term market direction)
  - vix_level                   (fear gauge)
  - vix_term_structure          (contango/backwardation)
  - breadth_pct_above_50sma     (market internals)
  - sector_rs_vs_spy_21d        (sector wind direction)

Target variable for training:
  - max_gain_5d: maximum (High[t+1:t+6] - Close[t]) / Close[t]
  - Label: 1 if max_gain_5d >= 3%, else 0
  - This is FORWARD-LOOKING during training only
  - At inference (live scan), we only use features
```

Training approach:
  - Use 3 years of historical data
  - Walk-forward: train on years 1-2, validate on year 3
  - Retrain monthly (not daily — avoid overfit)
  - Use NDCG as ranking metric, not accuracy
  - Feature importance monitoring: drop features that contribute < 2%

Why LightGBM over deep learning:
  - 500 stocks × 750 days = 375K samples. Not enough for neural nets.
  - LightGBM handles mixed feature types natively
  - Fast training (< 30 seconds), can retrain weekly
  - Interpretable: feature importance tells you WHY it ranked a stock high

#### 2c. Regime Detection: SMA Cross → Multi-Signal Regime Model

OLD: SMA50/SMA200 cross on SPY (lags by weeks)
NEW: Composite regime score using fast + slow signals

```
Fast Signals (react in 1-3 days):
  - VIX level (current)
  - VIX term structure: VIX / VIX3M ratio
    > 1.0 = backwardation = FEAR (market expects near-term vol)
    < 0.85 = steep contango = COMPLACENT (calm but fragile)
  - SPY 5-day ROC
  - NYSE advance/decline ratio (5-day)

Medium Signals (react in 1-2 weeks):
  - % of S&P 500 above their own 20-day SMA (breadth)
  - % of S&P 500 above their own 50-day SMA (breadth)
  - SPY vs its 21-day EMA

Slow Signals (trend confirmation):
  - SPY SMA50 vs SMA200 (keep this — it works for trend)
  - SPY 200-day SMA slope (rising/falling)

Regime Classification:
  RISK_ON:     Fast bullish + Medium bullish + Slow bullish
  RISK_ON_CAUTIOUS: Fast neutral + Medium bullish + Slow bullish
  TRANSITIONAL: Mixed signals (reduce size, don't stop trading)
  RISK_OFF:    Fast bearish OR Medium bearish + Slow bearish
  CRISIS:      VIX > 30 OR breadth < 30% (no new longs)
```

---

## 3. RISK MODEL

### What Changes

#### 3a. Adaptive Position Sizing (Replace Fixed $500 Risk)

```python
def calculate_position_risk(setup, regime, account_equity):
    # Base risk: half-Kelly, capped
    kelly_frac = setup.kelly_fraction * 0.5  # half-Kelly for safety
    kelly_risk = account_equity * kelly_frac

    # Regime scalar
    regime_scalars = {
        'RISK_ON': 1.0,
        'RISK_ON_CAUTIOUS': 0.7,
        'TRANSITIONAL': 0.5,
        'RISK_OFF': 0.25,
        'CRISIS': 0.0,
    }
    regime_risk = kelly_risk * regime_scalars[regime]

    # Confidence scalar (A=1.0, B=0.8, C=0.6, D=0.3, F=0)
    conf_scalar = {'A': 1.0, 'B': 0.8, 'C': 0.6, 'D': 0.3}.get(setup.grade, 0)
    sized_risk = regime_risk * conf_scalar

    # Hard caps
    max_risk = account_equity * 0.01  # never risk > 1% per trade
    max_position = account_equity * 0.12  # never > 12% in one name
    return min(sized_risk, max_risk)
```

#### 3b. Portfolio-Level Risk (New)

```
- Correlation-aware: Don't hold 4 energy breakouts simultaneously
  Max 2 positions in same sector
  Max 3 positions with pairwise correlation > 0.7

- Volatility targeting: Size the PORTFOLIO to target 8% annual vol
  If portfolio vol > 10%, reduce all positions proportionally
  If portfolio vol < 6%, allow slightly larger new positions

- Drawdown throttle: Progressive position reduction
  Drawdown 0-5%:   Normal sizing
  Drawdown 5-8%:   Half size
  Drawdown 8-12%:  Quarter size
  Drawdown >12%:   No new trades, manage exits only
```

#### 3c. Stop Loss: Fixed ATR → Volatility-Adaptive

```
OLD: stop = entry - (ATR * 2.0)  (same multiplier regardless)
NEW: stop = entry - (ATR * adaptive_mult)

Where adaptive_mult considers:
  - Stock's own volatility regime (ATR percentile over 60 days)
    Low vol (ATR pct < 30):  mult = 1.5 (tight stop, stock is calm)
    Normal (30-70):          mult = 2.0
    High vol (> 70):         mult = 2.5 (wider stop, stock is jumpy)

  - Market regime
    RISK_ON: use calculated mult
    CAUTIOUS/TRANSITIONAL: mult * 0.85 (tighter, protect capital)
    RISK_OFF: mult * 0.70

  - Setup quality
    A-grade: use calculated mult (trust the setup)
    B-grade: mult * 0.95
    C-grade: mult * 0.85 (tighter, less conviction)
```

---

## 4. BACKTESTING FRAMEWORK IMPROVEMENTS

### 4a. Fix Survivorship Bias

```
Solution: Historical S&P 500 constituent data

Option 1 (simple): Download historical constituent lists from
  Wikipedia's S&P 500 change history. For each backtest date,
  only include stocks that were IN the index at that time.

Option 2 (pragmatic): Apply a survivorship discount
  - Estimate 5-8% annual turnover in S&P 500
  - Apply a flat 3-5% penalty to backtest win rates
  - This is imperfect but honest

We'll implement Option 2 first (immediate), Option 1 later.
```

### 4b. Walk-Forward: 3 Folds → 5 Expanding Window Folds

```
OLD: 3 fixed-size folds, 20% test ratio
NEW: 5 expanding-window folds

Year 1-2 train → Year 3 test (fold 1)
Year 1-3 train → Year 4 test (fold 2, more training data)
Year 1-4 train → Year 5 test (fold 3)
...

This better simulates real trading: you always have MORE history
as time goes on, and you test on progressively recent data.
```

### 4c. Realistic Execution Simulation

```
OLD: Entry at open, flat slippage
NEW: Limit order fill simulation

For pre-breakout entries:
  - Place limit order at pivot price
  - Check if High[t+1:t+5] >= pivot (order would fill)
  - If fill: entry_price = pivot (limit order, exact price)
  - Apply 2bps slippage on limit fills (much less than market orders)
  - Track fill rate: what % of orders actually fill?
  - If fill rate < 30%, the pivot is too aggressive — adjust

For stop exits:
  - If Low[t] <= stop: exit at stop price (stop order)
  - Apply 5bps adverse slippage on stops (gap-through risk)
  - Check for gap-through: if Open[t] < stop, exit at Open (worse fill)

For target exits:
  - If High[t] >= target: exit at target (limit order, exact price)
  - No slippage on target limit orders
```

### 4d. Monte Carlo Confidence Intervals

```
After running walk-forward backtest:
  - Bootstrap the trade results (resample with replacement)
  - Run 1000 bootstrap iterations
  - Calculate 5th percentile return (worst realistic case)
  - Calculate 95th percentile return (best realistic case)
  - Report: "Expected annual return: 12-28% (90% CI)"
  - This gives honest uncertainty bounds instead of single-point estimates
```

---

## 5. OUTPUT FORMAT (What You See at 11:30 PM KST)

### The Nightly Scan Report

```
═══════════════════════════════════════════════════════════
  TITAN TRADE v10.0 — NIGHTLY STALK REPORT
  2026-03-27 23:30 KST (10:30 AM ET)
  Regime: RISK_ON_CAUTIOUS | VIX: 18.2 | Breadth: 62%
═══════════════════════════════════════════════════════════

  LIMIT ORDERS TO PLACE TONIGHT (3 signals):

  #1  NVDA  |  A-Grade  |  ENERGY SECTOR (TOP)
      Stalk Score: 94/100  |  ML Rank: #2 of 487
      Compression: 0.42 (very tight)  |  Dist to pivot: 0.8%
      ┌─ LIMIT BUY:  $892.50  (pivot)
      ├─ STOP LOSS:  $871.20  (2.4% risk, $21.30/share)
      ├─ PARTIAL:    $924.40  (1.5R, sell 40%)
      └─ TARGET:     $960.80  (3.2R, sell remainder)
      Shares: 23 starter + 12 add-on @ $895.00
      Risk: $490 (0.49% of account)
      Fill probability: ~65% within 3 days
      Backtest: WR 58% | PF 1.82 | OOS 1.65 | Rob 78

  #2  COST  |  B-Grade  |  CONSUMER STAPLES (TOP)
      ...

  STALKING (not ready yet, watch for 1-3 days):

  #1  MSFT  |  Compression tightening, 2.1% from pivot
      ...

  DO NOT TRADE (regime filter / weak setup):
      ...
```

### Dashboard Changes

The live dashboard should show:
- Active limit orders (pending fill)
- Stalking candidates (approaching pivot)
- Filled positions (with P&L and trail stop status)
- Regime gauge (fast/medium/slow signals)
- Fill rate tracker (what % of your limit orders fill)

---

## 6. IMPLEMENTATION PHASES

### Phase 1: Pre-Breakout Stalking Engine (Week 1)
- Restructure process_ticker() to output pre-breakout signals
- Lower early entry robustness threshold (85 → 65)
- Add compression trajectory tracking
- Add limit-order-at-pivot output format
- Add bracket order generation for Alpaca
- Modify dashboard to show stalking candidates
- NO ML yet — pure rule-based improvements

### Phase 2: Enhanced Regime + Adaptive Risk (Week 2)
- Add VIX term structure (VIX/VIX3M ratio)
- Add breadth indicators (% above 50-SMA)
- Build composite regime model
- Implement adaptive position sizing (half-Kelly)
- Implement drawdown throttle
- Implement volatility-adaptive stops

### Phase 3: ML Ranking Model (Week 3-4)
- Add lightgbm, scikit-learn to requirements
- Build feature extraction pipeline
- Train walk-forward ranker on 3 years of data
- Integrate ranker into scan pipeline
- A/B test: ML-ranked vs rule-ranked for 2 weeks
- Feature importance monitoring

### Phase 4: Backtester Upgrades (Week 4-5)
- Limit order fill simulation
- Survivorship bias adjustment
- 5-fold expanding window walk-forward
- Monte Carlo confidence intervals
- Correlation-aware portfolio risk

---

## 7. EXPECTED TRADE-OFFS

### What Gets Better
- Signal timing: catching stocks 1-3 days BEFORE the move
- Fill quality: limit orders at pivot vs market orders after gap
- Risk management: adaptive sizing instead of fixed $500
- Regime sensitivity: faster reaction to market shifts
- Honest backtests: survivorship discount, limit fill simulation
- Sleep quality: bracket orders manage themselves

### What Gets Worse (Honest Assessment)
- Fill rate: maybe only 40-60% of limit orders fill (vs 100% for market orders)
  → This is a feature, not a bug. Unfilled orders = the stock didn't break out = you avoided a bad trade
- Complexity: ML ranking adds a model to maintain and monitor
  → Mitigated by monthly retraining and feature importance checks
- Fewer trades: pre-breakout stalking is more selective
  → Matches your 3-4 day frequency perfectly
- Backtest results will look WORSE:
  → Because they're more honest (survivorship discount, limit fills)
  → Real-world P&L should actually improve despite worse backtest numbers

### What Stays the Same
- Core validation pipeline (OOS, walk-forward, robustness)
- Sector rotation logic
- Earnings blackout protection
- Portfolio heat management
- Alpaca integration
- Dashboard infrastructure

---

## 8. DEPENDENCIES TO ADD

```
# requirements.txt additions for v10
lightgbm>=4.0          # ML ranking model
scikit-learn>=1.4       # Feature preprocessing, metrics
joblib>=1.3             # Model serialization
```

No deep learning. No GPU required. Runs on any laptop.

---

## 9. SUCCESS METRICS

After 3 months of live trading v10:

| Metric | v9 Baseline | v10 Target |
|--------|------------|------------|
| Trades per month | 8-12 | 5-8 (more selective) |
| Win rate | ~50% (inflated) | 45-55% (honest) |
| Profit factor | ~1.3 | 1.5-2.0 |
| Average winner | 2-3% | 3-5% (better entries) |
| Average loser | 2-3% | 1.5-2% (tighter stops on pre-breakout) |
| Max drawdown | unknown | < 10% (drawdown throttle) |
| Fill rate | 100% (market orders) | 50-65% (limit orders) |
| Sleep quality | anxious | peaceful |
