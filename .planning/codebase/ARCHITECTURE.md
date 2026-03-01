# Architecture — Crypto Signal Model

## Overall Pattern

This is a supervised ML pipeline for generating directional trading signals on BTC/USDT (and ETH/USDT). The system is structured as a sequential, phase-based pipeline: raw market data flows through labeling, feature engineering, model training, validation, and backtesting before producing a live signal. The design is explicitly time-aware throughout — no random shuffling, no look-ahead leakage, all splits are chronological.

The pipeline is written in pure Python with pandas/sklearn/LightGBM and is executed via `main.py` with a `--phase` argument. Each phase is independently runnable but depends on artifacts (files) produced by prior phases.

---

## Pipeline Layers

### Layer 1 — Data Ingestion (`src/data_fetcher.py`, `src/external_data.py`)

**OHLCV data**: Fetched from the Binance public REST API (`/api/v3/klines`). Incremental: on each run, the fetcher checks the last stored timestamp in SQLite and resumes from there. Data is stored in `data/ohlcv.db` in a single `ohlcv` table keyed on `(symbol, interval, open_time)`. Supported symbols: BTCUSDT, ETHUSDT. Supported intervals: 1h, 4h. History depth: 730 days (`config.DAYS_HISTORY`).

**External data**: Five additional data sources are fetched and cached as Parquet files in `data/external/`:
- Fear & Greed Index (daily, alternative.me API) → `fear_greed.parquet`
- S&P500 / SPY ETF (hourly, yfinance) → `spx.parquet`
- EUR/USD exchange rate (hourly, yfinance) → `eurusd.parquet`
- BTC Perpetual Funding Rate (8-hourly, Binance Futures API) → `funding_rate.parquet`
- BTC Open Interest (hourly, Binance Futures API, limited to 30 days) → `open_interest.parquet`

Cache freshness threshold is 6 hours (`MAX_CACHE_AGE_H`). Lower-frequency data (daily F&G, 8-hourly funding) is forward-filled onto the 1h BTC index at join time.

---

### Layer 2 — P1/P2 Labeling (`src/p1p2_engine.py`)

The P1/P2 engine processes raw 1h OHLCV data into daily labels. For each UTC calendar day with at least 23 complete hours (`config.MIN_HOURS_PER_DAY`):
- P1 = the first intraday extreme (whichever of the daily high or daily low occurred first)
- P2 = the second intraday extreme (the opposite)

Each day produces a row with: `date`, `day_of_week`, `high_hour`, `low_hour`, `p1_hour`, `p1_type` (high/low), `p2_hour`, `p2_type`, `daily_high`, `daily_low`, `open_price`, `close_price`, `day_return`.

Output: `data/p1p2_labels.csv`

---

### Layer 3 — Statistical Heatmaps (`src/stats.py`)

Three heatmaps are computed from the P1/P2 label data, each indexed as (day_of_week × hour_of_day):

1. **P1 heatmap** — `P(P1 occurs at hour H | day D)`: conditional probability per day-of-week. Used as a feature (`p1_probability`) in model training.
2. **P2 heatmap** — same structure for P2. Used for visualization only.
3. **Direction bias** — `P(P1 = low | day D, hour H)`: fraction of days where P1 was a LOW (bullish day). Masked to NaN for cells with fewer than 5 observations.

**Critical anti-leakage design**: when called from the features phase, heatmaps are computed exclusively on the training portion of the data. The cutoff date is calculated as `df.index[-total_holdout_h].date()` where `total_holdout_h = (TEST_SIZE_DAYS + VALIDATION_SIZE_DAYS) * 24`. This prevents information from the validation or test period influencing the heatmaps that become model features.

Output CSVs: `data/stats/p1_heatmap.csv`, `data/stats/p2_heatmap.csv`, `data/stats/direction_bias.csv`
Output PNGs: same names with `.png` extension, plus `hourly_return_profile.png`

---

### Layer 4 — Feature Engineering (`src/features.py`)

The `build_features()` function takes the 1h OHLCV DataFrame, the P1/P2 labels, and the two heatmaps (P1 and direction_bias) and assembles the full feature matrix. It returns a DataFrame with columns = `FEATURE_COLS + ["target", "close"]`.

**Feature groups (38 total across 1h + 4h timeframes):**

- Time features: `hour`, `day_of_week`, `hour_of_week` (0-167 combined code), `session` (Asia/London/NY)
- P1/P2 statistics: `p1_probability` (looked up from heatmap per row)
- Price/volume rolling features: `volatility_24h`, `volume_ratio`, `volume_spike_48h`, `price_position`, `prev_day_return`
- Multi-horizon momentum: `return_2h`, `return_4h`, `return_6h`, `return_12h`
- Volatility regime: `vol_regime` (4h vol / 24h vol — detects expansions vs. squeezes)
- Trend quality: `trend_consistency_12h`, `buy_pressure`
- Technical indicators (1h): RSI(14), MACD(12,26,9), Bollinger %B(20), EMA ratios (20, 50), `price_vs_ema200`, ATR%
- Cross-asset features: `fear_greed`, `spx_return_24h`, `eurusd_return_24h`, `eth_btc_ratio`, `funding_rate`, `funding_momentum`
- Macro momentum: `return_7d`, `return_30d`, `ath_7d_distance`
- 4h timeframe features: `rsi_14_4h`, `macd_4h`, `bb_pct_4h`, `ema_ratio_20_4h`

**Look-ahead prevention**: 4h features are shifted by one 4h period (`shift(1)`) so that at T=04:00, only the already-closed 4h candle from 00:00 is visible. The `target` variable is computed as `close.shift(-PREDICTION_HORIZON_H)` and is therefore always in the future.

**Dead zone filtering**: rows where the absolute future price move is less than `TARGET_DEAD_ZONE_PCT` (0.3%) are labeled NaN and dropped. This removes near-neutral candles that are below the round-trip breakeven (2 × 0.1% fee = 0.2%), preventing the model from learning on noise.

- `target = 1` if `future_close > close * 1.003`
- `target = 0` if `future_close < close * 0.997`
- Neutral rows are removed

Output: `data/features.parquet`

---

### Layer 5 — Model Training (`src/model.py`, `src/model_compare.py`)

**Primary model** (`train_model()` in `src/model.py`): LightGBM classifier with fallback to RandomForest if LightGBM is not installed. LightGBM uses time-weighted training: sample weights increase linearly from 0.5 (oldest) to 1.0 (newest), giving recent market patterns more influence.

LightGBM hyperparameters: `n_estimators=400`, `max_depth=4`, `learning_rate=0.03`, `subsample=0.7`, `colsample_bytree=0.7`, `min_child_samples=80`, `reg_alpha=0.1`, `reg_lambda=1.0`.

**Model comparison** (`src/model_compare.py`): Trains RandomForest, XGBoost, and LightGBM on identical splits and computes a comparison table (ROC AUC, precision, recall, F1, Sharpe ratio, total return, win rate, trade counts). Also builds an AUC-weighted ensemble of all three models; ensemble weights are determined by each model's validation-set AUC (leakage-free because base models were not trained on the validation set).

**Threshold optimization**: Two thresholds are optimized separately on the validation set:
- Long threshold: grid search over [0.50, 0.75] maximizing Sharpe ratio with long-only signals
- Short threshold: grid search over [0.30, 0.46] maximizing Sharpe ratio with short-only signals

Optimal thresholds are saved as JSON and reloaded by the backtest and live signal phases.

Output: `data/model.pkl` (primary model), `data/model_best.pkl` (best from comparison), `data/optimal_threshold.json`, `data/stats/feature_importance.png`, `data/stats/roc_curve.png`, `data/stats/confusion_matrix.png`, `data/stats/model_comparison.csv`, `data/stats/roc_comparison.png`, `data/stats/model_comparison.png`

---

### Layer 6 — Backtest (`src/backtest.py`)

The `run_backtest()` function simulates a trading strategy on the test split:
- Long signal: model probability >= long threshold
- Short signal: model probability <= short threshold (when enabled)
- Regime filter: longs only when `price_vs_ema200 > 1.0` (price above EMA200); shorts only when below
- Position sizing: position size scales linearly with confidence: `(proba - 0.5) * 2` clipped to [0, 1]
- Stop-loss: returns are clipped at `-STOP_LOSS_PCT` (2%) per trade
- Fees: 2 × `TRADE_FEE` (0.1%) deducted per trade (entry + exit)

Metrics computed: total return, buy-and-hold return, annualized return, Sharpe ratio, max drawdown, win rate, trade counts (long/short), signal rate.

**Statistical significance**: `compute_random_baseline()` shuffles the signal column 500 times and computes the Sharpe distribution under random signals. The strategy must exceed the 95th percentile to be considered statistically significant.

Output: `data/stats/backtest_results.png` (cumulative return + drawdown chart)

---

### Layer 7 — Walk-Forward Validation (`src/backtest.py: run_walkforward()`)

Rolling-window retraining to estimate out-of-sample generalization. Configuration:
- Train window: 270 days (`WALKFORWARD_TRAIN_DAYS`)
- Test window: 30 days (`WALKFORWARD_TEST_DAYS`)
- Step size: 30 days (`WALKFORWARD_STEP_DAYS`)
- Validation window (internal to each fold): 60 days (`VALIDATION_SIZE_DAYS`)

For each fold: train on days [start, train_end - val_h], validate on [train_end - val_h, train_end], test on [train_end, train_end + test_h]. Both long and short thresholds are re-optimized per fold on the fold's internal validation slice. With 730 days of history and the above windows, approximately 11-14 folds are produced.

Output: `data/stats/walkforward_lightgbm.csv`, `data/stats/walkforward_lightgbm.png` (Sharpe per fold bar chart + combined cumulative return)

---

### Layer 8 — Horizon Scan (`src/horizon_scan.py`)

Trains and evaluates a RandomForest model for each prediction horizon in `config.HORIZON_SCAN` (default: 12h, 24h, 48h). The feature matrix is reused; only the target variable is recomputed per horizon. Uses the same train/val/test split logic. Produces a comparison table sorted by ROC AUC.

Output: `data/stats/horizon_scan.csv`, `data/stats/horizon_scan.png`

---

### Layer 9 — Live Signal (`src/backtest.py: generate_live_signal()`)

Rebuilds the feature matrix on the latest downloaded data, loads the saved model and thresholds, computes probabilities for the most recent hourly candle, and applies the regime filter and threshold logic to output one of: LONG, SHORT, WACHT (wait), or WACHT (onder EMA200 — long geblokkeerd).

Output: `data/latest_signal.json`

---

## Data Flow Between Modules

```
Binance API
    └─► data_fetcher.py ──────────────────────► data/ohlcv.db (SQLite)
                                                      │
External APIs (alternative.me, yfinance, Binance Futures)
    └─► external_data.py ────────────────────► data/external/*.parquet
                                                      │
data/ohlcv.db ──► p1p2_engine.py ───────────► data/p1p2_labels.csv
                                                      │
data/p1p2_labels.csv ──► stats.py ──────────► data/stats/{p1,p2,direction_bias}.csv
                                                      │
data/ohlcv.db + data/p1p2_labels.csv + heatmaps
    └─► features.py ─────────────────────────► data/features.parquet
                                                      │
data/features.parquet ──► model.py ─────────► data/model.pkl
                       │                    ► data/optimal_threshold.json
                       │
                       └─► model_compare.py ► data/model_best.pkl
                                            ► data/stats/model_comparison.csv
                                                      │
data/features.parquet + model.pkl + threshold.json
    └─► backtest.py ─────────────────────────► data/stats/backtest_results.png
                    └─► run_walkforward() ───► data/stats/walkforward_*.csv / *.png
                                                      │
                    └─► generate_live_signal() ► data/latest_signal.json
```

---

## Key Abstractions

| Abstraction | Location | Role |
|---|---|---|
| `build_features()` | `src/features.py` | Central transform: OHLCV + labels + heatmaps → feature matrix |
| `time_split_with_validation()` | `src/model.py` | Chronological 3-way split (train / val / test) |
| `time_split()` | `src/model.py` | 2-way split (train / test) for backward compatibility |
| `run_backtest()` | `src/backtest.py` | Unified backtest engine used by model training, walk-forward, and horizon scan |
| `compute_metrics()` | `src/backtest.py` | Sharpe, drawdown, win rate, trade counts from backtest results |
| `optimize_threshold()` | `src/model.py` | Sharpe-maximizing threshold search on validation set |
| `_get_models()` | `src/model_compare.py` | Returns all available classifier instances with lazy imports |
| `load_all_external()` | `src/external_data.py` | Aligns all external data sources to the BTC 1h index with forward-fill |
| `compute_p1p2()` | `src/p1p2_engine.py` | Transforms raw OHLCV into P1/P2 daily labels |
| `generate_live_signal()` | `src/backtest.py` | End-to-end inference on the latest candle |

---

## Entry Points — main.py Phases

`main.py` dispatches to phase functions via `--phase` argument:

| Phase flag | Function | Description |
|---|---|---|
| `data` | `fase_data()` | Download OHLCV for all symbols/intervals |
| `external_data` | `fase_external_data()` | Re-download all external data (clears cache) |
| `p1p2` | `fase_p1p2()` | Compute P1/P2 labels from OHLCV |
| `stats` | `fase_stats()` | Compute and plot heatmaps |
| `features` | `fase_features()` | Build feature matrix (Parquet) |
| `model` | `fase_model()` | Train primary model (LightGBM) + threshold optimization |
| `model_compare` | `fase_model_compare()` | Train RF/XGB/LGBM + ensemble, compare |
| `backtest` | `fase_backtest()` | Run backtest on test split |
| `walkforward` | `fase_walkforward()` | Rolling-window retraining validation (default: LightGBM) |
| `horizon_scan` | `fase_horizon_scan()` | Compare 12h/24h/48h prediction horizons |
| `signal` | `fase_signal()` | Generate live signal for the latest hour |
| `all` | (sequential) | Runs: data → p1p2 → stats → features → model → backtest → model_compare |

When run as `all`, phase functions pass in-memory DataFrames to the next phase to avoid redundant disk reads.

---

## Train / Validation / Test Split Design

The `time_split_with_validation()` function produces three non-overlapping chronological splits:

```
[─────────────────── TRAIN ───────────────────][──── VAL ────][─── TEST ───]
                                               ▲              ▲            ▲
                                         -150 days        -90 days      now
```

- **Test**: last `TEST_SIZE_DAYS` (90) days × 24 = 2,160 hourly candles
- **Validation**: preceding `VALIDATION_SIZE_DAYS` (60) days × 24 = 1,440 hourly candles
- **Train**: everything before the validation period

The validation set is used exclusively for threshold optimization (Sharpe-maximizing grid search). It is never used for model fitting. The test set is held out until final evaluation — it is not touched during training or threshold search. This three-way separation prevents the commonly seen error of fitting the threshold on the same set used for reporting.

---

## Walk-Forward Validation Design

Walk-forward uses a rolling window that simulates live retraining:

```
Fold 0: [─── TRAIN (270d) ──── VAL(60d)][─ TEST (30d) ─]
Fold 1:      [─── TRAIN (270d) ── VAL][─ TEST (30d) ─]
Fold 2:           [─── TRAIN ─── VAL][─ TEST (30d) ─]
...
```

Key properties:
- Each fold's validation slice is carved from the trailing 60 days of the training window, not from future data
- Thresholds (long and short) are re-optimized independently per fold
- Time weights (0.5 → 1.0) are applied for LightGBM and XGBoost within each fold
- All fold test results are concatenated into a single `all_results` DataFrame for plotting the combined equity curve
- Summary statistics (mean Sharpe, positive fold rate) give a generalization estimate robust to any single lucky or unlucky test period
