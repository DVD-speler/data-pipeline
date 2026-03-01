# CONCERNS.md — Technical Debt, Known Issues, and Areas of Concern

_Last updated: 2026-02-28_

---

## 1. Known Issues from lessons.md

The following bugs and design flaws were discovered and patched during development. They represent real failure modes that can recur or have left partial technical debt behind.

### 1.1 Binance OI 30-Day Cap (L1)
The `openInterestHist` endpoint returns HTTP 400 for any lookback beyond ~30 days at `period=1h`. The workaround is to cap the OI download at 30 days, which means only 4% of the 730-day training window has OI data — the remaining 96% are zeros. The feature `oi_change_24h` was therefore removed from `FEATURE_COLS` entirely. **Residual concern**: the feature is still downloaded and cached in `data/external/open_interest.parquet` and is still returned by `load_all_external()`. If it is ever re-added to `FEATURE_COLS` without fixing the coverage problem, the model will train on a near-constant feature, wasting capacity and potentially degrading importance of legitimate features (L7).

### 1.2 yfinance Timestamp Misalignment (L2)
yfinance 1h bars for US equities (SPY) start at :30 (09:30 ET = 13:30 UTC), not :00. Without the `df.index.floor("h")` normalization the join against BTC's :00-aligned index produces 0% matches and 100% NaN fill. The fix is in place in `_download_yfinance()`, but it depends on the `floor("h")` call never being removed. There is no assertion or test to guard this invariant. A future yfinance API change that returns different timestamps could silently re-introduce the bug with no warning beyond the >20% null-fraction diagnostic log line.

### 1.3 pandas 2.x datetime64 dtype Mismatch (L3)
pandas 2.x saves Parquet with `datetime64[ms, UTC]` but BTC OHLCV loaded from SQLite uses `datetime64[ns, UTC]`. When external Parquet caches are joined to the BTC index without dtype normalization, the join yields zero matches. The fix — `df_ext.index = df_ext.index.astype(base.index.dtype)` — is applied in `load_all_external()`. **Residual concern**: this is a silent failure. If pandas changes its Parquet timestamp resolution again in a future version, or if a new data source is added that skips this normalization, the bug returns invisibly. There is no test coverage for this join path.

### 1.4 Buy-and-Hold Benchmark Formula (L5)
The original B&H benchmark used overlapping h-period forward returns compounded together, producing nonsensical growth figures. The fix uses `close.pct_change()` (1h step) and `cumprod()`. The corrected formula is now in `run_backtest()`. However, the horizon scan (`horizon_scan.py` line 64) recomputes targets with a simpler formula (`close.shift(-h) > close`) without applying a dead zone, which means horizon scan results are not directly comparable to main backtest results — the target definitions differ.

### 1.5 P1/P2 Heatmap Data Leakage (L6)
The `p1_probability` and `direction_bias` features are historical statistics computed over (day_of_week, hour) pairs. If computed from the full 730-day dataset, the test period's patterns influence these training features — a form of target leakage. The fix computes heatmaps from the train-only slice using `train_cutoff = df.index[-holdout_h].date()`. **Residual concern**: the `main.py` pipeline phase ordering must ensure heatmaps are recomputed from train data before features are built. If someone runs the `stats` phase independently against the full dataset and then the `features` phase, stale/leaked heatmaps will be silently used. There is no artifact-versioning or provenance check.

### 1.6 Isotonic Calibration Overfitting (L4)
`CalibratedClassifierCV` with `method="isotonic"` collapses probabilities to step functions (0/1) when the validation set is small (~1440 rows). This forces the optimal threshold above 0.72, blocking all trades on the test set. Calibration was removed entirely as a result. If calibration is ever re-introduced, it must use a separate split — not the same set used for threshold optimization. The walk-forward loop (`run_walkforward()`) never had calibration in the first place, creating an inconsistency between the main model pipeline and walk-forward evaluation (L8).

### 1.7 Walk-forward Calibration Inconsistency (L8)
`run_walkforward()` instantiates fresh models per fold without calibration. The main `train_model()` path also now skips calibration, so currently there is no mismatch. However, the code architecture does not enforce this: calibration added to `train_model()` would not automatically propagate to `run_walkforward()`. This is structural debt — the two code paths are not unified.

### 1.8 Short Threshold Grid Floating-Point Artifacts (L13)
`np.arange(0.30, 0.46, 0.01)` can include 0.46 due to floating-point endpoint accumulation, producing a short threshold of 0.46 which is too permissive (L12). The fix uses `np.linspace(..., endpoint=False)`. Both `optimize_threshold` and `optimize_short_threshold` now use `linspace`. Any future threshold grid code that reverts to `arange` with float steps will silently re-introduce this off-by-one.

---

## 2. Model Performance Issues

### 2.1 Low Discriminative Power
The primary LightGBM model achieves a test-set ROC AUC of **0.5296** — barely above chance (0.50). This is the fundamental ceiling of the approach: BTC 1h price movements are close to a random walk. All downstream decisions (threshold at 0.74, trading at 53 longs over 90 days) are consequences of this weak signal.

### 2.2 Test Period Is a Single Regime (L10, L15)
The 90-day test period (November 2025 – February 2026) was a sustained bear market (B&H = -36.9%). The strategy's +16.1% return and Sharpe of +1.53 are impressive in relative terms but reflect regime-fitting, not all-weather robustness. The bear market was ideal for the short-signal component: walk-forward folds 7 and 8 (bear regime) show large Sharpe improvements from shorts, while fold 2 (May–June 2025 recovery) shows shorts actively hurting performance (Sharpe +4.2 → -13.3).

### 2.3 Walk-Forward Performance Is Poor
Walk-forward mean Sharpe is **-3.79** with only 2/9 positive folds. This is the honest all-weather assessment. The model does not generalize well across market regimes. Walk-forward improvement from RF long-only (-7.23) to LightGBM with shorts (-3.79) is real progress but the absolute level is still deeply negative.

### 2.4 High Long Threshold Creates Sparse Signal
The optimal long threshold of 0.74 is far from the 0.5 decision boundary. This is a regularization artefact: with a highly regularized model (max_depth=4, min_child_samples=80), the model rarely outputs probabilities above 0.74, producing only 53 long trades over 90 days (one every ~1.7 days). This sparsity reduces statistical power and makes the walk-forward fold-level Sharpe estimates very noisy.

### 2.5 Feature Importance Concentration
The top 5 features (prev_day_return 9.9%, spx_return_24h 8.5%, return_30d 7.0%, volatility_24h 6.5%, funding_momentum 5.0%) account for ~37% of total importance. The remaining 33 features share the other 63%. Features with near-zero importance risk being constant or near-constant (L7) rather than genuinely uninformative. No automated feature importance monitoring exists.

### 2.6 Regime Filter Blindspot (L14)
The EMA200 regime filter (long only above EMA200, short only below) cannot distinguish a sustained downtrend from a short-covering bounce. Short signals help in true bear markets but hurt during recoveries — both occur below EMA200. This is a structural limitation of a single binary regime signal with no momentum or velocity component.

---

## 3. Data Pipeline Fragility

### 3.1 Open Interest 30-Day History Cap
The Binance `openInterestHist` API enforces a ~30-day lookback hard limit for 1h granularity. This is an external API constraint with no workaround. If OI is ever reinstated as a feature, the model will train on 96% zeros, which is worse than omitting the feature entirely. A more robust approach would be to stream and accumulate OI data continuously, but no such persistence mechanism exists.

### 3.2 yfinance Fragility
SPY and EURUSD are sourced from yfinance with `period="730d"` and `interval="1h"`. yfinance is an unofficial reverse-engineered API that breaks with Yahoo Finance changes. Key fragility points:
- The 730-day 1h history limit may be enforced differently across yfinance versions.
- MultiIndex column flattening (`df.columns.get_level_values(0)`) is required for newer versions and may break again.
- The :30-minute timestamp behavior is undocumented and could change.
- No retry logic beyond the stale-cache fallback.

### 3.3 Datetime dtype Brittleness
The `load_all_external()` join path relies on `df_ext.index = df_ext.index.astype(base.index.dtype)` to normalize datetime precision. This is a fragile one-liner that silently succeeds or fails depending on pandas version behavior. There is no assertion that the dtypes match after the cast, and no test covering this code path.

### 3.4 Silent Fallback to Neutral Defaults
When any external data source fails, `load_all_external()` fills with neutral defaults (fear_greed=0.5, spx_return=0.0, etc.) and continues. This means a broken data source produces a silently degraded feature matrix. The 20% null-fraction warning is the only signal that something went wrong — it does not halt the pipeline or produce a clear error.

### 3.5 Cache Staleness Logic
The `MAX_CACHE_AGE_H = 6` cache policy means external data can be up to 6 hours stale during live operation. For the funding rate (8h resolution) and fear & greed (daily), this is acceptable. For SPY and EURUSD (1h), a 6-hour stale cache during live trading means the model may use significantly outdated market context.

### 3.6 SQLite Concurrency
The OHLCV data is stored in a SQLite file (`data/ohlcv.db`). SQLite supports only one writer at a time. If the data fetcher and any other process attempt concurrent writes (e.g., a scheduled refresh daemon alongside a pipeline run), writes will fail or block. There is no connection pooling or WAL mode configuration.

### 3.7 No Data Integrity Checks
There are no checksums, row-count assertions, or gap-detection checks on the OHLCV data after download. Missing candles (e.g., due to Binance downtime) would silently produce NaN-propagated features downstream. The `MIN_HOURS_PER_DAY` guard in `compute_p1p2()` is the only data completeness check in the pipeline.

---

## 4. ML / Statistical Concerns

### 4.1 Threshold Optimization on Validation Set Is In-Sample Fitting
The long threshold (0.74) and short threshold (0.45) are both selected by maximizing Sharpe on the validation set. This is a form of hyperparameter overfitting. The validation set has 1440 rows (60 days). With only ~50–80 trades occurring on the validation set at these thresholds, the Sharpe estimate is high-variance, and the optimal threshold may not generalize. There is no second-level hold-out to assess threshold stability.

### 4.2 Walk-Forward Threshold Re-Optimization Per Fold
Each walk-forward fold independently optimizes both the long and short thresholds on its internal validation slice. This means each fold uses a different threshold, which is realistic for a live system that would re-optimize periodically. However, it also means the walk-forward result aggregates strategies with different signal definitions — comparing Sharpe across folds conflates model quality with threshold variation.

### 4.3 Horizon Scan Uses Different Target Definition
`scan_horizons()` recomputes the target as `close.shift(-h) > close` — a simple binary without the dead zone. The main model uses a three-class target with a 0.3% dead zone that removes 21.6% of rows. The horizon scan results are therefore not directly comparable to main model results: they have different class balance, different row counts, and different implied fee viability.

### 4.4 Position Sizing Based on Raw Probability
When `use_position_sizing=True`, position size is computed as `(proba - 0.5) * 2` clipped to [0, 1]. At a threshold of 0.74, the minimum position size for a long entry is `(0.74 - 0.5) * 2 = 0.48` — roughly half a position. This means a signal just above threshold trades at 48% size, while a proba of 0.99 trades at full size. The scaling is linear and uncalibrated — it treats raw model probabilities as if they were well-calibrated confidence scores, which they are not (AUC 0.53 implies very poor calibration).

### 4.5 Random Baseline Only Tests Long Signals
`compute_random_baseline()` shuffles the signal column and applies fees only to positive (long) signals: `strat_ret = np.where(shuffled_signal > 0, trade_return - 2 * fee, 0.0)`. It ignores short signals in the shuffled baseline. This means the significance test for a strategy with both long and short components is not correctly specified — the p95 threshold of 0.508 was computed for a long-only random baseline.

### 4.6 Data Leakage Risk in Feature Construction
Several features use rolling windows that could introduce subtle forward-looking patterns if the index contains gaps. For example, `df["return_30d"] = df["close"].pct_change(30 * 24)` computes a 720-row lookback without checking whether those 720 rows are contiguous in time. If the OHLCV data has gaps (missing candles), `pct_change(720)` will compare the current close to a close that is more than 30 days earlier, inflating the apparent momentum signal. No gap detection is performed before feature engineering.

### 4.7 ETH/BTC Ratio Computed from Full History
The ETH/BTC ratio feature (`eth_btc_ratio`) is computed as `df_eth["close"] / df_ohlcv["close"]` joined on the full BTC index. There is no train/test awareness here — the ratio is computed across all available data before the split. While this does not directly cause label leakage (the ratio is used as a lagged momentum feature via `pct_change(24)`), the statistics of the full-history ratio are implicitly used when joining, which is a minor concern.

---

## 5. Live Signal Gaps

The `generate_live_signal()` function in `backtest.py` provides a stub for live operation, but several critical components are missing for production deployment.

### 5.1 No Live Data Ingestion Pipeline
`generate_live_signal()` calls `build_features(df_ohlcv, ...)` but does not specify how `df_ohlcv` is populated with the current candle. In a live deployment, a scheduler would need to fetch the latest completed 1h candle from Binance and append it to the SQLite database every hour, then re-run feature engineering and inference. No such scheduler, cron job, daemon, or streaming connection exists. The `download_ohlcv()` function is batch-only.

### 5.2 Model Staleness
The saved model (`data/model.pkl`) is a static artifact trained at a point in time. In live operation, the market regime shifts and the model degrades. Walk-forward results show significant variance across 30-day folds. There is no automated retraining schedule, no model versioning, and no mechanism to detect when model performance has decayed enough to trigger a retrain.

### 5.3 No Position State Tracking
The backtest simulates non-overlapping h-horizon positions. In reality, live signals can fire every hour and a new signal may occur before the previous position's 12h horizon expires. There is no position management layer: no tracking of open positions, no prevention of duplicate entries, and no logic to exit a position early based on a new opposing signal.

### 5.4 No Order Execution Integration
There is no exchange API integration for order placement. `generate_live_signal()` returns a Python dict with a signal label — it cannot place, modify, or cancel orders. Connecting to Binance Futures via their API (with authentication, order sizing, risk checks, and error handling) is entirely absent.

### 5.5 No Monitoring or Alerting
There is no runtime monitoring for:
- Signal delivery failures (e.g., data fetch timeout)
- Unexpected signal rate changes (e.g., the model suddenly generating 10x more longs)
- Drawdown thresholds being breached
- Model output distribution shifts (probability distribution drift)

No alerting channel (email, Slack, webhook) is configured.

### 5.6 Threshold Validity in Live Context
The optimal thresholds (long: 0.74, short: 0.45) were calibrated on the November 2024 – August 2025 validation period. In a live context (post-February 2026), these thresholds may no longer be appropriate if the regime has changed. There is no automatic threshold decay monitoring or re-optimization trigger.

---

## 6. Missing Infrastructure

### 6.1 No Automated Tests
There are zero test files in the repository. No unit tests, no integration tests, no regression tests. Every bug in lessons.md was discovered through manual inspection of outputs — not through a test suite. Given the number of subtle data bugs already found (dtype mismatches, timestamp misalignment, benchmark formula errors, leakage), the absence of tests represents significant ongoing risk.

Critical paths that lack test coverage include:
- `load_all_external()` join correctness with mismatched dtypes
- `build_features()` producing the correct number of rows and no look-ahead
- `run_backtest()` return calculation with stop-loss and position sizing
- `compute_metrics()` Sharpe annualization formula
- `optimize_threshold()` returning a value within [thr_min, thr_max]

### 6.2 No CI/CD Pipeline
There is no `.github/workflows/`, `Makefile`, `tox.ini`, or any other automated pipeline configuration. Pipeline phases are run manually by calling `python main.py --phase <phase>`. A developer introducing a regression has no automated safety net. Re-running the full pipeline takes substantial time (data download, feature engineering, model training, walk-forward across 9 folds), making ad-hoc testing slow.

### 6.3 No Dependency Pinning
There is no `requirements.txt` or `pyproject.toml` with pinned dependency versions. The codebase has already been affected by pandas 2.x API changes (L3) and yfinance API changes (L2). Without version pinning, the environment is not reproducible and future `pip install` or `pip upgrade` operations may silently break the pipeline.

### 6.4 No Logging Framework
All output is via `print()` statements. There is no structured logging (no `logging` module, no log levels, no log files). This makes it impossible to:
- Reproduce the exact output of a past pipeline run
- Filter warnings from informational output
- Capture errors to a persistent log for post-mortem analysis
- Configure verbosity without code changes

### 6.5 No Configuration Validation
`config.py` is imported directly as a module with no validation of parameter ranges or internal consistency. For example:
- `VALIDATION_SIZE_DAYS + TEST_SIZE_DAYS` must be less than `WALKFORWARD_TRAIN_DAYS` for the walk-forward to have sufficient data, but this is never asserted.
- `SIGNAL_THRESHOLD_SHORT` defaults to 0.0 (disabled) but the threshold optimization can set it to any value in [0.30, 0.45]; if config and the JSON file diverge, behavior is non-obvious.
- `TARGET_DEAD_ZONE_PCT` (0.003) must be above `TRADE_FEE * 2` (0.002) for the dead zone to be meaningful; this invariant is never checked.

### 6.6 No Reproducibility Guarantees
While `random_state=42` is set for all sklearn models and `seed=42` for the random baseline, LightGBM's parallel training (`n_jobs` equivalent via OpenMP threads) can produce non-deterministic results across runs on different hardware. There is no mechanism to fix the thread count for deterministic builds.

### 6.7 Global State Mutation in horizon_scan.py
`scan_horizons()` temporarily overwrites `config.FEATURE_COLS` (a module-level global) with `available_features` and restores it afterward (lines 93–96). This is a thread-unsafe pattern. If the horizon scan were ever run concurrently with another pipeline phase in the same Python process, it would corrupt the global feature list mid-execution with no error.

---

## 7. Summary of Highest-Priority Concerns

| Priority | Concern | Impact |
|----------|---------|--------|
| Critical | No automated tests | Any refactor can silently break data joins, leakage guards, or backtest math |
| Critical | Walk-forward Sharpe -3.79 (2/9 folds positive) | Core strategy does not generalize across regimes |
| High | No live data pipeline | Production deployment is not possible without significant new engineering |
| High | Threshold overfitting on 1440-row validation set | Live threshold validity is uncertain |
| High | Single-regime test period | Test-set +16.1% return is not evidence of all-weather performance |
| Medium | Silent fallback to neutral defaults | Data source failures produce degraded models without clear errors |
| Medium | No dependency pinning | Environment drift will break pipeline silently |
| Medium | Global state mutation in horizon_scan | Unsafe if any parallelism is introduced |
| Low | OI feature excluded due to 4% coverage | Signals a missing data source; alternative OI sources (e.g., Coinalyze) not explored |
| Low | No structured logging | Debugging pipeline failures in production requires code changes |
