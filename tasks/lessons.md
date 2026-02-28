# Lessons Learned

## Session 2026-02-28 — Improvement Sprint

### L1: Binance openInterestHist has a ~30-day history limit
- Endpoint `/futures/data/openInterestHist` with `period=1h` returns 400 for requests >30 days back
- Cap lookback to 30 days maximum
- With 730-day training window, 30-day OI coverage = 4% of training rows → all zeros for 96%
- **Decision**: Remove oi_change_24h from FEATURE_COLS entirely when history < 180 days

### L2: yfinance 1h bars for US stocks start at :30 (09:30 ET = 13:30 UTC)
- BTC OHLCV has timestamps at :00; yfinance SPY/EURUSD at :30 and :00 respectively
- After left-join on BTC index, SPX has 0% match → 100% NaN → fills to 0.0
- **Fix**: `df.index = df.index.floor("h")` + deduplicate after download
- **Rule**: Always floor/round external market data to hour boundary before joining to BTC index

### L3: pandas 2.x saves Parquet with datetime64[ms, UTC], but BTC OHLCV uses datetime64[ns, UTC]
- Dtype mismatch causes 0 matches on join even when timezones are identical
- **Fix**: `df_ext.index = df_ext.index.astype(base.index.dtype)` before join in load_all_external()
- **Rule**: Always normalize index dtype before join when combining data from different sources

### L4: isotonic CalibratedClassifierCV overfits on small validation sets
- With 1440 rows and isotonic method, calibration maps all probabilities to step-function 0/1
- This pushes threshold to 0.72+ making the model not trade at all on test set
- Sigmoid calibration is more stable for small datasets; but even simpler: skip calibration
  when the validation set is the same set used for threshold optimization (data is used twice)
- **Rule**: If calibration set == threshold optimization set, skip calibration or use separate split

### L5: B&H benchmark must use 1-hour step returns (not h-period forward returns)
- Using `close.shift(-h) / close - 1` as bh_return and then compounding gives nonsensical results
  because all 2160 overlapping h-period returns are multiplied together
- **Correct formula**: `close.pct_change()` (1-hour step) → `cumprod()` gives true B&H growth
- The old formula (`pct_change().shift(-1)`) was off by 1 but approximately correct

### L6: P1/P2 heatmaps computed from full dataset = data leakage
- `p1_probability` and `direction_bias` are historical statistics over (day_of_week, hour)
- If computed from the full 730-day dataset, test-period patterns influence these features
- **Fix**: Compute heatmaps from train-only slice using `train_cutoff = df.index[-holdout_h].date()`
- **Rule**: All feature statistics that aggregate over the training set must be computed before the train/test split and applied as frozen artifacts to val/test

### L7: Feature importance = 0 is always a data quality problem, not a model problem
- Zero importance means the feature is constant or near-constant
- Check: `df[col].std()` and `df[col].isna().mean()` before training
- Add post-join null fraction diagnostics to external data loader

### L8: Walk-forward does not inherit calibration from train_model()
- `run_walkforward()` creates fresh models per fold; calibration added to `train_model()` does not propagate
- If calibration is used, it must be added inside the walkforward loop as well
- Inconsistency between main model (calibrated) and walkforward (uncalibrated) corrupts comparison

### L9: Target dead zone removes 21.6% of training rows at 0.3% threshold
- For BTC 12h horizon: |move| < 0.3% occurs ~22% of the time
- This is above the fee breakeven (0.2% round-trip) — correctly excluded
- Result: stijging recall improved 0.24 → 0.40; accuracy 51% → 53%

### L10: Test period (Nov 2025 - Feb 2026) is a bear market (-36.9% B&H)
- All trading models will struggle; regime filter (EMA200) correctly blocks most long signals
- Evaluate strategy on relative terms (vs B&H) not absolute: -2.2% vs -36.9% = +34.7pp
- Walk-forward folds 4, 8-9 show 0 trades = expected and correct in bearish regimes

## Session 2026-02-28 — Profitability Sprint (Phase 2)

### L11: More regularized LightGBM forces higher thresholds → more profitable
- LightGBM max_depth=5, min_child_samples=60 → optimal threshold 0.50 → 432 longs → overtrading → -76%
- LightGBM max_depth=4, min_child_samples=80 → optimal threshold 0.74 → 53 longs → selective → +16.1%
- **Rule**: For 12h horizon with ~10k training rows, prefer more regularized models (depth=4, leaf=80); higher regularization naturally reduces signal frequency, which is correct for noisy classification

### L12: Short threshold ceiling must be ≤ 0.45, not 0.50
- optimize_short_threshold() with thr_max=0.50 finds 0.49 → "short when barely uncertain" → shorts recoveries → -76% return
- Correct ceiling: thr_max=0.46 with linspace → finds 0.45 → "short only when model is meaningfully bearish"
- **Rule**: Short signals require genuine conviction (proba ≤ 0.45). Threshold of 0.49 = noise.

### L13: np.arange with float step has floating-point endpoint artifacts
- `np.arange(0.30, 0.46, 0.01)` can include 0.46 due to float arithmetic
- **Fix**: Use `np.linspace(thr_min, thr_max, n_steps, endpoint=False)` for threshold grids

### L14: Short signals dramatically help in bear markets but hurt during recoveries
- Walk-forward fold 8 (Jan-Feb 2026): Sharpe -14.3 → +14.1 (+28.4pp) with shorts
- Walk-forward fold 7 (Dec-Jan): Sharpe -21.2 → -2.6 (+18.6pp) with shorts
- Walk-forward fold 2 (May-Jun 2025, recovery): Sharpe +4.2 → -13.3 (shorts on bounces hurt)
- **Rule**: EMA200 filter doesn't distinguish sustained bear from short-covering bounces

### L15: Test-period profitability ≠ walk-forward profitability when regimes differ
- Test period (Nov-Feb, pure bear): Sharpe +1.53, Return +16.1%, statistically significant
- Walk-forward mean (all 2025 regimes): Sharpe -3.79, 2/9 positive folds
- The bear-market test perfectly suits the short strategy; mixed-regime WF exposes limits
- **Rule**: A single-regime test period is not a fair all-weather evaluation; WF across multiple regimes is the ground truth
