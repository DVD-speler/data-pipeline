# Testing Patterns and Validation

Observed from direct analysis of `src/model.py`, `src/backtest.py`,
`tasks/lessons.md`, `config.py`, and the full source tree.

---

## Formal Test Suite: None

There is **no formal test suite** in this codebase. The following were verified
to be absent:

- No `test/` or `tests/` directory
- No `pytest.ini`, `setup.cfg`, `pyproject.toml`, or `conftest.py`
- No files matching `test_*.py` or `*_test.py`
- No `unittest` or `pytest` imports anywhere in `src/`

All quality assurance is done through runtime validation (backtesting, walk-
forward, and statistical significance checks) and print-driven inspection of
intermediate outputs. There are no unit tests, integration tests, or property-
based tests.

---

## Validation Strategy: Chronological Splits

The project enforces strict time-based data splits to prevent look-ahead bias.
Three regions are carved from the dataset in chronological order:

```
|<─────── Train ───────────────>|<── Validation ──>|<── Test ──>|
                                 60 days             90 days
                                 (threshold tuning)  (final report)
```

Implemented in `src/model.py:time_split_with_validation()`:

```python
test_h = config.TEST_SIZE_DAYS       * 24   # 90 days × 24h
val_h  = config.VALIDATION_SIZE_DAYS * 24   # 60 days × 24h
train  = df.iloc[: len(df) - total_held_out]
val    = df.iloc[len(df) - total_held_out : len(df) - test_h]
test   = df.iloc[len(df) - test_h :]
```

Key invariants enforced by this split:
- Threshold optimisation (`optimize_threshold`) runs only on `val`, never `test`.
- Feature heatmaps (P1 probability, direction bias) are computed from the
  train-only slice and frozen before the split — see `main.py:fase_features()`:
  ```python
  total_holdout_h = (config.TEST_SIZE_DAYS + config.VALIDATION_SIZE_DAYS) * 24
  train_cutoff_date = df.index[-total_holdout_h].date()
  train_p1p2 = p1p2[pd.to_datetime(p1p2["date"]).dt.date < train_cutoff_date]
  ```
  This was codified after lesson L6 (heatmaps from full dataset = data leakage).

---

## Backtest Metrics Computed

`src/backtest.py:compute_metrics()` returns the following for every evaluation:

| Metric              | Description                                              |
|---------------------|----------------------------------------------------------|
| `total_return`      | Cumulative strategy return (compounded)                  |
| `buy_hold_return`   | Buy-and-hold return over the same period                 |
| `annualized_return` | Annualised using `(1+r)^(8760/n_hours) - 1`             |
| `sharpe_ratio`      | Active-trade Sharpe, annualised by `sqrt(8760/horizon)`  |
| `max_drawdown`      | Maximum peak-to-trough drawdown                          |
| `win_rate`          | Fraction of active trades with positive return           |
| `n_trades`          | Total trades (long + short)                              |
| `n_long`            | Long signal count                                        |
| `n_short`           | Short signal count                                       |
| `signal_rate`       | Fraction of candles that generated any signal            |

Sharpe is computed on active trade returns only (bars where `signal != 0`),
not on the full time series. This avoids inflating Sharpe by counting idle
periods as zero-return days:

```python
active_returns = strat_returns[results["signal"] != 0]
sharpe = (active_returns.mean() / active_returns.std()) * np.sqrt(hours_per_year / h)
```

The B&H benchmark uses 1-hour step returns compounded — not h-period forward
returns — after lesson L5 identified that overlapping h-period returns produce
nonsensical benchmark values.

---

## How Threshold Optimisation Works (Validation-Driven)

Rather than using a fixed threshold, the project finds the Sharpe-maximising
long entry threshold on the validation set:

```python
for thr in np.linspace(thr_min, thr_max, n_steps, endpoint=False):
    r = run_backtest(val_df, probas, threshold=float(thr), use_short=False, ...)
    m = compute_metrics(r)
    if m["n_trades"] >= min_trades and m["sharpe_ratio"] > best_sharpe:
        best_sharpe = m["sharpe_ratio"]
        best_thr    = float(thr)
```

The minimum trade count guard (`min_trades=10` for longs, `min_trades=5` for
shorts) prevents selecting a threshold that fires on only one or two trades and
has a spuriously high Sharpe.

Short threshold is optimised separately with `thr_max=0.46` — a ceiling that
forces the model to be meaningfully bearish (proba <= 0.45) rather than merely
uncertain (lesson L12). Using `thr_max=0.50` previously found 0.49 as the
optimal short threshold, which amounted to shorting recoveries.

Thresholds are persisted to `data/optimal_threshold.json` and loaded at
backtest and live signal generation time.

---

## Walk-Forward Validation

`src/backtest.py:run_walkforward()` implements a rolling-window re-training
scheme to measure generalisation across multiple market regimes:

```
Config:
  WALKFORWARD_TRAIN_DAYS = 270  (9 months rolling train window)
  WALKFORWARD_TEST_DAYS  = 30   (1 month held-out test per fold)
  WALKFORWARD_STEP_DAYS  = 30   (advance by 1 month per fold)
```

Per fold:
1. Train on `[start : train_end - val_h]`
2. Optimise thresholds on internal validation slice `[train_end - val_h : train_end]`
3. Evaluate on `[train_end : train_end + test_h]`
4. Record Sharpe, return, win rate, long/short counts, and optimal thresholds

Summary metrics printed after all folds:
```
Gemiddelde Sharpe   : +X.XXXX
Gemiddeld Return    : +X.XX%
Gem. Win Rate       : XX.XX%
Positieve folds     : N/M
```

Fold-level results are saved to `data/stats/walkforward_<modelname>.csv` for
further inspection.

Walk-forward is considered the "ground truth" evaluation (lesson L15): a
single test period can be dominated by a single market regime (e.g., pure bear
market Nov 2025–Feb 2026), whereas walk-forward covers multiple regimes across
all of 2025.

---

## Statistical Significance Testing: Random Baseline

`src/backtest.py:compute_random_baseline()` tests whether the strategy Sharpe
is genuinely above chance:

```python
def compute_random_baseline(results, n_simulations=500, seed=42):
    rng = np.random.default_rng(seed)
    sharpes = []
    for _ in range(n_simulations):
        shuffled_signal = rng.permutation(signal_values)
        strat_ret = np.where(shuffled_signal > 0, trade_return - 2 * fee, 0.0)
        active = strat_ret[shuffled_signal != 0]
        s = (active.mean() / active.std()) * np.sqrt(hours_per_year / h)
        sharpes.append(s)
    return {"random_sharpe_p5": ..., "random_sharpe_mean": ..., "random_sharpe_p95": ...}
```

The signal vector is permuted 500 times (preserving signal frequency and market
return distribution but destroying timing). If the real strategy Sharpe exceeds
the 95th percentile of permuted Sharpes, it is declared statistically
significant.

Output in `train_model()`:
```
=== Significantiecheck (N=500 willekeurige signalen) ===
  Strategie Sharpe   : +1.530
  Random p5 / p95    : -0.412 / +0.837
  Statistisch signif.: JA
```

The significance label is `"JA"` or `"NEE"` — no p-value is computed; it is a
simple threshold crossing.

---

## Model Comparison as a Validation Step

`src/model_compare.py:compare_models()` runs all available classifiers
(RandomForest, XGBoost, LightGBM) and an AUC-weighted ensemble on the same
train/val/test split. Output is a table of:

- ROC AUC (classification quality)
- Sharpe Ratio, Total Return, Win Rate (trading quality)
- Long/short trade counts and thresholds

This serves as a sanity check that no single model is pathologically
outperforming due to lucky hyperparameters. Results are saved to
`data/stats/model_comparison.csv`.

The ensemble uses validation-set AUC to weight base learners, which avoids
leakage (lesson L8: calibration or weighting on the same set used for threshold
tuning corrupts the comparison).

---

## What `tasks/lessons.md` Documents

The lessons file is the project's primary quality log. It is written session-
by-session and contains 15 lessons as of 2026-02-28. Lessons are numbered L1–L15
and follow a consistent pattern:

```
### LN: Short title
- Description of what went wrong or what was discovered
- **Fix** or **Decision**: what was changed
- **Rule**: the generalised principle to apply going forward
```

Categories of lessons recorded:

| Category                  | Lessons                |
|---------------------------|------------------------|
| Data quality / API limits | L1 (OI 30-day cap), L2 (yfinance timestamp :30), L3 (pandas parquet dtype mismatch) |
| Data leakage              | L6 (heatmaps from full dataset), L8 (calibration not propagated to walkforward) |
| Numeric / float precision | L13 (np.arange float endpoint artifact → use np.linspace) |
| Model calibration         | L4 (isotonic calibration overfits on small val sets)       |
| Benchmark correctness     | L5 (B&H must use 1h step returns not h-period forward)     |
| Feature quality           | L7 (zero importance = constant/near-constant feature), L9 (dead zone removes 21.6% neutral rows) |
| Regime-specific results   | L10 (test period is bear market), L14 (shorts hurt on recoveries), L15 (WF is ground truth) |
| Hyperparameter insight    | L11 (more regularized LightGBM → higher threshold → fewer but better trades), L12 (short ceiling must be 0.45 not 0.50) |

The lessons file is a key quality artefact: it records decisions that are not
visible from the code alone (why certain hyperparameters were chosen, why
certain features were removed, why certain API endpoints are capped).

---

## Print-Driven Debugging and Inspection

There is no logging framework (no `logging` module usage found). All runtime
information is emitted via `print()`. The conventions are:

**Phase headers** in `main.py`:
```python
print("=" * 60)
print("FASE 5 — Model Trainen")
print("=" * 60)
```

**Progress within a phase** (two leading spaces):
```python
print(f"  Train      : {len(train):>6} rijen  ({...} → {...})")
print(f"  Validatie  : {len(val):>6} rijen  (...)")
```

**Results tables** with a header line:
```python
print(f"\n=== Test Set Resultaten ===")
print(classification_report(...))
print(f"ROC AUC : {roc_auc_score(y_test, probas):.4f}")
```

**Warnings with "Waarschuwing:" prefix** for non-fatal failures:
```python
print(f"  Waarschuwing: externe data laden mislukt ({e}) — defaults gebruikt")
print(f"  Waarschuwing: geen 4h data beschikbaar — 4h features weggelaten.")
```

**Save confirmations** for every file written:
```python
print(f"  Opgeslagen: {out.name}")
```

**Walk-forward fold lines** with aligned columns:
```python
print(
    f"  Fold {fold:>2}  [{test_start} → {test_end}]"
    f"  thr={opt_thr:.2f}"
    f"  Sharpe: {sharpe:+.3f}"
    f"  Return: {total_return:+.1%}"
    f"  L:{n_long} S:{n_short}"
)
```

The consistent use of `{value:+.3f}` (explicit sign) for Sharpe and returns
makes negative values immediately visible in the output. The `{n:>6}` right-
alignment for row counts keeps columns visually aligned.

This print-driven approach means that running `python main.py --phase model`
produces a complete audit trail of the training run to stdout, which is
sufficient for the project's current single-developer, research-oriented
workflow but would need to be replaced with structured logging before any
production or multi-user deployment.

---

## Feature Quality Checks (Manual, Not Automated)

No automated feature validation runs before training. Quality checks are
described in lessons and applied manually:

- **Lesson L7**: Zero feature importance = constant or near-constant feature.
  Check: `df[col].std()` and `df[col].isna().mean()`. A post-join null fraction
  diagnostic is recommended but not yet implemented as an automated pre-flight
  check.
- **Lesson L1**: OI feature removed because 96% of training rows had value 0.0
  (API history < training window). Detected by inspecting feature importance
  output after training.
- **Dead zone reporting**: `build_features()` prints how many neutral rows were
  removed, giving a quick sanity check on label balance:
  ```python
  print(f"  Dead zone: {n_removed} neutrale rijen verwijderd ({pct:.1%} van totaal)")
  ```
- **Target distribution**: `train_model()` prints the fraction of up vs. down
  labels in the training set:
  ```python
  print(f"Target (train): stijging={y_train.mean():.1%}  daling={(1-y_train.mean()):.1%}")
  ```

These are all eyeball checks during the training run, not assertions or
automated tests.
