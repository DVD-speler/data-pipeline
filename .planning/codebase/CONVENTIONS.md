# Coding Conventions

Observed conventions from direct analysis of the production codebase
(`config.py`, `src/model.py`, `src/backtest.py`, `src/features.py`,
`src/stats.py`, `src/data_fetcher.py`, `src/model_compare.py`, `main.py`).

---

## Language Mixing: Dutch Body, English Signatures

The most distinctive convention in this codebase is a deliberate split between
Dutch and English:

- **Function and method signatures** use English parameter names:
  `def run_backtest(test_df, probas, threshold, fee, stop_loss, use_short, ...)`
  `def time_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]`
- **Variable names inside function bodies** are Dutch:
  `drempelwaarde`, `stijging`, `kansen`, `rijen`, `venster`, `fold`, `best_thr`
  — but many short local variables follow Python idiom (`df`, `r`, `m`, `n`, `h`)
- **Comments** are Dutch throughout:
  `# Primair model: LightGBM met tijdsgewichten (recentere data zwaarder).`
  `# Blokkeer alle longs`
  `# Verschuif één 4h-candle terug`
- **Print output** (which doubles as runtime logging) is Dutch:
  `print(f"Train      : {len(train):>6} rijen  ({...} → {...})")`
  `"Statistisch signif.: {'JA' if significant else 'NEE'}"`
- **Module docstrings** are Dutch, with a consistent header format:
  `"""Fase N — <Phase Name>\n<Dutch description>\n"""`
- **Dict keys returned from functions** are English (snake_case):
  `"sharpe_ratio"`, `"total_return"`, `"win_rate"`, `"n_trades"`, `"signal_rate"`
- **Plot labels and chart titles** are Dutch.

**Practical rule**: If it appears in a public function signature or a returned
data structure, use English. If it lives inside the body of a function or in
a user-facing print statement, Dutch is the convention.

---

## Type Hints

Type hints are used on all public functions; internal helpers (`_`-prefixed) are
usually also annotated.

Patterns observed:

```python
def time_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
def run_backtest(test_df: pd.DataFrame, probas: np.ndarray, ...) -> pd.DataFrame:
def compute_metrics(results: pd.DataFrame, horizon: int = None) -> dict:
def _get_conn() -> sqlite3.Connection:
def _last_stored_timestamp(conn: sqlite3.Connection, symbol: str, interval: str):
```

- `pd.DataFrame` and `np.ndarray` are the primary container types.
- Return types use the built-in `tuple[...]` syntax (Python 3.9+ style, no
  `from typing import Tuple`).
- `dict` is used as a bare return type without parameterization.
- Optional parameters are typed with a default of `None` rather than `Optional[T]`.
- Some internal helpers omit return type annotations when the return is obvious
  (e.g., `_last_stored_timestamp` returns `int | None` implicitly).

---

## Docstring Pattern

All public functions have docstrings. The pattern is NumPy/SciPy style with
`Parameters` and `Returns` sections, written in Dutch:

```python
def optimize_threshold(model, val_df: pd.DataFrame, ...) -> float:
    """
    Zoek de drempelwaarde die de Sharpe Ratio maximaliseert op de validatieset.
    Gebruikt long-only zonder position sizing om de threshold puur te beoordelen.

    Parameters
    ----------
    model      : getraind classifier (sklearn-interface)
    val_df     : validatieset (uitvoer van time_split_with_validation)
    thr_min    : ondergrens zoekruimte
    thr_max    : bovengrens zoekruimte
    min_trades : minimaal aantal trades om een threshold te accepteren

    Returns
    -------
    float : optimale drempelwaarde
    """
```

Module-level docstrings follow a "Fase N" header with bullet points explaining
design decisions:

```python
"""
Fase 5 — ML Model
Traint een Random Forest classifier om te voorspellen of de prijs
in de komende N uur stijgt of daalt.

Stap G verbeteringen:
  - max_depth verlaagd van 8 → 5  (minder overfitting)
  ...
"""
```

Private helpers (`_`-prefixed) have shorter one-line Dutch docstrings:
```python
def _session(hour: int) -> int:
    """Codeer het handelsuur als sessie-integer."""
```

---

## Error Handling Patterns

Three distinct patterns are used:

**1. ValueError with Dutch message for domain violations:**
```python
raise ValueError(
    f"Niet genoeg data voor een {config.TEST_SIZE_DAYS}-daagse testperiode. "
    f"Dataset heeft {len(df)} rijen."
)
```

**2. FileNotFoundError for missing persisted artifacts:**
```python
raise FileNotFoundError(
    f"Geen model gevonden op {model_path}. Voer eerst train_model() uit."
)
```

**3. Silent except-with-fallback for optional external dependencies:**
```python
try:
    from src.external_data import load_all_external
    df_ext = load_all_external(df.index)
    for col in df_ext.columns:
        df[col] = df_ext[col]
except Exception as e:
    print(f"  Waarschuwing: externe data laden mislukt ({e}) — defaults gebruikt")
    for col in ["fear_greed", "spx_return_24h", ...]:
        if col not in df.columns:
            df[col] = 0.0
```

**4. ImportError with pip install hint for optional ML libraries:**
```python
try:
    import lightgbm as lgb
    model = lgb.LGBMClassifier(...)
except ImportError:
    model = RandomForestClassifier(...)
    print("\nModel trainen (RandomForest — LightGBM niet gevonden)...")
```

The pattern is: hard fail for data integrity problems; soft fail with a sensible
default and a printed warning for optional enrichment data or optional libraries.

---

## How `config.py` Is Used

`config.py` is the single source of truth for all tunable constants. It is
imported at the top of every module:

```python
import config
```

and accessed by attribute, never by local copy:

```python
test_h = config.TEST_SIZE_DAYS * 24
threshold = config.SIGNAL_THRESHOLD
fee = config.TRADE_FEE
```

Constants in `config.py` are organized into named sections with dashed comment
banners:

```
# ── Paths ──────────────────────────
# ── Binance API ────────────────────
# ── Multi-symbol & multi-timeframe ─
# ── Model instellingen ─────────────
# ── Walk-forward validatie ─────────
# ── Target dead zone ───────────────
# ── Features (1h timeframe) ────────
# ── Features (4h timeframe) ────────
# ── Backtest ───────────────────────
```

`DATA_DIR` is a `pathlib.Path` and all file paths are constructed with `/`
operator: `config.DATA_DIR / "model.pkl"`, `config.DATA_DIR / "stats" / "roc_curve.png"`.

Directory creation happens at import time: `DATA_DIR.mkdir(exist_ok=True)`.

Function signatures often expose config values as overridable defaults:
```python
def run_backtest(..., fee: float = config.TRADE_FEE, stop_loss: float = config.STOP_LOSS_PCT, ...):
```

This lets call sites override for experiments while keeping config as the default.

---

## Feature Column Registration via `FEATURE_COLS_1H`

Features are registered centrally in `config.py` as ordered Python lists:

```python
FEATURE_COLS_1H = [
    # Tijdsfeatures
    "hour", "day_of_week", "hour_of_week", "session",
    # P1/P2 statistieken
    "p1_probability",
    # Prijs & volume (rolling)
    "volatility_24h", "prev_day_return", "volume_ratio", ...
    # Technische indicatoren (1h)
    "rsi_14", "macd", "macd_signal", "bb_pct", ...
    # Cross-asset & externe features
    "fear_greed", "spx_return_24h", ...
]
FEATURE_COLS_4H = ["rsi_14_4h", "macd_4h", "bb_pct_4h", "ema_ratio_20_4h"]
FEATURE_COLS = FEATURE_COLS_1H + FEATURE_COLS_4H
```

To add a feature: (1) compute it in `src/features.py:build_features()`, (2) add
the column name to the appropriate list in `config.py`. The model training,
backtest, and live signal generation all reference `config.FEATURE_COLS` — no
other change is needed.

Removed features are commented out in-place with the reason:
```python
# Verwijderd (feature importance ≈ 0 of gebroken data):
#   "direction_bias"  — 0.000762 belang, biedt geen additioneel signaal
#   "oi_change_24h"   — Binance OI API limiteert tot 30 dagen history
```

---

## Pandas and NumPy Patterns

**DataFrame construction**: features are built by mutating a copy of the input
DataFrame, then selecting only the needed columns at the end:
```python
df = df_ohlcv.copy()
# ... many df["col"] = ... assignments ...
keep = config.FEATURE_COLS + ["target", "close"]
df_feat = df[keep].dropna()
```

**Rolling windows**: `df["col"].rolling(N).std()`, `.mean()`, `.max()` are the
standard patterns. A small epsilon `1e-10` is added to denominators to avoid
division by zero:
```python
df["volume_ratio"] = df["volume"] / (df["volume"].rolling(24).mean() + 1e-10)
```

**Forward fill for multi-timeframe join**: 4h features are joined to 1h data
with `how="left"` then `ffill()`:
```python
df = df.join(df_4h_feat, how="left")
df[cols_4h] = df[cols_4h].ffill()
```

**Target construction with dead zone**: `np.where` nested two levels deep:
```python
df["target"] = np.where(
    df["future_close"] > df["close"] * (1 + dead), 1,
    np.where(
        df["future_close"] < df["close"] * (1 - dead), 0,
        np.nan)
)
```

**Cumulative product for equity curve**:
```python
results["cum_strategy"] = (1 + results["strategy_return"].fillna(0)).cumprod()
```

**`np.linspace` over `np.arange` for float grids** (lesson L13):
```python
n_steps = round((thr_max - thr_min) / 0.01)
for thr in np.linspace(thr_min, thr_max, n_steps, endpoint=False):
```

**Alignment convention**: short variable names with aligned assignment operators
are used when initializing multiple related variables:
```python
test_h = config.TEST_SIZE_DAYS       * 24
val_h  = config.VALIDATION_SIZE_DAYS * 24
train  = df.iloc[: len(df) - total_held_out]
val    = df.iloc[len(df) - total_held_out : len(df) - test_h]
test   = df.iloc[len(df) - test_h :]
```

**Section banners**: horizontal rule comments separate logical sections within
a module:
```
# ── Train / validatie / test split ────────────────────────────────────────────
# ── Threshold optimalisatie ───────────────────────────────────────────────────
# ── Training ──────────────────────────────────────────────────────────────────
```

---

## Module Structure Pattern

Every source module follows the same structure:
1. Module docstring (Dutch, "Fase N" header)
2. Standard library imports
3. Third-party imports (numpy, pandas, sklearn, etc.)
4. `import config` (always last in imports)
5. Module-level constants (if any)
6. Private helpers (`_`-prefixed, grouped by concern with banner comment)
7. Public functions (grouped by concern with banner comment)
8. `if __name__ == "__main__":` block for standalone execution

The `if __name__ == "__main__":` block is present in every `src/` module,
allowing each phase to be run directly without going through `main.py`.
