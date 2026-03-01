# STACK.md — Technology Stack

## Primary Language

**Python 3.11**

The runtime environment is pinned to Python 3.11 via the devcontainer image
(`mcr.microsoft.com/devcontainers/python:3.11-bullseye`). The `__pycache__`
bytecode files confirm the interpreter version: all `.pyc` files are compiled
under `cpython-311`. No virtual environment configuration file (Pipfile,
poetry.lock, pyproject.toml) is present; dependency management is handled
exclusively through `requirements.txt`.

---

## Runtime Environment

- **Execution mode:** Local CLI application, run directly via `python main.py`
- **Container:** Microsoft Dev Container (`devcontainer.json`) based on
  `mcr.microsoft.com/devcontainers/python:3.11-bullseye` (Debian Bullseye)
- **IDE integration:** VS Code extensions defined in devcontainer — Python,
  Pylance, Black formatter, Jupyter, Jupyter Cell Tags
- **Jupyter support:** Port 8888 is forwarded; `jupyter` and `ipykernel` are
  installed, enabling notebook-based exploration alongside the CLI pipeline
- **No cloud deployment:** All computation runs locally; no containerization
  beyond the devcontainer, no Dockerfile, no cloud function configuration
- **Code formatter:** Black (enforced via VS Code `formatOnSave`)

---

## Dependencies (`requirements.txt`)

| Package | Version Constraint | Role |
|---|---|---|
| `pandas` | >=2.0.0 | Core data manipulation; DataFrames for OHLCV, features, results |
| `numpy` | >=1.24.0 | Numerical computation, array operations, random baseline |
| `requests` | >=2.31.0 | HTTP client for Binance REST API and Fear & Greed API |
| `scikit-learn` | >=1.3.0 | RandomForestClassifier, metrics (ROC AUC, classification report, confusion matrix) |
| `matplotlib` | >=3.7.0 | All charting: equity curves, drawdown, feature importance, ROC curves, heatmaps |
| `seaborn` | >=0.12.0 | Statistical heatmap visualizations (P1/P2 probability heatmaps) |
| `joblib` | >=1.3.0 | Model serialization/deserialization (`.pkl` files via `joblib.dump/load`) |
| `tqdm` | >=4.66.0 | Progress bars for Binance OHLCV batch downloads |
| `jupyter` | >=1.0.0 | Notebook runtime |
| `ipykernel` | >=6.25.0 | Jupyter kernel for Python 3.11 |
| `pyarrow` | >=14.0.0 | Parquet read/write backend used by pandas for external data cache files |
| `ta` | >=0.11.0 | Technical analysis library: RSI, MACD, Bollinger Bands, EMA, ATR |
| `yfinance` | >=0.2.0 | Yahoo Finance market data client (SPY/S&P500, EUR/USD hourly prices) |
| `xgboost` | >=2.0.0 | XGBoost gradient boosting classifier (optional; skipped gracefully if absent) |
| `lightgbm` | >=4.0.0 | LightGBM gradient boosting classifier (primary model; fallback to RF if absent) |

All dependencies are open source and freely available via PyPI. No proprietary
or licensed packages are required. No API keys are stored in `requirements.txt`
or any configuration file.

---

## Project Structure

```
crypto_signal_model/
├── config.py                  # Central configuration (paths, symbols, hyperparameters)
├── main.py                    # CLI entry point; orchestrates all pipeline phases
├── requirements.txt           # Python package dependencies
├── src/
│   ├── data_fetcher.py        # Phase 1: Binance OHLCV download + SQLite persistence
│   ├── external_data.py       # Phase 1b: Fear & Greed, SPX, EUR/USD, funding rate, OI
│   ├── p1p2_engine.py         # Phase 2: P1/P2 label computation
│   ├── stats.py               # Phase 3: probability heatmaps, direction bias
│   ├── features.py            # Phase 4: feature matrix construction
│   ├── model.py               # Phase 5a: LightGBM/RF training, threshold optimization
│   ├── model_compare.py       # Phase 5b: multi-model comparison + AUC-weighted ensemble
│   ├── backtest.py            # Phase 6: backtest engine, walk-forward, live signal
│   └── horizon_scan.py        # Phase 7: multi-horizon prediction scan
└── data/
    ├── ohlcv.db               # SQLite database (OHLCV candles)
    ├── p1p2_labels.csv        # P1/P2 daily labels
    ├── features.parquet       # Full feature matrix
    ├── model.pkl              # Serialized trained model (joblib)
    ├── model_best.pkl         # Best model from comparison run (joblib)
    ├── optimal_threshold.json # Optimized signal thresholds (long + short)
    ├── latest_signal.json     # Most recent live signal output
    ├── external/              # Cached external data (Parquet files, 6-hour TTL)
    └── stats/                 # Output charts and CSV reports (PNG + CSV)
```

---

## Configuration Management

All configuration is centralized in `/workspaces/crypto_signal_model/config.py`.
There is no `.env` file, no environment variable loading, and no secrets
management layer. The config module is a plain Python file with module-level
constants. Key categories:

- **Paths:** `ROOT_DIR`, `DATA_DIR` (auto-created with `mkdir`)
- **Binance API:** `BINANCE_BASE_URL = "https://api.binance.com"`
- **Symbol & timeframe config:** `SYMBOLS = ["BTCUSDT", "ETHUSDT"]`, `INTERVALS = ["1h", "4h"]`
- **History window:** `DAYS_HISTORY = 730` (two years of lookback)
- **Model hyperparameters:** prediction horizon, train/val/test split sizes, walk-forward window
- **Signal thresholds:** `SIGNAL_THRESHOLD = 0.58` (long), `SIGNAL_THRESHOLD_SHORT = 0.0` (short disabled by default)
- **Target dead zone:** `TARGET_DEAD_ZONE_PCT = 0.003` (0.3% minimum price move)
- **Backtest parameters:** `TRADE_FEE = 0.001`, `STOP_LOSS_PCT = 0.02`
- **Feature lists:** `FEATURE_COLS_1H` (34 features), `FEATURE_COLS_4H` (4 features)

---

## Data Formats Used

| Format | Location | Purpose |
|---|---|---|
| **SQLite** (`.db`) | `data/ohlcv.db` | Primary OHLCV candle store; table `ohlcv` with composite PK `(symbol, interval, open_time)` |
| **Parquet** (`.parquet`) | `data/features.parquet`, `data/external/*.parquet` | Feature matrix and external data cache; written/read via pandas + pyarrow |
| **CSV** (`.csv`) | `data/p1p2_labels.csv`, `data/stats/*.csv` | P1/P2 labels, walk-forward fold metrics, model comparison tables, heatmap exports |
| **Pickle/joblib** (`.pkl`) | `data/model.pkl`, `data/model_best.pkl` | Serialized scikit-learn / LightGBM / XGBoost model objects |
| **JSON** (`.json`) | `data/optimal_threshold.json`, `data/latest_signal.json` | Threshold configuration, live signal output |
| **PNG** (`.png`) | `data/stats/*.png` | Backtest equity curves, ROC curves, feature importance, heatmaps, confusion matrix |

---

## ML Framework and Pipeline Phases

The pipeline is structured into sequential phases, each runnable independently
via `python main.py --phase <name>`:

1. **data** — Binance OHLCV download (SQLite storage, incremental)
2. **external_data** — Fear & Greed, SPX, EUR/USD, funding rate, OI (Parquet cache)
3. **p1p2** — P1/P2 daily label computation
4. **stats** — Probability heatmaps and direction bias (seaborn/matplotlib)
5. **features** — Feature matrix assembly (pandas, ta library, 38 features total)
6. **model** — LightGBM primary classifier training with time-weighted samples; fallback to RandomForest
7. **model_compare** — Three-model comparison (RF, XGBoost, LightGBM) plus AUC-weighted ensemble
8. **backtest** — Strategy simulation with position sizing, stop-loss, regime filter (EMA200)
9. **walkforward** — Rolling window re-training validation (~11 folds)
10. **horizon_scan** — Comparison across 12h/24h/48h prediction horizons
11. **signal** — Live signal generation for the most recent candle
