# Crypto Signal Model — Directory Structure

## Root Files
- `main.py` — CLI dispatcher; 11 phases via `--phase <name>`
- `config.py` — All hyperparameters, paths, feature lists (single source of truth)
- `requirements.txt` — Python dependencies

## src/ Modules

| Module | Phase | Input | Output | Key Functions |
|--------|-------|-------|--------|---------------|
| `data_fetcher.py` | data | Binance REST | `data/ohlcv.db` (SQLite) | `download_ohlcv()`, `load_ohlcv()` |
| `external_data.py` | external_data | yfinance, Binance | `data/external/*.parquet` | `download_all_external()`, `load_all_external()` |
| `p1p2_engine.py` | p1p2 | OHLCV | `data/p1p2_labels.csv` | `compute_p1p2()` |
| `stats.py` | stats | P1/P2 + OHLCV | `data/stats/*.csv` + `.png` | `compute_p1_heatmap()`, `run_stats()` |
| `features.py` | features | OHLCV + P1/P2 + ext | `data/features.parquet` | `build_features()` |
| `model.py` | model | features.parquet | `model.pkl`, `optimal_threshold.json` | `train_model()`, `load_optimal_threshold()` |
| `model_compare.py` | model_compare | features.parquet | `data/stats/model_comparison.*` | `compare_models()` |
| `backtest.py` | backtest / walkforward | model + features | `backtest_results.png`, `walkforward_*.csv` | `run_backtest()`, `run_walkforward()`, `generate_live_signal()` |
| `horizon_scan.py` | horizon_scan | features.parquet | `data/stats/horizon_scan.*` | `scan_horizons()` |

## data/ Layout

```
data/
├── ohlcv.db                   # SQLite — BTC/ETH 1h/4h candles (730 days, ~17.5k rows)
├── p1p2_labels.csv            # Daily P1/P2 extremes (~730 rows)
├── features.parquet           # Feature matrix (42 cols, ~13.1k rows after dead zone)
├── model.pkl                  # Trained primary model (joblib)
├── model_best.pkl             # Best model from model_compare (currently LightGBM)
├── optimal_threshold.json     # {"threshold": 0.74, "threshold_short": 0.45, "model": "LightGBM"}
├── latest_signal.json         # Live signal output
├── external/
│   ├── fear_greed.parquet     # Fear & Greed Index (daily)
│   ├── spx.parquet            # S&P 500 hourly (floored to :00)
│   ├── eurusd.parquet         # EUR/USD 4h
│   ├── funding_rate.parquet   # BTC funding rate (8h, fwd-filled)
│   └── open_interest.parquet  # BTC OI (30-day max history, excluded from features)
└── stats/
    ├── *.csv                  # Heatmaps, model_comparison, walkforward, horizon_scan
    └── *.png                  # All matplotlib/plotly visualizations
```

## Naming Conventions

- **Modules:** `snake_case.py`
- **Functions:** `verb_noun()` — `build_features()`, `run_backtest()`, `load_model()`
- **Config constants:** `UPPER_CASE` — `FEATURE_COLS_1H`, `PREDICTION_HORIZON_H`
- **DataFrames:** short descriptive — `df`, `train`, `val`, `test`, `features`, `results`
- **Phase functions in main.py:** Dutch prefix `fase_X()` (legacy)
- **Comments/prints:** Dutch; function signatures: English

## Key Design Constraints

- **No data leakage:** Strict temporal split; heatmaps computed on train-only slice
- **No look-ahead:** 4h candles shifted by 1 period before feature merge
- **No random cross-validation:** Walk-forward is the correct validation for time series
- **All hyperparams in config.py:** Nothing hardcoded in module files
