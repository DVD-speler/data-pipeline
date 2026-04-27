# Architectuur — Crypto Signal Model

Huidige staat van het systeem. Update wanneer je structurele wijzigingen
maakt (nieuwe pipeline-fase, ander deploy-platform, ander signal-kanaal).

## Doel

Op basis van OHLCV + macro/sentiment-data voorspellen of BTC de komende
24 uur (resp. 4h of dagelijkse horizon) stijgt of daalt, en daar
handelssignalen uit genereren met long/short-richting, position sizing
en SL/TP. Resultaat gaat naar Discord. Live execution via Bybit is
gepland (Sprint 20, niet actief).

## Stack

- **Python 3.11** in devcontainer (`mcr.microsoft.com/devcontainers/python:3.11-bullseye`)
- **LightGBM** als hoofdmodel, met **Optuna** voor hyperparameter-tuning
- **scikit-learn** voor `CalibratedClassifierCV` en RF/comparison
- **pandas / pyarrow** voor data, **SQLite** voor OHLCV cache
- **Binance public REST API** voor OHLCV (geen key)
- **yfinance** voor SPX, EUR/USD, USD/JPY, DXY, VIX
- **Externe API's** (gratis): alternative.me (F&G), pytrends (Google Trends),
  Deribit (DVOL), blockchain.info (active addresses, hash rate),
  Bybit/Binance (OI, funding)
- **GitHub Actions** voor scheduled runs, **cron-job.org** voor exacte timing
- **Discord webhooks** voor signaal-notificatie

## Drie productie-modellen

Eén main model (BTCUSDT) op drie tijdframes. ETHUSDT is **niet** een
eigen model meer; ETH/BTC ratio wordt nog wel als BTC-feature gebruikt
(`eth_btc_ratio`, altcoin-season indicator).

| Timeframe | Horizon | Doel | Discord webhook |
|---|---|---|---|
| **1h** | 24h forward | hoofdsignaal, entry/exit-trigger | `DISCORD_WEBHOOK_URL` |
| **4h** | 24h forward | trager bevestigend signaal | `DISCORD_WEBHOOK_URL_4H` |
| **dagelijks** | 1d forward | bias / regime-gate (BTC only) | `DISCORD_WEBHOOK_URL_DAILY` |

Het dagelijkse model dient als **gate** voor het 1h model: als de daily
proba < `DAILY_GATE_THRESHOLD` (0.45), worden 1h longs geblokkeerd. Alleen
voor BTC actief — ETH daily AUC was te zwak (0.53).

## Pipeline (`main.py`)

Een fase-gestuurde flow. Elke fase produceert een artefact in `data/`
voor de volgende fase.

```
data → external_data → p1p2 → stats → features → model → backtest → signal/live_alert
```

| Fase | Output | Duur |
|---|---|---|
| `data` | `ohlcv.db` (SQLite) | ~1 min incrementeel |
| `external_data` | `data/external/*.parquet` | ~30 sec |
| `p1p2` | `{symbol}_p1p2_labels.csv` | ~5 sec |
| `stats` | heatmaps in `data/stats/`, P1-kanskaart | ~5 sec |
| `features` | `{symbol}_features.parquet` (~47 features) | ~10 sec |
| `model` | `{symbol}_model.pkl`, `_calibrated.pkl`, threshold + Kelly | ~5 min met Optuna |
| `backtest` | metrics + `walkforward_lightgbm.csv` | ~30 sec |
| `signal`/`live_alert` | `{symbol}_latest_signal.json` + Discord | ~10 sec |

Voor 4h en daily zijn er aparte fasen (`features_4h`, `model_4h`,
`live_alert_4h`, `features_daily`, etc.). Die patchen tijdelijk
`config.FEATURE_COLS` en backuppen de hourly model-artefacten tijdens
training (zie `fase_model_4h` / `fase_model_daily` in `main.py`).

## Live cron

Drie GitHub Actions workflows in `.github/workflows/`:

- **`signal.yml`** (unified) — `workflow_dispatch` met `inputs.type` =
  `hourly | daily | 4h`. Voert het juiste model uit afhankelijk van type.
- **`hourly_signal.yml`** — alleen 1h logica (legacy/redundant).
- **`daily_signal.yml`** — alleen dagelijkse logica (legacy/redundant).

Cron-job.org triggert via GitHub API:
- elk uur op `:05` UTC → `signal.yml` met `type=hourly`
- elk 4 uur op `:05` UTC → `signal.yml` met `type=4h`
- 07:05 UTC dagelijks → `signal.yml` met `type=daily`

Elke run committeert `data/external/`, `{symbol}_paper_trades.json` en
`{symbol}_latest_signal.json` terug naar `master` met `git pull --rebase
-X theirs` (cron wint van eventuele lokale wijzigingen op signal-files).

## Features (47 in 1h model + 4 in 4h)

Geclusterd:

- **Tijd**: hour, day_of_week, hour_of_week, p1_probability
- **Prijs/momentum**: return_2h, return_7d, return_30d, prev_day_return,
  ath_7d_distance, volatility_24h
- **Trendkwaliteit**: vol_regime, trend_consistency_12h, buy_pressure
- **Technisch (1h)**: rsi_14, macd, macd_signal, bb_pct, ema_ratio_20,
  price_vs_ema200, atr_pct, adx
- **Regime-detectie**: bb_width, macd_hist_stability, vwap_distance,
  poc_distance
- **Macro**: spx_return_24h, eurusd_return_24h, dxy_return_24h/7d,
  vix_level, usdjpy_return_24h/7d
- **Cross-asset**: eth_btc_ratio, fear_greed, fear_greed_7d_chg
- **Crypto-specifiek**: funding_rate, funding_momentum, oi_return_24h,
  oi_price_divergence, btc_dvol
- **On-chain**: active_addresses_7d_chg, hash_rate_7d_chg
- **Sentiment**: google_trends_btc, trends_momentum_4w, trends_spike
- **Patroon**: rsi_bull_divergence, rsi_bear_divergence
- **Ichimoku**: cloud_position, cloud_thickness, tk_cross
- **4h timeframe**: rsi_14_4h, macd_4h, bb_pct_4h, ema_ratio_20_4h

Het complete lijst staat in `config.FEATURE_COLS_1H` + `FEATURE_COLS_4H`.

## Active gates

Long-blokkers (1h):

| Gate | Drempel | Bron |
|---|---|---|
| DVOL | > 65 | Deribit |
| VIX | > 25 | yfinance |
| Funding extreme | > 0.05% per 8u | Binance/Bybit |
| Return 30d | < -10% | OHLCV |
| USDJPY 7d | < -3% | yfinance |
| 25D skew (live-only) | > 5% | Deribit |
| Daily model proba (BTC) | < 0.45 | eigen daily model |
| Drawdown circuit breaker | -15% peak → 168u pauze | backtest state |

Short-activatie (BTC only): `market_regime == -1` AND `proba <
SHORT_ENTRY_THRESHOLD` (0.30) AND `return_30d < -3%` AND `ranging_score < 2`.

## Risk management

- **Kelly half** — `KELLY_FRACTION = 0.5`, max 20% per trade
- **Regime SL/TP** — bull/ranging/bear hebben aparte SL/TP-percentages
  (`REGIME_SL_TP` in config.py)
- **ATR-trailing stop** — `ATR_STOP_MULTIPLIER × atr_pct`, geclipt 0.5–10%
- **Crash-mode 3-tier** — `CRASH_SIZE_FACTORS = {2: 0.50, 3: 0.25}`
  - Tier 2 (>2.5σ daling) → ×0.50
  - Tier 3 (>5σ of >10% in 24u) → ×0.25
- **MACD momentum sizing** — entry-size × (1 + macd_size_mult), max ×1.5
- **Correlatie guard** — niet relevant nu ETH-model weg is

## Bestanden

```
config.py              # 1h model + globaal
config_4h.py           # 4h model overrides
config_daily.py        # daily model overrides
main.py                # fase-runner CLI
src/
├── data_fetcher.py    # Binance OHLCV → SQLite
├── external_data.py   # macro/sentiment/on-chain
├── p1p2_engine.py     # P1/P2 extreem-labels
├── stats.py           # heatmaps + direction bias
├── features.py        # 1h feature matrix
├── features_4h.py     # 4h feature matrix
├── features_daily.py  # daily feature matrix
├── model.py           # LightGBM + Optuna + calibratie
├── model_compare.py   # RF/XGB/LGB vergelijking (research-only)
├── horizon_scan.py    # 12h/24h/48h horizon scan (research-only)
├── backtest.py        # backtest + walk-forward + run_backtest_be_trail
├── levels.py          # swing highs/lows voor structurele SL/TP
├── simulation.py      # paper-trading simulatie met SL/TP
├── live_alert.py      # 1h Discord alert + paper trade tracking
├── live_alert_4h.py   # 4h Discord alert
└── live_alert_daily.py # daily Discord alert (richtingsindicator)
```

## Belangrijke beperkingen

- **Geen lokale Python install** — alles draait in de devcontainer
  (Docker) of op GitHub Actions (Ubuntu runners). Lokale `python` op
  Windows lost niet op.
- **Single-symbol** — sinds de ETH-cleanup is BTCUSDT het enige
  productie-symbool. ETH OHLCV blijft gedownload omdat
  `eth_btc_ratio` als BTC-feature wordt gebruikt.
- **Geen automatic retraining** — `model.pkl` is een statisch artefact.
  Hertrainen gebeurt handmatig via `python main.py --phase model`.
