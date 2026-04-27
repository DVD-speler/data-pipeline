# Roadmap

Sprint-historiek (compact) + openstaande verbeteringen + experimenteel.

## Sprint progression (BTC, mediaan WF Sharpe)

| Sprint | Datum | Resultaat | Wat |
|---|---|---|---|
| 1–4 | mrt 2026 | +5.94 → +12.99 | DXY, Ichimoku, RSI-divergence, Kelly, regime SL/TP, Google Trends |
| 5 | mrt 2026 | regressie | HMM regime detection — uitgesloten (redundant met ADX) |
| 6 | mrt 2026 | +13.13 | 19 schadelijke features verwijderd (66 → 47) |
| 7 | mrt 2026 | +24.18 | Bybit OI + on-chain + F&G momentum |
| 8 | apr 2026 | +24.76 | Optuna 150 trials + Sharpe objective + daily gate infra |
| 9 | apr 2026 | — | Sharpe-based model selectie (geen meetbare winst) |
| 10 | apr 2026 | WF +0.76 | Walk-forward Sharpe rapport (3 folds) |
| 11 | apr 2026 | WF +2.21 (+191%) | Bear-regime short model — actief voor BTC |
| 12 | apr 2026 | WF +5.57 | Daily gate (BTC AUC 0.63), short threshold tuning |
| 13 | apr 2026 | WF +4.05 | ETH daily verbetering mislukt; gerevert naar 28 features |
| **14** | apr 2026 | **mediaan +6.14** | BB-breedte + MACD-stabiliteit → ranging_score → 71% minder false-shorts |
| 15 | apr 2026 | teruggedraaid | ADX-bonus + 4h-confluence — regressie -2.62 |
| 16 | apr 2026 | mediaan +7.33 | ATR-stops configureerbaar, crash-mode binary |
| 17 | apr 2026 | gem +6.36 | MACD momentum-sizing |
| 18 | apr 2026 | teruggedraaid | SHAP interactie-features + WF calibratie — beide instabiel |
| **19** | apr 2026 | **gem +5.97** | Crash-mode 3-tier (tier2+3 actief), Discord onderbouwing |

## Volgende: Sprint 20 — Live execution via Bybit

Status: **niet gestart**. Doel: nachtelijke Discord-signalen automatisch
laten uitvoeren via Bybit API.

- **S20-A** Order Executor module (`src/order_executor.py` met CCXT) — prio 1
- **S20-B** SQLite positie-tracking (`data/open_orders.db`) — prio 1
- **S20-C** Integratie in `live_alert.py` — prio 1
- **S20-D** Position monitor voor SL/TP (alternatief: Bybit conditional orders) — prio 2
- **S20-E** Sandbox-fase ≥ 2 weken vóór live-go — vereist

Risico's: KYC-tijd voor Bybit futures, slippage op marktorders, leverage
discipline. Begin met spot of max 2× leverage.

## Openstaande verbeteringen (medium prioriteit)

Items uit oude `verbeterplan.md` die nog niet gedaan zijn:

| Item | Status | Bestand(en) |
|---|---|---|
| **Per-regime drempel-optimalisatie (data-driven)** — vervangen van handmatige `REGIME_THRESHOLD_OFFSETS` door per-regime sweep op validatieset | open | `src/model.py`, `src/backtest.py` |
| **Signaalveroudering in positiegrootte** — size decay als trade vastzit zonder beweging | open | `src/live_alert.py`, `src/backtest.py` |
| **Glassnode/CryptoQuant on-chain integratie** — SOPR, NUPL, exchange netflow (vereist betaalde API) | open | `src/external_data.py`, `src/features.py` |
| **Drempeloptimalisatie per ADX-band** (S15-C, uitgesteld na S15 revert) | open | `src/backtest.py` |
| **Per-trade circuit breaker** — vereist trade-level simulatie (huidige backtest is vectorized) | uitgesteld | `src/backtest.py` |
| **Pyramiding / add-on logic** in sterke trends — zelfde reden, vereist trade-level sim | uitgesteld | `src/backtest.py` |

## Recent afgerond (al in productie)

| Verbetering | Sprint | Bestand |
|---|---|---|
| Probability calibratie (`CalibratedClassifierCV`) | — | `src/model.py`, `model_calibrated.pkl` |
| 4h confirmatie via apart 4h-model | — | `src/model.py`, `live_alert_4h.py` |
| Expanding-window walk-forward | — | `--expanding` flag in `main.py` |
| POC / volume profile als feature (`poc_distance`) | — | `src/features.py` |
| Asymmetrische dead zone | — | `TARGET_DEAD_ZONE_UP/DOWN` in config |
| Optuna auto-promotie | — | `data/stats/optuna_promotions.csv` |
| Put/Call ratio gate | — | `PUT_CALL_RATIO_GATE` in config |
| Regime SL/TP | Sprint 3 | `REGIME_SL_TP` in config |
| Kelly sizing (half-Kelly) | Sprint 3 | `USE_KELLY_SIZING` in config |
| ATR-trailing stop | — | `ATR_STOP_MULTIPLIER` in config |
| Drawdown circuit breaker | — | `MAX_DRAWDOWN_GATE` in config |

## Experimenteel / research-only — NIET in productie

Deze code blijft staan voor toekomstig onderzoek. Word **niet geladen
in de live flow** en zou bij een grootschalige refactor verwijderd
kunnen worden zonder gedragsverandering in productie.

### Regime sub-modellen (bull/ranging/bear)

`train_regime_models()` (in `src/model.py:838`) traint drie aparte
LightGBM-modellen op respectievelijk bull-, ranging- en bear-folds van
de trainset. Wordt automatisch aangeroepen aan het eind van
`fase_model` en schrijft `{symbol}_bull_model.pkl`,
`_ranging_model.pkl`, `_bear_model.pkl`.

Geladen via `load_regime_model()` (in `src/model.py:897`), maar **alleen
gebruikt wanneer `--phase walkforward --regime-models` wordt opgeroepen**
(default `False`). De live cron en de standaard `fase_backtest` laden de
sub-modellen niet — die gebruiken het gewone gecalibreerde model.

Dennis houdt de code voor toekomstig research naar
regime-conditionele modellen (in tegenstelling tot de huidige aanpak
met regime-conditionele *thresholds* en het Sprint-11 bear-short
schema, die wél actief zijn).

### Andere research-only artefacten

- `src/model_compare.py` — RF/XGBoost/LightGBM vergelijking. Auto-promotie
  naar `model.pkl` is uitgeschakeld; produceert alleen rapport-CSV.
- `src/horizon_scan.py` — 12h/24h/48h horizon vergelijking. Eénmalig
  draaien om te beslissen welke horizon te gebruiken.

## Niet meer relevant (voor de duidelijkheid)

- **ETH als eigen model** — vervalt. ETH bleek niet voorspelbaar genoeg
  (AUC 0.53). Cleanup in Fase C van repo-opschoning (april 2026). ETH
  OHLCV blijft gedownload omdat `eth_btc_ratio` een BTC-feature is.
- **HMM regime detection** — uitgesloten in Sprint 5, redundant met ADX.
- **BTC dominance feature** — datakwaliteit onvoldoende, 0 importance.
- **Candlestick patterns (hammer, engulfing, gap_up)** — 0 importance op
  24h horizon, te schaars.
