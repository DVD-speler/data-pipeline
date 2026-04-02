# Trader Roadmap

---

## Huidige staat van het model (2026-04-02)

| Categorie | Geïmplementeerd |
|---|---|
| **Technische indicatoren** | RSI-14, MACD+signal, Bollinger %B, EMA(20/200), ADX, ATR, VWAP, POC |
| **Multi-timeframe** | 1h model + 4h gate (rsi_14_4h, macd_4h, bb_pct_4h, ema_ratio_20_4h) |
| **Momentum** | return_2h, return_7d, return_30d, prev_day_return, ath_7d_distance |
| **Volume/regime** | vol_regime, buy_pressure, trend_consistency_12h |
| **Ichimoku** | cloud_position, cloud_thickness, tk_cross |
| **Funding** | funding_rate, funding_momentum |
| **Macro** | VIX, SPX, EUR/USD, USD/JPY (24h + 7d), DXY (24h + 7d), Fear & Greed |
| **Opties-markt** | btc_dvol (DVOL index), RSI-divergentie (bull + bear) |
| **Sentiment** | Google Trends (zoekvolume, 4w momentum, spike-indicator) |
| **P1/P2 statistieken** | uurlijkse kanskaart, direction bias, heatmap |
| **Live gates** | DVOL > 65, VIX > 25, funding > 0.05%, return_30d < -10%, JPY 7d > 3%, 25D skew > 5% |
| **Risicobeheer** | Kelly sizing (half-Kelly), regime SL/TP (bull/ranging/bear), correlatie guard, circuit breaker (-15% drawdown), ATR trailing stop, TP1/TP2 partieel |
| **Model infra** | LightGBM + Optuna, XGBoost/RF/Ensemble vergelijking, gekalibreerd model, walk-forward validatie, regime-specifieke modellen |

### Sharpe progressie (BTC, out-of-sample validatieset)

| Sprint | Sharpe | Delta | Wat gedaan |
|---|---|---|---|
| Baseline | +5.94 | — | origineel model |
| Sprint 1 | +6.21 | +4.5% | DXY, Ichimoku, RSI-divergentie, ATR-trail, TP1/TP2 |
| Sprint 2 | +6.29 | +1.3% | Funding gate, Optuna heroptimalisatie |
| Sprint 3 | +11.44 | +82% | Kelly sizing, regime SL/TP, correlatie guard |
| Sprint 4 | +12.99 | +14% | Google Trends (3 features) |
| Sprint 5 | +6.80* | -48% | HMM getest, code aanwezig maar uitgesloten (regressie) |
| Sprint 6 | +13.13 | +93% | 19 schadelijke features verwijderd (66 -> 47) |
| Sprint 7 | +24.18 | +84% | Bybit OI + Blockchain.info on-chain + Fear&Greed momentum (47 -> 52) |
| Sprint 8 | +24.76 | +2.4% | Optuna 150 trials + Sharpe objective + daily gate infra |
| Sprint 9 | ~15–25 | ±var | Model selectie op Sharpe in comparison; per-symbool config; geen structurele Sharpe winst |

*Sprint 5 regressie: nieuwe marktdata + HMM redundant met ADX market_regime.
ETH: Sprint 5 +5.57 -> Sprint 6 +7.83 (+41%) -> Sprint 7 +11.97 (+53%) -> Sprint 8 +7.77 (-35%)*
ETH Sprint 8 regressie: Optuna Sharpe objective overfits ETH validatieperiode. BF: RandomForest scoort +13.14 maar niet geselecteerd (AUC-selectie).

---

## Feature analyse (Sprint 6, 2026-03-28)

Permutation importance (20 iteraties, ROC AUC, out-of-sample validatieset).

### Top-15 features (BTC / ETH)

| Feature | BTC | ETH | Bron |
|---|---|---|---|
| google_trends_btc | +0.047 | +0.016 | pytrends (wekelijks) |
| eurusd_return_24h | +0.040 | +0.019 | yfinance |
| spx_return_24h | +0.016 | +0.026 | yfinance |
| dxy_return_24h | +0.030 | +0.020 | yfinance |
| trends_momentum_4w | +0.024 | +0.023 | pytrends |
| fear_greed | +0.006 | +0.021 | alternative.me |
| vix_level | +0.019 | +0.005 | yfinance |
| hour_of_week | +0.019 | +0.009 | berekend |
| btc_dvol | +0.016 | +0.010 | Deribit API |
| macd_4h | +0.016 | +0.004 | OHLCV (4h) |
| prev_day_return | +0.009 | +0.016 | OHLCV |
| usdjpy_return_7d | +0.013 | +0.011 | yfinance |
| dxy_return_7d | +0.010 | +0.016 | yfinance |
| return_30d | +0.011 | +0.009 | OHLCV |
| usdjpy_return_24h | +0.009 | +0.010 | yfinance |

Conclusie: Macro (DXY, EUR/USD, VIX, SPX, USD/JPY) en sentiment (Google Trends,
Fear & Greed) domineren. Technische indicatoren zijn ondersteunend, niet leidend.

### Verwijderde features (19 stuks — nul of negatief in BTC en ETH)

| Feature | BTC | ETH | Reden |
|---|---|---|---|
| btc_put_call_ratio | 0.000 | 0.000 | Geen signaal |
| btc_dominance | 0.000 | 0.000 | Datakwaliteit onvoldoende |
| btc_dominance_7d_chg | 0.000 | 0.000 | idem |
| is_hammer | 0.000 | 0.000 | Candlestick patronen te schaars voor 24h horizon |
| is_engulfing | 0.000 | 0.000 | idem |
| gap_up | 0.000 | 0.000 | idem |
| ema_ratio_50 | -0.00034 | -0.00045 | Redundant met ema_ratio_20 + price_vs_ema200 |
| chikou_position | -0.00012 | -0.00017 | Ichimoku redundantie (cloud + tk_cross volstaan) |
| volume_ratio | -0.00010 | -0.00008 | Ruis op 24h predictie-horizon |
| return_4h | -0.00007 | -0.00025 | Redundant met return_2h en return_7d |
| dxy_above_200ma | -0.00059 | -0.00008 | Regime-vlag; dxy_return_7d volstaat |
| session | -0.00001 | 0.000 | hour_of_week dekt tijdssessies al |
| return_12h | +0.00001 | -0.00051 | Conflicterend, netto negatief |
| return_6h | +0.00008 | -0.00008 | Netto nul |
| price_position | -0.00127 | +0.00026 | Duidelijk negatief voor BTC |
| lower_wick_pct | -0.00003 | 0.000 | Microstructuur niet relevant voor 24h horizon |
| upper_wick_pct | 0.000 | 0.000 | idem |
| candle_body_pct | +0.00012 | +0.00003 | Marginaal, ruis |
| volume_spike_48h | +0.00037 | -0.00029 | Conflicterend (BTC licht positief, ETH negatief) |

---

## Roadmap — Volgende sprints

### Sprint 7 — Nieuwe signalen met bewezen historische data

Doel: Aanvullen op het schone 47-feature fundament met kwalitatieve nieuwe bronnen.

#### S7-A Open Interest (OI) trend — VOLTOOID
- Signaal: OI + prijs stijgen samen = echte koop-interesse; OI stijgt + prijs daalt = distributie
- Features: oi_return_24h, oi_price_divergence (+1/0/-1)
- Bron: Coinglass gratis API (tot 180 dagen history zonder key)
- Blokkade: Binance Futures API slechts 30 dagen; evalueer Coinglass endpoint eerst
- Implementatie: src/external_data.py -> fetch_open_interest_coinglass()

#### S7-B On-chain basismetrics — VOLTOOID
- Signaal: Exchange netflow positief = coins naar beurs (verkoopdruk); SOPR > 1 = winst nemen
- Features: exchange_netflow_btc, sopr
- Bron: CryptoQuant Community (beperkt gratis) of Glassnode Community (gratis)
- Voordeel: Fundamenteel, lage correlatie met technische indicatoren

#### S7-C Fear & Greed momentum — VOLTOOID (geen CryptoPanic, API dood)
- Signaal: Sterk negatief nieuws = contrair entry of bevestigt bear
- Features: news_sentiment_score (-1 tot +1), news_volume_spike
- Bron: CryptoPanic API (gratis, 100 req/dag)
- Complementair aan Google Trends (nieuws vs. zoekgedrag)

#### S7-D USDT Dominance — wacht op data
- Code + data aanwezig maar slechts 25% datadekking (1 jaar CoinGecko history)
- Activeren wanneer 2+ jaar beschikbaar (apr 2027) of via CoinMarketCap
- Features: usdt_dominance, usdt_dominance_7d_chg

---

### Sprint 8 — Model architectuur verbetering — VOLTOOID

#### S8-A Optuna budget verhoging — VOLTOOID
- 50 trials -> 150 trials via `config.OPTUNA_N_TRIALS`
- BTC: stabielere params, +24.18 -> +24.76

#### S8-B Sharpe-based Optuna objective — VOLTOOID (gemengd resultaat)
- Objective: Sharpe op validatieset ipv ROC AUC
- Penalty: -10.0 als < 10 trades (voorkomt degeneratie naar 0 trades)
- BTC: lichte verbetering (+2.4%); ETH: regressie (-35%) door overfit validatieperiode

#### S8-C Daily alignment gate infra — VOLTOOID (wacht op daily model)
- Code aanwezig in backtest.py; blokkeert 1h longs als daily model bearish is
- Activeren zodra `BTCUSDT_1d_model.pkl` + `ETHUSDT_1d_model.pkl` beschikbaar zijn

---

### Sprint 9 — Infrastructuur (VOLTOOID, geen Sharpe winst)

#### S9-A Sharpe-gebaseerde model selectie — VOLTOOID
- model_compare.py selecteert nu model_best.pkl op Sharpe (met min_trades=20 guard)
- Automatische promotie naar model.pkl uitgeschakeld: model_compare Sharpe (zonder Kelly) ≠ productie-Sharpe
- Conclusie: model_best.pkl = correctere referentie; model.pkl = Optuna-LightGBM blijft primair

#### S9-B Symbool-specifieke Optuna objective — VOLTOOID (reverted)
- Bevinding: AUC objective voor ETH geeft lagere Sharpe (+6.55) dan Sharpe objective (+7.77)
- Conclusie: Sharpe objective (met penalty fix) is beter voor BEIDE symbolen
- Config: `OPTUNA_SHARPE_SYMBOLS = []` → valt terug op `OPTUNA_SHARPE_OBJECTIVE=True`

---

### Sprint 10 — Betrouwbaarheid & variantie reductie

**Kernprobleem ontdekt (Sprint 9):** run-to-run Sharpe variantie is ±30% (BTC: 15-25 range).
Dit maakt het onmogelijk om sprints eerlijk te vergelijken op single-run Sharpe.

#### S10-A Walk-forward Sharpe als primaire metric — prioriteit hoog
- Vervang single-run backtest door 3-fold walk-forward Sharpe (gemiddelde over 3 periodes)
- Elke fold = 90 dagen test; schuif 90 dagen per fold
- Geeft stabielere metric; onderscheidt echte signal van run-to-run ruis

#### S10-B Seed-fixing voor vergelijkbaarheid — prioriteit medium
- Voeg `random_state` toe aan alle Optuna runs (reproduceerbaar)
- LightGBM seed fixen: `seed=42` in params
- Maakt A/B vergelijking tussen sprints betrouwbaarder

#### S10-C Daily model trainen (1d timeframe) — prioriteit medium
- Train apart model op dagelijkse OHLCV + macro features
- Activeer daily gate in backtest.py (infra al aanwezig)
- Verwacht: filtert bear-markt entries, verbetert precision

#### S10-D Dynamische TP op ATR-basis — prioriteit laag
- Huidig: vaste TP% per regime (bull 8%, ranging 6%, bear 4%)
- Verbetering: TP = entry x (1 + N x ATR/close); N = 3.0/2.5/2.0 per regime

---

## Prioriteitenmatrix

| Taak | Impact | Moeite | Prioriteit | Status |
|---|---|---|---|---|
| S7-A OI trend (Bybit) | Hoog | Medium | *** | [x] |
| S7-B On-chain (blockchain.info) | Hoog | Medium | *** | [x] |
| S7-C Fear&Greed momentum | Medium | Laag | ** | [x] |
| S7-D USDT dominantie | Medium | Laag | * | [ ] wacht op data (apr 2027) |
| S8-A Optuna 150 trials | Hoog | Laag | *** | [x] |
| S8-B Sharpe objective | Medium | Laag | ** | [x] gemengd (BTC+, ETH-) |
| S8-C Daily gate infra | Hoog | Medium | *** | [x] code aanwezig, wacht op daily model |
| S9-A Sharpe model selectie | Hoog | Laag | *** | [x] infra OK, geen productie impact |
| S9-B Symbool-specifieke objective | Hoog | Laag | *** | [x] reverted: Sharpe obj. beter voor beiden |
| S10-A Walk-forward Sharpe metric | Hoog | Medium | *** | [ ] kern prioriteit |
| S10-B Seed-fixing reproduceerbaar | Medium | Laag | ** | [ ] |
| S10-C Daily model (1d timeframe) | Hoog | Medium | *** | [ ] |
| S10-D Dynamische ATR-TP | Medium | Laag | ** | [ ] |

---

## Geimplementeerd (archief)

| Sprint | Datum | Items | Sharpe BTC |
|---|---|---|---|
| Sprint 1 | 2026-03-27 | DXY, Ichimoku, candlestick features, RSI-divergentie, ATR trailing stop, TP1/TP2 partieel | +5.94 -> +6.21 |
| Sprint 2 | 2026-03-27 | Funding gate, Halving cyclus code, BTC-ETH correlatie code, Supertrend code, USDT dominantie code | +6.21 -> +6.29 |
| Sprint 3 | 2026-03-27 | Kelly Criterion sizing, Regime SL/TP (bull/ranging/bear), Multi-asset correlatie guard | +6.29 -> +11.44 |
| Sprint 4 | 2026-03-28 | Google Trends (3 features), Deribit 25D skew live gate | +11.44 -> +12.99 |
| Sprint 5 | 2026-03-28 | HMM regime detectie code (features.py); uitgesloten na regressietest (HMM +10.05, corr +7.76 vs baseline +12.99) | +12.99 -> +6.80 |
| Sprint 6 | 2026-03-28 | Permutation importance analyse (20 iteraties BTC+ETH), 19 schadelijke features verwijderd (66 -> 47) | +6.80 -> +13.13 |
| Sprint 7 | 2026-03-28 | Bybit OI (oi_return_24h, oi_price_divergence), blockchain.info (active_addresses, hash_rate), Fear&Greed 7d momentum | +13.13 -> +24.18 |
| Sprint 8 | 2026-04-02 | Optuna 150 trials, Sharpe objective (penalty <10 trades), daily gate infra | +24.18 -> +24.76 |
| Sprint 9 | 2026-04-02 | Model_compare selectie op Sharpe; OPTUNA_SHARPE_SYMBOLS config; per-symbool objective infra | geen meetbare winst; hoge run-to-run variantie ontdekt |
