# Trader Roadmap

---

## Huidige staat van het model (2026-03-28)

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

*Sprint 5 regressie: nieuwe marktdata + HMM redundant met ADX market_regime.
ETH: Sprint 5 +5.57 -> Sprint 6 +7.83 (+41%)

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

#### S7-A Open Interest (OI) trend — prioriteit hoog
- Signaal: OI + prijs stijgen samen = echte koop-interesse; OI stijgt + prijs daalt = distributie
- Features: oi_return_24h, oi_price_divergence (+1/0/-1)
- Bron: Coinglass gratis API (tot 180 dagen history zonder key)
- Blokkade: Binance Futures API slechts 30 dagen; evalueer Coinglass endpoint eerst
- Implementatie: src/external_data.py -> fetch_open_interest_coinglass()

#### S7-B On-chain basismetrics — prioriteit hoog
- Signaal: Exchange netflow positief = coins naar beurs (verkoopdruk); SOPR > 1 = winst nemen
- Features: exchange_netflow_btc, sopr
- Bron: CryptoQuant Community (beperkt gratis) of Glassnode Community (gratis)
- Voordeel: Fundamenteel, lage correlatie met technische indicatoren

#### S7-C News sentiment (CryptoPanic) — prioriteit medium
- Signaal: Sterk negatief nieuws = contrair entry of bevestigt bear
- Features: news_sentiment_score (-1 tot +1), news_volume_spike
- Bron: CryptoPanic API (gratis, 100 req/dag)
- Complementair aan Google Trends (nieuws vs. zoekgedrag)

#### S7-D USDT Dominance — wacht op data
- Code + data aanwezig maar slechts 25% datadekking (1 jaar CoinGecko history)
- Activeren wanneer 2+ jaar beschikbaar (apr 2027) of via CoinMarketCap
- Features: usdt_dominance, usdt_dominance_7d_chg

---

### Sprint 8 — Model architectuur verbetering

#### S8-A Optuna budget verhoging — prioriteit hoog
- Huidig: 50 trials -> verhogen naar 150 trials
- Effect: stabielere hyperparameters, minder variantie tussen trainruns

#### S8-B Walk-forward Optuna — prioriteit medium
- Optuna apart per rolling 6-maands venster
- Voorkomt overfit op één marktfase (bijv. 2022 bear of 2024 bull)

#### S8-C LSTM ensemble — prioriteit laag (complex)
- LightGBM behandelt features onafhankelijk per tijdstip
- LSTM leert sequentiële patronen (serie rode candles + hammer = reversal)
- Blend: 0.6 x LightGBM + 0.4 x LSTM
- Vereist: PyTorch, 48h lookback window

---

### Sprint 9 — Live trading optimalisatie

#### S9-A Dynamische TP op ATR-basis — prioriteit hoog
- Huidig: vaste TP% per regime (bull 8%, ranging 6%, bear 4%)
- Verbetering: TP = entry x (1 + N x ATR/close); N = 3.0/2.5/2.0 per regime

#### S9-B Daily alignment filter — prioriteit hoog
- Huidig: 4h confirmatie (threshold 0.52)
- Verbetering: blokkeer 1h longs als daily model bearish is

#### S9-C Value at Risk (VaR) tracking — prioriteit medium
- Bereken dagelijkse 95% VaR op basis van historische volatiliteit
- Blokkeer nieuwe entries als VaR > 5% van kapitaal

#### S9-D Retest entry timing — prioriteit medium
- Huidig: entry op eerste uur dat proba > threshold
- Verbetering: wacht op pullback naar VWAP of EMA(20) na signaal

---

## Prioriteitenmatrix

| Taak | Impact | Moeite | Prioriteit | Status |
|---|---|---|---|---|
| S7-A OI trend (Coinglass) | Hoog | Medium | *** | [ ] |
| S7-B On-chain (SOPR / netflow) | Hoog | Medium | *** | [ ] |
| S7-C News sentiment | Medium | Medium | ** | [ ] |
| S7-D USDT dominantie | Medium | Laag | * | [ ] wacht op data |
| S8-A Optuna 150 trials | Hoog | Laag | *** | [ ] |
| S8-B Walk-forward Optuna | Medium | Medium | ** | [ ] |
| S8-C LSTM ensemble | Hoog | Hoog | ** | [ ] |
| S9-A Dynamische ATR-TP | Medium | Laag | *** | [ ] |
| S9-B Daily alignment filter | Hoog | Medium | *** | [ ] |
| S9-C VaR tracking | Medium | Laag | ** | [ ] |
| S9-D Retest entry timing | Medium | Medium | ** | [ ] |

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
