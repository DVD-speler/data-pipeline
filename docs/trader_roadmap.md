# Trader Roadmap

---

## Huidige staat van het model (2026-04-03)

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
| Sprint 10 | WF +0.76 | — | Walk-forward Sharpe rapport (3 folds); WF = stabielere metric dan single-run |
| Sprint 11 | WF +2.21 | +191% | Bear-regime short model; bear-fold van 0 trades naar +9.76 |
| Sprint 12 | WF +5.57 | +152% | Short threshold tuning + BTC daily gate (AUC 0.63) |
| Sprint 13 | WF +4.05 | -27% | ETH daily verbetering mislukt; 28 features hersteld; run-to-run var |

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

### Sprint 10 — Betrouwbaarheid & variantie reductie — VOLTOOID

#### S10-A Walk-forward Sharpe rapport — VOLTOOID
- `wf_sharpe_report()` toegevoegd in model.py; wordt na elke backtest getoond
- 3 folds × 30 dagen = sep 2025 → apr 2026 (bull + correctie + ranging)
- Bevinding: BTC WF mean +0.76 (std 3.4), ETH WF mean +2.90 (std 5.5)
- Fold jan-feb = 0 trades = correct gedrag (bear markt + regime filter blokkeert longs)
- Single-run Sharpe > WF Sharpe doordat calibratie + Kelly + 4h gate significant bijdragen

#### S10-B Seed-fixing — VOLTOOID (was al goed)
- LightGBM, RF, XGBoost: allen `random_state=42` → run-to-run variantie was Optuna + data-afhankelijk
- Optuna: zelf stochastisch → variantie onvermijdelijk

---

### Sprint 11 — Bear-regime short model — VOLTOOID

#### S11-A Bear-regime short model — VOLTOOID
- Short signaal: `market_regime == -1` AND `proba < 0.30` AND `return_30d < -3%`
- Geen conflict met longs (short alleen bij signal_long == 0)
- Positie sizing: `(0.5 - proba) * 2 * vol_scale` (symetrisch met longs)
- Stop-loss: ATR-gebaseerd (zelfde als longs)
- Resultaat: bear-fold (jan-feb) van 0 trades naar +9.76 BTC / +6.97 ETH

#### S11-B WF rapport verbeteren — doorgeschoven naar Sprint 12
#### S11-C Daily model trainen — doorgeschoven naar Sprint 12

---

### Sprint 12 — Volgende verbeteringen

#### S12-A Short model tuning — VOLTOOID
- `optimize_short_threshold()` opnieuw geïmplementeerd (sweept 0.20–0.40 op bear val-data)
- BTC: val te bullish → fallback config 0.30. ETH: 0.26 gevonden maar short disabled
- `BEAR_REGIME_SHORT_SYMBOLS = ["BTCUSDT"]` — short alleen actief voor BTC

#### S12-B Daily model trainen — VOLTOOID (BTC only)
- BTC daily model: AUC 0.63 → gate actief, WF fold 3: -8.17→0.00 (blokkering werkt!)
- ETH daily model: AUC 0.53 → gate uitgeschakeld (te zwak, hurt performance)
- `DAILY_GATE_SYMBOLS = ["BTCUSDT"]` — gate alleen voor BTC
- Bug gefixed: `fase_model_daily()` overschreef hourly model → backup/restore logica

#### S12-C WF rapport verbeterd — doorgeschoven naar Sprint 13

---

### Sprint 13 — ETH verbetering + model diversificatie

#### S13-A ETH daily model verbeteren — VOLTOOID (geen verbetering)
- Poging 1: 38 features → BTC AUC 0.57, ETH AUC 0.49 (curse of dimensionality, <500 trainrows)
- Poging 2: 31 features (top-3 extra) → zelfde probleem
- Poging 3: revert 28 features + AUC objective (Sharpe obj had <10 trades penalty → degeneratie)
- Resultaat: BTC AUC 0.6302 (baseline hersteld), ETH AUC 0.5268 (geen verbetering)
- Conclusie: ETH daily gate blijft uitgeschakeld; ETH dagelijkse patronen te weinig predictief
- WF BTC S13: +4.05 (std 7.39); ETH S13: +1.91 (std 4.52) — run-to-run variantie

#### S13-B ETH short model herintroductie — VERVALT
- Focus verschoven naar BTC optimalisatie (zie analyse 2026-04-03)
- ETH AUC 0.53 maakt betrouwbare short-signalen onmogelijk
- Revisie: pas na structurele ETH model verbetering (niet voor Sprint 17+)

#### S13-C WF rapport met calibratie — doorgeschoven naar Sprint 18
- Deprioritized t.o.v. regime-detectie en drempel-optimalisatie
- Blijft relevant als referentie-metric, maar blokkeert geen hogere-impact sprints

---

## Analyse: tekortkomingen geïdentificeerd (2026-04-03)

Op basis van 30-fold walk-forward analyse (mrt 2023 → feb 2026) zijn drie structurele zwaktes ontdekt:

### Zwakte 1 — Model mist explosieve bull runs
- Folds 9 (BTC +61%, model -2%), 16 (BTC +29%, model -6%): threshold te hoog bij sterke trends
- ADX-regime offset slechts -0.05 in bull → te conservatief bij bevestigde opwaartse trend
- Oorzaak: traindata heeft weinig extreme bull periodes → model underfit op explosieve bewegingen

### Zwakte 2 — Vals-negatieve short-signalen in ranging/kantelende markten
- Folds 12, 21, 25: ranging of naar boven kantelende markt triggert bear-shorts → -14%, -15%, -14%
- ADX is een lagging indicator; regime-wissel wordt 2–4 candles te laat gedetecteerd
- BB-breedte en MACD-histogram stabiliteit zijn ongebruikte aanvullende regime-signalen

### Zwakte 3 — Cumulatieve verliezen bij crashes
- Fold 29 (BTC -28%, model -19%): beter dan B&H maar nog -19% absoluut verlies
- Fixed stop-loss -3% per trade; shorts in snelle crash genereren meerdere kleine verliezen
- Crash-modus (versnelde daling) wordt niet apart gedetecteerd

---

### Sprint 14 — Multi-indicator regime classificatie (PRIORITEIT 1)

**Doel:** Zwakte 2 aanpakken — betere onderscheid bull / ranging / bear voordat shorts worden geactiveerd

**Verwachte WF Sharpe winst:** +1.0–2.0

#### S14-A BB-breedte + MACD-stabiliteit als regime-features — VOLTOOID
- `bb_width` (al berekend in _add_ta_indicators) + `macd_hist_stability` (rolling std/5 van genorm. histogram) toegevoegd aan FEATURE_COLS_1H
- Volgorde-bug gefixed: macd_hist_stability moet vóór ranging_score berekend worden

#### S14-B Ensemble regime-score — VOLTOOID
- `ranging_score` = ADX<20 + bb_width<0.025 + macd_hist_stability>0.0002 (som 0–3)
- Toegevoegd aan FILTER_COLS (niet als model-input)
- Config params: RANGING_BB_WIDTH_THR=0.025, RANGING_MACD_STB_THR=0.0002, RANGING_SCORE_THR=2

#### S14-C Shorts filteren op ranging_score — VOLTOOID
- `not_ranging = (ranging_score < 2)` toegevoegd als extra filter in backtest.py short-signaal
- Probleem-folds: 0 false-shorts in fold12/21 (was 30/42 shorts → beide nu 0)

#### Evaluatie Sprint 14:
- **Mediaan Sharpe: +6.14** (was 0.0 in S13) — grote verbetering
- **Winstgevende folds: 20/31** (was 14/30)
- **Negatieve Sharpe folds: 10/31** (was 13/30)
- **Gem shorts per fold: 7.0** (was 24.2) — 71% minder false-short signalen
- **Gem longs per fold: 90.2** (was 26.5) — model heeft meer vertrouwen door betere features
- Probleem-fold12: -14.0% → **+44.6%** (0 shorts nu vs 30 eerder)
- Probleem-fold25: -13.9% → **+102.7%**
- Sprint geslaagd — alle acceptatiecriteria behaald

#### S14-A BB-breedte + MACD-stabiliteit als regime-features — HISTORISCH (zie boven)
- Hieronder stond de originele planning, ter referentie bewaard
- BB-breedte = (upper - lower) / SMA(20): laag = squeeze/ranging, hoog = trending
- MACD-histogram stabiliteit = rolling std(5) van MACD_hist: hoog = noisy/ranging, laag = trending
- Toevoegen aan `src/features.py` als `bb_width` en `macd_hist_stability`
- Retrain + permutation importance: verwacht top-10 feature bij bull/bear detectie

#### S14-B Ensemble regime-score — prioriteit hoog
- Combineer drie signalen: ADX(<20), BB-breedte(<drempel), MACD-stabiliteit(>drempel)
- `ranging_score` = som van de drie (0–3): ≥2 = RANGING, negeer short-signalen
- `strong_trend_score` = ADX>25 + BB-breedte>0.06 + MACD_hist stabiel: ≥2 = STERKE TREND
- Vervang huidige `market_regime` binaire check door ensemble score in backtest.py
- Validatie: manueel check folds 12, 21, 25 — worden false-short gevallen gefilterd?

#### S14-C Shorts uitschakelen bij ranging_score ≥ 2 — prioriteit hoog
- In `src/backtest.py`: voeg `ranging_score` als filter toe vóór short-signaal
- Short pas actief als: `market_regime == -1` AND `ranging_score < 2` AND `return_30d < -3%`
- Verwacht: folds 12/21/25 verbeteren van -14% naar ~0% (geen trades in ranging)

**Acceptatiecriteria:**
- WF mean Sharpe ≥ +1.0 t.o.v. S13
- Folds 12, 21, 25 geen negatieve returns meer door false shorts
- Geen regressie op bull-folds (6, 15, 26)

---

### Sprint 15 — Dynamische drempelaanpassing in bevestigde trends — TERUGGEDRAAID

**Doel:** Zwakte 1 aanpakken — model eerder laten instappen bij explosieve bull moves

**Verwachte WF Sharpe winst:** +0.5–1.5

#### S15-A ADX-geschaalde long-drempel — GEÏMPLEMENTEERD, daarna TERUGGEDRAAID
- ADX-band bonus geïmplementeerd: ADX 30–39 → -0.03, ADX 40+ → -0.07 extra drempelverlaging
- `config.ADX_THRESHOLD_OFFSETS = {30: -0.03, 40: -0.07}`

#### S15-B 4h confluence bonus — GEÏMPLEMENTEERD, daarna TERUGGEDRAAID
- `config.TREND_4H_THRESHOLD_BONUS = 0.03` (drempel -0.03 bij 4h proba > 0.65)

#### Evaluatie Sprint 15 — MISLUKT, teruggedraaid:
- **Mediaan Sharpe: +3.52** (S14 baseline: +6.14) — regressie -2.62
- **Gem Sharpe: +1.01** (S14: +3.86) — regressie -2.85
- **Winstgevende folds: 19/31** (S14: 20/31)
- **Oorzaak regressie:** lagere drempel genereert te veel marginale trades in bull-regime
  - Fold27: S14=0 trades (veilig, Sharpe +33.9) → S15=1 verliezende trade (Sharpe -23.8)
  - 21 van 31 folds slechter met S15 dan S14
- **Besluit:** S15 teruggedraaid — ADX-bonus verlaagt de selectiviteit te agressief
- **Nieuwe baseline:** mediaan Sharpe **+7.14** (32 folds, inclusief nieuwe data tot april 2026)
- Code: `ADX_THRESHOLD_OFFSETS = {}` en `TREND_4H_THRESHOLD_BONUS = 0.0` in config.py
- S15-A/B blokken verwijderd uit `src/backtest.py`

#### S15-C Drempeloptimalisatie per ADX-band — UITGESTELD naar Sprint 17
- Sweep: (ADX < 20), (20–30), (30–40), (>40) — welke offset-combinatie geeft beste WF?
- Optuna-sweep op validatieset per ADX-band — vereist zorgvuldigere aanpak dan vaste offsets

**Acceptatiecriteria (niet behaald):**
- ~~Folds 9 en 16 verbeteren~~
- ~~WF mean Sharpe ≥ +0.5 t.o.v. Sprint 14 baseline~~

---

### Sprint 16 — ATR-scaled stops + crash detector — VOLTOOID

**Doel:** Zwakte 3 aanpakken — minder cumulatief verlies bij snelle crashes

#### S16-A ATR-gebaseerde stop-loss — VOLTOOID (was al gedeeltelijk geïmplementeerd)
- Code gebruikte al `2.0 × atr_pct` als dynamische stop (hardcoded)
- Configureerbaar gemaakt via `config.ATR_STOP_MULTIPLIER = 2.0` in `config.py`
- `src/backtest.py`: hardcoded `2.0` vervangen door `getattr(config, "ATR_STOP_MULTIPLIER", 2.0)`

#### S16-B Crash-modus detector — VOLTOOID
- `crash_mode` binary feature toegevoegd aan `src/features.py`
  - Actief als: `return_1h < -2.5 × rolling_vol_24h` OF `return_24h < -10%`
  - 2.6% van alle uren actief — correct voor zeldzame crash-events
- Toegevoegd aan `config.FILTER_COLS` (niet als model-input)
- `src/backtest.py`: positiegrootte × 0.5 bij crash_mode=1 (binnen `use_position_sizing=True` blok)
- Config: `CRASH_SIGMA_THR=2.5`, `CRASH_RETURN_THR=0.10`, `CRASH_SIZE_FACTOR=0.5`

#### S16-C Per-trade circuit breaker — UITGESTELD
- Te complex voor vectorized backtest architectuur; vereist trade-level simulatie

#### Evaluatie Sprint 16 — AANVAARD (robuuster profiel):
- **Mediaan Sharpe: +7.33** (baseline: +7.14) — licht verbeterd (+0.19)
- **Gem Sharpe: +6.22** (baseline: +7.35) — gedaald door gedempt outlier fold24 (+44.9→+18.6)
- **Positieve folds: 20/32** — onveranderd
- **Crash-bescherming werkt:** fold4 -4.6→-2.6, fold11 -3.1→+8.5
- **Effect:** modereert uitschieters (minder variance), stabielere mediaan
- **Besluit:** aanvaard — robuuster profiel is meer waard dan hogere gemiddelde met extreme outliers
- WF resultaten: `data/stats/walkforward_lightgbm.csv`

---

### Sprint 17 — Pyramiding in sterke trends (PRIORITEIT 3)

**Doel:** Grotere positie opbouwen in bevestigde trends om meer van bull moves te profiteren

#### S17-A Add-on logica — UITGESTELD
- Vereist trade-level simulatie (weet "trade zit in winst") — niet mogelijk in vectorized backtest
- Doorgeschoven naar latere sprint als backtest-architectuur wordt uitgebreid

#### S17-B Momentum-gewogen sizing — VOLTOOID, AANVAARD (marginaal positief)
- `macd_size_mult` feature toegevoegd aan `src/features.py` (FILTER_COL)
  - `macd_size_mult = clip(macd_hist.clip(0) / rolling_mean_abs_20d, 0, 0.5)`
- `src/backtest.py`: `long_size *= (1 + macd_size_mult)` bij `MACD_MOMENTUM_SCALE=True`
- `config.py`: `MACD_MOMENTUM_SCALE = True`

#### Evaluatie Sprint 17 — AANVAARD (bescheiden verbetering):
- **Gem Sharpe: +6.357** (S16: +6.224) — +0.133
- **Mediaan Sharpe: +7.122** (S16: +7.333) — -0.211 (te klein om significant te zijn)
- **Positieve folds: 20/32** — onveranderd
- **16 van 32 folds verbeterd** (7 verslechterd, 9 onveranderd) — 2:1 verhouding
- **Bull-run captures verbeterd:** fold26 +291%→+366%, fold15 +154%→+184%
- **Kanttekening:** hogere returns in bull folds gaan gepaard met hogere variance → Sharpe neutraal
- Acceptatiecriterium (+0.3 Sharpe gem) **niet gehaald**, maar geen regressie → aanvaard
- WF resultaten: `data/stats/walkforward_lightgbm.csv`

---

### Sprint 18 — SHAP analyse + WF calibratie — VOLTOOID (analyse nuttig, code teruggedraaid)

**Doel:** Model blind spots identificeren; WF Sharpe vergelijkbaar maken met single-run Sharpe

#### S18-A SHAP feature interactie analyse — VOLTOOID, features TERUGGEDRAAID

**SHAP bevindingen (top-5 meest impactvolle features op testset):**
1. `dxy_return_24h` — 0.164 (sterkste predictor: dollar richting)
2. `trends_momentum_4w` — 0.142 (Google trends 4w momentum)
3. `dxy_return_7d` — 0.123 (dollar trend)
4. `google_trends_btc` — 0.120 (retail FOMO)
5. `eurusd_return_24h` — 0.118 (EUR/USD als DXY proxy)

**SHAP per regime (top-3):**
- BEAR: `dxy_return_24h`, `google_trends_btc`, `trends_momentum_4w`
- RANGING: `dxy_return_24h`, `trends_momentum_4w`, `spx_return_24h`
- BULL: `trends_momentum_4w`, `dxy_return_24h`, `bb_pct`

**3 sterkste interacties gevonden:**
1. `macro_risk_score` (DXY↓ + SPX↑ = risk-on): SHAP-delta +0.29 — sterkste interactie
2. `fomo_uptrend_score` (google_trends × return_30d>0): SHAP-delta +0.11
3. `dxy_momentum_align` (beide DXY timeframes negatief): aanhoudende dollarweakheid

**A/B test resultaat — TERUGGEDRAAID:**
- Mediaan Sharpe: 7.12 → 4.53 (-2.59) — regressie
- Oorzaak: LightGBM vangt interacties al op via boomsplitsingen; expliciete interactie-features voegen redundantie toe → verlegt drempeloptimalisatie verkeerd
- **Conclusie:** SHAP is waardevol voor interpretatie maar interactie-features niet toevoegen als model-input bij boom-modellen

#### S18-B WF calibratie — GEÏMPLEMENTEERD, daarna TERUGGEDRAAID
- Per-fold isotonische calibratie op validatieset geïmplementeerd
- Resultaat: fold4 Sharpe -82.9, fold27 Sharpe +99.6 (extreme instabiliteit)
- Oorzaak: kleine validatieset (≈1152 rijen) maakt isotone calibratie instabiel; verschuift drempeloptimalisatie
- **Conclusie:** calibratie vereist minstens 5000× val-rijen of Platt scaling in plaats van isotonic

#### Evaluatie Sprint 18 — GEEN CODE WIJZIGINGEN ACTIEF:
- Beide aanpassingen teruggedraaid na regressie-testen
- **Waardevol inzicht:** `dxy_return_24h` is dominante predictor; macro-omgeving > technische indicatoren
- **Aanbeveling voor toekomstige sprint:** betere DXY/SPX data coverage (huidig 28-31% null) verbeteren

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
| S10-A Walk-forward Sharpe rapport | Hoog | Medium | *** | [x] BTC +0.76, ETH +2.90 WF Sharpe |
| S10-B Seed-fixing reproduceerbaar | Medium | Laag | ** | [x] al gedaan |
| S11-A Bear-regime short model | Hoog | Medium | *** | [x] WF BTC +191%, ETH +67% |
| S12-A Short threshold optimalisatie | Hoog | Laag | *** | [x] per-symbool (BTC only) |
| S12-B Daily model BTC | Hoog | Medium | *** | [x] AUC 0.63, WF fold3: -8→0 |
| S12-C Daily model ETH | Medium | Medium | ** | [ ] AUC 0.53 te zwak — on hold |
| S13-A ETH daily model verbeteren | Hoog | Medium | *** | [x] geen verbetering; ETH AUC 0.53 (BTC 0.63 hersteld) |
| S13-B ETH short model herintroductie | Medium | Medium | ** | [x] VERVALT — focus naar BTC optimalisatie |
| S13-C WF rapport met calibratie | Laag | Medium | * | [ ] doorgeschoven → S18-B |
| **S14-A** BB-breedte + MACD-stabiliteit features | Hoog | Laag | **** | [x] mediaan Sharpe 0.0→+6.14 |
| **S14-B** Ensemble regime-score (3-indicator) | Hoog | Medium | **** | [x] ranging_score in FILTER_COLS |
| **S14-C** Shorts filteren op ranging_score | Hoog | Laag | **** | [x] 71% minder false-shorts |
| **S15-A** ADX-geschaalde long-drempel | Hoog | Laag | **** | [x] TERUGGEDRAAID — regressie |
| **S15-B** 4h confluence bonus (drempel verlaging) | Medium | Laag | *** | [x] TERUGGEDRAAID — regressie |
| **S15-C** Drempeloptimalisatie per ADX-band | Medium | Medium | *** | [ ] → S17 |
| **S16-A** ATR-gebaseerde stop-loss | Hoog | Medium | *** | [x] configureerbaar (ATR_STOP_MULTIPLIER=2.0) |
| **S16-B** Crash-modus detector | Hoog | Medium | *** | [x] crash_mode feature + positie halvering |
| **S16-C** Per-trade circuit breaker | Medium | Medium | ** | [ ] → uitgesteld (vereist trade-level sim) |
| **S17-A** Add-on logica (pyramiding) | Medium | Hoog | ** | [ ] → uitgesteld (vereist trade-sim) |
| **S17-B** Momentum-gewogen entry sizing | Laag | Medium | * | [x] MACD_MOMENTUM_SCALE +0.13 gem Sharpe |
| **S18-A** SHAP feature interactie analyse | Medium | Medium | ** | [x] TERUGGEDRAAID — LGB vangt interacties zelf op |
| **S18-B** WF rapport met calibratie | Laag | Medium | * | [x] TERUGGEDRAAID — instabiel op kleine val-set |

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
| Sprint 10 | 2026-04-03 | wf_sharpe_report() toegevoegd (3 folds); seed-fixing was al goed | WF BTC +0.76, ETH +2.90 (bear-markt fold = 0 trades, correct) |
| Sprint 11 | 2026-04-03 | Bear-regime short model (market_regime=-1, proba<0.30, return_30d<-3%) | WF BTC +0.76→+2.21 (+191%), ETH +2.90→+4.84 (+67%); bear-fold nu winstgevend |
| Sprint 12 | 2026-04-03 | Short threshold optim. op val-set; daily 1d model actief (BTC only, AUC 0.63); per-symbool gates | WF BTC +5.57 (+632% vs S10), ETH +2.97 (short disabled, ETH daily AUC 0.53 te zwak) |
| Sprint 13 | 2026-04-03 | S13-A: ETH daily model verbeteren mislukt (curse of dimensionality); revert 28 features + AUC obj; BTC AUC 0.63 hersteld | WF BTC +4.05 (run-to-run var), ETH +1.91; ETH daily gate uitgeschakeld |
| Sprint 14 | 2026-04-03 | BB-breedte + MACD-hist-stabiliteit features; ensemble ranging_score filter; false-shorts in ranging markten 71% gereduceerd | Mediaan WF Sharpe 0.0→+6.14; 20/31 folds winstgevend; fold12: -14%→+45% |
