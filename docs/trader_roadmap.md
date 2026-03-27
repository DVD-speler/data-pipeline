# Trader Roadmap — Ontbrekende Verbeteringen

> Analyse vanuit het perspectief van een echte crypto-trader: wat gebruiken professionele traders
> dat dit model nog niet heeft? Geordend op **impact × implementatie-gemak**.

---

## Wat het model al heeft (referentie)

| Categorie | Geïmplementeerd |
|---|---|
| Technische indicatoren | RSI, MACD, BB, EMA (20/50/200), ADX, ATR, VWAP, Stoch RSI |
| Volume | Volume ratio, spike, momentum, Volume Profile (POC) |
| Multi-timeframe | 1h model + 4h gate + dagelijks regime gate |
| Macro | VIX, USD/JPY, EUR/USD, SPX, Fear & Greed, BTC Dominance, **DXY** |
| Opties-markt | DVOL (implied vol), Put/Call ratio |
| Funding | Funding rate + momentum |
| Candlestick | **Ichimoku Cloud, body/wick ratios, hammer, engulfing, gap_up** |
| RSI | **RSI divergentie (bull + bear)** |
| SL/TP | Structureel (swing levels) + **ATR trailing stop**, breakeven trail, **TP1/TP2 partieel** |
| Risicobeheer | Circuit breaker, volatility-scaled sizing, signal decay, regime gates |
| Exit | Proba-exit, 168h timeout, model-driven |

---

## Tier 1 — Hoge impact, laag complex (snel te implementeren)

### T1-A · Ichimoku Cloud features
**Status:** `[x]` — geïmplementeerd 2026-03-27

**Waarom traders het gebruiken:**
Ichimoku geeft in één oogopslag trend, momentum, support/resistance én tijdsfilter.
Professionele traders kijken of prijs **boven de wolk** staat (bullish), de wolk dik is
(sterke trend) en of de TK-cross (Tenkan × Kijun) een entry bevestigt.

**Features:**
- `cloud_position`: close boven (+1), in (0), onder (−1) de kumo (wolk)
- `cloud_thickness`: (senkou_a - senkou_b) / close — dikke wolk = sterke trend
- `tk_cross`: Tenkan boven Kijun (+1), eronder (−1)
- `chikou_position`: chikou span t.o.v. de koers 26 perioden terug

**Implementatie:**
1. `src/features.py` — `_add_ichimoku()` (tenkan=9, kijun=26, senkou_b=52)
2. `config.py` — 4 features aan `FEATURE_COLS_1H`
3. Geen externe data nodig — puur OHLCV

**Bestanden:** `src/features.py`, `config.py`

---

### T1-B · Candlestick patroon encoding
**Status:** `[x]` — geïmplementeerd 2026-03-27 · LightGBM belang ≈ 0 (patronen worden door andere features afgedekt)

**Waarom traders het gebruiken:**
Hammer, Doji, Engulfing en Pin Bar zijn de meest betrouwbare reversal-signalen
die elke professionele trader herkent. Ze vangt micro-structuur op die RSI/MACD missen.

**Features:**
- `candle_body_pct`: body / total range — klein = onzekerheid (Doji)
- `upper_wick_pct`: upper wick / range — lange wick boven = rejection (bearish)
- `lower_wick_pct`: lower wick / range — lange wick onder = buy tail (bullish)
- `is_hammer`: 1 als lower wick > 2× body én close in top 30% van range
- `is_engulfing`: 1 als bullish engulfing (close > prev open, open < prev close)
- `gap_up`: open / prev close - 1 (positief = gap omhoog na overnight)

**Implementatie:**
1. `src/features.py` — vectorized berekening op OHLCV kolommen
2. `config.py` — 6 features toevoegen

**Bestanden:** `src/features.py`, `config.py`

---

### T1-C · DXY (Dollar Index) als macro feature
**Status:** `[x]` — geïmplementeerd 2026-03-27 · **#1 feature voor BTC én ETH** (importance 0.070 / 0.099)

**Waarom traders het gebruiken:**
De Dollar Index (DXY) is de sterkste macro-correlatie voor crypto. DXY stijgt →
risico-assets dalen (BTC, aandelen). Traders monitoren DXY nauwer dan EUR/USD
omdat het een breder beeld geeft van dollarkracht vs. 6 valuta's.

**Features:**
- `dxy_return_24h`: dagelijkse DXY return (negatief = dollar zwakker = bullish crypto)
- `dxy_return_7d`: wekelijkse trend (aanhoudende dollarstijging = macro headwind)
- `dxy_above_200ma`: 1 als DXY boven zijn eigen 200-daags MA (structureel dollar-bull)

**Bron:** yfinance ticker `DX-Y.NYB` (gratis, geen key)
**Cache:** 24h

**Implementatie:**
1. `src/external_data.py` — `fetch_dxy()` (zelfde patroon als fetch_usdjpy)
2. `config.py` — features + optionele gate (dxy_return_7d > +3% blokkeert longs)
3. `src/features.py` — join via `load_all_external()`
4. `src/backtest.py` — DXY gate toevoegen aan lange filter-blok

**Bestanden:** `src/external_data.py`, `config.py`, `src/backtest.py`

---

### T1-D · ATR-gebaseerde trailing stop (dynamisch)
**Status:** `[x]` — geïmplementeerd 2026-03-27 · 2×ATR in run_backtest() + run_backtest_be_trail()

**Waarom traders het gebruiken:**
De huidige trailing stop gaat alleen naar breakeven (entry prijs). Professionele traders
gebruiken een **ATR-trailing stop**: de stop volgt de prijs omhoog met een ATR-buffer,
zodat winst geconsolideerd wordt terwijl de trend loopt.
Voorbeeld: ATR(14) = €800 → trailing stop = close − 2 × €800.

**Implementatie:**
1. `src/live_alert.py` — voeg `trailing_sl` toe aan positie-dict, update elk uur:
   ```
   trailing_sl = max(current_sl, close - 2 × atr)
   ```
2. `src/backtest.py` `run_backtest_be_trail()` — simuleer per candle

**Configuratie in config.py:**
- `ATR_TRAIL_MULTIPLIER = 2.0` (2 × ATR)
- `ATR_TRAIL_ENABLED = True`

**Bestanden:** `src/live_alert.py`, `src/backtest.py`, `config.py`

---

### T1-E · Partiële exits (TP1 / TP2 systeem)
**Status:** `[x]` — geïmplementeerd 2026-03-27 · TP1=+3% (50%), TP2=+8% in run_backtest_be_trail()

**Waarom traders het gebruiken:**
Een vaste 6% TP laat winst op tafel liggen in sterke trends, of sluit te laat in zwakke.
Professionele traders nemen **partieel winst** op TP1 (bijv. +3%) en laten de rest lopen
met een hogere TP2 (bijv. +8%), beschermd door een trailing stop.

**Systeem:**
- **TP1** (50% van positie): close +3% → haal 50% eruit, zet SL naar entry
- **TP2** (resterende 50%): close +8% óf model-exit óf 168h timeout
- Effectief R/R: TP1 pakt zeker 3%, TP2 schiet voor meer

**Implementatie:**
1. `config.py` — `TP1_PCT = 0.03`, `TP2_PCT = 0.08`, `TP1_SIZE_FRACTION = 0.5`
2. `src/live_alert.py` — positie-dict uitbreiden met `partial_closed`, `tp1_hit`
3. `src/backtest.py` — partiële exit logica per candle in `run_backtest_be_trail()`

**Bestanden:** `config.py`, `src/live_alert.py`, `src/backtest.py`

---

### T1-F · RSI-divergentie als feature
**Status:** `[x]` — geïmplementeerd 2026-03-27 · belang laag (~0.001), model gebruikt het nauwelijks

**Waarom traders het gebruiken:**
RSI-divergentie (prijs maakt nieuw dieptepunt, RSI niet) is een van de sterkste
reversal-signalen. Traders wachten op bullish divergentie vóór ze een long openen.

**Features:**
- `rsi_bull_divergence`: 1 als close < close[N] EN rsi > rsi[N] — bullish reversal
- `rsi_bear_divergence`: 1 als close > close[N] EN rsi < rsi[N] — bearish reversal
- Lookback N = 24h (dagelijks), met vereiste: rsi < 40 voor bullish, > 60 voor bearish

**Implementatie:**
1. `src/features.py` — vergelijk close en rsi op rolling N-period basis
2. `config.py` — 2 features toevoegen

**Bestanden:** `src/features.py`, `config.py`

---

## Tier 2 — Hoge impact, medium complex

### T2-A · Liquidation Heatmap / grote liquidaties
**Status:** `[ ]`

**Waarom traders het gebruiken:**
Grote long-liquidaties (flash crashes) en short-liquidaties (short squeezes) zijn
de sterkste korte-termijn prijsdrivers in crypto. Binance biedt een **gratis publieke API**
voor geaggregeerde liquidaties. Traders zien een liquidatie-piek als entry-signaal
(snel herstel na overreactie) of als uitstap-signaal (begin van cascade).

**Features:**
- `liq_long_1h`: gecumuleerde long-liquidaties afgelopen 1 uur (USD)
- `liq_short_1h`: gecumuleerde short-liquidaties afgelopen 1 uur (USD)
- `liq_ratio_1h`: liq_long / (liq_long + liq_short) — >0.7 = long squeeze
- `liq_spike`: 1 als liq_long_1h > 3× 24h gemiddelde (extreme liquidatie)

**Bron:** Binance Futures public API:
```
GET https://fapi.binance.com/futures/data/globalLongShortAccountRatio
```
Echter: geen historische liquidatie-data zonder betaald account.
**Alternatief**: Coinglass public API heeft beperkte gratis liquidatiedata.

**Implementatie:**
1. `src/external_data.py` — `fetch_liquidations()` met Coinglass fallback
2. `config.py` — 3-4 features toevoegen + liquidation-spike gate
3. `src/backtest.py` — gate: blokkeer longs bij extreme long-liquidatie cascade

**Bestanden:** `src/external_data.py`, `config.py`, `src/backtest.py`
**Risico:** API-beschikbaarheid beperkt voor historische data

---

### T2-B · Funding rate extremen als reversal-signaal
**Status:** `[x]` — geïmplementeerd 2026-03-27 · T2-B funding gate actief (funding > 0.05% blokkeert longs)

**Waarom traders het gebruiken:**
Een **extreem positieve** funding rate (+0.1%+ per 8u) betekent dat longs zo dominant
zijn dat ze shorts betalen voor bescherming. Dit is een **contrair signaal**: te veel
bullishness = overbought. Professionele traders vermijden longs bij extreme funding.

**Huidig**: funding_rate als feature (lineair). Ontbreekt: extreme-funding gate.

**Features:**
- `funding_extreme_long`: 1 als funding > config.FUNDING_EXTREME_GATE (+0.05%)
- `funding_extreme_short`: 1 als funding < −0.03% (rare short squeeze setup)
- `funding_percentile_30d`: percentiel van huidige funding vs. afgelopen 30 dagen

**Implementatie:**
1. `config.py` — `FUNDING_EXTREME_GATE = 0.0005` (+0.05%)
2. `src/features.py` — rolling percentiel berekening
3. `src/backtest.py` — gate: blokkeer longs als funding_extreme_long == 1

**Bestanden:** `config.py`, `src/features.py`, `src/backtest.py`

---

### T2-C · Open Interest trend als bevestigingssignaal
**Status:** `[ ]`

**Waarom traders het gebruiken:**
OI + prijs stijgen tegelijk → echte koop-interesse (bullish confirmation).
OI stijgt + prijs daalt → distributie (bearish divergentie).
Dit is de **volume-equivalent voor futures** en mist vrijwel altijd in retail modellen.

**Huidig**: `oi_change_24h` zit in code maar is uitgeschakeld wegens slechte datadekking.

**Features:**
- `oi_price_divergence`: OI-richting vs. prijs-richting (−1/0/+1)
- `oi_trend_3d`: 3-daagse richting van OI (stijgend/dalend/neutraal)

**Bron-probleem**: Binance Futures geeft slechts 30 dagen historische OI terug → slecht voor training.
**Alternatief**: Coinglass API of CryptoQuant biedt langere OI-history (deels gratis).

**Implementatie:**
1. Eerst: evalueer Coinglass gratis OI-history endpoint
2. `src/external_data.py` — `fetch_open_interest_coinglass()`
3. `config.py` — features toevoegen

**Bestanden:** `src/external_data.py`, `config.py`, `src/features.py`

---

### T2-D · Kelly Criterion positiegroottes
**Status:** `[x]` — geïmplementeerd 2026-03-27 · compute_kelly_fraction() + save_kelly_sizing() in model.py; half-Kelly cap in live_alert.py; USE_KELLY_SIZING=True

**Waarom traders het gebruiken:**
De Kelly-formule bepaalt de theoretisch optimale positiegrootte op basis van
verwacht rendement en win-rate. Oversize → faillissement bij pech. Undersize → te
weinig rendement. Huidige aanpak (1% risico per trade) is conservatief maar niet
mathematisch geoptimaliseerd.

**Formule:**
```
Kelly fraction = (p × b − q) / b
Waarbij: p = win-rate, q = 1−p, b = odds (gem. winst / gem. verlies)
```

**Implementatie:**
1. `src/model.py` — bereken Kelly-fractie op validatieset na training
2. `src/live_alert.py` — gebruik Half-Kelly (Kelly/2) als positie-upper-bound
3. `config.py` — `USE_KELLY_SIZING = False` (opt-in flag)

**Bestanden:** `src/model.py`, `src/live_alert.py`, `config.py`

---

### T2-E · BTC Halving Cyclus feature
**Status:** `[x]` — geïmplementeerd 2026-03-27 · code aanwezig in features.py, tijdelijk uit FEATURE_COLS (overfitting test)

**Waarom traders het gebruiken:**
De BTC halving is de sterkste fundamentele cyclus in crypto. Historisch:
- 12-18 maanden na halving: piek
- Daarna: bear market 1-2 jaar

**Features:**
- `days_since_halving`: dagen since laatste halving (cyclische feature)
- `halving_cycle_phase`: 0–1 (fractie door de 4-jaar cyclus heen)
- `pre_halving_window`: 1 als binnen 90 dagen vóór halving (historisch bullish)

**Halvings:**
- 2012-11-28, 2016-07-09, 2020-05-11, 2024-04-20, **2028-03-? (geschat)**

**Implementatie:**
1. `src/features.py` — bereken cyclus-positie op basis van vaste datum-tabel
2. `config.py` — features toevoegen (geen externe data nodig)

**Bestanden:** `src/features.py`, `config.py`

---

### T2-F · Correlation regime (BTC-ETH ontkoppeling)
**Status:** `[x]` — geïmplementeerd 2026-03-27 · code aanwezig in features.py, tijdelijk uit FEATURE_COLS (gecombineerde regressie)

**Waarom traders het gebruiken:**
Normaal bewegen BTC en ETH samen (correlatie > 0.85). Wanneer de correlatie
**breekt** (ETH daalt terwijl BTC stijgt, of vice versa) is dit een vroeg signaal
van altcoin-seizoen of BTC-dominantie shift.

**Features:**
- `btc_eth_correlation_24h`: rolling 24h Pearson correlatie van returns
- `btc_eth_correlation_7d`: rolling 7d correlatie
- `correlation_breakdown`: 1 als 24h correlatie < 0.5 (ontkoppeling)

**Implementatie:**
1. `src/features.py` — laad beide OHLCV series en bereken rolling correlatie
2. `config.py` — 3 features toevoegen

**Bestanden:** `src/features.py`, `config.py`

---

### T2-G · Supertrend indicator
**Status:** `[x]` — geïmplementeerd 2026-03-27 · code aanwezig in features.py, tijdelijk uit FEATURE_COLS (corr -0.36 met proba)

**Waarom traders het gebruiken:**
Supertrend is een ATR-gebaseerde trendvolgend indicator die veel gebruikt wordt
op alle tijdframes. Geeft een duidelijk +1/−1 signaal (boven/onder de band)
en vangst trendchanges eerder dan EMA-crossovers.

**Features:**
- `supertrend_signal`: +1 (uptrend) / −1 (downtrend)
- `supertrend_distance`: (close − supertrend_line) / close

**Parameters:** ATR-multiplier 3, periode 14

**Implementatie:**
1. `src/features.py` — bereken Supertrend vanuit ATR
2. `config.py` — 2 features toevoegen

**Bestanden:** `src/features.py`, `config.py`

---

## Tier 3 — Hoge impact, hoog complex

### T3-A · Google Trends als sentiment proxy
**Status:** `[ ]`

**Waarom traders het gebruiken:**
Google Trends voor "bitcoin" correleert sterk met retail-FOMO toppen.
Een spike in zoekvolume → retail stroomt in → naderende correctie.
Gratis, geen API-key, bewezen in academisch onderzoek.

**Features:**
- `google_trends_btc`: wekelijks zoekvolume (genormaliseerd 0–100)
- `trends_momentum_4w`: 4-weeks verandering (stijgend = FOMO opbouw)
- `trends_spike`: 1 als huidige waarde > 90e percentiel (extreme FOMO)

**Bron:** `pytrends` library (inofficiële Google Trends API)
**Beperkingen:** Wekelijks granularity, rate limiting, wekelijks forward-filled

**Implementatie:**
1. `pip install pytrends`
2. `src/external_data.py` — `fetch_google_trends(keyword="bitcoin")`
3. `config.py` — features + gate (trends_spike blokkeert longs in FOMO-tops)

**Bestanden:** `src/external_data.py`, `config.py`, `src/backtest.py`

---

### T3-B · Stablecoin dominantie (USDT.D)
**Status:** `[x]` — geïmplementeerd 2026-03-27 · fetch_usdt_dominance() in external_data.py, uit FEATURE_COLS (76% null)

**Waarom traders het gebruiken:**
USDT Dominance (%) = kapitaal dat in stablecoins zit, buiten de markt.
Stijgend USDT.D → geld vlucht uit crypto (bearish).
Dalend USDT.D → droog poeder stroomt terug in markt (bullish).
Dit is complementair aan BTC Dominance.

**Features:**
- `usdt_dominance`: % van crypto market cap in stablecoins
- `usdt_dominance_7d_chg`: 7-daagse verandering

**Bron:** CoinGecko `/global` endpoint geeft stablecoin market cap percentage

**Implementatie:**
1. `src/external_data.py` — uitbreiden van `fetch_btc_dominance()` met USDT.D
2. `config.py` — 2 features toevoegen

**Bestanden:** `src/external_data.py`, `config.py`

---

### T3-C · Opties: 25-delta skew (bearish/bullish positioning)
**Status:** `[ ]`

**Waarom traders het gebruiken:**
De 25-delta skew meet het verschil in implied volatility tussen puts (bearish bescherming)
en calls (bullish exposure) op hetzelfde vervalmoment.
- Negatieve skew: puts duurder → markt verwacht crash
- Positieve skew: calls duurder → markt verwacht stijging
Dit is nauwkeuriger dan de Put/Call ratio omdat het prijs (IV) meet, niet volume.

**Bron:** Deribit public API `/api/v2/public/get_index_price` + options chain analyse

**Features:**
- `btc_skew_25d_1m`: 25-delta skew voor 1-maands opties
- `btc_skew_25d_3m`: 25-delta skew voor 3-maands opties

**Implementatie:**
1. `src/external_data.py` — `fetch_deribit_skew()` — complexe berekening op opties-chain
2. `config.py` — features + gate

**Bestanden:** `src/external_data.py`, `config.py`

---

### T3-D · Max Pain prijs (Deribit opties)
**Status:** `[ ]`

**Waarom traders het gebruiken:**
Max Pain = de uitoefenprijs waarbij de meeste opties waardeloos vervallen
(maximaal verlies voor opties-kopers, maximaal winst voor schrijvers).
Market makers hedgen richting max pain als vervaldag nadert.

**Features:**
- `max_pain_distance`: (close − max_pain_price) / close
- `days_to_expiry`: dagen tot de grootste expiratiedatum

**Implementatie:**
1. `src/external_data.py` — bereken max pain uit Deribit options chain
2. `config.py` — 2 features toevoegen

**Bestanden:** `src/external_data.py`, `config.py`

---

### T3-E · Sentimentanalyse (social media)
**Status:** `[ ]`

**Waarom traders het gebruiken:**
Retail-sentiment op Twitter/Reddit loopt 6-24 uur voor op prijs.
Bij extreme bullishness (iedereen roept "all-time high") is de top nabij.

**Opties (gratis of goedkoop):**
- **LunarCrush** — crypto social media metrics (beperkte gratis tier)
- **CryptoPanic** — nieuws sentiment API (gratis tier)
- **Santiment** — social volume (betaald, maar academisch bewezen)

**Features:**
- `social_volume_btc`: aantal mentions (log-schaal)
- `social_sentiment_btc`: positief/negatief ratio
- `news_sentiment_score`: gecombineerde nieuwsscore

**Implementatie:**
1. `src/external_data.py` — `fetch_cryptopanic_sentiment()` (gratis API)
2. `config.py` — features + gate (extreme positief sentiment blokkeert longs)

**Bestanden:** `src/external_data.py`, `config.py`

---

### T3-F · Hidden Markov Model regime detection
**Status:** `[ ]`

**Waarom traders het gebruiken:**
Het huidige regime-model gebruikt ADX (trendsterkte). HMM is statistisch robuuster:
het leert **verborgen staten** (bull/bear/ranging) uit het rendements-patroon zelf,
zonder handmatige drempelwaarden. Beter voor regime-overgangen.

**Implementatie:**
1. `pip install hmmlearn`
2. `src/model.py` — `train_hmm_regime(df)` → 3-state HMM op dagelijkse returns
3. `src/features.py` — `hmm_regime` als feature (vervangt of aanvult `market_regime`)
4. Walk-forward validatie om look-ahead te voorkomen

**Bestanden:** `src/model.py`, `src/features.py`, `config.py`

---

### T3-G · On-Chain data (Glassnode/CryptoQuant)
**Status:** `[ ]` — Vereist betaalde API-key

**Waarom traders het gebruiken:**
On-chain metrics zijn fundamentele gegevens over het echte gebruik van het netwerk.

| Metric | Bron | Betekenis |
|---|---|---|
| SOPR | Glassnode | > 1 = winst genomen (verkoopdruk) |
| NUPL | Glassnode | Net Unrealized P&L (markt-euforie maatstaf) |
| Exchange Netflow | Glassnode | Positief = coins naar beurs (verkoopdruk) |
| MVRV-Z Score | Glassnode | Over/ondergewaardeerd vs. on-chain waarde |
| Funding Composite | CryptoQuant | Gewogen funding over exchanges |

**Kosten:** Glassnode Hobbyist-tier ~$29/maand, CryptoQuant ~$40/maand
**Alternatief:** CryptoQuant heeft gratis eindpunten voor een beperkt aantal metrics

**Implementatie:**
1. Omgevingsvariabele `GLASSNODE_API_KEY` of `CRYPTOQUANT_API_KEY`
2. `src/external_data.py` — `fetch_glassnode(metric)` / `fetch_cryptoquant(metric)`
3. `config.py` — features toevoegen (SOPR, NUPL, exchange_netflow)

**Bestanden:** `src/external_data.py`, `config.py`, `src/features.py`

---

### T3-H · LSTM / Transformer tijdreeks model
**Status:** `[ ]`

**Waarom traders het gebruiken:**
LightGBM behandelt features onafhankelijk per tijdstip. Een LSTM/Transformer
modelleert **sequentiële patronen** — het leert dat "3 rode candles gevolgd door
een hammer" een reversal-patroon is. Dit vangt dingen die tabulaire modellen missen.

**Aanpak:**
- Gebruik huidige features als input-sequentie (window = 48h)
- Ensemble: blend LightGBM proba + LSTM proba (gewogen gemiddelde)
- LSTM als tweede gate: entry alleen als beide modellen bullish

**Implementatie:**
1. `pip install torch` of `tensorflow`
2. `src/model_lstm.py` — LSTM met 48h lookback window
3. `src/backtest.py` — blend_proba = 0.6 × lgbm + 0.4 × lstm

**Bestanden:** `src/model_lstm.py`, `src/backtest.py`, `config.py`

---

## Tier 4 — Portfolio-niveau verbeteringen

### T4-A · Correlatie-bewuste multi-asset uitvoering
**Status:** `[x]` — geïmplementeerd 2026-03-27 · _check_corr_guard() in live_alert.py; blokkeert bij 24h-corr > 90%, halveert bij > 70%

**Waarom traders het gebruiken:**
Wanneer BTC én ETH tegelijk een long-signaal geven, is dit gecorreleerd risico:
ze kunnen beiden tegelijk crashen. Professionele traders limiteren het totale exposure.

**Implementatie:**
- Wanneer BTC én ETH beide open posities hebben → halveer de positiegrootte per trade
- Wanneer BTC/ETH correlatie > 0.9 over afgelopen 24h → maximaal 1 actieve positie
- `src/live_alert.py` — check alle open posities voor sizing beslissing

**Bestanden:** `src/live_alert.py`, `config.py`

---

### T4-B · Value at Risk (VaR) tracking
**Status:** `[ ]`

**Waarom traders het gebruiken:**
VaR kwantificeert het risico in euro's: "met 95% kans verlies ik niet meer dan €X
in de komende 24 uur." Professionele risicoafdelingen gebruiken dit als harde grens.

**Implementatie:**
1. `src/live_alert.py` — bereken 1-dag 95%-VaR op basis van historische volatiliteit
2. Blokkeer nieuwe entries als VaR > 5% van kapitaal
3. Log VaR in Discord alert als extra context

**Bestanden:** `src/live_alert.py`

---

### T4-C · Automatische parameter-aanpassing op marktregime
**Status:** `[x]` — geïmplementeerd 2026-03-27 · REGIME_SL_TP tabel in config.py; _regime_sl_tp() in live_alert.py

**Waarom traders het gebruiken:**
De optimale SL/TP varieert sterk met het volatiliteitsregime:
- Bull-run: grotere TP nodig (trend loopt lang door)
- Ranging: kleinere TP halen eerder (geen trend)
- Bear: strengere SL nodig

**Implementatie:**
1. `config.py` — regime-afhankelijke SL/TP tabellen
2. `src/live_alert.py` — selecteer parameters op basis van dagelijks regime:
   ```python
   params = REGIME_PARAMS[daily_regime]
   sl_pct = params["sl_pct"]
   tp_pct = params["tp_pct"]
   ```

**Bestanden:** `config.py`, `src/live_alert.py`

---

## Prioriteitenmatrix

| Taak | Impact | Moeilijkheid | Prioriteit | Status |
|---|---|---|---|---|
| T1-A Ichimoku Cloud | Hoog | Laag | ⭐⭐⭐ | ✅ |
| T1-B Candlestick patronen | Medium | Laag | ⭐⭐⭐ | ✅ |
| T1-C DXY Dollar Index | Hoog | Laag | ⭐⭐⭐ | ✅ #1 feature |
| T1-D ATR trailing stop | Hoog | Medium | ⭐⭐⭐ | ✅ |
| T1-E Partiële exits (TP1/TP2) | Hoog | Medium | ⭐⭐⭐ | ✅ |
| T1-F RSI-divergentie | Medium | Medium | ⭐⭐ | ✅ laag belang |
| T2-A Liquidatie heatmap | Hoog | Medium | ⭐⭐⭐ | `[ ]` |
| T2-B Funding extremen gate | Medium | Laag | ⭐⭐⭐ | ✅ |
| T2-C OI trend | Medium | Medium | ⭐⭐ | `[ ]` |
| T2-D Kelly Criterion sizing | Medium | Medium | ⭐⭐ | ✅ |
| T2-E BTC Halving cyclus | Medium | Laag | ⭐⭐⭐ | ✅ (code, uit FEATURE_COLS) |
| T2-F BTC-ETH correlatie | Medium | Laag | ⭐⭐ | ✅ (code, uit FEATURE_COLS) |
| T2-G Supertrend | Medium | Laag | ⭐⭐ | ✅ (code, uit FEATURE_COLS) |
| T3-A Google Trends | Medium | Medium | ⭐⭐ | `[ ]` |
| T3-B USDT dominantie | Medium | Laag | ⭐⭐ | ✅ (code, uit FEATURE_COLS) |
| T3-C Opties 25d skew | Hoog | Hoog | ⭐⭐ | `[ ]` |
| T3-D Max Pain | Medium | Hoog | ⭐ | `[ ]` |
| T3-E Social media sentiment | Medium | Hoog | ⭐ | `[ ]` |
| T3-F HMM regime detectie | Hoog | Hoog | ⭐⭐ | `[ ]` |
| T3-G On-Chain (Glassnode) | Hoog | Hoog | ⭐ (betaald) | `[ ]` |
| T3-H LSTM/Transformer | Hoog | Zeer hoog | ⭐⭐ | `[ ]` |
| T4-A Multi-asset correlatie | Medium | Laag | ⭐⭐⭐ | ✅ |
| T4-B Value at Risk | Medium | Medium | ⭐⭐ | `[ ]` |
| T4-C Regime-afhankelijke params | Hoog | Medium | ⭐⭐⭐ | ✅ |

---

## Aanbevolen volgorde van uitvoering

### Sprint 1 ✅ VOLTOOID (2026-03-27)
- ~~**T1-C** DXY — #1 feature voor BTC & ETH (Sharpe BTC +4.5%)~~
- ~~**T1-A** Ichimoku Cloud — cloud_thickness top-3 feature~~
- ~~**T1-B** Candlestick patronen — geïmplementeerd, belang laag~~
- ~~**T1-D** ATR trailing stop — dynamische stop in beide backtest-varianten~~
- ~~**T1-E** Partiële exits TP1/TP2 — TP1=+3%, TP2=+8% in run_backtest_be_trail~~
- ~~**T1-F** RSI divergentie — geïmplementeerd, belang laag~~

**Resultaat Sprint 1:** BTC Sharpe +5.944 → +6.214 (+4.5%) | ETH val Sharpe +14.39

### Sprint 2 — Code geïmplementeerd, features geëvalueerd (2026-03-27)
- ~~**T2-B** Funding extreme gate~~ ✅ actief in backtest (funding > 0.05% blokkeert longs)
- ~~**T2-E** BTC Halving cyclus~~ ✅ code aanwezig, **tijdelijk uit FEATURE_COLS** (overfitting test: corr -0.28 met proba, verhoogde trades, lagere WR)
- ~~**T2-G** Supertrend~~ ✅ code aanwezig, **tijdelijk uit FEATURE_COLS** (corr -0.36 met proba; gecaptured door adx+ema)
- ~~**T2-F** BTC-ETH 7d correlatie~~ ✅ code aanwezig, **tijdelijk uit FEATURE_COLS** (gecombineerd met halving/supertrend: regressie; alleen toevoegen na Optuna re-run)
- ~~**T3-B** USDT dominantie~~ ✅ code aanwezig, **tijdelijk uit FEATURE_COLS** (76% null in training — slechts 1 jaar CoinGecko history)

**Sprint 2 bevindingen:**
- Sprint 2 features combinatie leidt tot 302 trades (vs 255), win rate daalt van 59% → 50% → Sharpe van 6.21 → 2.87
- Oorzaak: regime-threshold optimizer zet bull/ranging op 0.50 (te agressief met nieuwe features)
- T2-B funding gate heeft GEEN negatief effect (< 0.6% van candles geblokkeerd)
- **Volgende stap**: Optuna heroptimaliseren voor uitgebreide feature set voordat Sprint 2 features actief worden

**Resultaat Sprint 2:** BTC Sharpe +6.293 (T2-B gate + nieuwe Optuna params) vs baseline +5.944

### Sprint 3 ✅ VOLTOOID (2026-03-27)
- ~~**T4-A** Multi-asset correlatie guard~~ ✅ _check_corr_guard() in live_alert.py — blokkeert bij 24h-corr > 90%, halveert bij > 70%
- ~~**T4-C** Regime-afhankelijke SL/TP~~ ✅ REGIME_SL_TP tabel + _regime_sl_tp() — bull 2.5%/8%, ranging 2%/6%, bear 1.5%/4%
- ~~**T2-D** Kelly Criterion sizing~~ ✅ compute_kelly_fraction() in model.py — half-Kelly cap in live_alert; gesaved per symbool als {symbol}_kelly.json

**Sprint 3 bevindingen:**
- Kelly, regime SL/TP, en correlatie guard zijn puur live-trading verbeteringen — geen backtest Sharpe impact
- Kelly fraction wordt berekend op validatieset bij elke training en opgeslagen
- Regime SL/TP geeft bull-run meer ruimte (TP +8%) en bear-markt strakkere stop (SL 1.5%)

**Volgende prioriteit Sprint 4:**
1. **T2-A** Liquidatie data (Coinglass API) — sterkste ontbrekende feature
2. **T3-C** Opties 25-delta skew (Deribit) — nauwkeuriger dan P/C ratio
3. **T3-A** Google Trends sentiment
4. Sprint 2 features re-activeren na Optuna heroptimalisatie

### Sprint 4 (complex, transformatief)
- **T2-A** Liquidatie heatmap (Coinglass API)
- **T3-A** Google Trends sentiment
- **T3-C** Opties 25-delta skew

### Sprint 5 (geavanceerd)
- **T3-F** HMM regime detectie
- **T3-H** LSTM ensemble
- **T3-G** On-chain data (als API-key beschikbaar)
