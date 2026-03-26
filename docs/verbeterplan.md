# Verbeterplan — Crypto Signal Model

Overzicht van geplande verbeteringen, geordend op prioriteit.
Elke verbetering is een zelfstandige taak die onafhankelijk uitgerold kan worden.

Status-legenda: `[ ]` open · `[~]` in uitvoering · `[x]` afgerond

---

## Fase A — Hoge prioriteit

### A1 · Probability calibratie
**Status:** `[ ]`

**Probleem:**
LightGBM-probabilities zijn slecht gekalibreerd. Een output van 0.70 betekent in
de praktijk vaak 55–60% kans, niet 70%. Dit ondermijnt de betrouwbaarheid van:
- De entry-drempel (`SIGNAL_THRESHOLD`)
- De proba-exit drempels (`EXIT_PROBA_LONG / EXIT_PROBA_SHORT`)
- De positiegrootte (schaalt met `proba - 0.5`)

**Oplossing:**
Na het trainen een calibrator fitten op de validatieset:
- **Isotonic regression** — niet-parametrisch, flexibel (voorkeur bij >1000 samples)
- **Platt scaling** — logistische calibratie (sneller, minder data nodig)

De gecalibreerde wrapper vervangt `model.predict_proba()` overal in de pipeline.

**Implementatie:**
1. `src/model.py` — fit `CalibratedClassifierCV(model, method='isotonic', cv='prefit')` op val-set na training
2. Sla calibrated model op als `{symbol}_model_calibrated.pkl`
3. `src/backtest.py` `generate_live_signal()` — laad calibrated model indien aanwezig
4. `src/live_alert.py` — geen wijziging nodig (gebruikt generate_live_signal)
5. Backtest-vergelijking: ongekalibreerd vs. gekalibreerd op alle 4 marktperiodes

**Bestanden:** `src/model.py`, `src/backtest.py`

**Verwacht effect:** Scherpere exit-beslissingen, minder valse exits bij lichte proba-dip.

---

### A2 · Multi-timeframe signaalconfirmatie
**Status:** `[ ]`

**Probleem:**
Het 1h-model neemt beslissingen onafhankelijk van het 4h-tijdframe. Als beide
timeframes bullish zijn, is de historische trefkans significant hoger dan wanneer
alleen de 1h bullish is. Nu worden 4h-features wel meegenomen als inputs, maar
het 4h-model zelf niet als confirmatie gebruikt.

**Oplossing:**
Gebruik het bestaande `{symbol}_4h_model.pkl` als gate:
- Entry alleen als `proba_1h ≥ threshold` **EN** `proba_4h ≥ threshold_4h`
- `threshold_4h` apart optimaliseren op validatieset (verwacht lager dan 1h-drempel)
- Proba_4h ook toevoegen als feature aan het 1h-model (cross-timeframe momentum)

**Implementatie:**
1. `src/backtest.py` `run_backtest()` — laad 4h-model, bereken 4h-probas, voeg toe als gate
2. `src/backtest.py` `generate_live_signal()` — bereken beide probas, toon beide in output
3. `config.py` — `SIGNAL_THRESHOLD_4H` parameter (default 0.55)
4. `src/live_alert.py` — Discord-bericht toont 1h én 4h proba
5. Backtest-vergelijking: met/zonder 4h-confirmatie

**Bestanden:** `config.py`, `src/backtest.py`, `src/live_alert.py`

**Verwacht effect:** Minder valse entries, hogere win rate, minder trades (striktere filter).

---

### A3 · Regime-specifieke entry thresholds
**Status:** `[ ]`

**Probleem:**
De optimale entry-drempel varieert 42% across walk-forward folds (0.58–0.84).
Nu wordt één globale drempel gebruikt met vaste offsets per regime:
- Bull: threshold − 0.05
- Ranging: threshold
- Bear: threshold + 0.08

Deze offsets zijn handmatig gekozen, niet data-driven. In sterke bullmarkten is
de optimale drempel mogelijk 0.58, in ranging markten 0.72 — de vaste offset
corrigeert dit onvoldoende.

**Oplossing:**
Per regime een aparte drempel optimaliseren op de bijbehorende validatiedata:
- Filter validatieset op `market_regime == +1` → optimaliseer `threshold_bull`
- Filter validatieset op `market_regime == 0` → optimaliseer `threshold_ranging`
- Filter validatieset op `market_regime == -1` → optimaliseer `threshold_bear`
- Sla op in `{symbol}_regime_thresholds.json`

**Implementatie:**
1. `src/model.py` `optimize_threshold()` — voeg `regime` parameter toe
2. `src/model.py` `train_model()` — roep 3× `optimize_threshold()` aan na training
3. `src/backtest.py` `run_backtest()` — laad regime-thresholds en pas per rij toe
4. `src/backtest.py` `generate_live_signal()` — gebruik regime-specifieke drempel
5. `config.py` — verwijder `REGIME_THRESHOLD_OFFSETS` (vervangen door data-driven waarden)

**Bestanden:** `config.py`, `src/model.py`, `src/backtest.py`

**Verwacht effect:** Stabielere performance across regimes, minder over- en undertrading.

---

## Fase B — Medium prioriteit

### B1 · BTC Dominance als feature
**Status:** `[ ]`

**Probleem:**
Voor ETHUSDT (en andere alts) is BTC Dominance een sterkere predictor dan de
huidige `eth_btc_ratio`. BTC Dominance meet de bredere altcoin-cyclus: als
dominantie stijgt → kapitaal vloeit naar BTC → alts presteren slechter.
De huidige `eth_btc_ratio` mist dit macro-perspectief.

**Oplossing:**
BTC Dominance ophalen via CoinGecko (gratis, geen API key):
- `https://api.coingecko.com/api/v3/global` → `btc_dominance` veld
- Dagelijkse frequentie, forward-fill naar 1h (zoals VIX)
- Feature: `btc_dominance` (percentage, ~40–55%)
- Extra feature: `btc_dominance_7d_change` (richting van de verschuiving)

**Implementatie:**
1. `src/external_data.py` — voeg `fetch_btc_dominance()` toe
2. `src/features.py` — join op features, bereken `btc_dominance_7d_change`
3. `config.py` `FEATURE_COLS_1H` — beide features toevoegen
4. Model opnieuw trainen na feature-toevoeging

**Bestanden:** `src/external_data.py`, `src/features.py`, `config.py`

**Verwacht effect:** Betere ETHUSDT-signalen, met name bij alt-season detectie.

---

### B2 · Expanding window walk-forward
**Status:** `[ ]`

**Probleem:**
De huidige walk-forward gebruikt een rollend venster van 270 dagen. Bij elke
fold wordt de vroegste data weggegooid. Dit is inefficiënt voor zeldzame events
(2022 bearmarkt, 2020 corona-crash) — het model "vergeet" ze.

Een expanding window gebruikt alle data tot het testpunt, wat:
- De bear-performance verbetert (meer bear-voorbeelden in training)
- De variantie tussen folds verlaagt
- De parameter-stabiliteit vergroot

**Oplossing:**
`run_walkforward()` uitbreiden met `expanding=True` optie:
- `expanding=True`: train_start = altijd index 0, train_end schuift op
- `expanding=False`: huidige rolling behaviour (default behouden)
- Minimale trainset: 180 dagen (anders te weinig data in vroege folds)

**Implementatie:**
1. `src/backtest.py` `run_walkforward()` — voeg `expanding` parameter toe
2. `main.py` — `--expanding` CLI flag
3. Vergelijkingsrun: rolling vs. expanding op dezelfde testperiode

**Bestanden:** `src/backtest.py`, `main.py`

**Verwacht effect:** Betere bear-performance, stabielere metrics across folds.

---

### B3 · Volume Profile / Point of Control (POC)
**Status:** `[ ]`

**Probleem:**
De huidige structurele SL/TP (zie `src/levels.py`) gebruikt swing highs/lows als
niveaus. Volume Profile voegt een dimensie toe: het prijsniveau met het meeste
handelsvolume (POC) functioneert als sterke magnet/support/resistance — ongeacht
of er een swing high/low aanwezig is.

**Oplossing:**
Per lookback-periode (bijv. 168u = 1 week) de POC berekenen:
- Verdeel het prijsbereik in N buckets (bijv. 50)
- Tel het volume per bucket
- POC = bucket met het meeste volume

Features:
- `poc_distance` — (close − POC) / close (positie t.o.v. POC)
- `value_area_high` / `value_area_low` — grenzen van 70% van het volume
- `poc_7d` als extra support/resistance niveau in `src/levels.py`

**Implementatie:**
1. `src/features.py` — voeg `compute_volume_profile()` toe
2. `src/levels.py` — voeg POC toe als extra niveau naast swing highs/lows
3. `config.py` `FEATURE_COLS_1H` — `poc_distance` toevoegen
4. Model opnieuw trainen

**Bestanden:** `src/features.py`, `src/levels.py`, `config.py`

**Verwacht effect:** Scherpere SL/TP-plaatsing bij hoge-volume niveaus.

---

### B4 · Asymmetrische dead zone
**Status:** `[ ]`

**Probleem:**
Nu worden rijen verwijderd als `|move_24h| < 0.3%` (symmetrisch). In een
bullmarkt zijn kleine opwaartse moves informatief (rustige accumulatie), terwijl
kleine neerwaartse moves vaker ruis zijn. De huidige symmetrische dead zone
behandelt beide gelijk en gooit goede trainingsvoorbeelden weg.

**Oplossing:**
Aparte drempels per richting:
- `TARGET_DEAD_ZONE_UP = 0.002` (0.2% — kleine ups behouden)
- `TARGET_DEAD_ZONE_DOWN = 0.004` (0.4% — kleine downs verwijderen)

Ook testen: `TARGET_DEAD_ZONE_PCT` verhogen van 0.3% naar 0.5% om
meer ruis te verwijderen en cleaner labels te krijgen.

**Implementatie:**
1. `config.py` — `TARGET_DEAD_ZONE_UP`, `TARGET_DEAD_ZONE_DOWN` toevoegen
2. `src/features.py` target-definitie aanpassen
3. Vergelijkingsrun: symmetrisch 0.3% vs. asymmetrisch vs. symmetrisch 0.5%

**Bestanden:** `config.py`, `src/features.py`

**Verwacht effect:** Schonere labels, hogere signal-to-noise ratio in training.

---

### B5 · Shorts herdefiniëren of verwijderen
**Status:** `[ ]`

**Probleem:**
De walk-forward toont 0–250 short-trades per fold (chaotisch). In bullmarkten
worden shorts gevuurd tegen de trend. De short-logica staat in de code maar
`SIGNAL_THRESHOLD_SHORT = 0.0` schakelt het effectief uit — dead code die
complexiteit toevoegt zonder waarde.

**Optie 1 — Verwijderen:**
Alle short-logica uit `run_backtest()`, `run_backtest_be_trail()` en
`live_alert.py` verwijderen. Codebase wordt ~15% kleiner en minder foutgevoelig.

**Optie 2 — Dedicated short model:**
Een apart model trainen **alleen** op bearmarkten (bijv. 2022 data) met:
- Inverse target: `future_close < close × (1 - dead_zone)` = 1
- Eigen features: funding_rate negatief, return_30d < -0.15, etc.
- Striktere gates: only live during `market_regime == -1` AND `price < EMA200`

**Implementatie (optie 1 — aanbevolen voor nu):**
1. `src/backtest.py` — verwijder `signal_short`, `threshold_short` parameters
2. `src/live_alert.py` — verwijder SHORT-logica
3. `config.py` — verwijder `SIGNAL_THRESHOLD_SHORT`
4. Regressietest: check dat resultaten onveranderd zijn

**Bestanden:** `config.py`, `src/backtest.py`, `src/live_alert.py`

**Verwacht effect:** Schonere code, geen chaotische short-signalen.

---

## Fase C — Lagere prioriteit

### C1 · Optuna automatische promotie
**Status:** `[ ]`

**Probleem:**
Optuna vindt kandidaat-hyperparameters maar schrijft die naar
`lgb_optuna_params.json`. Promotie naar `lgb_best_params.json` (productie) is
handmatig — er is geen automatisch vergelijkingscriterium.

**Oplossing:**
Na Optuna-run een mini walk-forward (3 folds) draaien met zowel de huidige
als de kandidaat-params. Promoveer alleen als kandidaat-Sharpe > huidige Sharpe + 0.1.

**Implementatie:**
1. `src/model.py` `train_model()` — voeg `auto_promote_optuna=False` parameter toe
2. Bij `auto_promote_optuna=True`: vergelijk via 3-fold WF, promoveer indien beter
3. Log promotie-beslissing in `data/stats/optuna_promotions.csv`

**Bestanden:** `src/model.py`

---

### C2 · Signaalveroudering in positiegrootte
**Status:** `[ ]`

**Probleem:**
Positiegrootte wordt bepaald bij entry en verandert niet meer. Naarmate een trade
ouder wordt zonder dat de proba stijgt, neemt de kans op een winstgevende exit af.
Het heeft zin om de effectieve positie geleidelijk te reduceren als de trade
"vastzit" zonder beweging.

**Oplossing:**
Positiegrootte decay: `effective_size = initial_size × max(0.5, 1 - 0.02 × hours_open)`
- Na 24u: 52% van oorspronkelijke size
- Na 48u: 4% (minimaal)
- Gecombineerd met proba-exit: exit wordt sneller bij kleine size

**Implementatie:**
1. `src/live_alert.py` — sla `hours_open` op in positie-dict, pas size aan bij check
2. `src/backtest.py` `run_backtest_be_trail()` — simuleer decay per candle

**Bestanden:** `src/live_alert.py`, `src/backtest.py`

---

### C3 · On-chain data integratie
**Status:** `[ ]`
**Vereiste:** Glassnode of CryptoQuant API-key

**Probleem:**
On-chain metrics zijn bewezen voorspellende signalen op 24h+ horizons maar
vereisen betaalde API's. De meest waardevolle:

| Metric | Bron | Betekenis |
|---|---|---|
| Exchange Netflow | Glassnode | Positief = coins naar exchange (verkoopdruk) |
| SOPR | Glassnode | > 1 = winst genomen, < 1 = verlies genomen |
| NUPL | Glassnode | Net Unrealized P&L (markt-sentiment proxy) |
| Funding composite | CryptoQuant | Gewogen funding across exchanges |

**Implementatie:**
1. `src/external_data.py` — `fetch_glassnode()` met API-key uit environment
2. `src/features.py` — join on-chain data op hourly index (forward-fill)
3. `config.py` — features toevoegen aan `FEATURE_COLS_1H`
4. Model opnieuw trainen

**Bestanden:** `src/external_data.py`, `src/features.py`, `config.py`

---

### C4 · Put/Call ratio (Deribit opties)
**Status:** `[ ]`

**Probleem:**
De bestaande `btc_dvol` (implied volatility) meet het niveau van opties-angst,
maar niet de richting. Een hoge P/C ratio (veel meer puts dan calls) signaleert
bearish positionering bij grote spelers — aanvullende informatie op DVOL.

**Oplossing:**
Deribit biedt gratis publieke API voor opties-data:
- `GET /api/v2/public/get_book_summary_by_currency` → filter op puts vs. calls
- Feature: `btc_put_call_ratio` (dagelijks, forward-filled)
- Gate: `btc_put_call_ratio > 1.5` → blokkeert longs (extreme put-dominantie)

**Implementatie:**
1. `src/external_data.py` — `fetch_deribit_put_call_ratio()`
2. `src/features.py` — join op feature matrix
3. `config.py` — feature + gate toevoegen

**Bestanden:** `src/external_data.py`, `src/features.py`, `config.py`

---

## Afgeronde verbeteringen

| Verbetering | Afgerond | Beschrijving |
|---|---|---|
| Model-exit (proba-exit) | ✅ | LONG sluit als proba < EXIT_PROBA_LONG |
| 168h tijdsvangnet | ✅ | Vervangt vaste 24h timeout |
| Structurele SL/TP | ✅ | SL/TP op basis van swing highs/lows |
| VIX / USD-JPY features | ✅ | Macro gates op aandelenmarkt-angst + carry trade |
| Regime-geconditioneerde modellen | ✅ | Bull/ranging/bear submodellen |
| DAYS_HISTORY 730→1826 | ✅ | 5 jaar history voor volledige cyclus |
| Exit-proba sweep | ✅ | Backtest-optimalisatie van exit thresholds |
