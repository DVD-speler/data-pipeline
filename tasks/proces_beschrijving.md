# Proces: Crypto Signal Model — BTC/USDT

## Overzicht

Een end-to-end systeem dat automatisch handelssignalen genereert voor BTC/USDT,
gebaseerd op machine learning (LightGBM), en deze als Discord-alerts verstuurt via GitHub Actions.
Het model is getraind op 730 dagen historische data (Binance, 1h candles) en
live gevalideerd via paper trading (startkapitaal €1000).

**Eindresultaat (testperiode Nov 2025 – Feb 2026):**
- LightGBM AUC: **0.5847**
- Sharpe: **+3.19** (vs Buy & Hold −34.5%)
- Rendement: **+85.3%** vs B&H −34.5% → +119.8 procentpunt verschil
- Walk-forward: **6/10 positieve folds**, gemiddeld Sharpe +0.62

---

## Fase 1 — Data Verzameling (`src/data_fetcher.py`)

**Doel:** Historische OHLCV-candles ophalen en opslaan.

**Wat:**
- BTC/USDT en ETH/USDT, 1h en 4h timeframes
- 730 dagen historisch (Binance publieke REST API — geen API key nodig)
- Opgeslagen in lokale SQLite database (`data/ohlcv.db`)
- Incrementeel: bij herstart wordt verder gegaan vanaf het laatste opgeslagen tijdstip

**Hoe:**
```bash
python main.py --phase data
```

**Aandachtspunten:**
- Binance blokkeert US IP-adressen (HTTP 451) — GitHub Actions draait op Azure US-servers
- Oplossing: automatische fallback naar yfinance bij 451-error
- yfinance geeft 1h data; 4h wordt hieruit geresampeld (OHLCV-aggregatie)
- `ohlcv.db` wordt NIET in git opgeslagen (binary, wijzigt elke run)
- In GitHub Actions: wekelijkse cache zodat elke run alleen de laatste paren uur downloadt

---

## Fase 2 — Externe Data (`src/external_data.py`)

**Doel:** Macro- en sentimentdata toevoegen als extra features.

**Databronnen:**
| Bron | Data | Resolutie | API |
|------|------|-----------|-----|
| alternative.me | Fear & Greed Index | dagelijks | gratis REST |
| yfinance | SPY (S&P 500 proxy) | 1h | gratis |
| yfinance | EUR/USD | 1h | gratis |
| Binance Futures | Funding rate | 8h | gratis REST |

**Bewaard als:** Parquet-bestanden in `data/external/` (klein, zinvol → wél in git)

**Aandachtspunten:**
- yfinance 1h SPY-bars starten op :30 (09:30 ET) — vóór join gefloord naar het uur (anders 0 matches)
- pandas 2.x slaat Parquet op als `datetime64[ms, UTC]`; OHLCV gebruikt `datetime64[ns, UTC]`
  — dtype altijd normaliseren vóór join

```bash
python main.py --phase external_data
```

---

## Fase 3 — P1/P2 Labels (`src/p1p2_engine.py`)

**Doel:** Bepaal voor elk uur of de markt stijgt (P1) of daalt (P2) binnen het predictievenster.

**Definitie:**
- P1 (bullish): prijs over `PREDICTION_HORIZON_H=24` uur > +0.3% (boven fee breakeven)
- P2 (bearish): prijs over 24 uur < −0.3%
- Neutraal: |move| < 0.3% → verwijderd uit training (15.1% van rijen)

**Output:** `data/p1p2_labels.csv` met kolom `label` (1=P1, 0=P2)

**Aandachtspunten:**
- P1/P2 heatmaps (statistieken per dag/uur) worden berekend op uitsluitend de trainset
  (niet de volledige dataset — anders data leakage naar test)
- 24h horizon gekozen boven 12h: minder ruis, hogere AUC (+0.043), meer traindata

```bash
python main.py --phase p1p2
```

---

## Fase 4 — Feature Engineering (`src/features.py`)

**Doel:** Bouw de feature matrix die het model als input krijgt.

**40 features in totaal:**

*Tijdsfeatures:*
`hour`, `day_of_week`, `hour_of_week`, `session`

*P1/P2 statistieken:*
`p1_probability` (heatmap kans per dag×uur, berekend op trainset)

*Prijs & volume:*
`volatility_24h`, `prev_day_return`, `volume_ratio`, `volume_spike_48h`, `price_position`

*Multi-horizon momentum:*
`return_2h`, `return_4h`, `return_6h`, `return_12h`

*Trendkwaliteit:*
`vol_regime`, `trend_consistency_12h`, `buy_pressure`

*Technische indicatoren (1h):*
`rsi_14`, `macd`, `macd_signal`, `bb_pct`, `ema_ratio_20`, `ema_ratio_50`,
`price_vs_ema200`, `atr_pct`, `adx`, `vwap_distance`

*Cross-asset & extern:*
`fear_greed`, `spx_return_24h`, `eurusd_return_24h`, `eth_btc_ratio`,
`funding_rate`, `funding_momentum`

*Macro-momentum:*
`return_7d`, `return_30d`, `ath_7d_distance`

*4h timeframe (extra resolutie):*
`rsi_14_4h`, `macd_4h`, `bb_pct_4h`, `ema_ratio_20_4h`

*Filterkolom (niet in model, wel in feature matrix):*
`market_regime` — gebruikt in backtest voor long/short filter

**Splits:**
- Train: ~10666 rijen (tot ~9 maanden geleden)
- Validatie: 1440 rijen (2 maanden) — voor threshold-optimalisatie
- Test: 2160 rijen (3 maanden, nov 2025–feb 2026) — voor eindvalidatie

**Top features (24h horizon):**
`spx_return_24h` (9.3%), `prev_day_return` (8.6%), `return_30d` (7.9%),
`volatility_24h` (5.8%), `eurusd_return_24h` (5.6%)

```bash
python main.py --phase features
```

---

## Fase 5 — Model Trainen (`src/model.py`)

**Doel:** Train een LightGBM classificatiemodel op de feature matrix.

**Model:** LightGBM (gradient boosting, tijdsgewogen)
- Recente rijen krijgen meer gewicht zodat het model zich aanpast aan recente marktomstandigheden
- Hyperparameters opgeslagen in `data/lgb_best_params.json`
  (n_estimators=407, max_depth=5, lr=0.028, colsample_bytree=0.477, min_child_samples=175)

**Threshold-optimalisatie:**
- Long threshold: geoptimaliseerd op validatieset, bereik 0.65–0.85
  → selectief: alleen signalen met hoge model-conviction
- Short threshold: ceiling 0.45 (≤0.45 = model is genuinely bearish)
  → "short wanneer model betekenisvol bearish is, niet alleen onzeker"
- Opgeslagen in `data/optimal_threshold.json`

**Signaalfilters (backtest + live):**
- **Long**: alleen als prijs boven EMA(200) EN `market_regime != -1`
- **Short**: alleen als `return_30d < −3%` EN `return_7d < −1%` (dubbele macro-gate)
  → blokkeert shorts tijdens bull-herstel (7d positief ondanks 30d negatief)

**Output:** `data/model.pkl` (wél in git — nodig voor GitHub Actions live pipeline)

```bash
python main.py --phase model
```

---

## Fase 6 — Backtesten (`src/backtest.py`)

**Doel:** Simuleer de handelsstrategie op de testset.

**Strategie:**
- Entry bij LONG/SHORT signaal
- Stop-loss: 2% van entry
- Take-profit: 6% van entry
- Horizon: 24u (positie sluit automatisch na 24u op close-prijs)
- Kosten: 0.1% per kant (0.2% round-trip)

**Resultaten (testperiode, LightGBM):**

| Metric | Waarde |
|--------|--------|
| AUC | 0.5847 |
| Sharpe | +3.19 |
| Rendement | +85.3% |
| B&H (benchmark) | −34.5% |
| Verschil | +119.8 pp |
| Aantal trades | 204 |
| Statistisch significant | Ja |

```bash
python main.py --phase backtest
```

---

## Fase 7 — Walk-Forward Validatie (`src/backtest.py`)

**Doel:** Test of het model generaliseert over meerdere marktregimes (niet alleen de testperiode).

**Methode:**
- 10 opeenvolgende folds van 30 dagen
- Per fold: train 270 dagen, test 30 dagen
- Model en threshold opnieuw gefit per fold (geen leakage)

**Resultaten (LightGBM, 24h horizon):**

| Fold | Periode | Sharpe | Rendement | Opmerking |
|------|---------|--------|-----------|-----------|
| 0 | Feb 2025 | −5.46 | −11.3% | Shorts verliezen in herstelbeweging |
| 1 | Mrt 2025 | −24.69 | −25.8% | Slechtste fold, model verkeerde richting |
| 2 | Apr-Mei 2025 | +12.82 | +170.6% | Bull run, beste fold |
| 3 | Mei-Jun 2025 | +8.32 | +62.3% | |
| 4 | Jun-Aug 2025 | +4.36 | +4.6% | |
| 5 | Aug 2025 | −3.24 | −4.0% | |
| 6 | Sep-Okt 2025 | −12.74 | −31.4% | Te veel longs in bearmarkt |
| 7 | Okt-Nov 2025 | +7.49 | +50.1% | Short-gedomineerd |
| 8 | Nov-Dec 2025 | +8.62 | +45.1% | |
| 9 | Dec-Jan 2026 | +10.74 | +46.0% | Beste long-fold |

**Samenvatting:** 6/10 positieve folds, gemiddeld Sharpe +0.62, gemiddeld rendement +30.6%

```bash
python main.py --phase walkforward
```

---

## Fase 8 — Live Alert Pipeline (`src/live_alert.py`)

**Doel:** Automatisch signalen genereren en als Discord-alert versturen. Laptop hoeft niet aan.

**Architectuur:**

```
GitHub Actions (elke 2 uur, cron: '5 */2 * * *')
  ↓
1. Checkout repo (bevat model.pkl, external/*.parquet, paper_trades.json)
  ↓
2. Herstel ohlcv.db uit wekelijkse cache (of download 35 dagen vers bij cache miss)
  ↓
3. Incrementele OHLCV update (alleen laatste paar uur)
  ↓
4. Update externe data (Fear & Greed, SPY, EUR/USD, funding rate)
  ↓
5. Genereer live signaal via generate_live_signal()
  ↓
6. Paper trading: check open positie (SL/TP/24h horizon?) + open nieuw signaal
  ↓
7. Discord webhook alert sturen
  ↓
8. Commit paper_trades.json + latest_signal.json terug naar repo
```

**Paper trading state (`data/paper_trades.json`):**
```json
{
  "open_position": { "direction": "LONG", "entry_price": 85000, ... },
  "closed_trades": [ { "pnl_euro": 14.28, "gross_return_pct": 6.0, ... } ],
  "capital": 1014.28,
  "last_checked": "2026-03-02 21:00:00+00:00"
}
```

**Discord alert formaat (opening):**
```
🟢 LONG SIGNAAL — BTCUSDT
⏰ 02-03-2026 18:00 UTC
💰 Entry: $85,000 | SL: $83,300 (−2%) | TP: $90,100 (+6%)
📊 Proba: 71.2% | Regime: boven EMA200
💼 Positie: $500 (0.005882 BTC) | Risico: €10.00 (1% kapitaal)
```

**Discord alert formaat (sluiting):**
```
✅ TRADE GESLOTEN — BTCUSDT
📈 LONG | $85,000 → $90,100 | TP ✓ | +6.00%
💰 P&L: +€14.28 | Kapitaal: €1,014.28
```

**Risk management (live):**
- Positiegrootte: `kapitaal × 1% / 2%` = 50× risico (fixed fractional)
- Risico per trade: 1% van kapitaal (€10 bij start €1000)
- Kelly-optimaal berekend op backtest: f* ≈ 1.03% → vaste 1% is correct

**Setup (éénmalig):**
1. Push repo naar GitHub (publiek — geen Actions-minutenlimiet)
2. Maak Discord webhook aan: Serverinstellingen → Integraties → Webhooks
3. Voeg GitHub Secret toe: `DISCORD_WEBHOOK_URL`
4. Handmatig triggeren via Actions-tab om te testen

```bash
python main.py --phase live_alert
```

---

## Fase 9 (gepland) — Paper Trading Analyse

**Doel:** Na 30–50 gesloten live trades de prestaties evalueren en beslissen of het model
klaar is voor echte trading.

**Evaluatiemetrics:**
- Live winrate vs backtest (73.3%)
- Live Sharpe vs backtest (+3.19)
- Proba-kalibratie per bucket (betrouwbaarheidsdiagram)
- Equity curve (kapitaalgroei door de tijd)
- LONG vs SHORT verdeling

**Groen licht voor echte trading als:**
- Winrate > 55%
- Sharpe > 0.5
- Minimaal 50 trades

**Stop-criterium:** kapitaal daalt onder €800 (−20%) → volledige model review nodig

Zie: `tasks/fase6_paper_trading_analyse.md` voor de volledige analyse-code.

---

## Projectstructuur

```
crypto_signal_model/
├── config.py                        # Alle hyperparameters en paden
├── main.py                          # CLI-interface (--phase <fase>)
├── requirements.txt
├── src/
│   ├── data_fetcher.py              # Fase 1: OHLCV download (Binance + yfinance fallback)
│   ├── external_data.py             # Fase 2: Fear&Greed, SPY, EURUSD, funding
│   ├── p1p2_engine.py               # Fase 3: P1/P2 labels
│   ├── stats.py                     # P1/P2 heatmaps en direction bias
│   ├── features.py                  # Fase 4: Feature matrix
│   ├── model.py                     # Fase 5: LightGBM training + threshold
│   ├── backtest.py                  # Fase 6/7: Backtesten + walk-forward
│   ├── simulation.py                # Monte Carlo simulatie
│   ├── horizon_scan.py              # Vergelijking 12h vs 24h vs 48h horizon
│   ├── model_compare.py             # LightGBM vs XGBoost vs RandomForest vs Ensemble
│   └── live_alert.py                # Fase 8: Discord alerts + paper trading
├── data/
│   ├── ohlcv.db                     # SQLite OHLCV (NIET in git — te groot/binair)
│   ├── model.pkl                    # Getraind model (WEL in git — nodig voor Actions)
│   ├── lgb_best_params.json         # Stabiele LightGBM hyperparameters
│   ├── optimal_threshold.json       # Long/short drempelwaarden
│   ├── p1p2_labels.csv              # P1/P2 labels
│   ├── paper_trades.json            # Live paper trading state
│   ├── latest_signal.json           # Laatste signaal (voor debugging)
│   ├── external/                    # Parquet-bestanden externe data
│   └── stats/                       # Statistische output (grafieken, rapporten)
├── tasks/
│   ├── proces_beschrijving.md       # Dit document
│   ├── fase6_paper_trading_analyse.md  # Plan voor live evaluatie
│   └── lessons.md                   # Technische lessen per sprint
└── .github/
    └── workflows/
        └── hourly_signal.yml        # GitHub Actions workflow (elke 2 uur)
```

---

## Technische keuzes en waarom

| Keuze | Alternatief | Reden |
|-------|-------------|-------|
| LightGBM | XGBoost, RandomForest | Beste AUC (0.5847) + Sharpe (+3.19) op testset |
| 24h horizon | 12h, 48h | Top features zijn 24h+ van aard; hogere AUC (+0.043 vs 12h) |
| GitHub Actions | VPS, crontab | Gratis, geen eigen server nodig, werkt als laptop uit is |
| Discord webhook | Email, Telegram | Gratis, geen bot-setup nodig, directe push-notificatie |
| Paper trading in JSON | Database | Eenvoudig, comit-baar in git, geen extra infrastructuur |
| ohlcv.db in cache (niet git) | Git LFS, S3 | Binary wijzigt elke run; git-history wordt rommelig |
| model.pkl wél in git | Downloaden bij elke run | Nodig voor Actions zonder extra opslag/secrets |
| Vaste 1% risico | Kelly variabel | AUC 0.5847 + n=45 te klein voor betrouwbare proba-buckets |
| Dubbele macro-gate shorts | Enkel EMA200 | Voorkomt shorten tijdens bull-herstel (7d positief ondanks 30d negatief) |
