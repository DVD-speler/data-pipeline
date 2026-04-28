# Sports Roadmap

Eerste sport: **HBL (Handball-Bundesliga, Duitsland)**. Skellam-baseline,
1X2-markt, signalen op T-2h en T-4h voor kickoff via cron-getriggerde
GitHub Actions workflows naar Discord.

## Sprint-overzicht

| Sprint | Doel | Status |
|---|---|---|
| **S0** | Scaffold (directory-structuur, fase-runner stubs, config) | ✅ deze commit |
| **S1** | HBL data + odds collection | open |
| **S2** | Feature engineering (Elo, form, home advantage, rest, H2H) | open |
| **S3** | Skellam-model training + walk-forward backtest met CLV-metric | open |
| **S4** | Live signaal flow + Discord-webhook + paper trading | open |
| **S5** | Markt-uitbreiding: Total Goals (over/under) | open |

---

## S1 — HBL data + odds collection

### Match results + line-ups

Drie kandidaten, eerst onderzoeken voor we kiezen:

| Bron | Voor | Tegen |
|---|---|---|
| **handball.net** (officieel HBL) | Volledige historische resultaten, line-ups, statistieken. Officieel = stabiel. | Geen publieke API gevonden — vereist HTML-scrape. Risico op TOS-overtreding bij high-frequency scraping. |
| **Sofascore API** | Meerdere sporten, redelijke API, makkelijk uit te breiden naar latere sport-modules. | Reverse-engineered (geen officiële public API), risico op rate-limiting / blokkade. |
| **kicker.de** | Goed gestructureerde HTML, NL/DE perspectief, historische data terug tot 2010+. | Pure scrape, fragiel bij site-redesigns. |

**Onderzoeks-actie S1**: maak een minimale fetch-test per bron. Pak die met
de beste robuustheid + history-coverage. Cache match-resultaten lokaal in
`sports/hbl/data/matches.parquet`.

### Closing odds (backtest) + live odds (inference)

| Bron | Voor | Tegen |
|---|---|---|
| **Pinnacle** | Sharpest book, industriestandaard voor CLV-benchmark. | Vereist account; afhankelijkheid kan verdwijnen bij account-sluiting. |
| **Bet365** | Goede liquiditeit, redelijk transparent. | Niet sharp genoeg voor zuivere CLV; geo-blocking kan opspelen. |

**Beslissing**: Pinnacle als primaire CLV-referentie. Bet365 als tweede
data-punt en als fallback-bron voor live-odds wanneer Pinnacle 'm laat
liggen. Closing-snapshot = 2 minuten vóór kickoff.

### Output S1

- `sports/hbl/data/matches.parquet` — historische match-results (3+ seizoenen)
- `sports/hbl/data/odds_closing.parquet` — closing odds per match
- `sports/hbl/data/odds_live.parquet` — live odds-snapshots (alleen voor live-window)
- `crypto/src/data_fetcher.py`-stijl helpers in `sports/hbl/src/data_fetcher.py`

---

## S2 — Feature engineering

Features op basis van wat in handbal voorspellend is gebleken in
academische literatuur + bookmaker-modellen:

| Feature | Berekening | Verwachte waarde |
|---|---|---|
| `elo_diff` | Elo home − Elo away (handbal-tuned K-factor ~16) | Hoog — directe team-strength proxy |
| `recent_form_5` | Som goal-differences laatste 5 matches per team | Medium — momentum-effect |
| `home_advantage` | Multiplier op λ_home (start 1.05, empirisch tunen) | Medium — handbal heeft sterk thuisvoordeel |
| `rest_days_diff` | Rust-dagen home − rust-dagen away | Laag-medium — vermoeidheid effect |
| `h2h_last_4` | Goal-difference laatste 4 onderlinge wedstrijden | Laag — kleine sample, ruis |
| `season_progress` | % wedstrijden gespeeld in seizoen (0.0–1.0) | Laag — legacy van vroeg-seizoen-volatiliteit |

**Niet in S2** (mogelijk in S5+): line-up-strength (afwezige sterspelers),
weersinvloeden (n.v.t. binnensport), play-style-matchup, referee bias.

### Output S2

- `sports/hbl/data/features.parquet` — feature matrix per match
- `sports/hbl/src/features.py` — Elo-rating tracker + feature-engine

---

## S3 — Skellam-model + backtest

### Modeltraining

Per team schatten:
- `attack_strength` — gemiddelde gescoorde goals (gewogen recent-meer)
- `defense_factor` — gemiddelde tegenstander-aanval (gewogen)
- Plus: globale `home_advantage_multiplier`, geschat op trainset

Geen ML in deze sprint — pure statistische modellering. Waarom: handbal
heeft kleine sample-size (~600 matches over 4 seizoenen), Skellam met
maximum-likelihood is interpretabel en niet gevoelig voor overfit.

### Backtest

Walk-forward met **seizoens-folds** — train op seizoen 1+2, test op
seizoen 3. Schuif door. Minimum trainset 1.5 seizoenen.

**Primary metric: Closing Line Value (CLV)**

```
CLV = (model_implied_prob / market_implied_prob_closing − 1) × 100
```

Positief = model beat de markt's closing line consistent. Pinnacle is
de referentie. > 0 over voldoende trades = edge.

**Secondary metrics**: ROI, yield, log-loss, Brier score.

### Output S3

- `sports/hbl/data/model.pkl` — getraind model
- `sports/hbl/data/walkforward.csv` — fold-by-fold metrics
- `sports/hbl/src/model.py` — Skellam-fit + predict
- `sports/hbl/src/backtest.py` — walk-forward + CLV-berekening

---

## S4 — Live signaal + Discord + paper trading

### Cron-flow

GitHub Actions workflow `sports_signal.yml` — getriggerd elk half uur
(`:00` en `:30` UTC) via cron-job.org. Stappen:

1. Check upcoming matches in `[1h45m, 2h15m]` of `[3h45m, 4h15m]` venster
2. Voor elke match in venster: bereken Skellam-probabilities
3. Vergelijk met current Pinnacle/Bet365 odds → CLV-edge per uitkomst
4. Edge boven drempel (S3 te kalibreren) → Discord-bericht + paper-trade
5. Commit `latest_signal.json` + `paper_trades.json` terug naar master
   met prefix `S: signal: <date>`

### Discord-bericht-format (concept)

```
🟢 **HBL — TSV Hannover-Burgdorf vs Füchse Berlin**
⏰ 19:00 CEST | T-2h | Kickoff over 02:00:00

Skellam baseline:
  Home: 42% (Pinnacle: 38% — edge +4.0pp / +10.5%)
  Draw: 14% (Pinnacle: 12%)
  Away: 44% (Pinnacle: 50%)

🎯 Edge: HOME (+10.5% CLV vs Pinnacle close)
💰 Stake: half-Kelly @ 2.6% kapitaal
```

Webhook-env-var: `DISCORD_WEBHOOK_URL_SPORTS` (apart secret, andere
channel dan crypto).

### Paper trading

Hergebruik `shared/paper_state.py` — symbol-agnostisch, initial_capital
configureerbaar. State in `sports/hbl/data/paper_trades.json`.

### Output S4

- `.github/workflows/sports_signal.yml` — cron workflow
- `sports/hbl/src/live_alert.py` — signaal + Discord + paper update
- `sports/hbl/data/paper_trades.json` — running state

---

## S5 — Total Goals markt (over/under)

Skellam geeft P(home_goals + away_goals > N) gratis (Poisson-som heeft
zelf ook Poisson-distributie met λ_total = λ_home + λ_away). Dus:

- Geen nieuwe modeltraining nodig
- Wel: aparte CLV-evaluatie tegen Pinnacle's totals-markt
- Discord-bericht uitbreiden met over/under + edge

**Niet in scope hier**: handicap-markt (vereist andere modelling),
exact-score-markt (te lage edge bij Skellam-baseline), live-betting (
vergt minute-by-minute model-update).

---

## Open vragen

- **Scrape-toestemming**: handball.net en kicker.de TOS lezen vóór S1.
  Bij twijfel: eerst manueel in CSV.
- **Pinnacle-account**: heeft Dennis er al een, of moeten we eerst dat
  aanmaken voor odds-toegang?
- **Cron-job.org budget**: huidige cron-job.org gebruikt 4 jobs voor crypto
  (hourly, 4h, daily, signal). Sport-cron komt erbij — checken of
  free-tier nog ruim genoeg is.

---

## Referenties

- Skellam-distributie voor handbal: A. Groll et al., "Goal counts in
  team handball", _Stat. Modelling_ (2018).
- CLV als metric: Alex Belov, "Why CLV is the only metric that matters
  in sports betting" (2019).
- Pinnacle als sharp benchmark: standaard aanname in sports-betting-
  literatuur — zie Constantinou & Fenton (2012).
