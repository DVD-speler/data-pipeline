# Sports Roadmap

Eerste sport: **HBL (Handball-Bundesliga, 1. Liga, Duitsland)**.
Skellam-baseline, 1X2-markt MVP, signalen op T-2h en T-4h voor kickoff
via cron-getriggerde GitHub Actions workflows naar Discord.

## Design-overview & vastgelegde keuzes

| Onderdeel | Keuze | Waarom |
|---|---|---|
| **Eerste sport** | HBL (1. Bundesliga, Duitsland) | Hoge scoringsrate (~30/team/match) maakt Poisson-aannames realistisch; stabiele competitie 18 teams × 34 speeldagen → ~600 matches/seizoen; minder hedge-funds dan voetbal → realistische edge mogelijk. |
| **Historische scope** | 5 seizoenen 2020-21 t/m 2024-25 + huidige (2025-26) als out-of-sample | ≥ 3000 matches voor walk-forward training, lopend seizoen blijft volledig hold-out tot na S3 |
| **Liga-uitbreiding** | Alleen HBL voor MVP. **2. Bundesliga / EHF Champions League als optie** voor bredere training-set in latere sprints | Eerst proven edge in één liga; multi-liga voegt features-engineering complexiteit toe (liga-strength-multiplier) |
| **Model** | Skellam-distributie via Poisson-regression op features | `home_goals - away_goals ~ Skellam(λ_h, λ_a)` waarbij λ's gemodelleerd worden via Poisson-regression met Elo-diff + form + home advantage als covariaten. Geeft 1X2 én Total Goals direct |
| **Markt MVP** | 1X2 (home/draw/away). **Total Goals (S5)** komt gratis uit Skellam (Poisson-som) | Skellam levert P(diff > 0/= 0/< 0) én P(total > N) zonder hertraining |
| **Backtest-benchmark** | **Pinnacle closing line** | Sharpest book; standaard CLV-referentie in sports-betting-literatuur |
| **Primary metric** | Closing Line Value (CLV) > 0% over voldoende trades | Bewijst edge tegen de sharpste markt; ROI/yield zijn afhankelijk van bookies-keuze, CLV is universeel |
| **Execution-target (live)** | **Betfair Exchange** | Peer-to-peer, geen winner-limits, NL-grijs maar bruikbaar; past bij CLV-driven strategie omdat de exchange dichter bij sharp odds zit |
| **Cheap-first pad** | Scraping voor S1–S3 (gratis), paid odds API pas in S4 *als* backtest edge bewijst | Geen API-kosten investeren in een hypothese; promoveren wanneer waarde is aangetoond |
| **Discord-channel** | Eigen webhook `DISCORD_WEBHOOK_URL_SPORTS` | Aparte channel van crypto (3 webhooks daar al actief) |

## Sprint-overzicht

| Sprint | Doel | Status |
|---|---|---|
| **S0** | Scaffold (directory-structuur, fase-runner stubs, config) | ✅ |
| **S1** | HBL match-data 5 seizoenen (OpenLigaDB + Sofascore) | open |
| **S1.5** | Odds-historie 5 seizoenen (OddsPortal-scrape) | open |
| **S2** | Feature engineering (Elo, form, home advantage, rest, H2H) | open |
| **S3** | Skellam-model + walk-forward backtest met CLV-metric | open |
| **S4** | Live signaal + Discord + paper trading (+ paid odds als edge proven) | open |
| **S5** | Markt-uitbreiding: Total Goals (over/under ~55.5) | open |
| **S6+** | Volgende sport (vrouwenvoetbal Eredivisie / NBA totals / paardenrennen) | t.b.d. — pas na HBL-rendabiliteit |

---

## S1 — HBL match-data (5 seizoenen)

**Doel**: Volledige historische match-database (2020-21 t/m 2024-25) +
huidige seizoen, gecached lokaal in parquet per seizoen.

### Data-sources

| Bron | Rol | Beschikbaarheid |
|---|---|---|
| **OpenLigaDB API** | Primaire bron — fixtures + uitslagen + speeldata | Gratis, geen key, JSON REST. Stabiel sinds 2010 |
| **Sofascore-scrape** | Aanvulling — line-ups, shots, saves, possession | Gratis maar reverse-engineered API. Python-lib `sofascore-py` als startpunt |

OpenLigaDB endpoint-pattern:
```
https://api.openligadb.de/getmatchdata/hbl/{seizoen}
# {seizoen} = "2020", "2021", ..., "2025"
```

**Risico**: Sofascore-scrape is fragile bij site-updates → fallback naar
OpenLigaDB-only werkt nog (zonder line-up-features). Bij Sofascore-failure
gaat S2 gewoon door zonder line-up-strength feature; toevoegen wanneer
Sofascore weer stabiel is.

**Niet gekozen** (overwogen, verworpen):
- *handball.net* — geen publieke API, pure HTML-scrape, ToS onduidelijk
- *kicker.de* — fragiel bij site-redesigns, voegt weinig toe boven OpenLigaDB

### Output S1

```
sports/hbl/data/matches/
├── 2020-21.parquet
├── 2021-22.parquet
├── 2022-23.parquet
├── 2023-24.parquet
├── 2024-25.parquet
└── 2025-26.parquet     # current, growing per cron-tick
```

Helpers in `sports/hbl/src/data_fetcher.py` (zelfde stijl als
`crypto/src/data_fetcher.py` — incremental download, SQLite optioneel
als de parquet-per-seizoen-aanpak te traag wordt).

---

## S1.5 — Odds-historie (5 seizoenen)

**Doel**: Closing odds per match voor alle 5 historische seizoenen, plus
spread-data van 5–10 bookies voor mogelijke market-deviation features.

### Source

**OddsPortal-scrape** (`https://www.oddsportal.com/handball/germany/handball-bundesliga/`):
- Closing odds van Pinnacle (primaire CLV-referentie) + Bet365, William
  Hill, bet-at-home, Betway, Unibet, etc. (5–10 bookies)
- Closing-snapshot = laatste odds-update vóór kickoff
- Per-seizoen-archiefpagina's bevatten alle matches met definitieve odds

**Polite throttling** (essentieel — OddsPortal ToS niet scrape-vriendelijk):
- 1 req/sec maximum
- User-agent rotatie (3–5 strings)
- Retry-with-backoff op 429
- Stop bij eerste 403 — wacht tot volgende dag voor verdere downloads
- Hou referer-header op match-pagina's

### Output S1.5

```
sports/hbl/data/odds/
├── 2020-21.parquet     # match_id, bookie, home/draw/away closing odds
├── ...
└── 2025-26.parquet     # current, blijft groeien per nieuwe match
```

`match_id`-koppeling met `sports/hbl/data/matches/{seizoen}.parquet` via
date + team-namen (string-fuzzy-match — verschillende sites gebruiken
soms "Hannover-Burgdorf" vs "TSV Hannover-Burgdorf").

**Risico**: bij ToS-update of IP-blokkade van OddsPortal moeten we
overschakelen naar paid-fallback eerder dan gepland. Mitigatie: kleine
proof-of-concept eerst (1 seizoen scrapen, succes = green light voor de
overige 4).

---

## S2 — Feature engineering

**Doel**: feature matrix per match, klaar voor Poisson regression in S3.

### Features

| Feature | Berekening | Verwachte impact |
|---|---|---|
| `elo_diff` | Elo home − Elo away. **K-factor ~32**, start-rating 1500, **goal-diff bonus** in update (`K × log(1 + abs(goal_diff))`) | Hoog — directe team-strength proxy |
| `recent_form_5` | Gewogen som goal-differentials over laatste 5 matches (recent zwaarder via `0.9^i`-weighting) | Medium — momentum-effect |
| `home_advantage` | **Per team geleerd** uit historische home-vs-away goal-differential (≥ 30 thuiswedstrijden vereist, anders league-default) | Medium — handbal heeft sterk thuisvoordeel, varieert per team |
| `rest_days_diff` | (rest-dagen home) − (rest-dagen away) | Laag-medium — vermoeidheid effect zichtbaar bij ≤ 3 dagen rust |
| `h2h_last_3` | Goal-differential laatste 3 onderlinge wedstrijden | Laag — kleine sample, ruis-prone, maar gratis te berekenen |
| `season_progress` | % wedstrijden gespeeld in lopend seizoen (0.0–1.0) | Laag — vroeg-seizoen-volatiliteit indicator |

**Optioneel later (afhankelijk van Sofascore-stabiliteit)**:
- `lineup_strength_diff` — som van top-N spelers' minutes-weighted ratings
- Niet in S2 hard-vereist; toevoegen in S2.5 als feature-engineering-bonus

**Niet in scope**: weersinvloeden (n.v.t. binnensport),
play-style-matchup, referee bias, transfers (te zeldzaam mid-seizoen).

### Output S2

- `sports/hbl/data/features.parquet` — feature matrix per match (alle 5
  seizoenen + current)
- `sports/hbl/src/features.py` — Elo-rating tracker (incremental update
  per match), feature-engine

---

## S3 — Skellam-model + walk-forward backtest

### Modelaanpak

`home_goals - away_goals ~ Skellam(λ_home, λ_away)`. λ's worden
gemodelleerd via **Poisson regression** op features:

```
log(λ_home) = β₀ + β₁ × elo_diff + β₂ × recent_form_5_home
            + β₃ × home_advantage_team + β₄ × rest_days_home + ...
log(λ_away) = β₀' + β₁' × (-elo_diff) + β₂' × recent_form_5_away + ...
```

Twee gescheiden Poisson-regressies (één per team-side), parameters
gefit via maximum-likelihood. Voorspellingen:

```
P(home_win) = P(Skellam(λ_h, λ_a) > 0)
P(draw)     = P(Skellam(λ_h, λ_a) = 0)
P(away_win) = P(Skellam(λ_h, λ_a) < 0)
```

Geen ML/boosting in deze sprint — pure statistische regressie past bij
de sample-size (~3000 matches) en blijft interpreteerbaar (β-waarden
direct af te lezen als feature-importance).

### Walk-forward backtest

**Seizoens-folds**: train op n-1 seizoenen, test op seizoen n. Schuif door:

| Fold | Train | Test |
|---|---|---|
| 1 | 2020-21, 2021-22 | 2022-23 |
| 2 | 2020-21, 2021-22, 2022-23 | 2023-24 |
| 3 | 2020-21, 2021-22, 2022-23, 2023-24 | 2024-25 |

Lopend seizoen 2025-26 = volledige out-of-sample hold-out — tot na S3 niet
aangeraakt voor model-evaluatie.

### Metrics

**Primary: Closing Line Value (CLV) vs Pinnacle**
```
CLV = (model_implied_prob / pinnacle_closing_implied_prob − 1) × 100
```
> 0% over alle test-folds = model verslaat consistent de sharpste markt =
edge bewezen.

**Secondary**: ROI, hit-rate, Sharpe ratio (per fold), maximum drawdown,
log-loss, Brier score.

### Decision-gate na S3

| Resultaat | Vervolg |
|---|---|
| **CLV > 1% gemiddeld + ≥ 2 van 3 folds positief** | S4 met paid-odds-upgrade (The Odds API of Pinnacle direct) |
| **CLV > 0% maar < 1%, marginale edge** | S4 met goedkope OddsPortal-scrape voor live odds, herevalueer over 3 maanden |
| **CLV ≤ 0%** | Model herontwerpen of pivot naar andere sport. Geen S4 met deze baseline |

### Output S3

- `sports/hbl/data/models/skellam_v1.pkl` — getraind model (Poisson coefs)
- `sports/hbl/data/walkforward.csv` — fold-by-fold metrics
- `sports/hbl/data/walkforward.png` — CLV + ROI + drawdown plots
- `sports/hbl/src/model.py` — Poisson-regression fit + Skellam-predict
- `sports/hbl/src/backtest.py` — walk-forward + CLV-berekening tegen Pinnacle

---

## S4 — Live signaal + Discord + paper trading

**Voorwaarde**: S3 gate gepasseerd (CLV > 0%).

### Cron-flow

GitHub Actions workflow `sports_signal.yml`, getriggerd elk half uur
(`:00` en `:30` UTC) via cron-job.org. Stappen:

1. Check upcoming matches in `[1h45m, 2h15m]` of `[3h45m, 4h15m]` venster
2. Voor elke match in venster: bereken Skellam-probabilities via fitted model
3. Vergelijk met **current odds** (zie odds-bron-keuze hieronder) → CLV-edge
   per uitkomst
4. Edge boven drempel (S3-gekalibreerd, typisch +3pp implied-prob delta) →
   Discord-bericht + paper-trade
5. Commit `latest_signal.json` + `paper_trades.json` terug naar master
   met prefix `S: signal: <date>`

### Live-odds-bron — afhankelijk van S3-uitkomst

| S3-uitkomst | Live-odds-bron | Kosten |
|---|---|---|
| **Sterke edge (CLV > 1%)** | The Odds API (`the-odds-api.com`) of Pinnacle's eigen API | ~€30/maand |
| **Marginale edge** | OddsPortal-scrape voor live odds (zelfde infra als S1.5) | Gratis |
| **Geen edge** | n.v.t. — S4 wordt overgeslagen |

### Execution-target (informatie, geen automation in S4)

**Betfair Exchange** als geplande target:
- Peer-to-peer model — geen winner-limits zoals bij traditionele bookies
- NL-grijs (legaal grijs gebied), maar functioneel bruikbaar
- Past bij CLV-strategie omdat exchange-odds dicht bij sharp-odds zitten

S4 zelf doet alléén Discord-alert + paper trading. Auto-execution
naar Betfair is een separate sprint (S6+ analoog aan crypto's Sprint 20
voor Bybit-execution).

### Discord-bericht-format (concept)

```
🟢 **HBL — TSV Hannover-Burgdorf vs Füchse Berlin**
⏰ 19:00 CEST | T-2h | Kickoff over 02:00:00

Skellam baseline (5-season trained):
  Home: 42% (Pinnacle closing: 38% — edge +4.0pp / +10.5%)
  Draw: 14% (Pinnacle: 12%)
  Away: 44% (Pinnacle: 50%)

🎯 Edge: HOME (+10.5% CLV vs Pinnacle close)
💰 Stake: half-Kelly @ 2.6% kapitaal (Betfair-execution target)
📊 Confidence: 4/5 features stabiel; Sofascore-line-up missing
```

Webhook env-var: `DISCORD_WEBHOOK_URL_SPORTS` (apart secret, andere
channel dan crypto).

### Paper trading

Hergebruik `shared/paper_state.py` (zelfde module als crypto). Nieuwe
state-file `sports/hbl/data/paper/paper_trades.json`. `initial_capital`
configureerbaar (start 1000 EUR conform crypto-conventie).

### Output S4

- `.github/workflows/sports_signal.yml` — cron workflow met `on.push.paths:
  ['sports/**', 'shared/**', '.github/workflows/sports_*.yml',
  '!sports/hbl/data/**']`
- `sports/hbl/src/live_alert.py` — signaal + Discord + paper update
- `sports/hbl/data/paper/paper_trades.json` — running state

---

## S5 — Total Goals markt (over/under)

Skellam levert `P(home_goals + away_goals > N)` direct via som-van-Poisson:

```
total_goals ~ Poisson(λ_home + λ_away)
P(total > 55) = 1 − P(Poisson(λ_total) ≤ 55)
```

HBL-typische lijnen liggen rond **55.5** (gemiddelde totaal ~57 over 5
seizoenen, lijn iets onder gemiddelde voor over-bias). Backtest bewijst
of we Pinnacle's Total Goals-line ook kunnen verslaan.

### Werkwijze

1. Geen nieuwe modeltraining nodig — gebruik dezelfde S3 model
2. Bereken P(over/under) voor elke match in backtest
3. Aparte CLV-evaluatie tegen Pinnacle's totals-markt (kan andere edge
   geven dan 1X2)
4. Discord-bericht uitbreiden met over/under-edge

**Niet in scope**:
- Handicap-markt (vereist andere modelling-keuzes — handbal-handicap is
  meer gespecialiseerd dan voetbal)
- Exact-score-markt (te lage edge bij Skellam — outliers domineren)
- Live in-play betting (vergt minute-by-minute model-update)

---

## Future sports (S6+)

Pas relevant na bewezen HBL-rendabiliteit. Kandidaten:

| Sport / markt | Waarom interessant | Modelaanpak (concept) |
|---|---|---|
| **Vrouwenvoetbal Eredivisie** | Minder bookmaker-aandacht, hoge-scoringsrate genoeg voor Skellam | Skellam analoog aan HBL, lagere λ's (~2.5/team) |
| **NBA Totals (over/under)** | Diepe odds-markt + voorspelbare total-distributies | Normal-distributie of Poisson op pace × efficiency |
| **Paardenrennen (UK/IE)** | Markt-inefficiëntie in lower-grade races | Conditional logit / multinomial model |

Selectiecriterium: edge bewijsbaar tegen Pinnacle (of equivalent sharp
book) in walk-forward backtest, met data-source kosten < verwachte ROI
in eerste 3 maanden. Geen sport adopteren zonder S3-equivalent gate.

---

## Open punten — uit weg vóór S1

- **OpenLigaDB schema-stabiliteit**: één test-call per seizoen om response-
  format te valideren vóór bulk-download
- **Sofascore rate-limit**: officieel niet gepubliceerd; start met 1
  req/2sec en escaleer alleen bij stabiele 200-responses
- **OddsPortal-scrape ToS**: lees vóór S1.5 + bouw in alleen gebruik voor
  persoonlijke analyse (geen herpublicatie van scraped data)
- **Cron-job.org budget**: huidige 4 jobs voor crypto (hourly, 4h, daily,
  signal). Sport-cron komt erbij — checken of free-tier (5 jobs) nog ruim
  genoeg is, anders upgrade naar EUR 0.99/maand tier
- **Betfair-account**: NL-residenten kunnen registeren, KYC vereist —
  Dennis regelt dit pas bij S6 execution-sprint

---

## Referenties

- Skellam-distributie voor handbal: A. Groll et al., "Goal counts in
  team handball", _Stat. Modelling_ (2018)
- CLV als metric: Alex Belov, "Why CLV is the only metric that matters
  in sports betting" (2019)
- Pinnacle als sharp benchmark: standaard aanname in sports-betting-
  literatuur — zie Constantinou & Fenton (2012)
- Poisson regression voor sports: Dixon & Coles, "Modelling association
  football scores and inefficiencies in the football betting market"
  (1997) — klassieker; HBL-toepassing analoog
