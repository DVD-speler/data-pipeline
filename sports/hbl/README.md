# sports/hbl/

Skellam-distributie baseline-model voor de **Handball-Bundesliga (HBL)**.
Genereert pre-match 1X2-signalen op T-2h en T-4h voor kickoff,
distribueerde via Discord (S4), met paper-trading state.

## Status

**S0 — Scaffold (deze commit).** Alle fases in `main.py` raisen
NotImplementedError. Implementatie volgt:
- **S1** — data-bron + odds collection
- **S2** — feature engineering (Elo, form, home advantage, rest, H2H)
- **S3** — Skellam-model + walk-forward backtest met CLV-metric
- **S4** — live signaal + Discord + paper trading
- **S5** — Total Goals markt-uitbreiding

Zie [`docs/sports/ROADMAP.md`](../../docs/sports/ROADMAP.md) voor concrete
plannen per sprint.

## Waarom HBL als eerste sport

- **Hoge scoringsrate** (~30 goals/team/match) maakt Poisson-aannames
  realistisch — Skellam-baseline is wiskundig schoon en interpreteerbaar.
- **Stabiele competitie** (18 teams, 34 speelrondes per seizoen) — genoeg
  data voor walk-forward (meerdere seizoenen ≥ 600 matches).
- **Vloeibare odds-markt** bij Pinnacle / Bet365 — closing-line-value als
  primary metric is haalbaar.
- **Niche** in vergelijking met voetbal — minder hedge-fund-modellen
  jagen op edge, makkelijker om consistent boven de markt te modelleren.

## Modelaanpak — Skellam baseline

```
home_goals  ~ Poisson(λ_home)
away_goals  ~ Poisson(λ_away)
goal_diff   ~ Skellam(λ_home, λ_away)

P(home_win) = P(goal_diff > 0)
P(draw)     = P(goal_diff = 0)
P(away_win) = P(goal_diff < 0)
```

`λ` per team wordt geschat uit recent gemiddelde goals voor/tegen,
gewogen met Elo-difference en home advantage:

```
λ_home = base_attack_home  × def_factor_away × home_advantage_multiplier
λ_away = base_attack_away  × def_factor_home
```

Configuratie-parameters in [`config.py`](config.py):
- `SKELLAM_HOME_ADVANTAGE = 1.05` (startwaarde, empirisch tunen in S3)
- `SKELLAM_DEFAULT_LAMBDA = 28.5` (historisch HBL-gemiddelde)

## Markten

**MVP (S1–S4):** alleen 1X2 — home-win / draw / away-win. Skellam geeft
direct alle drie.

**Geplaatst voor S5+:** Total Goals (over/under bv. 56.5). Skellam levert
ook P(total > N) gratis op via Poisson-som, dus uitbreiding is laagdrempelig.

Niet in scope nu: handicap, exact-score, live-betting.

## Live-signaal timing

```python
SIGNAL_WINDOWS_HOURS = [(1.75, 2.25), (3.75, 4.25)]
CRON_INTERVAL_MINUTES = 30
```

Cron draait elke 30 min op `:00` en `:30`. Per tick checkt het script of
er matches zijn waarvan de kickoff binnen het venster `[1h45m, 2h15m]`
of `[3h45m, 4h15m]` valt. Eén signaal per match per venster — geen
race-conditions of gemiste matches mits de cron tijdig draait.

## Lokale ontwikkeling (zodra S1 klaar is)

Vanuit repo-root:

```bash
# Help (werkt al in S0)
python sports/hbl/main.py --help

# Data + odds vullen (S1)
python sports/hbl/main.py --phase data
python sports/hbl/main.py --phase odds

# Train Skellam baseline (S3)
python sports/hbl/main.py --phase features
python sports/hbl/main.py --phase model
python sports/hbl/main.py --phase backtest

# Live signaal-check (S4)
python sports/hbl/main.py --phase signal
DISCORD_WEBHOOK_URL_SPORTS=https://... python sports/hbl/main.py --phase live_alert
```

Imports werken vanuit elke cwd zolang `Path(__file__).resolve().parent`
gebruikt wordt voor paden — geen hard-coded paths.

## Cross-project utilities

- **Top-level `shared/`** — Discord notifier, paper-state, Kelly sizing
  (gedeeld met crypto). Direct importeerbaar mits repo-root op
  `PYTHONPATH` staat.
- **`sports/shared/`** — sport-overstijgende utils (odds-format, CLV, EV).
  Voor nu leeg; vullen vanaf S2 zodra een tweede sport het ook nodig heeft.
