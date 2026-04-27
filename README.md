# Signal Models

Monorepo voor kansstatistiek-gebaseerde signaalmodellen. Eén top-level
repo voor meerdere domeinen, met gedeelde infrastructuur.

> Repo-naam staat nog op `crypto_signal_model` — hernoeming naar
> `signal_models` volgt zodra de monorepo-structuur stabiel is.

## Projects

| Map | Status | Beschrijving |
|---|---|---|
| **`crypto/`** | actieve productie | BTC P1/P2 model met 1h, 4h en dagelijkse horizon. Live cron via GitHub Actions, signalen naar Discord. Sprint 19 voltooid (gem WF Sharpe BTC +5.97), Sprint 20 (Bybit live order execution) is volgende. |
| **`sports/`** | placeholder (S0) | Tegenhanger voor sport-events. Scaffold komt in Sprint S0. |
| **`shared/`** | actief | Cross-project utilities: Discord notifier, paper-trading state, Kelly sizing. |

## Documentatie

- **[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)** — monorepo-overzicht + crypto pipeline details
- **[`docs/crypto/ROADMAP.md`](docs/crypto/ROADMAP.md)** — crypto sprint-historie + Sprint 20 plan
- **[`docs/crypto/LESSONS.md`](docs/crypto/LESSONS.md)** — geleerde lessen + mislukte experimenten
- **[`docs/sports/ROADMAP.md`](docs/sports/ROADMAP.md)** — sports placeholder
- **[`crypto/README.md`](crypto/README.md)** — crypto quickstart + commands
- **[`shared/README.md`](shared/README.md)** — wanneer iets gedeeld is + module-overzicht

## Commit-prefix conventie

| Prefix | Scope |
|---|---|
| `C:` | crypto/ wijzigingen |
| `S:` | sports/ wijzigingen |
| `M:` | shared/ wijzigingen (multi-project) |
| `R:` | repo-tooling, workflows, docs cross-cutting |

Voorbeeld: `C: refactor: remove ETH model artifacts`, `R: monorepo migration`.

## Live signalen

Drie GitHub Actions workflows in `.github/workflows/crypto_signal_*.yml`:
- `crypto_signal_hourly.yml` — elk uur (`workflow_dispatch` via cron-job.org)
- `crypto_signal_4h.yml` — elk 4 uur
- `crypto_signal_daily.yml` — 07:05 UTC dagelijks

Alle drie posten naar verschillende Discord webhooks (3 secrets:
`DISCORD_WEBHOOK_URL`, `DISCORD_WEBHOOK_URL_4H`, `DISCORD_WEBHOOK_URL_DAILY`).

## Lokale ontwikkeling

Project draait in een devcontainer (`mcr.microsoft.com/devcontainers/python:3.11-bullseye`).
Vanuit repo-root:

```bash
# Installeer deps (eenmalig)
pip install -r requirements.txt

# Crypto signal genereren (vereist OHLCV cache + getrainde modellen)
python crypto/main.py --phase signal --symbol BTCUSDT
```
