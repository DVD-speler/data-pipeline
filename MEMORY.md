# Repo Memory

Project-level state. Niet hetzelfde als Claude Code auto-memory (die
zit in `~/.claude/projects/...`). Dit bestand is voor menselijke lezers
+ Claude die de repo leest tijdens een sessie.

## Wat dit is

Monorepo voor signaalmodellen op kansstatistische basis. Twee project-
mappen (crypto, sports) + één shared map.

## Status (2026-04-27)

- **crypto**: Sprint 19 voltooid, gem WF Sharpe BTC +5.97. Sprint 20
  (Bybit live order execution) is de volgende sprint maar nog niet
  gestart. Single-symbol (BTC) sinds april 2026 — ETH gerevert wegens
  AUC 0.53. ETH OHLCV blijft gedownload als cross-asset feature
  (`eth_btc_ratio` in BTC's feature matrix).
- **sports**: scaffold gepland in Sprint S0 (niet gestart).
- **shared**: drie modules actief — notifier (Discord), paper_state, kelly.
- **monorepo migration**: M1 (april 2026) — directories opgesplitst naar
  crypto/ + sports/ + shared/, workflows hernoemd naar `crypto_signal_*.yml`.

## Live cron

Drie workflows in `.github/workflows/crypto_signal_*.yml`. Cron-job.org
triggert ze via `workflow_dispatch`:
- `crypto_signal_hourly.yml` — elk uur :05 UTC → Discord `DISCORD_WEBHOOK_URL`
- `crypto_signal_4h.yml` — elke 4 uur :05 UTC → Discord `DISCORD_WEBHOOK_URL_4H`
- `crypto_signal_daily.yml` — 07:05 UTC dagelijks → Discord `DISCORD_WEBHOOK_URL_DAILY`

Werkt met `working-directory: crypto` + `PYTHONPATH=${{ github.workspace }}`
zodat `from src.foo` (binnen crypto) én `from shared.X` (cross-project)
beide werken.

## Commit-prefix conventie

- `C:` crypto/
- `S:` sports/
- `M:` shared/
- `R:` repo-tooling / cross-cutting

Cron-commits gebruiken `C: signal: ...`, `C: 4h-signal: ...`, `C: daily-signal: ...`.

## Belangrijke beperkingen

- **Geen Python op Windows host** — alles in devcontainer (Docker) of GH Actions Ubuntu.
- **Geen automatic retraining** — `model.pkl` is statisch artefact, hertrainen handmatig.
- **Workflows triggeren ook op push** met `paths: ['crypto/**', 'shared/**', '!crypto/data/**']`
  (excludes data-commits zodat cron-pushes de workflow niet retriggeren).
