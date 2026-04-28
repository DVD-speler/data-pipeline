# sports/

Sport-specifieke kansmodellen + signaalpijplijnen. Per sport een eigen
submap met een fase-runner die hetzelfde patroon volgt als
`crypto/main.py` (data → odds → features → model → backtest → signal →
live_alert).

## Per-sport submappen

| Map | Sport | Markt(en) MVP | Status |
|---|---|---|---|
| **`hbl/`** | Handball-Bundesliga (Duitsland) | 1X2 | scaffold (S0), implementatie in S1+ |
| _(later)_ | vrouwenvoetbal, horse racing, etc. | t.b.d. | niet gestart |

Elke sport heeft zijn eigen `config.py`, `main.py`, `src/` en `data/`
mappen. Ze delen niets met crypto behalve top-level `shared/` modules
(notifier, paper_state, kelly) en eventuele toekomstige
sport-overstijgende utilities in `sports/shared/`.

## Sport-overstijgende code

Code die door **meerdere sport-modules** gebruikt wordt komt in
`sports/shared/` (bijv. odds-format-conversie, CLV-berekening,
EV-vergelijking, market-line-parsing). Code die nog maar in één sport
gebruikt wordt blijft in die sport-map — pas promoten zodra een tweede
sport het ook nodig heeft.

Code die **ook door crypto** gebruikt wordt (zeldzaam voor sport-data,
maar denk aan Discord-notifier of paper-state) blijft in de top-level
`shared/`.

## Conventies

- **Phase-runner**: elke sport heeft een `main.py` met argparse
  `--phase X` flag, fasen identiek genoemd waar zinvol (`data`,
  `features`, `model`, `backtest`, `signal`, `live_alert`).
- **Discord-webhook**: aparte env-var per sport-domein (bijv.
  `DISCORD_WEBHOOK_URL_SPORTS` of fijner per sport).
- **Commit-prefix**: `S:` voor wijzigingen in `sports/`.
- **Config**: `Path(__file__).resolve().parent` voor alle paden — geen
  hardcoded absolute paden.

## Docs

- **`docs/sports/ROADMAP.md`** — concrete S1+ plan per sport
- **`sports/<sport>/README.md`** — sport-specifieke quickstart + design
  choices

## Lokale ontwikkeling

Vanuit repo-root:

```bash
# Help
python sports/hbl/main.py --help

# Een fase draaien (S0: alle fases raisen NotImplementedError tot S1+)
python sports/hbl/main.py --phase data
```
