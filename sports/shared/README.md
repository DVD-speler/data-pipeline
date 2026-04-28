# sports/shared/

Sport-overstijgende utils. **Voor nu leeg** — wordt gevuld zodra een
tweede sport-module hetzelfde patroon nodig heeft. Vroeg-promoten leidt
tot abstractie zonder concrete second-use-case.

## Geplande modules (vanaf S2)

| Module | Doel | Wanneer |
|---|---|---|
| `odds_format.py` | Decimaal ↔ Amerikaans ↔ fractioneel ↔ implied probability | bij eerste cross-sport gebruik |
| `clv.py` | Closing Line Value berekening (Pinnacle of Bet365 als referentie) | S3 (HBL backtest) — eerst HBL-only, daarna promoten |
| `ev.py` | Expected Value: model_proba × decimal_odds − 1 | S3 |
| `market_compare.py` | Vergelijk model-proba met markt-implied prob → edge-scoring | S3 |

## Conventie

Promoot een module hierheen wanneer:

1. **Twee of meer** sporten het zelfde nodig hebben (in de praktijk
   gebruikt, niet alleen "zou kunnen")
2. De API stabiel is (geen frequente refactors meer)
3. Geen sport-specifieke kennis in de code zit (geen "if HBL" branches)

Tot dan: hou het in de sport-map. Splitsen-na-bewijs is goedkoper dan
ontvouwen-na-aanname.
