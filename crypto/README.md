# crypto/

P1/P2-gebaseerd kansmodel + live signaalpijplijn voor BTCUSDT op 1h, 4h
en dagelijkse horizons. Output gaat naar Discord via cron-getriggerde
GitHub Actions workflows.

## Quickstart (devcontainer)

```bash
# Eerst data + features bouwen
python crypto/main.py --phase data
python crypto/main.py --phase external_data
python crypto/main.py --phase p1p2
python crypto/main.py --phase features

# Model trainen
python crypto/main.py --phase model

# Live signaal genereren (dry, naar JSON, geen Discord)
python crypto/main.py --phase signal --symbol BTCUSDT

# Volledige live alert (post naar Discord — env-var DISCORD_WEBHOOK_URL nodig)
DISCORD_WEBHOOK_URL=... python crypto/main.py --phase live_alert --symbol BTCUSDT
```

Alle commands draaien vanuit repo-root. Werken vanuit `crypto/`-cwd
werkt ook (`python main.py ...`); imports zijn identiek doordat
`crypto/` automatisch op sys.path staat.

## Belangrijkste docs

- **`docs/ARCHITECTURE.md`** (top-level) — pipeline, gates, risk-management
- **`docs/crypto/ROADMAP.md`** — sprint-historie + Sprint 20 (Bybit live execution)
- **`docs/crypto/LESSONS.md`** — geleerde lessen + mislukte experimenten

## Imports

Voor cross-project code (Discord, paper-state, Kelly) wordt
`shared/` gebruikt — zie `shared/README.md`.

```python
from shared.notifier import send_alert            # Discord webhook
from shared.paper_state import load_paper_state   # paper trading state
from shared.kelly import compute_half_kelly        # positiegrootte
```
