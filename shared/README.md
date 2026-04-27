# shared/

Cross-project utilities die in zowel `crypto/` als `sports/` (en latere
projects) gebruikt kunnen worden.

## Wanneer iets hier hoort

- Gebruikt door **2+ projects** of duidelijk generiek (geen
  domain-knowledge over crypto/sports)
- Geen project-specifieke configuratie of state in de logica
- Stabiele API — frequent gewijzigde code hoort in het project zelf

Als je twijfelt: hou het in het project. Promoten naar `shared/` is
makkelijker dan terug-splitsen.

## Modules

- **`notifier.py`** — Discord webhook poster. Leest `DISCORD_WEBHOOK_URL`
  uit env (of een per-project override). Eén `send_alert(content,
  webhook_url=None)` functie.
- **`paper_state.py`** — load/save voor paper-trading state JSON
  (capital, open_position, closed_trades). Symbol-agnostisch.
- **`kelly.py`** — half-Kelly positiegrootte berekening op basis van
  historische win rate / R-ratio. Gebruikt door crypto's
  `live_alert.py` en straks ook sports.

## Imports vanuit projects

Werkt mits repo-root op `PYTHONPATH` staat:

```python
from shared.notifier import send_alert
from shared.paper_state import load_paper_state, save_paper_state
from shared.kelly import compute_half_kelly
```

GitHub Actions workflows zetten `PYTHONPATH=${{ github.workspace }}`. In
de devcontainer staat repo-root automatisch op sys.path via VS Code's
python-interpreter-config (of voeg `PYTHONPATH=.` toe als je vanuit
crypto/ direct script-aanroepen doet).
