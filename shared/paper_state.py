"""
Paper-trading state load/save — gedeeld door crypto en (straks) sports.

State is een JSON-dict met:
  open_position   : dict | None
  closed_trades   : list[dict]
  capital         : float (startwaarde via initial_capital)
  last_checked    : str | None — ISO timestamp van laatste run

Identieke logica werd gedupliceerd in `crypto/src/live_alert.py`,
`live_alert_4h.py` en `live_alert_daily.py`. Bij promotie naar
`shared/` blijft het gedrag gelijk.
"""

import json
from pathlib import Path


def load_paper_state(path: Path, initial_capital: float = 1000.0) -> dict:
    """Laad paper trading state, of maak een lege state aan bij eerste run.

    Onleesbare/lege bestanden worden behandeld als "eerste run" — er wordt
    een nieuwe lege state teruggegeven (kapitaal = `initial_capital`).
    """
    if path.exists():
        try:
            with open(path) as f:
                content = f.read().strip()
                if content:
                    return json.loads(content)
        except (json.JSONDecodeError, ValueError):
            print(f"  Waarschuwing: {path.name} onleesbaar — nieuwe state aangemaakt.")
    return {
        "open_position": None,
        "closed_trades": [],
        "capital": initial_capital,
        "last_checked": None,
    }


def save_paper_state(state: dict, path: Path) -> None:
    """Sla paper trading state op als JSON met indent=2 voor leesbaarheid."""
    with open(path, "w") as f:
        json.dump(state, f, indent=2, default=str)
