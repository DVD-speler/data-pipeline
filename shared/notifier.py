"""
Discord webhook notifier — gedeeld door crypto en (straks) sports.

Eén entrypoint `send_alert(content, webhook_env_var=...)` die het bericht
post naar de webhook-URL uit de gegeven omgevingsvariabele. Caller kiest
welke env-var gebruikt wordt zodat verschillende channels parallel kunnen
draaien (bijv. uurmodel, 4h-model, dagmodel).

Bij missende env-var of HTTP-fout: print waarschuwing en return — nooit
exception. Het is essentieel dat live_alert-runs **niet** crashen op een
Discord-fout (cron exit-codet 0 → workflow draait door, paper state wordt
gewoon opgeslagen).
"""

import os

import requests


def send_alert(content: str, webhook_env_var: str = "DISCORD_WEBHOOK_URL") -> None:
    """Stuur een bericht naar Discord via de webhook-URL uit `webhook_env_var`."""
    webhook_url = os.environ.get(webhook_env_var)
    if not webhook_url:
        print(f"  [Discord] {webhook_env_var} niet ingesteld — alert overgeslagen.")
        return

    try:
        resp = requests.post(
            webhook_url,
            json={"content": content},
            timeout=10,
        )
        if resp.status_code in (200, 204):
            print("  [Discord] Alert verstuurd.")
        else:
            print(f"  [Discord] Fout {resp.status_code}: {resp.text[:200]}")
    except requests.RequestException as e:
        print(f"  [Discord] Verbindingsfout: {e}")
