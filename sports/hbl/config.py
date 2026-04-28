"""
HBL (Handball-Bundesliga) configuratie — paden, league-config, env-var refs.

Alle paden via `Path(__file__).resolve().parent` zodat scripts werken
ongeacht cwd. Volgt hetzelfde patroon als `crypto/config.py`.
"""
from pathlib import Path


# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)


def data_path(filename: str) -> Path:
    """Geef het pad terug voor een HBL data-bestand. Voorbeeld:
    data_path("matches.parquet") → sports/hbl/data/matches.parquet
    """
    return DATA_DIR / filename


# ── League-configuratie ───────────────────────────────────────────────────────
LEAGUE_NAME = "HBL"                  # Handball-Bundesliga
SEASON      = "2025-26"              # Huidig speelseizoen

# Markten in MVP (S1-S3). Total Goals / handicap / live komt in S5.
MARKETS     = ["1X2"]                # Win-Draw-Lose (home / draw / away)


# ── Modelaanpak ───────────────────────────────────────────────────────────────
# Skellam-distributie als baseline: P(home_goals − away_goals = k) waar
# beide team-totalen Poisson-verdeeld zijn met team-specifieke λ. Werkt
# uitstekend voor handbal door hoge gemiddelde scoringsrate (~30/team).
MODEL_TYPE                = "skellam"
SKELLAM_HOME_ADVANTAGE    = 1.05     # multiplier op home-team λ — startwaarde,
                                     # wordt empirisch geschat in S3 backtest
SKELLAM_DEFAULT_LAMBDA    = 28.5     # historisch HBL-gemiddelde goals/team/match


# ── Live-signaal timing ───────────────────────────────────────────────────────
# De cron draait elke 30 minuten op :00 en :30. Per run vraagt het script:
# "zijn er matches in [1h45m, 2h15m] of [3h45m, 4h15m] van nu?". Window-breedte
# 30 min zorgt dat elke match exact één signaal krijgt op T-2h en T-4h zonder
# per-match cron-magic. Ondergrens net onder 2h en 4h zodat een match die
# 02:00 begint en de cron draait om 24:01 (T-1h59m) niet gemist wordt.
SIGNAL_WINDOWS_HOURS = [
    (1.75, 2.25),                    # T-2h ± 15 min
    (3.75, 4.25),                    # T-4h ± 15 min
]

# Cron-tick interval (informatief; de daadwerkelijke cron-config zit in
# .github/workflows/sports_signal.yml zodra die is opgezet — zie S4).
CRON_INTERVAL_MINUTES = 30


# ── Discord webhook ───────────────────────────────────────────────────────────
# Niet actief gebruikt in S0/S1 — Discord-integratie volgt in S4.
# Verwijst naar GitHub-secret `DISCORD_WEBHOOK_URL_SPORTS`.
DISCORD_WEBHOOK_ENV_VAR = "DISCORD_WEBHOOK_URL_SPORTS"


# ── Data-bronnen (placeholder, definitief gekozen in S1) ──────────────────────
# Onderzocht in S1 (zie docs/sports/ROADMAP.md):
#   - handball.net (officieel HBL)
#   - sofascore.com API
#   - kicker.de (scrape)
# Closing odds: Pinnacle (referentie), Bet365 (alternatief).
DATA_SOURCE_RESULTS = None           # in S1 ingevuld
DATA_SOURCE_ODDS    = None           # in S1 ingevuld


# ── Backtest-instellingen ─────────────────────────────────────────────────────
# Walk-forward met seizoens-grenzen als folds. Primary metric: Closing Line
# Value (CLV) — kunnen we Pinnacle's closing line consistent verslaan?
WALKFORWARD_FOLD_BY    = "season"    # alternatief: "month"
PRIMARY_METRIC         = "clv"       # alternatieven: "roi", "yield", "log_loss"
MIN_TRADES_PER_FOLD    = 30          # < 30 trades = niet genoeg om CLV te
                                     # evalueren; fold wordt overgeslagen
