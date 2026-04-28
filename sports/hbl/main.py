"""
HBL Signal Model — Hoofdpipeline (S0 scaffold)

Volgt hetzelfde fase-runner-patroon als `crypto/main.py`. In S0 zijn alle
fases nog stubs die NotImplementedError raisen — implementatie volgt
sprint per sprint (zie docs/sports/ROADMAP.md).

Gebruik:
  python sports/hbl/main.py --phase data         # match results + line-ups (S1)
  python sports/hbl/main.py --phase odds         # closing + live odds      (S1)
  python sports/hbl/main.py --phase features     # Elo, form, H2H, rest     (S2)
  python sports/hbl/main.py --phase model        # Skellam baseline train   (S3)
  python sports/hbl/main.py --phase backtest     # walk-forward + CLV       (S3)
  python sports/hbl/main.py --phase signal       # pre-match signaal-window (S4)
  python sports/hbl/main.py --phase live_alert   # Discord + paper trade    (S4)
"""

import argparse


# ── Helpers ───────────────────────────────────────────────────────────────────


def _print_header(phase_label: str) -> None:
    """Print een fase-banner — zelfde stijl als crypto/main.py voor consistency."""
    print("=" * 60)
    print(f"HBL — {phase_label}")
    print("=" * 60)


# ── Fases (alle nog stubs, fillen we in S1+) ──────────────────────────────────


def fase_data() -> None:
    _print_header("FASE 1 — Data Verzameling (matches + line-ups)")
    raise NotImplementedError(
        "S1: implementeer match-results-fetcher voor HBL. "
        "Bron-keuze (handball.net / Sofascore / Kicker-scrape) in docs/sports/ROADMAP.md."
    )


def fase_odds() -> None:
    _print_header("FASE 2 — Odds (closing voor backtest, live voor inference)")
    raise NotImplementedError(
        "S1: closing odds van Pinnacle (referentie) + Bet365 (alternatief). "
        "Live odds via dezelfde bron, polling-interval afhankelijk van timing-window."
    )


def fase_features() -> None:
    _print_header("FASE 3 — Feature Engineering")
    raise NotImplementedError(
        "S2: Elo-rating, recent form (last-5 goal diff), home advantage, "
        "rest days, head-to-head. Zie docs/sports/ROADMAP.md voor designkeuzes."
    )


def fase_model() -> None:
    _print_header("FASE 4 — Skellam Baseline Training")
    raise NotImplementedError(
        "S3: train Skellam-distributie (Poisson home − Poisson away). "
        "Per team λ schatten op trainset, home advantage als multiplier."
    )


def fase_backtest() -> None:
    _print_header("FASE 5 — Walk-Forward Backtest (CLV-metric)")
    raise NotImplementedError(
        "S3: walk-forward over seizoens-folds. Primary: Closing Line Value "
        "vs Pinnacle. Aanvullend: ROI, yield, log-loss."
    )


def fase_signal() -> None:
    _print_header("FASE 6 — Pre-match Signaal (window-check)")
    raise NotImplementedError(
        "S4: bepaal welke matches in SIGNAL_WINDOWS_HOURS (T-2h, T-4h) vallen, "
        "genereer signaal per match, schrijf naar latest_signal.json."
    )


def fase_live_alert() -> None:
    _print_header("FASE 7 — Live Alert (Discord + paper trade)")
    raise NotImplementedError(
        "S4: Discord-post via shared.notifier (env DISCORD_WEBHOOK_URL_SPORTS), "
        "paper-trade tracking via shared.paper_state."
    )


# ── Entry point ───────────────────────────────────────────────────────────────


PHASES = {
    "data":       fase_data,
    "odds":       fase_odds,
    "features":   fase_features,
    "model":      fase_model,
    "backtest":   fase_backtest,
    "signal":     fase_signal,
    "live_alert": fase_live_alert,
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="HBL Signal Model — Skellam-baseline pipeline (S0 scaffold)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--phase",
        required=True,
        choices=list(PHASES.keys()),
        help="Welke fase uitvoeren (in S0 raisen alle fases NotImplementedError)",
    )
    args = parser.parse_args()
    PHASES[args.phase]()


if __name__ == "__main__":
    main()
