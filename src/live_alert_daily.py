"""
Dagelijks Signaal Pipeline — richtingsindicator (geen paper trading)

Stuurt elke dag een Discord-bericht met:
  - Verwachte richting: BULLISH / BEARISH / NEUTRAAL
  - Zekerheid (model proba)
  - Marktregime + key filters (death cross, EMA200)

Geen posities, geen SL/TP, geen paper trades.
Discord alerts via DISCORD_WEBHOOK_URL_DAILY (apart channel van uurmodel).

Gebruik:
  python main.py --phase live_alert_daily
  of lokaal testen:
  DISCORD_WEBHOOK_URL_DAILY=https://... python main.py --phase live_alert_daily
"""

import json
import os

import joblib
import pandas as pd
import requests

import config_daily as cfg
from src.data_fetcher import load_ohlcv
from src.features_daily import build_features_daily


# ── Discord ──────────────────────────────────────────────────────────────────


def send_discord_alert(content: str) -> None:
    """Stuur bericht naar het dagelijkse Discord channel (DISCORD_WEBHOOK_URL_DAILY)."""
    webhook_url = os.environ.get("DISCORD_WEBHOOK_URL_DAILY")
    if not webhook_url:
        print("  [Discord] DISCORD_WEBHOOK_URL_DAILY niet ingesteld — alert overgeslagen.")
        return
    try:
        resp = requests.post(webhook_url, json={"content": content}, timeout=10)
        if resp.status_code in (200, 204):
            print("  [Discord] Dagelijks alert verstuurd.")
        else:
            print(f"  [Discord] Fout {resp.status_code}: {resp.text[:200]}")
    except requests.RequestException as e:
        print(f"  [Discord] Verbindingsfout: {e}")


def send_alert(content: str) -> None:
    """Stuur alert naar Discord."""
    send_discord_alert(content)


# ── Signaal generatie ────────────────────────────────────────────────────────


def _generate_daily_signal(df_feat: pd.DataFrame, symbol: str) -> dict:
    """Genereer dagelijks richtingssignaal op basis van de meest recente dagcandle."""
    model_path = cfg.symbol_path_daily(symbol, "model.pkl")
    if not model_path.exists():
        return {
            "signaal":       "GEEN MODEL",
            "richting":      "NEUTRAAL",
            "kans_stijging": 0.5,
            "prijs":         0.0,
            "tijdstip":      "n/a",
        }

    model = joblib.load(model_path)

    last_row     = df_feat.iloc[[-1]]
    feature_cols = [c for c in cfg.FEATURE_COLS_DAILY if c in last_row.columns]
    proba        = float(model.predict_proba(last_row[feature_cols])[:, 1][0])

    market_regime   = float(last_row["market_regime"].iloc[0]) if "market_regime" in last_row else 0.0
    close_price     = float(last_row["close"].iloc[0])
    ts              = last_row.index[-1]
    price_vs_ema200 = float(last_row["price_vs_ema200"].iloc[0]) if "price_vs_ema200" in last_row else 1.0
    ema_ratio_50    = float(last_row["ema_ratio_50"].iloc[0])    if "ema_ratio_50"    in last_row else 1.0

    # Death cross: EMA50 < EMA200
    death_cross = ema_ratio_50 > price_vs_ema200

    # Richting bepalen op basis van ruwe proba (geen threshold — toon altijd iets)
    # >55%: BULLISH, <45%: BEARISH, anders: NEUTRAAL
    if proba >= 0.55:
        richting = "BULLISH"
    elif proba <= 0.45:
        richting = "BEARISH"
    else:
        richting = "NEUTRAAL"

    regime_labels = {1: "bull (ADX+)", 0: "ranging", -1: "bear (ADX-)"}
    regime_label  = regime_labels.get(int(market_regime), "onbekend")

    return {
        "signaal":          richting,
        "richting":         richting,
        "kans_stijging":    proba,
        "prijs":            close_price,
        "tijdstip":         str(ts),
        "market_regime":    market_regime,
        "regime_label":     regime_label,
        "death_cross":      death_cross,
        "boven_ema200":     price_vs_ema200 > 1.0,
    }


# ── Hoofdfunctie ─────────────────────────────────────────────────────────────


def run_live_alert_daily(symbol: str = cfg.SYMBOL) -> None:
    """
    Dagelijkse richtingsindicator:
    1. Laad 1d OHLCV + bouw dagelijkse features
    2. Genereer richtingssignaal via dagmodel
    3. Stuur Discord alert naar dagelijks channel
    4. Sla latest_signal.json op
    """
    print("\n── Dagelijks Signaal ────────────────────────────────────────────────────")

    # ── Laad data en bouw features ────────────────────────────────────────────
    df      = load_ohlcv(symbol=symbol, interval="1d")
    df_feat = build_features_daily(df, symbol=symbol)

    # ── Genereer signaal ──────────────────────────────────────────────────────
    signaal = _generate_daily_signal(df_feat, symbol=symbol)

    richting    = signaal["richting"]
    proba       = signaal["kans_stijging"]
    prijs       = signaal["prijs"]
    ts_str      = pd.Timestamp(signaal["tijdstip"]).strftime("%d-%m-%Y")
    regime      = signaal["regime_label"]
    death_cross = signaal["death_cross"]
    boven_ema200 = signaal["boven_ema200"]

    print(f"  Datum    : {ts_str}")
    print(f"  Richting : {richting}")
    print(f"  Proba    : {proba:.1%}")
    print(f"  Prijs    : ${prijs:,.0f}")
    print(f"  Regime   : {regime}")

    # ── Discord bericht ───────────────────────────────────────────────────────
    if richting == "BULLISH":
        icon    = "🟢"
        zekerheid_bar = "▓" * int(proba * 10) + "░" * (10 - int(proba * 10))
    elif richting == "BEARISH":
        icon    = "🔴"
        zekerheid_bar = "▓" * int((1 - proba) * 10) + "░" * (10 - int((1 - proba) * 10))
    else:
        icon    = "⚪"
        zekerheid_bar = "▒▒▒▒▒▒▒▒▒▒"

    zekerheid_pct = proba * 100 if richting == "BULLISH" else (1 - proba) * 100

    filters = []
    if death_cross:
        filters.append("⚠️ Death cross (EMA50 < EMA200)")
    if not boven_ema200:
        filters.append("📉 Prijs onder EMA200")
    filter_str = "\n".join(filters) if filters else "✅ Geen filters actief"

    msg = (
        f"{icon} **DAGELIJKS SIGNAAL** — {symbol}\n"
        f"📆 {ts_str} | Prijs: ${prijs:,.0f}\n"
        f"**{richting}** — {zekerheid_pct:.0f}% zekerheid\n"
        f"{zekerheid_bar}\n"
        f"📊 Regime: {regime} | Proba: {proba:.1%}\n"
        f"{filter_str}"
    )
    send_alert(msg)

    # ── Opslaan ───────────────────────────────────────────────────────────────
    out_path = cfg.symbol_path_daily(symbol, "latest_signal.json")
    with open(out_path, "w") as f:
        json.dump(signaal, f, indent=2, default=str)

    print(f"\n  Signaal opgeslagen: {out_path}")
    print("── Klaar ────────────────────────────────────────────────────────────────")
