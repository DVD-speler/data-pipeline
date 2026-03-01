"""
Live Alert Pipeline — Fase 5

Checkt elke run:
  1. Open positie → SL / TP / horizon bereikt?
  2. Nieuw signaal? → open positie + Discord alert

Paper trading state wordt bijgehouden in data/paper_trades.json.
Discord alerts via webhook URL in omgevingsvariabele DISCORD_WEBHOOK_URL.

Gebruik:
  python main.py --phase live_alert
  of lokaal testen:
  DISCORD_WEBHOOK_URL=https://... python main.py --phase live_alert
"""

import json
import os
from pathlib import Path

import pandas as pd
import requests

import config
from src.backtest import generate_live_signal
from src.data_fetcher import load_ohlcv
from src.stats import compute_direction_bias, compute_p1_heatmap

PAPER_TRADES_PATH = config.DATA_DIR / "paper_trades.json"


# ── State management ────────────────────────────────────────────────────────────

def load_paper_state() -> dict:
    """Laad paper trading state, of maak een lege state aan bij eerste run."""
    if PAPER_TRADES_PATH.exists():
        with open(PAPER_TRADES_PATH) as f:
            return json.load(f)
    return {
        "open_position": None,
        "closed_trades": [],
        "capital": 1000.0,
        "last_checked": None,
    }


def save_paper_state(state: dict) -> None:
    """Sla paper trading state op als JSON."""
    with open(PAPER_TRADES_PATH, "w") as f:
        json.dump(state, f, indent=2, default=str)


# ── Discord ─────────────────────────────────────────────────────────────────────

def send_discord_alert(content: str) -> None:
    """Stuur een bericht naar Discord via webhook URL uit omgevingsvariabele."""
    webhook_url = os.environ.get("DISCORD_WEBHOOK_URL")
    if not webhook_url:
        print("  [Discord] DISCORD_WEBHOOK_URL niet ingesteld — alert overgeslagen.")
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


# ── Positie management ──────────────────────────────────────────────────────────

def check_position_exit(
    pos: dict,
    latest_close: float,
    latest_high: float,
    latest_low: float,
    latest_ts,
) -> dict | None:
    """
    Controleer of de open positie geraakt is door SL, TP of het tijdshorizon.

    Returns exit-info dict als de positie gesloten wordt, anders None.
    """
    sl_price = pos["sl_price"]
    tp_price = pos["tp_price"]
    direction = pos["direction"]
    horizon_end = pd.Timestamp(pos["horizon_end"])
    now = pd.Timestamp(latest_ts)

    if direction == "LONG":
        if latest_low <= sl_price:
            return {"exit_price": sl_price, "exit_reason": "SL ✗", "exit_time": str(latest_ts)}
        if latest_high >= tp_price:
            return {"exit_price": tp_price, "exit_reason": "TP ✓", "exit_time": str(latest_ts)}
    else:  # SHORT
        if latest_high >= sl_price:
            return {"exit_price": sl_price, "exit_reason": "SL ✗", "exit_time": str(latest_ts)}
        if latest_low <= tp_price:
            return {"exit_price": tp_price, "exit_reason": "TP ✓", "exit_time": str(latest_ts)}

    # Horizon verlopen → sluit op huidige close
    if now >= horizon_end:
        return {"exit_price": latest_close, "exit_reason": "24h ⏰", "exit_time": str(latest_ts)}

    return None


def _close_position(state: dict, pos: dict, exit_info: dict) -> float:
    """
    Verwerk een gesloten positie: bereken P&L, update kapitaal en trade log.
    Retourneert de P&L in euro's.
    """
    entry = pos["entry_price"]
    exit_price = exit_info["exit_price"]
    pos_size = pos["position_size"]
    direction = pos["direction"]

    if direction == "LONG":
        gross_ret = (exit_price - entry) / entry
    else:
        gross_ret = (entry - exit_price) / entry

    pnl_euro = pos_size * gross_ret - pos_size * 2 * config.TRADE_FEE
    state["capital"] = round(state["capital"] + pnl_euro, 2)

    state["closed_trades"].append({
        **pos,
        **exit_info,
        "gross_return_pct": round(gross_ret * 100, 2),
        "pnl_euro": round(pnl_euro, 2),
        "capital_after": state["capital"],
    })
    state["open_position"] = None
    return pnl_euro


# ── Hoofdfunctie ────────────────────────────────────────────────────────────────

def run_live_alert(
    risk_pct: float = 0.01,
    sl_pct: float = 0.02,
    tp_pct: float = 0.06,
) -> None:
    """
    Voer één check-cyclus uit:
    1. Laad data + genereer signaal
    2. Check open positie (SL/TP/horizon)
    3. Open nieuwe positie bij signaal
    4. Stuur Discord alerts
    5. Sla state op
    """
    print("\n── Live Alert ──────────────────────────────────────────────────────────")

    # ── Laad data ─────────────────────────────────────────────────────────────
    df = load_ohlcv()
    p1p2 = pd.read_csv(config.DATA_DIR / "p1p2_labels.csv")
    p1_heatmap = compute_p1_heatmap(p1p2)
    direction_bias = compute_direction_bias(p1p2)

    # ── Genereer live signaal ─────────────────────────────────────────────────
    signaal = generate_live_signal(df, p1p2, p1_heatmap, direction_bias)

    latest_ts = df.index[-1]
    latest_row = df.iloc[-1]
    latest_close = float(latest_row["close"])
    latest_high = float(latest_row.get("high", latest_close))
    latest_low = float(latest_row.get("low", latest_close))

    print(f"  Tijdstip : {signaal['tijdstip']}")
    print(f"  Signaal  : {signaal['signaal']}")
    print(f"  Proba    : {signaal['kans_stijging']}")
    print(f"  Prijs    : ${signaal['prijs']:,.0f}")

    # ── Paper trading state ───────────────────────────────────────────────────
    state = load_paper_state()
    print(f"  Kapitaal : €{state['capital']:,.2f}")

    # ── Check open positie ────────────────────────────────────────────────────
    if state["open_position"] is not None:
        pos = state["open_position"]
        exit_info = check_position_exit(pos, latest_close, latest_high, latest_low, latest_ts)

        if exit_info:
            pnl_euro = _close_position(state, pos, exit_info)
            sign = "+" if pnl_euro >= 0 else ""
            icon = "✅" if pnl_euro > 0 else "❌"

            print(f"\n  ── TRADE GESLOTEN ──────────────────────────────────────────────────")
            print(f"  {pos['direction']} | ${pos['entry_price']:,.0f} → ${exit_info['exit_price']:,.0f}")
            print(f"  Reden: {exit_info['exit_reason']} | P&L: {sign}€{pnl_euro:.2f}")
            print(f"  Nieuw kapitaal: €{state['capital']:,.2f}")

            gross_pct = state["closed_trades"][-1]["gross_return_pct"]
            msg = (
                f"{icon} **TRADE GESLOTEN** — BTCUSDT\n"
                f"📈 **{pos['direction']}** | ${pos['entry_price']:,.0f} → ${exit_info['exit_price']:,.0f}\n"
                f"🎯 Reden: {exit_info['exit_reason']} | Rendement: {sign}{gross_pct:.2f}%\n"
                f"💰 P&L: {sign}€{pnl_euro:.2f} | Kapitaal: €{state['capital']:,.2f}"
            )
            send_discord_alert(msg)
        else:
            print(f"\n  Positie open: {pos['direction']} ${pos['entry_price']:,.0f} "
                  f"| SL ${pos['sl_price']:,.0f} | TP ${pos['tp_price']:,.0f}")

    # ── Nieuw signaal ─────────────────────────────────────────────────────────
    if state["open_position"] is None and signaal["signaal"] != "WACHT":
        direction = signaal["signaal"]
        # Gebruik de prijs en tijdstip van het signaal (features-laatste-rij),
        # niet df.index[-1] — die kan later liggen door dead zone filtering.
        entry_price = signaal["prijs"]
        signal_ts = pd.Timestamp(signaal["tijdstip"])
        capital = state["capital"]
        position_size = capital * risk_pct / sl_pct

        if direction == "LONG":
            sl_price = entry_price * (1 - sl_pct)
            tp_price = entry_price * (1 + tp_pct)
        else:
            sl_price = entry_price * (1 + sl_pct)
            tp_price = entry_price * (1 - tp_pct)

        horizon_end = signal_ts + pd.Timedelta(hours=config.PREDICTION_HORIZON_H)

        state["open_position"] = {
            "direction": direction,
            "entry_price": round(entry_price, 0),
            "sl_price": round(sl_price, 0),
            "tp_price": round(tp_price, 0),
            "open_time": str(signal_ts),
            "horizon_end": str(horizon_end),
            "position_size": round(position_size, 2),
            "risk_euro": round(capital * risk_pct, 2),
            "proba": signaal["kans_stijging"],
        }

        print(f"\n  ── NIEUW SIGNAAL ───────────────────────────────────────────────────")
        print(f"  {direction} | Entry ${entry_price:,.0f} | SL ${sl_price:,.0f} | TP ${tp_price:,.0f}")
        print(f"  Positie: ${position_size:,.0f} | Risico: €{capital*risk_pct:.2f}")

        icon = "🟢" if direction == "LONG" else "🔴"
        regime_label = "boven EMA200" if signaal.get("regime_boven_ema200") else "onder EMA200"
        msg = (
            f"{icon} **{direction} SIGNAAL** — BTCUSDT\n"
            f"⏰ {pd.Timestamp(latest_ts).strftime('%d-%m-%Y %H:%M')} UTC\n"
            f"💰 Entry: ${entry_price:,.0f} | SL: ${sl_price:,.0f} (−{sl_pct*100:.0f}%) | "
            f"TP: ${tp_price:,.0f} (+{tp_pct*100:.0f}%)\n"
            f"📊 Proba: {signaal['kans_stijging']} | Regime: {regime_label}\n"
            f"💼 Positie: ${position_size:,.0f} | Risico: €{capital*risk_pct:.2f} ({risk_pct*100:.0f}% kapitaal)"
        )
        send_discord_alert(msg)

    # ── Opslaan ───────────────────────────────────────────────────────────────
    state["last_checked"] = str(latest_ts)
    save_paper_state(state)
    print(f"\n  State opgeslagen: {PAPER_TRADES_PATH}")
    print("── Klaar ───────────────────────────────────────────────────────────────")
