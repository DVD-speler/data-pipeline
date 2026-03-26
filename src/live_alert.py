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
from datetime import timezone
from pathlib import Path

import pandas as pd
import requests

import config
import config_daily as cfg_daily
from src.backtest import generate_live_signal, load_exit_proba
from src.data_fetcher import load_ohlcv
from src.stats import compute_direction_bias, compute_p1_heatmap


# ── P2: Daily model regime gate ───────────────────────────────────────────────

def _load_daily_regime(symbol: str) -> dict:
    """
    Laad het meest recente dagelijkse regime signaal.

    Retourneert een dict met:
      is_bear     : bool — dagmodel ziet bearmarkt (proba < 0.40)
      proba       : float — dagelijkse stijgingskans
      richting    : str — "BULLISH" / "BEARISH" / "NEUTRAAL" / "ONBEKEND"
      is_fresh    : bool — signaal < 30 uur oud (anders stale = negeren)

    Als het bestand niet bestaat of verouderd is, wordt een neutraal
    resultaat (is_bear=False) teruggegeven zodat de 1h-bot gewoon doorloopt.
    """
    path = cfg_daily.symbol_path_daily(symbol, "latest_signal.json")
    neutral = {"is_bear": False, "proba": 0.5, "richting": "ONBEKEND", "is_fresh": False}

    if not path.exists():
        return neutral

    try:
        with open(path) as f:
            data = json.load(f)

        tijdstip = pd.Timestamp(data.get("tijdstip", ""))
        if tijdstip.tzinfo is None:
            tijdstip = tijdstip.tz_localize("UTC")
        now = pd.Timestamp.now(tz=timezone.utc)
        leeftijd_uur = (now - tijdstip).total_seconds() / 3600

        is_fresh = leeftijd_uur < 30   # max 30 uur oud

        proba    = float(data.get("kans_stijging", 0.5))
        richting = data.get("richting", data.get("signaal", "ONBEKEND"))
        is_bear  = (proba < 0.40) and is_fresh

        return {"is_bear": is_bear, "proba": proba, "richting": richting, "is_fresh": is_fresh}
    except Exception:
        return neutral

# ── State management ────────────────────────────────────────────────────────────

def load_paper_state(path) -> dict:
    """Laad paper trading state, of maak een lege state aan bij eerste run."""
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
        "capital": 1000.0,
        "last_checked": None,
    }


def save_paper_state(state: dict, path) -> None:
    """Sla paper trading state op als JSON."""
    with open(path, "w") as f:
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


def send_alert(content: str) -> None:
    """Stuur alert naar Discord."""
    send_discord_alert(content)


# ── Positie management ──────────────────────────────────────────────────────────

def check_position_exit(
    pos: dict,
    latest_close: float,
    latest_high: float,
    latest_low: float,
    latest_ts,
    current_proba: float = None,
    exit_proba_long: float = None,
    exit_proba_short: float = None,
) -> dict | None:
    """
    Controleer of de open positie gesloten moet worden.

    Prioriteit:
      1. SL geraakt  → harde stop, altijd
      2. TP geraakt  → winstdoel bereikt
      3. Model-exit  → proba zakt onder exit_proba_long (LONG) of stijgt boven
                       exit_proba_short (SHORT) — model ziet kans niet meer
      4. 168h vangnet → absolute maximale houdtijd (1 week)

    Returns exit-info dict als de positie gesloten wordt, anders None.
    """
    if exit_proba_long is None:
        exit_proba_long = config.EXIT_PROBA_LONG
    if exit_proba_short is None:
        exit_proba_short = config.EXIT_PROBA_SHORT

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
        if current_proba is not None and current_proba < exit_proba_long:
            return {"exit_price": latest_close,
                    "exit_reason": f"Model ↓ ({current_proba:.1%}<{exit_proba_long:.0%})",
                    "exit_time": str(latest_ts)}
    else:  # SHORT
        if latest_high >= sl_price:
            return {"exit_price": sl_price, "exit_reason": "SL ✗", "exit_time": str(latest_ts)}
        if latest_low <= tp_price:
            return {"exit_price": tp_price, "exit_reason": "TP ✓", "exit_time": str(latest_ts)}
        if current_proba is not None and current_proba > exit_proba_short:
            return {"exit_price": latest_close,
                    "exit_reason": f"Model ↑ ({current_proba:.1%}>{exit_proba_short:.0%})",
                    "exit_time": str(latest_ts)}

    # 168h tijdsvangnet → sluit op huidige close
    if now >= horizon_end:
        return {"exit_price": latest_close, "exit_reason": "168h ⏰", "exit_time": str(latest_ts)}

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
    symbol: str = config.SYMBOL,
) -> None:
    """
    Voer één check-cyclus uit:
    1. Laad data + genereer signaal
    2. Check open positie (SL/TP/horizon)
    3. Open nieuwe positie bij signaal
    4. Stuur Discord alerts
    5. Sla state op
    """
    paper_trades_path = config.symbol_path(symbol, "paper_trades.json")

    print("\n── Live Alert ──────────────────────────────────────────────────────────")

    # ── Laad data ─────────────────────────────────────────────────────────────
    df = load_ohlcv(symbol=symbol)
    p1p2 = pd.read_csv(config.symbol_path(symbol, "p1p2_labels.csv"))
    p1_heatmap = compute_p1_heatmap(p1p2)
    direction_bias = compute_direction_bias(p1p2)

    # ── Genereer live signaal ─────────────────────────────────────────────────
    signaal = generate_live_signal(df, p1p2, p1_heatmap, direction_bias, symbol=symbol)

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
    state = load_paper_state(paper_trades_path)
    had_open_pos = state["open_position"] is not None   # bewaar vóór eventuele sluiting
    print(f"  Kapitaal : €{state['capital']:,.2f}")

    # ── Laad geoptimaliseerde exit-drempelwaarden ─────────────────────────────
    exit_proba_long, exit_proba_short = load_exit_proba(symbol=symbol)

    # ── Check open positie ────────────────────────────────────────────────────
    if state["open_position"] is not None:
        pos = state["open_position"]
        current_proba = signaal.get("proba_raw")
        exit_info = check_position_exit(
            pos, latest_close, latest_high, latest_low, latest_ts,
            current_proba=current_proba,
            exit_proba_long=exit_proba_long,
            exit_proba_short=exit_proba_short,
        )

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
                f"{icon} **TRADE GESLOTEN** — {symbol}\n"
                f"📈 **{pos['direction']}** | ${pos['entry_price']:,.0f} → ${exit_info['exit_price']:,.0f}\n"
                f"🎯 Reden: {exit_info['exit_reason']} | Rendement: {sign}{gross_pct:.2f}%\n"
                f"💰 P&L: {sign}€{pnl_euro:.2f} | Kapitaal: €{state['capital']:,.2f}"
            )
            send_alert(msg)
        else:
            print(f"\n  Positie open: {pos['direction']} ${pos['entry_price']:,.0f} "
                  f"| SL ${pos['sl_price']:,.0f} | TP ${pos['tp_price']:,.0f}")

    # ── P2: Daily model regime gate ───────────────────────────────────────────
    # Wanneer het dagmodel BEARISH is (proba < 0.40), blokkeren we alle nieuwe longs.
    # De dagelijkse trend is een trager, stabieler signaal dan de 1h-indicatoren:
    # als de dagtrend negatief is, zijn 1h-longs statistisch minder betrouwbaar.
    daily_regime = _load_daily_regime(symbol)
    if daily_regime["is_bear"] and signaal["signaal"] == "LONG":
        print(
            f"\n  ⛔ Daily regime gate: LONG geblokkeerd\n"
            f"     Dagmodel: {daily_regime['richting']} "
            f"(proba {daily_regime['proba']:.1%} < 40%) — bearmarkt gedetecteerd"
        )
        signaal["signaal"] = f"WACHT (daily regime bear — proba {daily_regime['proba']:.1%})"

    # ── Nieuw signaal ─────────────────────────────────────────────────────────
    opened_new_position = False
    if state["open_position"] is None and signaal["signaal"] in ("LONG", "SHORT"):
        direction = signaal["signaal"]
        # Gebruik de actuele prijs/tijdstip van df.index[-1], NIET signaal["prijs"].
        # features.iloc[-1] ligt ~24 uur in het verleden (dropna verwijdert de laatste
        # PREDICTION_HORIZON_H rijen waarvan future_close NaN is). Als we signaal["prijs"]
        # als entry zouden gebruiken, is de prijs stale én is horizon_end ≈ nu, waardoor
        # de positie al bij de volgende run sluit (1 uur later).
        entry_price = latest_close
        signal_ts = pd.Timestamp(latest_ts)
        capital = state["capital"]
        position_size = capital * risk_pct / sl_pct

        if direction == "LONG":
            sl_price = entry_price * (1 - sl_pct)
            tp_price = entry_price * (1 + tp_pct)
        else:
            sl_price = entry_price * (1 + sl_pct)
            tp_price = entry_price * (1 - tp_pct)

        horizon_end = signal_ts + pd.Timedelta(hours=config.MAX_HOLD_HOURS)

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

        coin_name = symbol.replace("USDT", "")
        coin_amount = round(position_size / entry_price, 6)
        icon = "🟢" if direction == "LONG" else "🔴"
        regime_label = "boven EMA200" if signaal.get("regime_boven_ema200") else "onder EMA200"
        msg = (
            f"{icon} **{direction} SIGNAAL** — {symbol}\n"
            f"⏰ {pd.Timestamp(latest_ts).strftime('%d-%m-%Y %H:%M')} UTC\n"
            f"💰 Entry: ${entry_price:,.0f} | SL: ${sl_price:,.0f} (−{sl_pct*100:.0f}%) | "
            f"TP: ${tp_price:,.0f} (+{tp_pct*100:.0f}%)\n"
            f"📊 Proba: {signaal['kans_stijging']} | Regime: {regime_label}\n"
            f"💼 Positie: ${position_size:,.0f} ({coin_amount:.6f} {coin_name}) | Risico: €{capital*risk_pct:.2f} ({risk_pct*100:.0f}% kapitaal)"
        )
        opened_new_position = True
        send_discord_alert(msg)

    # ── WACHT bericht (geen nieuwe trade geopend deze run) ────────────────────
    if not opened_new_position:
        ts_str = pd.Timestamp(latest_ts).strftime("%d-%m-%Y %H:%M")
        sig_label = signaal["signaal"]

        if had_open_pos and state["open_position"] is not None:
            # Positie loopt nog — nieuw signaal (indien aanwezig) genegeerd
            pos = state["open_position"]
            hours_open = round(
                (pd.Timestamp(latest_ts) - pd.Timestamp(pos["open_time"])).total_seconds() / 3600, 1
            )
            reden = (
                f"Al een **{pos['direction']}** positie open "
                f"({hours_open}h geleden, entry ${pos['entry_price']:,.0f})\n"
                f"📍 SL: ${pos['sl_price']:,.0f} | TP: ${pos['tp_price']:,.0f} | "
                f"Vangnet: {pd.Timestamp(pos['horizon_end']).strftime('%d-%m %H:%M')} UTC (168h)"
            )
        elif "death cross" in sig_label.lower() or "EMA50" in sig_label:
            reden = (
                f"**Death cross actief** (EMA50 onder EMA200) → long geblokkeerd\n"
                f"📊 Proba: {signaal['kans_stijging']} | Drempel: {signaal['long_threshold']} "
                f"| Regime: {signaal.get('market_regime', '?')}"
            )
        elif "EMA200" in sig_label:
            reden = (
                f"Proba hoog genoeg ({signaal['kans_stijging']} ≥ {signaal['long_threshold']}) "
                f"maar prijs **onder EMA200** → long geblokkeerd door regime-filter"
            )
        else:
            regime = signaal.get("market_regime", "")
            base_thr = signaal.get("long_threshold_base", signaal["long_threshold"])
            regime_info = f" (regime: {regime}, offset toegepast)" if regime else ""
            reden = (
                f"Proba te laag voor signaal: **{signaal['kans_stijging']}** "
                f"(drempel: {signaal['long_threshold']}{regime_info})"
            )

        wacht_msg = (
            f"⏸️ **WACHT** — {symbol}\n"
            f"⏰ {ts_str} UTC | Prijs: ${latest_close:,.0f}\n"
            f"💡 {reden}"
        )
        send_alert(wacht_msg)

    # ── Opslaan ───────────────────────────────────────────────────────────────
    state["last_checked"] = str(latest_ts)
    save_paper_state(state, paper_trades_path)

    # Sla ook latest_signal.json op (voor workflow commit en debugging)
    import json as _json
    with open(config.symbol_path(symbol, "latest_signal.json"), "w") as f:
        _json.dump(signaal, f, indent=2, default=str)

    print(f"\n  State opgeslagen: {paper_trades_path}")
    print("── Klaar ───────────────────────────────────────────────────────────────")
