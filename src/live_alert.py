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

import numpy as np
import pandas as pd
import requests

import config
import config_daily as cfg_daily
from src.backtest import generate_live_signal, load_exit_proba
from src.data_fetcher import load_ohlcv
from src.levels import precompute_swings, get_recent_levels, compute_structural_sl_tp
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

# ── Kelly + Regime helpers ───────────────────────────────────────────────────────

def _load_kelly(symbol: str) -> dict:
    """Laad Kelly-fractie uit {symbol}_kelly.json. Geeft lege dict terug als niet gevonden."""
    path = config.symbol_path(symbol, "kelly.json")
    if path.exists():
        try:
            with open(path) as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def _regime_sl_tp(regime_label: str, sl_pct_default: float, tp_pct_default: float) -> tuple[float, float]:
    """
    Geef regime-afhankelijke SL/TP terug (T4-C).
    Gebruikt REGIME_SL_TP uit config als die beschikbaar is, anders fallback op defaults.
    """
    table = getattr(config, "REGIME_SL_TP", {})
    params = table.get(regime_label, {})
    sl = params.get("sl_pct", sl_pct_default)
    tp = params.get("tp_pct", tp_pct_default)
    return float(sl), float(tp)


def _check_corr_guard(symbol: str, df: pd.DataFrame) -> tuple[bool, float, str]:
    """
    Controleer of een open positie in een ander symbool het openen blokkeert (T4-A).

    Returns (blocked: bool, correlation: float, other_symbol: str)
    """
    for other in config.SYMBOLS:
        if other == symbol:
            continue
        other_path = config.symbol_path(other, "paper_trades.json")
        if not other_path.exists():
            continue
        try:
            with open(other_path) as f:
                other_state = json.load(f)
        except Exception:
            continue
        if other_state.get("open_position") is None:
            continue

        # Open positie in ander symbool — bereken 24h correlatie
        try:
            other_df = load_ohlcv(symbol=other)
            combined = pd.concat([
                df["close"].rename("a"),
                other_df["close"].rename("b"),
            ], axis=1).dropna()
            if len(combined) < 24:
                continue
            ret_a = combined["a"].pct_change().tail(24)
            ret_b = combined["b"].pct_change().tail(24)
            corr  = float(ret_a.corr(ret_b))

            block_thr = getattr(config, "CORR_GUARD_THRESHOLD_BLOCK", 0.90)
            halve_thr = getattr(config, "CORR_GUARD_THRESHOLD_HALVE", 0.70)

            if corr > block_thr:
                return True, corr, other
            if corr > halve_thr:
                return False, corr, other   # gebeld; beller halveert positie
        except Exception:
            continue

    return False, 0.0, ""


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
    exit_proba_short: float = None,  # legacy, genegeerd
) -> dict | None:
    """
    Controleer of de open LONG-positie gesloten moet worden.

    Prioriteit:
      1. SL geraakt  → harde stop, altijd
      2. TP geraakt  → winstdoel bereikt
      3. Model-exit  → proba zakt onder exit_proba_long — model ziet kans niet meer
      4. 168h vangnet → absolute maximale houdtijd (1 week)

    Returns exit-info dict als de positie gesloten wordt, anders None.
    """
    if exit_proba_long is None:
        exit_proba_long = config.EXIT_PROBA_LONG

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
            hours_open = round(
                (pd.Timestamp(exit_info["exit_time"]) - pd.Timestamp(pos["open_time"])).total_seconds() / 3600, 1
            )
            exit_reason = exit_info["exit_reason"]
            if "SL" in exit_reason:
                exit_explain = f"Stop-loss geraakt — prijs daalde tot ${exit_info['exit_price']:,.0f} (−{abs(pos['entry_price']-exit_info['exit_price'])/pos['entry_price']*100:.1f}%)"
            elif "TP" in exit_reason:
                exit_explain = f"Take-profit bereikt — prijs steeg tot ${exit_info['exit_price']:,.0f} (+{abs(exit_info['exit_price']-pos['entry_price'])/pos['entry_price']*100:.1f}%)"
            elif "Model" in exit_reason:
                exit_explain = f"Model-exit — stijgingskans daalde onder drempelwaarde ({exit_reason})"
            else:
                exit_explain = f"168h tijdsvangnet bereikt — positie gesloten op huidige prijs"

            msg = (
                f"{icon} **TRADE GESLOTEN** — {symbol}\n"
                f"📊 **{pos['direction']}** | Entry: ${pos['entry_price']:,.0f} → Exit: ${exit_info['exit_price']:,.0f}\n"
                f"🎯 **Reden:** {exit_explain}\n"
                f"⏱️ {hours_open:.0f}h open "
                f"(open: {pd.Timestamp(pos['open_time']).strftime('%d-%m %H:%M')}, "
                f"exit: {pd.Timestamp(exit_info['exit_time']).strftime('%d-%m %H:%M')} UTC)\n"
                f"💰 Rendement: {sign}{gross_pct:.2f}% | P&L: {sign}€{pnl_euro:.2f} | Kapitaal: €{state['capital']:,.2f}"
            )
            send_alert(msg)
        else:
            hours_open = (pd.Timestamp(latest_ts) - pd.Timestamp(pos["open_time"])).total_seconds() / 3600
            decay_factor = max(0.5, 1.0 - 0.02 * hours_open)
            print(f"\n  Positie open: {pos['direction']} ${pos['entry_price']:,.0f} "
                  f"| SL ${pos['sl_price']:,.0f} | TP ${pos['tp_price']:,.0f} "
                  f"| {hours_open:.0f}h open | size {decay_factor:.0%}")

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

    # ── Laad high/low data voor structurele SL/TP ─────────────────────────────
    try:
        _ohlcv_raw = load_ohlcv(symbol=symbol, interval="1h")
        _ohlcv_raw = _ohlcv_raw[_ohlcv_raw.index <= latest_ts].tail(500)
        _highs = _ohlcv_raw["high"].values
        _lows  = _ohlcv_raw["low"].values
        _swings = precompute_swings(_highs, _lows)
        _supports, _resistances = get_recent_levels(_swings, len(_highs))
        _use_structural = True
    except Exception as _e:
        print(f"  Waarschuwing: structurele niveaus niet beschikbaar ({_e}) — fallback op vaste pct")
        _use_structural = False
        _supports, _resistances = np.array([]), np.array([])

    # ── Nieuw signaal ─────────────────────────────────────────────────────────
    opened_new_position = False
    if state["open_position"] is None and signaal["signaal"] == "LONG":
        direction = signaal["signaal"]
        # Gebruik de actuele prijs/tijdstip van df.index[-1], NIET signaal["prijs"].
        # features.iloc[-1] ligt ~24 uur in het verleden (dropna verwijdert de laatste
        # PREDICTION_HORIZON_H rijen waarvan future_close NaN is). Als we signaal["prijs"]
        # als entry zouden gebruiken, is de prijs stale én is horizon_end ≈ nu, waardoor
        # de positie al bij de volgende run sluit (1 uur later).
        entry_price = latest_close
        signal_ts   = pd.Timestamp(latest_ts)
        capital     = state["capital"]

        # ── T4-A: Multi-asset correlatie guard ───────────────────────────────
        _corr_blocked, _corr_val, _corr_other = _check_corr_guard(symbol, df)
        if _corr_blocked:
            print(
                f"\n  ⛔ Correlatie guard: LONG geblokkeerd\n"
                f"     {_corr_other} heeft open positie, 24h correlatie {_corr_val:.1%} "
                f"> {getattr(config, 'CORR_GUARD_THRESHOLD_BLOCK', 0.90):.0%} limiet"
            )
            signaal["signaal"] = (
                f"WACHT (correlatie guard — {_corr_other} open, corr {_corr_val:.1%})"
            )
        else:
            # ── T4-C: Regime-afhankelijke SL/TP ─────────────────────────────
            regime_label = signaal.get("market_regime", "ranging")
            sl_pct_eff, tp_pct_eff = _regime_sl_tp(regime_label, sl_pct, tp_pct)
            if sl_pct_eff != sl_pct or tp_pct_eff != tp_pct:
                print(f"  Regime '{regime_label}': SL={sl_pct_eff:.1%}, TP={tp_pct_eff:.1%} "
                      f"(standaard SL={sl_pct:.1%}, TP={tp_pct:.1%})")
            sl_pct = sl_pct_eff
            tp_pct = tp_pct_eff

            position_size = capital * risk_pct / sl_pct

            # ── T2-D: Kelly Criterion positiegrootte cap ─────────────────────
            if getattr(config, "USE_KELLY_SIZING", False):
                kelly_data = _load_kelly(symbol)
                kelly_half = kelly_data.get("kelly_half", 0.0)
                if kelly_half > 0:
                    kelly_max_size = capital * kelly_half
                    if position_size > kelly_max_size:
                        print(f"  Kelly cap: ${position_size:,.0f} → ${kelly_max_size:,.0f} "
                              f"(half-Kelly {kelly_half:.1%} × kapitaal)")
                        position_size = kelly_max_size

            # Halveer positie als er een matig gecorreleerd symbool open staat
            if _corr_val > getattr(config, "CORR_GUARD_THRESHOLD_HALVE", 0.70):
                print(f"  Correlatie halvering: {_corr_other} open, corr {_corr_val:.1%} "
                      f"→ positie gehalveerd (${position_size:,.0f} → ${position_size/2:,.0f})")
                position_size /= 2

            # ── S16-B: Crash-modus positie-halvering ─────────────────────────
            # Flash-crash (>2.5σ candle) of -10% dag → halve positie.
            _crash_factor = getattr(config, "CRASH_SIZE_FACTOR", 0.5)
            if signaal.get("crash_mode") and _crash_factor < 1.0:
                new_size = position_size * _crash_factor
                print(f"  ⚠️  Crash-modus actief: positie gehalveerd "
                      f"(${position_size:,.0f} → ${new_size:,.0f})")
                position_size = new_size

            # ── S17-B: MACD momentum-gewogen positiescaling ───────────────────
            # Sterk positief MACD-momentum → tot 1.5× positiegroottegrootte.
            if getattr(config, "MACD_MOMENTUM_SCALE", False):
                _macd_mult = 1.0 + float(signaal.get("macd_size_mult", 0.0))
                if _macd_mult > 1.005:  # alleen printen bij merkbare verhoging
                    new_size = position_size * _macd_mult
                    print(f"  📈 MACD momentum ×{_macd_mult:.2f}: "
                          f"${position_size:,.0f} → ${new_size:,.0f}")
                    position_size = new_size

            if _use_structural:
                sl_price, tp_price = compute_structural_sl_tp(
                    entry_price, direction, _supports, _resistances,
                    fallback_sl_pct=sl_pct, fallback_tp_pct=tp_pct,
                )
                sl_pct_actual = abs(entry_price - sl_price) / entry_price
                tp_pct_actual = abs(tp_price - entry_price) / entry_price
                print(f"  Structurele niveaus: SL={sl_pct_actual:.1%} afstand, TP={tp_pct_actual:.1%} afstand "
                      f"(R/R={tp_pct_actual/max(sl_pct_actual,0.001):.1f})")
            else:
                sl_price = entry_price * (1 - sl_pct)
                tp_price = entry_price * (1 + tp_pct)

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
                "market_regime": regime_label,
            }

            print(f"\n  ── NIEUW SIGNAAL ───────────────────────────────────────────────────")
            print(f"  {direction} | Entry ${entry_price:,.0f} | SL ${sl_price:,.0f} | TP ${tp_price:,.0f}")
            print(f"  Positie: ${position_size:,.0f} | Risico: €{capital*risk_pct:.2f}")

            coin_name = symbol.replace("USDT", "")
            coin_amount = round(position_size / entry_price, 6)
            sl_pct_actual = abs(entry_price - sl_price) / entry_price
            tp_pct_actual = abs(tp_price - entry_price) / entry_price
            rr_ratio = tp_pct_actual / max(sl_pct_actual, 0.001)
            sl_type = "structureel swing-low" if _use_structural else f"ATR×{getattr(config, 'ATR_STOP_MULTIPLIER', 2.0):.1f}"
            tp_type = "structurele resistance" if _use_structural else "vaste TP-pct"

            # Onderbouwingsregels
            checks = []
            checks.append(
                f"• Model 1h: **{signaal['kans_stijging']}** ≥ {signaal['long_threshold']} "
                f"(regime-drempel {regime_label})"
            )
            if signaal.get("proba_4h", "n/a") != "n/a":
                c4h = "✅" if signaal.get("confirm_4h") else "⚠️"
                checks.append(f"• 4h bevestiging: {signaal['proba_4h']} {c4h}")
            ema_lbl = "✅ boven" if signaal.get("regime_boven_ema200") else "❌ onder"
            dc_lbl = "⚠️ ja" if signaal.get("death_cross") else "✅ nee"
            checks.append(f"• EMA200: {ema_lbl} | Death cross: {dc_lbl}")
            rs = int(signaal.get("ranging_score", 0))
            checks.append(
                f"• Ranging score: {rs}/3 — {'markt trending ✅' if rs < 2 else 'lichte squeeze ⚠️'}"
            )
            skew_lbl = f"⚠️ puts duur ({signaal.get('btc_skew_25d','?')})" if signaal.get("skew_blocked") else f"✅ normaal ({signaal.get('btc_skew_25d','?')})"
            checks.append(f"• 25D skew: {skew_lbl}")
            if signaal.get("crash_mode"):
                checks.append("• ⚠️ Crash-modus: positie gehalveerd")
            macd_mult = 1.0 + float(signaal.get("macd_size_mult", 0.0))
            if macd_mult > 1.01:
                checks.append(f"• 📈 MACD momentum ×{macd_mult:.2f}: positie vergroot")

            msg = (
                f"🟢 **LONG — {symbol}**\n"
                f"⏰ {pd.Timestamp(latest_ts).strftime('%d-%m-%Y %H:%M')} UTC | Entry: **${entry_price:,.0f}**\n"
                f"\n**Onderbouwing**\n"
                + "\n".join(checks) +
                f"\n\n**Niveaus**\n"
                f"• SL: **${sl_price:,.0f}** (−{sl_pct_actual*100:.1f}%) — {sl_type}\n"
                f"• TP: **${tp_price:,.0f}** (+{tp_pct_actual*100:.1f}%) — {tp_type}\n"
                f"• R/R: **{rr_ratio:.1f}** | Max houdtijd: 168h "
                f"(tot {horizon_end.strftime('%d-%m %H:%M')} UTC)\n"
                f"\n**Positie**\n"
                f"• **${position_size:,.0f}** ({coin_amount:.5f} {coin_name}) "
                f"| Risico: €{capital*risk_pct:.2f} ({risk_pct*100:.0f}% kapitaal)"
            )
            opened_new_position = True
            send_discord_alert(msg)

    # ── WACHT bericht (geen nieuwe trade geopend deze run) ────────────────────
    # Throttle: stuur max 1x per 4 uur, tenzij proba binnen 5% van drempel (near-miss)
    if not opened_new_position:
        ts_str = pd.Timestamp(latest_ts).strftime("%d-%m-%Y %H:%M")
        sig_label = signaal["signaal"]
        proba_raw = signaal.get("proba_raw", 0.0)
        try:
            long_thr_float = float(signaal["long_threshold"])
        except (ValueError, KeyError):
            long_thr_float = 0.58
        near_miss = proba_raw >= long_thr_float - 0.05
        hour_now = pd.Timestamp(latest_ts).hour
        send_wacht = near_miss or (hour_now % 4 == 0)

        if send_wacht:
            if had_open_pos and state["open_position"] is not None:
                pos = state["open_position"]
                hours_open = round(
                    (pd.Timestamp(latest_ts) - pd.Timestamp(pos["open_time"])).total_seconds() / 3600, 1
                )
                pnl_pct = (latest_close - pos["entry_price"]) / pos["entry_price"] * 100
                pnl_sign = "+" if pnl_pct >= 0 else ""
                reden = (
                    f"**{pos['direction']}** open ({hours_open:.0f}h) | Entry ${pos['entry_price']:,.0f}\n"
                    f"📍 SL: ${pos['sl_price']:,.0f} | TP: ${pos['tp_price']:,.0f} | "
                    f"Huidig: ${latest_close:,.0f} ({pnl_sign}{pnl_pct:.1f}%)\n"
                    f"⏱️ Vangnet: {pd.Timestamp(pos['horizon_end']).strftime('%d-%m %H:%M')} UTC"
                )
            elif "skew" in sig_label.lower():
                reden = (
                    f"**25D skew gate actief** — puts te duur voor veilige long\n"
                    f"📊 Proba: {signaal['kans_stijging']} ≥ {signaal['long_threshold']} ✅ | "
                    f"Skew: {signaal.get('btc_skew_25d','?')} (grens: {getattr(config,'SKEW_BEARISH_GATE',5.0):.0f}%)\n"
                    f"💡 Wacht tot skew < {getattr(config,'SKEW_BEARISH_GATE',5.0):.0f}% voor herstart long-signalen"
                )
            elif "death cross" in sig_label.lower():
                reden = (
                    f"**Death cross actief** (EMA50 < EMA200) — bull trap risico\n"
                    f"📊 Proba: {signaal['kans_stijging']} | Drempel: {signaal['long_threshold']} "
                    f"| Regime: {signaal.get('market_regime','?')}\n"
                    f"💡 Wacht op EMA50 > EMA200 herstellijn"
                )
            elif "EMA200" in sig_label or "ema200" in sig_label.lower():
                reden = (
                    f"Proba hoog genoeg (**{signaal['kans_stijging']}** ≥ {signaal['long_threshold']}) "
                    f"maar prijs **onder EMA200** — long geblokkeerd\n"
                    f"💡 Regime: {signaal.get('market_regime','?')} | "
                    f"Wacht op herstel boven de 200-uur gemiddelde"
                )
            elif "4h" in sig_label.lower():
                reden = (
                    f"1h model klaar (**{signaal['kans_stijging']}** ≥ {signaal['long_threshold']}) "
                    f"maar **4h bevestiging ontbreekt**\n"
                    f"📊 4h proba: {signaal.get('proba_4h','n/a')} | "
                    f"Regime: {signaal.get('market_regime','?')}\n"
                    f"💡 Beide tijdsframes moeten bevestigen voor entry"
                )
            elif "ranging" in sig_label.lower():
                reden = (
                    f"**Ranging market** gedetecteerd — short geblokkeerd\n"
                    f"📊 Ranging score: {signaal.get('ranging_score',0):.0f}/3 "
                    f"(ADX laag + BB squeeze + MACD instabiel)\n"
                    f"💡 Geen short in zijwaartse markt — wacht op trendbevestiging"
                )
            else:
                regime = signaal.get("market_regime", "")
                gap_pct = (long_thr_float - proba_raw) * 100
                near_str = f" — nog **{gap_pct:.1f}%** te gaan" if gap_pct < 5 else ""
                reden = (
                    f"Proba te laag: **{signaal['kans_stijging']}** < {signaal['long_threshold']} "
                    f"({regime} regime){near_str}\n"
                    f"📊 4h: {signaal.get('proba_4h','n/a')} | "
                    f"Skew: {signaal.get('btc_skew_25d','?')} | "
                    f"Ranging: {signaal.get('ranging_score',0):.0f}/3"
                )

            wacht_icon = "🔔" if near_miss else "⏸️"
            wacht_msg = (
                f"{wacht_icon} **WACHT** — {symbol}\n"
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
