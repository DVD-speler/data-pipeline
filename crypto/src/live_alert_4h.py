"""
4h Signaal Pipeline

BTC : paper trading (model statistisch significant — Sharpe +11.87 > p95 +6.11)
ETH : richtingsindicator (model NIET significant — Sharpe +7.73 < p95 +12.05)

Stuurt berichten naar een apart Discord channel via DISCORD_WEBHOOK_URL_4H.

Gebruik:
  python main.py --phase live_alert_4h --symbol BTCUSDT
  of lokaal:
  DISCORD_WEBHOOK_URL_4H=https://... python main.py --phase live_alert_4h --symbol BTCUSDT
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

import config_4h as cfg

from shared.notifier import send_alert as _shared_send_alert
from shared.paper_state import load_paper_state, save_paper_state


# ── Discord (4h channel) ──────────────────────────────────────────────────────

def send_alert(content: str) -> None:
    """Stuur bericht naar het 4h Discord channel via DISCORD_WEBHOOK_URL_4H."""
    _shared_send_alert(content, webhook_env_var="DISCORD_WEBHOOK_URL_4H")


# ── Signaal generatie ─────────────────────────────────────────────────────────

def _generate_4h_signal(df_feat: pd.DataFrame, symbol: str) -> dict:
    """
    Genereer 4h richtingssignaal op basis van de meest recente 4h candle.

    Returns dict met: signaal, kans_stijging, prijs, tijdstip, filters, etc.
    """
    model_path = cfg.symbol_path_4h(symbol, "model.pkl")
    thr_path   = cfg.symbol_path_4h(symbol, "optimal_threshold.json")

    if not model_path.exists():
        return {
            "signaal":       "GEEN MODEL",
            "richting":      "NEUTRAAL",
            "kans_stijging": 0.5,
            "prijs":         0.0,
            "tijdstip":      "n/a",
        }

    model = joblib.load(model_path)

    # Laad drempelwaarden
    threshold, threshold_short = 0.65, 0.0
    if thr_path.exists():
        with open(thr_path) as f:
            thr_data = json.load(f)
        threshold       = float(thr_data.get("threshold", 0.65))
        threshold_short = float(thr_data.get("threshold_short", 0.0))

    last_row     = df_feat.iloc[[-1]]
    feature_cols = [c for c in cfg.FEATURE_COLS_4H_MODEL if c in last_row.columns]
    proba        = float(model.predict_proba(last_row[feature_cols])[:, 1][0])

    close_price     = float(last_row["close"].iloc[0])
    ts              = last_row.index[-1]
    market_regime   = float(last_row["market_regime"].iloc[0]) if "market_regime" in last_row else 0.0
    price_vs_ema200 = float(last_row["price_vs_ema200"].iloc[0]) if "price_vs_ema200" in last_row else 1.0
    ema_ratio_50    = float(last_row["ema_ratio_50"].iloc[0])    if "ema_ratio_50"    in last_row else 1.0
    return_30d      = float(last_row["return_30d"].iloc[0])      if "return_30d"      in last_row else 0.0
    return_7d       = float(last_row["return_7d"].iloc[0])       if "return_7d"       in last_row else 0.0

    # Regime-adaptieve drempel (zelfde als 1h model)
    regime_int    = int(market_regime)
    regime_offset = cfg.REGIME_THRESHOLD_OFFSETS.get(regime_int, 0.0)
    eff_threshold = float(np.clip(threshold + regime_offset, 0.50, 0.95))

    # Filters
    regime_ok   = price_vs_ema200 > 1.0                      # prijs boven EMA200
    death_cross = ema_ratio_50 > price_vs_ema200              # EMA50 < EMA200
    bear_regime = market_regime == -1

    # Short macro gate: beide 30d én 7d negatief
    short_macro_ok = (return_30d < -0.03) and (return_7d < -0.01)

    # Signaal bepalen
    if proba >= eff_threshold and regime_ok and not death_cross and not bear_regime:
        signaal = "LONG"
    elif (proba <= threshold_short and threshold_short > 0
          and not regime_ok and short_macro_ok):
        signaal = "SHORT"
    elif death_cross and proba >= eff_threshold:
        signaal = "WACHT (death cross)"
    elif proba >= eff_threshold and not regime_ok:
        signaal = "WACHT (onder EMA200)"
    else:
        signaal = "WACHT"

    regime_labels = {1: "bull (ADX+)", 0: "ranging", -1: "bear (ADX-)"}
    regime_label  = regime_labels.get(regime_int, "onbekend")

    return {
        "signaal":          signaal,
        "richting":         "BULLISH" if proba >= 0.55 else ("BEARISH" if proba <= 0.45 else "NEUTRAAL"),
        "kans_stijging":    proba,
        "prijs":            close_price,
        "tijdstip":         str(ts),
        "market_regime":    market_regime,
        "regime_label":     regime_label,
        "death_cross":      death_cross,
        "boven_ema200":     regime_ok,
        "long_threshold":   eff_threshold,
        "long_threshold_base": threshold,
        "short_threshold":  threshold_short,
    }


# ── Positie management (BTC) ──────────────────────────────────────────────────

def check_position_exit(
    pos: dict,
    latest_close: float,
    latest_high: float,
    latest_low: float,
    latest_ts,
) -> dict | None:
    sl_price    = pos["sl_price"]
    tp_price    = pos["tp_price"]
    direction   = pos["direction"]
    horizon_end = pd.Timestamp(pos["horizon_end"])
    now         = pd.Timestamp(latest_ts)

    if direction == "LONG":
        if latest_low <= sl_price:
            return {"exit_price": sl_price, "exit_reason": "SL ✗", "exit_time": str(latest_ts)}
        if latest_high >= tp_price:
            return {"exit_price": tp_price, "exit_reason": "TP ✓", "exit_time": str(latest_ts)}
    else:
        if latest_high >= sl_price:
            return {"exit_price": sl_price, "exit_reason": "SL ✗", "exit_time": str(latest_ts)}
        if latest_low <= tp_price:
            return {"exit_price": tp_price, "exit_reason": "TP ✓", "exit_time": str(latest_ts)}

    if now >= horizon_end:
        return {"exit_price": latest_close, "exit_reason": "12h ⏰", "exit_time": str(latest_ts)}

    return None


def _close_position(state: dict, pos: dict, exit_info: dict) -> float:
    entry      = pos["entry_price"]
    exit_price = exit_info["exit_price"]
    pos_size   = pos["position_size"]
    direction  = pos["direction"]

    if direction == "LONG":
        gross_ret = (exit_price - entry) / entry
    else:
        gross_ret = (entry - exit_price) / entry

    pnl_euro       = pos_size * gross_ret - pos_size * 2 * cfg.TRADE_FEE
    state["capital"] = round(state["capital"] + pnl_euro, 2)

    state["closed_trades"].append({
        **pos,
        **exit_info,
        "gross_return_pct": round(gross_ret * 100, 2),
        "pnl_euro":         round(pnl_euro, 2),
        "capital_after":    state["capital"],
    })
    state["open_position"] = None
    return pnl_euro


# ── BTC: paper trading ────────────────────────────────────────────────────────

def _run_btc_paper_trading(
    symbol: str,
    df_feat: pd.DataFrame,
    df_ohlcv: pd.DataFrame,
    signaal: dict,
    risk_pct: float = 0.01,
    sl_pct: float = 0.02,
    tp_pct: float = 0.06,
) -> None:
    """Verwerk BTC 4h signaal met paper trading (SL/TP/horizon)."""
    paper_path = cfg.symbol_path_4h(symbol, "paper_trades.json")
    state      = load_paper_state(paper_path)
    had_open   = state["open_position"] is not None

    latest_ts    = df_ohlcv.index[-1]
    latest_close = float(df_ohlcv["close"].iloc[-1])
    latest_high  = float(df_ohlcv["high"].iloc[-1])
    latest_low   = float(df_ohlcv["low"].iloc[-1])

    print(f"  Kapitaal : €{state['capital']:,.2f}")

    # ── Check open positie ────────────────────────────────────────────────────
    if state["open_position"] is not None:
        pos       = state["open_position"]
        exit_info = check_position_exit(pos, latest_close, latest_high, latest_low, latest_ts)

        if exit_info:
            pnl_euro = _close_position(state, pos, exit_info)
            sign     = "+" if pnl_euro >= 0 else ""
            icon     = "✅" if pnl_euro > 0 else "❌"
            gross_pct = state["closed_trades"][-1]["gross_return_pct"]

            print(f"  TRADE GESLOTEN: {pos['direction']} ${pos['entry_price']:,.0f} → "
                  f"${exit_info['exit_price']:,.0f} | {exit_info['exit_reason']} | "
                  f"P&L: {sign}€{pnl_euro:.2f}")

            msg = (
                f"{icon} **TRADE GESLOTEN** — {symbol} (4h)\n"
                f"📈 **{pos['direction']}** | ${pos['entry_price']:,.0f} → ${exit_info['exit_price']:,.0f}\n"
                f"🎯 Reden: {exit_info['exit_reason']} | Rendement: {sign}{gross_pct:.2f}%\n"
                f"💰 P&L: {sign}€{pnl_euro:.2f} | Kapitaal: €{state['capital']:,.2f}"
            )
            send_alert(msg)
        else:
            pos_hrs = round(
                (pd.Timestamp(latest_ts) - pd.Timestamp(pos["open_time"])).total_seconds() / 3600, 1
            )
            print(f"  Positie open: {pos['direction']} ${pos['entry_price']:,.0f} ({pos_hrs}h)")

    # ── Nieuw signaal ─────────────────────────────────────────────────────────
    opened_new = False
    if state["open_position"] is None and signaal["signaal"] in ("LONG", "SHORT"):
        direction     = signaal["signaal"]
        entry_price   = latest_close
        signal_ts     = pd.Timestamp(latest_ts)
        capital       = state["capital"]
        position_size = capital * risk_pct / sl_pct

        if direction == "LONG":
            sl_price = entry_price * (1 - sl_pct)
            tp_price = entry_price * (1 + tp_pct)
        else:
            sl_price = entry_price * (1 + sl_pct)
            tp_price = entry_price * (1 - tp_pct)

        # Horizon: 3 × 4h = 12h
        horizon_end = signal_ts + pd.Timedelta(hours=cfg.PREDICTION_HORIZON_4H * 4)

        state["open_position"] = {
            "direction":    direction,
            "entry_price":  round(entry_price, 0),
            "sl_price":     round(sl_price, 0),
            "tp_price":     round(tp_price, 0),
            "open_time":    str(signal_ts),
            "horizon_end":  str(horizon_end),
            "position_size": round(position_size, 2),
            "risk_euro":    round(capital * risk_pct, 2),
            "proba":        signaal["kans_stijging"],
        }

        coin_name   = symbol.replace("USDT", "")
        coin_amount = round(position_size / entry_price, 6)
        icon        = "🟢" if direction == "LONG" else "🔴"
        regime_lbl  = "boven EMA200" if signaal["boven_ema200"] else "onder EMA200"

        print(f"  NIEUW SIGNAAL: {direction} | Entry ${entry_price:,.0f} | "
              f"SL ${sl_price:,.0f} | TP ${tp_price:,.0f}")

        msg = (
            f"{icon} **{direction} SIGNAAL** — {symbol} (4h)\n"
            f"⏰ {pd.Timestamp(latest_ts).strftime('%d-%m-%Y %H:%M')} UTC\n"
            f"💰 Entry: ${entry_price:,.0f} | SL: ${sl_price:,.0f} (−{sl_pct*100:.0f}%) | "
            f"TP: ${tp_price:,.0f} (+{tp_pct*100:.0f}%)\n"
            f"📊 Proba: {signaal['kans_stijging']:.1%} | Regime: {regime_lbl}\n"
            f"💼 Positie: ${position_size:,.0f} ({coin_amount:.6f} {coin_name}) | "
            f"Risico: €{capital*risk_pct:.2f}"
        )
        opened_new = True
        send_alert(msg)

    # ── Wacht bericht ─────────────────────────────────────────────────────────
    if not opened_new:
        ts_str  = pd.Timestamp(latest_ts).strftime("%d-%m-%Y %H:%M")
        sig_lbl = signaal["signaal"]

        if had_open and state["open_position"] is not None:
            pos      = state["open_position"]
            pos_hrs  = round(
                (pd.Timestamp(latest_ts) - pd.Timestamp(pos["open_time"])).total_seconds() / 3600, 1
            )
            reden = (
                f"Al een **{pos['direction']}** positie open ({pos_hrs}h, entry ${pos['entry_price']:,.0f})\n"
                f"📍 SL: ${pos['sl_price']:,.0f} | TP: ${pos['tp_price']:,.0f} | "
                f"Sluit uiterlijk: {pd.Timestamp(pos['horizon_end']).strftime('%d-%m %H:%M')} UTC"
            )
        elif "death cross" in sig_lbl.lower():
            reden = f"**Death cross actief** (EMA50 onder EMA200) → long geblokkeerd"
        elif "EMA200" in sig_lbl:
            reden = f"Proba hoog ({signaal['kans_stijging']:.1%}) maar prijs **onder EMA200** → geblokkeerd"
        else:
            regime = signaal.get("regime_label", "")
            reden  = (
                f"Proba te laag: **{signaal['kans_stijging']:.1%}** "
                f"(drempel: {signaal['long_threshold']:.2f}) | Regime: {regime}"
            )

        send_alert(
            f"⏸️ **WACHT** — {symbol} (4h)\n"
            f"⏰ {ts_str} UTC | Prijs: ${latest_close:,.0f}\n"
            f"💡 {reden}"
        )

    # ── Opslaan ───────────────────────────────────────────────────────────────
    state["last_checked"] = str(latest_ts)
    save_paper_state(state, paper_path)


# ── ETH: richtingsindicator ───────────────────────────────────────────────────

def _run_eth_direction_indicator(
    symbol: str,
    df_ohlcv: pd.DataFrame,
    signaal: dict,
) -> None:
    """ETH 4h richtingsindicator — geen paper trading, alleen Discord alert."""
    latest_ts = df_ohlcv.index[-1]
    proba     = signaal["kans_stijging"]
    richting  = signaal["richting"]
    prijs     = signaal["prijs"]
    ts_str    = pd.Timestamp(signaal["tijdstip"]).strftime("%d-%m-%Y %H:%M")
    regime    = signaal["regime_label"]
    death_x   = signaal["death_cross"]
    boven_200 = signaal["boven_ema200"]

    if richting == "BULLISH":
        icon = "🟢"
        bar  = "▓" * int(proba * 10) + "░" * (10 - int(proba * 10))
        pct  = proba * 100
    elif richting == "BEARISH":
        icon = "🔴"
        bar  = "▓" * int((1 - proba) * 10) + "░" * (10 - int((1 - proba) * 10))
        pct  = (1 - proba) * 100
    else:
        icon = "⚪"
        bar  = "▒▒▒▒▒▒▒▒▒▒"
        pct  = 50.0

    filters = []
    if death_x:
        filters.append("⚠️ Death cross (EMA50 < EMA200)")
    if not boven_200:
        filters.append("📉 Prijs onder EMA200")
    filter_str = "\n".join(filters) if filters else "✅ Geen filters actief"

    msg = (
        f"{icon} **4H SIGNAAL** — {symbol} *(richtingsindicator)*\n"
        f"📆 {ts_str} UTC | Prijs: ${prijs:,.2f}\n"
        f"**{richting}** — {pct:.0f}% zekerheid\n"
        f"{bar}\n"
        f"📊 Regime: {regime} | Proba: {proba:.1%}\n"
        f"{filter_str}"
    )
    send_alert(msg)


# ── Hoofdfunctie ──────────────────────────────────────────────────────────────

def run_live_alert_4h(
    symbol: str = cfg.SYMBOL,
    risk_pct: float = 0.01,
    sl_pct: float = 0.02,
    tp_pct: float = 0.06,
) -> None:
    """
    Voer één 4h check-cyclus uit:
    1. Laad 4h OHLCV + bouw features
    2. Genereer signaal
    3. BTC → paper trading; ETH → richtingsindicator
    4. Sla latest_signal.json op
    """
    from src.data_fetcher import load_ohlcv
    from src.features_4h import build_features_4h

    print(f"\n── 4h Alert ({symbol}) ──────────────────────────────────────────────────")

    df_ohlcv = load_ohlcv(symbol=symbol, interval="4h")
    df_feat  = build_features_4h(df_ohlcv, symbol=symbol)
    signaal  = _generate_4h_signal(df_feat, symbol=symbol)

    print(f"  Tijdstip : {signaal['tijdstip']}")
    print(f"  Signaal  : {signaal['signaal']}")
    print(f"  Proba    : {signaal['kans_stijging']:.1%}")
    print(f"  Prijs    : ${signaal['prijs']:,.2f}")
    print(f"  Regime   : {signaal['regime_label']}")

    if symbol == "BTCUSDT":
        _run_btc_paper_trading(
            symbol, df_feat, df_ohlcv, signaal,
            risk_pct=risk_pct, sl_pct=sl_pct, tp_pct=tp_pct,
        )
    else:
        _run_eth_direction_indicator(symbol, df_ohlcv, signaal)

    # ── Sla signaal op ────────────────────────────────────────────────────────
    out_path = cfg.symbol_path_4h(symbol, "latest_signal.json")
    with open(out_path, "w") as f:
        json.dump(signaal, f, indent=2, default=str)

    print(f"\n  Signaal opgeslagen: {out_path}")
    print("── Klaar ────────────────────────────────────────────────────────────────")
