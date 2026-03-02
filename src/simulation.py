"""
Trade Alert Simulatie
Simuleert maandelijkse tradingalerts met vast risico per trade, SL en TP.

Strategie:
  - Één positie tegelijk (non-overlapping): nieuwe signalen worden genegeerd zolang een trade open is
  - Positiegrootte = risk_amount / sl_pct   (bijv. €10 risico / 2% SL = €500 positie)
  - Exit volgorde: SL → TP → horizon (24h)
  - SL/TP worden gecontroleerd op basis van uurlijkse high/low (niet alleen close)

Gebruik:
  python main.py --phase simulation
  of:
  from src.simulation import simulate_month, print_month_report
"""

import textwrap
from datetime import timezone

import numpy as np
import pandas as pd

import config
from src.backtest import run_backtest
from src.data_fetcher import load_ohlcv
from src.model import load_model, load_optimal_threshold


# ── Simulatie kern ─────────────────────────────────────────────────────────────

def simulate_month(
    start_date: str,
    end_date: str,
    initial_capital: float = 1000.0,
    risk_pct: float = 0.01,
    sl_pct: float = 0.02,
    tp_pct: float = 0.06,
    symbol: str = config.SYMBOL,
) -> tuple[list[dict], float]:
    """
    Simuleer tradingalerts voor een gegeven periode.

    Returns
    -------
    (trades_list, eindkapitaal)
    """
    # ── Laad data ────────────────────────────────────────────────────────────
    features = pd.read_parquet(config.symbol_path(symbol, "features.parquet"))
    ohlcv    = load_ohlcv(symbol=symbol)

    model                = load_model(symbol=symbol)
    long_thr, short_thr  = load_optimal_threshold(symbol=symbol)

    # Normaliseer tijdzones voor vergelijking
    tz = features.index.tz
    start = pd.Timestamp(start_date, tz=tz)
    end   = pd.Timestamp(end_date,   tz=tz)

    period = features[(features.index >= start) & (features.index < end)].copy()
    if len(period) == 0:
        print(f"  Geen data voor {start_date} t/m {end_date}")
        return [], initial_capital

    # ── Signalen berekenen ───────────────────────────────────────────────────
    probas = model.predict_proba(period[config.FEATURE_COLS])[:, 1]
    period["proba"] = probas

    # Long: proba >= threshold EN boven EMA200 EN niet confirmed bear
    above_ema200    = period["price_vs_ema200"] > 1.0
    not_bear_regime = period.get("market_regime", pd.Series(1, index=period.index)) != -1
    long_signal     = (period["proba"] >= long_thr) & above_ema200 & not_bear_regime

    # Short: proba <= threshold_short EN onder EMA200 EN macro bearmarkt
    if short_thr > 0 and "return_30d" in period.columns and "return_7d" in period.columns:
        macro_bear   = (period["return_30d"] < -0.03) & (period["return_7d"] < -0.01)
        short_signal = (period["proba"] <= short_thr) & (~above_ema200) & macro_bear
    else:
        short_signal = pd.Series(False, index=period.index)

    # ── OHLCV voor SL/TP check (high/low per uur) ────────────────────────────
    # OHLCV heeft al UTC-aware DatetimeIndex vanuit load_ohlcv()
    if ohlcv.index.tz is None:
        ohlcv.index = ohlcv.index.tz_localize("UTC")
    elif str(ohlcv.index.tz) != "UTC":
        ohlcv.index = ohlcv.index.tz_convert("UTC")
    # Zorg dat OHLCV de goede kolommen heeft
    if "high" not in ohlcv.columns:
        ohlcv["high"] = ohlcv["close"]
        ohlcv["low"]  = ohlcv["close"]

    # ── Trade simulatie (non-overlapping) ────────────────────────────────────
    trades       = []
    capital      = initial_capital
    trade_num    = 0
    hold_until   = None   # Timestamp tot wanneer we in een positie zitten

    for ts in period.index:
        # Skip als we nog in een positie zitten
        if hold_until is not None and ts <= hold_until:
            continue

        # Bepaal richting
        direction = None
        if long_signal.loc[ts]:
            direction = "LONG"
        elif short_signal.loc[ts]:
            direction = "SHORT"

        if direction is None:
            continue

        entry_price = float(period.loc[ts, "close"])
        proba       = float(period.loc[ts, "proba"])

        if direction == "LONG":
            sl_price = entry_price * (1 - sl_pct)
            tp_price = entry_price * (1 + tp_pct)
        else:
            sl_price = entry_price * (1 + sl_pct)
            tp_price = entry_price * (1 - tp_pct)

        # Positiegrootte op basis van risico
        risk_amount   = capital * risk_pct
        position_size = risk_amount / sl_pct       # in euro's

        # ── Zoek exit: SL / TP / 24h ─────────────────────────────────────
        horizon_end = ts + pd.Timedelta(hours=config.PREDICTION_HORIZON_H)

        # Haal OHLCV op voor de holdperiode
        hold_ohlcv = ohlcv[(ohlcv.index > ts) & (ohlcv.index <= horizon_end)]

        exit_price  = None
        exit_time   = None
        exit_reason = f"{config.PREDICTION_HORIZON_H}h"

        for h_ts, row in hold_ohlcv.iterrows():
            h_high = float(row.get("high", row["close"]))
            h_low  = float(row.get("low",  row["close"]))
            h_close= float(row["close"])

            if direction == "LONG":
                if h_low <= sl_price:                     # SL geraakt
                    exit_price  = sl_price
                    exit_time   = h_ts
                    exit_reason = "SL ✗"
                    break
                if h_high >= tp_price:                    # TP geraakt
                    exit_price  = tp_price
                    exit_time   = h_ts
                    exit_reason = "TP ✓"
                    break
            else:  # SHORT
                if h_high >= sl_price:                    # SL geraakt
                    exit_price  = sl_price
                    exit_time   = h_ts
                    exit_reason = "SL ✗"
                    break
                if h_low <= tp_price:                     # TP geraakt
                    exit_price  = tp_price
                    exit_time   = h_ts
                    exit_reason = "TP ✓"
                    break

        # Geen SL/TP: sluit op horizon
        if exit_price is None:
            if len(hold_ohlcv) > 0:
                exit_price = float(hold_ohlcv.iloc[-1]["close"])
                exit_time  = hold_ohlcv.index[-1]
            else:
                continue   # Geen data beschikbaar

        # ── P&L berekenen ─────────────────────────────────────────────────
        if direction == "LONG":
            gross_return = (exit_price - entry_price) / entry_price
        else:
            gross_return = (entry_price - exit_price) / entry_price

        pnl_euro  = position_size * gross_return - position_size * 2 * config.TRADE_FEE
        pnl_pct   = pnl_euro / capital * 100    # % van kapitaal
        capital  += pnl_euro
        hold_until = exit_time

        trade_num += 1
        trades.append({
            "nr":            trade_num,
            "open_tijd":     ts,
            "richting":      direction,
            "entry":         round(entry_price, 0),
            "sl":            round(sl_price, 0),
            "tp":            round(tp_price, 0),
            "sluit_tijd":    exit_time,
            "exit":          round(exit_price, 0),
            "reden":         exit_reason,
            "proba":         round(proba, 2),
            "bruto_ret":     round(gross_return * 100, 2),
            "pos_eur":       round(position_size, 0),
            "pnl_eur":       round(pnl_euro, 2),
            "pnl_pct_kap":   round(pnl_pct, 2),
            "kapitaal":      round(capital, 2),
        })

    return trades, round(capital, 2)


# ── Rapportage ─────────────────────────────────────────────────────────────────

def print_month_report(
    label: str,
    trades: list[dict],
    initial_capital: float,
    final_capital: float,
    risk_pct: float,
    sl_pct: float,
    tp_pct: float,
) -> None:
    """Print een nette trade-overzichttabel voor één maand."""
    width = 120

    print()
    print("═" * width)
    titel = (
        f"  SIMULATIE: {label}  |  "
        f"Startkapitaal: €{initial_capital:,.2f}  |  "
        f"Risico/trade: {risk_pct*100:.0f}%  |  "
        f"SL: {sl_pct*100:.0f}%  |  TP: {tp_pct*100:.0f}%"
    )
    print(titel)
    print("═" * width)

    if not trades:
        print("  Geen trades in deze periode.")
        print("═" * width)
        return

    # Kolomheaders
    hdr = (
        f"  {'#':>2}  {'Open':^16}  {'Dir':^5}  {'Entry':>8}  "
        f"{'SL':>8}  {'TP':>8}  {'Sluit':^16}  {'Exit':>8}  "
        f"{'Reden':^8}  {'Proba':>5}  {'Ret%':>6}  {'P&L€':>8}  "
        f"{'%Kap':>6}  {'Kapitaal':>10}"
    )
    print(hdr)
    print("─" * width)

    n_wins = 0
    n_sl   = 0
    n_tp   = 0
    n_h    = 0
    best_trade  = None
    worst_trade = None
    total_pnl   = 0.0

    for t in trades:
        ot  = t["open_tijd"].strftime("%d-%m %H:%M")
        st  = t["sluit_tijd"].strftime("%d-%m %H:%M") if hasattr(t["sluit_tijd"], "strftime") else str(t["sluit_tijd"])[:11]
        win = t["pnl_eur"] > 0
        if win:
            n_wins += 1
        if "SL" in t["reden"]:
            n_sl += 1
        elif "TP" in t["reden"]:
            n_tp += 1
        else:
            n_h += 1

        total_pnl += t["pnl_eur"]
        if best_trade  is None or t["pnl_eur"] > best_trade["pnl_eur"]:
            best_trade = t
        if worst_trade is None or t["pnl_eur"] < worst_trade["pnl_eur"]:
            worst_trade = t

        # Kleur-indicator (tekst-gebaseerd)
        win_mark = "+" if win else "-"
        dir_col  = "LONG " if t["richting"] == "LONG" else "SHORT"

        row = (
            f"  {t['nr']:>2}  {ot:^16}  {dir_col:^5}  {t['entry']:>8,.0f}  "
            f"{t['sl']:>8,.0f}  {t['tp']:>8,.0f}  {st:^16}  {t['exit']:>8,.0f}  "
            f"{t['reden']:^8}  {t['proba']:>5.2f}  "
            f"{win_mark}{abs(t['bruto_ret']):>5.2f}%  "
            f"{win_mark}€{abs(t['pnl_eur']):>6.2f}  "
            f"{win_mark}{abs(t['pnl_pct_kap']):>5.2f}%  "
            f"€{t['kapitaal']:>9,.2f}"
        )
        print(row)

    print("─" * width)

    # Samenvatting
    n        = len(trades)
    win_rate = n_wins / n * 100
    end_pnl  = final_capital - initial_capital
    end_sign = "+" if end_pnl >= 0 else "-"

    print(f"\n  SAMENVATTING")
    print(f"  {'Trades':20}: {n}  ({n_wins} wins | {n - n_wins} losses)")
    print(f"  {'Win rate':20}: {win_rate:.1f}%")
    print(f"  {'Exit via SL':20}: {n_sl}×   |  TP: {n_tp}×   |  24h: {n_h}×")
    if best_trade:
        bt = best_trade
        print(f"  {'Beste trade':20}: #{bt['nr']} {bt['richting']} {bt['open_tijd'].strftime('%d-%m %H:%M')} → +€{bt['pnl_eur']:.2f} (+{bt['bruto_ret']:.2f}%)")
    if worst_trade:
        wt = worst_trade
        print(f"  {'Slechtste trade':20}: #{wt['nr']} {wt['richting']} {wt['open_tijd'].strftime('%d-%m %H:%M')} → €{wt['pnl_eur']:.2f} ({wt['bruto_ret']:.2f}%)")
    print(f"  {'Startkapitaal':20}: €{initial_capital:,.2f}")
    print(f"  {'Eindkapitaal':20}: €{final_capital:,.2f}  ({end_sign}€{abs(end_pnl):.2f}  {end_sign}{abs(end_pnl/initial_capital*100):.1f}%)")
    print("═" * width)


# ── Hoofdfunctie ───────────────────────────────────────────────────────────────

def run_simulation(
    initial_capital: float = 1000.0,
    risk_pct: float = 0.01,
    sl_pct: float = 0.02,
    tp_pct: float = 0.06,
    symbol: str = config.SYMBOL,
) -> None:
    """
    Draai simulaties voor meerdere representatieve maanden en print de resultaten.

    Periodes:
      1. April 2025     — stijgende markt (bull run)
      2. Sep–Okt 2025   — dalende markt (bear)
      3. Nov–Dec 2025   — herstel + bear (mix)
      4. Jan 2026       — meest recente testperiode
    """
    long_thr, short_thr = load_optimal_threshold(symbol=symbol)

    print("\n" + "═" * 120)
    print(f"  {symbol} HANDELSSIMULATIE  |  Model: LightGBM 24h horizon")
    print(f"  Beschikbare data: 2024-03-29 t/m 2026-02-26")
    print(f"  Thresholds: long ≥ {long_thr:.2f}  |  short ≤ {short_thr:.2f}  (short vereist ook macro bear filter)")
    print(f"  Positiegrootte: {risk_pct*100:.0f}% risico / {sl_pct*100:.0f}% SL = {risk_pct/sl_pct*100:.0f}% van kapitaal per trade")
    print("═" * 120)

    # Maanden om te simuleren
    periods = [
        ("April 2025 (bull markt)",          "2025-04-01", "2025-05-01"),
        ("Augustus 2025 (Japan-crash/herstel)","2025-08-01", "2025-09-01"),
        ("Oktober 2025 (bear markt)",         "2025-10-01", "2025-11-01"),
        ("December 2025 (bull herstel)",      "2025-12-01", "2026-01-01"),
        ("Januari 2026 (recent)",             "2026-01-01", "2026-02-01"),
    ]

    for label, start, end in periods:
        cap = initial_capital   # Reset kapitaal per maand (aparte simulatie)
        trades, final_cap = simulate_month(start, end, cap, risk_pct, sl_pct, tp_pct,
                                           symbol=symbol)
        print_month_report(label, trades, cap, final_cap, risk_pct, sl_pct, tp_pct)


if __name__ == "__main__":
    run_simulation()
