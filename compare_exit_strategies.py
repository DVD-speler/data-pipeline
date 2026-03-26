"""
Vergelijk exit-strategieën over meerdere marktomstandigheden.

Strategieën:
  A) Oud     — vaste 24h horizon, vaste 2%SL/6%TP
  B) Proba   — model-exit + 168h vangnet, vaste SL/TP
  C) Structuur — model-exit + 168h vangnet + structurele SL/TP

Marktperiodes (BTCUSDT als proxy):
  Bull     2023-01-01 → 2024-03-01  (BTC 16k → 69k)
  Ranging  2024-04-01 → 2024-09-30  (post-ATH consolidatie)
  Bear     2022-06-01 → 2022-12-31  (LUNA/FTX crash)
  Recent   2025-10-01 → 2026-03-18  (huidige daling)
"""
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import config
from src.backtest import (
    run_backtest_be_trail,
    compute_metrics,
    load_optimal_threshold,
    load_exit_proba,
)
from src.data_fetcher import load_ohlcv
from src.levels import precompute_swings
from src.model import load_model

MARKET_PERIODS = {
    "Bull  (2023-01 → 2024-03)": ("2023-01-01", "2024-03-01"),
    "Ranging(2024-04 → 2024-09)": ("2024-04-01", "2024-09-30"),
    "Bear  (2022-06 → 2022-12)":  ("2022-06-01", "2022-12-31"),
    "Recent(2025-10 → 2026-03)":  ("2025-10-01", "2026-03-18"),
}


def load_period(symbol: str, start: str, end: str):
    """Laad features + high/low voor een specifieke periode."""
    feat_path = config.symbol_path(symbol, "features.parquet")
    feat = pd.read_parquet(feat_path)
    # strip tz voor vergelijking als nodig
    idx = feat.index
    if idx.tz is not None:
        s = pd.Timestamp(start, tz="UTC")
        e = pd.Timestamp(end, tz="UTC")
    else:
        s = pd.Timestamp(start)
        e = pd.Timestamp(end)
    feat = feat[(idx >= s) & (idx <= e)].copy()
    if len(feat) < 100:
        return None, None, None

    # High/low uit SQLite
    ohlcv = load_ohlcv(symbol=symbol, interval="1h")
    ohlcv = ohlcv[(ohlcv.index >= s) & (ohlcv.index <= e)]
    # Align index
    common = feat.index.intersection(ohlcv.index)
    feat  = feat.loc[common]
    ohlcv = ohlcv.loc[common]

    highs = ohlcv["high"].values
    lows  = ohlcv["low"].values
    return feat, highs, lows


def run_strategy(label, feat, highs, lows, probas, thr, short_thr,
                 exit_long, exit_short,
                 use_structural, horizon, exit_proba_on):
    """Voer één strategie uit en return metrics."""
    el = exit_long  if exit_proba_on else 1.0
    es = exit_short if exit_proba_on else 0.0

    return run_backtest_be_trail(
        feat, probas,
        threshold=thr,
        threshold_short=short_thr,
        use_short=(short_thr > 0),
        exit_proba_long=el,
        exit_proba_short=es,
        horizon=horizon,
        use_structural_levels=use_structural,
        highs=highs if use_structural else None,
        lows=lows  if use_structural else None,
    )


def compare_symbol(symbol: str):
    print(f"\n{'#'*68}")
    print(f"##  {symbol}")
    print(f"{'#'*68}")

    model = load_model(symbol=symbol)
    thr, short_thr = load_optimal_threshold(symbol=symbol)
    exit_long, exit_short = load_exit_proba(symbol=symbol)

    header = f"  {'Periode':<30}  {'A:24h+vast':>10}  {'B:proba+vast':>12}  {'C:proba+struct':>14}"
    sub    = f"  {'':30}  {'Return':>10}  {'Return':>12}  {'Return':>14}  (Sharpe A/B/C)"

    for period_name, (start, end) in MARKET_PERIODS.items():
        feat, highs, lows = load_period(symbol, start, end)
        if feat is None or len(feat) < 200:
            print(f"\n  {period_name}  — onvoldoende data, overgeslagen")
            continue

        probas = model.predict_proba(feat[config.FEATURE_COLS])[:, 1]
        bh = feat["close"].iloc[-1] / feat["close"].iloc[0] - 1
        n_hours = len(feat)

        # Strategie A: oud (24h, vaste SL/TP, geen proba-exit)
        res_a = run_strategy("A", feat, highs, lows, probas, thr, short_thr,
                             exit_long, exit_short,
                             use_structural=False, horizon=24, exit_proba_on=False)
        m_a = compute_metrics(res_a, horizon=24)

        # Strategie B: proba-exit + 168h + vaste SL/TP
        res_b = run_strategy("B", feat, highs, lows, probas, thr, short_thr,
                             exit_long, exit_short,
                             use_structural=False, horizon=config.MAX_HOLD_HOURS,
                             exit_proba_on=True)
        m_b = compute_metrics(res_b, horizon=config.MAX_HOLD_HOURS)

        # Strategie C: proba-exit + 168h + structurele SL/TP
        swings = precompute_swings(highs, lows)
        res_c = run_strategy("C", feat, highs, lows, probas, thr, short_thr,
                             exit_long, exit_short,
                             use_structural=True, horizon=config.MAX_HOLD_HOURS,
                             exit_proba_on=True)
        m_c = compute_metrics(res_c, horizon=config.MAX_HOLD_HOURS)

        def fmt(m):
            r = m["total_return"]
            s = m["sharpe_ratio"]
            return f"{r:>+7.1%}", f"{s:>+6.2f}"

        ra, sa = fmt(m_a)
        rb, sb = fmt(m_b)
        rc, sc = fmt(m_c)

        # Winnaar markeren
        best_r = max(m_a["total_return"], m_b["total_return"], m_c["total_return"])
        mark_a = " ★" if m_a["total_return"] == best_r else "  "
        mark_b = " ★" if m_b["total_return"] == best_r else "  "
        mark_c = " ★" if m_c["total_return"] == best_r else "  "

        print(f"\n  ── {period_name}  ({n_hours}h, B&H: {bh:+.1%}) ──")
        print(f"  {'Strategie':<16} {'Return':>8} {'Sharpe':>8} {'Trades':>7} {'WinRate':>8} {'MaxDD':>8}")
        print(f"  {'-'*58}")

        def row(lbl, m, mark):
            print(f"  {lbl+mark:<16} {m['total_return']:>+8.1%} {m['sharpe_ratio']:>+8.3f} "
                  f"{m['n_trades']:>7} {m['win_rate']:>8.1%} {m['max_drawdown']:>+8.1%}")

        row("A: 24h+vast", m_a, mark_a)
        row("B: proba+vast", m_b, mark_b)
        row("C: proba+struct", m_c, mark_c)


if __name__ == "__main__":
    for sym in config.SYMBOLS:
        compare_symbol(sym)
    print("\nKlaar.")
