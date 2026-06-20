"""
Runner voor de event-driven SMC-backtest (docs/crypto/smc_backtest_spec.md, v1.1).

Executie 1h + bias 4h, gepoold over BTC + ETH. Draait standaard op de DEV-set
(t/m 2025-06-30). Holdout (2025-07-01 → nu) blijft vergrendeld — pas op een
mijlpaal met --holdout.

Gebruik (vanuit crypto/, na OHLCV 1h+4h in ohlcv.db):
  python scripts/smc_backtest_run.py            # dev, BTC+ETH gepoold
  python scripts/smc_backtest_run.py --dump     # + per-trade
  python scripts/smc_backtest_run.py --holdout  # PAS op een mijlpaal
"""

import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_fetcher import load_ohlcv  # noqa: E402
from src.smc_backtest import metrics, run_smc_backtest  # noqa: E402

DEV_END = pd.Timestamp("2025-06-30", tz="UTC")
HOLDOUT_START = pd.Timestamp("2025-07-01", tz="UTC")
SYMBOLS = ["BTCUSDT", "ETHUSDT"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--holdout", action="store_true",
                    help="Draai op de VERGRENDELDE holdout (alleen op mijlpalen!)")
    ap.add_argument("--dump", action="store_true", help="Print elke trade (diagnose)")
    args = ap.parse_args()

    all_trades = []
    for sym in SYMBOLS:
        ex = load_ohlcv(symbol=sym, interval="1h").sort_index()
        bi = load_ohlcv(symbol=sym, interval="4h").sort_index()
        if args.holdout:
            seg = ex[ex.index >= HOLDOUT_START]
        else:
            seg = ex[ex.index <= DEV_END]
        bi_seg = bi[bi.index <= seg.index[-1]]
        ts = run_smc_backtest(seg, bi_seg, avail_hours=4)
        for t in ts:
            t["sym"] = sym
        all_trades += ts

    label = (f"HOLDOUT ({HOLDOUT_START.date()} → nu)" if args.holdout
             else f"DEV (begin → {DEV_END.date()})")
    m = metrics(all_trades)

    print("=" * 60)
    print(f"SMC EVENT-DRIVEN BACKTEST v1.1 — BTC+ETH gepoold @ 1h/4h")
    print(f"Segment: {label}")
    print("=" * 60)
    if m["n_trades"] == 0:
        print("Geen trades gedetecteerd.")
        return
    if args.dump:
        for t in sorted(all_trades, key=lambda x: x["entry_time"]):
            print(f"  {t['sym']:8} {str(t['entry_time'])[:16]:16} {t['dir']:>2} "
                  f"risk{t['risk_pct']*100:5.2f}% {t['bars_held']:4d}b {t['reason']:>4} {t['r']:+6.2f}R")
        import numpy as _np
        reasons = {r: sum(t["reason"] == r for t in all_trades) for r in ("SL", "TP", "TIME")}
        print(f"  --- exit-redenen: {reasons}")
        print()
    print(f"  Trades        : {m['n_trades']}  (L {m['n_long']} / S {m['n_short']})")
    print(f"  Win rate      : {m['win_rate']*100:.1f}%")
    print(f"  Expectancy    : {m['expectancy_R']:+.3f} R / trade  (ná kosten)")
    print(f"  Total         : {m['total_R']:+.1f} R")
    print(f"  Profit factor : {m['profit_factor']:.2f}")
    print(f"  Beste/slechtste: {m['best_R']:+.2f} R / {m['worst_R']:+.2f} R")
    print()
    floor_ok = m["n_trades"] >= 30
    succ = (m["expectancy_R"] >= 0.15 and m["profit_factor"] > 1.3 and floor_ok)
    if not floor_ok:
        print("  -> < 30 trades: INCONCLUSIVE (te kleine steekproef)")
    elif succ:
        print("  -> VOLDOET aan succescriteria (exp >= +0.15R, PF > 1.3, >= 30 trades)")
    else:
        print("  -> voldoet NIET aan succescriteria")


if __name__ == "__main__":
    main()
