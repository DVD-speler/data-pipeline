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

    label = (f"HOLDOUT ({HOLDOUT_START.date()} → nu)" if args.holdout
             else f"DEV (begin → {DEV_END.date()})")
    print("=" * 60)
    print("SMC EVENT-DRIVEN BACKTEST v2 — BTC+ETH gepoold @ 1h/4h")
    print(f"Segment: {label}")
    print("=" * 60)

    def run(use_ind):
        trades = []
        for sym in SYMBOLS:
            ex = load_ohlcv(symbol=sym, interval="1h").sort_index()
            bi = load_ohlcv(symbol=sym, interval="4h").sort_index()
            seg = ex[ex.index >= HOLDOUT_START] if args.holdout else ex[ex.index <= DEV_END]
            bi_seg = bi[bi.index <= seg.index[-1]]
            ts = run_smc_backtest(seg, bi_seg, avail_hours=4, use_inducement=use_ind)
            for t in ts:
                t["sym"] = sym
            trades += ts
        return trades

    for use_ind, name in [(False, "1. ZONDER inducement (baseline)"),
                          (True, "2. MET inducement (stap 2)")]:
        trades = run(use_ind)
        m = metrics(trades)
        print(f"--- {name} ---")
        if m["n_trades"] == 0:
            print("  geen trades\n")
            continue
        floor_ok = m["n_trades"] >= 30
        succ = m["expectancy_R"] >= 0.15 and m["profit_factor"] > 1.3 and floor_ok
        print(f"  Trades        : {m['n_trades']}  (L {m['n_long']} / S {m['n_short']})")
        print(f"  Win rate      : {m['win_rate']*100:.1f}%   (random-2R baseline ~33.3%)")
        print(f"  Expectancy    : {m['expectancy_R']:+.3f} R/trade  | PF {m['profit_factor']:.2f}"
              f"  | totaal {m['total_R']:+.1f} R")
        verdict = ("INCONCLUSIVE (<30)" if not floor_ok else
                   ("VOLDOET" if succ else "voldoet NIET"))
        print(f"  -> {verdict}\n")
        if args.dump and use_ind:
            for t in sorted(trades, key=lambda x: x["entry_time"]):
                print(f"    {t['sym']:8} {str(t['entry_time'])[:16]:16} {t['dir']:>2} "
                      f"{t['reason']:>4} {t['r']:+6.2f}R")


if __name__ == "__main__":
    main()
