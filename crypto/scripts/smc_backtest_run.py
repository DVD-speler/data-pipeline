"""
Runner SMC event-driven backtest (docs/crypto/smc_backtest_spec.md).

Executie 1h + bias 4h, gepoold over BTC + ETH. Vergelijkt configuraties:
  inducement → + pool-targets → + SMT → bundel (pool+SMT).
Standaard DEV-set (t/m 2025-06-30); holdout met --holdout (alleen op mijlpaal).

Gebruik (vanuit crypto/):
  python scripts/smc_backtest_run.py
  python scripts/smc_backtest_run.py --holdout
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
    ap.add_argument("--holdout", action="store_true")
    args = ap.parse_args()

    ex = {s: load_ohlcv(symbol=s, interval="1h").sort_index() for s in SYMBOLS}
    bi = {s: load_ohlcv(symbol=s, interval="4h").sort_index() for s in SYMBOLS}
    other = {"BTCUSDT": "ETHUSDT", "ETHUSDT": "BTCUSDT"}

    def run(**kw):
        trades = []
        for s in SYMBOLS:
            seg = ex[s][ex[s].index >= HOLDOUT_START] if args.holdout else ex[s][ex[s].index <= DEV_END]
            bseg = bi[s][bi[s].index <= seg.index[-1]]
            smt = ex[other[s]][ex[other[s]].index <= seg.index[-1]]
            trades += run_smc_backtest(seg, bseg, avail_hours=4, df_smt=smt, **kw)
        return trades

    label = (f"HOLDOUT ({HOLDOUT_START.date()} → nu)" if args.holdout
             else f"DEV (begin → {DEV_END.date()})")
    print("=" * 64)
    print("SMC EVENT-DRIVEN — BTC+ETH gepoold @ 1h/4h  +  pool-targets / SMT")
    print(f"Segment: {label}   (random-2R baseline ~33.3%)")
    print("=" * 64)

    configs = [
        ("1. inducement (ref)", dict(use_inducement=True)),
        ("2. + pool-targets", dict(use_inducement=True, use_pool_targets=True)),
        ("3. + SMT", dict(use_inducement=True, use_smt=True)),
        ("4. bundel (pool + SMT)", dict(use_inducement=True, use_pool_targets=True, use_smt=True)),
    ]
    for name, kw in configs:
        m = metrics(run(**kw))
        if m["n_trades"] == 0:
            print(f"--- {name}: geen trades")
            continue
        floor_ok = m["n_trades"] >= 30
        succ = m["expectancy_R"] >= 0.15 and m["profit_factor"] > 1.3 and floor_ok
        verdict = "INCONCLUSIVE (<30)" if not floor_ok else ("VOLDOET ✓" if succ else "voldoet niet")
        print(f"--- {name} ---")
        print(f"    trades {m['n_trades']:3d} (L{m['n_long']}/S{m['n_short']}) | "
              f"win {m['win_rate']*100:4.1f}% | exp {m['expectancy_R']:+.3f}R | "
              f"PF {m['profit_factor']:.2f} | avgRR {m['avg_rr']:.2f} | tot {m['total_R']:+.1f}R")
        print(f"    -> {verdict}")


if __name__ == "__main__":
    main()
