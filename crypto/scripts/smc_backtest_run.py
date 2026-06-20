"""
Runner voor de event-driven SMC-backtest (docs/crypto/smc_backtest_spec.md).

Draait standaard op de DEV-set (t/m 2025-06-30). De holdout (2025-07-01 → nu)
blijft vergrendeld tijdens ontwikkeling — gebruik --holdout pas op een mijlpaal.

Gebruik (vanuit crypto/, na OHLCV 4h+1d in ohlcv.db):
  python scripts/smc_backtest_run.py            # dev-set
  python scripts/smc_backtest_run.py --holdout  # PAS aanraken op een mijlpaal
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="BTCUSDT")
    ap.add_argument("--holdout", action="store_true",
                    help="Draai op de VERGRENDELDE holdout (alleen op mijlpalen!)")
    ap.add_argument("--dump", action="store_true", help="Print elke trade (diagnose)")
    args = ap.parse_args()

    df4 = load_ohlcv(symbol=args.symbol, interval="4h").sort_index()
    df1 = load_ohlcv(symbol=args.symbol, interval="1d").sort_index()

    if args.holdout:
        seg4 = df4[df4.index >= HOLDOUT_START]
        label = f"HOLDOUT ({HOLDOUT_START.date()} → nu)"
    else:
        seg4 = df4[df4.index <= DEV_END]
        label = f"DEV (begin → {DEV_END.date()})"

    # daily bias mag de volledige (causale) daily-reeks t/m het segment gebruiken
    df1_seg = df1[df1.index <= seg4.index[-1]]

    trades = run_smc_backtest(seg4, df1_seg)
    m = metrics(trades)

    print("=" * 60)
    print(f"SMC EVENT-DRIVEN BACKTEST — {args.symbol} @ 4h/daily")
    print(f"Segment: {label}  ({len(seg4)} 4h-bars)")
    print("=" * 60)
    if m["n_trades"] == 0:
        print("Geen trades gedetecteerd.")
        return
    if args.dump:
        print(f"  {'entry_time':16} {'dir':3} {'risk%':>6} {'bars':>4} {'reason':>6} {'R':>7}")
        for t in trades:
            print(f"  {str(t['entry_time'])[:16]:16} {t['dir']:>3} "
                  f"{t['risk_pct']*100:6.2f} {t['bars_held']:4d} {t['reason']:>6} {t['r']:+7.2f}")
        import numpy as _np
        rp = _np.array([t["risk_pct"] for t in trades]) * 100
        bh = _np.array([t["bars_held"] for t in trades])
        ntp = sum(t["reason"] == "TP" for t in trades)
        print(f"  --- risk% med {_np.median(rp):.2f} (min {rp.min():.2f} / max {rp.max():.2f}) "
              f"| bars_held med {int(_np.median(bh))} | TP {ntp}/{len(trades)}")
        print()
    print(f"  Trades        : {m['n_trades']}  (L {m['n_long']} / S {m['n_short']})")
    print(f"  Win rate      : {m['win_rate']*100:.1f}%")
    print(f"  Expectancy    : {m['expectancy_R']:+.3f} R / trade  (ná kosten)")
    print(f"  Total         : {m['total_R']:+.1f} R")
    print(f"  Profit factor : {m['profit_factor']:.2f}")
    print(f"  Beste/slechtste: {m['best_R']:+.2f} R / {m['worst_R']:+.2f} R")
    yrs = (seg4.index[-1] - seg4.index[0]).days / 365.25
    print(f"  ~Trades/jaar  : {m['n_trades']/yrs:.1f}")
    print()
    # Pre-registered drempels (spec)
    floor_ok = m["n_trades"] >= 30
    succ = (m["expectancy_R"] >= 0.15 and m["profit_factor"] > 1.3 and floor_ok)
    if not floor_ok:
        print(f"  -> < 30 trades: INCONCLUSIVE (te kleine steekproef)")
    elif succ:
        print(f"  -> VOLDOET aan succescriteria (exp >= +0.15R, PF > 1.3, >= 30 trades)")
    else:
        print(f"  -> voldoet NIET aan succescriteria")


if __name__ == "__main__":
    main()
