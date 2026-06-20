"""
SMC-test — meet of de SMC-features (src/features_smc.py) de eerlijke
out-of-sample AUC verbeteren t.o.v. de leak-reduced baseline.

Zelfde honest harness als scripts/honest_holdout.py (expanding WF, default
params, purge=horizon). Vergelijkt drie kolomsets:
  1. baseline (leak-reduced)
  2. baseline + SMC
  3. SMC only

Gebruik (vanuit crypto/, na 'features'-fase + OHLCV in ohlcv.db):
  python scripts/smc_eval.py [--months 18] [--fold-days 30]
"""

import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config  # noqa: E402
from honest_holdout import (  # noqa: E402
    DEFAULT_PARAMS,
    LEAKY_COLS,
    _auc,
    verdict,
    walk_forward,
)
from src.data_fetcher import load_ohlcv  # noqa: E402
from src.features_smc import SMC_COLS, build_smc_features  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--months", type=int, default=18)
    ap.add_argument("--fold-days", type=int, default=30)
    ap.add_argument("--symbol", default=config.SYMBOL)
    args = ap.parse_args()

    horizon = config.PREDICTION_HORIZON_H
    feat = pd.read_parquet(config.symbol_path(args.symbol, "features.parquet")).sort_index()
    feat = feat.dropna(subset=["target"])

    ohlcv = load_ohlcv(symbol=args.symbol, interval="1h").sort_index()
    smc = build_smc_features(ohlcv)
    df = feat.join(smc, how="left")

    baseline = [c for c in config.FEATURE_COLS if c in df.columns and c not in LEAKY_COLS]
    smc_cols = [c for c in SMC_COLS if c in df.columns]

    print("=" * 64)
    print(f"SMC-TEST — {args.symbol} (1h, horizon {horizon}u)")
    print(f"Matrix: {len(df)} rijen, {df.index[0].date()} -> {df.index[-1].date()}")
    print(f"Baseline (leak-reduced): {len(baseline)} features | SMC: {len(smc_cols)} features")
    print(f"SMC-kolommen: {smc_cols}")
    # snelle sanity: niet-null aandeel + bias-verdeling
    print(f"smc_bias verdeling: {df['smc_bias'].value_counts(normalize=True).round(3).to_dict()}")
    print(f"WF: {args.months} folds × {args.fold_days}d, expanding, purge={horizon}u, DEFAULT params\n")

    runs = [
        ("1. baseline (leak-reduced)", baseline),
        ("2. baseline + SMC", baseline + smc_cols),
        ("3. SMC only", smc_cols),
    ]
    results = {}
    for name, cols in runs:
        py, pp, fa = walk_forward(df, cols, DEFAULT_PARAMS, args.months, args.fold_days, horizon)
        pooled = _auc(py, pp)
        med = sorted(a[2] for a in fa)[len(fa) // 2] if fa else None
        results[name] = pooled
        print(f"--- {name} ({len(cols)} feat) ---")
        print(f"  gepoolde OOS AUC : {pooled:.4f}" if pooled else "  gepoolde OOS AUC : n/a")
        print(f"  fold-AUC mediaan : {med:.4f}" if med else "  fold-AUC mediaan : n/a")
        print(f"  folds            : {len(fa)} ({len(py)} OOS-rijen)")
        print(f"  VERDICT          : {verdict(pooled)}\n")

    base = results["1. baseline (leak-reduced)"]
    combo = results["2. baseline + SMC"]
    if base and combo:
        delta = combo - base
        print("=" * 64)
        print(f"DELTA (baseline+SMC − baseline): {delta:+.4f} AUC")
        if delta >= 0.01:
            print("-> SMC voegt betekenisvol signaal toe. Doorbouwen op v2 gerechtvaardigd.")
        elif delta >= 0.003:
            print("-> Marginale verbetering. Mogelijk de moeite; meer SMC-features testen.")
        else:
            print("-> Geen betekenisvolle verbetering. SMC v1 voegt (nog) niets toe.")


if __name__ == "__main__":
    main()
