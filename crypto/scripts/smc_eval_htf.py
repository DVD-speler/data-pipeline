"""
SMC-test op hogere timeframes (4h / daily) — optie B.

Het SMC-framework leeft op HTF (MSB op de hoogste TF, executie M15+); 1h is
mogelijk te ruizig. Dit script meet of de SMC-features (src/features_smc.py)
de eerlijke OOS AUC verbeteren op 4h en daily, t.o.v. de native baseline van
dat timeframe (config_4h / config_daily FEATURE_COLS).

Timeframe-bewust: fold-sizing in BARS (niet uren), purge = target-horizon in
bars, lagere min-train voor de korte daily-reeks.

Gebruik (vanuit crypto/, na features_4h/features_daily + OHLCV in ohlcv.db):
  python scripts/smc_eval_htf.py --tf 4h
  python scripts/smc_eval_htf.py --tf 1d
"""

import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from honest_holdout import DEFAULT_PARAMS, LEAKY_COLS, _auc, verdict  # noqa: E402
from src.data_fetcher import load_ohlcv  # noqa: E402
from src.features_smc import SMC_COLS, build_smc_features  # noqa: E402


def wf(df, cols, params, n_folds, fold_bars, purge_bars, min_train):
    start = len(df) - n_folds * fold_bars
    pooled_y, pooled_p, fold_aucs = [], [], []
    for i in range(n_folds):
        a = start + i * fold_bars
        b = a + fold_bars
        if a - purge_bars <= 0:
            continue
        train = df.iloc[: a - purge_bars]
        test = df.iloc[a:b]
        if len(train) < min_train or len(test) == 0:
            continue
        import lightgbm as lgb
        m = lgb.LGBMClassifier(**params)
        m.fit(train[cols], train["target"])
        p = m.predict_proba(test[cols])[:, 1]
        y = test["target"].tolist()
        pooled_y += y
        pooled_p += list(p)
        fa = _auc(y, p)
        if fa is not None:
            fold_aucs.append(fa)
    return pooled_y, pooled_p, fold_aucs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tf", choices=["4h", "1d"], required=True)
    ap.add_argument("--symbol", default="BTCUSDT")
    args = ap.parse_args()

    if args.tf == "4h":
        import config_4h as cfg
        feat_path = cfg.symbol_path_4h(args.symbol, "features.parquet")
        feat_cols = cfg.FEATURE_COLS_4H_MODEL
        interval, purge = "4h", cfg.PREDICTION_HORIZON_4H
        n_folds, fold_bars, min_train = 18, 180, 2000   # 180 bars = 30d × 6
    else:
        import config_daily as cfg
        feat_path = cfg.symbol_path_daily(args.symbol, "features.parquet")
        feat_cols = cfg.FEATURE_COLS_DAILY
        interval, purge = "1d", cfg.PREDICTION_HORIZON_D
        n_folds, fold_bars, min_train = 15, 30, 400      # korte reeks

    feat = pd.read_parquet(feat_path).sort_index().dropna(subset=["target"])
    ohlcv = load_ohlcv(symbol=args.symbol, interval=interval).sort_index()
    smc = build_smc_features(ohlcv)
    df = feat.join(smc, how="left")

    baseline = [c for c in feat_cols if c in df.columns and c not in LEAKY_COLS]
    smc_cols = [c for c in SMC_COLS if c in df.columns]

    print("=" * 64)
    print(f"SMC-TEST HTF — {args.symbol} @ {args.tf} (horizon {purge} bars)")
    print(f"Matrix: {len(df)} rijen, {df.index[0].date()} -> {df.index[-1].date()}")
    print(f"Baseline: {len(baseline)} feat | SMC: {len(smc_cols)} feat")
    print(f"smc_bias verdeling: {df['smc_bias'].value_counts(normalize=True).round(3).to_dict()}")
    print(f"WF: {n_folds} folds × {fold_bars} bars, purge={purge}, min_train={min_train}\n")

    results = {}
    for name, cols in [
        ("1. baseline", baseline),
        ("2. baseline + SMC", baseline + smc_cols),
        ("3. SMC only", smc_cols),
    ]:
        py, pp, fa = wf(df, cols, DEFAULT_PARAMS, n_folds, fold_bars, purge, min_train)
        pooled = _auc(py, pp)
        med = sorted(fa)[len(fa) // 2] if fa else None
        results[name] = pooled
        print(f"--- {name} ({len(cols)} feat) ---")
        print(f"  gepoolde OOS AUC : {pooled:.4f}" if pooled else "  gepoolde OOS AUC : n/a")
        print(f"  fold-AUC mediaan : {med:.4f}" if med else "  fold-AUC mediaan : n/a")
        print(f"  folds            : {len(fa)} ({len(py)} OOS-rijen)")
        print(f"  VERDICT          : {verdict(pooled)}\n")

    base, combo = results["1. baseline"], results["2. baseline + SMC"]
    if base and combo:
        d = combo - base
        print("=" * 64)
        print(f"DELTA (baseline+SMC − baseline) @ {args.tf}: {d:+.4f} AUC")
        print("-> betekenisvol" if d >= 0.01 else
              ("-> marginaal" if d >= 0.003 else "-> geen verbetering"))


if __name__ == "__main__":
    main()
