"""
Honest holdout — Check 2 uit docs/crypto/HARNESS_FIX_PLAN.md

Doel: meet de EERLIJKE out-of-sample voorspellende inhoud van het 1h
feature/model-pakket, met de twee grootste backtest-leaks verwijderd:

  1. Hyperparams-op-de-evalset  -> we gebruiken DEFAULT params (geen Optuna).
  2. Threshold-op-de-testfold    -> we meten AUC (threshold-vrij), pure
     forward-predictie in een expanding walk-forward met purge tussen
     train en test (geen overlap via het 24u-label).

Rapporteert de GEPOOLDE OOS ROC AUC in drie varianten:
  A. default params, volledige FEATURE_COLS
  B. getunede params (lgb_best_params.json) — toont hoeveel de tuning oppompt
  C. default params, LEAK-REDUCED (zonder p1_probability + same-day macro)

Verdict-drempels (vooraf vastgelegd in het plan):
  AUC >= 0.55  -> signaal aanwezig, eerlijke harness rechtvaardigt verder werk
  0.52-0.55    -> marginaal
  AUC <  0.52  -> niet te onderscheiden van muntje -> stop BTC als geldverdiener

Gebruik (vanuit crypto/, na 'features'-fase):
  python scripts/honest_holdout.py [--months 18] [--fold-days 30]
"""

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd

# Zorg dat crypto/ (de parent van scripts/) op het pad staat, zodat dit script
# werkt vanuit elke cwd — zowel lokaal (python scripts/honest_holdout.py) als in
# CI (working-directory: crypto, PYTHONPATH = repo-root).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config  # noqa: E402  (na sys.path-bootstrap)

# Kolommen die de audit als leak-verdacht markeerde (Fase 3 van het plan):
LEAKY_COLS = [
    "p1_probability",        # heatmap-feature (op globale train-set gebouwd)
    "vix_level",             # same-day daily close op ochtendbars
    "usdjpy_return_24h",
    "usdjpy_return_7d",
    "dxy_return_24h",
    "dxy_return_7d",
]

DEFAULT_PARAMS = dict(
    n_estimators=400, learning_rate=0.03, num_leaves=31,
    min_child_samples=50, subsample=0.8, colsample_bytree=0.8,
    random_state=42, n_jobs=-1, verbose=-1,
)


def _auc(y_true, y_score):
    from sklearn.metrics import roc_auc_score
    if len(set(y_true)) < 2:
        return None
    return roc_auc_score(y_true, y_score)


def _fit_predict(train, test, cols, params):
    import lightgbm as lgb
    m = lgb.LGBMClassifier(**params)
    m.fit(train[cols], train["target"])
    return m.predict_proba(test[cols])[:, 1]


def walk_forward(df, cols, params, months, fold_days, horizon):
    """Expanding WF, purge=horizon. Retourneert (pooled_y, pooled_p, fold_aucs)."""
    test_h = fold_days * 24
    n_folds = months  # ~1 fold per maand
    start = len(df) - n_folds * test_h
    if start <= horizon:
        # te weinig data: verklein aantal folds
        n_folds = max(1, (len(df) - test_h - horizon) // test_h)
        start = len(df) - n_folds * test_h
    pooled_y, pooled_p, fold_aucs = [], [], []
    for i in range(n_folds):
        a = start + i * test_h
        b = a + test_h
        train = df.iloc[: a - horizon]
        test = df.iloc[a:b]
        if len(train) < 2000 or len(test) == 0:
            continue
        p = _fit_predict(train, test, cols, params)
        y = test["target"].tolist()
        pooled_y += y
        pooled_p += list(p)
        fa = _auc(y, p)
        if fa is not None:
            fold_aucs.append((str(test.index[0].date()), str(test.index[-1].date()), fa, len(test)))
    return pooled_y, pooled_p, fold_aucs


def verdict(auc):
    if auc is None:
        return "ONBEPAALD"
    if auc >= 0.55:
        return "SIGNAAL AANWEZIG (>=0.55)"
    if auc >= 0.52:
        return "MARGINAAL (0.52-0.55)"
    return "MUNTJE (<0.52) -> geen edge"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--months", type=int, default=18, help="aantal OOS-folds (~maanden)")
    ap.add_argument("--fold-days", type=int, default=30)
    ap.add_argument("--symbol", default=config.SYMBOL)
    args = ap.parse_args()

    horizon = config.PREDICTION_HORIZON_H
    feat_path = config.symbol_path(args.symbol, "features.parquet")
    if not feat_path.exists():
        raise SystemExit(f"features.parquet ontbreekt: {feat_path}\n"
                         f"Draai eerst: python main.py --phase features")

    df = pd.read_parquet(feat_path).sort_index()
    df = df.dropna(subset=["target"])
    cols_full = [c for c in config.FEATURE_COLS if c in df.columns]
    cols_lean = [c for c in cols_full if c not in LEAKY_COLS]

    print("=" * 64)
    print(f"HONEST HOLDOUT — {args.symbol} (1h, horizon {horizon}u)")
    print(f"Feature-matrix: {len(df)} rijen, {df.index[0].date()} -> {df.index[-1].date()}")
    print(f"FEATURE_COLS: {len(cols_full)} (leak-reduced: {len(cols_lean)})")
    up_rate = df["target"].mean()
    print(f"Up-rate (base rate): {up_rate*100:.1f}%  -> 'altijd long' accuracy = {max(up_rate,1-up_rate)*100:.1f}%")
    print(f"WF: {args.months} folds × {args.fold_days}d, expanding, purge={horizon}u, DEFAULT params\n")

    # Laad getunede params
    tuned = None
    tp = config.symbol_path(args.symbol, "lgb_best_params.json")
    if tp.exists():
        try:
            raw = json.loads(tp.read_text())
            tuned = {**raw, "random_state": 42, "n_jobs": -1, "verbose": -1}
            tuned.setdefault("n_estimators", 400)
        except Exception as e:
            print(f"  (lgb_best_params.json niet bruikbaar: {e})")

    runs = [("A. default / volledige features", cols_full, DEFAULT_PARAMS)]
    if tuned:
        runs.append(("B. GETUNED / volledige features", cols_full, tuned))
    runs.append(("C. default / LEAK-REDUCED", cols_lean, DEFAULT_PARAMS))

    report_lines = ["# Honest holdout — Check 2", "",
                    f"- Symbool: {args.symbol} (1h, horizon {horizon}u)",
                    f"- Matrix: {len(df)} rijen, {df.index[0].date()} → {df.index[-1].date()}",
                    f"- WF: {args.months} folds × {args.fold_days}d, expanding, purge {horizon}u",
                    f"- Up-rate (base): {up_rate*100:.1f}%", "",
                    "| Variant | Gepoolde OOS AUC | Fold-AUC mediaan | Folds | Verdict |",
                    "|---|---|---|---|---|"]

    pooled_results = {}
    for name, cols, params in runs:
        py, pp, fa = walk_forward(df, cols, params, args.months, args.fold_days, horizon)
        pooled = _auc(py, pp)
        med = sorted(a[2] for a in fa)[len(fa) // 2] if fa else None
        pooled_results[name] = pooled
        v = verdict(pooled)
        print(f"--- {name} ---")
        print(f"  gepoolde OOS AUC : {pooled:.4f}" if pooled else "  gepoolde OOS AUC : n/a")
        print(f"  fold-AUC mediaan : {med:.4f}" if med else "  fold-AUC mediaan : n/a")
        print(f"  folds            : {len(fa)} ({len(py)} OOS-rijen)")
        print(f"  VERDICT          : {v}\n")
        report_lines.append(
            f"| {name} | {pooled:.4f} | {med:.4f} | {len(fa)} | {v} |"
            if pooled and med else f"| {name} | n/a | n/a | {len(fa)} | {v} |"
        )

    # Tuning-inflatie
    if "B. GETUNED / volledige features" in pooled_results and pooled_results["A. default / volledige features"]:
        a = pooled_results["A. default / volledige features"]
        b = pooled_results["B. GETUNED / volledige features"]
        if a and b:
            report_lines += ["", f"**Tuning-delta (B − A):** {(b-a):+.4f} AUC "
                                 f"(positief = de Optuna-tuning pompt de AUC op)."]
            print(f"Tuning-delta (B-A): {(b-a):+.4f} AUC")

    report_lines += ["", "## Conclusie",
                     "Gepoolde OOS AUC < 0.52 in alle varianten → geen te onderscheiden edge; "
                     "BTC-model afsluiten als geldverdiener (zie HARNESS_FIX_PLAN.md). "
                     "AUC ≥ 0.55 (default of leak-reduced) → eerlijk signaal aanwezig; "
                     "eerlijke harness + nieuwe features (SMC) gerechtvaardigd.", ""]

    out = config.DATA_DIR / "stats" / "honest_holdout.md"
    out.write_text("\n".join(report_lines), encoding="utf-8")
    print(f"Rapport opgeslagen: {out}")


if __name__ == "__main__":
    main()
