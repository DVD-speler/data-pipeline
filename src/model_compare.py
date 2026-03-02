"""
Fase 5b — Model Vergelijking
Traint en vergelijkt drie classifiers op exact dezelfde train/validatie/test split:
  - Random Forest  (stap G: max_depth=5, min_samples_leaf=80)
  - XGBoost        (stap G: learning_rate verlaagd, regularisatie toegevoegd)
  - LightGBM       (stap G: tijdsgewogen training — recentere data zwaarder)

Outputt een vergelijkingstabel (CSV + PNG) met per model:
  ROC AUC, precision, recall, Sharpe ratio, total return, win rate, # trades.

Gebruik:
  python -m src.model_compare
  of via main.py --phase model_compare
"""

import warnings

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, roc_curve

import config
from src.model import (
    _optimize_threshold_from_probas,
    optimize_short_threshold,
    optimize_threshold,
    time_split_with_validation,
)

warnings.filterwarnings("ignore", category=UserWarning)


# ── Model definities ──────────────────────────────────────────────────────────

def _get_models(symbol: str = config.SYMBOL) -> dict:
    """
    Geeft een dict terug met naam → (model_instantie, kleur-voor-plot).
    XGBoost en LightGBM worden lazily geïmporteerd.
    """
    models = {
        "RandomForest": (
            RandomForestClassifier(
                n_estimators=400,
                max_depth=5,           # stap G: was 8
                min_samples_leaf=80,   # stap G: was 40
                max_features="sqrt",
                n_jobs=-1,
                random_state=42,
            ),
            "steelblue",
        ),
    }

    try:
        import xgboost as xgb
        models["XGBoost"] = (
            xgb.XGBClassifier(
                n_estimators=400,
                max_depth=4,           # stap G: was 6
                learning_rate=0.03,    # stap G: was 0.05 (trager leren = minder overfit)
                subsample=0.7,
                colsample_bytree=0.7,
                reg_alpha=0.1,         # stap G: L1 regularisatie (nieuw)
                reg_lambda=1.0,        # stap G: L2 regularisatie (nieuw)
                eval_metric="logloss",
                random_state=42,
                verbosity=0,
            ),
            "darkorange",
        )
    except ImportError:
        print("  XGBoost niet gevonden — sla over. (pip install xgboost)")

    try:
        import json
        import lightgbm as lgb

        # Standaard handgetuned params (fallback als Optuna nog niet gerund is)
        lgb_params: dict = {
            "n_estimators":      400,
            "max_depth":         4,
            "learning_rate":     0.03,
            "subsample":         0.7,
            "colsample_bytree":  0.7,
            "min_child_samples": 80,
            "reg_alpha":         0.1,
            "reg_lambda":        1.0,
            "random_state":      42,
            "verbose":           -1,
        }
        # Laad Optuna-geoptimaliseerde params als beschikbaar (gegenereerd door train_model)
        params_path = config.symbol_path(symbol, "lgb_best_params.json")
        if params_path.exists():
            with open(params_path) as _f:
                tuned = json.load(_f)
            lgb_params.update(tuned)
            lgb_params["random_state"] = 42
            lgb_params["verbose"]      = -1

        models["LightGBM"] = (
            lgb.LGBMClassifier(**lgb_params),
            "seagreen",
        )
    except ImportError:
        print("  LightGBM niet gevonden — sla over. (pip install lightgbm)")

    return models


# ── Vergelijking ──────────────────────────────────────────────────────────────

def compare_models(df: pd.DataFrame, symbol: str = config.SYMBOL) -> pd.DataFrame:
    """
    Train alle beschikbare modellen op dezelfde train/validatie/test split en
    vergelijk hun prestaties op classificatie- én trading-metrieken.

    Stap G — tijdsgewogen training voor LightGBM:
      Recentere trainsamples krijgen een hogere weging (lineair oplopend van 0.5 → 1.0).
      Dit laat het model meer leren van recente marktpatronen.

    Parameters
    ----------
    df : feature matrix (uitvoer van build_features)

    Returns
    -------
    pd.DataFrame : vergelijkingstabel gesorteerd op ROC AUC (aflopend)
    """
    from src.backtest import compute_metrics, run_backtest

    train, val, test = time_split_with_validation(df)
    X_train = train[config.FEATURE_COLS]
    y_train = train["target"]
    X_val   = val[config.FEATURE_COLS]
    y_val   = val["target"]
    X_test  = test[config.FEATURE_COLS]
    y_test  = test["target"]

    # Tijdsgewichten: lineair oplopend van 0.5 (oudste) naar 1.0 (nieuwste)
    n = len(X_train)
    time_weights = np.linspace(0.5, 1.0, n)

    print(f"Train      : {len(train):>6} rijen  ({train.index[0].date()} → {train.index[-1].date()})")
    print(f"Validatie  : {len(val):>6} rijen  ({val.index[0].date()} → {val.index[-1].date()})")
    print(f"Test       : {len(test):>6} rijen  ({test.index[0].date()} → {test.index[-1].date()})")

    models_dict = _get_models(symbol=symbol)
    rows     = []
    roc_data = {}
    # Bewaar val-kansen per model voor ensemble gewichten
    val_probas_per_model  = {}
    test_probas_per_model = {}

    for name, (model, color) in models_dict.items():
        print(f"\n  [{name}] Trainen...")

        # LightGBM en XGBoost ondersteunen sample_weight via fit() (stap G)
        if name in ("LightGBM", "XGBoost"):
            model.fit(X_train, y_train, sample_weight=time_weights)
        else:
            model.fit(X_train, y_train)

        # Long + short threshold optimaliseren op validatieset
        optimal_thr       = optimize_threshold(model, val)
        optimal_short_thr = optimize_short_threshold(model, val)
        short_str = f" / short={optimal_short_thr:.2f}" if optimal_short_thr > 0 else ""
        print(f"  [{name}] Threshold: long={optimal_thr:.2f}{short_str}")

        val_probas_per_model[name]  = model.predict_proba(X_val)[:, 1]
        test_probas_per_model[name] = model.predict_proba(X_test)[:, 1]
        probas = test_probas_per_model[name]
        y_pred = (probas >= optimal_thr).astype(int)

        auc = roc_auc_score(y_test, probas)
        report = classification_report(
            y_test, y_pred,
            target_names=["daling", "stijging"],
            output_dict=True,
            zero_division=0,
        )

        # Backtest met long + short signalen
        bt = run_backtest(
            test, probas,
            threshold=optimal_thr,
            threshold_short=optimal_short_thr,
            use_short=(optimal_short_thr > 0),
            use_position_sizing=False,
            regime_filter=True,
        )
        metrics = compute_metrics(bt)

        rows.append({
            "model":           name,
            "opt_threshold":   round(optimal_thr, 2),
            "opt_short_thr":   round(optimal_short_thr, 2),
            "roc_auc":         round(auc, 4),
            "precision_long":  round(report["stijging"]["precision"], 4),
            "recall_long":     round(report["stijging"]["recall"], 4),
            "f1_long":         round(report["stijging"]["f1-score"], 4),
            "sharpe_ratio":    round(metrics["sharpe_ratio"], 4),
            "total_return":    round(metrics["total_return"], 4),
            "win_rate":        round(metrics["win_rate"], 4),
            "n_trades":        metrics["n_trades"],
            "n_long":          metrics["n_long"],
            "n_short":         metrics["n_short"],
        })
        roc_data[name] = (model, probas, color)

        print(
            f"  [{name}] ROC AUC: {auc:.4f}  |  Sharpe: {metrics['sharpe_ratio']:.4f}"
            f"  |  Return: {metrics['total_return']:+.1%}"
            f"  |  L:{metrics['n_long']} S:{metrics['n_short']}"
        )

    # ── AUC-gewogen ensemble ──────────────────────────────────────────────────
    # Validatie-AUC per model bepaalt het gewicht; dit is leakage-vrij omdat
    # de base learners niet getraind zijn op de validatieset.
    print("\n  [Ensemble] AUC-gewogen gemiddelde berekenen...")
    val_aucs   = {name: roc_auc_score(y_val, vp)
                  for name, vp in val_probas_per_model.items()}
    total_auc  = sum(val_aucs.values())
    weights    = {k: v / total_auc for k, v in val_aucs.items()}

    print("  Ensemble gewichten (op val AUC):")
    for name, w in sorted(weights.items(), key=lambda x: -x[1]):
        print(f"    {name}: {w:.3f}  (val AUC = {val_aucs[name]:.4f})")

    ens_val_probas  = sum(weights[n] * val_probas_per_model[n]  for n in weights)
    ens_test_probas = sum(weights[n] * test_probas_per_model[n] for n in weights)

    ens_long_thr  = _optimize_threshold_from_probas(ens_val_probas, val)
    ens_short_thr = _optimize_threshold_from_probas(
        1 - ens_val_probas, val,    # invert: zoek drempel voor lage kansen
        thr_min=0.50, thr_max=0.70, min_trades=5,
    )
    # Converteer: ens_short_thr is de ceiling op de originele schaal
    ens_short_thr_orig = round(1 - ens_short_thr, 2)

    # Zoek direct met optimize_short_threshold variant via helper
    from src.model import optimize_short_threshold as _opt_short
    # Maak een wrapper die pre-computed kansen levert
    class _ProbaWrapper:
        def __init__(self, p): self._p = p
        def predict_proba(self, X): return np.column_stack([1 - self._p, self._p])
    _wrapper = _ProbaWrapper(ens_val_probas)
    ens_short_thr_opt = _opt_short(_wrapper, val)

    ens_auc  = roc_auc_score(y_test, ens_test_probas)
    y_ens_pred = (ens_test_probas >= ens_long_thr).astype(int)
    ens_report = classification_report(
        y_test, y_ens_pred,
        target_names=["daling", "stijging"],
        output_dict=True,
        zero_division=0,
    )
    bt_ens = run_backtest(
        test, ens_test_probas,
        threshold=ens_long_thr,
        threshold_short=ens_short_thr_opt,
        use_short=(ens_short_thr_opt > 0),
        use_position_sizing=False,
        regime_filter=True,
    )
    ens_metrics = compute_metrics(bt_ens)

    rows.append({
        "model":           "Ensemble",
        "opt_threshold":   round(ens_long_thr, 2),
        "opt_short_thr":   round(ens_short_thr_opt, 2),
        "roc_auc":         round(ens_auc, 4),
        "precision_long":  round(ens_report["stijging"]["precision"], 4),
        "recall_long":     round(ens_report["stijging"]["recall"], 4),
        "f1_long":         round(ens_report["stijging"]["f1-score"], 4),
        "sharpe_ratio":    round(ens_metrics["sharpe_ratio"], 4),
        "total_return":    round(ens_metrics["total_return"], 4),
        "win_rate":        round(ens_metrics["win_rate"], 4),
        "n_trades":        ens_metrics["n_trades"],
        "n_long":          ens_metrics["n_long"],
        "n_short":         ens_metrics["n_short"],
    })
    roc_data["Ensemble"] = (None, ens_test_probas, "mediumpurple")

    print(
        f"  [Ensemble] ROC AUC: {ens_auc:.4f}  |  Sharpe: {ens_metrics['sharpe_ratio']:.4f}"
        f"  |  Return: {ens_metrics['total_return']:+.1%}"
        f"  |  L:{ens_metrics['n_long']} S:{ens_metrics['n_short']}"
    )

    comparison = (
        pd.DataFrame(rows)
        .sort_values("roc_auc", ascending=False)
        .reset_index(drop=True)
    )

    best_name = comparison.iloc[0]["model"]
    print(f"\n  Beste model (op ROC AUC): {best_name}")

    out_dir  = config.DATA_DIR / "stats"
    csv_path = out_dir / "model_comparison.csv"
    comparison.to_csv(csv_path, index=False)
    print(f"\n  Vergelijkingstabel opgeslagen: {csv_path.name}")

    _plot_roc_comparison(y_test, roc_data, out_dir)
    _plot_metrics_comparison(comparison, out_dir)

    # Sla het beste individuele model op (Ensemble kan niet als joblib-object worden opgeslagen)
    best_single = comparison[comparison["model"] != "Ensemble"].iloc[0]["model"]
    best_model, _, _ = roc_data[best_single]
    best_path = config.symbol_path(symbol, "model_best.pkl")
    joblib.dump({"name": best_single, "model": best_model}, best_path)
    print(f"  Beste individuele model opgeslagen: {best_path.name}  ({best_single})")

    return comparison


# ── Visualisaties ─────────────────────────────────────────────────────────────

def _plot_roc_comparison(y_test, roc_data: dict, out_dir) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    for name, (model, probas, color) in roc_data.items():
        fpr, tpr, _ = roc_curve(y_test, probas)
        auc = roc_auc_score(y_test, probas)
        lw = 2.5 if name == "Ensemble" else 2
        ax.plot(fpr, tpr, lw=lw, label=f"{name} (AUC = {auc:.3f})", color=color)
    ax.plot([0, 1], [0, 1], "--", color="gray", label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve vergelijking — Testset", fontweight="bold")
    ax.legend(loc="lower right")
    plt.tight_layout()
    path = out_dir / "roc_comparison.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Opgeslagen: {path.name}")


def _plot_metrics_comparison(comparison: pd.DataFrame, out_dir) -> None:
    metrics_to_plot = {
        "roc_auc":      "ROC AUC",
        "sharpe_ratio": "Sharpe Ratio",
        "total_return": "Total Return",
        "win_rate":     "Win Rate",
    }
    fig, axes = plt.subplots(1, len(metrics_to_plot), figsize=(16, 5))
    fig.suptitle("Model Vergelijking — Testset Metrieken", fontsize=13, fontweight="bold")
    colors = ["steelblue", "darkorange", "seagreen", "mediumpurple"]

    for ax, (col, label) in zip(axes, metrics_to_plot.items()):
        vals  = comparison[col].tolist()
        names = comparison["model"].tolist()
        clrs  = colors[: len(names)]
        bars  = ax.bar(names, vals, color=clrs, edgecolor="white")
        ax.set_title(label, fontweight="bold", fontsize=11)
        ax.set_ylabel(label)
        spread = max(abs(v) for v in vals) if vals else 1
        for bar, val in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + spread * 0.02,
                f"{val:.3f}",
                ha="center", va="bottom", fontsize=9,
            )

    plt.tight_layout()
    path = out_dir / "model_comparison.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Opgeslagen: {path.name}")


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    df = pd.read_parquet(config.DATA_DIR / "features.parquet")
    comparison = compare_models(df)
    print("\n=== Model Vergelijkingstabel ===")
    print(comparison.to_string(index=False))
