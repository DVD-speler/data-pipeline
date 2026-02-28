"""
Fase 5 — ML Model
Traint een Random Forest classifier om te voorspellen of de prijs
in de komende N uur stijgt of daalt.

Splitsing op TIJD (niet random) om data-leakage te voorkomen:
  Train      → alle data t/m (totaal - VAL_DAYS - TEST_DAYS)
  Validatie  → de VALIDATION_SIZE_DAYS periode daarna   (threshold-optimalisatie)
  Test       → de laatste TEST_SIZE_DAYS dagen           (eindrapportage)

Stap G verbeteringen:
  - max_depth verlaagd van 8 → 5  (minder overfitting)
  - min_samples_leaf verhoogd van 40 → 80  (meer regularisatie)
  - n_estimators verhoogd van 300 → 400  (stabieler ensemble)

Stap C: optimize_threshold() vindt de Sharpe-maximaliserende drempelwaarde
        op de validatieset (NIET op de testset, om overfitting te vermijden).
"""

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    roc_auc_score,
    roc_curve,
)

import config


# ── Train / validatie / test split ────────────────────────────────────────────

def time_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Splits de feature matrix chronologisch in train en test.
    Test = laatste TEST_SIZE_DAYS × 24 uur candles.
    Compatibel met bestaande code die geen validatieset nodig heeft.
    """
    test_h   = config.TEST_SIZE_DAYS * 24
    split_idx = len(df) - test_h
    if split_idx <= 0:
        raise ValueError(
            f"Niet genoeg data voor een {config.TEST_SIZE_DAYS}-daagse testperiode. "
            f"Dataset heeft {len(df)} rijen."
        )
    return df.iloc[:split_idx], df.iloc[split_idx:]


def time_split_with_validation(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Splits de feature matrix in drie chronologische delen:
      train      → alles voor validatie + test
      validation → VALIDATION_SIZE_DAYS dagen (voor threshold-optimalisatie)
      test       → TEST_SIZE_DAYS dagen (voor eindrapportage)
    """
    test_h = config.TEST_SIZE_DAYS       * 24
    val_h  = config.VALIDATION_SIZE_DAYS * 24
    total_held_out = test_h + val_h

    if len(df) <= total_held_out:
        raise ValueError(
            f"Niet genoeg data. Dataset: {len(df)} rijen, "
            f"vereist minimaal {total_held_out + 100} rijen."
        )

    train = df.iloc[: len(df) - total_held_out]
    val   = df.iloc[len(df) - total_held_out : len(df) - test_h]
    test  = df.iloc[len(df) - test_h :]
    return train, val, test


# ── Threshold optimalisatie ───────────────────────────────────────────────────

def _optimize_threshold_from_probas(
    probas: np.ndarray,
    val_df: pd.DataFrame,
    thr_min: float = 0.50,
    thr_max: float = 0.75,
    min_trades: int = 10,
) -> float:
    """
    Kernfunctie: zoek de long-drempelwaarde die Sharpe maximaliseert,
    gegeven vooraf berekende kansen. Gebruikt long-only, geen position sizing.
    Gesepareerd van het model zodat ensemble-kansen ook geoptimaliseerd kunnen worden.
    """
    from src.backtest import compute_metrics, run_backtest

    best_sharpe = -np.inf
    best_thr    = config.SIGNAL_THRESHOLD

    n_steps = round((thr_max - thr_min) / 0.01)
    for thr in np.linspace(thr_min, thr_max, n_steps, endpoint=False):
        r = run_backtest(
            val_df, probas,
            threshold=float(thr),
            use_short=False,
            use_position_sizing=False,
            stop_loss=0.0,
        )
        m = compute_metrics(r)
        if m["n_trades"] >= min_trades and m["sharpe_ratio"] > best_sharpe:
            best_sharpe = m["sharpe_ratio"]
            best_thr    = float(thr)

    return round(best_thr, 2)


def optimize_threshold(
    model,
    val_df: pd.DataFrame,
    thr_min: float = 0.50,
    thr_max: float = 0.75,
    min_trades: int = 10,
) -> float:
    """
    Zoek de drempelwaarde die de Sharpe Ratio maximaliseert op de validatieset.
    Gebruikt long-only zonder position sizing om de threshold puur te beoordelen.

    Parameters
    ----------
    model      : getraind classifier (sklearn-interface)
    val_df     : validatieset (uitvoer van time_split_with_validation)
    thr_min    : ondergrens zoekruimte
    thr_max    : bovengrens zoekruimte
    min_trades : minimaal aantal trades om een threshold te accepteren

    Returns
    -------
    float : optimale drempelwaarde
    """
    probas = model.predict_proba(val_df[config.FEATURE_COLS])[:, 1]
    return _optimize_threshold_from_probas(probas, val_df, thr_min, thr_max, min_trades)


def optimize_short_threshold(
    model,
    val_df: pd.DataFrame,
    thr_min: float = 0.30,
    thr_max: float = 0.46,   # ≤0.45 = model genuinely bearish, not just uncertain
    min_trades: int = 5,
) -> float:
    """
    Zoek de kansbovengrens die Sharpe maximaliseert voor short-only signalen.
    Een short wordt getriggerd wanneer proba ≤ threshold_short EN prijs < EMA200.

    Returns 0.0 als geen drempelwaarde voldoende trades oplevert (val-periode is bullish).
    """
    from src.backtest import compute_metrics, run_backtest

    probas = model.predict_proba(val_df[config.FEATURE_COLS])[:, 1]

    best_sharpe = -np.inf
    best_thr    = 0.0   # Standaard: geen shorts

    # linspace vermijdt floating-point artefacten van np.arange met float-stap
    n_steps = round((thr_max - thr_min) / 0.01)
    for thr in np.linspace(thr_min, thr_max, n_steps, endpoint=False):
        r = run_backtest(
            val_df, probas,
            threshold=0.99,             # Blokkeer alle longs
            threshold_short=float(thr),
            use_short=True,
            use_position_sizing=False,
            stop_loss=0.0,
        )
        m = compute_metrics(r)
        if m["n_short"] >= min_trades and m["sharpe_ratio"] > best_sharpe:
            best_sharpe = m["sharpe_ratio"]
            best_thr    = float(thr)

    return round(best_thr, 2)


# ── Training ──────────────────────────────────────────────────────────────────

def train_model(df: pd.DataFrame) -> tuple:
    """
    Traint een LightGBM classifier (met fallback naar Random Forest) en evalueert
    op de testset met long + short signalen.

    Returns
    -------
    (model, test_df, probas)
    """
    train, val, test = time_split_with_validation(df)

    X_train = train[config.FEATURE_COLS]
    y_train = train["target"]
    X_test  = test[config.FEATURE_COLS]
    y_test  = test["target"]

    print(f"Train      : {len(train):>6} rijen  ({train.index[0].date()} → {train.index[-1].date()})")
    print(f"Validatie  : {len(val):>6} rijen  ({val.index[0].date()} → {val.index[-1].date()})")
    print(f"Test       : {len(test):>6} rijen  ({test.index[0].date()} → {test.index[-1].date()})")
    print(f"Target (train): stijging={y_train.mean():.1%}  daling={(1-y_train.mean()):.1%}")

    # Primair model: LightGBM met tijdsgewichten (recentere data zwaarder).
    # Fallback naar Random Forest als LightGBM niet geïnstalleerd is.
    try:
        import lightgbm as lgb
        model = lgb.LGBMClassifier(
            n_estimators=400,
            max_depth=4,           # diepere boom = meer overfit; 4 is stabieler voor ~10k rijen
            learning_rate=0.03,
            subsample=0.7,
            colsample_bytree=0.7,
            min_child_samples=80,  # hogere regularisatie = minder overfit, scherper threshold
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            verbose=-1,
        )
        n = len(X_train)
        time_weights = np.linspace(0.5, 1.0, n)
        print("\nModel trainen (LightGBM, tijdsgewichten 0.5→1.0)...")
        model.fit(X_train, y_train, sample_weight=time_weights)
        model_type = "LightGBM"
    except ImportError:
        model = RandomForestClassifier(
            n_estimators=400,
            max_depth=6,
            min_samples_leaf=60,
            max_features="sqrt",
            n_jobs=-1,
            random_state=42,
        )
        print("\nModel trainen (RandomForest — LightGBM niet gevonden)...")
        model.fit(X_train, y_train)
        model_type = "RandomForest"

    # Long threshold optimaliseren op validatieset
    print("\nLong threshold optimaliseren op validatieset...")
    optimal_thr = optimize_threshold(model, val)
    print(f"  Optimale long  threshold: {optimal_thr:.2f}")

    # Short threshold optimaliseren op validatieset
    print("Short threshold optimaliseren op validatieset...")
    optimal_short_thr = optimize_short_threshold(model, val)
    if optimal_short_thr > 0:
        print(f"  Optimale short threshold: {optimal_short_thr:.2f}")
    else:
        print("  Geen short threshold gevonden (val-periode te bullish) — shorts uitgeschakeld")

    # Sla beide thresholds op voor gebruik in backtest en live signaal
    import json
    thr_path = config.DATA_DIR / "optimal_threshold.json"
    with open(thr_path, "w") as f:
        json.dump({
            "threshold":       optimal_thr,
            "threshold_short": optimal_short_thr,
            "model":           model_type,
        }, f)

    # Evalueer op testset
    probas = model.predict_proba(X_test)[:, 1]
    y_pred = (probas >= optimal_thr).astype(int)

    print("\n=== Test Set Resultaten ===")
    print(classification_report(y_test, y_pred, target_names=["daling", "stijging"],
                                 zero_division=0))
    print(f"ROC AUC : {roc_auc_score(y_test, probas):.4f}")

    # Feature importance (genormaliseerd zodat LightGBM en RF vergelijkbaar zijn)
    raw_importance = pd.Series(model.feature_importances_, index=config.FEATURE_COLS)
    importance = (raw_importance / raw_importance.sum()).sort_values(ascending=False)
    print("\n=== Feature Importance ===")
    print(importance.to_string())

    # Willekeurig signaal baseline (statistisch significantiecheck)
    # Gebruikt long + short signalen voor een realistischer beeld
    from src.backtest import run_backtest, compute_metrics, compute_random_baseline
    bt_results = run_backtest(
        test, probas,
        threshold=optimal_thr,
        threshold_short=optimal_short_thr,
        use_short=(optimal_short_thr > 0),
        use_position_sizing=False,
        regime_filter=True,
    )
    bt_metrics    = compute_metrics(bt_results)
    rand_baseline = compute_random_baseline(bt_results)
    actual_sharpe = bt_metrics["sharpe_ratio"]
    significant   = actual_sharpe > rand_baseline["random_sharpe_p95"]

    print(f"\n=== Backtest Samenvatting (test) ===")
    print(f"  Long trades    : {bt_metrics['n_long']}")
    print(f"  Short trades   : {bt_metrics['n_short']}")
    print(f"  Win rate       : {bt_metrics['win_rate']:.1%}")
    print(f"  Totaal return  : {bt_metrics['total_return']:+.1%}  (B&H: {bt_metrics['buy_hold_return']:+.1%})")
    print(f"\n=== Significantiecheck (N=500 willekeurige signalen) ===")
    print(f"  Strategie Sharpe   : {actual_sharpe:+.3f}")
    print(f"  Random p5 / p95    : {rand_baseline['random_sharpe_p5']:+.3f} / "
          f"{rand_baseline['random_sharpe_p95']:+.3f}")
    print(f"  Statistisch signif.: {'JA' if significant else 'NEE'}")

    _plot_feature_importance(importance)
    _plot_roc_curve(y_test, probas)
    _plot_confusion_matrix(y_test, y_pred)

    model_path = config.DATA_DIR / "model.pkl"
    joblib.dump(model, model_path)
    print(f"\nModel opgeslagen: {model_path}  ({model_type})")
    print(f"Thresholds opgeslagen: {thr_path}")

    return model, test, probas


# ── Laden ─────────────────────────────────────────────────────────────────────

def load_model() -> RandomForestClassifier:
    model_path = config.DATA_DIR / "model.pkl"
    if not model_path.exists():
        raise FileNotFoundError(
            f"Geen model gevonden op {model_path}. Voer eerst train_model() uit."
        )
    return joblib.load(model_path)


def load_optimal_threshold() -> tuple[float, float]:
    """
    Laad de geoptimaliseerde long- en short-drempelwaarden.
    Valt terug op config.SIGNAL_THRESHOLD / 0.0 bij geen opgeslagen threshold.

    Returns
    -------
    (long_threshold, short_threshold)
    """
    import json
    thr_path = config.DATA_DIR / "optimal_threshold.json"
    if thr_path.exists():
        with open(thr_path) as f:
            data = json.load(f)
        return float(data["threshold"]), float(data.get("threshold_short", 0.0))
    return config.SIGNAL_THRESHOLD, 0.0


# ── Plots ─────────────────────────────────────────────────────────────────────

def _plot_feature_importance(importance: pd.Series) -> None:
    fig, ax = plt.subplots(figsize=(9, 6))
    colors = ["steelblue" if i < 5 else "lightsteelblue" for i in range(len(importance))]
    importance.plot(kind="barh", ax=ax, color=colors)
    ax.set_title("Feature Importance (LightGBM, genormaliseerd)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Relatief belang")
    ax.invert_yaxis()
    plt.tight_layout()
    out = config.DATA_DIR / "stats" / "feature_importance.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Opgeslagen: {out.name}")


def _plot_roc_curve(y_true, y_score) -> None:
    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc = roc_auc_score(y_true, y_score)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, lw=2, label=f"ROC (AUC = {auc:.3f})", color="steelblue")
    ax.plot([0, 1], [0, 1], "--", color="gray", label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve — Testset")
    ax.legend()
    plt.tight_layout()
    out = config.DATA_DIR / "stats" / "roc_curve.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Opgeslagen: {out.name}")


def _plot_confusion_matrix(y_true, y_pred) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred,
        display_labels=["daling", "stijging"],
        colorbar=False,
        ax=ax,
    )
    ax.set_title("Confusion Matrix — Testset")
    plt.tight_layout()
    out = config.DATA_DIR / "stats" / "confusion_matrix.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Opgeslagen: {out.name}")


if __name__ == "__main__":
    df = pd.read_parquet(config.DATA_DIR / "features.parquet")
    model, test_df, probas = train_model(df)
