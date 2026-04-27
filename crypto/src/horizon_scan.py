"""
Fase — Horizon Scan
Vergelijkt modelprestaties bij verschillende voorspellingstijdshorizons (12h, 24h, 48h).

De feature matrix blijft ongewijzigd — alleen de target variabele wordt herberekend
per horizon (geen extra feature-rebuild nodig). Dit maakt de scan snel.

Uitvoer: CSV + lijngrafieken per metriek in data/stats/horizon_scan.*

Gebruik:
  python -m src.horizon_scan
  of via main.py --phase horizon_scan
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

import config
from src.backtest import compute_metrics, run_backtest
from src.model import optimize_threshold, time_split_with_validation


def scan_horizons(
    df_features: pd.DataFrame,
    horizons: list = None,
) -> pd.DataFrame:
    """
    Train en evalueer een Random Forest voor elke horizon in `horizons`.
    De 'close' kolom in df_features wordt gebruikt om de target te herberekenen.

    Parameters
    ----------
    df_features : feature matrix met 'close' kolom (uitvoer van build_features)
    horizons    : lijst van horizons in uren (standaard: config.HORIZON_SCAN)

    Returns
    -------
    pd.DataFrame : vergelijkingstabel gesorteerd op ROC AUC
    """
    if horizons is None:
        horizons = config.HORIZON_SCAN

    if "close" not in df_features.columns:
        raise ValueError("df_features moet een 'close' kolom bevatten voor horizon herberekening.")

    # Gebruik alleen features die ook echt beschikbaar zijn
    available_features = [c for c in config.FEATURE_COLS if c in df_features.columns]
    if len(available_features) < len(config.FEATURE_COLS):
        missing = set(config.FEATURE_COLS) - set(available_features)
        print(f"  Waarschuwing: {len(missing)} features ontbreken — worden overgeslagen: {missing}")

    results = []

    for h in horizons:
        print(f"\n{'─'*50}")
        print(f"  Horizon {h}h")
        print(f"{'─'*50}")

        # Herbereken target voor deze horizon
        df = df_features.copy()
        df["target"] = (df["close"].shift(-h) > df["close"]).astype(int)
        df = df.dropna(subset=available_features + ["target", "close"])

        if len(df) < 2000:
            print(f"  Te weinig data ({len(df)} rijen) — sla over")
            continue

        try:
            train, val, test = time_split_with_validation(df)
        except ValueError as e:
            print(f"  Overgeslagen: {e}")
            continue

        print(f"  Train: {len(train)} rijen  ({train.index[0].date()} → {train.index[-1].date()})")
        print(f"  Test : {len(test)} rijen   ({test.index[0].date()} → {test.index[-1].date()})")
        print(f"  Target (train): stijging={train['target'].mean():.1%}")

        model = RandomForestClassifier(
            n_estimators=400,
            max_depth=5,
            min_samples_leaf=80,
            max_features="sqrt",
            n_jobs=-1,
            random_state=42,
        )
        model.fit(train[available_features], train["target"])

        # Threshold optimaliseren op validatieset
        # Tijdelijk FEATURE_COLS overschrijven zodat optimize_threshold de juiste kolommen gebruikt
        _orig = config.FEATURE_COLS
        config.FEATURE_COLS = available_features
        opt_thr = optimize_threshold(model, val)
        config.FEATURE_COLS = _orig

        probas  = model.predict_proba(test[available_features])[:, 1]
        auc     = roc_auc_score(test["target"], probas)

        # Backtest met de correcte horizon
        bt = run_backtest(
            test, probas,
            threshold=opt_thr,
            use_short=False,
            use_position_sizing=True,
            regime_filter=True,
            horizon=h,
        )
        m = compute_metrics(bt, horizon=h)

        startkapitaal = 1000
        eindkapitaal  = startkapitaal * bt["cum_strategy"].iloc[-1]
        bh_eind       = startkapitaal * bt["cum_buy_hold"].iloc[-1]

        print(f"  ROC AUC      : {auc:.4f}")
        print(f"  Opt. threshold: {opt_thr:.2f}")
        print(f"  Sharpe Ratio : {m['sharpe_ratio']:+.3f}")
        print(f"  Total Return : {m['total_return']:+.1%}")
        print(f"  Buy & Hold   : {m['buy_hold_return']:+.1%}")
        print(f"  Max Drawdown : {m['max_drawdown']:.1%}")
        print(f"  Win Rate     : {m['win_rate']:.1%}")
        print(f"  Trades       : {m['n_trades']}")
        print(f"  EUR 1000 → EUR {eindkapitaal:.2f}  (B&H: EUR {bh_eind:.2f})")

        results.append({
            "horizon_h":        h,
            "roc_auc":          round(auc, 4),
            "opt_threshold":    opt_thr,
            "sharpe_ratio":     round(m["sharpe_ratio"], 4),
            "total_return":     round(m["total_return"], 4),
            "annualized_return": round(m["annualized_return"], 4),
            "buy_hold_return":  round(m["buy_hold_return"], 4),
            "win_rate":         round(m["win_rate"], 4),
            "n_trades":         m["n_trades"],
            "max_drawdown":     round(m["max_drawdown"], 4),
        })

    if not results:
        print("Geen resultaten — controleer of er voldoende data is.")
        return pd.DataFrame()

    scan_df = pd.DataFrame(results).sort_values("roc_auc", ascending=False)

    # Samenvatting
    print(f"\n{'='*50}")
    print("  HORIZON SCAN SAMENVATTING")
    print(f"{'='*50}")
    best = scan_df.iloc[0]
    print(f"  Beste horizon (ROC AUC): {int(best['horizon_h'])}h  "
          f"(AUC={best['roc_auc']:.4f}, Sharpe={best['sharpe_ratio']:+.3f}, "
          f"Return={best['total_return']:+.1%})")

    # Opslaan
    out      = config.DATA_DIR / "stats"
    csv_path = out / "horizon_scan.csv"
    scan_df.to_csv(csv_path, index=False)
    print(f"\n  Opgeslagen: {csv_path.name}")

    _plot_horizon_scan(scan_df, out)
    return scan_df


# ── Visualisatie ──────────────────────────────────────────────────────────────

def _plot_horizon_scan(scan_df: pd.DataFrame, out_dir) -> None:
    """Lijngrafieken van de vier kernmetrieken per horizon."""
    metrics = {
        "roc_auc":       "ROC AUC",
        "sharpe_ratio":  "Sharpe Ratio",
        "total_return":  "Total Return",
        "win_rate":      "Win Rate",
    }

    fig, axes = plt.subplots(1, len(metrics), figsize=(16, 5))
    fig.suptitle(
        "Horizon Scan — Modelprestaties per Voorspellingstijdshorizon",
        fontsize=13, fontweight="bold",
    )

    for ax, (col, label) in zip(axes, metrics.items()):
        ax.plot(
            scan_df["horizon_h"], scan_df[col],
            marker="o", lw=2, color="steelblue", markersize=8,
        )
        # Markeer het beste punt
        best_idx = scan_df["roc_auc"].idxmax()
        ax.scatter(
            scan_df.loc[best_idx, "horizon_h"],
            scan_df.loc[best_idx, col],
            color="darkorange", s=120, zorder=5, label="Beste (ROC AUC)",
        )
        ax.set_title(label, fontweight="bold", fontsize=11)
        ax.set_xlabel("Horizon (uur)")
        ax.set_ylabel(label)
        ax.set_xticks(scan_df["horizon_h"])
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)

    plt.tight_layout()
    path = out_dir / "horizon_scan.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Opgeslagen: {path.name}")


if __name__ == "__main__":
    df = pd.read_parquet(config.DATA_DIR / "features.parquet")
    scan_df = scan_horizons(df)
    if not scan_df.empty:
        print("\n=== Horizon Scan Resultaten ===")
        print(scan_df.to_string(index=False))
