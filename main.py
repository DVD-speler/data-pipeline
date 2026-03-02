"""
Crypto Signal Model — Hoofdpipeline
Voer alle fases uit in volgorde, of selecteer een specifieke fase.

Gebruik:
  python main.py                           # alle fases
  python main.py --phase data              # OHLCV downloaden (alle symbolen & timeframes)
  python main.py --phase external_data     # externe data downloaden (F&G, SPX, EURUSD, funding, OI)
  python main.py --phase p1p2              # P1/P2 labels berekenen
  python main.py --phase stats             # statistieken + heatmaps
  python main.py --phase features          # feature matrix bouwen (incl. externe data)
  python main.py --phase model             # model trainen (Random Forest)
  python main.py --phase model_compare     # modellen vergelijken (RF, XGBoost, LightGBM)
  python main.py --phase backtest          # backtest uitvoeren
  python main.py --phase walkforward       # walk-forward validatie
  python main.py --phase horizon_scan      # horizons vergelijken (12h, 24h, 48h)
  python main.py --phase signal            # live signaal genereren (laatste uur)
  python main.py --phase simulation        # maand-simulaties met SL/TP, kapitaaloverzicht
  python main.py --phase live_alert       # live signaal + paper trade update + Discord alert
"""

import argparse
import json
import sys


# ── Fases ─────────────────────────────────────────────────────────────────────


def fase_data(symbol: str = None):
    print("=" * 60)
    print("FASE 1 — Data Verzameling")
    print("=" * 60)
    import config
    from src.data_fetcher import download_ohlcv, load_ohlcv

    print(f"Symbolen  : {config.SYMBOLS}")
    print(f"Timeframes: {config.INTERVALS}")
    download_ohlcv()
    # Laad primair symbool voor gebruik in volgende fases
    sym = symbol or config.SYMBOL
    df = load_ohlcv(symbol=sym)
    print(
        f"\nPrimair dataset geladen: {len(df)} candles  ({df.index[0].date()} → {df.index[-1].date()})"
    )
    return df


def fase_external_data(symbol: str = None):
    print("\n" + "=" * 60)
    print("FASE 1b — Externe Data Downloaden")
    print("=" * 60)
    from src.external_data import download_all_external

    download_all_external(force=True)


def fase_p1p2(df=None, symbol: str = None):
    print("\n" + "=" * 60)
    print("FASE 2 — P1/P2 Labels Berekenen")
    print("=" * 60)
    import config
    from src.data_fetcher import load_ohlcv
    from src.p1p2_engine import compute_p1p2

    sym = symbol or config.SYMBOL
    if df is None:
        df = load_ohlcv(symbol=sym)
    p1p2 = compute_p1p2(df)
    out = config.symbol_path(sym, "p1p2_labels.csv")
    p1p2.to_csv(out, index=False)
    print(f"Opgeslagen: {out}")
    return p1p2


def fase_stats(p1p2=None, symbol: str = None):
    print("\n" + "=" * 60)
    print("FASE 3 — Kansstatistieken & Heatmaps")
    print("=" * 60)
    import pandas as pd
    import config
    from src.data_fetcher import load_ohlcv
    from src.stats import run_stats

    sym = symbol or config.SYMBOL
    if p1p2 is None:
        p1p2 = pd.read_csv(config.symbol_path(sym, "p1p2_labels.csv"))
    df_ohlcv = load_ohlcv(symbol=sym)
    return run_stats(p1p2, df_ohlcv)


def fase_features(p1_heatmap=None, direction_bias=None, symbol: str = None):
    print("\n" + "=" * 60)
    print("FASE 4 — Feature Engineering")
    print("=" * 60)
    import pandas as pd
    import config
    from src.data_fetcher import load_ohlcv
    from src.features import build_features
    from src.stats import compute_direction_bias, compute_p1_heatmap

    sym = symbol or config.SYMBOL
    df = load_ohlcv(symbol=sym)
    p1p2 = pd.read_csv(config.symbol_path(sym, "p1p2_labels.csv"))

    if p1_heatmap is None or direction_bias is None:
        # Bepaal de train-cutoff op basis van de houdperiode (val + test).
        # De heatmaps worden uitsluitend op traindata berekend om leakage te voorkomen.
        total_holdout_h = (config.TEST_SIZE_DAYS + config.VALIDATION_SIZE_DAYS) * 24
        train_cutoff_date = df.index[-total_holdout_h].date()
        train_p1p2 = p1p2[pd.to_datetime(p1p2["date"]).dt.date < train_cutoff_date]
        print(f"  Heatmaps berekend op train-data t/m {train_cutoff_date} "
              f"({len(train_p1p2)}/{len(p1p2)} dagen)")
        if p1_heatmap is None:
            p1_heatmap = compute_p1_heatmap(train_p1p2)
        if direction_bias is None:
            direction_bias = compute_direction_bias(train_p1p2)

    features = build_features(df, p1p2, p1_heatmap, direction_bias, symbol=sym)
    out = config.symbol_path(sym, "features.parquet")
    features.to_parquet(out)
    print(f"Feature matrix: {features.shape[0]} rijen × {features.shape[1]} kolommen")
    print(f"Kolommen: {list(features.columns)}")
    print(f"Opgeslagen: {out}")
    return features


def fase_model(features=None, symbol: str = None):
    print("\n" + "=" * 60)
    print("FASE 5a — Model Trainen (LightGBM)")
    print("=" * 60)
    import pandas as pd
    import config
    from src.model import train_model

    sym = symbol or config.SYMBOL
    if features is None:
        features = pd.read_parquet(config.symbol_path(sym, "features.parquet"))
    return train_model(features, symbol=sym)


def fase_model_compare(features=None, symbol: str = None):
    print("\n" + "=" * 60)
    print("FASE 5b — Model Vergelijking (RF vs XGBoost vs LightGBM)")
    print("=" * 60)
    import pandas as pd
    import config
    from src.model_compare import compare_models

    sym = symbol or config.SYMBOL
    if features is None:
        features = pd.read_parquet(config.symbol_path(sym, "features.parquet"))

    comparison = compare_models(features, symbol=sym)

    print("\n=== Vergelijkingstabel ===")
    print(comparison.to_string(index=False))
    print(f"\nOpgeslagen in: {config.DATA_DIR / 'stats'}/")
    return comparison


def fase_backtest(model=None, test_df=None, probas=None, symbol: str = None):
    print("\n" + "=" * 60)
    print("FASE 6 — Backtest (long + short, position sizing, stop-loss)")
    print("=" * 60)
    import pandas as pd
    import config
    from src.backtest import compute_metrics, plot_results, run_backtest
    from src.model import load_model, load_optimal_threshold, time_split

    sym = symbol or config.SYMBOL
    if model is None:
        model = load_model(symbol=sym)
        features = pd.read_parquet(config.symbol_path(sym, "features.parquet"))
        _, test_df = time_split(features)
        probas = model.predict_proba(test_df[config.FEATURE_COLS])[:, 1]

    long_thr, short_thr = load_optimal_threshold(symbol=sym)
    results = run_backtest(
        test_df, probas,
        threshold=long_thr,
        threshold_short=short_thr,
        use_short=(short_thr > 0),
        use_position_sizing=True,
        regime_filter=True,
    )
    metrics = compute_metrics(results)

    print("\n=== Backtest Resultaten ===")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k:<22}: {v:.4f}")
        else:
            print(f"  {k:<22}: {v}")

    plot_results(results, metrics)
    return results, metrics


def fase_walkforward(model_name: str = "RandomForest", symbol: str = None):
    print("\n" + "=" * 60)
    print(f"FASE 6b — Walk-Forward Validatie ({model_name})")
    print("=" * 60)
    import pandas as pd
    import config
    from src.backtest import run_walkforward

    sym = symbol or config.SYMBOL
    features = pd.read_parquet(config.symbol_path(sym, "features.parquet"))
    fold_df, all_results = run_walkforward(features, model_name=model_name, symbol=sym)

    print(f"\nResultaten opgeslagen in: {config.DATA_DIR / 'stats'}/")
    return fold_df, all_results


def fase_horizon_scan(symbol: str = None):
    print("\n" + "=" * 60)
    print("FASE 7 — Horizon Scan (12h / 24h / 48h)")
    print("=" * 60)
    import pandas as pd
    import config
    from src.horizon_scan import scan_horizons

    sym = symbol or config.SYMBOL
    features = pd.read_parquet(config.symbol_path(sym, "features.parquet"))
    scan_df = scan_horizons(features)

    if not scan_df.empty:
        print("\n=== Horizon Scan Resultaten ===")
        print(scan_df.to_string(index=False))
    return scan_df


def fase_signal(symbol: str = None):
    print("\n" + "=" * 60)
    print("LIVE SIGNAAL — Meest recente uur")
    print("=" * 60)
    import pandas as pd
    import config
    from src.backtest import generate_live_signal
    from src.data_fetcher import download_ohlcv, load_ohlcv
    from src.stats import compute_direction_bias, compute_p1_heatmap

    sym = symbol or config.SYMBOL
    # Eerst updaten naar de laatste candle
    download_ohlcv()
    df = load_ohlcv(symbol=sym)
    p1p2 = pd.read_csv(config.symbol_path(sym, "p1p2_labels.csv"))
    p1_heatmap = compute_p1_heatmap(p1p2)
    direction_bias = compute_direction_bias(p1p2)

    signaal = generate_live_signal(df, p1p2, p1_heatmap, direction_bias, symbol=sym)

    print("\n" + "─" * 40)
    for k, v in signaal.items():
        print(f"  {k:<18}: {v}")
    print("─" * 40)

    # Ook opslaan als JSON
    out = config.symbol_path(sym, "latest_signal.json")
    with open(out, "w") as f:
        json.dump(signaal, f, indent=2)
    print(f"\nOpgeslagen: {out}")
    return signaal


# ── Entry point ───────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Crypto Signal Model — P1/P2 gebaseerde kansstatistieken",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--phase",
        default="all",
        choices=[
            "all",
            "data",
            "external_data",
            "p1p2",
            "stats",
            "features",
            "model",
            "model_compare",
            "backtest",
            "walkforward",
            "horizon_scan",
            "signal",
            "simulation",
            "live_alert",
        ],
        help="Welke fase uitvoeren (standaard: all)",
    )
    parser.add_argument(
        "--wf-model",
        default="LightGBM",
        choices=["RandomForest", "XGBoost", "LightGBM"],
        help="Model voor walk-forward validatie (standaard: LightGBM)",
    )
    parser.add_argument(
        "--symbol",
        default=None,
        choices=["BTCUSDT", "ETHUSDT"],
        help="Handelssymbool (standaard: BTCUSDT uit config)",
    )
    args = parser.parse_args()

    sym = args.symbol  # None = gebruik config.SYMBOL als default in elke fase

    if args.phase == "all":
        df = fase_data(symbol=sym)
        p1p2 = fase_p1p2(df, symbol=sym)
        p1_heatmap, _, direction_bias = fase_stats(p1p2, symbol=sym)
        features = fase_features(p1_heatmap, direction_bias, symbol=sym)
        model, test_df, probas = fase_model(features, symbol=sym)
        fase_backtest(model, test_df, probas, symbol=sym)
        fase_model_compare(features, symbol=sym)
        print("\n✓ Volledige pipeline succesvol afgerond.")
        print(f"  Heatmaps en grafieken staan in: data/stats/")
    elif args.phase == "data":
        fase_data(symbol=sym)
    elif args.phase == "external_data":
        fase_external_data(symbol=sym)
    elif args.phase == "p1p2":
        fase_p1p2(symbol=sym)
    elif args.phase == "stats":
        fase_stats(symbol=sym)
    elif args.phase == "features":
        fase_features(symbol=sym)
    elif args.phase == "model":
        fase_model(symbol=sym)
    elif args.phase == "model_compare":
        fase_model_compare(symbol=sym)
    elif args.phase == "backtest":
        fase_backtest(symbol=sym)
    elif args.phase == "walkforward":
        fase_walkforward(model_name=args.wf_model, symbol=sym)
    elif args.phase == "horizon_scan":
        fase_horizon_scan(symbol=sym)
    elif args.phase == "signal":
        fase_signal(symbol=sym)
    elif args.phase == "simulation":
        fase_simulation(symbol=sym)
    elif args.phase == "live_alert":
        fase_live_alert(symbol=sym)


def fase_simulation(symbol: str = None):
    print("\n" + "=" * 60)
    print("SIMULATIE — Maandoverzicht met alerts, SL/TP, kapitaal")
    print("=" * 60)
    import config
    from src.simulation import run_simulation
    run_simulation(
        initial_capital=1000.0,
        risk_pct=0.01,
        sl_pct=0.02,
        tp_pct=0.06,
        symbol=symbol or config.SYMBOL,
    )


def fase_live_alert(symbol: str = None):
    print("\n" + "=" * 60)
    print("LIVE ALERT — Signaal + paper trade update + Discord")
    print("=" * 60)
    import config
    from src.live_alert import run_live_alert
    run_live_alert(
        risk_pct=0.01,
        sl_pct=0.02,
        tp_pct=0.06,
        symbol=symbol or config.SYMBOL,
    )


if __name__ == "__main__":
    main()
