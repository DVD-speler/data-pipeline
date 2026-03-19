"""
Fase 6 — Backtest & Signaal Generatie
Simuleert een long-only strategie (short is uitgeschakeld tot model verbetert)
op basis van modelkansen en berekent prestatiemetrieken.

Strategie:
  Long  — koop bij close van uur T als model-kans ≥ threshold
  Exit  — na PREDICTION_HORIZON_H uur (of eerder bij stop-loss)
  Fees  — 2× TRADE_FEE per trade (entry + exit)

Stap B — Regime filter:
  Alleen long gaan wanneer close > EMA200 (price_vs_ema200 > 1.0).
  Voorkomt against-the-trend long posities in een dalende markt.

Walk-forward validatie:
  Traint maandelijks opnieuw op een rollend venster van WALKFORWARD_TRAIN_DAYS.
"""

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd

import config
from src.model import load_model, load_optimal_threshold, time_split


# ── Backtest logica ───────────────────────────────────────────────────────────

def run_backtest(
    test_df: pd.DataFrame,
    probas: np.ndarray,
    threshold: float = None,
    threshold_short: float = config.SIGNAL_THRESHOLD_SHORT,
    fee: float = config.TRADE_FEE,
    stop_loss: float = config.STOP_LOSS_PCT,
    use_short: bool = False,
    use_position_sizing: bool = True,
    regime_filter: bool = True,
    horizon: int = None,
) -> pd.DataFrame:
    """
    Voer de backtest uit op de testset.

    Parameters
    ----------
    test_df              : feature matrix testdeel (bevat 'close', 'target' en features)
    probas               : model voorspelde kansen op stijging (klasse 1)
    threshold            : minimale kans voor long-signaal (None = laad optimale drempel)
    threshold_short      : maximale kans voor short-signaal (0.0 = uitgeschakeld)
    fee                  : transactiekosten per kant
    stop_loss            : maximaal verlies per trade; 0 = geen stop-loss
    use_short            : False = alleen long
    use_position_sizing  : True = positie schaalt met kanszekerheid
    regime_filter        : True = alleen long boven EMA200
    horizon              : voorspellingstijdshorizon in uur (None = config.PREDICTION_HORIZON_H)
    """
    if threshold is None:
        threshold = load_optimal_threshold()

    h = horizon if horizon is not None else config.PREDICTION_HORIZON_H

    results = test_df[["close", "target"]].copy()
    results["proba"] = probas

    # ── Regime-adaptieve drempel ──────────────────────────────────────────────
    # In bull-regimes wordt de drempel verlaagd (meer kansen), in bear verhoogd
    # (alleen longs met zeer hoge modelzekerheid). Offsets uit config.
    if "market_regime" in test_df.columns:
        regime = test_df["market_regime"].reindex(results.index, fill_value=0).astype(int)
        offsets = config.REGIME_THRESHOLD_OFFSETS
        eff_thr = regime.map(lambda r: threshold + offsets.get(r, 0.0)).clip(0.50, 0.95)
        results["signal_long"] = (results["proba"] >= eff_thr).astype(int)
    else:
        results["signal_long"] = (results["proba"] >= threshold).astype(int)
    results["signal_short"] = (
        (results["proba"] <= threshold_short).astype(int)
        if use_short and threshold_short > 0
        else pd.Series(0, index=results.index)
    )

    # ── Regime filter ─────────────────────────────────────────────────────────
    if regime_filter and "price_vs_ema200" in test_df.columns:
        above_ema200 = (test_df["price_vs_ema200"] > 1.0).reindex(results.index, fill_value=True)
        # Longs: boven EMA200 + geen bevestigde beartrend (market_regime != -1)
        # Blokkeert longs in confirmed bear (ADX > 20 en -DI > +DI), ook al is prijs > EMA200
        if "market_regime" in test_df.columns:
            not_confirmed_bear = (test_df["market_regime"] != -1).reindex(results.index, fill_value=True)
            results["signal_long"] = results["signal_long"] * above_ema200.astype(int) * not_confirmed_bear.astype(int)
        else:
            results["signal_long"] = results["signal_long"] * above_ema200.astype(int)

        # Death cross filter: EMA50 < EMA200 → longs extra geblokkeerd.
        # Detectie: ema_ratio_50 (close/ema50) > price_vs_ema200 (close/ema200) → ema50 < ema200.
        # Vangt bear-transitie op vóór de prijs onder EMA200 zakt (bull trap voorkomen).
        if "ema_ratio_50" in test_df.columns:
            no_death_cross = (
                test_df["ema_ratio_50"].reindex(results.index, fill_value=1.0)
                <= test_df["price_vs_ema200"].reindex(results.index, fill_value=1.0)
            )
            results["signal_long"] = results["signal_long"] * no_death_cross.astype(int)

        # Shorts: dubbele macro-gate
        #   1. Onder EMA200 (prijs in downtrend)
        #   2. return_30d < -3% (macro 30-daagse trend negatief)
        #   3. return_7d  < -1% (recente week ook negatief — blokkeert recovery-bounces)
        # Blokkeert shorts bij kortstondige crashes met snelle recovery (zoals aug 2025 Japan-crash)
        if use_short and "return_30d" in test_df.columns and "return_7d" in test_df.columns:
            macro_bear = (
                (test_df["return_30d"] < -0.03) & (test_df["return_7d"] < -0.01)
            ).reindex(results.index, fill_value=False)
            results["signal_short"] = results["signal_short"] * (~above_ema200 & macro_bear).astype(int)
        elif use_short and "return_30d" in test_df.columns:
            macro_bear = (test_df["return_30d"] < -0.03).reindex(results.index, fill_value=False)
            results["signal_short"] = results["signal_short"] * (~above_ema200 & macro_bear).astype(int)
        else:
            results["signal_short"] = results["signal_short"] * (~above_ema200).astype(int)

    results["signal"] = results["signal_long"] - results["signal_short"]

    # ── Basisrendement ────────────────────────────────────────────────────────
    raw_return = results["close"].shift(-h) / results["close"] - 1

    # ── Stop-loss ─────────────────────────────────────────────────────────────
    if stop_loss > 0:
        long_return  = raw_return.clip(lower=-stop_loss)
        short_return = (-raw_return).clip(lower=-stop_loss)
    else:
        long_return  = raw_return
        short_return = -raw_return

    # ── Position sizing ───────────────────────────────────────────────────────
    if use_position_sizing:
        long_size  = ((results["proba"] - 0.5) * 2).clip(0, 1)
        short_size = ((0.5 - results["proba"]) * 2).clip(0, 1)
    else:
        long_size  = results["signal_long"].astype(float)
        short_size = results["signal_short"].astype(float)

    # ── Strategie rendement ───────────────────────────────────────────────────
    results["trade_return"]    = raw_return
    results["strategy_return"] = (
        results["signal_long"]  * long_size  * long_return
        - results["signal_long"]  * long_size  * 2 * fee
        + results["signal_short"] * short_size * short_return
        - results["signal_short"] * short_size * 2 * fee
    )

    # ── Benchmark ─────────────────────────────────────────────────────────────
    # Buy & Hold: uurlijks samengesteld rendement → cum_buy_hold[-1] = close[-1]/close[0]
    # Dit is de correcte vergelijking: hoeveel groeit €1 als je simpelweg houdt?
    results["bh_return"]    = results["close"].pct_change()
    results["cum_strategy"] = (1 + results["strategy_return"].fillna(0)).cumprod()
    results["cum_buy_hold"] = (1 + results["bh_return"].fillna(0)).cumprod()

    return results


# ── Metrieken ─────────────────────────────────────────────────────────────────

def compute_metrics(results: pd.DataFrame, horizon: int = None) -> dict:
    """
    Bereken prestatiemetrieken voor de strategie.

    Parameters
    ----------
    horizon : voorspellingstijdshorizon in uur (None = config.PREDICTION_HORIZON_H)
              Wordt gebruikt voor de Sharpe-annualisatiefactor.
    """
    h = horizon if horizon is not None else config.PREDICTION_HORIZON_H
    hours_per_year = 8760

    strat_returns  = results["strategy_return"].dropna()
    active_returns = strat_returns[results["signal"] != 0]

    total_return = float(results["cum_strategy"].iloc[-1] - 1)
    bh_return    = float(results["cum_buy_hold"].iloc[-1] - 1)
    n_hours      = len(results)
    annualized   = float((1 + total_return) ** (hours_per_year / n_hours) - 1)

    if len(active_returns) > 1 and active_returns.std() > 0:
        sharpe = float(
            (active_returns.mean() / active_returns.std())
            * np.sqrt(hours_per_year / h)
        )
    else:
        sharpe = 0.0

    cum      = results["cum_strategy"]
    drawdown = cum / cum.cummax() - 1
    max_dd   = float(drawdown.min())

    n_long   = int(results["signal_long"].sum())  if "signal_long"  in results.columns else 0
    n_short  = int(results["signal_short"].sum()) if "signal_short" in results.columns else 0
    win_rate = float((active_returns > 0).mean()) if len(active_returns) > 0 else 0.0

    return {
        "total_return":      total_return,
        "buy_hold_return":   bh_return,
        "annualized_return": annualized,
        "sharpe_ratio":      sharpe,
        "max_drawdown":      max_dd,
        "win_rate":          win_rate,
        "n_trades":          n_long + n_short,
        "n_long":            n_long,
        "n_short":           n_short,
        "signal_rate":       float(results["signal"].abs().mean()),
    }


# ── Willekeurig signaal baseline ─────────────────────────────────────────────

def compute_random_baseline(
    results: pd.DataFrame,
    n_simulations: int = 500,
    seed: int = 42,
) -> dict:
    """
    Schat de kansenverdeling van Sharpe Ratio onder willekeurige signalen.

    Shuffle de signal-kolom N keer, bereken elke keer de Sharpe Ratio,
    en geef het 5e en 95e percentiel terug.  Als de werkelijke strategie-Sharpe
    onder het 95e percentiel valt, is de prestatie statistisch niet significant.

    Parameters
    ----------
    results        : uitvoer van run_backtest() (bevat 'signal', 'strategy_return')
    n_simulations  : aantal shuffle-iteraties
    seed           : willekeurigheid reproduceerbaar maken

    Returns
    -------
    dict met p5, mean, p95 en een significantie-vlag
    """
    rng     = np.random.default_rng(seed)
    sharpes = []

    signal_values  = results["signal"].values
    trade_return   = results["trade_return"].fillna(0).values
    fee            = config.TRADE_FEE
    h              = config.PREDICTION_HORIZON_H
    hours_per_year = 8760

    for _ in range(n_simulations):
        shuffled_signal = rng.permutation(signal_values)
        # Herbereken strategie-rendement met willekeurig signaal.
        # Long (+1): trade_return - 2*fee  |  Short (-1): -(trade_return) - 2*fee
        strat_ret = np.where(
            shuffled_signal > 0,  trade_return - 2 * fee,
            np.where(
            shuffled_signal < 0, -trade_return - 2 * fee, 0.0)
        )
        active = strat_ret[shuffled_signal != 0]
        if len(active) > 1 and active.std() > 0:
            s = (active.mean() / active.std()) * np.sqrt(hours_per_year / h)
        else:
            s = 0.0
        sharpes.append(s)

    p5  = float(np.percentile(sharpes, 5))
    p95 = float(np.percentile(sharpes, 95))
    mean = float(np.mean(sharpes))

    return {"random_sharpe_p5": p5, "random_sharpe_mean": mean, "random_sharpe_p95": p95}


# ── Breakeven Trailing Stop Backtest ─────────────────────────────────────────

def run_backtest_be_trail(
    test_df: pd.DataFrame,
    probas: np.ndarray,
    threshold: float = None,
    threshold_short: float = config.SIGNAL_THRESHOLD_SHORT,
    fee: float = config.TRADE_FEE,
    stop_loss_pct: float = config.STOP_LOSS_PCT,
    tp_pct: float = 0.06,
    be_trigger_pct: float = 0.02,
    allow_second_entry: bool = True,
    use_short: bool = False,
    regime_filter: bool = True,
    horizon: int = None,
) -> pd.DataFrame:
    """
    Backtest met trailing stop-loss naar breakeven (BE).

    Logica per candle:
      - Wanneer close >= entry * (1 + be_trigger_pct) → SL verplaatst naar entry (BE)
      - Als allow_second_entry=True én alle open posities op BE staan → tweede trade
        toegestaan bij nieuw signaal (max 2 simultane posities)
      - SL/TP/horizon: gedetecteerd op candle-close (geen intrabar high/low beschikbaar)

    Signalen worden identiek berekend als run_backtest() (inclusief regime/death-cross filters).
    Strategy_return wordt bijgehouden per exit-candle (niet entry-candle).
    """
    if threshold is None:
        threshold, _ = load_optimal_threshold()

    h = horizon if horizon is not None else config.PREDICTION_HORIZON_H

    # Genereer signalen via run_backtest (alle regime-/death-cross filters ingebakken)
    base = run_backtest(
        test_df, probas,
        threshold=threshold,
        threshold_short=threshold_short,
        fee=fee,
        stop_loss=stop_loss_pct,
        use_short=use_short,
        use_position_sizing=True,
        regime_filter=regime_filter,
        horizon=h,
    )
    signals_long  = base["signal_long"].values.astype(int)
    signals_short = base["signal_short"].values.astype(int)

    closes = test_df["close"].values
    n      = len(closes)

    # Position sizing: identiek aan run_backtest
    long_sizes  = np.clip((probas - 0.5) * 2, 0.0, 1.0)
    short_sizes = np.clip((0.5 - probas) * 2, 0.0, 1.0)

    strategy_returns = np.zeros(n)
    trade_opened     = np.zeros(n, dtype=int)   # 1 = nieuw positie geopend op deze candle
    be_triggered     = np.zeros(n, dtype=int)   # 1 = SL naar BE verplaatst op deze candle
    positions = []  # actieve posities: list of dicts

    for i in range(n):
        close = closes[i]

        # ── 1. Verwerk exits ──────────────────────────────────────────────────
        remaining = []
        for pos in positions:
            direction = pos["direction"]
            sl        = pos["sl_price"]
            tp        = pos["tp_price"]
            size      = pos["size"]
            exit_price = None

            if direction == "LONG":
                if close <= sl:
                    exit_price = sl           # SL geraakt
                elif close >= tp:
                    exit_price = tp           # TP geraakt
                elif i >= pos["horizon_idx"]:
                    exit_price = close        # horizon verlopen
            else:  # SHORT
                if close >= sl:
                    exit_price = sl
                elif close <= tp:
                    exit_price = tp
                elif i >= pos["horizon_idx"]:
                    exit_price = close

            if exit_price is not None:
                entry = pos["entry_price"]
                gross_ret = (exit_price - entry) / entry if direction == "LONG" else (entry - exit_price) / entry
                strategy_returns[i] += size * gross_ret - size * 2 * fee
            else:
                # ── BE trigger: SL verplaatsen naar entry ────────────────────
                if not pos["is_be"]:
                    if direction == "LONG" and close >= pos["entry_price"] * (1 + be_trigger_pct):
                        pos["sl_price"] = pos["entry_price"]
                        pos["is_be"]    = True
                        be_triggered[i] = 1
                    elif direction == "SHORT" and close <= pos["entry_price"] * (1 - be_trigger_pct):
                        pos["sl_price"] = pos["entry_price"]
                        pos["is_be"]    = True
                        be_triggered[i] = 1
                remaining.append(pos)

        positions = remaining

        # ── 2. Nieuwe entry toestaan? ─────────────────────────────────────────
        n_open  = len(positions)
        all_be  = all(p["is_be"] for p in positions) if positions else True

        can_long  = (n_open == 0) or (allow_second_entry and all_be and n_open < 2)
        can_short = (n_open == 0)   # shorts: nooit tweede entry

        if can_long and signals_long[i]:
            positions.append({
                "direction":   "LONG",
                "entry_price": close,
                "sl_price":    close * (1 - stop_loss_pct),
                "tp_price":    close * (1 + tp_pct),
                "is_be":       False,
                "horizon_idx": i + h,
                "size":        long_sizes[i],
            })
            trade_opened[i] += 1

        if can_short and use_short and signals_short[i]:
            positions.append({
                "direction":   "SHORT",
                "entry_price": close,
                "sl_price":    close * (1 + stop_loss_pct),
                "tp_price":    close * (1 - tp_pct),
                "is_be":       False,
                "horizon_idx": i + h,
                "size":        short_sizes[i],
            })
            trade_opened[i] += 1

    # ── Resultaten DataFrame ──────────────────────────────────────────────────
    results = test_df[["close", "target"]].copy()
    results["proba"]           = probas
    results["signal_long"]     = signals_long
    results["signal_short"]    = signals_short
    results["signal"]          = signals_long - signals_short
    results["trade_return"]    = results["close"].shift(-h) / results["close"] - 1
    results["strategy_return"] = strategy_returns
    results["trade_opened"]    = trade_opened   # 1 = entry op deze candle
    results["be_triggered"]    = be_triggered   # 1 = SL naar BE op deze candle
    results["bh_return"]       = results["close"].pct_change()
    results["cum_strategy"]    = (1 + results["strategy_return"].fillna(0)).cumprod()
    results["cum_buy_hold"]    = (1 + results["bh_return"].fillna(0)).cumprod()
    # Metadata voor diagnose (opgeslagen als attrs)
    results.attrs["be_trail_params"] = {
        "be_trigger_pct":    be_trigger_pct,
        "allow_second_entry": allow_second_entry,
        "tp_pct":            tp_pct,
        "stop_loss_pct":     stop_loss_pct,
    }

    return results


# ── Walk-forward validatie ────────────────────────────────────────────────────

def run_walkforward(
    df: pd.DataFrame,
    model_name: str = "RandomForest",
    symbol: str = config.SYMBOL,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Walk-forward backtest: traint maandelijks opnieuw op een rollend venster.
    Gebruikt per fold ook threshold-optimalisatie op een interne validatieset.
    """
    from src.model import optimize_threshold
    from src.model_compare import _get_models

    train_h = config.WALKFORWARD_TRAIN_DAYS * 24
    test_h  = config.WALKFORWARD_TEST_DAYS  * 24
    step_h  = config.WALKFORWARD_STEP_DAYS  * 24
    val_h   = config.VALIDATION_SIZE_DAYS   * 24

    models_dict = _get_models(symbol=symbol)
    if model_name not in models_dict:
        raise ValueError(f"Model '{model_name}' niet gevonden. Beschikbaar: {list(models_dict.keys())}")

    print(f"Walk-forward validatie  ({model_name})")
    print(f"  Trainvenster : {config.WALKFORWARD_TRAIN_DAYS} dagen")
    print(f"  Testvenster  : {config.WALKFORWARD_TEST_DAYS}  dagen")

    fold_metrics_list = []
    all_results_list  = []
    fold  = 0
    start = 0

    while start + train_h + test_h <= len(df):
        # Reserveer de laatste val_h van het trainvenster als interne validatieset
        train_end = start + train_h
        train = df.iloc[start : train_end - val_h]
        val   = df.iloc[train_end - val_h : train_end]
        test  = df.iloc[train_end : train_end + test_h]

        if len(train) < 500 or len(val) < 100:
            start += step_h
            fold  += 1
            continue

        model_cls, _ = models_dict[model_name]
        model = model_cls.__class__(**model_cls.get_params())

        n = len(train)
        time_weights = np.linspace(0.5, 1.0, n)
        if model_name in ("LightGBM", "XGBoost"):
            model.fit(train[config.FEATURE_COLS], train["target"],
                      sample_weight=time_weights)
        else:
            model.fit(train[config.FEATURE_COLS], train["target"])

        # Long + short threshold per fold optimaliseren op interne validatieset
        from src.model import optimize_short_threshold
        opt_thr       = optimize_threshold(model, val)
        opt_short_thr = optimize_short_threshold(model, val)
        probas        = model.predict_proba(test[config.FEATURE_COLS])[:, 1]

        results = run_backtest(
            test, probas,
            threshold=opt_thr,
            threshold_short=opt_short_thr,
            use_short=(opt_short_thr > 0),
            use_position_sizing=True,
            regime_filter=True,
        )
        metrics = compute_metrics(results)
        metrics["fold"]          = fold
        metrics["opt_thr"]       = opt_thr
        metrics["opt_short_thr"] = opt_short_thr
        metrics["test_start"]    = str(test.index[0].date())
        metrics["test_end"]      = str(test.index[-1].date())

        fold_metrics_list.append(metrics)
        all_results_list.append(results)

        short_str = f"/s{opt_short_thr:.2f}" if opt_short_thr > 0 else ""
        print(
            f"  Fold {fold:>2}  [{metrics['test_start']} → {metrics['test_end']}]"
            f"  thr={opt_thr:.2f}{short_str}"
            f"  Sharpe: {metrics['sharpe_ratio']:+.3f}"
            f"  Return: {metrics['total_return']:+.1%}"
            f"  L:{metrics['n_long']} S:{metrics['n_short']}"
        )

        start += step_h
        fold  += 1

    if not fold_metrics_list:
        raise ValueError("Niet genoeg data voor walk-forward validatie.")

    fold_df = pd.DataFrame(fold_metrics_list)
    all_res = pd.concat(all_results_list)

    print(f"\n  Walk-forward samenvatting ({fold} folds):")
    print(f"    Gemiddelde Sharpe  : {fold_df['sharpe_ratio'].mean():+.4f}")
    print(f"    Gemiddeld Return   : {fold_df['total_return'].mean():+.2%}")
    print(f"    Gem. Win Rate      : {fold_df['win_rate'].mean():.2%}")
    print(f"    Positieve folds    : {(fold_df['sharpe_ratio'] > 0).sum()}/{len(fold_df)}")

    out      = config.DATA_DIR / "stats"
    csv_path = out / f"walkforward_{model_name.lower()}.csv"
    fold_df.to_csv(csv_path, index=False)
    print(f"\n  Fold-metrieken opgeslagen: {csv_path.name}")
    _plot_walkforward(fold_df, all_res, model_name, out)

    return fold_df, all_res


# ── Visualisaties ─────────────────────────────────────────────────────────────

def plot_results(results: pd.DataFrame, metrics: dict, title_suffix: str = "") -> None:
    fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True)

    ax = axes[0]
    ax.plot(results.index, results["cum_strategy"], lw=1.8,
            label="Strategie", color="steelblue")
    ax.plot(results.index, results["cum_buy_hold"], lw=1.2,
            label="Buy & Hold", color="orange", alpha=0.75)
    ax.set_title(f"Cumulatief Rendement: Strategie vs Buy & Hold{title_suffix}",
                 fontweight="bold")
    ax.set_ylabel("Groei van €1")
    ax.legend(loc="upper left")
    ax.yaxis.set_major_formatter(mtick.FormatStrFormatter("%.2fx"))

    lines = [
        f"Total Return   : {metrics['total_return']:+.1%}",
        f"Buy & Hold     : {metrics['buy_hold_return']:+.1%}",
        f"Ann. Return    : {metrics['annualized_return']:+.1%}",
        f"Sharpe Ratio   : {metrics['sharpe_ratio']:.2f}",
        f"Max Drawdown   : {metrics['max_drawdown']:.1%}",
        f"Win Rate       : {metrics['win_rate']:.1%}",
        f"Long Trades    : {metrics['n_long']}",
        f"Short Trades   : {metrics['n_short']}",
        f"Signal Rate    : {metrics['signal_rate']:.1%}",
    ]
    ax.text(
        0.01, 0.98, "\n".join(lines),
        transform=ax.transAxes,
        fontsize=8.5, verticalalignment="top",
        fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
    )

    cum      = results["cum_strategy"]
    drawdown = (cum / cum.cummax() - 1) * 100
    axes[1].fill_between(results.index, drawdown, 0, color="crimson", alpha=0.45)
    axes[1].set_title("Drawdown (%)")
    axes[1].set_ylabel("Drawdown %")
    axes[1].yaxis.set_major_formatter(mtick.FormatStrFormatter("%.1f%%"))

    plt.tight_layout()
    out = config.DATA_DIR / "stats" / "backtest_results.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Opgeslagen: {out.name}")


def _plot_walkforward(
    fold_df: pd.DataFrame,
    all_results: pd.DataFrame,
    model_name: str,
    out_dir,
) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(14, 9))

    ax = axes[0]
    colors = ["steelblue" if s >= 0 else "crimson" for s in fold_df["sharpe_ratio"]]
    ax.bar(fold_df["fold"], fold_df["sharpe_ratio"], color=colors)
    ax.axhline(0, color="gray", lw=0.8, linestyle="--")
    ax.axhline(fold_df["sharpe_ratio"].mean(), color="navy", lw=1.5,
               linestyle="--", label=f"Gemiddeld: {fold_df['sharpe_ratio'].mean():.3f}")
    ax.set_title(f"Walk-Forward — Sharpe Ratio per Fold ({model_name})", fontweight="bold")
    ax.set_xlabel("Fold")
    ax.set_ylabel("Sharpe Ratio")
    ax.legend()

    ax2 = axes[1]
    ax2.plot(all_results.index, all_results["cum_strategy"],
             lw=1.5, label="Strategie", color="steelblue")
    ax2.plot(all_results.index, all_results["cum_buy_hold"],
             lw=1.0, label="Buy & Hold", color="orange", alpha=0.75)
    ax2.set_title("Gecombineerd Cumulatief Rendement over alle Folds", fontweight="bold")
    ax2.set_ylabel("Groei van €1")
    ax2.yaxis.set_major_formatter(mtick.FormatStrFormatter("%.2fx"))
    ax2.legend()

    plt.tight_layout()
    path = out_dir / f"walkforward_{model_name.lower()}.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Opgeslagen: {path.name}")


# ── Live signaal ──────────────────────────────────────────────────────────────

def generate_live_signal(df_ohlcv, p1p2, p1_heatmap, direction_bias,
                         symbol: str = config.SYMBOL) -> dict:
    """
    Genereer een signaal voor het meest recente uur.
    Gebruikt de geoptimaliseerde drempelwaarde en regime filter.
    """
    from src.features import build_features

    features  = build_features(df_ohlcv, p1p2, p1_heatmap, direction_bias, symbol=symbol)
    model     = load_model(symbol=symbol)
    threshold, threshold_short = load_optimal_threshold(symbol=symbol)

    last_row  = features.iloc[[-1]]
    proba     = float(model.predict_proba(last_row[config.FEATURE_COLS])[0, 1])

    # Regime-check: prijs boven EMA200
    regime_ok = (
        float(last_row["price_vs_ema200"].iloc[0]) > 1.0
        if "price_vs_ema200" in last_row.columns
        else True
    )

    # Regime-adaptieve drempel: verhoog drempel in bear, verlaag in bull
    market_regime = 0
    if "market_regime" in last_row.columns:
        market_regime = int(last_row["market_regime"].iloc[0])
    regime_offset    = config.REGIME_THRESHOLD_OFFSETS.get(market_regime, 0.0)
    eff_threshold    = float(np.clip(threshold + regime_offset, 0.50, 0.95))

    # Death cross: EMA50 < EMA200 → extra blokkade (bull trap voorkomen)
    death_cross = False
    if "ema_ratio_50" in last_row.columns and "price_vs_ema200" in last_row.columns:
        death_cross = (
            float(last_row["ema_ratio_50"].iloc[0])
            > float(last_row["price_vs_ema200"].iloc[0])
        )

    if proba >= eff_threshold and regime_ok and not death_cross:
        signaal = "LONG"
    elif proba <= threshold_short and threshold_short > 0 and not regime_ok:
        signaal = "SHORT"
    elif death_cross and proba >= eff_threshold:
        signaal = "WACHT (death cross — EMA50 onder EMA200)"
    elif proba >= eff_threshold and not regime_ok:
        signaal = "WACHT (onder EMA200 — long geblokkeerd)"
    else:
        signaal = "WACHT"

    regime_labels = {1: "bull", 0: "ranging", -1: "bear"}
    return {
        "tijdstip":            str(last_row.index[0]),
        "signaal":             signaal,
        "kans_stijging":       f"{proba:.1%}",
        "long_threshold":      f"{eff_threshold:.2f}",
        "long_threshold_base": f"{threshold:.2f}",
        "short_threshold":     f"{threshold_short:.2f}" if threshold_short > 0 else "uitgeschakeld",
        "regime_boven_ema200": regime_ok,
        "market_regime":       regime_labels.get(market_regime, "onbekend"),
        "death_cross":         death_cross,
        "horizon":             f"{config.PREDICTION_HORIZON_H} uur",
        "prijs":               float(last_row["close"].iloc[0]),
    }


if __name__ == "__main__":
    features = pd.read_parquet(config.DATA_DIR / "features.parquet")
    _, test_df = time_split(features)

    model  = load_model()
    probas = model.predict_proba(test_df[config.FEATURE_COLS])[:, 1]
    long_thr, short_thr = load_optimal_threshold()

    results = run_backtest(test_df, probas, threshold=long_thr,
                           threshold_short=short_thr, use_short=(short_thr > 0))
    metrics = compute_metrics(results)

    print("\n=== Backtest Resultaten ===")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k:<22}: {v:.4f}")
        else:
            print(f"  {k:<22}: {v}")

    plot_results(results, metrics)
