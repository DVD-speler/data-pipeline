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


# ── Probability calibratie wrapper ────────────────────────────────────────────

class CalibratedWrapper:
    """
    Lichtgewicht wrapper die een isotonic-regression calibrator toepast bovenop
    een getraind model. Vervanger voor CalibratedClassifierCV(cv='prefit') dat
    verwijderd is in sklearn 1.6+.

    Serialiseerbaar via joblib (klasse op module-niveau).
    """
    def __init__(self, base_model, calibrator, feature_cols):
        self._model        = base_model
        self._calibrator   = calibrator
        self._feature_cols = feature_cols

    def predict_proba(self, X):
        import numpy as _np
        raw = self._model.predict_proba(X)[:, 1]
        cal = self._calibrator.predict(raw)
        return _np.column_stack([1 - cal, cal])

    def __getattr__(self, name):
        # Gebruik object.__getattribute__ om recursie te voorkomen:
        # self._model binnen __getattr__ zou opnieuw __getattr__ aanroepen
        # als _model nog niet in __dict__ staat (bijv. tijdens pickle restore).
        try:
            model = object.__getattribute__(self, '_model')
        except AttributeError:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
        return getattr(model, name)


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
    thr_min: float = 0.65,
    thr_max: float = 0.85,
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
    thr_min: float = 0.65,
    thr_max: float = 0.85,
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


def optimize_short_threshold(*args, **kwargs) -> float:
    """Verwijderd (B5: long-only codebase). Geeft altijd 0.0 terug."""
    return 0.0


# ── Optuna hyperparameter search ─────────────────────────────────────────────

def optuna_tune(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    val_df: pd.DataFrame,
    n_trials: int = None,
    symbol: str = config.SYMBOL,
) -> dict:
    """
    Gebruik Optuna om LightGBM hyperparameters te optimaliseren.

    S8-A: n_trials uit config.OPTUNA_N_TRIALS (default 150).
    S8-B: Objective is Sharpe op validatieset (config.OPTUNA_SHARPE_OBJECTIVE=True).
          Sharpe optimaliseert direct op handelsrendement i.p.v. discriminatievermogen.
          auto_promote_optuna() valideert vervolgens via 3-fold WF om regime-overfit te vangen.

    Returns dict met beste hyperparameters, of lege dict als Optuna/LightGBM niet beschikbaar is.
    """
    try:
        import optuna
        import lightgbm as lgb
    except ImportError:
        print("  Optuna of LightGBM niet beschikbaar — standaard parameters gebruikt.")
        return {}

    if n_trials is None:
        n_trials = getattr(config, "OPTUNA_N_TRIALS", 150)

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    n = len(X_train)
    time_weights = np.linspace(0.5, 1.0, n)
    use_sharpe = getattr(config, "OPTUNA_SHARPE_OBJECTIVE", True)

    from sklearn.metrics import roc_auc_score as _roc_auc

    def objective(trial: optuna.Trial) -> float:
        # Zoekruimte bewust beperkt tot geregulariseerde waarden.
        # Diepe bomen / lage min_child_samples → hoge val-AUC maar overfit op regimecluster val.
        # Conservatieve params → lager val-AUC maar betere walk-forward generalisatie.
        params = {
            "n_estimators":      trial.suggest_int("n_estimators", 200, 600),
            "max_depth":         trial.suggest_int("max_depth", 3, 5),
            "learning_rate":     trial.suggest_float("learning_rate", 0.01, 0.07, log=True),
            "subsample":         trial.suggest_float("subsample", 0.5, 0.85),
            "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.4, 0.70),
            "min_child_samples": trial.suggest_int("min_child_samples", 50, 200),
            "reg_alpha":         trial.suggest_float("reg_alpha", 0.0, 0.5),
            "reg_lambda":        trial.suggest_float("reg_lambda", 0.5, 4.0),
            "random_state":      42,
            "verbose":           -1,
        }
        m = lgb.LGBMClassifier(**params)
        m.fit(X_train[config.FEATURE_COLS], y_train, sample_weight=time_weights)
        probas = m.predict_proba(val_df[config.FEATURE_COLS])[:, 1]

        if use_sharpe:
            # S8-B: Sharpe-based objective — optimaliseert direct op handelsrendement.
            # Gebruik eenvoudige long-only strategie zonder gates voor snelheid.
            # Penaliseer oplossingen met < 10 trades om degeneratie (0 trades = 0.0) te voorkomen.
            try:
                from src.backtest import run_backtest, compute_metrics
                res = run_backtest(val_df, probas, use_position_sizing=False, stop_loss=0.0)
                m_stats = compute_metrics(res)
                if m_stats["n_trades"] < 10:
                    return -10.0  # hard penalty voor geen/weinig trades
                return m_stats["sharpe_ratio"]
            except Exception:
                return _roc_auc(val_df["target"], probas)
        else:
            return _roc_auc(val_df["target"], probas)

    print(f"\nOptuna hyperparameter search ({n_trials} trials, objective={'Sharpe' if use_sharpe else 'ROC AUC'})...")
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best = study.best_params
    best["random_state"] = 42
    best["verbose"]      = -1
    print(f"  Beste val-AUC: {study.best_value:.4f}")
    for k, v in best.items():
        if k not in ("random_state", "verbose"):
            print(f"    {k}: {v}")

    # Sla de Optuna-params op als CANDIDATE (niet als de actieve walkforward-params).
    import json
    saveable = {k: v for k, v in best.items() if k not in ("random_state", "verbose")}
    candidate_path = config.symbol_path(symbol, "lgb_optuna_params.json")
    with open(candidate_path, "w") as f:
        json.dump(saveable, f, indent=2)
    print(f"  Optuna kandidaat-params: {candidate_path.name}")

    return best


def _quick_wf_sharpe(df: pd.DataFrame, params: dict, n_folds: int = 3) -> float:
    """
    Mini walk-forward (n_folds) op df; geeft gemiddelde Sharpe over de folds.
    Gebruikt voor auto-promotie vergelijking (C1).
    """
    try:
        import lightgbm as lgb
        from src.backtest import compute_metrics, run_backtest
    except ImportError:
        return 0.0

    step_h  = config.WALKFORWARD_STEP_DAYS  * 24
    train_h = config.WALKFORWARD_TRAIN_DAYS * 24
    test_h  = config.WALKFORWARD_TEST_DAYS  * 24
    val_h   = config.VALIDATION_SIZE_DAYS   * 24

    n       = len(df)
    # Start zó dat we precies n_folds folds krijgen aan het einde van de dataset
    start   = n - test_h * n_folds - train_h
    if start < 0:
        return 0.0

    sharpes = []
    for _ in range(n_folds):
        train_end   = start + train_h
        train_start = max(0, train_end - train_h)
        train_data  = df.iloc[train_start : train_end - val_h]
        val_data    = df.iloc[train_end - val_h : train_end]
        test_data   = df.iloc[train_end : train_end + test_h]

        if len(train_data) < 300 or len(test_data) < 50:
            start += step_h
            continue

        full_params = {**params, "random_state": 42, "verbose": -1}
        m = lgb.LGBMClassifier(**full_params)
        n_tr = len(train_data)
        tw = np.linspace(0.5, 1.0, n_tr)
        m.fit(train_data[config.FEATURE_COLS], train_data["target"], sample_weight=tw)

        probas = m.predict_proba(test_data[config.FEATURE_COLS])[:, 1]
        from src.model import optimize_threshold as _ot
        thr = _ot(m, val_data)
        res = run_backtest(test_data, probas, threshold=thr)
        sharpes.append(compute_metrics(res)["sharpe_ratio"])
        start += step_h

    return float(np.mean(sharpes)) if sharpes else 0.0


def auto_promote_optuna(df: pd.DataFrame, symbol: str = config.SYMBOL,
                        min_improvement: float = 0.1) -> bool:
    """
    Vergelijk Optuna-kandidaat vs. huidige params via mini walk-forward (3 folds).
    Promoveer kandidaat naar lgb_best_params.json als Sharpe ≥ huidig + min_improvement.
    Logt de beslissing in data/stats/optuna_promotions.csv.

    Geeft True terug als promotie heeft plaatsgevonden.
    """
    import json, csv
    from datetime import datetime

    candidate_path = config.symbol_path(symbol, "lgb_optuna_params.json")
    stable_path    = config.symbol_path(symbol, "lgb_best_params.json")

    if not candidate_path.exists():
        print("  Geen kandidaat-params gevonden — auto-promotie overgeslagen.")
        return False

    with open(candidate_path) as f:
        candidate_params = json.load(f)

    if stable_path.exists():
        with open(stable_path) as f:
            current_params = json.load(f)
    else:
        current_params = {
            "n_estimators": 400, "max_depth": 4, "learning_rate": 0.03,
            "subsample": 0.7, "colsample_bytree": 0.6, "min_child_samples": 100,
            "reg_alpha": 0.1, "reg_lambda": 1.5,
        }

    print("\nAuto-promotie: 3-fold vergelijking huidige vs. kandidaat params...")
    current_sharpe   = _quick_wf_sharpe(df, current_params)
    candidate_sharpe = _quick_wf_sharpe(df, candidate_params)

    print(f"  Huidige params  Sharpe: {current_sharpe:+.4f}")
    print(f"  Kandidaat params Sharpe: {candidate_sharpe:+.4f}")

    promoted = candidate_sharpe >= current_sharpe + min_improvement
    if promoted:
        with open(stable_path, "w") as f:
            json.dump(candidate_params, f, indent=2)
        print(f"  Gepromoveerd naar lgb_best_params.json "
              f"(+{candidate_sharpe - current_sharpe:.4f} Sharpe verbetering)")
    else:
        print(f"  Niet gepromoveerd (verbetering {candidate_sharpe - current_sharpe:+.4f} "
              f"< drempel {min_improvement})")

    # Log beslissing
    log_path = config.DATA_DIR / "stats" / "optuna_promotions.csv"
    write_header = not log_path.exists()
    with open(log_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "timestamp", "symbol", "current_sharpe", "candidate_sharpe",
            "improvement", "promoted"
        ])
        if write_header:
            writer.writeheader()
        writer.writerow({
            "timestamp":        datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
            "symbol":           symbol,
            "current_sharpe":   round(current_sharpe, 4),
            "candidate_sharpe": round(candidate_sharpe, 4),
            "improvement":      round(candidate_sharpe - current_sharpe, 4),
            "promoted":         promoted,
        })

    return promoted


# ── Kelly Criterion positiegrootte (T2-D) ─────────────────────────────────────

def compute_kelly_fraction(win_rate: float, avg_win: float, avg_loss: float) -> float:
    """
    Bereken de Kelly-fractie: optimale positiegrootte als fractie van kapitaal.

    f = (p × b − q) / b
    Waarbij: p = win_rate, q = 1−p, b = avg_win / avg_loss (odds ratio)

    Geeft 0.0 terug als er onvoldoende data is of als de verwachte waarde negatief is.
    """
    if avg_loss <= 0 or win_rate <= 0 or win_rate >= 1:
        return 0.0
    b = avg_win / avg_loss
    q = 1.0 - win_rate
    kelly = (win_rate * b - q) / b
    return max(0.0, round(kelly, 4))


def save_kelly_sizing(
    model,
    val_df: pd.DataFrame,
    threshold: float,
    symbol: str = config.SYMBOL,
) -> dict:
    """
    Bereken Kelly-fractie op de validatieset en sla op als {symbol}_kelly.json.
    Gebruik half-Kelly als veilige positie-upper-bound in live_alert.

    Returns dict met kelly-statistieken, of leeg dict bij onvoldoende data.
    """
    import json as _json
    from src.backtest import run_backtest

    probas  = model.predict_proba(val_df[config.FEATURE_COLS])[:, 1]
    results = run_backtest(val_df, probas, threshold=threshold, use_position_sizing=False)

    active_returns = results["strategy_return"][results["signal"] != 0].dropna()
    if len(active_returns) < 10:
        print("  Kelly: onvoldoende trades op validatieset — standaard sizing gebruikt")
        return {}

    win_returns  = active_returns[active_returns > 0]
    loss_returns = active_returns[active_returns < 0]

    if len(win_returns) == 0 or len(loss_returns) == 0:
        print("  Kelly: geen verlies- of winst-trades — standaard sizing gebruikt")
        return {}

    win_rate = float((active_returns > 0).mean())
    avg_win  = float(win_returns.mean())
    avg_loss = float(abs(loss_returns.mean()))

    kelly_full = compute_kelly_fraction(win_rate, avg_win, avg_loss)
    kelly_frac = getattr(config, "KELLY_FRACTION",     0.5)
    kelly_max  = getattr(config, "KELLY_MAX_FRACTION", 0.20)
    kelly_half = min(kelly_full * kelly_frac, kelly_max)

    data = {
        "win_rate":   round(win_rate, 4),
        "avg_win":    round(avg_win,  4),
        "avg_loss":   round(avg_loss, 4),
        "kelly_full": round(kelly_full, 4),
        "kelly_half": round(kelly_half, 4),
        "n_trades":   int(len(active_returns)),
    }

    path = config.symbol_path(symbol, "kelly.json")
    with open(path, "w") as f:
        _json.dump(data, f, indent=2)

    print(f"  Kelly: win_rate={win_rate:.1%}, avg_win={avg_win:.4f}, avg_loss={avg_loss:.4f}")
    print(f"         full={kelly_full:.2%}, half={kelly_half:.2%}  (max {kelly_max:.0%} kapitaal)")
    print(f"  Opgeslagen: {path.name}")
    return data


# ── Training ──────────────────────────────────────────────────────────────────

def train_model(df: pd.DataFrame, symbol: str = config.SYMBOL) -> tuple:
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
    # Parameters worden geladen uit lgb_best_params.json (stabiel, gevalideerd via walkforward).
    # Optuna wordt als research-tool gedraaid; zijn output gaat naar lgb_optuna_params.json
    # maar overschrijft lgb_best_params.json NIET automatisch.
    # Fallback naar Random Forest als LightGBM niet geïnstalleerd is.
    try:
        import json
        import lightgbm as lgb

        # Laad stabiele params (handgetuned of eerder gevalideerd via walkforward)
        stable_path = config.symbol_path(symbol, "lgb_best_params.json")
        if stable_path.exists():
            with open(stable_path) as _f:
                lgb_params = json.load(_f)
            lgb_params["random_state"] = 42
            lgb_params["verbose"]      = -1
            print("\nStabiele LightGBM-params geladen uit lgb_best_params.json")
        else:
            lgb_params = {
                "n_estimators":      400,
                "max_depth":         4,
                "learning_rate":     0.03,
                "subsample":         0.7,
                "colsample_bytree":  0.6,
                "min_child_samples": 100,
                "reg_alpha":         0.1,
                "reg_lambda":        1.5,
                "random_state":      42,
                "verbose":           -1,
            }
            print("\nGebruik standaard LightGBM-params (geen lgb_best_params.json gevonden)")

        # Optuna: vindt kandidaat-params en probeert automatisch te promoveren (C1)
        optuna_tune(train, y_train, val, symbol=symbol)  # n_trials uit config.OPTUNA_N_TRIALS
        auto_promote_optuna(df, symbol=symbol, min_improvement=0.1)

        # Herlaad params na eventuele promotie
        if stable_path.exists():
            with open(stable_path) as _f:
                lgb_params = json.load(_f)
            lgb_params["random_state"] = 42
            lgb_params["verbose"]      = -1

        model = lgb.LGBMClassifier(**lgb_params)
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
    thr_path = config.symbol_path(symbol, "optimal_threshold.json")
    with open(thr_path, "w") as f:
        json.dump({
            "threshold":       optimal_thr,
            "threshold_short": optimal_short_thr,
            "model":           model_type,
        }, f)

    # Optimaliseer exit-proba drempelwaarden op validatieset
    from src.backtest import optimize_exit_proba
    optimize_exit_proba(model, val, optimal_thr, symbol=symbol)

    # Kelly Criterion positiegrootte berekenen op validatieset (T2-D)
    print("\nKelly Criterion positiegrootte berekenen op validatieset...")
    save_kelly_sizing(model, val, optimal_thr, symbol=symbol)

    # ── Regime-specifieke entry thresholds ────────────────────────────────────
    # Per regime (bull/ranging/bear) een aparte drempel optimaliseren op de
    # subset van de validatieset die overeenkomt met dat regime.
    # Vervangt de handmatige REGIME_THRESHOLD_OFFSETS in config.
    print("\nRegime-specifieke thresholds optimaliseren op validatieset...")
    regime_thresholds = {}
    regime_labels = {1: "bull", 0: "ranging", -1: "bear"}
    if "market_regime" in val.columns:
        for reg, lbl in regime_labels.items():
            subset = val[val["market_regime"] == reg]
            if len(subset) < 200:
                print(f"  {lbl:<8}: onvoldoende data ({len(subset)} rijen) — gebruik globale drempel")
                regime_thresholds[lbl] = optimal_thr
                continue
            reg_probas = model.predict_proba(subset[config.FEATURE_COLS])[:, 1]
            reg_thr = _optimize_threshold_from_probas(
                reg_probas, subset, thr_min=0.50, thr_max=0.90, min_trades=5
            )
            regime_thresholds[lbl] = reg_thr
            offset = reg_thr - optimal_thr
            print(f"  {lbl:<8}: {reg_thr:.2f}  (offset {offset:+.2f} vs globaal {optimal_thr:.2f})")
    else:
        for lbl in regime_labels.values():
            regime_thresholds[lbl] = optimal_thr
        print("  Geen market_regime kolom — globale drempel voor alle regimes")

    import json as _json
    reg_thr_path = config.symbol_path(symbol, "regime_thresholds.json")
    with open(reg_thr_path, "w") as f:
        _json.dump({**regime_thresholds, "global": optimal_thr}, f, indent=2)
    print(f"  Opgeslagen: {reg_thr_path.name}")

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

    model_path = config.symbol_path(symbol, "model.pkl")
    joblib.dump(model, model_path)
    print(f"\nModel opgeslagen: {model_path}  ({model_type})")
    print(f"Thresholds opgeslagen: {thr_path}")

    # ── Probability calibratie (isotonic regression op validatieset) ──────────
    # LightGBM-probabilities zijn slecht gekalibreerd. Isotonic regression
    # corrigeert dit zonder het model opnieuw te trainen.
    # Gebruik sklearn.isotonic.IsotonicRegression direct (cv='prefit' verwijderd in sklearn 1.6+).
    print("\nProbability calibratie (isotonic regression op validatieset)...")
    try:
        from sklearn.calibration import calibration_curve
        from sklearn.isotonic import IsotonicRegression

        probas_raw_val = model.predict_proba(val[config.FEATURE_COLS])[:, 1]
        ir = IsotonicRegression(out_of_bounds="clip")
        ir.fit(probas_raw_val, val["target"].values)
        calibrated = CalibratedWrapper(model, ir, config.FEATURE_COLS)

        probas_cal_val = calibrated.predict_proba(val[config.FEATURE_COLS])[:, 1]
        frac_pos, mean_pred_raw = calibration_curve(val["target"], probas_raw_val, n_bins=10)
        frac_pos_c, mean_pred_cal = calibration_curve(val["target"], probas_cal_val, n_bins=10)
        mae_raw = float(np.mean(np.abs(frac_pos   - mean_pred_raw)))
        mae_cal = float(np.mean(np.abs(frac_pos_c - mean_pred_cal)))
        print(f"  Calibratie MAE (ongekalibreerd): {mae_raw:.4f}")
        print(f"  Calibratie MAE (gekalibreerd)  : {mae_cal:.4f}  "
              f"({'beter' if mae_cal < mae_raw else 'geen verbetering'})")

        cal_path = config.symbol_path(symbol, "model_calibrated.pkl")
        joblib.dump(calibrated, cal_path)
        print(f"  Gekalibreerd model opgeslagen: {cal_path.name}")

        # Gebruik gekalibreerde probas voor verdere evaluatie op testset
        probas = calibrated.predict_proba(X_test)[:, 1]
    except Exception as e:
        print(f"  Waarschuwing: calibratie mislukt ({e}) — ongekalibreerd model gebruikt")

    return model, test, probas


# ── Regime-geconditioneerde training ──────────────────────────────────────────

def train_regime_models(df: pd.DataFrame, symbol: str = config.SYMBOL) -> dict:
    """
    Traint aparte LightGBM modellen per marktregime (bull/ranging/bear).
    Vereist kolom 'market_regime' in df (+1=bull, 0=ranging, -1=bear).

    Elke subset krijgt het algemene model als fallback als de regime-subset
    te klein is (<500 rijen na dead-zone filtering).

    Geeft dict terug: {1: bull_model, 0: ranging_model, -1: bear_model}
    Slaat modellen op als {symbol}_bull_model.pkl etc.
    """
    import json
    import lightgbm as lgb

    if "market_regime" not in df.columns:
        print("Geen market_regime kolom — regime-modellen overgeslagen.")
        return {}

    train, val, _ = time_split_with_validation(df)

    # Laad stabiele params
    stable_path = config.symbol_path(symbol, "lgb_best_params.json")
    if stable_path.exists():
        with open(stable_path) as f:
            base_params = json.load(f)
        base_params["random_state"] = 42
        base_params["verbose"]      = -1
    else:
        base_params = {
            "n_estimators": 400, "max_depth": 4, "learning_rate": 0.03,
            "subsample": 0.7, "colsample_bytree": 0.6, "min_child_samples": 50,
            "reg_alpha": 0.1, "reg_lambda": 1.5, "random_state": 42, "verbose": -1,
        }

    regime_labels = {1: "bull", 0: "ranging", -1: "bear"}
    regime_models = {}

    for regime, label in regime_labels.items():
        subset = train[train["market_regime"] == regime]
        if len(subset) < 500:
            print(f"  Regime {label:8s}: slechts {len(subset)} rijen — overgeslagen (te weinig data)")
            continue

        X_sub = subset[config.FEATURE_COLS]
        y_sub = subset["target"]
        n = len(X_sub)
        time_weights = np.linspace(0.5, 1.0, n)

        model = lgb.LGBMClassifier(**base_params)
        model.fit(X_sub, y_sub, sample_weight=time_weights)

        path = config.symbol_path(symbol, f"{label}_model.pkl")
        joblib.dump(model, path)
        regime_models[regime] = model
        print(f"  Regime {label:8s}: {n:>5} rijen — model opgeslagen: {path.name}")

    return regime_models


def load_regime_model(regime: int, symbol: str = config.SYMBOL):
    """
    Laad het regime-specifieke model voor het opgegeven regime.
    Valt terug op het algemene model als er geen regime-model bestaat.
    """
    regime_labels = {1: "bull", 0: "ranging", -1: "bear"}
    label = regime_labels.get(regime, "ranging")
    path = config.symbol_path(symbol, f"{label}_model.pkl")
    if path.exists():
        return joblib.load(path)
    return load_model(symbol)


# ── Laden ─────────────────────────────────────────────────────────────────────

def load_model(symbol: str = config.SYMBOL, calibrated: bool = True):
    """
    Laad het getrainde model. Geeft bij voorkeur het gekalibreerde model terug
    (betere probability estimates). Valt terug op het ongekalibreerde model.

    Parameters
    ----------
    calibrated : True = probeer gekalibreerde versie eerst (default)
    """
    if calibrated:
        cal_path = config.symbol_path(symbol, "model_calibrated.pkl")
        if cal_path.exists():
            return joblib.load(cal_path)
    model_path = config.symbol_path(symbol, "model.pkl")
    if not model_path.exists():
        raise FileNotFoundError(
            f"Geen model gevonden op {model_path}. Voer eerst train_model() uit."
        )
    return joblib.load(model_path)


def load_regime_thresholds(symbol: str = config.SYMBOL) -> dict:
    """
    Laad regime-specifieke entry drempelwaarden uit {symbol}_regime_thresholds.json.
    Geeft een dict: {'bull': float, 'ranging': float, 'bear': float, 'global': float}
    Valt terug op de globale threshold als het bestand niet bestaat.
    """
    import json
    path = config.symbol_path(symbol, "regime_thresholds.json")
    if path.exists():
        try:
            with open(path) as f:
                return json.load(f)
        except Exception:
            pass
    global_thr, _ = load_optimal_threshold(symbol=symbol)
    return {"bull": global_thr, "ranging": global_thr, "bear": global_thr, "global": global_thr}


def load_optimal_threshold(symbol: str = config.SYMBOL) -> tuple[float, float]:
    """
    Laad de geoptimaliseerde long- en short-drempelwaarden.
    Valt terug op config.SIGNAL_THRESHOLD / 0.0 bij geen opgeslagen threshold.

    Returns
    -------
    (long_threshold, short_threshold)
    """
    import json
    thr_path = config.symbol_path(symbol, "optimal_threshold.json")
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
