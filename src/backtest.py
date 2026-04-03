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
    fee: float = config.TRADE_FEE,
    stop_loss: float = config.STOP_LOSS_PCT,
    use_position_sizing: bool = True,
    regime_filter: bool = True,
    horizon: int = None,
    probas_4h: np.ndarray = None,
    threshold_4h: float = None,
    # Legacy kwargs — genegeerd, behouden voor achterwaartse compatibiliteit
    threshold_short: float = 0.0,
    use_short: bool = False,
) -> pd.DataFrame:
    """
    Voer de long-only backtest uit op de testset.

    Parameters
    ----------
    test_df              : feature matrix testdeel (bevat 'close', 'target' en features)
    probas               : model voorspelde kansen op stijging (klasse 1)
    threshold            : minimale kans voor long-signaal (None = laad optimale drempel)
    fee                  : transactiekosten per kant
    stop_loss            : maximaal verlies per trade; 0 = geen stop-loss
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
    # Voorkeur: data-driven regime_thresholds.json (per regime geoptimaliseerd).
    # Fallback: vaste offsets uit config.REGIME_THRESHOLD_OFFSETS.
    if "market_regime" in test_df.columns:
        regime = test_df["market_regime"].reindex(results.index, fill_value=0).astype(int)
        _regime_map = {1: "bull", 0: "ranging", -1: "bear"}
        try:
            from src.model import load_regime_thresholds as _lrt
            _rthr = _lrt()
            eff_thr = regime.map(
                lambda r: _rthr.get(_regime_map.get(r, "ranging"), threshold)
            ).astype(float)
        except Exception:
            offsets = config.REGIME_THRESHOLD_OFFSETS
            eff_thr = regime.map(lambda r: threshold + offsets.get(r, 0.0)).astype(float)
    else:
        regime = pd.Series(0, index=results.index)
        eff_thr = pd.Series(float(threshold), index=results.index)

    eff_thr = eff_thr.clip(0.48, 0.95)
    results["signal_long"] = (results["proba"] >= eff_thr).astype(int)
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

        # ── P1: DVOL gate ─────────────────────────────────────────────────────
        # Hoge BTC implied volatility (Deribit DVOL) → markt prijst staartrisico in.
        # btc_dvol > config.DVOL_GATE (0.65) blokkeert nieuwe longs.
        # Vangt flash-crash omgevingen (aug 2024) die EMA-filters missen.
        if "btc_dvol" in test_df.columns:
            low_dvol = (
                test_df["btc_dvol"].reindex(results.index, fill_value=0.45)
                < config.DVOL_GATE
            )
            results["signal_long"] = results["signal_long"] * low_dvol.astype(int)

        # ── P1: return_30d gate ───────────────────────────────────────────────
        # BTC > 10% gedaald in 30 dagen → sustained downtrend, geen nieuwe longs.
        # Vangt langzame bearmarkten (sep 2025) die boven EMA200 blijven hangen.
        if "return_30d" in test_df.columns:
            monthly_ok = (
                test_df["return_30d"].reindex(results.index, fill_value=0.0)
                >= config.RETURN_30D_LONG_GATE
            )
            results["signal_long"] = results["signal_long"] * monthly_ok.astype(int)

        # ── P2: VIX gate ──────────────────────────────────────────────────────
        # Aandelenmarkt-angst (VIX > config.VIX_GATE = 25) blokkeert longs.
        # VIX spikte naar 65+ tijdens aug 2024 crash; dagelijks forward-filled.
        if "vix_level" in test_df.columns:
            low_vix = (
                test_df["vix_level"].reindex(results.index, fill_value=20.0)
                < config.VIX_GATE
            )
            results["signal_long"] = results["signal_long"] * low_vix.astype(int)

        # ── P2: USD/JPY gate ──────────────────────────────────────────────────
        # Snelle yen-appreciatie → carry trade unwind → risk-off in alle markten.
        # usdjpy_return_7d < config.USDJPY_RETURN_7D_GATE (-3%) blokkeert longs.
        if "usdjpy_return_7d" in test_df.columns:
            jpy_ok = (
                test_df["usdjpy_return_7d"].reindex(results.index, fill_value=0.0)
                >= config.USDJPY_RETURN_7D_GATE
            )
            results["signal_long"] = results["signal_long"] * jpy_ok.astype(int)

        # ── C4: Put/Call ratio gate ────────────────────────────────────────────
        # Extreme put-dominantie (Deribit P/C > 1.5) → bearish positionering.
        if "btc_put_call_ratio" in test_df.columns:
            pc_gate = getattr(config, "PUT_CALL_RATIO_GATE", 1.5)
            low_pc = (
                test_df["btc_put_call_ratio"].reindex(results.index, fill_value=1.0)
                < pc_gate
            )
            results["signal_long"] = results["signal_long"] * low_pc.astype(int)

        # ── T2-B: Funding extreme gate ─────────────────────────────────────────
        # Extreem positieve funding rate → markt is te bullish (contrair signaal).
        # funding_rate > FUNDING_EXTREME_GATE (+0.05% per 8u) blokkeert nieuwe longs.
        if "funding_rate" in test_df.columns:
            funding_gate = getattr(config, "FUNDING_EXTREME_GATE", 0.0005)
            not_extreme_funding = (
                test_df["funding_rate"].reindex(results.index, fill_value=0.0)
                <= funding_gate
            )
            results["signal_long"] = results["signal_long"] * not_extreme_funding.astype(int)

    # ── Multi-timeframe 4h-confirmatie gate ──────────────────────────────────
    # 4h-model proba moet boven threshold_4h liggen voor een 1h-entry.
    # probas_4h moet op de 1h-index geïnterpoleerd zijn (forward-fill van 4h candle).
    if probas_4h is not None and len(probas_4h) == len(results):
        thr4 = threshold_4h if threshold_4h is not None else config.SIGNAL_THRESHOLD_4H
        if thr4 > 0:
            confirm_4h = (probas_4h >= thr4).astype(int)
            results["signal_long"] = results["signal_long"] * confirm_4h

    # ── S9-B: Dagelijks model alignment gate ─────────────────────────────────
    # Blokkeert 1h longs als het dagelijks model bearish is (proba < threshold).
    # Laad daily features + model on-the-fly als DAILY_GATE_ENABLED=True.
    # Toepasbaar op het testperiode-deel dat door de daily data wordt gedekt.
    # S12-B: daily gate alleen voor symbolen met voldoende model-kwaliteit (DAILY_GATE_SYMBOLS).
    _daily_symbols = getattr(config, "DAILY_GATE_SYMBOLS", [])
    _sym_now = getattr(config, "SYMBOL", "BTCUSDT")
    _daily_gate_ok = getattr(config, "DAILY_GATE_ENABLED", False) and (
        not _daily_symbols or _sym_now in _daily_symbols
    )
    if _daily_gate_ok and regime_filter:
        try:
            import config_daily
            from pathlib import Path
            sym_guess = _sym_now
            daily_feat_path = config_daily.symbol_path_daily(sym_guess, "features.parquet")
            if not daily_feat_path.exists():
                # Probeer uit test_df de symbol te achterhalen via kolom of pad
                raise FileNotFoundError(f"{daily_feat_path} niet gevonden")
            daily_feat = pd.read_parquet(daily_feat_path)
            daily_feat_cols = config_daily.FEATURE_COLS_DAILY
            if not all(c in daily_feat.columns for c in daily_feat_cols):
                raise ValueError("daily features kolommen niet compleet")
            from src.model import load_model as _load_model
            daily_model = _load_model(symbol=sym_guess, calibrated=False)
            # Gebruik 1d model als het beschikbaar is
            daily_model_path = config.DATA_DIR / f"{sym_guess}_1d_model.pkl"
            if daily_model_path.exists():
                import joblib as _jl
                daily_model = _jl.load(daily_model_path)
            daily_feat = daily_feat.dropna(subset=daily_feat_cols)
            daily_probas = daily_model.predict_proba(daily_feat[daily_feat_cols])[:, 1]
            daily_proba_s = pd.Series(daily_probas, index=daily_feat.index, name="daily_proba")
            # Forward-fill dagelijkse proba naar uurlijkse index
            daily_proba_h = (
                daily_proba_s
                .reindex(daily_proba_s.index.union(results.index))
                .sort_index()
                .ffill()
                .reindex(results.index)
            )
            daily_thr = getattr(config, "DAILY_GATE_THRESHOLD", 0.45)
            daily_ok  = (daily_proba_h >= daily_thr).fillna(True)  # NaN = geen data = niet blokkeren
            results["signal_long"] = results["signal_long"] * daily_ok.astype(int)
        except Exception:
            pass  # daily gate niet beschikbaar → stilzwijgend overslaan

    # ── S11-A: Bear-regime short signaal ──────────────────────────────────────
    # Actief uitsluitend bij market_regime == -1 (ADX > 20 en -DI > +DI).
    # Signaal: proba < SHORT_ENTRY_THRESHOLD (model ziet weinig kans op stijging).
    # Doel: verdient in bear-fases waar long volledig geblokkeerd is (0 trades).
    _short_symbols = getattr(config, "BEAR_REGIME_SHORT_SYMBOLS", [])
    bear_short_enabled = getattr(config, "BEAR_REGIME_SHORT_ENABLED", False) and (
        not _short_symbols or getattr(config, "SYMBOL", "BTCUSDT") in _short_symbols
    )
    # S12-A: laad geoptimaliseerde short threshold uit optimal_threshold.json als beschikbaar.
    # Fallback naar config.SHORT_ENTRY_THRESHOLD (handmatig ingesteld: 0.30).
    try:
        _, _saved_short_thr = load_optimal_threshold()
        short_thr = _saved_short_thr if _saved_short_thr > 0 else getattr(config, "SHORT_ENTRY_THRESHOLD", 0.30)
    except Exception:
        short_thr = getattr(config, "SHORT_ENTRY_THRESHOLD", 0.30)
    results["signal_short"] = 0
    if bear_short_enabled and regime_filter and "market_regime" in test_df.columns:
        confirmed_bear = (test_df["market_regime"].reindex(results.index, fill_value=0) == -1)
        bearish_proba  = (results["proba"] <= short_thr)
        # Extra filter: markt moet ook feitelijk in een neergaande trend zitten (return_30d < -3%).
        # Voorkomt shorts in bear-regime-detectie tijdens early recovery (ADX lagging).
        if "return_30d" in test_df.columns:
            in_downtrend = (test_df["return_30d"].reindex(results.index, fill_value=0.0) < -0.03)
        else:
            in_downtrend = pd.Series(True, index=results.index)
        # Geen short als er al een long-signaal is op dit uur (vermijdt conflicten)
        no_long_conflict = (results["signal_long"] == 0)
        # S14-C: ranging_score filter — geen shorts als markt mogelijk kantelend is.
        # ADX is een lagging indicator: ranging_score ≥ 2 betekent dat minstens 2 van 3
        # indicatoren (ADX, BB-breedte, MACD-stabiliteit) ranging signaleren → short overgeslagen.
        ranging_thr = getattr(config, "RANGING_SCORE_THR", 2)
        if "ranging_score" in test_df.columns:
            not_ranging = (test_df["ranging_score"].reindex(results.index, fill_value=0) < ranging_thr)
        else:
            not_ranging = pd.Series(True, index=results.index)
        results["signal_short"] = (
            confirmed_bear & bearish_proba & in_downtrend & no_long_conflict & not_ranging
        ).astype(int)

    results["signal"] = results["signal_long"] - results["signal_short"]

    # ── Basisrendement ────────────────────────────────────────────────────────
    raw_return = results["close"].shift(-h) / results["close"] - 1

    # ── Stop-loss (T1-D: dynamisch ATR-gebaseerd) ─────────────────────────────
    # Als atr_pct beschikbaar is: gebruik 2×ATR als dynamische stop (T1-D).
    # Dit geeft ruimere stops in hoge-vol periodes (voorkomt whipsaw) en
    # nauwere stops in rustige markten (betere risico/beloning verhouding).
    # Clip: minimum 0.5% (niet te eng), maximum 10% (niet te wijd).
    if "atr_pct" in test_df.columns and stop_loss > 0:
        _atr_mult = getattr(config, "ATR_STOP_MULTIPLIER", 2.0)
        atr_stop = (_atr_mult * test_df["atr_pct"].reindex(results.index)).clip(lower=0.005, upper=0.10)
        long_return  = raw_return.clip(lower=-atr_stop)
        short_return = (-raw_return).clip(lower=-atr_stop)  # short profiteert van daling
    elif stop_loss > 0:
        long_return  = raw_return.clip(lower=-stop_loss)
        short_return = (-raw_return).clip(lower=-stop_loss)
    else:
        long_return  = raw_return
        short_return = -raw_return

    # ── Position sizing ───────────────────────────────────────────────────────
    if use_position_sizing:
        base_long = ((results["proba"] - 0.5) * 2).clip(0, 1)
        # P1: Volatiliteit-geschaalde positiegrootte.
        # Hogere marktvolatiliteit → kleinere positie → minder verlies in crashes.
        if "volatility_24h" in test_df.columns:
            vol = test_df["volatility_24h"].reindex(
                results.index, fill_value=0.02
            ).clip(lower=0.001)
            vol_scale = (1.0 / (1.0 + config.VOL_SIZE_SCALE * vol)).clip(0.2, 1.0)
        else:
            vol_scale = 1.0
        long_size  = (base_long * vol_scale).clip(0, 1)
        # Short sizing: hoe lager de proba, hoe groter de short positie
        base_short = ((0.5 - results["proba"]) * 2).clip(0, 1)
        short_size = (base_short * vol_scale).clip(0, 1)

        # S16-B: Crash-modus positie-halvering.
        # Bij een scherpe 1h-daling (>2.5σ) of >10% dagverlies halveert de positie.
        # Beschermt kapitaal in flash-crash/cascadering zonder de trade te stoppen.
        _crash_factor = getattr(config, "CRASH_SIZE_FACTOR", 0.5)
        if "crash_mode" in test_df.columns and _crash_factor < 1.0:
            crash = test_df["crash_mode"].reindex(results.index, fill_value=0).astype(float)
            crash_scale = 1.0 - crash * (1.0 - _crash_factor)
            long_size  = (long_size  * crash_scale).clip(0, 1)
            short_size = (short_size * crash_scale).clip(0, 1)

        # S17-B: MACD momentum-gewogen positiescaling.
        # Sterk positief MACD-momentum → max 1.5× positiegrootte bij entry.
        # Alleen longs: short-positie niet vergroot bij bullish momentum.
        if getattr(config, "MACD_MOMENTUM_SCALE", False) and "macd_size_mult" in test_df.columns:
            macd_mult = 1.0 + test_df["macd_size_mult"].reindex(results.index, fill_value=0)
            long_size = (long_size * macd_mult).clip(0, 1)
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

    # ── P3: Drawdown circuit breaker ──────────────────────────────────────────
    # Zodra de cumulatieve drawdown de drempel overschrijdt (config.MAX_DRAWDOWN_GATE),
    # worden alle signalen geblokkeerd voor config.CIRCUIT_BREAKER_COOLDOWN_H uur.
    # Dit voorkomt dat een verliesgevende periode verder uitgehold wordt.
    #
    # Implementatie: herschrijf strategy_return naar 0 voor geblokkeerde periodes
    # en herbereken cum_strategy zodat de equity-curve klopt.
    max_dd_gate  = getattr(config, "MAX_DRAWDOWN_GATE",        -0.15)
    cooldown_h   = getattr(config, "CIRCUIT_BREAKER_COOLDOWN_H", 168)  # 7 dagen

    if max_dd_gate < 0:   # gate = 0.0 schakelt de breaker uit
        cum      = results["cum_strategy"].values.copy()
        signals  = results["signal"].values.copy()
        strat_r  = results["strategy_return"].values.copy()

        peak          = cum[0]
        blocked_until = -1   # index tot waar geblokkeerd

        for i in range(len(cum)):
            if i <= blocked_until:
                # Blokkeer signalen tijdens cooldown
                strat_r[i] = 0.0
            else:
                if cum[i] > peak:
                    peak = cum[i]
                dd = cum[i] / peak - 1
                if dd < max_dd_gate:
                    blocked_until = i + cooldown_h
                    strat_r[i]    = 0.0   # ook huidige candle blokkeren

        # Herbereken cumulatief met gecorrigeerde returns
        results["strategy_return"] = strat_r
        results["cum_strategy"]    = (1 + results["strategy_return"].fillna(0)).cumprod()

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
    fee: float = config.TRADE_FEE,
    stop_loss_pct: float = config.STOP_LOSS_PCT,
    tp_pct: float = 0.06,
    be_trigger_pct: float = 0.02,
    allow_second_entry: bool = True,
    regime_filter: bool = True,
    horizon: int = None,
    exit_proba_long: float = None,
    use_structural_levels: bool = False,
    highs: np.ndarray = None,
    lows: np.ndarray = None,
    use_atr_stop: bool = True,
    atr_multiplier: float = 2.0,
    use_partial_exits: bool = True,
    tp1_pct: float = 0.03,
    tp2_pct: float = 0.08,
    tp1_fraction: float = 0.50,
    # Legacy kwargs
    threshold_short: float = 0.0,
    use_short: bool = False,
    exit_proba_short: float = None,
) -> pd.DataFrame:
    """
    Long-only backtest met trailing stop-loss naar breakeven (BE) en model-gestuurd sluitmodel.

    T1-D ATR trailing stop (use_atr_stop=True):
      - Initiële SL = entry × (1 − atr_multiplier × atr_pct_at_entry)
      - Wordt bij iedere candle omhoog bijgesteld als close − atr_sl > huidige SL
        (trailing: SL volgt de prijs omhoog, nooit omlaag)
    T1-E Partial exits (use_partial_exits=True):
      - TP1 (tp1_pct=3%): sluit tp1_fraction (50%) van positie; rest houdt door naar TP2
      - Na TP1: SL verplaatst naar entry (breakeven)
      - TP2 (tp2_pct=8%): sluit de resterende helft

    Overige logica per candle:
      - Wanneer close >= entry * (1 + be_trigger_pct) → SL verplaatst naar entry (BE)
      - Als allow_second_entry=True én alle open posities op BE staan → tweede trade
        toegestaan bij nieuw signaal (max 2 simultane posities)
      - SL/TP: gedetecteerd op candle-close (geen intrabar high/low beschikbaar)
      - Model-exit: sluit LONG als proba < exit_proba_long
      - Tijdsvangnet (horizon): laatste exitoptie na MAX_HOLD_HOURS

    Signalen worden identiek berekend als run_backtest() (inclusief regime/death-cross filters).
    Strategy_return wordt bijgehouden per exit-candle (niet entry-candle).
    """
    if threshold is None:
        threshold, _ = load_optimal_threshold()
    if exit_proba_long is None:
        exit_proba_long = config.EXIT_PROBA_LONG
    if exit_proba_short is None:
        exit_proba_short = config.EXIT_PROBA_SHORT

    h = horizon if horizon is not None else config.MAX_HOLD_HOURS

    # ── Pre-compute swing levels voor structurele SL/TP ──────────────────────
    swings = None
    if use_structural_levels and highs is not None and lows is not None:
        from src.levels import precompute_swings
        swings = precompute_swings(highs, lows)

    # Genereer signalen via run_backtest (alle regime-/death-cross filters ingebakken)
    base = run_backtest(
        test_df, probas,
        threshold=threshold,
        fee=fee,
        stop_loss=stop_loss_pct,
        use_position_sizing=True,
        regime_filter=regime_filter,
        horizon=h,
    )
    signals_long = base["signal_long"].values.astype(int)

    closes    = test_df["close"].values
    atr_vals  = test_df["atr_pct"].values if "atr_pct" in test_df.columns else None
    n         = len(closes)

    # Position sizing: identiek aan run_backtest
    long_sizes = np.clip((probas - 0.5) * 2, 0.0, 1.0)

    strategy_returns = np.zeros(n)
    trade_opened     = np.zeros(n, dtype=int)   # 1 = nieuw positie geopend op deze candle
    be_triggered     = np.zeros(n, dtype=int)   # 1 = SL naar BE verplaatst op deze candle
    positions = []  # actieve posities: list of dicts

    for i in range(n):
        close = closes[i]

        # ── 1. Verwerk exits ──────────────────────────────────────────────────
        remaining = []
        for pos in positions:
            direction  = pos["direction"]
            sl         = pos["sl_price"]
            tp         = pos["tp_price"]
            hours_open = i - pos["entry_idx"]
            # C2: Signaalveroudering — positie-effectiviteit daalt lineair met tijd.
            decay_factor = max(0.5, 1.0 - 0.02 * hours_open)
            size       = pos["size"] * decay_factor
            exit_price = None

            proba_i = probas[i]
            if direction == "LONG":
                # ── T1-D: Update ATR trailing stop ────────────────────────────
                if use_atr_stop and atr_vals is not None:
                    atr_trail_sl = close - atr_multiplier * atr_vals[i] * close
                    if atr_trail_sl > pos["sl_price"]:
                        pos["sl_price"] = atr_trail_sl
                        sl = pos["sl_price"]

                if close <= sl:
                    exit_price = sl           # SL geraakt
                elif use_partial_exits and not pos.get("tp1_hit", False) and close >= pos["tp1_price"]:
                    # ── T1-E: TP1 geraakt → sluit tp1_fraction, zet SL naar BE ──
                    tp1_size = pos["size"] * tp1_fraction
                    tp1_ret  = (pos["tp1_price"] - pos["entry_price"]) / pos["entry_price"]
                    strategy_returns[i] += tp1_size * tp1_ret - tp1_size * 2 * fee
                    # Verkleun positie en zet SL naar entry (BE)
                    pos["size"]     = pos["size"] * (1 - tp1_fraction)
                    pos["tp1_hit"]  = True
                    pos["sl_price"] = pos["entry_price"]   # SL → BE na TP1
                    pos["is_be"]    = True
                    be_triggered[i] = 1
                    remaining.append(pos)
                    continue
                elif close >= tp:
                    exit_price = tp           # TP2 (of enkel TP als partial exits uit) geraakt
                elif proba_i < exit_proba_long:
                    exit_price = close        # Model-exit: proba onder drempel
                elif i >= pos["horizon_idx"]:
                    exit_price = close        # Tijdsvangnet (168h)
            else:  # SHORT (legacy — nooit actief bij long-only)
                if close >= sl:
                    exit_price = sl
                elif close <= tp:
                    exit_price = tp
                elif i >= pos["horizon_idx"]:
                    exit_price = close        # Tijdsvangnet (168h)

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
        n_open = len(positions)
        all_be = all(p["is_be"] for p in positions) if positions else True
        can_long = (n_open == 0) or (allow_second_entry and all_be and n_open < 2)

        if can_long and signals_long[i]:
            # T1-D: Dynamische stop op basis van ATR; fallback naar vaste stop
            if use_atr_stop and atr_vals is not None:
                atr_sl_pct = float(np.clip(atr_multiplier * atr_vals[i], 0.005, 0.10))
            else:
                atr_sl_pct = stop_loss_pct

            if swings is not None:
                from src.levels import get_recent_levels, compute_structural_sl_tp
                supports, resistances = get_recent_levels(swings, i)
                sl_p, tp_p = compute_structural_sl_tp(
                    close, "LONG", supports, resistances,
                    fallback_sl_pct=atr_sl_pct, fallback_tp_pct=tp_pct,
                )
            else:
                sl_p = close * (1 - atr_sl_pct)
                tp_p = close * (1 + (tp2_pct if use_partial_exits else tp_pct))

            positions.append({
                "direction":   "LONG",
                "entry_price": close,
                "sl_price":    sl_p,
                "tp_price":    tp_p,
                "tp1_price":   close * (1 + tp1_pct) if use_partial_exits else tp_p,
                "tp1_hit":     False,
                "is_be":       False,
                "horizon_idx": i + h,
                "entry_idx":   i,
                "size":        long_sizes[i],
            })
            trade_opened[i] += 1

    # ── Resultaten DataFrame ──────────────────────────────────────────────────
    results = test_df[["close", "target"]].copy()
    results["proba"]           = probas
    results["signal_long"]     = signals_long
    results["signal"]          = signals_long
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


# ── Exit-proba optimalisatie ──────────────────────────────────────────────────

def optimize_exit_proba(
    model,
    val_df: pd.DataFrame,
    threshold: float,
    candidates_long: list = None,
    symbol: str = config.SYMBOL,
    # Legacy kwargs
    threshold_short: float = 0.0,
    candidates_short: list = None,
) -> tuple[float, float]:
    """
    Zoek de optimale EXIT_PROBA_LONG via grid-sweep op val_df.

    Evalueert elke waarde met run_backtest_be_trail en selecteert de waarden
    met de hoogste Sharpe ratio. Slaat resultaat op in {symbol}_exit_proba.json.

    Parameters
    ----------
    model           : getraind model (predict_proba)
    val_df          : validatieset (feature matrix met close/target)
    threshold       : long entry drempel
    candidates_long : sweep-waarden voor exit_proba_long (default: 0.30–0.55 step 0.025)
    symbol          : voor opslaan exit_proba.json

    Returns
    -------
    (best_exit_long, best_exit_short) als floats (exit_short = 1 - exit_long)
    """
    import json

    if candidates_long is None:
        candidates_long = [round(x * 0.025 + 0.300, 3) for x in range(11)]  # 0.300–0.550

    probas_val = model.predict_proba(val_df[config.FEATURE_COLS])[:, 1]

    print("\nExit-proba optimalisatie (sweep op validatieset)...")
    best_sharpe = -np.inf
    best_long   = config.EXIT_PROBA_LONG

    results_log = []
    for el in candidates_long:
        try:
            res = run_backtest_be_trail(
                val_df, probas_val,
                threshold=threshold,
                exit_proba_long=el,
            )
            metrics = compute_metrics(res)
            sharpe  = metrics["sharpe_ratio"]
            results_log.append({"exit_long": el, "sharpe": sharpe,
                                 "n_trades": metrics["n_trades"],
                                 "total_return": metrics["total_return"]})
            if sharpe > best_sharpe:
                best_sharpe = sharpe
                best_long   = el
        except Exception:
            pass

    # Sorteer en toon top-5
    results_log.sort(key=lambda x: x["sharpe"], reverse=True)
    print(f"  {'exit_long':>10}  {'sharpe':>8}  {'trades':>6}  {'return':>8}")
    for row in results_log[:5]:
        print(f"  {row['exit_long']:>10.3f}  "
              f"{row['sharpe']:>+8.3f}  {row['n_trades']:>6}  {row['total_return']:>+8.1%}")
    print(f"  → Beste: exit_long={best_long:.3f}  Sharpe={best_sharpe:+.3f}")

    best_short = round(1.0 - best_long, 3)   # symmetrisch (niet actief, alleen voor compat)

    # Opslaan
    out_path = config.symbol_path(symbol, "exit_proba.json")
    with open(out_path, "w") as f:
        json.dump({
            "exit_proba_long":  best_long,
            "exit_proba_short": best_short,
            "sharpe":           round(best_sharpe, 4),
        }, f, indent=2)
    print(f"  Opgeslagen: {out_path.name}")

    return best_long, best_short


def load_exit_proba(symbol: str = config.SYMBOL) -> tuple[float, float]:
    """
    Laad geoptimaliseerde exit drempelwaarden uit {symbol}_exit_proba.json.
    Valt terug op config defaults als het bestand niet bestaat.
    """
    import json
    path = config.symbol_path(symbol, "exit_proba.json")
    if path.exists():
        try:
            with open(path) as f:
                data = json.load(f)
            return float(data["exit_proba_long"]), float(data["exit_proba_short"])
        except Exception:
            pass
    return config.EXIT_PROBA_LONG, config.EXIT_PROBA_SHORT


# ── Walk-forward validatie ────────────────────────────────────────────────────

def run_walkforward(
    df: pd.DataFrame,
    model_name: str = "RandomForest",
    symbol: str = config.SYMBOL,
    use_regime_models: bool = False,
    expanding: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Walk-forward backtest: traint maandelijks opnieuw.

    expanding=False (default) : rollend venster van WALKFORWARD_TRAIN_DAYS
    expanding=True            : expanding venster — alle data vanaf t=0 t/m fold start
                                Voordeel: model vergeet nooit zeldzame events (2022 bear).
                                Minimale trainset: 180 dagen.

    use_regime_models: traint aparte bull/ranging/bear modellen per fold en
    routeert elke rij naar het passende model bij het voorspellen.
    """
    from src.model import optimize_threshold
    from src.model_compare import _get_models

    train_h    = config.WALKFORWARD_TRAIN_DAYS * 24
    test_h     = config.WALKFORWARD_TEST_DAYS  * 24
    step_h     = config.WALKFORWARD_STEP_DAYS  * 24
    val_h      = config.VALIDATION_SIZE_DAYS   * 24
    min_train_h = 180 * 24   # minimale trainset bij expanding window

    models_dict = _get_models(symbol=symbol)
    if model_name not in models_dict:
        raise ValueError(f"Model '{model_name}' niet gevonden. Beschikbaar: {list(models_dict.keys())}")

    mode_str = "expanding" if expanding else f"rolling {config.WALKFORWARD_TRAIN_DAYS}d"
    print(f"Walk-forward validatie  ({model_name}, {mode_str})")
    print(f"  Trainvenster : {mode_str}")
    print(f"  Testvenster  : {config.WALKFORWARD_TEST_DAYS}  dagen")

    fold_metrics_list = []
    all_results_list  = []
    fold  = 0
    start = 0 if not expanding else train_h   # expanding: begin pas na minimale train

    while start + test_h <= len(df):
        # Expanding: train loopt altijd vanaf 0; rollend: vast venster van train_h
        if expanding:
            train_end = start
        else:
            train_end = start + train_h
        train_start = 0 if expanding else max(0, train_end - train_h)

        train = df.iloc[train_start : train_end - val_h]
        val   = df.iloc[train_end - val_h : train_end]
        test  = df.iloc[train_end : train_end + test_h]

        if len(train) < 500 or len(val) < 100 or len(test) == 0:
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

        # Regime-geconditioneerde voorspelling (optioneel)
        if use_regime_models and "market_regime" in train.columns and model_name == "LightGBM":
            import lightgbm as lgb
            import json as _json
            stable_path = config.symbol_path(symbol, "lgb_best_params.json")
            if stable_path.exists():
                with open(stable_path) as _f:
                    _rp = _json.load(_f)
                _rp["random_state"] = 42
                _rp["verbose"] = -1
            else:
                _rp = model.get_params()

            regime_mdls = {}
            for _reg, _lbl in {1: "bull", 0: "ranging", -1: "bear"}.items():
                _sub = train[train["market_regime"] == _reg]
                if len(_sub) >= 300:
                    _tw = np.linspace(0.5, 1.0, len(_sub))
                    _rm = lgb.LGBMClassifier(**_rp)
                    _rm.fit(_sub[config.FEATURE_COLS], _sub["target"], sample_weight=_tw)
                    regime_mdls[_reg] = _rm

            # Voorspel per rij met regime-model, val terug op algemeen model
            probas = np.empty(len(test))
            if "market_regime" in test.columns:
                for _i, (_idx, _row) in enumerate(test.iterrows()):
                    _reg = int(_row["market_regime"]) if not pd.isna(_row["market_regime"]) else 0
                    _mdl = regime_mdls.get(_reg, model)
                    probas[_i] = _mdl.predict_proba(_row[config.FEATURE_COLS].values.reshape(1, -1))[0, 1]
            else:
                probas = model.predict_proba(test[config.FEATURE_COLS])[:, 1]
        else:
            probas = model.predict_proba(test[config.FEATURE_COLS])[:, 1]

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
            f"  L:{metrics['n_long']}"
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
    Gebruikt de geoptimaliseerde drempelwaarde, regime filter én 4h-confirmatie.
    """
    from src.features import build_features

    features  = build_features(df_ohlcv, p1p2, p1_heatmap, direction_bias, symbol=symbol)
    model     = load_model(symbol=symbol)
    threshold, threshold_short = load_optimal_threshold(symbol=symbol)

    # ── 4h-model confirmatie ──────────────────────────────────────────────────
    proba_4h = None
    confirm_4h = True
    try:
        model_4h  = load_model(symbol=f"{symbol}_4h")
        thr_4h, _ = load_optimal_threshold(symbol=f"{symbol}_4h")
        feat_4h   = pd.read_parquet(config.symbol_path(f"{symbol}_4h", "features.parquet"))
        feat_cols_4h = [c for c in feat_4h.columns
                        if c not in ("target", "close", "future_close", "market_regime")]
        last_4h   = feat_4h.iloc[[-1]]
        proba_4h  = float(model_4h.predict_proba(last_4h[feat_cols_4h])[0, 1])
        thr_used  = thr_4h if thr_4h > 0 else config.SIGNAL_THRESHOLD_4H
        confirm_4h = proba_4h >= thr_used
    except Exception:
        pass  # geen 4h-model beschikbaar → gate uitgeschakeld

    last_row  = features.iloc[[-1]]
    proba     = float(model.predict_proba(last_row[config.FEATURE_COLS])[0, 1])

    # Regime-check: prijs boven EMA200
    regime_ok = (
        float(last_row["price_vs_ema200"].iloc[0]) > 1.0
        if "price_vs_ema200" in last_row.columns
        else True
    )

    # Regime-adaptieve drempel: data-driven per regime, fallback op config offsets
    market_regime = 0
    if "market_regime" in last_row.columns:
        market_regime = int(last_row["market_regime"].iloc[0])
    _regime_name_map = {1: "bull", 0: "ranging", -1: "bear"}
    try:
        from src.model import load_regime_thresholds as _lrt
        _rthr = _lrt(symbol=symbol)
        eff_threshold = float(np.clip(
            _rthr.get(_regime_name_map.get(market_regime, "ranging"), threshold), 0.50, 0.95
        ))
    except Exception:
        regime_offset = config.REGIME_THRESHOLD_OFFSETS.get(market_regime, 0.0)
        eff_threshold = float(np.clip(threshold + regime_offset, 0.50, 0.95))

    # Death cross: EMA50 < EMA200 → extra blokkade (bull trap voorkomen)
    death_cross = False
    if "ema_ratio_50" in last_row.columns and "price_vs_ema200" in last_row.columns:
        death_cross = (
            float(last_row["ema_ratio_50"].iloc[0])
            > float(last_row["price_vs_ema200"].iloc[0])
        )

    # ── T3-C: Deribit 25-delta skew gate (live-only) ─────────────────────────
    skew_25d    = 0.0
    skew_blocked = False
    try:
        from src.external_data import fetch_deribit_skew_live
        skew_df  = fetch_deribit_skew_live()
        if not skew_df.empty and "btc_skew_25d" in skew_df.columns:
            skew_25d = float(skew_df["btc_skew_25d"].iloc[-1])
            skew_gate = getattr(config, "SKEW_BEARISH_GATE", 5.0)
            skew_blocked = skew_25d > skew_gate
    except Exception:
        pass  # skew gate uitgeschakeld als Deribit onbereikbaar

    # S14-C: ranging_score filter — geen shorts als markt mogelijk kantelend is.
    ranging_score_val = 0.0
    if "ranging_score" in last_row.columns:
        ranging_score_val = float(last_row["ranging_score"].iloc[0])
    ranging_thr = getattr(config, "RANGING_SCORE_THR", 2)
    not_ranging = (ranging_score_val < ranging_thr)

    # S16-B: crash_mode — geef terug aan run_live_alert voor positie-halvering
    crash_mode_val = 0
    if "crash_mode" in last_row.columns:
        crash_mode_val = int(last_row["crash_mode"].iloc[0])

    # S17-B: macd_size_mult — geef terug voor momentum-scaling
    macd_size_mult_val = 0.0
    if "macd_size_mult" in last_row.columns:
        macd_size_mult_val = float(last_row["macd_size_mult"].iloc[0])

    if proba >= eff_threshold and regime_ok and not death_cross and confirm_4h and not skew_blocked:
        signaal = "LONG"
    elif proba <= threshold_short and threshold_short > 0 and not regime_ok and not_ranging:
        signaal = "SHORT"
    elif proba <= threshold_short and threshold_short > 0 and not regime_ok and not not_ranging:
        signaal = f"WACHT (ranging_score {ranging_score_val:.0f} ≥ {ranging_thr} — geen short in ranging markt)"
    elif skew_blocked and proba >= eff_threshold:
        signaal = f"WACHT (25D skew gate — puts te duur, skew {skew_25d:+.1f}%)"
    elif death_cross and proba >= eff_threshold:
        signaal = "WACHT (death cross — EMA50 onder EMA200)"
    elif proba >= eff_threshold and not regime_ok:
        signaal = "WACHT (onder EMA200 — long geblokkeerd)"
    elif proba >= eff_threshold and regime_ok and not confirm_4h:
        signaal = f"WACHT (4h bevestiging ontbreekt — 4h proba {proba_4h:.1%})"
    else:
        signaal = "WACHT"

    regime_labels = {1: "bull", 0: "ranging", -1: "bear"}
    return {
        "tijdstip":            str(last_row.index[0]),
        "signaal":             signaal,
        "kans_stijging":       f"{proba:.1%}",
        "proba_raw":           proba,
        "long_threshold":      f"{eff_threshold:.2f}",
        "long_threshold_base": f"{threshold:.2f}",
        "short_threshold":     f"{threshold_short:.2f}" if threshold_short > 0 else "uitgeschakeld",
        "regime_boven_ema200": regime_ok,
        "market_regime":       regime_labels.get(market_regime, "onbekend"),
        "death_cross":         death_cross,
        "proba_4h":            f"{proba_4h:.1%}" if proba_4h is not None else "n/a",
        "confirm_4h":          confirm_4h,
        "btc_skew_25d":        f"{skew_25d:+.1f}%",
        "skew_blocked":        skew_blocked,
        "horizon":             f"{config.PREDICTION_HORIZON_H} uur",
        "prijs":               float(last_row["close"].iloc[0]),
        # S14-C / S16-B / S17-B: waarden voor positie-sizing in run_live_alert
        "ranging_score":       ranging_score_val,
        "crash_mode":          crash_mode_val,
        "macd_size_mult":      macd_size_mult_val,
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
