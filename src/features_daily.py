"""
Dagelijkse Feature Engineering
Bouwt de feature matrix voor het dagelijkse BTC/ETH richtingsmodel.
Input : 1d OHLCV candles (load_ohlcv interval="1d")
Output: feature matrix voor ML-training (zelfde structuur als uurmodel)

Geen look-ahead bias: elke feature voor candle T gebruikt uitsluitend
data van vóór T. Target is de slotkoers van de volgende dagcandle.
"""

import numpy as np
import pandas as pd

import config_daily as cfg
from src.data_fetcher import load_ohlcv


def build_features_daily(
    df_ohlcv: pd.DataFrame,
    symbol: str = cfg.SYMBOL,
) -> pd.DataFrame:
    """
    Bouw de dagelijkse feature matrix voor ML-training.

    Parameters
    ----------
    df_ohlcv : UTC-geïndexeerde 1d OHLCV DataFrame (uitvoer van load_ohlcv interval="1d")
    symbol   : handelssymbool (BTCUSDT of ETHUSDT)

    Returns
    -------
    pd.DataFrame met kolommen = FEATURE_COLS_DAILY + FILTER_COLS_DAILY + ["target", "close"]
    """
    df = df_ohlcv.copy()

    # ── Tijdfeature ───────────────────────────────────────────────────────────
    df["day_of_week"] = df.index.dayofweek  # 0=Ma … 6=Zo

    # ── Prijs & volume (rolling, geen look-ahead) ─────────────────────────────
    df["returns"] = df["close"].pct_change()

    df["volatility_7d"]  = df["returns"].rolling(7).std()
    df["volatility_30d"] = df["returns"].rolling(30).std()

    df["volume_ratio_30d"] = df["volume"] / (df["volume"].rolling(30).mean() + 1e-10)
    df["volume_spike_30d"] = df["volume"] / (df["volume"].rolling(30).mean() + 1e-10)

    # Prijspositie in 7-daags high-low range
    high_7d = df["high"].rolling(7).max()
    low_7d  = df["low"].rolling(7).min()
    df["price_position_7d"] = (df["close"] - low_7d) / (high_7d - low_7d + 1e-10)

    # ── Vorige week return (5 dagcandles, geen look-ahead via shift) ──────────
    df["prev_week_return"] = df["close"].pct_change(5).shift(1)

    # ── Trendkwaliteit ────────────────────────────────────────────────────────
    df["trend_consistency_4w"] = (df["close"].diff() > 0).rolling(20).mean()
    df["buy_pressure_14d"]     = (df["close"] > df["open"]).rolling(14).mean()

    # ── Macro momentum ────────────────────────────────────────────────────────
    df["return_7d"]  = df["close"].pct_change(7)
    df["return_30d"] = df["close"].pct_change(30)

    ath_30d = df["close"].rolling(30).max()
    df["ath_30d_distance"] = df["close"] / ath_30d - 1

    # ── Technische indicatoren (op dagcandles) ────────────────────────────────
    try:
        import ta as ta_lib
    except ImportError:
        raise ImportError("ta is vereist. Installeer via: pip install ta")

    close = df["close"]

    # RSI(14)
    df["rsi_14"] = ta_lib.momentum.RSIIndicator(close=close, window=14).rsi()

    # MACD (12, 26, 9)
    macd_ind = ta_lib.trend.MACD(close=close, window_slow=26, window_fast=12, window_sign=9)
    df["macd"]        = macd_ind.macd()
    df["macd_signal"] = macd_ind.macd_signal()

    # Bollinger Bands (20, 2σ) — %B positie
    bb = ta_lib.volatility.BollingerBands(close=close, window=20, window_dev=2)
    df["bb_pct"] = bb.bollinger_pband()

    # EMA ratio's
    ema_20  = ta_lib.trend.EMAIndicator(close=close, window=20).ema_indicator()
    ema_50  = ta_lib.trend.EMAIndicator(close=close, window=50).ema_indicator()
    ema_200 = ta_lib.trend.EMAIndicator(close=close, window=200).ema_indicator()
    df["ema_ratio_20"]    = close / ema_20
    df["ema_ratio_50"]    = close / ema_50
    df["price_vs_ema200"] = close / ema_200  # >1.0 = boven EMA200 = bullish regime

    # ATR(14) genormaliseerd
    atr = ta_lib.volatility.AverageTrueRange(
        high=df["high"], low=df["low"], close=close, window=14
    ).average_true_range()
    df["atr_pct"] = atr / close

    # ADX(14) — trendsterkte en regime
    adx_ind = ta_lib.trend.ADXIndicator(
        high=df["high"], low=df["low"], close=close, window=14
    )
    df["adx"]     = adx_ind.adx()
    adx_pos = adx_ind.adx_pos()
    adx_neg = adx_ind.adx_neg()

    # market_regime: +1 bull, 0 ranging, -1 bear
    df["market_regime"] = np.where(
        (df["adx"] > 20) & (adx_pos > adx_neg),  1,
        np.where(
        (df["adx"] > 20) & (adx_neg > adx_pos), -1,
        0)
    ).astype(float)

    # ── ETH/BTC ratio (marktbreedte indicator) ────────────────────────────────
    try:
        if symbol == "BTCUSDT":
            df_other = load_ohlcv(symbol="ETHUSDT", interval="1d")
        else:
            df_other = load_ohlcv(symbol="BTCUSDT", interval="1d")
        ratio = (df_other["close"] / df_ohlcv["close"]).reindex(df.index)
        df["eth_btc_ratio"] = ratio.pct_change(1)
    except Exception:
        df["eth_btc_ratio"] = 0.0

    # ── Externe data (Fear & Greed, SPX, EURUSD, funding, btc_dvol) ──────────
    try:
        from src.external_data import load_all_external
        df_ext = load_all_external(df.index, symbol=symbol)
        for col in df_ext.columns:
            df[col] = df_ext[col]
    except (KeyError, ValueError, TypeError) as e:
        print(f"  Waarschuwing: externe data laden mislukt ({e}) — defaults gebruikt")
        import traceback; traceback.print_exc()

    # SPX en EURUSD via dagelijkse download (geen uurlimiet, matcht op middernacht UTC)
    # Uurdata (spx_return_24h) werkt niet voor dagindexen — join vindt nooit matches.
    try:
        from src.external_data import fetch_spx_daily, fetch_eurusd_daily
        df_spx = fetch_spx_daily()
        if not df_spx.empty and "spx_return_1w" in df_spx.columns:
            df_spx.index = df_spx.index.astype(df.index.dtype)
            df["spx_return_1w"] = df_spx["spx_return_1w"].reindex(df.index).ffill().fillna(0.0)
        else:
            df["spx_return_1w"] = 0.0

        df_eur = fetch_eurusd_daily()
        if not df_eur.empty and "eurusd_return_1w" in df_eur.columns:
            df_eur.index = df_eur.index.astype(df.index.dtype)
            df["eurusd_return_1w"] = df_eur["eurusd_return_1w"].reindex(df.index).ffill().fillna(0.0)
        else:
            df["eurusd_return_1w"] = 0.0
    except Exception as e:
        print(f"  Waarschuwing: dagelijkse SPX/EURUSD laden mislukt ({e})")
        df["spx_return_1w"]    = 0.0
        df["eurusd_return_1w"] = 0.0

    # Funding momentum: 7-daagse verandering (dagmodel)
    if "funding_rate" in df.columns:
        df["funding_momentum"] = df["funding_rate"].diff(7)
    else:
        df["funding_rate"]     = 0.0
        df["funding_momentum"] = 0.0

    # Defaults voor ontbrekende externe kolommen
    for col in ["fear_greed", "btc_dvol"]:
        if col not in df.columns:
            df[col] = 0.0

    # ── Target variabele ──────────────────────────────────────────────────────
    # target = 1 als volgende dagclose > huidig × (1 + dead)
    # target = 0 als volgende dagclose < huidig × (1 - dead)
    # NaN (dead zone) → verwijderd uit training
    h    = cfg.PREDICTION_HORIZON_D
    dead = cfg.TARGET_DEAD_ZONE_PCT
    df["future_close"] = df["close"].shift(-h)
    df["target"] = np.where(
        df["future_close"] > df["close"] * (1 + dead), 1,
        np.where(
        df["future_close"] < df["close"] * (1 - dead), 0,
        np.nan)
    )

    # ── Selecteer en schoon op ────────────────────────────────────────────────
    keep      = cfg.FEATURE_COLS_DAILY + cfg.FILTER_COLS_DAILY + ["target", "close"]
    available = [c for c in keep if c in df.columns]
    df_feat   = df[available].dropna()
    df_feat   = df_feat.copy()
    df_feat["target"] = df_feat["target"].astype(int)

    n_total   = len(df)
    n_removed = n_total - len(df_feat)
    if n_removed > 0:
        pct = n_removed / n_total
        print(f"  Dead zone + NaN: {n_removed} rijen verwijderd ({pct:.1%} van totaal)")

    missing = [c for c in keep if c not in df.columns]
    if missing:
        print(f"  Waarschuwing: ontbrekende kolommen: {missing}")

    return df_feat
