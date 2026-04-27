"""
4h Feature Engineering
Bouwt de feature matrix voor het 4h BTC/ETH richtingsmodel.
Input : 4h OHLCV candles (load_ohlcv interval="4h")
Output: feature matrix voor ML-training

Geen look-ahead bias: elke feature voor candle T gebruikt uitsluitend
data van vóór T. Target is de slotkoers 3 candles (12h) vooruit.
"""

import numpy as np
import pandas as pd

import config_4h as cfg
from src.data_fetcher import load_ohlcv


def build_features_4h(
    df_ohlcv: pd.DataFrame,
    symbol: str = cfg.SYMBOL,
) -> pd.DataFrame:
    """
    Bouw de 4h feature matrix voor ML-training.

    Parameters
    ----------
    df_ohlcv : UTC-geïndexeerde 4h OHLCV DataFrame (uitvoer van load_ohlcv interval="4h")
    symbol   : handelssymbool (BTCUSDT of ETHUSDT)

    Returns
    -------
    pd.DataFrame met kolommen = FEATURE_COLS_4H_MODEL + FILTER_COLS_4H + ["target", "close"]
    """
    df = df_ohlcv.copy()

    # ── Tijdfeatures ──────────────────────────────────────────────────────────
    df["day_of_week"] = df.index.dayofweek       # 0=Ma … 6=Zo
    df["hour_4h"]     = df.index.hour             # beginuur: 0, 4, 8, 12, 16, 20

    # ── Prijs & volume (rolling, geen look-ahead) ─────────────────────────────
    df["returns"] = df["close"].pct_change()

    # Volatiliteit (std van rendementen)
    df["volatility_24h"] = df["returns"].rolling(6).std()   # 6 candles = 24h
    df["volatility_3d"]  = df["returns"].rolling(18).std()  # 18 candles = 3d

    # Volume ratio
    df["volume_ratio_24h"] = df["volume"] / (df["volume"].rolling(6).mean() + 1e-10)

    # Prijspositie in 24h high-low range
    high_24h = df["high"].rolling(6).max()
    low_24h  = df["low"].rolling(6).min()
    df["price_position_24h"] = (df["close"] - low_24h) / (high_24h - low_24h + 1e-10)

    # Trendkwaliteit
    df["buy_pressure_24h"]     = (df["close"] > df["open"]).rolling(6).mean()
    df["trend_consistency_3d"] = (df["close"].diff() > 0).rolling(18).mean()

    # Multi-horizon momentum
    df["return_12h"] = df["close"].pct_change(3)    # 3 × 4h = 12h
    df["return_24h"] = df["close"].pct_change(6)    # 6 × 4h = 24h
    df["return_3d"]  = df["close"].pct_change(18)   # 18 × 4h = 3d
    df["return_7d"]  = df["close"].pct_change(42)   # 42 × 4h = 7d
    df["return_30d"] = df["close"].pct_change(180)  # 180 × 4h = 30d

    # ATH afstand (7-daags = 42 candles)
    ath_7d = df["close"].rolling(42).max()
    df["ath_7d_distance"] = df["close"] / ath_7d - 1

    # Volatiliteitsregime: korte vol / lange vol
    vol_6h  = df["returns"].rolling(2).std()   # 2 × 4h = 8h (korte vol)
    df["vol_regime"] = vol_6h / (df["volatility_24h"] + 1e-10)

    # ── Technische indicatoren ────────────────────────────────────────────────
    try:
        import ta as ta_lib
    except ImportError:
        raise ImportError("ta is vereist. Installeer via: pip install ta")

    close = df["close"]

    df["rsi_14"] = ta_lib.momentum.RSIIndicator(close=close, window=14).rsi()

    macd_ind = ta_lib.trend.MACD(close=close, window_slow=26, window_fast=12, window_sign=9)
    df["macd"]        = macd_ind.macd()
    df["macd_signal"] = macd_ind.macd_signal()

    bb = ta_lib.volatility.BollingerBands(close=close, window=20, window_dev=2)
    df["bb_pct"] = bb.bollinger_pband()

    ema_20  = ta_lib.trend.EMAIndicator(close=close, window=20).ema_indicator()
    ema_50  = ta_lib.trend.EMAIndicator(close=close, window=50).ema_indicator()
    ema_200 = ta_lib.trend.EMAIndicator(close=close, window=200).ema_indicator()
    df["ema_ratio_20"]    = close / ema_20
    df["ema_ratio_50"]    = close / ema_50
    df["price_vs_ema200"] = close / ema_200

    atr = ta_lib.volatility.AverageTrueRange(
        high=df["high"], low=df["low"], close=close, window=14
    ).average_true_range()
    df["atr_pct"] = atr / close

    adx_ind = ta_lib.trend.ADXIndicator(
        high=df["high"], low=df["low"], close=close, window=14
    )
    df["adx"]     = adx_ind.adx()
    adx_pos = adx_ind.adx_pos()
    adx_neg = adx_ind.adx_neg()

    # VWAP (dagelijks gereset, forward-fill binnen dag)
    df["_day"]     = df.index.normalize()
    df["_typical"] = (df["high"] + df["low"] + df["close"]) / 3
    df["_tpv"]     = df["_typical"] * df["volume"]
    cumsum_data    = df.groupby("_day")[["_tpv", "volume"]].cumsum()
    df["_cum_tpv"] = cumsum_data["_tpv"]
    df["_cum_vol"] = cumsum_data["volume"]
    vwap = df["_cum_tpv"] / df["_cum_vol"].replace(0, 1e-10)
    df["vwap_distance"] = (close - vwap) / close.replace(0, 1e-10)
    df.drop(columns=["_day", "_typical", "_tpv", "_cum_tpv", "_cum_vol"], inplace=True)

    # Market regime (+1 bull, 0 ranging, -1 bear)
    df["market_regime"] = np.where(
        (df["adx"] > 20) & (adx_pos > adx_neg),   1,
        np.where(
        (df["adx"] > 20) & (adx_neg > adx_pos),  -1,
        0)
    ).astype(float)

    # ── ETH/BTC ratio ─────────────────────────────────────────────────────────
    try:
        if symbol == "BTCUSDT":
            df_other = load_ohlcv(symbol="ETHUSDT", interval="4h")
        else:
            df_other = load_ohlcv(symbol="BTCUSDT", interval="4h")
        ratio = (df_other["close"] / df_ohlcv["close"]).reindex(df.index)
        df["eth_btc_ratio"] = ratio.pct_change(6)   # 24h verandering
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

    # Funding momentum: 18-candle verandering (= 3 dagen op 4h)
    if "funding_rate" in df.columns:
        df["funding_momentum"] = df["funding_rate"].diff(18)
    else:
        df["funding_rate"]     = 0.0
        df["funding_momentum"] = 0.0

    # Defaults voor ontbrekende externe kolommen
    for col in ["fear_greed", "spx_return_24h", "eurusd_return_24h", "btc_dvol"]:
        if col not in df.columns:
            df[col] = 0.0

    # ── Target variabele ──────────────────────────────────────────────────────
    h    = cfg.PREDICTION_HORIZON_4H   # 3 candles = 12h vooruit
    dead = cfg.TARGET_DEAD_ZONE_PCT    # 0.4%
    df["future_close"] = df["close"].shift(-h)
    df["target"] = np.where(
        df["future_close"] > df["close"] * (1 + dead),  1.0,
        np.where(
        df["future_close"] < df["close"] * (1 - dead),  0.0,
        np.nan)
    )

    # ── Feature matrix opschonen ──────────────────────────────────────────────
    available = [c for c in cfg.FEATURE_COLS_4H_MODEL + cfg.FILTER_COLS_4H
                 if c in df.columns]
    available += ["target", "close", "future_close"]

    n_before = len(df[available].dropna(subset=cfg.FEATURE_COLS_4H_MODEL))
    df_feat = df[available].dropna()
    n_removed = n_before - len(df_feat)
    print(f"  Dead zone: {n_removed} neutrale rijen verwijderd "
          f"({n_removed / max(len(df), 1):.1%} van totaal)")

    return df_feat
