"""
Fase 4 — Feature Engineering
Combineert ruwe OHLCV-data met P1/P2-statistieken en technische indicatoren
tot een feature matrix die als invoer dient voor het ML model.

Tijdframes:
  1h  — basisfeatures + P1/P2 statistieken + technische indicatoren
  4h  — hogere-timeframe context (RSI, MACD, BB, EMA)

Belangrijk: GEEN look-ahead bias. Elke feature voor candle op tijdstip T
gebruikt uitsluitend data van vóór T.
  - 4h features worden één periode verschoven (shift(1)) zodat op T=04:00
    uitsluitend de al afgesloten 4h-candle (00:00) zichtbaar is.
"""

import numpy as np
import pandas as pd

import config
from src.data_fetcher import load_ohlcv
from src.stats import compute_direction_bias, compute_p1_heatmap

DOW_LABELS = {0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri", 5: "Sat", 6: "Sun"}


# ── Hulpfuncties ──────────────────────────────────────────────────────────────

def _session(hour: int) -> int:
    """Codeer het handelsuur als sessie-integer."""
    if 0 <= hour < 8:
        return 0  # Aziatisch
    elif 8 <= hour < 13:
        return 1  # Londen
    else:
        return 2  # New York


def _lookup_heatmap(heatmap: pd.DataFrame, dow_int: int, hour: int) -> float:
    """Haal een waarde op uit een (dag × uur) heatmap. Geeft 0.0 bij missing."""
    label = DOW_LABELS.get(dow_int, "Mon")
    if label in heatmap.index and hour in heatmap.columns:
        val = heatmap.loc[label, hour]
        return float(val) if not np.isnan(val) else 0.0
    return 0.0


# ── Technische indicatoren ────────────────────────────────────────────────────

def _add_ta_indicators(df: pd.DataFrame, suffix: str = "") -> pd.DataFrame:
    """
    Voeg RSI, MACD, Bollinger Bands en EMA-ratio's toe aan een OHLCV DataFrame.
    Alle indicatoren zijn volledig gebaseerd op historische data (geen look-ahead).

    Gebruikt de `ta`-library (pip install ta), compatibel met Python 3.11.

    Parameters
    ----------
    df     : OHLCV DataFrame met een 'close'-kolom
    suffix : achtervoegsel voor kolomnamen (bijv. "_4h" voor 4h features)
    """
    try:
        import ta as ta_lib
    except ImportError:
        raise ImportError(
            "ta is vereist voor technische indicatoren. "
            "Installeer via: pip install ta"
        )

    close = df["close"]

    # RSI (14 perioden)
    df[f"rsi_14{suffix}"] = ta_lib.momentum.RSIIndicator(
        close=close, window=14
    ).rsi()

    # MACD (12, 26, 9)
    macd = ta_lib.trend.MACD(
        close=close, window_slow=26, window_fast=12, window_sign=9
    )
    df[f"macd{suffix}"]        = macd.macd()
    df[f"macd_signal{suffix}"] = macd.macd_signal()

    # Bollinger Bands (20 perioden, 2σ) — %B positie [0=onderband, 1=bovenband]
    bb = ta_lib.volatility.BollingerBands(close=close, window=20, window_dev=2)
    df[f"bb_pct{suffix}"] = bb.bollinger_pband()

    # EMA ratio's (close t.o.v. EMA — trendrichting en afstand)
    ema_20 = ta_lib.trend.EMAIndicator(close=close, window=20).ema_indicator()
    df[f"ema_ratio_20{suffix}"] = close / ema_20

    if suffix == "":
        # EMA50 en EMA200 alleen voor 1h (stap B+E)
        ema_50  = ta_lib.trend.EMAIndicator(close=close, window=50).ema_indicator()
        ema_200 = ta_lib.trend.EMAIndicator(close=close, window=200).ema_indicator()
        df["ema_ratio_50"]    = close / ema_50
        # price_vs_ema200 > 1.0 = boven EMA200 = bullish regime
        df["price_vs_ema200"] = close / ema_200

        # ATR genormaliseerd (stap E): True Range volatiliteit t.o.v. prijs
        atr = ta_lib.volatility.AverageTrueRange(
            high=df["high"], low=df["low"], close=close, window=14
        ).average_true_range()
        df["atr_pct"] = atr / close

    return df


# ── 4h features ───────────────────────────────────────────────────────────────

def _build_4h_features(df_4h: pd.DataFrame) -> pd.DataFrame:
    """
    Bereken technische indicatoren op 4h-candles en verschuif één periode
    om look-ahead bias te vermijden.

    Op T=04:00 (1h) wordt de 4h-candle van 00:00 gebruikt (al afgesloten).
    """
    df = df_4h.copy()
    df = _add_ta_indicators(df, suffix="_4h")

    # Shift één 4h-candle terug: op T=04:00 is de 4h-candle van 00:00 beschikbaar,
    # niet de net geopende 4h-candle van 04:00.
    cols_4h = config.FEATURE_COLS_4H
    available = [c for c in cols_4h if c in df.columns]
    df_shifted = df[available].shift(1)
    return df_shifted


# ── Hoofdfunctie ──────────────────────────────────────────────────────────────

def build_features(
    df_ohlcv: pd.DataFrame,
    p1p2: pd.DataFrame,
    p1_heatmap: pd.DataFrame,
    direction_bias: pd.DataFrame,
    df_4h: pd.DataFrame = None,
) -> pd.DataFrame:
    """
    Bouw de feature matrix voor ML-training.

    Parameters
    ----------
    df_ohlcv       : UTC-geïndexeerde 1h OHLCV DataFrame (uitvoer van load_ohlcv)
    p1p2           : dag-voor-dag P1/P2 labels (uitvoer van compute_p1p2)
    p1_heatmap     : kans-heatmap P1 (uitvoer van compute_p1_heatmap)
    direction_bias : richtingsbias heatmap (uitvoer van compute_direction_bias)
    df_4h          : 4h OHLCV DataFrame (optioneel; wordt geladen indien None)

    Returns
    -------
    pd.DataFrame met kolommen = FEATURE_COLS + ["target", "close"]
    """
    df = df_ohlcv.copy()

    # ── Tijdfeatures ──────────────────────────────────────────────────────────
    df["hour"]         = df.index.hour
    df["day_of_week"]  = df.index.dayofweek
    # hour_of_week (0–167): verfijnere tijdscodering dan dag + uur apart (stap E)
    df["hour_of_week"] = df["day_of_week"] * 24 + df["hour"]
    df["session"]      = df["hour"].map(_session)

    # ── P1/P2 statistieken als feature ────────────────────────────────────────
    df["p1_probability"] = df.apply(
        lambda r: _lookup_heatmap(p1_heatmap, r["day_of_week"], r["hour"]), axis=1
    )
    df["direction_bias"] = df.apply(
        lambda r: _lookup_heatmap(direction_bias, r["day_of_week"], r["hour"]), axis=1
    ).fillna(0.5)

    # ── Prijs- en volumefeatures (rolling, geen look-ahead) ───────────────────
    df["returns"]        = df["close"].pct_change()
    df["volatility_24h"] = df["returns"].rolling(24).std()
    df["volume_ratio"]   = df["volume"] / df["volume"].rolling(24).mean()
    df["volume_spike_48h"] = df["volume"] / (df["volume"].rolling(48).mean() + 1e-10)

    df["high_24h"]       = df["high"].rolling(24).max()
    df["low_24h"]        = df["low"].rolling(24).min()
    range_24h            = df["high_24h"] - df["low_24h"]
    df["price_position"] = (df["close"] - df["low_24h"]) / (range_24h + 1e-10)

    # Multi-horizon momentum: vangt korte-termijn momentum op de signaalhorizon
    df["return_2h"]  = df["close"].pct_change(2)
    df["return_4h"]  = df["close"].pct_change(4)
    df["return_6h"]  = df["close"].pct_change(6)
    df["return_12h"] = df["close"].pct_change(12)

    # Volatiliteitsregime: verhouding recente 4h-vol t.o.v. 24h-vol
    # > 1 = vol expandeert (breakout/capitulatie), < 1 = vol comprimeert (squeeze)
    df["vol_4h"]     = df["returns"].rolling(4).std()
    df["vol_regime"] = df["vol_4h"] / (df["volatility_24h"] + 1e-10)

    # Trendconsistentie: fractie stijgende close-to-close candles in afgelopen 12h
    df["trend_consistency_12h"] = (df["close"].diff() > 0).rolling(12).mean()

    # Macro-momentum: langetermijn trendrichting (voor bearmarktdetectie)
    df["return_7d"]  = df["close"].pct_change(7 * 24)   # 7-daags rendement
    df["return_30d"] = df["close"].pct_change(30 * 24)  # 30-daags rendement

    # Afstand van 7-daags ATH: negatief = in correctie, dichtbij 0 = near ATH
    df["ath_7d"]          = df["close"].rolling(7 * 24).max()
    df["ath_7d_distance"] = df["close"] / df["ath_7d"] - 1

    # buy_pressure: rolling fractie candles die hoger sloten dan openden (24h)
    df["buy_pressure"] = (df["close"] > df["open"]).rolling(24).mean()

    # ── Vorige dag richting ───────────────────────────────────────────────────
    p1p2 = p1p2.copy()
    p1p2["date"] = pd.to_datetime(p1p2["date"]).dt.date
    prev_return_map = p1p2.set_index("date")["day_return"].to_dict()

    def _prev_day_return(ts):
        prev_date = (ts - pd.Timedelta(days=1)).date()
        return prev_return_map.get(prev_date, 0.0)

    df["prev_day_return"] = [_prev_day_return(ts) for ts in df.index]

    # ── Technische indicatoren (1h) ───────────────────────────────────────────
    df = _add_ta_indicators(df, suffix="")

    # ── ETH/BTC ratio (marktbreedte / altcoin season indicator) ───────────────
    try:
        df_eth = load_ohlcv(symbol="ETHUSDT", interval="1h")
        eth_btc = (df_eth["close"] / df_ohlcv["close"]).reindex(df.index)
        # 24h rendement van het ETH/BTC ratio: positief = ETH outperformt = altcoin season
        df["eth_btc_ratio"] = eth_btc.pct_change(24)
    except Exception:
        df["eth_btc_ratio"] = 0.0

    # ── Externe features (Fear & Greed, SPX, EUR/USD, Funding Rate, OI) ───────
    try:
        from src.external_data import load_all_external
        df_ext = load_all_external(df.index)
        for col in df_ext.columns:
            df[col] = df_ext[col]
    except Exception as e:
        print(f"  Waarschuwing: externe data laden mislukt ({e}) — defaults gebruikt")
        for col in ["fear_greed", "spx_return_24h", "eurusd_return_24h",
                    "funding_rate", "oi_change_24h"]:
            if col not in df.columns:
                df[col] = 0.0

    # Funding rate momentum: 3-daagse verandering in funding rate
    # Positief → markt wordt meer bullish, negatief → shorts nemen het over
    if "funding_rate" in df.columns:
        df["funding_momentum"] = df["funding_rate"].diff(72)  # 72h = 3 days

    # ── 4h features (hogere timeframe context) ────────────────────────────────
    if df_4h is None:
        try:
            df_4h = load_ohlcv(symbol=config.SYMBOL, interval="4h")
        except Exception:
            df_4h = None

    if df_4h is not None and len(df_4h) > 50:
        df_4h_feat = _build_4h_features(df_4h)
        # Left-join op datetime index, forward-fill NaN (1h-candles tussen 4h-sluitingen)
        df = df.join(df_4h_feat, how="left")
        cols_4h = [c for c in config.FEATURE_COLS_4H if c in df.columns]
        df[cols_4h] = df[cols_4h].ffill()
    else:
        print("  Waarschuwing: geen 4h data beschikbaar — 4h features weggelaten.")
        for col in config.FEATURE_COLS_4H:
            df[col] = np.nan

    # ── Target variabele ──────────────────────────────────────────────────────
    # Dead zone: rijen waarbij de koersbeweging kleiner is dan TARGET_DEAD_ZONE_PCT
    # worden als "neutraal" beschouwd en uit training verwijderd.
    # Dit voorkomt dat de classifier op ruis leert: micro-bewegingen zijn
    # onvoorspelbaar en liggen onder de break-even drempel (2 × TRADE_FEE = 0.2%).
    #
    # target = 1  als future_close > close × (1 + drempel)
    # target = 0  als future_close < close × (1 - drempel)
    # NaN (neutraal) → verwijderd uit feature matrix
    h    = config.PREDICTION_HORIZON_H
    dead = config.TARGET_DEAD_ZONE_PCT
    df["future_close"] = df["close"].shift(-h)
    df["target"] = np.where(
        df["future_close"] > df["close"] * (1 + dead), 1,
        np.where(
        df["future_close"] < df["close"] * (1 - dead), 0,
        np.nan)
    )

    # ── Selecteer en schoon op ────────────────────────────────────────────────
    keep      = config.FEATURE_COLS + ["target", "close"]
    available = [c for c in keep if c in df.columns]
    df_feat   = df[available].dropna()
    df_feat["target"] = df_feat["target"].astype(int)

    n_removed = len(df[available].dropna(subset=config.FEATURE_COLS)) - len(df_feat)
    if n_removed > 0:
        pct = n_removed / (len(df_feat) + n_removed)
        print(f"  Dead zone: {n_removed} neutrale rijen verwijderd ({pct:.1%} van totaal)")

    return df_feat


if __name__ == "__main__":
    df    = load_ohlcv()
    p1p2  = pd.read_csv(config.DATA_DIR / "p1p2_labels.csv")
    p1_heatmap     = compute_p1_heatmap(p1p2)
    direction_bias = compute_direction_bias(p1p2)

    features = build_features(df, p1p2, p1_heatmap, direction_bias)
    out = config.DATA_DIR / "features.parquet"
    features.to_parquet(out)

    print(f"Feature matrix: {features.shape[0]} rijen × {features.shape[1]} kolommen")
    print(f"Opgeslagen: {out}")
    print(f"\nKolommen: {list(features.columns)}")
    print(f"\nTarget verdeling:\n{features['target'].value_counts(normalize=True)}")
    print(f"\nEerste rijen:\n{features.head()}")
