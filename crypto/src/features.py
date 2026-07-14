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


# ── Graceful degradation: neutrale defaults voor externe features ─────────────
# Wanneer externe API's (Binance funding, Bybit OI, CoinGecko, Deribit, pytrends)
# falen op de GH Actions runner door geo-block / rate-limit, blijven kolommen
# 100% null in de feature matrix. Pure dropna zou dan alle rijen verwijderen
# en `features.iloc[[-1]]` faalt met IndexError.
#
# Strategie: vóór de dropna doet `build_features` een ffill + fillna(default)
# op deze kolommen. Het model krijgt dan neutraal-conservatieve waarden i.p.v.
# een crash. Predictions verschuiven richting proba ≈ 0.5 — minder informatief
# maar pipeline blijft levend tot de API's weer beschikbaar zijn.
EXTERNAL_FEATURE_DEFAULTS = {
    # Macro (yfinance — gewoonlijk beschikbaar, maar feestdagen geven gaten)
    "spx_return_24h":          0.0,
    "eurusd_return_24h":       0.0,
    "dxy_return_24h":          0.0,
    "dxy_return_7d":           0.0,
    "usdjpy_return_24h":       0.0,
    "usdjpy_return_7d":        0.0,
    "vix_level":               20.0,   # historisch mediaan
    # Sentiment (alternative.me / pytrends)
    "fear_greed":              0.5,    # neutraal 50/100 genormaliseerd
    "fear_greed_7d_chg":       0.0,
    "google_trends_btc":       50.0,   # mediaan zoekvolume
    "trends_momentum_4w":      0.0,
    "trends_spike":            0.0,
    # Cross-asset
    "eth_btc_ratio":           0.0,
    # Crypto-specifiek (Binance/Bybit/Deribit — geo-gevoelig)
    "funding_rate":            0.0,    # neutrale funding
    "funding_momentum":        0.0,
    "oi_return_24h":           0.0,
    "oi_price_divergence":     0.0,
    "btc_dvol":                0.5,    # mediaan genormaliseerd DVOL
    # On-chain (blockchain.info)
    "active_addresses_7d_chg": 0.0,
    "hash_rate_7d_chg":        0.0,
}

# Drempel waarboven we een data-quality-waarschuwing printen voor de laatste
# rij (de inference-rij). 20% = als > 1/5 van de externe features imputed is.
DATA_QUALITY_WARN_THRESHOLD = 0.20


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


def _build_heatmap_lookup(heatmap: pd.DataFrame, default: float = 0.0) -> dict:
    """Bouw een {(dow_int, hour): value} dict voor vectorized heatmap lookup."""
    reverse_dow = {v: k for k, v in DOW_LABELS.items()}
    result = {}
    for label in heatmap.index:
        dow_int = reverse_dow.get(label)
        if dow_int is None:
            continue
        for hour in heatmap.columns:
            val = heatmap.loc[label, int(hour)]
            result[(dow_int, int(hour))] = float(val) if not np.isnan(val) else default
    return result


# ── Ichimoku Cloud ────────────────────────────────────────────────────────────

def _add_ichimoku(df: pd.DataFrame) -> pd.DataFrame:
    """
    Voeg Ichimoku Cloud features toe (T1-A).
    Standaard parameters: Tenkan=9, Kijun=26, Senkou B=52, displacement=26.

    cloud_position  : +1 close boven wolk, 0 in wolk, -1 onder wolk
    cloud_thickness : (senkou_a - senkou_b) / close — dikke wolk = sterke trend
    tk_cross        : +1 Tenkan boven Kijun, -1 eronder
    chikou_position : chikou span vs. koers 26 perioden terug (+1/0/-1)
    """
    high  = df["high"]
    low   = df["low"]
    close = df["close"]

    tenkan  = (high.rolling(9).max()  + low.rolling(9).min())  / 2
    kijun   = (high.rolling(26).max() + low.rolling(26).min()) / 2
    senkou_a = ((tenkan + kijun) / 2).shift(26)
    senkou_b = ((high.rolling(52).max() + low.rolling(52).min()) / 2).shift(26)

    cloud_top    = senkou_a.combine(senkou_b, max)
    cloud_bottom = senkou_a.combine(senkou_b, min)

    df["cloud_position"] = np.where(
        close > cloud_top,    1,
        np.where(close < cloud_bottom, -1, 0)
    ).astype(float)
    df["cloud_thickness"] = (senkou_a - senkou_b) / (close + 1e-10)
    df["tk_cross"] = np.where(tenkan > kijun, 1, -1).astype(float)

    # chikou vs close 26 perioden terug (shift +26 = kijk terug in de tijd)
    close_26ago = close.shift(26)
    df["chikou_position"] = np.where(
        close > close_26ago, 1,
        np.where(close < close_26ago, -1, 0)
    ).astype(float)

    return df


# ── RSI Divergentie (T1-F) ────────────────────────────────────────────────────

def _add_rsi_divergence(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    """
    Detecteer RSI-divergentie (T1-F) op basis van `rsi_14` kolom.

    rsi_bull_divergence : 1 als price maakt lower low maar RSI maakt higher low
                          (bullish reversal signaal)
    rsi_bear_divergence : 1 als price maakt higher high maar RSI maakt lower high
                          (bearish top signaal)
    """
    if "rsi_14" not in df.columns:
        df["rsi_bull_divergence"] = 0.0
        df["rsi_bear_divergence"] = 0.0
        return df

    close = df["close"]
    rsi   = df["rsi_14"]

    price_low_now   = close.rolling(window).min()
    price_low_prev  = price_low_now.shift(window)
    rsi_low_now     = rsi.rolling(window).min()
    rsi_low_prev    = rsi_low_now.shift(window)

    price_high_now  = close.rolling(window).max()
    price_high_prev = price_high_now.shift(window)
    rsi_high_now    = rsi.rolling(window).max()
    rsi_high_prev   = rsi_high_now.shift(window)

    df["rsi_bull_divergence"] = (
        (price_low_now  < price_low_prev) &
        (rsi_low_now    > rsi_low_prev)
    ).astype(float)

    df["rsi_bear_divergence"] = (
        (price_high_now  > price_high_prev) &
        (rsi_high_now    < rsi_high_prev)
    ).astype(float)

    return df


# ── Volume Profile / Point of Control ─────────────────────────────────────────

def _compute_poc(high_arr: np.ndarray, low_arr: np.ndarray,
                 close_arr: np.ndarray, vol_arr: np.ndarray,
                 window: int = 168, n_bins: int = 30) -> np.ndarray:
    """
    Bereken de Point of Control (POC) via een rollend volume profiel.

    De POC is het prijsniveau met het hoogste gecumuleerde volume in het
    rolling venster. Volume per candle wordt proportioneel verdeeld over
    de H-L range met numpy broadcasting (geen Python inner loops).

    Geeft (close - POC) / close terug — negatief = close onder POC (weerstand).
    """
    n = len(close_arr)
    poc_distance = np.full(n, np.nan)

    for i in range(window - 1, n):
        sl = slice(i - window + 1, i + 1)
        h = high_arr[sl]
        l = low_arr[sl]
        v = vol_arr[sl]

        price_lo = l.min()
        price_hi = h.max()
        if price_hi <= price_lo:
            poc_distance[i] = 0.0
            continue

        edges = np.linspace(price_lo, price_hi, n_bins + 1)
        # Broadcasting: overlap van elke candle (window,) met elke bin (n_bins,)
        overlap = np.maximum(
            0.0,
            np.minimum(h[:, None], edges[1:][None, :])
            - np.maximum(l[:, None], edges[:-1][None, :]),
        )
        candle_range = (h - l)[:, None] + 1e-10
        weight = overlap / candle_range
        vol_profile = (v[:, None] * weight).sum(axis=0)

        bin_centers = (edges[:-1] + edges[1:]) / 2
        poc_price = bin_centers[np.argmax(vol_profile)]
        poc_distance[i] = (close_arr[i] - poc_price) / (close_arr[i] + 1e-10)

    return poc_distance


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

    # Stochastic RSI (14 perioden, smooth 3/3) — sensitiever dan RSI
    # %K: positie van RSI binnen zijn eigen 14-perioden band [0=oversold, 1=overbought]
    stoch_rsi = ta_lib.momentum.StochRSIIndicator(
        close=close, window=14, smooth1=3, smooth2=3
    )
    df[f"stoch_rsi{suffix}"] = stoch_rsi.stochrsi_k()

    # MACD (12, 26, 9)
    macd = ta_lib.trend.MACD(
        close=close, window_slow=26, window_fast=12, window_sign=9
    )
    df[f"macd{suffix}"]        = macd.macd()
    df[f"macd_signal{suffix}"] = macd.macd_signal()

    # Bollinger Bands (20 perioden, 2σ) — %B positie en bandbreedte
    bb = ta_lib.volatility.BollingerBands(close=close, window=20, window_dev=2)
    df[f"bb_pct{suffix}"]   = bb.bollinger_pband()
    # BB breedte genormaliseerd door prijs: groeit bij volatiliteitsexpansie (breakout signaal)
    df[f"bb_width{suffix}"] = (bb.bollinger_hband() - bb.bollinger_lband()) / (close + 1e-10)

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

        # ADX (14) — trendsterkte en directional movement (Fase 1: regime detectie)
        adx_ind = ta_lib.trend.ADXIndicator(
            high=df["high"], low=df["low"], close=close, window=14
        )
        df["adx"]     = adx_ind.adx()      # trendsterkte 0-100 (>20 = trending)
        df["adx_pos"] = adx_ind.adx_pos()  # +DI: bull druk
        df["adx_neg"] = adx_ind.adx_neg()  # -DI: bear druk

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
    symbol: str = config.SYMBOL,
    keep_unlabeled: bool = False,
) -> pd.DataFrame:
    """
    Bouw de feature matrix voor ML-training of live inference.

    Parameters
    ----------
    df_ohlcv       : UTC-geïndexeerde 1h OHLCV DataFrame (uitvoer van load_ohlcv)
    p1p2           : dag-voor-dag P1/P2 labels (uitvoer van compute_p1p2)
    p1_heatmap     : kans-heatmap P1 (uitvoer van compute_p1_heatmap)
    direction_bias : richtingsbias heatmap (uitvoer van compute_direction_bias)
    df_4h          : 4h OHLCV DataFrame (optioneel; wordt geladen indien None)
    keep_unlabeled : `False` (default) drop de laatste PREDICTION_HORIZON_H rijen
                     waar `target` nog NaN is — gewenst voor training. `True`
                     houdt die rijen, zodat `df_feat.iloc[-1]` de meest recente
                     candle is (gewenst voor live inference). In `True`-mode
                     krijgen target-NaN rijen sentinel `-1` zodat astype(int)
                     niet faalt; consumers die op target trainen moeten zelf
                     `df[df.target != -1]` filteren.

    Returns
    -------
    pd.DataFrame met kolommen = FEATURE_COLS + FILTER_COLS + ["target", "close"]
    """
    df = df_ohlcv.copy()

    # ── Tijdfeatures ──────────────────────────────────────────────────────────
    df["hour"]         = df.index.hour
    df["day_of_week"]  = df.index.dayofweek
    # hour_of_week (0–167): verfijnere tijdscodering dan dag + uur apart (stap E)
    df["hour_of_week"] = df["day_of_week"] * 24 + df["hour"]
    df["session"]      = df["hour"].map(_session)

    # ── P1/P2 statistieken als feature (vectorized lookup) ────────────────────
    p1_dict  = _build_heatmap_lookup(p1_heatmap, default=0.0)
    dir_dict = _build_heatmap_lookup(direction_bias, default=0.5)
    dow_hour = list(zip(df["day_of_week"], df["hour"]))
    df["p1_probability"] = pd.Series(dow_hour, index=df.index).map(p1_dict).fillna(0.0)
    df["direction_bias"] = pd.Series(dow_hour, index=df.index).map(dir_dict).fillna(0.5)

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

    # Volume momentum: 7-daags vs 30-daags volume ratio
    # > 1 = groeiend volume t.o.v. baseline (institutionele activiteit / interesse)
    # < 1 = verkleind volume (stille markten, gebrek aan overtuiging)
    df["volume_momentum"] = (
        df["volume"].rolling(7 * 24).mean()
        / (df["volume"].rolling(30 * 24).mean() + 1e-10)
    )

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

    # ── Vorige dag richting (vectorized) ──────────────────────────────────────
    p1p2 = p1p2.copy()
    p1p2["date"] = pd.to_datetime(p1p2["date"]).dt.date
    prev_return_map = p1p2.set_index("date")["day_return"].to_dict()
    prev_dates = (df.index - pd.Timedelta(days=1)).date
    df["prev_day_return"] = pd.Series(prev_dates, index=df.index).map(prev_return_map).fillna(0.0)

    # ── Technische indicatoren (1h) ───────────────────────────────────────────
    df = _add_ta_indicators(df, suffix="")

    # ── Regime features (ADX-gebaseerd) ──────────────────────────────────────
    # adx_trend: gecombineerde richting × sterkte, bereik [-1, +1]
    #   positief = bull trend, negatief = bear trend, dichtbij 0 = ranging
    if "adx_pos" in df.columns and "adx_neg" in df.columns:
        df["adx_trend"] = (
            (df["adx_pos"] - df["adx_neg"])
            / (df["adx_pos"] + df["adx_neg"] + 1e-10)
        )
        # market_regime: bevestigd regime op basis van ADX drempel (>20 = trending)
        # +1 = bevestigde bull (ADX>20, +DI > -DI)
        #  0 = ranging (ADX≤20 of gemengde signalen)
        # -1 = bevestigde bear (ADX>20, -DI > +DI)  ← short filter gebruikt dit
        df["market_regime"] = np.where(
            (df["adx"] > 20) & (df["adx_pos"] > df["adx_neg"]),   1,
            np.where(
            (df["adx"] > 20) & (df["adx_neg"] > df["adx_pos"]), -1,
            0)
        ).astype(float)

    # MACD histogram + afgeleide features (S14-A — moet vóór ranging_score staan)
    if "macd" in df.columns and "macd_signal" in df.columns:
        macd_hist = df["macd"] - df["macd_signal"]
        df["macd_hist"]       = macd_hist                  # ruwe histogram (prijs-eenheden)
        df["macd_hist_slope"] = macd_hist.diff(3)          # 3h verandering (reversal indicator)
        # S14-A: MACD histogram stabiliteit — rolling std(5) van genormaliseerde histogram.
        # Laag = histogram beweegt gestaag in één richting (trending).
        # Hoog = histogram oscilleert sterk (ranging / kantelende markt).
        macd_hist_norm = macd_hist / (df["close"] + 1e-10)
        df["macd_hist_stability"] = macd_hist_norm.rolling(5).std()

        # S17-B: MACD momentum-gewogen positiescaling (FILTER_COL).
        # Sterk positief MACD-momentum → hogere positiegrootte bij entry.
        # Normalisatie: positief histogram / 20-daags gemiddelde van |histogram|.
        # Clip [0, 0.5] → size_multiplier = 1 + macd_size_mult ∈ [1.0, 1.5].
        mean_abs_hist = macd_hist.abs().rolling(20 * 24).mean()
        df["macd_size_mult"] = (
            macd_hist.clip(lower=0) / (mean_abs_hist + 1e-10)
        ).clip(0, 0.5)

    # S14-B: Ensemble ranging score (3 indicatoren gecombineerd).
    # Elk signaal voegt 1 punt toe als het een ranging/kantelende markt suggereert:
    #   adx_ranging  — ADX < 20: geen bevestigde trend
    #   bb_squeeze   — BB-breedte < drempel: prijs in nauwe band (lage vol = ranging)
    #   macd_noisy   — MACD-histogram stabiliteit > drempel: onstabiel momentum
    # ranging_score ≥ 2 → backtest filtert shorts (te groot risico false-bear signaal)
    if all(c in df.columns for c in ["adx", "bb_width", "macd_hist_stability"]):
        bb_thr   = getattr(config, "RANGING_BB_WIDTH_THR", 0.025)
        stb_thr  = getattr(config, "RANGING_MACD_STB_THR", 0.0002)
        adx_ranging = (df["adx"] < 20).astype(int)
        bb_squeeze  = (df["bb_width"] < bb_thr).astype(int)
        macd_noisy  = (df["macd_hist_stability"] > stb_thr).astype(int)
        df["ranging_score"] = adx_ranging + bb_squeeze + macd_noisy

    # S16-B / S19-A7: Crash-modus 3-tier detector (FILTER_COL — niet als model-input).
    # crash_mode = 0 (geen), 1 (mild >1σ), 2 (ernstig >2.5σ of >10% dag), 3 (extreem >5σ)
    # Backtest past positiegrootte aan per tier: 0.75× / 0.50× / 0.25×
    if "returns" in df.columns and "volatility_24h" in df.columns:
        sigma_mild    = getattr(config, "CRASH_SIGMA_THR_MILD",    1.0)
        sigma_thr     = getattr(config, "CRASH_SIGMA_THR",         2.5)
        sigma_extreme = getattr(config, "CRASH_SIGMA_THR_EXTREME", 5.0)
        ret_thr       = getattr(config, "CRASH_RETURN_THR",       0.10)
        vol_ref       = df["volatility_24h"].shift(1).fillna(0.02)
        ret_24h       = df["close"].pct_change(24)
        tier1 = df["returns"] < -(sigma_mild    * vol_ref)
        tier2 = df["returns"] < -(sigma_thr     * vol_ref)
        tier3 = df["returns"] < -(sigma_extreme * vol_ref)
        large_drop = ret_24h < -ret_thr
        import numpy as _np
        crash = _np.zeros(len(df), dtype=int)
        crash[tier1.values] = 1
        crash[tier2.values] = 2
        crash[(tier3 | large_drop).values] = 3
        df["crash_mode"] = crash
    else:
        df["crash_mode"] = 0

    # EMA alignment score: hoeveel EMAs zijn in bull-volgorde gestapeld (0–3)
    # Logica: ema_ratio = close/ema, dus kleinere ratio = EMA hoger dan bij grotere ratio
    #   ema20 > ema50  ↔  ema_ratio_20 < ema_ratio_50
    #   ema50 > ema200 ↔  ema_ratio_50 < price_vs_ema200
    if all(c in df.columns for c in ["ema_ratio_20", "ema_ratio_50", "price_vs_ema200"]):
        ema20_above_ema50  = (df["ema_ratio_20"]    < df["ema_ratio_50"]).astype(int)
        ema50_above_ema200 = (df["ema_ratio_50"]    < df["price_vs_ema200"]).astype(int)
        close_above_ema200 = (df["price_vs_ema200"] > 1.0).astype(int)
        df["ema_alignment"] = ema20_above_ema50 + ema50_above_ema200 + close_above_ema200

    # VWAP-afstand: dagelijks hersteld om 00:00 UTC
    # Positief = close boven daag-VWAP (intraday bullish), negatief = onder VWAP
    df["_day"]     = df.index.normalize()
    df["_typical"] = (df["high"] + df["low"] + df["close"]) / 3
    df["_tpv"]     = df["_typical"] * df["volume"]
    df["_cum_tpv"] = df.groupby("_day")["_tpv"].cumsum()
    df["_cum_vol"] = df.groupby("_day")["volume"].cumsum()
    df["vwap_distance"] = (
        (df["close"] - df["_cum_tpv"] / (df["_cum_vol"] + 1e-10))
        / (df["close"] + 1e-10)
    )
    df.drop(columns=["_day", "_typical", "_tpv", "_cum_tpv", "_cum_vol"],
            inplace=True, errors="ignore")

    # NB: HMM-regime, halving-cyclus, Supertrend, BTC-ETH-correlatie en candle-
    # patterns worden niet meer berekend — hun kolommen stonden niet in
    # FEATURE_COLS (verwijderd na Sprint 2/5-regressies, zie config.py) en
    # werden bij de selectie hieronder direct weggegooid.

    # ── Ichimoku Cloud (T1-A) ─────────────────────────────────────────────────
    df = _add_ichimoku(df)

    # ── RSI Divergentie (T1-F) ────────────────────────────────────────────────
    # Moet na _add_ta_indicators worden aangeroepen (rsi_14 vereist).
    df = _add_rsi_divergence(df)

    # ── Volume Profile / Point of Control (B3) ────────────────────────────────
    # POC = prijsniveau met het meeste volume in het afgelopen 168h venster.
    # poc_distance = (close − POC) / close:
    #   > 0 → close boven POC (POC als steun)
    #   < 0 → close onder POC (POC als weerstand)
    df["poc_distance"] = _compute_poc(
        df["high"].values, df["low"].values,
        df["close"].values, df["volume"].values,
        window=168, n_bins=30,
    )

    # ── ETH/BTC ratio (marktbreedte / dominantie indicator) ───────────────────
    # Voor BTC-model: ETH/BTC ratio — positief = ETH outperformt = altcoin season
    # Voor ETH-model: BTC/ETH ratio — BTC dominantie-indicator (omgekeerd perspectief)
    # Kolomnaam blijft "eth_btc_ratio" zodat de feature list identiek blijft.
    try:
        if symbol == "BTCUSDT":
            df_other = load_ohlcv(symbol="ETHUSDT", interval="1h")
            ratio = (df_other["close"] / df_ohlcv["close"]).reindex(df.index)
        else:
            df_other = load_ohlcv(symbol="BTCUSDT", interval="1h")
            ratio = (df_other["close"] / df_ohlcv["close"]).reindex(df.index)
        df["eth_btc_ratio"] = ratio.pct_change(24)
    except Exception:
        df["eth_btc_ratio"] = 0.0

    # ── Externe features (Fear & Greed, SPX, EUR/USD, Funding Rate, OI) ───────
    try:
        from src.external_data import load_all_external
        df_ext = load_all_external(df.index, symbol=symbol)
        new_cols = df_ext.columns.difference(df.columns)
        existing_cols = df_ext.columns.intersection(df.columns)
        # Update existing columns in-place, batch-concat new ones
        for col in existing_cols:
            df[col] = df_ext[col]
        if len(new_cols):
            df = pd.concat([df, df_ext[new_cols]], axis=1)
        df = df.copy()  # defragmenteer
    except (KeyError, ValueError, TypeError) as e:
        print(f"  Waarschuwing: externe data laden mislukt ({e}) — defaults gebruikt")
        import traceback; traceback.print_exc()
        for col in ["fear_greed", "spx_return_24h", "eurusd_return_24h",
                    "funding_rate", "oi_change_24h"]:
            if col not in df.columns:
                df[col] = 0.0

    # Funding rate momentum: 3-daagse verandering in funding rate
    # Positief → markt wordt meer bullish, negatief → shorts nemen het over
    if "funding_rate" in df.columns:
        df["funding_momentum"] = df["funding_rate"].diff(72)  # 72h = 3 days

    # Fear & Greed momentum: 7-daagse verandering in sentiment index (S7-C)
    # Positief = sentiment verbetert, negatief = sentiment verslechtert
    # Gebruik 168h diff (7 dagen × 24h) want daily data is forward-filled naar uurlijks
    if "fear_greed" in df.columns:
        df["fear_greed_7d_chg"] = df["fear_greed"].diff(168)  # 168h = 7 dagen

    # S7-A: OI price divergence — vergelijk OI richting met prijs richting
    # +1 = OI en prijs stijgen samen (bullish confirmation: nieuwe longs stromen in)
    # -1 = OI stijgt, prijs daalt (bearish: nieuwe shorts domineren)
    #  0 = OI daalt (minder conviction) of neutraal
    if "oi_return_24h" in df.columns:
        price_return_24h = df["close"].pct_change(24)
        oi_up    = df["oi_return_24h"] > 0.005   # >0.5% OI stijging
        oi_down  = df["oi_return_24h"] < -0.005  # >0.5% OI daling
        px_up    = price_return_24h > 0.002
        px_down  = price_return_24h < -0.002
        df["oi_price_divergence"] = np.where(
            oi_up  & px_up,   1,
            np.where(oi_up  & px_down, -1, 0)
        ).astype(np.int8)

    # ── 4h features (hogere timeframe context) ────────────────────────────────
    if df_4h is None:
        try:
            df_4h = load_ohlcv(symbol=symbol, interval="4h")
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
    # B4: Asymmetrische dead zone — aparte drempels voor ups en downs.
    # Kleine opwaartse moves zijn informatief (accumulatie), kleine neerwaartse
    # moves zijn vaker ruis. Symmetrische dead zone zou te veel goede long-
    # voorbeelden weggooien.
    #
    # target = 1  als future_close > close × (1 + dead_up)
    # target = 0  als future_close < close × (1 - dead_down)
    # NaN (neutraal) → verwijderd uit feature matrix
    df = df.copy()  # defragmenteer voor target-kolommen
    h         = config.PREDICTION_HORIZON_H
    dead_up   = getattr(config, "TARGET_DEAD_ZONE_UP",   config.TARGET_DEAD_ZONE_PCT)
    dead_down = getattr(config, "TARGET_DEAD_ZONE_DOWN",  config.TARGET_DEAD_ZONE_PCT)
    df["future_close"] = df["close"].shift(-h)
    df["target"] = np.where(
        df["future_close"] > df["close"] * (1 + dead_up), 1,
        np.where(
        df["future_close"] < df["close"] * (1 - dead_down), 0,
        np.nan)
    )

    # ── Graceful degradation — vul externe features met neutrale defaults ─────
    # Wanneer externe API's falen (Binance funding, Bybit OI, CoinGecko,
    # Deribit, pytrends) blijven kolommen NaN. ffill + fillna(default) zorgt
    # dat dropna niet alle rijen verwijdert. Tellen voor data-quality-warning
    # gebeurt op de laatste rij (de inference-candidaat).
    last_row_imputed = []
    for col, default in EXTERNAL_FEATURE_DEFAULTS.items():
        if col not in df.columns:
            continue
        last_val_pre = df[col].iloc[-1] if len(df) else None
        df[col] = df[col].ffill().fillna(default)
        last_val_post = df[col].iloc[-1] if len(df) else None
        # Imputed = laatste rij was NaN, nu defaulted (ffill kon de gap niet vullen)
        if pd.isna(last_val_pre) and last_val_post == default:
            last_row_imputed.append(col)

    n_external_present = sum(1 for c in EXTERNAL_FEATURE_DEFAULTS if c in df.columns)
    if n_external_present > 0 and last_row_imputed:
        imputed_pct = len(last_row_imputed) / n_external_present
        if imputed_pct >= DATA_QUALITY_WARN_THRESHOLD:
            preview = ", ".join(last_row_imputed[:5])
            extra   = f" (+{len(last_row_imputed) - 5} meer)" if len(last_row_imputed) > 5 else ""
            print(f"  ⚠️  Data quality: {len(last_row_imputed)}/{n_external_present} externe "
                  f"features imputed voor laatste rij ({imputed_pct:.0%}) — predictions "
                  f"kunnen minder informatief zijn.")
            print(f"      Imputed: {preview}{extra}")

    # ── Selecteer en schoon op ────────────────────────────────────────────────
    # FILTER_COLS (bijv. market_regime) worden ALTIJD meegenomen voor de backtest-filter,
    # maar staan NIET in FEATURE_COLS — worden dus niet als model-input gebruikt.
    filter_cols = getattr(config, "FILTER_COLS", [])
    keep      = config.FEATURE_COLS + filter_cols + ["target", "close"]
    available = [c for c in keep if c in df.columns]

    # Critical kolommen die ALTIJD non-null moeten zijn (price, market regime).
    # `target` is voor de laatste PREDICTION_HORIZON_H rijen NaN — we droppen die
    # standaard (training), maar NIET in inference-mode (`keep_unlabeled=True`)
    # zodat `df_feat.iloc[-1]` de actuele candle is i.p.v. een 24h-stale rij.
    critical_cols = [c for c in ["close"] + filter_cols if c in df.columns]

    df_feat = df[available].dropna(subset=critical_cols)
    if not keep_unlabeled:
        df_feat = df_feat.dropna(subset=["target"])

    # target naar int (sentinel -1 voor unlabeled rijen in inference-mode);
    # consumers die op target trainen filteren `df[df.target != -1]`.
    if "target" in df_feat.columns:
        df_feat["target"] = df_feat["target"].fillna(-1).astype(int)

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
