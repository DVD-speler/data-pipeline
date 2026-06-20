"""
SMC-features v1 — market structure / BOS / FVG (Smart Money Concepts).

Operationaliseert de kern van docs/crypto/smc_framework.md naar OHLC-features.
STRIKT CAUSAAL: een pivot op index j wordt pas gebruikt vanaf bar j+N
(als de N toekomst-bars die de pivot bevestigen bestaan en ≤ t zijn).
FVG's gebruiken alleen t-2..t. Geen enkele feature kijkt vooruit.

v1 bewust compact (6 features) zodat we incrementeel kunnen meten of SMC
iets toevoegt boven de leak-reduced baseline (~0.51 OOS AUC). Significante-
vs-insignificante structuur (retrospectief) is een v2-verfijning.

Features:
  smc_bias              +1 bullish / -1 bearish / 0 neutraal (laatste BOS-richting)
  smc_bars_since_bos    bars sinds de laatste break of structure (vers = klein)
  smc_dist_swing_high   (close - laatste bevestigde swing high) / close
  smc_dist_swing_low    (close - laatste bevestigde swing low)  / close
  smc_bull_fvg_K        aantal bullish FVG's in de laatste K bars
  smc_bear_fvg_K        aantal bearish FVG's in de laatste K bars
"""

import numpy as np
import pandas as pd

SMC_COLS = [
    "smc_bias",
    "smc_bars_since_bos",
    "smc_dist_swing_high",
    "smc_dist_swing_low",
    "smc_bull_fvg_12",
    "smc_bear_fvg_12",
]


def _pivots(high: np.ndarray, low: np.ndarray, n: int):
    """Swing high/low: extremum t.o.v. n bars links én rechts."""
    T = len(high)
    ph = np.zeros(T, bool)
    pl = np.zeros(T, bool)
    for i in range(n, T - n):
        if high[i] == high[i - n : i + n + 1].max():
            ph[i] = True
        if low[i] == low[i - n : i + n + 1].min():
            pl[i] = True
    return ph, pl


def build_smc_features(df: pd.DataFrame, n: int = 5, fvg_lookback: int = 12) -> pd.DataFrame:
    """OHLC-DataFrame (open/high/low/close, UTC-index) -> SMC-feature-matrix."""
    o = df["open"].to_numpy(float)
    h = df["high"].to_numpy(float)
    l = df["low"].to_numpy(float)
    c = df["close"].to_numpy(float)
    idx = df.index
    T = len(df)

    ph, pl = _pivots(h, l, n)

    bias = np.zeros(T)
    bars_since = np.full(T, np.nan)
    dist_hi = np.full(T, np.nan)
    dist_lo = np.full(T, np.nan)

    last_sh = np.nan   # laatste bevestigde, nog-niet-gebroken swing high
    last_sl = np.nan
    cur_bias = 0
    last_bos = -1

    for t in range(T):
        # Bevestig de pivot die op deze bar 'bekend' wordt (index t-n).
        j = t - n
        if j >= 0:
            if ph[j]:
                last_sh = h[j]
            if pl[j]:
                last_sl = l[j]

        # Break of structure: candle opent voorbij EN sluit voorbij de swing
        # (de close-rule uit het framework; wick-alleen telt niet als BOS).
        if not np.isnan(last_sh) and o[t] < last_sh and c[t] > last_sh:
            cur_bias = 1
            last_bos = t
            last_sh = np.nan          # gebroken — wacht op nieuwe swing
        elif not np.isnan(last_sl) and o[t] > last_sl and c[t] < last_sl:
            cur_bias = -1
            last_bos = t
            last_sl = np.nan

        bias[t] = cur_bias
        bars_since[t] = (t - last_bos) if last_bos >= 0 else np.nan
        dist_hi[t] = (c[t] - last_sh) / c[t] if not np.isnan(last_sh) else np.nan
        dist_lo[t] = (c[t] - last_sl) / c[t] if not np.isnan(last_sl) else np.nan

    # Fair Value Gaps (3-candle, alleen t-2..t -> causaal)
    bull_fvg = np.zeros(T)
    bear_fvg = np.zeros(T)
    for t in range(2, T):
        if c[t] > o[t] and c[t - 1] > o[t - 1] and c[t - 2] > o[t - 2] and l[t] > h[t - 2]:
            bull_fvg[t] = 1.0
        if c[t] < o[t] and c[t - 1] < o[t - 1] and c[t - 2] < o[t - 2] and h[t] < l[t - 2]:
            bear_fvg[t] = 1.0

    out = pd.DataFrame(index=idx)
    out["smc_bias"] = bias
    out["smc_bars_since_bos"] = np.clip(bars_since, 0, 200)
    out["smc_dist_swing_high"] = dist_hi
    out["smc_dist_swing_low"] = dist_lo
    out["smc_bull_fvg_12"] = (
        pd.Series(bull_fvg, index=idx).rolling(fvg_lookback, min_periods=1).sum().to_numpy()
    )
    out["smc_bear_fvg_12"] = (
        pd.Series(bear_fvg, index=idx).rolling(fvg_lookback, min_periods=1).sum().to_numpy()
    )
    return out
