"""
Event-driven SMC-backtest — v1 mechaniseerbare kern.

Volgt docs/crypto/smc_backtest_spec.md. Setup (long; short symmetrisch):
  bias (HTF) mee  +  liquidity sweep van een swing  +  MSB (close-rule)
  +  order-block-zone  ->  limit-entry op de zone, SL voorbij de sweep, TP = 2R.

STRIKT CAUSAAL: swings via pivot-lag N; daily-bias pas beschikbaar de dag ná de
daily-candle; alle detectie en simulatie gebruiken uitsluitend bars <= t.
Eén trade tegelijk (geen overlap). Intrabar SL-first.

Geeft een lijst trades (elk met rendement in R, ná kosten) + metrics.
"""

import numpy as np
import pandas as pd


def _atr(h, l, c, period=14):
    prev_c = np.roll(c, 1)
    prev_c[0] = c[0]
    tr = np.maximum(h - l, np.maximum(np.abs(h - prev_c), np.abs(l - prev_c)))
    return pd.Series(tr).rolling(period, min_periods=1).mean().to_numpy()


def _last_swings(high, low, n):
    """Per bar t: laatste BEVESTIGDE swing high/low (prijs), causaal (lag n)."""
    T = len(high)
    lsh = np.full(T, np.nan)
    lsl = np.full(T, np.nan)
    cur_h = np.nan
    cur_l = np.nan
    for t in range(T):
        j = t - n
        if j - n >= 0:
            if high[j] == high[j - n : j + n + 1].max():
                cur_h = high[j]
            if low[j] == low[j - n : j + n + 1].min():
                cur_l = low[j]
        lsh[t] = cur_h
        lsl[t] = cur_l
    return lsh, lsl


def daily_bias(df_1d, n=5):
    """+1/-1/0 dagelijkse structuur-bias, beschikbaar de dag NA de candle."""
    o = df_1d["open"].to_numpy(float)
    h = df_1d["high"].to_numpy(float)
    l = df_1d["low"].to_numpy(float)
    c = df_1d["close"].to_numpy(float)
    lsh, lsl = _last_swings(h, l, n)
    bias = np.zeros(len(c))
    cur = 0
    for t in range(len(c)):
        if not np.isnan(lsh[t]) and o[t] < lsh[t] and c[t] > lsh[t]:
            cur = 1
        elif not np.isnan(lsl[t]) and o[t] > lsl[t] and c[t] < lsl[t]:
            cur = -1
        bias[t] = cur
    s = pd.Series(bias, index=df_1d.index + pd.Timedelta(days=1))
    s = s[~s.index.duplicated(keep="last")]
    return s


def _sim_exit(h, l, sl, tp, direction, start, T):
    """Intrabar SL-first vanaf bar `start`. Retour (outcome_mult, exit_idx)."""
    for s in range(start, T):
        if direction == 1:
            if l[s] <= sl:
                return -1.0, s
            if h[s] >= tp:
                return +1.0, s   # × rr buiten deze functie
        else:
            if h[s] >= sl:
                return -1.0, s
            if l[s] <= tp:
                return +1.0, s
    return None, None


def run_smc_backtest(df_4h, df_1d, *, n=5, rr=2.0, sweep_mult=0.1,
                     msb_window=6, fill_window=10, sl_buf=0.0005,
                     fee=0.0006, slip=0.0002):
    o = df_4h["open"].to_numpy(float)
    h = df_4h["high"].to_numpy(float)
    l = df_4h["low"].to_numpy(float)
    c = df_4h["close"].to_numpy(float)
    idx = df_4h.index
    T = len(df_4h)
    atr = _atr(h, l, c)
    lsh, lsl = _last_swings(h, l, n)

    bias_s = daily_bias(df_1d, n).reindex(idx, method="ffill").fillna(0)
    bias = bias_s.to_numpy()

    trades = []
    i = 0
    while i < T:
        adv = 1
        b = bias[i]

        # ---- LONG setup ----
        if (b == 1 and not np.isnan(lsl[i]) and not np.isnan(lsh[i])
                and l[i] < lsl[i] and c[i] > lsl[i]
                and (lsl[i] - l[i]) >= sweep_mult * atr[i]):
            sweep_low, ref_high = l[i], lsh[i]
            msb = next((k for k in range(i + 1, min(i + msb_window + 1, T))
                        if o[k] < ref_high and c[k] > ref_high), None)
            if msb is not None:
                ob = next((q for q in range(msb, i - 1, -1) if c[q] < o[q]), None)
                if ob is not None:
                    entry = h[ob]
                    sl = sweep_low * (1 - sl_buf)
                    if entry > sl:
                        risk = entry - sl
                        tp = entry + rr * risk
                        fill = next((f for f in range(msb + 1, min(msb + fill_window + 1, T))
                                     if l[f] <= entry), None)
                        if fill is not None:
                            mult, ex = _sim_exit(h, l, sl, tp, 1, fill + 1, T)
                            if mult is not None:
                                r = (rr if mult > 0 else -1.0) - (fee + slip) * 2 * entry / risk
                                trades.append(dict(dir="L", entry_time=idx[fill],
                                                   exit_time=idx[ex], r=r,
                                                   win=mult > 0, entry=entry, sl=sl, tp=tp,
                                                   risk_pct=risk / entry, bars_held=ex - fill,
                                                   reason="TP" if mult > 0 else "SL"))
                                adv = ex + 1 - i

        # ---- SHORT setup ----
        elif (b == -1 and not np.isnan(lsh[i]) and not np.isnan(lsl[i])
                and h[i] > lsh[i] and c[i] < lsh[i]
                and (h[i] - lsh[i]) >= sweep_mult * atr[i]):
            sweep_high, ref_low = h[i], lsl[i]
            msb = next((k for k in range(i + 1, min(i + msb_window + 1, T))
                        if o[k] > ref_low and c[k] < ref_low), None)
            if msb is not None:
                ob = next((q for q in range(msb, i - 1, -1) if c[q] > o[q]), None)
                if ob is not None:
                    entry = l[ob]
                    sl = sweep_high * (1 + sl_buf)
                    if sl > entry:
                        risk = sl - entry
                        tp = entry - rr * risk
                        fill = next((f for f in range(msb + 1, min(msb + fill_window + 1, T))
                                     if h[f] >= entry), None)
                        if fill is not None:
                            mult, ex = _sim_exit(h, l, sl, tp, -1, fill + 1, T)
                            if mult is not None:
                                r = (rr if mult > 0 else -1.0) - (fee + slip) * 2 * entry / risk
                                trades.append(dict(dir="S", entry_time=idx[fill],
                                                   exit_time=idx[ex], r=r,
                                                   win=mult > 0, entry=entry, sl=sl, tp=tp,
                                                   risk_pct=risk / entry, bars_held=ex - fill,
                                                   reason="TP" if mult > 0 else "SL"))
                                adv = ex + 1 - i

        i += max(adv, 1)
    return trades


def metrics(trades):
    n = len(trades)
    if n == 0:
        return {"n_trades": 0}
    rs = np.array([t["r"] for t in trades])
    wins = rs[rs > 0]
    losses = rs[rs <= 0]
    gross_w = wins.sum()
    gross_l = -losses.sum()
    return {
        "n_trades": n,
        "n_long": sum(t["dir"] == "L" for t in trades),
        "n_short": sum(t["dir"] == "S" for t in trades),
        "win_rate": float((rs > 0).mean()),
        "expectancy_R": float(rs.mean()),
        "total_R": float(rs.sum()),
        "profit_factor": float(gross_w / gross_l) if gross_l > 0 else float("inf"),
        "best_R": float(rs.max()),
        "worst_R": float(rs.min()),
    }
