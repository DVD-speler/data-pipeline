"""
Event-driven SMC-backtest (docs/crypto/smc_backtest_spec.md).

Setup (long; short symmetrisch):
  HTF-bias mee + liquidity sweep + MSB (close-rule) + order-block-zone
  -> limit-entry op de zone, SL voorbij de sweep, TP = rr·R of liquidity-pool.

Vlaggen:
  use_inducement     front-run pivot vóór de zone-tap (twee-legs pullback)
  use_pool_targets   TP = eerstvolgende echte liquidity pool (oude swing) i.p.v. vaste 2R
  use_smt            SMT-divergentie met gecorreleerd asset (df_smt): neem de long
                     alleen als het andere asset zijn low NIET meeneemt (vals breakout)

STRIKT CAUSAAL. Eén trade tegelijk. Intrabar SL-first, beheer vanaf de bar ná fill.
"""

import numpy as np
import pandas as pd


def _atr(h, l, c, period=14):
    prev_c = np.roll(c, 1)
    prev_c[0] = c[0]
    tr = np.maximum(h - l, np.maximum(np.abs(h - prev_c), np.abs(l - prev_c)))
    return pd.Series(tr).rolling(period, min_periods=1).mean().to_numpy()


def _last_swings(high, low, n):
    T = len(high)
    lsh = np.full(T, np.nan)
    lsl = np.full(T, np.nan)
    cur_h = cur_l = np.nan
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


def _pivot_lists(high, low, n):
    """Bevestigde pivots als (confirm_bar, prijs)-lijsten, causaal (confirm = j+n)."""
    highs, lows = [], []
    T = len(high)
    for j in range(n, T - n):
        if high[j] == high[j - n : j + n + 1].max():
            highs.append((j + n, high[j]))
        if low[j] == low[j - n : j + n + 1].min():
            lows.append((j + n, low[j]))
    return highs, lows


def htf_bias(df_bias, avail_hours, n=5):
    o = df_bias["open"].to_numpy(float)
    h = df_bias["high"].to_numpy(float)
    l = df_bias["low"].to_numpy(float)
    c = df_bias["close"].to_numpy(float)
    lsh, lsl = _last_swings(h, l, n)
    bias = np.zeros(len(c))
    cur = 0
    for t in range(len(c)):
        if not np.isnan(lsh[t]) and o[t] < lsh[t] and c[t] > lsh[t]:
            cur = 1
        elif not np.isnan(lsl[t]) and o[t] > lsl[t] and c[t] < lsl[t]:
            cur = -1
        bias[t] = cur
    s = pd.Series(bias, index=df_bias.index + pd.Timedelta(hours=avail_hours))
    return s[~s.index.duplicated(keep="last")]


def _find_fill(low, high, entry, msb, end, direction, use_ind):
    ind_ok = not use_ind
    for f in range(msb + 2, end):
        if use_ind and not ind_ok:
            if direction == 1:
                if low[f - 1] < low[f - 2] and low[f - 1] < low[f] and low[f - 1] > entry:
                    ind_ok = True
                    continue
            else:
                if high[f - 1] > high[f - 2] and high[f - 1] > high[f] and high[f - 1] < entry:
                    ind_ok = True
                    continue
        if ind_ok:
            if direction == 1 and low[f] <= entry:
                return f
            if direction == -1 and high[f] >= entry:
                return f
    return None


def _pool_tp(direction, entry, risk, fill, highs, lows, rr_min):
    """Eerstvolgende liquidity pool (oude swing) die ≥ rr_min R geeft. None = skip."""
    if direction == 1:
        cand = sorted(pr for cb, pr in highs if cb <= fill and pr > entry)
        for pr in cand:
            if (pr - entry) / risk >= rr_min:
                return pr
    else:
        cand = sorted((pr for cb, pr in lows if cb <= fill and pr < entry), reverse=True)
        for pr in cand:
            if (entry - pr) / risk >= rr_min:
                return pr
    return None


def _sim_exit(h, l, c, entry, sl, tp, risk, rr, direction, start, T, max_hold):
    end = min(start + max_hold, T) if max_hold else T
    for s in range(start, end):
        if direction == 1:
            if l[s] <= sl:
                return -1.0, s, "SL"
            if h[s] >= tp:
                return rr, s, "TP"
        else:
            if h[s] >= sl:
                return -1.0, s, "SL"
            if l[s] <= tp:
                return rr, s, "TP"
    last = end - 1
    if last < start:
        return None, None, None
    r = (c[last] - entry) / risk if direction == 1 else (entry - c[last]) / risk
    return r, last, "TIME"


def run_smc_backtest(df_exec, df_bias, *, avail_hours, df_smt=None, n=5, rr=2.0,
                     sweep_mult=0.1, msb_window=24, fill_window=40, sl_buf=0.0005,
                     fee=0.0006, slip=0.0002, max_risk_pct=0.05, max_hold=240,
                     use_inducement=True, use_pool_targets=False, use_smt=False,
                     smt_window=3):
    o = df_exec["open"].to_numpy(float)
    h = df_exec["high"].to_numpy(float)
    l = df_exec["low"].to_numpy(float)
    c = df_exec["close"].to_numpy(float)
    idx = df_exec.index
    ts = idx.values.astype("int64")
    T = len(df_exec)
    atr = _atr(h, l, c)
    lsh, lsl = _last_swings(h, l, n)
    highs, lows = _pivot_lists(h, l, n)
    bias = htf_bias(df_bias, avail_hours, n).reindex(idx, method="ffill").fillna(0).to_numpy()

    # SMT (gecorreleerd asset)
    if use_smt and df_smt is not None:
        e_ts = df_smt.index.values.astype("int64")
        e_h = df_smt["high"].to_numpy(float)
        e_l = df_smt["low"].to_numpy(float)
        e_lsh, e_lsl = _last_swings(e_h, e_l, n)
    else:
        use_smt = False

    def smt_div(direction, t_ns):
        p = int(np.searchsorted(e_ts, t_ns, side="right")) - 1
        if p < 0:
            return False
        w0 = max(0, p - smt_window)
        if direction == 1:
            return (not np.isnan(e_lsl[p])) and e_l[w0:p + 1].min() > e_lsl[p]
        return (not np.isnan(e_lsh[p])) and e_h[w0:p + 1].max() < e_lsh[p]

    cost = lambda entry, risk: (fee + slip) * 2 * entry / risk  # noqa: E731
    trades = []
    i = 0
    while i < T:
        adv = 1
        b = bias[i]
        end = min(i + msb_window + 1, T)

        if (b == 1 and not np.isnan(lsl[i]) and not np.isnan(lsh[i])
                and l[i] < lsl[i] and c[i] > lsl[i]
                and (lsl[i] - l[i]) >= sweep_mult * atr[i]
                and (not use_smt or smt_div(1, ts[i]))):
            sweep_low, ref_high = l[i], lsh[i]
            msb = next((k for k in range(i + 1, end) if o[k] < ref_high and c[k] > ref_high), None)
            if msb is not None:
                ob = next((q for q in range(msb, i - 1, -1) if c[q] < o[q]), None)
                if ob is not None:
                    entry = h[ob]
                    sl = sweep_low * (1 - sl_buf)
                    if entry > sl and (entry - sl) / entry <= max_risk_pct:
                        risk = entry - sl
                        fill = _find_fill(l, h, entry, msb, min(msb + fill_window + 1, T),
                                          1, use_inducement)
                        if fill is not None:
                            tp = (_pool_tp(1, entry, risk, fill, highs, lows, rr)
                                  if use_pool_targets else entry + rr * risk)
                            if tp is not None:
                                rr_a = (tp - entry) / risk
                                rg, ex, reason = _sim_exit(h, l, c, entry, sl, tp, risk, rr_a,
                                                           1, fill + 1, T, max_hold)
                                if rg is not None:
                                    trades.append(dict(dir="L", entry_time=idx[fill],
                                                       exit_time=idx[ex], r=rg - cost(entry, risk),
                                                       win=rg > 0, risk_pct=risk / entry,
                                                       bars_held=ex - fill, reason=reason, rr=rr_a))
                                    adv = ex + 1 - i

        elif (b == -1 and not np.isnan(lsh[i]) and not np.isnan(lsl[i])
                and h[i] > lsh[i] and c[i] < lsh[i]
                and (h[i] - lsh[i]) >= sweep_mult * atr[i]
                and (not use_smt or smt_div(-1, ts[i]))):
            sweep_high, ref_low = h[i], lsl[i]
            msb = next((k for k in range(i + 1, end) if o[k] > ref_low and c[k] < ref_low), None)
            if msb is not None:
                ob = next((q for q in range(msb, i - 1, -1) if c[q] > o[q]), None)
                if ob is not None:
                    entry = l[ob]
                    sl = sweep_high * (1 + sl_buf)
                    if sl > entry and (sl - entry) / entry <= max_risk_pct:
                        risk = sl - entry
                        fill = _find_fill(l, h, entry, msb, min(msb + fill_window + 1, T),
                                          -1, use_inducement)
                        if fill is not None:
                            tp = (_pool_tp(-1, entry, risk, fill, highs, lows, rr)
                                  if use_pool_targets else entry - rr * risk)
                            if tp is not None:
                                rr_a = (entry - tp) / risk
                                rg, ex, reason = _sim_exit(h, l, c, entry, sl, tp, risk, rr_a,
                                                           -1, fill + 1, T, max_hold)
                                if rg is not None:
                                    trades.append(dict(dir="S", entry_time=idx[fill],
                                                       exit_time=idx[ex], r=rg - cost(entry, risk),
                                                       win=rg > 0, risk_pct=risk / entry,
                                                       bars_held=ex - fill, reason=reason, rr=rr_a))
                                    adv = ex + 1 - i

        i += max(adv, 1)
    return trades


def metrics(trades):
    n = len(trades)
    if n == 0:
        return {"n_trades": 0}
    rs = np.array([t["r"] for t in trades])
    gross_w = rs[rs > 0].sum()
    gross_l = -rs[rs <= 0].sum()
    return {
        "n_trades": n,
        "n_long": sum(t["dir"] == "L" for t in trades),
        "n_short": sum(t["dir"] == "S" for t in trades),
        "win_rate": float((rs > 0).mean()),
        "expectancy_R": float(rs.mean()),
        "total_R": float(rs.sum()),
        "profit_factor": float(gross_w / gross_l) if gross_l > 0 else float("inf"),
        "avg_rr": float(np.mean([t["rr"] for t in trades])),
        "best_R": float(rs.max()),
        "worst_R": float(rs.min()),
    }
