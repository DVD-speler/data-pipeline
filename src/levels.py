"""
Structurele SL/TP plaatsing op basis van swing highs/lows.

Detecteert pivot-punten in price action en plaatst SL net onder support,
TP net onder resistance — met minimale R/R als constraint.
"""
import numpy as np
import pandas as pd

import config


# ── Swing detectie ────────────────────────────────────────────────────────────

def find_swing_lows(lows: np.ndarray, order: int = 5) -> np.ndarray:
    """
    Detecteer swing lows: een candle is een pivot low als zijn low
    lager is dan alle lows in een venster van `order` candles aan elke kant.

    Returns array van zelfde lengte als lows, NaN waar geen pivot.
    """
    n = len(lows)
    result = np.full(n, np.nan)
    for i in range(order, n - order):
        window = lows[i - order: i + order + 1]
        if lows[i] <= window.min():
            result[i] = lows[i]
    return result


def find_swing_highs(highs: np.ndarray, order: int = 5) -> np.ndarray:
    """
    Detecteer swing highs: een candle is een pivot high als zijn high
    hoger is dan alle highs in een venster van `order` candles aan elke kant.
    """
    n = len(highs)
    result = np.full(n, np.nan)
    for i in range(order, n - order):
        window = highs[i - order: i + order + 1]
        if highs[i] >= window.max():
            result[i] = highs[i]
    return result


def precompute_swings(highs: np.ndarray, lows: np.ndarray) -> dict:
    """
    Berekent swing arrays voor meerdere orders (timeframe-proxies).

    Orders:
      5   → ~5h pivot  (intraday structuur)
      20  → ~20h pivot (dagelijkse structuur)
      120 → ~5d pivot  (wekelijkse structuur)

    Returns dict met 'lows_5', 'highs_5', 'lows_20', ... etc.
    """
    result = {}
    for order in (5, 20, 120):
        result[f"lows_{order}"]  = find_swing_lows(lows, order=order)
        result[f"highs_{order}"] = find_swing_highs(highs, order=order)
    return result


def get_recent_levels(
    swings: dict,
    idx: int,
    lookback: int = 500,
    n_levels: int = 8,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Verzamel de meest recente N support- en resistance-niveaus vóór idx.

    Combineert alle timeframe-orders (5, 20, 120) en geeft:
      supports    — swing lows gesorteerd van hoog naar laag (desc)
      resistances — swing highs gesorteerd van laag naar hoog (asc)
    """
    start = max(0, idx - lookback)

    all_supports    = []
    all_resistances = []

    for order in (5, 20, 120):
        lows_arr  = swings[f"lows_{order}"][start:idx]
        highs_arr = swings[f"highs_{order}"][start:idx]
        all_supports.extend(lows_arr[~np.isnan(lows_arr)].tolist())
        all_resistances.extend(highs_arr[~np.isnan(highs_arr)].tolist())

    # Meest recente N niveaus bewaren
    supports    = np.sort(np.unique(all_supports))[::-1][:n_levels]   # desc
    resistances = np.sort(np.unique(all_resistances))[:n_levels]       # asc

    return supports, resistances


# ── Structurele SL/TP berekening ──────────────────────────────────────────────

def compute_structural_sl_tp(
    entry_price: float,
    direction: str,
    supports: np.ndarray,
    resistances: np.ndarray,
    min_rr: float = 1.5,
    max_sl_pct: float = 0.05,
    min_sl_pct: float = 0.005,
    buffer: float = 0.003,
    fallback_sl_pct: float = config.STOP_LOSS_PCT,
    fallback_tp_pct: float = 0.06,
) -> tuple[float, float]:
    """
    Berekent structurele SL en TP op basis van swing niveaus.

    LONG:
      SL = net onder dichtstbijzijnde support onder entry
      TP = dichtstbijzijnde resistance boven entry met R/R >= min_rr

    SHORT:
      SL = net boven dichtstbijzijnde resistance boven entry
      TP = dichtstbijzijnde support onder entry met R/R >= min_rr

    Fallback op vaste percentages als geen geschikt niveau gevonden.

    Parameters
    ----------
    min_rr        : minimale reward/risk ratio (TP-afstand / SL-afstand)
    max_sl_pct    : SL nooit verder dan dit percentage (risico-cap)
    min_sl_pct    : SL nooit dichter dan dit (te strakke stops vermijden)
    buffer        : extra ruimte (0.3%) onder support / boven resistance
    fallback_sl_pct : vaste SL als geen structuur voldoet
    fallback_tp_pct : vaste TP als geen structuur voldoet

    Returns (sl_price, tp_price)
    """
    if direction == "LONG":
        # ── SL: dichtstbijzijnde swing low onder entry ────────────────────
        sl_price = None
        for level in supports:           # supports zijn desc gesorteerd (dichtbij eerst)
            if level >= entry_price:
                continue
            candidate = level * (1.0 - buffer)
            dist = (entry_price - candidate) / entry_price
            if dist < min_sl_pct:
                continue   # te strak
            if dist > max_sl_pct:
                break      # te ver — neem fallback
            sl_price = candidate
            break

        if sl_price is None:
            sl_price = entry_price * (1.0 - fallback_sl_pct)

        sl_dist = (entry_price - sl_price) / entry_price
        min_tp_dist = sl_dist * min_rr

        # ── TP: dichtstbijzijnde swing high boven entry met R/R >= min_rr ──
        tp_price = None
        for level in resistances:        # resistances zijn asc gesorteerd
            if level <= entry_price:
                continue
            candidate = level * (1.0 - buffer)   # net onder resistance
            tp_dist = (candidate - entry_price) / entry_price
            if tp_dist >= min_tp_dist:
                tp_price = candidate
                break

        if tp_price is None:
            # Geen resistance gevonden — projecteer vanuit SL-afstand
            rr = max(min_rr, fallback_tp_pct / max(fallback_sl_pct, 0.001))
            tp_price = entry_price * (1.0 + sl_dist * rr)

    else:  # SHORT
        # ── SL: dichtstbijzijnde swing high boven entry ───────────────────
        sl_price = None
        for level in resistances[::-1]:  # van dichtbij (groot) naar ver (klein)
            if level <= entry_price:
                continue
            candidate = level * (1.0 + buffer)
            dist = (candidate - entry_price) / entry_price
            if dist < min_sl_pct:
                continue
            if dist > max_sl_pct:
                break
            sl_price = candidate
            break

        if sl_price is None:
            sl_price = entry_price * (1.0 + fallback_sl_pct)

        sl_dist = (sl_price - entry_price) / entry_price
        min_tp_dist = sl_dist * min_rr

        # ── TP: dichtstbijzijnde swing low onder entry met R/R >= min_rr ──
        tp_price = None
        for level in supports:           # supports zijn desc gesorteerd (dichtbij eerst)
            candidate = level * (1.0 + buffer)   # net boven support
            tp_dist = (entry_price - candidate) / entry_price
            if tp_dist >= min_tp_dist:
                tp_price = candidate
                break

        if tp_price is None:
            rr = max(min_rr, fallback_tp_pct / max(fallback_sl_pct, 0.001))
            tp_price = entry_price * (1.0 - sl_dist * rr)

    return round(sl_price, 2), round(tp_price, 2)
