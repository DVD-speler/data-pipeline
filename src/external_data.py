"""
Externe data bronnen voor aanvullende features:

  1. Fear & Greed Index  — alternative.me (dagelijks, gratis)
  2. S&P500 (SPY ETF)   — yfinance (uurlijks, gratis)
  3. EUR/USD wisselkoers — yfinance (uurlijks, gratis) — proxy dollar sterkte
  4. BTC Funding Rate    — Binance Futures public API (8-uurlijks, gratis, geen key)
  5. BTC Open Interest   — Binance Futures public API (uurlijks, gratis, geen key)

ETH/BTC ratio wordt berekend in features.py vanuit de bestaande OHLCV database.

Alle data wordt gecached in data/external/ als Parquet bestanden.
De cache wordt automatisch ververst als hij ouder is dan MAX_CACHE_AGE_H uur.

Gebruik:
  from src.external_data import load_all_external
  df_ext = load_all_external(df_ohlcv.index)   # geeft DataFrame terug op 1h BTC-index
"""

import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests

import config

EXTERNAL_DIR    = config.DATA_DIR / "external"
EXTERNAL_DIR.mkdir(exist_ok=True)

FUTURES_BASE    = "https://fapi.binance.com"
FNG_URL         = "https://api.alternative.me/fng/"

# Per-bron cache-leeftijden (in uren)
# Fear & Greed is dagelijks → 24h; SPX/EURUSD zijn uurlijks → 1h;
# funding rate is 8-uurlijks → 8h; DVOL is uurlijks → 1h.
_CACHE_MAX_AGE_H = {
    "fear_greed":    24,
    "spx":            1,
    "eurusd":         1,
    "btc_dvol":       1,
    "spx_daily":     24,
    "eurusd_daily":  24,
}
_CACHE_AGE_DEFAULT = 6   # fallback voor funding_rate, open_interest, etc.


# ── Cache helpers ──────────────────────────────────────────────────────────────

def _cache_path(name: str) -> Path:
    return EXTERNAL_DIR / f"{name}.parquet"


def _cache_is_fresh(name: str) -> bool:
    path = _cache_path(name)
    if not path.exists():
        return False
    age_h = (time.time() - path.stat().st_mtime) / 3600
    max_age = _CACHE_MAX_AGE_H.get(name, _CACHE_AGE_DEFAULT)
    return age_h < max_age


def _save_cache(name: str, df: pd.DataFrame) -> None:
    df.to_parquet(_cache_path(name))


def _load_cache(name: str) -> pd.DataFrame:
    return pd.read_parquet(_cache_path(name))


# ── 1. Fear & Greed Index ─────────────────────────────────────────────────────

def _download_fear_greed() -> pd.DataFrame:
    resp = requests.get(
        FNG_URL,
        params={"limit": 2000, "format": "json"},   # 2000 = max (~5.5 jaar history)
        timeout=15,
    )
    resp.raise_for_status()
    data = resp.json()["data"]
    df = pd.DataFrame(data)[["timestamp", "value"]]
    df.index = pd.to_datetime(df["timestamp"].astype(int), unit="s", utc=True)
    df["fear_greed"] = df["value"].astype(float) / 100.0   # normaliseer naar 0–1
    df = df[["fear_greed"]].sort_index()
    df = df[~df.index.duplicated()]
    return df


def fetch_fear_greed() -> pd.DataFrame:
    """Fear & Greed Index (dagelijks). 0=extreme fear, 1=extreme greed."""
    name = "fear_greed"
    if not _cache_is_fresh(name):
        print("  Downloading Fear & Greed Index...")
        try:
            df = _download_fear_greed()
            _save_cache(name, df)
            return df
        except Exception as e:
            print(f"  Waarschuwing: F&G download mislukt ({e})")
    if _cache_path(name).exists():
        return _load_cache(name)
    return pd.DataFrame(columns=["fear_greed"])


# ── 2. S&P500 (SPY) ───────────────────────────────────────────────────────────

def _download_yfinance(ticker: str) -> pd.DataFrame:
    try:
        import yfinance as yf
    except ImportError:
        raise ImportError("yfinance niet gevonden. Installeer via: pip install yfinance")

    df = yf.download(ticker, interval="1h", period="730d", progress=False, auto_adjust=True)
    if df.empty:
        return pd.DataFrame()

    # Nieuwere yfinance-versies geven MultiIndex terug → platmaken
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Normalize to UTC regardless of whether yfinance returns tz-naive or tz-aware index
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")

    # SPX/EURUSD hourly bars beginnen op :30 (NYSE opent 09:30 ET = 13:30 UTC).
    # Floor naar uur zodat de timestamps overeenkomen met BTC's :00 candles.
    df.index = df.index.floor("h")
    df = df[~df.index.duplicated(keep="last")]
    return df[["Close"]]


def fetch_spx() -> pd.DataFrame:
    """S&P500 (SPY) uurlijkse 24h-rendement."""
    name = "spx"
    if not _cache_is_fresh(name):
        print("  Downloading S&P500 (SPY)...")
        try:
            df = _download_yfinance("SPY")
            df.columns = ["spx_close"]
            df["spx_return_24h"] = df["spx_close"].pct_change(24)
            df = df[["spx_return_24h"]]
            _save_cache(name, df)
            return df
        except Exception as e:
            print(f"  Waarschuwing: SPX download mislukt ({e})")
    if _cache_path(name).exists():
        return _load_cache(name)
    return pd.DataFrame(columns=["spx_return_24h"])


# ── 3. EUR/USD ────────────────────────────────────────────────────────────────

def fetch_eurusd() -> pd.DataFrame:
    """
    EUR/USD wisselkoers als proxy voor dollar sterkte.
    24h-rendement: negatief = dollar sterker = doorgaans bearish voor crypto.
    """
    name = "eurusd"
    if not _cache_is_fresh(name):
        print("  Downloading EUR/USD...")
        try:
            df = _download_yfinance("EURUSD=X")
            df.columns = ["eurusd_close"]
            df["eurusd_return_24h"] = df["eurusd_close"].pct_change(24)
            df = df[["eurusd_return_24h"]]
            _save_cache(name, df)
            return df
        except Exception as e:
            print(f"  Waarschuwing: EURUSD download mislukt ({e})")
    if _cache_path(name).exists():
        return _load_cache(name)
    return pd.DataFrame(columns=["eurusd_return_24h"])


def _download_yfinance_daily(ticker: str, years: int = 10) -> pd.DataFrame:
    """Download dagelijkse OHLCV via yfinance (geen uurlimiet — gaat jaren terug)."""
    try:
        import yfinance as yf
    except ImportError:
        raise ImportError("yfinance niet gevonden. Installeer via: pip install yfinance")

    df = yf.download(ticker, interval="1d", period=f"{years * 365}d",
                     progress=False, auto_adjust=True)
    if df.empty:
        return pd.DataFrame()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")

    # Normaliseer naar middernacht UTC zodat join met dagcandles matcht
    df.index = df.index.normalize()
    df = df[~df.index.duplicated(keep="last")]
    return df[["Close"]]


def fetch_spx_daily() -> pd.DataFrame:
    """S&P500 dagelijks rendement — 5-daags en 1-daags (voor dagmodel)."""
    name = "spx_daily"
    if not _cache_is_fresh(name):
        print("  Downloading S&P500 dagelijks (SPY)...")
        try:
            df = _download_yfinance_daily("SPY", years=10)
            df.columns = ["spx_close"]
            df["spx_return_1d"] = df["spx_close"].pct_change(1)
            df["spx_return_1w"] = df["spx_close"].pct_change(5)
            df = df[["spx_return_1d", "spx_return_1w"]]
            _save_cache(name, df)
            return df
        except Exception as e:
            print(f"  Waarschuwing: SPX dagelijks download mislukt ({e})")
    if _cache_path(name).exists():
        return _load_cache(name)
    return pd.DataFrame(columns=["spx_return_1d", "spx_return_1w"])


def fetch_vix() -> pd.DataFrame:
    """
    CBOE Volatility Index (VIX) — aandelenmarkt-angstmeter.

    VIX > 20 = verhoogde angst
    VIX > 25 = block longs (config.VIX_GATE)
    VIX > 30 = ernstige marktangst (crashes historisch)
    VIX > 50 = extreme paniek (COVID maart 2020, aug 2024 Japan-crash = 65+)

    Data: yfinance "^VIX" dagelijks, forward-filled naar uurlijks.
    Geeft de ruwe VIX-waarde terug (niet een rendement).
    """
    name = "vix"
    if not _cache_is_fresh(name):
        print("  Downloading VIX (CBOE Volatility Index)...")
        try:
            df = _download_yfinance_daily("^VIX", years=10)
            df.columns = ["vix_level"]
            _save_cache(name, df)
            return df
        except Exception as e:
            print(f"  Waarschuwing: VIX download mislukt ({e})")
    if _cache_path(name).exists():
        return _load_cache(name)
    return pd.DataFrame(columns=["vix_level"])


def fetch_usdjpy() -> pd.DataFrame:
    """
    USD/JPY wisselkoers — yen-carry-trade indicator.

    Sterke yen-appreciatie (USD/JPY daalt) = carry trade unwind =
    gedwongen liquidatie van risk assets (waaronder crypto).

    Biedt:
      usdjpy_return_24h : dagelijks rendement (proxy voor momentum)
      usdjpy_return_7d  : 7-daags rendement — gate trigger als < -3%
                          (yen > 3% gestegen in een week = gevaar)

    Data: yfinance "JPY=X" dagelijks, forward-filled naar uurlijks.
    """
    name = "usdjpy"
    if not _cache_is_fresh(name):
        print("  Downloading USD/JPY...")
        try:
            df = _download_yfinance_daily("JPY=X", years=10)
            df.columns = ["usdjpy_close"]
            df["usdjpy_return_24h"] = df["usdjpy_close"].pct_change(1)
            df["usdjpy_return_7d"]  = df["usdjpy_close"].pct_change(7)
            df = df[["usdjpy_return_24h", "usdjpy_return_7d"]]
            _save_cache(name, df)
            return df
        except Exception as e:
            print(f"  Waarschuwing: USD/JPY download mislukt ({e})")
    if _cache_path(name).exists():
        return _load_cache(name)
    return pd.DataFrame(columns=["usdjpy_return_24h", "usdjpy_return_7d"])


def fetch_eurusd_daily() -> pd.DataFrame:
    """EUR/USD dagelijks rendement — 5-daags en 1-daags (voor dagmodel)."""
    name = "eurusd_daily"
    if not _cache_is_fresh(name):
        print("  Downloading EUR/USD dagelijks...")
        try:
            df = _download_yfinance_daily("EURUSD=X", years=10)
            df.columns = ["eurusd_close"]
            df["eurusd_return_1d"] = df["eurusd_close"].pct_change(1)
            df["eurusd_return_1w"] = df["eurusd_close"].pct_change(5)
            df = df[["eurusd_return_1d", "eurusd_return_1w"]]
            _save_cache(name, df)
            return df
        except Exception as e:
            print(f"  Waarschuwing: EURUSD dagelijks download mislukt ({e})")
    if _cache_path(name).exists():
        return _load_cache(name)
    return pd.DataFrame(columns=["eurusd_return_1d", "eurusd_return_1w"])


# ── 4. BTC Funding Rate (Binance Futures) ────────────────────────────────────

def _download_funding_rate(symbol: str = "BTCUSDT") -> pd.DataFrame:
    url     = f"{FUTURES_BASE}/fapi/v1/fundingRate"
    now_ms  = int(datetime.now(timezone.utc).timestamp() * 1000)
    start_ms = now_ms - config.DAYS_HISTORY * 24 * 3600 * 1000

    all_rows  = []
    current   = start_ms

    while current < now_ms:
        resp = requests.get(
            url,
            params={"symbol": symbol, "startTime": current, "limit": 1000},
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
        if not data:
            break
        all_rows.extend(data)
        current = data[-1]["fundingTime"] + 1
        if len(data) < 1000:
            break
        time.sleep(0.1)

    if not all_rows:
        return pd.DataFrame(columns=["funding_rate"])

    df = pd.DataFrame(all_rows)
    df.index = pd.to_datetime(df["fundingTime"], unit="ms", utc=True)
    df["funding_rate"] = df["fundingRate"].astype(float)
    df = df[["funding_rate"]].sort_index()
    df = df[~df.index.duplicated()]
    return df


def fetch_funding_rate(symbol: str = "BTCUSDT") -> pd.DataFrame:
    """Perpetual futures funding rate (8-uurlijks → forward-fill naar 1h)."""
    name = f"{symbol}_funding_rate"
    if not _cache_is_fresh(name):
        print(f"  Downloading {symbol} Funding Rate...")
        try:
            df = _download_funding_rate(symbol=symbol)
            _save_cache(name, df)
            return df
        except Exception as e:
            print(f"  Waarschuwing: Funding Rate download mislukt ({e})")
    if _cache_path(name).exists():
        return _load_cache(name)
    return pd.DataFrame(columns=["funding_rate"])


# ── 5. BTC Open Interest (Binance Futures) ────────────────────────────────────

def _download_open_interest(symbol: str = "BTCUSDT") -> pd.DataFrame:
    # Binance openInterestHist endpoint enforces a maximum lookback of ~30 days
    # for the 1h period. Requesting further back returns a 400 error.
    url        = f"{FUTURES_BASE}/futures/data/openInterestHist"
    now_ms     = int(datetime.now(timezone.utc).timestamp() * 1000)
    max_oi_days = 30
    start_ms   = now_ms - max_oi_days * 24 * 3600 * 1000
    batch_ms   = 500 * 3600 * 1000   # 500 uur per batch (max limit=500)

    all_rows = []
    current  = start_ms

    while current < now_ms:
        end_ms = min(current + batch_ms, now_ms)
        resp = requests.get(
            url,
            params={
                "symbol":    symbol,
                "period":    "1h",
                "limit":     500,
                "startTime": current,
                "endTime":   end_ms,
            },
            timeout=15,
        )
        if resp.status_code != 200:
            print(f"  OI endpoint returned {resp.status_code}: {resp.text[:200]}")
            resp.raise_for_status()
        data = resp.json()
        if data:
            all_rows.extend(data)
        current = end_ms + 1
        time.sleep(0.12)

    if not all_rows:
        return pd.DataFrame(columns=["oi_change_24h"])

    df = pd.DataFrame(all_rows)
    df.index = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df["oi"] = pd.to_numeric(df["sumOpenInterest"], errors="coerce")
    df = df[["oi"]].sort_index()
    df = df[~df.index.duplicated()]
    df["oi_change_24h"] = df["oi"].pct_change(24)
    return df[["oi_change_24h"]]


def fetch_open_interest(symbol: str = "BTCUSDT") -> pd.DataFrame:
    """Open interest 24h-verandering (uurlijks, Binance futures public API)."""
    name = f"{symbol}_open_interest"
    if not _cache_is_fresh(name):
        print(f"  Downloading {symbol} Open Interest...")
        try:
            df = _download_open_interest(symbol=symbol)
            _save_cache(name, df)
            return df
        except Exception as e:
            print(f"  Waarschuwing: Open Interest download mislukt ({e})")
    if _cache_path(name).exists():
        return _load_cache(name)
    return pd.DataFrame(columns=["oi_change_24h"])


# ── 6. Deribit BTC Implied Volatility (DVOL) ─────────────────────────────────

DERIBIT_BASE = "https://www.deribit.com/api/v2/public"


def _download_deribit_dvol(days: int = 1000) -> pd.DataFrame:
    """
    Download Deribit BTC DVOL index (implied volatility) via gratis public API.
    DVOL waarden liggen typisch tussen 30 (lage vol) en 150 (hoge vol/angst).
    Genormaliseerd naar 0–1 door te delen door 100.
    """
    now_ms   = int(datetime.now(timezone.utc).timestamp() * 1000)
    start_ms = now_ms - days * 24 * 3600 * 1000
    resolution = 3600   # 1 uur in seconden

    # Deribit returns max 1000 candles per call — paginate with sliding window
    chunk_ms = 1000 * resolution * 1000   # 1000 uur in ms
    all_data = []
    current = start_ms

    while current < now_ms:
        chunk_end = min(current + chunk_ms, now_ms)
        resp = requests.get(
            f"{DERIBIT_BASE}/get_volatility_index_data",
            params={
                "currency":        "BTC",
                "resolution":      resolution,
                "start_timestamp": current,
                "end_timestamp":   chunk_end,
            },
            timeout=15,
        )
        resp.raise_for_status()
        result = resp.json().get("result", {})
        data   = result.get("data", [])
        if not data:
            current = chunk_end + resolution * 1000
            continue
        all_data.extend(data)
        current = data[-1][0] + resolution * 1000
        time.sleep(0.1)

    if not all_data:
        return pd.DataFrame(columns=["btc_dvol"])

    # Elk item: [timestamp_ms, open, high, low, close]
    df = pd.DataFrame(all_data, columns=["ts", "open", "high", "low", "close"])
    df.index = pd.to_datetime(df["ts"], unit="ms", utc=True)
    df = df[~df.index.duplicated(keep="last")].sort_index()
    # Normaliseer: DVOL 0-100 → 0-1 (typisch: 0.30–0.80 = normale range)
    df["btc_dvol"] = df["close"] / 100.0
    return df[["btc_dvol"]]


def fetch_deribit_dvol() -> pd.DataFrame:
    """Deribit BTC Implied Volatility Index (DVOL), uurlijks, genormaliseerd 0–1."""
    name = "btc_dvol"
    if not _cache_is_fresh(name):
        print("  Downloading Deribit BTC DVOL...")
        try:
            df = _download_deribit_dvol()
            _save_cache(name, df)
            return df
        except Exception as e:
            print(f"  Waarschuwing: Deribit DVOL download mislukt ({e})")
    if _cache_path(name).exists():
        return _load_cache(name)
    return pd.DataFrame(columns=["btc_dvol"])


# ── Gecombineerde loader ───────────────────────────────────────────────────────

# Neutrale defaults per feature (worden gebruikt als data ontbreekt)
def fetch_btc_dominance() -> pd.DataFrame:
    """
    Haal BTC Dominance op via CoinGecko public API (geen key vereist).
    Retourneert dagelijks DataFrame met kolommen: btc_dominance, btc_dominance_7d_chg.
    Cache: 24h (dagelijks signaal).
    """
    cache_path = EXTERNAL_DIR / "btc_dominance.parquet"
    if cache_path.exists() and (time.time() - cache_path.stat().st_mtime) / 3600 < 24:
        return pd.read_parquet(cache_path)

    print("  Downloading BTC Dominance (CoinGecko)...")
    try:
        # market_chart voor 365 dagen dominantie (via /global/market_cap_chart)
        url = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart"
        params = {"vs_currency": "usd", "days": "365", "interval": "daily"}
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()

        # Haal ook totale market cap op voor dominantie-berekening
        global_url = "https://api.coingecko.com/api/v3/global/market_cap_chart"
        g_resp = requests.get(global_url, params={"days": "365"}, timeout=15)
        g_resp.raise_for_status()

        btc_caps = resp.json().get("market_caps", [])
        total_caps = g_resp.json().get("market_cap_chart", {}).get("market_cap", [])

        if not btc_caps or not total_caps:
            raise ValueError("Lege response van CoinGecko")

        df_btc   = pd.DataFrame(btc_caps,   columns=["ts", "btc_cap"])
        df_total = pd.DataFrame(total_caps, columns=["ts", "total_cap"])
        df = df_btc.merge(df_total, on="ts", how="inner")
        df["datetime"]      = pd.to_datetime(df["ts"], unit="ms", utc=True)
        df                  = df.set_index("datetime").drop(columns=["ts", "btc_cap", "total_cap"])
        df["btc_dominance"] = (df_btc.set_index("ts")["btc_cap"] /
                                df_total.set_index("ts")["total_cap"] * 100
                               ).values
        df["btc_dominance_7d_chg"] = df["btc_dominance"].diff(7)
        df = df.dropna()
        df.to_parquet(cache_path)
        print(f"  BTC Dominance: {len(df)} datapunten geladen")
        return df

    except Exception as e:
        print(f"  Waarschuwing: BTC Dominance download mislukt ({e})")
        if cache_path.exists():
            return pd.read_parquet(cache_path)
        return pd.DataFrame(columns=["btc_dominance", "btc_dominance_7d_chg"])


def fetch_deribit_put_call_ratio() -> pd.DataFrame:
    """
    Haal BTC Put/Call ratio op via Deribit publieke API (geen key vereist).

    Methode: tel open interest van alle BTC puts en calls op uit
    /api/v2/public/get_book_summary_by_currency (valuta=BTC, kind=option).
    P/C ratio = sum(put_open_interest) / sum(call_open_interest).

    Hoge P/C ratio (> 1.5) → bearish positionering, blokkeert longs.
    Cache: 6h (opties-boek ververst traag).
    """
    cache_path = EXTERNAL_DIR / "btc_put_call_ratio.parquet"
    if cache_path.exists() and (time.time() - cache_path.stat().st_mtime) / 3600 < 6:
        return pd.read_parquet(cache_path)

    print("  Downloading Deribit BTC Put/Call ratio...")
    try:
        url = "https://www.deribit.com/api/v2/public/get_book_summary_by_currency"
        params = {"currency": "BTC", "kind": "option"}
        resp = requests.get(url, params=params, timeout=20)
        resp.raise_for_status()

        data = resp.json().get("result", [])
        if not data:
            raise ValueError("Lege response van Deribit options endpoint")

        put_oi  = sum(d.get("open_interest", 0) for d in data if d.get("instrument_name", "").endswith("-P"))
        call_oi = sum(d.get("open_interest", 0) for d in data if d.get("instrument_name", "").endswith("-C"))

        ratio = put_oi / call_oi if call_oi > 0 else 1.0

        now = pd.Timestamp.utcnow().floor("h")
        # Creëer hourly serie van laatste 30 dagen met huidige waarde (forward-fill)
        idx = pd.date_range(end=now, periods=30 * 24, freq="h", tz="UTC")
        df  = pd.DataFrame({"btc_put_call_ratio": ratio}, index=idx)
        df.to_parquet(cache_path)
        print(f"  Deribit P/C ratio: {ratio:.3f}")
        return df

    except Exception as e:
        print(f"  Waarschuwing: Deribit P/C ratio download mislukt ({e})")
        if cache_path.exists():
            return pd.read_parquet(cache_path)
        now = pd.Timestamp.utcnow().floor("h")
        idx = pd.date_range(end=now, periods=30 * 24, freq="h", tz="UTC")
        return pd.DataFrame({"btc_put_call_ratio": 1.0}, index=idx)


_DEFAULTS = {
    "fear_greed":        0.5,
    "spx_return_24h":    0.0,
    "eurusd_return_24h": 0.0,
    "funding_rate":      0.0,
    "oi_change_24h":     0.0,
    "btc_dvol":          0.45,   # ~45% IV = neutrale volatiliteit
    "vix_level":         20.0,   # historisch gemiddelde VIX ~20
    "usdjpy_return_24h":    0.0,
    "usdjpy_return_7d":     0.0,
    "btc_dominance":        50.0,   # historisch gemiddeld ~50%
    "btc_dominance_7d_chg": 0.0,
    "btc_put_call_ratio":   1.0,    # neutraal: evenveel puts als calls
}

_SOURCES_GLOBAL = {
    "fear_greed":           fetch_fear_greed,
    "spx":                  fetch_spx,
    "eurusd":               fetch_eurusd,
    "btc_dvol":             fetch_deribit_dvol,
    "vix":                  fetch_vix,
    "usdjpy":               fetch_usdjpy,
    "btc_dominance":        fetch_btc_dominance,
    "btc_put_call_ratio":   fetch_deribit_put_call_ratio,
}


def load_all_external(index: pd.DatetimeIndex, symbol: str = "BTCUSDT") -> pd.DataFrame:
    """
    Laad alle externe features en aligneer ze op de gegeven uurlijkse BTC-index.

    - Dagelijkse data (Fear & Greed) → forward-fill naar uurlijks
    - Marktuur-data (SPX, EURUSD)   → forward-fill voor gesloten uren/weekend
    - 8-uurlijkse data (funding)    → forward-fill
    - Ontbrekende waarden           → neutrale defaults

    Parameters
    ----------
    index : uurlijkse UTC DatetimeIndex van de BTC OHLCV data

    Returns
    -------
    pd.DataFrame met externe features, geïndexeerd op 'index'
    """
    sources = dict(_SOURCES_GLOBAL)
    sources["funding_rate"]  = lambda: fetch_funding_rate(symbol=symbol)
    sources["open_interest"] = lambda: fetch_open_interest(symbol=symbol)

    base = pd.DataFrame(index=index)

    for name, fetch_fn in sources.items():
        try:
            df_ext = fetch_fn()
            if df_ext.empty:
                for col, val in _DEFAULTS.items():
                    if col in [c for c in df_ext.columns] or name in col:
                        base[col] = val
                continue

            # Normaliseer het index-dtype van externe data naar dat van de BTC-index.
            # pandas 2.x slaat parquet op als datetime64[ms, UTC] maar BTC-data
            # gebruikt datetime64[ns, UTC]; dtype-mismatch geeft 0 matches na join.
            df_ext = df_ext.copy()
            df_ext.index = df_ext.index.astype(base.index.dtype)

            # Left-join op de BTC-index en forward-fill
            joined = base.join(df_ext, how="left")
            for col in df_ext.columns:
                if col in joined.columns:
                    base[col] = joined[col].ffill()
                    # Controleer na ffill: resterende NaN wijst op datumreikprobleem
                    null_frac = base[col].isna().mean()
                    if null_frac > 0.20:
                        print(f"  WAARSCHUWING: {col} is {null_frac:.0%} null na ffill "
                              f"— controleer datumbereik of bron")

        except Exception as e:
            print(f"  Fout bij laden van {name}: {e}")

    # Vul resterende NaN op met neutrale defaults
    for col, val in _DEFAULTS.items():
        if col in base.columns:
            base[col] = base[col].fillna(val)
        else:
            base[col] = val

    return base


def download_all_external(force: bool = True) -> None:
    """
    Download alle externe data opnieuw (overschrijft cache).
    Downloadt globale bronnen (F&G, SPX, EURUSD) éénmaal en
    symbool-specifieke bronnen (funding rate, OI) voor alle config.SYMBOLS.

    Parameters
    ----------
    force : True = verwijder eerst de cache om verse download te garanderen
    """
    import config as _cfg
    if force:
        for f in EXTERNAL_DIR.glob("*.parquet"):
            f.unlink()
        print("Cache gewist. Start verse download...")

    for name, fetch_fn in _SOURCES_GLOBAL.items():
        print(f"\n[{name}]")
        fetch_fn()

    # Dagelijkse SPX/EURUSD voor het dagmodel (geen uurlimiet, jaren terug)
    print("\n[spx_daily]")
    fetch_spx_daily()
    print("\n[eurusd_daily]")
    fetch_eurusd_daily()

    for sym in _cfg.SYMBOLS:
        print(f"\n[{sym} funding_rate]")
        fetch_funding_rate(symbol=sym)
        print(f"\n[{sym} open_interest]")
        fetch_open_interest(symbol=sym)

    print(f"\nKlaar. Gecached in: {EXTERNAL_DIR}/")


if __name__ == "__main__":
    download_all_external()
