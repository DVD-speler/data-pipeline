"""
Fase 1 — Data Verzameling
Haalt historische OHLCV-candles op via de gratis Binance public REST API
en slaat ze op in een lokale SQLite database (data/ohlcv.db).

Geen API key nodig voor publieke endpoints.

Multi-symbol & multi-interval ondersteuning:
  download_ohlcv() zonder argumenten downloadt alle symbolen en timeframes
  die in config.SYMBOLS en config.INTERVALS zijn geconfigureerd.
"""

import sqlite3
import time
from datetime import datetime, timezone

import pandas as pd
import requests
from tqdm import tqdm

import config

KLINES_URL = f"{config.BINANCE_BASE_URL}/api/v3/klines"
LIMIT_PER_REQUEST = 1000  # Binance maximum per call


# ── Database helpers ──────────────────────────────────────────────────────────

def _get_conn() -> sqlite3.Connection:
    return sqlite3.connect(config.DATA_DIR / "ohlcv.db")


def _create_table(conn: sqlite3.Connection) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS ohlcv (
            symbol       TEXT    NOT NULL,
            interval     TEXT    NOT NULL,
            open_time    INTEGER NOT NULL,
            open         REAL    NOT NULL,
            high         REAL    NOT NULL,
            low          REAL    NOT NULL,
            close        REAL    NOT NULL,
            volume       REAL    NOT NULL,
            close_time   INTEGER NOT NULL,
            PRIMARY KEY (symbol, interval, open_time)
        )
    """)
    conn.commit()


def _last_stored_timestamp(conn: sqlite3.Connection, symbol: str, interval: str):
    """Geeft de open_time (ms) van de meest recente opgeslagen candle terug."""
    row = conn.execute(
        "SELECT MAX(open_time) FROM ohlcv WHERE symbol=? AND interval=?",
        (symbol, interval),
    ).fetchone()
    return row[0]


# ── API fetch ─────────────────────────────────────────────────────────────────

def _fetch_klines(symbol: str, interval: str, start_ms: int) -> list:
    params = {
        "symbol": symbol,
        "interval": interval,
        "startTime": start_ms,
        "limit": LIMIT_PER_REQUEST,
    }
    resp = requests.get(KLINES_URL, params=params, timeout=15)
    resp.raise_for_status()
    return resp.json()


# ── yfinance fallback (voor US IPs waar Binance geblokkeerd is) ────────────────

_YF_SYMBOL_MAP = {
    "BTCUSDT": "BTC-USD",
    "ETHUSDT": "ETH-USD",
}


def _download_single_yfinance(
    symbol: str,
    interval: str,
    days: int,
    conn: sqlite3.Connection,
) -> int:
    """
    Fallback OHLCV-download via yfinance.
    Wordt gebruikt wanneer Binance een 451 (geo-blokkering) teruggeeft,
    zoals op GitHub Actions runners die op US-servers draaien.

    yfinance ondersteunt geen 4h interval — 4h wordt opgebouwd door 1h te
    resamplen via OHLCV-aggregatie.
    """
    import yfinance as yf

    yf_ticker = _YF_SYMBOL_MAP.get(symbol, symbol.replace("USDT", "-USD"))
    yf_interval = "1h"   # download altijd 1h; 4h wordt daarna geresampeld
    period_str  = f"{min(days, 729)}d"   # yfinance max voor 1h = 730 dagen

    print(f"  [{symbol} {interval}] yfinance fallback: {yf_ticker} {period_str} @ 1h")

    df = yf.download(
        yf_ticker,
        period=period_str,
        interval=yf_interval,
        auto_adjust=True,
        progress=False,
    )

    if df.empty:
        print(f"  Geen data van yfinance voor {yf_ticker}")
        return 0

    # Normaliseer kolommen (yfinance geeft soms MultiIndex of capitalized namen)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)
    df.columns = [c.lower() for c in df.columns]

    # Zorg voor UTC-aware index
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")

    # Resampel 1h → 4h indien gevraagd
    if interval == "4h":
        df = df.resample("4h", closed="left", label="left").agg(
            open=("open", "first"),
            high=("high", "max"),
            low=("low", "min"),
            close=("close", "last"),
            volume=("volume", "sum"),
        ).dropna(subset=["close"])

    interval_ms = 4 * 3600 * 1000 if interval == "4h" else 3600 * 1000

    rows = [
        (
            symbol,
            interval,
            int(ts.timestamp() * 1000),
            float(row["open"]),
            float(row["high"]),
            float(row["low"]),
            float(row["close"]),
            float(row["volume"]),
            int(ts.timestamp() * 1000) + interval_ms - 1,
        )
        for ts, row in df.iterrows()
    ]

    conn.executemany(
        "INSERT OR IGNORE INTO ohlcv VALUES (?,?,?,?,?,?,?,?,?)",
        rows,
    )
    conn.commit()
    print(f"  Klaar. {len(rows)} candles opgeslagen via yfinance.")
    return len(rows)


# ── Enkelvoudige download (één symbool / één interval) ────────────────────────

def _download_single(
    symbol: str,
    interval: str,
    days: int,
    conn: sqlite3.Connection,
) -> int:
    """
    Download historische candles voor één (symbool, interval)-combinatie.
    Hervat automatisch vanaf het laatste opgeslagen tijdstip.
    Geeft het aantal nieuw ingevoegde candles terug.
    """
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    default_start_ms = now_ms - days * 24 * 3600 * 1000

    last_ts = _last_stored_timestamp(conn, symbol, interval)
    if last_ts:
        start_ms = last_ts + 1
        print(
            f"  [{symbol} {interval}] Hervatten vanaf "
            f"{datetime.fromtimestamp(last_ts / 1000, tz=timezone.utc).strftime('%Y-%m-%d %H:%M')} UTC"
        )
    else:
        start_ms = default_start_ms
        print(
            f"  [{symbol} {interval}] Verse download vanaf "
            f"{datetime.fromtimestamp(start_ms / 1000, tz=timezone.utc).strftime('%Y-%m-%d %H:%M')} UTC"
        )

    hours_remaining = (now_ms - start_ms) // (3600 * 1000)
    estimated_batches = max(1, hours_remaining // LIMIT_PER_REQUEST + 1)

    total_inserted = 0
    current_start = start_ms

    with tqdm(
        total=estimated_batches,
        desc=f"  Downloading {symbol} {interval}",
        leave=False,
    ) as pbar:
        while current_start < now_ms:
            try:
                klines = _fetch_klines(symbol, interval, current_start)
            except requests.exceptions.HTTPError as exc:
                # 451 = geo-blokkering (bijv. GitHub Actions op US-servers)
                if exc.response is not None and exc.response.status_code == 451:
                    print(f"\n  Binance 451 (geo-blokkering) — overschakelen naar yfinance...")
                    return _download_single_yfinance(symbol, interval, days, conn)
                print(f"\n  API-fout: {exc}  — opnieuw proberen over 10s...")
                time.sleep(10)
                continue
            except requests.RequestException as exc:
                print(f"\n  API-fout: {exc}  — opnieuw proberen over 10s...")
                time.sleep(10)
                continue

            if not klines:
                break

            rows = [
                (
                    symbol,
                    interval,
                    int(k[0]),    # open_time ms
                    float(k[1]),  # open
                    float(k[2]),  # high
                    float(k[3]),  # low
                    float(k[4]),  # close
                    float(k[5]),  # volume
                    int(k[6]),    # close_time ms
                )
                for k in klines
            ]

            conn.executemany(
                "INSERT OR IGNORE INTO ohlcv VALUES (?,?,?,?,?,?,?,?,?)",
                rows,
            )
            conn.commit()
            total_inserted += len(rows)

            current_start = klines[-1][6] + 1
            pbar.update(1)
            time.sleep(0.12)  # ~8 req/s, ruim onder de Binance limiet van 1200/min

    return total_inserted


# ── Public interface ──────────────────────────────────────────────────────────

def download_ohlcv(
    symbols: list = None,
    intervals: list = None,
    days: int = config.DAYS_HISTORY,
) -> None:
    """
    Download historische OHLCV-candles van Binance voor alle geconfigureerde
    symbolen en timeframes. Hervat automatisch vanaf het laatste opgeslagen
    tijdstip (incrementeel).

    Parameters
    ----------
    symbols   : lijst van symbolen (standaard: config.SYMBOLS)
    intervals : lijst van timeframes (standaard: config.INTERVALS)
    days      : terugkijkvenster in dagen (standaard: config.DAYS_HISTORY)
    """
    if symbols is None:
        symbols = config.SYMBOLS
    if intervals is None:
        intervals = config.INTERVALS

    conn = _get_conn()
    _create_table(conn)

    total_pairs = len(symbols) * len(intervals)
    pair_num = 0

    for symbol in symbols:
        for interval in intervals:
            pair_num += 1
            print(f"\n[{pair_num}/{total_pairs}] {symbol} @ {interval}")
            n = _download_single(symbol, interval, days, conn)
            print(f"  Klaar. {n} nieuwe candles opgeslagen.")

    conn.close()


def load_ohlcv(
    symbol: str = config.SYMBOL,
    interval: str = config.INTERVAL,
) -> pd.DataFrame:
    """
    Laad alle opgeslagen candles uit SQLite als een pandas DataFrame.
    Index = UTC datetime, kolommen = open/high/low/close/volume.

    Parameters
    ----------
    symbol   : symbool om te laden (standaard: config.SYMBOL = "BTCUSDT")
    interval : timeframe om te laden (standaard: config.INTERVAL = "1h")
    """
    conn = _get_conn()
    df = pd.read_sql(
        "SELECT * FROM ohlcv WHERE symbol=? AND interval=? ORDER BY open_time",
        conn,
        params=(symbol, interval),
    )
    conn.close()

    df["datetime"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df = df.set_index("datetime").drop(columns=["symbol", "interval"])
    return df


if __name__ == "__main__":
    download_ohlcv()
    for sym in config.SYMBOLS:
        df = load_ohlcv(sym, "1h")
        print(f"\n{sym} 1h: {len(df)} candles  ({df.index[0].date()} → {df.index[-1].date()})")
    print(load_ohlcv().tail())
