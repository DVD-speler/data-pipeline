# INTEGRATIONS.md — External Integrations

## Overview

The project integrates with five external data sources, all of which are free
public APIs requiring no authentication. There are no cloud services, no
message queues, no webhooks, and no third-party SaaS platforms. All external
data is fetched on demand and cached locally as Parquet files with a 6-hour
freshness TTL.

---

## 1. Binance Spot REST API

**Purpose:** Historical OHLCV (candlestick) price data for BTC and ETH
**Source file:** `/workspaces/crypto_signal_model/src/data_fetcher.py`
**Base URL:** `https://api.binance.com`
**Endpoint:** `GET /api/v3/klines`

### Details

- **Authentication:** None. Uses the public unauthenticated endpoint; no API
  key or secret required.
- **Symbols fetched:** `BTCUSDT`, `ETHUSDT` (configurable via `config.SYMBOLS`)
- **Timeframes:** `1h`, `4h` (configurable via `config.INTERVALS`)
- **Lookback:** 730 days (configurable via `config.DAYS_HISTORY`)
- **Batch size:** 1000 candles per request (Binance maximum per call)
- **Incremental fetch:** On each run, the last stored `open_time` is queried
  from SQLite; subsequent requests resume from that timestamp + 1ms to avoid
  re-downloading existing data.

### Rate Limiting

The Binance public API allows 1200 requests per minute (weight-based). The
code self-throttles with `time.sleep(0.12)` between each batch request,
yielding approximately 8 requests/second — well within the limit. On API
errors (`requests.RequestException`), the code retries after a 10-second
backoff.

### Request Parameters

```python
params = {
    "symbol":    "BTCUSDT",   # or "ETHUSDT"
    "interval":  "1h",        # or "4h"
    "startTime": start_ms,    # Unix timestamp in milliseconds
    "limit":     1000,        # max candles per response
}
```

### Response Fields Used

Each kline array element: `[open_time, open, high, low, close, volume, close_time, ...]`
Fields stored: `open_time`, `open`, `high`, `low`, `close`, `volume`, `close_time`

### Persistence

Candles are stored in SQLite (`data/ohlcv.db`, table `ohlcv`) with a composite
primary key `(symbol, interval, open_time)`. The `INSERT OR IGNORE` strategy
prevents duplicates on repeated runs.

---

## 2. Binance Futures REST API — Funding Rate

**Purpose:** BTC perpetual futures funding rate (sentiment indicator)
**Source file:** `/workspaces/crypto_signal_model/src/external_data.py`
**Base URL:** `https://fapi.binance.com`
**Endpoint:** `GET /fapi/v1/fundingRate`

### Details

- **Authentication:** None. Public futures endpoint.
- **Symbol:** `BTCUSDT` (perpetual futures)
- **Frequency:** Funding rates are published every 8 hours by Binance.
- **Lookback:** `config.DAYS_HISTORY` days (730 days)
- **Batch size:** 1000 records per request; pagination via `fundingTime` cursor

### Rate Limiting

`time.sleep(0.1)` between pagination requests (~10 req/s).

### Request Parameters

```python
params = {
    "symbol":    "BTCUSDT",
    "startTime": current,    # milliseconds
    "limit":     1000,
}
```

### Feature Derivation

- `funding_rate`: raw 8-hourly rate, forward-filled to 1h BTC index
- `funding_momentum`: 72-hour difference of `funding_rate` (3-day shift)
  computed in `features.py`

### Caching

Written to `data/external/funding_rate.parquet`. Refreshed if cache age
exceeds 6 hours (`MAX_CACHE_AGE_H = 6`).

---

## 3. Binance Futures REST API — Open Interest

**Purpose:** BTC perpetual open interest (market positioning indicator)
**Source file:** `/workspaces/crypto_signal_model/src/external_data.py`
**Base URL:** `https://fapi.binance.com`
**Endpoint:** `GET /futures/data/openInterestHist`

### Details

- **Authentication:** None. Public futures data endpoint.
- **Symbol:** `BTCUSDT`
- **Period:** `1h` resolution
- **Hard API limit:** Maximum lookback of ~30 days enforced by Binance (the
  code caps at `max_oi_days = 30`); requests beyond 30 days return HTTP 400.
- **Batch size:** 500 records per request

### Important Limitation

Because of the 30-day maximum lookback, the `oi_change_24h` feature covers
only ~4% of the full training window (730 days). Per code comments, this
renders it effectively a constant-zero column over training and was removed
from `FEATURE_COLS` (commented out in `config.py`) to avoid adding noise.

### Request Parameters

```python
params = {
    "symbol":    "BTCUSDT",
    "period":    "1h",
    "limit":     500,
    "startTime": current,
    "endTime":   end_ms,
}
```

### Caching

Written to `data/external/open_interest.parquet`. Same 6-hour TTL as other
external sources.

---

## 4. Fear & Greed Index — alternative.me

**Purpose:** Daily crypto market sentiment score (0 = extreme fear, 100 = extreme greed)
**Source file:** `/workspaces/crypto_signal_model/src/external_data.py`
**API URL:** `https://api.alternative.me/fng/`

### Details

- **Authentication:** None. Fully public API.
- **Frequency:** Daily (one value per day).
- **Lookback:** Last 1000 days (`limit=1000`).
- **Response format:** JSON with a `data` array; each element contains
  `timestamp` (Unix seconds) and `value` (integer 0–100).

### Request

```python
params = {
    "limit":  1000,
    "format": "json",
}
```

### Feature Derivation

The raw value is normalized to `[0, 1]` by dividing by 100:
```python
df["fear_greed"] = df["value"].astype(float) / 100.0
```

Daily values are forward-filled across the hourly BTC index, so every 1h
candle carries the most recent available daily sentiment reading.

### Caching

Written to `data/external/fear_greed.parquet`. Refreshed every 6 hours.
Neutral default (0.5) is used if the download fails and no cache exists.

---

## 5. Yahoo Finance — yfinance

**Purpose:** S&P 500 (SPY ETF) and EUR/USD exchange rate — macro context features
**Source file:** `/workspaces/crypto_signal_model/src/external_data.py`
**Library:** `yfinance` (pip package wrapping Yahoo Finance undocumented API)

### Details

- **Authentication:** None. Yahoo Finance is accessed via yfinance's internal
  HTTP client; no API key needed.
- **Tickers fetched:**
  - `SPY` — SPDR S&P 500 ETF (proxy for US equity risk appetite)
  - `EURUSD=X` — EUR/USD forex rate (proxy for US dollar strength)
- **Resolution:** Hourly (`interval="1h"`)
- **Lookback:** `period="730d"` (2 years, matching Binance lookback)
- **Adjustment:** `auto_adjust=True` (split/dividend adjusted close prices)

### Data Handling

- yfinance may return a MultiIndex columns DataFrame in newer versions; the
  code flattens this with `df.columns.get_level_values(0)`.
- Timezone normalization: the index is coerced to UTC regardless of whether
  yfinance returns a tz-aware or tz-naive DatetimeIndex.
- Timestamp alignment: SPX and EUR/USD hourly bars start at :30 (NYSE opens
  09:30 ET = 13:30 UTC); timestamps are floored to the hour to align with
  BTC's :00 candles.

### Feature Derivation

```python
df["spx_return_24h"]   = df["spx_close"].pct_change(24)
df["eurusd_return_24h"] = df["eurusd_close"].pct_change(24)
```

Both features are forward-filled across weekends and non-trading hours so
every BTC 1h candle has a populated value.

### Caching

- `data/external/spx.parquet`
- `data/external/eurusd.parquet`

Both cached with 6-hour TTL. Neutral default (0.0) is applied on failure.

---

## Cache Architecture

All five external data sources share a unified caching pattern implemented in
`external_data.py`:

```
data/external/
├── fear_greed.parquet      # Fear & Greed Index (daily, ~1000 rows)
├── spx.parquet             # S&P500 hourly returns (2 years)
├── eurusd.parquet          # EUR/USD hourly returns (2 years)
├── funding_rate.parquet    # BTC funding rate (8-hourly, 2 years)
└── open_interest.parquet   # BTC open interest (hourly, 30 days max)
```

**TTL:** `MAX_CACHE_AGE_H = 6` hours — cache freshness checked via
`path.stat().st_mtime` at read time.

**Cache invalidation:** The `download_all_external(force=True)` function
deletes all `.parquet` files in `data/external/` before re-downloading.

**Failure handling:** Each external source is wrapped in a `try/except`
block. On failure, the system falls back to the cached file if available, or
applies a neutral default value per feature:

| Feature | Neutral Default |
|---|---|
| `fear_greed` | 0.5 (mid-range, no sentiment bias) |
| `spx_return_24h` | 0.0 (no S&P change) |
| `eurusd_return_24h` | 0.0 (no FX change) |
| `funding_rate` | 0.0 (neutral funding) |
| `oi_change_24h` | 0.0 (no OI change) |

---

## Authentication Patterns

**No authentication is used anywhere in this codebase.**

- All Binance endpoints used are public (`/api/v3/klines`, `/fapi/v1/fundingRate`,
  `/futures/data/openInterestHist`) — these require no API key or HMAC signature.
- The Fear & Greed API (`api.alternative.me`) is fully open.
- yfinance accesses Yahoo Finance without credentials.
- There is no `.env` file, no secrets manager, no environment variable loading,
  and no credential store of any kind.

This means the project is intentionally designed for read-only public data
access and cannot place orders or access private account information.

---

## Network Timeout Configuration

All `requests.get()` calls use `timeout=15` seconds. There is no global
session object or connection pooling; each batch request creates a new
connection.

---

## Summary Table

| Integration | API | Auth | Rate Limit Strategy | Cache TTL | Features Produced |
|---|---|---|---|---|---|
| Binance Spot | `api.binance.com/api/v3/klines` | None | sleep 0.12s/req; retry 10s on error | SQLite (persistent) | OHLCV, all price/volume features |
| Binance Futures Funding | `fapi.binance.com/fapi/v1/fundingRate` | None | sleep 0.1s/req | 6 hours | `funding_rate`, `funding_momentum` |
| Binance Futures OI | `fapi.binance.com/futures/data/openInterestHist` | None | sleep 0.12s/req | 6 hours | `oi_change_24h` (unused in training) |
| alternative.me Fear & Greed | `api.alternative.me/fng/` | None | Single request | 6 hours | `fear_greed` |
| Yahoo Finance (yfinance) | Internal Yahoo API | None | Library-managed | 6 hours | `spx_return_24h`, `eurusd_return_24h` |
