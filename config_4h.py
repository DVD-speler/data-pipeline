"""
4h model configuratie — BTCUSDT / ETHUSDT
Voorspelling 3 candles (12 uur) vooruit op 4-uurlijkse OHLCV candles.
Medium-term model tussen uurmodel (24h horizon) en dagmodel (1d horizon).
"""
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).parent
DATA_DIR = ROOT_DIR / "data"


def symbol_path_4h(symbol: str, filename: str) -> Path:
    """Geef pad terug voor 4h model-specifiek bestand.
    Voorbeeld: symbol_path_4h("BTCUSDT", "model.pkl") → data/BTCUSDT_4h_model.pkl
    """
    return DATA_DIR / f"{symbol}_4h_{filename}"


# ── Binance API ────────────────────────────────────────────────────────────────
BINANCE_BASE_URL = "https://api.binance.com"
INTERVAL_4H = "4h"
SYMBOLS = ["BTCUSDT", "ETHUSDT"]
SYMBOL = "BTCUSDT"
DAYS_HISTORY = 730  # 2 jaar × 6 candles/dag = ~4380 rijen

# ── Model instellingen ─────────────────────────────────────────────────────────
PREDICTION_HORIZON_4H = 3   # 3 × 4h = 12h vooruit

# model.py vermenigvuldigt TEST_SIZE_DAYS × 24 (ontworpen voor uurdata).
# Voor 4h data: 1 dag = 6 candles.
# TEST_SIZE_DAYS = 23  → 23 × 24 = 552 rijen ≈ 92 dagen × 6 candles/dag
# VALIDATION_SIZE_DAYS = 15 → 15 × 24 = 360 rijen = 60 dagen × 6 candles/dag
TEST_SIZE_DAYS = 23
VALIDATION_SIZE_DAYS = 15

# ── Target dead zone ───────────────────────────────────────────────────────────
# 4h horizon: grotere moves verwacht dan 1h (0.3%) maar kleiner dan 1d (0.5%)
TARGET_DEAD_ZONE_PCT = 0.004  # 0.4%

# ── Signaaldrempels ────────────────────────────────────────────────────────────
SIGNAL_THRESHOLD = 0.58
SIGNAL_THRESHOLD_SHORT = 0.0

# ── Walk-forward validatie ─────────────────────────────────────────────────────
WALKFORWARD_TRAIN_DAYS = 270   # ~270 × 6 = 1620 candles trainvenster
WALKFORWARD_TEST_DAYS  = 30    # 30 × 6 = 180 candles per fold
WALKFORWARD_STEP_DAYS  = 30

# ── Regime-adaptieve drempelwaarden ───────────────────────────────────────────
REGIME_THRESHOLD_OFFSETS = {1: -0.05, 0: 0.0, -1: 0.08}

# ── Backtest ───────────────────────────────────────────────────────────────────
TRADE_FEE    = 0.001
STOP_LOSS_PCT = 0.02

# ── Features (4h timeframe — 33 features) ─────────────────────────────────────
FEATURE_COLS_4H_MODEL = [
    # Tijdfeatures
    "day_of_week",          # dag van de week (0=Ma … 6=Zo)
    "hour_4h",              # beginuur van de 4h candle (0,4,8,12,16,20)
    # Prijs & volume (rolling, geen look-ahead)
    "volatility_24h",       # 6-candle rolling std van rendementen (24h)
    "volatility_3d",        # 18-candle rolling std (3 dagen)
    "volume_ratio_24h",     # volume t.o.v. 24h gemiddelde
    "price_position_24h",   # positie in 24h high-low range
    "buy_pressure_24h",     # fractie stijgende candles (6 candles)
    "trend_consistency_3d", # fractie stijgende close-to-close (18 candles)
    # Multi-horizon momentum
    "return_12h",           # 3-candle rendement (12h)
    "return_24h",           # 6-candle rendement (24h)
    "return_3d",            # 18-candle rendement (3 dagen)
    "return_7d",            # 42-candle rendement (7 dagen)
    "return_30d",           # 180-candle rendement (30 dagen)
    "ath_7d_distance",      # afstand van 7-daags ATH
    # Technische indicatoren (4h candles)
    "rsi_14",               # RSI(14)
    "macd",                 # MACD
    "macd_signal",          # MACD signaal
    "bb_pct",               # Bollinger %B
    "ema_ratio_20",         # close / EMA(20)
    "ema_ratio_50",         # close / EMA(50)
    "price_vs_ema200",      # close / EMA(200) — regime indicator
    "atr_pct",              # ATR(14) / close
    "adx",                  # ADX(14) trendsterkte
    "vwap_distance",        # afstand van VWAP (dagelijks gereset)
    # Cross-asset & extern
    "fear_greed",           # Fear & Greed Index (dagelijks, forward-filled)
    "spx_return_24h",       # S&P500 24h rendement
    "eurusd_return_24h",    # EUR/USD 24h rendement
    "funding_rate",         # BTC perp funding rate (8h, forward-filled)
    "funding_momentum",     # 3-daagse verandering funding rate (18 candles)
    "btc_dvol",             # Deribit implied vol (genormaliseerd 0-1)
    "eth_btc_ratio",        # ETH/BTC 24h rendement
    # Vol regime
    "vol_regime",           # korte vol / lange vol ratio
]

# Regime-only kolommen: voor backtest-filter, NIET als model-input
FILTER_COLS_4H = ["market_regime"]

# Aliassen voor generieke model/backtest code
FEATURE_COLS = FEATURE_COLS_4H_MODEL
FILTER_COLS  = FILTER_COLS_4H
