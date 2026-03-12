"""
Dagelijks model configuratie — BTCUSDT / ETHUSDT
Voorspelling 1 dag vooruit op dagelijkse OHLCV candles.
Aparte configuratie naast het uurmodel (config.py).
"""
from pathlib import Path

# ── Paths (zelfde data directory als uurmodel) ─────────────────────────────────
ROOT_DIR = Path(__file__).parent
DATA_DIR = ROOT_DIR / "data"


def symbol_path_daily(symbol: str, filename: str) -> Path:
    """Geef pad terug voor dagmodel-specifiek bestand.
    Voorbeeld: symbol_path_daily("BTCUSDT", "model.pkl") → data/BTCUSDT_1d_model.pkl
    """
    return DATA_DIR / f"{symbol}_1d_{filename}"


# ── Binance API ────────────────────────────────────────────────────────────────
BINANCE_BASE_URL = "https://api.binance.com"
INTERVAL_DAILY = "1d"
SYMBOLS = ["BTCUSDT", "ETHUSDT"]
SYMBOL = "BTCUSDT"
DAYS_HISTORY = 730  # 2 jaar dagelijkse candles

# ── Model instellingen ─────────────────────────────────────────────────────────
PREDICTION_HORIZON_D = 1     # 1 dag vooruit

# model.py vermenigvuldigt TEST_SIZE_DAYS × 24 (ontworpen voor uurdata).
# Voor dagdata compenseren we: echte dagcandles = waarde × 24.
# TEST_SIZE_DAYS = 3  → 3 × 24 = 72 dagcandles ≈ 2.5 maanden testperiode
# VALIDATION_SIZE_DAYS = 2 → 2 × 24 = 48 dagcandles ≈ 1.5 maanden validatie
TEST_SIZE_DAYS = 3
VALIDATION_SIZE_DAYS = 2

# ── Target dead zone ───────────────────────────────────────────────────────────
# Op dagbasis zijn bewegingen groter → ruimere dead zone dan uurmodel (0.3%)
TARGET_DEAD_ZONE_PCT = 0.005  # 0.5%

# ── Signaaldrempels (worden geoptimaliseerd via optimize_threshold) ────────────
SIGNAL_THRESHOLD = 0.58
SIGNAL_THRESHOLD_SHORT = 0.0

# ── Walk-forward validatie ─────────────────────────────────────────────────────
WALKFORWARD_TRAIN_DAYS = 365  # 1 jaar trainvenster (365 dagcandles)
WALKFORWARD_TEST_DAYS = 30    # 1 maand per fold
WALKFORWARD_STEP_DAYS = 30

# ── Regime-adaptieve drempelwaarden (zelfde logica als uurmodel) ───────────────
REGIME_THRESHOLD_OFFSETS = {1: -0.05, 0: 0.0, -1: 0.08}

# ── Backtest ───────────────────────────────────────────────────────────────────
TRADE_FEE = 0.001
STOP_LOSS_PCT = 0.02

# ── Features (dagelijks timeframe — 28 features) ──────────────────────────────
FEATURE_COLS_DAILY = [
    # Tijdfeature
    "day_of_week",
    # Prijs & volume (dagelijkse rolling windows)
    "volatility_7d",            # 7-daagse rolling std van dagrendementen
    "volatility_30d",           # 30-daagse rolling std (vol regime)
    "prev_week_return",         # vorige week return (5 dagcandles)
    "volume_ratio_30d",         # volume t.o.v. 30-daags gemiddelde
    "volume_spike_30d",         # volume vs 30d gemiddelde (spike detectie)
    "price_position_7d",        # positie van close in 7-daags high-low range
    # Trend kwaliteit
    "trend_consistency_4w",     # fractie stijgende dag-candles (20 dagen)
    "buy_pressure_14d",         # fractie candles close > open (14 dagen)
    # Macro momentum
    "return_7d",                # 7-daags BTC rendement
    "return_30d",               # 30-daags BTC rendement
    "ath_30d_distance",         # afstand van 30-daags ATH
    # Technische indicatoren (dagelijks)
    "rsi_14",                   # RSI(14) op dagcandles
    "macd",                     # MACD op dagcandles
    "macd_signal",              # MACD signaal op dagcandles
    "bb_pct",                   # Bollinger %B op dagcandles
    "ema_ratio_20",             # close / EMA(20) dagcandles
    "ema_ratio_50",             # close / EMA(50) dagcandles
    "price_vs_ema200",          # close / EMA(200) — regime indicator
    "atr_pct",                  # ATR(14) / close — genormaliseerde volatiliteit
    # Cross-asset & extern
    "fear_greed",               # Fear & Greed Index (dagelijks)
    "spx_return_1w",            # S&P500 1-weeks rendement
    "eurusd_return_1w",         # EUR/USD 1-weeks rendement
    "eth_btc_ratio",            # ETH/BTC dagelijks rendement
    "funding_rate",             # BTC perp funding rate
    "funding_momentum",         # 7-daagse verandering funding rate
    # Regime detectie
    "adx",                      # ADX(14) trendsterkte
    # Volatiliteitsregime (opties-markt)
    "btc_dvol",                 # Deribit BTC implied volatility (genormaliseerd 0–1)
]

# Regime-only kolommen: voor backtest-filter, NIET als model-input
FILTER_COLS_DAILY = ["market_regime"]

# Aliassen zodat generieke model/backtest code ook werkt met dagmodel
FEATURE_COLS = FEATURE_COLS_DAILY
FILTER_COLS = FILTER_COLS_DAILY
