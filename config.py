from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).parent
DATA_DIR = ROOT_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)
(DATA_DIR / "stats").mkdir(exist_ok=True)


def symbol_path(symbol: str, filename: str) -> Path:
    """Geef het pad terug voor een symbool-specifiek bestand.
    Voorbeeld: symbol_path("BTCUSDT", "model.pkl") → data/BTCUSDT_model.pkl
    """
    return DATA_DIR / f"{symbol}_{filename}"

# ── Binance API ────────────────────────────────────────────────────────────────
BINANCE_BASE_URL = "https://api.binance.com"

# ── Multi-symbol & multi-timeframe ─────────────────────────────────────────────
SYMBOL    = "BTCUSDT"
INTERVAL  = "1h"
SYMBOLS   = ["BTCUSDT", "ETHUSDT"]
INTERVALS = ["1h", "4h"]
DAYS_HISTORY = 730

# ── P1/P2 definitie ────────────────────────────────────────────────────────────
MIN_HOURS_PER_DAY = 23

# ── Model instellingen ─────────────────────────────────────────────────────────
# Stap D: horizon van 4h naar 12h — minder ruis, meer ruimte voor de predictie
# Sprint 3: test 24h horizon → top features zijn 24h+ in nature (prev_day_return, spx_return_24h, return_30d)
PREDICTION_HORIZON_H = 24

# Stap A: testset verkleind van 365 naar 90 dagen → model ziet meer marktcycli
TEST_SIZE_DAYS       = 90
# Stap C: aparte validatieset voor threshold-optimalisatie (niet de testset)
VALIDATION_SIZE_DAYS = 60

# Stap C: startpunt drempelwaarde (wordt geoptimaliseerd op validatieset)
SIGNAL_THRESHOLD = 0.58

# Stap F: short uitgeschakeld (threshold 0.0 is nooit bereikbaar voor probas ∈ (0,1))
SIGNAL_THRESHOLD_SHORT = 0.0

# ── Walk-forward validatie ─────────────────────────────────────────────────────
# 270d venster (ipv 365d) geeft meer folds (~11) voor betere generalisatieschatting
WALKFORWARD_TRAIN_DAYS = 270
WALKFORWARD_TEST_DAYS  = 30
WALKFORWARD_STEP_DAYS  = 30

# ── Target dead zone ───────────────────────────────────────────────────────────
# Minimale koersbeweging (%) om een uur als directional te labelen.
# Rijen met |move| < drempel worden als "neutraal" uit training verwijderd.
# Breakeven voor een round-trip: 2 × TRADE_FEE = 0.2%  → drempel iets hoger.
TARGET_DEAD_ZONE_PCT = 0.003   # 0.3%

# ── Features (1h timeframe) ────────────────────────────────────────────────────
FEATURE_COLS_1H = [
    # Tijdsfeatures
    "hour",
    "day_of_week",
    "hour_of_week",             # gecombineerde dag×uur code (0–167)
    "session",
    # P1/P2 statistieken
    "p1_probability",           # kans P1 op dit uur (train-only heatmap)
    # Prijs & volume (rolling)
    "volatility_24h",
    "prev_day_return",
    "volume_ratio",             # volume t.o.v. 24h gemiddelde
    "volume_spike_48h",         # volume t.o.v. 48h gemiddelde (spike detectie)
    "price_position",
    # Multi-horizon momentum
    "return_2h",
    "return_4h",
    "return_6h",
    "return_12h",
    # Volatiliteitsregime
    "vol_regime",               # recente 4h-vol / 24h-vol (>1=expanding, <1=squeeze)
    # Trendkwaliteit
    "trend_consistency_12h",    # fractie stijgende candles (close-to-close, 12h)
    "buy_pressure",             # fractie stijgende candles (open-to-close, 24h)
    # Technische indicatoren (1h)
    "rsi_14",
    "macd",
    "macd_signal",
    "bb_pct",
    "ema_ratio_20",
    "ema_ratio_50",
    "price_vs_ema200",          # marktregime — close / EMA(200)
    "atr_pct",                  # genormaliseerde volatiliteit — ATR(14) / close
    # Cross-asset & externe features
    "fear_greed",               # Fear & Greed Index (0=extreme fear, 1=extreme greed)
    "spx_return_24h",           # S&P500 24h rendement (risico-appetite indicator)
    "eurusd_return_24h",        # EUR/USD 24h rendement (dollar sterkte proxy)
    "eth_btc_ratio",            # ETH/BTC 24h rendement (altcoin season indicator)
    "funding_rate",             # BTC perp funding rate (sentiment: positief=longs domineren)
    "funding_momentum",         # 3-daagse verandering funding rate (sentiment shift)
    # Macro-momentum (bearmarktdetectie)
    "return_7d",                # 7-daags BTC rendement (weektrend)
    "return_30d",               # 30-daags BTC rendement (maandtrend)
    "ath_7d_distance",          # Afstand van 7-daags ATH (negatief = in correctie)
    # Regime detectie (Fase 1)
    "adx",                      # ADX(14) trendsterkte 0-100 (>20 = trending markt)
    "vwap_distance",            # (close − dag-VWAP) / close: positie vs. gewogen gemiddelde
    # Volatiliteitsregime (opties-markt)
    "btc_dvol",                 # Deribit BTC implied volatility index (genormaliseerd 0–1)
]
# Regime-only columns: in de feature matrix voor backtest-filter, NIET als model feature.
# adx_trend en market_regime geven expliciete richting → over-confidence in bullish val-periode
#   → threshold zakt naar 0.50 → model overfits op val-regime.
# Oplossing: model leert slechts trendsterkte (adx), backtest-filter gebruikt market_regime.
FILTER_COLS = ["market_regime"]   # altijd in feature matrix, nooit in model-input
# Verwijderd (feature importance ≈ 0 of gebroken data):
#   "return_4h"       — 0.000 belang (letterlijk nul) in meerdere runs → verwijderd Fase 2
#   "direction_bias"  — 0.000762 belang, biedt geen additioneel signaal boven p1_probability
#   "returns"         — 0.002264 belang, ruwe 1h return is te lawaaierig voor 12h-predictie
#   "oi_change_24h"   — Binance OI API limiteert tot 30 dagen history; slechts 4% dekking
#                       in de trainperiode → effectief constante 0-kolom, voegt ruis toe

# ── Features (4h timeframe) ────────────────────────────────────────────────────
FEATURE_COLS_4H = [
    "rsi_14_4h",
    "macd_4h",
    "bb_pct_4h",
    "ema_ratio_20_4h",
]

FEATURE_COLS = FEATURE_COLS_1H + FEATURE_COLS_4H

# ── Horizon scan ───────────────────────────────────────────────────────────────
HORIZON_SCAN = [12, 24, 48]   # te testen voorspellingstijdshorizons (in uren)

# ── Regime-adaptieve drempelwaarden ───────────────────────────────────────────
# Offset toegepast bovenop de geoptimaliseerde threshold, per marktregime.
# market_regime: +1=bevestigde bull (ADX>20, +DI>-DI), 0=ranging, -1=bevestigde bear
# Bull  → lagere drempel (meer trades in gunstige markt)
# Bear  → hogere drempel (alleen longs met extreem hoge zekerheid)
REGIME_THRESHOLD_OFFSETS = {1: -0.05, 0: 0.0, -1: 0.08}

# ── Backtest ───────────────────────────────────────────────────────────────────
TRADE_FEE     = 0.001
STOP_LOSS_PCT = 0.02
