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
DAYS_HISTORY = 1826  # 5 jaar (dekt volledige BTC 4-jaar cyclus incl. 2022 bearmarkt)

# ── P1/P2 definitie ────────────────────────────────────────────────────────────
MIN_HOURS_PER_DAY = 23

# ── Model instellingen ─────────────────────────────────────────────────────────
# Stap D: horizon van 4h naar 12h — minder ruis, meer ruimte voor de predictie
# Sprint 3: test 24h horizon → top features zijn 24h+ in nature (prev_day_return, spx_return_24h, return_30d)
PREDICTION_HORIZON_H = 24

# ── Optuna hyperparameter search (S8-A / S8-B / S9-B) ────────────────────────
# S8-A: trials verhoogd van 50 naar 150 voor stabielere params
OPTUNA_N_TRIALS       = 150
# S9-B: Sharpe objective alleen voor symbolen die er baat bij hebben.
# BTC: Sharpe-objective werkt goed (stabiele trends, voldoende trades in valperiode).
# ETH: AUC-objective werkt beter (kortere trends, minder trades → Optuna overfit).
OPTUNA_SHARPE_SYMBOLS = []   # lege lijst = gebruik OPTUNA_SHARPE_OBJECTIVE voor alle symbolen

# ── S11-A: Bear-regime short model ────────────────────────────────────────────
# Activeer short posities uitsluitend wanneer market_regime == -1 (confirmed bear).
# Signaal: proba <= SHORT_ENTRY_THRESHOLD (model ziet weinig kans op stijging).
# Doel: genereert rendement in bear-fases waar longs geblokkeerd zijn (0 trades).
BEAR_REGIME_SHORT_ENABLED = True
SHORT_ENTRY_THRESHOLD     = 0.30   # short wanneer proba < 0.30 (model sterk bearish)
# Short alleen voor symbolen met bewezen WF-verbetering (BTC: WF +0.76→+4.97 met daily gate).
# ETH: short model hurt performance (WF regressie + negatieve single-run) → uitgeschakeld.
BEAR_REGIME_SHORT_SYMBOLS = ["BTCUSDT"]  # lege lijst = short voor alle symbolen

# S8-C: model selectie op Sharpe i.p.v. ROC AUC (S9-A).
# ETH RandomForest had Sharpe +13.14 maar werd niet gekozen (AUC-selectie koos LightGBM +3.09).
# Guard: minimaal MODEL_SELECT_MIN_TRADES trades op testset nodig om in aanmerking te komen.
MODEL_SELECT_BY_SHARPE  = True
MODEL_SELECT_MIN_TRADES = 20

# ── S9-B: Dagelijks model alignment gate ──────────────────────────────────────
# Blokkeert 1h longs als het dagelijks model bearish is (proba < DAILY_GATE_THRESHOLD).
# Voorkomt longs die ingaan tegen de hogere-timeframe trend.
DAILY_GATE_ENABLED    = True
DAILY_GATE_THRESHOLD  = 0.45   # dagelijks proba onder 0.45 = bearish dagelijks model
# Activeer daily gate alleen voor symbolen met voldoende kwaliteit (AUC > ~0.57).
# BTC: AUC 0.63 → gate actief. ETH: AUC 0.53 → gate uitgeschakeld (te zwak).
DAILY_GATE_SYMBOLS    = ["BTCUSDT"]  # lege lijst = gate voor alle symbolen

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
TARGET_DEAD_ZONE_PCT  = 0.003   # 0.3% — symmetrische fallback (legacy)
# B4: asymmetrische dead zone — kleine ups behouden (accumulatie), kleine downs filteren (ruis)
TARGET_DEAD_ZONE_UP   = 0.002   # 0.2%: kleine opwaartse moves behouden (accumulatiefase)
TARGET_DEAD_ZONE_DOWN = 0.004   # 0.4%: kleine neerwaartse moves als ruis verwijderen

# ── Features (1h timeframe) ────────────────────────────────────────────────────
# Sprint 6 cleanup (2026-03-28): 66 → 47 features na permutation importance analyse
# Verwijderd: 19 features met nul of negatieve permutation importance (BTC + ETH beide)
#
# Top-5 features (permutation importance BTC / ETH):
#   google_trends_btc   +0.047 / +0.016    eurusd_return_24h  +0.040 / +0.019
#   dxy_return_24h      +0.030 / +0.020    trends_momentum_4w +0.024 / +0.022
#   spx_return_24h      +0.016 / +0.026
#
# Verwijderde features (nul of negatief in beide symbolen):
#   btc_put_call_ratio   0.000 / 0.000    btc_dominance        0.000 / 0.000
#   btc_dominance_7d_chg 0.000 / 0.000    is_hammer            0.000 / 0.000
#   is_engulfing         0.000 / 0.000    gap_up               0.000 / 0.000
#   ema_ratio_50        -0.00034/-0.00045  chikou_position     -0.00012/-0.00017
#   volume_ratio        -0.00010/-0.00008  return_4h           -0.00007/-0.00025
#   dxy_above_200ma     -0.00059/-0.00008  session             ~0.000 / 0.000
#   return_12h          +0.00001/-0.00051  return_6h           +0.00008/-0.00008
#   price_position      -0.00127/+0.00026  lower_wick_pct      ~0.000 / 0.000
#   upper_wick_pct       0.000 / 0.000    candle_body_pct      0.000 / 0.000
#   volume_spike_48h    +0.00037/-0.00029  (conflicterend, netto neutraal)

FEATURE_COLS_1H = [
    # Tijdsfeatures
    "hour",
    "day_of_week",
    "hour_of_week",             # gecombineerde dag×uur code (0–167)
    # P1/P2 statistieken
    "p1_probability",           # kans P1 op dit uur (train-only heatmap)
    # Prijs & volume
    "volatility_24h",
    "prev_day_return",
    # Momentum
    "return_2h",
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
    "price_vs_ema200",          # marktregime — close / EMA(200)
    "atr_pct",                  # genormaliseerde volatiliteit — ATR(14) / close
    # Cross-asset & externe features
    "fear_greed",               # Fear & Greed Index (0=extreme fear, 1=extreme greed)
    "spx_return_24h",           # S&P500 24h rendement (risico-appetite indicator)
    "eurusd_return_24h",        # EUR/USD 24h rendement (dollar sterkte proxy)
    "eth_btc_ratio",            # ETH/BTC 24h rendement (altcoin season indicator)
    "funding_rate",             # BTC perp funding rate (sentiment: positief=longs domineren)
    "funding_momentum",         # 3-daagse verandering funding rate (sentiment shift)
    # Macro-momentum
    "return_7d",                # 7-daags BTC rendement (weektrend)
    "return_30d",               # 30-daags BTC rendement (maandtrend)
    "ath_7d_distance",          # afstand van 7-daags ATH (negatief = in correctie)
    # Regime detectie
    "adx",                      # ADX(14) trendsterkte 0-100 (>20 = trending markt)
    "vwap_distance",            # (close − dag-VWAP) / close
    "poc_distance",             # (close − POC_168h) / close — wekelijks volume-zwaartepunt
    # Opties-markt (volatiliteit)
    "btc_dvol",                 # Deribit BTC implied volatility index (genormaliseerd 0–1)
    # Macro (VIX / valuta)
    "vix_level",                # CBOE VIX aandelenmarkt-angstmeter
    "usdjpy_return_24h",        # USD/JPY dagelijks rendement (risk-off indicator)
    "usdjpy_return_7d",         # USD/JPY 7-daags rendement — carry trade unwind
    # Ichimoku Cloud
    "cloud_position",           # +1 boven wolk, 0 in wolk, -1 onder wolk
    "cloud_thickness",          # (senkou_a − senkou_b) / close — dikke wolk = sterke trend
    "tk_cross",                 # +1 Tenkan boven Kijun (bull), -1 eronder (bear)
    # DXY Dollar Index
    "dxy_return_24h",           # dagelijks DXY rendement (negatief = bullish crypto)
    "dxy_return_7d",            # 7-daags DXY rendement (macro dollar trend)
    # RSI Divergentie
    "rsi_bull_divergence",      # price lower low + RSI higher low (bullish reversal)
    "rsi_bear_divergence",      # price higher high + RSI lower high (bearish top)
    # Google Trends sentiment
    "google_trends_btc",        # genormaliseerd BTC zoekvolume (0–100); hoog = retail FOMO
    "trends_momentum_4w",       # 4-weekse verandering zoekvolume; stijgend = FOMO opbouw
    "trends_spike",             # 1 als zoekvolume > 90e percentiel (extreme FOMO)
    # Sprint 7 — nieuwe signalen
    "oi_return_24h",            # S7-A: Bybit OI 24h % verandering (stijgend = grotere positie)
    "oi_price_divergence",      # S7-A: +1 OI+prijs stijgen (bullish), -1 OI stijgt+prijs daalt (bearish)
    "active_addresses_7d_chg",  # S7-B: blockchain.info wekelijkse verandering unieke adressen
    "hash_rate_7d_chg",         # S7-B: wekelijkse verandering BTC hash rate (miner vertrouwen)
    "fear_greed_7d_chg",        # S7-C: Fear & Greed 7-daags momentum (168h diff)
]
# Uitgesloten features (code aanwezig, niet in FEATURE_COLS):
#   btc_skew_25d         — LIVE-ONLY gate (geen historische Deribit data)
#   hmm_bull/bear_prob   — Sprint 5: regressie (HMM redundant met ADX market_regime)
#   btc_eth_corr_7d      — Sprint 5: regressie (-5 Sharpe vs baseline)
#   days_since_halving   — Sprint 2: corr -0.28, leidt tot extra trades/lagere WR
#   supertrend_distance  — Sprint 2: corr -0.36, gecaptured door adx+ema
#   usdt_dominance       — slechts 25% datadekking (CoinGecko ~1 jaar history)
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

# ── Multi-timeframe signaalconfirmatie ────────────────────────────────────────
# 4h-model proba moet boven deze drempel liggen voor een 1h-entry.
# 0.0 = uitgeschakeld (geen 4h-confirmatie vereist).
# Wordt apart geoptimaliseerd per symbool (zie {symbol}_4h_optimal_threshold.json).
SIGNAL_THRESHOLD_4H = 0.52   # lager dan 1h drempel (4h is trager signaal)

# ── Model-gestuurd sluitmodel ─────────────────────────────────────────────────
# Sluit LONG als proba daalt onder EXIT_PROBA_LONG (model ziet kans niet meer).
# Sluit SHORT als proba stijgt boven EXIT_PROBA_SHORT.
# Initiële waarden; worden geoptimaliseerd via backtest-sweep en opgeslagen in
# {symbol}_exit_proba.json (overschrijft deze defaults voor dat symbool).
EXIT_PROBA_LONG  = 0.45   # te optimaliseren: sweep 0.30–0.55
EXIT_PROBA_SHORT = 0.55   # symmetrisch voor shorts
MAX_HOLD_HOURS   = 168    # absolute tijdsvangnet: 1 week (vervangt 24h timeout)

# ── Macro gates (P1/P2 verbeteringen) ─────────────────────────────────────────
# P1 — DVOL gate: BTC implied volatility te hoog → markt prijst crash in
# btc_dvol is genormaliseerd op 0–1 (Deribit DVOL index / 100)
# P3 — Drawdown circuit breaker
# Zodra drawdown > MAX_DRAWDOWN_GATE → stop trading voor CIRCUIT_BREAKER_COOLDOWN_H uur
# Zet MAX_DRAWDOWN_GATE = 0.0 om uit te schakelen
MAX_DRAWDOWN_GATE            = -0.15   # -15% drawdown van peak triggert pauze
CIRCUIT_BREAKER_COOLDOWN_H   =  168    # 7 dagen cooldown (168 uur)

DVOL_GATE = 0.65          # Blokkeert longs als btc_dvol > 0.65 (DVOL index > 65)

# P1 — Maandelijkse terugval gate
# Blokkeert nieuwe longs als BTC de afgelopen 30 dagen > 10% gedaald is
RETURN_30D_LONG_GATE = -0.10

# P1 — Volatiliteit-gewogen positiegrootte
# Schalingsfactor: hogere waarde = kleiner positie bij hoge volatiliteit
# size = base_size / (1 + VOL_SIZE_SCALE * volatility_24h)
VOL_SIZE_SCALE = 5.0

# P2 — VIX gate: aandelenmarkt-angst blokkeert crypto longs
# VIX > 25 = elevated fear (historisch gemiddelde ~20, crashes >30)
VIX_GATE = 25.0

# P2 — USD/JPY gate: sterke yen-appreciatie signaleert carry trade unwind
# Blokkeert longs als JPY de afgelopen 7 dagen > 3% sterker is geworden
USDJPY_RETURN_7D_GATE = -0.03   # negatief = JPY wordt duurder (USD/JPY daalt)

# C4 — Deribit Put/Call ratio gate
# Extreme put-dominantie (P/C > 1.5) signaleert bearish positionering grote spelers
PUT_CALL_RATIO_GATE = 1.5

# T2-B — Funding extreme gate
# Extreem positieve funding (longs betalen te veel) = contrair bearish signaal
# Blokkeert nieuwe longs als funding_rate > FUNDING_EXTREME_GATE (per 8u)
FUNDING_EXTREME_GATE = 0.0005   # +0.05% per 8u = overbought signaal

# T3-C — Deribit 25-delta skew gate (LIVE-ONLY — geen historische data)
# Negatieve skew = calls duurder → bullish; positieve skew = puts duurder → bearish
# Blokkeert longs als put-IV > call-IV met meer dan SKEW_BEARISH_GATE (in % punten)
SKEW_BEARISH_GATE = 5.0   # blokkeert longs als 25D put-IV > 25D call-IV + 5%

# ── Kelly Criterion positiegrootte (T2-D) ─────────────────────────────────────
# Half-Kelly als upper-bound voor positiegroottes in live trading.
# kelly_half = min(full_Kelly × KELLY_FRACTION, KELLY_MAX_FRACTION)
# kelly_half wordt per symbool berekend op validatieset en opgeslagen in {symbol}_kelly.json
USE_KELLY_SIZING    = True   # schakel Kelly sizing in/uit
KELLY_FRACTION      = 0.5    # half-Kelly fractie (0.5 = conservatief, minder ruin-risk)
KELLY_MAX_FRACTION  = 0.20   # absolute maximale positie als fractie van kapitaal

# ── Regime-afhankelijke SL/TP (T4-C) ─────────────────────────────────────────
# Optimalere SL/TP per marktregime. Bull: meer ruimte voor de trend.
# Bear: strakke stop om verliezen te beperken. Ranging: standaard.
REGIME_SL_TP = {
    "bull":    {"sl_pct": 0.025, "tp_pct": 0.08},   # bull-run: ruimte voor de trend
    "ranging": {"sl_pct": 0.020, "tp_pct": 0.06},   # ranging: standaard (was tp=0.06)
    "bear":    {"sl_pct": 0.015, "tp_pct": 0.04},   # bear: strakke stop, kleiner TP
}

# ── Multi-asset correlatie guard (T4-A) ───────────────────────────────────────
# Wanneer BTC én ETH tegelijk open staan (gecorreleerd risico):
#   24h-correlatie > BLOCK → blokkeer tweede positie (max 1 actief)
#   24h-correlatie > HALVE → halveer positiegrootte tweede trade
CORR_GUARD_THRESHOLD_BLOCK = 0.90   # correlatie > 90% → blokkeer
CORR_GUARD_THRESHOLD_HALVE = 0.70   # correlatie > 70% → halveer
