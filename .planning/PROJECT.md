# Crypto Signal Model — Project Context

## Vision

Een ML-gedreven crypto trading signaal systeem dat **consistente winsten genereert over alle marktregimes** (bull, bear, ranging). Start als research platform; evolueert naar een volledig geautomatiseerde live trading bot op een demo-account.

## Core Value

**Een model dat winstgevend is ongeacht het marktregime** — niet alleen in bear markets maar ook in bull markets en zijwaartse bewegingen.

## Current State (2026-03-01)

### Wat werkt
- ✓ BTC 1h OHLCV pipeline (Binance, 730 dagen)
- ✓ Externe data: Fear & Greed, SPX, EUR/USD, funding rate, OI
- ✓ P1/P2 labeling systeem (dagelijkse extremen)
- ✓ 38 features over 1h + 4h timeframes
- ✓ LightGBM model met tijdsgewogen training
- ✓ Short signalen (threshold ≤ 0.45, EMA200 filter)
- ✓ Walk-forward validatie (9 folds, maandelijks)
- ✓ Backtest: test periode +16.1% return, Sharpe +1.53 (statistisch significant)

### Wat niet werkt
- ✗ Walk-forward: mean Sharpe -3.79, slechts 2/9 folds positief
- ✗ Model is bear-market afhankelijk: shorts helpen in bear, maar branden in bull recoveries
- ✗ AUC 0.53 — model heeft beperkte discriminatieve kracht
- ✗ EMA200 filter onderscheidt geen sustained bear van short-covering bounces
- ✗ Geen live pipeline, alerting of executie

## Goals

### Week 1 — Model dat werkt in alle regimes
- Walk-forward: ≥ 5/9 positieve folds (nu 2/9)
- Walk-forward mean Sharpe > 0 (nu -3.79)
- ROC AUC > 0.58 (nu 0.53)
- Test return positief in bull én bear simulatie

### Week 2 — Live pipeline & demo trading
- Binance live data feed (Binance WebSocket of REST polling)
- Telegram/Discord alerting per signaal
- Binance testnet paper trading
- Risk management: max drawdown limiet, positiegrootte, stop-loss
- Monitoring dashboard

## Constraints

- **Exchange:** Binance (USDT-margined futures voor shorts)
- **Timeframe:** 1h primair, 4h context
- **Data:** Maximaal 730 dagen history (Binance OHLCV), OI max 30 dagen
- **Model:** Python/scikit-learn ecosysteem (LightGBM, XGBoost, RF)
- **Kosten:** Geen extra API-kosten buiten gratis Binance + yfinance

## Key Decisions

| Beslissing | Reden | Status |
|-----------|-------|--------|
| LightGBM als primair model | Beste AUC (0.5296 vs RF 0.5323 — maar LightGBM depth=4 geeft threshold selectiviteit) | Vastgesteld |
| Short threshold ≤ 0.45 | 0.49 = noise; 0.45 = echt bearish signaal | Vastgesteld |
| 12h prediction horizon | Beste Sharpe in horizon scan (beter dan 24h/48h) | Vastgesteld |
| EMA200 regime filter | Blokkeert longs in bear market; blokkeert shorts in bull market | Deels — te grof |
| Target dead zone 0.3% | Verwijdert ~22% ruis boven fee breakeven | Vastgesteld |
| Geen calibratie | Isotonic overfits op 1440-row val set | Vastgesteld |

---
*Last updated: 2026-03-01 na initialisatie*
