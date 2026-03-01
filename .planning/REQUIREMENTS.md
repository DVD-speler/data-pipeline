# Requirements — Crypto Signal Model

## v1 Requirements (Week 1 — Model Verbetering)

### Regime Detection
- [ ] **REG-01**: Systeem detecteert marktregime (bull / bear / ranging) op basis van objectieve indicatoren (ADX, EMA structuur)
- [ ] **REG-02**: Short signalen worden alleen gegenereerd tijdens bevestigde beartrend (ADX > 20 + dalende EMA, niet alleen EMA200)
- [ ] **REG-03**: Regime-label is beschikbaar als feature voor het model
- [ ] **REG-04**: Walk-forward validatie toont ≥ 5/9 positieve folds na regime-filtering verbetering

### Feature Engineering
- [ ] **FEAT-01**: ADX indicator (14-periode) toegevoegd als feature (trendsterkte, geen richting)
- [ ] **FEAT-02**: Volume-profiel features: VWAP-afstand (close vs VWAP), volume-gewogen return
- [ ] **FEAT-03**: BTC dominance of ETH/BTC ratio versterkt (al aanwezig, eventueel met momentum)
- [ ] **FEAT-04**: Alle nieuwe features hebben importance > 0 na training (geen constante/bijna-constante waarden)
- [ ] **FEAT-05**: Feature matrix bevat geen data leakage (verificatie via permutation importance)

### Model Architectuur
- [ ] **MOD-01**: Hyperparameter optimalisatie via Optuna (LightGBM) op validatieset
- [ ] **MOD-02**: Regime-conditionele training: aparte modellen of regime als sterke feature
- [ ] **MOD-03**: ROC AUC > 0.57 op testset (verbetering van 0.5296)
- [ ] **MOD-04**: Walk-forward mean Sharpe > 0 (verbetering van -3.79)
- [ ] **MOD-05**: Model is statistisch significant in zowel bull als bear walk-forward folds

### Backtesting & Validatie
- [ ] **BT-01**: Backtest toont positieve returns in bull-regime simulatie (fold 1-3 walk-forward)
- [ ] **BT-02**: Backtest toont positieve returns in bear-regime simulatie (fold 7-9 walk-forward)
- [ ] **BT-03**: Total return > 0% op volledige walk-forward periode (alle 9 folds gecombineerd)

## v2 Requirements (Week 2 — Live Pipeline)

- [ ] **LIVE-01**: Binance live data feed (1h candle close trigger via REST polling)
- [ ] **LIVE-02**: Signaal alerting via Telegram of Discord webhook per signaal
- [ ] **LIVE-03**: Binance testnet paper trading integratie (automatische order plaatsing)
- [ ] **RISK-01**: Max drawdown circuit breaker (stop trading bij > X% dagelijks verlies)
- [ ] **RISK-02**: Positiegrootte risico-limiet (max % van portfolio per trade)
- [ ] **MON-01**: Monitoring dashboard (equity curve, open posities, recent signalen)
- [ ] **MON-02**: Automatische dagelijkse performance rapportage

## Out of Scope (v1)

- **On-chain data** (SOPR, MVRV, hash rate) — vereist betalende API; v3 mogelijk
- **Meerdere symbolen parallel traden** — eerst BTC-only stabiel krijgen
- **Reinforcement learning** — te complex voor korte tijdshorizon
- **Orderbook / tick data** — 1h candles zijn voldoende voor 12h horizon
- **ML-gebaseerde positiegrootte** — eenvoudige Kelly/fixed sizing is voldoende voor nu

## Traceability

| REQ-ID | Fase |
|--------|------|
| REG-01, REG-02, REG-03 | Fase 1: Regime Detectie |
| FEAT-01, FEAT-02, FEAT-03, FEAT-04, FEAT-05 | Fase 2: Feature Uitbreiding |
| MOD-01, MOD-02 | Fase 3: Model Optimalisatie |
| MOD-03, MOD-04, MOD-05, BT-01, BT-02, BT-03, REG-04 | Fase 4: Validatie & Evaluatie |
| LIVE-01, LIVE-02, LIVE-03, RISK-01, RISK-02, MON-01, MON-02 | Fase 5: Live Pipeline |
