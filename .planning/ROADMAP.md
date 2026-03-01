# Roadmap — Crypto Signal Model

**Doel:** Model dat winstgevend is in alle marktregimes → live trading pipeline
**Status:** Fase 1 klaar te starten

---

## Fase 1 — Regime Detectie & Slim Short Filter
**Goal:** Onderscheid sustained bear van tijdelijke correctie; shorts alleen bij echte downtrend
**Requirements:** REG-01, REG-02, REG-03
**Tijdsinschatting:** 1-2 dagen

### Taken
1. `ADX (14)` toevoegen aan `features.py` (trendsterkte indicator, 0-100; >25 = sterke trend)
2. `adx_trend` feature: ADX × sign(EMA20 - EMA50) → positief in bull, negatief in bear
3. Regime classificatie feature: `market_regime` = encoded bull/bear/ranging
4. Short filter aanscherpen: `optimize_short_threshold()` bouwt voort op regime — shorts alleen als `adx_trend < -threshold` (bevestigde bear)
5. `features.py` bijwerken + `config.py` FEATURE_COLS uitbreiden
6. Features herberekenen en model hertrainen

### Succes Criteria
- [ ] ADX feature heeft importance > 0 na training
- [ ] `market_regime` feature splitst walk-forward folds correct (folds 7-9 = bear gedetecteerd)
- [ ] Short trades in bull-regime folds (1-3) dalen significant (minder false shorts)
- [ ] Walk-forward folds 2-3 (bull recovery mei-jun 2025) verbeteren in Sharpe

---

## Fase 2 — Feature Uitbreiding
**Goal:** AUC verbeteren van 0.53 → >0.57 door rijkere feature set
**Requirements:** FEAT-01 t/m FEAT-05
**Tijdsinschatting:** 1-2 dagen

### Taken
1. **VWAP-afstand** feature: `(close - vwap) / close` per dag (gemiddeld gewogen prijs vs. huidige prijs)
2. **Volume momentum**: `volume_7d_avg` / `volume_30d_avg` — dalend volume in bear = bearish bevestiging
3. **Volatility clustering**: Garch-achtige feature: rolling std van returns normaliseren over 24h window
4. **Meerdere EMA ratios**: `ema_ratio_100`, `ema_ratio_200` trend alignment score
5. Feature selectie: verwijder features met importance < 0.005 (ruis vermijden)
6. Permutation importance check: verifieer geen leakage

### Succes Criteria
- [ ] Alle nieuwe features importance > 0
- [ ] ROC AUC stijgt met ≥ 0.01 t.o.v. baseline (0.53 → >0.54)
- [ ] Geen feature met std ≈ 0 (geen constante features)

---

## Fase 3 — Model Optimalisatie (Optuna)
**Goal:** LightGBM hyperparameters systematisch optimaliseren op validatieset
**Requirements:** MOD-01, MOD-02
**Tijdsinschatting:** 1 dag

### Taken
1. Optuna integratie in `model.py`: zoek naar beste `max_depth`, `num_leaves`, `min_child_samples`, `learning_rate`, `subsample`
2. 50-100 trials op validatieset (Sharpe als objective, niet AUC)
3. Beste parameters opslaan in `config.py` of als json artifact
4. Optioneel: Regime-conditionele modellen — train apart model op bull-folds en bear-folds, combineer via ensemble
5. Model vergelijking bijwerken in `model_compare.py`

### Succes Criteria
- [ ] Optuna vindt betere hyperparameters dan huidige hand-getuned waarden
- [ ] Val-Sharpe stijgt t.o.v. baseline bij geoptimaliseerde parameters
- [ ] Geen overfitting: test-Sharpe ≈ val-Sharpe (niet alleen val goed)

---

## Fase 4 — Walk-Forward Validatie & Beoordeling
**Goal:** Aantonen dat model werkt in alle 9 walk-forward folds (bull + bear + ranging)
**Requirements:** MOD-03, MOD-04, MOD-05, BT-01, BT-02, BT-03, REG-04
**Tijdsinschatting:** 0.5 dag (run + analyse)

### Taken
1. Volledige pipeline runnen: `features → model → model_compare → walkforward`
2. Walk-forward resultaten analyseren per regime-type:
   - Bull folds (1-3, apr-jul 2025): target ≥ 3/3 positief
   - Ranging folds (4-6, aug-okt 2025): target ≥ 2/3 positief
   - Bear folds (7-9, nov-feb 2026): target ≥ 3/3 positief
3. Als mean Sharpe < 0: terug naar Fase 1/2 met gericht debug
4. Lessons learned bijwerken in `tasks/lessons.md`

### Succes Criteria
- [ ] Walk-forward ≥ 5/9 positieve folds
- [ ] Mean walk-forward Sharpe > 0
- [ ] ROC AUC testset > 0.57
- [ ] Statistisch significant: test Sharpe > random p95

---

## Fase 5 — Live Pipeline (Week 2)
**Goal:** Signalen van model naar echte orders op Binance testnet
**Requirements:** LIVE-01 t/m MON-02
**Tijdsinschatting:** 4-5 dagen

### Taken
1. **Live data feed**: Binance REST polling elke uur na candle close (`:05` trigger)
2. **Signal pipeline**: Automatisch features berekenen → model.predict → signaal genereren
3. **Alerting**: Telegram bot of Discord webhook stuurt LONG/SHORT/NEUTRAL signaal
4. **Paper trading**: Binance testnet order plaatsing (market orders, correcte positiegrootte)
5. **Risk management module**: `src/risk_manager.py`
   - Max dagelijks verlies limiet (circuit breaker)
   - Max positiegrootte per trade (% van portfolio)
   - Trailing stop-loss management
6. **Monitoring**: Simpele dashboard (Streamlit of matplotlib auto-refresh)
   - Equity curve live bijwerken
   - Open posities tonen
   - Recente signalen log

### Succes Criteria
- [ ] Signaal gegenereerd binnen 5 minuten na candle close
- [ ] Telegram alert ontvangen bij elk signaal
- [ ] Testnet orders correct geplaatst en gesloten
- [ ] Circuit breaker stopt trading bij > 5% dagelijks verlies
- [ ] Dashboard toont live equity curve

---

## Status Overzicht

| Fase | Naam | Status | Folds doel |
|------|------|--------|-----------|
| 1 | Regime Detectie | ⏳ Klaar te starten | — |
| 2 | Feature Uitbreiding | ⏳ Na Fase 1 | — |
| 3 | Model Optimalisatie | ⏳ Na Fase 2 | — |
| 4 | Walk-Forward Validatie | ⏳ Na Fase 3 | ≥ 5/9 positief |
| 5 | Live Pipeline | ⏳ Week 2 | — |

**Huidige baseline:** 2/9 positieve folds, mean Sharpe -3.79, AUC 0.53, test return +16.1%
**Doelstelling:** ≥ 5/9 positieve folds, mean Sharpe > 0, AUC > 0.57
