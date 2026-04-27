# Lessons

Leerpunten uit experimenten + sprints. Tegen-voorbeelden zijn even
waardevol als positieve bevindingen — ze voorkomen herhaling.

## Data integratie

**L1: Binance `openInterestHist` heeft een ~30-dagen history limit.**
Endpoint `/futures/data/openInterestHist?period=1h` geeft 400 voor
requests > 30 dagen terug. Bij 730-daags trainvenster betekent dat 4%
dekking. Gebruik **Bybit OI** of Coinglass; Binance OI is onbruikbaar
voor langer dan een maand.

**L2: yfinance 1h-bars voor Amerikaanse aandelen starten op `:30` (09:30 ET = 13:30 UTC).**
BTC-OHLCV staat op `:00`. Left-join geeft 0% match → 100% NaN → fills
naar 0.0. **Fix:** `df.index = df.index.floor("h")` + dedupe na download.
Geldt voor SPX, EUR/USD, USD/JPY, DXY.

**L3: pandas 2.x slaat parquet op met `datetime64[ms, UTC]`, BTC-OHLCV gebruikt `[ns, UTC]`.**
Dtype-mismatch geeft 0 matches op join, ook al is de tijdzone identiek.
**Fix:** `df_ext.index = df_ext.index.astype(base.index.dtype)` vóór de join.

**L7: Feature importance == 0 is altijd een data-probleem, niet een
model-probleem.** Constant of bijna-constant. Check `df[col].std()` en
`df[col].isna().mean()` na join, vóór training.

## Target & labels

**L6: P1/P2-heatmaps berekenen op de hele dataset = data leakage.**
`p1_probability` en `direction_bias` zijn historische statistieken over
(day_of_week, hour). Bereken op train-only slice met `train_cutoff =
df.index[-holdout_h].date()`. Geldt voor alle features die over de
trainset aggregeren — bevriezen vóór de split.

**L9: Target dead zone op 0.3% verwijdert 21.6% van trainrijen.** Voor
BTC 24h horizon: |move| < 0.3% komt ~22% voor. Dit is boven de
fee-breakeven (0.2% round-trip) — correct uitgesloten. Resultaat:
recall stijging van 0.24 → 0.40, accuracy 51% → 53%.

## Model-keuzes

**L4: Isotonic `CalibratedClassifierCV` overfit op kleine validatiesets.**
Met 1440 rijen reduceert isotonic alle probabilities naar
step-functie 0/1. Threshold gaat naar 0.72+ → model trade niet meer.
**Regel:** als calibratieset == threshold-optimalisatieset, sla
calibratie over of gebruik aparte split. Sigmoid is stabieler, maar bij
zeer kleine sets is geen calibratie beter dan slechte calibratie.

**L8: Walk-forward erft géén calibratie van `train_model()`.**
`run_walkforward()` maakt verse modellen per fold. Calibratie die in
`train_model()` is toegevoegd, propageert niet. Inconsistentie tussen
hoofdmodel en WF-rapport corrumpeert vergelijking.

**L11: Meer geregulariseerde LightGBM dwingt hogere thresholds → meer winstgevend.**
Voor 24h horizon met ~10k trainrijen: prefer `max_depth=4`,
`min_child_samples=80`. Hogere regularisatie → hogere drempel → minder
maar selectievere trades. Onderregulariseerd model overtrades en
verliest aan fees.

## Walk-forward & evaluatie

**L10: Test-periode (nov 2025 – feb 2026) was een bear market (-36.9% B&H).**
Alle modellen die alleen op deze periode geëvalueerd worden lijken
tegenvallen. Vergelijk altijd **relatief**: -2.2% strategie vs -36.9%
B&H = +34.7pp. WF folds met 0 trades in bear-regimes = correct gedrag,
geen fout.

**L15: Test-periode profitability ≠ walk-forward profitability als
regimes verschillen.** Test (nov-feb, pure bear): Sharpe +1.53. WF mean
(alle 2025-regimes): Sharpe -3.79. Single-regime test is geen
all-weather evaluatie. **Trust WF over single-run.**

**L14: Shorts helpen dramatisch in bears, schaden tijdens recoveries.**
WF fold 8 (jan-feb 2026, bear): Sharpe +14.1 met shorts. Fold 2 (mei-jun
2025, recovery): -13.3 met shorts. Conclusie: EMA200-filter alleen is
niet genoeg om duurzame bear van short-covering bounce te scheiden.
Hierom heeft de Sprint-14 `ranging_score` filter de waarde.

## Backtest-mechanica

**L5: B&H-benchmark moet 1-uur step returns gebruiken, niet h-period forward returns.**
`close.shift(-h) / close - 1` gevolgd door cumprod compoundeert 2160
overlappende returns → onzin. **Correct:** `close.pct_change().cumprod()`.

**L13: `np.arange` met float-step heeft floating-point endpoint
artifacts.** `np.arange(0.30, 0.46, 0.01)` kan 0.46 includen door
float-arithmetic. Gebruik `np.linspace(min, max, n_steps,
endpoint=False)` voor threshold-grids.

## Mislukte experimenten (don't repeat)

**Sprint 5 — HMM regime detection:** redundant met ADX market_regime,
+10.05 vs baseline +12.99 → uitgesloten.

**Sprint 15 — ADX-bonus + 4h-confluence drempelverlaging:**
verlaagt selectiviteit te agressief. Mediaan Sharpe +6.14 → +3.52.
21/31 folds slechter. Reden: fold27 ging van 0 trades (veilig +33.9
Sharpe) naar 1 verliezende trade (-23.8). Lagere drempel introduceert
marginale trades zonder edge.

**Sprint 18 — SHAP interactie-features als model-input:** mediaan
Sharpe 7.12 → 4.53. LightGBM vangt interacties al op via
boomsplitsingen; expliciete features voegen redundantie toe en
verleggen drempeloptimalisatie verkeerd. **Regel:** voor boom-modellen
zijn handmatig interactie-features bijna altijd schadelijk.

**Sprint 18 — WF per-fold isotonic calibratie:** fold4 -82.9, fold27
+99.6. Kleine val-set (~1152 rijen) maakt isotonic onstabiel. Vereist
≥ 5000× val-rijen of Platt scaling.

**Sprint 19-A1 — Model-driven exit (`exit_proba` sweep):** WF mediaan
+3.15 vs baseline +4.76. Whipsaw bij tijdelijke proba-dips midden in
valide trends → trade exit te vroeg. `MODEL_EXIT_ENABLED = False`. Code
blijft voor toekomstig onderzoek.

**Crash-mode tier 1 (>1σ daling, ×0.75):** schaalde normale
buy-the-dip entries af. Verwijderd; alleen tier 2+3 (echte crashes)
schalen positie nu.

## Algemene werkregels

- **Bewaar onderbouwing van reverts** — een geslaagd revert is een
  geleerde les, niet een verloren sprint. Documenteer in deze file zodat
  het experiment niet over een half jaar opnieuw geprobeerd wordt.
- **WF over single-run** — voor model-vergelijking is mediaan WF Sharpe
  een betrouwbaardere metric dan single-run Sharpe (die varieert
  sterk met de gekozen testperiode).
- **Ten minste 20 trades op de testset** voor model-selectie. Lager =
  toeval/noise. `MODEL_SELECT_MIN_TRADES = 20`.
