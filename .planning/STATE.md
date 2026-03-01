# Project State

## Huidige Positie
- **Actieve fase:** Fase 1 — Regime Detectie (klaar te starten)
- **Laatste run:** 2026-03-01
- **Git branch:** master

## Beslissingen

| Beslissing | Reden | Datum |
|-----------|-------|-------|
| Regime detectie vóór feature uitbreiding | Root cause van bull-fold falen is slechte short-filtering, niet feature gebrek | 2026-03-01 |
| ADX als regime indicator | Geeft trendsterkte zonder richting; gecombineerd met EMA slope geeft richting | 2026-03-01 |
| Optuna voor hyperparameter search | Hand-tuning bereikt grenzen; Optuna vindt globaal optimum op validatieset | 2026-03-01 |
| Week 1 = model; Week 2 = live pipeline | Model moet eerst betrouwbaar zijn voor live testing zinvol is | 2026-03-01 |

## Bekende Blockers
- Geen actieve blockers

## Metrics Baseline (2026-03-01)
- Walk-forward: 2/9 positieve folds, mean Sharpe -3.79
- Test AUC: 0.5296
- Test return: +16.1%, Sharpe +1.53 (bear-regime only)
- Long threshold: 0.74, Short threshold: 0.45
- Model: LightGBM (max_depth=4, min_child_samples=80)
