# Code Review — crypto/ pipeline (2026-07-14)

> Bron: geautomatiseerde code-review (Claude Code) van `crypto/` + `shared/`.
> Top-bevindingen zijn handmatig geverifieerd tegen de code.
> **Status 2026-07-14:** 11 van 18 bevindingen gefixt op branch
> `fix/code-review-2026-07` (H1, H2, M1, M3, M4, L1, L2, L3, L4, L8, L9).
> Open: M2, M5, M6, M7, L5, L6, L7 — zie "Bewust open" onderaan.

## Algemene gezondheid
Volwassen, zorgvuldige codebase. Sterk: UTC-tijdzones consistent, chronologische
train/val/test-split, look-ahead bewust vermeden, API-fouten degraderen netjes naar
neutrale defaults. Zwakke plekken: dode feature-berekening in het hot path, één
overgeslagen bugfix (1h→4h/daily), en enkele erg lange functies.

## 🔴 HIGH
- [x] **H1 — 4h/daily live-signaal op verouderde candle** *(GEFIXT)*
  `crypto/src/features_4h.py:179` (en `features_daily.py:195`) deed kale
  `df[available].dropna()` → nieuwste candle (lege `future_close`) viel weg →
  live-signaal was 12h (4h) resp. 1 dag (daily) oud. Het 1h-pad loste dit op met
  `keep_unlabeled=True` (`features.py`); nooit doorgezet.
  **Fix:** `keep_unlabeled`-param toegevoegd aan `build_features_4h`/`_daily`
  (sentinel -1, zelfde conventie als 1h) en doorgegeven vanuit beide live-alerts.
  Trainingspad (default False) byte-identiek gebleven. Patroon gevalideerd met
  synthetische data (train: oude gedrag; inference: actuele candle behouden).
- [x] **H2 — Dode feature-berekening in hot path** *(GEFIXT)*
  `crypto/src/features.py` berekende `_add_hmm_regime`, `_add_halving_cycle`,
  `_add_supertrend`, `_add_btc_eth_correlation`, `_add_candle_patterns` — kolommen
  stonden niet in `FEATURE_COLS` (verwijderd na Sprint 2/5-regressies, zie
  config.py-commentaar) en werden direct weggegooid. Supertrend liep per rij
  (~44k), ETH-historie werd 2× geladen. **Fix:** 5 aanroepen + functies verwijderd
  (~260 regels); grep bevestigt nul resterende verwijzingen.

## 🟠 MEDIUM
- [x] **M1 — `threshold` werd een tuple** *(GEFIXT)*
  `crypto/src/backtest.py:61`: `load_optimal_threshold()` geeft `(long, short)`.
  **Fix:** `threshold, _ = load_optimal_threshold()`.
- [ ] **M2 — `download_all_external(force=True)` wist cache vóór download** *(OPEN)*
  `crypto/src/external_data.py:1140-1141`: pre-`unlink()` verslaat de bescherming
  van `_save_cache`; tijdelijke API-fout wist historie permanent.
  **Waarom open:** raakt force-refresh-semantiek van alle fetchers; verdient eigen
  ontwerp (temp-file → rename na succes) + test met live API's.
- [x] **M3 — Onbegrensde retry-loop bij niet-451 API-fouten** *(GEFIXT)*
  `crypto/src/data_fetcher.py`: `continue` + 10s sleep zonder cap.
  **Fix:** max 5 pogingen, teller reset na succesvolle fetch, daarna RuntimeError.
- [x] **M4 — `horizon_scan` verzon `target=0` voor laatste `h` rijen** *(GEFIXT)*
  `crypto/src/horizon_scan.py:64`: `(NaN > x)=False → 0`.
  **Fix:** target NaN houden waar future ontbreekt (`.where(future.notna())`),
  tail valt nu in de bestaande `dropna`, daarna cast naar int.
- [ ] **M5 — Dubbel-identieke feature (daily)** *(OPEN — vereist hertraining)*
  `crypto/src/features_daily.py:45-46`: `volume_ratio_30d` == `volume_spike_30d`,
  beide in `FEATURE_COLS_DAILY`. **Waarom open:** het opgeslagen dagmodel is op
  beide kolommen getraind; formule wijzigen zonder hertrainen = andere inputs bij
  live inference. Oppakken bij de volgende dagmodel-hertraining.
- [ ] **M6 — Overgrote/diep-geneste functies** *(OPEN — aparte refactor-PR)*
  `run_backtest` (~375 r.), `run_backtest_be_trail` (~217), `run_live_alert`
  (~380), `build_features` (~388).
- [ ] **M7 — Drievoudig gedupliceerde live-alert-logica** *(OPEN — aparte refactor-PR)*
  `check_position_exit`/`_close_position` bijna identiek in 1h/4h/daily alerts →
  kandidaat voor `shared/`.

## 🟢 LOW
- [x] **L1 — Dubbele config-restore** `crypto/main.py` *(GEFIXT — duplicaat weg)*
- [x] **L2 — Ongebruikte ensemble-vars** `model_compare.py` *(GEFIXT — `ens_short_thr`/`_orig` weg; `ens_long_thr` blijft, die wordt gebruikt)*
- [x] **L3 — Dode look-ahead-local** `features.py` `chikou = close.shift(-26)` *(GEFIXT)*
- [x] **L4 — `datetime.utcnow()` deprecated** `model.py` *(GEFIXT — `datetime.now(timezone.utc)`)*
- [ ] **L5 — Uitgeschakeld duur pad** `backtest.py` model-exit O(signals×h) loop *(OPEN — pas relevant bij re-enable; dan vectoriseren)*
- [ ] **L6 — Fragiele substring-matching** `external_data.py:1094` *(OPEN — meenemen met M2)*
- [ ] **L7 — ETH-restanten** config/guards voor tweede symbool *(OPEN — bewuste keuze nodig of multi-symbol scaffolding blijft)*
- [x] **L8 — `GEEN MODEL` dict miste keys** *(GEFIXT in 4h én daily — alle downstream-keys met veilige defaults)*
- [x] **L9 — Brede `except: pass` verborg gate-fouten** `backtest.py` *(GEFIXT — waarschuwing wordt geprint)*

## Bewust open (met reden)
| # | Reden | Wanneer oppakken |
|---|-------|------------------|
| M2 (+L6) | raakt force-refresh-semantiek alle fetchers; eigen ontwerp + live-API-test | aparte PR |
| M5 | vereist hertraining dagmodel (model is op beide kolommen getraind) | volgende hertraining |
| M6, M7 | grote refactors; risico concentreren in eigen PR met regressie-backtest | aparte PR |
| L5 | code staat uit (`MODEL_EXIT_ENABLED=False`) | bij re-enable |
| L7 | productkeuze: multi-symbol scaffolding houden of niet | bespreken |

## Validatie uitgevoerd
- `py_compile` op alle 11 gewijzigde bestanden: OK.
- H1-opschoonpatroon gevalideerd met synthetische data: trainingsmodus identiek
  aan oud gedrag; inference-modus behoudt actuele candle met sentinel -1.
- Grep-checks: geen verwijzingen meer naar verwijderde functies; H2-kolommen
  kwamen alleen voor in `features.py` (berekening) en config-commentaar.
- **Aanbevolen vóór merge:** één handmatige run `python crypto/main.py --phase
  live_alert_4h` (of de CI-workflow op de branch) om het 4h-signaal end-to-end
  te zien draaien met de actuele candle.
