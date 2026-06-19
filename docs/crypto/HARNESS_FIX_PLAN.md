# Harness-fix plan — eerlijke backtest vóór live execution

> Opgesteld 2026-06-19 na analyse van de live paper-trading-resultaten.
> **Aanleiding:** live paper trading (mrt–mei 2026) was negatief (hourly
> −6,5%, 4h −5,9%) terwijl de backtest "gem WF Sharpe +5,97" claimt. Een
> code-audit wees uit dat de backtest systematisch optimistischer is dan
> live én door leakage opgeblazen wordt. **Sprint 20 (Bybit live
> execution) staat op pauze tot de eerlijke baseline een echte
> out-of-sample edge laat zien.**

## Doel

Eén harness waarin backtest en live identiek rekenen, zonder look-ahead,
zodat een gemeten Sharpe te vertrouwen is. Pas daarna heeft het zin om
(a) te beslissen over live geld en (b) nieuwe features (bv. SMC/ICT:
market structure, liquidity, FVG, order blocks) te testen.

## Kernbevindingen die dit plan adresseert

| # | Bevinding | Bestand | Impact |
|---|---|---|---|
| A | Backtest rekent close-to-close, SL als floor-clip, **geen TP**, nooit intrabar high/low → realiseert intrabar-stopouts niet | `src/backtest.py` `run_backtest` (~270–322) | hoog |
| B | Live entry-pad evalueert 6 gates niet die de backtest wél heeft (DVOL/VIX/30d/USDJPY/put-call/funding) | `src/backtest.py` `generate_live_signal` (~1198) vs `run_backtest` (~110–224) | hoog |
| C | Hyperparams (`lgb_best_params.json`) getuned op exact de gerapporteerde WF-folds | `src/model.py` `auto_promote_optuna` (~455), `_quick_wf_sharpe` (~326) | hoog |
| D | Macro-gates (VIX/USDJPY/DXY) plakken dag-close op ochtendbars van dezelfde dag → deels helderziend | `src/external_data.py` (~202–224, ~1104) | hoog |
| E | P1-heatmap gebouwd op globale train-set, in elke fold-rij gebakken → toekomst in vroege folds | `main.py` (~116–126), `src/features.py` (~638) | hoog |
| F | Put/call-ratio: huidige snapshot teruggevuld over 30 dagen historie | `src/external_data.py` (~759–807) | midden |
| G | Sizing verschilt: backtest confidence-weighted (~0 op marginale trades); live volle fixed-risk notional | `src/backtest.py` (~291–322) vs `src/live_alert.py` (~392) | midden |
| H | "+5,97" = gemiddelde over 2 jaar, gedragen door enkele 2025-bull-folds; veel folds 0–7 trades; AUC 0,588 | `data/stats/walkforward_lightgbm.csv` | meet-artefact |

---

## Fase 0 — Goedkope falsificatie-checks vóór elke rebuild

> Aangescherpt 2026-06-19 na een Agent-Council-deliberatie. Kerninzicht:
> verwar "de backtest is ongeldig" niet met "het signaal is nul". De
> live-test is géén schone edge-meting (16+26 trades, uitgeschakelde
> gates, geen intrabar-stops). Beslis daarom met goedkope, falsifieerbare
> checks op **bestaande data** óf de dure Fasen 1–4 überhaupt zinvol zijn —
> elk met een **vooraf opgeschreven** go/no-go-drempel (niet achteraf
> herinterpreteren).

**Doel:** met een halve dag werk, zonder rebuild, vaststellen of er enig
restsignaal is dat een harness-verbouwing rechtvaardigt.

### Check 1 — Versloeg het model buy-and-hold over het live-raam? ✅ GEDAAN
- Drempel (vooraf): model ≥ B&H → zwak positief; model < B&H → negatief.
- **Resultaat (2026-06-19): GEFAALD.** Hourly −6,5% vs B&H **+9,7%**
  (−16,3 pp), 4h −5,9% vs B&H **+5,7%** (−11,6 pp). BTC **steeg** tijdens
  het handelsraam; het model verloor toch, met slechts 22% tijd-in-markt.
  → De "long-biased-in-een-daling"-alibi vervalt: het entry-signaal zelf
  is slecht (instappen op lokale toppen). Sterk no-go-signaal.

### Check 3 — Ruwe entry-timing zonder stops/gates ✅ GEDAAN (pure Python + Binance 1h klines)
- Doel: execution-bug scheiden van predictie-falen. Stops + gates VERWIJDERD,
  puur de forward-return van elke long-entry over zijn horizon gemeten.
- Drempel (vooraf): ruwe entries positief / ≥ random → plumbing-fix kan
  redden; ruwe entries negatief en < random → het signaal deugt niet.
- **Resultaat (2026-06-19): GEFAALD, beslissend.** Hourly ruwe forward-return
  **−1,55%/entry** (7% positief), 4h **−0,32%/entry**. Random-entry baseline
  in hetzelfde raam: **+0,22%** / **+0,095%**. Model-entries staan op
  **percentiel 0 resp. 3** van random — slechter dan vrijwel elke muntworp.
  Mét echte intrabar-stops: −1,02% / −0,33%. → **Geen execution-bug; het
  entry-signaal zelf is anti-predictief** (koopt lokale toppen). De
  random-baseline controleert voor het regime (random wérkte in dit raam).
  15/15 en 26/26 entries prijs-aligned met Binance → bron klopt.

### Check 2 — AUC op een schone hold-out (vereist OHLCV + retrain in devcontainer/CI)
- Drempel (vooraf): **≥ 0,55** op data die de hyperparams nooit raakten →
  signaal aanwezig. **< 0,52** → munt, stop.
- Status: **nog niet gedraaid** (vereist devcontainer/CI). Nu grotendeels
  bevestigend: Checks 1 en 3 tonen al onafhankelijk dat de out-of-sample
  entries slechter dan random zijn. Draai alleen als je een sluitende
  AUC-nagel wil; de go/no-go is feitelijk al beslist.

### Aggregaat correct herberekenen (bij Check 2/3)
- Pool alle trades over alle folds en bereken één Sharpe/PF/winrate over de
  gepoolde trades — niet het gemiddelde van per-fold-Sharpes (dat overweegt
  folds met 4 trades even zwaar als folds met 100; daar komt de "+5,97"
  vandaan).

**Beslisregel Fase 0:** alleen door naar Fasen 1–4 als Check 2 **of** Check 3
een fixbaar restsignaal laat zien. Anders → stop het BTC-model als
geldverdiener (hobby/leren mag door), en richt een eventuele eerlijke
harness op een vers domein (SMC-features of sports), niet op BTC-redding.
**Sprint 20 blijft geannuleerd, ongeacht de checks.**

> **UITKOMST (2026-06-19):** Checks 1 en 3 beide gefaald — en Check 3 toont
> dat het géén execution-bug is maar een anti-predictief entry-signaal
> (slechter dan random). Per de beslisregel: **Fasen 1–4 NIET uitvoeren als
> BTC-reddingsmissie.** BTC-model afsluiten als geldverdiener. Als je een
> eerlijke harness bouwt, doe het als standalone gereedschap gericht op een
> vers domein. Check 2 alleen nog voor een sluitende AUC-nagel; niet nodig
> voor het besluit.

## Fase 1 — Execution-engine samenvoegen (bevinding A, G)

**Doel:** backtest en live gebruiken dezelfde trade-simulatie.

- Til de intrabar-loop uit `src/simulation.py` (SL-first, dan TP, high/low
  per candle) tot de enige bron van waarheid.
- Laat `run_backtest` per trade die loop aanroepen i.p.v. close-to-close
  clip. Voeg TP toe. Houd fees (`2*TRADE_FEE`) en voeg een
  **slippage-parameter** toe (bv. 1–2 bp per fill).
- Lijn sizing uit: kies één model (advies: fixed-risk zoals live, want dat
  is wat je echt zou traden) en gebruik het in beide paden.
- Acceptatie: gegeven dezelfde input-bars produceren backtest en live
  een **identieke trade** (entry, exit-prijs, exit-reden, pnl). Schrijf
  hier een kleine regressietest voor.

## Fase 2 — Entry-gates samenvoegen (bevinding B)

**Doel:** één gate-functie die zowel backtest als live aanroept.

- Extraheer alle long/short-gates naar één functie
  `evaluate_gates(row) -> (long_ok, short_ok, reasons)`.
- Beide paden (`run_backtest`, `generate_live_signal`) roepen exact die
  functie aan. Verschil opheffen is een **bewuste keuze**: een gate die
  je niet live kunt evalueren (zie Fase 3) hoort ook niet in de backtest.
- Acceptatie: geen gate bestaat in maar één pad. Diff-test: voor een
  set rijen geven beide paden dezelfde gate-uitkomst.

## Fase 3 — Leakage dichten (bevindingen D, E, F)

**Doel:** geen enkele feature/gate gebruikt data die op beslismoment niet
bestond.

- **D — macro-alignment:** lag dagelijkse externe series (VIX/DXY/USDJPY/
  F&G/on-chain) met 1 dag, of gebruik expliciet de **vorige** dag-close.
  `merge_asof` met `direction="backward"` en een publicatie-lag.
- **E — P1-heatmap per fold:** bouw de heatmap binnen elke WF-fold alleen
  uit train-data (niet één globale heatmap voor de hele dataset).
- **F — put/call:** stop met de huidige snapshot 30 dagen terugvullen;
  laat historische waarden leeg of gebruik echte historische data.
- Acceptatie: voor elke feature geldt dat z'n waarde op tijdstip *t*
  alleen afhangt van data ≤ *t* (incl. realistische publicatie-lag).

## Fase 4 — Hyperparams ontkoppelen van de evaluatie (bevinding C)

**Doel:** de gerapporteerde WF mag niet het raam zijn waarop getuned is.

- Kies hyperparams op een **apart, ouder holdout-raam**; bevries ze.
- Rapporteer de WF over een raam dat **ná** het tuning-raam ligt en niet
  opnieuw getuned wordt.
- Acceptatie: `lgb_best_params.json` is niet geselecteerd m.b.v. de folds
  die in het eindrapport staan.

## Fase 5 — Eerlijke baseline meten (beslispunt)

**Doel:** de waarheid.

- Draai de volledige WF op de gefixte harness (Fasen 1–4).
- Rapporteer per fold mét trade-count, plus het gepoolde aggregaat uit
  Fase 0.
- **Beslissing:**
  - Edge duidelijk positief out-of-sample (na kosten/slippage) → door
    naar Fase 6, en Sprint 20 mag heroverwogen worden ná ≥ 4 weken
    paper op de eerlijke harness.
  - Edge ≈ 0 of negatief → dat is het echte signaal. Geen live geld.
    Terug naar de tekentafel (features/labels/horizon), niet naar execution.
- Acceptatie: één eerlijk, leakage-vrij, kosten-inclusief WF-rapport.

## Fase 6 — Pas hierna: nieuwe features (SMC/ICT) als research

**Alleen zinvol op een eerlijke harness.** SMC-strategieën hangen op
precieze SL-plaatsing op structuur — exact wat een close-to-close backtest
(vóór Fase 1) verkeerd meet.

- Voeg features toe (market structure / BOS-CHoCH, liquidity sweeps,
  FVG, order/breaker blocks, inducement) en meet **incrementeel** t.o.v.
  de Fase 5-baseline.
- Bijzondere hypothese om te toetsen: SMC-structuur als **long-bias-rem**
  (niet long onder bearish market structure / onder een breaker) — dat
  adresseert direct het waargenomen faalpatroon (model bleef long-en in
  een dalende markt).
- Acceptatie: elke feature wordt afgerekend op de eerlijke harness met
  een vooraf vastgelegde verbeterdrempel.

## Fase 7 — Sprint 20 (live execution) heropenen

Alleen ná Fase 5 met aangetoonde edge én ná de in de ROADMAP genoemde
sandbox-fase (≥ 2 weken), nu op de eerlijke harness gemeten.

---

## Volgorde & afhankelijkheden

```
Fase 0 (baseline)
   ├─ Fase 1 (execution)  ─┐
   ├─ Fase 2 (gates)      ─┤
   ├─ Fase 3 (leakage)    ─┤→ Fase 5 (eerlijke baseline) → beslispunt
   └─ Fase 4 (hyperparams)─┘                                  │
                                              Fase 6 (SMC) ────┤
                                              Fase 7 (Sprint 20)┘
```

Fasen 1–4 zijn grotendeels onafhankelijk; doe ze in volgorde van impact
(advies: **3-D macro-lag** eerst — goedkoop en hoge leakage-impact — dan
**1 execution**, dan **4 hyperparams**, dan **2 gates**, dan **3-E/F**).
Fase 5 is de barrière: pas daarna 6 en 7.
