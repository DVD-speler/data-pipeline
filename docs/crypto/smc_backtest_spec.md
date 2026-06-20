# SMC event-driven backtest — pre-registratie (v1)

> Vastgelegd 2026-06-19, vóór het zien van de holdout. Dit is de afspraak die
> "doorontwikkelen" eerlijk houdt (Agent-Council-discipline). Drempels en
> criteria worden **bevroren** voordat de holdout wordt aangeraakt. Wijzig
> hieronder vrij — maar NU, niet nadat je holdout-resultaten hebt gezien.

## Hypothese
Een event-driven SMC/ICT-setup (HTF-bias + engineered liquidity sweep + MSB +
valide inducement + FVG/OB-zone + schone invalidatie) levert op BTC een
positieve out-of-sample expectancy ná kosten — i.t.t. de per-bar classifier,
die geen edge toonde (die test was bovendien mis-gespecificeerd voor een
event-driven systeem; zie HARNESS_FIX_PLAN.md).

## Scope v1
- **Instrument:** BTCUSDT.
- **Timeframes:** bias op **daily**, setups/executie op **4h** (framework: MSB
  op hoogste TF waar de structuur zichtbaar is, executie op lagere TF). 1h als
  latere verfijning.
- **Richting:** long én short (framework is symmetrisch).

## Data-split (de kern van de eerlijkheid)
- **Dev-set:** alles t/m **2025-06-30**. Hierop ontwikkelen/tunen we vrij.
- **Holdout (vergrendeld):** **2025-07-01 → heden** (~12 mnd). Wordt tijdens
  ontwikkeling NIET aangeraakt; alleen bekeken op vooraf bepaalde mijlpalen.
  Elke holdout-check bevriest de op dat moment geldende rules.

## Setup-definitie (bevroren drempels v1)
Volgt `smc_framework.md`, met concrete waarden:
- **Pivots:** swing high/low met lookback **N = 5** bars (causaal: pas gebruikt
  vanaf bar j+N).
- **Significante structuur:** pivot high gevolgd door lower-low (resp. low →
  higher-high). Retrospectief, maar pas *geacteerd* zodra bevestigd (≤ t).
- **MSB:** candle **opent én sluit** voorbij de significante swing (geen wick-
  only). De close-rule.
- **Liquidity sweep:** wick voorbij een eerdere swing, daarna close terug erbinnen
  (sweep-diepte ≥ **0,1 × ATR(14)** om ruis te filteren).
- **FVG:** 3-candle, matching kleur, gap (`L_t > H_{t-2}` bullish / `H_t < L_{t-2}`
  bearish). Alleen high-probability (alle 3 dezelfde kleur).
- **Order block:** laatste tegengestelde candle vóór de impuls die de MSB maakte.
- **Inducement (Ada: BEVRIEZEN, niet weglaten):** een front-run pivot tussen de
  zone en de huidige prijs, geldig **alleen** als er tussen die pivot en nu ≥ 1
  fake break of structure (close voorbij een *insignificante* high/low) zat.
  Dit is causaal te coderen; als het niet causaal blijkt, vervalt de setup.
- **HTF-bias:** daily market-structure-richting (laatste daily-BOS). Setup moet
  met de bias mee (Rule 1 uit het framework).

## Entry / stop / target
- **Entry:** limit op de zone (OB of FVG) ná de inducement-sweep.
- **Stop:** voorbij de geswepte extreme (schone invalidatie — verplicht; geen
  schone SL = geen trade).
- **Target:** eerstvolgende tegengestelde liquidity pool; minimaal **RR ≥ 2**
  anders skip.

## Evaluatie (op de eerlijke intrabar-harness)
- Simuleer elke trade intrabar (SL-first), met **fees + slippage** (zelfde
  engine als `simulation.py`).
- **Metric:** out-of-sample **expectancy (gem. R per trade) ná kosten**, plus
  profit factor, winrate, # trades, en vergelijking met buy-and-hold over de
  holdout. **GEEN per-bar AUC.**

## Vooraf vastgelegde beslisregels
- **Event-floor:** < **30 trades** op de holdout → **"inconclusive"** (de
  steekproef is te klein; geen verdict, geen kapitaal).
- **Succes (edge-kandidaat):** holdout-expectancy ≥ **+0,15 R** ná kosten
  **én** profit factor > **1,3** **én** ≥ 30 trades **én** verslaat buy-and-hold.
  → door naar verlengde paper-fase (≥ 4 weken), nog steeds geen echt geld.
- **Falen:** expectancy ≤ 0 of PF ≤ 1,0 met ≥ 30 trades → setup-variant
  verworpen (niet "verdiepen tot het werkt").
- **Anti-overfit:** holdout alleen op mijlpalen; per check de rules bevriezen;
  geen her-tuning ná het zien van holdout-cijfers binnen dezelfde variant.

## Iteratie-protocol (zo blijft "doorontwikkelen" eerlijk)
- Itereer vrij op de **dev-set**. Elke nieuwe variant = nieuwe hypothese.
- Raak de holdout pas aan als je een variant af hebt. Houd een logboek van hoe
  vaak de holdout is bekeken (elke blik kost statistische zuiverheid).
- Geen echt geld vóór: holdout-succes **én** ≥ 4 weken live paper die het
  bevestigt.

## Build-volgorde
1. Event-driven backtest-engine: setups detecteren → trades simuleren
   (intrabar SL/TP, kosten) → expectancy-rapport. Begin met de mechaniseerbare
   kern (structuur + sweep + MSB + FVG/OB).
2. Inducement-laag toevoegen (bevroren causale regel).
3. Holdout-check #1.

## Amendement v1.1 (2026-06-19, vóór holdout — frequentie-fix)
De v1 4h-kern gaf maar ~4 trades/jaar → holdout zou ~4 trades opleveren,
hopeloos onder de 30-floor (council-voorspelling, empirisch bevestigd). Om een
verdict mogelijk te maken, aangepast (holdout nog onaangeraakt):
- **Executie naar 1h, bias naar 4h** (i.p.v. 4h/daily). Minder "puur HTF",
  maar ~3-4× meer setups. Framework staat LTF-executie toe.
- **Poolen over BTC + ETH** — zelfde setup-definitie, ~2× steekproef.
- **Guards** tegen ontaarde trades: **max-risk 5%** (skip setup als entry→SL
  > 5%), **max-hold** (time-exit, mark-to-market in R, als SL/TP niet raakt).
- **Next-bar fill-conventie**: SL/TP-beheer vanaf de candle ná de fill
  (same-candle fill+stop-artefact verwijderd; winrate 7%→14% op dev).
- Holdout-split (datums) ongewijzigd; succescriteria ongewijzigd.

## v1 implementatie-noten (eerlijk: vereenvoudigingen in de kern)
De mechaniseerbare kern (build-stap 1) gebruikt bewust simpele proxies; deze
worden op de **dev-set** verfijnd (iteratie-protocol), niet op de holdout:
- **Significantie:** v1 gebruikt de *laatste bevestigde swing* als BOS-referentie
  (standaard-operationalisatie), niet de volledige retrospectieve lower-low/
  higher-high-regel. Verfijning = v1.1 op dev.
- **Target:** v1 gebruikt **vaste RR = 2,0** i.p.v. "volgende liquidity pool".
  Dit maakt de eerste test schoon interpreteerbaar (winnen de setups > ~33%
  ná kosten?). "Volgende pool"-target = latere verfijning.
- **Eén trade tegelijk** (geen overlappende posities) in v1.
- **Inducement** zit nog NIET in stap 1 (komt in stap 2).
