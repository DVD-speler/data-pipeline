# Sports Lessons

Geleerde lessen + mislukte experimenten in het sports-domein. Tegen-
voorbeelden zijn even waardevol als positieve bevindingen — ze
voorkomen herhaling. Format analoog aan [`docs/crypto/LESSONS.md`](../crypto/LESSONS.md).

## Data-source onderzoek

### L1 — Live-verifieer iedere data-source vóór roadmap-vastlegging

**Datum**: 2026-04-28
**Sprint**: S1 (pre-implementatie)

**Misstap**: OpenLigaDB werd in de S1+ ROADMAP-update ([commit `e5544c6c`](
https://github.com/DVD-speler/data-pipeline/commit/e5544c6c)) gepland
als **primaire bron** voor HBL match-data, met een endpoint-pattern
`https://api.openligadb.de/getmatchdata/hbl/{season}` voor seizoenen
2020 t/m 2025. Aanname op papier: "gratis, geen key, JSON REST, stabiel
sinds 2010" — geen pre-flight-check uitgevoerd.

**Realiteit (geverifieerd 2026-04-28 via directe API-calls)**:

| Seizoen | Beschikbaar via `getmatchdata/hbl/{year}` |
|---|---|
| 2010 (`hbl1011`) | ✅ |
| 2011–2016 | ✅ |
| 2017–2022 | ❌ leeg `[]` |
| 2023 | ✅ (los, geen 2024-onwards) |
| 2024–2025 | ❌ leeg `[]` |

Alternatieve shortcuts (`hbl1`, `1hbl`, `dhb`, `lmh`, `hbf1`, `hb1`,
`handball`) eveneens leeg. Officiële `getavailableleagues`-listing toont
geen Bundesliga-handbal entries na 2016/2017.

**Gevolg**: S1a-implementatie (data-fetcher tegen OpenLigaDB) zou na ~30
min kosten 1 van 5 doelseizoenen opleveren. Bug pas zichtbaar bij eerste
download-run, niet bij code-review.

**Regel**: vóór elke nieuwe data-source in een sport-roadmap:

1. **Eén directe API-call** per beoogd seizoen / datapunt — niet aannemen
   wat het overzicht-endpoint suggereert
2. **Bytes-grootte van response checken** (`curl … | wc -c`) — `[]` =
   2 bytes is een silent-empty die niet als error oplevert
3. **Resultaat documenteren in de ROADMAP** vóór de bron als gekozen wordt
   gemarkeerd

Geldt analoog voor crypto's `data/external/*` bronnen — daar is dit
historisch al toegepast, zie de uitsluitingen in [`docs/crypto/LESSONS.md`](
../crypto/LESSONS.md) van btc_put_call_ratio (98% null), btc_dominance
(0% importance), usdt_dominance (25% datadekking).

**Mitigatie nu**: ROADMAP S1-sectie herschreven met expliciete
waarschuwing en open data-source-keuze (Liquimoly HBL official /
handball.net app-API / Sofascore / Sportmonks). Decision-point
verschoven naar laptop-sessie waar Dennis live kan inspecteren.
