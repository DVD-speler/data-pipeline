# sports/

**Status: scaffold komt in S0 (Sprint 0 van het sports-model).**

Geplande tegenhanger van `crypto/`: kansstatistiek-gebaseerd model +
signaal-pijplijn voor sport-events. Volgt vermoedelijk dezelfde
fase-structuur (data → features → model → signaal → live alert) maar
met sport-specifieke databronnen en horizons.

In de tussentijd is deze map een placeholder zodat:

- De top-level structuur stabiel is voordat er code in komt
- `shared/` API's getest kunnen worden met crypto als enige consumer
- De workflows-naming-conventie (`crypto_*.yml`, straks `sports_*.yml`)
  vooraf vaststaat
