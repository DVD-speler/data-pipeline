# Fase 6 — Paper Trading Analyse

## Doel
Na voldoende live paper trades (minimaal 30–50) de live prestaties vergelijken met de backtest
en beslissen of het model klaar is voor echte trading (of bijgestuurd moet worden).

## Wanneer starten?
- **Minimaal 30 gesloten trades** (bij ~1 signaal per 1–2 dagen = ca. 5–8 weken)
- Controleer tussentijds via `data/paper_trades.json` in de GitHub repo

---

## Stap 1 — Data laden en basisstatistieken

```python
import json, pandas as pd, numpy as np

with open("data/paper_trades.json") as f:
    state = json.load(f)

trades = pd.DataFrame(state["closed_trades"])
trades["open_time"]  = pd.to_datetime(trades["open_time"])
trades["exit_time"]  = pd.to_datetime(trades["exit_time"])
trades["proba"]      = trades["proba"].str.rstrip("%").astype(float)

print(f"Totaal trades      : {len(trades)}")
print(f"Winst trades       : {(trades['pnl_euro'] > 0).sum()}")
print(f"Winrate            : {(trades['pnl_euro'] > 0).mean():.1%}")
print(f"Gem. rendement     : {trades['gross_return_pct'].mean():.2f}%")
print(f"Gem. P&L           : €{trades['pnl_euro'].mean():.2f}")
print(f"Totale P&L         : €{trades['pnl_euro'].sum():.2f}")
print(f"Eindkapitaal       : €{state['capital']:.2f}")
print(f"Totaal rendement   : {(state['capital'] / 1000 - 1) * 100:.1f}%")
```

---

## Stap 2 — Sharpe ratio (live)

```python
# Dagelijks rendement op basis van kapitaalgroei per trade
returns = trades["pnl_euro"] / trades["capital_after"].shift(1).fillna(1000)

sharpe = returns.mean() / returns.std() * np.sqrt(365)
print(f"Live Sharpe (annualized): {sharpe:.2f}")
# Vergelijk met backtest Sharpe +3.19
```

---

## Stap 3 — Exit-reden breakdown

```python
exit_counts = trades["exit_reason"].value_counts()
print(exit_counts)
# Verwacht: TP ✓ = ~73% (backtest winrate), SL ✗ = ~27%, 24h ⏰ = klein deel

# P&L per exit-reden
print(trades.groupby("exit_reason")["pnl_euro"].agg(["count", "mean", "sum"]))
```

---

## Stap 4 — LONG vs SHORT verdeling

```python
print(trades.groupby("direction")[["gross_return_pct", "pnl_euro"]].agg(["count", "mean", "sum"]))
# Check: shorts presteren vergelijkbaar met backtest? (backtest: longs beter dan shorts in bull)
```

---

## Stap 5 — Proba-kalibratie check (na 50+ trades)

```python
# Splits in hoog/laag proba en kijk of winkans overeenkomt
trades["proba_bucket"] = pd.cut(trades["proba"], bins=[60, 65, 70, 75, 80, 100], labels=["60-65", "65-70", "70-75", "75-80", "80+"])
calibratie = trades.groupby("proba_bucket").apply(
    lambda g: pd.Series({
        "n": len(g),
        "winrate": (g["pnl_euro"] > 0).mean(),
        "gem_rendement": g["gross_return_pct"].mean(),
    })
)
print(calibratie)
# Als hogere proba-buckets echt hogere winrate geven → variabel position sizing overwegen
```

---

## Stap 6 — Equity curve plotten

```python
import matplotlib.pyplot as plt

equity = trades["capital_after"].tolist()
equity = [1000] + equity  # begin kapitaal

plt.figure(figsize=(12, 4))
plt.plot(equity, marker="o", markersize=3)
plt.axhline(1000, color="gray", linestyle="--", label="Start €1000")
plt.title("Paper Trading Equity Curve")
plt.ylabel("Kapitaal (€)")
plt.xlabel("Trade #")
plt.legend()
plt.tight_layout()
plt.savefig("data/stats/equity_curve_live.png", dpi=150)
plt.show()
```

---

## Stap 7 — Vergelijking backtest vs live

| Metric             | Backtest (test set) | Live (paper) | Verschil |
|--------------------|---------------------|--------------|---------|
| Winrate            | 73.3%               | ?            |         |
| Gem. rendement     | +1.8%               | ?            |         |
| Sharpe             | +3.19               | ?            |         |
| Totaal rendement   | +85.3%              | ?            |         |

**Acceptabel als live resultaten zijn:**
- Winrate > 55% (backtest 73% verwacht, maar live altijd lager door slippage/timing)
- Sharpe > 0.5 (backtest +3.19 — live zal lager zijn, dat is normaal)
- Gem. rendement > 0% (positief verwacht rendement)

---

## Stap 8 — Beslissing: doorgaan of bijsturen?

**Groen licht voor echte trading als:**
- Live winrate > 55%
- Live Sharpe > 0.5
- Minimaal 50 trades (statistisch significant genoeg)
- Geen grote afwijking van backtest-verdeling (geen aanwijzing voor data leakage of overfitting)

**Bijsturen als:**
- Winrate < 50% → threshold verhogen (longere drempel = minder maar betere signalen)
- Te veel SL-exits → sl_pct vergroten (van 2% naar 3%) of positiegrootte verkleinen
- Shorts verliezen structureel → short-filter aanscherpen of shorts uitschakelen

**Stop als:**
- Kapitaal daalt > 20% (€1000 → < €800) → model werkt niet live, volledige review nodig

---

## Implementatie in main.py (optioneel, later)

Toevoegen als aparte fase:
```
python main.py --phase paper_analyse
```

Genereert een rapport in `data/stats/paper_trading_report.txt` met alle bovenstaande statistieken
plus een equity curve PNG.

---

## Benodigdheden
- `data/paper_trades.json` met minimaal 30 gesloten trades
- Geen extra libraries nodig (pandas, numpy, matplotlib al aanwezig)
