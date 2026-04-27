"""
Fase 2 — P1/P2 Engine
Berekent per kalenderdag (UTC) het eerste en tweede dagelijks extreem.

Definities:
  P1 = het eerste extreem (high of low) van de dag, uitgedrukt als uur (0-23)
  P2 = het tweede extreem (het tegenovergestelde), uitgedrukt als uur (0-23)

Voorbeeld:
  Dag begint omlaag → P1=low op 03:00, P2=high op 14:00
  Dag begint omhoog → P1=high op 09:00, P2=low op 20:00
"""

import pandas as pd

import config
from src.data_fetcher import load_ohlcv


def compute_p1p2(df: pd.DataFrame) -> pd.DataFrame:
    """
    Verwerk een OHLCV DataFrame (1h candles, UTC-index) naar een
    dag-voor-dag P1/P2 label DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Output van load_ohlcv() — UTC-geïndexeerd, kolommen incl. high/low/open/close.

    Returns
    -------
    pd.DataFrame
        Eén rij per complete dag met kolommen:
        date, day_of_week, high_hour, low_hour,
        p1_hour, p1_type, p2_hour, p2_type,
        daily_high, daily_low, open_price, close_price, day_return
    """
    df = df.copy()
    df["_date"] = df.index.date
    df["_hour"] = df.index.hour

    results = []

    for date, group in df.groupby("_date"):
        # Sla onvolledige dagen over (bv. vandaag of een dag met datafout)
        if len(group) < config.MIN_HOURS_PER_DAY:
            continue

        daily_high = group["high"].max()
        daily_low = group["low"].min()

        # Eerste uur waarop het dagelijkse high/low bereikt werd
        high_hour = int(group.loc[group["high"] == daily_high, "_hour"].iloc[0])
        low_hour = int(group.loc[group["low"] == daily_low, "_hour"].iloc[0])

        # Edge case: high en low in hetzelfde uur → sla over
        if high_hour == low_hour:
            continue

        # P1 = wat het eerst optrad, P2 = wat daarna optrad
        if high_hour < low_hour:
            p1_hour, p1_type = high_hour, "high"
            p2_hour, p2_type = low_hour, "low"
        else:
            p1_hour, p1_type = low_hour, "low"
            p2_hour, p2_type = high_hour, "high"

        open_price = float(group.iloc[0]["open"])
        close_price = float(group.iloc[-1]["close"])
        day_return = (close_price - open_price) / open_price

        results.append(
            {
                "date": date,
                "day_of_week": int(group.index.dayofweek[0]),  # 0=ma, 6=zo
                "high_hour": high_hour,
                "low_hour": low_hour,
                "p1_hour": p1_hour,
                "p1_type": p1_type,
                "p2_hour": p2_hour,
                "p2_type": p2_type,
                "daily_high": daily_high,
                "daily_low": daily_low,
                "open_price": open_price,
                "close_price": close_price,
                "day_return": day_return,
            }
        )

    result_df = pd.DataFrame(results)
    print(
        f"P1/P2 berekend voor {len(result_df)} complete dagen "
        f"({len(df.groupby('_date')) - len(result_df)} dagen overgeslagen)"
    )
    return result_df


if __name__ == "__main__":
    df = load_ohlcv()
    p1p2 = compute_p1p2(df)

    out_path = config.DATA_DIR / "p1p2_labels.csv"
    p1p2.to_csv(out_path, index=False)
    print(f"Opgeslagen: {out_path}")
    print(p1p2.head(10).to_string())

    # Snelle sanity check
    print("\n--- P1 type verdeling ---")
    print(p1p2["p1_type"].value_counts())
    print("\n--- Gemiddeld dag-rendement ---")
    print(f"  P1=low  (bullish dag): {p1p2[p1p2['p1_type']=='low']['day_return'].mean():.2%}")
    print(f"  P1=high (bearish dag): {p1p2[p1p2['p1_type']=='high']['day_return'].mean():.2%}")
