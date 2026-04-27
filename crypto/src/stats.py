"""
Fase 3 — Kansstatistieken & Heatmaps
Berekent P1/P2-kansen per uur en dag van de week, vergelijkbaar met
het Brighter Data dashboard. Slaat resultaten op als CSV + PNG.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import config

DOW_LABELS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


# ── Statistieken ──────────────────────────────────────────────────────────────

def compute_p1_heatmap(p1p2: pd.DataFrame) -> pd.DataFrame:
    """
    Kans dat P1 (eerste dagelijkse extreem) optreedt op een gegeven
    (dag van de week, uur van de dag) combinatie.

    Waarden zijn conditioneel per dag van de week:
    P(P1 op uur H | dag D) = aantal keer / totaal dagen op dag D
    """
    total_per_dow = p1p2.groupby("day_of_week").size()
    counts = (
        p1p2.groupby(["day_of_week", "p1_hour"])
        .size()
        .reset_index(name="count")
    )
    counts = counts.merge(total_per_dow.rename("total"), on="day_of_week")
    counts["probability"] = counts["count"] / counts["total"]

    pivot = counts.pivot_table(
        index="day_of_week", columns="p1_hour", values="probability", fill_value=0
    )
    pivot = pivot.reindex(columns=range(24), fill_value=0)
    pivot.index = [DOW_LABELS[i] for i in pivot.index]
    return pivot


def compute_p2_heatmap(p1p2: pd.DataFrame) -> pd.DataFrame:
    """
    Kans dat P2 (tweede dagelijkse extreem) optreedt op een gegeven
    (dag van de week, uur van de dag) combinatie.
    """
    total_per_dow = p1p2.groupby("day_of_week").size()
    counts = (
        p1p2.groupby(["day_of_week", "p2_hour"])
        .size()
        .reset_index(name="count")
    )
    counts = counts.merge(total_per_dow.rename("total"), on="day_of_week")
    counts["probability"] = counts["count"] / counts["total"]

    pivot = counts.pivot_table(
        index="day_of_week", columns="p2_hour", values="probability", fill_value=0
    )
    pivot = pivot.reindex(columns=range(24), fill_value=0)
    pivot.index = [DOW_LABELS[i] for i in pivot.index]
    return pivot


def compute_direction_bias(p1p2: pd.DataFrame) -> pd.DataFrame:
    """
    Richtingsbias per (dag, uur): fractie van de tijd dat P1 een LOW was.
    > 0.5 = vaker bullisch (dag loopt omhoog na P1)
    < 0.5 = vaker bearisch (dag loopt omlaag na P1)

    Geeft NaN terug voor combinaties met < 5 observaties (te weinig data).
    """
    p1p2 = p1p2.copy()
    p1p2["p1_is_low"] = (p1p2["p1_type"] == "low").astype(float)

    agg = p1p2.groupby(["day_of_week", "p1_hour"])["p1_is_low"].agg(
        mean="mean", count="count"
    )
    # Maskeer te weinig data
    agg.loc[agg["count"] < 5, "mean"] = np.nan

    pivot = agg["mean"].unstack(level=1)
    pivot = pivot.reindex(columns=range(24), fill_value=np.nan)
    pivot.index = [DOW_LABELS[i] for i in pivot.index]
    return pivot


def compute_hourly_return_profile(p1p2: pd.DataFrame, df_ohlcv: pd.DataFrame) -> pd.DataFrame:
    """
    Gemiddeld uurrendement per (dag van de week, uur van de dag).
    Nuttig om te zien welke sessies historisch stijgend/dalend zijn.
    """
    df = df_ohlcv.copy()
    df["hour"] = df.index.hour
    df["day_of_week"] = df.index.dayofweek
    df["hour_return"] = df["close"].pct_change()

    profile = df.groupby(["day_of_week", "hour"])["hour_return"].mean().unstack(level=1)
    profile = profile.reindex(columns=range(24), fill_value=np.nan)
    profile.index = [DOW_LABELS[i] for i in profile.index]
    return profile


# ── Visualisatie ──────────────────────────────────────────────────────────────

def _plot_heatmap(
    data: pd.DataFrame,
    title: str,
    save_path: Path,
    cmap: str = "RdYlGn",
    fmt: str = ".2f",
    vmin=None,
    vmax=None,
) -> None:
    fig, ax = plt.subplots(figsize=(20, 5))
    sns.heatmap(
        data,
        annot=True,
        fmt=fmt,
        cmap=cmap,
        linewidths=0.4,
        ax=ax,
        vmin=vmin,
        vmax=vmax,
        cbar_kws={"label": "Waarde"},
    )
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Uur van de dag (UTC)", labelpad=8)
    ax.set_ylabel("Dag van de week", labelpad=8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Opgeslagen: {save_path.name}")


# ── Hoofdfunctie ──────────────────────────────────────────────────────────────

def run_stats(p1p2: pd.DataFrame, df_ohlcv: pd.DataFrame = None):
    """
    Bereken alle statistieken, sla op als CSV en genereer PNG heatmaps.

    Returns
    -------
    tuple : (p1_heatmap, p2_heatmap, direction_bias)  — DataFrames voor feature engineering
    """
    out = config.DATA_DIR / "stats"
    out.mkdir(exist_ok=True)

    print("Statistieken berekenen...")

    p1_heatmap = compute_p1_heatmap(p1p2)
    p2_heatmap = compute_p2_heatmap(p1p2)
    direction_bias = compute_direction_bias(p1p2)

    # CSV exports
    p1_heatmap.to_csv(out / "p1_heatmap.csv")
    p2_heatmap.to_csv(out / "p2_heatmap.csv")
    direction_bias.to_csv(out / "direction_bias.csv")

    # Heatmap visualisaties
    _plot_heatmap(
        p1_heatmap,
        "P1 Kans — Eerste dagelijks extreem per uur & dag (conditioneel per dag)",
        out / "p1_heatmap.png",
        cmap="Blues",
        fmt=".2f",
    )
    _plot_heatmap(
        p2_heatmap,
        "P2 Kans — Tweede dagelijks extreem per uur & dag (conditioneel per dag)",
        out / "p2_heatmap.png",
        cmap="Purples",
        fmt=".2f",
    )
    _plot_heatmap(
        direction_bias,
        "Richtingsbias — P(P1=Low) per uur & dag  [Groen=Bullish >0.5, Rood=Bearish <0.5]",
        out / "direction_bias.png",
        cmap="RdYlGn",
        fmt=".2f",
        vmin=0.0,
        vmax=1.0,
    )

    if df_ohlcv is not None:
        hourly_profile = compute_hourly_return_profile(p1p2, df_ohlcv)
        hourly_profile.to_csv(out / "hourly_return_profile.csv")
        _plot_heatmap(
            hourly_profile * 100,  # naar procenten
            "Gemiddeld uur-rendement (%) per uur & dag",
            out / "hourly_return_profile.png",
            cmap="RdYlGn",
            fmt=".3f",
        )

    print(f"\nAlle statistieken opgeslagen in {out}/")
    return p1_heatmap, p2_heatmap, direction_bias


if __name__ == "__main__":
    from src.data_fetcher import load_ohlcv

    p1p2 = pd.read_csv(config.DATA_DIR / "p1p2_labels.csv")
    df_ohlcv = load_ohlcv()
    p1_heatmap, p2_heatmap, direction_bias = run_stats(p1p2, df_ohlcv)

    print("\n--- Top 5 meest voorkomende P1-uren (over alle dagen) ---")
    print(p1_heatmap.mean().sort_values(ascending=False).head(5))
