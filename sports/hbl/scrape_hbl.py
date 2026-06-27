"""
HBL match-scraper — slaat 5 seizoenen results op als data/hbl_matches.csv.
Statische BetExplorer results-pagina's. Idempotent (skip als CSV bestaat, tenzij --force).
Persoonlijk/laag-volume onderzoeksgebruik.
"""

import re
import sys
import time
import urllib.request
from pathlib import Path

import pandas as pd

SEASONS = ["2020-2021", "2021-2022", "2022-2023", "2023-2024", "2024-2025"]
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/124.0 Safari/537.36")
OUT = Path(__file__).parent / "data" / "hbl_matches.csv"


def fetch(url):
    req = urllib.request.Request(url, headers={"User-Agent": UA,
                                               "Accept-Language": "en-US,en;q=0.9"})
    return urllib.request.urlopen(req, timeout=30).read().decode("utf-8", "replace")


def parse(html, season):
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(html, "lxml")
    rows = []
    for tr in soup.select("tr"):
        a = tr.select_one("a.in-match")
        if not a:
            continue
        spans = a.select("span")
        if len(spans) < 2:
            continue
        txt = tr.get_text(" ", strip=True)
        sc = re.search(r"\b(\d+):(\d+)\b", txt)
        dt = re.search(r"(\d{2}\.\d{2}\.\d{4})", txt)
        mid = re.search(r"/([A-Za-z0-9]{8})/?$", a.get("href", ""))
        if not (sc and dt):
            continue
        rows.append(dict(season=season, date=dt.group(1),
                         home=spans[0].get_text(strip=True),
                         away=spans[1].get_text(strip=True),
                         gh=int(sc.group(1)), ga=int(sc.group(2)),
                         match_id=mid.group(1) if mid else ""))
    return rows


def main():
    if OUT.exists() and "--force" not in sys.argv:
        print(f"{OUT} bestaat al ({len(pd.read_csv(OUT))} rijen). --force om te herscrapen.")
        return
    base = "https://www.betexplorer.com/handball/germany/bundesliga-{}/results/"
    allrows = []
    for s in SEASONS:
        allrows += parse(fetch(base.format(s)), s)
        print(f"{s}: {len([r for r in allrows if r['season']==s])} matches")
        time.sleep(2)
    df = pd.DataFrame(allrows)
    df["date"] = pd.to_datetime(df["date"], format="%d.%m.%Y")
    df = df.sort_values("date").reset_index(drop=True)
    OUT.parent.mkdir(exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"\nOpgeslagen: {OUT}  ({len(df)} matches, {df['date'].min().date()} → {df['date'].max().date()})")


if __name__ == "__main__":
    main()
