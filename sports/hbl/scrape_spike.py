"""
HBL scraper-spike (deel A) — scores + fixtures van BetExplorer results-pagina's.

Doel: hard bevestigen dat we 5 seizoenen HBL-uitslagen (met doelpunten, nodig
voor Skellam) kunnen ophalen. Statische HTML, robuust. Odds (deel B) volgt apart.

Persoonlijk/laag-volume onderzoeksgebruik; rate-limited. Niet voor productie/
hoog volume (ToS).
"""

import re
import time
import urllib.request

SEASONS = ["2020-2021", "2021-2022", "2022-2023", "2023-2024", "2024-2025"]
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/124.0 Safari/537.36")


def fetch(url):
    req = urllib.request.Request(url, headers={"User-Agent": UA,
                                               "Accept-Language": "en-US,en;q=0.9"})
    return urllib.request.urlopen(req, timeout=30).read().decode("utf-8", "replace")


def parse_results(html, season):
    """Parse match-rijen: (date, home, away, gh, ga, match_id)."""
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(html, "lxml")
    out = []
    for tr in soup.select("tr"):
        a = tr.select_one("a.in-match")
        if not a:
            continue
        spans = a.select("span")
        if len(spans) < 2:
            continue
        home, away = spans[0].get_text(strip=True), spans[1].get_text(strip=True)
        txt = tr.get_text(" ", strip=True)
        sc = re.search(r"\b(\d+):(\d+)\b", txt)
        dt = re.search(r"(\d{2}\.\d{2}\.\d{4})", txt)
        mid = re.search(r"/([A-Za-z0-9]{8})/?$", a.get("href", ""))
        if not (sc and dt):
            continue
        out.append((dt.group(1), home, away, int(sc.group(1)), int(sc.group(2)),
                    mid.group(1) if mid else ""))
    return out


def main():
    base = "https://www.betexplorer.com/handball/germany/bundesliga-{}/results/"
    total = 0
    for s in SEASONS:
        try:
            html = fetch(base.format(s))
            rows = parse_results(html, s)
        except Exception as e:
            print(f"{s}: FOUT {type(e).__name__}: {e}")
            continue
        total += len(rows)
        gemiddeld = (sum(r[3] + r[4] for r in rows) / len(rows)) if rows else 0
        print(f"{s}: {len(rows):3d} matches | gem. totaal doelpunten {gemiddeld:.1f}")
        for r in rows[:2]:
            print(f"    {r[0]}  {r[1]} {r[3]}-{r[4]} {r[2]}  [id {r[5]}]")
        time.sleep(2)  # beleefd
    print(f"\nTOTAAL: {total} matches over {len(SEASONS)} seizoenen "
          f"(verwacht ~306/seizoen = ~1530)")


if __name__ == "__main__":
    main()
