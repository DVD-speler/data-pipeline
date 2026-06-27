"""
HBL Stage 1 — skill-gate (geen odds nodig).

Walk-forward vergelijking op de 1604 matches:
  - BASE-RATE: voorspel de empirische [P(H),P(D),P(A)] uit verleden (skill-floor).
  - SKELLAM:   team attack/defense (rollend, causaal) -> λ_h, λ_a -> Skellam 1X2.
  - ELO:       Elo-rating -> win-kans, draw via empirische draw-rate (extra baseline).

Metric: multiclass log-loss + Brier + accuracy. Skellam heeft skill als z'n
log-loss duidelijk onder de base-rate ligt. Pas dán zijn odds (CLV) de moeite.
"""

import os
import sys
from collections import defaultdict, deque

import numpy as np
import pandas as pd
from scipy.stats import skellam

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "hbl_matches.csv")

WINDOW = 30          # rollende vorm-window per team
MIN_HIST = 10        # min. matches per team vóór evaluatie
ELO_K = 20
ELO_HA = 60          # thuisvoordeel in Elo-punten


def outcome(gh, ga):
    return 0 if gh > ga else (1 if gh == ga else 2)  # 0=H,1=D,2=A


def logloss(p, y):
    return -np.log(max(p[y], 1e-12))


def brier(p, y):
    t = np.zeros(3); t[y] = 1.0
    return float(np.sum((np.array(p) - t) ** 2))


def main():
    df = pd.read_csv(CSV, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    n = len(df)

    # state
    hist = defaultdict(lambda: deque(maxlen=WINDOW))  # team -> (scored, conceded)
    elo = defaultdict(lambda: 1500.0)
    cnt = defaultdict(int)
    n_h = n_d = n_a = 0          # base-rate tellers (verleden)
    sum_hg = sum_ag = 0.0        # voor league-avg + home-factor
    n_goals = 0

    results = {m: {"ll": [], "br": [], "acc": []} for m in ("base", "skellam", "elo")}

    for _, r in df.iterrows():
        h, a, gh, ga = r["home"], r["away"], int(r["gh"]), int(r["ga"])
        y = outcome(gh, ga)
        evaluable = cnt[h] >= MIN_HIST and cnt[a] >= MIN_HIST and (n_h + n_d + n_a) > 50

        if evaluable:
            tot = n_h + n_d + n_a
            p_base = [n_h / tot, n_d / tot, n_a / tot]

            lg = (sum_hg + sum_ag) / max(n_goals, 1)          # league avg goals/team/match
            hf = (sum_hg / max(sum_ag, 1e-9))                  # home/away goal-ratio
            def att(t):
                hh = hist[t]
                return (np.mean([s for s, _ in hh]) / lg) if hh else 1.0
            def dfn(t):
                hh = hist[t]
                return (np.mean([c for _, c in hh]) / lg) if hh else 1.0
            lam_h = lg * att(h) * dfn(a) * np.sqrt(hf)
            lam_a = lg * att(a) * dfn(h) / np.sqrt(hf)
            lam_h = float(np.clip(lam_h, 5, 60)); lam_a = float(np.clip(lam_a, 5, 60))
            pH = 1 - skellam.cdf(0, lam_h, lam_a)
            pD = skellam.pmf(0, lam_h, lam_a)
            pA = skellam.cdf(-1, lam_h, lam_a)
            s = pH + pD + pA
            p_skel = [pH / s, pD / s, pA / s]

            # Elo -> 1X2 (draw via empirische draw-rate)
            e_home = 1 / (1 + 10 ** (-(elo[h] + ELO_HA - elo[a]) / 400))
            draw_rate = n_d / tot
            pD_e = draw_rate
            pH_e = (1 - pD_e) * e_home
            pA_e = (1 - pD_e) * (1 - e_home)
            p_elo = [pH_e, pD_e, pA_e]

            for name, p in (("base", p_base), ("skellam", p_skel), ("elo", p_elo)):
                results[name]["ll"].append(logloss(p, y))
                results[name]["br"].append(brier(p, y))
                results[name]["acc"].append(int(np.argmax(p) == y))

        # update state ná predictie
        hist[h].append((gh, ga)); hist[a].append((ga, gh))
        cnt[h] += 1; cnt[a] += 1
        n_h += y == 0; n_d += y == 1; n_a += y == 2
        sum_hg += gh; sum_ag += ga; n_goals += 2
        exp_h = 1 / (1 + 10 ** (-(elo[h] + ELO_HA - elo[a]) / 400))
        sc_h = 1.0 if y == 0 else (0.5 if y == 1 else 0.0)
        elo[h] += ELO_K * (sc_h - exp_h)
        elo[a] += ELO_K * ((1 - sc_h) - (1 - exp_h))

    n_eval = len(results["base"]["ll"])
    print("=" * 60)
    print(f"HBL STAGE 1 — skill-gate ({n_eval} geëvalueerde matches van {n})")
    print(f"base rates verleden: H {n_h} / D {n_d} / A {n_a}  (draw-rate {n_d/(n_h+n_d+n_a)*100:.1f}%)")
    print("=" * 60)
    print(f"{'model':10} {'log-loss':>9} {'Brier':>8} {'accuracy':>9}")
    for name in ("base", "skellam", "elo"):
        ll = np.mean(results[name]["ll"]); br = np.mean(results[name]["br"])
        ac = np.mean(results[name]["acc"])
        print(f"{name:10} {ll:9.4f} {br:8.4f} {ac*100:8.1f}%")
    base_ll = np.mean(results["base"]["ll"])
    skel_ll = np.mean(results["skellam"]["ll"])
    print(f"\nSkellam vs base-rate log-loss: {skel_ll-base_ll:+.4f} "
          f"({'SKILL ✓' if skel_ll < base_ll - 0.005 else 'geen duidelijke skill'})")


if __name__ == "__main__":
    main()
