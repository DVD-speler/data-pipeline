"""
Kelly Criterion positiegrootte — gedeeld door crypto en (straks) sports.

Pure wiskunde, geen project-specifieke afhankelijkheden. De per-project
wrapper (in crypto: `save_kelly_sizing` in `crypto/src/model.py`) doet
het laden/opslaan van win-rate / avg_win / avg_loss op basis van
backtest-resultaten en roept `compute_kelly_fraction` aan voor de
feitelijke berekening.
"""


def compute_kelly_fraction(win_rate: float, avg_win: float, avg_loss: float) -> float:
    """
    Bereken de Kelly-fractie: optimale positiegrootte als fractie van kapitaal.

    f = (p × b − q) / b
    Waarbij: p = win_rate, q = 1 − p, b = avg_win / avg_loss (odds ratio)

    Geeft 0.0 terug als er onvoldoende data is of als de verwachte waarde
    negatief is — caller moet zelf besluiten wat te doen (bijv. fallback
    naar vaste positiegrootte).
    """
    if avg_loss <= 0 or win_rate <= 0 or win_rate >= 1:
        return 0.0
    b = avg_win / avg_loss
    q = 1.0 - win_rate
    kelly = (win_rate * b - q) / b
    return max(0.0, round(kelly, 4))
