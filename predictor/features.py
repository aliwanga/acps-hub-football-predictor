"""Chronological rating engine + leakage-free feature construction.

Every feature row for a match is computed from team state *before* that
match, then the state is updated with the result.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import pandas as pd

HOME_ADV = 100.0       # Elo points granted to the home side at non-neutral venues
START_ELO = 1500.0
AVG_GOALS = 1.35       # prior for per-team goals per game
EWMA_ALPHA = 0.08      # long-run smoothing of goals scored / conceded
FORM_WINDOW = 5
REST_CAP = 60


def tournament_k(tournament: str) -> float:
    """Elo K-factor: more important competitions move ratings more."""
    t = str(tournament).lower()
    if "friendly" in t:
        return 20.0
    if "qualif" in t:
        return 40.0
    if "world cup" in t:
        return 60.0
    if any(x in t for x in ("euro", "copa am", "african cup", "asian cup", "gold cup",
                            "nations cup", "confederations")):
        return 50.0
    if "nations league" in t:
        return 40.0
    return 30.0


def goal_multiplier(diff: int) -> float:
    d = abs(diff)
    if d <= 1:
        return 1.0
    if d == 2:
        return 1.5
    return (11.0 + d) / 8.0


@dataclass
class TeamState:
    elo: float = START_ELO
    gf: float = AVG_GOALS     # EWMA goals scored
    ga: float = AVG_GOALS     # EWMA goals conceded
    n: int = 0
    last_date: pd.Timestamp | None = None
    form: deque = field(default_factory=lambda: deque(maxlen=FORM_WINDOW))  # points (1/0.5/0)


class RatingEngine:
    def __init__(self) -> None:
        self.teams: dict[str, TeamState] = {}
        self.h2h: dict[tuple[str, str], deque] = {}

    # ------------------------------------------------------------------ helpers
    def state(self, team: str) -> TeamState:
        if team not in self.teams:
            self.teams[team] = TeamState()
        return self.teams[team]

    def known(self, team: str) -> bool:
        return team in self.teams

    @staticmethod
    def _key(a: str, b: str) -> tuple[str, str]:
        return (a, b) if a < b else (b, a)

    def _h2h_feature(self, home: str, away: str) -> tuple[float, int]:
        q = self.h2h.get(self._key(home, away))
        if not q:
            return 0.5, 0
        pts_first = float(np.mean(q))          # from the alphabetically-first team's view
        return (pts_first if home < away else 1.0 - pts_first), len(q)

    # ------------------------------------------------------------------ features
    def features(self, home: str, away: str, date, neutral: bool, tournament: str,
                 home_penalty: float = 0.0, away_penalty: float = 0.0) -> dict:
        """Features for a match given current state. Penalties are Elo points
        subtracted for injuries / absences."""
        h, a = self.state(home), self.state(away)
        date = pd.Timestamp(date)
        elo_h, elo_a = h.elo - home_penalty, a.elo - away_penalty
        h2h, h2h_n = self._h2h_feature(home, away)
        rest_h = min((date - h.last_date).days, REST_CAP) if h.last_date is not None else REST_CAP
        rest_a = min((date - a.last_date).days, REST_CAP) if a.last_date is not None else REST_CAP
        return {
            "elo_h": elo_h,
            "elo_a": elo_a,
            "elo_diff": elo_h - elo_a + (0.0 if neutral else HOME_ADV),
            "neutral": float(neutral),
            "form_h": float(np.mean(h.form)) if h.form else 0.5,
            "form_a": float(np.mean(a.form)) if a.form else 0.5,
            "gf_h": h.gf, "ga_h": h.ga, "gf_a": a.gf, "ga_a": a.ga,
            "n_h": h.n, "n_a": a.n,
            "rest_h": max(rest_h, 0), "rest_a": max(rest_a, 0),
            "h2h": h2h, "h2h_n": h2h_n,
            "importance": tournament_k(tournament),
        }

    # ------------------------------------------------------------------ update
    def update(self, home: str, away: str, hs: int, as_: int, date, neutral: bool,
               tournament: str) -> None:
        h, a = self.state(home), self.state(away)
        k = tournament_k(tournament) * goal_multiplier(hs - as_)
        dr = h.elo - a.elo + (0.0 if neutral else HOME_ADV)
        expected = 1.0 / (1.0 + 10 ** (-dr / 400.0))
        actual = 1.0 if hs > as_ else 0.5 if hs == as_ else 0.0
        delta = k * (actual - expected)
        h.elo += delta
        a.elo -= delta

        for st, gf, ga, pts in ((h, hs, as_, actual), (a, as_, hs, 1.0 - actual)):
            alpha = max(EWMA_ALPHA, 1.0 / (st.n + 2))   # fast start, then smooth
            st.gf = (1 - alpha) * st.gf + alpha * gf
            st.ga = (1 - alpha) * st.ga + alpha * ga
            st.n += 1
            st.last_date = pd.Timestamp(date)
            st.form.append(pts)

        key = self._key(home, away)
        self.h2h.setdefault(key, deque(maxlen=5)).append(actual if home < away else 1.0 - actual)


def build_table(matches: pd.DataFrame) -> tuple[pd.DataFrame, RatingEngine]:
    """Walk matches in date order; return feature table and the final engine."""
    eng = RatingEngine()
    rows = []
    for m in matches.itertuples(index=False):
        f = eng.features(m.home_team, m.away_team, m.date, m.neutral, m.tournament)
        f.update(date=m.date, home_team=m.home_team, away_team=m.away_team,
                 hs=m.home_score, as_=m.away_score,
                 y=0 if m.home_score > m.away_score else 1 if m.home_score == m.away_score else 2)
        rows.append(f)
        eng.update(m.home_team, m.away_team, m.home_score, m.away_score, m.date,
                   m.neutral, m.tournament)
    return pd.DataFrame(rows), eng
