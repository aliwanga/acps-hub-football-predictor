"""Loading match data.

Canonical match table columns:
    date, home_team, away_team, home_score, away_score, tournament, neutral

`data/matches.csv` is produced by `fetch_data.py` (real data). If it does not
exist, a clearly-labelled synthetic DEMO dataset with fictional teams is
generated so the whole pipeline can be exercised offline.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MATCHES_CSV = DATA_DIR / "matches.csv"

COLUMNS = ["date", "home_team", "away_team", "home_score", "away_score", "tournament", "neutral"]


def normalise(df: pd.DataFrame) -> pd.DataFrame:
    df = df[COLUMNS].copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["home_score"] = pd.to_numeric(df["home_score"], errors="coerce")
    df["away_score"] = pd.to_numeric(df["away_score"], errors="coerce")
    df["neutral"] = df["neutral"].astype(str).str.upper().isin(["TRUE", "1", "YES"])
    df["tournament"] = df["tournament"].fillna("").astype(str)
    df = df.dropna(subset=["date", "home_team", "away_team", "home_score", "away_score"])
    df = df[df["home_team"] != df["away_team"]]
    df["home_score"] = df["home_score"].astype(int)
    df["away_score"] = df["away_score"].astype(int)
    return df.sort_values("date", kind="stable").reset_index(drop=True)


def load_matches() -> tuple[pd.DataFrame, bool]:
    """Return (matches, is_demo)."""
    if MATCHES_CSV.exists():
        return normalise(pd.read_csv(MATCHES_CSV)), False
    return generate_demo(), True


_SYLLABLES = ["al", "bor", "cen", "dar", "el", "fen", "gal", "hor", "ist", "jor", "kal", "lum",
              "mar", "nor", "ost", "per", "qua", "ron", "sol", "tur", "um", "val", "wes", "yor", "zan"]


def generate_demo(n_teams: int = 48, seed: int = 7) -> pd.DataFrame:
    """Synthetic league data with fictional teams (NOT real results)."""
    rng = np.random.default_rng(seed)
    names: list[str] = []
    while len(names) < n_teams:
        nm = (rng.choice(_SYLLABLES) + rng.choice(_SYLLABLES)).capitalize() + " FC"
        if nm not in names:
            names.append(nm)
    att = rng.normal(0, 0.28, n_teams)
    dfn = rng.normal(0, 0.28, n_teams)
    start, end = pd.Timestamp("2012-08-01"), pd.Timestamp("2026-11-08")
    n_matches = 14000
    days = np.sort(rng.integers(0, (end - start).days, n_matches))
    rows = []
    last_day = 0
    for d in days:
        # slow random walk of team strengths
        if d - last_day >= 30:
            att += rng.normal(0, 0.02, n_teams)
            dfn += rng.normal(0, 0.02, n_teams)
            last_day = d
        h, a = rng.choice(n_teams, 2, replace=False)
        neutral = rng.random() < 0.08
        ha = 0.0 if neutral else 0.22
        lh = 1.3 * np.exp(att[h] - dfn[a] + ha)
        la = 1.3 * np.exp(att[a] - dfn[h])
        rows.append((start + pd.Timedelta(days=int(d)), names[h], names[a],
                     rng.poisson(lh), rng.poisson(la),
                     "Demo Cup" if neutral else "Demo League", neutral))
    return normalise(pd.DataFrame(rows, columns=COLUMNS))
