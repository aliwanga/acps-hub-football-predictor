"""Injury / absence handling.

Each absence has an `importance` (0-1: 1.0 = the team's best, hardest-to-replace
player; ~0.5 = regular starter; ~0.2 = rotation player) and a `status`.
The summed, availability-weighted impact is converted into an Elo penalty that
is fed to BOTH models (the ratings model and the ML model), so injured teams
get lower win probabilities and lower expected goals.

NOTE: the penalty size is a tunable assumption (no free historical injury
dataset exists to learn it from). Adjust ELO_PER_IMPACT / MAX_PENALTY below.
"""
from __future__ import annotations

import json
import threading
from pathlib import Path

from .data import DATA_DIR

STATUS_WEIGHT = {"out": 1.0, "doubtful": 0.5, "minor": 0.25}
ELO_PER_IMPACT = 60.0    # Elo points lost per 1.0 of availability-weighted importance
MAX_PENALTY = 150.0

INJURIES_JSON = DATA_DIR / "injuries.json"
_lock = threading.Lock()


def clean(entries) -> list[dict]:
    out = []
    for e in list(entries or [])[:40]:          # cap list size
        if not isinstance(e, dict):
            continue
        status = str(e.get("status", "out")).lower()
        if status not in STATUS_WEIGHT:
            status = "out"
        try:
            imp = float(e.get("importance", 0.5))
        except (TypeError, ValueError):
            imp = 0.5
        out.append({"player": str(e.get("player", "Unnamed"))[:60],
                    "importance": min(max(imp, 0.0), 1.0), "status": status})
    return out


def penalty(entries) -> tuple[float, float]:
    """Return (elo_penalty, impact)."""
    impact = sum(e["importance"] * STATUS_WEIGHT[e["status"]] for e in clean(entries))
    return min(MAX_PENALTY, ELO_PER_IMPACT * impact), impact


def load_all(path: Path = INJURIES_JSON) -> dict[str, list[dict]]:
    with _lock:
        if not path.exists():
            return {}
        try:
            return {k: clean(v) for k, v in json.loads(path.read_text()).items()}
        except json.JSONDecodeError:
            return {}


def save_team(team: str, entries, path: Path = INJURIES_JSON) -> list[dict]:
    entries = clean(entries)
    data = load_all(path)
    if entries:
        data[team] = entries
    else:
        data.pop(team, None)
    with _lock:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2))
    return entries
