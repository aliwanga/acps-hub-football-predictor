"""Training pipeline and the Predictor used by the API."""
from __future__ import annotations

import difflib
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from . import injuries as inj
from .data import ROOT, load_matches
from .features import build_table
from .models import (MLModel, RatingsModel, baselines, blend, metrics, recency_weights,
                     score_matrix, matrix_to_probs, tune_weight)

MODEL_PATH = ROOT / "model" / "predictor.joblib"
MIN_GAMES = 5


def train(test_days: int = 365, verbose: bool = True) -> dict:
    matches, is_demo = load_matches()
    table, engine = build_table(matches)
    usable = table[(table.n_h >= MIN_GAMES) & (table.n_a >= MIN_GAMES)].reset_index(drop=True)
    last = usable["date"].max()
    cutoff = last - pd.Timedelta(days=test_days)
    train_all = usable[usable["date"] < cutoff].reset_index(drop=True)
    test = usable[usable["date"] >= cutoff].reset_index(drop=True)
    if len(train_all) < 500 or len(test) < 50:
        raise SystemExit("Not enough data for a train/test split; lower --test-days or add data.")

    # 1) choose blend weight on the most recent 15% of the training period
    split = int(len(train_all) * 0.85)
    fit_part, val_part = train_all.iloc[:split], train_all.iloc[split:]
    w_fit = recency_weights(fit_part["date"])
    r, m = RatingsModel().fit(fit_part, w_fit), MLModel().fit(fit_part, w_fit)
    w_ml = tune_weight(r.predict_proba(val_part), m.predict_proba(val_part), val_part["y"].to_numpy())

    # 2) honest hold-out evaluation: refit on all pre-cutoff data, score the last `test_days`
    w_all = recency_weights(train_all["date"])
    r, m = RatingsModel().fit(train_all, w_all), MLModel().fit(train_all, w_all)
    pr, pm = r.predict_proba(test), m.predict_proba(test)
    y = test["y"].to_numpy()
    report = {
        "is_demo": is_demo,
        "data_through": str(last.date()),
        "train_matches": int(len(train_all)),
        "test_matches": int(len(test)),
        "test_window_days": test_days,
        "blend_weight_ml": w_ml,
        "ratings": metrics(pr, y),
        "ml": metrics(pm, y),
        "ensemble": metrics(blend(pr, pm, w_ml), y),
        "baselines": baselines(train_all, test),
    }

    # 3) final production models use every usable match
    w_full = recency_weights(usable["date"])
    ratings, ml = RatingsModel().fit(usable, w_full), MLModel().fit(usable, w_full)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    # write to a temp file then rename, so a running server never reads a half-written model
    tmp = MODEL_PATH.with_suffix(".tmp")
    joblib.dump({"engine": engine, "ratings": ratings, "ml": ml, "w_ml": w_ml,
                 "report": report, "last_date": last}, tmp)
    os.replace(tmp, MODEL_PATH)
    if verbose:
        _print_report(report)
    return report


def _print_report(r: dict) -> None:
    if r["is_demo"]:
        print("\n*** DEMO DATA (fictional teams). Run fetch_data.py for real results. ***")
    print(f"\nTrained through {r['data_through']}: {r['train_matches']} train / "
          f"{r['test_matches']} hold-out matches ({r['test_window_days']} days)")
    print(f"Blend weight on ML model: {r['blend_weight_ml']:.2f}\n")
    print(f"{'model':<22}{'accuracy':>10}{'log loss':>10}{'brier':>9}")
    rows = [("ratings (Elo+Poisson)", r["ratings"]), ("ml (grad. boosting)", r["ml"]),
            ("ENSEMBLE", r["ensemble"])] + [(f"baseline: {k}", v) for k, v in r["baselines"].items()]
    for name, v in rows:
        print(f"{name:<22}{v['accuracy']:>10.3f}{v['log_loss']:>10.4f}{v['brier']:>9.4f}")


class Predictor:
    def __init__(self, path: Path = MODEL_PATH) -> None:
        blob = joblib.load(path)
        self.engine = blob["engine"]
        self.ratings: RatingsModel = blob["ratings"]
        self.ml: MLModel = blob["ml"]
        self.w_ml: float = blob["w_ml"]
        self.report: dict = blob["report"]
        self.last_date: pd.Timestamp = blob["last_date"]

    # ------------------------------------------------------------------ teams
    def teams(self, active_days: int = 540) -> list[dict]:
        out = []
        for name, st in self.engine.teams.items():
            if st.last_date is None or (self.last_date - st.last_date).days > active_days:
                continue
            out.append({"name": name, "elo": round(st.elo), "games": st.n,
                        "last_played": str(st.last_date.date())})
        return sorted(out, key=lambda t: -t["elo"])

    def resolve(self, name: str) -> str:
        if self.engine.known(name):
            return name
        lower = {t.lower(): t for t in self.engine.teams}
        if name.lower() in lower:
            return lower[name.lower()]
        close = difflib.get_close_matches(name, list(self.engine.teams), n=3, cutoff=0.6)
        hint = f" Did you mean: {', '.join(close)}?" if close else ""
        raise ValueError(f"Unknown team '{name}'.{hint}")

    # ------------------------------------------------------------------ predict
    def predict(self, home: str, away: str, neutral: bool = False, tournament: str = "",
                date=None, home_injuries=None, away_injuries=None) -> dict:
        home, away = self.resolve(home), self.resolve(away)
        if home == away:
            raise ValueError("Pick two different teams.")
        date = pd.Timestamp(date) if date else pd.Timestamp.today().normalize()
        store = inj.load_all()
        h_list = inj.clean(home_injuries) if home_injuries is not None else store.get(home, [])
        a_list = inj.clean(away_injuries) if away_injuries is not None else store.get(away, [])
        h_pen, h_imp = inj.penalty(h_list)
        a_pen, a_imp = inj.penalty(a_list)

        f = self.engine.features(home, away, date, neutral, tournament, h_pen, a_pen)
        t = pd.DataFrame([f])
        lh, la = (float(x[0]) for x in self.ratings.rates(t))
        mat = score_matrix(lh, la)
        p_r = matrix_to_probs(mat)
        p_m = self.ml.predict_proba(t)[0]
        p = blend(p_r[None, :], p_m[None, :], self.w_ml)[0]

        top = np.dstack(np.unravel_index(np.argsort(-mat.ravel())[:5], mat.shape))[0]
        hs, as_ = self.engine.state(home), self.engine.state(away)
        pct = lambda v: round(float(v) * 100, 1)
        return {
            "home": home, "away": away, "neutral": neutral, "date": str(date.date()),
            "probabilities": {"home_win": pct(p[0]), "draw": pct(p[1]), "away_win": pct(p[2])},
            "models": {
                "ratings": {"home_win": pct(p_r[0]), "draw": pct(p_r[1]), "away_win": pct(p_r[2])},
                "ml": {"home_win": pct(p_m[0]), "draw": pct(p_m[1]), "away_win": pct(p_m[2])},
                "weight_ml": self.w_ml,
            },
            "expected_goals": {"home": round(lh, 2), "away": round(la, 2)},
            "likely_scores": [{"score": f"{i}-{j}", "probability": pct(mat[i, j])} for i, j in top],
            "pick": ["home_win", "draw", "away_win"][int(p.argmax())],
            "factors": {
                "elo": {"home": round(hs.elo), "away": round(as_.elo)},
                "elo_after_injuries": {"home": round(f["elo_h"]), "away": round(f["elo_a"])},
                "recent_form_points_per_game": {"home": round(f["form_h"] * 3, 2),
                                                "away": round(f["form_a"] * 3, 2)},
                "rest_days": {"home": f["rest_h"], "away": f["rest_a"]},
                "injuries": {
                    "home": {"players": h_list, "impact": round(h_imp, 2), "elo_penalty": round(h_pen)},
                    "away": {"players": a_list, "impact": round(a_imp, 2), "elo_penalty": round(a_pen)},
                },
            },
            "data_is_demo": self.report["is_demo"],
        }
