"""Export the trained models as a static website (for GitHub Pages).

    python train.py
    python export_static.py            # writes site/  (index.html, engine.js, model.json)

The browser runs the same maths as the Python service: Poisson ratings model,
gradient-boosted trees (exported node-by-node), the blend and the injury
adjustment. A self-check compares the exported trees with scikit-learn and
refuses to publish if they disagree. Demo (fictional) data is refused unless
--allow-demo is passed, so a failed download can never publish fake teams.
"""
import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from predictor import injuries as inj
from predictor.features import AVG_GOALS, HOME_ADV, tournament_k
from predictor.models import DC_RHO, MAX_GOALS, ML_FEATURES
from predictor.service import Predictor

ROOT = Path(__file__).resolve().parent
SITE = ROOT / "site"
BUILD = ROOT / "build"
RATE_CLIP = (0.15, 4.0)


def export_trees(clf) -> dict:
    baseline = np.asarray(clf._baseline_prediction, dtype=float).ravel().tolist()
    trees = []
    for it in clf._predictors:
        for k, pred in enumerate(it):
            n = pred.nodes
            leaf = n["is_leaf"].astype(bool)
            trees.append({
                "k": k,
                "f": [-1 if l else int(f) for l, f in zip(leaf, n["feature_idx"])],
                "t": [0 if l else round(float(t), 9) for l, t in zip(leaf, n["num_threshold"])],
                "l": [int(x) for x in n["left"]],
                "r": [int(x) for x in n["right"]],
                "v": [round(float(v), 9) for v in n["value"]],
            })
    return {"features": ML_FEATURES, "baseline": baseline, "trees": trees}


def tree_raw(model_ml: dict, x: np.ndarray) -> np.ndarray:
    """Reference evaluator mirroring engine.js (used for the self-check)."""
    raw = np.array(model_ml["baseline"], dtype=float)
    for tr in model_ml["trees"]:
        i = 0
        while tr["f"][i] >= 0:
            i = tr["l"][i] if x[tr["f"][i]] <= tr["t"][i] else tr["r"][i]
        raw[tr["k"]] += tr["v"][i]
    return raw


def build_model(p: Predictor) -> dict:
    teams = sorted(p.teams(), key=lambda t: t["name"])          # code-point order == Python's `<`
    index = {t["name"]: i for i, t in enumerate(teams)}
    out_teams = []
    for t in teams:
        st = p.engine.teams[t["name"]]
        out_teams.append({
            "n": t["name"], "elo": round(st.elo, 4), "gf": round(st.gf, 5), "ga": round(st.ga, 5),
            "games": st.n, "form": round(float(np.mean(st.form)), 5) if st.form else 0.5,
            "last": str(st.last_date.date()),
        })
    h2h = {}
    for (a, b), q in p.engine.h2h.items():                      # a < b alphabetically
        if a in index and b in index and len(q):
            h2h[f"{index[a]}|{index[b]}"] = [round(float(np.mean(q)), 5), len(q)]

    rp = p.ratings
    return {
        "version": 1,
        "data_through": p.report["data_through"],
        "is_demo": p.report["is_demo"],
        "w_ml": p.w_ml,
        "report": {k: p.report[k] for k in ("ratings", "ml", "ensemble", "baselines",
                                            "train_matches", "test_matches", "test_window_days")},
        "consts": {
            "home_adv": HOME_ADV, "avg_goals": AVG_GOALS, "dc_rho": DC_RHO, "max_goals": MAX_GOALS,
            "rate_clip": list(RATE_CLIP), "default_importance": tournament_k(""), "rest_cap": 60,
            "status_weight": inj.STATUS_WEIGHT, "elo_per_impact": inj.ELO_PER_IMPACT,
            "max_penalty": inj.MAX_PENALTY, "max_injury_entries": 40,
        },
        "poisson": {
            "home": {"b": float(rp.h.intercept_), "w": [float(x) for x in rp.h.coef_]},
            "away": {"b": float(rp.a.intercept_), "w": [float(x) for x in rp.a.coef_]},
        },
        "ml": export_trees(p.ml.clf),
        "teams": out_teams,
        "h2h": h2h,
    }


def self_check(p: Predictor, model: dict) -> None:
    rng = np.random.default_rng(0)
    X = pd.DataFrame({
        "elo_h": rng.normal(1500, 150, 200), "elo_a": rng.normal(1500, 150, 200),
        "neutral": rng.integers(0, 2, 200).astype(float),
        "form_h": rng.random(200), "form_a": rng.random(200),
        "gf_h": rng.uniform(0.4, 2.8, 200), "ga_h": rng.uniform(0.4, 2.8, 200),
        "gf_a": rng.uniform(0.4, 2.8, 200), "ga_a": rng.uniform(0.4, 2.8, 200),
        "n_h": rng.integers(5, 500, 200), "n_a": rng.integers(5, 500, 200),
        "rest_h": rng.integers(0, 61, 200), "rest_a": rng.integers(0, 61, 200),
        "h2h": rng.random(200), "h2h_n": rng.integers(0, 6, 200),
        "importance": rng.choice([20, 30, 40, 60], 200)})
    X["elo_diff"] = X["elo_h"] - X["elo_a"] + (1 - X["neutral"]) * HOME_ADV
    X = X[ML_FEATURES]
    want = p.ml.clf.predict_proba(X)
    for row, w in zip(X.to_numpy(dtype=float), want):
        raw = tree_raw(model["ml"], row)
        e = np.exp(raw - raw.max())
        if np.abs(e / e.sum() - w).max() > 1e-6:
            raise SystemExit("Exported trees do not reproduce scikit-learn predictions "
                             "(scikit-learn internals may have changed). Aborting.")
    print("Self-check passed: exported trees match scikit-learn on 200 random rows.")


def make_cases(p: Predictor, model: dict, date: str) -> list[dict]:
    """Python reference predictions used by tools/verify_static.js."""
    names = [t["n"] for t in model["teams"]]
    top = [t["name"] for t in p.teams()[:12]]
    rng = np.random.default_rng(1)
    pairs = [(top[0], top[1], False, None), (top[1], top[0], True, None), (top[2], top[7], False, None)]
    pairs += [tuple(rng.choice(names, 2, replace=False)) + (bool(rng.integers(0, 2)), None) for _ in range(25)]
    hurt = [{"player": "Star", "importance": 1.0, "status": "out"},
            {"player": "Starter", "importance": 0.6, "status": "doubtful"}]
    cases = []
    for h, a, neutral, _ in pairs:
        cases.append({"home": h, "away": a, "neutral": neutral, "date": date, "injuries": None})
    cases.append({"home": top[0], "away": top[1], "neutral": False, "date": date,
                  "injuries": {"home": hurt, "away": []}})
    cases.append({"home": top[3], "away": top[4], "neutral": True, "date": date,
                  "injuries": {"home": [], "away": hurt}})
    for c in cases:
        inj_ = c["injuries"] or {}
        c["expected"] = p.predict(c["home"], c["away"], c["neutral"], "", date,
                                  home_injuries=inj_.get("home", []), away_injuries=inj_.get("away", []))
    return cases


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--allow-demo", action="store_true", help="allow fictional demo data (testing only)")
    a = ap.parse_args()

    p = Predictor()
    if p.report["is_demo"] and not a.allow_demo:
        raise SystemExit("Model was trained on DEMO data. Refusing to publish. "
                         "Run fetch_data.py and train.py first (or pass --allow-demo for testing).")
    model = build_model(p)
    self_check(p, model)

    if SITE.exists():
        shutil.rmtree(SITE)
    SITE.mkdir()
    (SITE / "model.json").write_text(json.dumps(model, separators=(",", ":")))
    shutil.copy(ROOT / "site_src" / "engine.js", SITE / "engine.js")
    shutil.copy(ROOT / "static" / "index.html", SITE / "index.html")
    (SITE / ".nojekyll").write_text("")

    BUILD.mkdir(exist_ok=True)
    date = str((p.last_date + pd.Timedelta(days=3)).date())
    (BUILD / "cases.json").write_text(json.dumps({"cases": make_cases(p, model, date)}))
    size = (SITE / "model.json").stat().st_size / 1e6
    print(f"Wrote site/ ({len(model['teams'])} teams, {len(model['h2h'])} head-to-head pairs, "
          f"{len(model['ml']['trees'])} trees, model.json {size:.2f} MB)")


if __name__ == "__main__":
    main()
