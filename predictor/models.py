"""Two models and their blend.

RatingsModel  : Elo difference + EWMA attack/defence -> Poisson goal rates
                -> scoreline matrix -> home/draw/away probabilities.
MLModel       : gradient-boosted classifier on the full feature set.
Ensemble      : convex blend, weight tuned on a time-ordered validation slice.

Class order everywhere: 0 = home win, 1 = draw, 2 = away win.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import poisson
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, PoissonRegressor

from .features import AVG_GOALS

ML_FEATURES = ["elo_h", "elo_a", "elo_diff", "neutral", "form_h", "form_a",
               "gf_h", "ga_h", "gf_a", "ga_a", "n_h", "n_a", "rest_h", "rest_a",
               "h2h", "h2h_n", "importance"]
MAX_GOALS = 10
DC_RHO = -0.05          # Dixon-Coles low-score correlation


def recency_weights(dates: pd.Series, half_life_days: float = 1460.0) -> np.ndarray:
    age = (dates.max() - dates).dt.days.to_numpy(dtype=float)
    return 0.5 ** (age / half_life_days)


def score_matrix(lh: float, la: float) -> np.ndarray:
    g = np.arange(MAX_GOALS + 1)
    m = np.outer(poisson.pmf(g, lh), poisson.pmf(g, la))
    m[0, 0] *= 1 - lh * la * DC_RHO
    m[0, 1] *= 1 + lh * DC_RHO
    m[1, 0] *= 1 + la * DC_RHO
    m[1, 1] *= 1 - DC_RHO
    return m / m.sum()


def matrix_to_probs(m: np.ndarray) -> np.ndarray:
    return np.array([np.tril(m, -1).sum(), np.trace(m), np.triu(m, 1).sum()])


def _poisson_X(t: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    lg = lambda s: np.log(np.clip(s, 0.15, None) / AVG_GOALS)
    d = t["elo_diff"].to_numpy() / 400.0
    home = 1.0 - t["neutral"].to_numpy()
    xh = np.column_stack([lg(t["gf_h"]), lg(t["ga_a"]), d, home])
    xa = np.column_stack([lg(t["gf_a"]), lg(t["ga_h"]), -d, home])
    return xh, xa


class RatingsModel:
    def __init__(self) -> None:
        self.h = PoissonRegressor(alpha=1e-4, max_iter=500)
        self.a = PoissonRegressor(alpha=1e-4, max_iter=500)

    def fit(self, t: pd.DataFrame, w: np.ndarray | None = None) -> "RatingsModel":
        xh, xa = _poisson_X(t)
        self.h.fit(xh, t["hs"], sample_weight=w)
        self.a.fit(xa, t["as_"], sample_weight=w)
        return self

    def rates(self, t: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        xh, xa = _poisson_X(t)
        # clamp so extreme mismatches can't extrapolate to absurd goal rates
        return np.clip(self.h.predict(xh), 0.15, 4.0), np.clip(self.a.predict(xa), 0.15, 4.0)

    def predict_proba(self, t: pd.DataFrame) -> np.ndarray:
        lh, la = self.rates(t)
        return np.vstack([matrix_to_probs(score_matrix(x, y)) for x, y in zip(lh, la)])


class MLModel:
    def __init__(self) -> None:
        self.clf = HistGradientBoostingClassifier(
            max_depth=4, learning_rate=0.05, max_iter=250, l2_regularization=2.0,
            min_samples_leaf=40, random_state=0)

    def fit(self, t: pd.DataFrame, w: np.ndarray | None = None) -> "MLModel":
        self.clf.fit(t[ML_FEATURES], t["y"], sample_weight=w)
        return self

    def predict_proba(self, t: pd.DataFrame) -> np.ndarray:
        return self.clf.predict_proba(t[ML_FEATURES])


def blend(p_ratings: np.ndarray, p_ml: np.ndarray, w_ml: float) -> np.ndarray:
    p = w_ml * p_ml + (1 - w_ml) * p_ratings
    return p / p.sum(axis=-1, keepdims=True)


def log_loss(p: np.ndarray, y: np.ndarray) -> float:
    return float(-np.mean(np.log(np.clip(p[np.arange(len(y)), y], 1e-12, 1))))


def metrics(p: np.ndarray, y: np.ndarray) -> dict:
    onehot = np.eye(3)[y]
    return {"accuracy": float((p.argmax(1) == y).mean()),
            "log_loss": log_loss(p, y),
            "brier": float(((p - onehot) ** 2).sum(1).mean()),
            "n": int(len(y))}


def tune_weight(p_ratings: np.ndarray, p_ml: np.ndarray, y: np.ndarray) -> float:
    grid = np.linspace(0, 1, 21)
    return float(grid[int(np.argmin([log_loss(blend(p_ratings, p_ml, w), y) for w in grid]))])


def baselines(train: pd.DataFrame, test: pd.DataFrame) -> dict:
    prior = np.bincount(train["y"], minlength=3) / len(train)
    out = {"class_prior": metrics(np.tile(prior, (len(test), 1)), test["y"].to_numpy())}
    lr = LogisticRegression(max_iter=500).fit(train[["elo_diff"]], train["y"])
    out["elo_only_logistic"] = metrics(lr.predict_proba(test[["elo_diff"]]), test["y"].to_numpy())
    return out
