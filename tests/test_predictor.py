"""Run with:  python -m unittest discover -s tests"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from predictor.data import generate_demo
from predictor.features import build_table
from predictor.models import matrix_to_probs, score_matrix
from predictor import injuries as inj


class TestCore(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.matches = generate_demo(n_teams=12, seed=1).head(1500)

    def test_score_matrix_probs_sum_to_one(self):
        p = matrix_to_probs(score_matrix(1.6, 0.9))
        self.assertAlmostEqual(p.sum(), 1.0, places=6)
        self.assertGreater(p[0], p[2])

    def test_no_leakage_in_features(self):
        """Changing a match's own result must not change that match's features."""
        base, _ = build_table(self.matches)
        altered = self.matches.copy()
        i = 900
        altered.loc[i, "home_score"] += 5
        alt, _ = build_table(altered)
        cols = [c for c in base.columns if c not in ("hs", "as_", "y")]
        pd.testing.assert_series_equal(base.loc[i, cols], alt.loc[i, cols])
        # ...but it should change later rows
        self.assertFalse(base.loc[i + 100, cols].equals(alt.loc[i + 100, cols]))

    def test_injury_penalty_monotonic_and_capped(self):
        small = inj.penalty([{"importance": 0.2, "status": "out"}])[0]
        big = inj.penalty([{"importance": 1.0, "status": "out"}])[0]
        huge = inj.penalty([{"importance": 1.0, "status": "out"}] * 10)[0]
        self.assertLess(small, big)
        self.assertEqual(huge, inj.MAX_PENALTY)
        self.assertEqual(inj.penalty([]), (0.0, 0.0))


@unittest.skipUnless((Path(__file__).resolve().parent.parent / "model" / "predictor.joblib").exists(),
                     "train a model first")
class TestPredictor(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from predictor.service import Predictor
        cls.p = Predictor()
        cls.a, cls.b = (t["name"] for t in cls.p.teams()[:2])

    def test_probabilities_sum_to_100(self):
        r = self.p.predict(self.a, self.b, home_injuries=[], away_injuries=[])
        self.assertAlmostEqual(sum(r["probabilities"].values()), 100, delta=0.2)

    def test_injuries_lower_win_probability(self):
        clean = self.p.predict(self.a, self.b, home_injuries=[], away_injuries=[])
        hurt = self.p.predict(self.a, self.b, away_injuries=[], home_injuries=[
            {"player": "Star", "importance": 1.0, "status": "out"},
            {"player": "Starter", "importance": 0.6, "status": "out"}])
        self.assertLess(hurt["probabilities"]["home_win"], clean["probabilities"]["home_win"])
        self.assertLess(hurt["expected_goals"]["home"], clean["expected_goals"]["home"])

    def test_stronger_team_favoured(self):
        top, bottom = self.p.teams()[0]["name"], self.p.teams()[-1]["name"]
        r = self.p.predict(top, bottom, neutral=True, home_injuries=[], away_injuries=[])
        self.assertGreater(r["probabilities"]["home_win"], r["probabilities"]["away_win"])

    def test_unknown_team(self):
        with self.assertRaises(ValueError):
            self.p.predict("Nowhere United", self.a)


@unittest.skipUnless((Path(__file__).resolve().parent.parent / "model" / "predictor.joblib").exists(),
                     "train a model first")
class TestApi(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import os
        import app as app_module
        cls.os = os
        cls.client = app_module.app.test_client()
        cls.teams = [t["name"] for t in app_module.get_predictor().teams()[:2]]

    def test_predict_endpoint(self):
        r = self.client.post("/api/predict", json={"home": self.teams[0], "away": self.teams[1]})
        self.assertEqual(r.status_code, 200)
        self.assertIn("probabilities", r.get_json())

    def test_predict_validation(self):
        self.assertEqual(self.client.post("/api/predict", json={"home": "x"}).status_code, 400)
        self.assertEqual(self.client.post("/api/predict", json={"home": "Nowhere", "away": "Else"}).status_code, 400)

    def test_injury_write_requires_token(self):
        url = f"/api/injuries/{self.teams[0]}"
        self.os.environ.pop("ADMIN_TOKEN", None)
        self.assertEqual(self.client.put(url, json={"players": []}).status_code, 403)
        self.os.environ["ADMIN_TOKEN"] = "s3cret"
        try:
            self.assertEqual(self.client.put(url, json={"players": []}).status_code, 401)
            ok = self.client.put(url, json={"players": []}, headers={"Authorization": "Bearer s3cret"})
            self.assertEqual(ok.status_code, 200)
        finally:
            self.os.environ.pop("ADMIN_TOKEN", None)


if __name__ == "__main__":
    unittest.main()
