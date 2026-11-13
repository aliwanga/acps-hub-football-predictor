// Verifies that the browser engine reproduces the Python predictions.
//   node tools/verify_static.js
const fs = require("fs");
const path = require("path");
const { createEngine } = require("../site_src/engine.js");

const root = path.resolve(__dirname, "..");
const model = JSON.parse(fs.readFileSync(path.join(root, "site", "model.json"), "utf8"));
const { cases } = JSON.parse(fs.readFileSync(path.join(root, "build", "cases.json"), "utf8"));
const engine = createEngine(model);

let worst = 0, bad = 0;
const diff = (a, b) => Math.abs(a - b);
for (const c of cases) {
  const got = engine.predict({ home: c.home, away: c.away, neutral: c.neutral, date: c.date, injuries: c.injuries });
  const exp = c.expected;
  const pairs = [];
  for (const k of ["home_win", "draw", "away_win"]) {
    pairs.push(diff(got.probabilities[k], exp.probabilities[k]));
    pairs.push(diff(got.models.ratings[k], exp.models.ratings[k]));
    pairs.push(diff(got.models.ml[k], exp.models.ml[k]));
  }
  pairs.push(diff(got.expected_goals.home, exp.expected_goals.home), diff(got.expected_goals.away, exp.expected_goals.away));
  for (let i = 0; i < 3; i++) pairs.push(diff(got.likely_scores[i].probability, exp.likely_scores[i].probability));
  pairs.push(diff(got.factors.elo_after_injuries.home, exp.factors.elo_after_injuries.home));
  pairs.push(diff(got.factors.rest_days.home, exp.factors.rest_days.home));
  const m = Math.max(...pairs);
  worst = Math.max(worst, m);
  if (m > 0.2) { bad++; console.error(`MISMATCH ${c.home} vs ${c.away}: max diff ${m.toFixed(3)}`); }
}
console.log(`Checked ${cases.length} predictions; worst difference ${worst.toFixed(3)} percentage points.`);
process.exit(bad ? 1 : 0);
