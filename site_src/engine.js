/* In-browser prediction engine (mirrors predictor/*.py).
 * Works in browsers (loads model.json) and in Node (module.exports.createEngine), used for verification. */
(function () {
  "use strict";

  function createEngine(M) {
    const C = M.consts;
    const teams = M.teams;
    const byName = new Map(teams.map((t, i) => [t.n, i]));
    const byLower = new Map(teams.map((t, i) => [t.n.toLowerCase(), i]));
    const lg = (v) => Math.log(Math.max(v, C.rate_clip[0]) / C.avg_goals);
    const clip = (v) => Math.min(Math.max(v, C.rate_clip[0]), C.rate_clip[1]);
    const pct = (v) => Math.round(v * 1000) / 10;

    function resolve(name) {
      name = String(name || "").trim();
      if (byName.has(name)) return byName.get(name);
      const low = name.toLowerCase();
      if (byLower.has(low)) return byLower.get(low);
      const close = teams.filter((t) => t.n.toLowerCase().includes(low)).slice(0, 3).map((t) => t.n);
      throw new Error(`Unknown team '${name}'.` + (close.length && low ? ` Did you mean: ${close.join(", ")}?` : ""));
    }

    function cleanInjuries(list) {
      const out = [];
      for (const e of (list || []).slice(0, C.max_injury_entries)) {
        if (!e || typeof e !== "object") continue;
        let status = String(e.status || "out").toLowerCase();
        if (!(status in C.status_weight)) status = "out";
        let imp = parseFloat(e.importance);
        if (!isFinite(imp)) imp = 0.5;
        out.push({ player: String(e.player || "Unnamed").slice(0, 60), importance: Math.min(Math.max(imp, 0), 1), status });
      }
      return out;
    }

    function penalty(list) {
      const impact = list.reduce((s, e) => s + e.importance * C.status_weight[e.status], 0);
      return { pen: Math.min(C.max_penalty, C.elo_per_impact * impact), impact };
    }

    function h2hFeature(i, j) {
      const lo = Math.min(i, j), hi = Math.max(i, j);
      const rec = M.h2h[lo + "|" + hi];
      if (!rec) return [0.5, 0];
      return [i < j ? rec[0] : 1 - rec[0], rec[1]];
    }

    function restDays(lastIso, dateIso) {
      const d = (Date.parse(dateIso + "T00:00:00Z") - Date.parse(lastIso + "T00:00:00Z")) / 86400000;
      return Math.max(Math.min(Math.floor(d), C.rest_cap), 0);
    }

    function poissonPmf(lam, n) {
      const p = new Array(n + 1);
      p[0] = Math.exp(-lam);
      for (let k = 1; k <= n; k++) p[k] = (p[k - 1] * lam) / k;
      return p;
    }

    function scoreMatrix(lh, la) {
      const n = C.max_goals, rho = C.dc_rho;
      const ph = poissonPmf(lh, n), pa = poissonPmf(la, n);
      const m = [];
      let total = 0;
      for (let i = 0; i <= n; i++) {
        m.push([]);
        for (let j = 0; j <= n; j++) {
          let v = ph[i] * pa[j];
          if (i === 0 && j === 0) v *= 1 - lh * la * rho;
          else if (i === 0 && j === 1) v *= 1 + lh * rho;
          else if (i === 1 && j === 0) v *= 1 + la * rho;
          else if (i === 1 && j === 1) v *= 1 - rho;
          m[i].push(v);
          total += v;
        }
      }
      for (const row of m) for (let j = 0; j <= n; j++) row[j] /= total;
      return m;
    }

    function matrixProbs(m) {
      let hw = 0, dr = 0, aw = 0;
      for (let i = 0; i < m.length; i++)
        for (let j = 0; j < m.length; j++) {
          if (i > j) hw += m[i][j]; else if (i === j) dr += m[i][j]; else aw += m[i][j];
        }
      return [hw, dr, aw];
    }

    function mlProbs(x) {
      const raw = M.ml.baseline.slice();
      for (const t of M.ml.trees) {
        let i = 0;
        while (t.f[i] >= 0) i = x[t.f[i]] <= t.t[i] ? t.l[i] : t.r[i];
        raw[t.k] += t.v[i];
      }
      const mx = Math.max(...raw);
      const e = raw.map((r) => Math.exp(r - mx));
      const s = e.reduce((a, b) => a + b, 0);
      return e.map((v) => v / s);
    }

    function today() { return new Date().toISOString().slice(0, 10); }

    function predict(req) {
      const hi = resolve(req.home), ai = resolve(req.away);
      if (hi === ai) throw new Error("Pick two different teams.");
      const neutral = !!req.neutral;
      const date = req.date || today();
      const injs = req.injuries || {};
      const hList = cleanInjuries(injs.home), aList = cleanInjuries(injs.away);
      const hp = penalty(hList), ap = penalty(aList);
      const H = teams[hi], A = teams[ai];

      const eloH = H.elo - hp.pen, eloA = A.elo - ap.pen;
      const eloDiff = eloH - eloA + (neutral ? 0 : C.home_adv);
      const [h2h, h2hN] = h2hFeature(hi, ai);
      const restH = restDays(H.last, date), restA = restDays(A.last, date);

      // ratings model (Poisson)
      const d = eloDiff / 400, home = neutral ? 0 : 1;
      const xh = [lg(H.gf), lg(A.ga), d, home], xa = [lg(A.gf), lg(H.ga), -d, home];
      const lin = (m, x) => m.b + m.w.reduce((s, w, k) => s + w * x[k], 0);
      const lh = clip(Math.exp(lin(M.poisson.home, xh))), la = clip(Math.exp(lin(M.poisson.away, xa)));
      const mat = scoreMatrix(lh, la);
      const pr = matrixProbs(mat);

      // ML model
      const feat = { elo_h: eloH, elo_a: eloA, elo_diff: eloDiff, neutral: neutral ? 1 : 0, form_h: H.form, form_a: A.form,
        gf_h: H.gf, ga_h: H.ga, gf_a: A.gf, ga_a: A.ga, n_h: H.games, n_a: A.games, rest_h: restH, rest_a: restA,
        h2h, h2h_n: h2hN, importance: C.default_importance };
      const pm = mlProbs(M.ml.features.map((f) => feat[f]));

      const w = M.w_ml;
      let p = pr.map((v, k) => w * pm[k] + (1 - w) * v);
      const s = p[0] + p[1] + p[2];
      p = p.map((v) => v / s);

      const cells = [];
      for (let i = 0; i < mat.length; i++) for (let j = 0; j < mat.length; j++) cells.push([i, j, mat[i][j]]);
      cells.sort((x, y) => y[2] - x[2]);

      const trip = (v) => ({ home_win: pct(v[0]), draw: pct(v[1]), away_win: pct(v[2]) });
      const keys = ["home_win", "draw", "away_win"];
      return {
        home: H.n, away: A.n, neutral, date,
        probabilities: trip(p),
        models: { ratings: trip(pr), ml: trip(pm), weight_ml: w },
        expected_goals: { home: Math.round(lh * 100) / 100, away: Math.round(la * 100) / 100 },
        likely_scores: cells.slice(0, 5).map(([i, j, v]) => ({ score: `${i}-${j}`, probability: pct(v) })),
        pick: keys[p.indexOf(Math.max(...p))],
        factors: {
          elo: { home: Math.round(H.elo), away: Math.round(A.elo) },
          elo_after_injuries: { home: Math.round(eloH), away: Math.round(eloA) },
          recent_form_points_per_game: { home: Math.round(H.form * 300) / 100, away: Math.round(A.form * 300) / 100 },
          rest_days: { home: restH, away: restA },
          injuries: {
            home: { players: hList, impact: Math.round(hp.impact * 100) / 100, elo_penalty: Math.round(hp.pen) },
            away: { players: aList, impact: Math.round(ap.impact * 100) / 100, elo_penalty: Math.round(ap.pen) },
          },
        },
        data_is_demo: M.is_demo,
      };
    }

    function listTeams() {
      return {
        teams: teams.map((t) => ({ name: t.n, elo: Math.round(t.elo), games: t.games, last_played: t.last }))
          .sort((a, b) => b.elo - a.elo),
        data_is_demo: M.is_demo, data_through: M.data_through, static_edition: true,
      };
    }

    return { predict, teams: listTeams };
  }

  if (typeof module !== "undefined" && module.exports) module.exports = { createEngine };

  if (typeof window !== "undefined") {
    const ready = fetch("model.json").then((r) => {
      if (!r.ok) throw new Error("Could not load model.json");
      return r.json();
    }).then(createEngine);
    window.StaticEngine = {
      teams: async () => (await ready).teams(),
      predict: async (req) => (await ready).predict(req),
    };
  }
})();
