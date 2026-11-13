# ACPS-HUB FOOTBALL PREDICTOR

Predicts home / draw / away probabilities, expected goals and likely scorelines for football matches worldwide (national teams and club leagues). Two models are blended:

| Component | What it does |
|---|---|
| **Ratings model** | Elo ratings (tournament-weighted, goal-difference-aware, home advantage) plus time-smoothed attack/defence strengths feed a Poisson goals model with a Dixon-Coles low-score correction. |
| **ML model** | Gradient-boosted classifier on Elo, recent form, goals for/against, rest days, head-to-head, venue and competition importance. |
| **Blend** | Weight between the two is tuned on a time-ordered validation slice. |
| **Injuries** | Each absence has an importance (0-1) and a status (out / doubtful / minor). The total becomes an Elo penalty applied to **both** models. |

## Quick start

```bash
pip install -r requirements.txt
python fetch_data.py      # downloads real results into data/matches.csv
python train.py           # trains, prints a hold-out evaluation, saves model/
python app.py             # open http://localhost:5000
python -m unittest discover -s tests
```

Requires Python 3.10+. The dev server (`python app.py`) binds to 127.0.0.1; set `HOST=0.0.0.0` to expose it, or use Docker (below).

Without `data/matches.csv`, `train.py` uses a synthetic **demo** dataset with fictional teams so you can try the pipeline offline. The UI shows a banner when demo data is in use.

## Free hosting on GitHub Pages (static edition)

The same predictor runs entirely in the browser, hosted free by GitHub Pages. `export_static.py` writes `site/` (`index.html`, `engine.js`, `model.json`): the ratings model, the exported gradient-boosted trees, the blend and the injury adjustment all run in JavaScript, and `tools/verify_static.js` checks them against the Python models.

The workflow `.github/workflows/pages.yml` downloads fresh data, trains, exports, verifies and deploys, on every push to `main` and daily at 03:00 UTC.

One-time setup: **Settings → Pages → Build and deployment → Source: GitHub Actions**, then **Actions → Deploy website (GitHub Pages) → Run workflow**. The site appears at `https://<username>.github.io/<repository>/`.

Notes: on a free GitHub account Pages needs a **public** repository; stored injury lists (`data/injuries.json`, the PUT endpoint) don't exist in the static edition, but injuries entered on the page work; the export refuses to publish models trained on demo data. To try it locally: `python train.py && python export_static.py --allow-demo`, then serve `site/` with `python -m http.server -d site`.

## Deploy with Docker (auto-refreshing)

```bash
docker compose up -d --build
```

- `web`: gunicorn serving the site on port 5000. On first boot with an empty volume it tries `fetch_data.py` for real data, and falls back to demo data if the download fails.
- `retrainer`: runs `refresh.py --loop --hour 3`, so every day at 03:00 UTC it downloads fresh results, retrains and swaps the model in atomically. The web service hot-reloads it with no restart.
- A failed or partial download (under 80% of the existing row count) is rejected and the current data and model keep serving.
- Data, injuries and models live in Docker volumes and survive rebuilds. Health check: `/healthz`.
- Force a refresh now: `docker compose exec retrainer python refresh.py`.
- Put a reverse proxy (Caddy, nginx) with HTTPS in front for a public site.
- Writing stored injury lists (`PUT /api/injuries/<team>`) is **disabled unless you set `ADMIN_TOKEN`** (copy `.env.example` to `.env`), then send `Authorization: Bearer <token>`. Predictions and per-request injuries need no token.

## Data

`fetch_data.py` pulls international results (martj42/international_results on GitHub) and club leagues from football-data.co.uk. Check their terms before commercial use. To use your own data, write `data/matches.csv` with columns:

`date, home_team, away_team, home_score, away_score, tournament, neutral`

Re-run `fetch_data.py` and `train.py` regularly (e.g. nightly cron) to keep ratings current.

## API

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/teams` | Teams with Elo |
| GET | `/api/rankings?top=50` | Elo leaderboard |
| POST | `/api/predict` | `{"home","away","neutral","injuries":{"home":[...],"away":[...]}}` |
| GET | `/api/injuries` | Stored injury lists |
| PUT | `/api/injuries/<team>` | `{"players":[...]}`; needs `ADMIN_TOKEN` bearer token |
| GET | `/api/model` | Hold-out evaluation |

Injuries not passed in the request fall back to the lists in `data/injuries.json`.

## Data sources and licences

This repo contains **no match data**; `fetch_data.py` downloads it at run time from third-party sources (martj42/international_results, football-data.co.uk). Those sources have their own licences and terms, so check them before redistributing the data or using it commercially. The licence in this repository covers this code only.

## Honest limitations

- **Injuries are an assumption, not a learned effect.** No free historical injury dataset exists, so the penalty size (`ELO_PER_IMPACT` in `predictor/injuries.py`) is hand-set. If you get historical injury data (e.g. from a paid API), add it to the feature builder to learn the real effect. Injury lists must be entered manually or fed from an API you license.
- **Football is noisy.** On real data, good models reach roughly 50-55% three-way accuracy; treat output as probabilities, not certainties. Check `train.py`'s hold-out table against the baselines (class prior, Elo-only) to see whether the ML model actually adds value on your data.
- **Not tested against live downloads in the build environment** (no outside network there): `fetch_data.py` is written against the sources' current formats, so run it and check its output.
- Not betting advice.

## Ideas for next steps

xG data, lineups and squad market values as features, league-specific home advantage, calibration plots, scheduled retraining, Docker + gunicorn deployment.

## Contributing

Contributions are by arrangement with ACPS HUB (see [LICENSE](LICENSE)). If you have been invited to contribute, please run `python train.py && python -m unittest discover -s tests` before submitting; CI runs the same on Python 3.10 and 3.12.

## Licence

© 2026 ACPS HUB. All rights reserved. See [LICENSE](LICENSE).

---

**© 2026 ACPS HUB. All rights reserved.**  
Developed by Allan.  
*Building smart solutions for better communities through technology and innovation*
