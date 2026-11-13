"""Flask API + web UI.

    python app.py            # http://localhost:5000

Endpoints
    GET  /api/teams                  teams with Elo (active in the last ~18 months)
    GET  /api/rankings?top=50        Elo leaderboard
    POST /api/predict                {home, away, neutral?, tournament?, date?, injuries?: {home:[...], away:[...]}}
    GET  /api/injuries               all stored injury lists
    PUT  /api/injuries/<team>        replace a team's injury list (requires ADMIN_TOKEN, see below)
    GET  /api/model                  hold-out evaluation report

Environment
    ADMIN_TOKEN   secret for writing stored injury lists: send  Authorization: Bearer <token>.
                  If unset, the write endpoint is disabled. (Per-request injuries passed to
                  /api/predict need no token.)
    HOST / PORT   dev server bind address (defaults 127.0.0.1 / 5000)
"""
import hmac
import os
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory

from predictor import injuries as inj
from predictor.service import MODEL_PATH, Predictor

app = Flask(__name__, static_folder="static")
app.config["MAX_CONTENT_LENGTH"] = 64 * 1024     # requests are tiny JSON bodies

if not MODEL_PATH.exists():
    raise SystemExit("No trained model found. Run:  python train.py")

_state = {"predictor": Predictor(), "mtime": MODEL_PATH.stat().st_mtime}


def get_predictor() -> Predictor:
    """Hot-reload: pick up a freshly retrained model without restarting the server."""
    try:
        mtime = MODEL_PATH.stat().st_mtime
        if mtime != _state["mtime"]:
            _state["predictor"], _state["mtime"] = Predictor(), mtime
    except Exception as exc:  # keep serving the old model if the new one is unreadable
        app.logger.warning("model reload failed: %s", exc)
    return _state["predictor"]


def bad(msg: str, code: int = 400):
    return jsonify({"error": msg}), code


@app.get("/")
def index():
    return send_from_directory(Path(app.static_folder), "index.html")


@app.get("/api/teams")
def teams():
    p = get_predictor()
    return jsonify({"teams": p.teams(), "data_is_demo": p.report["is_demo"],
                    "data_through": p.report["data_through"]})


@app.get("/healthz")
def healthz():
    p = get_predictor()
    return jsonify({"status": "ok", "data_through": p.report["data_through"],
                    "data_is_demo": p.report["is_demo"]})


@app.get("/api/rankings")
def rankings():
    top = min(max(request.args.get("top", 50, type=int), 1), 500)
    return jsonify({"rankings": get_predictor().teams()[:top]})


@app.post("/api/predict")
def predict():
    body = request.get_json(silent=True) or {}
    if not body.get("home") or not body.get("away"):
        return bad("Both 'home' and 'away' are required.")
    injuries = body.get("injuries") or {}
    try:
        return jsonify(get_predictor().predict(
            body["home"], body["away"], bool(body.get("neutral", False)),
            body.get("tournament", ""), body.get("date"),
            home_injuries=injuries.get("home"), away_injuries=injuries.get("away")))
    except ValueError as exc:
        return bad(str(exc))


@app.get("/api/injuries")
def get_injuries():
    return jsonify(inj.load_all())


@app.put("/api/injuries/<path:team>")
def put_injuries(team: str):
    token = os.environ.get("ADMIN_TOKEN", "")
    if not token:
        return bad("Writing stored injuries is disabled: set the ADMIN_TOKEN environment variable.", 403)
    supplied = request.headers.get("Authorization", "").removeprefix("Bearer ").strip()
    if not hmac.compare_digest(supplied, token):
        return bad("Unauthorized.", 401)
    try:
        team = get_predictor().resolve(team)
    except ValueError as exc:
        return bad(str(exc), 404)
    players = (request.get_json(silent=True) or {}).get("players", [])
    return jsonify({"team": team, "players": inj.save_team(team, players)})


@app.get("/api/model")
def model_report():
    return jsonify(get_predictor().report)


if __name__ == "__main__":
    app.run(host=os.environ.get("HOST", "127.0.0.1"), port=int(os.environ.get("PORT", 5000)), debug=False)
