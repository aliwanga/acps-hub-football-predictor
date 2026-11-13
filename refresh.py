"""Refresh real data and retrain. Run once, or as a long-lived scheduler.

    python refresh.py                      # fetch + train now
    python refresh.py --loop --hour 3      # every day at 03:00 UTC (Docker 'retrainer' service)
    python refresh.py --skip-fetch         # retrain only

A failed fetch never breaks serving: the existing data and model stay in place
(the model file is replaced atomically and the server hot-reloads it).
"""
import argparse
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent


def run(script: str, *args: str) -> bool:
    print(f"[{datetime.now(timezone.utc):%Y-%m-%d %H:%M:%S}Z] running {script} {' '.join(args)}", flush=True)
    return subprocess.run([sys.executable, str(HERE / script), *args]).returncode == 0


def refresh(skip_fetch: bool, since: int, no_clubs: bool) -> bool:
    if not skip_fetch:
        args = ["--since", str(since)] + (["--no-clubs"] if no_clubs else [])
        if not run("fetch_data.py", *args):
            print("Fetch failed or was rejected; keeping existing data and model.", flush=True)
            return False
    ok = run("train.py")
    print("Retrain " + ("succeeded." if ok else "FAILED; old model still serving."), flush=True)
    return ok


def seconds_until(hour: int) -> float:
    now = datetime.now(timezone.utc)
    nxt = now.replace(hour=hour, minute=0, second=0, microsecond=0)
    if nxt <= now:
        nxt += timedelta(days=1)
    return (nxt - now).total_seconds()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--loop", action="store_true", help="run daily instead of once")
    ap.add_argument("--hour", type=int, default=3, help="UTC hour for the daily run")
    ap.add_argument("--skip-fetch", action="store_true")
    ap.add_argument("--since", type=int, default=2012)
    ap.add_argument("--no-clubs", action="store_true")
    ap.add_argument("--run-now", action="store_true", help="with --loop: also run once at startup")
    a = ap.parse_args()

    if not a.loop:
        sys.exit(0 if refresh(a.skip_fetch, a.since, a.no_clubs) else 1)

    if a.run_now:
        refresh(a.skip_fetch, a.since, a.no_clubs)
    while True:
        wait = seconds_until(a.hour)
        print(f"Next refresh in {wait / 3600:.1f} h (daily at {a.hour:02d}:00 UTC)", flush=True)
        time.sleep(wait)
        refresh(a.skip_fetch, a.since, a.no_clubs)
