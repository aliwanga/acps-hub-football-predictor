#!/bin/sh
set -e
# First boot on an empty volume: try real data, fall back to the demo dataset.
if [ ! -f model/predictor.joblib ]; then
  if [ ! -f data/matches.csv ]; then
    echo "No data yet: trying to download real results..."
    python fetch_data.py || echo "Download failed; training on DEMO data (fictional teams)."
  fi
  python train.py
fi
exec gunicorn --bind 0.0.0.0:5000 --workers "${WEB_WORKERS:-2}" --timeout 60 app:app
