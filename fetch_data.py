"""Download real historical results into data/matches.csv.

Sources (free, public):
  * International matches since 1872   - github.com/martj42/international_results
  * Club leagues worldwide             - football-data.co.uk (main + "extra" leagues)

    python fetch_data.py                 # everything
    python fetch_data.py --since 2014    # only matches from 2014 onwards
    python fetch_data.py --no-clubs      # international teams only

Check each source's terms before using the data commercially. Sources that fail
to download are skipped with a warning.
"""
import argparse
import io
import urllib.request
from datetime import date

import pandas as pd

from predictor.data import COLUMNS, DATA_DIR, MATCHES_CSV, normalise

UA = {"User-Agent": "Mozilla/5.0 acps-hub-football-predictor"}
INTL_URL = "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"
MAIN_LEAGUES = ["E0", "E1", "E2", "E3", "EC", "SC0", "SC1", "D1", "D2", "I1", "I2", "SP1", "SP2",
                "F1", "F2", "N1", "B1", "P1", "T1", "G1"]
EXTRA_LEAGUES = ["ARG", "AUT", "BRA", "CHN", "DNK", "FIN", "IRL", "JPN", "MEX", "NOR", "POL",
                 "ROU", "RUS", "SWE", "SWZ", "USA"]


def get(url: str) -> pd.DataFrame | None:
    try:
        raw = urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60).read()
        return pd.read_csv(io.StringIO(raw.decode("latin-1")), on_bad_lines="skip")
    except Exception as exc:  # network errors, 404 for seasons not yet started, etc.
        print(f"  skipped {url} ({exc.__class__.__name__}: {exc})")
        return None


def season_codes(first_year: int) -> list[str]:
    today = date.today()
    last = today.year if today.month >= 7 else today.year - 1
    return [f"{y % 100:02d}{(y + 1) % 100:02d}" for y in range(first_year, last + 1)]


def international() -> pd.DataFrame | None:
    df = get(INTL_URL)
    if df is None:
        return None
    return df[COLUMNS]


def main_leagues(first_year: int) -> list[pd.DataFrame]:
    out = []
    for season in season_codes(first_year):
        for code in MAIN_LEAGUES:
            df = get(f"https://www.football-data.co.uk/mmz4281/{season}/{code}.csv")
            if df is None or not {"Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"} <= set(df.columns):
                continue
            out.append(pd.DataFrame({
                "date": pd.to_datetime(df["Date"], dayfirst=True, errors="coerce"),
                "home_team": df["HomeTeam"], "away_team": df["AwayTeam"],
                "home_score": df["FTHG"], "away_score": df["FTAG"],
                "tournament": f"League {code}", "neutral": False}))
    return out


def extra_leagues() -> list[pd.DataFrame]:
    out = []
    for code in EXTRA_LEAGUES:
        df = get(f"https://www.football-data.co.uk/new/{code}.csv")
        if df is None or not {"Date", "Home", "Away", "HG", "AG"} <= set(df.columns):
            continue
        out.append(pd.DataFrame({
            "date": pd.to_datetime(df["Date"], dayfirst=True, errors="coerce"),
            "home_team": df["Home"], "away_team": df["Away"],
            "home_score": df["HG"], "away_score": df["AG"],
            "tournament": f"League {code}", "neutral": False}))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", type=int, default=2012, help="first year to keep")
    ap.add_argument("--no-clubs", action="store_true")
    args = ap.parse_args()

    frames = []
    print("International results...")
    intl = international()
    if intl is not None:
        frames.append(intl)
    if not args.no_clubs:
        print("Club leagues (main)...")
        frames += main_leagues(args.since)
        print("Club leagues (extra)...")
        frames += extra_leagues()
    if not frames:
        raise SystemExit("Nothing downloaded. Check your internet connection / firewall.")

    df = normalise(pd.concat(frames, ignore_index=True))
    df = df[df["date"].dt.year >= args.since].drop_duplicates(
        ["date", "home_team", "away_team"]).reset_index(drop=True)
    DATA_DIR.mkdir(exist_ok=True)
    # Safety: a partial download (e.g. a source was down) must not replace good data.
    if MATCHES_CSV.exists():
        old_rows = sum(1 for _ in open(MATCHES_CSV, encoding="utf-8")) - 1
        if len(df) < 0.8 * old_rows:
            raise SystemExit(f"New download has {len(df):,} rows vs {old_rows:,} existing "
                             f"(<80%). Keeping the existing file.")
    tmp = MATCHES_CSV.with_suffix(".tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(MATCHES_CSV)
    print(f"\nWrote {len(df):,} matches ({df.date.min().date()} to {df.date.max().date()}) "
          f"covering {len(set(df.home_team) | set(df.away_team)):,} teams -> {MATCHES_CSV}")
