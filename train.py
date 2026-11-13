"""Train the models and print a hold-out evaluation.

    python train.py                  # uses data/matches.csv (or demo data if missing)
    python train.py --test-days 540  # longer hold-out window
"""
import argparse
import json

from predictor.service import MODEL_PATH, train

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-days", type=int, default=365, help="size of the hold-out window")
    args = ap.parse_args()
    report = train(test_days=args.test_days)
    (MODEL_PATH.parent / "report.json").write_text(json.dumps(report, indent=2))
    # (the server reads the report from the model file itself)
    print(f"\nSaved model to {MODEL_PATH}")
