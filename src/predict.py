"""
predict.py
----------
Simple inference helper that loads the full sklearn pipeline
(encoder + scaler + XGB) and returns default probability.

Usage (CLI):
    python -m src.predict --csv path/to/new_applicants.csv

Or from Python:
    from src.predict import load_pipeline, predict_proba
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PIPELINE_PATH = ROOT / "models" / "full_pipeline.pkl"


def load_pipeline(path: Path = PIPELINE_PATH):
    if not path.exists():
        raise FileNotFoundError(
            f"Pipeline not found at {path}. Run `python -m src.train` first."
        )
    return joblib.load(path)


def predict_proba(df: pd.DataFrame, pipeline=None) -> pd.Series:
    """
    Return probability of default (loan_status = 1) for each row.
    `df` must contain the same columns that were present at training time
    (see NUMERIC + CATEGORICAL in train.py).
    """
    if pipeline is None:
        pipeline = load_pipeline()
    proba = pipeline.predict_proba(df)[:, 1]
    return pd.Series(proba, index=df.index, name="default_proba")


def main():
    parser = argparse.ArgumentParser(description="Score new loan applications")
    parser.add_argument("--csv", required=True, help="CSV of applicants")
    parser.add_argument(
        "--out",
        default="scored.csv",
        help="Output path for scored file",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    proba = predict_proba(df)
    out = df.copy()
    out["default_proba"] = proba
    out["predicted_default"] = (proba >= 0.5).astype(int)
    out.to_csv(args.out, index=False)
    print(f"Scored {len(out)} rows → {args.out}")


if __name__ == "__main__":
    main()
