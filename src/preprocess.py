"""
preprocess.py
-------------
Data cleaning pipeline for the Loan Credit Risk dataset.

Responsibilities:
1. Load raw data
2. Basic quality checks
3. Remove extreme outliers
4. Apply feature engineering
5. Save cleaned dataset
6. Save X and y separately

Usage:
    python -m src.preprocess
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from src.features import clip_age, engineer_features, preprocess_features

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

RAW_PATH = ROOT / "data" / "raw" / "loan_data.csv"

PROCESSED_DIR = ROOT / "data" / "processed"

CLEAN_PATH = PROCESSED_DIR / "loan_data_clean.csv"
X_RAW_PATH = PROCESSED_DIR / "X_raw_features.csv"
Y_PATH = PROCESSED_DIR / "y_target.csv"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_raw(path: Path = RAW_PATH) -> pd.DataFrame:
    """Load raw dataset."""
    logger.info("Loading raw data from %s", path)

    df = pd.read_csv(path)

    logger.info("Dataset shape: %s", df.shape)

    return df


def basic_qc(df: pd.DataFrame) -> None:
    """Basic quality checks."""

    duplicates = df.duplicated().sum()
    nulls = df.isnull().sum().sum()

    logger.info("Duplicates: %d", duplicates)
    logger.info("Null cells: %d", nulls)

    if duplicates or nulls:
        logger.warning("Dataset contains duplicates or missing values.")


def remove_extreme_income_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove unrealistic income values.
    """

    df = df.copy()

    mask = df["person_income"] < 3_000_000

    removed = (~mask).sum()

    if removed:
        logger.info("Removed %d extreme income outliers.", removed)

    return df.loc[mask].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------------------------

def run_preprocess(
    raw_path: Path = RAW_PATH,
    clean_path: Path = CLEAN_PATH,
    x_path: Path = X_RAW_PATH,
    y_path: Path = Y_PATH,
) -> pd.DataFrame:

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # Load
    df = load_raw(raw_path)

    # QC
    basic_qc(df)

    # Cleaning
    df = remove_extreme_income_outliers(df)

    # Feature Engineering
    df = preprocess_features(df)

    logger.info("Final dataset shape: %s", df.shape)

    # Save cleaned dataset
    df.to_csv(clean_path, index=False)

    logger.info("Saved cleaned dataset -> %s", clean_path)

    # Split Features / Target
    X = df.drop(columns=["loan_status"])
    y = df["loan_status"]

    # Save
    X.to_csv(x_path, index=False)
    y.to_csv(y_path, index=False)

    logger.info("Saved X -> %s", x_path)
    logger.info("Saved y -> %s", y_path)

    return df


if __name__ == "__main__":
    run_preprocess()
