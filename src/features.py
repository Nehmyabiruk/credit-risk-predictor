"""Feature engineering shared by batch preprocessing and API inference."""

from __future__ import annotations

import numpy as np
import pandas as pd


def clip_age(df: pd.DataFrame, upper: float = 80) -> pd.DataFrame:
    """Cap implausible ages while leaving the caller's frame untouched."""
    if "person_age" not in df.columns:
        raise ValueError("Missing required column: person_age")

    result = df.copy()
    result["person_age"] = result["person_age"].clip(upper=upper)
    return result


def engineer_features(
    df: pd.DataFrame, income_log_quantile: float = 0.999
) -> pd.DataFrame:
    """Add the engineered columns expected by the trained pipeline.

    The formulas match the existing processed training data.  A small
    denominator offset prevents invalid values for zero income or credit
    history records.
    """
    required = {
        "person_income",
        "loan_amnt",
        "person_emp_exp",
        "person_age",
        "cb_person_cred_hist_length",
    }
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")

    result = df.copy()
    income = result["person_income"].clip(lower=0)
    result["income_log"] = np.log1p(income)
    cap = result["income_log"].quantile(income_log_quantile)
    result["income_log_capped"] = result["income_log"].clip(upper=cap)
    result["debt_income"] = result["loan_amnt"] / (income + 1)
    result["age_ratio"] = result["person_emp_exp"] / result["person_age"].replace(0, np.nan)
    result["age_ratio"] = result["age_ratio"].fillna(0)
    result["loan_history"] = result["loan_amnt"] / (
        result["cb_person_cred_hist_length"].replace(0, np.nan)
    )
    result["loan_history"] = result["loan_history"].fillna(0)
    return result


def preprocess_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the complete deterministic feature transformation."""
    return engineer_features(clip_age(df))
