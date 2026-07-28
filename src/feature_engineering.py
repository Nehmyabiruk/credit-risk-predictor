import numpy as np
import pandas as pd

AGE_CAP = 80
INCOME_LOG_QUANTILE = 0.999


def preprocess_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all feature engineering used during training.
    This function should be used by both training and the API.
    """
    df = df.copy()

    # Cap unrealistic ages
    df["person_age"] = df["person_age"].clip(upper=AGE_CAP)

    # Log income
    df["income_log"] = np.log1p(df["person_income"])

    # Cap extreme log incomes
    upper = df["income_log"].quantile(INCOME_LOG_QUANTILE)
    df["income_log_capped"] = np.clip(df["income_log"], None, upper)

    # Engineered features
    df["debt_income"] = (
        df["loan_amnt"] /
        (df["person_income"] + 1)
    )

    df["age_ratio"] = (
        df["person_emp_exp"] /
        df["person_age"].replace(0, np.nan)
    )

    df["loan_history"] = (
        df["loan_amnt"] /
        df["cb_person_cred_hist_length"]
    )

    return df