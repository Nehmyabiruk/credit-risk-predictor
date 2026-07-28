"""
train.py
--------
Train / tune an XGBoost classifier for loan default prediction.

Pipeline:
  1. Load cleaned data
  2. Train / test split (stratified)
  3. One-hot encode categoricals + StandardScaler on numerics
  4. RandomizedSearchCV on XGBClassifier
  5. Evaluate on hold-out set
  6. Persist model + scaler + feature names

Usage:
    python -m src.train
"""

from __future__ import annotations

import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier
from src.config import CATEGORICAL, NUMERIC, TARGET

# ---------------------------------------------------------------------------
# Paths & config
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
CLEAN_PATH = ROOT / "data" / "processed" / "loan_data_clean.csv"
MODEL_DIR = ROOT / "models"
MODEL_PATH = MODEL_DIR / "credit-risk-model.pkl"
SCALER_PATH = MODEL_DIR / "scaler.pkl"          # kept for backward compat
PIPELINE_PATH = MODEL_DIR / "full_pipeline.pkl"  # preferred: encoder+scaler+model

RANDOM_STATE = 42
TEST_SIZE = 0.2

#CATEGORICAL = [
  #  "person_gender",
   # "person_education",
   #"person_home_ownership",
    #"loan_intent",
    #"previous_loan_defaults_on_file",
#]

#NUMERIC = [
 #   "person_age",
  #  "person_income",
   # "person_emp_exp",
   # "loan_amnt",
    #"loan_int_rate",
    #"loan_percent_income",
    #"cb_person_cred_hist_length",
    #"credit_score",
    #"income_log",
    #"income_log_capped",
    #"debt_income",
    #"age_ratio",
    #"loan_history",
#]

# Hyper-parameter search space (matches notebook)
PARAM_DIST = {
    "n_estimators": [100, 200, 300, 400, 500],
    "max_depth": [3, 5, 7, 9],
    "learning_rate": [0.01, 0.05, 0.1],
    "subsample": [0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.6, 0.8, 1.0],
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def load_clean(path: Path = CLEAN_PATH) -> tuple[pd.DataFrame, pd.Series]:
    logger.info("Loading cleaned data from %s", path)
    df = pd.read_csv(path)
    y = df[TARGET]
    X = df.drop(columns=[TARGET])
    logger.info("X shape %s | positive rate %.3f", X.shape, y.mean())
    return X, y


def split(X: pd.DataFrame, y: pd.Series):
    return train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
        shuffle=True,
    )


# ---------------------------------------------------------------------------
# Preprocessing transformers
# ---------------------------------------------------------------------------
def build_preprocessor() -> ColumnTransformer:
    """
    One-hot encode categoricals (drop first to avoid dummy trap)
    and standard-scale all numeric features.
    """
    return ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", drop="first", sparse_output=False),
                CATEGORICAL,
            ),
            ("num", StandardScaler(), NUMERIC),
        ],
        remainder="drop",
    )


# ---------------------------------------------------------------------------
# Model + search
# ---------------------------------------------------------------------------
def build_search(preprocessor: ColumnTransformer) -> RandomizedSearchCV:
    xgb = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        enable_categorical=False,
    )

    # We search only over the classifier hyper-params;
    # the preprocessor is fixed.
    pipe = Pipeline(
        steps=[
            ("pre", preprocessor),
            ("clf", xgb),
        ]
    )

    # Prefix parameter names for the pipeline
    param_dist = {f"clf__{k}": v for k, v in PARAM_DIST.items()}

    search = RandomizedSearchCV(
        estimator=pipe,
        param_distributions=param_dist,
        n_iter=20,
        scoring="roc_auc",
        cv=5,
        n_jobs=-1,
        random_state=RANDOM_STATE,
        verbose=1,
    )
    return search


# ---------------------------------------------------------------------------
# Train / evaluate / save
# ---------------------------------------------------------------------------
def train():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    X, y = load_clean()
    X_train, X_test, y_train, y_test = split(X, y)

    logger.info(
        "Train: %s | Test: %s | pos rate train=%.3f",
        X_train.shape,
        X_test.shape,
        y_train.mean(),
    )

    preprocessor = build_preprocessor()
    search = build_search(preprocessor)

    logger.info("Starting RandomizedSearchCV …")
    search.fit(X_train, y_train)

    best = search.best_estimator_
    logger.info("Best params: %s", search.best_params_)
    logger.info("Best CV ROC-AUC: %.4f", search.best_score_)

    # Hold-out evaluation
    y_pred = best.predict(X_test)
    y_proba = best.predict_proba(X_test)[:, 1]

    logger.info("\n%s", classification_report(y_test, y_pred, digits=3))
    logger.info("Hold-out ROC-AUC: %.4f", roc_auc_score(y_test, y_proba))

    # Persist artefacts
    joblib.dump(best, PIPELINE_PATH)
    logger.info("Saved full pipeline → %s", PIPELINE_PATH)

    # Also keep the raw XGB + a standalone scaler for notebook compatibility
    # (the notebook saved them separately)
    clf = best.named_steps["clf"]
    joblib.dump(clf, MODEL_PATH)
    # Reconstruct a scaler that was fitted on the numeric columns only
    # (useful if someone wants to re-use the old inference code)
    num_scaler = best.named_steps["pre"].named_transformers_["num"]
    joblib.dump(num_scaler, SCALER_PATH)
    logger.info("Saved XGB model → %s", MODEL_PATH)
    logger.info("Saved numeric scaler → %s", SCALER_PATH)

    return best


if __name__ == "__main__":
    train()
