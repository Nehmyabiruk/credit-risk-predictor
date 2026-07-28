"""Minimal smoke tests – run with: pytest tests/"""

from pathlib import Path

import pandas as pd
import pytest

from src.preprocess import clip_age, engineer_features


def test_clip_age():
    df = pd.DataFrame({"person_age": [22.0, 90.0, 144.0]})
    out = clip_age(df, upper=80)
    assert out["person_age"].max() == 80.0
    assert out["person_age"].min() == 22.0


def test_engineer_features():
    df = pd.DataFrame(
        {
            "loan_amnt": [1000.0, 5000.0],
            "person_income": [50000.0, 20000.0],
            "person_emp_exp": [2, 5],
            "person_age": [25.0, 30.0],
            "cb_person_cred_hist_length": [4.0, 10.0],
        }
    )
    out = engineer_features(df)
    assert "debt_income" in out.columns
    assert "age_ratio" in out.columns
    assert "loan_history" in out.columns
    assert out["debt_income"].iloc[0] == pytest.approx(1000 / 50001)
