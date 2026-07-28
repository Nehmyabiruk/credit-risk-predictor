"""Credit-risk API with SQLite prediction history and SHAP explanations."""
from __future__ import annotations
import json
import os
import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
import joblib
import pandas as pd
import xgboost as xgb
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from src.features import preprocess_features

ROOT = Path(__file__).resolve().parents[1]
DATABASE_PATH = Path(os.getenv("DATABASE_PATH", ROOT / "Data" / "predictions.sqlite3"))
model = joblib.load(ROOT / "models" / "full_pipeline.pkl")
preprocessor, classifier = model.named_steps["pre"], model.named_steps["clf"]
FEATURE_NAMES = list(preprocessor.get_feature_names_out())
app = FastAPI(title="Credit Risk Prediction API", description="Loan default prediction service", version="1.1")
origins = [origin.strip() for origin in os.getenv("FRONTEND_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173").split(",") if origin.strip()]
app.add_middleware(CORSMiddleware, allow_origins=origins, allow_credentials=False, allow_methods=["GET", "POST"], allow_headers=["*"])

class LoanData(BaseModel):
    person_age: int = Field(ge=18, le=100)
    person_gender: str
    person_education: str
    person_income: float = Field(gt=0)
    person_emp_exp: int = Field(ge=0)
    person_home_ownership: str
    loan_amnt: float = Field(gt=0)
    loan_intent: str
    loan_int_rate: float = Field(ge=0, le=100)
    loan_percent_income: float = Field(ge=0, le=10)
    cb_person_cred_hist_length: int = Field(ge=0)
    credit_score: int = Field(ge=0, le=1000)
    previous_loan_defaults_on_file: str

class FeatureExplanation(BaseModel):
    feature: str
    shap_value: float
    direction: str

class PredictionResponse(BaseModel):
    id: int
    prediction: int
    default_probability: float
    risk: str
    explanations: list[FeatureExplanation]

def initialize_database() -> None:
    DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(DATABASE_PATH)) as connection:
        connection.execute("CREATE TABLE IF NOT EXISTS predictions (id INTEGER PRIMARY KEY AUTOINCREMENT, created_at TEXT NOT NULL, input_data TEXT NOT NULL, prediction INTEGER NOT NULL, default_probability REAL NOT NULL, risk TEXT NOT NULL, explanations TEXT NOT NULL)")
        connection.commit()

@app.on_event("startup")
def startup() -> None: initialize_database()

def original_feature_name(name: str) -> str:
    name = name.split("__", maxsplit=1)[-1]
    for field in LoanData.model_fields:
        if name == field or name.startswith(f"{field}_"): return field
    return name

def explain(features) -> list[dict[str, object]]:
    values = classifier.get_booster().predict(xgb.DMatrix(features, feature_names=FEATURE_NAMES), pred_contribs=True)[0][:-1]
    grouped: dict[str, float] = {}
    for name, value in zip(FEATURE_NAMES, values, strict=True):
        field = original_feature_name(name); grouped[field] = grouped.get(field, 0.0) + float(value)
    return [{"feature": field.replace("_", " ").title(), "shap_value": round(value, 4), "direction": "increases default risk" if value > 0 else "reduces default risk"} for field, value in sorted(grouped.items(), key=lambda item: abs(item[1]), reverse=True)[:5]]

def save_prediction(data: LoanData, prediction: int, probability: float, risk: str, explanations: list[dict[str, object]]) -> int:
    with closing(sqlite3.connect(DATABASE_PATH)) as connection:
        cursor = connection.execute("INSERT INTO predictions (created_at, input_data, prediction, default_probability, risk, explanations) VALUES (?, ?, ?, ?, ?, ?)", (datetime.now(timezone.utc).isoformat(), data.model_dump_json(), prediction, probability, risk, json.dumps(explanations)))
        connection.commit(); return int(cursor.lastrowid)

@app.get("/")
def home() -> dict[str, str]: return {"message": "Credit Risk API is running"}

@app.post("/predict", response_model=PredictionResponse)
def predict(data: LoanData) -> PredictionResponse:
    frame = preprocess_features(pd.DataFrame([data.model_dump()])); transformed = preprocessor.transform(frame)
    prediction = int(classifier.predict(transformed)[0]); probability = round(float(classifier.predict_proba(transformed)[0][1]), 4); risk = "Default" if prediction else "No Default"
    explanations = explain(transformed); prediction_id = save_prediction(data, prediction, probability, risk, explanations)
    return PredictionResponse(id=prediction_id, prediction=prediction, default_probability=probability, risk=risk, explanations=explanations)

@app.get("/predictions", response_model=list[PredictionResponse])
def prediction_history(limit: int = 20) -> list[PredictionResponse]:
    safe_limit = min(max(limit, 1), 100)
    with closing(sqlite3.connect(DATABASE_PATH)) as connection: rows = connection.execute("SELECT id, prediction, default_probability, risk, explanations FROM predictions ORDER BY id DESC LIMIT ?", (safe_limit,)).fetchall()
    return [PredictionResponse(id=row[0], prediction=row[1], default_probability=row[2], risk=row[3], explanations=json.loads(row[4])) for row in rows]
