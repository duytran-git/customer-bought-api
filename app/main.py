# app/main.py
from __future__ import annotations

from pathlib import Path
from typing import Optional

import joblib
from fastapi import FastAPI
from pydantic import BaseModel, Field

from app.preprocessing import PreprocessConfig, preprocess_one

# -------------------------
# Load artifacts robustly (works no matter where you run from)
# -------------------------
BASE_DIR = Path(__file__).resolve().parent
ARTIFACT_DIR = BASE_DIR / "artifacts"

MODEL_PATH = ARTIFACT_DIR / "final_model.joblib"
FEATURES_PATH = ARTIFACT_DIR / "feature_columns.joblib"

model = joblib.load(MODEL_PATH)
feature_columns = joblib.load(FEATURES_PATH)

config = PreprocessConfig(feature_columns=feature_columns)

app = FastAPI(title="Customer Bought Prediction API", version="1.0")


# -------------------------
# Request schema
# -------------------------
class Customer(BaseModel):
    age: Optional[float] = Field(None, description="Customer age")
    income: Optional[float] = Field(None, description="Yearly income")
    country: Optional[str] = Field(None, description="Country name (e.g., USA, France)")
    price: Optional[float] = Field(None, description="Product price")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(customer: Customer):
    X = preprocess_one(
        age=customer.age,
        income=customer.income,
        country=customer.country,
        price=customer.price,
        config=config,
    )

    pred = int(model.predict(X)[0])

    proba_1 = None
    if hasattr(model, "predict_proba"):
        proba_1 = float(model.predict_proba(X)[0][1])

    return {
        "prediction": pred,  # 0 or 1
        "probability_bought_1": proba_1,  # may be None
        "model": type(model).__name__,
    }
