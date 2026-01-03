from typing import Optional
import numpy as np
import pandas as pd
import joblib

from fastapi import FastAPI
from pydantic import BaseModel, Field

# -------------------------
# Load model + feature schema at startup
# -------------------------
MODEL_PATH = "models/final_model.joblib"
FEATURES_PATH = "models/feature_columns.joblib"

model = joblib.load(MODEL_PATH)
feature_columns = joblib.load(FEATURES_PATH)

app = FastAPI(title="Customer Bought Prediction API", version="1.0")


# -------------------------
# Request schema (what JSON must look like)
# -------------------------
class Customer(BaseModel):
    age: Optional[float] = Field(None, description="Customer age")
    income: Optional[float] = Field(None, description="Yearly income")
    country: Optional[str] = Field(None, description="Country name (e.g., USA, France)")
    price: Optional[float] = Field(None, description="Product price")


def preprocess_one(customer: Customer) -> pd.DataFrame:
    """
    Convert raw JSON -> model-ready features with the SAME logic used in training:
    - fill missing values (basic)
    - feature engineering: income_log1p, income_per_age
    - one-hot encode categoricals
    - align columns to training feature_columns
    """

    # 1) Build a 1-row DataFrame (raw)
    row = pd.DataFrame(
        [
            {
                "age": customer.age,
                "income": customer.income,
                "country": customer.country,
                "price": customer.price,
            }
        ]
    )

    # 2) Fill missing values in the simplest safe way
    # NOTE: In production, you typically save imputers from training.
    # Here we use conservative defaults.
    if row["age"].isna().any():
        row["age"] = row["age"].fillna(0)
    if row["income"].isna().any():
        row["income"] = row["income"].fillna(0)
    if row["price"].isna().any():
        row["price"] = row["price"].fillna(0)
    if row["country"].isna().any():
        row["country"] = row["country"].fillna("Unknown")

    # 3) Feature engineering (must match training)
    row["income_log1p"] = np.log1p(row["income"])
    row["income_per_age"] = row["income"] / np.maximum(row["age"], 1)

    # 4) One-hot encode country (and any other categoricals if present)
    row_enc = pd.get_dummies(row)

    # 5) Align columns exactly to training schema
    row_enc = row_enc.reindex(columns=feature_columns, fill_value=0)

    return row_enc


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(customer: Customer):
    X = preprocess_one(customer)

    pred = int(model.predict(X)[0])

    # Not all sklearn models support predict_proba, but yours should.
    proba = None
    if hasattr(model, "predict_proba"):
        proba = float(model.predict_proba(X)[0][1])  # probability of class 1

    return {
        "prediction": pred,  # 0 or 1
        "probability_bought_1": proba,  # may be None if model doesn't support it
        "model": type(model).__name__,
    }


"""
uvicorn app:app --reload --host 0.0.0.0 --port 8000

Open:

http://localhost:8000/health

http://localhost:8000/docs (interactive Swagger UI)

Example curl command on new bash terminal:
 
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"age": 23, "income": 10000, "country": "USA", "price": 150}'

# Expected response:
{
  "prediction": 1,
  "probability_bought_1": 0.85,
  "model": "LogisticRegression"
}

"""
