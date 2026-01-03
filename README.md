# Customer Bought Prediction API

A **Dockerized FastAPI machine learning inference service** that predicts whether a customer will buy a product based on basic demographic and pricing information.

This project demonstrates an **end-to-end ML engineering workflow**:
- data preprocessing
- model training (offline)
- model serialization
- API inference
- automated testing
- Docker packaging

---

## Project Overview

This API predicts a binary outcome:

- `0` → customer is **unlikely** to buy  
- `1` → customer is **likely** to buy  

It uses a **scikit-learn model** trained on processed customer data and exposes predictions via a **FastAPI REST endpoint**.

---

## Project Structure

```
customer-bought-api/
├─ app/
│  ├─ artifacts/
│  │  ├─ final_model.joblib
│  │  └─ feature_columns.joblib
│  ├─ __init__.py
│  ├─ main.py
│  ├─ preprocessing.py
├─ tests/
│  ├─ test_health.py
│  └─ test_predict.py
├─ client.py
├─ Dockerfile
├─ requirements.txt
├─ .dockerignore
└─ README.md
```

---

## ML Design

This project uses an **old-style ML deployment approach**:

- `final_model.joblib` → trained model  
- `feature_columns.joblib` → feature schema  
- `preprocessing.py` → transforms raw input into model-ready features  

⚠️ If preprocessing logic changes, the model **must be retrained** and artifacts regenerated.

---

## Run with Docker

### Build image
```bash
docker build -t customer-bought-api .
```

### Run API
```bash
docker run -p 8000:8000 customer-bought-api
```

Endpoints:
- Health: http://localhost:8000/health
- Docs: http://localhost:8000/docs

---

## Run Tests

```bash
docker run --rm customer-bought-api pytest -q
```

---

## Example Request

```bash
curl -X POST http://localhost:8000/predict   -H "Content-Type: application/json"   -d '{"age":40,"income":70000,"country":"USA","price":150}'
```

Example response:
```json
{
  "prediction": 1,
  "probability_bought_1": 0.73,
  "model": "DecisionTreeClassifier"
}
```

---

## Tech Stack

- Python
- FastAPI
- scikit-learn
- pandas / numpy
- pytest
- Docker

---

## Author

Duy Tran | Aspiring AI and Machine Learning Engineer
