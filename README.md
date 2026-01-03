# Customer Bought API (FastAPI + sklearn)

A simple ML inference service that predicts whether a customer will buy (0/1).

## Project structure

- `app/main.py`: FastAPI app + endpoints
- `app/preprocessing.py`: transforms raw input into model-ready features
- `app/artifacts/`: `final_model.joblib` and `feature_columns.joblib`
- `tests/`: pytest tests

## Requirements

- Docker (recommended), OR Python 3.11+

---

## Run with Docker (recommended)

### 1) Build
bash
docker build -t customer-bought-api .

### 2) Run API
bash
docker run -p 8000:8000 customer-bought-api

### 3) Run tests (inside Docker)
bash
docker run --rm customer-bought-api pytest -q

## Test by another computer
### Make sure: 
docker run -p 8000:8000 customer-bought-api 
or 
ontainers port run
### Powercell:
wsl -d Ubuntu (turn into bash)

curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"age": 40, "income": 70000, "country": "USA", "price": 150}'

output:
{"prediction":1,"probability_bought_1":1.0,"model":"DecisionTreeClassifier"}