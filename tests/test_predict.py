# tests/test_predict.py
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_predict_happy_path():
    payload = {"age": 40, "income": 70000, "country": "USA", "price": 150}
    r = client.post("/predict", json=payload)
    assert r.status_code == 200

    data = r.json()
    assert "prediction" in data
    assert data["prediction"] in (0, 1)

    assert "probability_bought_1" in data
    if data["probability_bought_1"] is not None:
        assert 0.0 <= data["probability_bought_1"] <= 1.0


def test_predict_missing_fields_ok():
    # Optional fields should not crash preprocessing
    payload = {"country": "France"}
    r = client.post("/predict", json=payload)
    assert r.status_code == 200
    data = r.json()
    assert data["prediction"] in (0, 1)
