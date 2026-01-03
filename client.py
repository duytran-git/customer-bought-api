import requests

url = "http://localhost:8000/predict"

payload = {"age": 40, "income": 70000, "country": "USA", "price": 150}

r = requests.post(url, json=payload)
print("Status:", r.status_code)
print("Response:", r.json())
