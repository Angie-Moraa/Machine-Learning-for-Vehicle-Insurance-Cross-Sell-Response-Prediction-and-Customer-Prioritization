from fastapi.testclient import TestClient

from app import app

client = TestClient(app)

CUSTOMER = {
    "gender": "Male",
    "customer_date_of_birth": "1977-03-07",
    "driving_license": 1,
    "customer_residence_sub_county": "EMBAKASI",
    "previously_insured": 0,
    "vehicle_year_of_manufacture": 2020,
    "vehicle_damage": "Yes",
    "annual_premium": 38294,
    "agent_name": "Jomo Kenyatta",
    "life_policy_start_date": "2023-12-21",
}


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_predict_high_response_customer():
    r = client.post("/predict", json=CUSTOMER)
    assert r.status_code == 200
    body = r.json()
    assert 0 <= body["probability"] <= 1
    assert body["contact"] is True          # damaged + not insured should be contacted


def test_predict_already_insured_customer():
    r = client.post("/predict", json={**CUSTOMER, "previously_insured": 1, "vehicle_damage": "No"})
    assert r.status_code == 200
    assert r.json()["contact"] is False      # insured + no damage almost never responds


def test_rejects_bad_input():
    r = client.post("/predict", json={**CUSTOMER, "gender": "Unknown"})
    assert r.status_code == 422             # Pydantic validation error