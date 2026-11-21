from fastapi.testclient import TestClient
from app import app

client = TestClient(app)

def test_home_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "online"

def test_triage_billing_complaint():
    response = client.post(
        "/triage",
        json={"complaint_text": "I was charged twice on my monthly postpaid bill"}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["department"] == "Billing"
    assert data["urgency"] == "High"
    assert 0.0 <= data["confidence"] <= 1.0

def test_empty_complaint_rejected():
    response = client.post("/triage", json={"complaint_text": "   "})
    assert response.status_code == 400