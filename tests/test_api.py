
import pytest
from fastapi.testclient import TestClient
from api.main import app


SAMPLE = {
    "MONTH": 7,
    "DAY": 15,
    "DAY_OF_WEEK": 3,
    "AIRLINE": "AA",
    "ORIGIN_AIRPORT": "JFK",
    "DESTINATION_AIRPORT": "LAX",
    "SCHEDULED_TIME": 360,
    "DISTANCE": 2475,
    "scheduled_departure": 1830
}


@pytest.fixture(scope="module")
def client():
    # Context manager executes FastAPI startup and shutdown events
    with TestClient(app) as test_client:
        yield test_client


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_metadata(client):
    response = client.get("/metadata")
    assert response.status_code == 200
    assert response.json()["model_loaded"] is True


def test_prediction(client):
    response = client.post("/predict", json=SAMPLE)

    assert response.status_code == 200

    data = response.json()
    assert data["prediction"] in (0, 1)
    assert 0 <= data["delay_probability"] <= 1
    assert "prediction_label" in data


def test_invalid_departure_time(client):
    payload = SAMPLE.copy()
    payload["scheduled_departure"] = 1865

    response = client.post("/predict", json=payload)

    assert response.status_code == 422


def test_batch_prediction(client):
    response = client.post(
        "/predict/batch",
        json={"flights": [SAMPLE, SAMPLE]}
    )

    assert response.status_code == 200
    assert response.json()["count"] == 2
