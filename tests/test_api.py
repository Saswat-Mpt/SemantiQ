import pytest
from fastapi.testclient import TestClient
from app.api import app


@pytest.fixture(scope="module")
def client():
    return TestClient(app)


def test_live_probe(client):
    response = client.get("/live")
    assert response.status_code == 200
    assert response.json() == {"status": "alive"}


def test_ready_probe(client):
    response = client.get("/ready")
    assert response.status_code == 200
    assert response.json()["status"] == "ready"


def test_model_info_endpoint(client):
    response = client.get("/api/v1/model-info")
    assert response.status_code == 200
    data = response.json()
    assert data["model_name"] == "SemantIQ"
    assert data["num_features"] == 19
    assert "threshold_policy" in data


def test_predict_endpoint(client):
    payload = {
        "question1": "What is the best way to learn Python?",
        "question2": "How can I study Python programming effectively?",
    }
    response = client.post("/api/v1/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "decision" in data
    assert "score" in data
    assert "critical_tokens" in data
    assert response.headers.get("X-Request-ID") is not None


def test_batch_predict_endpoint(client):
    payload = {
        "pairs": [
            {
                "question1": "How to learn machine learning?",
                "question2": "What are best resources for machine learning?",
            },
            {
                "question1": "Why is the sky blue?",
                "question2": "How do commercial planes fly?",
            },
        ]
    }
    response = client.post("/api/v1/batch-predict", json=payload)
    assert response.status_code == 200
    results = response.json()
    assert len(results) == 2
    assert results[0]["score"] > results[1]["score"]
