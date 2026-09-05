"""Taste-test the restaurant door (api/main.py).

6yo version: /health = 'are you awake?', /predict = 'how long is my ride?'
We use TestClient with mocked model (from conftest) so no real training needed.
Covers: monkeypatch/mock usage required by mini-p1.md Step 06.
"""

from unittest.mock import MagicMock

import numpy as np


def test_health_returns_200(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert "status" in r.json()
    assert r.headers.get("X-Request-ID")  # correlation ID echoed


def test_predict_happy_path(client, sample_features):
    # API expects trip_distance + optional fields; sample_features has all
    payload = {"trip_distance": 2.5, "PULocationID": "100", "DOLocationID": "200", "fare_amount": 10.0}
    r = client.post("/predict", json=payload)
    assert r.status_code == 200, r.text
    body = r.json()
    assert "prediction" in body and "latency_ms" in body
    assert isinstance(body["prediction"], float)
    assert r.headers.get("X-Request-ID")


def test_predict_invalid_returns_422(client):
    # Missing required trip_distance -> 422, not 500 stack trace
    r = client.post("/predict", json={"fare_amount": 10.0})
    assert r.status_code == 422


def test_response_schema_matches(client, sample_features):
    r = client.post("/predict", json={"trip_distance": 3.0})
    assert r.status_code == 200
    body = r.json()
    # PredictResponse must at least have these; may also have model_version, correlation_id
    assert "prediction" in body and "latency_ms" in body
    assert isinstance(body["prediction"], float)


def test_metadata_returns_id_card(client):
    r = client.get("/metadata")
    assert r.status_code == 200
    body = r.json()
    assert "model_version" in body and "feature_names" in body and "artifact_hash" in body


def test_predict_batch(client):
    r = client.post("/predict/batch", json={"rides": [{"trip_distance": 1.0}, {"trip_distance": 2.0}]})
    assert r.status_code == 200
    assert len(r.json()["predictions"]) == 2


def test_predict_uses_mock_not_real_training(client, monkeypatch):
    """Proves mock usage: patch predictor.predict_one, ensure API calls it."""
    import prodml.api.main as main_module

    called = {}

    def _fake_predict_one(self, features):
        called["yes"] = True
        return 42.0

    monkeypatch.setattr(main_module.DurationPredictor, "predict_one", _fake_predict_one)
    r = client.post("/predict", json={"trip_distance": 1.0})
    assert r.status_code == 200
    assert r.json()["prediction"] == 42.0
    assert called.get("yes") is True
