"""API contract, startup loading, leakage, and failure behavior."""

from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient

from fraud_scoring import api
from fraud_scoring.config import load_config
from fraud_scoring.features import SERVING_FEATURES
from fraud_scoring.predict import LoadedChampion

VALID_REQUEST = {
    "transaction_id": "txn-1001",
    "amount": 125.50,
    "merchant_category": "electronics",
    "hour_of_day": 14,
    "device_risk": 0.23,
}


@pytest.fixture
def client(monkeypatch):
    """Provide a test client backed by a deterministic approved-model stub.

    Args:
        monkeypatch: Pytest fixture used to replace registry model loading.

    Yields:
        Test client, model stub, and model-loading stub.
    """
    model = Mock()
    model.classes_ = [0, 1]
    model.predict_proba.return_value = [[0.13, 0.87]]
    load = Mock(return_value=LoadedChampion(model=model, version="7"))
    monkeypatch.setattr(api, "load_champion", load)
    with TestClient(api.create_app(load_config("configs/dev.yaml"))) as test_client:
        yield test_client, model, load


def test_valid_score_uses_serving_features_and_loads_once(client):
    test_client, model, load = client
    response = test_client.post("/score", json=VALID_REQUEST)
    assert response.status_code == 200
    assert response.json() == {"risk_score": 0.87, "model_version": "7"}
    assert test_client.get("/health").json() == {"status": "healthy", "model_loaded": True}
    assert test_client.post("/score", json=VALID_REQUEST).status_code == 200
    load.assert_called_once()
    features = model.predict_proba.call_args.args[0]
    assert list(features.columns) == list(SERVING_FEATURES)
    assert "transaction_id" not in features.columns
    assert "is_fraud" not in features.columns


def test_model_info_reports_active_non_sensitive_metadata(client):
    test_client, _, _ = client
    response = test_client.get("/model-info")
    assert response.status_code == 200
    assert response.json() == {
        "model_name": "paysafe-fraud-detector",
        "model_alias": "champion",
        "model_version": "7",
        "environment": "dev",
    }


def test_alias_category_is_normalized_before_model_inference(client):
    test_client, model, _ = client
    response = test_client.post("/score", json={**VALID_REQUEST, "merchant_category": "Phone Shop"})
    assert response.status_code == 200
    features = model.predict_proba.call_args.args[0]
    assert features.loc[0, "merchant_category"] == "electronics"


@pytest.mark.parametrize(
    "changes",
    [
        {"amount": 0},
        {"amount": -1},
        {"hour_of_day": 24},
        {"hour_of_day": -1},
        {"hour_of_day": 14.5},
        {"device_risk": -0.1},
        {"device_risk": 1.1},
        {"merchant_category": "unknown"},
        {"transaction_id": "   "},
        {"is_fraud": 1},
        {"chargeback_amount": 10},
    ],
)
def test_invalid_requests_are_rejected(client, changes):
    test_client, model, _ = client
    response = test_client.post("/score", json={**VALID_REQUEST, **changes})
    assert response.status_code == 422
    model.predict_proba.assert_not_called()


def test_missing_field_is_rejected(client):
    test_client, model, _ = client
    request = {key: value for key, value in VALID_REQUEST.items() if key != "device_risk"}
    assert test_client.post("/score", json=request).status_code == 422
    model.predict_proba.assert_not_called()


def test_openapi_has_serving_schema_without_target(client):
    test_client, _, _ = client
    document = test_client.get("/openapi.json").json()
    assert "/score" in document["paths"]
    properties = document["components"]["schemas"]["ScoreRequest"]["properties"]
    assert set(properties) == {*SERVING_FEATURES, "transaction_id"}
    assert "is_fraud" not in properties
    assert test_client.get("/docs").status_code == 200


def test_model_load_failure_prevents_startup(monkeypatch):
    monkeypatch.setattr(api, "load_champion", Mock(side_effect=OSError("store unavailable")))
    with pytest.raises(RuntimeError, match="startup aborted"):
        with TestClient(api.create_app(load_config("configs/dev.yaml"))):
            pass


def test_prediction_failure_is_safe_for_client(client):
    test_client, model, _ = client
    model.predict_proba.side_effect = RuntimeError("private internal error")
    response = test_client.post("/score", json=VALID_REQUEST)
    assert response.status_code == 500
    assert response.json() == {"detail": "Prediction failed"}


def test_uninitialized_model_returns_503(monkeypatch):
    monkeypatch.setattr(api, "load_champion", Mock())
    test_client = TestClient(api.create_app(load_config("configs/dev.yaml")))
    assert test_client.get("/health").status_code == 503
    assert test_client.get("/model-info").status_code == 503
    assert test_client.post("/score", json=VALID_REQUEST).status_code == 503
