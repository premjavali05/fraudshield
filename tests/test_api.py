
from fastapi.testclient import TestClient

from fraudshield.api.main import app

client = TestClient(app)


def test_health():
    response = client.get("/health")

    assert response.status_code == 200

    body = response.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert body["feature_count"] == 13


def test_prediction():
    response = client.post(
        "/predict",
        json={
            "step": 100,
            "type": "TRANSFER",
            "amount": 500,
            "oldbalanceOrg": 1000,
            "oldbalanceDest": 0,
        },
    )

    assert response.status_code == 200

    body = response.json()
    assert 0 <= body["risk_score"] <= 1
    assert body["model_version"] == "v2_full_training_weighted"
    assert body["decision_threshold"] == app_threshold()
    assert isinstance(body["flag_for_review"], bool)

    expected_decision = (
        "FLAG_FOR_REVIEW"
        if body["flag_for_review"]
        else "NOT_FLAGGED"
    )
    assert body["decision"] == expected_decision


def app_threshold():
    # Import the same configured threshold used by the API.
    from fraudshield.api import main
    return main.THRESHOLD


def test_invalid_transaction_type():
    response = client.post(
        "/predict",
        json={
            "step": 100,
            "type": "UNKNOWN",
            "amount": 500,
            "oldbalanceOrg": 1000,
            "oldbalanceDest": 0,
        },
    )

    assert response.status_code == 422


def test_negative_amount_is_rejected():
    response = client.post(
        "/predict",
        json={
            "step": 100,
            "type": "TRANSFER",
            "amount": -500,
            "oldbalanceOrg": 1000,
            "oldbalanceDest": 0,
        },
    )

    assert response.status_code == 422


def test_unexpected_fields_are_rejected():
    response = client.post(
        "/predict",
        json={
            "step": 100,
            "type": "TRANSFER",
            "amount": 500,
            "oldbalanceOrg": 1000,
            "oldbalanceDest": 0,
            "isFraud": 1,
        },
    )

    assert response.status_code == 422
