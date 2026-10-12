
from fastapi.testclient import TestClient

from fraudshield.api.main import app

client = TestClient(app)


def test_prediction_is_saved_to_audit_trail():
    payload = {
        "step": 100,
        "type": "TRANSFER",
        "amount": 500,
        "oldbalanceOrg": 1000,
        "oldbalanceDest": 0,
    }

    # 1. Submit a transaction for scoring.
    prediction_response = client.post("/predict", json=payload)

    assert prediction_response.status_code == 200

    prediction = prediction_response.json()
    audit_id = prediction["audit_id"]

    # 2. Retrieve recent audit records.
    audit_response = client.get(
        "/audit/recent",
        params={"limit": 20},
    )

    assert audit_response.status_code == 200

    records = audit_response.json()

    # 3. Verify that the prediction was persisted.
    matching_records = [
        record for record in records
        if record["id"] == audit_id
    ]

    assert len(matching_records) == 1

    record = matching_records[0]

    assert record["transaction_type"] == "TRANSFER"
    assert record["risk_score"] == prediction["risk_score"]
    assert record["decision_threshold"] == prediction[
        "decision_threshold"
    ]
    assert record["flag_for_review"] == prediction["flag_for_review"]
    assert record["model_version"] == prediction["model_version"]

    # 4. Ensure raw transaction balances weren't stored.
    assert "amount" not in record
    assert "oldbalanceOrg" not in record
    assert "oldbalanceDest" not in record


def test_audit_rejects_invalid_limit():
    response = client.get(
        "/audit/recent",
        params={"limit": 0},
    )

    assert response.status_code == 422
