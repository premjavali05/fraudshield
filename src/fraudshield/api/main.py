
import logging
from pathlib import Path

import joblib
from fastapi import FastAPI

from fraudshield.api.schemas import (
    PredictionResponse,
    TransactionRequest,
)
from fraudshield.features.engineering import (
    MODEL_FEATURES,
    build_feature_frame,
)

from fastapi import FastAPI, Query

from fraudshield.api.schemas import AuditRecordResponse
from fraudshield.storage.database import (
    get_recent_predictions,
    initialize_database,
    record_prediction,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("fraudshield")

# main.py -> api -> fraudshield -> src -> project root
PROJECT_ROOT = Path(__file__).resolve().parents[3]
MODEL_PATH = PROJECT_ROOT / "artifacts" / "fraud_model.joblib"

if not MODEL_PATH.is_file():
    raise FileNotFoundError(
        f"Required model artifact not found: {MODEL_PATH}"
    )

# Load once when the application starts, not on each request.
bundle = joblib.load(MODEL_PATH)

MODEL = bundle["model"]
FEATURES = tuple(bundle["features"])
THRESHOLD = float(bundle["threshold"])
MODEL_VERSION = bundle["feature_version"]

# Fail at startup if the saved model differs from our API contract.
if FEATURES != MODEL_FEATURES:
    raise RuntimeError(
        "Saved model features do not match API feature engineering."
    )

if MODEL.n_features_in_ != len(MODEL_FEATURES):
    raise RuntimeError(
        "Saved model expects an unexpected number of features."
    )

app = FastAPI(
    title="FraudShield API",
    description=(
        "Transaction risk scoring using a trained "
        "HistGradientBoostingClassifier."
    ),
    version="1.0.0",
)

initialize_database()

@app.get("/")
def root():
    return {
        "application": "FraudShield",
        "message": "Transaction risk scoring API",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": True,
        "model_version": MODEL_VERSION,
        "feature_count": len(FEATURES),
    }



@app.post("/predict", response_model=PredictionResponse)
def predict(transaction: TransactionRequest):
    """Score a transaction and save prediction metadata."""

    feature_frame = build_feature_frame(transaction)
    feature_frame = feature_frame.loc[:, list(FEATURES)]

    risk_score = float(
        MODEL.predict_proba(feature_frame)[0, 1]
    )

    flag_for_review = risk_score >= THRESHOLD

    # Save the result and obtain a traceable audit ID.
    audit_id = record_prediction(
        transaction_type=transaction.type,
        risk_score=risk_score,
        decision_threshold=THRESHOLD,
        flag_for_review=flag_for_review,
        model_version=MODEL_VERSION,
    )

    logger.info(
        "Prediction recorded: audit_id=%s, review=%s",
        audit_id,
        flag_for_review,
    )

    return PredictionResponse(
        audit_id=audit_id,
        model_version=MODEL_VERSION,
        feature_version=MODEL_VERSION,
        risk_score=risk_score,
        decision_threshold=THRESHOLD,
        flag_for_review=flag_for_review,
        decision=(
            "FLAG_FOR_REVIEW"
            if flag_for_review
            else "NOT_FLAGGED"
        ),
    )


@app.get(
    "/audit/recent",
    response_model=list[AuditRecordResponse],
)
def recent_audit(
    limit: int = Query(default=20, ge=1, le=100),
):
    """Retrieve recent prediction metadata."""
    return get_recent_predictions(limit)

