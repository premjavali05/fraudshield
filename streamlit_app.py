
import logging
import sys
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
from pydantic import ValidationError

# Make the src/ package available locally and on Streamlit Cloud.
ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fraudshield.api.schemas import TransactionRequest
from fraudshield.features.engineering import (
    MODEL_FEATURES,
    build_feature_frame,
)
from fraudshield.storage.database import (
    get_recent_predictions,
    initialize_database,
    record_prediction,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("fraudshield.streamlit")

MODEL_PATH = ROOT / "artifacts" / "fraud_model.joblib"


@st.cache_resource(show_spinner="Loading the trained FraudShield model...")
def load_model_bundle():
    """Load the saved model once and validate its feature contract."""
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Model artifact not found: {MODEL_PATH}"
        )

    bundle = joblib.load(MODEL_PATH)

    if tuple(bundle["features"]) != MODEL_FEATURES:
        raise RuntimeError(
            "Saved model features do not match feature engineering."
        )

    if bundle["model"].n_features_in_ != len(MODEL_FEATURES):
        raise RuntimeError(
            "Unexpected number of input features in saved model."
        )

    return bundle


@st.cache_resource
def setup_audit_database():
    """Initialize the local audit database once per app process."""
    initialize_database()
    return True


st.set_page_config(
    page_title="FraudShield",
    page_icon="🛡️",
    layout="wide",
)

st.title("🛡️ FraudShield")
st.caption(
    "Financial transaction risk scoring powered by machine learning"
)

st.info(
    "Research prototype evaluated on synthetic PaySim data. "
    "The model risk score is not a calibrated real-world "
    "fraud probability."
)

# Load the actual trained model—no HTTP API call required.
try:
    bundle = load_model_bundle()
    setup_audit_database()
except Exception:
    logger.exception("FraudShield initialization failed")
    st.error(
        "FraudShield could not load its model or initialize storage. "
        "Check the application logs."
    )
    st.stop()

MODEL = bundle["model"]
FEATURES = list(bundle["features"])
THRESHOLD = float(bundle["threshold"])
MODEL_VERSION = bundle["feature_version"]

st.success(
    f"Model loaded · {MODEL_VERSION} · "
    f"{len(FEATURES)} validated features"
)

# --------------------------------------------------
# Transaction input
# --------------------------------------------------
st.subheader("Analyse a transaction")

with st.form("transaction_form"):
    left, right = st.columns(2)

    with left:
        step = st.number_input(
            "Transaction step",
            min_value=1,
            max_value=32767,
            value=100,
            step=1,
            help="PaySim time-step index, not a calendar timestamp.",
        )

        transaction_type = st.selectbox(
            "Transaction type",
            ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"],
            index=4,
        )

        amount = st.number_input(
            "Transaction amount",
            min_value=0.0,
            value=500.0,
            step=100.0,
            format="%.2f",
        )

    with right:
        origin_balance = st.number_input(
            "Origin balance before transaction",
            min_value=0.0,
            value=1000.0,
            step=100.0,
            format="%.2f",
        )

        destination_balance = st.number_input(
            "Destination balance before transaction",
            min_value=0.0,
            value=0.0,
            step=100.0,
            format="%.2f",
        )

    submitted = st.form_submit_button(
        "Score transaction",
        type="primary",
        use_container_width=True,
    )

# --------------------------------------------------
# Inference directly in Streamlit
# --------------------------------------------------
if submitted:
    try:
        transaction = TransactionRequest(
            step=int(step),
            type=transaction_type,
            amount=float(amount),
            oldbalanceOrg=float(origin_balance),
            oldbalanceDest=float(destination_balance),
        )

        # Exactly the same feature-engineering function as FastAPI.
        features = build_feature_frame(transaction)
        features = features.loc[:, FEATURES]

        risk_score = float(
            MODEL.predict_proba(features)[0, 1]
        )

        flag_for_review = risk_score >= THRESHOLD

        audit_id = record_prediction(
            transaction_type=transaction.type,
            risk_score=risk_score,
            decision_threshold=THRESHOLD,
            flag_for_review=flag_for_review,
            model_version=MODEL_VERSION,
        )

        st.session_state["last_result"] = {
            "audit_id": audit_id,
            "model_version": MODEL_VERSION,
            "risk_score": risk_score,
            "decision_threshold": THRESHOLD,
            "flag_for_review": flag_for_review,
            "decision": (
                "FLAG_FOR_REVIEW"
                if flag_for_review
                else "NOT_FLAGGED"
            ),
        }

        logger.info(
            "Prediction recorded: audit_id=%s",
            audit_id,
        )

    except ValidationError as exc:
        st.error(f"Invalid transaction: {exc}")

    except Exception:
        logger.exception("Prediction failed")
        st.error(
            "Prediction failed. Check the application logs."
        )

# --------------------------------------------------
# Prediction result
# --------------------------------------------------
result = st.session_state.get("last_result")

if result:
    st.divider()
    st.subheader("Prediction result")

    score_col, threshold_col, id_col = st.columns(3)

    score = result["risk_score"]
    threshold = result["decision_threshold"]

    score_col.metric("Model risk score", f"{score:.4f}")
    threshold_col.metric("Review threshold", f"{threshold:.4f}")
    id_col.metric("Audit record ID", result["audit_id"])

    st.progress(
        score,
        text="Model score (0–1; not a calibrated probability)",
    )

    if result["flag_for_review"]:
        st.warning(
            "🚨 FLAG FOR REVIEW — the model score meets "
            "or exceeds the configured threshold."
        )
    else:
        st.success(
            "No review flag — the score is below the threshold. "
            "This does not guarantee that the transaction is legitimate."
        )

    with st.expander("Prediction details"):
        st.json(result)

# --------------------------------------------------
# Recent audit history
# --------------------------------------------------
st.divider()
st.subheader("Recent prediction history")

try:
    records = get_recent_predictions(limit=20)

    if records:
        history = pd.DataFrame(records)
        history["flag_for_review"] = (
            history["flag_for_review"]
            .map({1: "YES", 0: "NO"})
        )

        st.dataframe(
            history,
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("No predictions have been recorded yet.")

except Exception:
    logger.exception("Could not load prediction history")
    st.warning("Could not load recent prediction history.")

st.caption(
    "FraudShield · Synthetic PaySim dataset · "
    "Scores are for research and demonstration purposes."
)
