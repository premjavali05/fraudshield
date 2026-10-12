
import os

import requests
import streamlit as st


API_URL = os.getenv(
    "FRAUDSHIELD_API_URL",
    "http://127.0.0.1:8000",
).rstrip("/")

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
    "FraudShield is a research prototype evaluated on synthetic "
    "PaySim transactions. Its risk score is not a calibrated "
    "real-world fraud probability."
)

# -----------------------------------------------
# Backend connection status
# -----------------------------------------------
try:
    health_response = requests.get(
        f"{API_URL}/health",
        timeout=5,
    )
    health_response.raise_for_status()
    health = health_response.json()

    st.success(
        f"API connected · Model: {health['model_version']}"
    )
except requests.RequestException:
    st.error(
        "Cannot connect to the FraudShield API. "
        "Start FastAPI in another terminal and refresh this page."
    )
    st.stop()


# -----------------------------------------------
# Transaction input
# -----------------------------------------------
st.subheader("Analyse a transaction")

with st.form("transaction_form"):
    col1, col2 = st.columns(2)

    with col1:
        step = st.number_input(
            "Transaction step",
            min_value=1,
            max_value=32767,
            value=100,
            step=1,
            help="PaySim time-step index; not a calendar timestamp.",
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

    with col2:
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


# -----------------------------------------------
# Submit to FastAPI
# -----------------------------------------------
if submitted:
    payload = {
        "step": int(step),
        "type": transaction_type,
        "amount": float(amount),
        "oldbalanceOrg": float(origin_balance),
        "oldbalanceDest": float(destination_balance),
    }

    try:
        response = requests.post(
            f"{API_URL}/predict",
            json=payload,
            timeout=15,
        )
        response.raise_for_status()
        result = response.json()

        st.session_state["last_result"] = result
        st.session_state["last_transaction"] = payload

    except requests.HTTPError:
        try:
            detail = response.json()
        except ValueError:
            detail = response.text

        st.error(f"API rejected the request: {detail}")

    except requests.RequestException as exc:
        st.error(f"Prediction request failed: {exc}")


# -----------------------------------------------
# Display the latest prediction
# -----------------------------------------------
result = st.session_state.get("last_result")

if result:
    st.divider()
    st.subheader("Prediction result")

    score_col, threshold_col, id_col = st.columns(3)

    score = float(result["risk_score"])
    threshold = float(result["decision_threshold"])

    score_col.metric("Model risk score", f"{score:.4f}")
    threshold_col.metric("Review threshold", f"{threshold:.4f}")
    id_col.metric("Audit record ID", result["audit_id"])

    st.progress(
        min(max(score, 0.0), 1.0),
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
        st.json({
            "decision": result["decision"],
            "model_version": result["model_version"],
            "feature_version": result["feature_version"],
            "risk_score": result["risk_score"],
            "decision_threshold": result["decision_threshold"],
            "flag_for_review": result["flag_for_review"],
            "audit_id": result["audit_id"],
        })


# -----------------------------------------------
# Recent audit records
# -----------------------------------------------
st.divider()
st.subheader("Recent prediction history")

if st.button("Refresh history"):
    st.session_state["refresh_audit"] = True

try:
    audit_response = requests.get(
        f"{API_URL}/audit/recent",
        params={"limit": 20},
        timeout=5,
    )
    audit_response.raise_for_status()
    records = audit_response.json()

    if records:
        st.dataframe(
            records,
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("No predictions have been recorded yet.")

except requests.RequestException as exc:
    st.warning(f"Could not retrieve audit history: {exc}")

st.caption("FraudShield · Research prototype · PaySim synthetic dataset")
