"""Optional Streamlit client for the FastAPI fraud-scoring service.

The UI deliberately has no MLflow, model, feature-engineering, or promotion
code. It only sends the authorization-time request to the public API contract.
"""

from __future__ import annotations

import os
from typing import Any

import requests
import streamlit as st

DEFAULT_API_URL = "http://127.0.0.1:8000"
REQUEST_TIMEOUT_SECONDS = 10


def _api_url() -> str:
    return os.getenv("FRAUD_SCORING_API_URL", DEFAULT_API_URL).rstrip("/")


def _get_json(path: str) -> tuple[dict[str, Any] | None, str | None]:
    try:
        response = requests.get(f"{_api_url()}{path}", timeout=REQUEST_TIMEOUT_SECONDS)
        response.raise_for_status()
        return response.json(), None
    except requests.RequestException as exc:
        return None, str(exc)


def _risk_label(risk_score: float) -> str:
    if risk_score <= 0.33:
        return "Low"
    if risk_score <= 0.66:
        return "Medium"
    return "High"


def main() -> None:
    st.set_page_config(page_title="PaySafe Fraud Scoring", page_icon="🛡️")
    st.title("PaySafe Fraud Scoring")
    st.caption("Optional client for the FastAPI `/score` endpoint")

    api_url = _api_url()
    st.sidebar.subheader("API status")
    st.sidebar.caption(api_url)
    health, health_error = _get_json("/health")
    if health_error:
        st.sidebar.error("API is unreachable")
        st.sidebar.caption(health_error)
    else:
        st.sidebar.success(f"{health['status']} (model loaded: {health['model_loaded']})")

    st.sidebar.subheader("Active model")
    model_info, model_error = _get_json("/model-info")
    if model_error:
        st.sidebar.info("Model information is unavailable until the API is healthy.")
    elif model_info:
        st.sidebar.write(f"**Model:** {model_info['model_name']}")
        st.sidebar.write(f"**Version:** {model_info['model_version']}")
        st.sidebar.write(f"**Alias:** {model_info['model_alias']}")
        st.sidebar.write(f"**Environment:** {model_info['environment']}")

    with st.form("score-transaction"):
        transaction_id = st.text_input("Transaction ID", value="txn-demo-001")
        amount = st.number_input("Amount", min_value=0.01, value=125.50, step=0.01)
        merchant_category = st.text_input("Merchant category", value="electronics")
        hour_of_day = st.number_input("Hour of day", min_value=0, max_value=23, value=14, step=1)
        device_risk = st.number_input(
            "Device risk", min_value=0.0, max_value=1.0, value=0.23, step=0.01
        )
        submitted = st.form_submit_button("Score Transaction")

    if not submitted:
        return

    payload = {
        "transaction_id": transaction_id,
        "amount": amount,
        "merchant_category": merchant_category,
        "hour_of_day": hour_of_day,
        "device_risk": device_risk,
    }
    try:
        response = requests.post(f"{api_url}/score", json=payload, timeout=REQUEST_TIMEOUT_SECONDS)
    except requests.RequestException as exc:
        st.error("Prediction request failed because the API is unreachable.")
        st.caption(str(exc))
        return

    if response.status_code != 200:
        detail = response.json().get("detail", "The API rejected the request.")
        st.error(f"Prediction request failed ({response.status_code}): {detail}")
        return

    result = response.json()
    risk_score = float(result["risk_score"])
    st.success("Prediction request completed")
    st.metric("Fraud risk score", f"{risk_score:.4f}")
    st.write(f"**Model version:** {result['model_version']}")
    st.caption(
        f"UI-only interpretation: {_risk_label(risk_score)} risk. "
        "This is not a business approval threshold."
    )


if __name__ == "__main__":
    main()
