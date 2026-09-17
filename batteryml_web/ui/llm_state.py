from __future__ import annotations

from dataclasses import dataclass

import streamlit as st


LLM_STATUS_NOT_CONFIGURED = "not_configured"
LLM_STATUS_CONFIGURED = "configured"
LLM_STATUS_CONNECTED = "connected"
LLM_STATUS_FAILED = "failed"


@dataclass
class LLMState:
    base_url: str
    api_key: str
    model: str
    status: str
    status_message: str


def ensure_llm_state_defaults() -> None:
    st.session_state.setdefault("llm_base_url", "")
    st.session_state.setdefault("llm_api_key", "")
    st.session_state.setdefault("llm_model", "")
    st.session_state.setdefault("llm_status", LLM_STATUS_NOT_CONFIGURED)
    st.session_state.setdefault("llm_status_message", "")


def is_llm_configured() -> bool:
    ensure_llm_state_defaults()
    return (
        bool(st.session_state.get("llm_base_url", "").strip())
        and bool(st.session_state.get("llm_api_key", "").strip())
        and bool(st.session_state.get("llm_model", "").strip())
    )


def refresh_llm_status_from_config() -> None:
    ensure_llm_state_defaults()
    if not is_llm_configured():
        st.session_state["llm_status"] = LLM_STATUS_NOT_CONFIGURED
        if not st.session_state.get("llm_status_message"):
            st.session_state["llm_status_message"] = "Missing Base URL, API key, or model."
        return
    if st.session_state.get("llm_status") in (LLM_STATUS_NOT_CONFIGURED, "", None):
        st.session_state["llm_status"] = LLM_STATUS_CONFIGURED
        st.session_state["llm_status_message"] = "LLM settings saved. Connection not tested yet."


#: How each status reads to a user. One mapping, so every surface says the
#: same word for the same state. "Configured" is deliberately not
#: "Connected": filling three fields is not evidence that the endpoint
#: answers, and reporting it as connected was a claim the app had not
#: checked.
LLM_STATUS_LABELS = {
    LLM_STATUS_NOT_CONFIGURED: "Not configured",
    LLM_STATUS_CONFIGURED: "Configured (not tested)",
    LLM_STATUS_CONNECTED: "Connected",
    LLM_STATUS_FAILED: "Connection failed",
}


def describe_llm_status() -> str:
    """The current status in the user's words, from the one stored status."""
    return LLM_STATUS_LABELS.get(get_llm_status(), "Unknown")


def get_llm_status() -> str:
    refresh_llm_status_from_config()
    return st.session_state.get("llm_status", LLM_STATUS_NOT_CONFIGURED)


def set_llm_settings(base_url: str, api_key: str, model: str) -> None:
    ensure_llm_state_defaults()
    st.session_state["llm_base_url"] = str(base_url or "").strip()
    st.session_state["llm_api_key"] = str(api_key or "").strip()
    st.session_state["llm_model"] = str(model or "").strip()
    if is_llm_configured():
        st.session_state["llm_status"] = LLM_STATUS_CONFIGURED
        st.session_state["llm_status_message"] = "LLM settings saved. Connection not tested yet."
    else:
        st.session_state["llm_status"] = LLM_STATUS_NOT_CONFIGURED
        st.session_state["llm_status_message"] = "Missing Base URL, API key, or model."


def get_llm_state() -> LLMState:
    ensure_llm_state_defaults()
    refresh_llm_status_from_config()
    return LLMState(
        base_url=st.session_state.get("llm_base_url", ""),
        api_key=st.session_state.get("llm_api_key", ""),
        model=st.session_state.get("llm_model", ""),
        status=st.session_state.get("llm_status", LLM_STATUS_NOT_CONFIGURED),
        status_message=st.session_state.get("llm_status_message", ""),
    )
