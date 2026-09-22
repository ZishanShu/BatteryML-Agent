"""The one place the app keeps its LLM configuration.

Every surface — the sidebar Settings form, Agent Workflow, Agent
explanation, Review Results explanation — reads and writes *these* session
keys and no others. The rule that makes "configure it once" true is narrow
and worth stating plainly:

    Nothing outside this module may read ``st.session_state["llm_*"]``.

Not because a direct read is wrong in itself, but because a direct read is
how a second copy starts. Four call sites used to pass
``st.session_state["llm_base_url"]`` into a client by hand; each one was a
place where a key could be renamed, defaulted, or quietly missed, and a
``KeyError`` on a page the user had not visited was the mildest way that
could go wrong. :func:`llm_credentials` is the one way to reach the values,
and :func:`set_llm_settings` the one way to change them.

**Scope: session, not project.** These keys live in Streamlit's session
state for the life of the browser session. Loading, creating or switching a
project does not touch them, and nothing here is written to a project file,
a run manifest, an export bundle or a log. A browser restart asks for the
key again, which is the intended trade and not a defect to work around.

**The API key never leaves this module except through**
:func:`llm_credentials`, which exists to hand it to the HTTP client.
:func:`llm_settings_summary`-style callers get ``api_key_set`` — a boolean —
so nothing built for display can carry the value.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import streamlit as st


LLM_STATUS_NOT_CONFIGURED = "not_configured"
LLM_STATUS_CONFIGURED = "configured"
LLM_STATUS_CONNECTED = "connected"
LLM_STATUS_FAILED = "failed"

#: Session key for the chosen provider preset. It lives here, with the
#: credentials it belongs to, rather than beside the form that happens to
#: render it: the provider is part of the configuration, so a surface asking
#: "what is configured" must be able to get it from one call.
PROVIDER_KEY = "llm_provider_preset"

#: The provider name meaning "I will type the Base URL myself". Also the
#: default, because that is the honest description of an empty form: no
#: preset has been chosen.
CUSTOM_PROVIDER = "Custom OpenAI-compatible"

#: The credential keys, in the order the form shows them. Kept here because
#: this module owns them.
CREDENTIAL_KEYS = ("llm_base_url", "llm_api_key", "llm_model")


@dataclass
class LLMState:
    """A snapshot of the configuration. Never stored — always re-read."""

    provider: str
    base_url: str
    api_key: str
    model: str
    status: str
    status_message: str


def ensure_llm_state_defaults() -> None:
    st.session_state.setdefault(PROVIDER_KEY, CUSTOM_PROVIDER)
    st.session_state.setdefault("llm_base_url", "")
    st.session_state.setdefault("llm_api_key", "")
    st.session_state.setdefault("llm_model", "")
    st.session_state.setdefault("llm_status", LLM_STATUS_NOT_CONFIGURED)
    st.session_state.setdefault("llm_status_message", "")


def is_llm_configured() -> bool:
    ensure_llm_state_defaults()
    return (
        bool(str(st.session_state.get("llm_base_url", "") or "").strip())
        and bool(str(st.session_state.get("llm_api_key", "") or "").strip())
        and bool(str(st.session_state.get("llm_model", "") or "").strip())
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


def get_llm_provider() -> str:
    ensure_llm_state_defaults()
    return str(st.session_state.get(PROVIDER_KEY) or CUSTOM_PROVIDER)


def _credential_tuple() -> tuple[str, ...]:
    return tuple(
        str(st.session_state.get(key, "") or "") for key in CREDENTIAL_KEYS
    )


def set_llm_settings(
    base_url: str,
    api_key: str,
    model: str,
    provider: Optional[str] = None,
) -> None:
    """Write the configuration. The only writer of the credential keys.

    ``provider`` is optional so a caller that only has the three credential
    fields does not have to invent one; passing ``None`` leaves the stored
    provider alone rather than resetting it to the default.

    **An unchanged save keeps a verified connection.** Save used to drop the
    status to "Configured (not tested)" unconditionally, so re-saving the
    same settings — which is what pressing Save twice does — threw away the
    evidence that Connect had gathered. A change to any of the four fields
    still invalidates it, because that evidence was about the old endpoint.
    """
    ensure_llm_state_defaults()
    before = _credential_tuple()
    before_provider = get_llm_provider()

    st.session_state["llm_base_url"] = str(base_url or "").strip()
    st.session_state["llm_api_key"] = str(api_key or "").strip()
    st.session_state["llm_model"] = str(model or "").strip()
    if provider is not None:
        st.session_state[PROVIDER_KEY] = str(provider or CUSTOM_PROVIDER)

    if not is_llm_configured():
        st.session_state["llm_status"] = LLM_STATUS_NOT_CONFIGURED
        st.session_state["llm_status_message"] = "Missing Base URL, API key, or model."
        return

    unchanged = (
        _credential_tuple() == before and get_llm_provider() == before_provider
    )
    if unchanged and st.session_state.get("llm_status") in (
        LLM_STATUS_CONNECTED,
        LLM_STATUS_FAILED,
    ):
        # Nothing moved, so the last connection test still describes this
        # endpoint. Re-asserting "not tested yet" would be a claim that the
        # evidence had expired, and it has not.
        return
    st.session_state["llm_status"] = LLM_STATUS_CONFIGURED
    st.session_state["llm_status_message"] = "LLM settings saved. Connection not tested yet."


def set_llm_connection_result(ok: bool, message: str) -> None:
    """Record the outcome of one connection test.

    The only writer of ``llm_status`` besides :func:`set_llm_settings`.
    Surfaces used to assign the raw strings ``"connected"`` / ``"failed"``
    inline after an unrelated prompt succeeded, which put a value into the
    status that the status vocabulary in this module had never agreed to.
    """
    ensure_llm_state_defaults()
    st.session_state["llm_status"] = (
        LLM_STATUS_CONNECTED if ok else LLM_STATUS_FAILED
    )
    st.session_state["llm_status_message"] = str(message)


def get_llm_state() -> LLMState:
    ensure_llm_state_defaults()
    refresh_llm_status_from_config()
    return LLMState(
        provider=get_llm_provider(),
        base_url=str(st.session_state.get("llm_base_url", "")),
        api_key=str(st.session_state.get("llm_api_key", "")),
        model=str(st.session_state.get("llm_model", "")),
        status=str(st.session_state.get("llm_status", LLM_STATUS_NOT_CONFIGURED)),
        status_message=str(st.session_state.get("llm_status_message", "")),
    )


def llm_credentials() -> Dict[str, str]:
    """The three arguments every LLM call needs, from the one store.

    Returned as a mapping so a call site reads
    ``run_llm_prompt(prompt, **llm_credentials())`` and cannot pick up two
    fields from the store and a third from somewhere else. This is the only
    function that hands out the API key, and it exists solely to reach the
    HTTP client — nothing that renders, logs or serialises should call it.
    """
    ensure_llm_state_defaults()
    return {
        "base_url": str(st.session_state.get("llm_base_url", "")),
        "api_key": str(st.session_state.get("llm_api_key", "")),
        "model": str(st.session_state.get("llm_model", "")),
    }


def llm_settings_summary() -> Dict[str, Any]:
    """What a page may say about the configuration — never the key itself.

    ``api_key_set`` is a boolean. The key's value does not appear in the
    returned structure, so nothing built from this can leak it into a
    caption, a log line or a test recorder.
    """
    ensure_llm_state_defaults()
    return {
        "configured": is_llm_configured(),
        "provider": get_llm_provider(),
        "base_url": str(st.session_state.get("llm_base_url", "")),
        "model": str(st.session_state.get("llm_model", "")),
        "api_key_set": bool(str(st.session_state.get("llm_api_key", "") or "").strip()),
        "status": str(st.session_state.get("llm_status", LLM_STATUS_NOT_CONFIGURED)),
    }


def llm_settings_fingerprint() -> str:
    """Identity of the current configuration, safe to store and compare.

    Built from provider, endpoint and model plus *whether* a key is set —
    never from the key itself. Used to label an explanation with the
    configuration that produced it, so a later model change cannot silently
    relabel an old answer as the new model's work.
    """
    summary = llm_settings_summary()
    return "|".join(
        [
            str(summary["provider"]),
            str(summary["base_url"]),
            str(summary["model"]),
            "key" if summary["api_key_set"] else "nokey",
        ]
    )
