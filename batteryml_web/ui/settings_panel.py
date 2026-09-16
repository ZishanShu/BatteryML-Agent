"""User-facing settings, in the left sidebar and wherever they are needed.

Two problems this solves at once.

The app leaned on Streamlit's top-right menu as if it were product UI. It is
not: it holds *developer* actions ("Clear caches", "Rerun", a screencast
recorder) that mean nothing to a battery scientist and one of which can throw
away a run. Application settings belong somewhere we own, which is the
sidebar.

And "Explain computed result" was a dead end: disabled, with a caption
telling the user to configure an assistant the page gave them no way to
configure. The button is now always clickable; when the assistant is not
configured it opens the same configuration form the sidebar shows.

**One source of truth.** Both entry points read and write the existing LLM
session state through :mod:`batteryml_web.ui.llm_state` — the same
``llm_base_url`` / ``llm_api_key`` / ``llm_model`` keys the rest of the app
already uses. There is no second settings system.

Secrets: the API key is rendered with ``type="password"``, is never written
to a caption, a log, a toast or a debug expander, and is never persisted by
this module. It lives in Streamlit's session state for the session, which is
the mechanism the app already used.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional

import streamlit as st

from .llm_state import (
    LLM_STATUS_CONFIGURED,
    LLM_STATUS_NOT_CONFIGURED,
    ensure_llm_state_defaults,
    is_llm_configured,
    set_llm_settings,
)

#: Session key holding the pending "what to do after saving" intent.
PENDING_ACTION_KEY = "llm_settings_pending_action"

#: Session key that opens the inline fallback form.
FORM_OPEN_KEY = "llm_settings_form_open"

DIALOG_TITLE = "LLM configuration"

#: Field labels. The names come from the existing contract, not a new one.
FIELD_LABELS: Dict[str, str] = {
    "llm_base_url": "Base URL",
    "llm_api_key": "API key",
    "llm_model": "Model",
}


def llm_settings_summary() -> Dict[str, Any]:
    """What the page may say about the LLM settings — never the key itself.

    ``api_key_set`` is a boolean. The key's value does not appear in the
    returned structure, so nothing built from this can leak it into a
    caption, a log line or a test recorder.
    """
    ensure_llm_state_defaults()
    return {
        "configured": is_llm_configured(),
        "base_url": str(st.session_state.get("llm_base_url", "")),
        "model": str(st.session_state.get("llm_model", "")),
        "api_key_set": bool(str(st.session_state.get("llm_api_key", "")).strip()),
        "status": str(st.session_state.get("llm_status", LLM_STATUS_NOT_CONFIGURED)),
    }


def dialog_supported() -> bool:
    """Whether this Streamlit build has ``st.dialog``.

    Older builds do not; they get the inline expander fallback instead of an
    exception. The feature is a nicety, and losing it must not break a page.
    """
    return callable(getattr(st, "dialog", None))


def render_llm_settings_form(
    *,
    key_prefix: str,
    on_saved: Optional[Callable[[], None]] = None,
    offer_save_and_run: bool = False,
    save_and_run_label: str = "Save & explain",
) -> Optional[str]:
    """The LLM configuration form. Returns the action the user took.

    Args:
        key_prefix: namespace for the widget keys, so the sidebar copy and
            the dialog copy can coexist in one session.
        on_saved: called after settings are written, whichever button saved.
        offer_save_and_run: adds the "Save & explain" button.
        save_and_run_label: its label.

    Returns ``"save"``, ``"save_and_run"``, ``"cancel"`` or None (nothing
    pressed). Writing the settings is all this does — it never calls the
    assistant, so opening the form cannot send anything anywhere.
    """
    ensure_llm_state_defaults()
    base_url = st.text_input(
        FIELD_LABELS["llm_base_url"],
        value=str(st.session_state.get("llm_base_url", "")),
        key=f"{key_prefix}_base_url",
    )
    api_key = st.text_input(
        FIELD_LABELS["llm_api_key"],
        value=str(st.session_state.get("llm_api_key", "")),
        type="password",
        key=f"{key_prefix}_api_key",
    )
    model = st.text_input(
        FIELD_LABELS["llm_model"],
        value=str(st.session_state.get("llm_model", "")),
        key=f"{key_prefix}_model",
    )
    st.caption(
        "Optional. Without it the platform still runs the full deterministic "
        "workflow. The key is held for this session only and is never written "
        "to a file, a log or the debug view."
    )

    columns = st.columns(3 if offer_save_and_run else 2)
    action: Optional[str] = None
    if columns[0].button("Cancel", key=f"{key_prefix}_cancel", use_container_width=True):
        action = "cancel"
    if columns[1].button(
        "Save settings", key=f"{key_prefix}_save", use_container_width=True
    ):
        set_llm_settings(base_url, api_key, model)
        action = "save"
    if offer_save_and_run and columns[2].button(
        save_and_run_label, key=f"{key_prefix}_save_run", use_container_width=True
    ):
        set_llm_settings(base_url, api_key, model)
        action = "save_and_run"

    if action in ("save", "save_and_run") and on_saved is not None:
        on_saved()
    return action


def render_sidebar_settings() -> None:
    """The Settings section of the sidebar.

    Application settings only. Streamlit's developer actions are deliberately
    absent: they are not product features, and ``.streamlit/config.toml``
    already hides them from end users.
    """
    with st.expander("Settings", expanded=False):
        summary = llm_settings_summary()
        st.caption(
            "LLM assistant: configured"
            if summary["configured"]
            else "LLM assistant: not configured"
        )
        render_llm_settings_form(key_prefix="sidebar_llm")


def request_llm_configuration(pending_action: str) -> None:
    """Ask for the configuration form to open, remembering why."""
    st.session_state[PENDING_ACTION_KEY] = str(pending_action)
    st.session_state[FORM_OPEN_KEY] = True


def clear_llm_configuration_request() -> None:
    st.session_state.pop(PENDING_ACTION_KEY, None)
    st.session_state.pop(FORM_OPEN_KEY, None)


def llm_configuration_requested() -> bool:
    return bool(st.session_state.get(FORM_OPEN_KEY))


def pending_action() -> Optional[str]:
    value = st.session_state.get(PENDING_ACTION_KEY)
    return str(value) if value else None


def render_llm_configuration_prompt(
    *,
    save_and_run_label: str = "Save & explain",
) -> Optional[str]:
    """Show the configuration form, as a dialog where one is available.

    Returns the action the user took, or None. The caller decides what
    ``"save_and_run"`` means — nothing is sent anywhere from here.
    """
    if not llm_configuration_requested():
        return None
    outcome: Dict[str, Optional[str]] = {"action": None}

    def body() -> None:
        st.write(
            "The assistant is not configured yet. Add the endpoint below to "
            "enable result explanation."
        )
        outcome["action"] = render_llm_settings_form(
            key_prefix="dialog_llm",
            offer_save_and_run=True,
            save_and_run_label=save_and_run_label,
        )

    if dialog_supported():
        decorated = st.dialog(DIALOG_TITLE)(body)
        decorated()
    else:
        # Inline fallback for builds without st.dialog: same form, same
        # single source of truth, no exception.
        with st.expander(DIALOG_TITLE, expanded=True):
            body()
    return outcome["action"]
