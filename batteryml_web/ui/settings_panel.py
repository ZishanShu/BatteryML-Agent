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

from ..agent.failure_taxonomy import classify_llm_failure
from ..agent.llm_client import test_llm_connection
from .feedback import notify_success
from .llm_state import (
    LLM_STATUS_CONFIGURED,
    describe_llm_status,
    LLM_STATUS_CONNECTED,
    LLM_STATUS_FAILED,
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

#: The two actions the form offers, in render order. Save stores; Connect
#: stores *and* tests. Neither is "Cancel": closing the form is not an
#: action the user needs a button for, and "Save settings" beside "Cancel"
#: hid the connection test the earlier UX had.
SAVE_LABEL = "Save"
CONNECT_LABEL = "Connect"

#: Transient confirmations (ui/feedback.py emits each once per occurrence).
SAVED_MESSAGE = "LLM settings saved"
CONNECTED_MESSAGE = "LLM connected"

#: Feedback event keys, so a rerun does not repeat a toast.
EVENT_LLM_SAVED = "llm_settings_saved"
EVENT_LLM_CONNECTED = "llm_connected"

#: Shown when Connect is pressed with a field still empty. Names what is
#: missing; never echoes any value.
INCOMPLETE_MESSAGE = (
    "Base URL, API key and model are all required before the connection can "
    "be tested."
)

#: Provider presets: a name and the Base URL that provider documents for its
#: OpenAI-compatible endpoint. This is a *convenience for filling one field*,
#: not a claim of integration — the client speaks one protocol, and any
#: endpoint that speaks it works, including one not listed here.
#:
#: Model names are deliberately absent. They change faster than this file
#: can, and a stale dropdown of model names that no longer exist is worse
#: than a text box. The user types the model.
PROVIDER_PRESETS: Dict[str, str] = {
    "OpenAI": "https://api.openai.com/v1",
    "Gemini (OpenAI-compatible)": "https://generativelanguage.googleapis.com/v1beta/openai",
    "Claude (OpenAI-compatible)": "https://api.anthropic.com/v1",
    "Kimi / Moonshot": "https://api.moonshot.cn/v1",
    "GLM / Zhipu": "https://open.bigmodel.cn/api/paas/v4",
    "Custom OpenAI-compatible": "",
}

CUSTOM_PROVIDER = "Custom OpenAI-compatible"

PROVIDER_LABEL = "Provider"

#: Session key for the chosen preset. Part of the same single settings
#: state as the credentials themselves.
PROVIDER_KEY = "llm_provider_preset"

#: What this client actually supports, stated precisely. It speaks the
#: OpenAI chat-completions protocol; that is an endpoint-compatibility claim
#: and not a promise of any provider's own extensions.
SUPPORTED_API_NOTE = (
    "This client speaks the OpenAI-compatible chat-completions API. Any "
    "endpoint offering that protocol works — OpenAI, Gemini, Claude, Kimi "
    "and GLM all publish one, as do most self-hosted gateways. Presets only "
    "fill the Base URL; provider-specific features beyond chat completions "
    "are not implemented."
)

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


def connect_llm(base_url: str, api_key: str, model: str) -> tuple[bool, str]:
    """Store these credentials, then test them through the existing client.

    The connection test is :func:`batteryml_web.agent.llm_client.test_llm_connection`
    — the one the platform already had. Nothing here opens a socket, builds a
    request or parses a response: a second connection implementation is
    exactly what this must not become.

    Returns ``(ok, message)``. The message is the client's own, so a failure
    names what actually went wrong; it is never composed from the API key, and
    the key is not logged, printed or returned.
    """
    ensure_llm_state_defaults()
    set_llm_settings(base_url, api_key, model)
    _invalidate_all_form_syncs()
    if not is_llm_configured():
        st.session_state["llm_status"] = LLM_STATUS_NOT_CONFIGURED
        st.session_state["llm_status_message"] = INCOMPLETE_MESSAGE
        return False, INCOMPLETE_MESSAGE

    ok, message = test_llm_connection(
        base_url=str(st.session_state.get("llm_base_url", "")),
        api_key=str(st.session_state.get("llm_api_key", "")),
        model=str(st.session_state.get("llm_model", "")),
    )
    st.session_state["llm_status"] = LLM_STATUS_CONNECTED if ok else LLM_STATUS_FAILED
    st.session_state["llm_status_message"] = str(message)
    return bool(ok), str(message)


#: Per-form marker recording which stored values that form's widgets were
#: last seeded from.
_SYNCED_SUFFIX = "_synced_from"


#: Marker value meaning "the store was written elsewhere; re-read it". Not a
#: deletion: a missing marker means "this form has never synced", which is a
#: different situation and must not clobber a half-typed field.
_STALE = ["__stale__"]


def _invalidate_all_form_syncs() -> None:
    """Force every settings form to re-read the store on the next run.

    Called after a write, so a value saved on one surface is what every other
    surface shows — which is the whole point of having one store.
    """
    for key in [k for k in st.session_state if str(k).endswith(_SYNCED_SUFFIX)]:
        st.session_state[key] = list(_STALE)


def _sync_widgets_from_store(key_prefix: str) -> None:
    """Make this form's widgets a view of the one credential store.

    Each surface needs its own widget keys — the sidebar form and an inline
    form render in the same script run, and Streamlit rejects duplicate keys.
    But a keyed widget's session value *wins over* its ``value=`` argument,
    so per-surface keys quietly become per-surface copies: settings saved in
    the sidebar never appeared in the Home form, which is the "enter it
    again" problem this milestone exists to fix. A browser check caught it
    after the obvious duplicate form had already been removed.

    So the widget keys are re-seeded from the store whenever the store has
    changed since this form last looked. Typing is never clobbered: while the
    user edits, the store is unchanged, so no re-seed happens.
    """
    canonical = [
        str(st.session_state.get(field, "") or "") for field in FIELD_LABELS
    ]
    marker = f"{key_prefix}{_SYNCED_SUFFIX}"
    previous = st.session_state.get(marker)
    if previous == canonical:
        return
    # Overwrite an existing widget value only when the store really changed
    # under this form (previous marker present and different). A form that
    # has never synced fills in what is missing and leaves anything already
    # in the box alone — half-typed text is not something to discard.
    external_change = previous is not None and previous != canonical
    for field, value in zip(FIELD_LABELS, canonical):
        widget_key = f"{key_prefix}_{field[len('llm_'):]}"
        if widget_key not in st.session_state or external_change:
            st.session_state[widget_key] = value
    st.session_state[marker] = canonical


def render_llm_settings_form(
    *,
    key_prefix: str,
    on_saved: Optional[Callable[[], None]] = None,
    on_connected: Optional[Callable[[], None]] = None,
) -> Optional[str]:
    """The LLM configuration form: Save and Connect.

    Args:
        key_prefix: namespace for the widget keys, so the sidebar copy and
            the dialog copy can coexist in one session.
        on_saved: called after Save writes the settings.
        on_connected: called after Connect writes them *and* the test
            succeeded. Not called on a failed test.

    Returns ``"save"``, ``"connect"``, ``"connect_failed"`` or None.

    **Save** stores what is in the fields and stops there: no network call,
    no connection test. **Connect** stores the same values through the same
    function and then runs the existing connection test. A failed test leaves
    a persistent error on the page — a toast would vanish before it could be
    acted on — while a success is transient.
    """
    ensure_llm_state_defaults()
    _sync_widgets_from_store(key_prefix)
    st.caption(SUPPORTED_API_NOTE)

    names = list(PROVIDER_PRESETS)
    stored = str(st.session_state.get(PROVIDER_KEY) or CUSTOM_PROVIDER)
    provider = st.selectbox(
        PROVIDER_LABEL,
        names,
        index=names.index(stored) if stored in names else names.index(CUSTOM_PROVIDER),
        key=f"{key_prefix}_provider",
    )
    if provider != stored:
        # A preset fills the Base URL; it never locks it. The field below
        # stays editable, and a gateway with its own URL is normal.
        st.session_state[PROVIDER_KEY] = provider
        default_url = PROVIDER_PRESETS.get(provider, "")
        if default_url:
            st.session_state[f"{key_prefix}_base_url"] = default_url

    base_url = st.text_input(
        FIELD_LABELS["llm_base_url"],
        value=str(st.session_state.get("llm_base_url", "")),
        key=f"{key_prefix}_base_url",
        help="Editable. A preset fills this in; any OpenAI-compatible URL works.",
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

    columns = st.columns(2)
    action: Optional[str] = None
    if columns[0].button(SAVE_LABEL, key=f"{key_prefix}_save", use_container_width=True):
        # Stores only. Whether these credentials work is a separate question,
        # asked by Connect, so saving can never block on a network call.
        set_llm_settings(base_url, api_key, model)
        _invalidate_all_form_syncs()
        notify_success(
            SAVED_MESSAGE,
            event_key=EVENT_LLM_SAVED,
            signature=llm_settings_signature(),
        )
        action = "save"
        if on_saved is not None:
            on_saved()
    if columns[1].button(
        CONNECT_LABEL, key=f"{key_prefix}_connect", use_container_width=True
    ):
        ok, message = connect_llm(base_url, api_key, model)
        if not ok and message == INCOMPLETE_MESSAGE:
            # Not a connection failure: nothing was sent, because a field is
            # empty. Classifying this as an LLM error would report "the cause
            # was not recognised" for a form the user simply has not finished.
            st.error(message)
            return "connect_failed"
        if ok:
            notify_success(
                CONNECTED_MESSAGE,
                event_key=EVENT_LLM_CONNECTED,
                signature=llm_settings_signature(),
            )
            action = "connect"
            if on_connected is not None:
                on_connected()
        else:
            # Persistent, and actionable. The reason is classified rather
            # than shown raw, so "Network is unreachable" reads as a
            # connectivity problem and not as a generic failure — and the
            # classifier redacts anything key-shaped.
            finding = classify_llm_failure(message, configured=True)
            st.error(f"{finding['short_title']} {finding['user_message']}")
            if finding["recommended_action"]:
                st.caption(finding["recommended_action"])
            if finding["evidence_line"]:
                with st.expander("Show technical details", expanded=False):
                    st.code(str(finding["evidence_line"]), language="text")
            action = "connect_failed"
    st.caption(f"Status: {describe_llm_status()}")
    return action


def llm_settings_signature() -> str:
    """Identity of the current settings, for toast de-duplication.

    Built from the endpoint and model plus *whether* a key is set — never
    from the key itself, so nothing derived from this can carry it.
    """
    summary = llm_settings_summary()
    return "|".join(
        [
            summary["base_url"],
            summary["model"],
            "key" if summary["api_key_set"] else "nokey",
        ]
    )


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


def render_llm_configuration_prompt() -> Optional[str]:
    """Show the configuration form, as a dialog where one is available.

    The same two actions as the sidebar — Save and Connect — because there is
    one form and one credential store. Returns what the user did; nothing is
    sent anywhere from here, and in particular a successful Connect does not
    explain anything. It only makes explaining possible: the user presses
    Explain again, deliberately.
    """
    if not llm_configuration_requested():
        return None
    outcome: Dict[str, Optional[str]] = {"action": None}

    def body() -> None:
        st.write(
            "The assistant is not configured yet. Add the endpoint below to "
            "enable result explanation."
        )
        outcome["action"] = render_llm_settings_form(key_prefix="dialog_llm")

    if dialog_supported():
        decorated = st.dialog(DIALOG_TITLE)(body)
        decorated()
    else:
        # Inline fallback for builds without st.dialog: same form, same
        # single source of truth, no exception.
        with st.expander(DIALOG_TITLE, expanded=True):
            body()
    return outcome["action"]
