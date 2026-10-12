"""The LLM configuration form. There is exactly one, and it lives in Settings.

The app leaned on Streamlit's top-right menu as if it were product UI. It is
not: it holds *developer* actions ("Clear caches", "Rerun", a screencast
recorder) that mean nothing to a battery scientist and one of which can throw
away a run. Application settings belong somewhere we own, which is the
sidebar.

**One form, in one place.** Five surfaces used to be able to render this
form — the sidebar, Home, Run Model, Review Results, and a dialog on the
Agent page. They all wrote the same store, so the *values* never diverged;
what diverged was the user's model of the app. Being asked for a provider, a
Base URL and an API key on the Review Results page reads as "this page wants
its own credentials", whatever the implementation does underneath, and that
is the complaint this module now answers. Every other surface shows
:data:`CONFIGURE_IN_SETTINGS_MESSAGE` and sends the user here.

**Opening Settings from a page** (UI-14). The sidebar may be collapsed —
Streamlit collapses it on a narrow window — so expanding its Settings block
from a page is not reliably visible. A page therefore *requests* Settings
(:func:`request_llm_settings`) and the same settings body opens in a small,
dismissible dialog (:func:`render_requested_llm_settings`): the one form,
its one call site, its one store. It opens when the user asks (Configure
LLM), when Summarize/Explain is pressed without a verified connection (no
provider call is made), and once per session on arriving at Review Results
from a completed run with nothing configured. A request is consumed when
the dialog opens, so no rerun reopens it.

Everything is read and written through :mod:`batteryml_web.ui.llm_state`,
which owns the session keys. This module contributes the provider presets
and the widgets; it stores nothing of its own.

Secrets: the API key is rendered with ``type="password"``, is never written
to a caption, a log, a toast or a debug expander, and is never persisted by
this module. It lives in Streamlit's session state for the session only.
"""

from __future__ import annotations

from typing import Callable, Dict, Optional

import streamlit as st

from ..agent.failure_taxonomy import classify_llm_failure
from ..agent.llm_client import test_llm_connection
from .feedback import notify_success
from .surfaces import NAV_ROLE, PRIMARY_ROLE, SECONDARY_ROLE
from .llm_state import (
    CUSTOM_PROVIDER,
    LLM_STATUS_FAILED,
    LLM_STATUS_NOT_CONFIGURED,
    PROVIDER_KEY,
    describe_llm_status,
    ensure_llm_state_defaults,
    get_llm_provider,
    get_llm_state,
    get_llm_status,
    is_llm_configured,
    is_llm_connected,
    llm_credentials,
    llm_settings_fingerprint,
    llm_settings_summary,
    set_llm_connection_result,
    set_llm_settings,
)

#: What every surface that is *not* Settings says when the assistant is not
#: configured. One sentence, one destination: a page that explains how to
#: configure an assistant is a page that has started to become a second
#: settings screen.
CONFIGURE_IN_SETTINGS_MESSAGE = (
    "LLM Assistant is not configured. Configure it in Settings (left sidebar)."
)

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
    CUSTOM_PROVIDER: "",
}

PROVIDER_LABEL = "Provider"

#: One line, because it is a hint and not a specification. The precise claim
#: it summarises: this client speaks the OpenAI chat-completions protocol, so
#: any endpoint offering that protocol works; presets only fill the Base URL,
#: and provider-specific features beyond chat completions are not
#: implemented. That detail belongs here, where a maintainer reads it, rather
#: than in four lines above every settings form.
SUPPORTED_API_NOTE = (
    "Supports OpenAI-compatible APIs: OpenAI, Gemini, Claude, Kimi, GLM, and "
    "custom endpoints."
)

#: Field labels. The names come from the existing contract, not a new one.
FIELD_LABELS: Dict[str, str] = {
    "llm_base_url": "Base URL",
    "llm_api_key": "API key",
    "llm_model": "Model",
}


def connect_llm(
    base_url: str,
    api_key: str,
    model: str,
    provider: Optional[str] = None,
) -> tuple[bool, str]:
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
    set_llm_settings(base_url, api_key, model, provider=provider)
    _invalidate_all_form_syncs()
    if not is_llm_configured():
        st.session_state["llm_status"] = LLM_STATUS_NOT_CONFIGURED
        st.session_state["llm_status_message"] = INCOMPLETE_MESSAGE
        return False, INCOMPLETE_MESSAGE

    # Credentials come back out of the store, not from the arguments: the
    # store is what every other surface will use, so the test must be of
    # what was actually written.
    ok, message = test_llm_connection(**llm_credentials())
    set_llm_connection_result(bool(ok), str(message))
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
        key_prefix: namespace for the widget keys. Only Settings renders
            this form now, but the parameter stays: a keyed widget's session
            value wins over its ``value=`` argument, so two forms sharing
            widget keys would silently become one form with two positions.
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
    stored = get_llm_provider()
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
        "Optional — the deterministic workflow works without an LLM. "
        "API keys stay in this session only."
    )

    columns = st.columns(2)
    action: Optional[str] = None
    # Connect is this panel's primary action (the site's button hierarchy
    # lists it with Run and Preprocess): it stores the fields *and* proves
    # they work, which is what every LLM feature waits for. Save stays
    # secondary — it stores without a network call — and a panel with two
    # filled buttons side by side ranks neither of them.
    #
    # The role is passed as `type=` rather than through
    # `surfaces.primary_action`, because the helpers wrap `st.button` and
    # this call goes through a column receiver. Same native parameter,
    # same rendered `data-testid`, so the stylesheet treats it identically.
    if columns[0].button(
        SAVE_LABEL,
        key=f"{key_prefix}_save",
        type=SECONDARY_ROLE,
        use_container_width=True,
    ):
        # Stores only. Whether these credentials work is a separate question,
        # asked by Connect, so saving can never block on a network call.
        set_llm_settings(base_url, api_key, model, provider=provider)
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
        CONNECT_LABEL,
        key=f"{key_prefix}_connect",
        type=PRIMARY_ROLE,
        use_container_width=True,
    ):
        ok, message = connect_llm(base_url, api_key, model, provider=provider)
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

    The one implementation lives in :mod:`batteryml_web.ui.llm_state`, beside
    the values it is derived from. Built from provider, endpoint and model
    plus *whether* a key is set — never from the key itself, so nothing
    derived from this can carry it.
    """
    return llm_settings_fingerprint()


def render_configure_in_settings_notice() -> None:
    """What a non-Settings surface shows when nothing is configured.

    One caption, no fields. This is the whole of the "second form" removal:
    a page that needs an assistant says so and names where to set one up,
    and a page that tries to be helpful by offering the fields is a page
    that has become a second settings screen.
    """
    st.caption(CONFIGURE_IN_SETTINGS_MESSAGE)


#: Widget namespaces of the two places the one settings body is shown. Two,
#: because the sidebar renders on every run and a dialog may open in the
#: same run, and Streamlit rejects duplicate keys; both are views of the one
#: store (see :func:`_sync_widgets_from_store`).
SIDEBAR_FORM_PREFIX = "sidebar_llm"
DIALOG_FORM_PREFIX = "dialog_llm"

DIALOG_TITLE = "LLM settings"
CLOSE_LABEL = "Close"


def _render_last_failure() -> None:
    """Case D: the last connection test failed. Say so briefly, redacted.

    The fields keep their values (the store is unchanged) and nothing is
    retried; the user corrects a field and presses Connect.
    """
    if get_llm_status() != LLM_STATUS_FAILED:
        return
    message = get_llm_state().status_message
    if not message:
        return
    finding = classify_llm_failure(message, configured=True)
    st.caption(
        f"Last connection test failed: {finding['short_title']} "
        "Correct the settings above and press Connect; nothing is retried automatically."
    )


def _render_settings_body(
    key_prefix: str, *, on_connected: Optional[Callable[[], None]] = None
) -> Optional[str]:
    """Everything Settings shows, wherever it is opened. The form's one call site."""
    summary = llm_settings_summary()
    st.caption(
        "LLM assistant: configured"
        if summary["configured"]
        else "LLM assistant: not configured"
    )
    action = render_llm_settings_form(key_prefix=key_prefix, on_connected=on_connected)
    if action is None:
        _render_last_failure()
    return action


def render_sidebar_settings() -> None:
    """The Settings section of the sidebar.

    Application settings only. Streamlit's developer actions are deliberately
    absent: they are not product features, and ``.streamlit/config.toml``
    already hides them from end users.
    """
    with st.expander("Settings", expanded=False):
        _render_settings_body(SIDEBAR_FORM_PREFIX)


# --- opening Settings from a page (UI-14) ------------------------------------

#: One-shot request to show Settings on this script run, holding the reason.
#: Consumed when the dialog opens, so a later rerun does not reopen it.
REVEAL_REQUEST_KEY = "llm_settings_reveal_request"

#: Set by an intentional "view the results of the run that just completed";
#: consumed by Review Results on arrival.
RESULTS_INTENT_KEY = "llm_results_view_intent"

#: The once-per-session automatic reveal has happened (whether the user then
#: connected, closed or dismissed it). Never reset within the session.
RESULTS_REVEAL_DONE_KEY = "llm_results_reveal_done"

REVEAL_CONFIGURE = "configure"
REVEAL_ACTION = "action"
REVEAL_ARRIVAL = "arrival"

#: What the dialog says first, per reason. No reason claims a result is
#: missing: every BatteryML number is already on the page without an LLM.
REVEAL_NOTES: Dict[str, str] = {
    REVEAL_ACTION: (
        "Summaries and explanations need a connected LLM. Nothing was sent. "
        "Connect below, then press the button again."
    ),
    REVEAL_ARRIVAL: (
        "Your run has finished. Optionally connect an LLM to summarize or "
        "explain it — every BatteryML result is already on the page without one."
    ),
}


def request_llm_settings(reason: str) -> None:
    """Ask for Settings to open on this run (see :func:`render_requested_llm_settings`)."""
    st.session_state[REVEAL_REQUEST_KEY] = str(reason)


def mark_results_intent() -> None:
    """Record an intentional "view results" of a run that just completed."""
    st.session_state[RESULTS_INTENT_KEY] = True


def arrival_reveal_due(*, intent: bool, status: str, already_revealed: bool) -> bool:
    """Whether arriving at Review Results opens Settings by itself. Pure.

    Only on an intentional "view results" of a completed run, only when
    nothing is configured (a saved or failed configuration is the user's to
    act on — they have seen Settings already), and only once a session.
    """
    return bool(intent) and status == LLM_STATUS_NOT_CONFIGURED and not already_revealed


def consume_results_intent() -> bool:
    """On Review Results: decide, once, whether this arrival reveals Settings."""
    intent = bool(st.session_state.pop(RESULTS_INTENT_KEY, False))
    due = arrival_reveal_due(
        intent=intent,
        status=get_llm_status(),
        already_revealed=bool(st.session_state.get(RESULTS_REVEAL_DONE_KEY)),
    )
    if due:
        st.session_state[RESULTS_REVEAL_DONE_KEY] = True
        request_llm_settings(REVEAL_ARRIVAL)
    return due


def llm_action_allowed() -> bool:
    """Whether Summarize/Explain may call the provider: a verified connection only."""
    return is_llm_connected()


def _finish_reveal() -> None:
    """Close Settings opened from a page: forget the request, redraw the page."""
    st.session_state.pop(REVEAL_REQUEST_KEY, None)
    st.rerun()


def _settings_dialog_body(reason: str) -> None:
    note = REVEAL_NOTES.get(reason)
    if note:
        st.caption(note)
    # A successful Connect closes the dialog and changes nothing else: the
    # selected result stays selected and no request is sent on its behalf.
    _render_settings_body(DIALOG_FORM_PREFIX, on_connected=_finish_reveal)
    if st.button(CLOSE_LABEL, key=f"{DIALOG_FORM_PREFIX}_close", type=NAV_ROLE):
        _finish_reveal()


def render_requested_llm_settings() -> Optional[str]:
    """Open Settings if a page asked for it on this run. Returns the reason.

    A dialog where this Streamlit has one: dismissible (Close, the corner
    cross, Escape) and consumed on opening, so nothing reopens it on a
    rerun. Without ``st.dialog`` the same body is drawn in place, and stays
    until Close or a successful Connect, because there typing reruns the
    whole page.
    """
    reason = st.session_state.get(REVEAL_REQUEST_KEY)
    if not reason:
        return None
    dialog = getattr(st, "dialog", None)
    if dialog is None:
        with st.container(border=True):
            st.markdown(f"**{DIALOG_TITLE}**")
            _settings_dialog_body(str(reason))
        return str(reason)
    st.session_state.pop(REVEAL_REQUEST_KEY, None)
    dialog(DIALOG_TITLE)(_settings_dialog_body)(str(reason))
    return str(reason)
