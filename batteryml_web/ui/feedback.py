"""Transient success feedback, shown once per event.

Streamlit reruns the whole script on every widget interaction. A success
message written straight into the page therefore either persists as a green
box long after the action it reported, or is re-emitted on every rerun. Both
are noise, and a page full of stale green boxes trains the user to ignore
the one that matters.

The convention here:

* **success is transient** — one ``st.toast``, once, for the event that just
  happened;
* **failure is persistent** — ``st.error`` / ``st.warning`` stay on the page
  until the user acts on them. Nothing in this module touches failures;
* **an event is identified, not counted** — the caller supplies an event key
  and a signature describing *which* occurrence this is. The same signature
  never toasts twice, so reruns are silent, and a genuinely new occurrence
  (a different project, a different plan, a different run) toasts once.

Signatures are the whole mechanism, so choose them to match the action a
user would call "a thing that happened". A signature that includes a device
or a checkbox would fire on every fiddle; one that includes only the
dataset, task and model fires when the user actually changes the subject.

Pure helpers (:func:`should_notify`, :func:`plan_signature`, ...) hold no
Streamlit reference so the deduplication logic is testable without a
runtime.
"""

from __future__ import annotations

import json
from typing import Any, Dict, Mapping, MutableMapping, Optional

import streamlit as st

#: Session-state key holding ``{event_key: last_signature}``.
SEEN_KEY = "_transient_feedback_seen"

#: Event keys used across the app. Named here so the set is visible in one
#: place and a test can assert the app covers them.
EVENT_PROJECT_CREATED = "project_created"
EVENT_QUICKSTART_READY = "quickstart_ready"
EVENT_REQUEST_ANALYZED = "request_analyzed"
EVENT_PLAN_READY = "plan_ready"
EVENT_RUN_COMPLETED = "run_completed"
EVENT_RESULT_LOADED = "result_loaded"
EVENT_EXPLANATION_READY = "explanation_ready"
EVENT_PREPROCESS_DONE = "preprocess_done"

EVENT_KEYS: tuple[str, ...] = (
    EVENT_PROJECT_CREATED,
    EVENT_QUICKSTART_READY,
    EVENT_REQUEST_ANALYZED,
    EVENT_PLAN_READY,
    EVENT_RUN_COMPLETED,
    EVENT_RESULT_LOADED,
    EVENT_EXPLANATION_READY,
    EVENT_PREPROCESS_DONE,
)


def should_notify(
    seen: MutableMapping[str, str],
    event_key: str,
    signature: str,
) -> bool:
    """Whether this occurrence has not been reported yet.

    Records the signature when it returns True, so a rerun with the same
    signature returns False. Pure apart from the mapping it is handed, which
    is what makes the rerun behaviour testable without Streamlit.
    """
    key = str(event_key or "").strip()
    if not key:
        raise ValueError("should_notify: event_key is empty.")
    value = str(signature)
    if seen.get(key) == value:
        return False
    seen[key] = value
    return True


def signature_of(payload: Any) -> str:
    """A stable signature string for any small JSON-serializable value."""
    try:
        return json.dumps(payload, sort_keys=True, default=str)
    except (TypeError, ValueError):
        return repr(payload)


def plan_signature(plan: Optional[Mapping[str, Any]]) -> str:
    """Identify a plan by its *subject*, not by every knob on the page.

    Dataset, task, model and validated config only: changing the device or
    ticking the smoke box rebuilds the plan but does not make it a different
    thing to report, and toasting on those would be exactly the per-widget
    noise this module exists to avoid.
    """
    if not isinstance(plan, Mapping):
        return ""
    return signature_of(
        [
            plan.get("dataset"),
            plan.get("task"),
            plan.get("model"),
            plan.get("config_path"),
            bool(plan.get("can_execute")),
        ]
    )


def _seen_map() -> MutableMapping[str, str]:
    seen = st.session_state.get(SEEN_KEY)
    if not isinstance(seen, dict):
        seen = {}
        st.session_state[SEEN_KEY] = seen
    return seen


def toast(message: str, *, icon: Optional[str] = None) -> bool:
    """Emit one toast; returns whether the installed Streamlit had toasts.

    ``st.toast`` arrived in Streamlit 1.27. On an older build this is a
    no-op rather than an exception: transient feedback is a nicety, and
    losing it must never break a page.
    """
    emit = getattr(st, "toast", None)
    if not callable(emit):
        return False
    try:
        emit(message, icon=icon) if icon else emit(message)
    except TypeError:
        emit(message)
    return True


def notify_success(
    message: str,
    *,
    event_key: str,
    signature: Any,
    icon: str = "✅",
) -> bool:
    """Toast ``message`` once for this occurrence of ``event_key``.

    Returns True when the toast was emitted, False when this occurrence had
    already been reported (a rerun) or the installed Streamlit has no
    toasts. Never writes a persistent success box.
    """
    if not should_notify(_seen_map(), event_key, signature_of(signature)):
        return False
    return toast(message, icon=icon)


def reset_event(event_key: str) -> None:
    """Forget one event, so its next occurrence toasts again.

    Used where the same subject can legitimately recur — re-running the same
    plan, for instance, is a new run worth reporting even though the plan
    signature is unchanged.
    """
    _seen_map().pop(str(event_key), None)


def seen_signatures() -> Dict[str, str]:
    """A copy of what has been reported, for tests and debugging."""
    return dict(_seen_map())
