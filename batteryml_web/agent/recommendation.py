"""What to show the user about models, given what they actually asked for.

The page used to answer every request the same way: dump every runnable
model plus every blocked one, and let the user find the relevant row. That
is wrong twice over — it buries the model the user named, and on a platform
with two dozen registered models it buries everything else too.

This module turns one resolved request plus the authoritative readiness
context into a *presentation*: a short primary section that answers the
request, and the rest behind an expander. Five shapes, decided by what the
request contained:

``requested_ready``
    A registered model was named and readiness says it can run. Show that
    model. Nothing else needs to compete with it.
``requested_blocked``
    A known model was named and readiness says it cannot run. Show its exact
    status and blockers, then offer runnable alternatives — and require the
    user to pick one, because substituting a model behind their back would
    produce a result for a question they did not ask.
``requested_unknown``
    A name was given that the catalog does not contain. Say it is not
    registered, offer the closest runnable models as *suggestions*, and
    require an explicit choice.
``recommendation``
    No model was named and the request asked which to use. Propose from the
    agent's suggestions where they survive readiness, otherwise from the
    context's own runnable list.
``browse``
    No model, no recommendation asked for. Offer the runnable models, capped.

No model name appears anywhere in this module. Every key it emits was read
from the readiness context or the catalog, and a key that readiness does not
list as runnable is never placed where the UI can select it.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

#: How many models the primary section may name. Beyond this the page stops
#: being an answer and becomes a catalogue; the rest lives in the expander.
PRIMARY_LIMIT = 3

#: The presentation shapes, in the order they are decided.
MODE_REQUESTED_READY = "requested_ready"
MODE_REQUESTED_BLOCKED = "requested_blocked"
MODE_REQUESTED_UNKNOWN = "requested_unknown"
MODE_RECOMMENDATION = "recommendation"
MODE_BROWSE = "browse"

#: Shown in place of a preselected model when the user must choose for
#: themselves. Not a model key, and never accepted as one.
NO_SELECTION = ""


def _runnable_entries(workflow_context: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [dict(entry) for entry in workflow_context.get("runnable_models", [])]


def _blocked_entries(workflow_context: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [dict(entry) for entry in workflow_context.get("blocked_models", [])]


def _keys(entries: Sequence[Dict[str, Any]]) -> List[str]:
    return [str(entry.get("model")) for entry in entries]


def _entry_for(entries: Sequence[Dict[str, Any]], key: Optional[str]) -> Optional[Dict[str, Any]]:
    if not key:
        return None
    for entry in entries:
        if str(entry.get("model")) == str(key):
            return dict(entry)
    return None


def _take(entries: Sequence[Dict[str, Any]], keys: Sequence[str]) -> List[Dict[str, Any]]:
    """Entries for ``keys``, in the order of ``keys``, skipping unknown ones."""
    by_key = {str(entry.get("model")): dict(entry) for entry in entries}
    return [by_key[str(key)] for key in keys if str(key) in by_key]


def build_model_presentation(
    *,
    resolution: Optional[Dict[str, Any]],
    workflow_context: Dict[str, Any],
    model_request: Optional[Dict[str, Any]] = None,
    suggestions: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Decide what the model section should show for this request.

    Args:
        resolution: the structured intent from
            :func:`batteryml_web.agent.llm_intent.resolve_workflow_intent_with_llm`.
            Read for ``requested_model``, ``requested_model_name`` and
            ``recommendation_requested`` only; its prose decides nothing.
        workflow_context: the authoritative readiness context. The only
            source of what may be offered as runnable.
        model_request: the result of
            :func:`batteryml_web.agent.llm_intent.resolve_model_request`,
            which already judged the requested name against this context.
        suggestions: the result of
            :func:`batteryml_web.agent.llm_intent.filter_suggestions` — the
            agent's proposals, already split by readiness.

    Returns a dict with ``mode``, ``headline``, ``primary`` (runnable entries,
    at most :data:`PRIMARY_LIMIT`), ``blocked_focus`` (the blocked entries this
    request is actually about), ``alternatives`` (runnable keys offered in
    place of something unavailable), ``requires_confirmation``,
    ``default_model`` (None whenever the user must choose deliberately),
    ``selectable`` (every runnable key, so the choice is never restricted),
    ``all_runnable``, ``all_blocked`` and ``notes``.
    """
    resolution = resolution or {}
    runnable = _runnable_entries(workflow_context)
    blocked = _blocked_entries(workflow_context)
    runnable_keys = _keys(runnable)

    presentation: Dict[str, Any] = {
        "mode": MODE_BROWSE,
        "headline": "",
        "primary": [],
        "blocked_focus": [],
        "alternatives": [],
        "requires_confirmation": False,
        "default_model": None,
        "selectable": list(runnable_keys),
        "all_runnable": runnable,
        "all_blocked": blocked,
        "notes": [],
    }

    status = str((model_request or {}).get("status") or "none")
    requested_key = (model_request or {}).get("model_key")
    requested_name = (
        (model_request or {}).get("requested")
        or resolution.get("requested_model_name")
        or resolution.get("requested_model")
    )

    if status == "runnable" and requested_key:
        entry = _entry_for(runnable, requested_key)
        presentation.update(
            {
                "mode": MODE_REQUESTED_READY,
                "headline": "The model you asked for is ready to run.",
                "primary": [entry] if entry else [],
                "default_model": str(requested_key),
            }
        )
        others = [key for key in runnable_keys if key != str(requested_key)]
        if others:
            presentation["notes"].append(
                f"{len(others)} other model(s) are also runnable for this dataset "
                "and task; open the full list to see them."
            )
        return presentation

    if status == "blocked":
        focus = _entry_for(blocked, requested_key)
        alternatives = runnable_keys[:PRIMARY_LIMIT]
        presentation.update(
            {
                "mode": MODE_REQUESTED_BLOCKED,
                "headline": (
                    f"{requested_key or requested_name} cannot run for this dataset "
                    "and task."
                ),
                "blocked_focus": [focus] if focus else [],
                "primary": _take(runnable, alternatives),
                "alternatives": alternatives,
                "requires_confirmation": True,
                "default_model": None,
            }
        )
        reason = str((model_request or {}).get("reason") or "")
        if reason:
            presentation["notes"].append(reason)
        presentation["notes"].append(
            "Nothing was substituted. Choose one of the runnable models above to "
            "continue, or resolve the blocker and analyse again."
        )
        return presentation

    if status == "unknown":
        suggested = [
            key
            for key in (model_request or {}).get("alternatives", [])
            if key in runnable_keys
        ][:PRIMARY_LIMIT]
        presentation.update(
            {
                "mode": MODE_REQUESTED_UNKNOWN,
                "headline": f"'{requested_name}' is not registered on this platform.",
                "primary": _take(runnable, suggested),
                "alternatives": suggested,
                "requires_confirmation": True,
                "default_model": None,
            }
        )
        reason = str((model_request or {}).get("reason") or "")
        if reason:
            presentation["notes"].append(reason)
        if suggested:
            presentation["notes"].append(
                "These are the closest models that can run now. Select one "
                "deliberately; none was chosen for you."
            )
        return presentation

    # No model was named.
    agent_runnable = [
        key
        for key in (suggestions or {}).get("runnable", [])
        if key in runnable_keys
    ][:PRIMARY_LIMIT]
    agent_blocked = [
        key for key in (suggestions or {}).get("blocked", []) if key in _keys(blocked)
    ]

    if resolution.get("recommendation_requested"):
        chosen = agent_runnable or runnable_keys[:PRIMARY_LIMIT]
        presentation.update(
            {
                "mode": MODE_RECOMMENDATION,
                "headline": (
                    "Recommended for this dataset and task"
                    if chosen
                    else "No model can run for this dataset and task."
                ),
                "primary": _take(runnable, chosen),
                "blocked_focus": _take(blocked, agent_blocked),
                "default_model": chosen[0] if chosen else None,
            }
        )
        if agent_runnable:
            presentation["notes"].append(
                "Proposed by the agent and kept only where readiness says the model "
                "can run now."
            )
        elif chosen:
            presentation["notes"].append(
                "The agent proposed nothing runnable, so these are the models "
                "readiness reports as ready, in the backend's own order."
            )
        if agent_blocked:
            presentation["notes"].append(
                "Models the agent mentioned that cannot run are shown as blocked, "
                "never as options."
            )
        return presentation

    chosen = runnable_keys[:PRIMARY_LIMIT]
    presentation.update(
        {
            "mode": MODE_BROWSE,
            "headline": (
                "Models that can run for this dataset and task"
                if chosen
                else "No model can run for this dataset and task."
            ),
            "primary": _take(runnable, chosen),
            "default_model": chosen[0] if chosen else None,
        }
    )
    return presentation


def selectbox_options(presentation: Dict[str, Any]) -> List[str]:
    """Options for the model control, with a placeholder where needed.

    When the presentation requires confirmation the list starts with
    :data:`NO_SELECTION`, so nothing is preselected and the Run button stays
    disabled until the user has chosen a model themselves.
    """
    options = [str(key) for key in presentation.get("selectable", [])]
    if presentation.get("requires_confirmation"):
        return [NO_SELECTION] + options
    return options


def default_option_index(presentation: Dict[str, Any]) -> int:
    """Index of the option the control should open on."""
    options = selectbox_options(presentation)
    default = presentation.get("default_model")
    if default and str(default) in options:
        return options.index(str(default))
    return 0


def hidden_runnable_count(presentation: Dict[str, Any]) -> int:
    """How many runnable models the primary section did not name."""
    shown = {str(entry.get("model")) for entry in presentation.get("primary", [])}
    return len([key for key in presentation.get("selectable", []) if str(key) not in shown])


def hidden_blocked_count(presentation: Dict[str, Any]) -> int:
    """How many blocked models the primary section did not name."""
    shown = {str(entry.get("model")) for entry in presentation.get("blocked_focus", [])}
    return len(
        [
            entry
            for entry in presentation.get("all_blocked", [])
            if str(entry.get("model")) not in shown
        ]
    )
