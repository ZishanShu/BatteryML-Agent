"""Streamlit presentation layer for the verified agent workflow.

Boundary rules (milestone: agent web integration):

* readiness, validation, planning, execution and metrics all come from the
  existing backend APIs (``agent.workflow`` / ``agent.execution`` /
  ``agent.executor`` / ``agent.results`` / ``agent.result_agent``); nothing
  is re-derived here;
* no subprocess, no shell strings, no stdout metric parsing, no metric
  computation in this module. The explicit Run button hands the validated
  plan to :class:`batteryml_web.core.run_manager.RunManager`, which executes
  it through ``execute_plan_streaming`` on its own thread. The page then
  only *polls* that run's structured state, which is what leaves it free to
  draw a Cancel button that can actually be clicked. Live progress is
  rendered from the events the run reported; the output is never a source of
  metrics, and no percentage is drawn that the run did not report;
* a run has an identity (``core/run_identity.py``) and a persisted record
  (``core/run_manifest.py``). The page reads history back from those files
  rather than from session state, so a refresh or a restart does not lose
  it, and a run that cannot be proven alive is reported INTERRUPTED rather
  than left claiming to be running;
* session state holds small JSON-serializable values only (request text,
  workflow context, selected model key, plan, execution result, result
  context, explanation) — never tensors, checkpoints or DataBundle objects.

The pure helpers (:func:`runnable_model_keys`, :func:`run_is_enabled`,
:func:`display_metrics`, ...) are deliberately free of Streamlit calls so
``tests/test_agent_web_ui.py`` can verify the UI/backend boundary without a
Streamlit runtime.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import streamlit as st

from ..agent.catalog import (
    REQUEST_TEMPLATE_STRUCTURE,
    build_intent_catalog,
    build_request_template,
)
from ..agent.execution import build_execution_plan
from ..agent.llm_intent import (
    filter_suggestions,
    resolve_model_request,
    resolve_workflow_intent_with_llm,
)
from ..agent.prompts import build_agent_workflow_context
from ..agent.recommendation import (
    MODE_BROWSE,
    MODE_RECOMMENDATION,
    NO_SELECTION,
    build_model_presentation,
    default_option_index,
    hidden_blocked_count,
    hidden_runnable_count,
    selectbox_options,
)
from ..agent.result_agent import explain_run_result
from ..agent.results import METRIC_KEYS, SMOKE_RUN_WARNING, build_result_context
from ..agent.run_policy import (
    SMOKE_EPOCHS,
    SMOKE_NOTICE,
    gpu_status_line,
    smoke_device_recommendation,
    smoke_epochs_for,
)
from ..agent.streaming import (
    HEARTBEAT_MESSAGE,
    STAGE_FAILED,
    execute_plan_streaming,
)
from ..agent.workflow import guide_workflow
from ..agent.progress_estimate import (
    EpochTimer,
    progress_line,
)
from ..agent.provenance import (
    artifact_display,
    build_run_context,
    context_rows,
)
from ..core.device_inventory import (
    build_device_selection,
    describe_selection,
    device_choices_with_manual_gpu,
    device_label,
    device_string_for_selection,
    gpu_option_label,
    inventory_rows,
    query_gpu_inventory,
    selection_from_devices,
)
from ..agent.failure_taxonomy import (
    CATEGORY_NONE,
    classify_llm_failure,
    classify_run_failure,
    explanation_failure_is_not_a_run_failure,
    redact,
)
from ..core import run_manifest as rm
from ..core.run_manifest import write_manifest
from ..core.run_identity import new_run_id
from ..core.run_manager import CANCELLED_MESSAGE, get_run_manager
from ..core.project_context import (
    agent_workspace,
    run_workspace,
    display_path,
    resolve_project,
)
from .formatting import metric_display
from .settings_panel import (
    clear_llm_configuration_request,
    llm_configuration_requested,
    render_llm_configuration_prompt,
    request_llm_configuration,
)
from .errors import render_copy_control
from .feedback import (
    EVENT_EXPLANATION_READY,
    EVENT_PLAN_READY,
    EVENT_REQUEST_ANALYZED,
    EVENT_RESULT_LOADED,
    EVENT_RUN_COMPLETED,
    notify_success,
    plan_signature,
)
from .llm_state import is_llm_configured
from .progress_utils import get_safe_runtime_env

REPO_ROOT = Path(__file__).resolve().parents[2]

RUN_BUTTON_LABEL = "Run"
#: Offered for any run that is executing, whatever the model. Stopping is
#: independent of whether that model can later be resumed — conflating them
#: would hide Stop from exactly the long runs that need it. The backend state
#: remains CANCELLED; this is the user-facing word for it.
CANCEL_BUTTON_LABEL = "Stop run"

#: Session key holding the run id reserved for the *prepared* plan. It is
#: minted once a runnable plan exists, so the page can show which run it is
#: about to create, and consumed when Run is pressed. Nothing is written to
#: disk until then: a reservation that is never used names no execution.
PENDING_RUN_ID_KEY = "agent_pending_run_id"
#: The plan the reserved id belongs to. Paired with PENDING_RUN_ID_KEY and
#: only ever written or cleared together with it; see clear_pending_run.
PENDING_RUN_SUBJECT_KEY = "agent_pending_run_subject"

#: Session key holding the id of the run this page is currently watching.
ACTIVE_RUN_KEY = "agent_active_run_id"

#: How often the progress fragment re-draws while a run is in flight. A
#: polled refresh, not a rerun loop: the script is not re-executed whole.
PROGRESS_POLL_SECONDS = 2.0
#: Label of the one request control. There are no tabs: an example gallery
#: was a second way to do the same thing, and the page now offers one.
REQUEST_LABEL = "Workflow request"

#: Fills the request box with a template drawn from this installation.
#: Replaces the earlier Tab-to-accept idea, which no stable Streamlit API
#: supports (docs/STREAMLIT_KEYBOARD_AUDIT.md), and the example gallery
#: before it. One helper, not a list of curated cases.
USE_TEMPLATE_LABEL = "Use template"
#: Session key holding the canonical request text. One key, written by
#: typing and by accepting the suggestion alike (milestone section 6).
REQUEST_STATE_KEY = "agent_original_request"

#: The request text area's own widget key. Named because accepting the
#: suggestion has to write it before the widget is instantiated.
REQUEST_WIDGET_KEY = "agent_request_input"

#: Set by the template button, consumed on the next run. A one-shot flag,
#: not a second request state: the text it produces goes into
#: REQUEST_STATE_KEY, the same key typing writes.
USE_TEMPLATE_KEY = "agent_use_template"
#: Exact wording milestone A requires wherever a smoke run is offered or shown.
SMOKE_CHECKBOX_LABEL = SMOKE_NOTICE
#: Raw stdout/stderr are display tails only; metrics never come from them.
OUTPUT_TAIL_CHARS = 4000
#: stderr carries the traceback, so it keeps a much larger tail than stdout
#: (which is mostly tqdm progress lines).
ERROR_TAIL_CHARS = 20000
#: Minimum seconds between two live-progress repaints. The backend emits an
#: event per tqdm update; repainting on every one would spend the run's time
#: in Streamlit rather than in BatteryML.
PROGRESS_REDRAW_INTERVAL_S = 0.25
#: How many trailing output lines the live log shows while a run is going.
LIVE_LOG_LINES = 12
#: The two compute modes, as the user picks them.
COMPUTE_MODES: tuple[str, str] = ("CPU", "GPU")
COMPUTE_LABEL = "Compute device"
#: Label of the single-GPU selector. The product supports CPU and one GPU;
#: distributed execution stays experimental infrastructure and is not offered
#: here (docs/DDP_CONTRACT.md).
GPU_SELECT_LABEL = "GPU"
EXPLAIN_BUTTON_LABEL = "Explain computed result"


def _display_name(entry: dict[str, Any]) -> str:
    return str(entry.get("model_display") or entry.get("model"))


def runnable_model_keys(context: dict[str, Any]) -> list[str]:
    """Canonical backend model keys that may be offered for execution."""
    return [str(entry.get("model")) for entry in context.get("runnable_models", [])]


def runnable_display_names(context: dict[str, Any]) -> dict[str, str]:
    """Canonical model key -> backend display name, from context entries only."""
    return {
        str(entry.get("model")): _display_name(entry)
        for entry in context.get("runnable_models", [])
    }


def blocked_model_entries(context: dict[str, Any]) -> list[dict[str, Any]]:
    return list(context.get("blocked_models", []))


def has_runnable_model(context: dict[str, Any]) -> bool:
    return bool(runnable_model_keys(context))


def run_is_enabled(plan: dict[str, Any] | None) -> bool:
    """The single gate behind the Run button: the backend planner must allow it."""
    return isinstance(plan, dict) and plan.get("can_execute") is True


def smoke_epochs_value(enabled: bool) -> int | None:
    """The only execution override the UI may request: epochs=1 or nothing.

    Delegates to the backend policy so the smoke definition lives in exactly
    one place (:mod:`batteryml_web.agent.run_policy`).
    """
    return smoke_epochs_for(enabled)


def device_options(inventory: dict[str, Any] | None) -> list[str]:
    """Selectable devices: cpu plus every GPU the backend actually enumerated.

    Never a hardcoded GPU count. When enumeration failed, a single manual
    ``cuda:0`` entry is kept so an explicit GPU choice remains possible, and
    the caller shows the inventory error beside it.
    """
    return device_choices_with_manual_gpu(inventory)


def gpu_device_options(inventory: dict[str, Any] | None) -> list[str]:
    """The GPUs a user may tick. CPU is the empty selection, not an entry."""
    return [name for name in device_options(inventory) if name != "cpu"]


def default_gpu_selection(model_key: str, smoke: bool) -> list[str]:
    """The GPUs the control opens with: none, i.e. CPU.

    CPU is the platform default, and for a smoke run of a model whose smoke
    policy recommends CPU it is also the recommendation. Returning an empty
    list means "run on CPU" — the user adds GPUs deliberately.
    """
    return []


def smoke_device_note(model_key: str, smoke: bool) -> str:
    """The backend's reason for recommending a CPU smoke run, or ''."""
    advice = smoke_device_recommendation(
        model_key, smoke=smoke, available_devices=["cpu"]
    )
    return str(advice.get("reason") or "")


def raw_only_notice(context: dict[str, Any]) -> str | None:
    """The backend notice for a dataset whose preprocessing route is unwired.

    Returns the exact ``recommended_next_steps`` text produced by the
    readiness backend — never a UI-constructed command.
    """
    if context.get("dataset_status") != "RAW_ONLY":
        return None
    if context.get("preprocessing_supported"):
        return None
    return next(
        (
            str(step)
            for step in context.get("recommended_next_steps", [])
            if "not yet wired" in str(step)
        ),
        None,
    )


def display_metrics(result_context: dict[str, Any] | None) -> dict[str, float]:
    """Canonical metrics a failed run may show: none, ever.

    Only a context whose ``success`` is exactly True may surface metrics, and
    only under the canonical RMSE/MAE/MAPE keys, verbatim from
    ``build_result_context``. MAPE stays the stored ratio.
    """
    if not isinstance(result_context, dict) or result_context.get("success") is not True:
        return {}
    metrics = result_context.get("metrics") or {}
    return {key: metrics[key] for key in METRIC_KEYS if key in metrics}


def mape_percent(value: float) -> str:
    """Supplementary percent reading; the raw ratio remains canonical."""
    return f"{float(value) * 100:.2f}%"


def smoke_warnings(result_context: dict[str, Any] | None) -> list[str]:
    """Warnings that must be shown next to the metrics for a short run."""
    if not isinstance(result_context, dict):
        return []
    return [
        str(warning)
        for warning in result_context.get("warnings", [])
        if str(warning) == SMOKE_RUN_WARNING
        or ("smoke run" in str(warning).lower() and "benchmark" in str(warning).lower())
    ]


def provenance_warnings(result_context: dict[str, Any] | None) -> list[str]:
    """Warnings shown when the result is not a clean verified run."""
    if not isinstance(result_context, dict) or result_context.get("provenance_verified") is not False:
        return []
    return [str(warning) for warning in result_context.get("provenance_warnings", [])]


def result_summary_rows(result_context: dict[str, Any] | None) -> list[tuple[str, Any]]:
    """Structured result fields the UI shows, verbatim from the context.

    Metrics are excluded here: they render through :func:`display_metrics`
    so the canonical ratio/float values stay distinct from the display.
    """
    if not isinstance(result_context, dict):
        return []
    figures = result_context.get("figure_files") or []
    return [
        ("Dataset", result_context.get("dataset")),
        ("Task", result_context.get("task")),
        ("Model", result_context.get("model")),
        ("Number of predictions", result_context.get("num_predictions")),
        ("Predictions artifact", result_context.get("predictions_file")),
        ("Figures", ", ".join(str(item) for item in figures) if figures else "None"),
    ]


def explanation_text(parsed: dict[str, Any]) -> str:
    """Human text from the explanation JSON, ignoring injected data keys."""
    for key in ("summary", "explanation", "interpretation"):
        value = parsed.get(key)
        if value:
            return str(value)
    payload = {k: v for k, v in parsed.items() if k not in ("metrics", "result_context")}
    return json.dumps(payload, indent=2, default=str)


def _store_json_state(key: str, value: dict[str, Any] | None) -> None:
    """Persist only small JSON-serializable workflow state."""
    if value is None:
        st.session_state.pop(key, None)
    else:
        st.session_state[key] = json.loads(json.dumps(value, default=str))


def _tail_text(text: Any, limit: int = OUTPUT_TAIL_CHARS) -> str:
    value = str(text or "")
    if len(value) <= limit:
        return value
    return f"...({len(value) - limit} earlier chars omitted)...\n" + value[-limit:]


def _compact_execution_output(result: dict[str, Any]) -> dict[str, Any]:
    """Trim raw stdout/stderr; session state keeps only display tails.

    stderr keeps the larger budget because a CUDA OOM traceback ends up
    there and milestone A requires it to stay readable in the debug view.
    """
    compacted = dict(result)
    compacted["stdout"] = _tail_text(compacted.get("stdout"), OUTPUT_TAIL_CHARS)
    compacted["stderr"] = _tail_text(compacted.get("stderr"), ERROR_TAIL_CHARS)
    return compacted


def _render_intent(resolution: dict[str, Any] | None) -> None:
    """Show how the request was understood, and what was refused.

    Everything displayed here came back from the backend resolver as a
    canonical identifier or as an explicit rejection; the LLM's prose is
    shown as rationale only and decides nothing.
    """
    if not isinstance(resolution, dict):
        return
    if resolution.get("source") == "llm":
        st.caption("Request interpreted by the LLM, then checked against the platform catalog.")
    elif resolution.get("llm_available"):
        st.caption("Request resolved by the deterministic resolver; no LLM call was needed.")
    else:
        st.caption(
            "No LLM is configured, so advanced natural-language understanding is "
            "unavailable. The deterministic resolver is being used; name the dataset "
            "and the task explicitly if it cannot tell."
        )
    rationale = str(resolution.get("user_facing_rationale") or "")
    if rationale:
        st.write(rationale)
    for warning in resolution.get("warnings", []):
        st.warning(str(warning))
    rejected = resolution.get("rejected") or []
    if rejected:
        with st.expander("Values the platform refused", expanded=False):
            for item in rejected:
                st.write(
                    f"- {item.get('field')}: {item.get('value')!r} — {item.get('kind')}"
                )
    question = str(resolution.get("clarification_question") or "")
    if resolution.get("needs_clarification") and question:
        st.info(question)


def _render_model_presentation(presentation: dict[str, Any]) -> None:
    """Show the models this request is about, and hide the rest.

    Everything named here came from the readiness context; the section never
    lists a model the backend did not report, and never presents a blocked
    one as an option. The full catalogues stay behind expanders so the
    primary view answers the request instead of enumerating the platform.
    """
    if not isinstance(presentation, dict):
        return

    headline = str(presentation.get("headline") or "")
    if headline:
        if presentation.get("requires_confirmation"):
            st.warning(headline)
        else:
            st.markdown(f"**{headline}**")

    for entry in presentation.get("blocked_focus", []):
        missing = ", ".join(str(item) for item in entry.get("missing", []))
        st.write(
            f"- {entry.get('model')} — {entry.get('status', 'UNKNOWN')}"
            + (f"; missing: {missing}" if missing else "")
        )
        action = entry.get("recommended_action")
        if action:
            st.caption(str(action))

    primary = presentation.get("primary", [])
    if primary:
        for entry in primary:
            st.write(
                f"- {entry.get('model_display') or entry.get('model')} "
                f"({entry.get('status', 'READY')})"
            )
    elif presentation.get("mode") in (MODE_RECOMMENDATION, MODE_BROWSE):
        st.info("No model is runnable for this dataset/task right now.")

    for note in presentation.get("notes", []):
        st.caption(str(note))

    hidden_runnable = hidden_runnable_count(presentation)
    if hidden_runnable:
        with st.expander(
            f"Show all compatible models ({hidden_runnable} more)", expanded=False
        ):
            for entry in presentation.get("all_runnable", []):
                st.write(
                    f"- {entry.get('model_display') or entry.get('model')} "
                    f"({entry.get('status', 'READY')})"
                )

    hidden_blocked = hidden_blocked_count(presentation)
    if hidden_blocked:
        with st.expander(
            f"Show all blocked models ({hidden_blocked})", expanded=False
        ):
            for entry in presentation.get("all_blocked", []):
                missing = ", ".join(str(item) for item in entry.get("missing", []))
                st.write(
                    f"- {entry.get('model')} — {entry.get('status', 'UNKNOWN')}"
                    + (f"; missing: {missing}" if missing else "")
                )


def _render_context(context: dict[str, Any]) -> None:
    st.subheader("Workflow analysis", anchor=False)
    cols = st.columns(3)
    cols[0].metric("Dataset", str(context.get("dataset") or "Unknown"))
    cols[1].metric("Task", str(context.get("task") or "Unknown"))
    cols[2].metric("Dataset readiness", str(context.get("dataset_status") or "UNKNOWN"))

    notice = raw_only_notice(context)
    if notice:
        st.warning(notice)

    runnable = context.get("runnable_models", [])
    blocked = blocked_model_entries(context)
    st.caption(
        f"{len(runnable)} model(s) runnable, {len(blocked)} blocked for this "
        "dataset and task."
    )
    # Which models to *name* is decided by _render_model_presentation from
    # what the request asked for. Listing every runnable model plus every
    # blocked one here buried the answer under the catalogue.

    steps = [str(step) for step in context.get("recommended_next_steps", [])]
    if steps:
        with st.expander("Backend recommended next steps", expanded=False):
            for step in steps:
                st.write(f"- {step}")


def _gpu_inventory(refresh: bool = False) -> dict[str, Any]:
    """Cached GPU inventory for this session.

    ``nvidia-smi`` is queried once per session (and on an explicit refresh)
    rather than on every Streamlit rerun. The value is a small
    JSON-serializable dict from the backend; nothing here inspects the GPU
    itself.
    """
    cached = st.session_state.get("gpu_inventory")
    if refresh or not isinstance(cached, dict):
        cached = query_gpu_inventory()
        st.session_state["gpu_inventory"] = cached
    return cached


def _render_gpu_inventory(inventory: dict[str, Any]) -> None:
    """Show what the machine actually reports; never a hardcoded GPU count."""
    rows = inventory_rows(inventory)
    if rows:
        st.caption(
            f"{len(rows)} CUDA device(s) reported by nvidia-smi. Free memory is a "
            "snapshot of a shared machine."
        )
        st.dataframe(rows, hide_index=True, use_container_width=True)
    elif inventory.get("error"):
        st.caption(f"GPU inventory unavailable: {inventory['error']}")


def _render_run_context(
    run_context: dict[str, Any] | None,
    *,
    title: str = "Run context",
) -> None:
    """Where this run writes and what it runs, from the plan itself.

    Shown before execution and preserved with the result, so the answer never
    changes between the two. Paths are repo-relative here; the exact absolute
    form stays in the details expander, because it is what is really on disk.
    """
    rows = context_rows(run_context)
    if not rows:
        return
    st.markdown(f"**{title}**")
    if run_context and run_context.get("project_is_default"):
        st.caption(
            "No project is loaded, so this run goes to the default project. "
            "Create or load a project on Home to keep runs separate."
        )
    st.dataframe(
        [{"Field": label, "Value": value} for label, value in rows],
        hide_index=True,
        use_container_width=True,
    )
    absolute = (run_context or {}).get("workspace")
    config = (run_context or {}).get("config")
    if absolute or config:
        with st.expander("Exact paths", expanded=False):
            if absolute:
                st.caption("Run directory")
                st.code(str(absolute), language="text")
            if config:
                st.caption("Config")
                st.code(str(config), language="text")


def _render_plan(plan: dict[str, Any], device: str, seed: int) -> None:
    st.subheader("Execution plan", anchor=False)
    rows = [
        {"Field": "Dataset", "Value": str(plan.get("dataset"))},
        {"Field": "Task", "Value": str(plan.get("task"))},
        {"Field": "Model", "Value": str(plan.get("model_display") or plan.get("model"))},
        {"Field": "Validated config", "Value": str(plan.get("config_path") or "Not available")},
        {"Field": "Device", "Value": device},
        {"Field": "Seed", "Value": str(seed)},
        {
            "Field": "Epochs override",
            "Value": str(plan.get("epochs")) if plan.get("epochs") is not None else "Config default",
        },
        {"Field": "Can execute", "Value": str(run_is_enabled(plan))},
    ]
    st.dataframe(rows, hide_index=True, use_container_width=True)
    if not run_is_enabled(plan):
        st.warning(str(plan.get("reason") or "The backend planner refused execution."))
    elif plan.get("epochs") == SMOKE_EPOCHS:
        st.info(SMOKE_NOTICE)


def progress_caption(
    stage: str,
    current: int | None,
    total: int | None,
    elapsed_s: float,
    *,
    eta_s: float | None = None,
    estimating: bool = False,
) -> str:
    """One status line: stage, the observed counter, elapsed, and an ETA.

    A counter appears only when the run itself reported one, and an ETA only
    when enough of that counter has completed to time. Where a counter exists
    but not enough of it, the line says "estimating…" rather than a number;
    where there is no counter at all it says neither. There is no fallback
    percentage and no invented countdown.
    """
    return progress_line(
        stage,
        current=current,
        total=total,
        elapsed_s=elapsed_s,
        eta_s=eta_s,
        estimating=estimating,
    )


class _LiveProgress:
    """Renders backend progress events into three Streamlit placeholders.

    Holds no execution logic: it only formats what
    :func:`batteryml_web.agent.streaming.execute_plan_streaming` reports. It
    never derives a percentage the run did not provide, and never reads a
    metric out of the output — ``fraction`` is drawn only when the event
    carries one.
    """

    def __init__(self, status_slot: Any, bar_slot: Any, log_slot: Any) -> None:
        self._status = status_slot
        self._bar = bar_slot
        self._log = log_slot
        self._started = time.monotonic()
        self._last_draw = 0.0
        self._stage = ""
        self._lines: list[str] = []
        # Times the run's own epoch counter. Produces an ETA only once enough
        # epochs have completed to measure one.
        self._timer = EpochTimer()

    def on_event(self, event: dict[str, Any]) -> None:
        message = str(event.get("message") or "")
        if event.get("type") in ("output", "training_progress") and message:
            self._lines.append(message)
            del self._lines[:-LIVE_LOG_LINES]

        stage = str(event.get("stage") or "")
        now = time.monotonic()
        forced = (
            event.get("type") == "completed"
            or (stage and stage != self._stage and message != HEARTBEAT_MESSAGE)
        )
        if not forced and (now - self._last_draw) < PROGRESS_REDRAW_INTERVAL_S:
            return
        self._stage = stage
        self._last_draw = now
        self._draw(event)

    def _draw(self, event: dict[str, Any]) -> None:
        elapsed = time.monotonic() - self._started
        self._timer.observe(event.get("current"), event.get("total"), elapsed)
        eta = self._timer.eta_seconds()
        has_counter = isinstance(event.get("current"), int) and isinstance(
            event.get("total"), int
        )
        caption = progress_caption(
            str(event.get("stage") or ""),
            event.get("current"),
            event.get("total"),
            elapsed,
            eta_s=eta,
            # A counter exists but has not produced enough completed units to
            # time yet: say so instead of guessing.
            estimating=has_counter and eta is None,
        )
        if event.get("stage") == STAGE_FAILED:
            self._status.error(caption)
        else:
            self._status.info(caption)
        fraction = event.get("fraction")
        # bool is an int subclass; True must never be read as 100%.
        if isinstance(fraction, (int, float)) and not isinstance(fraction, bool):
            self._bar.progress(min(max(float(fraction), 0.0), 1.0))
        if self._lines:
            self._log.code("\n".join(self._lines), language="text")


def _render_execution_status(
    result: dict[str, Any],
    *,
    device: str = "cpu",
    inventory: dict[str, Any] | None = None,
    smoke: bool = False,
) -> None:
    st.subheader("Execution status", anchor=False)
    st.write(f"**Started:** {'Yes' if result.get('started') else 'No'}")
    st.write(f"**Completed:** {'Yes' if result.get('completed') else 'No'}")
    st.write(f"**Success:** {bool(result.get('success'))}")
    st.write(f"**Return code:** {result.get('returncode', 'N/A')}")

    finding = classify_run_failure(
        result,
        device=device,
        inventory=inventory,
        smoke=smoke,
        status=str(st.session_state.get("last_run_status", "")).upper() or None,
    )
    if finding["category"] != CATEGORY_NONE:
        # What happened, in one line, and what the user can do — before any
        # traceback. The details stay one click away, never removed.
        st.error(f"Run failed — {finding['short_title']}")
        if finding["user_message"]:
            st.write(finding["user_message"])
        if finding["recommended_action"]:
            st.markdown("**Suggested action:**")
            st.markdown(finding["recommended_action"])
        if finding["evidence_line"]:
            st.caption(f"From the run's own output: {finding['evidence_line']}")
        # Stated explicitly, because "did it silently try something else?" is
        # the first question a scientist asks about a failed run.
        st.caption(
            "Nothing was retried and no device was changed on your behalf."
        )
    elif result.get("error"):
        st.error(redact(str(result["error"])))

    with st.expander("Show technical details", expanded=False):
        command = result.get("command")
        if command:
            st.caption("Exact argv executed (built by the backend planner):")
            st.code(" ".join(str(part) for part in command), language="bash")
        if result.get("error"):
            st.caption("Executor error")
            st.text(redact(str(result["error"])))
        st.caption("stdout")
        st.text(redact(result.get("stdout") or "No stdout captured."))
        st.caption("stderr (full traceback, as produced by the run)")
        st.text(redact(result.get("stderr") or "No stderr captured."))
        # One copy affordance for the whole block, with the same transient
        # confirmation every other copy in the app uses. Redacted before it
        # reaches the clipboard, not only before it reaches the screen.
        st.caption(
            "Redacted: API keys and other credentials are removed before this "
            "is shown or copied."
        )
        render_copy_control(
            _run_failure_details(result),
            key="agent_run_failure_copy",
        )


def _run_failure_details(result: dict[str, Any]) -> str:
    """The failed run's technical block as one redacted, copyable text."""
    command = result.get("command")
    sections = [
        f"returncode: {result.get('returncode', 'N/A')}",
        "",
        "command:",
        " ".join(str(part) for part in command) if command else "(not recorded)",
        "",
        "error:",
        str(result.get("error") or "(none)"),
        "",
        "stdout:",
        str(result.get("stdout") or "(empty)"),
        "",
        "stderr:",
        str(result.get("stderr") or "(empty)"),
    ]
    return redact("\n".join(sections))


def _explain_result(result_context: dict[str, Any]) -> None:
    """Call the existing explanation path. Never called by opening a form."""
    try:
        parsed = explain_run_result(
            result_context,
            base_url=st.session_state["llm_base_url"],
            api_key=st.session_state["llm_api_key"],
            model=st.session_state["llm_model"],
        )
        st.session_state.pop("agent_explanation_failure", None)
        st.session_state["agent_explanation"] = explanation_text(parsed)
        st.session_state["agent_explanation_serial"] = (
            int(st.session_state.get("agent_explanation_serial", 0)) + 1
        )
    except Exception as exc:  # noqa: BLE001 - reported, never escalated
        # An explanation is optional. Failing to produce one says nothing
        # about the run: the metrics came from BatteryML's own saved scores
        # and are untouched, and the run's status is not changed here.
        finding = classify_llm_failure(
            exc, configured=is_llm_configured()
        )
        st.session_state["agent_explanation_failure"] = finding
        st.session_state.pop("agent_explanation", None)


def _render_result(result_context: dict[str, Any]) -> None:
    st.subheader("Run result", anchor=False)
    for warning in provenance_warnings(result_context):
        st.warning(f"Provenance not verified: {warning}")
    for warning in smoke_warnings(result_context):
        st.warning(warning)
    if not result_context.get("success"):
        st.error("The run did not succeed; no metrics are reported.")
        for warning in result_context.get("warnings", []):
            st.caption(str(warning))
        return

    # Compact row. The canonical values stay exactly as build_result_context
    # returned them; only the strings drawn here are shortened, and the full
    # precision is one expander away.
    metrics = display_metrics(result_context)
    cells = [metric_display(metrics, key) for key in METRIC_KEYS]
    st.markdown(
        " &nbsp;·&nbsp; ".join(
            f"**{key}** {cell['compact']}"
            + (f" ({cell['percent']})" if cell["percent"] else "")
            if cell["present"]
            else f"**{key}** —"
            for key, cell in zip(METRIC_KEYS, cells)
        )
    )
    st.caption(
        "MAPE percentage is a display reading; the canonical value is the ratio."
    )

    artifact = artifact_display(result_context)
    if artifact["display"]:
        st.caption("Predictions artifact")
        st.code(artifact["display"], language="text")

    predictions = result_context.get("num_predictions")
    if predictions is not None:
        st.caption(f"{predictions} prediction(s)")

    with st.expander("Result details", expanded=False):
        for key, cell in zip(METRIC_KEYS, cells):
            if cell["present"]:
                st.caption(f"{key} (canonical): {cell['exact']}")
        for label, value in result_summary_rows(result_context):
            st.write(f"**{label}:** {value if value is not None else 'Unavailable'}")
        if artifact["absolute"]:
            st.caption("Predictions artifact (absolute)")
            st.code(artifact["absolute"], language="text")
        for warning in result_context.get("warnings", []):
            st.caption(str(warning))


def _resolve_intent(trimmed: str, catalog: dict[str, Any]) -> dict[str, Any]:
    """Intent resolution for one request, LLM-assisted where configured.

    Always returns the backend's structured result: canonical dataset, task
    and model key (or None), never free prose the caller has to parse. The
    LLM is consulted only when the deterministic resolver cannot answer, and
    its output has already been checked against the canonical catalog before
    it gets here.
    """
    if is_llm_configured():
        return resolve_workflow_intent_with_llm(
            trimmed,
            base_url=st.session_state["llm_base_url"],
            api_key=st.session_state["llm_api_key"],
            model=st.session_state["llm_model"],
            catalog=catalog,
        )
    return resolve_workflow_intent_with_llm(trimmed, catalog=catalog)


def _analyze_request(trimmed: str) -> None:
    """Resolve one request through the backend and store the result.

    Analysis only: it resolves intent, reads readiness and builds the compact
    workflow context, and optionally asks the LLM to explain it. Nothing is
    executed here — execution needs the explicit Run button below.
    """
    st.session_state[REQUEST_STATE_KEY] = trimmed
    for key in (
        "agent_selected_model",
        "agent_execution_plan",
        "agent_execution_result",
        "agent_result_context",
        "agent_explanation",
        "agent_explanation_failure",
        "agent_run_signature",
        "agent_model_request",
        "agent_suggestions",
        # C5: a new request must not leave the previous run's identity or
        # provenance on screen beside a fresh selection.
        "agent_run_context",
    ):
        st.session_state.pop(key, None)
    # The reservation belongs to the previous request. Released through the
    # one helper, so the id and its subject go together.
    clear_pending_run(st.session_state)
    if st.session_state.get("agent_model_choice") is not None:
        st.session_state.pop("agent_model_choice", None)

    # Built once: it scans the config registry and the dataset directories,
    # and both the resolver and the model-request check need the same menu.
    catalog = build_intent_catalog()
    resolution = _resolve_intent(trimmed, catalog)
    _store_json_state("agent_intent", resolution)

    if not resolution.get("dataset") or not resolution.get("task"):
        _store_json_state("agent_workflow_context", None)
        st.session_state["agent_explanation"] = str(
            resolution.get("clarification_question")
            or resolution.get("user_facing_rationale")
            or resolution.get("message")
            or "The request could not be resolved to a dataset and a task."
        )
        return

    context = build_agent_workflow_context(
        str(resolution["dataset"]), str(resolution["task"])
    )
    _store_json_state("agent_workflow_context", context)

    # What the request asked for, judged against the authoritative context.
    model_request = resolve_model_request(
        resolution.get("requested_model"), context, catalog
    )
    _store_json_state("agent_model_request", model_request)
    if model_request.get("status") == "runnable" and model_request.get("model_key"):
        st.session_state["agent_selected_model"] = str(model_request["model_key"])

    _store_json_state(
        "agent_suggestions",
        filter_suggestions(list(resolution.get("suggested_model_keys") or []), context),
    )
    notify_success(
        f"Request resolved: {context.get('dataset')} · {context.get('task')}",
        event_key=EVENT_REQUEST_ANALYZED,
        signature=[trimmed, context.get("dataset"), context.get("task")],
    )

    if not is_llm_configured():
        return
    try:
        guidance = guide_workflow(
            trimmed,
            base_url=st.session_state["llm_base_url"],
            api_key=st.session_state["llm_api_key"],
            model=st.session_state["llm_model"],
        )
        st.session_state["agent_explanation"] = str(guidance.get("guidance") or "")
    except Exception as exc:
        st.session_state["agent_explanation"] = f"LLM guidance failed: {exc}"


def request_template() -> str:
    """The one request template this installation offers.

    Derived from the live catalog and, where readiness is known, from the
    models actually runnable — so the filled sentence names nothing this
    platform cannot run. It is a sentence pattern, not a curated example and
    not a second recommendation engine: the real model/task recommendation
    stays with the agent backend.

    Cached per session because it is an input aid, not a live readout.
    """
    cached = st.session_state.get("agent_request_template")
    if isinstance(cached, str) and cached:
        return cached
    context = st.session_state.get("agent_workflow_context")
    runnable = runnable_model_keys(context) if isinstance(context, dict) else None
    # The task comes from the resolved workflow when there is one, so the
    # filled sentence names the task actually in play rather than assuming
    # every request is a lifetime prediction.
    task = context.get("task") if isinstance(context, dict) else None
    template = build_request_template(
        build_intent_catalog(), runnable_keys=runnable, task=task
    )
    st.session_state["agent_request_template"] = str(template)
    return str(template)


def accept_template(template: str, current_request: str) -> bool:
    """Whether filling with ``template`` may overwrite the request field.

    Pure, so the rule is testable without Streamlit. Filling is allowed only
    into an empty field: text the user typed is never overwritten, however
    the fill was triggered.
    """
    return bool(str(template or "").strip()) and not str(current_request or "").strip()


def pending_run_subject(
    *,
    dataset: str,
    model: str,
    seed: Any,
    device: str,
    smoke: bool,
) -> str:
    """What makes this plan *this* plan, as a comparable string.

    Pure, so the invalidation rule is testable without Streamlit. Two plans
    with the same subject are the same experiment and may share a reserved
    id; any difference means a different run.
    """
    return json.dumps(
        [dataset, model, str(seed), device, bool(smoke)], sort_keys=True
    )


def clear_pending_run(session_state) -> None:
    """Release the reservation — **both** keys, always together.

    The id and the subject are one value in two slots: the subject is the
    cache key for the id. Clearing one without the other leaves the pair
    inconsistent, and that is exactly what crashed the page — see
    :func:`reserved_run_id`. Every release goes through here so the pair
    cannot drift apart again.
    """
    for key in (PENDING_RUN_ID_KEY, PENDING_RUN_SUBJECT_KEY):
        session_state.pop(key, None)


def reserved_run_id(
    session_state,
    *,
    dataset: str,
    model: str,
    seed: Any,
    device: str,
    smoke: bool,
) -> str:
    """The id reserved for the currently prepared plan.

    Kept stable while the plan is unchanged, so the page does not appear to
    renumber the run on every rerun, and re-minted when any part of the plan
    that identifies the experiment changes. Reserving is not creating: no
    directory and no manifest exist until Run is pressed.

    **Re-mints when the id is missing, even if the subject still matches.**
    That is the repaired case: pressing Run used to pop the id alone, so the
    next rerun of the same plan matched the stale subject, skipped minting,
    and then read a key that was no longer there —

        KeyError: st.session_state has no key "agent_pending_run_id"

    — taking the whole page down with Streamlit's native traceback. The
    condition below is an ``or`` rather than a lone subject comparison so
    that a half-cleared pair heals itself instead of raising.

    Reads through ``.get`` and returns the local value, never a second
    lookup, and never stringifies ``None``: a missing id is falsy and mints a
    real one rather than becoming the literal ``"None"``.
    """
    subject = pending_run_subject(
        dataset=dataset, model=model, seed=seed, device=device, smoke=smoke
    )
    reserved = str(session_state.get(PENDING_RUN_ID_KEY) or "").strip()
    if not reserved or session_state.get(PENDING_RUN_SUBJECT_KEY) != subject:
        reserved = str(new_run_id(dataset=dataset, model=model, seed=seed))
        session_state[PENDING_RUN_SUBJECT_KEY] = subject
        session_state[PENDING_RUN_ID_KEY] = reserved
    return reserved


def _reserved_run_id(
    *,
    dataset: str,
    model: str,
    seed: Any,
    device: str,
    smoke: bool,
) -> str:
    """:func:`reserved_run_id` against the live Streamlit session."""
    return reserved_run_id(
        st.session_state,
        dataset=dataset,
        model=model,
        seed=seed,
        device=device,
        smoke=smoke,
    )


def _launch_run(
    *,
    run_id: str,
    plan: dict[str, Any],
    project: dict[str, Any],
    run_context: dict[str, Any],
    resumed_from: dict[str, Any] | None = None,
) -> None:
    """Create the run on disk, then start it in the background.

    Order matters: the manifest is written *before* anything executes, so a
    crash between the two leaves a PLANNED run on disk rather than a silent
    gap. The page then returns immediately — the job runs on the manager's
    own thread, which is what keeps Cancel clickable.
    """
    workspace = str(plan.get("command_workspace") or run_context.get("workspace") or "")
    if not workspace:
        workspace = str(_workspace_from_plan_command(plan) or "")
    run_dir = REPO_ROOT / workspace if workspace else None
    if run_dir is None:
        st.error(
            "The validated plan carries no workspace, so there is nowhere to "
            "record this run. Rebuild the plan before running."
        )
        return

    manifest = rm.build_manifest(
        run_id=run_id,
        project=str(project.get("name") or ""),
        workspace=workspace,
        dataset=plan.get("dataset"),
        task=plan.get("task"),
        model_key=plan.get("model"),
        model_display=plan.get("model_display"),
        config_path=plan.get("config_path"),
        seed=run_context.get("seed"),
        epochs=plan.get("epochs"),
        device=run_context.get("device"),
        device_selection=st.session_state.get("agent_device_selection") or {},
        smoke=bool(run_context.get("smoke")),
        command=plan.get("command") or [],
        git_commit=_git_commit(),
        # Present only for a resumed run. The source run is never touched:
        # this is a new run that records where it continued from.
        resumed_from=resumed_from,
    )
    write_manifest(run_dir, manifest)

    get_run_manager().start(
        run_id=run_id,
        run_dir=run_dir,
        plan=plan,
        repo_root=REPO_ROOT,
        env=get_safe_runtime_env(),
    )
    st.session_state[ACTIVE_RUN_KEY] = run_id


def _workspace_from_plan_command(plan: dict[str, Any]) -> str | None:
    """The ``--workspace`` value in the plan's own argv."""
    command = plan.get("command")
    if not isinstance(command, (list, tuple)):
        return None
    parts = [str(part) for part in command]
    if "--workspace" not in parts:
        return None
    index = parts.index("--workspace")
    return parts[index + 1] if index + 1 < len(parts) else None


def _git_commit() -> str | None:
    """The repository commit, read from .git without shelling out.

    Returns None when it cannot be read; a run that cannot state its commit
    says so rather than carrying a plausible-looking wrong one.
    """
    head = REPO_ROOT / ".git" / "HEAD"
    try:
        text = head.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    if text.startswith("ref:"):
        ref = (REPO_ROOT / ".git" / text.split(" ", 1)[1].strip())
        try:
            return ref.read_text(encoding="utf-8").strip() or None
        except OSError:
            return None
    return text or None


def _finish_active_run(run_id: str, snapshot: dict[str, Any], *, smoke: bool) -> None:
    """Move a finished background run into the page's result state.

    A result context is built **only** for a run that completed. A cancelled
    or failed run has measured nothing, so none is built and no metric is
    offered for it.
    """
    manager = get_run_manager()
    handle = manager.get(run_id)
    result = dict(getattr(handle, "result", None) or {})
    status = str(snapshot.get("status") or "")
    result_context = None
    if status == rm.STATUS_COMPLETED and result.get("success"):
        workspace = result.get("workspace")
        plan = st.session_state.get("agent_execution_plan") or {}
        if workspace:
            result_context = build_result_context(
                workspace,
                dataset=plan.get("dataset"),
                task=plan.get("task"),
                model=plan.get("model"),
                config_path=plan.get("config_path"),
                run_status="SUCCESS",
                execution_result=result,
                smoke_run=smoke,
            )
            metrics = (result_context or {}).get("metrics") or {}
            if metrics:
                try:
                    rm.update_manifest(
                        handle.run_dir, {"metrics": dict(metrics)},
                        enforce_transition=False,
                    )
                except (FileNotFoundError, ValueError):
                    pass
    _store_json_state("agent_execution_result", _compact_execution_output(result))
    _store_json_state("agent_result_context", result_context)
    st.session_state["agent_run_serial"] = (
        int(st.session_state.get("agent_run_serial", 0)) + 1
    )
    st.session_state["last_run_status"] = status.lower()
    st.session_state["last_run_code"] = result.get("returncode")
    st.session_state.pop(ACTIVE_RUN_KEY, None)
    manager.forget(run_id)


def _render_active_run(
    run_id: str,
    *,
    plan: dict[str, Any],
    smoke: bool,
    device: str,
) -> None:
    """Status, live progress and a working Cancel for the run in flight.

    The refresh is a fragment on a timer, not a rerun loop: only this block
    re-executes, on a fixed interval, so the rest of the page (and the user's
    typing) is untouched and the server is not spun.
    """
    manager = get_run_manager()
    snapshot = manager.snapshot(run_id)
    if snapshot is None:
        # The manager has no memory of this run — this process did not start
        # it, or it was already reaped. Say so rather than showing a spinner
        # forever; Run History below still has the persisted truth.
        st.warning(
            f"Run {run_id} is no longer tracked by this server process. Its "
            "recorded state is in Run History below."
        )
        st.session_state.pop(ACTIVE_RUN_KEY, None)
        return

    if snapshot.get("finished"):
        _finish_active_run(run_id, snapshot, smoke=smoke)
        st.rerun()
        return

    st.caption(f"Run {run_id} is executing (started by an explicit user action).")

    @st.fragment(run_every=PROGRESS_POLL_SECONDS)
    def _live() -> None:
        current = manager.snapshot(run_id)
        if current is None:
            return
        event = current.get("event") or {}
        elapsed = float(current.get("elapsed_s") or 0.0)
        timer = _progress_timer(run_id)
        timer.observe(event.get("current"), event.get("total"), elapsed)
        eta = timer.eta_seconds()
        has_counter = isinstance(event.get("current"), int) and isinstance(
            event.get("total"), int
        )
        st.caption(
            progress_caption(
                str(event.get("stage") or ""),
                event.get("current"),
                event.get("total"),
                elapsed,
                eta_s=eta,
                estimating=has_counter and eta is None,
            )
        )
        fraction = event.get("fraction")
        if isinstance(fraction, (int, float)):
            # Drawn only when the run itself reported a counter.
            st.progress(min(max(float(fraction), 0.0), 1.0))
        tail = current.get("tail") or []
        if tail:
            st.code("\n".join(str(line) for line in tail[-LIVE_LOG_LINES:]), language="text")
        if current.get("finished"):
            st.rerun()

    _live()

    if st.session_state.get(f"cancelling_{run_id}"):
        st.warning("Stopping this run…")
    if st.button(
        CANCEL_BUTTON_LABEL,
        key=f"agent_cancel_{run_id}",
        use_container_width=True,
    ):
        st.session_state[f"cancelling_{run_id}"] = True
        # Terminates this run's process group and nothing else.
        manager.cancel(run_id)
        st.rerun()


def _progress_timer(run_id: str) -> EpochTimer:
    """One ETA estimator per run, kept across fragment redraws."""
    timers = st.session_state.setdefault("_agent_progress_timers", {})
    if run_id not in timers:
        timers[run_id] = EpochTimer()
    return timers[run_id]


def _render_result_figures(result_context: dict[str, Any] | None) -> None:
    """Observed-vs-predicted, residuals and the prediction table.

    Only for a successful run, and only from the predictions artifact the
    backend result context names — the same file the canonical metrics came
    from. Nothing here recomputes a metric; a run whose artifact lacks
    observed labels gets the stated reason instead of a figure.
    """
    if not isinstance(result_context, dict) or result_context.get("success") is not True:
        return
    predictions_file = result_context.get("predictions_file")
    if not predictions_file:
        return
    from ..agent.result_views import RUL_SCHEMA, TRAJECTORY_SCHEMA, load_view
    from .run_history import (
        render_prediction_table,
        render_rul_figures,
        render_trajectory_figure,
    )

    try:
        views = load_view(str(predictions_file))
    except Exception as exc:  # noqa: BLE001 - a figure must not break a result
        st.warning(
            f"The predictions artifact could not be read "
            f"({type(exc).__name__}: {exc}); no figure is drawn from it. The "
            "metrics above are unaffected."
        )
        return
    if views["schema"] == RUL_SCHEMA:
        render_rul_figures(views["rul"])
        render_prediction_table(views["rul"])
    elif views["schema"] == TRAJECTORY_SCHEMA:
        render_trajectory_figure(views["trajectory"])


def render_agent_workflow() -> None:
    """Analyze -> select -> plan -> explicit Run -> structured result -> explain."""
    st.header("Agent Workflow", anchor=False)
    st.caption(
        "A natural-language request is resolved by the deterministic backend workflow; "
        "readiness, validation and metrics are authoritative. The optional LLM agent "
        "explains; it never triggers execution."
    )

    # A request handed over from Home. Analysed once, on arrival, so the page
    # opens on the backend's answer instead of an empty form. Popped first so
    # a rerun cannot re-analyse it.
    pending = st.session_state.pop("agent_pending_request", None)
    if isinstance(pending, str) and pending.strip():
        _analyze_request(pending.strip())

    # One request control. The suggestion is grey placeholder text while the
    # field is empty: it is a hint, not the user's request, and nothing is
    # analysed or executed until Analyze is pressed.
    template = request_template()

    # The fill is applied *here*, before the text area is instantiated.
    # A keyed widget takes its content from its own session key, and that key
    # may only be written before the widget exists; writing it afterwards (or
    # popping it and relying on ``value=``) leaves the box empty, which is
    # exactly what a browser check of the first attempt showed.
    if st.session_state.pop(USE_TEMPLATE_KEY, False):
        # Re-checked against the value the widget actually committed.
        # Streamlit commits a text area on blur, so a user can be typing
        # while the helper is still on screen; clicking it then blurs the box
        # and commits their text. Without this second check that text would
        # be overwritten — which a browser check caught.
        committed = str(st.session_state.get(REQUEST_WIDGET_KEY, "") or "")
        if accept_template(template, committed):
            st.session_state[REQUEST_STATE_KEY] = template
            st.session_state[REQUEST_WIDGET_KEY] = template

    request = st.text_area(
        REQUEST_LABEL,
        value=st.session_state.get(REQUEST_STATE_KEY, ""),
        # The generic shape, so the box teaches the pattern without
        # pretending a curated example is waiting there.
        placeholder=REQUEST_TEMPLATE_STRUCTURE,
        key=REQUEST_WIDGET_KEY,
        height=90,
    )
    if accept_template(template, request):
        # Only reachable while the field is empty, so typed text can never be
        # overwritten. A button rather than the Tab key: see
        # docs/STREAMLIT_KEYBOARD_AUDIT.md.
        if st.button(USE_TEMPLATE_LABEL, key="agent_use_template"):
            # Only records the intent and reruns; the write happens above, on
            # the next run, before the widget is built. Filling the box is all
            # it does: no analysis, no LLM call, no plan, no execution.
            st.session_state[USE_TEMPLATE_KEY] = True
            st.rerun()
    if st.button("Analyze", key="agent_analyze", use_container_width=True):
        if not str(request or "").strip():
            st.error("Enter a workflow request before analyzing.")
        else:
            _analyze_request(str(request).strip())
            st.rerun()

    context = st.session_state.get("agent_workflow_context")
    if not isinstance(context, dict):
        _render_intent(st.session_state.get("agent_intent"))
        st.info("Enter a request and press Analyze to load the backend workflow state.")
        if st.session_state.get("agent_explanation"):
            st.write(st.session_state["agent_explanation"])
        return

    _render_intent(st.session_state.get("agent_intent"))
    _render_context(context)

    presentation = build_model_presentation(
        resolution=st.session_state.get("agent_intent"),
        workflow_context=context,
        model_request=st.session_state.get("agent_model_request"),
        suggestions=st.session_state.get("agent_suggestions"),
    )
    _render_model_presentation(presentation)

    options = runnable_model_keys(context)
    plan: dict[str, Any] | None = None
    smoke = False
    selected_key: str = NO_SELECTION
    # The project decides where a run is written. Resolved once, here, so the
    # workspace the page shows and the workspace the plan carries cannot come
    # from different places.
    project = resolve_project(
        project_root=st.session_state.get("project_root"),
        project_name=st.session_state.get("project_name"),
    )
    inventory = _gpu_inventory()
    devices = device_options(inventory)
    device = devices[0]
    seed = 0

    if options:
        display_by_key = runnable_display_names(context)
        choices = selectbox_options(presentation)
        if st.session_state.get("agent_model_choice") not in choices:
            st.session_state.pop("agent_model_choice", None)
        default_key = st.session_state.get("agent_selected_model")
        index = (
            choices.index(default_key)
            if default_key in choices
            else default_option_index(presentation)
        )
        selected_key = st.selectbox(
            "Model (runnable models only; blocked models cannot be selected)",
            options=choices,
            index=index,
            format_func=lambda key: (
                "— select a model —" if key == NO_SELECTION else display_by_key.get(key, key)
            ),
            key="agent_model_choice",
        )
    if options and selected_key == NO_SELECTION:
        # The request named something unavailable, so nothing is chosen for
        # the user. No plan is built, so Run is unreachable until they select
        # a model themselves. The rest of the page still renders: an earlier
        # run's results must not vanish because the new request needs a
        # decision.
        st.session_state.pop("agent_selected_model", None)
        st.info("Select one of the runnable models above to build an execution plan.")

    if options and selected_key != NO_SELECTION:
        st.session_state["agent_selected_model"] = str(selected_key)

        # Smoke mode first: it is what makes CPU the recommended default for
        # the models whose smoke run must not compete for GPU memory.
        smoke = st.checkbox(SMOKE_CHECKBOX_LABEL, key="agent_smoke_mode")

        st.markdown(f"**{COMPUTE_LABEL}**")
        device_reason = smoke_device_note(str(selected_key), smoke)
        if device_reason:
            st.info(device_reason)

        gpu_options = gpu_device_options(inventory)
        # CPU is an explicit choice, not "the absence of a GPU choice".
        mode = st.radio(
            COMPUTE_LABEL,
            list(COMPUTE_MODES),
            index=0,
            horizontal=True,
            label_visibility="collapsed",
            key=f"agent_compute_mode_{selected_key}_{int(bool(smoke))}",
        )
        if mode == "GPU" and not gpu_options:
            st.warning(
                "No CUDA device was enumerated on this machine, so GPU mode is "
                "unavailable. Runs will use the CPU."
            )
            mode = "CPU"

        chosen_gpu: str | None = None
        if mode == "GPU":
            # Exactly one GPU. The product supports CPU and a single GPU;
            # distributed execution is experimental infrastructure and is not
            # offered here (docs/DDP_CONTRACT.md). Offering a multi-select for
            # something no model can execute put an error box in front of
            # every user who tried it.
            chosen_gpu = st.selectbox(
                GPU_SELECT_LABEL,
                gpu_options,
                index=0,
                format_func=lambda name: gpu_option_label(name, inventory),
                key=f"agent_gpu_choice_{selected_key}_{int(bool(smoke))}",
            )

        selection = selection_from_devices([chosen_gpu] if chosen_gpu else [])
        _store_json_state("agent_device_selection", selection)
        device_value = device_string_for_selection(selection)
        # The readable form for the provenance block ("CPU", "GPU 3"), which
        # is separate from the executable --device string below.
        device_label_text = describe_selection(selection)

        if chosen_gpu:
            # One compact line. A warning box only when the figure deserves
            # one — a full-width warning for an idle GPU teaches people to
            # ignore warnings.
            status = gpu_status_line(str(chosen_gpu), inventory)
            if status["text"]:
                if status["severity"] == "warning":
                    st.warning(status["text"])
                else:
                    st.caption(status["text"])
        else:
            st.caption(device_label_text)

        if device_value is not None:
            device = str(device_value)
            st.session_state["agent_device"] = device
            # The run this page is about to create. Reserved (not created):
            # minting happens once a runnable plan exists so the id can be
            # shown before Run, and the directory and manifest appear only
            # when Run is pressed. Pressing Run consumes it, so a second run
            # of the identical plan gets a different id.
            pending_run_id = _reserved_run_id(
                dataset=str(context.get("dataset")),
                model=str(selected_key),
                seed=seed,
                device=device,
                smoke=smoke,
            )
            plan = build_execution_plan(
                str(context.get("dataset")),
                str(context.get("task")),
                str(selected_key),
                workflow_context=context,
                device=device,
                seed=seed,
                # The run-specific directory, so the plan the user reviews is
                # exactly the plan that executes, and two runs of the same
                # model cannot overwrite each other.
                workspace=run_workspace(
                    project,
                    str(context.get("dataset")),
                    str(selected_key),
                    pending_run_id,
                    smoke=smoke,
                ),
                epochs=smoke_epochs_value(smoke),
            )
            _store_json_state("agent_execution_plan", plan)
            run_context = build_run_context(
                project=project,
                plan=plan,
                device_label=device_label_text,
                seed=seed,
                smoke=smoke,
            )
            run_context["run_id"] = pending_run_id
            _store_json_state("agent_run_context", run_context)
            if run_is_enabled(plan):
                # Keyed on the plan's subject, not on the device or the smoke
                # box, so fiddling with those does not toast (ui/feedback.py).
                notify_success(
                    f"Execution plan ready for "
                    f"{plan.get('model_display') or plan.get('model')}",
                    event_key=EVENT_PLAN_READY,
                    signature=plan_signature(plan),
                )
            _render_run_context(run_context, title="Run context")
            _render_plan(plan, device=device, seed=seed)

    if not options:
        # The presentation above already named the blocked models and why;
        # this says what it means for running anything at all.
        st.info(
            "No model can be executed for this dataset/task. Blocked models are "
            "displayed with their reason and cannot be run."
        )

    execution_result = st.session_state.get("agent_execution_result")
    result_context = st.session_state.get("agent_result_context")

    if options and plan is not None:
        current_signature = json.dumps(plan.get("command") or [], sort_keys=True, default=str)
        stored_signature = st.session_state.get("agent_run_signature")
        if stored_signature is not None and stored_signature != current_signature:
            # The plan changed since the last execution: results shown would
            # be stale, so they are dropped until an explicit new Run.
            st.session_state.pop("agent_execution_result", None)
            st.session_state.pop("agent_result_context", None)
            execution_result = None
            result_context = None
        active_run_id = st.session_state.get(ACTIVE_RUN_KEY)
        if active_run_id:
            # A run this page started is in flight (or has just ended). The
            # script is not blocked on it, so Cancel below is a live control.
            _render_active_run(str(active_run_id), plan=plan, smoke=smoke, device=device)
        elif st.button(
            RUN_BUTTON_LABEL,
            disabled=not run_is_enabled(plan),
            use_container_width=True,
            key="agent_run",
        ):
            # Explicit user action. Everything before this point was planning.
            run_id = str(st.session_state.get(PENDING_RUN_ID_KEY) or "")
            _launch_run(
                run_id=run_id,
                plan=plan,
                project=project,
                run_context=st.session_state.get("agent_run_context") or {},
            )
            # The reservation is spent: a repeat of this plan is a new run.
            # Both keys, or the next rerun of this same plan would match the
            # stale subject, skip minting and read an id that is gone.
            clear_pending_run(st.session_state)
            st.session_state["agent_run_signature"] = current_signature
            st.rerun()

    serial = st.session_state.get("agent_run_serial", 0)
    if isinstance(execution_result, dict) and execution_result.get("success"):
        notify_success(
            "Run completed",
            event_key=EVENT_RUN_COMPLETED,
            signature=[serial, execution_result.get("workspace"), execution_result.get("returncode")],
        )
    if isinstance(result_context, dict) and result_context.get("success"):
        notify_success(
            "Results loaded from the run workspace",
            event_key=EVENT_RESULT_LOADED,
            signature=[serial, result_context.get("workspace")],
        )

    if isinstance(execution_result, dict):
        _render_execution_status(
            execution_result,
            device=str(execution_result.get("device") or device),
            inventory=inventory,
            smoke=bool(execution_result.get("smoke", smoke)),
        )

    if isinstance(result_context, dict):
        # The provenance the page showed before the run, preserved with the
        # result so "where did this come from" has one answer either side.
        _render_run_context(
            st.session_state.get("agent_run_context"), title="Run provenance"
        )
        _render_result(result_context)
        # The scientific views for the run that just finished, from its own
        # saved artifact and through the same reader the history view uses.
        _render_result_figures(result_context)

        # Always clickable. When the assistant is not configured the click
        # opens the configuration form instead of doing nothing.
        if st.button(
            EXPLAIN_BUTTON_LABEL,
            use_container_width=True,
            key="agent_explain_result",
        ):
            if is_llm_configured():
                _explain_result(result_context)
                st.rerun()
            else:
                request_llm_configuration("explain")
                st.rerun()

        if llm_configuration_requested():
            action = render_llm_configuration_prompt()
            if action in ("save", "connect"):
                # Settings are stored (and, for Connect, tested). Close the
                # form and stay on the result: Explain is now usable, and the
                # user presses it again if they want an explanation. Nothing
                # is sent to the LLM on their behalf.
                clear_llm_configuration_request()
                st.rerun()
            # "connect_failed" leaves the form open with its error showing,
            # so the endpoint can be corrected without reopening anything.

    explanation_failure = st.session_state.get("agent_explanation_failure")
    if isinstance(explanation_failure, dict) and isinstance(context, dict):
        # Shown beside the result, never instead of it: the metrics above came
        # from BatteryML's own saved scores and are unaffected by the
        # assistant being unreachable.
        st.warning(
            f"{explanation_failure.get('short_title')} "
            f"{explanation_failure.get('user_message')}"
        )
        if explanation_failure.get("recommended_action"):
            st.caption(str(explanation_failure["recommended_action"]))
        st.caption(
            "The computed result above is unaffected: it comes from the run's "
            "own saved scores, not from the assistant."
        )

    if st.session_state.get("agent_explanation") and isinstance(context, dict):
        explanation_serial = st.session_state.get("agent_explanation_serial")
        if explanation_serial:
            notify_success(
                "Result explanation ready",
                event_key=EVENT_EXPLANATION_READY,
                signature=explanation_serial,
            )
        with st.expander("Agent explanation", expanded=True):
            st.caption("Explanation only — never a source of metrics or execution.")
            st.write(st.session_state["agent_explanation"])
