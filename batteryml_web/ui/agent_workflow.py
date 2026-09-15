"""Streamlit presentation layer for the verified agent workflow.

Boundary rules (milestone: agent web integration):

* readiness, validation, planning, execution and metrics all come from the
  existing backend APIs (``agent.workflow`` / ``agent.execution`` /
  ``agent.executor`` / ``agent.results`` / ``agent.result_agent``); nothing
  is re-derived here;
* no subprocess, no shell strings, no stdout metric parsing, no metric
  computation in this module — execution goes through
  ``execute_plan_streaming`` only, and only from the explicit Run button
  inside :func:`render_agent_workflow`. Live progress is rendered from the
  structured events that function emits; the output is never a source of
  metrics, and no percentage is drawn that the run did not report;
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

from ..agent.catalog import build_example_requests, build_intent_catalog
from ..agent.execution import build_execution_plan
from ..agent.llm_intent import (
    filter_suggestions,
    resolve_model_request,
    resolve_workflow_intent_with_llm,
)
from ..agent.prompts import build_agent_workflow_context
from ..agent.result_agent import explain_run_result
from ..agent.results import METRIC_KEYS, SMOKE_RUN_WARNING, build_result_context
from ..agent.run_policy import (
    SMOKE_EPOCHS,
    SMOKE_NOTICE,
    classify_execution_failure,
    gpu_memory_advisory,
    smoke_device_recommendation,
    smoke_epochs_for,
)
from ..agent.streaming import (
    HEARTBEAT_MESSAGE,
    STAGE_FAILED,
    execute_plan_streaming,
)
from ..agent.workflow import guide_workflow
from ..core.device_inventory import (
    device_choices_with_manual_gpu,
    device_label,
    inventory_rows,
    query_gpu_inventory,
)
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
#: The two ways to supply a request. Ordered as rendered.
REQUEST_TABS: tuple[str, str] = ("Describe request", "Examples")
#: Placeholder for the request box. A template-free sentence: the clickable
#: examples, which do name a model, are built from the backend catalog.
REQUEST_PLACEHOLDER = "Use CPTransformer on MATR to predict battery lifetime."
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


def default_device_index(
    options: list[str], model_key: str, smoke: bool
) -> tuple[int, str]:
    """Index of the recommended default device, plus the reason to show.

    The recommendation is advice only: the returned index is the *initial*
    selection of an editable control, and any listed device may be chosen.
    """
    advice = smoke_device_recommendation(
        model_key, smoke=smoke, available_devices=options
    )
    device = str(advice.get("device") or "")
    index = options.index(device) if device in options else 0
    return index, str(advice.get("reason") or "")


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


def workspace_for_request(
    dataset: str, model: str, smoke: bool, project_root: str | None = None
) -> str:
    """Workspace path inside the repository runs/ area for one agent run."""
    base = str(project_root or "").strip("/") or "runs/web_projects/default"
    if not Path(base).is_relative_to("runs"):
        base = "runs/web_projects/default"
    safe_dataset = "".join(char if char.isalnum() else "_" for char in str(dataset)).lower().strip("_")
    safe_model = "".join(char if char.isalnum() else "_" for char in str(model)).lower().strip("_")
    suffix = "_smoke" if smoke else ""
    return f"{base}/agent_{safe_dataset}_{safe_model}{suffix}"


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


def _render_model_request(model_request: dict[str, Any] | None) -> None:
    """Explain what happened to an explicitly requested model.

    A blocked or unknown model is never swapped for another one: the
    alternatives are listed, and the user selects one deliberately below.
    """
    if not isinstance(model_request, dict):
        return
    status = model_request.get("status")
    if status in (None, "none", "runnable"):
        return
    alternatives = ", ".join(str(key) for key in model_request.get("alternatives", []))
    st.warning(str(model_request.get("reason") or "The requested model is unavailable."))
    if alternatives:
        st.write(
            f"Runnable alternatives: {alternatives}. Select one below to continue — "
            "nothing was substituted for you."
        )
    else:
        st.write("No model is runnable for this dataset and task right now.")


def _render_suggestions(suggestions: dict[str, Any] | None) -> None:
    """Show the agent's suggestions after the backend has filtered them."""
    if not isinstance(suggestions, dict):
        return
    runnable = [str(key) for key in suggestions.get("runnable", [])]
    blocked = [str(key) for key in suggestions.get("blocked", [])]
    if runnable:
        st.write(f"**Agent suggests (runnable now):** {', '.join(runnable)}")
    if blocked:
        st.write(
            f"**Also mentioned, but blocked:** {', '.join(blocked)} — "
            "these cannot be run; see the blocked models above for why."
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
    if runnable:
        st.markdown("**Runnable models**")
        st.write(", ".join(_display_name(entry) for entry in runnable))
    else:
        st.info("No model is runnable for this dataset/task right now.")

    blocked = blocked_model_entries(context)
    if blocked:
        st.markdown("**Blocked models** (not selectable for execution)")
        for entry in blocked:
            missing = ", ".join(str(item) for item in entry.get("missing", []))
            reason = f" — missing: {missing}" if missing else ""
            st.write(f"- {_display_name(entry)} — {entry.get('status', 'UNKNOWN')}{reason}")

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
) -> str:
    """One status line: the stage, the observed counter, and elapsed time.

    A counter appears only when the run itself reported one. There is no
    fallback percentage: an unknown position is shown as the stage plus the
    elapsed clock, never as an invented number.
    """
    parts = [str(stage or "Running")]
    if isinstance(current, int) and isinstance(total, int) and total > 0:
        parts.append(f"step {current}/{total}")
    parts.append(f"elapsed {_format_elapsed(elapsed_s)}")
    return " — ".join(parts)


def _format_elapsed(seconds: float) -> str:
    total = max(0, int(seconds))
    hours, rest = divmod(total, 3600)
    minutes, secs = divmod(rest, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


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
        caption = progress_caption(
            str(event.get("stage") or ""),
            event.get("current"),
            event.get("total"),
            elapsed,
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

    diagnosis = classify_execution_failure(
        result, device=device, inventory=inventory, smoke=smoke
    )
    if diagnosis["kind"] != "none":
        st.error(diagnosis["headline"])
        for action in diagnosis["actions"]:
            st.write(f"- {action}")
        if diagnosis["evidence_line"]:
            st.caption(f"From the run's own output: {diagnosis['evidence_line']}")
    elif result.get("error"):
        st.error(str(result["error"]))

    with st.expander("Execution debug output", expanded=False):
        command = result.get("command")
        if command:
            st.caption("Exact argv executed (built by the backend planner):")
            st.code(" ".join(str(part) for part in command), language="bash")
        if result.get("error"):
            st.caption("Executor error")
            st.text(str(result["error"]))
        st.caption("stdout")
        st.text(result.get("stdout") or "No stdout captured.")
        st.caption("stderr (full traceback, as produced by the run)")
        st.text(result.get("stderr") or "No stderr captured.")


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

    metrics = display_metrics(result_context)
    cols = st.columns(len(METRIC_KEYS))
    for col, key in zip(cols, METRIC_KEYS):
        if key in metrics:
            col.metric(key, str(metrics[key]))
            if key == "MAPE":
                col.caption(f"= {mape_percent(metrics[key])} (display only; ratio is canonical)")
        else:
            col.metric(key, "Unavailable")

    for label, value in result_summary_rows(result_context):
        st.write(f"**{label}:** {value if value is not None else 'Unavailable'}")

    warnings = result_context.get("warnings", [])
    if warnings:
        with st.expander("All backend warnings", expanded=False):
            for warning in warnings:
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
    st.session_state["agent_original_request"] = trimmed
    for key in (
        "agent_selected_model",
        "agent_execution_plan",
        "agent_execution_result",
        "agent_result_context",
        "agent_explanation",
        "agent_run_signature",
        "agent_model_request",
        "agent_suggestions",
    ):
        st.session_state.pop(key, None)
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


def _example_requests() -> list[str]:
    """Example requests for this installation, computed once per session.

    Built from the backend catalog and, where a workflow context has already
    been loaded, from the models readiness currently reports as runnable —
    so an example names a dataset and a model that really exist here. No
    model name is written into the UI.
    """
    cached = st.session_state.get("agent_example_requests")
    if isinstance(cached, list):
        return [str(item) for item in cached]
    context = st.session_state.get("agent_workflow_context")
    runnable = (
        runnable_model_keys(context) if isinstance(context, dict) else None
    )
    examples = build_example_requests(build_intent_catalog(), runnable_keys=runnable)
    st.session_state["agent_example_requests"] = list(examples)
    return list(examples)


def _render_examples() -> None:
    """Clickable examples that fill the request box and stop there.

    Clicking writes the example into the same session-state field the
    Describe request tab reads, and reruns so the text box shows it. It does
    not analyse and it does not execute: both remain explicit actions.
    """
    st.caption(
        "Pick one to fill the request box. Nothing is analysed or run until you "
        "press Analyze."
    )
    examples = _example_requests()
    if not examples:
        st.info(
            "No example can be offered: the config registry declares no dataset "
            "this platform could run a workflow on."
        )
        return
    for index, example in enumerate(examples):
        if st.button(example, key=f"agent_example_{index}", use_container_width=True):
            st.session_state["agent_original_request"] = example
            # The text box is a keyed widget: clearing its stored value lets
            # the freshly chosen example become its content on the rerun.
            st.session_state.pop("agent_request_input", None)
            st.rerun()


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

    describe_tab, examples_tab = st.tabs(list(REQUEST_TABS))
    with describe_tab:
        request = st.text_area(
            "Workflow request",
            value=st.session_state.get("agent_original_request", ""),
            placeholder=REQUEST_PLACEHOLDER,
            key="agent_request_input",
            height=90,
        )
        if st.button("Analyze", key="agent_analyze", use_container_width=True):
            if not str(request or "").strip():
                st.error("Enter a workflow request before analyzing.")
            else:
                _analyze_request(str(request).strip())
                st.rerun()
    with examples_tab:
        _render_examples()

    context = st.session_state.get("agent_workflow_context")
    if not isinstance(context, dict):
        _render_intent(st.session_state.get("agent_intent"))
        st.info("Enter a request and press Analyze to load the backend workflow state.")
        if st.session_state.get("agent_explanation"):
            st.write(st.session_state["agent_explanation"])
        return

    _render_intent(st.session_state.get("agent_intent"))
    _render_context(context)
    _render_model_request(st.session_state.get("agent_model_request"))
    _render_suggestions(st.session_state.get("agent_suggestions"))

    options = runnable_model_keys(context)
    plan: dict[str, Any] | None = None
    smoke = False
    inventory = _gpu_inventory()
    devices = device_options(inventory)
    device = devices[0]
    seed = 0

    if options:
        display_by_key = runnable_display_names(context)
        if st.session_state.get("agent_model_choice") not in options:
            st.session_state.pop("agent_model_choice", None)
        default_key = st.session_state.get("agent_selected_model")
        selected_key = st.selectbox(
            "Model (runnable models only; blocked models cannot be selected)",
            options=options,
            index=options.index(default_key) if default_key in options else 0,
            format_func=lambda key: display_by_key.get(key, key),
            key="agent_model_choice",
        )
        st.session_state["agent_selected_model"] = str(selected_key)

        # Smoke mode first: it is what makes CPU the recommended default for
        # the models whose smoke run must not compete for GPU memory.
        smoke = st.checkbox(SMOKE_CHECKBOX_LABEL, key="agent_smoke_mode")

        st.markdown("**Device**")
        _render_gpu_inventory(inventory)
        default_index, device_reason = default_device_index(
            devices, str(selected_key), smoke
        )
        if device_reason:
            st.info(device_reason)
        # The widget key encodes (model, smoke) so that switching model or
        # toggling smoke re-applies the backend's recommended default instead
        # of silently keeping a device chosen for a different combination.
        device = st.selectbox(
            "Device",
            devices,
            index=default_index,
            format_func=lambda name: device_label(name, inventory),
            key=f"agent_device_select_{selected_key}_{int(bool(smoke))}",
        )
        st.session_state["agent_device"] = str(device)
        advisory = gpu_memory_advisory(str(device), inventory)
        if advisory:
            st.warning(advisory)

        plan = build_execution_plan(
            str(context.get("dataset")),
            str(context.get("task")),
            str(selected_key),
            workflow_context=context,
            device=str(device),
            seed=seed,
            workspace=workspace_for_request(
                str(context.get("dataset")),
                str(selected_key),
                smoke,
                project_root=st.session_state.get("project_root"),
            ),
            epochs=smoke_epochs_value(smoke),
        )
        _store_json_state("agent_execution_plan", plan)
        if run_is_enabled(plan):
            # Keyed on the plan's subject, not on the device or the smoke box,
            # so fiddling with those does not toast (see ui/feedback.py).
            notify_success(
                f"Execution plan ready for {plan.get('model_display') or plan.get('model')}",
                event_key=EVENT_PLAN_READY,
                signature=plan_signature(plan),
            )
        _render_plan(plan, device=str(device), seed=seed)
    else:
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
        if st.button(
            RUN_BUTTON_LABEL,
            disabled=not run_is_enabled(plan),
            use_container_width=True,
            key="agent_run",
        ):
            st.caption("Executing the validated plan (explicit user action).")
            progress = _LiveProgress(st.empty(), st.empty(), st.empty())
            result = execute_plan_streaming(
                plan,
                on_event=progress.on_event,
                repo_root=REPO_ROOT,
                env=get_safe_runtime_env(),
            )
            # Provenance for the failure diagnosis only; neither field is read
            # back by the executor or by build_result_context.
            result["device"] = str(device)
            result["smoke"] = bool(smoke)
            workspace = result.get("workspace")
            if result.get("success") and workspace:
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
            _store_json_state("agent_execution_result", _compact_execution_output(result))
            _store_json_state("agent_result_context", result_context)
            st.session_state["agent_run_signature"] = current_signature
            # A repeat of the identical plan is still a new run worth
            # reporting, so the feedback signature carries a serial.
            st.session_state["agent_run_serial"] = (
                int(st.session_state.get("agent_run_serial", 0)) + 1
            )
            st.session_state["last_run_status"] = "success" if result.get("success") else "failed"
            st.session_state["last_run_code"] = result.get("returncode")
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
        _render_result(result_context)
        explain_disabled = not is_llm_configured()
        if explain_disabled:
            st.caption("Configure the optional LLM assistant to enable result explanation.")
        if st.button(
            "Explain computed result",
            disabled=explain_disabled,
            use_container_width=True,
            key="agent_explain_result",
        ):
            try:
                parsed = explain_run_result(
                    result_context,
                    base_url=st.session_state["llm_base_url"],
                    api_key=st.session_state["llm_api_key"],
                    model=st.session_state["llm_model"],
                )
                st.session_state["agent_explanation"] = explanation_text(parsed)
                st.session_state["agent_explanation_serial"] = (
                    int(st.session_state.get("agent_explanation_serial", 0)) + 1
                )
            except Exception as exc:
                st.session_state["agent_explanation"] = f"LLM explanation failed: {exc}"
            st.rerun()

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
