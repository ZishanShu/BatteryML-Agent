"""Streamlit presentation layer for the verified agent workflow.

Boundary rules (milestone: agent web integration):

* readiness, validation, planning, execution and metrics all come from the
  existing backend APIs (``agent.workflow`` / ``agent.execution`` /
  ``agent.executor`` / ``agent.results`` / ``agent.result_agent``); nothing
  is re-derived here;
* no subprocess, no shell strings, no stdout metric parsing, no metric
  computation in this module — execution goes through ``execute_plan`` only,
  and only from the explicit Run button inside :func:`render_agent_workflow`;
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
from pathlib import Path
from typing import Any

import streamlit as st

from ..agent.execution import build_execution_plan
from ..agent.executor import execute_plan
from ..agent.intent import resolve_workflow_intent
from ..agent.result_agent import explain_run_result
from ..agent.results import METRIC_KEYS, SMOKE_RUN_WARNING, build_result_context
from ..agent.workflow import guide_workflow, workflow_guidance_preview
from .llm_state import is_llm_configured
from .progress_utils import get_safe_runtime_env

REPO_ROOT = Path(__file__).resolve().parents[2]

RUN_BUTTON_LABEL = "Run"
SMOKE_CHECKBOX_LABEL = (
    "1-epoch smoke test — validates the workflow only; not benchmark performance."
)
SMOKE_EPOCHS = 1
DEVICE_OPTIONS = ("cpu", "cuda:0")
#: Raw stdout/stderr are display tails only; metrics never come from them.
OUTPUT_TAIL_CHARS = 4000


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
    """The only execution override the UI may request: epochs=1 or nothing."""
    return SMOKE_EPOCHS if enabled else None


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
    """Trim raw stdout/stderr; session state keeps only small display tails."""
    compacted = dict(result)
    for key in ("stdout", "stderr"):
        compacted[key] = _tail_text(compacted.get(key))
    return compacted


def _render_context(context: dict[str, Any]) -> None:
    st.markdown("#### Workflow analysis")
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


def _render_plan(plan: dict[str, Any], device: str, seed: int) -> None:
    st.markdown("#### Execution plan")
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
        st.info(SMOKE_CHECKBOX_LABEL)


def _render_execution_status(result: dict[str, Any]) -> None:
    st.markdown("#### Execution status")
    st.write(f"**Started:** {'Yes' if result.get('started') else 'No'}")
    st.write(f"**Completed:** {'Yes' if result.get('completed') else 'No'}")
    st.write(f"**Success:** {bool(result.get('success'))}")
    st.write(f"**Return code:** {result.get('returncode', 'N/A')}")
    if result.get("error"):
        st.error(str(result["error"]))
    with st.expander("Execution debug output", expanded=False):
        command = result.get("command")
        if command:
            st.caption("Exact argv executed (built by the backend planner):")
            st.code(" ".join(str(part) for part in command), language="bash")
        st.caption("stdout")
        st.text(result.get("stdout") or "No stdout captured.")
        st.caption("stderr")
        st.text(result.get("stderr") or "No stderr captured.")


def _render_result(result_context: dict[str, Any]) -> None:
    st.markdown("#### Run result")
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


def render_agent_workflow() -> None:
    """Analyze -> select -> plan -> explicit Run -> structured result -> explain."""
    st.markdown("## 🤖 Agent Workflow")
    st.caption(
        "A natural-language request is resolved by the deterministic backend workflow; "
        "readiness, validation and metrics are authoritative. The optional LLM agent "
        "explains; it never triggers execution."
    )

    request = st.text_area(
        "Workflow request",
        value=st.session_state.get("agent_original_request", ""),
        placeholder="Use CPTransformer on MATR to predict battery lifetime.",
        key="agent_request_input",
        height=90,
    )
    if st.button("Analyze", key="agent_analyze", use_container_width=True):
        if not str(request or "").strip():
            st.error("Enter a workflow request before analyzing.")
        else:
            trimmed = str(request).strip()
            preview = workflow_guidance_preview(trimmed)
            st.session_state["agent_original_request"] = trimmed
            for key in (
                "agent_selected_model",
                "agent_execution_plan",
                "agent_execution_result",
                "agent_result_context",
                "agent_explanation",
            ):
                st.session_state.pop(key, None)
            if st.session_state.get("agent_model_choice") is not None:
                st.session_state.pop("agent_model_choice", None)
            if preview.get("workflow_context") is None:
                intent = resolve_workflow_intent(trimmed)
                _store_json_state("agent_workflow_context", None)
                st.session_state["agent_explanation"] = str(intent.get("message"))
            else:
                _store_json_state("agent_workflow_context", preview["workflow_context"])
                if is_llm_configured():
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
            st.rerun()

    context = st.session_state.get("agent_workflow_context")
    if not isinstance(context, dict):
        st.info("Enter a request and press Analyze to load the backend workflow state.")
        if st.session_state.get("agent_explanation"):
            st.write(st.session_state["agent_explanation"])
        return

    _render_context(context)

    options = runnable_model_keys(context)
    plan: dict[str, Any] | None = None
    smoke = False
    device = st.session_state.get("agent_device", DEVICE_OPTIONS[0])
    if device not in DEVICE_OPTIONS:
        device = DEVICE_OPTIONS[0]
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

        device = st.selectbox(
            "Device",
            list(DEVICE_OPTIONS),
            key="agent_device_select",
        )
        st.session_state["agent_device"] = str(device)

        smoke = st.checkbox(SMOKE_CHECKBOX_LABEL, key="agent_smoke_mode")

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
            with st.spinner("Executing the validated plan (explicit user action)..."):
                result = execute_plan(plan, repo_root=REPO_ROOT, env=get_safe_runtime_env())
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
            st.session_state["last_run_status"] = "success" if result.get("success") else "failed"
            st.session_state["last_run_code"] = result.get("returncode")
            st.rerun()

    if isinstance(execution_result, dict):
        _render_execution_status(execution_result)

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
            except Exception as exc:
                st.session_state["agent_explanation"] = f"LLM explanation failed: {exc}"
            st.rerun()

    if st.session_state.get("agent_explanation") and isinstance(context, dict):
        with st.expander("Agent explanation", expanded=True):
            st.caption("Explanation only — never a source of metrics or execution.")
            st.write(st.session_state["agent_explanation"])
