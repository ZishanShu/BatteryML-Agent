from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from ..core.schema_validator import OPTIONAL_FIELDS, REQUIRED_FIELDS


def _join_list(items: List[str]) -> str:
    return "\n".join(f"- {item}" for item in items)


def build_schema_columns_prompt(dataset_summary: Dict[str, Any]) -> str:
    return (
        "Describe the columns in this dataset for a BatteryML workflow. "
        "Return JSON only with keys: column_descriptions, data_quality_notes.\n\n"
        f"Columns: {dataset_summary.get('columns', [])}\n"
        f"Dtypes: {dataset_summary.get('dtypes', {})}\n"
        f"Missing values: {dataset_summary.get('missing_values', {})}\n"
        f"Preview: {dataset_summary.get('preview', [])}"
    )


def build_schema_mapping_prompt(dataset_summary: Dict[str, Any]) -> str:
    return (
        "Given the following dataset summary, map BatteryML schema fields to the best matching raw columns. "
        "Return JSON only with the structure: {\"mapping\": {field: column_name_or_empty_string}, \"rationale\": {field: explanation}}. "
        "Do not include markdown, code fences, or extra text.\n\n"
        f"Required schema fields: {REQUIRED_FIELDS}\n"
        f"Optional schema fields: {OPTIONAL_FIELDS}\n"
        f"Columns: {dataset_summary.get('columns', [])}\n"
        f"Dtypes: {dataset_summary.get('dtypes', {})}\n"
        f"Missing values: {dataset_summary.get('missing_values', {})}\n"
        f"Dataset summary: {dataset_summary}"
    )


def build_data_quality_prompt(dataset_summary: Dict[str, Any]) -> str:
    return (
        "Explain obvious data quality issues in this BatteryML dataset summary. "
        "Return JSON only with keys: issues, severity, recommended_checks.\n\n"
        f"Summary: {dataset_summary}"
    )


def build_config_explain_prompt(config_summary: Dict[str, Any]) -> str:
    return (
        "Explain this BatteryML config in simple terms. Return JSON only with keys: overview, key_components, caveats.\n\n"
        f"Config summary: {config_summary}"
    )


def build_model_task_prompt(config_summary: Dict[str, Any]) -> str:
    return (
        "Suggest whether the current BatteryML task/model pair looks reasonable and propose alternatives if needed. "
        "Return JSON only with keys: suggested_task, suggested_model, rationale.\n\n"
        f"Config summary: {config_summary}"
    )


def build_log_explain_prompt(log_text: str) -> str:
    return (
        "Explain this BatteryML run log. Return JSON only with keys: summary, likely_error, next_action.\n\n"
        f"Log excerpt:\n{log_text[:8000]}"
    )


def build_results_summary_prompt(results_summary: Dict[str, Any]) -> str:
    return (
        "Summarize these BatteryML results. Return JSON only with keys: summary, caveats, interpretation.\n\n"
        f"Results summary: {results_summary}"
    )


def build_agent_workflow_context(
    dataset: str | None = None,
    task: str | None = None,
    *,
    repo_root: str | Path | None = None,
    split_hint: str | None = None,
) -> Dict[str, Any]:
    """Compact deterministic workflow readiness for the LLM agent.

    Calls :func:`batteryml_web.core.readiness.get_workflow_context` and keeps
    only the fields the LLM needs to decide: which dataset/task was resolved,
    whether a deterministic preprocessing command is supported, which models
    are runnable (with their validator-approved config), which are blocked
    (with their missing inputs), and the ordered next steps.

    The full readiness dict is deliberately not passed through: it carries
    per-model prose, absolute paths and registry detail the prompt must not
    see. ``validated_config`` is copied verbatim from the readiness
    ``details`` — it is never invented here — and only entries whose status
    is READY appear under ``runnable_models``. ``repo_root`` is forwarded to
    the readiness layer so the helper is testable against a temporary
    repository root; it is never silently ignored. A caller must treat this
    structure as read-only state: the LLM may explain or recommend among
    runnable models, but the validator/readiness layer stays authoritative.
    """
    from ..core.readiness import ModelStatus, get_workflow_context

    context = get_workflow_context(dataset, task, repo_root=repo_root, split_hint=split_hint)
    data = context.get("dataset", {})

    runnable: List[Dict[str, Any]] = []
    for item in context.get("available_models", []):
        if item.get("status") != ModelStatus.READY.value or not item.get("ready"):
            continue
        details = item.get("details") or {}
        validated_config = details.get("validated_config")
        runnable.append(
            {
                "model": item.get("model"),
                "model_display": item.get("model_display"),
                "status": item.get("status"),
                "validated_config": validated_config,
                "config_selection": details.get("config_selection") or "chosen",
                "config_choices": list(details.get("config_choices") or []),
                "config_message": item.get("recommended_action"),
            }
        )
    runnable.sort(key=lambda entry: str(entry.get("model_display") or entry.get("model")))

    blocked: List[Dict[str, Any]] = []
    for item in context.get("blocked_models", []):
        if item.get("status") == ModelStatus.READY.value and item.get("ready"):
            continue
        blocked.append(
            {
                "model": item.get("model"),
                "status": item.get("status"),
                "missing": list(item.get("missing") or []),
                "recommended_action": item.get("recommended_action"),
            }
        )
    blocked.sort(key=lambda entry: str(entry.get("model")))

    steps = [str(step) for step in context.get("recommended_next_steps", [])]

    return {
        "dataset": data.get("dataset"),
        "dataset_status": data.get("status"),
        "task": context.get("task"),
        "preprocessing_supported": bool(data.get("preprocessing_supported", False)),
        "runnable_models": runnable,
        "blocked_models": blocked,
        "recommended_next_steps": steps,
    }
