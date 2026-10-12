"""Deterministic alternatives for a request the platform cannot run as asked.

"Dataset READY" describes the data only. Whether a *dataset + task + model*
combination can run is a different question, answered by the configuration
registry (is there a run config for it at all?) and by readiness/validation
(is that config's data present and does the config pass the gate?). A user
who asked for "ElasticNet on HUST to predict SOH" and saw "HUST READY, 0
models runnable" was given the second answer without the first.

This module answers both, from the authoritative layers only:

* the **configuration registry** (:func:`batteryml_web.core.config_registry.
  get_config_registry`) says which (dataset, task, model) triples have a run
  config at all — nothing here invents one;
* the **readiness context** (:func:`batteryml_web.agent.prompts.
  build_agent_workflow_context`) says, for one (dataset, task), which models
  are READY with a validator-approved config. Only those are ever offered as
  runnable; a dataset that is RAW_ONLY or MISSING contributes nothing
  runnable, whatever the registry declares for it.

It returns three clearly separate kinds of alternative — same dataset and
task with another model, same task on another dataset, same dataset with
another task — plus the diagnosis and the small set of actions a page may
offer. No LLM is involved, so all of it works without one; an LLM may only
be *told* this result (see :func:`alternatives_prompt_block`).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

#: Diagnosis codes, in the order they are checked.
RUNNABLE = "runnable"
NO_CONFIGURATION = "no_configuration"
DATASET_NOT_READY = "dataset_not_ready"
MODEL_UNAVAILABLE = "model_unavailable"
NO_RUNNABLE_MODEL = "no_runnable_model"

READY = "READY"

Triple = Tuple[str, str, str]
ContextFor = Callable[[str, str], Dict[str, Any]]


def configured_triples(repo_root: str | Path | None = None) -> List[Triple]:
    """Every (dataset, task, model key) that has a run config in the registry."""
    from ..core.config_registry import get_config_registry

    registry = get_config_registry(str(repo_root) if repo_root is not None else None)
    return sorted(
        {
            (str(entry.dataset), str(entry.task), str(entry.model_key))
            for entry in registry.entries
            if entry.dataset and entry.task and entry.model_key
        }
    )


def _runnable(context: Dict[str, Any]) -> List[Dict[str, str]]:
    """READY models of a readiness context, as {model, model_display}."""
    return [
        {
            "model": str(item.get("model")),
            "model_display": str(item.get("model_display") or item.get("model")),
        }
        for item in context.get("runnable_models") or []
        if item.get("model")
    ]


def _status_phrase(status: str) -> str:
    return "ready" if status == READY else f"{status}"


def _alternative_entries(
    pairs: Iterable[Tuple[str, str]],
    configured: Sequence[Triple],
    context_for: ContextFor,
) -> List[Dict[str, Any]]:
    """Readiness for each (dataset, task) pair the registry configures."""
    entries = []
    for dataset, task in pairs:
        context = context_for(dataset, task)
        runnable = _runnable(context)
        entries.append(
            {
                "dataset": dataset,
                "task": task,
                "dataset_status": str(context.get("dataset_status") or "UNKNOWN"),
                "runnable": runnable,
                # Registered here but not runnable now (data not processed,
                # dependency missing, validation failed): named, never offered.
                "configured_not_runnable": sorted(
                    {m for d, t, m in configured if d == dataset and t == task}
                    - {entry["model"] for entry in runnable}
                ),
            }
        )
    return entries


def build_workflow_alternatives(
    workflow_context: Dict[str, Any],
    *,
    model_request: Optional[Dict[str, Any]] = None,
    configured: Optional[Sequence[Triple]] = None,
    context_for: Optional[ContextFor] = None,
    repo_root: str | Path | None = None,
) -> Dict[str, Any]:
    """Diagnose one resolved request and list what the backend can run instead.

    Args:
        workflow_context: :func:`build_agent_workflow_context` for the
            resolved dataset and task (the page's own context).
        model_request: :func:`resolve_model_request` for the model the
            request named, if any.
        configured: the registry's (dataset, task, model) triples; read from
            the registry when omitted (tests inject them).
        context_for: ``(dataset, task) -> readiness context``; the real
            :func:`build_agent_workflow_context` when omitted.
        repo_root: forwarded to the registry and readiness layers.

    Returns a JSON-serialisable mapping::

        {"requested": {"dataset", "task", "model", "model_display"},
         "dataset_status", "diagnosis": {"code", "message"},
         "same_dataset_task": [{model, model_display}],
         "same_task_other_datasets": [entry], "same_dataset_other_tasks": [entry],
         "requested_model_configured_for": [dataset], "actions": [action]}

    where an *entry* is ``{"dataset", "task", "dataset_status", "runnable",
    "configured_not_runnable"}`` and an *action* is ``{"label", "dataset",
    "task", "model", "model_display"}`` — offered only for a combination with at least one
    runnable model, and naming the requested model only when it is itself
    runnable there.
    """
    dataset = str(workflow_context.get("dataset") or "")
    task = str(workflow_context.get("task") or "")
    status = str(workflow_context.get("dataset_status") or "UNKNOWN")
    if configured is None:
        configured = configured_triples(repo_root)
    if context_for is None:
        from .prompts import build_agent_workflow_context

        def context_for(d: str, t: str) -> Dict[str, Any]:
            return build_agent_workflow_context(d, t, repo_root=repo_root)

    request = model_request if isinstance(model_request, dict) else {}
    requested_name = str(request.get("requested") or "")
    requested_key = str(request.get("model_key") or "") if request.get("status") in ("runnable", "blocked") else ""
    requested_display = str(request.get("model_display") or requested_name or "")

    runnable_here = _runnable(workflow_context)
    runnable_keys = {entry["model"] for entry in runnable_here}
    configured_here = {m for d, t, m in configured if d == dataset and t == task}

    result: Dict[str, Any] = {
        "requested": {
            "dataset": dataset,
            "task": task,
            "model": requested_key or None,
            "model_display": requested_display or None,
        },
        "dataset_status": status,
        "same_dataset_task": [e for e in runnable_here if e["model"] != requested_key],
        "same_task_other_datasets": [],
        "same_dataset_other_tasks": [],
        "requested_model_configured_for": sorted(
            {d for d, t, m in configured if t == task and requested_key and m == requested_key}
        ),
        "actions": [],
    }

    satisfied = (requested_key in runnable_keys) if requested_name else bool(runnable_keys)
    if satisfied:
        result["diagnosis"] = {"code": RUNNABLE, "message": ""}
        return result

    other_datasets = sorted({d for d, t, _ in configured if t == task and d != dataset})
    other_tasks = sorted({t for d, t, _ in configured if d == dataset and t != task})
    result["same_task_other_datasets"] = _alternative_entries(
        [(d, task) for d in other_datasets], configured, context_for
    )
    result["same_dataset_other_tasks"] = _alternative_entries(
        [(dataset, t) for t in other_tasks], configured, context_for
    )
    result["diagnosis"] = _diagnose(
        dataset=dataset, task=task, status=status, request=request,
        requested_key=requested_key, requested_display=requested_display,
        runnable_here=runnable_here, configured_here=configured_here,
        workflow_context=workflow_context, alternatives=result,
    )
    result["actions"] = _actions(result, requested_key)
    return result


def _diagnose(
    *,
    dataset: str,
    task: str,
    status: str,
    request: Dict[str, Any],
    requested_key: str,
    requested_display: str,
    runnable_here: List[Dict[str, str]],
    configured_here: set,
    workflow_context: Dict[str, Any],
    alternatives: Dict[str, Any],
) -> Dict[str, str]:
    elsewhere = [e["dataset"] for e in alternatives["same_task_other_datasets"] if e["runnable"]]
    configured_elsewhere = [e["dataset"] for e in alternatives["same_task_other_datasets"]]
    if not configured_here:
        joiner = "but" if status == READY else "and"
        message = (
            f"{dataset} data is {_status_phrase(status)}, {joiner} no registered {task} "
            f"configuration is currently available for {dataset}, so no model can run "
            "this combination."
        )
        if elsewhere:
            message += (
                f" To perform {task}, choose a dataset with a supported {task} "
                f"configuration: {', '.join(elsewhere)}."
            )
        elif configured_elsewhere:
            message += (
                f" {task} configurations are registered for {', '.join(configured_elsewhere)}, "
                "but none of those datasets is ready here; prepare one in Prepare Data first."
            )
        else:
            message += f" No dataset currently has a registered {task} configuration."
        return {"code": NO_CONFIGURATION, "message": message}

    if status != READY:
        detail = (
            "raw data is present but has not been preprocessed"
            if status == "RAW_ONLY"
            else f"its data is {status}"
        )
        return {
            "code": DATASET_NOT_READY,
            "message": (
                f"{task} configurations are registered for {dataset}, but {detail}; "
                "models run on processed data only. Prepare the dataset in Prepare Data "
                "first, then analyze again."
            ),
        }

    if request.get("status") == "unknown":
        name = str(request.get("requested") or "")
        lead = f"'{name}' is not a registered model on this platform."
    elif requested_key and requested_key not in configured_here:
        lead = f"{requested_display} has no registered {task} configuration for {dataset}."
    elif requested_key:
        blocked = next(
            (b for b in workflow_context.get("blocked_models") or [] if str(b.get("model")) == requested_key),
            {},
        )
        reason = "; ".join(str(m) for m in blocked.get("missing") or []) or str(blocked.get("status") or "blocked")
        lead = f"{requested_display} is registered for {dataset} {task} but cannot run now: {reason}."
    else:
        lead = ""

    if runnable_here:
        names = ", ".join(e["model_display"] for e in runnable_here)
        return {
            "code": MODEL_UNAVAILABLE,
            "message": (lead + " " if lead else "")
            + f"{len(runnable_here)} other model(s) can run {task} on {dataset}: {names}.",
        }
    return {
        "code": NO_RUNNABLE_MODEL,
        "message": (lead + " " if lead else "")
        + f"{task} configurations are registered for {dataset}, but none of them passes "
        "readiness right now; see each blocked model's reason.",
    }


def _actions(result: Dict[str, Any], requested_key: str) -> List[Dict[str, Any]]:
    actions: List[Dict[str, Any]] = []
    for entry in result["same_task_other_datasets"]:
        if not entry["runnable"]:
            continue
        keep = next((e for e in entry["runnable"] if e["model"] == requested_key), None)
        label = (
            f"Use {keep['model_display']} on {entry['dataset']} for {entry['task']}"
            if keep
            else f"Use {entry['dataset']} for {entry['task']}"
        )
        actions.append({"label": label, "dataset": entry["dataset"], "task": entry["task"],
                        "model": keep["model"] if keep else None,
                        "model_display": keep["model_display"] if keep else None})
    for entry in result["same_dataset_other_tasks"]:
        if not entry["runnable"]:
            continue
        keep = next((e for e in entry["runnable"] if e["model"] == requested_key), None)
        label = (
            f"Change task to {entry['task']} with {keep['model_display']}"
            if keep
            else f"Change task to {entry['task']}"
        )
        actions.append({"label": label, "dataset": entry["dataset"], "task": entry["task"],
                        "model": keep["model"] if keep else None,
                        "model_display": keep["model_display"] if keep else None})
    return actions


def alternatives_prompt_block(alternatives: Optional[Dict[str, Any]]) -> str:
    """The alternatives, as authoritative text for an explanation prompt.

    Every model named here is one the readiness layer reported READY with a
    validated config; configured-but-not-runnable ones are marked as such.
    """
    if not isinstance(alternatives, dict):
        return "backend_alternatives: (not computed)"
    diagnosis = alternatives.get("diagnosis") or {}

    def models(entries: List[Dict[str, str]]) -> str:
        return ", ".join(f"{e['model_display']} ({e['model']})" for e in entries) or "(none)"

    lines = [
        "backend_alternatives (authoritative; only models listed as runnable can run):",
        f"diagnosis: {diagnosis.get('code')} — {diagnosis.get('message') or 'requested combination is runnable'}",
        f"same dataset and task, other runnable models: {models(alternatives.get('same_dataset_task') or [])}",
    ]
    for label, key in (("same task on other datasets", "same_task_other_datasets"),
                       ("other tasks on the same dataset", "same_dataset_other_tasks")):
        entries = alternatives.get(key) or []
        if not entries:
            lines.append(f"{label}: (none registered)")
        for entry in entries:
            lines.append(
                f"{label}: {entry['dataset']} / {entry['task']} (dataset {entry['dataset_status']}): "
                f"runnable {models(entry['runnable'])}; registered but not runnable now: "
                f"{', '.join(entry['configured_not_runnable']) or '(none)'}"
            )
    return "\n".join(lines)


def alternative_request_text(action: Dict[str, Any], model_display: Optional[str] = None) -> str:
    """The request sentence an action puts in the request box (shown, not hidden)."""
    subject = f"Use {model_display} on {action['dataset']}" if model_display else f"Use {action['dataset']}"
    return f"{subject} for {action['task']}."
