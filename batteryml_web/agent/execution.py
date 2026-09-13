"""Deterministic execution planning between agent recommendation and runs.

Smallest backend-only layer wiring the existing pieces together:

1. ``readiness`` / ``build_agent_workflow_context`` is authoritative for
   whether a model is currently runnable.
2. :func:`batteryml_web.core.config_validator.load_and_validate_config_file`
   with ``require_data=True`` is the final deterministic gate, re-run here
   even when readiness says READY.
3. The plan builder emits an argv list for the existing CLI wrapper
   (``batteryml_web/core/run_batteryml_cli.py run <config> --train --eval
   --device ... --seed ... --workspace ...``). It never calls subprocess.

Model selection never parses LLM free text: callers pass an explicit model
key (or ``plan_for_request`` matches a model mention deterministically
against the context's own entries plus the registry's existing alias
tables). Anything blocked — NEEDS_ASSETS, NEEDS_DATA, CONFIG_INVALID,
unknown — yields ``can_execute=False`` with no fabricated config and no
command.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .intent import resolve_workflow_intent
from .prompts import build_agent_workflow_context

WRAPPER = "batteryml_web/core/run_batteryml_cli.py"


def _registry_tables() -> Tuple[Dict[str, str], Dict[str, str], Any]:
    from ..core.config_registry import (
        MODEL_CLASS_KEYS,
        MODEL_KEY_ALIASES,
        normalize_model_token,
    )

    normalized_aliases = {
        normalize_model_token(key): alias for key, alias in MODEL_KEY_ALIASES.items()
    }
    return MODEL_CLASS_KEYS, normalized_aliases, normalize_model_token


def _canonical_model_key(token: str) -> str:
    """Fold a requested model token onto a canonical key using existing aliases."""
    _, normalized_aliases, normalize_model_token = _registry_tables()
    from ..core.config_registry import MODEL_CLASS_KEYS

    normalized = normalize_model_token(token)
    if normalized in normalized_aliases:
        return normalized_aliases[normalized]
    squashed = normalized.replace(" ", "")
    if squashed in MODEL_CLASS_KEYS:
        return MODEL_CLASS_KEYS[squashed]
    return normalized.replace(" ", "_")


def _aliases_for_entry(
    model_key: str, model_display: Optional[str]
) -> List[str]:
    """All existing names that may refer to one context entry."""
    from ..core.config_registry import MODEL_CLASS_KEYS, MODEL_KEY_ALIASES

    aliases = {model_key, model_key.replace("_", " ")}
    for alias, target in MODEL_KEY_ALIASES.items():
        if target == model_key:
            aliases.add(alias)
    for class_name, target in MODEL_CLASS_KEYS.items():
        if target == model_key:
            aliases.add(class_name)
    if model_display:
        aliases.add(model_display)
    return sorted(aliases, key=len, reverse=True)


def find_model_entry(
    workflow_context: Dict[str, Any], requested: str
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Find the exact context entry for ``requested``.

    Matches the canonical key, any registry alias for it, or the entry's
    display name — all of which already exist in the registry or the
    context. Never fuzzy-matches into an unrelated model: no entry means
    ``(None, None)``.

    Returns ``(entry, side)`` with ``side`` in ``{"runnable", "blocked"}``.
    """
    if not str(requested or "").strip():
        raise ValueError("find_model_entry: requested model is empty.")
    canonical = _canonical_model_key(str(requested))
    runnable = {str(e.get("model")): e for e in workflow_context.get("runnable_models", [])}
    blocked = {str(e.get("model")): e for e in workflow_context.get("blocked_models", [])}
    if canonical in runnable:
        return runnable[canonical], "runnable"
    if canonical in blocked:
        return blocked[canonical], "blocked"
    for side, entries in (("runnable", runnable), ("blocked", blocked)):
        for key, entry in entries.items():
            for alias in _aliases_for_entry(key, entry.get("model_display")):
                if _canonical_model_key(alias) == canonical:
                    return entry, side
    lowered = str(requested).strip().lower()
    for side, entries in (("runnable", runnable), ("blocked", blocked)):
        for key, entry in entries.items():
            display = str(entry.get("model_display") or "").lower()
            if lowered and (lowered == key.lower() or (display and lowered == display)):
                return entry, side
    return None, None


def resolve_model_mention(
    text: str, workflow_context: Dict[str, Any]
) -> Optional[str]:
    """Match a model mention in free text against this context's own entries.

    Only names drawn from the context entries (plus their registry aliases)
    can match, longest alias first on word boundaries, so "rf" cannot match
    inside an unrelated word and an unknown name yields None instead of a
    neighboring model.
    """
    lowered = str(text or "").lower()
    candidates: List[Tuple[int, str]] = []
    for side in ("runnable_models", "blocked_models"):
        for entry in workflow_context.get(side, []):
            key = str(entry.get("model"))
            for alias in _aliases_for_entry(key, entry.get("model_display")):
                pattern = r"\b" + re.escape(alias.lower()) + r"\b"
                if re.search(pattern, lowered):
                    candidates.append((len(alias), key))
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0][1]


def _default_workspace(dataset: str, model: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", f"agent_{dataset}_{model}".lower()).strip("_")
    return f"runs/{safe or 'agent_run'}"


def _validate_epochs(epochs: int | None) -> Optional[int]:
    """Validate the planner-owned epoch override.

    Only ``None`` (preserve the config's epoch schedule) or a positive
    ``int`` is accepted. ``bool`` is rejected explicitly because it is an
    ``int`` subclass in Python and ``True`` would otherwise mean 1 epoch.
    No other CLI flag overrides exist; this is the single allowed one.
    """
    if epochs is None:
        return None
    if isinstance(epochs, bool) or not isinstance(epochs, int):
        raise ValueError(
            f"epochs must be a positive int or None, got {epochs!r}."
        )
    if epochs <= 0:
        raise ValueError(f"epochs must be a positive int or None, got {epochs!r}.")
    return epochs


def build_execution_plan(
    dataset: str,
    task: str,
    model: str,
    *,
    repo_root: str | Path | None = None,
    device: str = "cpu",
    seed: int = 0,
    workspace: str | None = None,
    workflow_context: Optional[Dict[str, Any]] = None,
    epochs: int | None = None,
) -> Dict[str, Any]:
    """Build a validated, non-executing run plan for one model.

    Args:
        dataset / task / model: explicit backend-selected identifiers. The
            model is matched via :func:`find_model_entry`; free-text LLM
            output must never be passed here as authority.
        repo_root: forwarded to the readiness layer when the context is
            built inside this call.
        device / seed / workspace: CLI run parameters (informational only).
        workflow_context: optional prebuilt compact context (test seam for
            mocked READY state). When omitted it is built from
            ``(dataset, task)``; only its ``validated_config`` values may
            ever appear in the plan.
        epochs: optional positive-int ``--epochs`` override owned by the
            planner (``bin/batteryml.py run`` accepts ``--epochs`` as a
            number-of-epochs override). ``None`` preserves the config's own
            schedule exactly. This is the only CLI flag override permitted;
            no arbitrary extra flags are accepted.

    Returns a dict with ``dataset``, ``task``, ``model``, ``model_status``,
    ``dataset_status``, ``config_path`` (verbatim readiness value or None),
    ``config_valid``, ``can_execute``, ``validation_errors``, ``command``
    (argv list or None), ``epochs`` and ``reason``. No subprocess is ever
    started.
    """
    if not str(dataset or "").strip():
        raise ValueError("build_execution_plan: dataset is empty.")
    if not str(task or "").strip():
        raise ValueError("build_execution_plan: task is empty.")
    if not str(model or "").strip():
        raise ValueError("build_execution_plan: model is empty.")
    epochs_value = _validate_epochs(epochs)

    from ..core.config_validator import load_and_validate_config_file

    context = (
        workflow_context
        if workflow_context is not None
        else build_agent_workflow_context(dataset, task, repo_root=repo_root)
    )
    dataset_status = str(context.get("dataset_status") or "UNKNOWN")
    resolved_task = str(context.get("task") or task)

    entry, side = find_model_entry(context, model)
    if entry is None:
        return {
            "dataset": dataset,
            "task": resolved_task,
            "model": str(model),
            "model_display": None,
            "model_status": "UNKNOWN",
            "dataset_status": dataset_status,
            "config_path": None,
            "config_valid": False,
            "can_execute": False,
            "validation_errors": [f"Model {model!r} is not known for this dataset/task."],
            "command": None,
            "reason": (
                f"Model {model!r} matches no runnable or blocked entry, so "
                "nothing may be executed."
            ),
        }

    model_key = str(entry.get("model"))
    model_status = str(entry.get("status") or "UNKNOWN")
    if side == "blocked" or model_status != "READY":
        missing = list(entry.get("missing") or [])
        return {
            "dataset": dataset,
            "task": resolved_task,
            "model": model_key,
            "model_display": entry.get("model_display"),
            "model_status": model_status,
            "dataset_status": dataset_status,
            "config_path": None,
            "config_valid": False,
            "can_execute": False,
            "validation_errors": [],
            "command": None,
            "reason": (
                f"Model {model_key!r} is {model_status} and must not execute: "
                f"{entry.get('recommended_action') or '; '.join(missing) or 'see blocked_models'}"
            ),
        }

    validated = entry.get("validated_config")
    if not validated:
        return {
            "dataset": dataset,
            "task": resolved_task,
            "model": model_key,
            "model_display": entry.get("model_display"),
            "model_status": model_status,
            "dataset_status": dataset_status,
            "config_path": None,
            "config_valid": False,
            "can_execute": False,
            "validation_errors": ["Runnable entry carries no validated_config."],
            "command": None,
            "reason": f"Model {model_key!r} is READY but supplies no validated config.",
        }

    config_path = str(validated)
    report = load_and_validate_config_file(config_path, require_data=True)
    errors = [str(e) for e in report.get("errors", [])]
    config_valid = bool(report.get("valid"))
    runnable_data = dataset_status == "READY"
    can_execute = bool(config_valid and runnable_data and model_status == "READY")
    if can_execute:
        ws = workspace or _default_workspace(dataset, model_key)
        command: Optional[List[str]] = [
            sys.executable,
            WRAPPER,
            "run",
            config_path,
            "--workspace",
            ws,
            "--train",
            "--eval",
            "--device",
            str(device),
            "--seed",
            str(seed),
        ]
        if epochs_value is not None:
            command.extend(["--epochs", str(epochs_value)])
        reason = f"Model {model_key!r} is READY with a validated config; plan only, nothing executed."
    else:
        command = None
        if not config_valid:
            reason = f"Validator rejected {config_path}; execution is refused."
        elif not runnable_data:
            reason = (
                f"Dataset {dataset!r} is {dataset_status}, so no run may start "
                "even though the model entry is READY."
            )
        else:
            reason = f"Model {model_key!r} cannot execute; see validation_errors."
    return {
        "dataset": dataset,
        "task": resolved_task,
        "model": model_key,
        "model_display": entry.get("model_display"),
        "model_status": model_status,
        "dataset_status": dataset_status,
        "config_path": config_path,
        "config_valid": config_valid,
        "can_execute": can_execute,
        "validation_errors": errors,
        "command": command,
        "epochs": epochs_value,
        "reason": reason,
    }


def plan_for_request(
    user_request: str,
    *,
    repo_root: str | Path | None = None,
    device: str = "cpu",
    seed: int = 0,
    workspace: str | None = None,
    epochs: int | None = None,
) -> Dict[str, Any]:
    """End-to-end deterministic plan for one natural-language request.

    Resolves dataset + task via intent, builds the compact context, matches
    a model mention against that context only, and returns
    :func:`build_execution_plan`. When intent or the model mention cannot be
    resolved, or the dataset is not READY, the result has
    ``can_execute=False`` and states what is needed — the LLM is not
    consulted and no fuzzy fallback model is picked. ``epochs`` is the
    planner-owned positive-int ``--epochs`` override (None preserves the
    config schedule); it is validated by :func:`build_execution_plan`.
    """
    if not str(user_request or "").strip():
        raise ValueError("plan_for_request: user_request is empty.")
    intent = resolve_workflow_intent(user_request)
    dataset = intent.get("dataset")
    task = intent.get("task")
    if dataset is None or task is None:
        return {
            "dataset": dataset,
            "task": task,
            "model": None,
            "model_display": None,
            "model_status": None,
            "dataset_status": None,
            "config_path": None,
            "config_valid": False,
            "can_execute": False,
            "validation_errors": [],
            "command": None,
            "reason": intent["message"],
        }
    context = build_agent_workflow_context(dataset, task, repo_root=repo_root)
    mentioned = resolve_model_mention(user_request, context)
    if mentioned is None:
        runnable = [str(e.get("model")) for e in context.get("runnable_models", [])]
        return {
            "dataset": dataset,
            "task": task,
            "model": None,
            "model_display": None,
            "model_status": None,
            "dataset_status": context.get("dataset_status"),
            "config_path": None,
            "config_valid": False,
            "can_execute": False,
            "validation_errors": [],
            "command": None,
            "reason": (
                f"Dataset={dataset}, task={task} resolved, but no known model "
                f"is named in the request. Runnable now: "
                f"{', '.join(runnable) or '(none)'}."
            ),
        }
    plan = build_execution_plan(
        dataset,
        task,
        mentioned,
        repo_root=repo_root,
        device=device,
        seed=seed,
        workspace=workspace,
        workflow_context=context,
        epochs=epochs,
    )
    plan["reason"] = f"Request resolved to dataset={dataset}, task={task}. " + str(plan.get("reason"))
    return plan
