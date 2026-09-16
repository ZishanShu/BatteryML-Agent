"""The run's identity: which project, where it writes, and with what.

Two questions the page must be able to answer at any moment, and must answer
the *same* way before and after a run:

* where will this be written — and where *was* it written;
* what exactly was run — dataset, task, model, config, seed, epochs, device.

Both are assembled here, from the validated execution plan and the resolved
project, so the fields the page shows before Run are the same objects it
shows afterwards. A provenance block that is rebuilt from different sources
on either side of an execution is how a page ends up claiming project A while
the artifacts sit under project B.

Every value is copied from its owner: the plan (built by the planner and
checked by the validator), the resolved project, and the device selection.
Nothing here derives, defaults or prettifies a value into existence — an
absent field is reported as absent.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from ..core.project_context import display_path

#: Field order, as the page shows it. The first two answer "where", the rest
#: answer "what".
FIELD_ORDER: tuple[str, ...] = (
    "project",
    "workspace",
    "dataset",
    "task",
    "model",
    "config",
    "seed",
    "epochs",
    "device",
)

#: Shown for a field the backend did not supply. Never a guess.
UNKNOWN = "Not available"


def build_run_context(
    *,
    project: Optional[Dict[str, Any]],
    plan: Optional[Dict[str, Any]],
    device_label: str = "",
    seed: Any = None,
    smoke: bool = False,
) -> Dict[str, Any]:
    """The provenance block for one planned or completed run.

    Args:
        project: the resolved project (:func:`resolve_project`).
        plan: the validated execution plan, or None before one exists.
        device_label: how the device was chosen, in the user's terms
            ("CPU", "GPU 3", "GPUs 0, 1"). The plan's own ``--device`` value
            is the executable form; this is the readable one.
        seed: the seed the run uses.
        smoke: whether the 1-epoch smoke override is on.

    Returns a dict keyed by :data:`FIELD_ORDER` plus ``workspace_display``
    (repo-relative where possible) and ``smoke``. Values are strings or None;
    None means the backend had nothing, and the page shows
    :data:`UNKNOWN` rather than inventing a value.
    """
    plan = plan or {}
    project = project or {}
    workspace = _workspace_from_plan(plan)
    epochs = plan.get("epochs")
    return {
        "project": project.get("name"),
        "project_is_default": bool(project.get("is_default")),
        "workspace": workspace,
        "workspace_display": display_path(workspace) if workspace else None,
        "dataset": plan.get("dataset"),
        "task": plan.get("task"),
        "model": plan.get("model_display") or plan.get("model"),
        "model_key": plan.get("model"),
        "config": plan.get("config_path"),
        "config_display": display_path(plan.get("config_path"))
        if plan.get("config_path")
        else None,
        "seed": None if seed is None else str(seed),
        "epochs": str(epochs) if epochs is not None else "Config default",
        "device": device_label or None,
        "smoke": bool(smoke),
    }


def _workspace_from_plan(plan: Dict[str, Any]) -> Optional[str]:
    """The ``--workspace`` value inside the plan's own argv.

    Read from the command rather than from a separate field, because the
    command is what actually executes. If the two ever disagreed, this
    reports what will really happen.
    """
    command = plan.get("command")
    if not isinstance(command, (list, tuple)):
        return None
    parts = [str(part) for part in command]
    try:
        index = parts.index("--workspace")
    except ValueError:
        return None
    return parts[index + 1] if index + 1 < len(parts) else None


def context_rows(
    run_context: Optional[Dict[str, Any]],
    *,
    prefer_display_paths: bool = True,
) -> List[tuple[str, str]]:
    """``(label, value)`` pairs in :data:`FIELD_ORDER`, ready to render.

    Paths are shown repo-relative by default; the absolute form stays in the
    context for a details view. A missing value renders as :data:`UNKNOWN`,
    never as an empty cell that reads like "nothing to see here".
    """
    if not isinstance(run_context, dict):
        return []
    labels = {
        "project": "Project",
        "workspace": "Run directory",
        "dataset": "Dataset",
        "task": "Task",
        "model": "Model",
        "config": "Config",
        "seed": "Seed",
        "epochs": "Epochs",
        "device": "Device",
    }
    rows: List[tuple[str, str]] = []
    for field in FIELD_ORDER:
        value = run_context.get(field)
        if prefer_display_paths and field in ("workspace", "config"):
            value = run_context.get(f"{field}_display") or value
        rows.append((labels[field], UNKNOWN if value in (None, "") else str(value)))
    return rows


def artifact_display(
    result_context: Optional[Dict[str, Any]],
) -> Dict[str, Optional[str]]:
    """The predictions artifact, in both the readable and the exact form.

    ``display`` is repo-relative where the file is inside the repository;
    ``absolute`` is what the backend actually recorded and is what a details
    view must show. The file's location is not changed by either.
    """
    if not isinstance(result_context, dict):
        return {"display": None, "absolute": None}
    absolute = result_context.get("predictions_file")
    if not absolute:
        return {"display": None, "absolute": None}
    return {"display": display_path(absolute), "absolute": str(absolute)}


def matches_run(
    run_context: Optional[Dict[str, Any]],
    result_context: Optional[Dict[str, Any]],
) -> bool:
    """Whether a result really belongs to the run the page is showing.

    Compares the workspace, which is the one field that ties a result to a
    place on disk. Used to refuse to display a result under provenance it
    does not belong to.
    """
    if not isinstance(run_context, dict) or not isinstance(result_context, dict):
        return False
    planned = run_context.get("workspace")
    actual = result_context.get("workspace")
    if not planned or not actual:
        return False
    return str(planned).rstrip("/") == str(actual).rstrip("/")
