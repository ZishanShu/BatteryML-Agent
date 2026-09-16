"""Which project a run belongs to, and therefore where it is written.

The bug this module exists to prevent: the page showed one project while the
run was written under another. The old workspace helper silently fell back to
``runs/web_projects/default`` whenever the session's project root was absent
or did not sit under ``runs/`` — so a project whose base directory the user
had typed as an absolute path displayed as "alpha" and executed as "default",
with no message either way.

Here the resolution is explicit and total: a project either resolves to a
workspace or the call fails naming the reason. There is no answer that is
silently different from what the page is showing.

Design constraints (CLAUDE.md §3, §6): no Streamlit import, no filesystem
writes, no hidden state. Everything is a pure function of its arguments, so
"project alpha writes under runs/web_projects/alpha" is a unit test rather
than something only a running server can tell you.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, Optional

#: Every project lives under this directory, relative to the repository root.
PROJECTS_ROOT = "runs/web_projects"

#: The project used when the user has not created or loaded one. It is a real,
#: named project — not a hidden fallback — so the page can say which project a
#: run will be written to before the user presses anything.
DEFAULT_PROJECT_NAME = "default"

#: Characters kept in a path segment built from a user-supplied name.
_SAFE_SEGMENT = re.compile(r"[^A-Za-z0-9._-]+")


def safe_segment(value: str) -> str:
    """One path segment from arbitrary text, or '' when nothing survives.

    Never returns ``.`` or ``..``: a name that reduces to either would let a
    workspace escape its project directory.
    """
    cleaned = _SAFE_SEGMENT.sub("_", str(value or "")).strip("._-")
    return "" if cleaned in {"", ".", ".."} else cleaned


def project_root_for(name: str) -> str:
    """The repo-relative root of the project called ``name``.

    Raises:
        ValueError: the name has no usable characters. Guessing a directory
            for an unnameable project is how a run ends up somewhere the user
            was not shown.
    """
    segment = safe_segment(name)
    if not segment:
        raise ValueError(
            f"project name {name!r} contains no usable characters; a project "
            "name must contain letters, digits, '.', '_' or '-'."
        )
    return f"{PROJECTS_ROOT}/{segment}"


def resolve_project(
    *,
    project_root: Optional[str] = None,
    project_name: Optional[str] = None,
) -> Dict[str, Any]:
    """The project a run belongs to, from whatever the session holds.

    Args:
        project_root: the session's stored project directory, if any. May be
            absolute (the user may have typed an absolute base directory).
        project_name: the session's stored project name, if any.

    Returns ``{"name", "root", "is_default", "source"}``. ``root`` is always
    repo-relative and always under :data:`PROJECTS_ROOT`; ``source`` says
    which input decided it (``"root"``, ``"name"`` or ``"default"``), so a
    caller can tell a real choice from the default.

    A project root that points outside the projects area is **not** silently
    replaced: its final segment is taken as the project name and the project
    resolves under the projects area, which is where every workspace must
    live for the executor's containment rule to hold. The ``source`` field
    still reports ``"root"``, and the returned ``root`` is what the page must
    display — never the rejected input.
    """
    raw_root = str(project_root or "").strip()
    if raw_root:
        name = Path(raw_root.rstrip("/")).name
        segment = safe_segment(name)
        if segment:
            return {
                "name": segment,
                "root": f"{PROJECTS_ROOT}/{segment}",
                "is_default": segment == DEFAULT_PROJECT_NAME,
                "source": "root",
            }

    raw_name = str(project_name or "").strip()
    if raw_name:
        segment = safe_segment(raw_name)
        if segment:
            return {
                "name": segment,
                "root": f"{PROJECTS_ROOT}/{segment}",
                "is_default": segment == DEFAULT_PROJECT_NAME,
                "source": "name",
            }

    return {
        "name": DEFAULT_PROJECT_NAME,
        "root": f"{PROJECTS_ROOT}/{DEFAULT_PROJECT_NAME}",
        "is_default": True,
        "source": "default",
    }


def agent_workspace(
    project: Dict[str, Any],
    dataset: str,
    model: str,
    *,
    smoke: bool = False,
) -> str:
    """The run workspace for one (project, dataset, model) triple.

    ``runs/web_projects/<project>/agent_<dataset>_<model>[_smoke]``.

    Raises:
        ValueError: the project is malformed, or the dataset or model reduces
            to nothing. A workspace built from an empty component would
            collide with another run's directory.
    """
    root = str((project or {}).get("root") or "").strip()
    if not root.startswith(PROJECTS_ROOT + "/"):
        raise ValueError(
            f"project root {root!r} is not under {PROJECTS_ROOT}/; resolve the "
            "project with resolve_project() before building a workspace."
        )
    safe_dataset = safe_segment(dataset).lower()
    safe_model = safe_segment(model).lower()
    if not safe_dataset:
        raise ValueError(f"dataset {dataset!r} yields no usable path segment.")
    if not safe_model:
        raise ValueError(f"model {model!r} yields no usable path segment.")
    suffix = "_smoke" if smoke else ""
    return f"{root}/agent_{safe_dataset}_{safe_model}{suffix}"


def display_path(path: Any, *, repo_root: str | Path | None = None) -> str:
    """A path as a reader wants to see it: repo-relative where it can be.

    ``/root/share/szs/BatteryML/runs/web_projects/1/...`` becomes
    ``runs/web_projects/1/...``. A path outside the repository is returned
    unchanged — shortening it would be a lie about where the file is. The
    absolute form stays available to the caller; this only chooses what to
    show.
    """
    text = str(path or "").strip()
    if not text:
        return ""
    root = Path(repo_root) if repo_root is not None else _repo_root()
    try:
        return Path(text).resolve().relative_to(root.resolve()).as_posix()
    except (ValueError, OSError):
        return text


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]
