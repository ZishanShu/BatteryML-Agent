"""The persisted record of one run, and the history built from many.

A run's truth lives on disk, not in Streamlit's session state: a browser
refresh, a server restart or a second tab must all see the same runs. Each
run directory carries one ``run.json`` describing what was attempted, what
happened, and where the artifacts are.

Two rules this module exists to enforce.

**Status is recorded, never inferred from a directory.** A directory exists
the moment a run starts; existence says nothing about whether it finished.
So every state is written explicitly, and the one state that cannot be
trusted after a restart — ``RUNNING``, whose process this interpreter no
longer owns — reconciles to ``INTERRUPTED``. It never becomes ``COMPLETED``:
inventing a completion is the worst thing this file could do.

**Writes are atomic.** A manifest is rewritten on every transition, and the
app can be killed mid-write. Writing to a temporary file in the same
directory and then ``os.replace`` makes the swap atomic on POSIX and
Windows, so a reader sees either the old manifest or the new one, never a
truncated one.

Secrets are structurally absent: :func:`build_manifest` takes named
reproducibility fields and copies nothing wholesale, so an API key has no
route into the file.

No Streamlit import, by design — the manager, the verifier and the tests all
call this.
"""

from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

MANIFEST_NAME = "run.json"

STDOUT_LOG = "stdout.log"
STDERR_LOG = "stderr.log"

# --- status model ---------------------------------------------------------

STATUS_PLANNED = "PLANNED"
STATUS_RUNNING = "RUNNING"
STATUS_COMPLETED = "COMPLETED"
STATUS_FAILED = "FAILED"
STATUS_CANCELLED = "CANCELLED"
STATUS_INTERRUPTED = "INTERRUPTED"

ALL_STATUSES: tuple[str, ...] = (
    STATUS_PLANNED,
    STATUS_RUNNING,
    STATUS_COMPLETED,
    STATUS_FAILED,
    STATUS_CANCELLED,
    STATUS_INTERRUPTED,
)

#: States a run can never leave.
TERMINAL_STATUSES: frozenset[str] = frozenset(
    {STATUS_COMPLETED, STATUS_FAILED, STATUS_CANCELLED, STATUS_INTERRUPTED}
)

#: Only these transitions are legal. In particular nothing reaches
#: COMPLETED except from RUNNING, and no terminal state has an exit.
ALLOWED_TRANSITIONS: Dict[str, frozenset[str]] = {
    STATUS_PLANNED: frozenset({STATUS_RUNNING, STATUS_FAILED, STATUS_CANCELLED}),
    STATUS_RUNNING: frozenset(
        {STATUS_COMPLETED, STATUS_FAILED, STATUS_CANCELLED, STATUS_INTERRUPTED}
    ),
    STATUS_COMPLETED: frozenset(),
    STATUS_FAILED: frozenset(),
    STATUS_CANCELLED: frozenset(),
    STATUS_INTERRUPTED: frozenset(),
}


def can_transition(current: str, target: str) -> bool:
    """Whether ``current -> target`` is a legal move."""
    return str(target) in ALLOWED_TRANSITIONS.get(str(current), frozenset())


def utc_now_iso() -> str:
    """Timestamps are UTC and ISO-8601, so runs from any machine sort."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# --- writing --------------------------------------------------------------


def build_manifest(
    *,
    run_id: str,
    project: str,
    workspace: str,
    dataset: Any,
    task: Any,
    model_key: Any,
    model_display: Any = None,
    config_path: Any = None,
    seed: Any = None,
    epochs: Any = None,
    device: Any = None,
    device_selection: Optional[Dict[str, Any]] = None,
    smoke: bool = False,
    command: Optional[Iterable[str]] = None,
    git_commit: Any = None,
    created_at: Optional[str] = None,
) -> Dict[str, Any]:
    """A fresh PLANNED manifest.

    Every field is named explicitly. Nothing is copied from an ambient dict,
    so there is no path by which a credential, a token or an unrelated
    environment value could arrive in this file.
    """
    return {
        "schema": 1,
        "run_id": str(run_id),
        "project": str(project),
        "status": STATUS_PLANNED,
        "dataset": None if dataset is None else str(dataset),
        "task": None if task is None else str(task),
        "model_key": None if model_key is None else str(model_key),
        "model_display": None if model_display is None else str(model_display),
        "config_path": None if config_path is None else str(config_path),
        "config_snapshot": None,
        "seed": None if seed is None else str(seed),
        "epochs": None if epochs is None else str(epochs),
        "device": None if device is None else str(device),
        "device_selection": dict(device_selection or {}),
        "smoke": bool(smoke),
        "workspace": str(workspace),
        "command": [str(part) for part in (command or [])],
        "created_at": created_at or utc_now_iso(),
        "started_at": None,
        "finished_at": None,
        "elapsed_seconds": None,
        "return_code": None,
        "artifacts": {},
        "metrics": {},
        "git_commit": None if git_commit is None else str(git_commit),
        "failure": None,
    }


def manifest_path(run_dir: str | Path) -> Path:
    return Path(run_dir) / MANIFEST_NAME


def write_manifest(run_dir: str | Path, manifest: Dict[str, Any]) -> Path:
    """Write ``manifest`` atomically into ``run_dir``.

    Temp file in the *same directory* (so the replace cannot cross a
    filesystem boundary), flushed and fsync'd before the swap, then
    ``os.replace``. A reader concurrent with this sees one whole manifest or
    the other, never a partial one.
    """
    directory = Path(run_dir)
    directory.mkdir(parents=True, exist_ok=True)
    target = manifest_path(directory)
    payload = json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False)
    handle = tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=str(directory),
        prefix=".run-manifest-",
        suffix=".tmp",
        delete=False,
    )
    try:
        with handle as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(handle.name, target)
    except BaseException:
        # Never leave the scratch file behind on a failed write.
        try:
            os.unlink(handle.name)
        except OSError:
            pass
        raise
    return target


def read_manifest(run_dir: str | Path) -> Optional[Dict[str, Any]]:
    """The manifest in ``run_dir``, or None.

    A file that is missing, unreadable or not an object returns None rather
    than raising: one damaged run must not make a whole project's history
    unopenable. A caller that needs to know the difference can look itself.
    """
    path = manifest_path(run_dir)
    if not path.is_file():
        return None
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return loaded if isinstance(loaded, dict) else None


def update_manifest(
    run_dir: str | Path,
    changes: Dict[str, Any],
    *,
    status: Optional[str] = None,
    enforce_transition: bool = True,
) -> Dict[str, Any]:
    """Apply ``changes`` (and optionally a new ``status``) and rewrite.

    Raises ValueError on an illegal transition, so a bug that tried to call
    a cancelled run completed fails loudly here rather than silently
    rewriting history.
    """
    current = read_manifest(run_dir)
    if current is None:
        raise FileNotFoundError(
            f"No {MANIFEST_NAME} in {run_dir}; a run must be created before it is updated."
        )
    if status is not None:
        present = str(current.get("status") or STATUS_PLANNED)
        if status != present and enforce_transition and not can_transition(present, status):
            raise ValueError(
                f"Illegal run status transition {present} -> {status} for "
                f"{current.get('run_id')}. Allowed from {present}: "
                f"{sorted(ALLOWED_TRANSITIONS.get(present, frozenset())) or 'none (terminal)'}."
            )
        current["status"] = status
    current.update(changes)
    write_manifest(run_dir, current)
    return current


# --- discovery ------------------------------------------------------------


def iter_run_dirs(project_root: str | Path) -> List[Path]:
    """Every directory under ``project_root`` holding a manifest.

    Found by looking for the manifest itself rather than by assuming a
    directory layout, so a run written by an older or newer layout is still
    discovered. Nothing outside ``project_root`` is examined.
    """
    root = Path(project_root)
    if not root.is_dir():
        return []
    return sorted({path.parent for path in root.rglob(MANIFEST_NAME) if path.is_file()})


def reconcile_status(
    manifest: Dict[str, Any],
    *,
    is_live: bool = False,
) -> Dict[str, Any]:
    """Resolve a manifest read from disk against what is actually running.

    A manifest saying ``RUNNING`` is only believable while this process can
    still see the job. After a restart it cannot, so the run is reported
    ``INTERRUPTED`` — an explicit "we do not know that it finished", which
    is the honest answer. It is never upgraded to ``COMPLETED``, and the
    stored file is left alone: reporting is not rewriting.
    """
    resolved = dict(manifest)
    if str(resolved.get("status")) == STATUS_RUNNING and not is_live:
        resolved["status"] = STATUS_INTERRUPTED
        resolved["reconciled_from"] = STATUS_RUNNING
        resolved["reconcile_reason"] = (
            "The application restarted or the job is no longer tracked, so this "
            "run could not be confirmed alive. It is not known to have finished."
        )
    return resolved


def list_runs(
    project_root: str | Path,
    *,
    live_run_ids: Optional[Iterable[str]] = None,
) -> List[Dict[str, Any]]:
    """Every run under ``project_root``, newest first.

    Args:
        project_root: the project directory. Only runs inside it are listed,
            so one project's history can never show another's.
        live_run_ids: ids this process is currently tracking. Anything
            claiming RUNNING that is not in this set is reconciled (see
            :func:`reconcile_status`).

    Each entry carries ``run_dir`` beside the manifest fields. Ordering is by
    ``created_at`` descending, with the run id as the tiebreaker so the order
    is total and stable rather than dependent on directory iteration.
    """
    live = {str(item) for item in (live_run_ids or ())}
    runs: List[Dict[str, Any]] = []
    for directory in iter_run_dirs(project_root):
        manifest = read_manifest(directory)
        if not manifest:
            continue
        entry = reconcile_status(manifest, is_live=str(manifest.get("run_id")) in live)
        entry["run_dir"] = str(directory)
        runs.append(entry)
    runs.sort(
        key=lambda item: (str(item.get("created_at") or ""), str(item.get("run_id") or "")),
        reverse=True,
    )
    return runs


def find_run(
    project_root: str | Path,
    run_id: str,
    *,
    live_run_ids: Optional[Iterable[str]] = None,
) -> Optional[Dict[str, Any]]:
    """One run by id, searched only inside ``project_root``."""
    wanted = str(run_id)
    for entry in list_runs(project_root, live_run_ids=live_run_ids):
        if str(entry.get("run_id")) == wanted:
            return entry
    return None
