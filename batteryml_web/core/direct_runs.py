"""Run records for runs started from the Run Model page or Home's MATR demo.

Review Results lists a project's runs from the ``run.json`` files under it
(:func:`batteryml_web.core.run_manifest.list_runs`). Only Agent Workflow used
to write one, so a run started from **Run Model → Run BatteryML** or from
**Home → Run MATR Demo** never appeared there, even though Home sends the user
straight to Review Results after the demo. Both pages also wrote every run
into the same ``<project>/workspace`` folder, so a record there would have
been overwritten by the next run anyway.

This module gives those runs what Agent Workflow runs have:

* their own folder, ``runs/web_projects/<project>/<source>/<run_id>/``, which
  is also the ``--workspace`` BatteryML writes into, so one run's predictions
  can never be mixed with another's;
* a ``run.json`` in the same schema (:func:`run_manifest.build_manifest`),
  written PLANNED → RUNNING *before* BatteryML starts, so a crash or a
  closed browser tab leaves a record that history reports as INTERRUPTED
  rather than a gap;
* COMPLETED or FAILED afterwards, with the return code, duration, the real
  artifact list and only the metrics BatteryML saved (the caller reads them
  with the same functions Agent Workflow uses). A failed run records no
  metrics.

No Streamlit import and no BatteryML import: the page passes everything in,
so this file can be tested on a temporary folder.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, MutableMapping, Optional

from . import run_manifest as rm
from .run_identity import is_run_id, new_run_id

#: Which page started the run → the project sub-folder its runs live in.
SOURCE_DIRS: Dict[str, str] = {
    "run_model": "run_model",
    "quickstart": "quickstart",
}

#: The run's combined stdout and stderr, as the page captured it.
OUTPUT_LOG = "output.log"


def direct_run_dir(project_root: str | Path, source: str, run_id: str) -> Path:
    """``<project_root>/<source folder>/<run_id>``.

    Raises:
        ValueError: ``source`` is not a known page, or ``run_id`` is not an id
            minted by :func:`run_identity.new_run_id` (anything else could
            contain a path separator or collide with another run).
    """
    if source not in SOURCE_DIRS:
        raise ValueError(
            f"Unknown run source {source!r}; expected one of {sorted(SOURCE_DIRS)}."
        )
    if not is_run_id(run_id):
        raise ValueError(
            f"{run_id!r} is not a run id minted by run_identity.new_run_id(); "
            "refusing to build a run folder from it."
        )
    return Path(project_root) / SOURCE_DIRS[source] / run_id


def reserve_run_id(
    session: MutableMapping[str, Any],
    *,
    key: str,
    dataset: Any,
    model: Any,
    seed: Any,
    device: Any,
    config_path: Any,
) -> str:
    """The run id the page shows in its command before Run is pressed.

    The Run Model page displays the exact command — including ``--workspace``
    — before the click, so the id has to exist by then. It is kept while the
    selection stays the same (a rerun must not change the command on screen)
    and replaced as soon as dataset, model, seed, device or config changes,
    so an id never names a different configuration than the one it ran.
    Call :func:`release_run_id` once the run has been recorded, so the next
    run gets a new id.
    """
    subject = [str(part) for part in (dataset, model, seed, device, config_path)]
    stored = session.get(key)
    if (
        isinstance(stored, dict)
        and stored.get("subject") == subject
        and is_run_id(stored.get("run_id"))
    ):
        return str(stored["run_id"])
    run_id = new_run_id(dataset=dataset, model=model, seed=seed)
    session[key] = {"subject": subject, "run_id": run_id}
    return run_id


def release_run_id(session: MutableMapping[str, Any], *, key: str) -> None:
    """Forget the reserved id, so the next run is a new run."""
    session.pop(key, None)


def begin_direct_run(
    run_dir: str | Path,
    *,
    workspace: str,
    run_id: str,
    project: str,
    dataset: Any,
    task: Any,
    model_key: Any,
    model_display: Any,
    config_path: Any,
    seed: Any,
    device: Any,
    command: Iterable[str],
    git_commit: Optional[str],
) -> Dict[str, Any]:
    """Write the run's ``run.json`` and mark it RUNNING. Call before starting.

    ``run_dir`` is where the file is written (the page passes it anchored at
    the repository root); ``workspace`` is the repo-relative form recorded in
    the manifest and passed to BatteryML as ``--workspace``, exactly as an
    Agent Workflow run records it.

    Raises:
        FileExistsError: ``run_dir`` already holds a run record. A recorded
            run is never overwritten; mint a new id instead.
    """
    directory = Path(run_dir)
    if rm.manifest_path(directory).exists():
        raise FileExistsError(
            f"{rm.manifest_path(directory)} already exists; run {run_id} was "
            "already recorded. A new run needs a new run id."
        )
    manifest = rm.build_manifest(
        run_id=run_id,
        project=project,
        workspace=str(workspace),
        dataset=dataset,
        task=task,
        model_key=model_key,
        model_display=model_display,
        config_path=config_path,
        seed=seed,
        epochs=None,
        device=device,
        smoke=False,
        command=command,
        git_commit=git_commit,
    )
    rm.write_manifest(directory, manifest)
    return rm.update_manifest(
        directory, {"started_at": rm.utc_now_iso()}, status=rm.STATUS_RUNNING
    )


def finish_direct_run(
    run_dir: str | Path,
    *,
    return_code: Optional[int],
    elapsed_seconds: float,
    output: str,
    artifacts: Optional[Dict[str, Any]] = None,
    metrics: Optional[Dict[str, Any]] = None,
    metrics_error: Optional[str] = None,
) -> Dict[str, Any]:
    """Record how the run ended: COMPLETED for return code 0, else FAILED.

    Args:
        output: the combined stdout/stderr the page captured; written to
            :data:`OUTPUT_LOG` in the run folder.
        artifacts: from ``agent.executor.discover_artifacts`` — real files
            only. Ignored for a failed run.
        metrics: only what BatteryML saved, read by the caller. Ignored for a
            failed run, which has measured nothing.
        metrics_error: why reading the metrics failed, if it did. Recorded so
            a completed run without metrics says why instead of looking as
            if BatteryML had produced none.
    """
    directory = Path(run_dir)
    (directory / OUTPUT_LOG).write_text(str(output or ""), encoding="utf-8")
    succeeded = return_code == 0
    changes: Dict[str, Any] = {
        "finished_at": rm.utc_now_iso(),
        "elapsed_seconds": round(float(elapsed_seconds), 3),
        "return_code": return_code,
        "artifacts": dict(artifacts or {}) if succeeded else {},
        "metrics": dict(metrics or {}) if succeeded else {},
    }
    if succeeded and metrics_error:
        changes["metrics_error"] = str(metrics_error)
    if not succeeded:
        changes["failure"] = {
            "category": "execution_failed",
            "message": (
                f"The run exited with code {return_code}. Its output is in "
                f"{(directory / OUTPUT_LOG).as_posix()}."
            ),
        }
    return rm.update_manifest(
        directory,
        changes,
        status=rm.STATUS_COMPLETED if succeeded else rm.STATUS_FAILED,
    )
