"""Report every run's recorded state, and flag any that never recorded one.

For when a user says a workflow looks stuck. It answers one question from
the durable record: **did this run ever write an outcome?** A manifest still
saying ``PLANNED`` or ``RUNNING`` has not, and that is the shape of a stuck
run.

What this tool deliberately does not claim: whether a process is alive.
Liveness is knowable only inside the server process that started the run —
that is what ``RunManager.live_run_ids()`` is, and it is why
``run_manifest.reconcile_status`` takes it as an argument. The manifest
records no pid, so a pid check here would be invented. Run this alongside
the server: a run listed below while the server that started it is still up
may simply be in flight; the same run after that server has stopped is one
that will never report.

It reads only. A run this tool cannot see the end of is reported, never
relabelled — writing an outcome for it would be inventing one.

Usage::

    python -m scripts.inspect_run_state runs/web_projects/<project>
    python -m scripts.inspect_run_state runs/web_projects/<project> --no-outcome-only
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from batteryml_web.core import run_manifest as rm  # noqa: E402

#: Statuses that mean "this run has not reported an outcome".
NO_OUTCOME = (rm.STATUS_PLANNED, rm.STATUS_RUNNING)


def collect(project_root: Path) -> List[Dict[str, Any]]:
    """One row per run, newest first, from the manifests alone.

    ``live_run_ids=[]`` because this process started none of these runs, so
    claiming any of them live would be false. The *recorded* status is read
    from the manifest rather than from the reconciled entry, so the row
    shows what is actually on disk.
    """
    rows: List[Dict[str, Any]] = []
    for entry in rm.list_runs(project_root, live_run_ids=[]):
        manifest = rm.read_manifest(Path(entry["run_dir"])) or {}
        recorded = str(manifest.get("status") or "")
        rows.append(
            {
                "run_id": str(entry.get("run_id") or ""),
                "run_dir": str(entry.get("run_dir") or ""),
                "recorded_status": recorded,
                "return_code": manifest.get("return_code"),
                "started_at": manifest.get("started_at"),
                "finished_at": manifest.get("finished_at"),
                "no_outcome": recorded in NO_OUTCOME,
            }
        )
    return rows


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("project_root", help="runs/web_projects/<project>")
    parser.add_argument(
        "--no-outcome-only",
        action="store_true",
        help="print only runs whose manifest never reached a terminal status",
    )
    args = parser.parse_args(argv)

    root = Path(args.project_root)
    if not root.is_dir():
        print(f"No such project directory: {root}", file=sys.stderr)
        return 2

    rows = collect(root)
    if args.no_outcome_only:
        rows = [row for row in rows if row["no_outcome"]]
    if not rows:
        print(
            "Every run recorded an outcome."
            if args.no_outcome_only
            else f"No runs found under {root}."
        )
        return 0

    print(f"{'run_id':<46} {'recorded':<11} {'rc':>4}  finished_at")
    for row in rows:
        code = row["return_code"]
        marker = "   <-- no outcome recorded" if row["no_outcome"] else ""
        print(
            f"{row['run_id']:<46} {row['recorded_status']:<11} "
            f"{str(code if code is not None else '-'):>4}  "
            f"{row['finished_at'] or '-'}{marker}"
        )

    pending = [row for row in rows if row["no_outcome"]]
    if pending:
        print(
            f"\n{len(pending)} run(s) never recorded an outcome. If the server "
            "that started them is no longer running, the page reports these as "
            "INTERRUPTED — 'not known to have finished' — rather than as still "
            "going, and their run directories keep whatever output was written."
        )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
