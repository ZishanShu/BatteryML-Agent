"""Project-scoped run history, rendered from the manifests on disk.

Everything shown here is read back from ``run.json`` files under the current
project. Nothing is remembered in session state, so a browser refresh, a
second tab and a restarted server all show the same list — and a run that
this process never started is still visible, because the filesystem, not the
session, is the record.

The row builder is a pure function so it can be tested without Streamlit,
and so the "no hardcoded model names or metrics" rule is checkable: every
value in a row comes from the manifest it was built from.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import streamlit as st

from ..core import run_manifest as rm
from ..core.run_manager import get_run_manager
from ..ui.formatting import compact_metric

HISTORY_TITLE = "Run history"

#: Column order for the history table.
COLUMNS: tuple[str, ...] = (
    "Run ID",
    "Started",
    "Dataset",
    "Task",
    "Model",
    "Seed",
    "Epochs",
    "Device",
    "Status",
    "Duration",
    "Metrics",
)

NOT_AVAILABLE = "—"


def format_duration(seconds: Any) -> str:
    """``mm:ss`` (or ``h:mm:ss``) from a number of seconds, else a dash."""
    try:
        total = int(round(float(seconds)))
    except (TypeError, ValueError):
        return NOT_AVAILABLE
    if total < 0:
        return NOT_AVAILABLE
    hours, rest = divmod(total, 3600)
    minutes, secs = divmod(rest, 60)
    return f"{hours}:{minutes:02d}:{secs:02d}" if hours else f"{minutes:02d}:{secs:02d}"


def format_metrics(metrics: Any) -> str:
    """A short metric reading for the row, from the manifest's own values.

    Only a run that recorded metrics shows any; a cancelled or failed run
    records none, so its cell is a dash rather than a number borrowed from
    somewhere else.
    """
    if not isinstance(metrics, dict) or not metrics:
        return NOT_AVAILABLE
    return "  ".join(
        f"{name} {compact_metric(value)}"
        for name, value in metrics.items()
        if isinstance(value, (int, float))
    ) or NOT_AVAILABLE


def history_row(manifest: Dict[str, Any]) -> Dict[str, str]:
    """One table row. Every value comes from ``manifest``."""
    started = str(manifest.get("started_at") or manifest.get("created_at") or "")
    return {
        "Run ID": str(manifest.get("run_id") or NOT_AVAILABLE),
        "Started": started or NOT_AVAILABLE,
        "Dataset": str(manifest.get("dataset") or NOT_AVAILABLE),
        "Task": str(manifest.get("task") or NOT_AVAILABLE),
        "Model": str(
            manifest.get("model_display") or manifest.get("model_key") or NOT_AVAILABLE
        ),
        "Seed": str(manifest.get("seed") or NOT_AVAILABLE),
        "Epochs": str(manifest.get("epochs") or "Config default"),
        "Device": str(manifest.get("device") or NOT_AVAILABLE),
        "Status": str(manifest.get("status") or NOT_AVAILABLE),
        "Duration": format_duration(manifest.get("elapsed_seconds")),
        "Metrics": format_metrics(manifest.get("metrics")),
    }


def history_rows(manifests: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """Rows for a whole history, in the order given (already newest-first)."""
    return [history_row(manifest) for manifest in manifests]


def load_history(project_root: str) -> List[Dict[str, Any]]:
    """The current project's runs, newest first, reconciled against reality.

    ``live_run_ids`` comes from the manager, so a manifest left saying
    RUNNING by a previous server process is reported INTERRUPTED rather than
    pretending the job is still going.
    """
    return rm.list_runs(project_root, live_run_ids=get_run_manager().live_run_ids())


def render_run_history(project_root: str, *, project_name: str = "") -> Optional[str]:
    """Draw the history table and return the run the user opened, if any."""
    st.subheader(HISTORY_TITLE, anchor=False)
    runs = load_history(project_root)
    if not runs:
        st.caption(
            f"No runs recorded yet for this project ({project_root}). A run "
            "appears here as soon as it is started."
        )
        return None

    st.caption(
        f"{len(runs)} run(s) recorded for project "
        f"{project_name or project_root}, newest first. Read from the run "
        "manifests on disk, so this survives a refresh or a restart."
    )
    st.dataframe(history_rows(runs), hide_index=True, use_container_width=True)

    ids = [str(entry.get("run_id")) for entry in runs]
    chosen = st.selectbox(
        "Open a run",
        ids,
        index=0,
        key="run_history_selection",
    )
    return str(chosen) if chosen else None


def render_run_detail(manifest: Dict[str, Any]) -> None:
    """Provenance and outcome for one historical run. Never re-runs it."""
    status = str(manifest.get("status") or "")
    st.markdown(f"**{manifest.get('run_id')}** — {status}")
    if manifest.get("reconcile_reason"):
        st.warning(str(manifest["reconcile_reason"]))

    rows = [
        {"Field": "Project", "Value": str(manifest.get("project") or NOT_AVAILABLE)},
        {"Field": "Run directory", "Value": str(manifest.get("run_dir") or manifest.get("workspace") or NOT_AVAILABLE)},
        {"Field": "Dataset", "Value": str(manifest.get("dataset") or NOT_AVAILABLE)},
        {"Field": "Task", "Value": str(manifest.get("task") or NOT_AVAILABLE)},
        {"Field": "Model", "Value": str(manifest.get("model_display") or manifest.get("model_key") or NOT_AVAILABLE)},
        {"Field": "Config", "Value": str(manifest.get("config_path") or NOT_AVAILABLE)},
        {"Field": "Seed", "Value": str(manifest.get("seed") or NOT_AVAILABLE)},
        {"Field": "Epochs", "Value": str(manifest.get("epochs") or "Config default")},
        {"Field": "Device", "Value": str(manifest.get("device") or NOT_AVAILABLE)},
        {"Field": "Status", "Value": status or NOT_AVAILABLE},
        {"Field": "Duration", "Value": format_duration(manifest.get("elapsed_seconds"))},
        {"Field": "Return code", "Value": str(manifest.get("return_code")) if manifest.get("return_code") is not None else NOT_AVAILABLE},
        {"Field": "Git commit", "Value": str(manifest.get("git_commit") or NOT_AVAILABLE)},
    ]
    st.dataframe(rows, hide_index=True, use_container_width=True)

    if status == rm.STATUS_COMPLETED:
        metrics = manifest.get("metrics") or {}
        if metrics:
            st.markdown("**Metrics**")
            st.dataframe(
                [
                    {"Metric": name, "Value": compact_metric(value)}
                    for name, value in metrics.items()
                ],
                hide_index=True,
                use_container_width=True,
            )
        else:
            st.caption(
                "This run completed but recorded no metrics in its manifest."
            )
    else:
        failure = manifest.get("failure") or {}
        if failure:
            st.error(str(failure.get("message") or status))
        # Deliberately no metrics for a run that did not complete: a
        # cancelled or failed run has not measured anything.
        if manifest.get("partial_artifacts"):
            st.caption(
                "Artifacts from this run are partial — it was stopped before "
                "it finished."
            )
