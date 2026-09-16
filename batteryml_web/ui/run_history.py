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

from pathlib import Path
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

    run_dir = str(manifest.get("run_dir") or manifest.get("workspace") or "")
    if run_dir:
        if status == rm.STATUS_COMPLETED:
            _render_scientific_views(manifest)
            render_bundle_download(run_dir, manifest)
        render_artifacts(run_dir)


def _render_scientific_views(manifest: Dict[str, Any]) -> None:
    """Figures and the prediction table for a completed run.

    Built from the run's own saved predictions artifact through the shared
    reader; nothing here recomputes a metric, and a run whose artifact does
    not carry the needed values gets the reason rather than a chart.
    """
    from ..agent.result_views import RUL_SCHEMA, TRAJECTORY_SCHEMA, load_view

    files = (manifest.get("artifacts") or {}).get("prediction_files") or []
    if not files:
        st.caption(
            "This run recorded no predictions artifact, so there is nothing "
            "to visualize."
        )
        return
    try:
        views = load_view(files[0])
    except Exception as exc:  # noqa: BLE001 - a display path must not break a result
        st.warning(
            f"The predictions artifact could not be read "
            f"({type(exc).__name__}: {exc}); no figure is drawn from it."
        )
        return

    if views["schema"] == RUL_SCHEMA:
        render_rul_figures(views["rul"])
        render_prediction_table(views["rul"])
    elif views["schema"] == TRAJECTORY_SCHEMA:
        render_trajectory_figure(views["trajectory"])
    else:
        st.info(
            "This run's predictions artifact is not in a shape this platform "
            "knows how to visualize."
        )


# --- scientific views, artifacts and export -------------------------------

def _figure(draw) -> Any:
    """Build one matplotlib figure with ``draw(ax)`` and hand it back.

    matplotlib is already a platform dependency (the viz contract, FIG-01),
    so no new visualization framework is introduced for these two plots.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(figsize=(4.5, 4.0), dpi=120)
    draw(axes)
    figure.tight_layout()
    return figure


def render_rul_figures(view: Dict[str, Any]) -> None:
    """Observed-vs-predicted and residuals, or an explicit refusal.

    A run whose artifact lacks the values gets the reason, never a chart
    drawn from something else.
    """
    if not view.get("available"):
        st.info(str(view.get("reason") or "This visualization is unavailable."))
        return
    if view.get("note"):
        st.caption(str(view["note"]))

    observed = view["observed"]
    predicted = view["predicted"]
    residuals = view["residuals"]

    left, right = st.columns(2)
    with left:
        st.caption("Observed vs predicted")

        def _scatter(axes) -> None:
            axes.scatter(observed, predicted, s=18, alpha=0.75)
            low = min(min(observed), min(predicted))
            high = max(max(observed), max(predicted))
            # The reference line is y = x: perfect agreement, drawn from the
            # data's own range rather than a fixed scale.
            axes.plot([low, high], [low, high], linestyle="--", linewidth=1)
            axes.set_xlabel("Observed")
            axes.set_ylabel("Predicted")

        st.pyplot(_figure(_scatter), clear_figure=True)
    with right:
        st.caption("Residuals (predicted − observed)")

        def _residuals(axes) -> None:
            axes.scatter(observed, residuals, s=18, alpha=0.75)
            axes.axhline(0.0, linestyle="--", linewidth=1)
            axes.set_xlabel("Observed")
            axes.set_ylabel("Residual")

        st.pyplot(_figure(_residuals), clear_figure=True)


def render_trajectory_figure(view: Dict[str, Any], *, limit: int = 8) -> None:
    """Predicted (and observed, where stored) degradation trajectories."""
    if not view.get("available"):
        st.info(str(view.get("reason") or "This visualization is unavailable."))
        return
    predicted = view["predicted"][:limit]
    observed = view.get("observed") or []

    def _draw(axes) -> None:
        for index, series in enumerate(predicted):
            axes.plot(range(len(series)), series, linewidth=1,
                      label="predicted" if index == 0 else None)
        for index, series in enumerate(observed[:limit]):
            axes.plot(range(len(series)), series, linewidth=1, linestyle="--",
                      label="observed" if index == 0 else None)
        axes.set_xlabel("Cycle index")
        axes.set_ylabel("Value")
        if predicted or observed:
            axes.legend(loc="best", fontsize="small")

    st.caption(f"{view['count']} trajectory(ies), {view['length']} point(s) each")
    st.pyplot(_figure(_draw), clear_figure=True)


def render_prediction_table(view: Dict[str, Any], *, limit: int = 200) -> None:
    """The per-sample table, straight from the stored arrays."""
    from ..agent.result_views import prediction_table

    rows = prediction_table(view, limit=limit)
    if not rows:
        return
    st.markdown("**Predictions**")
    if view["count"] > len(rows):
        st.caption(f"Showing the first {len(rows)} of {view['count']} predictions.")
    st.dataframe(rows, hide_index=True, use_container_width=True)


def render_artifacts(run_dir: str) -> None:
    """The files this run really produced, downloadable within its own root."""
    from ..core.artifact_browser import list_artifacts, read_artifact

    entries = list_artifacts(run_dir)
    if not entries:
        st.caption("No artifact files were found in this run's directory.")
        return
    st.markdown("**Artifacts**")
    st.dataframe(
        [
            {
                "File": entry["relative_path"],
                "Kind": entry["category"],
                "Size": f"{entry['size_bytes']:,} bytes",
            }
            for entry in entries
        ],
        hide_index=True,
        use_container_width=True,
    )
    downloadable = [entry for entry in entries if entry["downloadable"]]
    if not downloadable:
        return
    choice = st.selectbox(
        "Download a file",
        [entry["relative_path"] for entry in downloadable],
        key=f"artifact_choice_{run_dir}",
    )
    if choice:
        # Path safety is the browser's job: anything resolving outside this
        # run's directory raises rather than being served.
        payload = read_artifact(run_dir, str(choice))
        if payload is not None:
            st.download_button(
                "Download selected file",
                data=payload,
                file_name=Path(str(choice)).name,
                key=f"artifact_download_{run_dir}",
            )


def render_bundle_download(run_dir: str, manifest: Dict[str, Any]) -> None:
    """Offer the reproducibility zip for a completed run."""
    import tempfile

    from ..agent.result_views import build_rul_view, load_view
    from ..core.reproducibility import predictions_csv_rows, write_bundle

    if str(manifest.get("status")) != rm.STATUS_COMPLETED:
        # A run that did not complete has nothing reproducible to export.
        return
    if not st.button("Prepare run bundle", key=f"bundle_{run_dir}"):
        return

    predictions_csv = None
    files = (manifest.get("artifacts") or {}).get("prediction_files") or []
    if files:
        try:
            view = load_view(files[0])["rul"]
            predictions_csv = predictions_csv_rows(view)
        except Exception:  # noqa: BLE001 - a missing CSV must not block the zip
            predictions_csv = None

    with tempfile.TemporaryDirectory() as tmp:
        target = Path(tmp) / f"{manifest.get('run_id')}_bundle.zip"
        summary = write_bundle(
            run_dir, target, manifest=manifest, predictions_csv=predictions_csv
        )
        payload = target.read_bytes()
    st.download_button(
        "Download run bundle",
        data=payload,
        file_name=target.name,
        key=f"bundle_download_{run_dir}",
    )
    if summary["omitted"]:
        st.caption(
            f"{len(summary['omitted'])} file(s) were left out of the bundle and "
            "are named in its README with the path where they remain."
        )
