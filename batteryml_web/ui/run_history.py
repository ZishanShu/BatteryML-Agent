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

#: Shown only for a run that can actually be continued.
RESUME_LABEL = "Resume training"

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

    lineage = manifest.get("resumed_from")
    if isinstance(lineage, dict) and lineage:
        # A resumed run says where it continued from. The source run is
        # untouched and still in this history under its own id.
        st.info(
            f"Resumed from {lineage.get('run_id')} at epoch "
            f"{lineage.get('completed_epochs')} — this run trained epochs "
            f"{lineage.get('starting_epoch')}–{lineage.get('target_total_epochs')}."
        )

    _render_resume_offer(manifest)

    run_dir = str(manifest.get("run_dir") or manifest.get("workspace") or "")
    if run_dir:
        if status == rm.STATUS_COMPLETED:
            _render_scientific_views(manifest)
            render_bundle_download(run_dir, manifest)
        render_artifacts(run_dir)


def resume_offer(manifest: Dict[str, Any]) -> Dict[str, Any]:
    """Whether this run may be resumed, and from what.

    Pure, so the gate is testable without Streamlit. Every condition is
    backend-derived and all must hold:

    * the model's training loop can resume (``resume_capability``);
    * at least one checkpoint passes the resumable-checkpoint schema — a
      filename is never evidence, and an older weights-only ``.ckpt`` is not
      enough;
    * the run belongs to a project, so the resumed run has somewhere to go.

    Returns ``{"available", "checkpoint", "completed_epochs",
    "suggested_target"}``. ``available`` False means the control is simply
    not rendered — an unsupported run stays uncluttered.
    """
    from ..agent.run_policy import resume_capability
    from ..core.training_checkpoint import KIND_RESUMABLE, classify_checkpoint

    empty = {
        "available": False,
        "checkpoint": None,
        "completed_epochs": 0,
        "suggested_target": 1,
    }
    capability = resume_capability(
        str(manifest.get("model_key") or ""),
        family=str(manifest.get("model_family") or ""),
    )
    if not capability["supports_resume"]:
        return empty

    resumable = [
        verdict
        for verdict in (
            classify_checkpoint(path)
            for path in (manifest.get("artifacts") or {}).get("checkpoint_files") or []
        )
        if verdict["kind"] == KIND_RESUMABLE
    ]
    if not resumable:
        return empty

    latest = max(resumable, key=lambda v: v.get("epoch") or 0)
    completed = int(latest.get("epoch") or 0)
    stored = int(latest.get("target_total_epochs") or 0)
    return {
        "available": True,
        "checkpoint": latest,
        "completed_epochs": completed,
        "suggested_target": max(completed + 1, stored),
    }


def _render_resume_offer(manifest: Dict[str, Any]) -> None:
    """The Resume panel: three lines, and a button that really starts a run.

    Deliberately small. Optimizer, RNG and schema internals are what make
    resume correct, not what a user needs to see before pressing it; they
    stay in docs/TRAINING_STATE_AUDIT.md and in the checkpoint itself.
    """
    offer = resume_offer(manifest)
    if not offer["available"]:
        return

    completed = offer["completed_epochs"]
    run_id = str(manifest.get("run_id"))
    with st.expander(RESUME_LABEL, expanded=False):
        st.caption(f"Last checkpoint: epoch {completed}")
        target = int(
            st.number_input(
                "Target total epochs",
                min_value=completed + 1,
                value=offer["suggested_target"],
                step=1,
                key=f"resume_target_{run_id}",
            )
        )
        st.caption(
            f"{target - completed} epoch(s) remaining — continues from epoch "
            f"{completed + 1} in a new run. This run is not modified."
        )
        if st.button(
            "Resume",
            key=f"resume_confirm_{run_id}",
            use_container_width=True,
        ):
            _start_resumed_run(manifest, offer, target)


def _start_resumed_run(
    manifest: Dict[str, Any],
    offer: Dict[str, Any],
    target_total_epochs: int,
) -> None:
    """Validate once more, then start a new run through the normal path.

    The same planner, the same RunManager, the same manifest lifecycle as any
    other run — a resumed run is an ordinary run that happens to start from a
    checkpoint, so it gets live progress, Stop run, artifacts and a result
    without a second execution system.
    """
    from ..agent.execution import build_execution_plan
    from ..core.project_context import resolve_project, run_workspace
    from ..core.run_identity import new_run_id
    from ..core.run_manifest import build_resume_lineage
    from ..core.training_checkpoint import validate_resume
    from .agent_workflow import _launch_run

    checkpoint = offer["checkpoint"]
    completed = offer["completed_epochs"]
    dataset = str(manifest.get("dataset") or "")
    model_key = str(manifest.get("model_key") or "")

    # The identity the checkpoint must agree with. Re-checked here rather
    # than trusted from the panel: the number box is user input.
    validation = validate_resume(
        checkpoint,
        {
            "model_key": model_key,
            "dataset": dataset,
            "task": str(manifest.get("task") or ""),
            "seed": str(manifest.get("seed") or ""),
        },
        target_total_epochs=target_total_epochs,
    )
    if not validation["ok"]:
        for message in validation["errors"]:
            st.error(message)
        return
    for message in validation["warnings"]:
        # What could not be compared, said rather than hidden. The checkpoint
        # lives inside this run's own directory and the plan is built from
        # this run's manifest, so identity is anchored either way.
        st.caption(message)

    project = resolve_project(
        project_root=st.session_state.get("project_root"),
        project_name=st.session_state.get("project_name")
        or st.session_state.get("current_project"),
    )
    run_id = new_run_id(dataset=dataset, model=model_key, seed=manifest.get("seed"))
    workspace = run_workspace(project, dataset, model_key, run_id)

    plan = build_execution_plan(
        dataset,
        str(manifest.get("task") or ""),
        model_key,
        device=str(manifest.get("device") or "cpu"),
        seed=manifest.get("seed") or 0,
        workspace=workspace,
        # Total target, not additional: the loop runs start_epoch..target.
        epochs=target_total_epochs,
        resume_from=str(checkpoint["path"]),
    )
    if not plan.get("can_execute"):
        st.error(str(plan.get("reason") or "The resumed run could not be planned."))
        for message in plan.get("validation_errors") or []:
            st.caption(str(message))
        return

    _launch_run(
        run_id=run_id,
        plan=plan,
        project=project,
        run_context={
            "workspace": workspace,
            "seed": manifest.get("seed"),
            "device": manifest.get("device"),
            "smoke": False,
        },
        resumed_from=build_resume_lineage(
            source_run_id=str(manifest.get("run_id")),
            checkpoint_path=str(checkpoint["path"]),
            starting_epoch=int(validation["start_epoch"]),
            target_total_epochs=target_total_epochs,
            completed_epochs=completed,
        ),
    )
    st.success(
        f"Resumed run started from epoch {validation['start_epoch']}. It "
        "appears in Run History as a new run."
    )
    st.rerun()


def _render_scientific_views(manifest: Dict[str, Any]) -> None:
    """Figures and the prediction table for a completed run.

    Built from the run's own saved predictions artifact through the shared
    reader; nothing here recomputes a metric, and a run whose artifact does
    not carry the needed values gets the reason rather than a chart.
    """
    from ..agent.result_views import (
        RUL_CYCLEWISE_SCHEMA,
        SCALAR_SCHEMAS,
        SOH_SCHEMA,
        TRAJECTORY_SCHEMA,
        load_view,
    )

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

    if views["schema"] in SCALAR_SCHEMAS:
        render_rul_figures(views["rul"])
        # Only the cycle-wise schema. A scalar per-cell RUL result used to
        # reach this too: the view refused itself, but the renderer still
        # drew a disclosure headed "Cycle-wise RUL curve" to say so, and a
        # reader scanning headings saw a cycle-wise curve announced on a run
        # that has one lifetime estimate per cell. Saying nothing is the
        # honest answer — there is no such view to offer and nothing for the
        # reader to act on.
        if views["schema"] == RUL_CYCLEWISE_SCHEMA:
            render_rul_cycle_figure(views["rul_cycles"])
        render_prediction_table(views["rul"])
    elif views["schema"] == SOH_SCHEMA:
        render_soh_figures(views["soh"])
        render_prediction_table(views["soh"])
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


def _show_figure(draw) -> None:
    """Render a chart at the available tab width instead of clipping it."""
    st.pyplot(_figure(draw), clear_figure=True, use_container_width=True)


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

        _show_figure(_scatter)
    with right:
        st.caption("Residuals (predicted − observed)")

        def _residuals(axes) -> None:
            axes.scatter(observed, residuals, s=18, alpha=0.75)
            axes.axhline(0.0, linestyle="--", linewidth=1)
            axes.set_xlabel("Observed")
            axes.set_ylabel("Residual")

        _show_figure(_residuals)


def render_rul_cycle_figure(view: Dict[str, Any], *, max_cells: int = 6) -> None:
    """Remaining useful life against the cycle each estimate was made at.

    Drawn only for an artifact that genuinely holds repeated RUL estimates
    for the same cell at several observed cycles. For a per-cell lifetime
    result the view refuses itself, and this draws the refusal as a quiet
    caption inside a closed disclosure rather than a warning: nothing is
    wrong with such a run, and there is nothing for the reader to fix.

    One line pair per cell, never a single series across cells: the view
    hands over per-cell entries precisely so two cells cannot be joined.
    """
    if not view.get("available"):
        reason = str(view.get("reason") or "")
        if reason:
            with st.expander("Cycle-wise RUL curve", expanded=False):
                st.caption(reason)
        return
    if view.get("note"):
        st.caption(str(view["note"]))

    cells = view.get("cells") or []
    st.caption(
        f"{view['count']:,} remaining-life estimate(s) across "
        f"{view.get('n_cells', 0)} test cell(s); "
        f"first {min(len(cells), max_cells)} shown."
    )

    def _curves(axes) -> None:
        for position, cell in enumerate(cells[:max_cells]):
            # Each cell is plotted from its own arrays, so no line can span
            # two cells. The view already sorted the cycles ascending.
            axes.plot(
                cell["cycles"], cell["observed"],
                linewidth=1,
                label="observed / true RUL" if position == 0 else None,
            )
            axes.plot(
                cell["cycles"], cell["predicted"],
                linewidth=1, linestyle="--",
                label="estimated RUL" if position == 0 else None,
            )
        axes.set_xlabel("Observed cycle")
        axes.set_ylabel("Remaining useful life (cycles)")
        axes.legend(loc="best", fontsize="small")

    _show_figure(_curves)


def render_soh_figures(view: Dict[str, Any], *, max_cells: int = 6) -> None:
    """SOH against cycle, observed vs estimated, plus the range diagnostic.

    A cycle-level result is read per cell: a scatter of all samples answers
    "is the model calibrated", but only a per-cell curve answers "does it
    track degradation". Both are drawn, from the stored arrays alone.

    The diagnostic below them is the honest part. A ratio below 0 or well
    above 1 is physically impossible, and an average error can hide a few of
    them completely. They are *counted and shown*, never clipped away: the
    values plotted are the values the metrics came from.
    """
    if not view.get("available"):
        st.info(str(view.get("reason") or "This visualization is unavailable."))
        return
    if view.get("note"):
        st.caption(str(view["note"]))

    observed = view["observed"]
    predicted = view["predicted"]
    cell_ids = view.get("cell_ids") or []
    cycles = view.get("cycle_indices") or []
    st.caption(
        f"{view['count']:,} cycle-level sample(s) across "
        f"{view.get('n_cells', 0)} test cell(s)."
    )

    left, right = st.columns(2)
    with left:
        st.caption("Observed vs estimated SOH (ratio)")

        def _scatter(axes) -> None:
            axes.scatter(observed, predicted, s=8, alpha=0.4)
            low = min(min(observed), min(predicted))
            high = max(max(observed), max(predicted))
            axes.plot([low, high], [low, high], linestyle="--", linewidth=1)
            axes.set_xlabel("Observed SOH (ratio)")
            axes.set_ylabel("Estimated SOH (ratio)")

        _show_figure(_scatter)
    with right:
        st.caption(f"SOH across cycles, first {max_cells} test cell(s)")

        def _per_cell(axes) -> None:
            # Insertion order, so the cells shown are the first the run
            # actually evaluated rather than an alphabetical sample of them.
            selected: List[str] = []
            for cell in cell_ids:
                if cell not in selected:
                    selected.append(cell)
                if len(selected) >= max_cells:
                    break
            for position, cell in enumerate(selected):
                rows = [i for i, name in enumerate(cell_ids) if name == cell]
                rows.sort(key=lambda i: cycles[i])
                axes.plot(
                    [cycles[i] for i in rows], [observed[i] for i in rows],
                    linewidth=1,
                    label="observed" if position == 0 else None,
                )
                axes.plot(
                    [cycles[i] for i in rows], [predicted[i] for i in rows],
                    linewidth=1, linestyle="--",
                    label="estimated" if position == 0 else None,
                )
            axes.set_xlabel("Cycle")
            axes.set_ylabel("SOH (ratio)")
            if selected:
                axes.legend(loc="best", fontsize="small")

        _show_figure(_per_cell)

    render_soh_range_diagnostics(view)


def render_soh_range_diagnostics(view: Dict[str, Any]) -> None:
    """How many estimates fall outside the physical range of a ratio.

    Shown for every SOH run, including a clean one: "0 outside the range" is
    information, and a panel that appears only when something is wrong
    teaches the reader that its absence means nothing was checked.
    """
    from ..agent.result_views import largest_residuals

    summary = view.get("range") or {}
    if not summary.get("count"):
        return
    st.markdown("**Estimate range check**")
    columns = st.columns(4)
    figures = (
        ("Below 0", summary.get("below_zero", 0)),
        ("Above 1.0", summary.get("above_one", 0)),
        ("Above 1.1", summary.get("above_implausible", 0)),
        ("Outside range", summary.get("out_of_range", 0)),
    )
    for column, (label, value) in zip(columns, figures):
        with column:
            st.metric(label, f"{int(value):,}")
    low, high = summary.get("min"), summary.get("max")
    if low is not None and high is not None:
        st.caption(
            f"Estimated SOH spans {low:.4f} to {high:.4f} across "
            f"{int(summary['count']):,} sample(s); "
            f"{summary.get('fraction_out_of_range', 0.0) * 100:.2f}% lie "
            "outside (0, 1.0]."
        )
    st.caption(str(summary.get("note") or ""))

    rows = largest_residuals(view, limit=10)
    if rows:
        st.markdown("**Largest absolute residuals**")
        st.caption(
            "The ten samples where the estimate misses the observed SOH by "
            "the most. A selection of stored rows, ordered; no value is "
            "recomputed."
        )
        st.dataframe(rows, hide_index=True, use_container_width=True)


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
    _show_figure(_draw)


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


# --- structured file browsing ---------------------------------------------

#: Order the artifact groups are shown in: what a user looks for first.
GROUP_ORDER: tuple[str, ...] = (
    "predictions",
    "metrics",
    "figures",
    "config",
    "manifest",
    "logs",
    "checkpoints",
    "other",
)

GROUP_TITLES: Dict[str, str] = {
    "predictions": "Predictions",
    "metrics": "Metrics and summaries",
    "figures": "Figures",
    "config": "Config",
    "manifest": "Manifest",
    "logs": "Logs",
    "checkpoints": "Checkpoints",
    "other": "Other files",
}


def _human_size(size: Any) -> str:
    try:
        value = float(size)
    except (TypeError, ValueError):
        return "—"
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024 or unit == "GB":
            return f"{value:,.0f} {unit}" if unit == "B" else f"{value:,.1f} {unit}"
        value /= 1024
    return f"{value:,.1f} GB"


def _modified(path: str) -> str:
    from datetime import datetime, timezone

    try:
        stamp = Path(path).stat().st_mtime
    except OSError:
        return "—"
    return datetime.fromtimestamp(stamp, tz=timezone.utc).strftime("%Y-%m-%d %H:%M")


def group_artifacts(entries: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """Group discovered artifacts by category, newest-looking order kept.

    Purely a rearrangement of what ``list_artifacts`` found: no group is
    created for files that do not exist, and nothing is added to make a
    group look complete.
    """
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for entry in entries:
        grouped.setdefault(str(entry.get("category") or "other"), []).append(entry)
    return {
        name: grouped[name]
        for name in GROUP_ORDER
        if name in grouped
    } | {
        name: rows for name, rows in grouped.items() if name not in GROUP_ORDER
    }


def render_run_location(manifest: Dict[str, Any]) -> None:
    """Say plainly where this run's output went.

    Project, workspace and run directory, each on its own line. Inferring
    the run directory from a truncated path in a table was the thing this
    replaces.
    """
    run_dir = str(manifest.get("run_dir") or manifest.get("workspace") or "")
    workspace = str(Path(run_dir).parent) if run_dir else ""
    st.markdown("**Where this run wrote**")
    st.dataframe(
        [
            {"": "Project", "Path": str(manifest.get("project") or NOT_AVAILABLE)},
            {"": "Workspace", "Path": workspace or NOT_AVAILABLE},
            {"": "Run directory", "Path": run_dir or NOT_AVAILABLE},
        ],
        hide_index=True,
        use_container_width=True,
    )
    if run_dir:
        st.code(run_dir, language="text")


def render_file_browser(run_dir: str) -> None:
    """The Files view: grouped, with metadata, not a flat list of paths."""
    from ..core.artifact_browser import list_artifacts, read_artifact

    entries = list_artifacts(run_dir)
    if not entries:
        st.caption(
            f"No files were found under {run_dir}. Nothing is listed that "
            "does not exist."
        )
        return

    grouped = group_artifacts(entries)
    st.caption(
        f"{len(entries)} file(s) in {len(grouped)} group(s), discovered under "
        "this run's own directory."
    )
    for name, rows in grouped.items():
        with st.expander(
            f"{GROUP_TITLES.get(name, name.title())} ({len(rows)})",
            expanded=(name == "predictions"),
        ):
            st.dataframe(
                [
                    {
                        "File": row["relative_path"],
                        "Size": _human_size(row["size_bytes"]),
                        "Modified (UTC)": _modified(
                            str(Path(run_dir) / row["relative_path"])
                        ),
                    }
                    for row in rows
                ],
                hide_index=True,
                use_container_width=True,
            )
            downloadable = [row for row in rows if row["downloadable"]]
            if not downloadable:
                st.caption(
                    "These files are larger than the download limit and stay "
                    "in the run directory."
                )
                continue
            choice = st.selectbox(
                "File",
                [row["relative_path"] for row in downloadable],
                key=f"files_{run_dir}_{name}",
                label_visibility="collapsed",
            )
            if choice:
                st.code(str(Path(run_dir) / str(choice)), language="text")
                payload = read_artifact(run_dir, str(choice))
                if payload is not None:
                    st.download_button(
                        "Download",
                        data=payload,
                        file_name=Path(str(choice)).name,
                        key=f"files_dl_{run_dir}_{name}",
                    )


def render_predictions_panel(manifest: Dict[str, Any]) -> None:
    """The Predictions view: one artifact selected, described and previewed."""
    from ..agent.artifact_preview import (
        KIND_BATTERYML_PREDICTION,
        KIND_UNREADABLE,
        KIND_UNSUPPORTED_PICKLE,
        SUPPORTED_PREVIEW_FORMATS,
        preview_prediction_file,
    )
    from ..agent.result_views import (
        RUL_CYCLEWISE_SCHEMA,
        SCALAR_SCHEMAS,
        SOH_SCHEMA,
        TRAJECTORY_SCHEMA,
    )

    files = [
        str(path)
        for path in (manifest.get("artifacts") or {}).get("prediction_files") or []
    ]
    if not files:
        st.info("This run recorded no predictions artifact.")
        return

    chosen = files[0]
    if len(files) > 1:
        chosen = st.selectbox(
            "Prediction artifact",
            files,
            key=f"prediction_choice_{manifest.get('run_id')}",
        )
    preview = preview_prediction_file(str(chosen))

    if preview["kind"] == KIND_BATTERYML_PREDICTION:
        st.markdown("**Artifact summary**")
        st.dataframe(
            [{"Field": key, "Value": value} for key, value in preview["summary"].items()],
            hide_index=True,
            use_container_width=True,
        )
        if preview["schema"] in SCALAR_SCHEMAS and preview["rul"]:
            render_rul_figures(preview["rul"])
            # As above: the heading belongs only to an artifact that has the
            # curve.
            if preview["schema"] == RUL_CYCLEWISE_SCHEMA:
                render_rul_cycle_figure(preview.get("rul_cycles") or {})
            render_prediction_table(preview["rul"])
        elif preview["schema"] == SOH_SCHEMA and preview.get("soh"):
            render_soh_figures(preview["soh"])
            render_prediction_table(preview["soh"])
        elif preview["schema"] == TRAJECTORY_SCHEMA and preview["trajectory"]:
            render_trajectory_figure(preview["trajectory"])
    elif preview["kind"] in (KIND_UNSUPPORTED_PICKLE, KIND_UNREADABLE):
        # Graceful, and useful: say what it is not, then show what can be
        # determined about it without interpreting anything.
        st.info(preview["message"])
        structure = preview.get("structure") or {}
        if structure:
            st.caption(
                f"Top-level type: {structure.get('type')}"
                + (f" · {structure.get('length')} entries" if structure.get("length") is not None else "")
            )
            if structure.get("fields"):
                st.dataframe(
                    [
                        {"Field": f["name"], "Type": f["type"], "Shape": f["shape"]}
                        for f in structure["fields"]
                    ],
                    hide_index=True,
                    use_container_width=True,
                )
    else:
        st.caption(
            "This artifact is not a pickle payload; use the Files view to "
            "download it."
        )

    st.code(str(chosen), language="text")
    with st.expander("Which formats can be previewed?", expanded=False):
        for line in SUPPORTED_PREVIEW_FORMATS:
            st.caption(f"- {line}")
        st.caption(
            "Anything else is listed and downloadable, but not interpreted."
        )
