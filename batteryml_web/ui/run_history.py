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

import html
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import streamlit as st

from ..core import run_manifest as rm
from ..core.run_identity import SUFFIX_CHARS, is_run_id
from ..core.run_manager import get_run_manager
from ..ui.formatting import compact_metric

HISTORY_TITLE = "Run history"

#: Shown only for a run that can actually be continued.
RESUME_LABEL = "Resume training"

#: Column order for the history table: compact and readable (UI-13). The
#: canonical run id and the recorded timestamp are not shortened anywhere
#: they are used — the selector's value, the details, the manifest — only
#: in this table's cells, and both appear in full in the run's details.
COLUMNS: tuple[str, ...] = (
    "Run",
    "Started",
    "Experiment",
    "Seed",
    "Device",
    "Status",
    "Duration",
    "Metrics",
)

NOT_AVAILABLE = "—"

#: A status word → the passive status pill's modifier (home_style.py,
#: ``.bml-status--*``). An unknown status gets the neutral pill, never a
#: guessed colour.
STATUS_KINDS: Dict[str, str] = {
    rm.STATUS_COMPLETED: "completed",
    rm.STATUS_FAILED: "failed",
    rm.STATUS_INTERRUPTED: "failed",
    rm.STATUS_CANCELLED: "cancelled",
    rm.STATUS_RUNNING: "running",
    rm.STATUS_PLANNED: "running",
}


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


def short_run_id(run_id: Any) -> str:
    """A run id shortened for reading in a table cell. Never used to find a run.

    An id this platform minted ends in a random hexadecimal suffix, which is
    what tells two runs started in the same second apart; that suffix is
    shown. Any other id is shown whole when short, otherwise as its head and
    tail around an ellipsis.
    """
    text = str(run_id or "")
    if not text:
        return NOT_AVAILABLE
    if is_run_id(text):
        return "…" + text[-SUFFIX_CHARS:]
    return text if len(text) <= 18 else f"{text[:8]}…{text[-SUFFIX_CHARS:]}"


def recorded_start(manifest: Dict[str, Any]) -> str:
    """The run's start as the manifest recorded it (``started_at``, else ``created_at``)."""
    return str(manifest.get("started_at") or manifest.get("created_at") or "")


def format_timestamp(value: Any) -> str:
    """``2026-10-12 17:29 UTC`` from a recorded ISO-8601 time, zone explicit.

    The zone shown is the one the stamp carries — no conversion to the
    server's or the browser's local time, so the cell says exactly what was
    recorded, minus the seconds. A stamp without a zone says so; a value
    that is not ISO-8601 is shown as recorded rather than guessed at.
    """
    text = str(value or "").strip()
    if not text:
        return NOT_AVAILABLE
    try:
        moment = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return text
    offset = moment.utcoffset()
    if offset is None:
        return f"{moment:%Y-%m-%d %H:%M} (time zone not recorded)"
    if offset == timedelta(0):
        return f"{moment:%Y-%m-%d %H:%M} UTC"
    minutes = int(offset.total_seconds() // 60)
    sign = "+" if minutes >= 0 else "-"
    hours, mins = divmod(abs(minutes), 60)
    return f"{moment:%Y-%m-%d %H:%M} UTC{sign}{hours:02d}:{mins:02d}"


def short_model_name(manifest: Dict[str, Any]) -> str:
    """The model's name without its long form: ``"XGBoost"`` for
    ``"XGBoost — Extreme Gradient Boosting"``. The full display name stays in
    the run's details."""
    display = str(manifest.get("model_display") or "").split(" — ", 1)[0].strip()
    return display or str(manifest.get("model_key") or "")


def experiment_summary(manifest: Dict[str, Any]) -> str:
    """``model · dataset · task`` from the manifest's own values."""
    parts = [
        short_model_name(manifest),
        str(manifest.get("dataset") or ""),
        str(manifest.get("task") or ""),
    ]
    return " · ".join(part for part in parts if part) or NOT_AVAILABLE


def status_badge(status: Any) -> str:
    """The run status as a passive pill (HTML). A word to read, not a control."""
    word = str(status or "").strip() or "UNKNOWN"
    kind = STATUS_KINDS.get(word.upper())
    modifier = f" bml-status--{kind}" if kind else ""
    return f'<span class="bml-status{modifier}">{html.escape(word)}</span>'


def history_row(manifest: Dict[str, Any]) -> Dict[str, str]:
    """One table row. Every value comes from ``manifest``; none is invented."""
    return {
        "Run": short_run_id(manifest.get("run_id")),
        "Started": format_timestamp(recorded_start(manifest)),
        "Experiment": experiment_summary(manifest),
        "Seed": str(manifest.get("seed") if manifest.get("seed") not in (None, "") else NOT_AVAILABLE),
        "Device": str(manifest.get("device") or NOT_AVAILABLE),
        "Status": str(manifest.get("status") or NOT_AVAILABLE),
        "Duration": format_duration(manifest.get("elapsed_seconds")),
        "Metrics": format_metrics(manifest.get("metrics")),
    }


def history_rows(manifests: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """Rows for a whole history, in the order given (already newest-first)."""
    return [history_row(manifest) for manifest in manifests]


def run_option_label(manifest: Dict[str, Any]) -> str:
    """What the "Open a run" selector shows for one run; its value stays the full id."""
    return " · ".join((
        short_run_id(manifest.get("run_id")),
        format_timestamp(recorded_start(manifest)),
        experiment_summary(manifest),
        str(manifest.get("status") or NOT_AVAILABLE),
    ))


def _history_column_config() -> Optional[Dict[str, Any]]:
    """Column widths and the full-id hint, where this Streamlit has column_config."""
    config = getattr(st, "column_config", None)
    if config is None or not hasattr(config, "TextColumn"):
        return None
    # Widths are left to the content except the short id: fixed widths cut
    # "COMPLETED" and the metrics off at 1440px (measured in a browser).
    return {
        "Run": config.TextColumn("Run", width="small",
                                 help="End of the run id. The full id is in the run's details below."),
        "Started": config.TextColumn("Started",
                                     help="Recorded start time, in the time zone the run recorded."),
        "Experiment": config.TextColumn("Experiment", help="Model · dataset · task. The full model name is in the run's details."),
    }


def load_history(project_root: str) -> List[Dict[str, Any]]:
    """The current project's runs, newest first, reconciled against reality.

    ``live_run_ids`` comes from the manager, so a manifest left saying
    RUNNING by a previous server process is reported INTERRUPTED rather than
    pretending the job is still going.
    """
    return rm.list_runs(project_root, live_run_ids=get_run_manager().live_run_ids())


def projects_with_runs(projects_root: Path, *, exclude: str = "") -> List[tuple]:
    """``[(project name, run count)]`` for every other project that has runs.

    Read-only and name-sorted; used only to tell a reader of an empty history
    where runs actually are. ``rm.list_runs`` is the same reader the history
    uses, so a folder that is not a project yields nothing.
    """
    root = Path(projects_root)
    if not root.is_dir():
        return []
    found = []
    for child in sorted(root.iterdir(), key=lambda p: p.name.lower()):
        if child.is_dir() and child.name != exclude:
            count = len(rm.list_runs(child))
            if count:
                found.append((child.name, count))
    return found


def render_run_history(project_root: str, *, project_name: str = "") -> Optional[str]:
    """Draw the history table and return the run the user opened, if any."""
    st.subheader(HISTORY_TITLE, anchor=False)
    runs = load_history(project_root)
    if not runs:
        st.caption(
            f"No runs recorded yet for this project ({project_root}). A run "
            "appears here as soon as it is started."
        )
        others = projects_with_runs(Path(project_root).parent, exclude=Path(project_root).name)
        if others:
            # Runs are never pulled in from another project; the reader is
            # told where they are and how to open them.
            listed = ", ".join(f"{name} ({count})" for name, count in others)
            st.caption(
                f"Runs are recorded in other projects: {listed}. Load one from "
                "Home → Existing project to review its runs here."
            )
        return None

    st.caption(
        f"{len(runs)} run(s) recorded for project "
        f"{project_name or project_root}, newest first. Read from the run "
        "manifests on disk, so this survives a refresh or a restart."
    )
    table: Dict[str, Any] = {"hide_index": True, "use_container_width": True}
    column_config = _history_column_config()
    if column_config is not None:
        table["column_config"] = column_config
    st.dataframe(history_rows(runs), **table)

    # The value is the canonical run id — that is what the rest of the page
    # looks the run up by; only the label is shortened.
    ids = [str(entry.get("run_id")) for entry in runs]
    labels = {str(entry.get("run_id")): run_option_label(entry) for entry in runs}
    chosen = st.selectbox(
        "Open a run",
        ids,
        index=0,
        key="run_history_selection",
        format_func=lambda run_id: labels.get(str(run_id), str(run_id)),
    )
    return str(chosen) if chosen else None


def render_run_detail(manifest: Dict[str, Any]) -> None:
    """Provenance and outcome for one historical run. Never re-runs it."""
    status = str(manifest.get("status") or "")
    # Status is shown as status — a pill to read — beside what was run; the
    # full run id sits in a code block so it can be copied whole.
    st.markdown(
        f"{status_badge(status)}&nbsp;&nbsp;{html.escape(experiment_summary(manifest))}",
        unsafe_allow_html=True,
    )
    st.code(str(manifest.get("run_id") or NOT_AVAILABLE), language="text")
    if manifest.get("reconcile_reason"):
        st.warning(str(manifest["reconcile_reason"]))

    rows = [
        {"Field": "Run ID", "Value": str(manifest.get("run_id") or NOT_AVAILABLE)},
        {"Field": "Started (as recorded)", "Value": recorded_start(manifest) or NOT_AVAILABLE},
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
            st.markdown("#### Metrics")
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
            st.markdown("#### Scientific plots")
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
    chosen = str(files[0])
    if len(files) > 1:
        # Never silently "the first": a run with several artifacts asks.
        picked = st.selectbox(
            "Prediction artifact", [str(f) for f in files],
            key=f"history_artifact::{manifest.get('run_id')}",
        )
        if picked in [str(f) for f in files]:
            chosen = str(picked)
    try:
        views = load_view(chosen)
    except Exception as exc:  # noqa: BLE001 - a display path must not break a result
        st.warning(
            f"The predictions artifact could not be read "
            f"({type(exc).__name__}: {exc}); no figure is drawn from it."
        )
        return

    if views["schema"] in SCALAR_SCHEMAS:
        render_rul_figures(views["rul"], placement="history")
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
        render_soh_figures(views["soh"], placement="history")
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


def _artifact_scope(view: Dict[str, Any], placement: str) -> str:
    """A widget-key scope naming exactly one artifact revision and one placement.

    Run A and run B never share a key, so a cell or zoom choice made on A
    cannot survive as the wrong cell on B; a rewritten artifact (size or
    mtime changed) is a new scope too. The placement keeps the Run History
    detail and the Predictions tab — both on Review Results — apart.
    """
    path = str(view.get("artifact_path") or "")
    revision = ""
    if path:
        try:
            stat = Path(path).stat()
            revision = f"{stat.st_size}:{stat.st_mtime_ns}"
        except OSError:
            revision = "missing"
    return f"{placement}::{path}::{revision}"


def _figure_badge(view: Dict[str, Any]) -> Optional[str]:
    """``smoke run · not benchmark performance`` when run.json says so."""
    path = view.get("artifact_path")
    if not path:
        return None
    from ..agent.artifact_provenance import artifact_provenance
    from ..viz.result_export import SMOKE_BADGE

    try:
        record = artifact_provenance(path, hash_files=False)
    except FileNotFoundError:
        return None
    return SMOKE_BADGE if record.get("smoke") is True else None


ZOOM_OPTIONS = ("Full range", "Zoomed view")


def _zoom_choice(scope: str) -> bool:
    """True for the zoomed viewport. Full range is the default and always one click away."""
    choice = st.radio(
        "Axis range", ZOOM_OPTIONS, index=0, horizontal=True,
        key=f"viz_zoom::{scope}",
        help="Zoomed view changes only the visible range; no value is clipped, "
             "and the figure states how many points fall outside it.",
    )
    return choice == ZOOM_OPTIONS[1]


def _pyplot_full_width(figure: Any) -> None:
    """Draw a figure at the available tab width instead of clipping it.

    Newer Streamlit takes ``width="stretch"`` (``use_container_width`` is
    deprecated there); older releases only know ``use_container_width``.
    """
    import inspect

    try:
        accepts_width = "width" in inspect.signature(st.pyplot).parameters
    except (TypeError, ValueError):
        accepts_width = False
    if accepts_width:
        st.pyplot(figure, clear_figure=True, width="stretch")
    else:
        st.pyplot(figure, clear_figure=True, use_container_width=True)


def _show(panel: Dict[str, Any]) -> None:
    _pyplot_full_width(panel["figure"])


def _show_figure(draw) -> None:
    """Render a chart drawn by ``draw`` at the available tab width."""
    _pyplot_full_width(_figure(draw))


def _render_export_action(view: Dict[str, Any], scope: str, *, cell_id: Optional[str]) -> None:
    """Compact: the exact CLI command, and the same files as a zip on request."""
    path = view.get("artifact_path")
    if not path:
        return
    from ..viz.result_export import ExportError, build_export, export_command, zip_bytes

    with st.expander("Export figures", expanded=False):
        st.caption(
            "SVG and PNG panels, the exact plotted values (CSV) and a provenance "
            "record (figure_sources.json), built from this artifact only."
        )
        st.code(export_command(path, cell_id=cell_id), language="bash")
        if st.button("Prepare figure package", key=f"viz_export::{scope}"):
            try:
                # What actually happened, as a structured record. No command was
                # run here, so none is written: the run's own command comes from
                # its run.json inside build_export.
                result = build_export(
                    path, cell_id=cell_id, dpi=300.0,
                    invocation={"interface": "website", "action": "Prepare figure package",
                                "artifact": str(path), "cell_id": cell_id, "dpi": 300.0},
                )
            except ExportError as exc:
                st.error(str(exc))
                return
            st.download_button(
                "Download figure package (.zip)", zip_bytes(result),
                file_name=f"figures_{Path(str(path)).stem}.zip",
                mime="application/zip", on_click="ignore",
                key=f"viz_export_download::{scope}",
            )


def _granularity_caption(view: Dict[str, Any]) -> str:
    target = view.get("target") or {}
    parts = [
        f"{target.get('quantity', 'Target')} in {target.get('unit', 'unit not recorded')}",
        f"{int(view.get('count') or 0):,} plotted point(s)"
        + (f" from {len(set(view.get('cell_ids') or []))} cell(s)" if view.get("cell_ids") else ""),
        str(target.get("granularity") or ""),
    ]
    return " · ".join(part for part in parts if part)


def render_rul_figures(view: Dict[str, Any], *, placement: str = "result") -> None:
    """Observed-vs-predicted and residuals, or an explicit refusal.

    A run whose artifact lacks the values gets the reason, never a chart
    drawn from something else. The figures come from
    :func:`batteryml_web.viz.result_figures.build_panels` — the same function
    the export uses — so what is shown is what is exported.
    """
    if not view.get("available"):
        st.info(str(view.get("reason") or "This visualization is unavailable."))
        return
    from ..viz.result_figures import FigureDataError, build_panels

    target = view.get("target") or {}
    st.caption(_granularity_caption(view))
    if target.get("definition"):
        st.caption(str(target["definition"]))
    if view.get("note"):
        st.caption(str(view["note"]))

    scope = _artifact_scope(view, placement)
    zoomed = _zoom_choice(scope)
    try:
        panels = build_panels(
            view, saved_metrics=view.get("saved_metrics"), badge=_figure_badge(view),
            include_zoom=zoomed,
        )
    except FigureDataError as exc:
        st.info(str(exc))
        return
    suffix = "_zoom" if zoomed else ""
    left, right = st.columns(2)
    with left:
        _show(panels["parity" + suffix])
    with right:
        _show(panels["residuals" + suffix])
    if zoomed:
        st.caption(
            f"Zoomed view: {panels['parity_zoom']['meta']['outside']:,} point(s) lie "
            "outside the parity viewport and "
            f"{panels['residuals_zoom']['meta']['outside']:,} outside the residual "
            "viewport. They are still in the data and in the full-range view."
        )
    _render_export_action(view, scope, cell_id=None)


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


def render_soh_figures(view: Dict[str, Any], *, placement: str = "result") -> None:
    """One selected cell's SOH across observed cycles, parity and residuals.

    The cycle panel shows **one** cell at a time — six overlaid cells with a
    two-entry legend could not say which line was which. The default cell is
    the first ID in ascending order, a rule that does not look at prediction
    error; the selector lists only cells this artifact really holds, and its
    key is scoped to this artifact, so a cell picked on another run never
    carries over. Parity and residuals use every plotted sample. All panels
    come from ``build_panels``, the export's own function.

    The range diagnostic below is the honest part: estimates far outside the
    usual range are counted and shown, never clipped away.
    """
    if not view.get("available"):
        st.info(str(view.get("reason") or "This visualization is unavailable."))
        return
    from ..viz.result_figures import (
        DEFAULT_CELL_RULE, PER_CELL_NOTE, FigureDataError, build_panels,
        cell_choices, per_cell_errors,
    )

    target = view.get("target") or {}
    st.caption(_granularity_caption(view))
    if target.get("definition"):
        st.caption(str(target["definition"]))
    if view.get("note"):
        st.caption(str(view["note"]))

    scope = _artifact_scope(view, placement)
    choices = cell_choices(view)
    chosen = st.selectbox(
        "Cell", choices, index=0, key=f"viz_cell::{scope}",
        help=f"Default: {DEFAULT_CELL_RULE}.",
    ) if choices else None
    if chosen not in choices:
        chosen = choices[0] if choices else None
    zoomed = _zoom_choice(scope)
    try:
        panels = build_panels(
            view, cell_id=chosen, saved_metrics=view.get("saved_metrics"),
            badge=_figure_badge(view), include_zoom=zoomed,
        )
    except FigureDataError as exc:
        st.info(str(exc))
        return

    suffix = "_zoom" if zoomed else ""
    cell_panel = panels.get("soh_cycle" + suffix)
    duplicates = (cell_panel or {}).get("meta", {}).get("duplicates") or []
    left, right = st.columns(2)
    with left:
        if cell_panel is not None:
            _show(cell_panel)
    with right:
        _show(panels["parity" + suffix])
    left, right = st.columns(2)
    with left:
        _show(panels["residuals" + suffix])
    with right:
        st.caption(
            "SOH estimates across observed cycles: each point is an estimate made "
            "at that recorded cycle, joined in cycle order for one cell only. It is "
            "not a forecast; nothing is smoothed or interpolated."
        )
        if chosen is not None and choices and chosen == choices[0]:
            st.caption(f"Cell shown by default: {DEFAULT_CELL_RULE}.")
        if target.get("residual_note"):
            st.caption(str(target["residual_note"]))
        if zoomed:
            st.caption(
                f"Zoomed view: {panels['parity_zoom']['meta']['outside']:,} of "
                f"{int(view.get('count') or 0):,} point(s) lie outside the parity "
                "viewport; they are still in the data and in the full-range view."
            )
    if duplicates:
        st.warning(
            f"Cell {chosen} has repeated cycle records "
            + ", ".join(f"cycle {d['cycle_index']} ×{d['count']}" for d in duplicates)
            + ". They are drawn as stored, not averaged."
        )
    with st.expander("Per-cell error summary (display diagnostic)", expanded=False):
        st.caption(PER_CELL_NOTE)
        st.dataframe(per_cell_errors(view), hide_index=True, use_container_width=True)

    render_soh_range_diagnostics(view)
    _render_export_action(view, scope, cell_id=chosen)


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
        ("Outside [0, 1.0]", summary.get("out_of_range", 0)),
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
            "outside [0, 1.0] (below 0, or above the reference capacity)."
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
    st.markdown("#### Artifacts")
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
            render_rul_figures(preview["rul"], placement="predictions")
            # As above: the heading belongs only to an artifact that has the
            # curve.
            if preview["schema"] == RUL_CYCLEWISE_SCHEMA:
                render_rul_cycle_figure(preview.get("rul_cycles") or {})
            render_prediction_table(preview["rul"])
        elif preview["schema"] == SOH_SCHEMA and preview.get("soh"):
            render_soh_figures(preview["soh"], placement="predictions")
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
