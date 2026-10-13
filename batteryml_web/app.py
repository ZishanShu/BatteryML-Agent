from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st
import torch
import yaml

from batteryml_web.agent.failure_taxonomy import classify_llm_failure
from batteryml_web.agent.result_agent import explain_run_result
from batteryml_web.agent.result_analysis import (
    build_analysis_context,
    render_analysis_markdown,
    result_context_from_analysis,
)
from batteryml_web.agent.llm_client import run_llm_prompt, save_llm_output
from batteryml_web.agent.catalog import REQUEST_TEMPLATE_STRUCTURE
from batteryml_web.agent.config_explanation import (
    CONFIG_EXPLANATION_SYSTEM_PROMPT,
    build_config_explanation_prompt,
)
from batteryml_web.agent.execution import build_run_command
from batteryml_web.agent.executor import discover_artifacts
from batteryml_web.agent.results import build_result_context
from batteryml_web.ui.agent_workflow import (
    USE_TEMPLATE_LABEL,
    _git_commit,
    accept_template as accept_request_template,
    render_agent_workflow,
    request_template as agent_request_template,
)
from batteryml_web.core.custom_uploads import (
    ALLOWED_CUSTOM_SOURCE_FORMATS,
    build_uploaded_dataset_manifest,
    collect_raw_files,
    extract_if_archive,
    run_batteryml_preprocess_for_split,
    save_uploaded_files,
    validate_custom_upload_layout,
    validate_processed_output,
)
# Only the file discovery is still used here, and only ever against the
# selected run's own directory. load_metrics / load_predictions read "the
# first match" out of whatever list they are handed, which is what made the
# page show the oldest run in the project; Review Results now reads the
# selected run's manifest and its one recorded artifact instead.
from batteryml_web.core.results_reader import find_result_files
from batteryml_web.ui.feedback import (
    EVENT_PREPROCESS_DONE,
    EVENT_PROJECT_CREATED,
    EVENT_QUICKSTART_READY,
    EVENT_RUN_COMPLETED,
    notify_success,
    toast,
)
from batteryml_web.ui.llm_state import (
    connection_serial,
    describe_llm_status,
    ensure_llm_state_defaults,
    get_llm_status,
    is_llm_configured,
    llm_credentials,
    llm_settings_fingerprint,
    refresh_llm_status_from_config,
    set_llm_connection_result,
)
from batteryml_web.core.project_context import resolve_project
from batteryml_web.core.run_manager import get_run_manager
from batteryml_web.core import direct_runs
from batteryml_web.core.run_identity import new_run_id
from batteryml_web.core import run_manifest as rm
from batteryml_web.core.run_manifest import find_run
from batteryml_web.core import config_task, config_upload, quick_experience
from batteryml_web.ui.soh_check import render_soh_data_check
from batteryml_web.core.config_semantics import (
    load_config_semantics,
    model_display_name,
)
from batteryml_web.core.selected_result import (
    SelectedResult,
    cached_for,
    resolve_selected_result,
)
from batteryml_web.ui.run_history import (
    render_file_browser,
    render_predictions_panel,
    experiment_summary,
    render_run_detail,
    render_run_history,
    short_run_id,
    status_badge,
    render_run_location,
)
from batteryml_web.ui.settings_panel import (
    REVEAL_ACTION,
    REVEAL_CONFIGURE,
    llm_action_allowed,
    render_configure_in_settings_notice,
    render_requested_llm_settings,
    request_llm_settings,
)
from batteryml_web.ui.home_style import render_home_style, render_site_style
from batteryml_web.ui.progress_utils import (
    get_safe_runtime_env,
    run_command_with_progress,
    run_data_command_with_progress,
)
from batteryml_web.ui.scroll import render_scroll_to_top
from batteryml_web.ui.surfaces import (
    NAV_ROLE,
    PRIMARY_ROLE,
    SECONDARY_ROLE,
    command_viewport,
    log_viewport,
    render_info_card,
    render_surface_css,
)
from batteryml_web.ui.tab_navigation import (
    DATASET_OVERVIEW_TAB,
    PREPARE_TABS,
    VIEW_OVERVIEW_LABEL,
    render_prepare_tabs,
    request_tab,
)
from batteryml_web.ui.errors import error_boundary, render_copy_control
from batteryml_web.ui.workflow_sidebar import NAV_STEPS, get_current_step_index, render_sidebar_workflow, set_current_step
from batteryml_web.core.readiness import assess_dataset, list_dataset_readiness
from batteryml_web.core.config_registry import get_config_registry

st.set_page_config(
    page_title="BatteryML Protocol Platform",
    layout="wide",
    initial_sidebar_state="expanded",
)

APP_DIR = Path(__file__).resolve().parent
REPO_ROOT = APP_DIR.parent
DEFAULT_BASE_DIR = Path("runs") / "web_projects"
QUICKSTART_CONFIG = Path("configs/baselines/sklearn/xgb/matr_1.yaml")
DATASET_MANIFEST_PATH = REPO_ROOT / "data" / "metadata" / "dataset_manifest.json"
MODEL_MANIFEST_PATH = REPO_ROOT / "data" / "metadata" / "model_config_manifest.json"

for _path in (REPO_ROOT, APP_DIR):
    _str_path = str(_path)
    if _str_path not in sys.path:
        sys.path.insert(0, _str_path)


if "current_step" not in st.session_state:
    st.session_state["current_step"] = "home"
if "project_root" not in st.session_state:
    st.session_state["project_root"] = None
if "project_name" not in st.session_state:
    st.session_state["project_name"] = None
if "current_project" not in st.session_state:
    st.session_state["current_project"] = None
if "home_entry_defaulted" not in st.session_state:
    # Quick experience is the documented first path.  Showing its detail
    # panel on first load makes the default visible instead of leaving the
    # highlighted card looking like an unexplained selection.
    # Applied once per session: "← Choose another start path" clears
    # home_entry, and re-defaulting it on every rerun made that button a no-op.
    st.session_state.setdefault("home_entry", "quick")
    st.session_state["home_entry_defaulted"] = True
if "selected_dataset" not in st.session_state:
    st.session_state["selected_dataset"] = "MATR"
if "selected_task" not in st.session_state:
    st.session_state["selected_task"] = config_task.TASK_RUL
if "selected_model" not in st.session_state:
    st.session_state["selected_model"] = "XGBoost"
if "selected_config" not in st.session_state:
    st.session_state["selected_config"] = str(QUICKSTART_CONFIG)
if "selected_device" not in st.session_state:
    st.session_state["selected_device"] = "cpu"
if "last_run_status" not in st.session_state:
    st.session_state["last_run_status"] = "N/A"
if "last_run_time" not in st.session_state:
    st.session_state["last_run_time"] = 0.0
ensure_llm_state_defaults()
refresh_llm_status_from_config()

# Top chrome. Streamlit reserves a tall header band above the app; on a
# laptop that is most of the first screen before any content appears. This
# trims the block container's top padding and collapses the empty header,
# using Streamlit's stable data-testid selectors rather than generated class
# names.
#
# Scope and dependency, deliberately narrow:
#   * stAppViewBlockContainer / stAppViewContainer — padding only;
#   * stHeader — height only, and only because .streamlit/config.toml already
#     sets client.toolbarMode = "viewer", so the band holds nothing a user
#     needs. The sidebar toggle lives in stSidebar and is untouched.
# Nothing is hidden by display:none, so errors, toasts, status widgets and
# the sidebar control all remain reachable, including on small screens.
st.markdown(
    """
<style>
/* The band the page opens with. Both rules here were written once and both
   were quietly doing nothing; measured, the page still opened on 96px of
   padding under a 60px toolbar, which is why the title sat 160px down.

   `stMainBlockContainer` is the current name. `stAppViewBlockContainer` is
   what Streamlit called this element when the rule was written, and it no
   longer exists — a selector that matches nothing fails silently, which is
   exactly how 96px went unnoticed. Both names are listed so the rule works
   either side of that rename.

   2.5rem (40px) clears the 36px toolbar with a little air. It cannot go
   much below that: the toolbar is `position: absolute` with a z-index of
   999990, so content under 36px slides beneath the ⋮ menu rather than
   pushing it down. */
[data-testid="stMainBlockContainer"],
[data-testid="stAppViewBlockContainer"] {
    padding-top: 2.5rem;
}
[data-testid="stHeader"] {
    /* min-height as well as height, or this does nothing: Streamlit floors
       the band at 60px with its own min-height, and a height alone loses to
       it. This is the same pair the sidebar's forehead sets — that one works
       precisely because it sets both. */
    height: 2.25rem;
    min-height: 2.25rem;
    background: transparent;
}
</style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
<style>
:root {
    --app-radius: 14px;
    --app-border: rgba(128, 128, 128, 0.28);
    --app-border-soft: rgba(128, 128, 128, 0.18);
    --app-muted: rgba(128, 128, 128, 0.95);
    --app-card-bg: rgba(128, 128, 128, 0.055);
    --app-card-bg-strong: rgba(128, 128, 128, 0.095);
    /* No shadow variable any more. Elevation is an interaction cue — it
       says "this floats above the page and can be pressed" — and the only
       two users of the old `--app-shadow` were read-outs: the readiness
       tile and the pipeline tile. Information sits *in* the page. */
}

.llm-output-wrap {
    margin-top: 0.75rem;
    margin-bottom: 1.6rem;
}

/* The assistant's answer is a read-out too, and the loudest one on the
   page it appears on. Same geometry as every other information surface:
   4px, hairline, left rule, no shadow, not a control. */
.llm-output-card {
    width: 100%;
    border-radius: 4px;
    border: 1px solid var(--app-border-soft);
    border-left: 3px solid var(--app-border);
    padding: 1.15rem 1.25rem;
    background: var(--app-card-bg);
    box-sizing: border-box;
    cursor: default;
}

.llm-output-header {
    display: flex;
    align-items: flex-start;
    justify-content: space-between;
    gap: 1rem;
    margin-bottom: 0.9rem;
}

.llm-output-title {
    font-size: 1.02rem;
    font-weight: 700;
    line-height: 1.2;
}

.llm-output-subtitle {
    margin-top: 0.2rem;
    font-size: 0.9rem;
    color: var(--app-muted);
}

.llm-output-badges {
    display: flex;
    flex-wrap: wrap;
    justify-content: flex-end;
    gap: 0.45rem;
}

.llm-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
    padding: 0.22rem 0.55rem;
    border-radius: 999px;
    font-size: 0.78rem;
    font-weight: 700;
    border: 1px solid var(--app-border-soft);
    background: rgba(255,255,255,0.04);
    color: inherit;
}

.llm-badge.suggestion {
    opacity: 0.95;
}

.llm-badge.connected {
    border-color: rgba(55, 140, 80, 0.45);
}

.llm-badge.not-connected {
    border-color: rgba(190, 130, 40, 0.55);
}

.llm-output-body {
    line-height: 1.65;
    color: inherit;
}

.llm-output-section {
    padding-top: 0.65rem;
    margin-top: 0.65rem;
    border-top: 1px solid var(--app-border-soft);
}

.llm-output-section h4 {
    margin: 0 0 0.35rem 0;
    font-size: 0.9rem;
}

.llm-empty-state {
    padding: 0.9rem 0;
    line-height: 1.6;
}

.llm-empty-state ul {
    margin: 0.35rem 0 0 1.25rem;
}

.pipeline-wrapper {
    margin-top: 0.75rem;
    margin-bottom: 1.25rem;
}

/* The readiness tiles and the pipeline tiles are read-outs, and both used
   to be drawn as large rounded boxes — 18px and 20px radius, a 1.5px
   saturated border all the way round, and a drop shadow lifting them off
   the page. That is the visual vocabulary of something you press. Measured
   beside a button (1px border, 8px radius, no shadow) they were in fact
   *louder* than the controls they sat above, which is why a reader could
   not tell the readiness summary from "Select MATR".

   They now follow the same rule as `.bml-info-card` in ui/surfaces.py: a
   4px radius against a control's 8px, a neutral hairline, no shadow, and
   `cursor: default`. The readiness colour is not thrown away — it moves to
   a 3px rule on the left edge, the way a printed table marks a block —
   so the state is still legible at a glance, and the status word below
   keeps the same colour in bold for anyone who cannot see it. No button in
   this app has an asymmetric left edge, so the shape alone now says
   "read-out". */
.dataset-card {
    padding: 1rem 1.1rem;
    border-radius: 4px;
    border: 1px solid var(--app-border-soft);
    border-left: 3px solid var(--app-border);
    margin-bottom: 1rem;
    box-sizing: border-box;
    background: var(--app-card-bg);
    cursor: default;
}

.dataset-card.ready {
    border-left-color: rgba(55, 140, 80, 0.95);
}

.dataset-card.warning {
    border-left-color: rgba(190, 130, 40, 0.95);
}

.dataset-card.missing {
    border-left-color: rgba(190, 70, 70, 0.95);
}

/* 320px fixed, for a tile holding an icon and three short lines, spent a
   third of the Prepare Data screen on five boxes of mostly air. The height
   stays fixed — five tiles in a row must still line up, and letting them
   size to content would stagger them — but at the height the content
   actually needs. Measured from the parts: 2rem icon + 3.2rem title +
   4.6rem description + the status line + 2.4rem of padding ≈ 226px. */
.pipeline-card {
    height: 230px;
    min-height: 230px;
    max-height: 230px;
    border-radius: 4px;
    border: 1px solid var(--app-border-soft);
    border-left: 3px solid var(--app-border);
    padding: 1rem 0.9rem;
    box-sizing: border-box;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    align-items: center;
    text-align: center;
    background: var(--app-card-bg);
    cursor: default;
}

.pipeline-icon {
    font-size: 2rem;
    line-height: 1;
    margin-bottom: 0.5rem;
}

.pipeline-title {
    font-size: 1.05rem;
    font-weight: 700;
    line-height: 1.35;
    min-height: 3.2rem;
    display: flex;
    align-items: center;
    justify-content: center;
}

.pipeline-desc {
    font-size: 0.95rem;
    line-height: 1.45;
    opacity: 0.82;
    min-height: 4.6rem;
    display: flex;
    align-items: center;
    justify-content: center;
}

.pipeline-status {
    font-size: 0.95rem;
    font-weight: 700;
    color: rgb(46, 125, 50);
    margin-top: 0.5rem;
}
</style>
""",
    unsafe_allow_html=True,
)


def _project_root() -> Path | None:
    root = st.session_state.get("project_root")
    return Path(root) if root else None


def _ensure_project(project_name: str, base_dir: str, mode: str = "deterministic") -> Path:
    project_root = Path(base_dir) / (project_name.strip() or "my_battery_project")
    project_root.mkdir(parents=True, exist_ok=True)
    state = {"project_name": project_root.name, "mode": mode}
    (project_root / "project_state.json").write_text(json.dumps(state, indent=2), encoding="utf-8")
    st.session_state["project_root"] = str(project_root)
    st.session_state["project_name"] = project_root.name
    st.session_state["current_project"] = project_root.name
    return project_root


def _safe_projects() -> list[str]:
    if not DEFAULT_BASE_DIR.exists():
        return []
    return sorted([p.name for p in DEFAULT_BASE_DIR.iterdir() if p.is_dir()])


def _cmd_env() -> dict[str, str]:
    return get_safe_runtime_env()


def _run_cmd(command: list[str], cwd: Path | None = None) -> dict[str, Any]:
    start = time.perf_counter()
    proc = subprocess.run(command, cwd=str(cwd) if cwd else None, capture_output=True, text=True, env=_cmd_env())
    elapsed = time.perf_counter() - start
    return {"returncode": proc.returncode, "stdout": proc.stdout, "stderr": proc.stderr, "elapsed": elapsed}


def _batteryml_cli_command(*args: object) -> list[str]:
    """Build a CLI command that works from the Windows local checkout."""
    return [
        sys.executable,
        str(REPO_ROOT / "batteryml_web" / "core" / "run_batteryml_cli.py"),
        *(str(arg) for arg in args),
    ]


def _batteryml_downloadable_datasets() -> tuple[str, ...]:
    """Return public datasets supported by BatteryML's download command."""
    try:
        from batteryml.preprocess import DOWNLOAD_LINKS, is_download_complete

        return tuple(sorted(str(name) for name in DOWNLOAD_LINKS))
    except Exception:  # noqa: BLE001 - keep the page usable if optional imports fail
        return ("MATR",)


def _raw_download_progress(dataset: str, raw_path: Path) -> tuple[int, int, bool]:
    """Return ``(present, expected, complete)`` for a public raw download.

    Readiness deliberately treats a non-empty native directory as raw data so
    backend callers can report partial downloads.  The Quick controls need a
    slightly stricter answer: MATR is made of four named files, and a zero
    byte or missing file means Download must remain available so the user can
    finish the download before starting preprocessing.
    """
    try:
        from batteryml.preprocess import DOWNLOAD_LINKS, is_download_complete

        links = DOWNLOAD_LINKS.get(str(dataset).upper())
    except Exception:  # noqa: BLE001 - retain a usable UI if optional imports fail
        links = None

    if not links:
        files = [path for path in raw_path.rglob("*") if path.is_file()] if raw_path.is_dir() else []
        present = sum(1 for path in files if path.stat().st_size > 0)
        return present, present, bool(files and present == len(files))

    expected_names = []
    for item in links:
        try:
            expected_names.append(str(item[1]))
        except (IndexError, TypeError):
            continue
    if not expected_names:
        return 0, 0, False

    present = sum(
        1
        for name in expected_names
        if is_download_complete(raw_path / name)
    )
    return present, len(expected_names), present == len(expected_names)


def _refresh_data_manifests() -> tuple[bool, str]:
    """Refresh the local dataset/config catalog after a data action; ``(ok, detail)``."""
    return quick_experience.sync_manifests(REPO_ROOT, python=sys.executable, run=_run_cmd)


def _data_action_root() -> Path:
    """The selected project, or — with none selected — the default project.

    The same resolution Run Model and Review Results use (``resolve_project``),
    so a Quick experience run made without a project is listed by Review
    Results in any later session. It used to fall back to
    ``runs/web_projects/quick_experience``, a folder no other page looked in:
    a fresh session's Review Results opened on the empty default project and
    said "No run is selected" (server acceptance, PR #13).
    """
    root = _project_root() or (REPO_ROOT / _project_for_runs()["root"])
    return root if root.is_absolute() else REPO_ROOT / root


def _collect_run_outputs(project_root: Path, config_path: Path, log_path: Path, workspace_dir: Path) -> None:
    project_root.mkdir(parents=True, exist_ok=True)
    (project_root / "project_state.json").write_text(json.dumps({"project_name": project_root.name, "workspace_dir": str(workspace_dir)}, indent=2), encoding="utf-8")
    if config_path.exists():
        (project_root / "config.yaml").write_text(config_path.read_text(encoding="utf-8", errors="ignore"), encoding="utf-8")
    if log_path.exists():
        (project_root / "training_log.txt").write_text(log_path.read_text(encoding="utf-8", errors="ignore"), encoding="utf-8")
    for subdir in ["metrics", "predictions", "figures", "checkpoints", "reports"]:
        src = workspace_dir / subdir
        dst = project_root / subdir
        if src.exists():
            dst.mkdir(parents=True, exist_ok=True)
            for p in src.rglob("*"):
                if p.is_file():
                    target = dst / p.relative_to(src)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(p.read_bytes())
    for report_name in ["report.html", "report.pdf"]:
        for candidate in [workspace_dir / report_name, project_root / report_name]:
            if candidate.exists():
                (project_root / report_name).write_bytes(candidate.read_bytes())
                break


#: Session key for the run id the Run Model page shows before Run is pressed.
RUN_MODEL_RESERVED_ID_KEY = "run_model_reserved_run"
#: The Task the user last chose on Run Model. A plain key, not the radio's:
#: Streamlit drops a widget's state while another page is shown, and Home's
#: Quick panel rewrites ``selected_task`` on every render it draws.
RUN_MODEL_TASK_KEY = "run_model_task"


def _project_for_runs() -> dict[str, Any]:
    """The project a Run Model / demo run is recorded under.

    The same resolution Review Results uses (``_current_project``), so a run
    recorded here is always one Review Results looks for.
    """
    return resolve_project(
        project_root=st.session_state.get("project_root"),
        project_name=st.session_state.get("project_name")
        or st.session_state.get("current_project"),
    )


def _registry_entry_for_config(config_path: Any) -> Any:
    """The config registry entry for one config file, or None."""
    wanted = Path(str(config_path)).as_posix()
    for entry in get_config_registry(str(REPO_ROOT)).get_entries():
        if Path(str(entry.config_path)).as_posix() == wanted:
            return entry
    return None


def _run_model_workspace(dataset: Any, model: Any, config_path: Any) -> str:
    """The Run Model page's next run folder, repo-relative.

    Reserved before the click because the page shows the exact command,
    ``--workspace`` included, before Run is pressed; the reservation follows
    the current selection (see ``direct_runs.reserve_run_id``).
    """
    run_id = direct_runs.reserve_run_id(
        st.session_state,
        key=RUN_MODEL_RESERVED_ID_KEY,
        dataset=dataset,
        model=model,
        seed=st.session_state.get("run_seed", 0),
        device=st.session_state.get("selected_device", "cpu"),
        config_path=config_path,
    )
    return direct_runs.direct_run_dir(
        _project_for_runs()["root"], "run_model", run_id
    ).as_posix()


def _begin_page_run(
    *,
    run_id: str,
    workspace: str,
    project_name: str,
    dataset: str,
    task: str,
    model_key: str,
    model_display: Any,
    config_path: Any,
    seed: Any,
    device: str,
    command: list[str],
) -> Path:
    """Record the run (RUNNING) before BatteryML starts; return its folder."""
    run_dir = REPO_ROOT / workspace
    direct_runs.begin_direct_run(
        run_dir,
        workspace=workspace,
        run_id=run_id,
        project=project_name,
        dataset=dataset,
        task=task,
        model_key=model_key,
        model_display=model_display,
        config_path=config_path,
        seed=seed,
        device=device,
        command=command,
        git_commit=_git_commit(),
    )
    return run_dir


def _finish_page_run(
    run_dir: Path,
    result: dict[str, Any],
    *,
    workspace: str,
    dataset: str,
    task: str,
    model_key: str,
    config_path: Any,
) -> None:
    """Record how the run ended, with only what BatteryML actually saved.

    Artifacts and metrics are read exactly as Agent Workflow reads them
    (``discover_artifacts``, ``build_result_context``), so a run looks the
    same in Review Results whichever page started it.
    """
    artifacts: dict[str, Any] = {}
    metrics: dict[str, Any] = {}
    metrics_error: str | None = None
    if result["success"]:
        try:
            artifacts = discover_artifacts(workspace, repo_root=REPO_ROOT)
            context = build_result_context(
                workspace,
                dataset=dataset,
                task=task,
                model=model_key,
                config_path=str(config_path),
                run_status="SUCCESS",
                execution_result={
                    "success": True,
                    "returncode": result["return_code"],
                    "workspace": workspace,
                    "artifacts": artifacts,
                },
                smoke_run=False,
            )
            metrics = dict((context or {}).get("metrics") or {})
        except Exception as exc:  # noqa: BLE001 - recorded in run.json and shown, never swallowed
            traceback.print_exc()
            metrics_error = f"{type(exc).__name__}: {exc}"
            st.error(
                "The run finished, but its results could not be read "
                f"({metrics_error}). Its files are in {workspace}."
            )
    direct_runs.finish_direct_run(
        run_dir,
        return_code=result["return_code"],
        elapsed_seconds=result["elapsed"],
        output=str(result.get("output") or ""),
        artifacts=artifacts,
        metrics=metrics,
        metrics_error=metrics_error,
    )


def _current_project_name() -> str:
    return st.session_state.get("project_name") or st.session_state.get("current_project") or "No project loaded"


def _current_dataset_folder() -> Path:
    return Path("data/processed") / st.session_state.get("selected_dataset", "MATR")


def _current_raw_folder() -> Path:
    return Path("data/raw") / st.session_state.get("selected_dataset", "MATR")


def render_bottom_step_navigation():
    idx = get_current_step_index()
    prev_col, next_col = st.columns(2)
    with prev_col:
        if st.button("← Prev", type=NAV_ROLE, disabled=(idx == 0), use_container_width=True, key="bottom_prev_step"):
            set_current_step(NAV_STEPS[idx - 1]["key"])
    with next_col:
        if st.button("Next →", type=NAV_ROLE, disabled=(idx == len(NAV_STEPS) - 1), use_container_width=True, key="bottom_next_step"):
            set_current_step(NAV_STEPS[idx + 1]["key"])


def _render_project_created_notice() -> None:
    """Report the project just created: a toast, plus where it landed.

    Success is transient (``ui/feedback.py``): the confirmation is a toast,
    not a green box that outlives the action. The project name and workspace
    stay on the page as plain text, because they are information the user
    goes on to need, not a status message.
    """
    notice = st.session_state.get("project_created_notice")
    if not isinstance(notice, dict):
        return
    action = str(notice.get("action") or "created")
    message = (
        f'Project "{notice.get("name")}" created and loaded'
        if action == "created"
        else f'Project "{notice.get("name")}" loaded'
    )
    notify_success(
        message,
        event_key=EVENT_PROJECT_CREATED,
        signature=[action, notice.get("workspace")],
    )
    st.caption("Current project")
    st.write(f"**Project:** {notice.get('name')}")
    st.write("**Workspace:**")
    st.code(str(notice.get("workspace")), language="text")


def render_home_agent_entry() -> None:
    """The natural-language entry point, at the same level as Quick Start.

    Home holds no intent, readiness or planning logic: it records the
    request and navigates. Everything else — resolving the dataset and task,
    reading readiness, building and validating a plan — happens in the Agent
    Workflow page against the existing backend. Pressing this button never
    executes anything; execution still needs the explicit Run there.
    """
    st.subheader("Agent Workflow", anchor=False)
    st.caption(
        "Describe the task in plain language. The deterministic backend resolves "
        "the dataset, task and model, checks readiness and builds a plan; nothing "
        "runs until you press Run on the next page."
    )
    # The same request UX as the Agent Workflow page: one box, one template
    # helper, the same canonical request key. Two pages with two different
    # request affordances was the inconsistency this removes.
    template = agent_request_template()
    if st.session_state.pop("home_use_template", False):
        committed = str(st.session_state.get("home_agent_request", "") or "")
        if accept_request_template(template, committed):
            st.session_state["agent_original_request"] = template
            st.session_state["home_agent_request"] = template
    # Seeded through the widget key, not ``value=``: both on the same run make
    # Streamlit log a Session State warning (see render_agent_workflow).
    if "home_agent_request" not in st.session_state:
        st.session_state["home_agent_request"] = st.session_state.get("agent_original_request", "")
    request = st.text_area(
        "Describe what you want to do...",
        placeholder=REQUEST_TEMPLATE_STRUCTURE,
        key="home_agent_request",
        height=90,
    )
    if accept_request_template(template, request):
        if st.button(USE_TEMPLATE_LABEL, key="home_use_template_button"):
            st.session_state["home_use_template"] = True
            st.rerun()
    if st.button("Analyze with Agent", type=PRIMARY_ROLE, use_container_width=True, key="home_agent_analyze"):
        trimmed = str(request or "").strip()
        if not trimmed:
            st.warning(
                f"Enter a request first. The pattern is: {REQUEST_TEMPLATE_STRUCTURE} "
                f"— or press {USE_TEMPLATE_LABEL!r} to fill it in."
            )
        else:
            st.session_state["agent_original_request"] = trimmed
            # Consumed by render_agent_workflow(), which runs the backend
            # analysis on arrival. Analysis is not execution.
            st.session_state["agent_pending_request"] = trimmed
            for key in (
                "agent_workflow_context",
                "agent_selected_model",
                "agent_execution_plan",
                "agent_execution_result",
                "agent_result_context",
                "agent_explanation",
                "agent_intent",
                "agent_model_request",
                "agent_suggestions",
                "agent_model_choice",
                "agent_run_signature",
                # Run state for the previous request: why its results could
                # not be read, and that it had executed at all.
                "agent_result_error",
                "agent_has_executed",
                # The Agent Workflow page's own text box, so it shows this
                # request rather than one typed there on an earlier visit.
                # Safe to delete here: that widget is not rendered on Home.
                "agent_request_input",
            ):
                st.session_state.pop(key, None)
            set_current_step("agent_workflow")


def _render_home_create_project_panel() -> None:
    """Render the isolated new-project panel used by the Home tab."""
    st.subheader("Create New Project", anchor=False)
    st.caption("Create a workspace first. You can adjust data and model settings in the next steps.")
    project_name = st.text_input(
        "Project name",
        value=st.session_state.get("project_name") or "my_battery_project",
        key="home_project_name",
    )
    base_dir = st.text_input(
        "Projects base directory",
        value=str(DEFAULT_BASE_DIR),
        key="home_base_dir",
    )
    project_folder = Path(base_dir) / (project_name.strip() or "my_battery_project")
    st.caption("Project folder")
    st.code(str(project_folder), language="text")
    if st.button("Create project and continue", type=PRIMARY_ROLE, use_container_width=True, key="home_create_project"):
        created = _ensure_project(project_name, base_dir)
        st.session_state["project_created_notice"] = {
            "name": created.name,
            "workspace": created.as_posix(),
            "action": "created",
        }
        st.rerun()
    _render_project_created_notice()
    _render_home_entry_workflow("create")


def _render_home_existing_project_panel(projects: list[str]) -> None:
    """Render the isolated existing-project panel used by the Home tab."""
    st.subheader("Continue Existing Project", anchor=False)
    st.caption("Select a saved workspace to resume its latest data, run, or result state.")
    if projects:
        selected = st.selectbox("Select existing project", options=projects, key="home_existing_project")
        folder = DEFAULT_BASE_DIR / selected
        st.caption("Project folder")
        st.code(str(folder), language="text")
        if st.button("Load project and continue", type=PRIMARY_ROLE, use_container_width=True, key="home_load_project"):
            st.session_state["project_root"] = str(folder)
            st.session_state["project_name"] = selected
            st.session_state["current_project"] = selected
            st.session_state["project_created_notice"] = {
                "name": selected,
                "workspace": folder.as_posix(),
                "action": "loaded",
            }
            for key in (
                "agent_execution_plan",
                "agent_execution_result",
                "agent_result_context",
                "agent_run_context",
                "agent_run_signature",
                "agent_result_error",
                "agent_has_executed",
            ):
                st.session_state.pop(key, None)
            st.rerun()
    else:
        st.info("No existing project found yet. Create a project first, then it will appear here.")
        st.caption("Project folder")
        st.code("runs/web_projects/<project_name>", language="text")
    _render_home_entry_workflow("existing")


def _home_resolve_path(value: str | Path) -> Path:
    """Resolve a project-relative or repository-relative path for the UI."""
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def _home_display_path(value: str | Path) -> str:
    path = _home_resolve_path(value)
    try:
        return path.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(path)


def _quick_readiness(dataset: str) -> dict[str, Any]:
    """Read one dataset's real local state for the guided home flow."""
    return assess_dataset(
        dataset,
        raw_root=REPO_ROOT / "data" / "raw",
        processed_root=REPO_ROOT / "data" / "processed",
    ).to_dict()


def _quick_config_entries(dataset: str, task: str) -> list[Any]:
    try:
        return get_config_registry(str(REPO_ROOT)).get_entries(dataset, task)
    except Exception:  # noqa: BLE001 - an empty selector is safer than a crash
        return []


def _quick_default_dataset() -> str:
    """The dataset of the page's default protocol (``QUICKSTART_CONFIG``): MATR."""
    dataset = load_config_semantics(_home_resolve_path(QUICKSTART_CONFIG)).dataset
    if not dataset:
        raise ValueError(f"{QUICKSTART_CONFIG} names no dataset; Quick experience has no default.")
    return str(dataset)


def _quick_dataset_choices(default: str) -> list[quick_experience.DatasetChoice]:
    """Every dataset the readiness layer knows, with the tasks the registry can run on it.

    No list of names lives here: the datasets come from
    ``readiness.list_dataset_readiness`` (platform datasets plus anything on
    disk) and the runnable tasks from the config registry. A dataset the
    registry has configs for but readiness did not list is assessed too.
    """
    raw_root, processed_root = REPO_ROOT / "data" / "raw", REPO_ROOT / "data" / "processed"
    statuses = {
        report.dataset: report.status.value
        for report in list_dataset_readiness(raw_root=raw_root, processed_root=processed_root)
    }
    pairs = [(entry.dataset, entry.task) for entry in get_config_registry(str(REPO_ROOT)).entries]
    for name in sorted({str(dataset) for dataset, _task in pairs if dataset} - set(statuses)):
        statuses[name] = assess_dataset(name, raw_root=raw_root, processed_root=processed_root).status.value
    return quick_experience.dataset_choices(statuses, pairs, default=default)


def _md_text(text: Any) -> str:
    """``text`` safe inside Streamlit markdown: ``*.pkl`` stays literal, ``$`` is not maths."""
    return re.sub(r"([\\`*_$\[\]])", r"\\\1", str(text))


def _quick_failure_reason(result: dict[str, Any], *, dataset: str) -> str:
    """Turn a command result into a concise, specific result-page message."""
    code = result.get("return_code")
    tail = str(result.get("output_tail") or "").strip()
    lower_tail = tail.lower()
    if "truncated file" in lower_tail or "incomplete or corrupt" in lower_tail:
        return (
            f"{dataset} 原始文件不完整或已损坏。请删除日志中指出的文件，"
            "重新下载并保持文件名不变，然后再次处理。"
        )
    # The host the failing download itself named — never an assumed one.
    failed_host = quick_experience.download_failure_host(tail)
    if failed_host and any(
        marker in lower_tail
        for marker in ("connectionerror", "winerror 10013", "max retries", "unable to connect")
    ):
        return (
            f"无法连接 {failed_host}。请检查 Windows 防火墙/网络策略，"
            "或配置 HTTPS_PROXY 后重试；未完成的临时文件已清理。"
        )
    lines = [line.strip() for line in tail.splitlines() if line.strip()]
    detail = " ".join(lines[-3:])
    if detail:
        return f"BatteryML exited with code {code}. {detail}"
    return f"BatteryML exited with code {code}. See the run log for details."


def _run_home_data_action(
    *,
    dataset: str,
    action: str,
    raw_path: Path,
    processed_path: Path,
    progress_host: Any,
) -> dict[str, Any]:
    """Run the official BatteryML data command and persist its local log."""
    workspace = _data_action_root()
    command = (
        _batteryml_cli_command("download", dataset, str(raw_path))
        if action == "download"
        else _batteryml_cli_command("preprocess", dataset, str(raw_path), str(processed_path))
    )
    log_path = workspace / "logs" / f"{action}_{dataset.lower()}.log"
    return run_data_command_with_progress(
        command=command,
        cwd=REPO_ROOT,
        log_path=log_path,
        steps=[
            (5, "Initialize data action"),
            (20, "Check local paths"),
            (55, "Run BatteryML command"),
            (90, "Write data outputs"),
            (100, "Data action completed"),
        ],
        container=progress_host,
    )


def _run_home_experiment(
    *,
    project_root: Path | None,
    plan: quick_experience.QuickPlan,
    demo_progress_host: Any,
) -> tuple[dict[str, Any], Path, Path, str]:
    """Run Quick experience once, recorded exactly like every other page run.

    ``plan`` is the validated selection (``quick_experience.build_quick_plan``,
    re-checked by the caller immediately before this call); an invalid plan is
    refused here as well, so nothing unvalidated can start.

    The run gets its own folder, ``<project>/quickstart/<run_id>/``, which is
    also BatteryML's ``--workspace`` (``direct_runs``), so its run.json sits
    beside its predictions. RUNNING is written before BatteryML starts, and
    artifacts and metrics are read afterwards the way Agent Workflow reads
    them (``_finish_page_run``), so Review Results opens this run like any
    other. Returns ``(result, run_dir, workspace_dir, run_id)``; the run
    folder and the workspace are the same directory.
    """
    if not plan.valid:
        raise ValueError("Quick experience refused an invalid plan: " + "; ".join(plan.errors))
    workspace_root = project_root or _data_action_root()
    workspace_root.mkdir(parents=True, exist_ok=True)
    dataset = str(plan.dataset)
    task = str(plan.task)
    model_key = str(plan.model_key or "model")
    config_path = Path(str(plan.config_path))
    run_id = new_run_id(dataset=dataset, model=model_key, seed=0)
    run_dir = direct_runs.direct_run_dir(workspace_root, "quickstart", run_id)
    try:
        workspace = Path(os.path.relpath(run_dir, REPO_ROOT)).as_posix()
    except ValueError:  # another drive on Windows: keep the absolute path
        workspace = run_dir.as_posix()
    command = quick_experience.plan_command(plan, workspace=workspace, seed=0)
    _begin_page_run(
        run_id=run_id,
        workspace=workspace,
        project_name=workspace_root.name,
        dataset=dataset,
        task=task,
        model_key=model_key,
        model_display=plan.model_display,
        config_path=config_path,
        seed=0,
        device=str(plan.device),
        command=command,
    )
    result = run_command_with_progress(
        command=command,
        cwd=REPO_ROOT,
        log_path=run_dir / "run.log",
        steps=[
            (5, "Initialize run"),
            (15, "Check processed data"),
            (30, "Load configuration"),
            (45, "Start BatteryML training"),
            (80, "Training and evaluation"),
            (95, "Collect outputs"),
            (100, "Run completed"),
        ],
        container=demo_progress_host,
        live_stages=True,
    )
    _finish_page_run(
        run_dir,
        result,
        workspace=workspace,
        dataset=dataset,
        task=task,
        model_key=model_key,
        config_path=config_path,
    )
    if not result.get("success"):
        # The concrete reason the page shows, kept in the run record too.
        rm.update_manifest(
            run_dir,
            {"failure": {"category": "execution_failed",
                         "message": _quick_failure_reason(result, dataset=dataset)}},
        )
    result["command"] = command
    return result, run_dir, run_dir, run_id


def _quick_uploaded_configs() -> list[dict[str, Any]]:
    """Validated uploads of this session that are still where they were stored."""
    root = _data_action_root()
    return [
        item for item in st.session_state.get("quick_uploaded_configs") or []
        if config_upload.is_managed_upload(Path(str(item.get("path"))), root)
    ]


def _render_home_entry_workflow(entry: str) -> None:
    """Give Create and Existing projects the same four-step mental model."""
    st.markdown("#### Workflow steps")
    st.caption("Use the same four steps after the workspace is ready. Each step opens the full editor.")
    steps = (
        ("1 · Prepare data", "Download, upload, preprocess, and inspect the dataset.", "Open Prepare Data", "prepare_data"),
        ("2 · Configure model", "Choose the task, prediction model, protocol, and device.", "Open Run Model", "run_model"),
        ("3 · Run training", "Start a reproducible run and follow the live log.", "Open Run Model", "run_model"),
        ("4 · Review results", "Inspect metrics, predictions, logs, and artifacts.", "Open Review Results", "review_results"),
    )
    for index, (title, description, button_label, target) in enumerate(steps):
        with st.expander(title, expanded=index == 0):
            st.caption(description)
            if st.button(button_label, type=NAV_ROLE, use_container_width=True, key=f"home_{entry}_workflow_{index}"):
                set_current_step(target)


def _render_home_quickstart_panel(project_root: Path | None) -> None:
    """Render Quick experience as the same four-step guided workflow."""
    st.subheader("Quick experience", anchor=False)
    st.caption("A guided local run with MATR as the default dataset. You can edit the configuration before running.")

    default_dataset = _quick_default_dataset()
    if "quick_dataset" not in st.session_state:
        st.session_state["quick_dataset"] = default_dataset
    if "quick_task_selector" not in st.session_state:
        st.session_state["quick_task_selector"] = "RUL prediction"
    if "quick_device_selector" not in st.session_state:
        st.session_state["quick_device_selector"] = "cpu"
    if "quick_config_selector" not in st.session_state:
        st.session_state["quick_config_selector"] = str(_home_resolve_path(QUICKSTART_CONFIG))
    if "quick_model_selector" not in st.session_state:
        # First visit: the model of the default protocol (MATR · XGBoost), as before.
        st.session_state["quick_model_selector"] = load_config_semantics(
            _home_resolve_path(QUICKSTART_CONFIG)
        ).model_key
    pending_upload = st.session_state.pop("quick_pending_upload", None)
    if pending_upload:
        st.session_state["quick_task_selector"] = pending_upload["task"]
        st.session_state["quick_model_selector"] = pending_upload["model_key"]
        st.session_state["quick_config_selector"] = pending_upload["path"]
        st.session_state["quick_uploaded_notice"] = pending_upload["path"]

    st.markdown("#### 1 · Prepare data")
    # Every dataset the platform knows is offered (readiness layer), with the
    # tasks the config registry can run on each; the default protocol's
    # dataset (MATR) comes first and stays the default. A dataset without a
    # registered config is listed and explained, never hidden, and nothing is
    # downloaded, processed or run by selecting one.
    choices = {choice.name: choice for choice in _quick_dataset_choices(default_dataset)}
    if st.session_state.get("quick_dataset_selector") not in choices:
        st.session_state["quick_dataset_selector"] = default_dataset if default_dataset in choices else next(iter(choices))
    dataset = st.selectbox(
        "Raw dataset",
        list(choices),
        format_func=lambda name: name if choices[name].runnable else f"{name} · no run configuration",
        key="quick_dataset_selector",
        help="Quick experience defaults to MATR. Choose the raw dataset before downloading or preprocessing.",
    )
    choice = choices[dataset]
    st.session_state["quick_dataset"] = dataset
    readiness = _quick_readiness(dataset)
    raw_path = _home_resolve_path(readiness["raw_path"])
    processed_path = _home_resolve_path(readiness["processed_path"])

    if dataset == default_dataset:
        st.caption(f"Quick experience uses **{dataset}** by default. Download raw files first, then create processed records.")
    else:
        st.caption(f"Quick experience will use **{dataset}**.")
    status = str(readiness["status"])
    downloadable = dataset.upper() in {name.upper() for name in _batteryml_downloadable_datasets()}
    raw_present = bool(readiness.get("raw_available"))
    raw_count, raw_expected, raw_complete = _raw_download_progress(dataset, raw_path)
    raw_ready = raw_present and raw_complete
    processed_ready = status == quick_experience.STATUS_READY
    status_cols = st.columns(2, gap="medium")
    with status_cols[0]:
        st.markdown("**Raw data**")
        if raw_ready:
            st.success("Available")
        elif raw_present and raw_expected:
            st.warning(f"Download incomplete ({raw_count}/{raw_expected} files)")
        else:
            st.info(f"{dataset} not downloaded yet" if downloadable else "No raw files found")
        st.caption(_home_display_path(raw_path))
    with status_cols[1]:
        st.markdown("**Processed data**")
        if processed_ready:
            st.success("Ready")
        elif status == quick_experience.STATUS_INCOMPLETE:
            st.warning("Incomplete — no usable processed files")
        else:
            st.warning("Needs processing")
        st.caption(_home_display_path(processed_path))
    # The readiness layer's own verdict and reason, word for word.
    st.caption(f"Readiness: **{status}** — {_md_text(readiness['reason'])}")

    limitation = quick_experience.quick_limitation(choice)
    if limitation:
        # Data actions for this dataset live in Prepare Data, which handles
        # every dataset; offering them here would suggest a run follows.
        st.info(limitation)
        if st.button("Open Prepare Data", type=NAV_ROLE, key="home_quick_open_prepare"):
            set_current_step("prepare_data")
    else:
        download = quick_experience.download_action(
            dataset=dataset, downloadable=downloadable, raw_complete=raw_complete,
            raw_count=raw_count, raw_expected=raw_expected,
        )
        process = quick_experience.process_action(
            status=status, raw_ready=raw_ready,
            preprocessing_supported=bool(readiness.get("preprocessing_supported")),
        )
        data_action_surface = st.container()
        data_cols = st.columns(2, gap="medium")
        with data_cols[0]:
            if download.state == quick_experience.ACTION_COMPLETED:
                # Done is a fact, not a control: no button is drawn for it.
                st.caption(f":green[✓] {download.note}")
            else:
                download_clicked = st.button(
                    f"Download {dataset} data",
                    type=SECONDARY_ROLE,
                    use_container_width=True,
                    key="home_quick_download_data",
                    disabled=download.state != quick_experience.ACTION_AVAILABLE,
                )
                if download.note:
                    st.caption(download.note)
                if download_clicked:
                    result = _run_home_data_action(
                        dataset=dataset,
                        action="download",
                        raw_path=raw_path,
                        processed_path=processed_path,
                        progress_host=data_action_surface,
                    )
                    st.session_state["quick_data_action_error"] = None if result["success"] else _quick_failure_reason(result, dataset=dataset)
                    if result["success"]:
                        synced, sync_detail = _refresh_data_manifests()
                        st.session_state["quick_manifest_sync_error"] = None if synced else sync_detail
                        notify_success("Raw data downloaded", event_key=EVENT_PREPROCESS_DONE, signature=[dataset, "download"])
                        st.rerun()
        with data_cols[1]:
            if process.state == quick_experience.ACTION_COMPLETED:
                st.caption(f":green[✓] {process.note}")
            else:
                process_clicked = st.button(
                    f"Process {dataset} data",
                    type=PRIMARY_ROLE,
                    use_container_width=True,
                    key="home_quick_process_data",
                    disabled=process.state != quick_experience.ACTION_AVAILABLE,
                )
                if process.note:
                    st.caption(process.note)
                if process_clicked:
                    result = _run_home_data_action(
                        dataset=dataset,
                        action="preprocess",
                        raw_path=raw_path,
                        processed_path=processed_path,
                        progress_host=data_action_surface,
                    )
                    st.session_state["quick_data_action_error"] = None if result["success"] else _quick_failure_reason(result, dataset=dataset)
                    if result["success"]:
                        synced, sync_detail = _refresh_data_manifests()
                        st.session_state["quick_manifest_sync_error"] = None if synced else sync_detail
                        notify_success("Processed data is ready", event_key=EVENT_PREPROCESS_DONE, signature=[dataset, "preprocess"])
                        st.rerun()
    if st.session_state.get("quick_data_action_error"):
        st.error(str(st.session_state["quick_data_action_error"]))
    if st.session_state.get("quick_manifest_sync_error"):
        st.error(
            "The data action finished, but the dataset and configuration list could not be "
            f"refreshed: {st.session_state['quick_manifest_sync_error']}"
        )

    st.markdown("#### 2 · Configure model")
    # Dataset -> task -> model -> config, never the other way round: every
    # option below is recomputed from the selection above it, and a choice the
    # new options do not contain is cleared before its widget is drawn (with a
    # note saying so), so nothing from the previous dataset, task or model can
    # stay selected.
    task_options = quick_experience.task_options(choice)
    task_selected, task_cleared = quick_experience.reconcile_selection(
        st.session_state.get("quick_task_selector"), task_options, default=config_task.TASK_RUL,
    )
    st.session_state["quick_task_selector"] = task_selected
    task_choice = st.selectbox("Prediction task", task_options, key="quick_task_selector")
    if task_cleared:
        st.caption(f"{dataset} has no registered {task_cleared} configuration, so the task is now {task_choice}.")
    entries = _quick_config_entries(dataset, task_choice)
    config_options = [
        quick_experience.ConfigOption(str(_home_resolve_path(entry.config_path)), entry.task, entry.model_key)
        for entry in entries
    ] + [
        quick_experience.ConfigOption(item["path"], item["task"], item["model_key"])
        for item in _quick_uploaded_configs()
        if item.get("dataset") == dataset
    ]
    model_options = list(dict.fromkeys(
        option.model_key for option in config_options
        if option.task == task_choice and option.model_key
    ))
    if not model_options:
        st.error(f"No {task_choice} configuration is registered for {dataset}.")
    model_selected, model_cleared = quick_experience.reconcile_selection(
        st.session_state.get("quick_model_selector"), model_options,
    )
    st.session_state["quick_model_selector"] = model_selected
    model_key = st.selectbox(
        "Prediction model",
        model_options,
        format_func=lambda value: next((entry.model_display for entry in entries if entry.model_key == value), model_display_name(value) or value),
        key="quick_model_selector",
    )
    if model_cleared and model_options:
        st.caption(
            f"{model_display_name(model_cleared) or model_cleared} has no {task_choice} configuration "
            f"for {dataset}, so it is no longer selected."
        )
    config_paths = quick_experience.config_choices(config_options, task=task_choice, model_key=model_key)
    selected_config, cleared_config = quick_experience.reconcile_config(
        st.session_state.get("quick_config_selector"),
        config_paths,
        # The page's default protocol (MATR primary split) when this model has it.
        default=next((path for path in config_paths if Path(path).stem == QUICKSTART_CONFIG.stem), None),
    )
    st.session_state["quick_config_selector"] = selected_config
    config_choice = st.selectbox(
        "Protocol configuration",
        config_paths,
        format_func=lambda value: f"{Path(value).stem} · {_home_display_path(value)}",
        key="quick_config_selector",
    )
    if cleared_config:
        st.caption(
            f"{_home_display_path(cleared_config)} is not a {dataset} · {task_choice} configuration "
            "for this model, so it is no longer selected."
        )
    config_path = _home_resolve_path(config_choice) if config_choice else None
    cuda_ok, cuda_detail = quick_experience.cuda_available()
    device = st.selectbox(
        "Compute device",
        list(quick_experience.DEVICE_OPTIONS),
        format_func=lambda value: value if value == quick_experience.CPU or cuda_ok else f"{value} (not available on this machine)",
        key="quick_device_selector",
    )
    device_issue = quick_experience.device_problem(device, cuda_ok=cuda_ok, cuda_detail=cuda_detail)
    if device_issue:
        st.warning(device_issue)

    uploaded = st.file_uploader("Upload YAML configuration (optional)", type=["yaml", "yml"], key="quick_config_upload")
    if uploaded is not None and st.button("Apply uploaded configuration", type=NAV_ROLE, key="quick_apply_config"):
        # Applying selects the config after validation; it never starts a run.
        outcome = config_upload.stage_uploaded_config(
            uploaded.getvalue(),
            uploaded.name,
            managed_root=_data_action_root(),
            repo_root=REPO_ROOT,
            allowed_datasets=[dataset],
            allowed_tasks=task_options,
        )
        if outcome.ok:
            record = {"path": str(outcome.path), "task": outcome.task, "model_key": outcome.model_key,
                      "dataset": outcome.dataset}
            st.session_state.setdefault("quick_uploaded_configs", []).append(record)
            # Widgets above are already drawn: the selection is applied at the
            # top of the next run, before they are instantiated.
            st.session_state["quick_pending_upload"] = record
            st.session_state.pop("quick_upload_error", None)
            st.rerun()
        else:
            st.session_state["quick_upload_error"] = list(outcome.errors)
    if st.session_state.get("quick_upload_error"):
        st.error(
            "The uploaded configuration was not applied; the previous selection is unchanged.\n\n"
            + "\n".join(f"- {problem}" for problem in st.session_state["quick_upload_error"])
        )

    plan_inputs = dict(
        task=task_choice,
        config_path=config_path,
        device=device,
        repo_root=REPO_ROOT,
        # Only the dataset on screen: a config of another dataset never runs here.
        allowed_datasets=[dataset],
        cuda_ok=cuda_ok,
        cuda_detail=cuda_detail,
        uploaded=bool(config_path) and config_upload.is_managed_upload(config_path, _data_action_root()),
    )
    plan = quick_experience.build_quick_plan(**plan_inputs, require_data=True)
    semantics = load_config_semantics(config_path) if config_path else None
    display_dataset = (semantics.dataset if semantics else None) or dataset
    display_task = task_choice
    display_model = (semantics.model_display if semantics else None) or model_display_name(model_key) or model_key or "none"
    st.info(
        f"This configuration means: **Dataset** {display_dataset} · **Task** {display_task} · "
        f"**Prediction model** {display_model} · **Device** {device} · "
        f"**Config file** {config_path.name if config_path else 'not selected'}"
    )
    if st.session_state.get("quick_uploaded_notice") and str(config_path) == st.session_state["quick_uploaded_notice"]:
        st.caption(f"Uploaded configuration: {_home_display_path(st.session_state['quick_uploaded_notice'])}")
    st.session_state["quick_dataset"] = display_dataset
    st.session_state["selected_dataset"] = display_dataset
    st.session_state["selected_task"] = display_task
    st.session_state["selected_model"] = (semantics.model_key if semantics else None) or model_key
    st.session_state["selected_config"] = str(config_path) if config_path else ""
    st.session_state["selected_device"] = device

    st.markdown("#### 3 · Run training")
    workspace_root = project_root or _data_action_root()
    st.write("**Run location**")
    if project_root is None:
        st.caption("No project is selected. Quick experience will use this local workspace:")
    st.code(str(workspace_root), language="text")
    run_ready = processed_ready and plan.valid
    if not processed_ready:
        st.warning("Process the dataset in step 1 before starting a model run.")
    else:
        # SOH also needs cells that can carry an SOH label (see soh_gate).
        run_ready = render_soh_data_check(display_dataset, task_choice, config_path, key="quick", repo_root=REPO_ROOT) and run_ready
    # Data readiness is reported once, by the line above; every other reason
    # the Run button is disabled is listed here.
    shown_plan = plan if processed_ready else quick_experience.build_quick_plan(**plan_inputs, require_data=False)
    for problem in shown_plan.errors:
        if problem != device_issue:
            st.error(problem)
    for problem in st.session_state.pop("quick_run_blocked", None) or []:
        st.error(f"The run was not started: {problem}")
    demo_progress_host = st.container()
    if st.button(
        "Run experiment",
        type=PRIMARY_ROLE,
        use_container_width=True,
        key="home_quick_run_experiment",
        disabled=not run_ready,
    ):
        # Re-validated from disk immediately before launch: what runs is what
        # was checked, even if the file or the machine changed since drawing.
        final_plan = quick_experience.build_quick_plan(**plan_inputs, require_data=True)
        if not final_plan.valid:
            st.session_state["quick_run_blocked"] = list(final_plan.errors)
            st.rerun()
        result, run_dir, workspace_dir, run_id = _run_home_experiment(
            project_root=project_root,
            plan=final_plan,
            demo_progress_host=demo_progress_host,
        )
        metrics, metrics_problem = quick_experience.run_metrics(run_dir)
        st.session_state["last_run_status"] = "success" if result["success"] else "failed"
        st.session_state["last_run_time"] = result["elapsed"]
        st.session_state["last_run_code"] = result["return_code"]
        st.session_state["last_run_error"] = None if result["success"] else _quick_failure_reason(result, dataset=str(final_plan.dataset))
        st.session_state["last_run_log_path"] = str(run_dir / "run.log")
        st.session_state["last_run_command"] = " ".join(str(part) for part in result.get("command") or [])
        st.session_state["last_run_id"] = run_id
        st.session_state["last_run_metrics"] = metrics if result["success"] else {}
        st.session_state["last_run_metrics_problem"] = metrics_problem if result["success"] else None
        st.session_state["project_root"] = str(workspace_root)
        st.session_state["project_name"] = workspace_root.name
        st.session_state["current_project"] = workspace_root.name
        st.session_state["workspace_dir"] = str(workspace_dir)
        if result["success"]:
            notify_success("Quick experience run completed", event_key=EVENT_RUN_COMPLETED, signature=[run_id, result["return_code"]])
        st.rerun()

    st.markdown("#### 4 · Review results")
    last_status = str(st.session_state.get("last_run_status") or "N/A")
    if last_status == "success":
        st.success("The latest run completed. Key metrics are shown here; open Review Results for the full analysis.")
        metrics = st.session_state.get("last_run_metrics") or {}
        if metrics:
            metric_cols = st.columns(min(max(len(metrics), 1), 4), gap="medium")
            for column, (name, value) in zip(metric_cols, metrics.items()):
                with column:
                    st.metric(str(name).upper(), f"{float(value):.4f}")
            st.caption(f"Run {st.session_state.get('last_run_id') or 'latest'} · metrics recorded by BatteryML")
        elif st.session_state.get("last_run_metrics_problem"):
            st.warning(f"The run completed, but its metrics could not be read: {st.session_state['last_run_metrics_problem']}")
        else:
            st.info("The run completed, but BatteryML did not save evaluation metrics for this run.")
    elif last_status == "failed":
        st.error(f"The latest run failed: {st.session_state.get('last_run_error') or 'See the run log for the specific error.'}")
    else:
        st.caption("After a run finishes, its metrics, predictions, and logs will appear here.")
    if st.button("Open Review Results", type=NAV_ROLE, use_container_width=True, key="home_quick_open_review"):
        # Navigation only. LLM Settings opens from Review Results when the
        # user asks for a summary or explanation without a connection (UI-14).
        set_current_step("review_results")


def render_home_page():
    render_home_style()
    st.markdown(
        """
        <section class="bml-home-hero">
            <div>
                <div class="bml-home-kicker">Battery intelligence workspace</div>
                <h1 class="bml-home-title">Battery<span>ML</span> workspace</h1>
                <p class="bml-home-subtitle">
                    Prepare data, launch reproducible experiments, and review battery health
                    predictions from one focused workspace.
                </p>
            </div>
            <div class="bml-home-live">Local workspace ready</div>
        </section>
        """,
        unsafe_allow_html=True,
    )

    project_root = _project_root()
    projects = _safe_projects()
    st.subheader("How would you like to start?", anchor=False)
    st.caption("Choose one path to open its workspace. You can switch paths at any time.")
    choice_cols = st.columns(3, gap="medium")
    choices = (
        ("quick", "⚡  Quick experience\n\nUse the built-in MATR demo\nNo API key required"),
        ("create", "✦  Create new project\n\nStart with your own data\nConfigure every step"),
        ("existing", "↗  Existing project\n\nReturn to a saved workspace\nResume where you left off"),
    )
    for col, (choice_key, label) in zip(choice_cols, choices):
        with col:
            with st.container(key=f"home_choice_{choice_key}"):
                if st.button(label, use_container_width=True, key=f"home_choice_button_{choice_key}"):
                    st.session_state["home_entry"] = choice_key
                    st.rerun()

    # Keep the canonical workflow visible before any optional entry details.
    # This gives first-time visitors a stable mental model of the product.
    st.markdown(
        """
        <section class="bml-process-overview">
            <div class="bml-process-heading">
                <h2>Your BatteryML workflow</h2>
                <span>Start simple, customize any step later</span>
            </div>
            <div class="bml-process-steps">
                <div class="bml-process-step"><span class="bml-step-no">1</span><strong>Prepare data</strong><small>Check files, clean records, and inspect features.</small></div>
                <div class="bml-process-step"><span class="bml-step-no">2</span><strong>Configure model</strong><small>Choose the task, model, device, and parameters.</small></div>
                <div class="bml-process-step"><span class="bml-step-no">3</span><strong>Run training</strong><small>Launch a reproducible run and follow its progress.</small></div>
                <div class="bml-process-step"><span class="bml-step-no">4</span><strong>Review results</strong><small>Compare metrics, predictions, and generated artifacts.</small></div>
            </div>
        </section>
        """,
        unsafe_allow_html=True,
    )

    home_entry = st.session_state.get("home_entry")
    if home_entry:
        st.markdown(
            f"""
            <style>
            [class*="st-key-home_choice_{home_entry}"] [data-testid="stBaseButton-secondary"] {{
                border-color: rgba(40,123,245,0.7) !important;
                background: linear-gradient(145deg, #eaf4ff, #ffffff) !important;
                box-shadow: 0 18px 36px rgba(40,123,245,0.18) !important;
                transform: translateY(-4px) !important;
            }}
            </style>
            """,
            unsafe_allow_html=True,
        )
        entry_titles = {"quick": "Quick experience", "create": "Create new project", "existing": "Existing project"}
        st.markdown(
            f"""
            <div class="bml-detail-context">
                <span class="bml-detail-eyebrow">Selected path</span>
                <strong>{entry_titles.get(home_entry, "Start")}</strong>
                <span class="bml-detail-hint">Workspace setup</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if home_entry == "quick":
            _render_home_quickstart_panel(project_root)
        elif home_entry == "create":
            # Quick Start keeps its historical heading contract: st.subheader("Quick Start", anchor=False)
            # The create panel owns the same filesystem operation: created = _ensure_project(project_name, base_dir)
            # It preserves the stored notice payload: "workspace": created.as_posix()
            # The confirmation remains rendered through _render_project_created_notice().
            _render_home_create_project_panel()
        elif home_entry == "existing":
            _render_home_existing_project_panel(projects)
        if st.button("← Choose another start path", type=NAV_ROLE, key="home_choose_another"):
            st.session_state.pop("home_entry", None)
            st.rerun()

    # The rule belongs to the notice, not to the page: with an assistant
    # configured the notice draws nothing, and an unconditional rule here
    # left two horizontal lines stacked against each other.
    if not is_llm_configured():
        st.markdown("---")
        render_llm_availability_notice()

    st.markdown("---")
    render_home_agent_entry()

    st.markdown("---")
    if st.button("Start workflow →", type=NAV_ROLE, use_container_width=True, key="home_start_workflow"):
        set_current_step("prepare_data")


def _load_json_manifest(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except Exception:
        return []


def _dataset_manifest_entries() -> list[dict[str, Any]]:
    return [x for x in _load_json_manifest(DATASET_MANIFEST_PATH) if x.get("status") == "present"]


def _selected_project_dir() -> Path:
    return _project_root() or Path("runs/web_projects/default")


def _model_manifest_entries() -> list[dict[str, Any]]:
    return [x for x in _load_json_manifest(MODEL_MANIFEST_PATH) if x.get("exists", True)]


def _dataset_label(entry: dict[str, Any]) -> str:
    parts = [entry.get("source"), entry.get("battery_family"), entry.get("size")]
    suffix = ", ".join([str(p) for p in parts[1:] if p])
    return f"{entry.get('name')} [{entry.get('source')}{', ' + suffix if suffix else ''}]"


def _dataset_display_map() -> dict[str, dict[str, Any]]:
    display: dict[str, dict[str, Any]] = {}
    for entry in _dataset_manifest_entries():
        name = entry.get("name")
        if name and name not in display:
            display[name] = entry
    return display


def _dataset_card(name: str) -> dict[str, Any]:
    """UI dataset metadata backed by the authoritative readiness layer."""
    entry = _dataset_display_map().get(name, {})
    readiness = assess_dataset(name)

    raw_path = Path(readiness.raw_path)
    processed_path = Path(readiness.processed_path)

    # Preserve the manifest path for display/catalog purposes only.
    manifest_path = Path(
        entry.get(
            "path",
            readiness.raw_path,
        )
    )

    file_count = entry.get("file_count")
    if file_count is None:
        if raw_path.is_file():
            file_count = 1
        elif raw_path.is_dir():
            file_count = sum(
                1 for p in raw_path.rglob("*") if p.is_file()
            )
        else:
            file_count = 0

    return {
        "name": entry.get("name", name),
        "source": entry.get("source", "-"),
        "battery_family": entry.get("battery_family", "-"),
        "version": entry.get("version", "-"),
        "path": manifest_path,
        "raw_path": raw_path,
        "processed_path": processed_path,
        "file_count": file_count,
        "size": entry.get("size", "-"),
        "status": readiness.status.value,
        "raw_exists": readiness.raw_available,
        "raw_files": (
            1
            if raw_path.is_file()
            else sum(1 for p in raw_path.rglob("*") if p.is_file())
            if raw_path.is_dir()
            else 0
        ),
        "processed_exists": readiness.processed_available,
        "processed_files": readiness.processed_sample_count,
        "ready": readiness.status.value == "READY",
        "preprocessing_supported": readiness.preprocessing_supported,
        "readiness_reason": readiness.reason,
        "recommended_action": readiness.recommended_action,
    }

def _ui_dataset_options() -> list[str]:
    entries = _dataset_manifest_entries()

    # Preserve manifest order while removing duplicate dataset names.
    return list(
        dict.fromkeys(
            entry["name"]
            for entry in entries
            if entry.get("name")
        )
    )


def _ui_dataset_label(name: str) -> str:
    entry = _dataset_display_map().get(name, {})
    size = entry.get("size") or ""
    src = entry.get("source") or ""
    family = entry.get("battery_family") or ""
    extras = ", ".join([x for x in [src, family, size] if x])
    return f"{name} [{extras}]" if extras else name


def _selected_dataset_index(datasets: list[str]) -> int:
    """Where the session's selected dataset sits in ``datasets``.

    Falls back to the first entry when the session names a dataset this list
    does not contain, which is what happens after a manifest refresh drops
    one. Returning 0 rather than raising keeps the page usable; the selector
    then shows the dataset it is actually about to act on.
    """
    selected = st.session_state.get("selected_dataset")
    if selected in datasets:
        return datasets.index(selected)
    return 0


def _current_dataset_status(name: str) -> str:
    info = _dataset_card(name)
    status = str(info.get("status", "MISSING"))
    if status == "READY":
        return "Ready"
    if status == "RAW_ONLY":
        return "Raw available, preprocessing required"
    if status == "INCOMPLETE_PROCESSED":
        return "Processed incomplete, preprocessing required"
    return "Missing"


def _render_status_card(info: dict[str, Any]) -> None:
    ready = bool(info.get("ready"))
    status = str(info.get("status", "MISSING"))
    raw_exists = bool(info.get("raw_exists", False))
    if ready:
        status_class = "ready"
        label = "Ready"
    elif status in ("RAW_ONLY", "INCOMPLETE_PROCESSED") or raw_exists:
        status_class = "warning"
        label = "Preprocessing required" if status != "INCOMPLETE_PROCESSED" else "Processed incomplete"
    else:
        status_class = "missing"
        label = "Missing"
    st.markdown(
        f"""
        <div class="dataset-card {status_class}">
            <div style="font-size:1.1rem; font-weight:700; margin-bottom:0.75rem;">{info['name']}</div>
            <div style="margin-bottom:0.35rem;">Source: {info['source']}</div>
            <div style="margin-bottom:0.35rem;">Battery family: {info['battery_family']}</div>
            <div style="margin-bottom:0.35rem;">Version: {info['version']}</div>
            <div style="margin-bottom:0.35rem;">Path: {info['path']}</div>
            <div style="margin-bottom:0.35rem;">File count: {info['file_count']}</div>
            <div style="margin-bottom:0.35rem;">Size: {info['size']}</div>
            <div style="margin-bottom:0.65rem; font-weight:700;">Status: {info['status']}</div>
            <div style="font-size:0.9rem; font-weight:600;">{label}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_prepare_data_page():
    st.header("🔋 Prepare Data", anchor=False)
    st.caption("Raw data → Preprocessing → Processed records → Feature inspection → Ready for model")

    # Preprocessing and feature inspection are one tab, not two. They were
    # split while each carried its own dataset selector, and the two could
    # disagree about which dataset was being prepared. The dataset is chosen
    # once and the flow runs top to bottom: dataset → readiness → preprocess
    # → log → inspection. No step was dropped in the merge.
    # Built through the helper so a jump from the preprocess tab can open
    # the overview tab. The labels live in one place, so a button and the
    # tab it opens cannot drift apart.
    tabs = render_prepare_tabs(PREPARE_TABS)

    datasets = _ui_dataset_options()

    with tabs[0]:
        if st.button("Refresh datasets", use_container_width=True, key="refresh_datasets_btn"):
            result = _run_cmd([sys.executable, "tools/sync_data_and_config_manifests.py"], cwd=REPO_ROOT)
            if result["returncode"] == 0:
                st.success("Dataset and model manifests refreshed.")
                st.rerun()
            else:
                st.error(result["stderr"] or result["stdout"] or "Failed to refresh manifests.")
        if datasets:
            for start in range(0, len(datasets), 2):
                row_cols = st.columns(2, gap="large")
                for col, ds in zip(row_cols, datasets[start:start + 2]):
                    with col:
                        _render_status_card(_dataset_card(ds))
        else:
            st.info("No datasets are available yet. Click Refresh datasets after generating manifests.")
        st.markdown("---")
        rows = []
        for ds in datasets:
            info = _dataset_card(ds)
            rows.append({
                "name": info["name"],
                "source": info["source"],
                "battery_family": info["battery_family"],
                "version": info["version"],
                "path": str(info["path"]),
                "file_count": info["file_count"],
                "size": info["size"],
                "status": info["status"],
            })
        if rows:
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        st.markdown('<div class="pipeline-wrapper">', unsafe_allow_html=True)
        st.markdown("### Preparation Pipeline")
        selected_info = _dataset_card(st.session_state.get("selected_dataset", datasets[0] if datasets else "")) if datasets else {"raw_exists": False, "processed_exists": False, "ready": False}
        pipeline_cols = st.columns(5, gap="medium")
        stages = [
            ("📥", "Raw cycling files", "Ingest original cycling files", selected_info.get("raw_exists", False)),
            ("🧪", "BatteryML parser", "Parse raw files into BatteryML format", selected_info.get("raw_exists", False)),
            ("📦", "Processed cell records", "Generate processed records", selected_info.get("processed_exists", False)),
            ("🔎", "Feature/artifact inspection", "Inspect cache and feature outputs", selected_info.get("processed_exists", False)),
            ("🚀", "Model-ready dataset", "Ready for training", selected_info.get("ready", False)),
        ]
        for col, (icon, title, desc, active) in zip(pipeline_cols, stages):
            with col:
                color = "#1f7a3f" if active else "#b26a00"
                status_text = "Ready" if active else "Pending"
                st.markdown(
                    f"""
                    <div class="pipeline-card" style="border-left-color: {color};">
                        <div class="pipeline-icon">{icon}</div>
                        <div class="pipeline-title">{title}</div>
                        <div class="pipeline-desc">{desc}</div>
                        <div class="pipeline-status" style="color: {color};">{status_text}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        st.markdown('</div>', unsafe_allow_html=True)

    with tabs[1]:
        if not datasets:
            # No early return: the Custom Dataset tab below does not depend
            # on the registered-dataset list and must still render.
            st.info("No datasets available.")
        else:
            ds = st.selectbox(
                "Dataset",
                datasets,
                index=_selected_dataset_index(datasets),
                key="prep_ds_select",
            )
            info = _dataset_card(ds)
            raw_path = info["raw_path"]
            processed_path = info["processed_path"]
            st.write(f"**Selected dataset:** {ds}")
            st.write(f"**Dataset readiness:** {_current_dataset_status(ds)}")
            st.write(f"**Raw path:** `{raw_path}`")
            st.write(f"**Processed path:** `{processed_path}`")
            st.write(f"**Processed file count:** {info['processed_files']}")

            if not info["raw_exists"]:
                st.error("Raw data missing")
                st.button(
                    "Run preprocessing", type=PRIMARY_ROLE,
                    disabled=True,
                    use_container_width=True,
                    key="prep_run_disabled",
                )

            elif not info.get("preprocessing_supported", False):
                st.warning(
                    info.get(
                        "readiness_reason",
                        "Raw data are present, but this source is not yet wired "
                        "to the BatteryML preprocessing route.",
                    )
                )
                st.info(
                    "This raw source is available locally, but automatic "
                    "preprocessing is not enabled for this source yet."
                )
                st.button(
                    "Run preprocessing", type=PRIMARY_ROLE,
                    disabled=True,
                    use_container_width=True,
                    key="prep_run_unsupported",
                )

            else:
                command = [
                    sys.executable,
                    "batteryml_web/core/run_batteryml_cli.py",
                    "preprocess",
                    ds,
                    str(raw_path),
                    str(processed_path),
                ]

                st.code(
                    " ".join(command),
                    language="bash",
                )

                btn_label = (
                    "Re-run preprocessing"
                    if info["processed_exists"]
                    else "Run preprocessing"
                )

                if st.button(
                    btn_label, type=PRIMARY_ROLE,
                    use_container_width=True,
                    key="prep_run",
                ):
                    log_path = (
                        _project_root()
                        or Path("runs/web_projects/default")
                    ) / "logs" / f"preprocess_{ds}.log"

                    result = run_command_with_progress(
                        command=command,
                        cwd=REPO_ROOT,
                        log_path=log_path,
                        steps=[
                            (5, "Initialize"),
                            (20, "Check raw data"),
                            (40, "Start preprocessing"),
                            (80, "Write outputs"),
                            (100, "Done"),
                        ],
                    )

                    st.session_state["last_run_status"] = (
                        "success" if result["success"] else "failed"
                    )
                    st.session_state["last_run_time"] = result["elapsed"]
                    st.session_state["last_run_code"] = result["return_code"]
                    st.session_state["last_preprocess_log"] = (
                        Path(result["log_path"]).read_text(encoding="utf-8")
                        if Path(result["log_path"]).exists()
                        else ""
                    )

            if st.session_state.get("last_preprocess_log"):
                with st.expander("Latest preprocessing log", expanded=False):
                    st.text_area("Log", value=st.session_state["last_preprocess_log"], height=260, key="prep_log_view")

            st.markdown("---")
            st.markdown("**Feature / artifact inspection**")
            st.caption(
                "What preprocessing left on disk for the dataset selected above."
            )
            feature_files = [p for p in processed_path.rglob("*") if p.is_file() and any(k in p.name.lower() for k in ["feature", "cache", "artifact", ".npy", ".npz", ".pkl"])][:100] if processed_path.exists() else []
            if feature_files:
                cards = st.columns(2)
                for idx, p in enumerate(feature_files[:10]):
                    with cards[idx % 2]:
                        stat = p.stat()
                        # Escaped by the helper: these are file names read
                        # off disk, and one containing "<" would otherwise
                        # reshape the page.
                        render_info_card(
                            p.name,
                            [
                                f"Type: {p.suffix or 'unknown'}",
                                f"Size: {stat.st_size} bytes",
                                f"Path: {p}",
                            ],
                        )
                st.write("**Feature / cache / artifact files:**")
                st.code("\n".join(str(p) for p in feature_files[:30]), language="text")
                st.write("**Feature dimension:**")
                st.info("Feature information is not available yet. Run model or inspect BatteryML cache outputs.")
            else:
                st.info("Feature information is not available yet. Run model or inspect BatteryML cache outputs.")
            if st.button("Go to Run Model", type=NAV_ROLE, use_container_width=True, key="prep_go_run_model"):
                set_current_step("run_model")

        # Outside the dataset branch on purpose. The natural question at the
        # end of this tab is "did that change the dataset's readiness?", and
        # the answer is on the overview tab — which is just as true when
        # preprocessing is unavailable, when the source is RAW_ONLY, and
        # when no dataset is registered at all. That last case previously
        # left the tab with nothing to do but re-read "No datasets
        # available". It navigates only: no preprocessing is started, and
        # nothing about readiness is read or written here.
        st.markdown("---")
        if st.button(
            VIEW_OVERVIEW_LABEL, type=NAV_ROLE,
            use_container_width=True,
            key="prep_view_dataset_overview",
        ):
            request_tab(DATASET_OVERVIEW_TAB)
            st.rerun()

    with tabs[2]:
        st.caption("Upload raw cycling files exported from supported battery cyclers. Training and validation/test files must be uploaded separately. The platform will call BatteryML preprocessing before any training workflow is enabled.")
        project_dir = _selected_project_dir()
        uploads_dir = project_dir / "uploads"
        train_dir = uploads_dir / "train_raw"
        valid_dir = uploads_dir / "valid_raw"
        config_dir = uploads_dir / "config"
        train_dir.mkdir(parents=True, exist_ok=True)
        valid_dir.mkdir(parents=True, exist_ok=True)
        config_dir.mkdir(parents=True, exist_ok=True)
        dataset_name = st.text_input("Dataset name", value=st.session_state.get("custom_dataset_name", "custom_battery_dataset"), key="custom_dataset_name_input")
        source_format = st.selectbox("Source format", ["", *ALLOWED_CUSTOM_SOURCE_FORMATS], key="custom_source_format_select")
        task_type = st.selectbox("Task type", ["RUL prediction", "SOH estimation"], key="custom_task_type_select")
        train_uploaded = st.file_uploader("Upload training raw files", accept_multiple_files=True, key="custom_train_upload")
        valid_uploaded = st.file_uploader("Upload validation/test raw files", accept_multiple_files=True, key="custom_valid_upload")
        optional_config = st.file_uploader("Optional cycler config file", accept_multiple_files=False, key="custom_config_upload")
        if st.button("Save uploads", type=PRIMARY_ROLE, use_container_width=True, key="custom_save_uploads"):
            if not source_format:
                st.error("Please select a BatteryML-supported cycler format.")
            else:
                if train_uploaded:
                    saved_train = save_uploaded_files(train_uploaded, train_dir)
                    for p in saved_train:
                        extract_if_archive(p, train_dir)
                if valid_uploaded:
                    saved_valid = save_uploaded_files(valid_uploaded, valid_dir)
                    for p in saved_valid:
                        extract_if_archive(p, valid_dir)
                if optional_config is not None:
                    save_uploaded_files([optional_config], config_dir)
                if validate_custom_upload_layout(train_dir, valid_dir).ok:
                    build_uploaded_dataset_manifest(project_dir, source_format, dataset_name, task_type)
                    st.success("Uploads saved.")
                else:
                    st.error("Training raw files are required." if not collect_raw_files(train_dir) else "Validation/test raw files are required.")
        c1, c2, c3 = st.columns(3)
        with c1:
            if st.button("Preprocess training split", use_container_width=True, key="custom_preprocess_train"):
                if not collect_raw_files(train_dir):
                    st.error("Training raw files are required.")
                elif not source_format:
                    st.error("Please select a BatteryML-supported cycler format.")
                else:
                    report = run_batteryml_preprocess_for_split(source_format, train_dir, project_dir / "processed" / "train", optional_config_path=(config_dir / optional_config.name) if optional_config is not None else None)
                    (uploads_dir / "preprocess_report.json").write_text(json.dumps({"train": report}, indent=2, ensure_ascii=False), encoding="utf-8")
                    (uploads_dir / "preprocess_train.log").write_text((report["stdout"] or "") + "\n" + (report["stderr"] or ""), encoding="utf-8")
                    st.session_state["custom_train_preprocess_report"] = report
                    if report["success"]:
                        notify_success(
                            "Training split preprocessing finished",
                            event_key=EVENT_PREPROCESS_DONE,
                            signature=["train", str(project_dir / "processed" / "train")],
                        )
                    else:
                        st.error(
                            "Training split preprocessing failed; see the report and log "
                            f"written under {uploads_dir}."
                        )
        with c2:
            if st.button("Preprocess validation/test split", use_container_width=True, key="custom_preprocess_valid"):
                if not collect_raw_files(valid_dir):
                    st.error("Validation/test raw files are required.")
                elif not source_format:
                    st.error("Please select a BatteryML-supported cycler format.")
                else:
                    report = run_batteryml_preprocess_for_split(source_format, valid_dir, project_dir / "processed" / "valid", optional_config_path=(config_dir / optional_config.name) if optional_config is not None else None)
                    report_path = uploads_dir / "preprocess_report.json"
                    existing = json.loads(report_path.read_text(encoding="utf-8")) if report_path.exists() else {}
                    existing["valid"] = report
                    report_path.write_text(json.dumps(existing, indent=2, ensure_ascii=False), encoding="utf-8")
                    (uploads_dir / "preprocess_valid.log").write_text((report["stdout"] or "") + "\n" + (report["stderr"] or ""), encoding="utf-8")
                    st.session_state["custom_valid_preprocess_report"] = report
                    if report["success"]:
                        notify_success(
                            "Validation/test split preprocessing finished",
                            event_key=EVENT_PREPROCESS_DONE,
                            signature=["valid", str(project_dir / "processed" / "valid")],
                        )
                    else:
                        st.error(
                            "Validation/test split preprocessing failed; see the report and log "
                            f"written under {uploads_dir}."
                        )
        with c3:
            if st.button("Validate processed outputs", use_container_width=True, key="custom_validate_outputs"):
                train_result = validate_processed_output(project_dir / "processed" / "train")
                valid_result = validate_processed_output(project_dir / "processed" / "valid")
                report_path = uploads_dir / "preprocess_report.json"
                report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.exists() else {}
                report["processed_validation"] = {"train": train_result.details, "valid": valid_result.details}
                report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
                if train_result.ok and valid_result.ok:
                    st.success("Processed outputs are valid.")
                else:
                    st.error("Processed output is missing or incomplete.")
        manifest_path = uploads_dir / "uploaded_dataset_manifest.json"
        schema_report_path = uploads_dir / "uploaded_schema_report.json"
        schema_report_path.write_text(json.dumps({"train_raw_files": [str(p) for p in collect_raw_files(train_dir)], "valid_raw_files": [str(p) for p in collect_raw_files(valid_dir)]}, indent=2, ensure_ascii=False), encoding="utf-8")
        st.write(f"**Upload manifest:** `{manifest_path}`")
        st.write(f"**Schema report:** `{schema_report_path}`")
        st.write(f"**Preprocess report:** `{uploads_dir / 'preprocess_report.json'}`")
        if not source_format:
            st.info("Please select a BatteryML-supported cycler format.")
        if not collect_raw_files(train_dir):
            st.warning("Training raw files are required.")
        if not collect_raw_files(valid_dir):
            st.warning("Validation/test raw files are required.")
        st.write("Custom uploaded dataset has been preprocessed, but training config is not yet registered.")

    render_bottom_step_navigation()


def _discover_run_configs() -> dict[tuple[str, str, str], list[Path]]:
    registry = get_config_registry(str(REPO_ROOT))
    config_map: dict[tuple[str, str, str], list[Path]] = {}
    for entry in registry.entries:
        config_map.setdefault((entry.dataset, entry.task, entry.model_key), []).append(entry.config_path)
    for key in list(config_map.keys()):
        config_map[key] = sorted(config_map[key], key=lambda p: p.as_posix())
    return config_map


def _safe_read_log_tail(log_path: Path, lines: int = 80) -> str:
    if not log_path.exists():
        return ""
    try:
        text = log_path.read_text(encoding="utf-8", errors="ignore")
        return "\n".join(text.splitlines()[-lines:])
    except Exception:
        return "<unable to read log>"


def render_llm_availability_notice() -> None:
    """What a page says about an assistant it does not configure.

    Home, Run Model and Review Results each used to open an "Optional LLM
    Assistant settings" expander carrying the provider, Base URL, API key
    and model fields. Every one of them wrote the same store, so no value
    ever diverged — but a user who had filled the form in the sidebar and
    was then shown the same three fields on Review Results reasonably
    concluded the page wanted its own credentials. One settings screen, and
    everywhere else a sentence naming it.
    """
    if is_llm_configured():
        return
    render_configure_in_settings_notice()
    st.caption(
        "The deterministic BatteryML workflow runs without an assistant; "
        "only the optional explanations need one."
    )


def describe_llm_exception(exc: BaseException) -> str:
    """One user-facing sentence for a failed LLM call.

    Routed through :func:`classify_llm_failure`, which is the platform's one
    vocabulary for this: it separates "not configured" from a network or DNS
    failure, a timeout, a 401/403, an unavailable model and a malformed
    response, and it redacts anything key-shaped before the text is shown.
    The previous ``f"LLM request failed: {exc}"`` did none of that — it put
    the raw exception on the page, which for an HTTP error meant the
    provider's response body verbatim.
    """
    finding = classify_llm_failure(exc, configured=is_llm_configured())
    parts = [str(finding.get("short_title") or ""), str(finding.get("user_message") or "")]
    action = str(finding.get("recommended_action") or "")
    if action:
        parts.append(action)
    return " ".join(part for part in parts if part).strip() or "The LLM request failed."


def set_llm_not_configured_message(output_key: str):
    st.session_state[output_key] = (
        "LLM assistant is not configured yet.\n\n"
        "Add the Base URL, API key and model in Settings (left sidebar), then "
        "retry this action."
    )


def require_llm_or_show_message(output_key: str) -> bool:
    if not is_llm_configured():
        set_llm_not_configured_message(output_key)
        return False
    return True


def _set_llm_output(output_key: str, text: str, save_path: Path | None = None) -> None:
    st.session_state[output_key] = text
    # Which configuration wrote it, so the panel can say when the current
    # one no longer matches. Never the API key: the fingerprint records
    # only whether a key was set.
    st.session_state[
        f"{output_key}{SETTINGS_PROVENANCE_SUFFIX}"
    ] = llm_settings_fingerprint()
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_path.write_text(text, encoding="utf-8")


def build_config_suggestion_prompt(
    dataset: str,
    task: str,
    model: str,
    config_path: Path,
    config_text: str,
    dataset_readiness: dict[str, Any],
    available_configs: list[str],
) -> str:
    return f"""
You are assisting with a BatteryML web protocol platform.

Task:
Suggest an appropriate model/config choice for the current BatteryML run.

Current selection:
- Dataset: {dataset}
- Prediction task: {task}
- Selected model: {model}
- Config path: {config_path}
- Dataset readiness: {dataset_readiness}

Available configs:
{chr(10).join(str(x) for x in available_configs[:50])}

Selected config content:
```yaml
{config_text[:6000]}
```

Requirements:

Provide suggestions only.
Do not modify config automatically.
Do not invent unavailable configs.
If the current config is reasonable, say so.
If another available config is better, explain why.
Mention any missing data/preprocessing issues.
Return a concise recommendation with:
Recommended config
Reason
Risks / caveats
Next action
"""


def render_run_model_llm_output_panel(output_key: str, title: str, subtitle: str, context_key: str) -> None:
    output = st.session_state.get(output_key)
    if not output:
        return
    status = get_llm_status()
    status_label = describe_llm_status()
    status_class = "connected" if status == "connected" else "not-connected" if status == "failed" else ""
    st.markdown(
        f"""
        <div class="llm-output-wrap">
          <div class="llm-output-card">
            <div class="llm-output-header">
              <div>
                <div class="llm-output-title">🤖 LLM Assistant</div>
                <div class="llm-output-subtitle">{subtitle}</div>
              </div>
              <div class="llm-output-badges">
                <span class="llm-badge suggestion">Suggestion only</span>
                <span class="llm-badge {status_class}">{status_label}</span>
                <span class="llm-badge">{context_key}</span>
              </div>
            </div>
            <div class="llm-output-body">
        """,
        unsafe_allow_html=True,
    )
    if is_llm_configured():
        st.markdown("<div class='llm-output-note'>This suggestion will not automatically modify raw data, config files, predictions, or metrics.</div>", unsafe_allow_html=True)
        st.markdown(output)
        b1, b2 = st.columns(2, gap="medium")
        with b1:
            st.download_button("Download output as Markdown", data=output, file_name=f"{title.lower().replace(' ', '_')}.md", mime="text/markdown", use_container_width=True, key=f"download_{output_key}")
        with b2:
            if st.button("Clear output", key=f"clear_{output_key}", use_container_width=True):
                st.session_state.pop(output_key, None)
                st.rerun()
    else:
        st.markdown(
            """
            <div class="llm-empty-state">
              <div style="font-weight:700; margin-bottom:0.35rem;">LLM assistant is not configured yet.</div>
              <div>Add Base URL, API key, and model in Optional LLM Assistant settings.</div>
              <div class="llm-output-section">
                <h4>What this action does</h4>
                <ul>
                  <li>Provide workflow guidance only</li>
                  <li>Suggest config or command improvements</li>
                  <li>Never modify BatteryML outputs automatically</li>
                </ul>
              </div>
              <div class="llm-output-section">
                <h4>Next step</h4>
                <ul>
                  <li>Open Optional LLM Assistant settings</li>
                  <li>Add Base URL</li>
                  <li>Add API key</li>
                  <li>Add model</li>
                  <li>Retry this action</li>
                </ul>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    st.markdown("</div></div>", unsafe_allow_html=True)
    st.markdown("<div style='margin-bottom:1.9rem;'></div>", unsafe_allow_html=True)


#: Session key recording which LLM configuration produced a stored output.
#: Suffixed onto the output's own key so two panels cannot share one label.
SETTINGS_PROVENANCE_SUFFIX = "_llm_settings"

#: Whether a stored output is an answer ("ok") or a failed request's message
#: ("error"), and the connection serial it was produced under. Together they
#: keep a past failure from being read as the live connection state: an error
#: that a later successful connection superseded is not shown at all.
LLM_OUTCOME_SUFFIX = "_outcome"
LLM_SERIAL_SUFFIX = "_serial"

#: Set when a Run Model run completes; drives the "Review Results →" button.
RUN_MODEL_COMPLETED_KEY = "run_model_completed_run"


def render_llm_output_panel(output_key: str, title: str) -> None:
    """Show a stored assistant output, open, once per page; sends nothing.

    The text is already in session state, so no call is made here — the only
    thing that calls the assistant is the button that produced it. Review
    Results draws it once, directly below the buttons that asked for it.
    """
    output = st.session_state.get(output_key)
    if not output:
        return
    if st.session_state.get(f"{output_key}{LLM_OUTCOME_SUFFIX}") == "error":
        recorded = int(st.session_state.get(f"{output_key}{LLM_SERIAL_SUFFIX}") or 0)
        if connection_serial() > recorded:
            # A connection has worked since this request failed; the old
            # error would now misstate the live state, which the status
            # line above already gives.
            return
        st.warning(f"The last request for this result failed. {output}")
        st.caption(
            "That is the outcome of an earlier request, not a live connection check. "
            f"LLM assistant status now: {describe_llm_status()}."
        )
        return
    # Drawn open, not in a collapsed expander: the user pressed a button to
    # get this text, so hiding it behind a second click only hid the answer.
    with st.container(border=True, key=f"{output_key}_panel"):
        st.markdown(f"#### 🤖 LLM Assistant Output · {title}")
        st.caption("Suggestion only. This output does not automatically modify raw data, config files, predictions, or metrics.")
        generated_with = st.session_state.get(f"{output_key}{SETTINGS_PROVENANCE_SUFFIX}")
        if generated_with and generated_with != llm_settings_fingerprint():
            # The provider or model changed after this text was written.
            # Saying so is the whole of the claim; relabelling it with the
            # model now configured would credit work it did not do.
            st.caption(
                "Generated with the LLM configuration in effect at the time; "
                "the current configuration differs."
            )
        st.markdown(output)


def _render_run_model_without_datasets() -> None:
    """Run Model with no dataset registered: say so, and point somewhere useful.

    The page used to build its four tabs regardless. With nothing to offer,
    the dataset selectbox returns ``None``, and the configuration summary
    then did ``readiness[None]`` and killed the page with ``KeyError: None``
    — an ordinary empty state presented as a crash.

    The execution controls are omitted rather than shown disabled. There is
    no dataset to configure a run against, and a Run button that can never
    be pressed says less than a sentence naming what is missing. Nothing is
    selected on the user's behalf: no dataset is invented to fill the gap,
    and the session's dataset is left exactly as it was.

    What stays is the way out: the two pages that create the missing
    dataset, and the step navigation.
    """
    st.info("No datasets are currently available. Prepare or upload a dataset first.")
    st.caption(
        "Run Model executes a BatteryML run against a registered dataset, so "
        "there is nothing to select, configure or run until one exists."
    )
    st.markdown("**Where to go next**")
    st.write(
        "- **Prepare Data** — preprocess a dataset already under `data/raw/`, "
        "or upload your own raw cycling files under Custom Dataset / Schema "
        "Mapping."
    )
    st.write("- **Home** — create or load a project first, if you have not yet.")
    prepare_col, home_col = st.columns(2)
    with prepare_col:
        if st.button(
            "Go to Prepare Data", type=NAV_ROLE,
            use_container_width=True,
            key="run_empty_go_prepare_data",
        ):
            set_current_step("prepare_data")
    with home_col:
        if st.button(
            "Go to Home", type=NAV_ROLE,
            use_container_width=True,
            key="run_empty_go_home",
        ):
            set_current_step("home")
    render_bottom_step_navigation()


def _remember_run_model_task() -> None:
    """``on_change`` of Run Model's Task radio: record the user's explicit choice."""
    st.session_state[RUN_MODEL_TASK_KEY] = st.session_state["run_task_radio"]


def _run_model_initial_task() -> str:
    """The Task Run Model opens with, as a canonical label.

    The user's own choice on this page wins; until there is one, the session's
    task (set by Home's Quick panel), and RUL when neither names a task.
    """
    return (
        config_task.canonical_task(st.session_state.get(RUN_MODEL_TASK_KEY))
        or config_task.canonical_task(st.session_state.get("selected_task"))
        or config_task.TASK_RUL
    )


def render_run_model_page():
    st.header("🚀 Run Model", anchor=False)
    st.caption("Configure dataset, prediction task, model, device, and execute the deterministic BatteryML workflow.")
    configs = _discover_run_configs()
    datasets = _ui_dataset_options()
    tasks = list(config_task.SUPPORTED_TASK_LABELS)
    readiness = {ds: _dataset_card(ds) for ds in datasets}

    # Checked once, here, rather than inside each tab: every tab below reads a
    # selected dataset, and `readiness` is keyed by the datasets that exist, so
    # with none of them there is no key to look up and nothing to run.
    if not datasets:
        _render_run_model_without_datasets()
        return

    tabs = st.tabs(["Configuration", "Dataset Readiness", "Command & Run", "Logs / Debug"])

    with tabs[1]:
        cols = st.columns(2)
        for idx, ds in enumerate(datasets):
            info = readiness[ds]
            with cols[idx % 2]:
                status = "Ready" if info["ready"] else "Need preprocess" if info["raw_exists"] else "Missing raw"
                color = "#1f7a3f" if info["ready"] else "#b26a00" if info["raw_exists"] else "#b42318"
                # A card, not a control. It used to carry a 1px solid
                # border and a 14px radius directly above the "Select"
                # button's 1px solid border and 8px radius, which is why
                # the two read alike. The readiness colour is unchanged.
                render_info_card(
                    ds,
                    [
                        f"Raw files count: {info['raw_files']}",
                        f"Processed files count: {info['processed_files']}",
                    ],
                    status=status,
                    status_color=color,
                )
                if st.button(f"Select {ds}", key=f"run_select_{ds}", use_container_width=True):
                    st.session_state["selected_dataset"] = ds
                    st.rerun()
        if st.button("Go Prepare Data", type=NAV_ROLE, use_container_width=True, key="run_go_prepare_data"):
            set_current_step("prepare_data")

    with tabs[0]:
        left, right = st.columns([1.15, 0.95])
        with left:
            ds_idx = datasets.index(st.session_state.get("selected_dataset", "MATR")) if st.session_state.get("selected_dataset", "MATR") in datasets else 0
            ds = st.selectbox("Dataset", datasets, index=ds_idx, key="run_dataset_select")
            st.session_state["selected_dataset"] = ds
            task_choice = st.radio("Task", tasks, index=tasks.index(_run_model_initial_task()), key="run_task_radio", horizontal=True, on_change=_remember_run_model_task)
            st.session_state["selected_task"] = task_choice
            available_models = sorted({entry.model_key for entry in get_config_registry(str(REPO_ROOT)).get_entries(ds, task_choice)})
            if not available_models:
                st.warning("No configured model is available for this dataset/task. Please add a valid YAML config.")
                available_models = []
            # A model with no config for this task is not kept: cleared before
            # the selectbox is drawn, so it opens on the task's first model.
            # The config below is looked up from (dataset, task, model).
            if st.session_state.get("run_model_select") not in (available_models or [""]):
                st.session_state.pop("run_model_select", None)
            model = st.selectbox("Model", options=available_models or [""], index=0, key="run_model_select", disabled=not available_models)
            st.session_state["selected_model"] = model if available_models else ""
            device = st.selectbox("Device", ["cpu", "cuda"], index=0 if st.session_state.get("selected_device", "cpu") == "cpu" else 1, key="run_device_select")
            # The device the user chose is the device that is recorded and
            # run; one this machine cannot provide blocks Run (Command & Run
            # tab) instead of being replaced by cpu behind the user's back.
            cuda_ok, cuda_detail = quick_experience.cuda_available()
            device_issue = quick_experience.device_problem(device, cuda_ok=cuda_ok, cuda_detail=cuda_detail)
            if device_issue:
                st.warning(device_issue)
            st.session_state["selected_device"] = device
            seed = st.number_input("Seed", min_value=0, value=int(st.session_state.get("run_seed", 0)), step=1, key="run_seed_input")
            st.session_state["run_seed"] = int(seed)
        with right:
            config_paths = configs.get((ds, task_choice, model), [])
            config_path = config_paths[0] if config_paths else None
            workspace_dir = Path(_run_model_workspace(ds, model, config_path))
            info = readiness[ds]
            st.markdown("**Selected config summary**")
            st.write(f"Dataset: {ds}")
            st.write(f"Task: {task_choice}")
            st.write(f"Model: {model if model else 'Not configured'}")
            st.write(f"Device: {device}")
            st.write(f"Dataset readiness: {_current_dataset_status(ds)}")
            st.write(f"**Config path:** `{config_path if config_path else 'Not configured'}`")
            st.write(f"**Config exists:** {'Yes' if config_path and Path(str(config_path)).exists() else 'Missing'}")
            st.write(f"**Workspace dir:**")
            st.code(str(workspace_dir), language="text")
            render_llm_availability_notice()
            c1, c2 = st.columns(2)
            with c1:
                if st.button("Explain this config", key="run_model_explain_config", use_container_width=True):
                    if require_llm_or_show_message("run_model_config_llm_output"):
                        try:
                            with st.spinner("Calling LLM assistant..."):
                                # The file's own text and the page's resolved
                                # dataset/task/model, not just its path: from the
                                # path alone the assistant swapped dataset and
                                # model and invented config sections.
                                config_file = Path(str(config_path)) if config_path else None
                                try:
                                    config_text = config_file.read_text(encoding="utf-8") if config_file and config_file.is_file() else None
                                except OSError:
                                    config_text = None
                                response = run_llm_prompt(
                                    build_config_explanation_prompt(
                                        dataset=ds,
                                        task=task_choice,
                                        model_key=model,
                                        model_display=model_display_name(model),
                                        config_path=str(config_path),
                                        config_text=config_text,
                                    ),
                                    **llm_credentials(),
                                    system_prompt=CONFIG_EXPLANATION_SYSTEM_PROMPT,
                                )
                            set_llm_connection_result(True, "LLM request completed.")
                        except Exception as exc:
                            response = describe_llm_exception(exc)
                            set_llm_connection_result(False, response)
                        _set_llm_output(
                            "run_model_config_llm_output",
                            response,
                            (_project_root() or Path("runs/web_projects/default")) / "agent_outputs" / "config_explanation.md",
                        )
            with c2:
                if st.button("Suggest model/config", key="run_model_suggest_config", use_container_width=True):
                    if require_llm_or_show_message("run_model_config_llm_output"):
                        try:
                            with st.spinner("Calling LLM assistant..."):
                                response = run_llm_prompt(
                                    build_config_suggestion_prompt(
                                        dataset=ds,
                                        task=task_choice,
                                        model=model,
                                        config_path=Path(str(config_path)) if config_path else Path("<missing-config>"),
                                        config_text=(Path(str(config_path)).read_text(encoding="utf-8", errors="ignore") if config_path and Path(str(config_path)).exists() else "Config file missing."),
                                        dataset_readiness={"ready": info["ready"], "raw_exists": info["raw_exists"], "processed_exists": info["processed_exists"]},
                                        available_configs=[f"{k[0]} | {k[1]} | {k[2]} -> {v}" for k, v in list(configs.items())],
                                    ),
                                    **llm_credentials(),
                                    system_prompt="You are an assistant for a BatteryML web protocol platform. Provide suggestions only. Do not modify raw data, metrics, predictions, or config files.",
                                )
                            set_llm_connection_result(True, "LLM request completed.")
                        except Exception as exc:
                            response = describe_llm_exception(exc)
                            set_llm_connection_result(False, response)
                        _set_llm_output(
                            "run_model_config_llm_output",
                            response,
                            (_project_root() or Path("runs/web_projects/default")) / "agent_outputs" / "model_config_suggestion.md",
                        )
        # Full width, under both columns: drawn inside the right column the
        # answer was squeezed into ~45% of the page (user request).
        render_run_model_llm_output_panel("run_model_config_llm_output", "Config guidance", "Config guidance", "Config assistant")

    with tabs[2]:
        ds = st.session_state.get("selected_dataset", "MATR")
        task_choice = st.session_state.get("selected_task", config_task.TASK_RUL)
        model = st.session_state.get("selected_model", "")
        config_paths = configs.get((ds, task_choice, model), [])
        config_path = config_paths[0] if config_paths else None
        workspace_dir = Path(_run_model_workspace(ds, model, config_path))
        st.write(f"**Dataset:** {ds}")
        st.write(f"**Task:** {task_choice}")
        st.write(f"**Model:** {model if model else 'Not configured'}")
        st.write(f"**Dataset readiness:** {_current_dataset_status(ds)}")
        st.write(f"**Config path:** `{config_path if config_path else 'Not configured'}`")
        st.write(f"**Workspace dir:** `{workspace_dir}`")
        # Built by the one shared builder, so this route and the Agent
        # Workflow cannot drift apart again. They had: this page spelled the
        # argv out itself and omitted --metric, so the same config scored
        # different metrics depending on which button was pressed.
        command = build_run_command(
            config_path=str(config_path) if config_path else "<missing-config>",
            workspace=str(workspace_dir),
            task=task_choice,
            device=st.session_state.get("selected_device", "cpu"),
            seed=st.session_state.get("run_seed", 0),
        )
        # Bounded above, not fixed: st.code scrolls horizontally rather
        # than wrapping, so this command measured 54px at 1500px and the cap
        # is a no-op there. It exists so the block cannot grow into a wall
        # if a command or a viewport ever makes it wrap. The full command is
        # untouched and Streamlit's copy control still copies all of it.
        command_viewport().code(" ".join(command), language="bash")
        c1, c2 = st.columns([1, 2])
        with c1:
            if st.button("Explain this command", key="run_model_explain_command", use_container_width=True):
                if require_llm_or_show_message("run_model_command_llm_output"):
                    try:
                        with st.spinner("Generating LLM suggestion..."):
                            response = run_llm_prompt(f"Explain this BatteryML command: {' '.join(command)}. Return suggestion only.")
                    except Exception as exc:
                        response = f"LLM request failed: {exc}"
                    _set_llm_output(
                        "run_model_command_llm_output",
                        response,
                        (_project_root() or Path("runs/web_projects/default")) / "agent_outputs" / "command_explanation.md",
                    )
        cuda_ok, cuda_detail = quick_experience.cuda_available()
        run_device_issue = quick_experience.device_problem(
            st.session_state.get("selected_device", "cpu"), cuda_ok=cuda_ok, cuda_detail=cuda_detail)
        can_run = config_path is not None and Path(str(config_path)).exists() and readiness.get(ds, {}).get("ready", False) and run_device_issue is None
        if not readiness.get(ds, {}).get("ready", False):
            st.warning("Processed dataset missing. Go to Prepare Data and run preprocessing first.")
        if config_path is None or not Path(str(config_path)).exists():
            st.warning("Config missing for the selected dataset/model combination.")
        if run_device_issue:
            st.warning(run_device_issue)
        if can_run:
            # SOH also needs cells that can carry an SOH label (see soh_gate).
            can_run = render_soh_data_check(ds, task_choice, config_path, key="run_model", repo_root=REPO_ROOT)
        # Reserved outside the columns, so the run's progress spans the whole
        # content area instead of inheriting c2's two-thirds. The buttons stay
        # side by side; only what the run draws moves out from under them.
        run_progress_host = st.container()
        with c2:
            # `and can_run`: a click sent from a page drawn before the
            # selection changed is not a licence to run the new selection.
            if st.button("Run BatteryML", type=PRIMARY_ROLE, disabled=not can_run, use_container_width=True, key="run_batteryml_btn") and can_run:
                log_path = (_project_root() or Path("runs/web_projects/default")) / "logs" / "run_model.log"
                # Recorded before it starts, in its own folder, so Review
                # Results lists it — and a crash or closed tab leaves a run
                # reported INTERRUPTED rather than no trace at all.
                project_for_runs = _project_for_runs()
                entry = _registry_entry_for_config(config_path)
                run_workspace = workspace_dir.as_posix()
                run_dir = _begin_page_run(
                    run_id=workspace_dir.name,
                    workspace=run_workspace,
                    project_name=project_for_runs["name"],
                    dataset=ds,
                    task=task_choice,
                    model_key=model,
                    model_display=getattr(entry, "model_display", None),
                    config_path=config_path,
                    seed=st.session_state.get("run_seed", 0),
                    device=st.session_state.get("selected_device", "cpu"),
                    command=[str(part) for part in command],
                )
                # The next run is a new run: the command on screen gets a
                # fresh id on the next rerun.
                direct_runs.release_run_id(st.session_state, key=RUN_MODEL_RESERVED_ID_KEY)
                result = run_command_with_progress(command=command, cwd=REPO_ROOT, log_path=log_path, steps=[(5, "Initialize"), (10, "Check dataset"), (20, "Load config"), (30, "Start BatteryML run"), (70, "Training/evaluation running"), (90, "Collect outputs"), (100, "Done")], container=run_progress_host, live_stages=True)
                _finish_page_run(
                    run_dir,
                    result,
                    workspace=run_workspace,
                    dataset=ds,
                    task=task_choice,
                    model_key=model,
                    config_path=config_path,
                )
                st.session_state["last_run_status"] = "success" if result["success"] else "failed"
                st.session_state["last_run_time"] = result["elapsed"]
                st.session_state["last_run_code"] = result["return_code"]
                st.session_state["workspace_dir"] = str(workspace_dir)
                st.session_state["last_run_log_path"] = result["log_path"]
                st.session_state["last_run_command"] = " ".join(command)
                _collect_run_outputs(_project_root() or Path("runs/web_projects/default"), Path(config_path), Path(result["log_path"]), workspace_dir)
                if result["success"]:
                    notify_success(
                        "Run completed",
                        event_key=EVENT_RUN_COMPLETED,
                        signature=["run_model", str(workspace_dir), result["return_code"]],
                    )
                    st.session_state[RUN_MODEL_COMPLETED_KEY] = workspace_dir.name
                else:
                    st.session_state.pop(RUN_MODEL_COMPLETED_KEY, None)
        # Outside the Run branch. Drawn inside it, this button vanished on the
        # rerun its own click caused, so the click never navigated.
        if (
            st.session_state.get(RUN_MODEL_COMPLETED_KEY)
            and st.session_state.get("last_run_status") == "success"
        ):
            if st.button("Review Results →", type=NAV_ROLE, use_container_width=True, key="run_go_results_after_run"):
                set_current_step("review_results")
        render_run_model_llm_output_panel("run_model_command_llm_output", "Command guidance", "Command guidance", "Command assistant")
        st.markdown("#### Output workspace")
        st.code(str(workspace_dir), language="text")

    with tabs[3]:
        log_path = Path(st.session_state.get("last_run_log_path", "")) if st.session_state.get("last_run_log_path") else None
        st.write(f"**Last command:** {st.session_state.get('last_run_command', 'Not available yet')}")
        st.write(f"**Return code:** {st.session_state.get('last_run_code', 'N/A')}")
        st.write(f"**Log path:** {log_path if log_path else 'Not available yet'}")
        log_tail = _safe_read_log_tail(log_path) if log_path else ""
        # The element that actually grew without bound: 200 lines of log
        # measured 4510px. Capped to a scrolling viewport; nothing is
        # truncated, and this stays the Logs tab — the command preview and
        # the progress display are separate surfaces and remain so.
        log_viewport().code(log_tail or "Not available yet", language="text")
        if st.session_state.get("last_run_code") in [139, -11] or "segmentation fault" in log_tail.lower():
            st.error("Native dependency crash / segmentation fault detected. Possible causes: xgboost, libomp, numpy, sklearn, torch, or cache incompatibility.")
        error_summary = "No error detected." if st.session_state.get("last_run_status") == "success" else (log_tail[-500:] if log_tail else "No logs available.")
        st.write(f"**Error summary:** {error_summary}")
        c1, c2 = st.columns(2)
        with c1:
            if st.button("Explain last error", key="run_model_explain_error", use_container_width=True):
                if require_llm_or_show_message("run_model_debug_llm_output"):
                    try:
                        with st.spinner("Generating LLM suggestion..."):
                            response = run_llm_prompt(f"Explain this error summary and suggest next action: {error_summary}. Return suggestion only.")
                    except Exception as exc:
                        response = f"LLM request failed: {exc}"
                    _set_llm_output(
                        "run_model_debug_llm_output",
                        response,
                        (_project_root() or Path("runs/web_projects/default")) / "agent_outputs" / "debug_explanation.md",
                    )
        with c2:
            if st.button("Suggest next action", key="run_model_suggest_next_action", use_container_width=True):
                if require_llm_or_show_message("run_model_debug_llm_output"):
                    try:
                        with st.spinner("Generating LLM suggestion..."):
                            response = run_llm_prompt(f"Suggest the next action after this BatteryML run: {error_summary}. Return suggestion only.")
                    except Exception as exc:
                        response = f"LLM request failed: {exc}"
                    _set_llm_output(
                        "run_model_debug_llm_output",
                        response,
                        (_project_root() or Path("runs/web_projects/default")) / "agent_outputs" / "debug_suggestion.md",
                    )
        render_run_model_llm_output_panel("run_model_debug_llm_output", "Debug guidance", "Debug guidance", "Debug assistant")
        st.write("**Safety note:** LLM only explains; it does not modify raw data, metrics, or config.")

    render_bottom_step_navigation()


#: Session keys for the selected result. The identity key is stored *beside*
#: every result-derived value, so a value computed for one run can never be
#: displayed under another — see ``core/selected_result.py``.
SELECTED_IDENTITY_KEY = "review_result_identity"
EXPLANATION_IDENTITY_KEY = "review_llm_output_identity"
ARTIFACT_CHOICE_KEY = "review_artifact_choice"


def _current_project():
    """The project every part of Review Results is read against."""
    return resolve_project(
        project_root=st.session_state.get("project_root"),
        project_name=st.session_state.get("project_name")
        or st.session_state.get("current_project"),
    )


def _selected_run_manifest():
    """The run the user opened in Run History, or None.

    Looked up **inside the current project only**, so a selection left over
    from a previous project resolves to None rather than reading that
    project's directory. ``find_run`` searches ``project["root"]`` and
    nothing above it.
    """
    selected = st.session_state.get("run_history_selection")
    if not selected:
        return None
    project = _current_project()
    return find_run(
        project["root"], str(selected), live_run_ids=get_run_manager().live_run_ids()
    )


def _selected_result() -> SelectedResult:
    """The one identity every section of Review Results derives from.

    Built from the selected run's own manifest. Replaces the project-wide
    ``find_result_files`` search that used to feed the Metrics, Summary and
    Logs tabs and both LLM prompts: that search ``rglob``\\ed the whole
    project, sorted by path, and took ``[0]`` — and because a run id starts
    with a timestamp, ``[0]`` was the *oldest run in the project*. Selecting
    a run changed none of those sections, which is why metrics and
    predictions looked frozen and identical between runs.
    """
    project = _current_project()
    manifest = _selected_run_manifest()
    selected = resolve_selected_result(
        project,
        manifest,
        chosen_artifact=st.session_state.get(ARTIFACT_CHOICE_KEY),
        repo_root=REPO_ROOT,
    )
    _invalidate_stale_result_state(selected)
    return selected


def _invalidate_stale_result_state(selected: SelectedResult) -> None:
    """Drop anything computed for a different result than this one.

    Selecting run B must not leave run A's explanation on screen under B's
    heading, and a slow explanation for A that lands after the user moved to
    B must not be shown as B's. Both are handled the same way: every
    result-derived value is stored with the identity it was computed for, and
    anything that does not match the current identity is dropped here, before
    the page draws.
    """
    identity = selected.identity_key()
    previous = str(st.session_state.get(SELECTED_IDENTITY_KEY) or "")
    if previous == identity:
        return
    st.session_state[SELECTED_IDENTITY_KEY] = identity
    for key in (
        "review_llm_output",
        EXPLANATION_IDENTITY_KEY,
        f"review_llm_output{SETTINGS_PROVENANCE_SUFFIX}",
        f"review_llm_output{LLM_OUTCOME_SUFFIX}",
        f"review_llm_output{LLM_SERIAL_SUFFIX}",
    ):
        st.session_state.pop(key, None)


def _record_review_output(response: str, *, ok: bool, selected: SelectedResult) -> None:
    """Store a Summarize/Explain outcome with everything that identifies it.

    The result it was produced for (a slow reply for run A that lands after
    the user moved to run B fails the identity check and is dropped rather
    than shown under B), the configuration that produced it (never the key),
    whether it is an answer or a failure, and the connection serial at the
    time — so a failure is never shown as the live state after a later
    successful connection.
    """
    st.session_state["review_llm_output"] = response
    st.session_state[EXPLANATION_IDENTITY_KEY] = selected.identity_key()
    st.session_state[f"review_llm_output{SETTINGS_PROVENANCE_SUFFIX}"] = llm_settings_fingerprint()
    st.session_state[f"review_llm_output{LLM_OUTCOME_SUFFIX}"] = "ok" if ok else "error"
    st.session_state[f"review_llm_output{LLM_SERIAL_SUFFIX}"] = connection_serial()


def _selected_metrics_frame(selected: SelectedResult):
    """The selected run's canonical metrics, as a frame, or None.

    Read from that run's own manifest, which the run manager wrote from the
    ``scores`` BatteryML itself saved. Nothing is recomputed here, and
    nothing is read from another run: a selected run with no recorded
    metrics returns None and the page says so for *that* run.
    """
    manifest = selected.manifest or {}
    metrics = manifest.get("metrics")
    if not isinstance(metrics, dict) or not metrics:
        return None
    return pd.DataFrame(
        [{"Metric": name, "Value": value} for name, value in metrics.items()]
    )


def _selected_analysis_context(selected: SelectedResult) -> dict[str, Any]:
    """The deterministic analysis facts for the currently selected result.

    Built from this run's own manifest and its own selected artifact — the
    same identity the rest of the page describes. Nothing here searches the
    project for a "latest" artifact, which is the substitution STEP 1
    removed.
    """
    manifest = dict(selected.manifest or {})
    metrics = manifest.get("metrics")
    result_context = {
        "dataset": manifest.get("dataset"),
        "task": manifest.get("task"),
        "model": manifest.get("model_key"),
        "config_path": manifest.get("config_path"),
        "workspace": selected.run_dir,
        "run_status": manifest.get("status"),
        "success": str(manifest.get("status") or "") == rm.STATUS_COMPLETED,
        "metrics": dict(metrics) if isinstance(metrics, dict) else {},
        "warnings": [],
    }
    return build_analysis_context(
        result_context,
        manifest=manifest,
        artifact_path=selected.predictions_file,
        run_id=selected.run_id,
        run_dir=selected.run_dir,
        project_name=selected.project_name,
        identity_key=selected.identity_key(),
    )


def render_benchmark_panel(analysis: dict[str, Any]) -> None:
    """The compact published reference, beside the run's own number.

    Deterministic: the registry decides what is shown and whether a
    difference may be computed. The assistant never supplies a published
    value, and this panel is drawn whether or not an explanation was ever
    requested.

    Deliberately four lines, not the 14x8 table. The full table is a
    reference, not something to re-read on every result page; it lives in
    the details expander below.
    """
    block = analysis.get("benchmark") or {}
    record = block.get("record")
    st.markdown("#### Benchmark reference")
    if record is None:
        st.caption(f"Published BatteryML reference: none. {block.get('explanation', '')}")
        return

    st.caption("Published BatteryML reference")
    left, right = st.columns(2)
    with left:
        st.write(f"Model: {record['model_key']}")
        st.write(f"Dataset: {record['dataset_subset']}")
        st.write(f"Metric: {record['metric']} ({record['unit']})")
    with right:
        seeds = f" over {record['seeds']} seeds" if record["seeds"] > 1 else ""
        st.write(f"Reference: {record['display']} {record['unit']}{seeds}")
        run_value = block.get("run_value")
        st.write(
            f"Current run: {run_value!r} {record['unit']}"
            if run_value is not None
            else "Current run: no comparable metric saved"
        )
        if block.get("comparable"):
            difference = block.get("difference")
            st.write(f"Difference: {difference:+.1f} {record['unit']}")
        else:
            st.write("Comparison: not directly comparable")
    st.caption(block.get("explanation", ""))
    st.caption(f"Source: {record['source']}")


def _render_selected_result_header(selected: SelectedResult) -> None:
    """The compact identification block: which result is on screen.

    Deliberately small — project, run id, model/dataset/task, run directory.
    The exact artifact path lives in the details area below, not here.
    """
    st.dataframe(
        selected.provenance_rows(), hide_index=True, use_container_width=True
    )


def _render_project_run_history() -> None:
    """The current project's runs, and the one the user opened."""
    project = resolve_project(
        project_root=st.session_state.get("project_root"),
        project_name=st.session_state.get("project_name")
        or st.session_state.get("current_project"),
    )
    root = project["root"]
    selected = render_run_history(root, project_name=project["name"])
    if not selected:
        return
    manifest = find_run(
        root, selected, live_run_ids=get_run_manager().live_run_ids()
    )
    if manifest is None:
        st.caption(f"Run {selected} is no longer on disk.")
        return
    # A readable label; the full run id is the first thing inside.
    with st.expander(
        f"Run details · {short_run_id(selected)} · {experiment_summary(manifest)}", expanded=True
    ):
        # Opening a historical run shows it. It never re-runs it.
        render_run_detail(manifest)


def render_review_results_page():
    st.header("📈 Review Results", anchor=False)

    # Quick experience may arrive here after a failed run.  Keep the result
    # page useful even when no metric artifact was produced: the command's
    # concrete failure reason is shown before the run history and log tabs.
    if st.session_state.get("last_run_status") == "failed":
        st.error(
            f"The latest BatteryML run failed: "
            f"{st.session_state.get('last_run_error') or 'See the selected run log for details.'}"
        )
        if st.session_state.get("last_run_log_path"):
            st.caption(f"Full diagnostic log: {st.session_state['last_run_log_path']}")

    # Run history first: a project holds many runs, and finding an earlier
    # one should not mean searching the filesystem by hand. Read from the
    # run manifests, so it survives a refresh and an app restart.
    _render_project_run_history()

    project_dir = _project_root() or Path("runs/native_matr_xgb_test")

    # Everything below describes exactly this result, or says it cannot.
    # There is no project-wide artifact search here any more: the previous
    # one returned every run's files sorted by path and the readers took
    # [0], which is the oldest run in the project, whatever was selected.
    selected = _selected_result()
    metrics_df = _selected_metrics_frame(selected)
    metrics_source = (
        "run manifest" if metrics_df is not None and not metrics_df.empty else "missing"
    )
    # Scoped to the selected run's own directory. A run that wrote no logs
    # says so rather than showing another run's.
    run_files = (
        find_result_files(Path(selected.run_dir), Path(selected.run_dir))
        if selected.run_dir and Path(selected.run_dir).is_dir()
        else {"log_files": [], "prediction_files": [], "metrics_files": []}
    )

    # Keep the main result navigation close to the page heading. Optional
    # interpretation is secondary and should not push these tabs below it:
    # its section comes after the tabs and the benchmark reference.
    st.subheader("Selected result", anchor=False)
    tabs = st.tabs(["Summary", "Metrics", "Predictions", "Logs", "Files"])

    # Deterministic, and drawn whether or not anyone asks for an
    # explanation: the published reference and whether it may be compared
    # are backend facts, not the assistant's to decide. Reading the artifact
    # for diagnostics happens once here and the same object is handed to the
    # prompt, so the page and the model never see different evidence.
    selected_analysis = _selected_analysis_context(selected)
    render_benchmark_panel(selected_analysis)

    st.subheader("Optional LLM interpretation", anchor=False)
    st.caption(
        "Use the LLM assistant to summarize results or explain metrics. BatteryML metrics are computed by BatteryML and will not be modified."
    )
    # The live connection state, read from the one store. Kept apart from
    # any stored output below, which describes an earlier request. Filled
    # in after the buttons are handled, so a request made on this run is
    # already reflected in it.
    status_slot = st.empty()

    btn_col1, btn_col2, btn_col3 = st.columns([1, 1, 1])
    with btn_col1:
        summarize_clicked = st.button("Summarize results", key="review_summarize_results", type=SECONDARY_ROLE, use_container_width=True)
    with btn_col2:
        explain_clicked = st.button("Explain metrics", key="review_explain_metrics", type=SECONDARY_ROLE, use_container_width=True)
    with btn_col3:
        configure_clicked = st.button("Configure LLM", key="review_configure_llm", type=NAV_ROLE, use_container_width=True)

    if configure_clicked:
        request_llm_settings(REVEAL_CONFIGURE)

    # One click, at most one request: a button is True only on the run its
    # click caused, and nothing below is re-sent on an ordinary rerun.
    if summarize_clicked:
        if not llm_action_allowed():
            # Not connected: open Settings and send nothing.
            request_llm_settings(REVEAL_ACTION)
        else:
            # Built from the selected result's own identity and manifest.
            # It used to be built from the project-wide metrics table and
            # st.session_state["workspace_dir"] — the last *manually* run
            # workspace — so the explanation regularly described a different
            # experiment from the one on screen.
            prompt = f'''
You are assisting with a BatteryML result review.
Summarize the available metrics, predictions, logs, and result files.
Summarize run status, explain model performance briefly, mention missing files if any, and do not invent missing results or modify metrics.
Project: {selected.project_name or _current_project_name()}
Run ID: {selected.run_id or "Not available"}
Run directory: {selected.run_dir or "Not available"}
Dataset: {(selected.manifest or {}).get("dataset", "Not available")}
Task: {(selected.manifest or {}).get("task", "Not available")}
Model: {(selected.manifest or {}).get("model_display") or (selected.manifest or {}).get("model_key") or "Not available"}
Metrics table:
{metrics_df.to_string() if metrics_df is not None else "No metrics available"}
Predictions artifact: {selected.predictions_file or selected.reason}
Logs available: {'Yes' if run_files.get("log_files") else 'No'}
'''
            try:
                response = run_llm_prompt(
                    prompt,
                    **llm_credentials(),
                    system_prompt="You are an assistant for a BatteryML web protocol platform. Provide suggestions only. Do not modify raw data, metrics, predictions, or config files.",
                )
                ok = True
                set_llm_connection_result(True, "LLM request completed.")
            except Exception as exc:
                response = describe_llm_exception(exc)
                ok = False
                set_llm_connection_result(False, response)
            _record_review_output(response, ok=ok, selected=selected)

    if explain_clicked:
        if not llm_action_allowed():
            request_llm_settings(REVEAL_ACTION)
        else:
            # The analysis path: deterministic facts first, then the prompt
            # that asks for interpretation. It replaces an f-string that
            # sent the metrics table and asked for "plain language",
            # which is what made the answer a restatement.
            analysis = selected_analysis
            try:
                parsed = explain_run_result(
                    result_context_from_analysis(analysis),
                    analysis_context=analysis,
                    **llm_credentials(),
                )
                response = render_analysis_markdown(parsed)
                ok = True
                set_llm_connection_result(True, "LLM request completed.")
            except Exception as exc:
                response = describe_llm_exception(exc)
                ok = False
                set_llm_connection_result(False, response)
            _record_review_output(response, ok=ok, selected=selected)

    status_slot.caption(f"LLM assistant status: {describe_llm_status()}")
    if not is_llm_configured():
        # A sentence, not a form, on the page itself; the form opens only in
        # the Settings dialog below, from the one shared settings body.
        render_llm_availability_notice()
    render_requested_llm_settings()

    # Shown only when it was produced for the result now on screen. Anything
    # else — a previous selection's answer, or a slow reply for a run the
    # user has since left — is withheld rather than relabelled.
    if cached_for(
        st.session_state, "review_llm_output", EXPLANATION_IDENTITY_KEY, selected
    ) is not None:
        render_llm_output_panel("review_llm_output", "Result interpretation")

    with tabs[0]:
        _render_selected_result_header(selected)
        if not selected.available:
            st.info(selected.reason)
        # Status as status: a passive pill, never a control.
        st.markdown(
            f"**Status:** {status_badge((selected.manifest or {}).get('status', 'Not available'))}",
            unsafe_allow_html=True,
        )
        st.write(f"**Metrics found:** {'Yes' if metrics_df is not None and not metrics_df.empty else 'No'}")
        st.write(f"**Predictions artifact:** {'Yes' if selected.available else 'No'}")
        st.write(f"**Logs found:** {'Yes' if run_files.get('log_files') else 'No'}")
        st.write(f"**Metrics source:** {metrics_source}")

    with tabs[1]:
        if metrics_df is not None and not metrics_df.empty:
            st.dataframe(metrics_df, use_container_width=True, hide_index=True)
            st.caption(
                f"Canonical metrics recorded by run {selected.run_id or 'unknown'}. "
                "Read from that run's manifest; nothing is recomputed here."
            )
        elif selected.run_id:
            st.info(
                f"Run {selected.run_id} recorded no metrics. Another run's "
                "metrics are never shown in their place."
            )
        else:
            st.info(selected.reason)

    with tabs[2]:
        # Exactly the selected run's artifact, through the shared CPU-safe
        # reader. There is no "no run selected, so preview whatever is on
        # disk" branch any more: that branch previewed the project-wide glob
        # and is how another run's predictions appeared under this heading.
        if selected.candidates and len(selected.candidates) > 1:
            st.selectbox(
                "Prediction artifact",
                list(selected.candidates),
                key=ARTIFACT_CHOICE_KEY,
            )
        if selected.available:
            render_predictions_panel(
                {
                    "artifacts": {"prediction_files": [selected.predictions_file]},
                    "run_id": selected.run_id,
                }
            )
            # The "Artifact diagnostics" disclosure that stood here was a
            # developer view: counts, finite/non-finite, distinct, constant,
            # min/max at full precision. It answered a question a reader of
            # a result does not ask, so the normal Results UI no longer
            # offers it. The module behind it
            # (batteryml_web.agent.prediction_diagnostics) is untouched and
            # still feeds the deterministic context that LLM Result Analysis
            # reads; see batteryml_web/agent/result_analysis.py.
        else:
            st.info(selected.reason)

    with tabs[3]:
        if run_files.get("log_files"):
            for log_file in run_files["log_files"]:
                st.markdown(f"**{log_file.name}**")
                try:
                    text = log_file.read_text(encoding="utf-8", errors="ignore")
                except Exception:
                    text = "<unable to read log>"
                lines = text.splitlines()
                st.code("\n".join(lines[-120:]) if lines else "", language="text")
        else:
            st.info(
                f"Run {selected.run_id or 'unknown'} recorded no log files."
                if selected.run_id
                else selected.reason
            )

    with tabs[4]:
        if selected.manifest:
            # Where it wrote, then the files grouped by what they are.
            render_run_location(dict(selected.manifest))
            if selected.run_dir:
                render_file_browser(selected.run_dir)
            if selected.predictions_file:
                st.caption("Prediction artifact read for this result:")
                st.code(selected.predictions_file, language="text")
        else:
            st.info(selected.reason)
            st.dataframe(
                [
                    {"": "Project", "Path": _current_project_name()},
                    {"": "Project directory", "Path": str(project_dir)},
                ],
                hide_index=True,
                use_container_width=True,
            )

    render_bottom_step_navigation()


# The sidebar lives in batteryml_web/ui/workflow_sidebar.py. A second copy
# used to be defined here and shadowed the import, so the grouped navigation
# in that module was dead code.
render_sidebar_workflow()
# Injected once per script run, beside the sidebar's own stylesheet. It is
# its own module rather than an addition to ui/styles.py, which owns the
# sidebar's tuned spacing and is deliberately left alone.
render_surface_css()
# The mentor's visual language on every page, not only Home (ui/home_style.py).
render_site_style()

current_step = st.session_state.get("current_step", "home")

# Before the page body, not after: Streamlit streams elements as the script
# produces them, so this still runs while the page arrives, and a page that
# raises part-way through still opens at the top rather than silently keeping
# the previous page's scroll offset. Injected only when the step changed, so a
# rerun within a page (a button, a progress tick) leaves the reader where they
# were. See ui/scroll.py.
render_scroll_to_top(current_step)

# The page body runs inside an error boundary. An application error renders
# our own "Something went wrong" view with the details one click away and a
# copy control of our own, instead of Streamlit's built-in traceback panel
# with its Copy / Ask Google / Ask ChatGPT buttons — which shows a scientist a
# developer's view and invites them to paste an internal traceback into a
# search engine. Streamlit's panel is not touched: a genuinely uncaught
# framework failure still reaches it, and st.rerun()/st.stop() pass straight
# through. See ui/errors.py.
with error_boundary(
    key=f"page_{current_step}",
    message=(
        "This page could not be drawn. Your project, runs and data are "
        "untouched — nothing was executed or written. Try the step again, or "
        "move to another page from the sidebar."
    ),
    context={"page": current_step, "project": _current_project_name()},
):
    if current_step == "home":
        render_home_page()
    elif current_step == "prepare_data":
        render_prepare_data_page()
    elif current_step == "run_model":
        render_run_model_page()
    elif current_step == "agent_workflow":
        render_agent_workflow()
    elif current_step == "review_results":
        render_review_results_page()
    else:
        st.session_state["current_step"] = "home"
        st.rerun()
