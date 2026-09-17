from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st
import torch

from batteryml_web.agent.llm_client import run_llm_prompt, save_llm_output
from batteryml_web.agent.catalog import REQUEST_TEMPLATE_STRUCTURE
from batteryml_web.ui.agent_workflow import (
    USE_TEMPLATE_LABEL,
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
from batteryml_web.core.results_reader import find_result_files, load_metrics, load_metrics_from_logs, load_predictions
from batteryml_web.ui.feedback import (
    EVENT_PREPROCESS_DONE,
    EVENT_PROJECT_CREATED,
    EVENT_QUICKSTART_READY,
    EVENT_RUN_COMPLETED,
    notify_success,
    toast,
)
from batteryml_web.ui.llm_state import describe_llm_status, ensure_llm_state_defaults, get_llm_status, is_llm_configured, refresh_llm_status_from_config, set_llm_settings
from batteryml_web.core.project_context import resolve_project
from batteryml_web.core.run_manager import get_run_manager
from batteryml_web.core.run_manifest import find_run
from batteryml_web.ui.run_history import (
    render_file_browser,
    render_predictions_panel,
    render_run_detail,
    render_run_history,
    render_run_location,
)
from batteryml_web.ui.settings_panel import render_llm_settings_form
from batteryml_web.ui.progress_utils import get_safe_runtime_env, run_command_with_progress
from batteryml_web.ui.scroll import render_scroll_to_top
from batteryml_web.ui.errors import error_boundary, render_copy_control
from batteryml_web.ui.workflow_sidebar import NAV_STEPS, get_current_step_index, render_sidebar_workflow, set_current_step
from batteryml_web.core.readiness import assess_dataset
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
if "selected_dataset" not in st.session_state:
    st.session_state["selected_dataset"] = "MATR"
if "selected_task" not in st.session_state:
    st.session_state["selected_task"] = "RUL"
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
[data-testid="stAppViewBlockContainer"] {
    padding-top: 1.5rem;
}
[data-testid="stHeader"] {
    height: 2.25rem;
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
    --app-shadow: 0 8px 24px rgba(0, 0, 0, 0.08);
}

.llm-output-wrap {
    margin-top: 0.75rem;
    margin-bottom: 1.6rem;
}

.llm-output-card {
    width: 100%;
    border-radius: 18px;
    border: 1px solid var(--app-border);
    padding: 1.15rem 1.25rem;
    background: var(--app-card-bg);
    box-sizing: border-box;
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

.dataset-card {
    padding: 1.25rem 1.35rem;
    border-radius: 18px;
    border: 1.5px solid rgba(80, 160, 100, 0.85);
    min-height: 280px;
    margin-bottom: 1rem;
    box-sizing: border-box;
    background: var(--app-card-bg);
    box-shadow: var(--app-shadow);
}

.dataset-card.ready {
    border-color: rgba(55, 140, 80, 0.95);
}

.dataset-card.warning {
    border-color: rgba(190, 130, 40, 0.95);
}

.dataset-card.missing {
    border-color: rgba(190, 70, 70, 0.95);
}

.pipeline-card {
    height: 320px;
    min-height: 320px;
    max-height: 320px;
    border-radius: 20px;
    border: 1.5px solid rgba(55, 140, 80, 0.95);
    padding: 1.2rem 1rem;
    box-sizing: border-box;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    align-items: center;
    text-align: center;
    background: var(--app-card-bg);
    box-shadow: var(--app-shadow);
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
        if st.button("← Prev", disabled=(idx == 0), use_container_width=True, key="bottom_prev_step"):
            set_current_step(NAV_STEPS[idx - 1]["key"])
    with next_col:
        if st.button("Next →", disabled=(idx == len(NAV_STEPS) - 1), use_container_width=True, key="bottom_next_step"):
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
    request = st.text_area(
        "Describe what you want to do...",
        value=st.session_state.get("agent_original_request", ""),
        placeholder=REQUEST_TEMPLATE_STRUCTURE,
        key="home_agent_request",
        height=90,
    )
    if accept_request_template(template, request):
        if st.button(USE_TEMPLATE_LABEL, key="home_use_template_button"):
            st.session_state["home_use_template"] = True
            st.rerun()
    if st.button("Analyze with Agent", use_container_width=True, key="home_agent_analyze"):
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


def render_home_page():
    st.title("🔋 LLM-guided BatteryML Protocol Platform", anchor=False)
    st.caption("BatteryML performs numerical prediction. The optional LLM agent assists workflow guidance only.")

    project_root = _project_root()
    current_dataset = st.session_state.get("selected_dataset", "MATR")
    last_run = st.session_state.get("last_run_status", "N/A")
    # From the one status the credential store maintains — not re-derived
    # from whether two fields happen to be non-empty, which reported
    # "Connected" for settings that had never been tested.
    llm_status = describe_llm_status()

    summary_items = [
        ("Project", _current_project_name()),
        ("Dataset", current_dataset),
        ("Current page", "Home"),
        ("Last run", last_run),
        ("LLM", llm_status),
    ]
    cols = st.columns(len(summary_items))
    for col, (label, value) in zip(cols, summary_items):
        with col:
            st.markdown(
                f"""
                <div class='summary-card'>
                    <div class='summary-label'>{label}</div>
                    <div class='summary-value'>{value}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("---")
    left_col, right_col = st.columns(2, gap="large")

    with left_col:
        st.subheader("Create New Project", anchor=False)
        project_name = st.text_input("Project name", value=st.session_state.get("project_name") or "my_battery_project", key="home_project_name")
        base_dir = st.text_input("Projects base directory", value=str(DEFAULT_BASE_DIR), key="home_base_dir")
        project_folder = Path(base_dir) / (project_name.strip() or "my_battery_project")
        st.caption("Project folder")
        st.code(str(project_folder), language="text")
        if st.button("Create New Project", use_container_width=True, key="home_create_project"):
            created = _ensure_project(project_name, base_dir)
            # The filesystem semantics are unchanged; only the confirmation is
            # new. Stored rather than printed because the rerun below would
            # otherwise discard the message before it is seen.
            st.session_state["project_created_notice"] = {
                "name": created.name,
                "workspace": created.as_posix(),
                "action": "created",
            }
            st.rerun()
        _render_project_created_notice()

    with right_col:
        st.subheader("Load Existing Project", anchor=False)
        projects = _safe_projects()
        if projects:
            selected = st.selectbox("Select existing project", options=projects, key="home_existing_project")
            folder = DEFAULT_BASE_DIR / selected
            st.caption("Project folder")
            st.code(str(folder), language="text")
            if st.button("Load Project", use_container_width=True, key="home_load_project"):
                st.session_state["project_root"] = str(folder)
                st.session_state["project_name"] = selected
                st.session_state["current_project"] = selected
                st.session_state["project_created_notice"] = {
                    "name": selected,
                    "workspace": folder.as_posix(),
                    "action": "loaded",
                }
                # A different project means a different run workspace; any
                # plan or result still on the page belongs to the old one.
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
            st.info("No existing project found yet.")
            st.caption("Project folder")
            st.code("runs/web_projects/<project_name>", language="text")
            st.button("Load Project", disabled=True, use_container_width=True, key="home_load_disabled")

    st.markdown("---")
    # One settings surface, shared with the sidebar and the Explain prompt.
    # This block used to carry its own three text inputs bound straight to
    # the session keys, which bypassed set_llm_settings entirely: the status
    # was never updated, and settings entered here did not register as
    # configured elsewhere. That is what made the app ask twice.
    render_inline_llm_settings("home", expanded=False)

    st.markdown("---")
    st.subheader("Quick Start", anchor=False)
    st.info("Run a built-in MATR RUL example to verify the full local BatteryML workflow: dataset → processed files → config → model run → metrics/results.")
    raw_path = Path("data/raw/MATR")
    processed_path = Path("data/processed/MATR")
    config_path = QUICKSTART_CONFIG
    st.markdown("##### Preflight checklist")
    checklist = pd.DataFrame([
        {"Item": "Raw data: data/raw/MATR", "Status": "exists" if raw_path.exists() else "missing"},
        {"Item": "Processed data: data/processed/MATR", "Status": "exists" if processed_path.exists() else "missing"},
        {"Item": "Config: configs/baselines/sklearn/xgb/matr_1.yaml", "Status": "exists" if config_path.exists() else "missing"},
        {"Item": "Device", "Status": "cpu"},
    ])
    st.dataframe(checklist, use_container_width=True, hide_index=True)
    st.caption("Dataset: MATR | Task: RUL | Model: XGBoost | Device: CPU")
    st.code(
        "\n".join([
            "python batteryml_web/core/run_batteryml_cli.py run",
            str(config_path),
            "--workspace <workspace_dir>",
            "--train",
            "--eval",
            "--device cpu",
            "--seed 0",
        ]),
        language="bash",
    )
    preflight_ok = raw_path.exists() and processed_path.exists() and config_path.exists()
    if not raw_path.exists():
        st.error("Raw dataset missing: data/raw/MATR")
    if not processed_path.exists():
        st.error("Processed dataset missing: data/processed/MATR. Please go to Prepare Data and run preprocessing first.")
        if st.button("Go Prepare Data", use_container_width=True, key="home_go_prepare_data"):
            set_current_step("prepare_data")
    if not config_path.exists():
        st.error("Config file missing: configs/baselines/sklearn/xgb/matr_1.yaml")
    q1, q2 = st.columns(2)
    with q1:
        if st.button("Load MATR Demo", use_container_width=True, key="home_load_matr_demo"):
            st.session_state["selected_dataset"] = "MATR"
            st.session_state["selected_task"] = "RUL"
            st.session_state["selected_model"] = "XGBoost"
            st.session_state["selected_config"] = str(QUICKSTART_CONFIG)
            st.session_state["selected_device"] = "cpu"
            st.session_state["quickstart_loaded_serial"] = (
                int(st.session_state.get("quickstart_loaded_serial", 0)) + 1
            )
    with q2:
        if st.button("Run MATR Demo", use_container_width=True, key="home_run_matr_demo", disabled=not preflight_ok):
            workspace_root = project_root or Path("runs/web_projects/default")
            workspace_dir = workspace_root / "workspace"
            log_path = workspace_root / "logs" / "matr_demo.log"
            result = run_command_with_progress(
                command=[
                    sys.executable,
                    "batteryml_web/core/run_batteryml_cli.py",
                    "run",
                    str(config_path),
                    "--workspace",
                    str(workspace_dir),
                    "--train",
                    "--eval",
                    "--device",
                    "cpu",
                    "--seed",
                    "0",
                ],
                cwd=REPO_ROOT,
                log_path=log_path,
                steps=[
                    (5, "Load MATR demo"),
                    (10, "Check raw data"),
                    (20, "Check processed data"),
                    (30, "Check config"),
                    (40, "Start BatteryML run"),
                    (80, "Running train/eval"),
                    (95, "Collect outputs"),
                    (100, "Demo completed"),
                ],
            )
            st.session_state["last_run_status"] = "success" if result["success"] else "failed"
            st.session_state["last_run_time"] = result["elapsed"]
            st.session_state["project_root"] = str(workspace_root)
            st.session_state["workspace_dir"] = str(workspace_dir)
            st.session_state["last_run_code"] = result["return_code"]
            if result["success"]:
                notify_success(
                    "Quick-start run completed",
                    event_key=EVENT_RUN_COMPLETED,
                    signature=["quickstart", str(workspace_dir), result["return_code"]],
                )
            st.session_state["current_step"] = "review_results"
            st.rerun()

    quickstart_serial = st.session_state.get("quickstart_loaded_serial")
    if quickstart_serial:
        notify_success(
            "MATR quick-start demo loaded into the workspace settings",
            event_key=EVENT_QUICKSTART_READY,
            signature=quickstart_serial,
        )

    st.markdown("---")
    render_home_agent_entry()

    st.markdown("---")
    if st.button("Start workflow →", use_container_width=True, key="home_start_workflow"):
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
    tabs = st.tabs([
        "Dataset Overview",
        "Preprocess & Inspect",
        "Custom Dataset / Schema Mapping",
    ])

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
                    <div class="pipeline-card" style="border-color: {color};">
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
                    "Run preprocessing",
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
                    "Run preprocessing",
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
                    btn_label,
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
                        st.markdown(
                            f"""
                            <div style="border:1px solid rgba(255,255,255,0.12); border-radius:14px; padding:12px; min-height:110px;">
                                <div style="font-weight:700; word-break:break-all;">{p.name}</div>
                                <div style="font-size:0.85rem; opacity:0.85; margin-top:6px;">Type: {p.suffix or 'unknown'}</div>
                                <div style="font-size:0.85rem; opacity:0.85;">Size: {stat.st_size} bytes</div>
                                <div style="font-size:0.75rem; opacity:0.75; word-break:break-all; margin-top:6px;">Path: {p}</div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                st.write("**Feature / cache / artifact files:**")
                st.code("\n".join(str(p) for p in feature_files[:30]), language="text")
                st.write("**Feature dimension:**")
                st.info("Feature information is not available yet. Run model or inspect BatteryML cache outputs.")
            else:
                st.info("Feature information is not available yet. Run model or inspect BatteryML cache outputs.")
            if st.button("Go to Run Model", use_container_width=True, key="prep_go_run_model"):
                set_current_step("run_model")

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
        if st.button("Save uploads", use_container_width=True, key="custom_save_uploads"):
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


def render_inline_llm_settings(location_key: str, expanded: bool = False):
    """The LLM settings, wherever a page offers them inline.

    One form, one credential store: this delegates to
    :func:`batteryml_web.ui.settings_panel.render_llm_settings_form`, which is
    the same form the sidebar and the Explain prompt show. It used to carry
    its own copy of the fields and its own Save/Test handling — and that copy
    had a real defect: ``test_llm_connection`` *returns* ``(ok, message)`` and
    does not raise, so its ``except Exception`` never fired and a failed test
    was reported as "connected". Delegating removes the second
    implementation and the defect with it.
    """
    with st.expander("Optional LLM Assistant settings", expanded=expanded):
        st.caption(
            "LLM assistant is optional. Without an LLM endpoint, the platform "
            "still runs the deterministic BatteryML workflow."
        )
        # The form prints the status itself, from the one status vocabulary.
        render_llm_settings_form(key_prefix=location_key)


def set_llm_not_configured_message(output_key: str):
    st.session_state[output_key] = (
        "LLM assistant is not configured yet.\n\n"
        "Please add Base URL, API key, and model in the Optional LLM Assistant settings below, then retry this action."
    )


def require_llm_or_show_message(output_key: str) -> bool:
    if not is_llm_configured():
        set_llm_not_configured_message(output_key)
        return False
    return True


def _set_llm_output(output_key: str, text: str, save_path: Path | None = None) -> None:
    st.session_state[output_key] = text
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
    status_label = {
        "connected": "Connected",
        "failed": "Connection failed",
        "configured": "Configured / Not tested",
    }.get(status, "Not connected")
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


def render_llm_output_panel(output_key: str, title: str) -> None:
    output = st.session_state.get(output_key)
    if not output:
        return
    st.markdown("---")
    with st.expander(f"🤖 LLM Assistant Output · {title}", expanded=True):
        st.caption("Suggestion only. This output does not automatically modify raw data, config files, predictions, or metrics.")
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
            "Go to Prepare Data",
            use_container_width=True,
            key="run_empty_go_prepare_data",
        ):
            set_current_step("prepare_data")
    with home_col:
        if st.button(
            "Go to Home",
            use_container_width=True,
            key="run_empty_go_home",
        ):
            set_current_step("home")
    render_bottom_step_navigation()


def render_run_model_page():
    st.header("🚀 Run Model", anchor=False)
    st.caption("Configure dataset, prediction task, model, device, and execute the deterministic BatteryML workflow.")
    configs = _discover_run_configs()
    datasets = _ui_dataset_options()
    tasks = ["RUL prediction", "SOH estimation"]
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
                st.markdown(f"<div style='border:1px solid {color}; border-radius:14px; padding:12px; margin-bottom:10px;'><b>{ds}</b><br>Raw files count: {info['raw_files']}<br>Processed files count: {info['processed_files']}<br><b>{status}</b></div>", unsafe_allow_html=True)
                if st.button(f"Select {ds}", key=f"run_select_{ds}", use_container_width=True):
                    st.session_state["selected_dataset"] = ds
                    st.rerun()
        if st.button("Go Prepare Data", use_container_width=True, key="run_go_prepare_data"):
            set_current_step("prepare_data")

    with tabs[0]:
        left, right = st.columns([1.15, 0.95])
        with left:
            ds_idx = datasets.index(st.session_state.get("selected_dataset", "MATR")) if st.session_state.get("selected_dataset", "MATR") in datasets else 0
            ds = st.selectbox("Dataset", datasets, index=ds_idx, key="run_dataset_select")
            st.session_state["selected_dataset"] = ds
            task_choice = st.radio("Task", tasks, index=0 if st.session_state.get("selected_task", "RUL") == "RUL" else 1, key="run_task_radio", horizontal=True)
            st.session_state["selected_task"] = task_choice
            available_models = sorted({entry.model_key for entry in get_config_registry(str(REPO_ROOT)).get_entries(ds, task_choice)})
            if not available_models:
                st.warning("No configured model is available for this dataset/task. Please add a valid YAML config.")
                available_models = []
            model = st.selectbox("Model", options=available_models or [""], index=0, key="run_model_select", disabled=not available_models)
            st.session_state["selected_model"] = model if available_models else ""
            device = st.selectbox("Device", ["cpu", "cuda"], index=0 if st.session_state.get("selected_device", "cpu") == "cpu" else 1, key="run_device_select")
            if device == "cuda" and not torch.cuda.is_available():
                st.warning("CUDA is not available in the current environment. Falling back to CPU.")
                device = "cpu"
            st.session_state["selected_device"] = device
            seed = st.number_input("Seed", min_value=0, value=int(st.session_state.get("run_seed", 0)), step=1, key="run_seed_input")
            st.session_state["run_seed"] = int(seed)
        with right:
            config_paths = configs.get((ds, task_choice, model), [])
            config_path = config_paths[0] if config_paths else None
            workspace_dir = (_project_root() or Path("runs/web_projects/default")) / "workspace"
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
            if not is_llm_configured():
                st.caption("Need LLM help? Configure assistant below.")
                render_inline_llm_settings("run_model", expanded=False)
            c1, c2 = st.columns(2)
            with c1:
                if st.button("Explain this config", key="run_model_explain_config", use_container_width=True):
                    if require_llm_or_show_message("run_model_config_llm_output"):
                        try:
                            with st.spinner("Calling LLM assistant..."):
                                response = run_llm_prompt(
                                    f"Explain this BatteryML config: {config_path}. Return suggestion only.",
                                    base_url=st.session_state["llm_base_url"],
                                    api_key=st.session_state["llm_api_key"],
                                    model=st.session_state["llm_model"],
                                    system_prompt="You are an assistant for a BatteryML web protocol platform. Provide suggestions only. Do not modify raw data, metrics, predictions, or config files.",
                                )
                            st.session_state["llm_status"] = "connected"
                            st.session_state["llm_status_message"] = "LLM request completed."
                        except Exception as exc:
                            response = f"LLM request failed: {exc}"
                            st.session_state["llm_status"] = "failed"
                            st.session_state["llm_status_message"] = response
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
                                        task="RUL prediction",
                                        model=model,
                                        config_path=Path(str(config_path)) if config_path else Path("<missing-config>"),
                                        config_text=(Path(str(config_path)).read_text(encoding="utf-8", errors="ignore") if config_path and Path(str(config_path)).exists() else "Config file missing."),
                                        dataset_readiness={"ready": info["ready"], "raw_exists": info["raw_exists"], "processed_exists": info["processed_exists"]},
                                        available_configs=[f"{k[0]} | {k[1]} | {k[2]} -> {v}" for k, v in list(configs.items())],
                                    ),
                                    base_url=st.session_state["llm_base_url"],
                                    api_key=st.session_state["llm_api_key"],
                                    model=st.session_state["llm_model"],
                                    system_prompt="You are an assistant for a BatteryML web protocol platform. Provide suggestions only. Do not modify raw data, metrics, predictions, or config files.",
                                )
                            st.session_state["llm_status"] = "connected"
                            st.session_state["llm_status_message"] = "LLM request completed."
                        except Exception as exc:
                            response = f"LLM request failed: {exc}"
                            st.session_state["llm_status"] = "failed"
                            st.session_state["llm_status_message"] = response
                        _set_llm_output(
                            "run_model_config_llm_output",
                            response,
                            (_project_root() or Path("runs/web_projects/default")) / "agent_outputs" / "model_config_suggestion.md",
                        )
            render_run_model_llm_output_panel("run_model_config_llm_output", "Config guidance", "Config guidance", "Config assistant")

    with tabs[2]:
        ds = st.session_state.get("selected_dataset", "MATR")
        task_choice = st.session_state.get("selected_task", "RUL prediction")
        model = st.session_state.get("selected_model", "")
        config_paths = configs.get((ds, task_choice, model), [])
        config_path = config_paths[0] if config_paths else None
        workspace_dir = (_project_root() or Path("runs/web_projects/default")) / "workspace"
        st.write(f"**Dataset:** {ds}")
        st.write(f"**Task:** {task_choice}")
        st.write(f"**Model:** {model if model else 'Not configured'}")
        st.write(f"**Dataset readiness:** {_current_dataset_status(ds)}")
        st.write(f"**Config path:** `{config_path if config_path else 'Not configured'}`")
        st.write(f"**Workspace dir:** `{workspace_dir}`")
        command = [sys.executable, "batteryml_web/core/run_batteryml_cli.py", "run", str(config_path) if config_path else "<missing-config>", "--workspace", str(workspace_dir), "--train", "--eval", "--device", st.session_state.get("selected_device", "cpu"), "--seed", str(st.session_state.get("run_seed", 0))]
        st.code(" ".join(command), language="bash")
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
        can_run = config_path is not None and Path(str(config_path)).exists() and readiness.get(ds, {}).get("ready", False)
        if not readiness.get(ds, {}).get("ready", False):
            st.warning("Processed dataset missing. Go to Prepare Data and run preprocessing first.")
        if config_path is None or not Path(str(config_path)).exists():
            st.warning("Config missing for the selected dataset/model combination.")
        with c2:
            if st.button("Run BatteryML", disabled=not can_run, use_container_width=True, key="run_batteryml_btn"):
                log_path = (_project_root() or Path("runs/web_projects/default")) / "logs" / "run_model.log"
                result = run_command_with_progress(command=command, cwd=REPO_ROOT, log_path=log_path, steps=[(5, "Initialize"), (10, "Check dataset"), (20, "Load config"), (30, "Start BatteryML run"), (70, "Training/evaluation running"), (90, "Collect outputs"), (100, "Done")])
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
                    if st.button("Review Results →", use_container_width=True, key="run_go_results_after_run"):
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
        st.code(log_tail or "Not available yet", language="text")
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


def _selected_run_manifest():
    """The run the user opened in Run History, or None.

    Returned to the Predictions and Files tabs so they describe *that run's*
    artifacts rather than whatever the workspace glob happened to find. None
    means nothing is selected — the tabs then fall back to the workspace and
    say so, instead of silently showing a different run's files.
    """
    selected = st.session_state.get("run_history_selection")
    if not selected:
        return None
    project = resolve_project(
        project_root=st.session_state.get("project_root"),
        project_name=st.session_state.get("project_name")
        or st.session_state.get("current_project"),
    )
    return find_run(
        project["root"], str(selected), live_run_ids=get_run_manager().live_run_ids()
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
    with st.expander(f"Run {selected}", expanded=True):
        # Opening a historical run shows it. It never re-runs it.
        render_run_detail(manifest)


def render_review_results_page():
    st.header("📈 Review Results", anchor=False)

    # Run history first: a project holds many runs, and finding an earlier
    # one should not mean searching the filesystem by hand. Read from the
    # run manifests, so it survives a refresh and an app restart.
    _render_project_run_history()

    st.markdown("#### Optional LLM interpretation")
    st.caption(
        "Use the LLM assistant to summarize results or explain metrics. BatteryML metrics are computed by BatteryML and will not be modified."
    )

    project_dir = _project_root() or Path("runs/native_matr_xgb_test")
    session_workspace = Path(st.session_state.get("workspace_dir", "")) if st.session_state.get("workspace_dir") else None
    fallback_workspace = Path("runs/native_matr_xgb_test")
    searched_paths = [project_dir, project_dir / "workspace"]
    if session_workspace:
        searched_paths.append(session_workspace)
    searched_paths.append(fallback_workspace)
    files = find_result_files(project_dir, session_workspace or project_dir / "workspace")
    metrics_df = load_metrics(files["metrics_files"])
    if metrics_df is None or metrics_df.empty:
        metrics_df = load_metrics_from_logs(files["log_files"])
    preds_df = load_predictions(files["prediction_files"])
    metrics_source = "metrics file" if files["metrics_files"] and metrics_df is not None else ("log file" if metrics_df is not None else "missing")

    btn_col1, btn_col2, btn_col3 = st.columns([1, 1, 1])
    with btn_col1:
        summarize_clicked = st.button("Summarize results", key="review_summarize_results", use_container_width=True)
    with btn_col2:
        explain_clicked = st.button("Explain metrics", key="review_explain_metrics", use_container_width=True)
    with btn_col3:
        configure_clicked = st.button("Configure LLM", key="review_configure_llm", use_container_width=True)

    if configure_clicked:
        st.session_state["review_show_llm_settings"] = True

    if summarize_clicked:
        if not is_llm_configured():
            set_llm_not_configured_message("review_llm_output")
            st.session_state["review_show_llm_settings"] = True
        else:
            prompt = f'''
You are assisting with a BatteryML result review.
Summarize the available metrics, predictions, logs, and result files.
Summarize run status, explain model performance briefly, mention missing files if any, and do not invent missing results or modify metrics.
Project: {_current_project_name()}
Workspace: {st.session_state.get("workspace_dir", "Not available")}
Metrics table:
{metrics_df.to_string() if metrics_df is not None else "No metrics available"}
Predictions available: {'Yes' if preds_df is not None and not preds_df.empty else 'No'}
Logs available: {'Yes' if files.get("log_files") else 'No'}
Reports available: {'Yes' if files.get("report_files") else 'No'}
'''
            try:
                response = run_llm_prompt(
                    prompt,
                    base_url=st.session_state["llm_base_url"],
                    api_key=st.session_state["llm_api_key"],
                    model=st.session_state["llm_model"],
                    system_prompt="You are an assistant for a BatteryML web protocol platform. Provide suggestions only. Do not modify raw data, metrics, predictions, or config files.",
                )
                st.session_state["llm_status"] = "connected"
                st.session_state["llm_status_message"] = "LLM request completed."
            except Exception as exc:
                response = f"LLM request failed: {exc}"
                st.session_state["llm_status"] = "failed"
                st.session_state["llm_status_message"] = response
            st.session_state["review_llm_output"] = response

    if explain_clicked:
        if not is_llm_configured():
            set_llm_not_configured_message("review_llm_output")
            st.session_state["review_show_llm_settings"] = True
        else:
            prompt = f'''
Explain the following BatteryML evaluation metrics in plain language.
Focus on what RMSE, MAE, and MAPE mean for RUL prediction, what lower or higher values mean, and whether metrics are available or missing.
Do not recalculate or alter the metric values.
Metrics:
{metrics_df.to_string() if metrics_df is not None else "No metrics available"}
'''
            try:
                response = run_llm_prompt(
                    prompt,
                    base_url=st.session_state["llm_base_url"],
                    api_key=st.session_state["llm_api_key"],
                    model=st.session_state["llm_model"],
                    system_prompt="You are an assistant for a BatteryML web protocol platform. Provide suggestions only. Do not modify raw data, metrics, predictions, or config files.",
                )
                st.session_state["llm_status"] = "connected"
                st.session_state["llm_status_message"] = "LLM request completed."
            except Exception as exc:
                response = f"LLM request failed: {exc}"
                st.session_state["llm_status"] = "failed"
                st.session_state["llm_status_message"] = response
            st.session_state["review_llm_output"] = response

    if st.session_state.get("review_show_llm_settings") or not is_llm_configured():
        render_inline_llm_settings("review_results", expanded=st.session_state.get("review_show_llm_settings", False))

    render_llm_output_panel("review_llm_output", "Result interpretation")

    tabs = st.tabs(["Summary", "Metrics", "Predictions", "Logs", "Files"])

    with tabs[0]:
        st.write(f"**Project:** {_current_project_name()}")
        st.write(f"**Project dir:** {project_dir}")
        st.write(f"**Workspace dir:** {session_workspace or (project_dir / 'workspace')}")
        st.write("**Searched result locations:**")
        for p in searched_paths:
            st.code(str(p), language="text")
        st.write(f"**Last run:** {st.session_state.get('last_run_status', 'N/A')}")
        st.write(f"**Metrics found:** {'Yes' if metrics_df is not None and not metrics_df.empty else 'No'}")
        st.write(f"**Predictions found:** {'Yes' if files['prediction_files'] else 'No'}")
        st.write(f"**Logs found:** {'Yes' if files['log_files'] else 'No'}")
        st.write(f"**Metrics source:** {metrics_source}")

    with tabs[1]:
        if metrics_df is not None and not metrics_df.empty:
            st.dataframe(metrics_df, use_container_width=True, hide_index=True)
            st.caption(f"Metrics source: {metrics_source}")
        else:
            st.info("Not available")
            st.write("**Searched paths:**")
            for p in searched_paths:
                st.code(str(p), language="text")
            st.write("**Next action:** Run training or inspect logs for RMSE / MAE / MAPE output.")

    with tabs[2]:
        # A selected run's predictions, described through the shared reader.
        # The old branch below only understood a CSV with guessable column
        # names and called every .pkl "parser not implemented" — which is
        # almost every artifact this platform writes.
        selected_manifest = _selected_run_manifest()
        if selected_manifest is not None:
            render_predictions_panel(selected_manifest)
        elif preds_df is not None and not preds_df.empty:
            st.dataframe(preds_df, use_container_width=True, hide_index=True)
            cols = {c.lower(): c for c in preds_df.columns}
            y_true_col = next((cols[c] for c in cols if c in {"y_true", "true", "gt", "label", "ground_truth"}), None)
            y_pred_col = next((cols[c] for c in cols if c in {"y_pred", "pred", "prediction", "predictions"}), None)
            if y_true_col and y_pred_col:
                plot_df = preds_df[[y_true_col, y_pred_col]].rename(columns={y_true_col: "GT", y_pred_col: "Pred"})
                st.scatter_chart(plot_df)
                st.caption("GT vs Pred")
            else:
                st.info("Prediction file found but parser not implemented for this format.")
                for p in files["prediction_files"]:
                    st.code(str(p), language="text")
        elif files["prediction_files"]:
            # No run selected in history, but artifacts exist on disk: preview
            # the first one through the same reader rather than declaring it
            # unparseable.
            render_predictions_panel(
                {"artifacts": {"prediction_files": [str(p) for p in files["prediction_files"]]},
                 "run_id": "workspace"}
            )
        else:
            st.info("No predictions artifact was found.")
            st.write("**Searched paths:**")
            for p in searched_paths:
                st.code(str(p), language="text")
            st.write("**Next action:** Run training and re-open Review Results.")

    with tabs[3]:
        if files["log_files"]:
            for log_file in files["log_files"]:
                st.markdown(f"**{log_file.name}**")
                try:
                    text = log_file.read_text(encoding="utf-8", errors="ignore")
                except Exception:
                    text = "<unable to read log>"
                lines = text.splitlines()
                st.code("\n".join(lines[-120:]) if lines else "", language="text")
        else:
            st.info("Not available")
        render_llm_output_panel("review_llm_output", "Result interpretation")

    with tabs[4]:
        selected_manifest = _selected_run_manifest()
        if selected_manifest is not None:
            # Where it wrote, then the files grouped by what they are.
            render_run_location(selected_manifest)
            run_dir = str(
                selected_manifest.get("run_dir")
                or selected_manifest.get("workspace")
                or ""
            )
            if run_dir:
                render_file_browser(run_dir)
        else:
            workspace = str(session_workspace or (project_dir / "workspace"))
            st.markdown("**Where results are being read from**")
            st.dataframe(
                [
                    {"": "Project", "Path": _current_project_name()},
                    {"": "Project directory", "Path": str(project_dir)},
                    {"": "Workspace", "Path": workspace},
                ],
                hide_index=True,
                use_container_width=True,
            )
            st.caption(
                "Select a run in Run History above to browse that run's own "
                "files, grouped by artifact type."
            )
            render_file_browser(workspace)
        render_llm_output_panel("review_llm_output", "Result interpretation")

    render_bottom_step_navigation()


# The sidebar lives in batteryml_web/ui/workflow_sidebar.py. A second copy
# used to be defined here and shadowed the import, so the grouped navigation
# in that module was dead code.
render_sidebar_workflow()

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
