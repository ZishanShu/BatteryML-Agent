from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

APP_DIR = Path(__file__).resolve().parent
REPO_ROOT = APP_DIR.parent
for _path in (REPO_ROOT, APP_DIR):
    _str_path = str(_path)
    if _str_path not in sys.path:
        sys.path.insert(0, _str_path)

import pandas as pd
import streamlit as st
import yaml

from agent.llm_client import LLMClient
from agent.result_agent import explain_results
from agent.schema_agent import suggest_schema_mapping
from agent.troubleshoot_agent import diagnose_failure, run_validator_after_fix
from core.batteryml_runner import run_batteryml, run_dry_run
from core.config_builder import build_quickstart_config, build_task_config, save_config_yaml
from core.config_validator import validate_config
from core.dataset_tools import inspect_tabular_file
from core.gpu_checker import get_gpu_readiness_report, validate_device_choice, list_available_configs, get_gpu_demo_configs
from core.report_builder import generate_html_report
from core.results_reader import discover_metrics_artifacts, discover_prediction_artifacts, list_figures, list_workspace_files, read_log, read_metrics, read_predictions
from core.schema_validator import OPTIONAL_FIELDS, REQUIRED_FIELDS, validate_schema_mapping
from core.state import ensure_project_state, get_project_root, save_project_state, set_project_mode
from core.catalog import get_catalog, TaskType, Status
from core.dataset_inspector import (
    inspect_dataset,
    list_dataset_files,
    preview_file,
    format_file_table,
    detect_file_type,
)
from ui.progress_utils import get_safe_runtime_env, render_staged_progress, run_command_with_progress

APP_TITLE = "LLM-guided BatteryML Protocol Platform"
DEFAULT_BASE_DIR = Path("runs") / "web_projects"
QUICKSTART_CONFIG = Path("configs/baselines/sklearn/xgb/matr_1.yaml")
SUPPORTED_DATASETS = ["MATR"]
SUPPORTED_TASKS = ["RUL", "SOH"]
SUPPORTED_MODELS = ["XGBoostRULPredictor", "DummyRULPredictor"]
SUPPORTED_SPLITS = ["PRIMARY"]

# Page config is defined once in batteryml_web/app.py before importing this module.

# Initialize session state
if "llm_base_url" not in st.session_state:
    st.session_state["llm_base_url"] = ""
if "llm_api_key" not in st.session_state:
    st.session_state["llm_api_key"] = ""
if "llm_model" not in st.session_state:
    st.session_state["llm_model"] = ""
if "active_section" not in st.session_state:
    st.session_state["active_section"] = "project"
if "current_step" not in st.session_state:
    st.session_state["current_step"] = "home"
if "project_root" not in st.session_state:
    st.session_state["project_root"] = None
if "project_state" not in st.session_state:
    st.session_state["project_state"] = {}
if "selected_dataset" not in st.session_state:
    st.session_state["selected_dataset"] = "MATR"
if "selected_task" not in st.session_state:
    st.session_state["selected_task"] = "RUL"
if "selected_model" not in st.session_state:
    st.session_state["selected_model"] = "XGBoostRULPredictor"
if "selected_config" not in st.session_state:
    st.session_state["selected_config"] = None
if "selected_device" not in st.session_state:
    st.session_state["selected_device"] = "cpu"

st.markdown("""
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

.block-container {
    max-width: 1280px;
    padding-top: 1.4rem !important;
    padding-bottom: 2rem !important;
}

.app-title {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    font-size: clamp(2rem, 4vw, 2.55rem);
    font-weight: 800;
    line-height: 1.12;
    letter-spacing: -0.03em;
    margin-bottom: 0.35rem;
}

.app-subtitle {
    opacity: 0.88;
    font-size: 1.05rem;
    line-height: 1.55;
    margin-bottom: 0.85rem;
}

.app-card,
.project-bar,
.step-card,
.workflow-card,
div[data-testid="stMetric"],
div[data-testid="stDataFrame"] {
    border: 1px solid var(--app-border-soft);
    border-radius: var(--app-radius);
    background: var(--app-card-bg);
    box-shadow: var(--app-shadow);
}

.project-bar {
    display: grid;
    grid-template-columns: 1.4fr 1fr 1fr 1fr 1fr;
    gap: 0.75rem;
    padding: 0.75rem 0.9rem;
    margin: 0.75rem 0 1rem 0;
}

.project-bar div {
    min-width: 0;
    font-size: 0.86rem;
    opacity: 0.9;
    overflow-wrap: anywhere;
}

.project-bar b {
    opacity: 0.96;
    font-size: 0.78rem;
    text-transform: uppercase;
    letter-spacing: 0.02em;
}

.step-card,
.workflow-card {
    padding: 0.85rem;
    margin-bottom: 0.7rem;
}

.workflow-card.current {
    border-color: rgba(59, 130, 246, 0.72);
    background: rgba(59, 130, 246, 0.12);
}

.status-pill {
    display: inline-block;
    padding: 0.16rem 0.5rem;
    border-radius: 999px;
    font-size: 0.74rem;
    font-weight: 700;
    border: 1px solid var(--app-border-soft);
}

.status-ready {
    background: rgba(34, 197, 94, 0.16);
    color: inherit;
}

.status-action {
    background: rgba(245, 158, 11, 0.18);
    color: inherit;
}

.status-missing {
    background: rgba(239, 68, 68, 0.16);
    color: inherit;
}

h1, h2, h3 {
    letter-spacing: 0 !important;
}

h3 {
    font-size: 1.1rem !important;
    font-weight: 650 !important;
}

p, .stCaption, div[data-testid="stCaptionContainer"] {
    opacity: 0.88;
}

code, pre, div[data-testid="stCodeBlock"] {
    white-space: pre-wrap !important;
    word-break: break-word !important;
    font-size: 0.85rem !important;
    border-radius: 12px !important;
}

.stTextInput input,
.stTextArea textarea,
div[data-baseweb="select"] > div {
    border-color: var(--app-border) !important;
}

.stButton > button {
    border-radius: 10px;
    border-color: var(--app-border) !important;
}

.stButton > button:hover {
    border-color: rgba(59, 130, 246, 0.7) !important;
}

div[data-testid="stMetric"] {
    padding: 0.55rem 0.75rem;
}

div[data-testid="stMetricValue"] {
    font-size: 1.1rem !important;
    line-height: 1.2 !important;
    font-weight: 650 !important;
}

div[data-testid="stMetricLabel"] {
    font-size: 0.8rem !important;
    opacity: 0.86;
}

hr {
    border-color: var(--app-border-soft) !important;
}
</style>
""", unsafe_allow_html=True)

# Compact title
st.markdown("<div class='app-title'>🔋 LLM-guided BatteryML Protocol Platform</div>", unsafe_allow_html=True)
st.markdown("<div class='app-subtitle'>BatteryML performs numerical prediction. The optional LLM agent assists workflow guidance only.</div>", unsafe_allow_html=True)


def _ensure_project(project_name: str, base_dir: str, mode: str) -> Path:
    project_root = get_project_root(project_name, Path(base_dir))
    state = ensure_project_state(project_root=project_root, project_name=project_name, mode=mode)
    set_project_mode(project_root, mode)
    st.session_state["project_root"] = str(project_root)
    st.session_state["project_state"] = state
    return project_root


def _load_state(project_root: Path) -> Dict[str, Any]:
    state_path = project_root / "project_state.json"
    if state_path.exists():
        return json.loads(state_path.read_text(encoding="utf-8"))
    return {}


def _path_block(label: str, path: str | Path | None) -> None:
    value = str(path) if path else "Not available"
    st.markdown(f"**{label}**  \n`{value}`")


def _status_badge(status: str) -> str:
    mapping = {
        "success": "success",
        "running": "warning",
        "failed": "error",
        "initialized": "secondary",
        "unknown": "secondary",
    }
    return mapping.get(status, "secondary")


def _open_command(path: Path) -> str:
    return f"open {path}"


def _copyable_path(path: Path) -> None:
    st.code(str(path), language="text")
    st.caption(f"open folder command: `{_open_command(path)}`")


def _dataset_status(dataset: str) -> Dict[str, Any]:
    raw_path = Path("data/raw") / dataset
    processed_path = Path("data/processed") / dataset
    config_path = Path("configs/baselines/sklearn/xgb") / f"{dataset.lower()}_1.yaml"
    return {
        "Dataset": dataset,
        "Raw path": str(raw_path),
        "Raw exists": raw_path.exists(),
        "Processed path": str(processed_path),
        "Processed exists": processed_path.exists(),
        "Ready for training": raw_path.exists() and processed_path.exists() and config_path.exists(),
        "Action": "Run preprocessing" if not processed_path.exists() else "Ready",
        "config_path": config_path,
    }


def _run_process(command: List[str], cwd: Path) -> Dict[str, Any]:
    started = time.perf_counter()
    proc = subprocess.run(command, cwd=str(cwd), capture_output=True, text=True)
    elapsed = time.perf_counter() - started
    return {
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
        "elapsed_seconds": elapsed,
    }


def _load_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}


def _config_details(config_path: Path) -> Dict[str, Any]:
    cfg = _load_yaml(config_path)
    return {
        "Dataset": cfg.get("dataset") or cfg.get("train_test_split", {}).get("cell_data_path", "MATR"),
        "Train/test splitter": cfg.get("train_test_split", {}).get("name") if isinstance(cfg.get("train_test_split"), dict) else cfg.get("train_test_split"),
        "Label annotator": cfg.get("label", {}).get("name") if isinstance(cfg.get("label"), dict) else cfg.get("label"),
        "Feature extractor": cfg.get("feature", {}).get("name") if isinstance(cfg.get("feature"), dict) else cfg.get("feature"),
        "Model class": cfg.get("model", {}).get("name") if isinstance(cfg.get("model"), dict) else cfg.get("model"),
        "Task": cfg.get("task") or ("RUL" if "RUL" in str(cfg.get("label", {})) else "SOH"),
        "Seed": cfg.get("seed", 0),
        "Device": cfg.get("device", "cpu"),
    }


def _extract_log_excerpts(log_text: str) -> Dict[str, str]:
    excerpts = {}
    for key, pattern in {"feature": r"(?im)^.*feature.*$", "model": r"(?im)^.*model.*$", "training": r"(?im)^.*training.*$"}.items():
        lines = re.findall(pattern, log_text)
        if lines:
            excerpts[key] = "\n".join(lines[:5])
    return excerpts


project_name = st.session_state.get("project_name_input", "my_battery_project")
base_dir = str(DEFAULT_BASE_DIR)
llm_client = LLMClient(
    base_url=st.session_state.get("llm_base_url") or None,
    api_key=st.session_state.get("llm_api_key") or None,
    model=st.session_state.get("llm_model") or None,
)

# Helper function to get project status
def get_project_status():
    if st.session_state["project_root"] is None:
        return None, "Not set", "⚪"
    root = Path(st.session_state["project_root"])
    name = root.name
    return root, name, "🟢"

def get_dataset_status():
    ds = st.session_state.get("selected_dataset", "MATR")
    raw_exists = Path(f"data/raw/{ds}").exists()
    proc_exists = Path(f"data/processed/{ds}").exists()
    if proc_exists:
        return f"{ds} Ready", "status-ready"
    elif raw_exists:
        return f"{ds} Need prep", "status-action"
    return f"{ds} Missing", "status-missing"

def get_current_step():
    ds_status, _ = get_dataset_status()
    if "Missing" in ds_status or "Need prep" in ds_status:
        return "Prepare Data", "prepare"
    if st.session_state.get("last_run_status") == "success":
        return "View Results", "results"
    if st.session_state.get("selected_config"):
        return "Run Model", "run"
    return "Visualize Features", "features"

NAV_STEPS = [
    {
        "key": "home",
        "label": "Home",
        "emoji": "🏠",
        "description": "Create/load project and run quick-start demo",
    },
    {
        "key": "prepare_data",
        "label": "Prepare Data",
        "emoji": "🧪",
        "description": "Import and preprocess data",
    },
    {
        "key": "visualize_features",
        "label": "Visualize Features",
        "emoji": "📊",
        "description": "Inspect features and artifacts",
    },
    {
        "key": "run_model",
        "label": "Run Model",
        "emoji": "🚀",
        "description": "Configure and execute model",
    },
    {
        "key": "review_results",
        "label": "Review Results",
        "emoji": "📈",
        "description": "Review metrics and reports",
    },
]

WORKFLOW_STEPS = NAV_STEPS[1:]

NAV_TO_SECTION = {
    "prepare_data": "prepare",
    "visualize_features": "features",
    "run_model": "run",
    "review_results": "results",
}

SECTION_TO_NAV = {value: key for key, value in NAV_TO_SECTION.items()}


def set_current_step(step_key: str):
    if step_key not in {s["key"] for s in NAV_STEPS}:
        step_key = "home"
    st.session_state["current_step"] = step_key
    st.session_state["active_section"] = NAV_TO_SECTION.get(step_key, "project")
    st.rerun()


def get_current_step_index() -> int:
    current = st.session_state.get("current_step", "home")
    keys = [s["key"] for s in NAV_STEPS]
    if current not in keys:
        st.session_state["current_step"] = "home"
        st.session_state["active_section"] = "project"
        return 0
    return keys.index(current)


def render_sidebar_workflow():
    with st.sidebar:
        st.markdown("### 🔋 BatteryML")
        st.caption("Protocol Navigator")

        current_project = st.session_state.get("project_name") or st.session_state.get("current_project") or "No project loaded"
        if st.session_state.get("project_root"):
            current_project = Path(st.session_state["project_root"]).name
        st.markdown("**Project**")
        st.caption(str(current_project))

        st.divider()

        current_step = st.session_state.get("current_step", "home")

        for step in NAV_STEPS:
            label = f"{step['emoji']} {step['label']}"
            is_current = current_step == step["key"]
            if is_current:
                st.markdown(f"**▶ {label}**")
                st.caption(step["description"])
            else:
                if st.button(label, key=f"sidebar_go_{step['key']}", use_container_width=True):
                    set_current_step(step["key"])

        st.divider()

        idx = get_current_step_index()
        prev_col, next_col = st.columns(2)
        with prev_col:
            if st.button("← Prev", disabled=(idx == 0), use_container_width=True, key="sidebar_prev_step"):
                set_current_step(NAV_STEPS[idx - 1]["key"])
        with next_col:
            if st.button("Next →", disabled=(idx == len(NAV_STEPS) - 1), use_container_width=True, key="sidebar_next_step"):
                set_current_step(NAV_STEPS[idx + 1]["key"])

        st.divider()
        llm_connected = bool(st.session_state.get("llm_base_url")) and bool(st.session_state.get("llm_api_key"))
        st.caption(f"LLM: {'Connected' if llm_connected else 'Optional / Not connected'}")


def go_to_step(step_name: str) -> None:
    set_current_step(step_name)


def render_step_navigation(current_step: str | None = None, compact: bool = False) -> None:
    idx = get_current_step_index()
    prev_label = "← Prev" if compact else "← Previous step"
    next_label = "Next →" if compact else "Next step →"
    prev_col, next_col = st.columns(2)
    with prev_col:
        if st.button(prev_label, use_container_width=True, disabled=idx == 0, key=f"prev_step_btn_{'compact' if compact else 'main'}"):
            set_current_step(NAV_STEPS[idx - 1]["key"])
    with next_col:
        if st.button(next_label, use_container_width=True, disabled=idx == len(NAV_STEPS) - 1, key=f"next_step_btn_{'compact' if compact else 'main'}"):
            set_current_step(NAV_STEPS[idx + 1]["key"])


# Project Bar (if project exists)
root, name, icon = get_project_status()
if root:
    ds_status, ds_class = get_dataset_status()
    step, step_key = get_current_step()
    last_run = st.session_state.get("last_run_status", "N/A")
    llm_status = "🟢" if llm_client.is_configured() else "⚪"

    st.markdown(f"""
    <div class="project-bar">
      <div><b>Project</b><br>{name}</div>
      <div><b>Dataset</b><br>{ds_status}</div>
      <div><b>Current Step</b><br>{step}</div>
      <div><b>Last Run</b><br>{last_run}</div>
      <div><b>LLM</b><br>{llm_status}</div>
    </div>
    """, unsafe_allow_html=True)

def render_home_page():
    st.markdown("---")
    st.markdown("#### Create or Load Project")
    st.caption("Start a new local project or continue an existing one.")

    create_col, load_col = st.columns(2, gap="large")
    existing_projects = []
    if DEFAULT_BASE_DIR.exists():
        existing_projects = sorted([d.name for d in DEFAULT_BASE_DIR.iterdir() if d.is_dir()])
    load_has_projects = bool(existing_projects)

    with create_col:
        st.markdown("**Create New Project**")
        project_name_input = st.text_input("Project name", value="my_battery_project", key="project_name_input")
        project_folder = Path("runs/web_projects") / (project_name_input.strip() or "my_battery_project")
        st.caption("Project folder")
        st.code(str(project_folder), language="text")
        if st.button("Create New Project", use_container_width=True, key="create_new_project_main"):
            project_root = _ensure_project(project_name_input, str(DEFAULT_BASE_DIR), "deterministic")
            st.session_state["project_root"] = str(project_root)
            st.session_state["project_name"] = project_root.name
            st.session_state["current_project"] = project_root.name
            st.session_state["current_step"] = "prepare_data"
            st.session_state["active_section"] = "prepare"
            st.rerun()

    with load_col:
        st.markdown("**Load Existing Project**")
        if load_has_projects:
            selected_existing = st.selectbox("Select existing project", options=existing_projects, key="load_existing_project_select")
            selected_folder = Path("runs/web_projects") / selected_existing
            st.caption("Project folder")
            st.code(str(selected_folder), language="text")
            if st.button("Load Project", use_container_width=True, key="load_existing_project_main"):
                root = DEFAULT_BASE_DIR / selected_existing
                st.session_state["project_root"] = str(root)
                st.session_state["project_name"] = selected_existing
                st.session_state["current_project"] = selected_existing
                st.session_state["project_state"] = _load_state(root)
                st.session_state["current_step"] = "prepare_data"
                st.session_state["active_section"] = "prepare"
                st.rerun()
        else:
            st.info("No existing project found yet.")
            st.caption("Project folder")
            st.code("runs/web_projects/<project_name>", language="text")
            st.button("Load Project", use_container_width=True, key="load_existing_disabled", disabled=True)

    st.markdown("---")
    with st.expander("Optional LLM Assistant", expanded=False):
        st.caption("LLM assistant is optional. Without an LLM endpoint, the platform still runs the deterministic BatteryML workflow.")
        st.text_input("Base URL", key="llm_base_url")
        st.text_input("API key", key="llm_api_key", type="password")
        st.text_input("Model", key="llm_model")
        llm_client = LLMClient(base_url=st.session_state.get("llm_base_url") or None, api_key=st.session_state.get("llm_api_key") or None, model=st.session_state.get("llm_model") or None)
        st.caption(f"LLM status: {'Connected' if llm_client.is_configured() else 'Not connected'}")
        if st.button("Test connection", use_container_width=True, key="llm_test_connection"):
            if llm_client.is_configured():
                st.success("LLM assistant is connected.")
            else:
                st.warning("LLM assistant is not connected.")

    st.markdown("---")
    st.markdown("#### ⚡ Quick Start")
    st.info("Run a built-in MATR RUL example to verify the full local BatteryML workflow: dataset → processed files → config → model run → metrics/results.")
    st.caption("Quick Start runs a built-in MATR RUL example using the existing BatteryML configuration. It is intended to verify that the local dataset, processed files, config, and model runner are working end-to-end.")

    demo_ready = Path("data/processed/MATR").exists() and Path("configs/baselines/sklearn/xgb/matr_1.yaml").exists()
    demo_command = f"python batteryml_web/core/run_batteryml_cli.py run {QUICKSTART_CONFIG} --workspace <project_workspace> --train --eval --device cpu --seed 0"
    st.caption("Demo output: metrics, logs, workspace files, and report if available.")

    demo_col1, demo_col2 = st.columns(2, gap="large")
    with demo_col1:
        st.markdown("**MATR CPU Demo**")
        st.caption("Dataset: MATR | Task: RUL | Model: XGBoostRULPredictor | Device: CPU")
        st.code(str(QUICKSTART_CONFIG), language="text")
        if st.button("Load MATR Demo", use_container_width=True, key="quick_load_matr_demo"):
            render_staged_progress([(10, "Load MATR dataset setting"), (30, "Load RUL task setting"), (55, "Load XGBoostRULPredictor model setting"), (80, "Load CPU device and config path"), (100, "MATR demo settings loaded")], title="Load MATR Demo")
            st.session_state["selected_dataset"] = "MATR"
            st.session_state["selected_task"] = "RUL"
            st.session_state["selected_model"] = "XGBoostRULPredictor"
            st.session_state["selected_config"] = str(QUICKSTART_CONFIG)
            st.session_state["config_path_for_run"] = str(QUICKSTART_CONFIG)
            st.session_state["dataset_for_run"] = "MATR"
            st.session_state["task_for_run"] = "RUL"
            st.session_state["model_for_run"] = "XGBoostRULPredictor"
            st.session_state["selected_device"] = "cpu"
            st.session_state["device_for_run"] = "cpu"
            st.success("MATR demo loaded into the workspace settings.")
    with demo_col2:
        st.markdown("**One-click Run**")
        st.caption("Runs the demo and writes outputs to the current project workspace.")
        if st.button("Run MATR Demo", use_container_width=True, key="quick_run_matr_demo", disabled=not demo_ready):
            st.session_state["selected_dataset"] = "MATR"
            st.session_state["selected_task"] = "RUL"
            st.session_state["selected_model"] = "XGBoostRULPredictor"
            st.session_state["selected_config"] = str(QUICKSTART_CONFIG)
            st.session_state["selected_device"] = "cpu"
            st.session_state["run_quick_demo"] = True
            st.session_state["current_step"] = "run_model"
            st.session_state["active_section"] = "run"
            st.rerun()

    with st.expander("Show demo command"):
        st.code(demo_command, language="bash")

    if st.button("Start workflow →", use_container_width=True, key="home_start_workflow"):
        st.session_state["current_step"] = "prepare_data"
        st.session_state["active_section"] = "prepare"
        st.rerun()


# Section content renderer
current_step = st.session_state.get("current_step", "home")
if current_step not in {s["key"] for s in NAV_STEPS}:
    current_step = "home"
    st.session_state["current_step"] = current_step
current_section = NAV_TO_SECTION.get(current_step, "project")
st.session_state["active_section"] = current_section
current_root = root

main_context = st.container()

with main_context:
    if current_step == "home":
        render_home_page()
    elif current_step == "prepare_data" and root:
        st.markdown("---")
    elif current_step in {"visualize_features", "run_model", "review_results"}:
        pass
    else:
        render_home_page()

render_step_navigation(current_step)
    # Legacy workflow block removed.
    st.caption("Complete workflow for battery degradation prediction from raw data to results")

    # Check current workflow state
    matr_proc_exists = Path("data/processed/MATR").exists()
    matr_raw_exists = Path("data/raw/MATR").exists()
    config_selected = st.session_state.get("selected_config") is not None
    last_run = st.session_state.get("last_run_status")

    # Determine step statuses
    step1_status = "complete" if matr_proc_exists else "ready" if matr_raw_exists else "pending"
    step2_status = "complete" if matr_proc_exists else "pending"
    step3_status = "ready" if matr_proc_exists else "pending"
    step4_status = "complete" if last_run == "success" else "pending" if config_selected else "pending"

    # Helper function for workflow card
    def workflow_card(emoji, title, status, description, action_label, action_tab, disabled=False):
        border_class = {
            "complete": "border-left: 4px solid #38a169;",
            "ready": "border-left: 4px solid #3182ce;",
            "pending": "border-left: 4px solid #a0aec0; opacity: 0.7;"
        }.get(status, "")

        status_icon = {"complete": "✅", "ready": "⏳", "pending": "⏸️"}.get(status, "⏸️")

        with st.container():
            st.markdown(f"""
            <div style="background: white; border-radius: 10px; padding: 1rem; margin: 0.5rem 0;
                        border: 1px solid #e2e8f0; {border_class}">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <span style="font-size: 1.5rem; margin-right: 0.5rem;">{emoji}</span>
                        <strong>{title}</strong>
                        <span style="margin-left: 1rem;">{status_icon}</span>
                    </div>
                </div>
                <p style="margin: 0.5rem 0; color: #4a5568; font-size: 0.9rem;">{description}</p>
            </div>
            """, unsafe_allow_html=True)

            if not disabled and st.button(action_label, key=f"goto_{action_tab}"):
                st.session_state["active_section"] = action_tab
                st.rerun()

    # Step 1: Dataset Preparation
    workflow_card(
        "📁", "1. Dataset Preparation",
        step1_status,
        "Import and preprocess battery cycling data. Convert raw files to processed cell records.",
        "Go to Preparation →", "prepare",
        disabled=False
    )

    # Step 2: Feature Visualization
    workflow_card(
        "🔍", "2. Feature Visualization",
        step2_status,
        "Explore extracted features: voltage curves, capacity fade, feature matrix preview.",
        "Go to Features →", "features",
        disabled=not matr_proc_exists
    )

    # Step 3: Model Configuration
    workflow_card(
        "⚙️", "3. Model Configuration",
        step3_status,
        "Select dataset, task (RUL/SOH), model (XGBoost, MLP, LSTM), and device (CPU/GPU).",
        "Go to Configuration →", "run",
        disabled=not matr_proc_exists
    )

    # Step 4: Run Training
    workflow_card(
        "🚀", "4. Run Training / Evaluation",
        "ready" if config_selected else "pending",
        "Execute BatteryML training with preflight validation. View logs in real-time.",
        "Go to Run →", "run",
        disabled=not matr_proc_exists or not config_selected
    )

    # Step 5: Results & Report
    workflow_card(
        "📊", "5. Results & Report",
        step4_status,
        "View metrics, prediction curves, error analysis, and download HTML report.",
        "Go to Results →", "results",
        disabled=last_run != "success"
    )

    st.markdown("---")

    # Quick Actions
    st.markdown("<div class='section-header'>⚡ Quick Actions</div>", unsafe_allow_html=True)

    qa_col1, qa_col2, qa_col3 = st.columns(3)

    with qa_col1:
        if st.button("📂 Load MATR Config", use_container_width=True, key="quick_load_matr"):
            st.session_state["demo_dataset"] = "MATR"
            st.session_state["demo_task"] = "RUL"
            st.session_state["demo_model"] = "XGBoostRULPredictor"
            st.session_state["demo_config_path"] = str(QUICKSTART_CONFIG)
            st.session_state["selected_dataset"] = "MATR"
            st.session_state["selected_task"] = "RUL"
            st.session_state["selected_model"] = "XGBoostRULPredictor"
            st.session_state["selected_config"] = str(QUICKSTART_CONFIG)
            st.success("MATR config loaded. Go to 'Model Configuration' or 'Run Training'.")

    with qa_col2:
        if st.button("🔄 Run Preprocessing", use_container_width=True, key="quick_preprocess",
                      disabled=not Path("data/raw/MATR").exists()):
            st.session_state["active_tab"] = "Dataset Preparation"
            st.rerun()

    with qa_col3:
        demo_ready = Path("data/processed/MATR").exists() and Path("configs/baselines/sklearn/xgb/matr_1.yaml").exists()
        if st.button("▶️ Run Demo", use_container_width=True, type="primary", key="quick_run_demo",
                    disabled=not demo_ready):
            st.session_state["run_quick_demo"] = True
            st.session_state["active_tab"] = "Run Training"
            st.rerun()

    # Handle quick demo execution
    if st.session_state.get("run_quick_demo"):
        st.session_state["run_quick_demo"] = False
        st.markdown("---")
        st.markdown("<div class='section-header'>🚀 Running MATR Quick Demo</div>", unsafe_allow_html=True)

        # Validation
        errors = []
        config_path = Path("configs/baselines/sklearn/xgb/matr_1.yaml")
        if not config_path.exists():
            errors.append(f"Config not found: {config_path}")
        if not Path("data/raw/MATR").exists():
            errors.append("Raw data missing: data/raw/MATR")
        if not Path("data/processed/MATR").exists():
            errors.append("Processed data missing. Run preprocessing first.")

        if errors:
            st.error("Cannot run demo:")
            for e in errors:
                st.write(f"- {e}")
        else:
            workspace_path = current_root / "workspace"
            workspace_path.mkdir(parents=True, exist_ok=True)

            import time
            start = time.perf_counter()

            cmd = [
                sys.executable, "batteryml_web/core/run_batteryml_cli.py", "run",
                str(config_path), "--workspace", str(workspace_path),
                "--train", "--eval", "--device", "cpu", "--seed", "0"
            ]

            result = run_command_with_progress(
                command=cmd,
                log_path=current_root / "logs" / "matr_quick_demo.log",
                steps=[
                    (5, "Load MATR demo settings"),
                    (10, "Check raw data"),
                    (20, "Check processed data"),
                    (30, "Check config"),
                    (40, "Start BatteryML run"),
                    (75, "Running train/eval"),
                    (90, "Collect outputs"),
                    (100, "Demo completed"),
                ],
                cwd=Path("."),
                env=get_safe_runtime_env(),
            )

            elapsed = time.perf_counter() - start
            st.session_state["last_run_status"] = "success" if result["returncode"] == 0 else "failed"
            st.session_state["last_run_time"] = elapsed

            # Display results
            res_cols = st.columns(4)
            res_cols[0].metric("Status", "✅ Success" if result["returncode"] == 0 else "❌ Failed")
            res_cols[1].metric("Runtime", f"{elapsed:.1f}s")
            res_cols[2].metric("Exit Code", result["returncode"])
            res_cols[3].metric("Output", str(workspace_path))

            if result["returncode"] == 0:
                st.success(f"Demo completed in {elapsed:.1f}s. Go to 'Results & Report' to view outputs.")
            else:
                st.error("Demo failed. Check logs below.")
                with st.expander("Error Details"):
                    st.code(result.get("stderr", "")[-1000:], language="text")

if current_section == "prepare" and root:
    st.markdown("<div class='section-header'>🔋 Dataset Preparation</div>", unsafe_allow_html=True)
    st.caption("Import raw battery data, run preprocessing, and inspect processed cell records")

    # Two-column layout
    prep_col1, prep_col2 = st.columns([2, 3])

    with prep_col1:
        # Dataset selector
        st.markdown("#### 📖 Select Dataset")
        inspect_datasets = ["MATR", "HUST", "CALCE", "RWTH", "SNL", "HNEI", "UL_PUR"]
        selected_inspect_dataset = st.selectbox(
            "Dataset to inspect",
            options=inspect_datasets,
            key="dataset_prep_selector_v2"
        )

        ds_info = inspect_dataset(selected_inspect_dataset, current_root)

        # Battery status card
        st.markdown("#### 🔋 Battery Data Status")

        batt_cols = st.columns(2)
        with batt_cols[0]:
            raw_icon = "🔋" if ds_info["raw"]["exists"] else "🔌"
            st.metric(f"{raw_icon} Raw", f"{ds_info['raw']['file_count']} files" if ds_info["raw"]["exists"] else "Missing")
        with batt_cols[1]:
            proc_icon = "🔋" if ds_info["processed"]["exists"] else "🔌"
            st.metric(f"{proc_icon} Processed", f"{ds_info['processed']['file_count']} files" if ds_info["processed"]["exists"] else "Missing")

        # Overall status with action
        overall = ds_info["overall_status"]
        if overall == "Ready":
            st.success("✅ Dataset ready for training")
            if st.button("📊 View Features", use_container_width=True, key="goto_features_prep"):
                st.session_state["current_step"] = "visualize_features"
                st.rerun()
        elif overall == "Need Preprocess":
            st.warning("⚠️ Need preprocessing")
        else:
            st.error("❌ Data missing")
            st.caption(f"Expected: data/raw/{selected_inspect_dataset}/")

    with prep_col2:
        # Preparation Flow Visualization
        st.markdown("#### 📊 Preparation Pipeline")
        st.markdown("""
        ```
        🔌 Raw Cycling Files (CSV/MAT)
            ↓ [BatteryML Parser]
        🔋 Processed Cell Records (PKL)
            ↓ [Feature Extractor]
        📊 Feature Matrix (NPY/NPZ)
            ↓ [Model Trainer]
        🎯 Trained Model + Predictions
        ```
        """)

        # Compact dataset status table
        st.markdown("#### 📁 All Datasets Status")
        all_ds_check = ["MATR", "HUST", "CALCE", "RWTH"]
        ds_summary_data = []
        for ds in all_ds_check:
            ds_check = inspect_dataset(ds, current_root)
            ds_summary_data.append({
                "Dataset": ds,
                "Raw": "✅" if ds_check["raw"]["exists"] else "❌",
                "Processed": "✅" if ds_check["processed"]["exists"] else "❌",
                "Status": ds_check["overall_status"],
            })
        st.dataframe(pd.DataFrame(ds_summary_data), use_container_width=True, hide_index=True)

    # Preprocessing Action Section
    st.markdown("---")
    st.markdown("#### ▶️ Preprocess Action")

    raw_path = Path(f"data/raw/{selected_inspect_dataset}")
    processed_path = Path(f"data/processed/{selected_inspect_dataset}")

    if not raw_path.exists():
        st.button("Run Preprocessing", disabled=True, use_container_width=True, key="preprocess_disabled_v2")
        st.error("Raw data missing")
    else:
        btn_label = "🔄 Re-run Preprocessing" if processed_path.exists() else "▶️ Run Preprocessing"
        if st.button(btn_label, use_container_width=True, key="preprocess_action_v2"):
            with st.spinner(f"Preprocessing {selected_inspect_dataset}..."):
                logs_dir = current_root / "logs"
                logs_dir.mkdir(parents=True, exist_ok=True)

                preprocess_cmd = [
                    sys.executable,
                    "batteryml_web/core/run_batteryml_cli.py",
                    "preprocess",
                    selected_inspect_dataset,
                    str(raw_path),
                    str(processed_path)
                ]
                log_path = logs_dir / f"preprocess_{selected_inspect_dataset}.log"
                result = run_command_with_progress(
                    command=preprocess_cmd,
                    log_path=log_path,
                    steps=[
                        (0, "Initialize preprocessing task"),
                        (10, "Check raw data path"),
                        (20, "Check processed output path"),
                        (30, "Start BatteryML preprocessing command"),
                        (70, f"BatteryML preprocessing is running for {selected_inspect_dataset}"),
                        (85, "Verify processed files"),
                        (95, "Write preprocessing log"),
                        (100, "Preprocessing completed"),
                    ],
                    cwd=Path("."),
                    env=get_safe_runtime_env(),
                )
                st.session_state["last_preprocess_log"] = Path(result["log_path"]).read_text(encoding="utf-8")
                st.session_state["last_preprocess_result"] = result

            if result['returncode'] == 0:
                st.success("✅ Preprocessing complete!")
            else:
                st.error("❌ Preprocessing failed")

    # File Inspector
    st.markdown("---")
    st.markdown("<div class='section-header'>🔍 Dataset File Inspector</div>", unsafe_allow_html=True)

    all_files = []
    for location, info in [
        ("raw", ds_info["raw"]),
        ("processed", ds_info["processed"]),
        ("cache", ds_info["cache"]),
        ("workspace", ds_info["runs"]),
    ]:
        if info["exists"]:
            for f in info["files"]:
                f["location"] = location
                all_files.append(f)

    if all_files:
        file_types_found = sorted(set(f["file_type"] for f in all_files))
        st.markdown("**File Types:** " + " ".join([f"`{t}`" for t in file_types_found[:10]]))

        file_df = pd.DataFrame([
            {
                "📁 Location": f["location"],
                "📄 File Name": f["name"],
                "🏷️ Type": f["file_type"],
                "📏 Size": f["size_human"],
                "👁️ Preview": "✓" if f["can_preview"] else "—",
            }
            for f in all_files[:100]
        ])
        st.dataframe(file_df, use_container_width=True, hide_index=True)

        # File Preview
        st.markdown("#### 👁️ File Preview")
        preview_options = [f"{f['location']}: {f['name']}" for f in all_files if f["can_preview"]]

        if preview_options:
            preview_sel = st.selectbox("Select file", options=preview_options, key="file_preview_v2")
            if preview_sel:
                location, filename = preview_sel.split(": ", 1)
                selected_file = next((f for f in all_files if f["location"] == location and f["name"] == filename), None)
                if selected_file:
                    file_path = Path(selected_file["full_path"])
                    pcols = st.columns(3)
                    pcols[0].metric("Type", selected_file['file_type'])
                    pcols[1].metric("Size", selected_file['size_human'])
                    pcols[2].metric("Modified", selected_file['modified_time'][:10])

                    st.code(f"cd {file_path.parent}\nls -lh {file_path.name}", language="bash")

                    preview = preview_file(file_path, max_rows=20)
                    if "error" in preview:
                        st.error(preview["error"])
                    elif preview["preview_type"] == "binary":
                        st.info(f"📦 Binary: {preview['preview'].get('note', 'Cannot preview')}")
                    elif preview["preview_type"] in ["csv", "xlsx", "parquet"]:
                        if preview.get("shape"):
                            st.write(f"**Shape:** {preview['shape']}")
                        if preview.get("preview"):
                            st.dataframe(pd.DataFrame(preview["preview"]), use_container_width=True)
                    elif preview["preview_type"] in ["json", "yaml"]:
                        st.json(preview["preview"])
                    elif preview["preview_type"] in ["txt", "log"]:
                        st.text_area("Content", value=preview.get("preview", ""), height=250, key="txt_preview_v2")
    else:
        st.info(f"📂 No files found for {selected_inspect_dataset}")
        st.code(f"data/raw/{selected_inspect_dataset}/       ← Place raw files here\ndata/processed/{selected_inspect_dataset}/   → Outputs here")

    # Post-preprocessing Battery Summary
    if ds_info["processed"]["exists"]:
        st.markdown("---")
        st.markdown("<div class='section-header'>🔋 Post-Preprocessing Summary</div>", unsafe_allow_html=True)
        try:
            processed_files = [f for f in all_files if f["location"] == "processed"]
            cell_count = len([f for f in processed_files if "cell" in f["name"].lower()])

            batt_cols = st.columns(4)
            batt_cols[0].metric("🔋 Cells", cell_count if cell_count > 0 else "N/A")
            batt_cols[1].metric("⚡ Cycles", "N/A")
            batt_cols[2].metric("📊 Voltage Range", "N/A")
            batt_cols[3].metric("🔋 Capacity", "N/A")

            st.caption("Battery curve visualization requires dedicated parser. Go to Feature Visualization for extracted features.")
        except Exception as e:
            st.caption(f"Summary not available: {e}")

    # Preprocessing Log
    if st.session_state.get("last_preprocess_log"):
        with st.expander("📄 Last Preprocessing Log"):
            st.text_area("Log", value=st.session_state["last_preprocess_log"], height=300, key="preprocess_log_v2")

# Removed Schema Mapping tab - functionality merged into Dataset Preparation or removed as redundant

    # Paths display
    with st.expander("View Paths"):
        st.code(f"""
Raw:        {ds_info['raw']['path']}
Processed:  {ds_info['processed']['path']}
Cache:      {ds_info['cache']['path']}
Workspace:  {ds_info['runs']['path']}
        """.strip())

    # File browser
    st.markdown("#### File Browser")

    # Combine all files for display
    all_files = []
    for location, info in [
        ("raw", ds_info["raw"]),
        ("processed", ds_info["processed"]),
        ("cache", ds_info["cache"]),
        ("workspace", ds_info["runs"]),
    ]:
        if info["exists"]:
            for f in info["files"]:
                f["location"] = location
                all_files.append(f)

    if all_files:
        # Create file table
        file_df = pd.DataFrame([
            {
                "Location": f["location"],
                "File Name": f["name"],
                "Type": f["file_type"],
                "Size": f["size_human"],
                "Modified": f["modified_time"],
                "Preview": "✓" if f["can_preview"] else "✗",
            }
            for f in all_files
        ])

        # File type filter
        file_types = ["All"] + sorted(set(file_df["Type"].tolist()))
        selected_type = st.selectbox("Filter by file type", options=file_types, key="file_type_filter")

        if selected_type != "All":
            file_df = file_df[file_df["Type"] == selected_type]

        st.dataframe(file_df, use_container_width=True, hide_index=True)

        # File preview selector
        st.markdown("#### File Preview")
        preview_options = [f"{f['location']}: {f['name']}" for f in all_files if f["can_preview"]]

        if preview_options:
            selected_file_str = st.selectbox("Select file to preview", options=preview_options, key="file_preview_selector")

            if selected_file_str:
                # Parse selection
                location, filename = selected_file_str.split(": ", 1)
                selected_file = next((f for f in all_files if f["location"] == location and f["name"] == filename), None)

                if selected_file:
                    file_path = Path(selected_file["full_path"])

                    # Show file info
                    info_cols = st.columns(3)
                    info_cols[0].write(f"**Type:** {selected_file['file_type']}")
                    info_cols[1].write(f"**Size:** {selected_file['size_human']}")
                    info_cols[2].write(f"**Modified:** {selected_file['modified_time']}")

                    # Show path and copy command
                    st.code(f"cd {file_path.parent}\nls -lh {file_path.name}")

                    # Preview content
                    preview = preview_file(file_path, max_rows=20)

                    if "error" in preview:
                        st.error(preview["error"])
                    elif preview["preview_type"] == "binary":
                        st.info(f"Binary file: {preview['preview'].get('note', 'Cannot preview binary files')}")
                        if "shape" in preview["preview"]:
                            st.write(f"**Array Shape:** {preview['preview']['shape']}")
                        if "dtype" in preview["preview"]:
                            st.write(f"**Data Type:** {preview['preview']['dtype']}")
                    elif preview["preview_type"] in ["csv", "xlsx", "parquet"]:
                        if preview.get("shape"):
                            st.write(f"**Shape:** {preview['shape']}")
                        if preview.get("columns"):
                            st.write(f"**Columns:** {', '.join(preview['columns'])}")
                        if preview.get("dtypes"):
                            with st.expander("Column Types"):
                                st.json(preview["dtypes"])
                        if preview.get("preview"):
                            preview_df = pd.DataFrame(preview["preview"])
                            st.dataframe(preview_df, use_container_width=True)
                    elif preview["preview_type"] in ["json", "yaml"]:
                        st.json(preview["preview"])
                    elif preview["preview_type"] in ["txt", "log"]:
                        st.text_area(
                            "Content (first 20 lines)",
                            value=preview.get("preview", ""),
                            height=300,
                            key="txt_preview_content"
                        )
                        if preview.get("total_lines"):
                            st.caption(f"Total lines: {preview['total_lines']}")
                    else:
                        st.info(f"Preview not implemented for {selected_file['file_type']} files.")
        else:
            st.info("No previewable files found. Supported formats: CSV, XLSX, Parquet, JSON, YAML, TXT, LOG")
    else:
        st.info(f"No files found for {selected_inspect_dataset}. Expected paths:")
        st.code(f"""
data/raw/{selected_inspect_dataset}/     - Raw data files
data/processed/{selected_inspect_dataset}/ - Processed data files
        """.strip())

    st.divider()

    # Original dataset status table
    st.markdown("#### All Datasets Status")
    datasets = ["MATR", "HUST", "CALCE", "RWTH", "SNL", "HNEI", "UL_PUR"]
    status_rows = []
    for ds in datasets:
        ds_status = inspect_dataset(ds, current_root)
        status_rows.append({
            "Dataset": ds,
            "Status": ds_status["overall_status"],
            "Raw": f"{ds_status['raw']['file_count']} files" if ds_status['raw']['exists'] else "Missing",
            "Processed": f"{ds_status['processed']['file_count']} files" if ds_status['processed']['exists'] else "Missing",
            "Can Run": "✅" if ds_status["can_run"] else "❌",
        })
    st.dataframe(pd.DataFrame(status_rows), use_container_width=True, hide_index=True)

    st.markdown("#### Preparation flow")
    prep_cols = st.columns(5)
    prep_cols[0].write("Raw cycling files")
    prep_cols[1].write("→ BatteryML preprocessing")
    prep_cols[2].write("→ BatteryData / processed files")
    prep_cols[3].write("→ Feature extraction")
    prep_cols[4].write("→ Model training")

    st.divider()
    st.markdown("#### Dataset Preprocessing")
    st.caption("Select a dataset to preprocess. Raw data must exist.")

    # Dataset selector for preprocessing
    preprocess_datasets = ["MATR", "HUST", "CALCE", "RWTH", "SNL", "HNEI", "UL_PUR"]
    selected_preprocess_ds = st.selectbox(
        "Select dataset to preprocess",
        options=preprocess_datasets,
        key="preprocess_dataset_selector"
    )

    # Show paths and status
    raw_path = Path(f"data/raw/{selected_preprocess_ds}")
    processed_path = Path(f"data/processed/{selected_preprocess_ds}")

    path_cols = st.columns(2)
    with path_cols[0]:
        st.write("**Raw path:**")
        st.code(str(raw_path))
        raw_exists = raw_path.exists() and any(raw_path.iterdir())
        if raw_exists:
            raw_files = list(raw_path.rglob("*"))
            raw_file_count = len([f for f in raw_files if f.is_file()])
            st.success(f"✅ Raw data exists ({raw_file_count} files)")
        else:
            st.error("❌ Raw data missing")

    with path_cols[1]:
        st.write("**Processed path:**")
        st.code(str(processed_path))
        processed_exists = processed_path.exists() and any(processed_path.iterdir())
        if processed_exists:
            processed_files = list(processed_path.rglob("*"))
            processed_file_count = len([f for f in processed_files if f.is_file()])
            st.success(f"✅ Processed data exists ({processed_file_count} files)")
        else:
            st.warning("⚠️ Processed data missing")

    # Preprocess button
    st.write("**Preprocess action:**")

    if not raw_exists:
        st.button("Run preprocessing", disabled=True, key="preprocess_btn_disabled")
        st.error(f"Cannot preprocess: Raw data missing at {raw_path}")
        st.info("Please place raw data files in the expected location.")
    else:
        preprocess_btn_label = "Re-run preprocessing" if processed_exists else "Run preprocessing"
        if st.button(preprocess_btn_label, key="preprocess_btn_run"):
            # Create logs directory
            logs_dir = current_root / "logs"
            logs_dir.mkdir(parents=True, exist_ok=True)
            preprocess_log_path = logs_dir / "preprocess.log"

            with st.spinner(f"Running preprocessing for {selected_preprocess_ds}..."):
                preprocess_cmd = [
                    sys.executable,
                    "batteryml_web/core/run_batteryml_cli.py",
                    "preprocess",
                    selected_preprocess_ds,
                    str(raw_path),
                    str(processed_path)
                ]
                result = run_command_with_progress(
                    command=preprocess_cmd,
                    log_path=preprocess_log_path,
                    steps=[
                        (0, "Initialize preprocessing task"),
                        (10, "Check raw data path"),
                        (20, "Check processed output path"),
                        (30, "Start BatteryML preprocessing command"),
                        (70, f"BatteryML preprocessing is running for {selected_preprocess_ds}"),
                        (85, "Verify processed files"),
                        (95, "Write preprocessing log"),
                        (100, "Preprocessing completed"),
                    ],
                    cwd=Path("."),
                    env=get_safe_runtime_env(),
                )

            log_content = Path(result["log_path"]).read_text(encoding="utf-8")

            # Display results
            st.divider()
            st.write("**Preprocessing Results:**")

            result_cols = st.columns(3)
            result_cols[0].metric("Status", "Success" if result['returncode'] == 0 else "Failed")
            result_cols[1].metric("Exit Code", result['returncode'])
            result_cols[2].metric("Runtime", f"{result['elapsed_seconds']:.1f}s")

            if result['returncode'] == 0:
                st.success("✅ Preprocessing completed successfully!")
            else:
                st.error("❌ Preprocessing failed!")

            st.write(f"**Log file:** `{preprocess_log_path}`")

            with st.expander("View log", expanded=result['returncode'] != 0):
                st.text_area("Log content", value=log_content, height=400, key="preprocess_log_viewer")

            # Check processed data status
            if processed_path.exists():
                new_processed_files = [p for p in processed_path.rglob("*") if p.is_file()]
                st.write(f"**Processed files:** {len(new_processed_files)}")
            else:
                st.warning("Processed directory not created.")

    st.divider()
    st.subheader("Dataset inspection and schema mapping")
    uploaded_prep = st.file_uploader("Upload raw CSV/XLSX", type=["csv", "xlsx", "xls"], key="prep_dataset_upload_raw_csv_xlsx")
    inspect_result_prep = None
    source_path_prep = None
    if uploaded_prep is not None:
        raw_dir = current_root / "raw_inputs"
        raw_dir.mkdir(parents=True, exist_ok=True)
        source_path_prep = raw_dir / uploaded_prep.name
        source_path_prep.write_bytes(uploaded_prep.getbuffer())
        inspect_result_prep = inspect_tabular_file(source_path_prep)
        st.success(f"Uploaded file saved to `{source_path_prep}`")
    elif (current_root / "raw_inputs").exists():
        st.caption("No new upload. Existing raw inputs will be used if present.")

    if inspect_result_prep:
        st.dataframe(pd.DataFrame(inspect_result_prep["preview"]))
        left_prep, right_prep = st.columns(2)
        with left_prep:
            st.json({"columns": inspect_result_prep["columns"], "dtypes": inspect_result_prep["dtypes"]})
        with right_prep:
            st.json({"missing_values": inspect_result_prep["missing_values"]})

        helper_cols_prep = st.columns(3)
        base_url_prep = helper_cols_prep[0].text_input("LLM base URL for mapping", value=st.session_state.get("llm_base_url", ""), key="prep_map_base_url")
        api_key_prep = helper_cols_prep[1].text_input("LLM API key for mapping", value=st.session_state.get("llm_api_key", ""), type="password", key="prep_map_api_key")
        model_name_prep = helper_cols_prep[2].text_input("LLM model for mapping", value=st.session_state.get("llm_model", ""), key="prep_map_model")
        if st.button("Use LLM to suggest mapping", key="prep_use_llm_suggest_mapping_btn"):
            client_prep = LLMClient(base_url=base_url_prep or None, api_key=api_key_prep or None, model=model_name_prep or None)
            if not client_prep.is_configured():
                st.warning("No API key or endpoint configured. Manual mapping remains available.")
            else:
                render_staged_progress(
                    [
                        (10, "Connecting to LLM endpoint"),
                        (30, "Sending schema mapping prompt"),
                        (70, "Waiting for response"),
                        (90, "Parsing suggestion"),
                        (100, "LLM suggestion ready"),
                    ],
                    title="LLM schema mapping",
                )
                suggestion_prep = suggest_schema_mapping(inspect_result_prep, llm_client=client_prep)
                st.session_state["llm_suggested_mapping_prep"] = suggestion_prep.get("mapping", {})
                st.session_state["llm_suggestion_raw_prep"] = suggestion_prep
                st.success("LLM suggestion ready for review.")
                st.json(suggestion_prep)

        suggested_mapping_prep = st.session_state.get("llm_suggested_mapping_prep", {})
        available_columns_prep = ["<unmapped>"] + inspect_result_prep["columns"]
        mapping_prep = {}
        cols_prep = st.columns(2)
        for i, field in enumerate(REQUIRED_FIELDS + OPTIONAL_FIELDS):
            default_value_prep = suggested_mapping_prep.get(field, "<unmapped>") if suggested_mapping_prep else "<unmapped>"
            if default_value_prep not in available_columns_prep:
                default_value_prep = "<unmapped>"
            with cols_prep[i % 2]:
                mapping_prep[field] = st.selectbox(f"Map `{field}` to column", available_columns_prep, index=available_columns_prep.index(default_value_prep), key=f"prep_mapping_{field}")

        if st.button("Validate schema mapping", key="prep_validate_schema_mapping_btn"):
            cleaned_mapping_prep = {k: v for k, v in mapping_prep.items() if v != "<unmapped>"}
            report_prep = validate_schema_mapping(cleaned_mapping_prep)
            mapping_path_prep = current_root / "schema_mapping.json"
            report_path_prep = current_root / "schema_validation_report.json"
            mapping_path_prep.write_text(json.dumps(cleaned_mapping_prep, indent=2, ensure_ascii=False), encoding="utf-8")
            report_path_prep.write_text(json.dumps(report_prep, indent=2, ensure_ascii=False), encoding="utf-8")
            st.success(f"Schema mapping saved to `{mapping_path_prep}`")
            st.json(report_prep)
    else:
        st.caption("Upload a dataset to inspect columns and create a validated schema mapping.")

if current_section == "run" and root:
    # Tab 2: Model Configuration (moved from Tab 3)
    st.markdown("<div class='section-header'>⚙️ Model Configuration</div>", unsafe_allow_html=True)
    st.caption("Choose dataset, task, model, and execution device")

    # Initialize catalog
    catalog = get_catalog(root_dir=Path("."), include_debug=False)

    # Dataset selection with status
    all_datasets = catalog.get_datasets()
    if not all_datasets:
        st.error("No datasets found in catalog.")
        st.stop()

    st.markdown("#### Dataset & Task Selection")
    sel_cols = st.columns(3)

    with sel_cols[0]:
        default_ds = st.session_state.get("selected_dataset", all_datasets[0] if all_datasets else None)
        default_ds_idx = all_datasets.index(default_ds) if default_ds in all_datasets else 0

        def format_dataset_option(ds_name):
            ds_status = inspect_dataset(ds_name, current_root)
            icon = {"Ready": "🟢", "Need Preprocess": "🟡", "Missing": "🔴"}.get(ds_status["overall_status"], "⚪")
            return f"{icon} {ds_name}"

        selected_dataset = st.selectbox(
            "Dataset",
            options=all_datasets,
            index=default_ds_idx,
            key="model_config_dataset",
            format_func=format_dataset_option
        )
        st.session_state["selected_dataset"] = selected_dataset

        ds_inspect = inspect_dataset(selected_dataset, current_root)
        status_color = {"Ready": "🟢", "Need Preprocess": "🟡", "Missing": "🔴"}.get(ds_inspect["overall_status"], "⚪")
        st.write(f"**Status:** {status_color} {ds_inspect['overall_status']}")

        if not ds_inspect["processed"]["exists"]:
            st.warning("⚠️ Processed data missing")
            if st.button("🔋 Go to Preparation", key="goto_prep_from_config"):
                st.session_state["active_tab"] = "Dataset Preparation"
                st.rerun()

    with sel_cols[1]:
        available_tasks = catalog.get_tasks_for_dataset(selected_dataset)
        task_options = [t.value for t in available_tasks] if available_tasks else ["RUL"]
        default_task = st.session_state.get("selected_task", task_options[0] if task_options else "RUL")
        default_task_idx = task_options.index(default_task) if default_task in task_options else 0
        selected_task_str = st.selectbox(
            "Task",
            options=task_options,
            index=default_task_idx,
            key="model_config_task"
        )
        st.session_state["selected_task"] = selected_task_str
        selected_task = TaskType(selected_task_str)

    with sel_cols[2]:
        available_models = catalog.get_models_for_dataset_task(selected_dataset, selected_task)
        model_options = list(dict.fromkeys([m.model_class for m in available_models])) if available_models else []

        if not model_options:
            st.warning("No models found")

        default_model = st.session_state.get("selected_model", model_options[0] if model_options else None)
        default_model_idx = model_options.index(default_model) if default_model in model_options else 0
        selected_model = st.selectbox(
            "Model",
            options=model_options,
            index=default_model_idx if model_options else 0,
            key="model_config_model"
        )
        st.session_state["selected_model"] = selected_model

    # Config selection
    st.markdown("#### Configuration")
    if selected_model:
        matching_configs = catalog.get_configs_for_model(selected_dataset, selected_task, selected_model)
        if matching_configs:
            config_options = []
            for cfg in matching_configs:
                icon = {Status.READY: "✅", Status.NEED_PREPROCESS: "⚠️", Status.EXPERIMENTAL: "🔬"}.get(cfg.status, "❓")
                label = f"{icon} {cfg.config_path.name} ({', '.join(cfg.device_support)})"
                config_options.append((label, cfg))

            selected_label = st.selectbox("Select config", options=[o[0] for o in config_options], key="model_config_select")
            selected_config = next((cfg for label, cfg in config_options if label == selected_label), None)

            if selected_config:
                st.session_state["selected_config"] = str(selected_config.config_path)
                st.session_state["config_path_for_run"] = str(selected_config.config_path)
                st.session_state["dataset_for_run"] = selected_dataset
                st.session_state["task_for_run"] = selected_task_str
                st.session_state["model_for_run"] = selected_model

                with st.container(border=True):
                    cfg_cols = st.columns([2, 1])
                    with cfg_cols[0]:
                        st.code(str(selected_config.config_path))
                        st.caption(f"Feature: {selected_config.feature_extractor}")
                    with cfg_cols[1]:
                        st.write(f"**Status:** {selected_config.status.value}")
                        st.write(f"**Devices:** {', '.join(selected_config.device_support)}")

                if selected_config.status == Status.NEED_PREPROCESS:
                    st.warning("⚠️ Need preprocessing")
        else:
            st.warning("No configs found")

    # Device selection
    st.markdown("#### Device")
    if st.session_state.get("selected_config"):
        device_options = ["cpu"]
        gpu_val = validate_device_choice("cuda")
        if gpu_val["can_run"]:
            device_options.append("cuda")

        device = st.radio("Execution device", options=device_options, horizontal=True, key="model_config_device")
        st.session_state["selected_device"] = device
        st.session_state["device_for_run"] = device

        if device == "cuda":
            st.success(f"🚀 GPU ready: {gpu_val.get('gpu_report', {}).get('torch_cuda', {}).get('device_names', ['CUDA'])[0]}")
    else:
        st.info("Select a config first")

    # Ready indicator
    st.markdown("---")
    config_ready = st.session_state.get("selected_config") and ds_inspect["processed"]["exists"]
    if config_ready:
        st.success("✅ Configuration ready. Go to 'Run Training' to start.")
        if st.button("🚀 Go to Run Training", type="primary", use_container_width=True, key="goto_run_from_config"):
            st.session_state["active_tab"] = "Run Training"
            st.rerun()
    else:
        st.warning("⚠️ Configuration incomplete")

    # Debug models in expander
    with st.expander("🐛 Debug Models"):
        debug_models = catalog.get_debug_models()
        if debug_models:
            for dm in debug_models[:3]:
                st.caption(f"- {dm.model_class}: {dm.config_path}")

    # Roadmap
    with st.expander("📋 Roadmap / Planned Methods"):
        st.caption("BatLiNet (Inter-cell pair feature learning) - Experimental, not yet implemented")

if current_section == "features" and root:
    # Tab 3: Feature Visualization
    st.markdown("<div class='section-header'>📊 Feature Visualization</div>", unsafe_allow_html=True)
    st.caption("Explore extracted features, feature matrix, and battery characteristics")
    render_staged_progress(
        [
            (10, "Load processed data path"),
            (35, "Inspect processed feature files"),
            (65, "Parse feature cache metadata"),
            (90, "Prepare feature summary"),
            (100, "Feature view loaded"),
        ],
        title="Feature loading",
    )

    # Check for processed data
    selected_viz_dataset = st.selectbox(
        "Select dataset",
        options=["MATR", "HUST", "CALCE", "RWTH"],
        key="feature_viz_dataset"
    )
    proc_path = Path(f"data/processed/{selected_viz_dataset}")

    if not proc_path.exists():
        st.warning("⚠️ Processed data not found")
        st.info(f"Expected: {proc_path}")
        if st.button("🔋 Go to Dataset Preparation", key="goto_prep_viz"):
            st.session_state["active_tab"] = "Dataset Preparation"
            st.rerun()
    else:
        st.success("✅ Processed data available")

        # Feature Matrix Preview (placeholder - would need actual feature files)
        st.markdown("#### Feature Matrix Preview")
        feature_files = list(proc_path.rglob("*feature*")) + list(proc_path.rglob("*.npy"))
        if feature_files:
            st.caption(f"Found {len(feature_files)} potential feature files")
            for f in feature_files[:5]:
                st.code(str(f.name))
        else:
            st.info("Feature files not found. Features are extracted during training.")

        # Cell-level statistics placeholder
        st.markdown("#### Battery Cell Overview")
        viz_cols = st.columns(4)
        viz_cols[0].metric("🔋 Cells", "N/A")
        viz_cols[1].metric("⚡ Cycles", "N/A")
        viz_cols[2].metric("📊 Features", "N/A")
        viz_cols[3].metric("🎯 Labels", "N/A")

        st.caption("Full feature visualization requires feature extraction to be completed. Run training to generate features.")

        # Placeholder for future visualizations
        with st.expander("Planned Visualizations"):
            st.markdown("""
            - Feature correlation heatmap
            - Capacity fade curves
            - Voltage curves
            - Early-cycle feature importance
            """)

if current_section == "run" and root:
    # Tab 4: Run Training (merged Preflight + Execution)
    st.markdown("<div class='section-header'>🚀 Run Training</div>", unsafe_allow_html=True)
    st.caption("Validate configuration and execute BatteryML training")

    # Get current configuration
    exec_config_path = st.session_state.get("config_path_for_run")
    exec_dataset = st.session_state.get("dataset_for_run")
    exec_task = st.session_state.get("task_for_run")
    exec_model = st.session_state.get("model_for_run")
    exec_device = st.session_state.get("device_for_run", "cpu")

    # Validation Section
    st.markdown("#### ✅ Pre-flight Validation")

    checks = []
    all_valid = True

    # Check 1: Config
    if exec_config_path and Path(exec_config_path).exists():
        checks.append(("Config file", True, str(exec_config_path)))
    else:
        checks.append(("Config file", False, "Not selected or not found"))
        all_valid = False

    # Check 2: Dataset processed data
    if exec_dataset:
        proc_path = Path(f"data/processed/{exec_dataset}")
        if proc_path.exists():
            checks.append(("Processed data", True, str(proc_path)))
        else:
            checks.append(("Processed data", False, f"Missing: {proc_path}"))
            all_valid = False
    else:
        checks.append(("Dataset", False, "Not selected"))
        all_valid = False

    # Check 3: Model/Task
    if exec_model and exec_task:
        checks.append(("Model/Task", True, f"{exec_model} / {exec_task}"))
    else:
        checks.append(("Model/Task", False, "Not configured"))
        all_valid = False

    # Display checks
    check_cols = st.columns(2)
    for i, (name, passed, detail) in enumerate(checks):
        with check_cols[i % 2]:
            icon = "✅" if passed else "❌"
            if passed:
                st.success(f"{icon} **{name}**: {detail}")
            else:
                st.error(f"{icon} **{name}**: {detail}")

    # Configuration summary
    st.markdown("#### ⚙️ Configuration Summary")
    if all(check[1] for check in checks[:3]):
        cfg_cols = st.columns(4)
        cfg_cols[0].metric("Dataset", exec_dataset or "N/A")
        cfg_cols[1].metric("Task", exec_task or "N/A")
        cfg_cols[2].metric("Model", exec_model or "N/A")
        cfg_cols[3].metric("Device", exec_device.upper())

        st.code(exec_config_path or "No config selected", language="text")

        # Device validation
        if exec_device == "cuda":
            gpu_val = validate_device_choice("cuda")
            if gpu_val["can_run"]:
                st.success(f"🚀 GPU ready")
            else:
                st.error("❌ GPU not available, switching to CPU")
                exec_device = "cpu"
    else:
        st.warning("⚠️ Configuration incomplete")
        if st.button("⚙️ Go to Model Configuration", key="goto_config_from_run"):
            st.session_state["active_tab"] = "Model Configuration"
            st.rerun()

    # Run section
    st.markdown("---")
    st.markdown("#### ▶️ Execute Training")

    workspace = current_root / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)

    if all_valid and exec_config_path:
        st.success("✅ Ready to run")

        if st.button("🚀 Run BatteryML Training", type="primary", use_container_width=True, key="run_training_btn"):
            import time
            start_time = time.perf_counter()

            cmd = [
                sys.executable, "batteryml_web/core/run_batteryml_cli.py", "run",
                str(exec_config_path), "--workspace", str(workspace),
                "--train", "--eval", "--device", exec_device, "--seed", "0"
            ]

            result = run_command_with_progress(
                command=cmd,
                log_path=current_root / "logs" / "batteryml_training.log",
                steps=[
                    (0, "Initialize run"),
                    (10, "Check dataset readiness"),
                    (20, "Load config"),
                    (30, "Start BatteryML run command"),
                    (50, "Training model"),
                    (70, "Evaluating predictions"),
                    (85, "Collecting outputs"),
                    (95, "Writing report"),
                    (100, "Completed"),
                ],
                cwd=Path("."),
                env=get_safe_runtime_env(),
            )

            elapsed = time.perf_counter() - start_time
            st.session_state["last_run_result"] = result
            st.session_state["last_run_status"] = "success" if result["returncode"] == 0 else "failed"
            st.session_state["last_run_time"] = elapsed

            # Display results
            res_cols = st.columns(4)
            res_cols[0].metric("Status", "✅ Success" if result["returncode"] == 0 else "❌ Failed")
            res_cols[1].metric("Exit Code", result["returncode"])
            res_cols[2].metric("Runtime", f"{elapsed:.1f}s")
            res_cols[3].metric("Output", str(workspace))

            if result["returncode"] == 0:
                st.success(f"Training completed in {elapsed:.1f}s")
                if st.button("📈 View Results", type="primary", key="goto_results_after_run"):
                    st.session_state["active_tab"] = "Results & Report"
                    st.rerun()
            else:
                st.error("Training failed")
                with st.expander("Error Details"):
                    st.code(result.get("stderr", "")[-2000:], language="text")

            # Save log
            log_path = current_root / "training_log.txt"
            log_content = f"""Training Log
Command: {' '.join(cmd)}
Timestamp: {pd.Timestamp.now()}
Exit code: {result['returncode']}
Runtime: {elapsed:.1f}s

=== STDOUT ===
{result['stdout']}

=== STDERR ===
{result['stderr']}
"""
            log_path.write_text(log_content, encoding="utf-8")
    else:
        st.error("❌ Cannot run - validation failed")
        st.info("Complete configuration in 'Model Configuration' tab first")

if current_section == "results" and root:
    # Tab 6: Results & Report (includes Troubleshooting features)
    st.markdown("<div class='section-header'>📈 Results & Report</div>", unsafe_allow_html=True)
    st.caption("View training results, metrics, predictions, and download reports")
    render_staged_progress(
        [
            (10, "Locate workspace"),
            (30, "Load metrics"),
            (50, "Load predictions"),
            (70, "Generate GT vs Pred plot"),
            (90, "Prepare report links"),
            (100, "Results loaded"),
        ],
        title="Results loading",
    )

    # Results sub-tabs
    result_tabs = st.tabs(["📊 Overview", "📈 Metrics", "🔮 Predictions", "📋 Logs", "📉 Curves", "📄 Report", "🔍 Troubleshoot"])

    with result_tabs[0]:
        st.markdown("#### Run Summary")
        last_status = st.session_state.get("last_run_status", "N/A")
        last_time = st.session_state.get("last_run_time", 0)

        overview_cols = st.columns(4)
        overview_cols[0].metric("Status", "✅ Success" if last_status == "success" else "❌ Failed" if last_status == "failed" else "⚪ N/A")
        overview_cols[1].metric("Runtime", f"{last_time:.1f}s" if last_time > 0 else "N/A")
        overview_cols[2].metric("Dataset", st.session_state.get("dataset_for_run", "N/A"))
        overview_cols[3].metric("Model", st.session_state.get("model_for_run", "N/A"))

        st.markdown("#### Output Paths")
        st.code(f"""
Workspace:   {current_root}
Processed:   data/processed/{st.session_state.get('dataset_for_run', 'MATR')}
Config:      {st.session_state.get('config_path_for_run', 'N/A')}
        """)

    with result_tabs[1]:
        st.markdown("#### Model Metrics")

        # Try to load metrics from session or results reader
        ts_metrics = read_metrics(current_root)

        if ts_metrics.get("metrics"):
            metrics_df = pd.DataFrame(ts_metrics["metrics"])
            st.dataframe(metrics_df, use_container_width=True)

            # Metrics cards
            if "RMSE" in ts_metrics["metrics"]:
                m = ts_metrics["metrics"]
                met_cols = st.columns(4)
                met_cols[0].metric("RMSE", f"{m.get('RMSE', 'N/A')}")
                met_cols[1].metric("MAE", f"{m.get('MAE', 'N/A')}")
                met_cols[2].metric("MAPE", f"{m.get('MAPE', 'N/A')}")
                met_cols[3].metric("R²", f"{m.get('R2', 'N/A')}")
        else:
            st.info("📊 No metrics available yet. Run training first.")
            if st.button("🚀 Go to Run Training", key="goto_run_from_results"):
                st.session_state["active_tab"] = "Run Training"
                st.rerun()

    with result_tabs[2]:
        st.markdown("#### Predictions")

        ts_preds = read_predictions(current_root)
        if ts_preds.get("predictions"):
            st.json(ts_preds["predictions"])
            st.caption("Prediction visualization would show GT vs Pred scatter here.")
        else:
            st.info("🔮 No predictions available yet.")

    with result_tabs[3]:
        st.markdown("#### Training Log")

        ts_log = read_log(current_root)
        if ts_log:
            st.text_area("Log", value=ts_log[-5000:], height=400, key="results_log_viewer")
        else:
            st.info("📋 No log available yet.")

    with result_tabs[4]:
        st.markdown("#### Curves & Visualizations")

        figs = list_figures(current_root)
        if figs:
            st.caption(f"Found {len(figs)} figures")
            for fig in figs[:10]:
                st.write(fig)
        else:
            st.info("📉 No curves available yet.")
            st.caption("After training, this would show:")
            st.markdown("""
            - GT vs Pred scatter plot
            - Error histogram
            - Capacity fade curves
            - Training loss curves
            """)

    with result_tabs[5]:
        st.markdown("#### Report")

        report_html = generate_html_report(current_root)
        report_path = current_root / "report.html"
        report_path.write_text(report_html, encoding="utf-8")

        st.download_button(
            "📥 Download report.html",
            data=report_html,
            file_name="report.html",
            mime="text/html",
            use_container_width=True,
            key="download_report_btn"
        )
        st.caption(f"Saved to: `{report_path}`")

    with result_tabs[6]:
        st.markdown("#### 🔍 Troubleshooting")

        # LLM Configuration
        st.markdown("**LLM Configuration**")
        ts_cols = st.columns(3)
        ts_cols[0].text_input("Base URL", key="troubleshoot_llm_url")
        ts_cols[1].text_input("API Key", type="password", key="troubleshoot_llm_key")
        ts_cols[2].text_input("Model", key="troubleshoot_llm_model")

        ts_client = LLMClient(
            base_url=st.session_state.get("troubleshoot_llm_url") or None,
            api_key=st.session_state.get("troubleshoot_llm_key") or None,
            model=st.session_state.get("troubleshoot_llm_model") or None
        )

        if ts_client.is_configured():
            st.success("🟢 LLM configured")
        else:
            st.info("⚪ LLM not configured (optional)")

        st.divider()

        # Error diagnosis
        st.markdown("**Error Diagnosis**")
        fail_log = read_log(current_root)
        st.text_area("Log", value=fail_log[-2000:] if fail_log else "No log", height=200, key="troubleshoot_log")

        if st.button("🔍 Explain with LLM", key="explain_error_btn"):
            if not ts_client.is_configured():
                st.warning("Configure LLM first")
            elif not fail_log:
                st.warning("No log to analyze")
            else:
                render_staged_progress(
                    [
                        (10, "Connecting to LLM endpoint"),
                        (30, "Sending error log prompt"),
                        (70, "Waiting for response"),
                        (90, "Parsing diagnosis"),
                        (100, "Error explanation ready"),
                    ],
                    title="LLM error explanation",
                )
                diag = diagnose_failure(fail_log, config_path=current_root / "config.yaml", llm_client=ts_client)
                st.json(diag)

        # Result explanation
        st.divider()
        st.markdown("**Result Explanation**")
        if st.button("📖 Summarize Results", key="summarize_results_btn"):
            if not ts_client.is_configured():
                st.warning("Configure LLM first")
            else:
                render_staged_progress(
                    [
                        (10, "Connecting to LLM endpoint"),
                        (30, "Sending result summary prompt"),
                        (70, "Waiting for response"),
                        (90, "Parsing summary"),
                        (100, "Result summary ready"),
                    ],
                    title="LLM result summary",
                )
                expl = explain_results(ts_metrics, fail_log, llm_client=ts_client)
                st.json(expl)
