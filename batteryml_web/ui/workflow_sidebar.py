"""Sidebar navigation.

Two ways into the platform, shown as two: the **Agent Workflow** (describe
the task in natural language) and the numbered **Protocol Workflow** (the
manual route a reviewer follows without any LLM — Route B in `CLAUDE.md`
§1). The agent is not numbered inside the protocol sequence, because it is
not a stage of it; both routes reach the same validated execution path.

Label convention: the Agent Workflow entry is written exactly
``Agent Workflow`` — no emoji, no caret, no numbering — wherever it appears
(Home, sidebar, page heading, navigation buttons). Decorative prefixes on
that entry carried no information and made the same destination read as
three different things.
"""

from __future__ import annotations

import streamlit as st

from ..core.project_context import resolve_project
from .settings_panel import render_sidebar_settings
from .styles import render_layout_css

#: Ordered for Prev/Next. ``group`` decides where a step is drawn:
#: ``home`` at the top, ``agent`` in its own block, ``workflow`` in the
#: numbered protocol list. ``step_no`` exists only for ``workflow`` steps.
NAV_STEPS = [
    {
        "key": "home",
        "label": "Home",
        "emoji": "🏠",
        "description": "Create/load a project, run the quick-start demo, or describe a task",
        "group": "home",
    },
    {
        "key": "agent_workflow",
        "label": "Agent Workflow",
        # Deliberately empty: this entry carries no emoji anywhere.
        "emoji": "",
        "description": "Describe a task in plain language; the backend resolves, plans and runs it on an explicit click",
        "group": "agent",
    },
    {
        "key": "prepare_data",
        "label": "Prepare Data",
        "emoji": "🧪",
        "description": "Inspect datasets, preprocess raw data, and visualize features",
        "group": "workflow",
        "step_no": 1,
    },
    {
        "key": "run_model",
        "label": "Run Model",
        "emoji": "🚀",
        "description": "Select dataset, task, model, config, and execute BatteryML",
        "group": "workflow",
        "step_no": 2,
    },
    {
        "key": "review_results",
        "label": "Review Results",
        "emoji": "📈",
        "description": "Review metrics, predictions, logs, files, and reports",
        "group": "workflow",
        "step_no": 3,
    },
]

WORKFLOW_STEPS = [step for step in NAV_STEPS if step["group"] == "workflow"]
AGENT_STEPS = [step for step in NAV_STEPS if step["group"] == "agent"]
STEP_KEYS = [s["key"] for s in NAV_STEPS]


def set_current_step(step_key: str):
    if step_key not in STEP_KEYS:
        step_key = "home"
    st.session_state["current_step"] = step_key
    st.rerun()


def get_current_step_index() -> int:
    current = st.session_state.get("current_step", "home")
    if current not in STEP_KEYS:
        st.session_state["current_step"] = "home"
        return 0
    return STEP_KEYS.index(current)


def current_project_label() -> str:
    """The project a run would be written to, resolved the same way it is.

    Reads through :func:`resolve_project`, which is what the Agent Workflow
    page uses to build a workspace — so the name in the sidebar is the name
    the next run will actually use, not a separately-derived display string
    that can drift from it.
    """
    project = resolve_project(
        project_root=st.session_state.get("project_root"),
        project_name=st.session_state.get("project_name")
        or st.session_state.get("current_project"),
    )
    if project["source"] == "default":
        return f"{project['name']} (no project loaded)"
    return str(project["name"])


def llm_status_badge() -> tuple[str, str]:
    """(label, colour) for the sidebar LLM line, derived from session state."""
    configured = (
        bool(st.session_state.get("llm_base_url"))
        and bool(st.session_state.get("llm_api_key"))
        and bool(st.session_state.get("llm_model"))
    )
    status = st.session_state.get("llm_status", "not_configured")
    if not configured:
        return "Not connected", "#8a8f98"
    if status == "connected":
        return "Connected", "#1f8f4d"
    if status == "failed":
        return "Connection failed", "#c0392b"
    return "Configured / Not tested", "#b7791f"


def nav_label(step: dict) -> str:
    """The label for one navigation entry.

    A step with no emoji gets none: the Agent Workflow entry is exactly
    ``Agent Workflow``. Numbered protocol steps keep their number, because
    the number is information — it is the order a reviewer follows.
    """
    parts = []
    if step.get("step_no") is not None:
        parts.append(f"{step['step_no']}.")
    if step.get("emoji"):
        parts.append(str(step["emoji"]))
    parts.append(str(step["label"]))
    return " ".join(parts)


def _nav_button(step: dict, current_step: str) -> None:
    """One sidebar entry: the current step is text, the others are buttons.

    The current entry is marked by weight alone. A caret prefix decorated
    the label without saying anything the bold text did not.
    """
    label = nav_label(step)
    if current_step == step["key"]:
        st.markdown(f"**{label}**")
        st.caption(step["description"])
    else:
        if st.button(label, key=f"sidebar_go_{step['key']}", use_container_width=True):
            set_current_step(step["key"])


def render_sidebar_workflow():
    # Spacing only, and injected before the sidebar is drawn so the first
    # paint is already compact. See ui/styles.py for what it may not do.
    render_layout_css()
    with st.sidebar:
        st.markdown("### 🔋 BatteryML")
        st.caption("Protocol Navigator")

        st.markdown("**Current project**")
        st.caption(current_project_label())

        st.divider()
        _nav_button(NAV_STEPS[0], st.session_state.get("current_step", "home"))

        st.divider()
        current_step = st.session_state.get("current_step", "home")
        for step in AGENT_STEPS:
            _nav_button(step, current_step)

        st.divider()
        st.markdown("**Protocol Workflow**")
        st.caption("The manual route; works offline, with no API key.")
        for step in WORKFLOW_STEPS:
            _nav_button(step, current_step)

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
        render_sidebar_settings()

        st.divider()
        label, color = llm_status_badge()
        st.markdown(
            f"<div style='font-size:0.9rem; color:{color}; font-weight:600;'>LLM: {label}</div>",
            unsafe_allow_html=True,
        )
