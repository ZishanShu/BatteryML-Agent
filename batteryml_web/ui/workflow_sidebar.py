"""Sidebar navigation.

Two ways into the platform, deliberately shown as two:

* the **Agent Workflow** — describe the task in natural language and let the
  deterministic backend resolve, plan and (on an explicit click) run it;
* the **Protocol Workflow** — the numbered manual route a reviewer follows
  without any LLM (Route B in `CLAUDE.md` §1).

The agent is an alternative entry point, not step 3 of 4: numbering it
inside the sequence made it look like a mandatory stage of the manual
route, which it is not. Both routes reach the same validated execution
path.
"""

from __future__ import annotations

import streamlit as st

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
        "emoji": "🤖",
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
    """The project name to show, from session state only."""
    root = st.session_state.get("project_root")
    if root:
        return str(root).rstrip("/").split("/")[-1]
    return (
        st.session_state.get("project_name")
        or st.session_state.get("current_project")
        or "No project loaded"
    )


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


def _nav_button(step: dict, current_step: str) -> None:
    """One sidebar entry: the current step is text, the others are buttons."""
    if step["group"] == "workflow":
        label = f"{step['step_no']}. {step['emoji']} {step['label']}"
    else:
        label = f"{step['emoji']} {step['label']}"
    if current_step == step["key"]:
        st.markdown(f"**▶ {label}**")
        st.caption(step["description"])
    else:
        if st.button(label, key=f"sidebar_go_{step['key']}", use_container_width=True):
            set_current_step(step["key"])


def render_sidebar_workflow():
    with st.sidebar:
        st.markdown("### 🔋 BatteryML")
        st.caption("Protocol Navigator")

        st.markdown("**Project**")
        st.caption(current_project_label())

        st.divider()
        _nav_button(NAV_STEPS[0], st.session_state.get("current_step", "home"))

        st.divider()
        st.markdown("**Intelligent entry point**")
        st.caption("An alternative to the numbered route below — not a step in it.")
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
        label, color = llm_status_badge()
        st.markdown(
            f"<div style='font-size:0.9rem; color:{color}; font-weight:600;'>LLM: {label}</div>",
            unsafe_allow_html=True,
        )
