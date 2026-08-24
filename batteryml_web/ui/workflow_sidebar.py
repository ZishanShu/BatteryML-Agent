from __future__ import annotations

import streamlit as st

NAV_STEPS = [
    {
        "key": "home",
        "label": "Home",
        "emoji": "🏠",
        "description": "Create/load project and run quick-start demo",
        "group": "home",
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


def render_sidebar_workflow():
    with st.sidebar:
        st.markdown("### 🔋 BatteryML")
        st.caption("Protocol Navigator")

        current_project = st.session_state.get("project_name") or st.session_state.get("current_project") or "No project loaded"
        if st.session_state.get("project_root"):
            current_project = st.session_state["project_root"].split("/")[-1]
        st.markdown("**Project**")
        st.caption(str(current_project))

        st.divider()

        current_step = st.session_state.get("current_step", "home")
        home = NAV_STEPS[0]
        home_label = f"{home['emoji']} {home['label']}"
        if current_step == home["key"]:
            st.markdown(f"**▶ {home_label}**")
            st.caption(home["description"])
        else:
            if st.button(home_label, key="sidebar_go_home", use_container_width=True):
                set_current_step(home["key"])

        st.divider()
        st.markdown("**Protocol Workflow**")
        for step in WORKFLOW_STEPS:
            label = f"{step['step_no']}. {step['emoji']} {step['label']}"
            if current_step == step["key"]:
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
        status = st.session_state.get("llm_status", "not_configured")
        if not bool(st.session_state.get("llm_base_url")) or not bool(st.session_state.get("llm_api_key")) or not bool(st.session_state.get("llm_model")):
            color = "#8a8f98"
            label = "Not connected"
        elif status == "connected":
            color = "#1f8f4d"
            label = "Connected"
        elif status == "failed":
            color = "#c0392b"
            label = "Connection failed"
        else:
            color = "#b7791f"
            label = "Configured / Not tested"
        st.markdown(f"<div style='font-size:0.9rem; color:{color}; font-weight:600;'>LLM: {label}</div>", unsafe_allow_html=True)
