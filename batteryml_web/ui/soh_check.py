"""The SOH data check every run surface shows before an SOH run.

Run Model, Quick experience and Agent Workflow all call
:func:`render_soh_data_check` next to their Run button, so the same verdict,
wording and "Check SOH data" control appear wherever an SOH run can start.
The decision itself is :mod:`batteryml_web.core.soh_gate`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import streamlit as st
import yaml

from batteryml_web.core import soh_gate
from batteryml_web.core.config_task import TASK_SOH
from batteryml_web.ui.surfaces import SECONDARY_ROLE


def render_soh_data_check(dataset: str, task: str, config_path: Any, *, key: str, repo_root: str | Path) -> bool:
    """For an SOH run: show whether the dataset's data supports it; return whether it may run.

    A config existing for (dataset, SOH) says nothing about the cells: SOH
    needs a valid reference capacity and enough cycles with both a discharge
    capacity and a voltage trace. The verdict comes from
    :mod:`batteryml_web.core.soh_readiness`, computed on request and cached
    against the processed folder's fingerprint and the config's label policy
    (:mod:`batteryml_web.core.soh_gate`). Any other task returns True at once.
    """
    if task != TASK_SOH or not dataset or not config_path:
        return True
    try:
        policy = soh_gate.policy_from_config(config_path)
    except (OSError, ValueError, yaml.YAMLError) as exc:
        st.error(f"SOH data check unavailable: {exc}")
        return False
    verdict = soh_gate.cached_verdict(repo_root, dataset, policy)
    allowed, message = soh_gate.gate(verdict)
    if allowed:
        st.success(message)
        st.caption(f"Checked {verdict.get('checked_at', '')} for {dataset}; re-checked automatically after the processed data changes.")
        return True
    if allowed is False:
        st.error(message)
    else:
        st.info(message)
    if st.button("Check SOH data", key=f"{key}_soh_check", type=SECONDARY_ROLE):
        with st.spinner(f"Reading {dataset}'s processed cells…"):
            soh_gate.check_and_store(repo_root, dataset, policy)
        st.rerun()
    return False
