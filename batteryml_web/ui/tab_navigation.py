"""Moving between the tabs of one page, from a button.

Prepare Data is three tabs of a single page — Dataset Overview, Preprocess
& Inspect, Custom Dataset / Schema Mapping — so "go and look at the dataset
status now that preprocessing has run" is a move *within* the page, not a
page change. ``workflow_sidebar.set_current_step`` is the abstraction for
the other kind and is deliberately not used here: Prepare Data is already
the current step.

**Why this looks the way it does.** Streamlit 1.64's ``st.tabs`` takes both
``key`` and ``default``, which suggests two ways to select a tab from code.
Measured in a browser, neither works on its own:

* writing the label into ``st.session_state[key]`` before the widget is
  built does nothing — the tab stays where it was;
* passing a new ``default=`` on a rerun does nothing either, because
  ``default`` is the *initial* value of a widget that already exists.

What does work is changing the widget's key so Streamlit treats it as a new
widget, which then honours ``default``. That is the mechanism here, and it
is confined to this module so the rest of the app asks for a tab by name.

The cost is that the tab strip is rebuilt on a jump, which resets nothing
but the tab strip: the project, the LLM settings, the selected dataset and
every other session key are untouched, because this writes only its own two
keys.
"""

from __future__ import annotations

from typing import Any, Sequence

import streamlit as st

#: The Prepare Data tab labels, in order. The page builds its tabs from
#: this, so a label and its navigation target cannot drift apart.
PREPARE_TABS: tuple[str, ...] = (
    "Dataset Overview",
    "Preprocess & Inspect",
    "Custom Dataset / Schema Mapping",
)

#: The tab a jump asked for. Absent means "whatever the user last clicked".
REQUEST_KEY = "prepare_tab_request"

#: Bumped on every jump, purely to give the widget a new identity.
NONCE_KEY = "prepare_tab_nonce"

#: The label of the overview tab, named once so a button and the tab it
#: opens cannot disagree.
DATASET_OVERVIEW_TAB = PREPARE_TABS[0]

#: What the navigation action is called wherever it appears.
VIEW_OVERVIEW_LABEL = "View Dataset Overview"


def request_tab(label: str) -> None:
    """Ask for ``label`` to be the active tab on the next script run.

    Records the request and nothing else. It does not rerun: the caller
    decides that, because a caller that has just written other state wants
    to finish doing so first.

    An unknown label is ignored rather than stored. Storing it would leave
    the page with a request it can never satisfy, and silently opening the
    first tab instead would be a different page than the one asked for.
    """
    if label in PREPARE_TABS:
        st.session_state[REQUEST_KEY] = label
        st.session_state[NONCE_KEY] = int(st.session_state.get(NONCE_KEY, 0)) + 1


def pending_tab() -> str | None:
    """The requested tab, if a jump is pending."""
    value = st.session_state.get(REQUEST_KEY)
    return str(value) if value in PREPARE_TABS else None


def render_prepare_tabs(labels: Sequence[str] = PREPARE_TABS) -> Sequence[Any]:
    """Build the Prepare Data tab strip, honouring any pending jump.

    The request is consumed here: it opens the tab once and then stops
    applying, so the user's own next click is not fought by a stale jump.
    """
    requested = pending_tab()
    nonce = int(st.session_state.get(NONCE_KEY, 0))
    tabs = st.tabs(
        list(labels),
        default=requested or labels[0],
        key=f"prepare_tabs_{nonce}",
    )
    if requested is not None:
        st.session_state.pop(REQUEST_KEY, None)
    return tabs
