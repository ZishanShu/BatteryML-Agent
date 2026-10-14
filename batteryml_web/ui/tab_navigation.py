"""Moving between the tabs of one page, from a button.

Prepare Data and Run Model are each one page made of tabs (Prepare Data:
Dataset Overview → Preprocess & Inspect → Custom Dataset / Schema Mapping;
Run Model: Configuration → Dataset Readiness → Command & Run → Logs / Debug).
Two kinds of move happen inside them:

* a jump to a named tab ("View Dataset Overview" after preprocessing);
* the page's own **← Prev / Next →** buttons, which step through the tabs in
  order and only leave the page at its first or last tab.

``workflow_sidebar.set_current_step`` is the abstraction for changing
*page*; this module is the one for changing *tab*, and the bottom navigation
in ``app.py`` combines the two.

**How a tab is selected from code.** Streamlit 1.64's ``st.tabs`` takes
``key`` and ``default``. Measured in a browser: writing a label into
``st.session_state[key]`` before an untracked strip is built does nothing,
and a new ``default=`` on an existing widget is ignored (it is an *initial*
value). What works is changing the widget's key, so Streamlit builds a new
widget that honours ``default``. A jump therefore bumps a per-page nonce.

**How the open tab is known.** Prev/Next must start from the tab the user is
looking at, including one they clicked themselves. The strip is built with
``on_change="rerun"`` so it tracks state: a tab click reruns the script and
``st.session_state[key]`` holds the open label. That was measured in a
browser too, together with the jump above: both a clicked tab and a
button-driven move are reported correctly.

**Why a busy strip exists.** A tracked strip reruns the script on a tab
click, and a rerun stops a script that is in the middle of a blocking run.
Measured in a browser: clicking another tab during an 8-second job stopped
the job's script when the strip tracked state, and did not when it did not.
Run Model and Prepare Data start BatteryML and preprocessing *inside* the
script, so on the run a Run/Preprocess button triggers, the caller passes
``busy=True`` and the strip is built untracked under a different key. The
open tab is remembered across that run, so the next ordinary run rebuilds
the tracked strip on the same tab.

**Which tab a page opens on.** The tab last open on this visit to the page;
a page entered anew (another page was rendered in between) opens on its
first tab, unless a jump asked for a specific one. "A new visit" is read
from the page-change serial ``ui/scroll.py`` already keeps.

Every key this module writes is its own; the project, LLM settings,
selected dataset and other session state are never touched.
"""

from __future__ import annotations

from typing import Any, Mapping, MutableMapping, Optional, Sequence

import streamlit as st

from .scroll import SCROLL_SERIAL_KEY

#: Page identifiers for the tabbed pages.
PREPARE_PAGE = "prepare"
RUN_MODEL_PAGE = "run_model"

#: The Prepare Data tab labels, in order. The page builds its tabs from
#: this, so a label and its navigation target cannot drift apart.
PREPARE_TABS: tuple[str, ...] = (
    "Dataset Overview",
    "Preprocess & Inspect",
    "Custom Dataset / Schema Mapping",
)

#: The Run Model tab labels, in order.
RUN_MODEL_TABS: tuple[str, ...] = (
    "Configuration",
    "Dataset Readiness",
    "Command & Run",
    "Logs / Debug",
)

#: Every tabbed page and its tabs, in order.
PAGE_TABS: Mapping[str, tuple[str, ...]] = {
    PREPARE_PAGE: PREPARE_TABS,
    RUN_MODEL_PAGE: RUN_MODEL_TABS,
}

#: Which workflow step (``workflow_sidebar.NAV_STEPS`` key) is which tabbed
#: page. A step not listed here has no tabs.
STEP_PAGES: Mapping[str, str] = {
    "prepare_data": PREPARE_PAGE,
    "run_model": RUN_MODEL_PAGE,
}

#: Prepare Data's jump keys keep their historical names.
REQUEST_KEY = "prepare_tab_request"
NONCE_KEY = "prepare_tab_nonce"

#: The label of the overview tab, named once so a button and the tab it
#: opens cannot disagree.
DATASET_OVERVIEW_TAB = PREPARE_TABS[0]

#: What the navigation action is called wherever it appears.
VIEW_OVERVIEW_LABEL = "View Dataset Overview"


def _request_key(page: str) -> str:
    return f"{page}_tab_request"


def _nonce_key(page: str) -> str:
    return f"{page}_tab_nonce"


def _active_key(page: str) -> str:
    return f"{page}_tab_active"


def _labels(page: str) -> tuple[str, ...]:
    try:
        return PAGE_TABS[page]
    except KeyError:
        raise ValueError(f"Unknown tabbed page {page!r}; expected one of {sorted(PAGE_TABS)}.") from None


def neighbour(labels: Sequence[str], current: Optional[str], step: int) -> Optional[str]:
    """The tab ``step`` places away from ``current``, or None past either end.

    Pure, so the Prev/Next rule is testable without Streamlit. An unknown
    ``current`` counts as the first tab, which is what the page shows then.
    """
    if not labels:
        return None
    index = list(labels).index(current) if current in labels else 0
    target = index + step
    if 0 <= target < len(labels):
        return labels[target]
    return None


def request_tab(label: str, page: str = PREPARE_PAGE) -> None:
    """Ask for ``label`` to be the open tab of ``page`` on its next render.

    Records the request and nothing else. It does not rerun: the caller
    decides that, because a caller that has just written other state wants
    to finish doing so first.

    An unknown label is ignored rather than stored. Storing it would leave
    the page with a request it can never satisfy, and silently opening the
    first tab instead would be a different page than the one asked for.
    """
    if label in _labels(page):
        st.session_state[_request_key(page)] = label
        st.session_state[_nonce_key(page)] = int(st.session_state.get(_nonce_key(page), 0)) + 1


def pending_tab(page: str = PREPARE_PAGE) -> Optional[str]:
    """The requested tab, if a jump is pending."""
    value = st.session_state.get(_request_key(page))
    return str(value) if value in _labels(page) else None


def _visit(state: Mapping[str, Any]) -> int:
    return int(state.get(SCROLL_SERIAL_KEY, 0) or 0)


def active_tab(page: str) -> Optional[str]:
    """The tab open on this visit to ``page``, or None on a fresh visit."""
    stored = st.session_state.get(_active_key(page))
    if not isinstance(stored, dict) or stored.get("visit") != _visit(st.session_state):
        return None
    label = stored.get("tab")
    return label if label in _labels(page) else None


def _remember(state: MutableMapping[str, Any], page: str, label: str) -> None:
    state[_active_key(page)] = {"tab": label, "visit": _visit(state)}


def render_page_tabs(page: str, *, busy: bool = False) -> Sequence[Any]:
    """Build ``page``'s tab strip, honouring a pending jump.

    ``busy`` must be True on a run that executes a blocking job inside the
    page (see the module docstring): the strip is then built untracked, so a
    tab click cannot stop the job.

    The request is consumed here: it opens the tab once and then stops
    applying, so the user's own next click is not fought by a stale jump.
    """
    labels = _labels(page)
    requested = pending_tab(page)
    current = requested or active_tab(page) or labels[0]
    nonce = int(st.session_state.get(_nonce_key(page), 0))
    if busy:
        tabs = st.tabs(list(labels), default=current, key=f"{page}_tabs_busy_{nonce}")
        shown = current
    else:
        key = f"{page}_tabs_{nonce}"
        tabs = st.tabs(list(labels), default=current, key=key, on_change="rerun")
        tracked = st.session_state.get(key)
        shown = tracked if tracked in labels else current
    _remember(st.session_state, page, shown)
    if requested is not None:
        st.session_state.pop(_request_key(page), None)
    return tabs


def render_prepare_tabs(labels: Sequence[str] = PREPARE_TABS, *, busy: bool = False) -> Sequence[Any]:
    """Build the Prepare Data tab strip (see :func:`render_page_tabs`)."""
    if tuple(labels) != PREPARE_TABS:
        raise ValueError("Prepare Data's tabs are defined by PREPARE_TABS; pass those labels.")
    return render_page_tabs(PREPARE_PAGE, busy=busy)


def any_pressed(state: Mapping[str, Any], keys: Sequence[str]) -> bool:
    """Whether a button among ``keys`` triggered this run.

    A button's session value is True only on the run its click caused, and
    it is readable before the button is drawn, which is what lets a page
    choose a busy strip before it reaches the button.
    """
    return any(state.get(key) is True for key in keys)
