"""Open every page at the top.

Streamlit reruns the script on navigation but does not reset the browser's
scroll position, so arriving on a new page from the sidebar could land the
user halfway down it — at whatever offset the *previous* page happened to be
left at. Nothing in the page reports this; it just looks like content is
missing above the fold.

The fix is split in two so that the decision is checkable without a browser:

* :func:`page_changed` is pure. It compares the step about to be rendered
  with the one the previous completed run rendered, records the new one, and
  returns whether the page changed. No Streamlit, no DOM, no timing.
* :func:`render_scroll_to_top` is the thin browser half. When the pure half
  says the page changed, it injects a zero-height component whose script
  scrolls the parent document back to the top.

Why a component and not ``st.markdown``: Streamlit strips ``<script>`` from
markdown even with ``unsafe_allow_html=True``, so markdown cannot run this.
A component iframe can, and it is the only sanctioned way to reach the host
page. The iframe is zero-height and the app's CSS collapses its container,
so it occupies no space.

Why a serial number: Streamlit re-uses a component iframe whose parameters
are unchanged, and a re-used iframe does not re-execute its script. Stamping
each injection with an incrementing serial makes every injection a distinct
iframe, so the scroll fires on *every* navigation rather than only the first.

Detecting the change rather than being told about it is deliberate: it
covers every route into a page — the sidebar buttons, Prev/Next, the bottom
navigation, and the places that assign ``current_step`` directly — without
each of them having to remember to ask.

Injected *before* the page body for the same reason. Streamlit streams
elements to the browser as the script produces them, so an iframe emitted
first still runs while the rest of the page arrives — and a page that raises
half-way through (a missing dataset, an unreadable run directory) then still
opens at the top instead of silently keeping the old scroll offset. The
retries below cover the layout that is still arriving.
"""

from __future__ import annotations

from typing import MutableMapping

import streamlit as st
from streamlit.components import v1 as components

#: Session key holding the step rendered by the last *completed* script run.
#: A run that ends in ``st.rerun()`` never reaches the end of the script, so
#: it never updates this — which is exactly right: the click run and the
#: render run must not both count as "rendered".
LAST_RENDERED_STEP_KEY = "_last_rendered_step"

#: Session key for the injection counter described in the module docstring.
SCROLL_SERIAL_KEY = "_scroll_to_top_serial"

#: Emitted only on the run that opens a new page (see render_scroll_to_top).
#: Streamlit marks an element of the previous run ``data-stale="true"`` until
#: the new run re-draws its position. Layout blocks (tabs, column rows,
#: expanders) carry no such attribute themselves, so one is hidden while
#: everything in it is stale and nothing of the new page has arrived in it —
#: otherwise the previous page's tab bar stayed on screen, greyed, under the
#: new title.
HIDE_STALE_PAGE_CSS = """<style>
[data-testid="stMain"] [data-stale="true"],
[data-testid="stMain"] :is([data-testid="stTabs"], [data-testid="stLayoutWrapper"], [data-testid="stExpander"]):has([data-stale="true"]):not(:has([data-stale="false"])) {
    display: none !important;
}
</style>"""


def page_changed(session_state: MutableMapping, step: str) -> bool:
    """Whether ``step`` differs from the step rendered on the previous run.

    Records ``step`` as the newly rendered page, so the caller must invoke
    this exactly once per completed run.

    The first ever run returns False: there is nothing above the fold to
    scroll away from, and scrolling a freshly loaded page would be a no-op
    that is only ever visible as a flicker.
    """
    previous = session_state.get(LAST_RENDERED_STEP_KEY)
    session_state[LAST_RENDERED_STEP_KEY] = step
    return previous is not None and previous != step


def scroll_script(serial: int) -> str:
    """The injected page-top script, stamped with ``serial``.

    Several scroll targets are tried because the element that actually
    scrolls has changed name across Streamlit versions, and this repository
    cannot pin the version running on the server. A selector that matches
    nothing yields ``null`` and is skipped; the document's own scrolling
    element is the backstop that works regardless of markup.

    The retries exist because the new page's content is still being laid out
    when the iframe loads: scrolling to the top before the tall element
    above arrives leaves the viewport where it was. The window is bounded at
    ~0.9s, which is the trade-off — long enough for a page that reads the
    filesystem before drawing, short enough that it cannot fight a user who
    has already started scrolling the page they landed on.
    """
    return f"""
<!-- scroll-to-top #{serial} -->
<script>
(function () {{
    var host = window.parent;
    var doc = host.document;
    function toTop() {{
        var targets = [
            doc.querySelector('section.main'),
            doc.querySelector('[data-testid="stMain"]'),
            doc.querySelector('[data-testid="stAppViewContainer"] > section'),
            doc.scrollingElement,
            doc.documentElement,
        ];
        for (var i = 0; i < targets.length; i++) {{
            var el = targets[i];
            if (!el) {{ continue; }}
            if (typeof el.scrollTo === 'function') {{ el.scrollTo(0, 0); }}
            el.scrollTop = 0;
        }}
        host.scrollTo(0, 0);
    }}
    toTop();
    [0, 80, 200, 450, 900].forEach(function (delay) {{ host.setTimeout(toTop, delay); }});
}})();
</script>
"""


def render_scroll_to_top(step: str) -> bool:
    """Scroll the app back to the top when ``step`` is a newly opened page.

    Returns whether the scroll was injected, so a caller (and a test) can
    tell the two cases apart rather than guessing from the absence of a
    visible effect.
    """
    # One slot on *every* run, filled only on a page change. Emitting the
    # iframe on some runs and nothing on others shifted every element below
    # it by one position on the first rerun after navigating; Streamlit then
    # treated the page's st.tabs as a new widget (first tab selected) and
    # left the old page greyed out underneath. Measured on Run Model: the
    # first "Run BatteryML" click jumped to "Configuration", hid the run's
    # progress in the unselected "Command & Run" tab, and kept a faded copy
    # of the previous page below until the run ended.
    slot = st.empty()
    if not page_changed(st.session_state, step):
        return False
    serial = int(st.session_state.get(SCROLL_SERIAL_KEY, 0)) + 1
    st.session_state[SCROLL_SERIAL_KEY] = serial
    with slot.container():
        components.html(scroll_script(serial), height=0)
        # Only on this navigation run: the previous page's elements are
        # "stale" until the new page re-draws their positions, and a page
        # that reads a large data tree before drawing (Run Model's dataset
        # cards, Review Results' run history) left the previous page's
        # content greyed out under the new title for the whole time. Hiding
        # stale elements in the main area removes it at once. The rule lives
        # in this slot, so the next rerun within the page drops it and a
        # button press keeps Streamlit's usual behaviour.
        st.markdown(HIDE_STALE_PAGE_CSS, unsafe_allow_html=True)
    return True
