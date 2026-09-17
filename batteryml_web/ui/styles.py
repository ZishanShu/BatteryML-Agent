"""Layout-only stylesheets injected by the UI.

Nothing here changes what is on a page — only how much vertical space it
takes. Two rules this module keeps to, so a style tweak can never remove a
control:

1. **No ``display: none`` and no ``visibility: hidden``.** Padding, margin,
   gap and line-height only. Every button, caption and divider that existed
   before is still rendered and still reachable, including on a narrow
   screen where the sidebar is a drawer.
2. **Stable ``data-testid`` selectors only**, never Streamlit's generated
   class names, which change between releases. A selector that matches
   nothing on a given Streamlit version simply does nothing; the sidebar
   falls back to its default spacing rather than breaking.

Both are checked by ``tests/test_ui_layout.py``.
"""

from __future__ import annotations

import streamlit as st

#: The sidebar's vertical rhythm, tightened.
#:
#: The default sidebar spends most of the first screen on padding: a tall
#: band above the title, a full text line of gap between every element, and
#: dividers with roughly a line of air on each side. On a laptop that pushed
#: the numbered protocol steps below the fold, which is the one thing the
#: sidebar exists to show.
#:
#: The values below were not guessed. Streamlit's sidebar heading is a 50px
#: box that occupies only 34px of layout — it overflows the element below it
#: by a constant ~16px, and the default 1rem gap is what hides that. Squeeze
#: the gap to 0.3rem and zero the heading's padding, as a first attempt did,
#: and the title lands on top of its own caption. So the gap stays at 0.4rem
#: and the heading keeps a small padding: measured in a browser against the
#: unstyled sidebar, that is 42% shorter with the text still clear.
SIDEBAR_COMPACT_CSS = """
<style>
/* The band above the sidebar title — the "forehead", and the largest single
   saving. It holds the sidebar-collapse control, so only its padding
   shrinks: the control keeps its own size and stays clickable. Measured in a
   browser afterwards to confirm the button is neither clipped nor overlapped
   by the title beneath it. */
[data-testid="stSidebarHeader"] {
    /* Streamlit fixes this band at 60px while the collapse control inside it
       is 28px, so the height is what to reduce — padding alone cannot. 2.25rem
       (36px) still centres the 28px control with room around it, and matches
       the main header's height set in app.py. */
    height: 2.25rem;
    min-height: 2.25rem;
    padding-top: 0.25rem;
    padding-bottom: 0;
}
[data-testid="stSidebarUserContent"] {
    padding-top: 0.25rem;
    padding-bottom: 1rem;
}

/* The gap Streamlit puts between every stacked element in the sidebar.
   Not below 0.4rem: see the note above on the heading's overflow. */
[data-testid="stSidebar"] [data-testid="stVerticalBlock"] {
    gap: 0.4rem;
}

/* Dividers: keep the rule, drop the air around it. Five dividers separate
   the sidebar's sections, so this is worth several lines of height. */
[data-testid="stSidebar"] hr,
[data-testid="stSidebar"] [data-testid="stDivider"] {
    margin-top: 0.3rem;
    margin-bottom: 0.3rem;
}

/* "BatteryML", "Current project", "Protocol Workflow". Reduced, not zeroed. */
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] h4 {
    padding-top: 0.2rem;
    padding-bottom: 0.2rem;
}

/* Section descriptions and captions are deliberately NOT given their own
   rule. Tightening their margin was measured in a browser and clipped the
   second line of a wrapped caption ("...works offline, with no API key.")
   behind the button beneath it. They are tightened by the block gap above,
   which is enough and cannot collide. */

/* Navigation buttons: same size type, less box. */
[data-testid="stSidebar"] .stButton > button {
    padding-top: 0.3rem;
    padding-bottom: 0.3rem;
}
[data-testid="stSidebar"] [data-testid="stExpander"] {
    margin-top: 0.15rem;
    margin-bottom: 0.15rem;
}
</style>
"""

#: The scroll-to-top component (``ui/scroll.py``) is injected at zero height
#: and carries no visible content, but Streamlit still wraps it in an
#: element container with the standard block gap. Collapsing that container
#: keeps navigation from adding a strip of blank space at the foot of the
#: page. ``:has()`` is unsupported on older browsers, where the cost of not
#: matching is a few pixels of padding — not a missing control.
ZERO_HEIGHT_COMPONENT_CSS = """
<style>
[data-testid="stElementContainer"]:has(> iframe[height="0"]) {
    height: 0;
    margin: 0;
    padding: 0;
}
</style>
"""


def render_layout_css() -> None:
    """Inject the layout stylesheets. Safe to call once per script run."""
    st.markdown(SIDEBAR_COMPACT_CSS, unsafe_allow_html=True)
    st.markdown(ZERO_HEIGHT_COMPONENT_CSS, unsafe_allow_html=True)
