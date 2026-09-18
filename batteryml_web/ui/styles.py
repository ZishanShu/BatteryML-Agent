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

#: The sidebar's vertical rhythm.
#:
#: The default sidebar spends most of the first screen on padding: a tall
#: band above the title, a full text line of gap between every element, and
#: dividers with roughly a line of air on each side. The band above the title
#: is worth reclaiming. The gap between controls is not — it was cut to 4px
#: here once and the navigation read as one welded strip, so it is back to a
#: comfortable 0.7rem and only the header band stays tight.
#:
#: The values below were not guessed. Streamlit's sidebar heading is a 50px
#: box that occupies only 34px of layout — it overflows the element below it
#: by a constant ~16px. Zeroing the heading's padding, as a first attempt
#: did, lands the title on top of its own caption; the brand rule below now
#: pins that overflow explicitly instead of relying on the gap to hide it.
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
    /* Streamlit hangs a 16px bottom margin off this band, which is the gap
       between the collapse control and the brand title — measured, the
       forehead ended at 36px and the first content row began at 56px. The
       band's own height was already cut; this is the space *after* it. */
    margin-bottom: 0;
}
/* The content column's own gutters. The panel is 300px wide and left 240px
   for content whose widest line — "default (no project loaded)" — measures
   146px, so every button stretched across ~90px of slack and the column read
   as half empty.

   The gutter is what narrows here, not the panel. Streamlit writes the
   panel's width as an *inline* style because the sidebar is drag-resizable,
   so a stylesheet can only win with `!important` — and that was measured:
   the panel does go to 17rem, and dragging the handle then does nothing at
   all, because the inline width the drag writes is overridden on every
   frame. Twenty-eight pixels is not worth silently removing a control the
   user has, and anyone who wants the panel narrower can simply drag it.
   A gutter costs nothing and fixes what the eye was actually objecting to. */
[data-testid="stSidebarUserContent"] {
    /* 4px, which lands the brand title at 40px: the same line the main
       column's content starts on, so the two columns open level. */
    padding-top: 0.25rem;
    padding-bottom: 1rem;
    padding-left: 0.75rem;
    padding-right: 0.75rem;
}

/* The gap Streamlit puts between every stacked element in the sidebar, and
   the single value that sets the sidebar's vertical rhythm: measured in a
   browser, *every* top-level block is separated by exactly this and nothing
   else, from the project label down to the LLM status line.
   0.7rem (11.2px). It was taken down to 0.25rem (4px) to save height, and
   that was the wrong trade: at 4px the navigation buttons read as one
   welded strip rather than as a list of separate controls, which is the
   complaint this value now answers. A 40px control needs roughly a quarter
   of its own height around it before the eye accepts it as standing alone,
   and saving 90px in a column that had 600px of empty space beneath it
   bought nothing.

   The old floor of 0.4rem was not about rhythm. It existed because
   Streamlit's sidebar heading overflows its container by ~16px, so a
   smaller gap let the title land on its own caption. That overflow is now
   pinned by the brand rule below, whose negative margin is kept equal and
   opposite to this value — so the header's spacing is unchanged by anything
   here. Change one and the other must change with it; the test and UI-01
   both check the pair, not the number. */
[data-testid="stSidebar"] [data-testid="stVerticalBlock"] {
    gap: 0.7rem;
}

/* Dividers: keep the rule, drop the air around it. Held below the block gap
   so the rule never adds height of its own — measured, these margins collapse
   against the gap and contribute nothing, and the section break the divider
   marks comes out at the group margin plus a gap, against the 11.2px between
   two buttons. That contrast is what reads as a break rather than as one
   more row. */
[data-testid="stSidebar"] hr,
[data-testid="stSidebar"] [data-testid="stDivider"] {
    margin-top: 0.2rem;
    margin-bottom: 0.2rem;
}

/* "BatteryML", "Current project", "Protocol Workflow". Reduced, not zeroed. */
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] h4 {
    padding-top: 0.2rem;
    padding-bottom: 0.2rem;
}

/* The brand block at the very top: "BatteryML" over "Protocol Navigator".
   These two lines are one lockup and must read as one, which means the space
   between them has to be smaller than the space below them — and, before
   this rule, it was negative.

   Streamlit gives every markdown container `margin-bottom: -16px`. On a
   caption that is harmless; on the heading it collapses a 28px title into a
   12px container, so the element after it starts *inside* the title's own
   line box. Measured in a browser: the title's ink ran to y=84.0 while the
   caption beneath it began at y=74.4 — a 9.6px overlap, which is why
   "Protocol Navigator" sat jammed under "BatteryML" with its capitals
   touching the title's descenders.

   Replacing it with a known value lets the heading occupy the height it
   actually draws. Negative, not 0: the pair has to read as one, which means
   the space inside it must be smaller than the space below it. At 0 both
   came out the same and the subtitle was equally attached to the title above
   it and to "Current project" below — no grouping at all.

   **This value is kept equal and opposite to the block gap above.** Cancelling
   exactly one gap lands the subtitle against the heading's own 0.2rem bottom
   padding whatever that gap is, so the brand lockup holds still while the
   rest of the sidebar's rhythm is tuned: measured at 2.8px between the two
   lines of text at gap 0.4rem, and the same 2.8px at 0.25rem and at 0.7rem.
   The heading's
   padding is what guarantees that clearance, which is why this can tighten
   the pair without ever reaching the title's ink. If the gap changes again,
   change this with it — it is the pairing, not either number, that keeps the
   title off its caption.

   Scoped by `:has(> h3)` because the sidebar contains exactly one heading —
   the brand title — so this cannot reach "Current project", "Protocol
   Workflow", any caption or any navigation entry. Where `:has()` is
   unsupported the rule simply does not match and the header keeps its
   previous spacing, which is the same graceful failure as every other
   selector here. */
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"]:has(> h3) {
    margin-bottom: -0.7rem;
}

/* Captions — the grey secondary lines under each entry. Type size only.

   Their *spacing* is deliberately left alone: tightening a caption's margin
   was measured in a browser and clipped the second line of a wrapped caption
   ("...works offline, with no API key.") behind the button beneath it. So
   no margin, no height, no overflow here, and the line-height stays above 1
   so a wrapped caption still has room for both its lines. The block gap
   above does the spacing, and it cannot collide.

   Scoped to the sidebar. st.caption is used across the whole application —
   under metrics, beside figures, for the transform note on a result — and
   shrinking it globally would re-typeset pages this milestone is not
   about. */
[data-testid="stSidebar"] [data-testid="stCaptionContainer"],
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] p {
    font-size: 0.82rem;
    line-height: 1.35;
}

/* Navigation buttons: same size type, less box. */
[data-testid="stSidebar"] .stButton > button {
    padding-top: 0.3rem;
    padding-bottom: 0.3rem;
}
/* The settings disclosure was the one block that broke the rhythm: it carried
   0.15rem of its own margin on each side, so measured border-to-border it sat
   6.4px from the Prev/Next row above and 6.4px from the status line below
   while every other pair in the sidebar was 4px. Zeroed, not reduced — the
   block gap already separates it, and a control that supplies its own margin
   on top of that is the definition of an uneven list. Its own box, border and
   padding are untouched, so the control is the same size it was. */
[data-testid="stSidebar"] [data-testid="stExpander"] {
    margin-top: 0;
    margin-bottom: 0;
}

/* Group rhythm. The block gap above is the space *inside* a group — a label
   and its own value, a heading and its own list — and it is deliberately
   tight. This is the space *between* two groups, and it has to be clearly
   larger or the sidebar reads as one undifferentiated column: at a uniform
   gap, "Protocol Navigator" sat as close to "Current project" as "Current
   project" did to the project name belonging to it.

   1.6rem against the 0.7rem gap measures 36.8px between groups against
   11.2px within one — about 3.3:1, enough that the grouping is read rather
   than measured. The first group takes no top margin: its space above is
   the header band's, already set.

   Keyed off the class Streamlit writes for `st.container(key=...)`, which is
   the author's own name (see SIDEBAR_GROUP_KEYS in ui/workflow_sidebar.py),
   not one of the generated `css-1x2y3z` hashes this file may not rely on. A
   group whose key is ever renamed simply stops matching and falls back to
   the block gap — no overlap, nothing hidden. */
[data-testid="stSidebar"] .st-key-sb_project,
[data-testid="stSidebar"] .st-key-sb_entry,
[data-testid="stSidebar"] .st-key-sb_protocol,
[data-testid="stSidebar"] .st-key-sb_pager,
[data-testid="stSidebar"] .st-key-sb_foot {
    margin-top: 1.6rem;
}

/* The protocol group opens with the one rule in the sidebar, which already
   announces the boundary. Giving it the full group margin *as well* stacked
   two separations on top of each other, so it takes a smaller one and the
   rule does the rest. */
[data-testid="stSidebar"] .st-key-sb_protocol {
    margin-top: 0.7rem;
}

/* Prev/Next is the protocol list's own control, not a new subject, so it
   sits closer to the list than the groups do to each other. */
[data-testid="stSidebar"] .st-key-sb_pager {
    margin-top: 1.4rem;
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
