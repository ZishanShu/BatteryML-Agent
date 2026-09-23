"""Telling a control apart from a read-out.

The page had one shape for two meanings. A Streamlit button is a 40px box
with a 1px solid border and an 8px radius; the status cards were hand-built
``<div>``s with a 1px solid border and a 14px radius, sitting directly above
those buttons. Measured in a browser: border ``1px solid`` on both, radius
8px against 14px. Nothing else distinguished them, so "Select MATR" and the
readiness card above it read as two of the same thing and the eye had to
try one to find out.

The fix is to make the *information* quieter rather than to make the
buttons louder. Streamlit's button already carries the affordances that
matter — a real ``<button>``, keyboard focus, a pointer cursor, a hover
state — and restyling every button on the site to win a contrast it already
has would be a redesign. So the cards get a filled surface, a lower-contrast
border and an explicit ``cursor: default``, and the buttons are left alone.

Three rules this module keeps to:

1. **Never colour alone.** A card differs from a button in fill, border
   weight, radius, cursor and the absence of hover. Someone who cannot see
   the colour difference still has four other signals.
2. **No fake controls.** A card is a ``<div>`` and stays one. Where
   something must be clickable it is a real ``st.button``, so it keeps
   native semantics, focus and keyboard activation.
3. **App-owned class names only**, never Streamlit's generated hashes,
   which change between releases. A selector that matches nothing degrades
   to the default appearance rather than breaking a page.
"""

from __future__ import annotations

import html
from typing import Iterable, Mapping, Optional, Sequence, Tuple

import streamlit as st

#: Class on every informational card. Namespaced, so it cannot collide with
#: a Streamlit class now or after an upgrade.
INFO_CARD_CLASS = "bml-info-card"

#: Container keys whose code block is capped. ``st.container(key=...)``
#: emits ``st-key-<key>``, an author-chosen name rather than a generated
#: hash, which is why the rule below can be written at all.
COMMAND_VIEWPORT_KEY = "bml_command_viewport"
LOG_VIEWPORT_KEY = "bml_log_viewport"

#: The ceiling for a code read-out, in lines.
#:
#: **A maximum, not a height.** Streamlit's ``st.code(height=...)`` is a
#: fixed box: it would pad a two-line command out to nine lines of empty
#: space. ``max-height`` bounds the tall case and leaves the short one
#: exactly as it was — measured, a 200-line log went from 4510px to the cap
#: while a one-line command stayed at 54px.
VIEWPORT_MAX_LINES = 9

#: One line of Streamlit's code block, in rem. **Measured, not assumed**:
#: the rendered ``line-height`` is 19.2px against a 16px root, and a first
#: attempt that guessed 1.6rem produced a 12-line viewport instead of a
#: 9-line one. Deriving the cap from this keeps the constant above honest.
_CODE_LINE_REM = 1.2

_VIEWPORT_MAX_REM = round(VIEWPORT_MAX_LINES * _CODE_LINE_REM, 2)

SURFACE_CSS = f"""
<style>
/* An informational card. Quiet on purpose: a filled surface with a
   low-contrast hairline reads as a panel, while the 1px solid border and
   8px radius of a button read as a control. Both borders were 1px solid
   before this, which is why the two were confusable. */
.{INFO_CARD_CLASS} {{
    border: 1px solid rgba(128, 128, 128, 0.22);
    border-radius: 10px;
    padding: 0.75rem 0.9rem;
    margin-bottom: 0.6rem;
    background: rgba(128, 128, 128, 0.06);
    /* Says "not a control" to the pointer, which is the one signal a
       screenshot cannot carry and a mouse user gets for free. */
    cursor: default;
}}
/* Deliberately no :hover rule. A surface that lights up under the cursor
   is claiming to be clickable, and this one is not. */
.{INFO_CARD_CLASS} .{INFO_CARD_CLASS}-title {{
    font-weight: 600;
    margin-bottom: 0.3rem;
    word-break: break-word;
}}
.{INFO_CARD_CLASS} .{INFO_CARD_CLASS}-row {{
    font-size: 0.86rem;
    opacity: 0.85;
    word-break: break-word;
}}
/* The status word keeps its own colour — those are the scientific
   readiness colours and are not this module's to retune — but it is also
   bolded, so the state survives being read without colour. */
.{INFO_CARD_CLASS} .{INFO_CARD_CLASS}-status {{
    font-weight: 700;
    margin-top: 0.35rem;
}}

/* Code read-outs: bounded above, never padded out.

   The log was the element that actually grew without limit — 200 lines of
   it measured 4510px, which is the "huge log-looking box". The command
   block does not grow at all (st.code scrolls horizontally), so the same
   cap is a no-op there today and exists to keep it that way if a command
   or a viewport ever makes it wrap. */
[class*="st-key-{COMMAND_VIEWPORT_KEY}"] pre,
[class*="st-key-{LOG_VIEWPORT_KEY}"] pre {{
    max-height: {_VIEWPORT_MAX_REM}rem;
    overflow: auto;
}}
</style>
"""


def render_surface_css() -> None:
    """Inject the surface stylesheet. Safe to call once per script run."""
    st.markdown(SURFACE_CSS, unsafe_allow_html=True)


def render_info_card(
    title: str,
    rows: Sequence[str] = (),
    *,
    status: Optional[str] = None,
    status_color: Optional[str] = None,
) -> None:
    """One informational card. Never clickable, never a control.

    Args:
        title: the card's subject, e.g. a dataset or file name.
        rows: secondary lines, rendered small and quiet.
        status: an optional state word, bolded so it reads without colour.
        status_color: the scientific readiness colour for ``status``. Passed
            through unchanged — readiness colours belong to the readiness
            layer, not to this one.

    Everything is HTML-escaped: these strings carry file names and paths
    from disk, and a path containing ``<`` would otherwise reshape the page.
    """
    parts = [
        f'<div class="{INFO_CARD_CLASS}">',
        f'<div class="{INFO_CARD_CLASS}-title">{html.escape(str(title))}</div>',
    ]
    for row in rows:
        parts.append(
            f'<div class="{INFO_CARD_CLASS}-row">{html.escape(str(row))}</div>'
        )
    if status:
        style = f' style="color:{html.escape(status_color)};"' if status_color else ""
        parts.append(
            f'<div class="{INFO_CARD_CLASS}-status"{style}>'
            f"{html.escape(str(status))}</div>"
        )
    parts.append("</div>")
    st.markdown("".join(parts), unsafe_allow_html=True)


def command_viewport():
    """A container whose code block is capped at :data:`VIEWPORT_MAX_LINES`."""
    return st.container(key=COMMAND_VIEWPORT_KEY)


def log_viewport():
    """A container whose log block is capped at :data:`VIEWPORT_MAX_LINES`."""
    return st.container(key=LOG_VIEWPORT_KEY)
