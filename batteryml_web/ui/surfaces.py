"""Telling a control apart from a read-out, and one control apart from another.

The page had one shape for two meanings. A Streamlit button is a 40px box
with a 1px solid border and an 8px radius; the status cards were hand-built
``<div>``s with a 1px solid border and a 14px radius, sitting directly above
those buttons. Measured in a browser: border ``1px solid`` on both, radius
8px against 14px. Nothing else distinguished them, so "Select MATR" and the
readiness card above it read as two of the same thing and the eye had to
try one to find out.

The first pass made the *information* quieter: cards got a filled surface,
a lower-contrast border and an explicit ``cursor: default``. That fixed
card-versus-button and left a second confusion untouched, which is what
this module now also answers.

**Measured before this pass**, on the live app at 1500px — every action on
every page:

===========================  ==========  =====  ======  ========
role                         background  border radius  weight
===========================  ==========  =====  ======  ========
"Run BatteryML" (primary)    white       1px    8px     400
"View Dataset Overview"      white       1px    8px     400
"Go to Run Model" (nav)      white       1px    8px     400
===========================  ==========  =====  ======  ========

Three roles, one appearance, down to the pixel — because not one of the 54
action controls in the app passed ``type=`` to Streamlit, so every one of
them rendered as the default ``secondary``. Starting a two-hour training
run and navigating to another page were the same white box, and the eye had
no way to rank them.

So this module now carries a small, closed set of **semantic roles**, and
each role has exactly one appearance:

:data:`PRIMARY_ROLE`
    The one thing this section is for — Run, Preprocess, Analyze, Save.
    A filled control: the only filled thing on the page.
:data:`SECONDARY_ROLE`
    Supporting — Download, Refresh, Explain, Cancel. An outlined box.
    Streamlit's default, so a control that says nothing gets this.
:data:`NAV_ROLE`
    Goes somewhere else — Prev/Next, "Go to Run Model", the sidebar
    entries. A faint surface that never competes with a real action.

A call site names its role by passing the constant straight to Streamlit::

    st.button("Run BatteryML", type=PRIMARY_ROLE, key="run")

and not through a wrapper in this module. That is deliberate. This repo's
tests drive pages by replacing the *page module's* own ``st`` with a
recorder; a helper here would call ``surfaces.st`` instead, so the button
would vanish from every one of those doubles. Three role constants and one
stylesheet give the same vocabulary with none of that indirection, and the
rendered element stays the identical native ``<button>``.

Four rules this module keeps to:

1. **Never colour alone.** Primary differs from secondary in fill, border,
   weight *and* hover; a card differs from a button in fill, border weight,
   radius, cursor and the absence of hover. Someone who cannot see the
   colour difference still has several other signals.
2. **No fake controls.** A card is a ``<div>`` and stays one. Where
   something must be clickable it is a real ``st.button`` carrying a
   ``type=``, never a div restyled to look pressable.
3. **App-owned or stable-testid selectors only**, never Streamlit's
   generated hashes, which change between releases. ``stBaseButton-primary``
   and friends are Streamlit's own ``data-testid`` values, which is why the
   rules below can key off the role at all. A selector that matches nothing
   degrades to the default appearance rather than breaking a page.
4. **Neutral, not branded.** Streamlit's own primary fill is
   ``rgb(255, 75, 75)`` — measured — which is a consumer-app red and wrong
   for an instrument. The fill here is derived from the page's own text
   colour, so it stays neutral and follows a light or dark theme without
   this module knowing which one is active.
"""

from __future__ import annotations

import html
from typing import Any, Iterable, Mapping, Optional, Sequence, Tuple

import streamlit as st

#: Class on every informational card. Namespaced, so it cannot collide with
#: a Streamlit class now or after an upgrade.
INFO_CARD_CLASS = "bml-info-card"

#: The three action roles, as the ``type=`` value Streamlit understands.
#:
#: These are not decoration: ``type=`` is the native parameter, so a button
#: built through the helpers below is the same real ``<button>`` element it
#: always was, with the same keyboard behaviour and the same
#: ``data-testid``. Only the role — and therefore the appearance — is new.
PRIMARY_ROLE = "primary"
SECONDARY_ROLE = "secondary"
NAV_ROLE = "tertiary"

#: Streamlit's own ``data-testid`` for each role's rendered button. Stable
#: names, published by the framework, unlike the ``st-emotion-cache-*``
#: hashes that change on every release and that this file may not use.
PRIMARY_TESTID = "stBaseButton-primary"
SECONDARY_TESTID = "stBaseButton-secondary"
NAV_TESTID = "stBaseButton-tertiary"

#: The same three roles inside an ``st.form``, where Streamlit appends
#: ``FormSubmit`` to the kind.
PRIMARY_FORM_TESTID = "stBaseButton-primaryFormSubmit"
SECONDARY_FORM_TESTID = "stBaseButton-secondaryFormSubmit"

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
/* ---------------------------------------------------------------- actions

   One rule per role, keyed off Streamlit's own data-testid. Everything
   here is restrained on purpose: the strongest control on a page is a
   filled neutral box, not a coloured one.

   `color-mix` and the `Canvas` system colour are what keep this theme-
   agnostic. `currentColor` is the page's text colour, so a fill derived
   from it is dark on a light theme and light on a dark one; `Canvas` is
   the page's own background, so label-on-fill inverts with it. Neither
   needs this file to know which theme is active, and on a browser too old
   for `color-mix` the declaration is dropped and the button falls back to
   Streamlit's own appearance — quieter than intended, never invisible. */

/* PRIMARY — "the one thing this section is for". The only filled control
   on the page, so it is found without reading. */
/* `CanvasText`/`Canvas`, not `currentColor`, and the distinction matters:
   this rule sets `color` itself, and `currentColor` in any other property
   on the same element resolves to *that* new value. A fill written as
   `color-mix(… currentColor …)` beside `color: Canvas` therefore mixes the
   page background with itself and comes out invisible. The system colour
   pair is the same theme-following value without the circularity —
   CanvasText is the page's text colour, Canvas its background. */
[data-testid="{PRIMARY_TESTID}"],
[data-testid="{PRIMARY_FORM_TESTID}"] {{
    background: color-mix(in srgb, CanvasText 86%, transparent);
    border: 1px solid transparent;
    color: Canvas;
    font-weight: 600;
    cursor: pointer;
}}
/* `border-color` is restated, not inherited. Streamlit's own hover rule
   sets the border to its theme primary — measured on this build as
   `rgb(255, 0, 0)` — and the resting border here is transparent, so
   without this a pure red outline snapped around the button under the
   cursor: the one consumer-app colour this whole module avoids. */
[data-testid="{PRIMARY_TESTID}"]:hover:enabled,
[data-testid="{PRIMARY_FORM_TESTID}"]:hover:enabled {{
    background: CanvasText;
    border-color: transparent;
    color: Canvas;
}}
[data-testid="{PRIMARY_TESTID}"]:active:enabled,
[data-testid="{PRIMARY_FORM_TESTID}"]:active:enabled {{
    background: color-mix(in srgb, CanvasText 70%, transparent);
}}
/* Disabled says "not now" by losing the fill, not only by dimming: a flat
   dashed outline beside a filled sibling is legible without colour vision. */
[data-testid="{PRIMARY_TESTID}"]:disabled,
[data-testid="{PRIMARY_FORM_TESTID}"]:disabled {{
    background: transparent;
    border: 1px dashed color-mix(in srgb, CanvasText 30%, transparent);
    color: color-mix(in srgb, CanvasText 45%, transparent);
    font-weight: 500;
    cursor: not-allowed;
}}

/* SECONDARY — supporting. Still unmistakably a control: a real box with a
   border, just not a filled one. The border is firmer than Streamlit's
   default 0.2 alpha, which is what let it fade into a card. */
[data-testid="{SECONDARY_TESTID}"],
[data-testid="{SECONDARY_FORM_TESTID}"] {{
    border: 1px solid color-mix(in srgb, currentColor 38%, transparent);
    font-weight: 500;
    cursor: pointer;
}}
/* `color: inherit` first, and the order matters. Streamlit's hover rule
   repaints the label in its theme primary; `currentColor` then resolves to
   that red, so a fill mixed from `currentColor` came out a red wash —
   measured as `rgba(255, 75, 75, 0.1)` on a button whose rule asks only for
   a neutral tint. Pinning the label back to the inherited page colour keeps
   both the text and everything mixed from it neutral. */
[data-testid="{SECONDARY_TESTID}"]:hover:enabled,
[data-testid="{SECONDARY_FORM_TESTID}"]:hover:enabled {{
    color: inherit;
    border-color: currentColor;
    background: color-mix(in srgb, currentColor 7%, transparent);
}}

/* NAVIGATION — goes somewhere else rather than doing something.

   A faint resting surface, not a bare label. Streamlit's own `tertiary` is
   pure text with no box, and that was measured on the live sidebar: with
   no fill and no border the six destinations read as a list of headings
   rather than as six things to click, which is the opposite of the
   complaint this pass is answering. A hairline at 14% against secondary's
   38%, over a 4% wash, keeps navigation clearly third in the ranking while
   still looking pressable. */
[data-testid="{NAV_TESTID}"] {{
    background: color-mix(in srgb, currentColor 4%, transparent);
    border: 1px solid color-mix(in srgb, currentColor 14%, transparent);
    font-weight: 500;
    cursor: pointer;
}}
[data-testid="{NAV_TESTID}"]:hover:enabled {{
    color: inherit;
    background: color-mix(in srgb, currentColor 10%, transparent);
    border-color: color-mix(in srgb, currentColor 32%, transparent);
}}
/* In the sidebar a destination is a row, so its label starts at the left
   edge like the headings above it rather than floating in the centre.
   Geometry only — no padding, height or margin is touched here, because
   the sidebar's vertical rhythm is set in `ui/styles.py` and this pass is
   about affordance, not spacing. */
[data-testid="stSidebar"] [data-testid="{NAV_TESTID}"],
[data-testid="stSidebar"] [data-testid="{NAV_TESTID}"] > div {{
    justify-content: flex-start;
    text-align: left;
}}

/* Focus. Deliberately one rule for all three roles and deliberately not
   `:focus` — `:focus-visible` shows the ring for keyboard users and keeps
   it off a mouse click. Without this the filled primary above would have
   swallowed Streamlit's own focus treatment. */
[data-testid="{PRIMARY_TESTID}"]:focus-visible,
[data-testid="{SECONDARY_TESTID}"]:focus-visible,
[data-testid="{NAV_TESTID}"]:focus-visible,
[data-testid="{PRIMARY_FORM_TESTID}"]:focus-visible,
[data-testid="{SECONDARY_FORM_TESTID}"]:focus-visible {{
    outline: 2px solid currentColor;
    outline-offset: 2px;
}}

/* ------------------------------------------------------------ disclosure

   An expander summary was a white box with a 1px border and an 8px radius
   — measured, the same three values as a button — so "Show technical
   details" read as an action. Flattening the box leaves the caret and the
   label, which is what a disclosure should look like.

   Scoped to the main column. The sidebar's Settings expander is spaced by
   `ui/styles.py`, and this pass is not allowed to disturb it. */
[data-testid="stMain"] [data-testid="stExpander"] details {{
    border: 1px solid color-mix(in srgb, currentColor 16%, transparent);
    border-radius: 4px;
    background: transparent;
}}
[data-testid="stMain"] [data-testid="stExpander"] summary {{
    font-weight: 500;
    cursor: pointer;
}}
[data-testid="stMain"] [data-testid="stExpander"] summary:hover {{
    background: color-mix(in srgb, currentColor 6%, transparent);
}}

/* ---------------------------------------------------------------- cards */

/* An informational card. Quiet on purpose: a filled surface with a
   low-contrast hairline reads as a panel, while the border and radius of a
   button read as a control. Both borders were 1px solid before this, which
   is why the two were confusable.

   The radius is the second signal and the reason it is 4px against the
   button's 8px: a card and a control now differ in *shape*, not only in
   fill, so the distinction survives a greyscale screenshot. The left edge
   is heavier than the other three — a margin rule, the way a printed table
   marks a block of read-out — which no button in this app has. */
.{INFO_CARD_CLASS} {{
    border: 1px solid rgba(128, 128, 128, 0.22);
    border-left: 2px solid rgba(128, 128, 128, 0.42);
    border-radius: 4px;
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
   bolded, so the state survives being read without colour.

   `inline-block` with its own tight padding is what stops a status reading
   as a control: it shrinks to its text instead of spanning the card, and a
   full-width tinted bar is exactly what a button looks like. Letter-spacing
   and a small type size mark it as a label rather than a label *on* a
   control. No background of its own, so there is no rainbow of status
   chips — the colour is the word, and the weight carries it in greyscale. */
.{INFO_CARD_CLASS} .{INFO_CARD_CLASS}-status {{
    display: inline-block;
    font-weight: 700;
    font-size: 0.78rem;
    letter-spacing: 0.04em;
    margin-top: 0.35rem;
    padding: 0.05rem 0;
    cursor: default;
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
