"""The BatteryML visual language: the mentor's design, on every page.

Two stylesheets, both from the mentor's Home (``codex/ui-optimization``):

* :data:`SITE_STYLE_CSS` — the shared language every page uses: the bright
  canvas, the content width, ink headings and muted body copy, the cobalt
  primary action, rounded inputs, and calm disclosures and tables. Injected
  once per run by ``app.py`` right after ``surfaces.render_surface_css``, so
  it refines that sheet's role rules rather than replacing them. It used to
  ship only with Home, which is why every other page looked like a different
  application (measured: neutral 8px buttons, grey captions, Streamlit's
  default canvas and width there; cobalt 12px buttons on Home).
* :data:`HOME_STYLE_CSS` — what only Home has: the hero, the start-path
  cards and the four-step overview. Injected by ``render_home_page``.

The rules keep Streamlit's native controls and the existing workflow logic,
and select only stable ``data-testid``/``data-baseweb`` attributes and the
app's own ``bml-``/``st-key-`` names — never Streamlit's generated classes.

Interaction hierarchy, one appearance per role (``surfaces.py`` names them):

=================  ===============================================
primary            filled cobalt gradient, white label
secondary          white surface, cobalt hairline and label
navigation         tinted cobalt wash, no outline
interrupting       Stop run: red hairline and label on white
disabled (any)     neutral, dashed outline, muted label, no lift
information        flat panels (4px, left rule, no hover, no shadow)
status             a pill with a dot and the word; no fill, no hover
disclosure         hairline panel, 10px (calmer than a 12px control)
=================  ===============================================

Type scale (one hierarchy for every page; the Home hero keeps its own):

=================  ======================================================
page title         ``st.header`` (h2) / ``h1``: 2rem, 700
section            ``st.subheader`` (h3): 1.5rem, 650
subsection         ``####`` (h4): 1.25rem, 600 — the mentor's own component
                   heading size
body / label       1rem / 0.875rem, muted ink; captions 0.875rem
metric value       1.75rem, ink, tabular digits
code, run ids      the theme's monospace, wrapped
family             the theme's (Source Sans), never overridden
=================  ======================================================

All values are ``--bml-*`` tokens in ``:root`` below, derived from the
mentor's seven palette values; the rules use the tokens.

Scope: the main column (``stMain``), plus — for form controls only — the
sidebar Settings block (``st-key-sb_foot``) and dialogs, so the one LLM
settings form looks the same wherever it is opened. The sidebar's layout,
spacing and navigation stay with ``sidebar_style.py``/``styles.py``.
"""

from __future__ import annotations

import streamlit as st

#: Present in :data:`SITE_STYLE_CSS` so a page test can tell it was injected.
SITE_STYLE_MARKER = "bml-site-style"

SITE_STYLE_CSS = """
<style>
/* bml-site-style: the mentor's visual language, shared by every page. */
:root {
    /* The mentor's palette (codex/ui-optimization, f058d51): the source of
       every token below. */
    --bml-ink: #14224a;
    --bml-muted: #66728b;
    --bml-blue: #287bf5;
    --bml-blue-deep: #1557d6;
    --bml-mint: #31c79b;
    --bml-line: rgba(53, 83, 135, 0.13);
    --bml-soft: rgba(238, 246, 255, 0.76);

    /* Colour roles. A page uses these names and never invents its own
       blue or grey. */
    --bml-primary: var(--bml-blue);
    --bml-primary-hover: var(--bml-blue-deep);
    --bml-primary-muted: rgba(40, 123, 245, 0.07);
    --bml-background: #fbfdff;
    --bml-surface: rgba(255, 255, 255, 0.78);
    --bml-surface-secondary: var(--bml-soft);
    --bml-surface-selected: rgba(40, 123, 245, 0.13);
    --bml-border: rgba(40, 123, 245, 0.24);
    --bml-border-subtle: var(--bml-line);
    --bml-text-primary: var(--bml-ink);
    --bml-text-secondary: var(--bml-muted);
    --bml-text-muted: #7a8599;
    --bml-focus-ring: var(--bml-blue-deep);
    --bml-success: #137a62;
    --bml-warning: #8a5300;
    --bml-error: #b42318;

    /* Type. The family is the theme's (Streamlit's Source Sans) on every
       page and no rule here overrides it; code and run ids use the theme's
       monospace. */
    --bml-font-mono: "Source Code Pro", monospace;
    --bml-fs-page-title: 2rem;
    --bml-fs-section-title: 1.5rem;
    --bml-fs-subsection-title: 1.25rem;
    --bml-fs-body: 1rem;
    --bml-fs-label: 0.875rem;
    --bml-fs-caption: 0.875rem;
    --bml-fs-metric: 1.75rem;
    --bml-fw-page-title: 700;
    --bml-fw-section-title: 650;
    --bml-fw-subsection-title: 600;

    /* Geometry. Spacing stays the mentor's (Streamlit's 1rem block gap, the
       clamp() side padding below); these are the only new measures. */
    --bml-page-max-width: 1480px;
    --bml-card-radius: 16px;
    --bml-button-radius: 12px;
    --bml-input-radius: 12px;
    --bml-disclosure-radius: 10px;
    --bml-control-height: 2.75rem;
    --bml-section-gap: 1.5rem;
    --bml-component-gap: 1rem;
    --bml-card-padding: 0.75rem 0.9rem;

    /* Interaction states. */
    --bml-state-hover: rgba(40, 123, 245, 0.06);
    --bml-state-hover-border: rgba(40, 123, 245, 0.42);
    --bml-state-selected: var(--bml-surface-selected);
    --bml-state-disabled-bg: rgba(20, 34, 74, 0.035);
    --bml-state-disabled-border: rgba(102, 114, 139, 0.38);
    --bml-state-loading: var(--bml-primary-hover);   /* 6.2:1 on white; the lighter blue is 4.0:1 */
    --bml-state-completed: var(--bml-success);
    --bml-state-failed: var(--bml-error);

    /* The app's read-out tokens (app.py: dataset, pipeline and assistant
       tiles) take the same palette; their flat, passive shape is unchanged. */
    --app-border: rgba(53, 83, 135, 0.24);
    --app-border-soft: var(--bml-border-subtle);
    --app-muted: var(--bml-text-secondary);
    --app-card-bg: var(--bml-surface);
    --app-card-bg-strong: var(--bml-surface-secondary);
}

[data-testid="stAppViewContainer"] {
    background:
        radial-gradient(circle at 92% 4%, rgba(170, 222, 255, 0.24), transparent 25rem),
        linear-gradient(180deg, #ffffff 0%, var(--bml-background) 58%, #f4f8ff 100%);
}

[data-testid="stMainBlockContainer"] {
    max-width: var(--bml-page-max-width);
    padding-left: clamp(1.2rem, 4vw, 4.5rem);
    padding-right: clamp(1.2rem, 4vw, 4.5rem);
}

/* ------------------------------------------------------------ typography */

[data-testid="stMain"] h1,
[data-testid="stMain"] h2,
[data-testid="stMain"] h3,
[data-testid="stMain"] h4 {
    color: var(--bml-text-primary);
    letter-spacing: -0.035em;
}

/* One hierarchy for every page. Measured before: page titles at 36px/600
   beside 28px sections and 24px subsections, so Review Results stacked
   three oversized headings. Home's own components (hero, start cards, step
   overview) keep the mentor's sizes: their rules are more specific or come
   later. */
[data-testid="stMain"] h1:not(.bml-home-title),
[data-testid="stMain"] h2 {
    font-size: var(--bml-fs-page-title);
    font-weight: var(--bml-fw-page-title);
    line-height: 1.2;
    letter-spacing: -0.04em;
}

[data-testid="stMain"] h3 {
    font-size: var(--bml-fs-section-title);
    font-weight: var(--bml-fw-section-title);
    line-height: 1.3;
}

[data-testid="stMain"] h4 {
    font-size: var(--bml-fs-subsection-title);
    font-weight: var(--bml-fw-subsection-title);
    line-height: 1.35;
}

[data-testid="stMain"] p,
[data-testid="stMain"] label,
[data-testid="stMain"] [data-testid="stCaptionContainer"] {
    color: var(--bml-text-secondary);
}

/* Code, paths and run ids: one monospace token, wrapped rather than
   pushing the page wider than the window. */
[data-testid="stMain"] code {
    font-family: var(--bml-font-mono);
    overflow-wrap: anywhere;
}

/* A status message keeps its own colour: READY, RAW_ONLY, FAILED must not
   all fade to the same muted grey-blue as body copy. */
[data-testid="stMain"] [data-testid="stAlert"] p {
    color: inherit;
}

/* A metric value is a result, not body copy: it stays in ink, one size on
   every page, with digits that line up. */
[data-testid="stMain"] [data-testid="stMetricValue"],
[data-testid="stMain"] [data-testid="stMetricValue"] * {
    color: var(--bml-text-primary);
    font-size: var(--bml-fs-metric);
    font-variant-numeric: tabular-nums;
}

/* --------------------------------------------------------------- actions */

[data-testid="stMain"] [data-testid="stButton"] > button,
[data-testid="stMain"] [data-testid="stDownloadButton"] > button,
[data-testid="stMain"] [data-testid="stFormSubmitButton"] > button {
    min-height: var(--bml-control-height);
    border-radius: var(--bml-button-radius);
    border-color: var(--bml-border);
    color: var(--bml-primary-hover);
    font-weight: 700;
    transition: transform 140ms ease, box-shadow 140ms ease, background 140ms ease;
}

/* Lift and glow only for a control that can act. */
[data-testid="stMain"] [data-testid="stButton"] > button:hover:enabled,
[data-testid="stMain"] [data-testid="stDownloadButton"] > button:hover:enabled,
[data-testid="stMain"] [data-testid="stFormSubmitButton"] > button:hover:enabled {
    border-color: var(--bml-state-hover-border);
    background: var(--bml-state-hover);
    box-shadow: 0 7px 18px rgba(40,123,245,0.12);
    transform: translateY(-1px);
}

/* PRIMARY: the one filled control in a section. */
[data-testid="stMain"] [data-testid="stBaseButton-primary"],
[data-testid="stMain"] [data-testid="stBaseButton-primaryFormSubmit"] {
    border-color: var(--bml-primary);
    background: linear-gradient(135deg, var(--bml-primary), var(--bml-primary-hover));
    color: #ffffff !important;
    box-shadow: 0 8px 18px rgba(40,123,245,0.22);
}

[data-testid="stMain"] [data-testid="stBaseButton-primary"],
[data-testid="stMain"] [data-testid="stBaseButton-primary"] *,
[data-testid="stMain"] [data-testid="stBaseButton-primaryFormSubmit"],
[data-testid="stMain"] [data-testid="stBaseButton-primaryFormSubmit"] * {
    color: #ffffff !important;
    -webkit-text-fill-color: #ffffff;
    fill: currentColor;
}

[data-testid="stMain"] [data-testid="stBaseButton-primary"]:hover:enabled,
[data-testid="stMain"] [data-testid="stBaseButton-primary"]:active:enabled,
[data-testid="stMain"] [data-testid="stBaseButton-primary"]:focus-visible,
[data-testid="stMain"] [data-testid="stBaseButton-primaryFormSubmit"]:hover:enabled {
    border-color: var(--bml-primary-hover);
    background: linear-gradient(135deg, #3a8aff, var(--bml-primary-hover));
    color: #ffffff !important;
}

/* SECONDARY (Download, Refresh, Summarize, Explain) is the shared rule
   above, as the mentor wrote it: an outlined surface with a cobalt hairline
   and label. */

/* NAVIGATION: goes to another page. A tinted wash with no outline — quieter
   than secondary, still visibly pressable. One step more specific than the
   shared rule above, which would otherwise outline it like a secondary. */
[data-testid="stMain"] [data-testid="stButton"] > [data-testid="stBaseButton-tertiary"] {
    border-color: transparent;
    background: var(--bml-primary-muted);
    font-weight: 650;
}

[data-testid="stMain"] [data-testid="stButton"] > [data-testid="stBaseButton-tertiary"]:hover:enabled {
    border-color: transparent;
    background: var(--bml-state-selected);
    box-shadow: none;
    transform: none;
}

[data-testid="stMain"] [data-testid="stButton"] > button:focus-visible,
[data-testid="stMain"] [data-testid="stDownloadButton"] > button:focus-visible,
[data-testid="stMain"] [data-testid="stFormSubmitButton"] > button:focus-visible {
    outline: 2px solid var(--bml-focus-ring);
    outline-offset: 2px;
}

/* DISABLED, any role: neutral, not a pale version of the action. The
   dashed outline comes from surfaces.py and is kept, so "not now" reads
   without colour; the fill and the lift are removed. The reason sits
   beneath the control on the page. */
[data-testid="stMain"] [data-testid="stButton"] > button:disabled,
[data-testid="stMain"] [data-testid="stDownloadButton"] > button:disabled,
[data-testid="stMain"] [data-testid="stFormSubmitButton"] > button:disabled {
    border-style: dashed !important;
    border-color: var(--bml-state-disabled-border) !important;
    background: var(--bml-state-disabled-bg) !important;
    color: var(--bml-text-muted) !important;
    -webkit-text-fill-color: var(--bml-text-muted);
    box-shadow: none !important;
    transform: none !important;
    cursor: not-allowed;
}

/* The label inside takes the muted colour and no fill of its own, so the
   control reads as one flat surface rather than a band behind the text. */
[data-testid="stMain"] [data-testid="stButton"] > button:disabled *,
[data-testid="stMain"] [data-testid="stDownloadButton"] > button:disabled *,
[data-testid="stMain"] [data-testid="stFormSubmitButton"] > button:disabled * {
    background: transparent !important;
    color: var(--bml-text-muted) !important;
    -webkit-text-fill-color: var(--bml-text-muted);
}

/* INTERRUPTING (Stop run): clearly not a forward action, without alarm —
   a red hairline and label on white, tinted only under the pointer. */
[data-testid="stMain"] [class*="st-key-agent_cancel_"] [data-testid="stButton"] > button {
    border-color: rgba(180, 35, 24, 0.38);
    background: #ffffff;
    color: var(--bml-error);
}

[data-testid="stMain"] [class*="st-key-agent_cancel_"] [data-testid="stButton"] > button:hover:enabled {
    border-color: var(--bml-error);
    background: rgba(180, 35, 24, 0.06);
    box-shadow: none;
}

[data-testid="stMain"] [class*="st-key-agent_cancel_"] [data-testid="stButton"] > button *,
[data-testid="stMain"] [class*="st-key-agent_cancel_"] [data-testid="stButton"] > button:hover:enabled * {
    color: var(--bml-error) !important;
    -webkit-text-fill-color: var(--bml-error);
}

/* ---------------------------------------------------------- selection */

[data-testid="stMain"] [data-testid="stTextInput"] input,
[data-testid="stMain"] [data-testid="stTextArea"] textarea,
[data-testid="stMain"] [data-testid="stNumberInput"] input,
[data-testid="stMain"] [data-testid="stSelectbox"] [data-baseweb="select"] > div,
[data-testid="stMain"] [data-testid="stMultiSelect"] [data-baseweb="select"] > div,
/* Streamlit 1.5x+ draws a selectbox as a react-aria combobox group, which the
   BaseWeb selector above (the mentor's, for older builds) no longer reaches. */
[data-testid="stMain"] [data-testid="stSelectbox"] [role="group"] {
    border-color: var(--bml-border-subtle);
    border-radius: var(--bml-input-radius);
    background: rgba(255,255,255,0.76);
}

/* THE LLM SETTINGS FORM, wherever it is opened (sidebar Settings, dialog):
   the same controls as the main column. Colours, borders and radii only —
   the sidebar's spacing belongs to ui/styles.py. */
[data-testid="stSidebar"] .st-key-sb_foot [data-testid="stTextInput"] input,
[data-testid="stSidebar"] .st-key-sb_foot [data-testid="stSelectbox"] [role="group"],
[data-testid="stSidebar"] .st-key-sb_foot [data-testid="stSelectbox"] [data-baseweb="select"] > div,
[role="dialog"] [data-testid="stTextInput"] input,
[role="dialog"] [data-testid="stSelectbox"] [role="group"],
[role="dialog"] [data-testid="stSelectbox"] [data-baseweb="select"] > div {
    border-color: var(--bml-border-subtle);
    border-radius: var(--bml-input-radius);
    background: #ffffff;
}

[data-testid="stSidebar"] .st-key-sb_foot [data-testid="stBaseButton-primary"],
[role="dialog"] [data-testid="stBaseButton-primary"] {
    border-color: var(--bml-primary);
    border-radius: var(--bml-button-radius);
    background: linear-gradient(135deg, var(--bml-primary), var(--bml-primary-hover));
    color: #ffffff;
}

[data-testid="stSidebar"] .st-key-sb_foot [data-testid="stBaseButton-primary"] *,
[role="dialog"] [data-testid="stBaseButton-primary"] * {
    color: #ffffff !important;
    -webkit-text-fill-color: #ffffff;
}

[data-testid="stSidebar"] .st-key-sb_foot [data-testid="stBaseButton-secondary"],
[role="dialog"] [data-testid="stBaseButton-secondary"] {
    border-color: var(--bml-border);
    border-radius: var(--bml-button-radius);
    background: #ffffff;
    color: var(--bml-primary-hover);
}

[data-testid="stSidebar"] .st-key-sb_foot [data-testid="stBaseButton-primary"]:disabled,
[data-testid="stSidebar"] .st-key-sb_foot [data-testid="stBaseButton-secondary"]:disabled,
[role="dialog"] [data-testid="stBaseButton-primary"]:disabled,
[role="dialog"] [data-testid="stBaseButton-secondary"]:disabled {
    border-style: dashed;
    border-color: var(--bml-state-disabled-border);
    background: var(--bml-state-disabled-bg);
    color: var(--bml-text-muted);
}

[data-testid="stSidebar"] .st-key-sb_foot [data-testid="stBaseButton-primary"]:disabled *,
[data-testid="stSidebar"] .st-key-sb_foot [data-testid="stBaseButton-secondary"]:disabled *,
[role="dialog"] [data-testid="stBaseButton-primary"]:disabled *,
[role="dialog"] [data-testid="stBaseButton-secondary"]:disabled * {
    color: var(--bml-text-muted) !important;
    -webkit-text-fill-color: var(--bml-text-muted);
}

[data-testid="stSidebar"] .st-key-sb_foot [data-testid="stBaseButton-primary"]:focus-visible,
[data-testid="stSidebar"] .st-key-sb_foot [data-testid="stBaseButton-secondary"]:focus-visible,
[role="dialog"] [data-testid="stBaseButton-primary"]:focus-visible,
[role="dialog"] [data-testid="stBaseButton-secondary"]:focus-visible {
    outline: 2px solid var(--bml-focus-ring);
    outline-offset: 2px;
}

/* -------------------------------------------- information and disclosure */

[data-testid="stMain"] [data-testid="stDataFrame"] {
    border-color: var(--bml-border-subtle);
    border-radius: var(--bml-card-radius);
    background: rgba(255,255,255,0.72);
}

/* A disclosure is calmer than a control: a hairline panel with a smaller
   radius than a button (10px against 12px) and no lift. */
[data-testid="stMain"] [data-testid="stExpander"] details {
    border-color: var(--bml-border-subtle);
    border-radius: var(--bml-disclosure-radius);
    background: rgba(255,255,255,0.72);
}

[data-testid="stMain"] [data-testid="stExpander"] summary {
    color: var(--bml-text-primary);
    font-weight: 600;
}

/* Information cards keep their flat read-out shape (surfaces.py: 4px, left
   rule, no hover); only the palette follows the mentor's. */
[data-testid="stMain"] .bml-info-card {
    border-color: var(--bml-border-subtle);
    border-left-color: rgba(40,123,245,0.32);
    background: var(--bml-surface);
    color: var(--bml-text-primary);
}

/* STATUS (a run's COMPLETED / FAILED / CANCELLED / RUNNING): a word with a
   dot, read, never pressed. A pill rather than the 12px box of a control,
   no fill to press, no hover, the default cursor. The word carries the
   state, so it reads without colour. */
[data-testid="stMain"] .bml-status {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    padding: 0.1rem 0.6rem;
    border: 1px solid currentColor;
    border-radius: 999px;
    font-size: var(--bml-fs-caption);
    font-weight: 700;
    letter-spacing: 0.04em;
    color: var(--bml-text-secondary);
    cursor: default;
    user-select: text;
}

[data-testid="stMain"] .bml-status::before {
    content: "";
    width: 0.5rem;
    height: 0.5rem;
    border-radius: 50%;
    background: currentColor;
}

[data-testid="stMain"] .bml-status--completed { color: var(--bml-state-completed); }
[data-testid="stMain"] .bml-status--failed { color: var(--bml-state-failed); }
[data-testid="stMain"] .bml-status--cancelled { color: var(--bml-warning); }
[data-testid="stMain"] .bml-status--running { color: var(--bml-state-loading); }
</style>
"""

HOME_STYLE_CSS = """
<style>
.bml-home-hero {
    position: relative;
    overflow: hidden;
    display: flex;
    align-items: flex-end;
    justify-content: space-between;
    gap: 2rem;
    min-height: 178px;
    padding: 2.2rem 2.4rem;
    margin: 0 0 1.25rem;
    border: 1px solid rgba(126, 181, 238, 0.28);
    border-radius: 24px;
    background:
        linear-gradient(115deg, rgba(255,255,255,0.96), rgba(238,248,255,0.86)),
        radial-gradient(circle at 82% 10%, rgba(68, 203, 170, 0.3), transparent 18rem);
    box-shadow: 0 18px 50px rgba(71, 125, 190, 0.12);
}

.bml-home-hero::after {
    content: "";
    position: absolute;
    width: 310px;
    height: 310px;
    right: -88px;
    top: -150px;
    border: 34px solid rgba(48, 198, 162, 0.13);
    border-radius: 50%;
    transform: rotate(-24deg);
}

.bml-home-kicker {
    position: relative;
    z-index: 1;
    margin-bottom: 0.55rem;
    color: var(--bml-blue-deep);
    font-size: 0.78rem;
    font-weight: 800;
    letter-spacing: 0.13em;
    text-transform: uppercase;
}

.bml-home-title {
    position: relative;
    z-index: 1;
    margin: 0;
    color: var(--bml-ink);
    font-size: clamp(2.1rem, 4vw, 4rem);
    font-weight: 800;
    letter-spacing: -0.065em;
    line-height: 0.98;
}

.bml-home-title span { color: var(--bml-blue); }

.bml-home-subtitle {
    position: relative;
    z-index: 1;
    max-width: 620px;
    margin: 0.85rem 0 0;
    color: var(--bml-muted);
    font-size: 1rem;
}

.bml-home-live {
    position: relative;
    z-index: 1;
    display: inline-flex;
    align-items: center;
    gap: 0.55rem;
    align-self: flex-start;
    padding: 0.62rem 0.9rem;
    border: 1px solid rgba(49, 199, 155, 0.25);
    border-radius: 999px;
    color: #137a62;
    background: rgba(255,255,255,0.66);
    font-size: 0.8rem;
    font-weight: 700;
    white-space: nowrap;
}

.bml-home-live::before {
    content: "";
    width: 0.52rem;
    height: 0.52rem;
    border-radius: 50%;
    background: var(--bml-mint);
    box-shadow: 0 0 0 5px rgba(49,199,155,0.12);
}

.bml-start-card {
    min-height: 154px;
    padding: 1.2rem 1.25rem 0.85rem;
    border: 1px solid var(--bml-line);
    border-radius: 18px;
    background: rgba(255,255,255,0.84);
    box-shadow: 0 10px 28px rgba(75, 117, 170, 0.08);
}

/* Start choices are real Streamlit buttons, styled as one-piece cards. */
[class*="st-key-home_choice_"] [data-testid="stBaseButton-secondary"] {
    width: 100%;
    min-height: 142px;
    padding: 1.15rem 1.2rem;
    border: 1px solid var(--bml-line);
    border-radius: 18px;
    background: rgba(255,255,255,0.84);
    color: var(--bml-ink);
    box-shadow: 0 10px 28px rgba(75,117,170,0.08);
    text-align: left;
    white-space: pre-wrap;
    transition: transform 180ms ease, box-shadow 180ms ease, border-color 180ms ease, background 180ms ease;
}

/* Streamlit renders the label inside one or more child nodes (usually a
   <p>).  Pin the colour on the whole button subtree so a theme rule cannot
   repaint a card label when the pointer moves over it.  Entry cards are
   light surfaces, so their copy stays dark in every non-selected state. */
[class*="st-key-home_choice_"] [data-testid="stBaseButton-secondary"] *,
[class*="st-key-home_choice_"] [data-testid="stBaseButton-secondary"] {
    color: var(--bml-ink) !important;
    -webkit-text-fill-color: var(--bml-ink);
}

[class*="st-key-home_choice_"] [data-testid="stBaseButton-secondary"]:hover {
    transform: translateY(-4px);
    border-color: rgba(40,123,245,0.42);
    background: linear-gradient(145deg, rgba(239,247,255,0.98), rgba(255,255,255,0.94));
    box-shadow: 0 18px 34px rgba(40,123,245,0.14);
    color: var(--bml-ink);
}

/* Selection is state owned by render_home_page.  Do not key this treatment
   to the Quick card: Quick is the default path, but it must stop looking
   selected as soon as the user chooses Create or Existing.  The explicit
   marker selectors support both the current inline state rule and a future
   data-selected/class marker without coupling the visual layer to session
   state. */
[class*="st-key-home_choice_"].bml-choice-selected [data-testid="stBaseButton-secondary"],
[class*="st-key-home_choice_"][data-selected="true"] [data-testid="stBaseButton-secondary"],
[class*="st-key-home_choice_"][aria-current="true"] [data-testid="stBaseButton-secondary"] {
    border-color: rgba(40,123,245,0.7) !important;
    background: linear-gradient(145deg, #eaf4ff, #ffffff) !important;
    box-shadow: 0 18px 36px rgba(40,123,245,0.18) !important;
    transform: translateY(-4px);
}

[class*="st-key-home_choice_"].bml-choice-selected [data-testid="stBaseButton-secondary"]:focus-visible,
[class*="st-key-home_choice_"][data-selected="true"] [data-testid="stBaseButton-secondary"]:focus-visible,
[class*="st-key-home_choice_"][aria-current="true"] [data-testid="stBaseButton-secondary"]:focus-visible {
    outline: 2px solid var(--bml-blue-deep);
    outline-offset: 3px;
}

.bml-choice-caption {
    margin-top: 0.55rem;
    color: var(--bml-muted);
    font-size: 0.78rem;
    text-align: center;
}

.bml-detail-context {
    display: flex;
    align-items: baseline;
    gap: 0.7rem;
    margin: 2rem 0 0.35rem;
    padding: 0.75rem 0 0.7rem;
    border-top: 1px solid rgba(40,123,245,0.18);
    border-bottom: 1px solid rgba(40,123,245,0.12);
    animation: bml-panel-enter 260ms ease-out both;
}

@keyframes bml-panel-enter {
    from { opacity: 0; transform: translateY(12px) scale(0.99); }
    to { opacity: 1; transform: translateY(0) scale(1); }
}

.bml-detail-context .bml-detail-eyebrow {
    color: var(--bml-blue-deep);
    font-size: 0.7rem;
    font-weight: 800;
    letter-spacing: 0.1em;
    text-transform: uppercase;
}

.bml-detail-context strong {
    color: var(--bml-ink);
    font-size: 1rem;
    font-weight: 750;
}

.bml-detail-context .bml-detail-hint {
    margin-left: auto;
    color: var(--bml-muted);
    font-size: 0.78rem;
}

.bml-start-card.recommended {
    border-color: rgba(40,123,245,0.35);
    background: linear-gradient(145deg, rgba(239,247,255,0.98), rgba(255,255,255,0.88));
    box-shadow: 0 14px 32px rgba(40,123,245,0.13);
}

.bml-start-card h3 {
    margin: 0;
    color: var(--bml-ink);
    font-size: 1.1rem;
    letter-spacing: -0.02em;
}

.bml-start-card p {
    min-height: 2.7rem;
    margin: 0.55rem 0 0;
    color: var(--bml-muted);
    font-size: 0.86rem;
    line-height: 1.45;
}

.bml-start-card .bml-tag {
    display: inline-flex;
    margin-bottom: 0.65rem;
    padding: 0.22rem 0.55rem;
    border-radius: 999px;
    color: var(--bml-blue-deep);
    background: rgba(40,123,245,0.1);
    font-size: 0.68rem;
    font-weight: 800;
    letter-spacing: 0.06em;
    text-transform: uppercase;
}

.bml-process-overview {
    padding: 1.35rem 1.45rem 1.2rem;
    border: 1px solid var(--bml-line);
    border-radius: 20px;
    background: rgba(255,255,255,0.7);
}

.bml-process-heading {
    display: flex;
    align-items: baseline;
    justify-content: space-between;
    gap: 1rem;
    margin-bottom: 1rem;
}

.bml-process-heading h2 {
    margin: 0;
    color: var(--bml-ink);
    font-size: 1.25rem;
    /* Pinned to what this heading measured before the shared type scale
       existed (600, -0.035em, 1.2): it is a component heading, not a page
       title, and the shared h2 rule would otherwise make it 700. */
    font-weight: 600;
    letter-spacing: -0.035em;
    line-height: 1.2;
}

.bml-process-heading span {
    color: var(--bml-muted);
    font-size: 0.82rem;
}

.bml-process-steps {
    display: grid;
    grid-template-columns: repeat(4, minmax(0, 1fr));
    gap: 0.7rem;
}

.bml-process-step {
    position: relative;
    min-height: 104px;
    padding: 0.9rem;
    border: 1px solid rgba(53,83,135,0.1);
    border-radius: 14px;
    background: rgba(247,250,255,0.9);
}

.bml-process-step::after {
    content: "→";
    position: absolute;
    top: 1.55rem;
    right: -0.7rem;
    color: #9db3d2;
    font-size: 1.1rem;
    font-weight: 700;
}

.bml-process-step:last-child::after { content: ""; }

.bml-process-step .bml-step-no {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 1.55rem;
    height: 1.55rem;
    margin-bottom: 0.45rem;
    border-radius: 50%;
    color: white;
    background: var(--bml-blue);
    font-size: 0.72rem;
    font-weight: 800;
}

.bml-process-step strong {
    display: block;
    color: var(--bml-ink);
    font-size: 0.88rem;
}

.bml-process-step small {
    display: block;
    margin-top: 0.28rem;
    color: var(--bml-muted);
    font-size: 0.75rem;
    line-height: 1.35;
}

/* "← Choose another start path": a navigation button sized to its label
   had no horizontal padding (165px, text touching both edges). 240px wide
   (user request) with side padding so the label never touches the edge;
   capped by the column so it never overflows. */
.st-key-home_choose_another [data-testid="stButton"] > button {
    width: 240px;
    max-width: 100%;
    padding-left: 1rem;
    padding-right: 1rem;
    justify-content: center;
}

@media (max-width: 900px) {
    .bml-process-steps { grid-template-columns: repeat(2, minmax(0, 1fr)); }
    .bml-process-step::after { content: ""; }
}
</style>
"""


def render_site_style() -> None:
    """Inject the shared visual language. Every page, after the surface styles."""
    st.markdown(SITE_STYLE_CSS, unsafe_allow_html=True)


def render_home_style() -> None:
    """Inject Home's own components (hero, start paths, step overview)."""
    st.markdown(HOME_STYLE_CSS, unsafe_allow_html=True)
