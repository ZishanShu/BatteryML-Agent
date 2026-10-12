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
disabled (any)     neutral, dashed outline, muted label, no lift
information        flat panels (4px, left rule, no hover, no shadow)
=================  ===============================================

The sidebar is styled by ``sidebar_style.py`` and is not touched here: every
rule below is scoped to the main column (``stMain``) or to page-level
tokens.
"""

from __future__ import annotations

import streamlit as st

#: Present in :data:`SITE_STYLE_CSS` so a page test can tell it was injected.
SITE_STYLE_MARKER = "bml-site-style"

SITE_STYLE_CSS = """
<style>
/* bml-site-style: the mentor's visual language, shared by every page. */
:root {
    --bml-ink: #14224a;
    --bml-muted: #66728b;
    --bml-blue: #287bf5;
    --bml-blue-deep: #1557d6;
    --bml-mint: #31c79b;
    --bml-line: rgba(53, 83, 135, 0.13);
    --bml-soft: rgba(238, 246, 255, 0.76);
    /* The app's read-out tokens (app.py: dataset, pipeline and assistant
       tiles) take the same palette; their flat, passive shape is unchanged. */
    --app-border: rgba(53, 83, 135, 0.24);
    --app-border-soft: rgba(53, 83, 135, 0.13);
    --app-muted: #66728b;
    --app-card-bg: rgba(255, 255, 255, 0.78);
    --app-card-bg-strong: rgba(238, 246, 255, 0.76);
}

[data-testid="stAppViewContainer"] {
    background:
        radial-gradient(circle at 92% 4%, rgba(170, 222, 255, 0.24), transparent 25rem),
        linear-gradient(180deg, #ffffff 0%, #fbfdff 58%, #f4f8ff 100%);
}

[data-testid="stMainBlockContainer"] {
    max-width: 1480px;
    padding-left: clamp(1.2rem, 4vw, 4.5rem);
    padding-right: clamp(1.2rem, 4vw, 4.5rem);
}

/* ------------------------------------------------------------ typography */

[data-testid="stMain"] h1,
[data-testid="stMain"] h2,
[data-testid="stMain"] h3,
[data-testid="stMain"] h4 {
    color: var(--bml-ink);
    letter-spacing: -0.035em;
}

[data-testid="stMain"] p,
[data-testid="stMain"] label,
[data-testid="stMain"] [data-testid="stCaptionContainer"] {
    color: var(--bml-muted);
}

/* A status message keeps its own colour: READY, RAW_ONLY, FAILED must not
   all fade to the same muted grey-blue as body copy. */
[data-testid="stMain"] [data-testid="stAlert"] p {
    color: inherit;
}

/* A metric value is a result, not body copy: it stays in ink. */
[data-testid="stMain"] [data-testid="stMetricValue"],
[data-testid="stMain"] [data-testid="stMetricValue"] * {
    color: var(--bml-ink);
}

/* --------------------------------------------------------------- actions */

[data-testid="stMain"] [data-testid="stButton"] > button,
[data-testid="stMain"] [data-testid="stDownloadButton"] > button,
[data-testid="stMain"] [data-testid="stFormSubmitButton"] > button {
    min-height: 2.75rem;
    border-radius: 12px;
    border-color: rgba(40,123,245,0.24);
    color: var(--bml-blue-deep);
    font-weight: 700;
    transition: transform 140ms ease, box-shadow 140ms ease, background 140ms ease;
}

/* Lift and glow only for a control that can act. */
[data-testid="stMain"] [data-testid="stButton"] > button:hover:enabled,
[data-testid="stMain"] [data-testid="stDownloadButton"] > button:hover:enabled,
[data-testid="stMain"] [data-testid="stFormSubmitButton"] > button:hover:enabled {
    border-color: rgba(40,123,245,0.42);
    background: rgba(40,123,245,0.06);
    box-shadow: 0 7px 18px rgba(40,123,245,0.12);
    transform: translateY(-1px);
}

/* PRIMARY: the one filled control in a section. */
[data-testid="stMain"] [data-testid="stBaseButton-primary"],
[data-testid="stMain"] [data-testid="stBaseButton-primaryFormSubmit"] {
    border-color: var(--bml-blue);
    background: linear-gradient(135deg, var(--bml-blue), var(--bml-blue-deep));
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
    border-color: var(--bml-blue-deep);
    background: linear-gradient(135deg, #3a8aff, var(--bml-blue-deep));
    color: #ffffff !important;
}

/* SECONDARY (Download, Refresh, Explain) is the shared rule above, as the
   mentor wrote it: an outlined surface with a cobalt hairline and label. */

/* NAVIGATION: goes to another page. A tinted wash with no outline — quieter
   than secondary, still visibly pressable. One step more specific than the
   shared rule above, which would otherwise outline it like a secondary. */
[data-testid="stMain"] [data-testid="stButton"] > [data-testid="stBaseButton-tertiary"] {
    border-color: transparent;
    background: rgba(40,123,245,0.07);
    font-weight: 650;
}

[data-testid="stMain"] [data-testid="stButton"] > [data-testid="stBaseButton-tertiary"]:hover:enabled {
    border-color: transparent;
    background: rgba(40,123,245,0.13);
    box-shadow: none;
    transform: none;
}

[data-testid="stMain"] [data-testid="stButton"] > button:focus-visible,
[data-testid="stMain"] [data-testid="stDownloadButton"] > button:focus-visible,
[data-testid="stMain"] [data-testid="stFormSubmitButton"] > button:focus-visible {
    outline: 2px solid var(--bml-blue-deep);
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
    border-color: rgba(102,114,139,0.38) !important;
    background: rgba(20,34,74,0.035) !important;
    color: #7a8599 !important;
    -webkit-text-fill-color: #7a8599;
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
    color: #7a8599 !important;
    -webkit-text-fill-color: #7a8599;
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
    border-color: var(--bml-line);
    border-radius: 12px;
    background: rgba(255,255,255,0.76);
}

/* -------------------------------------------- information and disclosure */

[data-testid="stMain"] [data-testid="stDataFrame"] {
    border-color: var(--bml-line);
    border-radius: 16px;
    background: rgba(255,255,255,0.72);
}

/* A disclosure is calmer than a control: a hairline panel with a smaller
   radius than a button (10px against 12px) and no lift. */
[data-testid="stMain"] [data-testid="stExpander"] details {
    border-color: var(--bml-line);
    border-radius: 10px;
    background: rgba(255,255,255,0.72);
}

[data-testid="stMain"] [data-testid="stExpander"] summary {
    color: var(--bml-ink);
    font-weight: 600;
}

/* Information cards keep their flat read-out shape (surfaces.py: 4px, left
   rule, no hover); only the palette follows the mentor's. */
[data-testid="stMain"] .bml-info-card {
    border-color: var(--bml-line);
    border-left-color: rgba(40,123,245,0.32);
    background: rgba(255,255,255,0.78);
    color: var(--bml-ink);
}
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
