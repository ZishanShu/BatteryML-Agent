"""Visual treatment for the persistent BatteryML navigation rail.

The sidebar is rendered on every page, while ``home_style`` is only injected
by the landing page.  Keeping this sheet separate makes the sidebar use the
same light, cobalt-and-mint language everywhere without reopening the compact
layout rules in :mod:`ui.styles`.

Selectors use Streamlit's stable ``data-testid`` attributes and the explicit
``st.container(key=...)`` names from ``workflow_sidebar.py``.  The sheet is
purely visual: navigation structure, copy, and button callbacks stay in the
existing renderer.
"""

from __future__ import annotations


SIDEBAR_THEME_CSS = r"""
<style>
/* The main workspace uses a white canvas with cobalt actions.  Give the
   navigation rail its own softly tinted canvas so it reads as one product,
   while still separating it from the content column. */
[data-testid="stSidebar"] {
    --bml-side-ink: #14224a;
    --bml-side-muted: #66728b;
    --bml-side-blue: #287bf5;
    --bml-side-blue-deep: #1557d6;
    --bml-side-line: rgba(53, 83, 135, 0.15);
    background:
        radial-gradient(circle at 12% 2%, rgba(180, 224, 255, 0.44), transparent 17rem),
        linear-gradient(180deg, #f8fbff 0%, #f1f6ff 58%, #edf4ff 100%);
    border-right: 1px solid rgba(87, 137, 205, 0.14);
}

[data-testid="stSidebar"] > div:first-child {
    background: transparent;
}

[data-testid="stSidebar"] [data-testid="stSidebarUserContent"] {
    color: var(--bml-side-ink);
}

[data-testid="stSidebar"] [data-testid="stMarkdownContainer"],
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] {
    color: var(--bml-side-muted);
}

/* A compact brand tile gives the rail the same raised-card language as the
   home hero and entry cards. */
[data-testid="stSidebar"] .st-key-sb_brand {
    padding: 0.88rem 0.95rem 0.8rem;
    border: 1px solid rgba(126, 181, 238, 0.28);
    border-radius: 18px;
    background: linear-gradient(145deg, rgba(255,255,255,0.94), rgba(239,248,255,0.82));
    box-shadow: 0 12px 28px rgba(75, 117, 170, 0.09);
}

[data-testid="stSidebar"] .st-key-sb_brand h3 {
    margin: 0;
    color: var(--bml-side-ink);
    font-size: 1.42rem;
    font-weight: 800;
    letter-spacing: -0.045em;
}

[data-testid="stSidebar"] .st-key-sb_brand [data-testid="stCaptionContainer"] {
    color: var(--bml-side-muted);
    font-size: 0.76rem;
    font-weight: 650;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}

/* Current-project context is information, not another navigation button. */
[data-testid="stSidebar"] .st-key-sb_project {
    padding: 0.8rem 0.95rem;
    border: 1px solid var(--bml-side-line);
    border-radius: 16px;
    background: rgba(255,255,255,0.68);
    box-shadow: 0 8px 22px rgba(75, 117, 170, 0.06);
}

[data-testid="stSidebar"] .st-key-sb_project [data-testid="stMarkdownContainer"] strong {
    color: var(--bml-side-blue-deep);
    font-size: 0.71rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
}

[data-testid="stSidebar"] .st-key-sb_project [data-testid="stCaptionContainer"] {
    color: var(--bml-side-ink);
    font-size: 0.9rem;
}

/* The two navigation groups use the same one-piece, raised controls as the
   home choice cards.  Text is pinned on every state so the theme cannot turn
   a light-blue hover or keyboard focus into low-contrast grey copy. */
[data-testid="stSidebar"] .st-key-sb_entry [data-testid="stButton"] > button,
[data-testid="stSidebar"] .st-key-sb_protocol [data-testid="stButton"] > button {
    min-height: 2.72rem;
    padding: 0.58rem 0.8rem;
    border: 1px solid var(--bml-side-line);
    border-radius: 14px;
    background: rgba(255,255,255,0.78);
    color: var(--bml-side-ink) !important;
    box-shadow: 0 7px 18px rgba(75, 117, 170, 0.06);
    font-weight: 680;
    text-align: left;
    transition: transform 160ms ease, border-color 160ms ease,
        background 160ms ease, box-shadow 160ms ease;
}

[data-testid="stSidebar"] .st-key-sb_entry [data-testid="stButton"] > button *,
[data-testid="stSidebar"] .st-key-sb_protocol [data-testid="stButton"] > button * {
    color: var(--bml-side-ink) !important;
    -webkit-text-fill-color: var(--bml-side-ink);
}

[data-testid="stSidebar"] .st-key-sb_entry [data-testid="stButton"] > button:hover,
[data-testid="stSidebar"] .st-key-sb_protocol [data-testid="stButton"] > button:hover {
    border-color: rgba(40,123,245,0.46);
    background: linear-gradient(145deg, rgba(239,247,255,0.98), rgba(255,255,255,0.94));
    box-shadow: 0 12px 24px rgba(40,123,245,0.13);
    transform: translateX(2px);
}

[data-testid="stSidebar"] .st-key-sb_entry [data-testid="stButton"] > button:focus-visible,
[data-testid="stSidebar"] .st-key-sb_protocol [data-testid="stButton"] > button:focus-visible {
    outline: 2px solid var(--bml-side-blue-deep);
    outline-offset: 2px;
}

/* ``_nav_button`` deliberately keeps the accessible bold-markdown path for
   the current item.  A blue accent and tinted surface make that state clear;
   the Protocol heading itself remains an eyebrow-sized label. */
[data-testid="stSidebar"] .st-key-sb_entry [data-testid="stMarkdownContainer"] strong {
    display: block;
    padding: 0.62rem 0.8rem;
    border: 1px solid rgba(40,123,245,0.42);
    border-left: 3px solid var(--bml-side-blue);
    border-radius: 14px;
    background: linear-gradient(145deg, #eaf4ff, #ffffff);
    color: var(--bml-side-ink) !important;
    box-shadow: 0 11px 24px rgba(40,123,245,0.12);
}

[data-testid="stSidebar"] .st-key-sb_protocol [data-testid="stMarkdownContainer"] strong {
    color: var(--bml-side-blue-deep);
}

/* Protocol Workflow is a section label; its first strong line should stay a
   label instead of becoming a fourth button. */
[data-testid="stSidebar"] .st-key-sb_protocol [data-testid="stMarkdownContainer"] p > strong {
    letter-spacing: 0.04em;
}

[data-testid="stSidebar"] .st-key-sb_protocol [data-testid="stMarkdownContainer"]:has(+ *) strong {
    color: var(--bml-side-ink);
}

/* Settings is a disclosure, so it shares the cards' border and radius while
   retaining Streamlit's native expand/collapse affordance. */
[data-testid="stSidebar"] .st-key-sb_foot [data-testid="stExpander"] {
    border: 1px solid var(--bml-side-line);
    border-radius: 14px;
    background: rgba(255,255,255,0.72);
    box-shadow: 0 8px 20px rgba(75,117,170,0.05);
}

[data-testid="stSidebar"] .st-key-sb_foot [data-testid="stExpander"] summary,
[data-testid="stSidebar"] .st-key-sb_foot [data-testid="stExpander"] summary * {
    color: var(--bml-side-ink) !important;
    font-weight: 700;
}

/* Prev/Next are secondary controls: quieter than the route entries, but still
   unmistakably clickable and legible when enabled. */
[data-testid="stSidebar"] .st-key-sb_pager [data-testid="stButton"] > button {
    min-height: 2.58rem;
    border: 1px solid rgba(40,123,245,0.22);
    border-radius: 12px;
    background: rgba(255,255,255,0.7);
    color: var(--bml-side-blue-deep) !important;
    font-weight: 700;
}

/* Prev / Next are two buttons, not rows of a list: their labels sit in the
   middle. surfaces.py left-aligns every sidebar navigation button so the
   page list reads like headings; this more specific rule exempts the pager. */
[data-testid="stSidebar"] .st-key-sb_pager [data-testid="stButton"] > button,
[data-testid="stSidebar"] .st-key-sb_pager [data-testid="stButton"] > button > div {
    justify-content: center;
    text-align: center;
}

[data-testid="stSidebar"] .st-key-sb_pager [data-testid="stButton"] > button:hover {
    border-color: rgba(40,123,245,0.44);
    background: rgba(234,244,255,0.96);
}

[data-testid="stSidebar"] .st-key-sb_pager [data-testid="stButton"] > button:disabled {
    border-color: rgba(53,83,135,0.1);
    background: rgba(255,255,255,0.42);
    color: #91a0b5 !important;
    -webkit-text-fill-color: #91a0b5;
    box-shadow: none;
}

[data-testid="stSidebar"] .st-key-sb_foot [data-testid="stMarkdownContainer"] div[style*="LLM"] {
    padding: 0.18rem 0.2rem;
    color: var(--bml-side-muted) !important;
    font-size: 0.78rem !important;
    letter-spacing: 0.02em;
}

@media (max-width: 900px) {
    [data-testid="stSidebar"] .st-key-sb_brand,
    [data-testid="stSidebar"] .st-key-sb_project {
        padding-left: 0.78rem;
        padding-right: 0.78rem;
    }

    [data-testid="stSidebar"] .st-key-sb_entry [data-testid="stButton"] > button,
    [data-testid="stSidebar"] .st-key-sb_protocol [data-testid="stButton"] > button {
        min-height: 2.6rem;
        padding-left: 0.72rem;
        padding-right: 0.72rem;
    }
}

@media (prefers-reduced-motion: reduce) {
    [data-testid="stSidebar"] * {
        transition-duration: 0.01ms !important;
    }
}
</style>
"""


__all__ = ["SIDEBAR_THEME_CSS"]
