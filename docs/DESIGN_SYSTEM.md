# Design system

One visual language for every page of the web app: the mentor's Home
(`codex/ui-optimization`). Spec IDs `UI-10`, `UI-11`, `UI-12`, `UI-13`.

Where it lives:

| What | File |
| --- | --- |
| Tokens and site-wide rules (every page) | `batteryml_web/ui/home_style.py` → `SITE_STYLE_CSS` |
| Home's own components (hero, start paths, step overview) | `batteryml_web/ui/home_style.py` → `HOME_STYLE_CSS` |
| Action roles (`PRIMARY_ROLE`, `SECONDARY_ROLE`, `NAV_ROLE`), info cards | `batteryml_web/ui/surfaces.py` |
| Sidebar layout and navigation | `batteryml_web/ui/sidebar_style.py`, `batteryml_web/ui/styles.py` |
| Streamlit widget accent (tabs, radios, focus) | `.streamlit/config.toml` → `primaryColor` |

## Tokens

All are CSS custom properties in `:root`, derived from the mentor's palette
(`--bml-ink #14224a`, `--bml-muted #66728b`, `--bml-blue #287bf5`,
`--bml-blue-deep #1557d6`, `--bml-mint #31c79b`, `--bml-line`, `--bml-soft`).

| Group | Tokens |
| --- | --- |
| Colour | `--bml-primary` (#287bf5), `--bml-primary-hover` (#1557d6), `--bml-primary-muted`, `--bml-background`, `--bml-surface`, `--bml-surface-secondary`, `--bml-surface-selected`, `--bml-border`, `--bml-border-subtle`, `--bml-text-primary` (#14224a), `--bml-text-secondary` (#66728b), `--bml-text-muted`, `--bml-focus-ring`, `--bml-success`, `--bml-warning`, `--bml-error` |
| Type | `--bml-fs-page-title` 2rem / 700, `--bml-fs-section-title` 1.5rem / 650, `--bml-fs-subsection-title` 1.25rem / 600, `--bml-fs-body` 1rem, `--bml-fs-label` and `--bml-fs-caption` 0.875rem, `--bml-fs-metric` 1.75rem, `--bml-font-mono` |
| Geometry | `--bml-page-max-width` 1480px, `--bml-card-radius` 16px, `--bml-button-radius` and `--bml-input-radius` 12px, `--bml-disclosure-radius` 10px, `--bml-control-height` 2.75rem, `--bml-section-gap`, `--bml-component-gap`, `--bml-card-padding` |
| States | `--bml-state-hover`, `--bml-state-selected`, `--bml-state-disabled-bg`, `--bml-state-disabled-border`, `--bml-state-loading`, `--bml-state-completed`, `--bml-state-failed` |

The type family is the theme's (Streamlit's Source Sans) everywhere; no
sheet overrides it. Code and run ids use the theme's monospace.

## Headings

| Level | Call | Size / weight |
| --- | --- | --- |
| Page title | `st.header("…", anchor=False)` (h2) | 2rem / 700 |
| Section | `st.subheader("…", anchor=False)` (h3) | 1.5rem / 650 |
| Subsection | `st.markdown("#### …")` (h4) | 1.25rem / 600 |

Home's hero title and its component headings keep the mentor's sizes.

## Roles

| Role | How to ask for it | Looks like |
| --- | --- | --- |
| Primary (Run, Preprocess, Analyze) | `type=PRIMARY_ROLE` | filled cobalt gradient |
| Secondary (Download, Refresh, Summarize, Explain) | default / `type=SECONDARY_ROLE` | white, cobalt hairline |
| Navigation (pages, Prev/Next) | `type=NAV_ROLE` | cobalt wash, no outline |
| Interrupting (Stop run) | key `agent_cancel_<run_id>` | red hairline on white |
| Disabled (any) | `disabled=True`, reason beneath | dashed, neutral, `not-allowed` |
| Information | `surfaces.render_info_card` | flat 4px panel, no hover |
| Status | `<span class="bml-status bml-status--completed">` | pill with a dot and the word |
| Disclosure | `st.expander` | 10px hairline panel |

## Review Results and Run History

Page title, then three sections: **Run history**, **Selected result** (the
result tabs, then the *Benchmark reference* subsection) and **Optional LLM
interpretation**. The history table is compact — short run id, start time
with its recorded zone (`2026-10-12 17:29 UTC`), `model · dataset · task`,
seed, device, status, duration, metrics — and only the table and the
selector's labels are shortened: the selector's value, the session state and
every section use the full run id, which the run's details show in a
copyable code block next to the start time as recorded
(`batteryml_web/ui/run_history.py`).

## Checking it in a browser

`scripts/ui_audit/viewport_audit.py` opens a running app at 1440, 1024 and
760px, visits every page and the sidebar Settings, and reports horizontal
page overflow, exceptions and the computed heading sizes. It needs
Playwright, which is optional tooling and not a platform dependency:

```bash
pip install playwright==1.63.0
python -m playwright install chromium
streamlit run batteryml_web/app.py --server.port 8501 --server.headless true &
python scripts/ui_audit/viewport_audit.py --url http://localhost:8501
```

Without Playwright, the same overflow check is one line in the browser's
developer console at each window width:

```js
document.documentElement.scrollWidth - document.documentElement.clientWidth   // expect 0
```
