"""UI-01 — navigation and layout: page-top reset, one Prepare Data flow, compact sidebar.

Establishes `UX-08`, `UX-09` and `UX-10` without a browser and without
Streamlit, so the same three claims can be re-checked on the server in a
second:

* **`UX-08`** — the page-top reset exists as a pure decision plus a thin
  injection, is imported and called by ``app.py``, and is called *before*
  every page body so a page that raises part-way through cannot skip it.
  The decision itself is executed here against a plain dict: a first render
  does not scroll, a page change does, and a rerun within a page does not.
* **`UX-09`** — Prepare Data builds three tabs with no "Preprocessing" /
  "Feature Inspection" split, uses exactly one dataset selector, and still
  contains every capability the two tabs had.
* **`UX-10`** — the sidebar stylesheet adjusts spacing only, is scoped to
  the sidebar, hides nothing, relies on no generated class name, and keeps
  the values a browser run pinned down: the block gap at or above
  `0.25rem`, a non-zero heading padding, and a brand margin that cancels
  exactly one block gap. The sidebar heading overflows its own container by
  ~16px; that last pairing is what stops it landing on its caption, and it
  holds at any gap.

This check renders nothing, starts no server and writes nothing outside its
own report.
"""

from __future__ import annotations

import argparse
import ast
import re
from pathlib import Path

from .common import FAIL, PASS, CheckResult, repo_root, run_check, write_json

CHECK_ID = "UI-01"
CHECK_NAME = "navigation_layout"

APP = Path("batteryml_web") / "app.py"
SCROLL = Path("batteryml_web") / "ui" / "scroll.py"
STYLES = Path("batteryml_web") / "ui" / "styles.py"
SIDEBAR = Path("batteryml_web") / "ui" / "workflow_sidebar.py"

#: Every page body the reset must precede.
PAGE_CALLS = (
    "render_home_page()",
    "render_prepare_data_page()",
    "render_run_model_page()",
    "render_agent_workflow()",
    "render_review_results_page()",
)

#: Capabilities the merged Prepare Data flow must still offer (UX-09).
PREPARE_DATA_CAPABILITIES = (
    "Dataset readiness",
    "Raw path",
    "Processed path",
    "Processed file count",
    "'prep_run'",
    "Latest preprocessing log",
    "Feature / cache / artifact files",
    "'prep_go_run_model'",
)

#: Spacing only. Anything outside this set changes appearance, not layout.
ALLOWED_CSS_PROPERTIES = {
    "padding", "padding-top", "padding-bottom", "padding-left", "padding-right",
    "margin", "margin-top", "margin-bottom", "margin-left", "margin-right",
    "gap", "line-height", "height", "min-height",
    # Type size, for the sidebar's grey secondary lines (UX-11). A size is
    # not a disappearance: the text still renders, still wraps and is still
    # selectable. Colour, display, visibility, content and overflow stay out
    # — those are the properties that can remove a control.
    "font-size",
}

#: The band UX-11 fixes the sidebar's secondary text to. A floor as well as a
#: ceiling: below this it stops being readable, above it nothing was gained.
SIDEBAR_CAPTION_REM = (0.80, 0.85)

#: Rules the sidebar caption selector may never carry — the properties that
#: clipped the second line of a wrapped caption behind the button beneath it.
CAPTION_FORBIDDEN_ROOTS = {"margin", "padding", "height", "overflow"}

#: The sidebar's whole vertical rhythm: measured in a browser, every
#: top-level block is separated by this and nothing else. The floor is about
#: legibility, not collision — below it the controls stop reading as separate
#: rows. It used to be 0.4rem for a different reason: the heading's ~16px
#: overflow had nothing else holding it back, so a smaller gap put the title
#: on its own caption. The brand rule now cancels exactly one block gap, which
#: holds the header steady at any value, and the pairing is checked below.
MINIMUM_BLOCK_GAP_REM = 0.25

_GENERATED_CLASS = re.compile(r"\.(css|st-emotion)-[0-9a-z]{4,}")


def _function(tree: ast.Module, name: str) -> ast.FunctionDef | None:
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    return None


def _strip_docstrings(node: ast.AST) -> str:
    """The node's executable source. Prose is documentation, not behaviour."""
    clone = ast.parse(ast.unparse(node))
    for inner in ast.walk(clone):
        if not isinstance(
            inner, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
        ):
            continue
        body = inner.body
        if (
            body
            and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)
        ):
            inner.body = body[1:] or [ast.Pass()]
    return ast.unparse(clone)


def _tab_labels(page: ast.FunctionDef) -> list[str] | None:
    for node in ast.walk(page):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "tabs"
            and node.args
            and isinstance(node.args[0], ast.List)
        ):
            return [
                element.value
                for element in node.args[0].elts
                if isinstance(element, ast.Constant)
            ]
    return None


def _css_rules(stylesheet: str) -> list[tuple[str, str]]:
    """(selector, declarations) pairs, with comments removed first."""
    body = stylesheet.split("<style>", 1)[-1].split("</style>", 1)[0]
    body = re.sub(r"/\*.*?\*/", "", body, flags=re.DOTALL)
    rules: list[tuple[str, str]] = []
    for block in body.split("}"):
        if "{" not in block:
            continue
        selector, declarations = block.split("{", 1)
        rules.append((selector.strip(), declarations.strip()))
    return rules


def _check_page_top_reset(root: Path, errors: list[str], observed: dict) -> None:
    app_source = (root / APP).read_text(encoding="utf-8")
    lines = app_source.splitlines()

    if "from batteryml_web.ui.scroll import render_scroll_to_top" not in app_source:
        errors.append(
            f"{APP.as_posix()} does not import render_scroll_to_top; navigation "
            "would keep the previous page's scroll offset (UX-08)."
        )
        return

    reset_lines = [i for i, line in enumerate(lines) if "render_scroll_to_top(current_step)" in line]
    if not reset_lines:
        errors.append(
            f"{APP.as_posix()} never calls render_scroll_to_top(current_step); "
            "the page-top reset is imported but not wired (UX-08)."
        )
        return
    reset = min(reset_lines)
    observed["reset_line"] = reset + 1

    for page_call in PAGE_CALLS:
        drawn = [i for i, line in enumerate(lines) if page_call in line]
        if not drawn:
            errors.append(
                f"{APP.as_posix()} no longer dispatches {page_call}; the UI-01 "
                "scan is out of date with the app's page list."
            )
            continue
        if reset > max(drawn):
            errors.append(
                f"render_scroll_to_top runs after {page_call} (line "
                f"{reset + 1} > {max(drawn) + 1}). It must precede every page "
                "body, or a page that raises part-way through skips the reset "
                "and the app keeps the previous page's scroll offset (UX-08)."
            )

    from batteryml_web.ui.scroll import LAST_RENDERED_STEP_KEY, page_changed

    session: dict = {}
    first = page_changed(session, "home")
    changed = page_changed(session, "run_model")
    again = page_changed(session, "run_model")
    observed["page_changed"] = {"first_render": first, "page_change": changed, "same_page": again}
    if first:
        errors.append("page_changed scrolls on the first render, which can only flicker (UX-08).")
    if not changed:
        errors.append("page_changed does not report a page change; navigation would not reset (UX-08).")
    if again:
        errors.append(
            "page_changed reports a change for a rerun within the same page; a "
            "button press would yank the reader to the top (UX-08)."
        )
    if session.get(LAST_RENDERED_STEP_KEY) != "run_model":
        errors.append("page_changed does not record the step it rendered (UX-08).")


def _check_prepare_data(root: Path, errors: list[str], observed: dict) -> None:
    app_source = (root / APP).read_text(encoding="utf-8")
    tree = ast.parse(app_source)
    page = _function(tree, "render_prepare_data_page")
    if page is None:
        errors.append(f"{APP.as_posix()} no longer defines render_prepare_data_page (UX-09).")
        return
    code = _strip_docstrings(page)

    labels = _tab_labels(page)
    if labels is None:
        errors.append(f"render_prepare_data_page no longer builds a tab list (UX-09).")
        return
    observed["prepare_data_tabs"] = labels
    for split in ("Preprocessing", "Feature Inspection"):
        if split in labels:
            errors.append(
                f"Prepare Data still shows a separate {split!r} tab. The flow is "
                "linear once a dataset is chosen, so the two are one tab (UX-09)."
            )

    selectors = code.count("'prep_ds_select'")
    observed["dataset_selectors"] = selectors
    if selectors != 1:
        errors.append(
            f"Prepare Data has {selectors} 'prep_ds_select' selector(s), expected 1. "
            "Two selectors can disagree about which dataset is being prepared (UX-09)."
        )
    if "feature_ds_select" in app_source:
        errors.append(
            "feature_ds_select still exists: the merged flow must have one "
            "dataset selector, not a second one for inspection (UX-09)."
        )

    missing = [capability for capability in PREPARE_DATA_CAPABILITIES if capability not in code]
    if missing:
        errors.append(
            "the merged Prepare Data flow lost " + ", ".join(missing) + ". "
            "Simplifying the structure may not drop functionality (UX-09)."
        )


def _check_sidebar_stylesheet(root: Path, errors: list[str], observed: dict) -> None:
    from batteryml_web.ui.styles import SIDEBAR_COMPACT_CSS

    sidebar_source = (root / SIDEBAR).read_text(encoding="utf-8")
    sidebar_tree = ast.parse(sidebar_source)
    render = _function(sidebar_tree, "render_sidebar_workflow")
    if render is None or "render_layout_css()" not in _strip_docstrings(render):
        errors.append(
            f"{SIDEBAR.as_posix()} does not call render_layout_css(); the sidebar "
            "would keep Streamlit's default spacing (UX-10)."
        )

    for forbidden in ("display: none", "display:none", "visibility: hidden", "visibility:hidden"):
        if forbidden in SIDEBAR_COMPACT_CSS:
            errors.append(
                f"the sidebar stylesheet uses {forbidden!r}. A control the user "
                "cannot reach is not 'compact' — spacing only (UX-10)."
            )

    if _GENERATED_CLASS.search(SIDEBAR_COMPACT_CSS):
        errors.append(
            "the sidebar stylesheet targets a generated Streamlit class name. "
            "Those change between releases; use data-testid selectors (UX-10)."
        )

    rules = _css_rules(SIDEBAR_COMPACT_CSS)
    observed["sidebar_rules"] = len(rules)
    if len(rules) < 5:
        errors.append(
            f"the sidebar stylesheet declares only {len(rules)} rule(s); the "
            "UI-01 scan found almost nothing to check (UX-10)."
        )

    for selector, declarations in rules:
        for part in selector.split(","):
            if part.strip() and "stSidebar" not in part:
                errors.append(
                    f"sidebar rule {part.strip()!r} is not scoped to the sidebar; "
                    "it would re-space the whole app (UX-10)."
                )
        for declaration in declarations.split(";"):
            if ":" not in declaration:
                continue
            prop = declaration.split(":", 1)[0].strip()
            if prop not in ALLOWED_CSS_PROPERTIES:
                errors.append(
                    f"sidebar rule {selector!r} sets {prop!r}, which is not a "
                    "spacing or type-size property. This stylesheet adjusts "
                    "layout only (UX-10, UX-11)."
                )

    caption_rules = [
        (selector, declarations)
        for selector, declarations in rules
        if "stCaptionContainer" in selector
    ]
    if not caption_rules:
        errors.append(
            "the sidebar stylesheet no longer sets the secondary text size "
            "(UX-11)."
        )
    for selector, declarations in caption_rules:
        for declaration in declarations.split(";"):
            if ":" not in declaration:
                continue
            prop = declaration.split(":", 1)[0].strip()
            # Matched on the property root, so "line-height" is read as type
            # rather than as a "height".
            if prop.split("-")[0] in CAPTION_FORBIDDEN_ROOTS:
                errors.append(
                    f"sidebar caption rule sets {prop!r}; a caption spacing "
                    "rule clipped wrapped captions behind the next button "
                    "(UX-11)."
                )
        sizes = re.findall(r"font-size:\s*([0-9.]+)rem", declarations)
        observed["sidebar_caption_rem"] = sizes
        low, high = SIDEBAR_CAPTION_REM
        for size in sizes:
            if not (low <= float(size) <= high):
                errors.append(
                    f"the sidebar secondary text is {size}rem, outside the "
                    f"{low}-{high}rem band UX-11 fixes."
                )

    gap = re.search(r"stVerticalBlock\"\]\s*\{[^}]*gap:\s*([0-9.]+)rem", SIDEBAR_COMPACT_CSS)
    if gap is None:
        errors.append("the sidebar stylesheet no longer sets the block gap (UX-10).")
    else:
        observed["block_gap_rem"] = float(gap.group(1))
        if float(gap.group(1)) < MINIMUM_BLOCK_GAP_REM:
            errors.append(
                f"the sidebar block gap is {gap.group(1)}rem, below "
                f"{MINIMUM_BLOCK_GAP_REM}rem. This one value sets the spacing "
                "between every block in the sidebar; below the floor the "
                "controls stop reading as separate rows (UX-10)."
            )
        brand = re.search(
            r":has\(> h3\)\s*\{[^}]*margin-bottom:\s*(-?[0-9.]+)rem",
            SIDEBAR_COMPACT_CSS,
        )
        if brand is None:
            errors.append(
                "the sidebar brand heading no longer pins its bottom margin. "
                "Streamlit's heading overflows its container by ~16px, and "
                "that rule is what stops the title landing on its own caption "
                "(UX-10, UX-11)."
            )
        else:
            observed["brand_margin_rem"] = float(brand.group(1))
            if abs(float(brand.group(1)) + float(gap.group(1))) > 1e-4:
                errors.append(
                    f"the sidebar brand margin is {brand.group(1)}rem but the "
                    f"block gap is {gap.group(1)}rem. The margin must cancel "
                    "exactly one gap, or 'Protocol Navigator' drifts into or "
                    "away from 'BatteryML' whenever the rhythm is retuned "
                    "(UX-10, UX-11)."
                )

    headings = re.search(r"stSidebar\"\]\s*h4\s*\{([^}]*)\}", SIDEBAR_COMPACT_CSS)
    if headings is None:
        errors.append("the sidebar stylesheet no longer tightens the headings (UX-10).")
    else:
        paddings = re.findall(r"padding-\w+:\s*([0-9.]+)\s*(?:rem|em|px)?", headings.group(1))
        observed["heading_paddings"] = paddings
        if not paddings:
            errors.append("the sidebar heading rule sets no padding (UX-10).")
        for value in paddings:
            if float(value) <= 0.0:
                errors.append(
                    "the sidebar heading padding is zero, which collapses its box "
                    "onto the caption beneath it (UX-10)."
                )


def check(output_dir: str | Path | None = None) -> CheckResult:
    """Run UI-01.

    Args:
        output_dir: where ``navigation_layout.json`` is written. Defaults to
            ``runs/verify/``.

    Returns:
        The check result.
    """
    root = repo_root()
    out_dir = Path(output_dir) if output_dir else root / "runs" / "verify"

    errors: list[str] = []
    observed: dict = {}

    _check_page_top_reset(root, errors, observed)
    _check_prepare_data(root, errors, observed)
    _check_sidebar_stylesheet(root, errors, observed)

    evidence = {"observed": observed, "errors": errors}
    artifact = write_json(out_dir / "navigation_layout.json", evidence)

    if errors:
        return CheckResult(
            CHECK_ID, CHECK_NAME, FAIL,
            f"{len(errors)} navigation/layout contract failure(s). First: {errors[0]} "
            "See evidence.errors for the full list.",
            evidence, [artifact],
        )

    return CheckResult(
        CHECK_ID, CHECK_NAME, PASS,
        "Navigation resets the page to the top before any page body is drawn and "
        "not on a rerun within a page; Prepare Data is one flow with one dataset "
        f"selector and {len(PREPARE_DATA_CAPABILITIES)} retained capabilities "
        f"across tabs {observed.get('prepare_data_tabs')}; the sidebar stylesheet "
        f"adjusts spacing only, is scoped to the sidebar, hides nothing, and keeps "
        f"the measured block gap at {observed.get('block_gap_rem')}rem.",
        evidence, [artifact],
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args(argv)
    output_dir = Path(args.output_dir) if args.output_dir else None
    result = run_check(CHECK_ID, CHECK_NAME, check, output_dir=output_dir)
    print(f"[{result.status}] {result.check_id} {result.name}: {result.message}")
    return 1 if result.status == FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
