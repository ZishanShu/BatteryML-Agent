"""Navigation opens at the top; Prepare Data is one flow; the sidebar is tight.

Three layout claims, each checked where it is actually decided rather than
where it is visible:

* **page-top reset** — the decision is a pure function of session state
  (:func:`batteryml_web.ui.scroll.page_changed`), so it is tested directly;
  the injection is tested by rendering the real
  :func:`~batteryml_web.ui.scroll.render_scroll_to_top` against a stand-in
  Streamlit and asserting a component was emitted on a page change and not
  otherwise. The wiring into ``app.py`` is checked by AST, because a helper
  nothing calls resets nothing.
* **one Prepare Data flow** — read from ``app.py``'s AST: the tab list no
  longer splits Preprocessing from Feature Inspection, there is one dataset
  selector, and every piece of the old two-tab functionality is still in the
  merged block. Simplification is only a win if nothing was quietly lost.
* **compact sidebar** — the stylesheet is checked for what it may *not* do
  (hide anything, depend on generated class names) as much as for what it
  does, and the sidebar is checked to actually inject it.

Run from the repository root:

    conda run -n batteryml python -m unittest tests.test_ui_layout -v
"""

from __future__ import annotations

import ast
import re
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

APP_PATH = REPO_ROOT / "batteryml_web" / "app.py"
SIDEBAR_PATH = REPO_ROOT / "batteryml_web" / "ui" / "workflow_sidebar.py"
STYLES_PATH = REPO_ROOT / "batteryml_web" / "ui" / "styles.py"
SCROLL_PATH = REPO_ROOT / "batteryml_web" / "ui" / "scroll.py"


def _streamlit_available() -> bool:
    try:
        import streamlit  # noqa: F401
    except Exception:
        return False
    return True


STREAMLIT_AVAILABLE = _streamlit_available()


def _function(tree: ast.Module, name: str) -> ast.FunctionDef | None:
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    return None


def _strip_docstrings(node: ast.AST) -> ast.AST:
    """A copy of ``node`` with every docstring removed.

    Prose is documentation, not behaviour: a docstring that explains what a
    tab used to be called must not make a "the old tab is gone" test fail.
    """
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
    return clone


def _code_of(node: ast.AST) -> str:
    """The node's executable source: docstrings and comments both gone."""
    return ast.unparse(_strip_docstrings(node))


class _Component:
    """Records what :func:`render_scroll_to_top` injects."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def html(self, html: str, **kwargs) -> None:
        self.calls.append({"html": str(html), "height": kwargs.get("height")})


class _St:
    """The two attributes ``ui/scroll.py`` touches."""

    def __init__(self, session: dict) -> None:
        self.session_state = session


class TestPageTopResetDecision(unittest.TestCase):
    """The pure half: when is a page 'newly opened'?"""

    def setUp(self) -> None:
        from batteryml_web.ui import scroll

        self.scroll = scroll

    def test_first_render_does_not_scroll(self) -> None:
        # A freshly loaded app is already at the top; scrolling it would be
        # a no-op visible only as a flicker.
        session: dict = {}
        self.assertFalse(self.scroll.page_changed(session, "home"))

    def test_changing_page_scrolls(self) -> None:
        session: dict = {}
        self.scroll.page_changed(session, "home")
        self.assertTrue(self.scroll.page_changed(session, "run_model"))

    def test_rerun_within_a_page_does_not_scroll(self) -> None:
        """A button press must not yank the reader back to the top."""
        session: dict = {}
        self.scroll.page_changed(session, "run_model")
        for _ in range(3):
            self.assertFalse(self.scroll.page_changed(session, "run_model"))

    def test_every_page_pair_is_treated_the_same(self) -> None:
        pages = ["home", "agent_workflow", "prepare_data", "run_model", "review_results"]
        for origin in pages:
            for destination in pages:
                if origin == destination:
                    continue
                with self.subTest(origin=origin, destination=destination):
                    session: dict = {}
                    self.scroll.page_changed(session, origin)
                    self.assertTrue(self.scroll.page_changed(session, destination))

    def test_the_rendered_step_is_recorded(self) -> None:
        session: dict = {}
        self.scroll.page_changed(session, "prepare_data")
        self.assertEqual(
            session[self.scroll.LAST_RENDERED_STEP_KEY], "prepare_data"
        )


class TestPageTopResetInjection(unittest.TestCase):
    """The browser half: what actually reaches the page."""

    def setUp(self) -> None:
        from batteryml_web.ui import scroll

        self.scroll = scroll
        self.session: dict = {}
        self.components = _Component()
        self._saved = (scroll.st, scroll.components)
        scroll.st = _St(self.session)
        scroll.components = self.components

    def tearDown(self) -> None:
        self.scroll.st, self.scroll.components = self._saved

    def test_a_component_is_injected_only_on_a_page_change(self) -> None:
        self.assertFalse(self.scroll.render_scroll_to_top("home"))
        self.assertEqual(self.components.calls, [])

        self.assertTrue(self.scroll.render_scroll_to_top("run_model"))
        self.assertEqual(len(self.components.calls), 1)

        self.assertFalse(self.scroll.render_scroll_to_top("run_model"))
        self.assertEqual(len(self.components.calls), 1)

    def test_the_injection_takes_no_vertical_space(self) -> None:
        self.scroll.render_scroll_to_top("home")
        self.scroll.render_scroll_to_top("review_results")
        self.assertEqual(self.components.calls[0]["height"], 0)

    def test_each_injection_is_distinct_so_it_re_executes(self) -> None:
        """Streamlit re-uses an unchanged iframe, and a re-used one is inert."""
        self.scroll.render_scroll_to_top("home")
        self.scroll.render_scroll_to_top("run_model")
        self.scroll.render_scroll_to_top("prepare_data")
        bodies = [call["html"] for call in self.components.calls]
        self.assertEqual(len(bodies), 2)
        self.assertNotEqual(bodies[0], bodies[1])

    def test_the_script_scrolls_the_host_page_to_the_top(self) -> None:
        self.scroll.render_scroll_to_top("home")
        self.scroll.render_scroll_to_top("run_model")
        html = self.components.calls[0]["html"]
        self.assertIn("<script>", html)
        self.assertIn("window.parent", html)
        self.assertIn("scrollTo(0, 0)", html)
        # Retried, because the new page is still laying out when this loads.
        self.assertIn("setTimeout", html)
        # And it must never scroll anywhere but the top.
        offsets = re.findall(r"scrollTo\(([^)]*)\)", html)
        for offset in offsets:
            self.assertEqual(offset.replace(" ", ""), "0,0")


class TestPageTopResetIsWiredIn(unittest.TestCase):
    """Wired into the real page, not only into a test."""

    def setUp(self) -> None:
        self.source = APP_PATH.read_text(encoding="utf-8")
        self.tree = ast.parse(self.source)

    def test_the_app_imports_and_calls_the_reset(self) -> None:
        self.assertIn(
            "from batteryml_web.ui.scroll import render_scroll_to_top", self.source
        )
        module_level = ast.unparse(
            ast.Module(
                body=[n for n in self.tree.body if not isinstance(n, (ast.FunctionDef, ast.ClassDef))],
                type_ignores=[],
            )
        )
        self.assertIn("render_scroll_to_top(current_step)", module_level)

    def test_the_reset_cannot_be_skipped_by_a_failing_page(self) -> None:
        """Placed before the dispatch, verified in a browser.

        With the reset after the page body, a page that raised part-way
        through — which a real one does, on a missing dataset — never
        reached it, and the app kept the previous page's scroll offset with
        nothing to say so. Streamlit streams elements as the script produces
        them, so injecting first still works and cannot be skipped.
        """
        lines = self.source.splitlines()
        reset = min(
            i for i, line in enumerate(lines) if "render_scroll_to_top(current_step)" in line
        )
        # The dispatch now sits inside the error boundary, so it is indented.
        dispatch = min(
            i for i, line in enumerate(lines)
            if line.strip().startswith("if current_step ==")
        )
        self.assertLess(reset, dispatch)
        # ...and the boundary itself starts after the reset, so a page that
        # raises is caught having already been scrolled to the top.
        boundary = min(
            i for i, line in enumerate(lines)
            if line.strip().startswith("with error_boundary(")
        )
        self.assertLess(reset, boundary)
        for page_call in (
            "render_home_page()",
            "render_prepare_data_page()",
            "render_run_model_page()",
            "render_agent_workflow()",
            "render_review_results_page()",
        ):
            with self.subTest(page=page_call):
                drawn = max(i for i, line in enumerate(lines) if page_call in line)
                self.assertLess(reset, drawn)

    def test_the_reset_covers_navigation_from_the_sidebar(self) -> None:
        """Every sidebar entry routes through the step key the reset watches."""
        sidebar = SIDEBAR_PATH.read_text(encoding="utf-8")
        tree = ast.parse(sidebar)
        setter = _function(tree, "set_current_step")
        self.assertIsNotNone(setter)
        body = _code_of(setter)
        self.assertIn('st.session_state[\'current_step\'] = step_key', body)
        # The sidebar buttons and Prev/Next all go through that one setter,
        # so change-detection on current_step sees every one of them.
        nav = _function(tree, "_nav_button")
        self.assertIn("set_current_step(step['key'])", _code_of(nav))


class TestPrepareDataIsOneFlow(unittest.TestCase):
    """2: the redundant Preprocessing / Feature Inspection split is gone."""

    def setUp(self) -> None:
        self.source = APP_PATH.read_text(encoding="utf-8")
        self.tree = ast.parse(self.source)
        page = _function(self.tree, "render_prepare_data_page")
        self.assertIsNotNone(page, "the Prepare Data page must still exist")
        self.page = page
        self.code = _code_of(page)

    def _tab_labels(self) -> list[str]:
        for node in ast.walk(self.page):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "tabs"
            ):
                return [
                    element.value
                    for element in node.args[0].elts
                    if isinstance(element, ast.Constant)
                ]
        self.fail("Prepare Data no longer builds a tab list")

    def test_the_two_tabs_are_merged_into_one(self) -> None:
        labels = self._tab_labels()
        self.assertNotIn("Preprocessing", labels)
        self.assertNotIn("Feature Inspection", labels)
        self.assertEqual(len(labels), 3, f"expected three tabs, got {labels}")
        self.assertIn("Dataset Overview", labels)
        self.assertIn("Custom Dataset / Schema Mapping", labels)

    def test_one_dataset_selector_drives_the_whole_flow(self) -> None:
        """Two selectors could disagree about which dataset was being prepared."""
        self.assertNotIn("feature_ds_select", self.source)
        self.assertEqual(self.code.count("'prep_ds_select'"), 1)

    def test_the_selector_honours_the_session_dataset(self) -> None:
        self.assertIn("_selected_dataset_index(datasets)", self.code)
        helper = _function(self.tree, "_selected_dataset_index")
        self.assertIsNotNone(helper, "the index helper must exist")
        body = _code_of(helper)
        self.assertIn("selected_dataset", body)
        self.assertIn("datasets.index(selected)", body)

    def test_no_functionality_was_dropped_in_the_merge(self) -> None:
        """Each capability the two tabs had is still reachable on the page."""
        for kept in (
            "Dataset readiness",          # readiness line
            "Raw path",                   # raw path
            "Processed path",             # processed path
            "Processed file count",       # from the old inspection tab
            "'preprocess'",               # the preprocess command
            "'prep_run'",                 # the preprocess action
            "Latest preprocessing log",   # the log expander
            "Feature / cache / artifact files",  # artifact inspection
            "'prep_go_run_model'",        # the hand-off to Run Model
        ):
            with self.subTest(kept=kept):
                self.assertIn(kept, self.code)

    def test_the_page_order_is_the_linear_one(self) -> None:
        """Selector → readiness → preprocess → log → inspection."""
        order = [
            "'prep_ds_select'",
            "Dataset readiness",
            "'prep_run'",
            "Latest preprocessing log",
            "Feature / cache / artifact files",
        ]
        positions = [self.code.index(token) for token in order]
        self.assertEqual(
            positions,
            sorted(positions),
            f"Prepare Data steps are out of order: {order}",
        )

    def test_an_empty_dataset_list_no_longer_hides_the_custom_tab(self) -> None:
        """The old early ``return`` skipped the rest of the page entirely."""
        self.assertIn("No datasets available.", self.code)
        merged = self.code.split("No datasets available.", 1)[1]
        self.assertIn("Custom", self.code)
        # The branch reports and continues rather than returning out of the
        # page, so the Custom Dataset tab below still renders.
        first_statement = merged.strip().splitlines()[1].strip()
        self.assertFalse(
            first_statement.startswith("return"),
            "an empty dataset list must not abandon the rest of the page",
        )


class TestSidebarIsCompact(unittest.TestCase):
    """3: tighter spacing, with nothing removed."""

    def setUp(self) -> None:
        self.css_source = STYLES_PATH.read_text(encoding="utf-8")
        from batteryml_web.ui import styles

        self.styles = styles
        self.css = styles.SIDEBAR_COMPACT_CSS

    def test_the_sidebar_injects_the_stylesheet(self) -> None:
        sidebar = SIDEBAR_PATH.read_text(encoding="utf-8")
        tree = ast.parse(sidebar)
        render = _function(tree, "render_sidebar_workflow")
        self.assertIsNotNone(render)
        self.assertIn("render_layout_css()", _code_of(render))
        self.assertIn("from .styles import render_layout_css", sidebar)

    def test_render_layout_css_emits_both_stylesheets(self) -> None:
        emitted: list[str] = []

        class _Recorder:
            session_state: dict = {}

            @staticmethod
            def markdown(text, **kwargs):
                emitted.append(str(text))

        saved = self.styles.st
        try:
            self.styles.st = _Recorder
            self.styles.render_layout_css()
        finally:
            self.styles.st = saved
        self.assertEqual(len(emitted), 2)
        self.assertIn(self.styles.SIDEBAR_COMPACT_CSS, emitted)
        self.assertIn(self.styles.ZERO_HEIGHT_COMPONENT_CSS, emitted)

    def test_nothing_in_the_sidebar_is_hidden(self) -> None:
        """Spacing only: a control the user cannot reach is not 'compact'."""
        for forbidden in ("display: none", "display:none", "visibility: hidden", "visibility:hidden"):
            self.assertNotIn(forbidden, self.css, f"the sidebar CSS must not use {forbidden}")

    def test_the_sidebar_keeps_every_navigation_entry(self) -> None:
        """The stylesheet tightens the current sidebar; it does not redesign it."""
        sidebar = SIDEBAR_PATH.read_text(encoding="utf-8")
        for entry in (
            '"label": "Home"',
            '"label": "Agent Workflow"',
            '"label": "Prepare Data"',
            '"label": "Run Model"',
            '"label": "Review Results"',
        ):
            self.assertIn(entry, sidebar)
        self.assertIn("Protocol Workflow", sidebar)
        self.assertIn("Current project", sidebar)

    def test_the_blocks_named_in_the_brief_are_all_tightened(self) -> None:
        """Header, project block, headings, buttons and dividers each shrink.

        Captions and section descriptions have no rule of their own on
        purpose — they are tightened by the block gap. See the next test.
        """
        for selector in (
            'data-testid="stSidebarHeader"',       # the top band
            'data-testid="stSidebarUserContent"',  # the content's own padding
            'data-testid="stVerticalBlock"',       # the gap between blocks
            'data-testid="stDivider"',             # dividers
            ".stButton > button",                  # the nav entries
        ):
            with self.subTest(selector=selector):
                self.assertIn(selector, self.css)
        self.assertIn('[data-testid="stSidebar"] hr', self.css)
        # The header block the brief calls out by name.
        self.assertIn('[data-testid="stSidebar"] h3', self.css)

    def test_the_two_values_a_browser_run_pinned_down(self) -> None:
        """Regression guard for the two rules that were measured, not guessed.

        A first attempt used ``gap: 0.3rem`` with the heading padding zeroed.
        Streamlit's sidebar heading is a 50px box that occupies 34px of
        layout, so it overflows the element below it by ~16px and the
        default gap is what hides that: the title landed on top of its own
        caption. A caption ``margin-bottom`` rule clipped the second line of
        a wrapped caption behind the button under it. Both were found by
        rendering the page, and both are cheap to reintroduce by eye.
        """
        gap = re.search(r"stVerticalBlock\"\]\s*\{[^}]*gap:\s*([0-9.]+)rem", self.css)
        self.assertIsNotNone(gap, "the sidebar must still set the block gap")
        self.assertGreaterEqual(
            float(gap.group(1)), 0.25,
            "below 0.25rem the sidebar's controls stop reading as separate rows",
        )

    def test_the_brand_margin_stays_coupled_to_the_block_gap(self) -> None:
        """What now keeps the title off its own caption, in place of a floor.

        The gap used to be floored at 0.4rem because the heading's ~16px
        overflow had nothing else holding it back. It does now: the brand
        rule cancels exactly one block gap, which lands the subtitle against
        the heading's own padding *whatever the gap is*. Measured in a
        browser at both values — 2.8px between the two lines of text at
        0.4rem, and the same 2.8px at 0.25rem.

        So the invariant is the pairing, not either number, and this asserts
        the pairing. Tuning the sidebar's rhythm without moving the brand
        margin with it is the regression this catches — and it is a stronger
        guarantee than the floor it replaces, because it holds at any gap.
        """
        gap = re.search(r"stVerticalBlock\"\]\s*\{[^}]*gap:\s*([0-9.]+)rem", self.css)
        self.assertIsNotNone(gap, "the sidebar must still set the block gap")
        brand = re.search(
            r":has\(> h3\)\s*\{[^}]*margin-bottom:\s*(-?[0-9.]+)rem", self.css
        )
        self.assertIsNotNone(
            brand, "the brand heading rule must still set a bottom margin"
        )
        self.assertAlmostEqual(
            float(brand.group(1)), -float(gap.group(1)), places=4,
            msg=(
                f"the brand margin ({brand.group(1)}rem) no longer cancels the "
                f"block gap ({gap.group(1)}rem); 'Protocol Navigator' will "
                "drift into or away from 'BatteryML'"
            ),
        )
        headings = re.search(
            r"stSidebar\"\]\s*h4\s*\{([^}]*)\}", self.css
        )
        self.assertIsNotNone(headings, "the heading rule must still exist")
        paddings = re.findall(r"padding-\w+:\s*([0-9.]+)\s*(rem|em|px)?", headings.group(1))
        self.assertTrue(paddings, "the heading rule must still set a padding")
        for value, _unit in paddings:
            # A bare ``0`` carries no unit, which is exactly the value that
            # collapsed the heading's box — so the unit is optional here.
            self.assertGreater(
                float(value), 0.0,
                "zeroing the heading padding collapses its box onto the caption",
            )
        # The caption rule may set type, never space: a caption
        # ``margin-bottom`` is what clipped a wrapped caption behind the
        # button beneath it, so the selector is allowed back but those
        # properties are not.
        caption_rules = re.findall(
            r"stCaptionContainer[^{]*\{([^}]*)\}", self._declaration_body()
        )
        for body in caption_rules:
            declared = {
                declaration.split(":", 1)[0].strip()
                for declaration in body.split(";")
                if ":" in declaration
            }
            for property_name in declared:
                # Matched on the property, not on a substring: "line-height"
                # is type, and must not be read as a "height".
                root = property_name.split("-")[0]
                self.assertNotIn(
                    root, {"margin", "height", "overflow", "padding"},
                    f"a caption {property_name} rule clipped wrapped captions "
                    "behind the next button",
                )
            line_height = re.search(r"line-height:\s*([0-9.]+)", body)
            if line_height:
                self.assertGreater(
                    float(line_height.group(1)), 1.0,
                    "a line-height at or below 1 crops a wrapped caption",
                )

    def _declaration_body(self) -> str:
        """The stylesheet with its comments removed.

        Comments are prose about the rules, not rules: a comment naming a
        sidebar block must not be parsed as a selector.
        """
        body = self.css.split("<style>", 1)[1].split("</style>", 1)[0]
        return re.sub(r"/\*.*?\*/", "", body, flags=re.DOTALL)

    def test_every_rule_is_scoped_to_the_sidebar(self) -> None:
        """A global rule here would re-space the whole app by accident."""
        body = self._declaration_body()
        selectors = [
            part.strip()
            for block in body.split("}")
            if "{" in block
            for part in block.split("{", 1)[0].split(",")
            if part.strip()
        ]
        self.assertGreater(len(selectors), 5, "the stylesheet scan found almost nothing")
        for selector in selectors:
            with self.subTest(selector=selector):
                self.assertIn("stSidebar", selector)

    def test_only_spacing_and_type_size_properties_are_set(self) -> None:
        """Nothing here may change colour, content or behaviour.

        ``font-size`` joined the allowed list when the sidebar's grey
        secondary lines were reduced to 0.82rem. It is a size, not a
        disappearance: the text is still rendered, still selectable and still
        wraps. Colour, ``display``, ``visibility``, ``content`` and
        ``overflow`` remain out — those are the properties that can remove a
        control rather than fit one.
        """
        body = self._declaration_body()
        properties = {
            declaration.split(":", 1)[0].strip()
            for block in body.split("}")
            if "{" in block
            for declaration in block.split("{", 1)[1].split(";")
            if ":" in declaration
        }
        allowed = {
            "padding", "padding-top", "padding-bottom", "padding-left", "padding-right",
            "margin", "margin-top", "margin-bottom", "margin-left", "margin-right",
            "gap", "line-height", "height", "min-height", "font-size",
            # The sidebar's own width, in the same family as height/min-height
            # and admitted on the same reasoning: a box that is narrower still
            # renders every control, still wraps, still scrolls. The properties
            # that stay out are the ones that can make a control unreachable —
            # display, visibility, overflow, content, colour.
            "width", "min-width",
        }
        self.assertTrue(properties, "the stylesheet declares nothing")
        self.assertEqual(
            properties - allowed,
            set(),
            "the sidebar stylesheet may only adjust spacing and type size",
        )

    def test_the_sidebar_secondary_text_is_smaller_but_still_readable(self) -> None:
        """The brief's range, and a floor so it cannot shrink into illegibility."""
        sizes = re.findall(
            r"stCaptionContainer[^{]*\{[^}]*font-size:\s*([0-9.]+)rem",
            self._declaration_body(),
        )
        self.assertTrue(sizes, "the sidebar caption size rule is missing")
        for size in sizes:
            self.assertGreaterEqual(float(size), 0.80)
            self.assertLessEqual(float(size), 0.85)

    def test_the_forehead_is_reduced_to_fit_its_control(self) -> None:
        """The top band, measured in a browser at 60px before this change.

        Streamlit fixes it with a `height`, not padding, so padding alone
        could not shrink it. The collapse control inside is 28px, so the band
        may not go below that plus its own padding, or the control would be
        clipped.
        """
        match = re.search(
            r'stSidebarHeader"\]\s*\{([^}]*)\}', self.css, re.DOTALL
        )
        self.assertIsNotNone(match, "the sidebar header rule is gone")
        block = match.group(1)
        height = re.search(r"\bheight:\s*([0-9.]+)rem", block)
        self.assertIsNotNone(height, "the header height must be set, not just padding")
        rem = float(height.group(1))
        self.assertLessEqual(rem, 2.5, "the forehead is still as tall as Streamlit's default")
        self.assertGreaterEqual(
            rem, 2.0,
            "below ~32px the 28px sidebar-collapse control would be clipped",
        )

    def test_the_sidebar_captions_are_short_enough_not_to_need_clipping(self) -> None:
        """Shortened copy, not CSS that hides wrapped text.

        ``description`` is optional — an entry may carry none at all — but an
        entry that does carry one may not carry an empty string, which would
        draw a caption container with nothing in it.
        """
        from batteryml_web.ui import workflow_sidebar

        described = 0
        for step in workflow_sidebar.NAV_STEPS:
            with self.subTest(step=step["key"]):
                if "description" not in step:
                    continue
                description = str(step["description"])
                self.assertTrue(
                    description,
                    f"{step['key']} declares an empty description; drop the key "
                    "instead, so no caption is drawn at all",
                )
                described += 1
                self.assertLessEqual(
                    len(description), 80,
                    f"{step['key']}'s caption is long enough to wrap to three "
                    "lines in a 300px sidebar; shorten it rather than clipping it",
                )
        # Non-vacuity: if every description were dropped the loop above would
        # pass without checking a single string.
        self.assertGreater(described, 0, "no navigation entry carries a description")

    def test_nothing_clips_overflowing_text(self) -> None:
        """A hidden caption is worse than a tall one."""
        for forbidden in ("overflow: hidden", "overflow:hidden",
                          "text-overflow", "white-space: nowrap", "-webkit-line-clamp"):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(
                    forbidden, self.css,
                    "the sidebar stylesheet must not clip text; shorten the "
                    "copy instead",
                )

    def test_dividers_are_tighter_than_the_block_gap_allows_alone(self) -> None:
        """The rules that remain still separate the sidebar's sections."""
        match = re.search(
            r"stDivider\"\]\s*\{([^}]*)\}", self.css, re.DOTALL
        )
        self.assertIsNotNone(match)
        values = re.findall(r"margin-\w+:\s*([0-9.]+)rem", match.group(1))
        self.assertTrue(values)
        for value in values:
            self.assertLessEqual(float(value), 0.35)
            self.assertGreater(float(value), 0.0, "a divider still needs some air")

    def test_no_generated_class_names_are_relied_on(self) -> None:
        """Streamlit's ``css-1x2y3z`` classes change between releases."""
        self.assertIsNone(
            re.search(r"\.(css|st-emotion)-[0-9a-z]{4,}", self.css),
            "use stable data-testid selectors, not generated class names",
        )


class _SidebarRecorder:
    """Records the sidebar's element sequence, in order.

    A stand-in rather than a source scan: what matters is the order the
    elements come out in, which is the thing a reader sees and the thing a
    refactor can break without touching any of the strings a scan looks for.
    """

    def __init__(self, session: dict) -> None:
        self.session_state = session
        self.sequence: list[tuple[str, str]] = []
        self.sidebar = self

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False

    def markdown(self, text="", **kwargs):
        self.sequence.append(("markdown", str(text)))

    def caption(self, text="", **kwargs):
        self.sequence.append(("caption", str(text)))

    def divider(self, **kwargs):
        self.sequence.append(("divider", ""))

    def button(self, label="", **kwargs):
        self.sequence.append(("button", str(label)))
        return False

    def columns(self, spec, **kwargs):
        count = spec if isinstance(spec, int) else len(spec)
        return [self for _ in range(count)]

    def container(self, **kwargs):
        """A group wrapper, recorded by key and kept transparent.

        Returning ``self`` keeps the recorded sequence flat, which is what
        the order assertions below read: grouping the sidebar changed where
        elements sit relative to each other in *space*, not in order, and a
        recorder that nested them would make those assertions about the
        recorder rather than about the sidebar.
        """
        key = kwargs.get("key")
        if key is not None:
            self.sequence.append(("container", str(key)))
        return self


class TestSidebarStructure(unittest.TestCase):
    """One rule, in the one place a boundary exists.

    Every divider that used to sit around the header, Home, Agent Workflow,
    Prev/Next, settings and the status line said "a different kind of thing
    follows" about boundaries that are not there: a list and its own
    Prev/Next controls are one thing, and so are a single-line settings
    control and a single-line status readout. The rule that remains marks
    the real boundary — the two ways in, then the numbered protocol route.
    """

    def _render(self, current_step: str = "home") -> list[tuple[str, str]]:
        from batteryml_web.ui import workflow_sidebar as sidebar

        recorder = _SidebarRecorder({"current_step": current_step})
        saved = (sidebar.st, sidebar.render_layout_css,
                 sidebar.render_sidebar_settings)
        sidebar.st = recorder
        sidebar.render_layout_css = lambda: None
        # Recorded rather than dropped: "no rule immediately above Settings"
        # is a statement about where Settings sits in the sequence, so the
        # sequence has to contain it.
        sidebar.render_sidebar_settings = lambda: recorder.sequence.append(
            ("settings", "")
        )
        try:
            sidebar.render_sidebar_workflow()
        finally:
            (sidebar.st, sidebar.render_layout_css,
             sidebar.render_sidebar_settings) = saved
        return recorder.sequence

    def test_no_rule_separates_the_header_from_home_or_the_agent_entry(self) -> None:
        sequence = self._render()
        kinds = [kind for kind, _ in sequence]
        labels = [text for _, text in sequence]
        agent = labels.index("Agent Workflow")
        # Everything from the title down to the Agent Workflow entry is one
        # uninterrupted block.
        self.assertNotIn("divider", kinds[:agent + 1])

    def test_the_only_rule_is_the_one_above_the_protocol_list(self) -> None:
        sequence = self._render()
        kinds = [kind for kind, _ in sequence]
        labels = [text for _, text in sequence]
        heading = labels.index("**Protocol Workflow**")
        # A rule immediately precedes the numbered list: that boundary is
        # real, and losing it would be the opposite regression.
        self.assertEqual(
            kinds[heading - 1], "divider",
            "the rule above the Protocol Workflow heading is the one that "
            "marks a real boundary; it may not be removed with the others",
        )
        self.assertEqual(
            kinds.count("divider"), 1,
            "the sidebar draws exactly one rule; found "
            f"{kinds.count('divider')} at indices "
            f"{[i for i, k in enumerate(kinds) if k == 'divider']}",
        )

    def test_no_rule_sits_immediately_above_settings(self) -> None:
        """The rule the user asked for by name.

        Asserted positionally rather than by counting, so it keeps holding if
        a rule is ever added back somewhere else in the sidebar.
        """
        sequence = self._render()
        kinds = [kind for kind, _ in sequence]
        self.assertIn("settings", kinds, "the settings block is no longer drawn")
        settings = kinds.index("settings")
        self.assertNotEqual(
            kinds[settings - 1], "divider",
            "a rule is drawn immediately above Settings",
        )

    def test_no_rule_sits_immediately_above_prev_next_or_the_llm_status(self) -> None:
        sequence = self._render()
        kinds = [kind for kind, _ in sequence]
        labels = [text for _, text in sequence]
        prev = labels.index("← Prev")
        self.assertNotEqual(
            kinds[prev - 1], "divider",
            "a rule is drawn immediately above the Prev/Next pair",
        )
        status = next(
            index for index, text in enumerate(labels) if text.startswith("<div") and "LLM:" in text
        )
        self.assertNotEqual(
            kinds[status - 1], "divider",
            "a rule is drawn immediately above the LLM status line",
        )

    def test_the_helper_copy_the_brief_named_is_gone(self) -> None:
        """Both strings, checked against what is actually drawn.

        Home's caption only appears while Home is the current step, so the
        sidebar is rendered on both pages rather than once.
        """
        removed = (
            "Create/load a project or start a workflow.",
            "The manual route; works offline, with no API key.",
        )
        for step in ("home", "agent_workflow"):
            drawn = [text for _, text in self._render(current_step=step)]
            for copy in removed:
                with self.subTest(step=step, copy=copy):
                    self.assertNotIn(
                        copy, drawn,
                        f"{copy!r} is still drawn in the sidebar on {step}",
                    )

    def test_the_helper_copy_is_not_replaced_by_another_paragraph(self) -> None:
        """Removed, not reworded.

        Home carries no caption of its own at all now, and the Protocol
        Workflow heading is followed straight by its first numbered step.
        """
        from batteryml_web.ui import workflow_sidebar as sidebar

        home = next(step for step in sidebar.NAV_STEPS if step["key"] == "home")
        self.assertNotIn(
            "description", home,
            "Home carries a caption again; the brief removed it rather than "
            "asking for shorter copy",
        )

        sequence = self._render()
        labels = [text for _, text in sequence]
        kinds = [kind for kind, _ in sequence]
        heading = labels.index("**Protocol Workflow**")
        self.assertNotEqual(
            kinds[heading + 1], "caption",
            f"a caption follows the Protocol Workflow heading: {labels[heading + 1]!r}",
        )

    def test_the_sidebar_is_drawn_as_groups_not_as_one_flat_list(self) -> None:
        """Every row used to sit the same distance from its neighbour.

        That is even, but it says nothing: "Protocol Navigator" read as
        closely related to "Current project" as "Current project" did to the
        project name belonging to it. The groups are what let the stylesheet
        space *between* subjects differently from *within* one.
        """
        from batteryml_web.ui import workflow_sidebar as sidebar

        opened = [text for kind, text in self._render() if kind == "container"]
        self.assertEqual(
            opened, list(sidebar.SIDEBAR_GROUP_KEYS),
            "the sidebar's groups, in order, are what the stylesheet spaces",
        )

    def test_each_group_key_is_styled_and_each_styled_key_exists(self) -> None:
        """A key with no rule, or a rule with no key, silently does nothing."""
        from batteryml_web.ui import workflow_sidebar as sidebar

        from batteryml_web.ui.styles import SIDEBAR_COMPACT_CSS as css

        styled = set(re.findall(r"\.st-key-(sb_\w+)", css))
        declared = set(sidebar.SIDEBAR_GROUP_KEYS)
        # The first group takes no top margin: its space above is the header
        # band's, so it is deliberately not styled here.
        self.assertEqual(
            styled, declared - {"sb_brand"},
            "a group is styled that does not exist, or exists unstyled",
        )

    #: The one group whose separation is not its margin: it opens with the
    #: sidebar's single rule, which announces the boundary on its own. Its
    #: margin is deliberately *smaller* so the two do not stack — measured,
    #: it still lands at ~30px from the entry group above, comparable to the
    #: groups that use a margin alone.
    RULE_SEPARATED_GROUP = "sb_protocol"

    def test_the_gap_between_groups_is_clearly_larger_than_the_gap_within_one(self) -> None:
        """The whole point of the grouping, as a number.

        Measured in a browser: 11.2px inside a group against 36.8px between
        two — a ratio the eye reads as grouping rather than has to measure.
        Anything under about 2:1 stops reading as a break.

        The gap itself is not floored here on purpose. It was once cut to
        4px to save height and the navigation buttons welded into a strip;
        what that proved is that the *ratio* is the invariant and the
        absolute value is a judgement, so only the ratio is asserted.
        """
        from batteryml_web.ui.styles import SIDEBAR_COMPACT_CSS as css

        gap = re.search(r"stVerticalBlock\"\]\s*\{[^}]*gap:\s*([0-9.]+)rem", css)
        self.assertIsNotNone(gap, "the sidebar must still set the block gap")
        within = float(gap.group(1))

        # The margin each group's selector actually ends up with: later rules
        # override earlier ones, so a per-group override is what counts.
        margins: dict[str, float] = {}
        for block in css.split("}"):
            if ".st-key-sb_" not in block:
                continue
            found = re.search(r"margin-top:\s*([0-9.]+)rem", block)
            if not found:
                continue
            for key in re.findall(r"\.st-key-(sb_\w+)", block):
                margins[key] = float(found.group(1))
        self.assertTrue(margins, "no group carries a separating margin")

        for key, margin in sorted(margins.items()):
            if key == self.RULE_SEPARATED_GROUP:
                continue
            with self.subTest(group=key):
                self.assertGreaterEqual(
                    margin / within, 2.0,
                    f"{key} separates itself by {margin}rem against a "
                    f"{within}rem gap within a group, so it stops reading as "
                    "a separate group",
                )

    def test_the_rule_separated_group_leans_on_its_rule_not_a_big_margin(self) -> None:
        """Non-vacuity for the exemption above: it is exempt for a reason.

        Stacking a full group margin on top of the divider would separate
        that one boundary twice as hard as every other, which is its own kind
        of uneven. So its margin is smaller than the rest — and that is
        asserted, rather than the group simply being skipped.
        """
        from batteryml_web.ui.styles import SIDEBAR_COMPACT_CSS as css

        margins: dict[str, float] = {}
        for block in css.split("}"):
            if ".st-key-sb_" not in block:
                continue
            found = re.search(r"margin-top:\s*([0-9.]+)rem", block)
            if not found:
                continue
            for key in re.findall(r"\.st-key-(sb_\w+)", block):
                margins[key] = float(found.group(1))

        exempt = margins.get(self.RULE_SEPARATED_GROUP)
        self.assertIsNotNone(exempt, "the rule-separated group carries no margin")
        others = [m for key, m in margins.items() if key != self.RULE_SEPARATED_GROUP]
        self.assertTrue(others)
        self.assertLess(
            exempt, max(others),
            "the group that opens with the rule takes the full group margin "
            "as well, so that boundary is separated twice",
        )
        # And it really does open with the rule.
        sidebar = SIDEBAR_PATH.read_text(encoding="utf-8")
        block = sidebar.split(f'key="{self.RULE_SEPARATED_GROUP}"', 1)[1]
        self.assertLess(
            block.index("st.divider()"), block.index("Protocol Workflow"),
            "the rule is no longer the first thing in that group",
        )

    def test_the_content_column_has_gutters_so_controls_are_not_stretched(self) -> None:
        """Why the sidebar looked too wide: the panel is not what changed.

        Streamlit writes the panel's width inline because the sidebar is
        drag-resizable, so pinning it in CSS needs ``!important`` — measured,
        that does narrow the panel and does stop the drag handle working.
        The gutter narrows the content column instead and leaves the control
        alone.
        """
        from batteryml_web.ui.styles import SIDEBAR_COMPACT_CSS as css

        match = re.search(
            r'stSidebarUserContent"\]\s*\{([^}]*)\}', css, re.DOTALL
        )
        self.assertIsNotNone(match, "the sidebar content rule is gone")
        for side in ("padding-left", "padding-right"):
            found = re.search(rf"{side}:\s*([0-9.]+)rem", match.group(1))
            with self.subTest(side=side):
                self.assertIsNotNone(found, f"the content column has no {side}")
                self.assertGreater(float(found.group(1)), 0.0)
        # Declarations only: the rule above explains in prose *why* the
        # panel is not pinned with !important, and that explanation must not
        # be what trips this.
        declarations = re.sub(r"/\*.*?\*/", "", css, flags=re.DOTALL)
        self.assertNotIn(
            "!important", declarations,
            "!important on the sidebar pins its width and stops the user "
            "dragging it; the gutter achieves the same look without that",
        )

    def test_the_agent_entry_keeps_the_one_description_that_carries_a_promise(self) -> None:
        """Requirement 6: one short description may stay where it adds value.

        Agent Workflow's says nothing runs until Run is pressed — the
        execution contract from CLAUDE.md §3, which the label cannot carry.
        """
        from batteryml_web.ui import workflow_sidebar as sidebar

        agent = sidebar.AGENT_STEPS[0]
        description = agent.get("description", "")
        self.assertIn("Run", description)
        self.assertLessEqual(len(description), 80)

    def test_every_navigation_entry_is_still_reachable(self) -> None:
        """Non-vacuity: removing rules must not remove destinations."""
        from batteryml_web.ui import workflow_sidebar as sidebar

        labels = [text for _, text in self._render()]
        for step in sidebar.NAV_STEPS:
            expected = sidebar.nav_label(step)
            with self.subTest(step=step["key"]):
                # Home is the current step here, so it is bold text rather
                # than a button; every other entry is a button.
                self.assertTrue(
                    expected in labels or f"**{expected}**" in labels,
                    f"{expected} is no longer drawn in the sidebar",
                )


class TestRunProgressSpansTheContentArea(unittest.TestCase):
    """A progress bar is read as a proportion of the page, so it gets the page.

    Both Run buttons sit in ``st.columns`` so they can share a row with
    another control, and anything drawn under a button inherits that column's
    width. Measured in a browser before the fix: the bar came out at 66% of
    the content area on Run Model (``columns([1, 2])``) and 49% on the Home
    demo (``columns(2)``). The fix is a container reserved *outside* the
    columns and handed to the renderer — not a width forced onto the widget,
    which is why it stays responsive and cannot overflow.
    """

    def test_the_renderer_draws_everything_into_the_container_it_is_given(self) -> None:
        """All of it or none of it: a half-redirected renderer still splits.

        If the bar moved out of the column but its status line, log box or
        error stayed behind, the run's output would be torn across two
        widths — worse than the narrow bar it replaced.
        """
        import inspect
        import textwrap

        from batteryml_web.ui import progress_utils

        source = textwrap.dedent(
            inspect.getsource(progress_utils.run_command_with_progress)
        )
        tree = ast.parse(source)

        # Read by AST, not by substring: "st.error(" is a substring of
        # "host.error(", so a text scan passes or fails for the wrong reason.
        bare_st = {
            node.func.attr
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "st"
        }
        # `st.code` is the single legitimate one: it sits inside
        # `with host.expander(...)`, where the active container is already
        # the expander, so it lands in the caller's container anyway.
        self.assertEqual(
            bare_st, {"code"},
            f"{sorted(bare_st - {'code'})} bypass the caller's container, so "
            "those elements would stay behind in the narrow column",
        )
        inside_expander = {
            inner.func.attr
            for node in ast.walk(tree) if isinstance(node, ast.With)
            for item in node.items
            if isinstance(item.context_expr, ast.Call)
            and isinstance(item.context_expr.func, ast.Attribute)
            and item.context_expr.func.attr == "expander"
            and isinstance(item.context_expr.func.value, ast.Name)
            and item.context_expr.func.value.id == "host"
            for inner in ast.walk(node)
            if isinstance(inner, ast.Call)
            and isinstance(inner.func, ast.Attribute)
            and isinstance(inner.func.value, ast.Name)
            and inner.func.value.id == "st"
        }
        self.assertEqual(
            inside_expander, {"code"},
            "st.code must be the one inside `with host.expander(...)`",
        )
        self.assertIn("host = container if container is not None else st", source)

    def test_the_container_is_reserved_outside_the_button_columns(self) -> None:
        """The whole point: reserved *before* the ``with`` block, not inside.

        Reserving it inside ``with c2:`` would compile and run and look
        correct, and the bar would still be two-thirds width — so the
        position is what this asserts, by AST.
        """
        tree = ast.parse(APP_PATH.read_text(encoding="utf-8"))

        def _hosts(scope: ast.AST) -> set[str]:
            """`<name>_progress_host = st.container()` anywhere under ``scope``."""
            return {
                node.targets[0].id
                for node in ast.walk(scope)
                if isinstance(node, ast.Assign)
                and isinstance(node.value, ast.Call)
                and isinstance(node.value.func, ast.Attribute)
                and node.value.func.attr == "container"
                and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id.endswith("progress_host")
            }

        # The names bound by `a, b = st.columns(...)`. Only these matter: an
        # enclosing `with tabs[2]:` is a tab, not a width constraint, and the
        # container is legitimately inside one of those.
        column_names: set[str] = set()
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Assign)
                and isinstance(node.value, ast.Call)
                and isinstance(node.value.func, ast.Attribute)
                and node.value.func.attr == "columns"
                and isinstance(node.targets[0], ast.Tuple)
            ):
                column_names.update(
                    element.id for element in node.targets[0].elts
                    if isinstance(element, ast.Name)
                )
        self.assertTrue(column_names, "no st.columns unpacking found in app.py")

        inside_a_column = set()
        for node in ast.walk(tree):
            if not isinstance(node, ast.With):
                continue
            if any(
                isinstance(item.context_expr, ast.Name)
                and item.context_expr.id in column_names
                for item in node.items
            ):
                inside_a_column |= _hosts(node)

        self.assertEqual(
            _hosts(tree), {"run_progress_host", "demo_progress_host"},
            "both Run buttons must reserve a full-width progress container",
        )
        self.assertEqual(
            inside_a_column, set(),
            f"{sorted(inside_a_column)} is reserved inside a column's `with` "
            "block, so it inherits that column's width and the bar is narrow "
            "again",
        )

    def test_every_column_bound_run_passes_its_container(self) -> None:
        """Reserving a container helps nothing if the call does not use it."""
        tree = ast.parse(APP_PATH.read_text(encoding="utf-8"))
        calls = [
            node for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "run_command_with_progress"
        ]
        self.assertEqual(len(calls), 3, "the known run/preprocess call sites")
        with_container = [
            call for call in calls
            if any(kw.arg == "container" for kw in call.keywords)
        ]
        # Two of the three are inside columns and must pass one; the
        # preprocessing call is already at full width and needs nothing.
        self.assertEqual(
            len(with_container), 2,
            "the two column-bound Run buttons must pass container=",
        )
        passed = {
            kw.value.id
            for call in with_container for kw in call.keywords
            if kw.arg == "container" and isinstance(kw.value, ast.Name)
        }
        self.assertEqual(passed, {"run_progress_host", "demo_progress_host"})

    def test_no_pixel_or_viewport_width_is_forced_on_the_progress_bar(self) -> None:
        """Responsiveness comes from filling a container, not from a number.

        ``100vw`` in particular includes the sidebar, so it would overflow
        horizontally rather than fit the content area.
        """
        for path in (APP_PATH, REPO_ROOT / "batteryml_web" / "ui" / "progress_utils.py",
                     STYLES_PATH):
            text = path.read_text(encoding="utf-8")
            with self.subTest(path=path.name):
                self.assertNotIn("100vw", text)
                self.assertNotIn("stProgress", text,
                                 "the widget is not targeted directly; its "
                                 "container is what decides its width")


if __name__ == "__main__":
    unittest.main(verbosity=2)
