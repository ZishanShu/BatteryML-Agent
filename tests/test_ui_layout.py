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
            float(gap.group(1)), 0.4,
            "a gap below 0.4rem lets the sidebar heading overlap its caption",
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


if __name__ == "__main__":
    unittest.main(verbosity=2)
