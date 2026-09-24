"""Can a reader tell a control from a read-out, and rank two controls?

The complaint these guards answer was not about one button. Measured on the
running app before this pass, at 1500px, *every* action on *every* page
rendered identically::

    "Run BatteryML"          white  1px  8px radius  weight 400
    "View Dataset Overview"  white  1px  8px radius  weight 400
    "Go to Run Model"        white  1px  8px radius  weight 400

Three semantic roles, one appearance, because not one of the app's 54
action controls passed ``type=`` to Streamlit and every one of them fell
back to the default ``secondary``.

So these are not pixel tests. They assert the three things that make the
hierarchy real and that a later edit could silently undo:

1. every role still resolves to a *different* native Streamlit kind, so
   the three are distinguishable at all;
2. the stylesheet still says something different about each kind, and says
   it through stable selectors rather than generated hashes;
3. an information surface is still a ``<div>`` with no control affordance,
   and a control is still a real ``<button>``.

What is deliberately *not* asserted: exact colours. A palette is allowed to
move. The relations — filled beats outlined beats text, card radius under
control radius — are what must hold.
"""

from __future__ import annotations

import ast
import re
import unittest
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1]
SURFACES_PATH = REPO_ROOT / "batteryml_web" / "ui" / "surfaces.py"
STYLES_PATH = REPO_ROOT / "batteryml_web" / "ui" / "styles.py"
APP_PATH = REPO_ROOT / "batteryml_web" / "app.py"

UI_DIR = REPO_ROOT / "batteryml_web" / "ui"

#: Streamlit's own generated class names. Any of these in an app stylesheet
#: is a latent breakage: they change between releases.
_GENERATED_CLASS = re.compile(r"\.(st-emotion-cache-|css-)[A-Za-z0-9_-]+")

#: The names a call site may pass as ``type=``. Names, not values: the
#: call sites import the constants, so this is what the AST shows.
_ROLE_CONSTANTS = ("PRIMARY_ROLE", "SECONDARY_ROLE", "NAV_ROLE")


class _RecordingSt:
    """Captures what a surfaces call asks Streamlit to draw."""

    def __init__(self) -> None:
        self.buttons: List[Dict[str, Any]] = []
        self.markdown_html: List[str] = []
        self.session_state: dict = {}

    def button(self, label: str, **kwargs: Any) -> bool:
        self.buttons.append({"label": label, **kwargs})
        return False

    def markdown(self, body: str, **_kwargs: Any) -> None:
        self.markdown_html.append(str(body))


def _decomment(css: str) -> str:
    """CSS with ``/* … */`` removed.

    The rule scanner below reads "everything since the last brace" as the
    selector, and these stylesheets are heavily commented, so without this
    a selector arrives as the paragraph explaining it followed by the
    selector itself — and never compares equal to anything.
    """
    return re.sub(r"/\*.*?\*/", "", css, flags=re.S)


def _rule_body(css: str, selector: str) -> str:
    """The declarations that win for ``selector``.

    The **last** matching rule, not the first: CSS cascades, and several of
    these selectors appear twice — once in a grouped rule that sets a shared
    value and again in a later override. Reading the first match reported
    the group's value for a selector that overrides it.

    Returns "" when no rule matches, so a caller can assert absence too.
    """
    body = ""
    for block in re.finditer(r"([^{}]+)\{([^{}]*)\}", _decomment(css)):
        if selector in block.group(1):
            body = block.group(2)
    return body


def _exact_rule(css: str, selector: str) -> str:
    """The declarations of the rule whose selector is exactly ``selector``.

    ``_rule_body`` is not enough for a base rule. Every one of these
    selectors also appears inside longer ones — ``…-primary`` inside
    ``…-primary:hover``, ``.bml-info-card`` inside
    ``.bml-info-card .bml-info-card-status``, ``.dataset-card`` inside
    ``.dataset-card.ready`` — and substring matching returns whichever
    comes last. A question like "is primary filled?" was being answered by
    the shared ``:focus-visible`` block, which only sets an outline.
    Comparing the selector part exactly is what asks about the base rule.
    """
    body = ""
    for block in re.finditer(r"([^{}]+)\{([^{}]*)\}", _decomment(css)):
        for part in block.group(1).split(","):
            # The last line of the part, because the very first rule in a
            # sheet carries the literal "<style>" opening tag ahead of its
            # selector and would otherwise never compare equal.
            tail = part.strip().splitlines()[-1].strip() if part.strip() else ""
            if tail == selector:
                body = block.group(2)
    return body


def _role_rule(css: str, testid: str) -> str:
    """The base rule for one action role."""
    return _exact_rule(css, f'[data-testid="{testid}"]')


def _call_labels(role_constant: str) -> List[str]:
    """Every literal label of a control carrying ``type=<role_constant>``.

    By AST rather than by text: these calls are wrapped across lines by the
    formatter, so ``st.button("Go to Home", type=NAV_ROLE`` is simply not a
    substring of the source even when the call is exactly that.
    """
    labels: List[str] = []
    for path in [APP_PATH] + sorted(UI_DIR.glob("*.py")):
        if path.name == "surfaces.py":
            continue
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if not (isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr.endswith("button")
                    and node.args
                    and isinstance(node.args[0], ast.Constant)
                    and isinstance(node.args[0].value, str)):
                continue
            for kw in node.keywords:
                if kw.arg == "type" and ast.unparse(kw.value) == role_constant:
                    labels.append(node.args[0].value)
    return labels


# --------------------------------------------------------------------------
# 1. the three roles are actually three
# --------------------------------------------------------------------------


class TestRolesAreDistinct(unittest.TestCase):
    """The defect in one assertion: three roles must not collapse to one."""

    def setUp(self) -> None:
        from batteryml_web.ui import surfaces

        self.surfaces = surfaces

    def test_each_role_maps_to_a_different_streamlit_kind(self) -> None:
        roles = [
            self.surfaces.PRIMARY_ROLE,
            self.surfaces.SECONDARY_ROLE,
            self.surfaces.NAV_ROLE,
        ]
        self.assertEqual(
            len(set(roles)), 3, f"roles collapsed onto each other: {roles}"
        )

    def test_the_kinds_are_the_ones_streamlit_accepts(self) -> None:
        """A typo here would silently render every button as secondary again."""
        import streamlit as st
        import inspect

        annotation = str(
            inspect.signature(st.button).parameters["type"].annotation
        )
        for role in (
            self.surfaces.PRIMARY_ROLE,
            self.surfaces.SECONDARY_ROLE,
            self.surfaces.NAV_ROLE,
        ):
            with self.subTest(role=role):
                self.assertIn(
                    f"'{role}'",
                    annotation,
                    f"{role!r} is not a button type this Streamlit accepts",
                )

    def test_each_role_has_its_own_testid(self) -> None:
        ids = [
            self.surfaces.PRIMARY_TESTID,
            self.surfaces.SECONDARY_TESTID,
            self.surfaces.NAV_TESTID,
        ]
        self.assertEqual(len(set(ids)), 3, f"testids collapsed: {ids}")
        for role, testid in (
            (self.surfaces.PRIMARY_ROLE, self.surfaces.PRIMARY_TESTID),
            (self.surfaces.SECONDARY_ROLE, self.surfaces.SECONDARY_TESTID),
            (self.surfaces.NAV_ROLE, self.surfaces.NAV_TESTID),
        ):
            with self.subTest(role=role):
                self.assertTrue(
                    testid.endswith(role),
                    f"{testid!r} does not name the {role!r} kind, so the "
                    "stylesheet would style the wrong control",
                )


# --------------------------------------------------------------------------
# 2. the helpers produce native controls, not decorated divs
# --------------------------------------------------------------------------


class TestActionsStayNativeButtons(unittest.TestCase):
    """Accessibility: keyboard, focus and tab order come from the element.

    The roles are passed as ``type=`` on ``st.button`` rather than through
    a wrapper in ``surfaces``. An earlier draft did use wrappers, and the
    full suite caught the cost: this repo's page tests replace the *page
    module's* ``st`` with a recorder, so a wrapper calling ``surfaces.st``
    made the button invisible to fifteen existing tests. These assertions
    pin the decision so it is not quietly undone.
    """

    def test_every_role_marked_control_is_a_real_streamlit_button(self) -> None:
        for path in [APP_PATH] + sorted(UI_DIR.glob("*.py")):
            for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
                if not isinstance(node, ast.Call):
                    continue
                kwargs = {k.arg: k.value for k in node.keywords if k.arg}
                if "type" not in kwargs:
                    continue
                value = ast.unparse(kwargs["type"])
                if value not in _ROLE_CONSTANTS:
                    continue
                with self.subTest(where=f"{path.name}:{node.lineno}"):
                    self.assertIsInstance(
                        node.func,
                        ast.Attribute,
                        "a role was passed to something that is not st.button",
                    )
                    self.assertIn(
                        node.func.attr,
                        ("button", "download_button", "form_submit_button"),
                        f"{node.func.attr} is not a native Streamlit control",
                    )

    def test_no_page_routes_a_control_through_a_surfaces_wrapper(self) -> None:
        """The indirection that blinded the existing test doubles."""
        for path in [APP_PATH] + sorted(UI_DIR.glob("*.py")):
            if path.name == "surfaces.py":
                continue
            for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                    with self.subTest(where=f"{path.name}:{node.lineno}"):
                        self.assertNotIn(
                            node.func.id,
                            ("primary_action", "secondary_action", "nav_action"),
                            "a control is routed through a surfaces wrapper; "
                            "page tests that patch this module's `st` would "
                            "stop seeing it",
                        )

    def test_a_role_is_never_spelled_as_a_bare_string(self) -> None:
        """A literal type="primary" drifts silently when the vocabulary moves."""
        offenders = []
        for path in [APP_PATH] + sorted(UI_DIR.glob("*.py")):
            if path.name == "surfaces.py":
                continue
            for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
                if not (isinstance(node, ast.Call)
                        and isinstance(node.func, ast.Attribute)
                        and node.func.attr.endswith("button")):
                    continue
                for kw in node.keywords:
                    if kw.arg == "type" and isinstance(kw.value, ast.Constant):
                        offenders.append(f"{path.name}:{node.lineno}")
        self.assertEqual(offenders, [], f"hard-coded role strings: {offenders}")


# --------------------------------------------------------------------------
# 3. information surfaces are not controls
# --------------------------------------------------------------------------


class TestInformationIsNotAControl(unittest.TestCase):
    """The user's main complaint: a card must not read as a button."""

    def setUp(self) -> None:
        from batteryml_web.ui import surfaces

        self.surfaces = surfaces
        self.css = surfaces.SURFACE_CSS

    def test_a_card_renders_as_a_div_and_never_as_a_button(self) -> None:
        fake = _RecordingSt()
        saved = self.surfaces.st
        try:
            self.surfaces.st = fake
            self.surfaces.render_info_card(
                "MATR", ["raw: 180 files"], status="READY", status_color="#2e7d32"
            )
        finally:
            self.surfaces.st = saved
        html = "".join(fake.markdown_html)
        self.assertIn("<div", html)
        self.assertNotIn("<button", html.lower())
        self.assertNotIn("onclick", html.lower())
        self.assertNotIn("role=\"button\"", html.lower())
        self.assertEqual(fake.buttons, [], "a card must not emit a control")

    def test_a_card_tells_the_pointer_it_is_not_a_control(self) -> None:
        body = _exact_rule(self.css, f".{self.surfaces.INFO_CARD_CLASS}")
        self.assertIn("cursor: default", body)

    def test_a_card_has_no_hover_state(self) -> None:
        """A surface that lights up under the cursor is claiming to be clickable."""
        hover = re.findall(
            rf"\.{re.escape(self.surfaces.INFO_CARD_CLASS)}[^{{,]*:hover", self.css
        )
        self.assertEqual(hover, [], f"info card grew a hover rule: {hover}")

    def test_a_card_is_a_different_shape_from_a_control(self) -> None:
        """Shape, so the distinction survives a greyscale screenshot.

        Measured: a Streamlit button is an 8px radius. The card has to be
        visibly flatter than that, not merely a different colour.
        """
        card = _exact_rule(self.css, f".{self.surfaces.INFO_CARD_CLASS}")
        radius = re.search(r"border-radius:\s*(\d+)px", card)
        self.assertIsNotNone(radius, "the card declares no radius")
        self.assertLess(
            int(radius.group(1)),
            8,
            "the info card is no flatter than a button, which is the "
            "confusion this module exists to remove",
        )

    def test_the_status_word_does_not_span_its_container(self) -> None:
        """A full-width tinted bar is what a button looks like."""
        body = _exact_rule(
            self.css,
            f".{self.surfaces.INFO_CARD_CLASS} .{self.surfaces.INFO_CARD_CLASS}-status",
        )
        self.assertIn("inline-block", body)
        self.assertIn("cursor: default", body)

    def test_every_card_class_in_the_app_is_passive(self) -> None:
        """The legacy tiles too, not only the ones this module draws.

        `.dataset-card` and `.pipeline-card` were 18px and 20px rounded
        boxes with a saturated 1.5px border and a drop shadow — louder than
        the controls beneath them.
        """
        app_css = APP_PATH.read_text(encoding="utf-8")
        for klass in (".dataset-card", ".pipeline-card", ".llm-output-card"):
            with self.subTest(card=klass):
                body = _exact_rule(app_css, klass)
                self.assertTrue(body, f"{klass} has no rule any more")
                self.assertIn("cursor: default", body, f"{klass} is not passive")
                self.assertNotIn("box-shadow", body, f"{klass} still floats")
                radius = re.search(r"border-radius:\s*(\d+)px", body)
                self.assertIsNotNone(radius, f"{klass} declares no radius")
                self.assertLess(
                    int(radius.group(1)), 8, f"{klass} is not flatter than a button"
                )

    def test_no_elevation_variable_survives(self) -> None:
        """Elevation says 'pressable'; its only users were read-outs."""
        app_css = APP_PATH.read_text(encoding="utf-8")
        self.assertNotIn(
            "box-shadow: var(--app-shadow)",
            app_css,
            "a read-out is floating above the page again",
        )


# --------------------------------------------------------------------------
# 4. the stylesheet ranks the three roles, through stable selectors
# --------------------------------------------------------------------------


class TestStylesheetRanksTheRoles(unittest.TestCase):
    def setUp(self) -> None:
        from batteryml_web.ui import surfaces

        self.surfaces = surfaces
        self.css = surfaces.SURFACE_CSS

    def test_every_role_is_styled(self) -> None:
        for testid in (
            self.surfaces.PRIMARY_TESTID,
            self.surfaces.SECONDARY_TESTID,
            self.surfaces.NAV_TESTID,
        ):
            with self.subTest(testid=testid):
                self.assertIn(
                    f'[data-testid="{testid}"]',
                    self.css,
                    f"{testid} has no rule, so it falls back to the default "
                    "appearance and the ranking collapses",
                )

    def test_only_primary_is_filled(self) -> None:
        """Filled-beats-outlined is the ranking. Two filled roles rank nothing."""
        primary = _role_rule(self.css, self.surfaces.PRIMARY_TESTID)
        self.assertRegex(primary, r"background:\s*color-mix|background:\s*CanvasText")
        secondary = _role_rule(self.css, self.surfaces.SECONDARY_TESTID)
        self.assertNotIn("background:", secondary)

    def test_primary_is_the_heaviest_text(self) -> None:
        weights = {}
        for name, testid in (
            ("primary", self.surfaces.PRIMARY_TESTID),
            ("secondary", self.surfaces.SECONDARY_TESTID),
            ("nav", self.surfaces.NAV_TESTID),
        ):
            body = _role_rule(self.css, testid)
            found = re.search(r"font-weight:\s*(\d+)", body)
            self.assertIsNotNone(found, f"{name} declares no weight")
            weights[name] = int(found.group(1))
        self.assertGreater(weights["primary"], weights["secondary"])
        self.assertGreaterEqual(weights["secondary"], weights["nav"])

    def test_navigation_is_quieter_than_a_secondary_action(self) -> None:
        """Third in the ranking: nav must not out-shout a real action."""
        def alpha(testid: str) -> float:
            body = _role_rule(self.css, testid)
            found = re.search(r"border:[^;]*currentColor\s+(\d+)%", body)
            self.assertIsNotNone(found, f"{testid} declares no border mix")
            return int(found.group(1))

        self.assertLess(
            alpha(self.surfaces.NAV_TESTID),
            alpha(self.surfaces.SECONDARY_TESTID),
            "navigation's border is as strong as a secondary action's",
        )

    def test_hover_never_lets_the_theme_accent_back_in(self) -> None:
        """Found in a browser, not by reading the sheet.

        Streamlit's own ``:hover`` repaints a button's label in the theme
        primary. Every fill and border here is mixed from ``currentColor``,
        which then resolved to that accent, so hovering a neutral button
        washed it red — measured as ``rgba(255, 75, 75, 0.1)`` on the
        navigation role and ``rgb(255, 0, 0)`` on the primary's border.
        Each hover rule must therefore pin the colour it mixes from.
        """
        for testid in (
            self.surfaces.SECONDARY_TESTID,
            self.surfaces.NAV_TESTID,
        ):
            with self.subTest(testid=testid):
                body = _exact_rule(
                    self.css, f'[data-testid="{testid}"]:hover:enabled'
                )
                self.assertIn(
                    "color: inherit",
                    body,
                    "this hover mixes from currentColor without pinning it, "
                    "so the theme accent leaks into the fill",
                )
        primary = _exact_rule(
            self.css, f'[data-testid="{self.surfaces.PRIMARY_TESTID}"]:hover:enabled'
        )
        self.assertIn(
            "border-color:",
            primary,
            "the primary's hover leaves its border to Streamlit, which "
            "paints it in the theme accent",
        )

    def test_no_accent_colour_is_hard_coded(self) -> None:
        """Neutral by construction: no literal colour belongs in a role rule."""
        for testid in (
            self.surfaces.PRIMARY_TESTID,
            self.surfaces.SECONDARY_TESTID,
            self.surfaces.NAV_TESTID,
        ):
            with self.subTest(testid=testid):
                body = _role_rule(self.css, testid)
                self.assertNotRegex(body, r"#[0-9a-fA-F]{3,8}\b")
                self.assertNotRegex(body, r"rgb\(")

    def test_each_role_offers_hover_feedback(self) -> None:
        for testid in (
            self.surfaces.PRIMARY_TESTID,
            self.surfaces.SECONDARY_TESTID,
            self.surfaces.NAV_TESTID,
        ):
            with self.subTest(testid=testid):
                self.assertIn(f'[data-testid="{testid}"]:hover', self.css)

    def test_keyboard_focus_is_visible_for_every_role(self) -> None:
        """The filled primary would otherwise swallow Streamlit's own ring."""
        for testid in (
            self.surfaces.PRIMARY_TESTID,
            self.surfaces.SECONDARY_TESTID,
            self.surfaces.NAV_TESTID,
        ):
            with self.subTest(testid=testid):
                self.assertIn(f'[data-testid="{testid}"]:focus-visible', self.css)
        self.assertIn("outline:", self.css)

    def test_disabled_is_marked_without_relying_on_colour(self) -> None:
        body = _rule_body(
            self.css, f'[data-testid="{self.surfaces.PRIMARY_TESTID}"]:disabled'
        )
        self.assertIn("dashed", body, "disabled differs by dimming alone")
        self.assertIn("not-allowed", body)

    def test_the_stylesheet_uses_no_generated_class_name(self) -> None:
        hit = _GENERATED_CLASS.search(self.css)
        self.assertIsNone(
            hit,
            f"surfaces.py targets a generated Streamlit class ({hit.group(0) if hit else ''}); "
            "those change between releases",
        )

    def test_the_stylesheet_hides_nothing(self) -> None:
        for forbidden in ("display: none", "display:none", "visibility: hidden"):
            self.assertNotIn(forbidden, self.css)


# --------------------------------------------------------------------------
# 5. the roles are actually applied, and applied sparingly
# --------------------------------------------------------------------------


def _role_calls() -> Dict[str, List[str]]:
    """Every role-carrying control in the UI, by role constant, as ``file:line``.

    Read from the ``type=`` keyword, which is where a role now lives.
    """
    found: Dict[str, List[str]] = {c: [] for c in _ROLE_CONSTANTS}
    found["SECONDARY_DEFAULT"] = []
    for path in [APP_PATH] + sorted(UI_DIR.glob("*.py")):
        if path.name == "surfaces.py":
            continue
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if not (isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr in
                    ("button", "download_button", "form_submit_button")):
                continue
            kwargs = {k.arg: ast.unparse(k.value) for k in node.keywords if k.arg}
            role = kwargs.get("type", "SECONDARY_DEFAULT")
            found.setdefault(role, []).append(f"{path.name}:{node.lineno}")
    return found


class TestTheRolesAreApplied(unittest.TestCase):
    """A vocabulary nothing uses would leave the pages exactly as they were."""

    def setUp(self) -> None:
        self.calls = _role_calls()

    def test_the_pages_actually_use_the_primary_role(self) -> None:
        self.assertGreaterEqual(
            len(self.calls["PRIMARY_ROLE"]),
            8,
            "almost nothing is marked primary; the ranking would not be "
            f"visible on most pages. Found: {self.calls['PRIMARY_ROLE']}",
        )

    def test_the_pages_actually_use_the_navigation_role(self) -> None:
        self.assertGreaterEqual(
            len(self.calls["NAV_ROLE"]),
            8,
            f"navigation is still styled as an action: {self.calls['NAV_ROLE']}",
        )

    def test_primary_stays_rare(self) -> None:
        """"Everything is primary" is the same failure with a darker palette."""
        total = sum(len(v) for v in self.calls.values())
        primary = len(self.calls["PRIMARY_ROLE"])
        self.assertLess(
            primary / max(total, 1),
            0.35,
            f"{primary} of {total} controls are primary; a page with several "
            "dominant actions ranks none of them",
        )

    def test_the_main_run_control_is_primary(self) -> None:
        """The one action the whole platform exists for."""
        self.assertIn(
            "Run BatteryML",
            _call_labels("PRIMARY_ROLE"),
            "the Run control is no longer the page's primary action",
        )

    def test_navigation_between_pages_is_navigation(self) -> None:
        marked = _call_labels("NAV_ROLE")
        for label in ("Go to Run Model", "Go to Home", "Review Results →"):
            with self.subTest(label=label):
                self.assertIn(
                    label, marked, f"{label!r} is not marked as navigation"
                )

    def test_no_page_move_is_still_dressed_as_an_action(self) -> None:
        """A "Go to …" left as a plain button outranks the real action."""
        stragglers = [
            label
            for label in _call_labels("PRIMARY_ROLE")
            + _call_labels("SECONDARY_ROLE")
            if label.startswith(("Go to ", "Go Prepare"))
        ]
        self.assertEqual(
            stragglers, [], f"these moves are styled as actions: {stragglers}"
        )


class TestNavigationStillWorks(unittest.TestCase):
    """Affordance only: the wiring behind each control is untouched."""

    def test_the_sidebar_entry_still_sets_the_step(self) -> None:
        source = (UI_DIR / "workflow_sidebar.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        fn = next(
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.FunctionDef) and n.name == "_nav_button"
        )
        body = ast.unparse(fn)
        self.assertIn("type=NAV_ROLE", body, "the sidebar entry lost its role")
        self.assertIn("set_current_step(step['key'])", body)

    def test_the_current_entry_is_still_text_rather_than_a_control(self) -> None:
        """Selected location is marked by weight, not by a pressable box."""
        source = (UI_DIR / "workflow_sidebar.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        fn = next(
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.FunctionDef) and n.name == "_nav_button"
        )
        body = ast.unparse(fn)
        self.assertIn("st.markdown(f'**{label}**')", body)


class TestSidebarSpacingIsUntouched(unittest.TestCase):
    """This pass is affordance. The tuned sidebar rhythm must not move."""

    def setUp(self) -> None:
        from batteryml_web.ui import styles

        self.css = styles.SIDEBAR_COMPACT_CSS

    def test_the_block_gap_is_unchanged(self) -> None:
        found = re.search(
            r'stVerticalBlock"\]\s*\{[^}]*gap:\s*([0-9.]+)rem', self.css
        )
        self.assertIsNotNone(found)
        self.assertEqual(found.group(1), "0.7", "the sidebar rhythm moved")

    def test_the_brand_pin_is_unchanged(self) -> None:
        body = _rule_body(self.css, ":has(> h3)")
        self.assertIn("margin-bottom: 0", body)

    def test_the_group_margins_are_unchanged(self) -> None:
        for key, expected in (
            ("st-key-sb_protocol", "0.7rem"),
            ("st-key-sb_pager", "1.4rem"),
        ):
            with self.subTest(key=key):
                body = _rule_body(self.css, key)
                self.assertIn(f"margin-top: {expected}", body)

    def test_the_new_rules_add_no_sidebar_spacing(self) -> None:
        """surfaces.py may restyle the sidebar's look, never its metrics."""
        from batteryml_web.ui import surfaces

        for block in re.finditer(r"([^{}]+)\{([^{}]*)\}", surfaces.SURFACE_CSS):
            selector, body = block.group(1), block.group(2)
            if "stSidebar" not in selector:
                continue
            for prop in ("padding", "margin", "height", "gap", "line-height"):
                with self.subTest(selector=selector.strip()[:60], prop=prop):
                    self.assertNotRegex(
                        body,
                        rf"(^|[;\s]){prop}\s*:",
                        f"a sidebar rule in surfaces.py sets {prop}; spacing "
                        "belongs to ui/styles.py",
                    )


class TestDisclosuresReadAsDisclosures(unittest.TestCase):
    """"Show technical details" was a white 1px 8px box — a button's shape."""

    def setUp(self) -> None:
        from batteryml_web.ui import surfaces

        self.css = surfaces.SURFACE_CSS

    def test_the_expander_is_flatter_than_a_control(self) -> None:
        body = _exact_rule(
            self.css,
            '[data-testid="stMain"] [data-testid="stExpander"] details',
        )
        self.assertTrue(body, "no expander rule")
        radius = re.search(r"border-radius:\s*(\d+)px", body)
        self.assertIsNotNone(radius)
        self.assertLess(int(radius.group(1)), 8)

    def test_the_expander_rule_stays_out_of_the_sidebar(self) -> None:
        """The sidebar's Settings disclosure is spaced by ui/styles.py."""
        for block in re.finditer(r"([^{}]+)\{([^{}]*)\}", self.css):
            selector = block.group(1)
            if "stExpander" in selector:
                self.assertIn(
                    "stMain",
                    selector,
                    "an expander rule reaches the sidebar, where spacing is "
                    "already tuned elsewhere",
                )

    def test_the_summary_stays_clickable(self) -> None:
        body = _exact_rule(
            self.css,
            '[data-testid="stMain"] [data-testid="stExpander"] summary',
        )
        self.assertIn("cursor: pointer", body)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
