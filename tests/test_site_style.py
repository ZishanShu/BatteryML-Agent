"""One visual language on every page: the mentor's, from ``ui/home_style.py``.

The mentor's stylesheet used to be injected only by ``render_home_page``, so
Home was cobalt and rounded while every other page kept Streamlit's defaults
(measured in a browser before this change: neutral 8px buttons and grey text
elsewhere, 12px cobalt buttons on Home). These tests pin the split into a
site-wide sheet and Home's own components, and the interaction hierarchy the
shared sheet draws.

Run from the repository root::

    python -m unittest tests.test_site_style -v
"""

from __future__ import annotations

import ast
import re
import sys
import unittest
from pathlib import Path
from typing import Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from batteryml_web.ui import home_style, surfaces  # noqa: E402
from tests.test_quick_experience_fixes import AppTest, _PageHarness  # noqa: E402

APP = REPO_ROOT / "batteryml_web" / "app.py"


def _clean(css: str) -> str:
    """The rules alone: no comments, no ``<style>`` wrapper."""
    css = re.sub(r"/\*.*?\*/", "", css, flags=re.S)
    return re.sub(r"</?style>", "", css)


SITE = _clean(home_style.SITE_STYLE_CSS)
HOME = _clean(home_style.HOME_STYLE_CSS)
SURFACE = _clean(surfaces.SURFACE_CSS)
_GENERATED_CLASS = re.compile(r"\.(css|st-emotion)-[0-9a-z]{4,}")

#: The mentor's palette, as committed on codex/ui-optimization (f058d51).
MENTOR_TOKENS = {
    "--bml-ink": "#14224a",
    "--bml-muted": "#66728b",
    "--bml-blue": "#287bf5",
    "--bml-blue-deep": "#1557d6",
    "--bml-mint": "#31c79b",
}


def _rules(css: str) -> List[tuple]:
    return [(m.group(1).strip(), m.group(2)) for m in re.finditer(r"([^{}]+)\{([^{}]*)\}", css)]


def _body(css: str, selector: str) -> str:
    """Declarations of every rule whose selector list contains ``selector``."""
    found = [body for sel, body in _rules(css) if selector in [s.strip() for s in sel.split(",")]]
    return "\n".join(found)


ROLE = '[data-testid="stMain"] [data-testid="stBaseButton-{}"]'


class TestOneLanguageEverywhere(unittest.TestCase):
    def test_the_mentor_palette_is_kept(self) -> None:
        for token, value in MENTOR_TOKENS.items():
            with self.subTest(token=token):
                self.assertRegex(SITE, rf"{re.escape(token)}:\s*{re.escape(value)};")

    def test_the_theme_accent_is_the_mentor_blue(self) -> None:
        import tomllib

        config = tomllib.loads((REPO_ROOT / ".streamlit" / "config.toml").read_text(encoding="utf-8"))
        self.assertEqual(config["theme"]["primaryColor"].lower(), MENTOR_TOKENS["--bml-blue"])
        self.assertEqual(config["theme"]["base"], "light")

    def test_the_site_sheet_is_injected_for_every_page_and_home_keeps_its_own(self) -> None:
        tree = ast.parse(APP.read_text(encoding="utf-8"))
        module_calls = [n.value.func.id for n in tree.body
                        if isinstance(n, ast.Expr) and isinstance(n.value, ast.Call)
                        and isinstance(n.value.func, ast.Name)]
        self.assertIn("render_site_style", module_calls)
        self.assertLess(module_calls.index("render_surface_css"), module_calls.index("render_site_style"),
                        "the site sheet must refine the surface roles, so it comes after them")
        home = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "render_home_page")
        self.assertIn("render_home_style", ast.unparse(home))

    def test_home_components_stay_with_home(self) -> None:
        for klass in (".bml-home-hero", ".bml-start-card", ".bml-process-overview", "st-key-home_choice_"):
            with self.subTest(klass=klass):
                self.assertIn(klass, HOME)
                self.assertNotIn(klass, SITE)


class TestInteractionHierarchy(unittest.TestCase):
    def test_primary_secondary_and_navigation_look_different(self) -> None:
        shared = _body(SITE, '[data-testid="stMain"] [data-testid="stButton"] > button')
        primary = _body(SITE, ROLE.format("primary"))
        nav = _body(SITE, '[data-testid="stMain"] [data-testid="stButton"] > [data-testid="stBaseButton-tertiary"]')
        # Secondary is the shared rule as the mentor wrote it: outlined, no fill.
        self.assertIn("border-color: rgba(40,123,245,0.24)", shared)
        self.assertNotRegex(shared, r"(?m)^\s*background\s*:")
        self.assertEqual(_body(SITE, ROLE.format("secondary")), "", "secondary grew its own fill")
        self.assertIn("linear-gradient", primary)                       # filled
        # Navigation outranks the shared rule (one more attribute selector) and
        # drops its outline for a wash.
        self.assertIn("border-color: transparent", nav)
        self.assertIn("background: rgba(40,123,245,0.07)", nav)
        self.assertNotIn("gradient", nav)

    def test_only_a_control_that_can_act_reacts_to_the_pointer(self) -> None:
        for selector, _body_text in _rules(SITE):
            for part in selector.split(","):
                if ":hover" in part:
                    with self.subTest(selector=part.strip()):
                        self.assertIn(":enabled", part)

    def test_disabled_is_neutral_still_and_not_a_pale_action(self) -> None:
        body = _body(SITE, '[data-testid="stMain"] [data-testid="stButton"] > button:disabled')
        for declaration in ("cursor: not-allowed", "transform: none", "box-shadow: none",
                            "border-style: dashed"):
            self.assertIn(declaration, body)
        self.assertNotIn("gradient", body)
        self.assertNotIn("#eaf3ff", SITE, "the pale-blue disabled fill is back")
        self.assertIn("dashed", _body(SURFACE, '[data-testid="stBaseButton-primary"]:disabled'))

    def test_keyboard_focus_is_visible(self) -> None:
        self.assertIn('[data-testid="stButton"] > button:focus-visible', SITE)
        self.assertIn("outline: 2px solid", SITE)


class TestInformationStaysPassive(unittest.TestCase):
    def test_info_cards_change_palette_not_shape(self) -> None:
        body = _body(SITE, '[data-testid="stMain"] .bml-info-card')
        self.assertTrue(body)
        for prop in ("border-radius", "box-shadow", "cursor", "transform"):
            self.assertNotIn(prop, body)
        self.assertNotRegex(SITE, r"bml-info-card[^{,]*:hover")

    def test_app_read_out_tokens_take_colour_only(self) -> None:
        root = _body(SITE, ":root")
        for token in ("--app-border", "--app-border-soft", "--app-card-bg", "--app-muted"):
            self.assertIn(token, root)
        self.assertNotIn("--app-radius", root)
        self.assertNotIn("--app-shadow", root)

    def test_a_disclosure_is_flatter_than_a_control(self) -> None:
        expander = _body(SITE, '[data-testid="stMain"] [data-testid="stExpander"] details')
        button = _body(SITE, '[data-testid="stMain"] [data-testid="stButton"] > button')
        e = int(re.search(r"border-radius:\s*(\d+)px", expander).group(1))
        b = int(re.search(r"border-radius:\s*(\d+)px", button).group(1))
        self.assertLess(e, b)

    def test_status_messages_and_metric_values_keep_their_colour(self) -> None:
        self.assertIn("color: inherit", _body(SITE, '[data-testid="stMain"] [data-testid="stAlert"] p'))
        self.assertIn("color: var(--bml-ink)", _body(SITE, '[data-testid="stMain"] [data-testid="stMetricValue"]'))


class TestScopedAndStable(unittest.TestCase):
    def test_no_generated_class_and_nothing_hidden(self) -> None:
        for name, css in (("site", SITE), ("home", HOME)):
            with self.subTest(sheet=name):
                self.assertIsNone(_GENERATED_CLASS.search(css))
                for forbidden in ("display: none", "display:none", "visibility: hidden"):
                    self.assertNotIn(forbidden, css)

    def test_the_sidebar_is_left_to_its_own_sheet(self) -> None:
        self.assertNotIn("stSidebar", SITE)
        for selector, _ in _rules(SITE):
            for part in selector.split(","):
                part = part.strip()
                if part.startswith((":root", "[data-testid=\"stAppViewContainer\"]",
                                    "[data-testid=\"stMainBlockContainer\"]")):
                    continue
                with self.subTest(selector=part):
                    self.assertTrue(part.startswith('[data-testid="stMain"]'), part)


@unittest.skipIf(AppTest is None, "streamlit.testing is not available")
class TestEveryPageCarriesIt(_PageHarness, unittest.TestCase):
    PAGES = ("home", "agent_workflow", "prepare_data", "run_model", "review_results")

    def test_every_page_renders_with_the_site_sheet(self) -> None:
        for page in self.PAGES:
            with self.subTest(page=page):
                at = self.app()
                at.session_state["current_step"] = page
                at.run()
                self.assertHealthy(at)
                markdown = [m.value for m in at.markdown]
                self.assertTrue(any(home_style.SITE_STYLE_MARKER in m for m in markdown), page)
                has_home_sheet = any(".bml-home-hero" in m for m in markdown)
                self.assertEqual(has_home_sheet, page == "home", page)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
