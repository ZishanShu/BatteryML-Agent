"""One design system for every page: tokens, type scale, roles, controls.

Regression tests A1–A12 of the global UI/UX unification (A7, status that
cannot be pressed, is completed by the Run History tests). Everything here
reads the stylesheets and the page source, or renders a page with
``streamlit.testing``; no browser runs. The layout claims a stylesheet cannot
settle (A12: no horizontal page overflow) are measured in a browser by
``scripts/ui_audit/viewport_audit.py``, which these tests only keep honest.

    python -m unittest tests.test_design_system -v
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

from batteryml_web.ui import home_style, sidebar_style, styles, surfaces  # noqa: E402
from tests.test_quick_experience_fixes import AppTest, _PageHarness  # noqa: E402
from tests.test_site_style import (  # noqa: E402
    HOME, MENTOR_TOKENS, SITE, SURFACE, _body, _clean, _rules, resolve, tokens,
)

APP = REPO_ROOT / "batteryml_web" / "app.py"
AGENT = REPO_ROOT / "batteryml_web" / "ui" / "agent_workflow.py"
SIDEBAR_SHEETS = {
    "sidebar_theme": _clean(sidebar_style.SIDEBAR_THEME_CSS),
    "sidebar_compact": _clean(styles.SIDEBAR_COMPACT_CSS),
}
ALL_SHEETS = {"site": SITE, "home": HOME, "surface": SURFACE, **SIDEBAR_SHEETS}
MAIN_BUTTON = '[data-testid="stMain"] [data-testid="stButton"] > button'

#: The token families the design system must name (the task's list).
REQUIRED_TOKENS = (
    "--bml-primary", "--bml-primary-hover", "--bml-primary-muted", "--bml-background",
    "--bml-surface", "--bml-surface-secondary", "--bml-surface-selected", "--bml-border",
    "--bml-border-subtle", "--bml-text-primary", "--bml-text-secondary", "--bml-text-muted",
    "--bml-focus-ring", "--bml-success", "--bml-warning", "--bml-error",
    "--bml-font-mono", "--bml-fs-page-title", "--bml-fs-section-title", "--bml-fs-subsection-title",
    "--bml-fs-body", "--bml-fs-label", "--bml-fs-caption", "--bml-fs-metric",
    "--bml-page-max-width", "--bml-card-radius", "--bml-button-radius", "--bml-input-radius",
    "--bml-control-height", "--bml-section-gap", "--bml-component-gap", "--bml-card-padding",
    "--bml-state-hover", "--bml-state-selected", "--bml-state-disabled-bg", "--bml-state-loading",
    "--bml-state-completed", "--bml-state-failed",
)


def _rem(value: str) -> float:
    value = value.strip()
    if value.endswith("rem"):
        return float(value[:-3])
    if value.endswith("px"):
        return float(value[:-2]) / 16
    raise AssertionError(f"not a length: {value!r}")


def _decl(body: str, prop: str) -> str:
    found = re.findall(rf"(?m)^\s*{re.escape(prop)}\s*:\s*([^;]+);", body)
    if not found:
        raise AssertionError(f"{prop} not declared in:\n{body}")
    return found[-1].strip()


def _hex_outside_root(css: str) -> set:
    rest = "\n".join(body for sel, body in _rules(css) if sel.strip() != ":root")
    return {h.lower() for h in re.findall(r"#[0-9a-fA-F]{3,8}\b", rest)}


class TestA01SharedTokens(unittest.TestCase):
    def test_the_token_set_is_complete_and_built_on_the_mentor_palette(self) -> None:
        table = tokens()
        for token, value in MENTOR_TOKENS.items():
            self.assertEqual(table[token].lower(), value)
        for token in REQUIRED_TOKENS:
            with self.subTest(token=token):
                self.assertIn(token, table)
                resolve(f"var({token})")                       # defined, no cycle
        self.assertEqual(resolve("var(--bml-primary)"), MENTOR_TOKENS["--bml-blue"])
        self.assertEqual(resolve("var(--bml-text-primary)"), MENTOR_TOKENS["--bml-ink"])
        self.assertEqual(resolve("var(--bml-text-secondary)"), MENTOR_TOKENS["--bml-muted"])

    def test_rules_use_tokens_not_new_colours(self) -> None:
        """Outside ``:root`` the only literal colours are white and the mentor's two highlights."""
        self.assertLessEqual(_hex_outside_root(SITE), {"#ffffff", "#3a8aff", "#f4f8ff"})
        for selector, body in _rules(SITE):
            for name in re.findall(r"var\((--[\w-]+)\)", body):
                with self.subTest(selector=selector.strip()[:60], token=name):
                    self.assertIn(name, tokens())


class TestA02HeadingHierarchy(unittest.TestCase):
    PAGE_TITLES = {
        APP: ("render_prepare_data_page", "render_run_model_page", "render_review_results_page"),
        AGENT: ("render_agent_workflow",),
    }

    def test_every_workflow_page_has_one_page_title_of_the_same_kind(self) -> None:
        for path, names in self.PAGE_TITLES.items():
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for name in names:
                with self.subTest(page=name):
                    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
                    calls = [n for n in ast.walk(fn) if isinstance(n, ast.Call)
                             and isinstance(n.func, ast.Attribute) and n.func.attr in ("header", "title")]
                    self.assertEqual([c.func.attr for c in calls], ["header"])
        for path in (APP, AGENT):
            self.assertNotIn("st.title(", path.read_text(encoding="utf-8"))

    def test_the_scale_steps_down_page_section_subsection(self) -> None:
        page = resolve(_body(SITE, '[data-testid="stMain"] h2'))
        section = resolve(_body(SITE, '[data-testid="stMain"] h3'))
        subsection = resolve(_body(SITE, '[data-testid="stMain"] h4'))
        sizes = [_rem(_decl(b, "font-size")) for b in (page, section, subsection)]
        weights = [int(_decl(b, "font-weight")) for b in (page, section, subsection)]
        self.assertEqual(sizes, sorted(sizes, reverse=True))
        self.assertEqual(len(set(sizes)), 3)
        self.assertEqual(weights, sorted(weights, reverse=True))
        self.assertLessEqual(max(weights), 700, "no page title heavier than the mentor's Home headings")
        # The page-title rule covers both h1 and h2, except the Home hero.
        self.assertIn('[data-testid="stMain"] h1:not(.bml-home-title)',
                      [s.strip() for sel, _ in _rules(SITE) for s in sel.split(",")])
        # The subsection is the mentor's own component-heading size.
        self.assertEqual(sizes[2], _rem(_decl(_body(HOME, ".bml-process-heading h2"), "font-size")))


class TestA03OneTypeFamily(unittest.TestCase):
    def test_no_sheet_overrides_the_theme_family_except_for_code(self) -> None:
        for name, css in ALL_SHEETS.items():
            for selector, body in _rules(css):
                if "font-family" not in body:
                    continue
                with self.subTest(sheet=name, selector=selector.strip()[:60]):
                    self.assertEqual(_decl(body, "font-family"), "var(--bml-font-mono)")
                    self.assertTrue(all(p.strip().endswith(("code", "pre")) for p in selector.split(",")))
        self.assertIn("monospace", resolve("var(--bml-font-mono)"))


class TestA04SharedBackground(unittest.TestCase):
    def test_one_canvas_for_every_page(self) -> None:
        canvas = [sel for sel, body in _rules(SITE) if "background" in body
                  and sel.strip() == '[data-testid="stAppViewContainer"]']
        self.assertEqual(len(canvas), 1)
        for name in ("home", "surface", "sidebar_theme", "sidebar_compact"):
            with self.subTest(sheet=name):
                self.assertNotIn("stAppViewContainer", ALL_SHEETS[name])
        self.assertIn(resolve("var(--bml-background)"), resolve(_body(SITE, '[data-testid="stAppViewContainer"]')))


class TestA05ActionRoles(unittest.TestCase):
    def test_primary_secondary_navigation_and_interrupting_differ_in_more_than_colour(self) -> None:
        secondary = resolve(_body(SITE, MAIN_BUTTON))
        primary = resolve(_body(SITE, '[data-testid="stMain"] [data-testid="stBaseButton-primary"]'))
        nav = resolve(_body(SITE, '[data-testid="stMain"] [data-testid="stButton"] > '
                                  '[data-testid="stBaseButton-tertiary"]'))
        stop = resolve(_body(SITE, '[data-testid="stMain"] [class*="st-key-agent_cancel_"] '
                                   '[data-testid="stButton"] > button'))
        self.assertIn("linear-gradient", primary)                        # fill
        self.assertNotRegex(secondary, r"(?m)^\s*background\s*:")        # outline only
        self.assertIn("border-color: transparent", nav)                  # wash, no outline
        self.assertIn("background: #ffffff", stop)
        self.assertIn("rgba(180, 35, 24", stop)                          # red hairline, not a fill
        self.assertNotIn("gradient", stop)
        # The Stop control keeps a stable key the rule can reach.
        self.assertIn('key=f"agent_cancel_{run_id}"', AGENT.read_text(encoding="utf-8"))

    def test_settings_form_controls_take_the_same_roles(self) -> None:
        main_primary = resolve(_body(SITE, '[data-testid="stMain"] [data-testid="stBaseButton-primary"]'))
        for scope in ('[data-testid="stSidebar"] .st-key-sb_foot ', '[role="dialog"] '):
            with self.subTest(scope=scope):
                body = resolve(_body(SITE, scope + '[data-testid="stBaseButton-primary"]'))
                self.assertEqual(_decl(body, "background"), _decl(main_primary, "background"))
                self.assertEqual(_decl(body, "border-radius"),
                                 _decl(resolve(_body(SITE, MAIN_BUTTON)), "border-radius"))


class TestA06PassiveCards(unittest.TestCase):
    def test_an_info_card_is_a_div_with_no_pointer_behaviour(self) -> None:
        card = _body(SURFACE, ".bml-info-card")
        self.assertIn("cursor: default", card)
        for name, css in ALL_SHEETS.items():
            with self.subTest(sheet=name):
                self.assertNotRegex(css, r"bml-info-card[^{,]*:(hover|active)")
        source = Path(surfaces.__file__).read_text(encoding="utf-8")
        fn = source[source.index("def render_info_card"):source.index("def command_viewport")]
        self.assertNotIn("<button", fn)
        self.assertNotIn("onclick", fn.lower())
        self.assertNotIn("st.button", fn)


class TestA07StatusIsNotAControl(unittest.TestCase):
    def test_the_status_pill_is_read_not_pressed(self) -> None:
        pill = resolve(_body(SITE, '[data-testid="stMain"] .bml-status'))
        self.assertIn("cursor: default", pill)
        self.assertEqual(_decl(pill, "border-radius"), "999px")          # not a 12px control box
        self.assertNotRegex(pill, r"(?m)^\s*background\s*:")              # nothing to press
        self.assertNotRegex(SITE, r"bml-status[^{,]*:(hover|active|focus)")
        for state in ("completed", "failed", "cancelled", "running"):
            with self.subTest(state=state):
                self.assertTrue(_body(SITE, f'[data-testid="stMain"] .bml-status--{state}'))


class TestA08NativeDisclosures(unittest.TestCase):
    def test_no_sheet_takes_over_expand_and_collapse(self) -> None:
        forbidden = re.compile(r"(?m)^\s*(display|height|max-height|overflow|pointer-events|visibility)\s*:")
        for name, css in ALL_SHEETS.items():
            for selector, body in _rules(css):
                if not re.search(r"stExpander|(?<![\w-])(details|summary)\b", selector):
                    continue
                with self.subTest(sheet=name, selector=selector.strip()[:70]):
                    self.assertIsNone(forbidden.search(body))
        self.assertIn("cursor: pointer", _body(SURFACE, '[data-testid="stMain"] [data-testid="stExpander"] summary'))


class TestA09InputsShareOneStyle(unittest.TestCase):
    def test_main_column_and_settings_form_inputs_resolve_alike(self) -> None:
        main = resolve(_body(SITE, '[data-testid="stMain"] [data-testid="stSelectbox"] [role="group"]'))
        for selector in (
            '[data-testid="stMain"] [data-testid="stTextInput"] input',
            '[data-testid="stMain"] [data-testid="stNumberInput"] input',
            '[data-testid="stSidebar"] .st-key-sb_foot [data-testid="stTextInput"] input',
            '[data-testid="stSidebar"] .st-key-sb_foot [data-testid="stSelectbox"] [role="group"]',
            '[role="dialog"] [data-testid="stTextInput"] input',
            '[role="dialog"] [data-testid="stSelectbox"] [role="group"]',
        ):
            with self.subTest(selector=selector):
                body = resolve(_body(SITE, selector))
                self.assertEqual(_decl(body, "border-radius"), _decl(main, "border-radius"))
                self.assertEqual(_decl(body, "border-color"), _decl(main, "border-color"))
        self.assertEqual(_decl(main, "border-radius"), resolve("var(--bml-button-radius)"))


class TestA10MentorHomeKept(unittest.TestCase):
    #: The mentor's Home values (codex/ui-optimization), unchanged by the shared sheet.
    MENTOR_HOME = {
        ".bml-home-hero": {"border-radius": "24px", "padding": "2.2rem 2.4rem"},
        ".bml-home-title": {"font-size": "clamp(2.1rem, 4vw, 4rem)", "font-weight": "800"},
    }

    def test_home_components_keep_their_values(self) -> None:
        for selector, wanted in self.MENTOR_HOME.items():
            body = _body(HOME, selector)
            for prop, value in wanted.items():
                with self.subTest(selector=selector, prop=prop):
                    self.assertEqual(_decl(body, prop), value)

    def test_home_is_injected_after_the_site_sheet_so_it_wins_ties(self) -> None:
        source = APP.read_text(encoding="utf-8")
        tree = ast.parse(source)
        home = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "render_home_page")
        self.assertIn("render_home_style", ast.unparse(home))
        self.assertIn('<h1 class="bml-home-title">', source)
        for marker in ('class="bml-process-overview"', 'key=f"home_choice_{choice_key}"'):
            self.assertIn(marker, source)
        self.assertIn(".bml-start-card", HOME)


class TestA11NoGeneratedSelectors(unittest.TestCase):
    def test_every_sheet(self) -> None:
        generated = re.compile(r"\.(css|st-emotion)-[0-9a-z]{4,}|\.e[0-9a-z]{6,}\b")
        for name, css in ALL_SHEETS.items():
            with self.subTest(sheet=name):
                self.assertIsNone(generated.search(css))


class TestA12NoPageOverflowGuards(unittest.TestCase):
    """What the stylesheet contributes to "no horizontal page overflow".

    The overflow itself is measured in a browser at 1440, 1024 and 760px by
    ``scripts/ui_audit/viewport_audit.py``; these keep the static half true.
    """

    def test_width_is_bounded_and_long_values_wrap(self) -> None:
        container = resolve(_body(SITE, '[data-testid="stMainBlockContainer"]'))
        self.assertEqual(_decl(container, "max-width"), "1480px")
        self.assertIn("clamp(", _decl(container, "padding-left"))
        self.assertIn("overflow-wrap: anywhere", _body(SITE, '[data-testid="stMain"] code'))
        for name, css in ALL_SHEETS.items():
            for selector, body in _rules(css):
                for match in re.finditer(r"(?m)^\s*(min-width|width)\s*:\s*(\d+)px", body):
                    with self.subTest(sheet=name, selector=selector.strip()[:60]):
                        self.assertLessEqual(int(match.group(2)), 360, "a fixed width wider than a phone")

    def test_the_browser_audit_exists_and_measures_the_page(self) -> None:
        script = REPO_ROOT / "scripts" / "ui_audit" / "viewport_audit.py"
        text = script.read_text(encoding="utf-8")
        for needle in ("1440", "1024", "760", "scrollWidth", "clientWidth"):
            self.assertIn(needle, text)


@unittest.skipIf(AppTest is None, "streamlit.testing is not available")
class TestA01EveryPageRendersTheSheet(_PageHarness, unittest.TestCase):
    def test_every_page(self) -> None:
        for page in ("home", "agent_workflow", "prepare_data", "run_model", "review_results"):
            with self.subTest(page=page):
                at = self.app()
                at.session_state["current_step"] = page
                at.run()
                self.assertHealthy(at)
                self.assertTrue(any(home_style.SITE_STYLE_MARKER in m.value for m in at.markdown), page)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
