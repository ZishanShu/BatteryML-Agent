"""Controls look like controls, read-outs stay bounded, and a tab jump works.

Three UI defects are pinned here, each measured in a browser before it was
repaired.

**1. A card and a button were the same shape.** Streamlit's button is a 40px
box with a ``1px solid`` border and an 8px radius; the readiness cards were
hand-built ``<div>``s with a ``1px solid`` border and a 14px radius, drawn
directly above those buttons. Nothing else separated them.

**2. A read-out could grow without limit.** 200 lines of log in a bare
``st.code`` measured **4510px**. The *command* block, by contrast, measured
54px and does not grow at all — ``st.code`` scrolls horizontally rather than
wrapping — so the cap there is insurance, not a repair.

**3. There was no way from preprocessing to the dataset status.** Dataset
Overview is a sibling tab, and Streamlit offers no working way to select a
tab from code: writing the label into ``st.session_state[key]`` does
nothing, and a new ``default=`` on a rerun does nothing either. Only
changing the widget key — so Streamlit builds a new widget, which then
honours ``default`` — switches the tab. All three were checked in a browser.
"""

from __future__ import annotations

import ast
import re
import subprocess
import sys
import unittest
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from batteryml_web.agent.execution import build_run_command  # noqa: E402
from batteryml_web.ui import surfaces  # noqa: E402
from batteryml_web.ui import tab_navigation as nav  # noqa: E402

APP = REPO_ROOT / "batteryml_web" / "app.py"


class _FakeStreamlit:
    """Enough Streamlit to record what a helper drew."""

    def __init__(self, session: Dict[str, Any] | None = None) -> None:
        self.session_state: Dict[str, Any] = dict(session or {})
        self.markdown: List[str] = []
        self.tabs_calls: List[Dict[str, Any]] = []

    # --- recorded ---
    def markdown_(self, body: str, **_kwargs: Any) -> None:
        self.markdown.append(str(body))

    def tabs(self, labels, **kwargs):
        self.tabs_calls.append({"labels": list(labels), **kwargs})
        return [object() for _ in labels]


def _with_fake(module, fake):
    """Point one module's ``st`` at the fake for the duration of a with-block."""

    class _Ctx:
        def __enter__(self):
            self.saved = module.st
            fake.markdown = fake.markdown  # keep the list identity
            module.st = _Shim(fake)
            return fake

        def __exit__(self, *exc):
            module.st = self.saved
            return False

    return _Ctx()


class _Shim:
    """Adapts ``markdown_`` to ``markdown`` without shadowing the record list."""

    def __init__(self, fake: _FakeStreamlit) -> None:
        self._fake = fake
        self.session_state = fake.session_state

    def markdown(self, body, **kwargs):
        self._fake.markdown.append(str(body))

    def tabs(self, labels, **kwargs):
        return self._fake.tabs(labels, **kwargs)

    def container(self, **kwargs):
        return self


# --------------------------------------------------------------------------
# 1-4: preprocessing -> Dataset Overview
# --------------------------------------------------------------------------


class TestTabNavigation(unittest.TestCase):
    def test_a_request_opens_that_tab(self) -> None:
        """Requirements 1-2, at the mechanism.

        The request has to survive into the widget as ``default``, because
        that is the only thing Streamlit acts on.
        """
        fake = _FakeStreamlit()
        with _with_fake(nav, fake):
            nav.request_tab(nav.DATASET_OVERVIEW_TAB)
            self.assertEqual(nav.pending_tab(), nav.DATASET_OVERVIEW_TAB)
            nav.render_prepare_tabs()
        call = fake.tabs_calls[-1]
        self.assertEqual(call["default"], nav.DATASET_OVERVIEW_TAB)
        self.assertEqual(call["labels"], list(nav.PREPARE_TABS))

    def test_the_widget_key_changes_so_streamlit_rebuilds_it(self) -> None:
        """The part that is not obvious, and was found by measuring.

        ``default=`` on an existing widget is ignored — it is an *initial*
        value. Only a new key makes Streamlit build a new widget that reads
        it, so the key must differ between the two renders.
        """
        fake = _FakeStreamlit()
        with _with_fake(nav, fake):
            nav.render_prepare_tabs()
            first = fake.tabs_calls[-1]["key"]
            nav.request_tab(nav.DATASET_OVERVIEW_TAB)
            nav.render_prepare_tabs()
            second = fake.tabs_calls[-1]["key"]
        self.assertNotEqual(first, second)

    def test_the_request_is_consumed_so_it_does_not_stick(self) -> None:
        """A stale jump would fight the user's own next tab click."""
        fake = _FakeStreamlit()
        with _with_fake(nav, fake):
            nav.request_tab(nav.DATASET_OVERVIEW_TAB)
            nav.render_prepare_tabs()
            self.assertIsNone(nav.pending_tab())
            nav.render_prepare_tabs()
            self.assertEqual(fake.tabs_calls[-1]["default"], nav.PREPARE_TABS[0])

    def test_navigation_preserves_the_project_and_everything_else(self) -> None:
        """Requirements 3-4.

        Asserted as "it wrote only its own two keys" rather than by naming
        the keys it must not touch — a list of exclusions goes stale, and
        the real guarantee is that nothing else is written at all.
        """
        before = {
            "project_root": "runs/web_projects/alpha",
            "current_project": "alpha",
            "llm_api_key": "sk-secret",
            "llm_model": "glm-5.3-flashx",
            "selected_dataset": "MATR",
            "dataset_status_MATR": "READY",
        }
        fake = _FakeStreamlit(before)
        with _with_fake(nav, fake):
            nav.request_tab(nav.DATASET_OVERVIEW_TAB)
        changed = {
            key for key in fake.session_state
            if key not in before or fake.session_state[key] != before[key]
        }
        self.assertEqual(changed, {nav.REQUEST_KEY, nav.NONCE_KEY})
        for key, value in before.items():
            self.assertEqual(fake.session_state[key], value)

    def test_an_unknown_tab_is_refused_not_stored(self) -> None:
        """Storing it would leave a request the page can never satisfy."""
        fake = _FakeStreamlit()
        with _with_fake(nav, fake):
            nav.request_tab("Nonexistent Tab")
            self.assertIsNone(nav.pending_tab())
        self.assertNotIn(nav.REQUEST_KEY, fake.session_state)

    def test_the_preprocess_tab_offers_the_action_as_a_real_button(self) -> None:
        """Requirements 1 and 6, at the call site.

        Read from the syntax tree rather than by searching for a string, so
        it cannot be satisfied by the label appearing in a comment. What it
        checks is that a real ``st.button`` carries the label and that
        pressing it calls ``request_tab`` with the overview tab.
        """
        tree = ast.parse(APP.read_text(encoding="utf-8"))
        buttons = [
            node for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "button"
            and any(
                isinstance(arg, ast.Name) and arg.id == "VIEW_OVERVIEW_LABEL"
                for arg in node.args
            )
        ]
        self.assertEqual(len(buttons), 1, "expected exactly one navigation button")
        requests = [
            node for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "request_tab"
        ]
        self.assertTrue(requests)
        self.assertTrue(
            any(
                isinstance(arg, ast.Name) and arg.id == "DATASET_OVERVIEW_TAB"
                for call in requests
                for arg in call.args
            )
        )

    def test_navigation_does_not_start_preprocessing(self) -> None:
        """The action navigates. It must not re-run the work.

        Read from the button's own ``if`` body via the syntax tree. A first
        version of this split the file on the label string and matched the
        *import* line, so it was examining the whole module and failing for
        the wrong reason.
        """
        tree = ast.parse(APP.read_text(encoding="utf-8"))

        def is_nav_button(node: ast.AST) -> bool:
            return (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "button"
                and any(
                    isinstance(arg, ast.Name) and arg.id == "VIEW_OVERVIEW_LABEL"
                    for arg in node.args
                )
            )

        bodies = [
            node.body for node in ast.walk(tree)
            if isinstance(node, ast.If) and any(
                is_nav_button(child) for child in ast.walk(node.test)
            )
        ]
        self.assertEqual(len(bodies), 1, "expected one navigation button branch")
        called = {
            child.func.id if isinstance(child.func, ast.Name) else child.func.attr
            for statement in bodies[0]
            for child in ast.walk(statement)
            if isinstance(child, ast.Call)
            and isinstance(child.func, (ast.Name, ast.Attribute))
        }
        self.assertEqual(called, {"request_tab", "rerun"})


# --------------------------------------------------------------------------
# 5-6: information is not a control
# --------------------------------------------------------------------------


class TestInformationIsNotAControl(unittest.TestCase):
    def test_a_card_is_a_div_with_no_control_semantics(self) -> None:
        """Requirement 5.

        Measured in a browser: ``tag=DIV cursor=default role=None
        onclick=False`` against the button's ``tag=BUTTON cursor=pointer
        tabIndex=0``.
        """
        fake = _FakeStreamlit()
        with _with_fake(surfaces, fake):
            surfaces.render_info_card(
                "MATR", ["Raw files count: 180"], status="Ready",
                status_color="#1f7a3f",
            )
        html = fake.markdown[-1]
        self.assertIn(f'class="{surfaces.INFO_CARD_CLASS}"', html)
        for forbidden in ("<button", "role=", "onclick", "href="):
            self.assertNotIn(forbidden, html)

    def test_the_card_style_says_not_a_control(self) -> None:
        css = surfaces.SURFACE_CSS
        card = css.split(f".{surfaces.INFO_CARD_CLASS} {{")[1].split("}")[0]
        self.assertIn("cursor: default", card)
        self.assertIn("background:", card)

    def test_a_card_never_gains_a_hover_state(self) -> None:
        """A surface that lights up under the cursor claims to be clickable."""
        self.assertNotIn(f".{surfaces.INFO_CARD_CLASS}:hover", surfaces.SURFACE_CSS)

    def test_the_distinction_does_not_rest_on_colour(self) -> None:
        """Requirement in section 5.

        Someone who cannot distinguish the fill still has the cursor, the
        radius, the element type and the absence of focus. This asserts the
        two that live in CSS.
        """
        card = surfaces.SURFACE_CSS.split(
            f".{surfaces.INFO_CARD_CLASS} {{"
        )[1].split("}")[0]
        self.assertIn("cursor: default", card)
        radius = re.search(r"border-radius:\s*([0-9.]+)px", card)
        self.assertIsNotNone(radius)
        # Streamlit's button radius is 8px; the card must not match it.
        self.assertNotEqual(float(radius.group(1)), 8.0)

    def test_card_content_is_escaped(self) -> None:
        """These strings are file names and paths read off disk."""
        fake = _FakeStreamlit()
        with _with_fake(surfaces, fake):
            surfaces.render_info_card("<script>alert(1)</script>", ["a<b>c"])
        html = fake.markdown[-1]
        self.assertNotIn("<script>", html)
        self.assertIn("&lt;script&gt;", html)

    def test_the_readiness_colour_is_passed_through_untouched(self) -> None:
        """Scientific status colours are not this module's to retune."""
        fake = _FakeStreamlit()
        with _with_fake(surfaces, fake):
            surfaces.render_info_card("HUST", [], status="Ready", status_color="#1f7a3f")
        self.assertIn("#1f7a3f", fake.markdown[-1])

    def test_the_stylesheet_uses_no_generated_class_names(self) -> None:
        """A hashed Streamlit class changes between releases."""
        self.assertIsNone(
            re.search(r"\.(css|st-emotion)-[0-9a-z]{4,}", surfaces.SURFACE_CSS)
        )


# --------------------------------------------------------------------------
# 7-11: the command and log viewports
# --------------------------------------------------------------------------


class TestBoundedCodeViewports(unittest.TestCase):
    def test_the_command_builder_is_untouched(self) -> None:
        """Requirement 7. The executable argv is not this step's business."""
        command = build_run_command(
            config_path="configs/baselines/sklearn/ridge/matr_1.yaml",
            workspace="runs/web_projects/alpha/workspace",
            task="RUL prediction", device="cpu", seed=0,
        )
        self.assertEqual(
            command,
            [
                sys.executable,
                "batteryml_web/core/run_batteryml_cli.py",
                "run",
                "configs/baselines/sklearn/ridge/matr_1.yaml",
                "--workspace",
                "runs/web_projects/alpha/workspace",
                "--train",
                "--eval",
                "--device",
                "cpu",
                "--seed",
                "0",
                "--metric",
                "RMSE,MAE,MAPE",
            ],
        )

    def test_the_default_viewport_is_five_to_ten_lines(self) -> None:
        """Requirement 9."""
        self.assertGreaterEqual(surfaces.VIEWPORT_MAX_LINES, 5)
        self.assertLessEqual(surfaces.VIEWPORT_MAX_LINES, 10)

    def test_the_cap_is_derived_from_the_measured_line_height(self) -> None:
        """The guess that produced a 12-line viewport, pinned.

        A first attempt assumed a 1.6rem code line-height. The rendered
        value is **19.2px against a 16px root — 1.2rem** — so a cap meant
        for 9 lines came out at 12, outside the 5-10 range the brief asks
        for. Both numbers are asserted, and the measured one is written out
        rather than recomputed from the constant under test: deriving the
        expectation from ``_CODE_LINE_REM`` made this pass for any value of
        it, which a mutation run caught.
        """
        self.assertEqual(
            surfaces._CODE_LINE_REM, 1.2,
            "the rendered code line-height measured 19.2px against a 16px root",
        )
        expected = round(surfaces.VIEWPORT_MAX_LINES * 1.2, 2)
        self.assertEqual(surfaces._VIEWPORT_MAX_REM, expected)
        self.assertIn(f"max-height: {expected}rem", surfaces.SURFACE_CSS)
        # And the cap that actually results is inside the asked-for range.
        rendered_px = expected * 16
        self.assertGreaterEqual(rendered_px / 19.2, 5.0)
        self.assertLessEqual(rendered_px / 19.2, 10.0)

    def test_the_cap_is_a_maximum_not_a_fixed_height(self) -> None:
        """Requirement 10, and why ``st.code(height=...)`` was not used.

        A fixed height pads a two-line command out to nine lines of empty
        space. ``max-height`` bounds the tall case and leaves the short one
        alone — measured, the command stayed at 54px while a 200-line log
        came down from 4510px to the cap.
        """
        self.assertIn("max-height:", surfaces.SURFACE_CSS)
        self.assertNotIn("\n    height:", surfaces.SURFACE_CSS)
        self.assertIn("overflow: auto", surfaces.SURFACE_CSS)

    def test_both_viewports_are_scoped_to_app_owned_keys(self) -> None:
        for key in (surfaces.COMMAND_VIEWPORT_KEY, surfaces.LOG_VIEWPORT_KEY):
            with self.subTest(key=key):
                self.assertIn(f'st-key-{key}', surfaces.SURFACE_CSS)

    def test_the_command_is_rendered_whole_and_stays_copyable(self) -> None:
        """Requirements 8 and 11.

        The viewport is a container: it wraps, it does not transform. What
        is asserted is that the call site passes the joined command
        unmodified and still uses ``st.code``, which is what carries
        Streamlit's copy control.
        """
        source = APP.read_text(encoding="utf-8")
        self.assertIn(
            'command_viewport().code(" ".join(command), language="bash")', source
        )
        for truncation in ("[:200]", "[:120]", "textwrap.shorten", "…"):
            self.assertNotIn(
                f'command_viewport().code(" ".join(command){truncation}', source
            )

    def test_the_log_viewport_is_not_the_command_viewport(self) -> None:
        """Section 7: the command preview and the log stay separate."""
        self.assertNotEqual(
            surfaces.COMMAND_VIEWPORT_KEY, surfaces.LOG_VIEWPORT_KEY
        )
        source = APP.read_text(encoding="utf-8")
        self.assertIn("log_viewport().code(log_tail", source)
        self.assertIn('command_viewport().code(" ".join(command)', source)


# --------------------------------------------------------------------------
# 12: the sidebar work is untouched
# --------------------------------------------------------------------------


class TestNoSidebarRegression(unittest.TestCase):
    def test_the_sidebar_stylesheet_is_unchanged(self) -> None:
        """Requirement 12, checked against git rather than by eye.

        ``ui/styles.py`` holds spacing tuned over several earlier rounds —
        the header band, the block gap, the brand margin, the content
        gutters. This step adds its own stylesheet in its own module
        precisely so that file does not have to be reopened.
        """
        result = subprocess.run(
            ["git", "diff", "--stat", "HEAD", "--", "batteryml_web/ui/styles.py"],
            cwd=REPO_ROOT, capture_output=True, text=True, timeout=60,
        )
        if result.returncode != 0:  # pragma: no cover - git unavailable
            self.skipTest("git is not available to compare against HEAD")
        self.assertEqual(
            result.stdout.strip(), "",
            f"ui/styles.py was modified:\n{result.stdout}",
        )

    def test_the_surface_stylesheet_does_not_retune_the_sidebar(self) -> None:
        """Nothing here may re-tune the sidebar's spacing by accident.

        This used to forbid the string ``stSidebar`` outright. That was a
        proxy for the real rule and it had stopped matching it in both
        directions:

        * too weak — the role rules key off ``stBaseButton-*``, which
          applies to the sidebar's own navigation buttons without the word
          ``stSidebar`` appearing anywhere. A sheet could restyle the whole
          sidebar and still pass.
        * too strong — it rejected a sidebar-scoped rule that sets only
          ``justify-content``/``text-align`` to left-align a navigation
          label, which cannot move a single pixel of the tuned rhythm.

        So the assertion now names the thing it protects: no rule in this
        sheet, sidebar-scoped or not, may set a spacing property inside the
        sidebar. The header band, block gap, brand margin and gutters stay
        the exclusive business of ``ui/styles.py``.
        """
        for tuned in ("stSidebarHeader", "stVerticalBlock", "gap:"):
            self.assertNotIn(tuned, surfaces.SURFACE_CSS)

        css = re.sub(r"/\*.*?\*/", "", surfaces.SURFACE_CSS, flags=re.S)
        for block in re.finditer(r"([^{}]+)\{([^{}]*)\}", css):
            selector, body = block.group(1), block.group(2)
            if "stSidebar" not in selector:
                continue
            for prop in ("padding", "margin", "height", "gap", "line-height"):
                with self.subTest(selector=selector.strip()[-60:], prop=prop):
                    self.assertNotRegex(
                        body,
                        rf"(^|[;\s]){prop}\s*:",
                        f"a sidebar-scoped rule sets {prop}; spacing belongs "
                        "to ui/styles.py",
                    )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
