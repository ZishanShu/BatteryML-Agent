"""Caught errors get our view, not Streamlit's traceback panel.

Streamlit's built-in exception panel shows a scientist a red traceback with
*Copy*, *Ask Google* and *Ask ChatGPT* — a developer's view, and one that
invites pasting an internal traceback into a search engine. For errors the
application can catch, it renders its own instead: a headline, one actionable
sentence, and the technical detail behind a disclosure in a copyable block.

Two boundaries these tests hold:

* **Streamlit's own panel is never touched.** No monkey-patching of private
  Streamlit internals, no JavaScript reaching into the host page to find its
  Copy button. A genuinely uncaught framework failure still shows the native
  panel, which is the correct outcome — the alternative breaks on the next
  release and hides errors when it does.
* **Secrets never reach the details.** They are redacted before being
  displayed — and the displayed block is the one that gets copied — through
  the same filter the run failure path already uses.

Run from the repository root:

    conda run -n batteryml python -m unittest tests.test_error_copy_ux -v
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

ERRORS_PATH = REPO_ROOT / "batteryml_web" / "ui" / "errors.py"
APP_PATH = REPO_ROOT / "batteryml_web" / "app.py"


def _strip_docstrings(source: str) -> str:
    """The module's executable source, with every docstring removed."""
    import ast

    tree = ast.parse(source)
    for node in ast.walk(tree):
        if not isinstance(
            node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
        ):
            continue
        body = node.body
        if (
            body
            and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)
        ):
            node.body = body[1:] or [ast.Pass()]
    return ast.unparse(tree)


def _streamlit_available() -> bool:
    try:
        import streamlit  # noqa: F401
    except Exception:
        return False
    return True


STREAMLIT_AVAILABLE = _streamlit_available()


class _Recorder:
    """The Streamlit surface ui/errors.py touches."""

    def __init__(self) -> None:
        self.session_state: dict = {}
        self.errors: list[str] = []
        self.writes: list[str] = []
        self.captions: list[str] = []
        self.codes: list[str] = []
        self.expanders: list[str] = []
        self.buttons: list[str] = []
        self.toasts: list[str] = []
        self.successes: list[str] = []
        self.clicked: set[str] = set()

    # --- surface ---
    def error(self, text, **kwargs):
        self.errors.append(str(text))

    def write(self, *args, **kwargs):
        self.writes.append(" ".join(str(a) for a in args))

    def caption(self, text, **kwargs):
        self.captions.append(str(text))

    def code(self, text, **kwargs):
        self.codes.append(str(text))

    def success(self, text, **kwargs):
        self.successes.append(str(text))

    def toast(self, message, **kwargs):
        self.toasts.append(str(message))

    def button(self, label, **kwargs):
        self.buttons.append(str(label))
        return str(label) in self.clicked

    def expander(self, label, **kwargs):
        self.expanders.append(str(label))
        return _Ctx()


class _Ctx:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


@unittest.skipUnless(STREAMLIT_AVAILABLE, "streamlit is not importable in this environment")
class ErrorUITestCase(unittest.TestCase):
    def setUp(self) -> None:
        from batteryml_web.ui import errors, feedback

        self.errors_mod = errors
        self.feedback = feedback
        self.st = _Recorder()
        self._saved = (errors.st, feedback.st)
        errors.st = self.st
        feedback.st = self.st

    def tearDown(self) -> None:
        self.errors_mod.st, self.feedback.st = self._saved


class TestTheFriendlyErrorView(ErrorUITestCase):
    def test_a_caught_error_gets_a_headline_and_an_action(self) -> None:
        self.errors_mod.render_error(
            "Try the step again, or move to another page.",
            key="t",
            error=ValueError("boom"),
        )
        self.assertIn(self.errors_mod.ERROR_HEADLINE, self.st.errors)
        self.assertIn("Try the step again, or move to another page.", self.st.writes)

    def test_the_technical_detail_is_behind_a_disclosure(self) -> None:
        self.errors_mod.render_error("do this", key="t", error=ValueError("boom"))
        self.assertIn("Show technical details", self.st.expanders)
        body = "\n".join(self.st.codes)
        self.assertIn("ValueError: boom", body)

    def test_the_traceback_is_not_the_headline(self) -> None:
        self.errors_mod.render_error("do this", key="t", error=ValueError("boom"))
        for shown in self.st.errors:
            self.assertNotIn("Traceback", shown)
            self.assertNotIn("ValueError", shown)

    def test_context_is_included_in_the_details(self) -> None:
        self.errors_mod.render_error(
            "do this", key="t", error=ValueError("boom"),
            context={"page": "agent_workflow", "project": "demo"},
        )
        body = "\n".join(self.st.codes)
        self.assertIn("page: agent_workflow", body)
        self.assertIn("project: demo", body)

    def test_no_permanent_success_box_is_written(self) -> None:
        self.errors_mod.render_error("do this", key="t", error=ValueError("boom"))
        self.assertEqual(self.st.successes, [])


class TestSecretsAreRedacted(ErrorUITestCase):
    SECRET = "sk-abcdefghijklmnopqrstuvwxyz0123456789ABCD"

    def test_details_do_not_carry_an_api_key(self) -> None:
        details = self.errors_mod.format_details(
            RuntimeError(f"auth failed for key {self.SECRET}"),
            context={"base_url": "https://api.example.com"},
        )
        self.assertNotIn(self.SECRET, details)

    def test_what_is_displayed_is_what_is_copied(self) -> None:
        """The clipboard must not receive more than the screen showed."""
        self.st.clicked.add(self.errors_mod.COPY_LABEL)
        self.errors_mod.render_error(
            "do this", key="t",
            error=RuntimeError(f"token {self.SECRET} rejected"),
        )
        displayed = "\n".join(self.st.codes)
        self.assertTrue(displayed, "no details block was rendered")
        self.assertNotIn(self.SECRET, displayed)

    def test_the_redaction_is_the_existing_shared_filter(self) -> None:
        source = ERRORS_PATH.read_text(encoding="utf-8")
        self.assertIn("from ..agent.failure_taxonomy import redact", source)


class TestCopyFeedbackIsTransient(ErrorUITestCase):
    """The control is honest about what it did, and says so transiently."""

    def test_pressing_it_reveals_the_text_and_toasts_once(self) -> None:
        self.st.clicked.add(self.errors_mod.COPY_LABEL)
        acted = self.errors_mod.render_copy_control("details here", key="k")
        self.assertTrue(acted)
        self.assertIn("details here", self.st.codes)
        self.assertEqual(self.st.toasts, [self.errors_mod.COPIED_MESSAGE])
        self.assertEqual(self.st.successes, [], "must not leave a success box")

    def test_a_rerun_without_a_press_does_not_toast(self) -> None:
        self.errors_mod.render_copy_control("details", key="k")
        self.assertEqual(self.st.toasts, [])

    def test_the_revealed_block_survives_a_rerun_without_re_announcing(self) -> None:
        self.st.clicked.add(self.errors_mod.COPY_LABEL)
        self.errors_mod.render_copy_control("details", key="k")
        self.st.clicked.clear()
        self.st.toasts.clear()
        self.errors_mod.render_copy_control("details", key="k")
        self.assertIn("details", self.st.codes)
        self.assertEqual(self.st.toasts, [], "a rerun re-announced a finished action")

    def test_it_points_at_the_working_copy_affordance(self) -> None:
        self.st.clicked.add(self.errors_mod.COPY_LABEL)
        self.errors_mod.render_copy_control("details", key="k")
        self.assertIn(self.errors_mod.COPY_HINT, self.st.captions)

    def test_it_does_not_claim_a_clipboard_write(self) -> None:
        """Measured in a browser: the component clipboard write did nothing.

        A control that toasts "copied" while the clipboard stays empty is
        worse than no control, so the message says the details are ready.
        """
        self.assertNotIn("copied", self.errors_mod.COPIED_MESSAGE.lower().split()[-1])
        self.assertIn("ready", self.errors_mod.COPIED_MESSAGE.lower())

    def test_nothing_is_shown_when_there_is_nothing_to_copy(self) -> None:
        self.assertFalse(self.errors_mod.render_copy_control("", key="k"))
        self.assertEqual(self.st.codes, [])


class TestTheErrorBoundary(ErrorUITestCase):
    def test_it_renders_our_view_instead_of_raising(self) -> None:
        with self.errors_mod.error_boundary(key="page", message="try again"):
            raise ValueError("page exploded")
        self.assertIn(self.errors_mod.ERROR_HEADLINE, self.st.errors)
        self.assertIn("try again", self.st.writes)

    def test_a_successful_block_renders_nothing_extra(self) -> None:
        with self.errors_mod.error_boundary(key="page", message="try again"):
            pass
        self.assertEqual(self.st.errors, [])

    def test_streamlit_control_flow_passes_through(self) -> None:
        """st.rerun() and st.stop() raise; swallowing them strands the page."""

        class RerunException(Exception):
            pass

        with self.assertRaises(RerunException):
            with self.errors_mod.error_boundary(key="page", message="try again"):
                raise RerunException("this is the framework working")
        self.assertEqual(self.st.errors, [])

    def test_keyboard_interrupt_passes_through(self) -> None:
        with self.assertRaises(KeyboardInterrupt):
            with self.errors_mod.error_boundary(key="page", message="try again"):
                raise KeyboardInterrupt


class TestStreamlitInternalsAreNotTouched(unittest.TestCase):
    """The approach the brief rules out, asserted as absent."""

    def setUp(self) -> None:
        self.source = ERRORS_PATH.read_text(encoding="utf-8")
        # The executable source. The module docstring *describes* Streamlit's
        # panel — "Copy, Ask Google and Ask ChatGPT" — to say what is being
        # replaced and why. That is documentation, and it must not read as
        # code that reaches for those buttons.
        self.code = _strip_docstrings(self.source)
        self.app = APP_PATH.read_text(encoding="utf-8")

    def test_no_streamlit_private_api_is_patched(self) -> None:
        for forbidden in (
            "streamlit.error_util",
            "streamlit.elements.exception",
            "_handle_uncaught_app_exception",
            "st.exception =",
            "monkeypatch",
            "setattr(st,",
        ):
            with self.subTest(forbidden=forbidden):
                self.assertFalse(
                    forbidden in self.code,
                    f"ui/errors.py patches Streamlit internals: {forbidden!r}",
                )

    def test_no_javascript_reaches_into_streamlits_own_dom(self) -> None:
        """Our component writes the clipboard; it does not hunt for buttons."""
        for forbidden in (
            "stException",
            "querySelector('button",
            'querySelector("button',
            "Ask Google",
            "Ask ChatGPT",
            "window.parent.document",
        ):
            with self.subTest(forbidden=forbidden):
                self.assertFalse(
                    forbidden in self.code,
                    f"the copy component reaches into Streamlit's DOM: {forbidden!r}",
                )
    def test_the_module_ships_no_javascript_at_all(self) -> None:
        """The strongest form of the boundary.

        A browser run showed a component-based clipboard write doing nothing
        while its button had already claimed success, so the JavaScript was
        removed rather than made more elaborate. With none left, there is no
        script that could reach the host page even by accident.
        """
        for forbidden in ("<script", "navigator.clipboard", "execCommand",
                          "components.html", "document."):
            with self.subTest(forbidden=forbidden):
                self.assertFalse(
                    forbidden in self.code,
                    f"ui/errors.py ships JavaScript again: {forbidden!r}",
                )

    def test_the_page_dispatch_runs_inside_the_boundary(self) -> None:
        self.assertIn("with error_boundary(", self.app)
        for page in (
            "render_home_page()",
            "render_prepare_data_page()",
            "render_run_model_page()",
            "render_agent_workflow()",
            "render_review_results_page()",
        ):
            with self.subTest(page=page):
                self.assertIn(f"        {page}", self.app)

    def test_the_boundary_message_does_not_promise_a_traceback(self) -> None:
        start = self.app.index("with error_boundary(")
        block = self.app[start:start + 700]
        self.assertIn("nothing was executed or written", block)
        self.assertNotIn("Traceback", block)


if __name__ == "__main__":
    unittest.main(verbosity=2)
