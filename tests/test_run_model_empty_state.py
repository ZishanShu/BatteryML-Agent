"""Run Model with no dataset registered renders instead of crashing.

The page built its four tabs regardless of whether any dataset existed.
With none, the dataset selectbox returns ``None`` and the configuration
summary did ``readiness[None]``:

    File "batteryml_web/app.py", line 1439, in render_run_model_page
        info = readiness[ds]
    KeyError: None

An ordinary empty state — a fresh checkout before any preprocessing —
presented as a crash, and one that also swallowed the page's own way out.

These tests *render the page*, they do not read its source. ``app.py`` is
imported (it is importable in bare mode) and its Streamlit surface is
replaced by the test double from ``tests/test_agent_web_ui.py``, so the
claim under test is that calling ``render_run_model_page()`` with zero
datasets completes — which is the thing that used to raise.

Run from the repository root:

    conda run -n batteryml python -m unittest tests.test_run_model_empty_state -v
"""

from __future__ import annotations

import sys
import unittest
import warnings
from contextlib import contextmanager
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

EMPTY_STATE_MESSAGE = (
    "No datasets are currently available. Prepare or upload a dataset first."
)


def _streamlit_available() -> bool:
    try:
        import streamlit  # noqa: F401
    except Exception:
        return False
    return True


STREAMLIT_AVAILABLE = _streamlit_available()


def _load_app():
    """Import ``batteryml_web.app`` with its bare-mode noise suppressed.

    Importing the page module also executes its module body, which draws the
    Home page against the real Streamlit in bare mode. That is loud but
    harmless: it writes nothing and only warns about the missing script-run
    context.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        import batteryml_web.app as app
    return app


@unittest.skipUnless(STREAMLIT_AVAILABLE, "streamlit is not importable in this environment")
class RunModelPageTestCase(unittest.TestCase):
    """Shared plumbing: render the real page against the test double."""

    @classmethod
    def setUpClass(cls) -> None:
        from tests.test_agent_web_ui import FakeStreamlit, _Col

        cls.app = _load_app()

        class _ContextCol(_Col):
            """A column usable as ``with col:`` — the page does that."""

            def __enter__(self):
                return self

            def __exit__(self, *exc_info):
                return False

        class _RunModelStreamlit(FakeStreamlit):
            """The double, plus the few widgets Run Model reaches for."""

            def columns(self, spec, **kwargs):
                count = spec if isinstance(spec, int) else len(spec)
                return [_ContextCol(self) for _ in range(count)]

            def container(self, **kwargs):
                """A reserved slot, recorded so its position can be checked.

                Run Model reserves one outside its button columns so the run's
                progress bar spans the content area instead of inheriting a
                column's width. A stand-in that draws into the page directly
                is enough here: nothing in this file asserts on where the
                progress *goes*, only that the page renders.
                """
                self.rec.setdefault("containers", []).append(len(self.rec["markdown"]))
                return _ContextCol(self)

            def number_input(self, label, value=0, **kwargs):
                self.rec.setdefault("number_inputs", []).append(str(label))
                key = kwargs.get("key")
                if key is not None and str(key) in self.session_state:
                    return self.session_state[str(key)]
                return value

            def divider(self, **kwargs):
                self.rec["markdown"].append("---")

        cls.Fake = _RunModelStreamlit

    @contextmanager
    def _rendered(self, datasets: list[str], session: dict | None = None):
        """Render Run Model with exactly ``datasets`` registered.

        ``_ui_dataset_options`` is the seam: it is the one function that
        answers "which datasets does this installation have?", so patching
        it reproduces the empty state without touching the filesystem and
        without depending on whether this machine happens to have data.
        """
        from batteryml_web.ui import feedback as ui_feedback
        from batteryml_web.ui import llm_state as ui_llm_state
        from batteryml_web.ui import settings_panel as ui_settings
        from batteryml_web.ui import workflow_sidebar as ui_sidebar

        fake = self.Fake(session if session is not None else {})
        modules = (self.app, ui_feedback, ui_llm_state, ui_settings, ui_sidebar)
        saved = [(module, module.st) for module in modules]
        original_options = self.app._ui_dataset_options
        try:
            for module, _ in saved:
                module.st = fake
            self.app._ui_dataset_options = lambda: list(datasets)
            self.app.render_run_model_page()
            yield fake
        finally:
            self.app._ui_dataset_options = original_options
            for module, original in saved:
                module.st = original


class TestZeroDatasets(RunModelPageTestCase):
    """The regression: zero datasets must render, not raise."""

    def test_the_page_renders_without_raising(self) -> None:
        with self._rendered([]) as fake:
            self.assertTrue(fake.rec["headings"], "the page rendered nothing at all")

    def test_readiness_is_never_indexed_with_none(self) -> None:
        """The exact defect, stated as the exception it used to raise."""
        try:
            with self._rendered([]):
                pass
        except KeyError as exc:  # pragma: no cover - this is the regression
            self.fail(f"Run Model still indexes readiness with a missing key: {exc!r}")

    def test_the_empty_state_is_explained(self) -> None:
        with self._rendered([]) as fake:
            self.assertIn(EMPTY_STATE_MESSAGE, fake.rec["infos"])

    def test_execution_controls_are_omitted(self) -> None:
        """No run may be configured or launched when there is nothing to run."""
        with self._rendered([]) as fake:
            labels = [button["label"] for button in fake.rec["buttons"]]
            for control in ("Run BatteryML", "Explain this config", "Suggest model/config"):
                self.assertNotIn(control, labels)
            self.assertEqual(fake.rec["tabs"], [], "the configuration tabs must be omitted")
            self.assertEqual(
                [box["label"] for box in fake.rec["selectboxes"]], [],
                "nothing is selectable with no dataset registered",
            )

    def test_no_dataset_is_fabricated(self) -> None:
        """An empty state is not an excuse to assume MATR."""
        session: dict = {}
        with self._rendered([], session=session):
            pass
        self.assertNotIn("selected_dataset", session)

    def test_a_pre_existing_selection_is_left_alone(self) -> None:
        session = {"selected_dataset": "MATR"}
        with self._rendered([], session=session):
            pass
        self.assertEqual(session["selected_dataset"], "MATR")

    def test_the_way_out_is_preserved(self) -> None:
        """Upload, project and navigation stay reachable."""
        with self._rendered([]) as fake:
            labels = [button["label"] for button in fake.rec["buttons"]]
            self.assertIn("Go to Prepare Data", labels)
            self.assertIn("Go to Home", labels)
            # The bottom step navigation the page always ends with.
            self.assertIn("← Prev", labels)
            self.assertIn("Next →", labels)
            # And the message names where an upload lives.
            body = " ".join(fake.rec["writes"] + fake.rec["markdown"])
            self.assertIn("Custom Dataset", body)


class TestDatasetsStillPresent(RunModelPageTestCase):
    """Normal behaviour is unchanged when a dataset exists."""

    def test_the_configuration_tabs_are_still_built(self) -> None:
        with self._rendered(["MATR"]) as fake:
            self.assertEqual(
                fake.rec["tabs"],
                [["Configuration", "Dataset Readiness", "Command & Run", "Logs / Debug"]],
            )

    def test_the_empty_state_notice_does_not_appear(self) -> None:
        with self._rendered(["MATR"]) as fake:
            self.assertNotIn(EMPTY_STATE_MESSAGE, fake.rec["infos"])
            labels = [button["label"] for button in fake.rec["buttons"]]
            self.assertNotIn("Go to Prepare Data", labels)

    def test_the_dataset_is_selectable_and_recorded(self) -> None:
        session: dict = {}
        with self._rendered(["MATR"], session=session) as fake:
            self.assertIn(
                "Dataset", [box["label"] for box in fake.rec["selectboxes"]]
            )
            self.assertEqual(session.get("selected_dataset"), "MATR")

    def test_the_run_control_is_still_offered(self) -> None:
        with self._rendered(["MATR"]) as fake:
            labels = [button["label"] for button in fake.rec["buttons"]]
            self.assertIn("Run BatteryML", labels)


if __name__ == "__main__":
    unittest.main(verbosity=2)
