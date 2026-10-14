"""The page's own ← Prev / Next → step through its tabs (UX-13).

On Prepare Data and Run Model the bottom Prev/Next move one *tab*, and leave
the page only at its first or last tab. The tab strip tracks the open tab,
except on a run that executes a blocking job, where it is built untracked so
a tab click cannot stop the job.

    python -m unittest tests.test_tab_step_navigation -v
"""

from __future__ import annotations

import ast
import sys
import unittest
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from batteryml_web.ui import tab_navigation as nav  # noqa: E402
from batteryml_web.ui.scroll import SCROLL_SERIAL_KEY  # noqa: E402

APP = REPO_ROOT / "batteryml_web" / "app.py"


class _FakeStreamlit:
    def __init__(self, state: Dict[str, Any] | None = None) -> None:
        self.session_state: Dict[str, Any] = dict(state or {})
        self.tabs_calls: List[Dict[str, Any]] = []

    def tabs(self, labels, **kwargs):
        self.tabs_calls.append({"labels": list(labels), **kwargs})
        key = kwargs.get("key")
        if kwargs.get("on_change") is not None and key is not None:
            # A tracked strip holds its open label in session state.
            self.session_state.setdefault(key, kwargs.get("default") or labels[0])
        return [object() for _ in labels]


class _Fake:
    def __init__(self, fake: _FakeStreamlit) -> None:
        self.fake = fake

    def __enter__(self) -> _FakeStreamlit:
        self.saved = nav.st
        nav.st = self.fake
        return self.fake

    def __exit__(self, *exc) -> bool:
        nav.st = self.saved
        return False


class TestNeighbour(unittest.TestCase):
    def test_steps_inside_the_page_and_stops_at_its_ends(self) -> None:
        tabs = nav.RUN_MODEL_TABS
        self.assertEqual(nav.neighbour(tabs, "Configuration", +1), "Dataset Readiness")
        self.assertEqual(nav.neighbour(tabs, "Command & Run", -1), "Dataset Readiness")
        self.assertIsNone(nav.neighbour(tabs, "Configuration", -1))
        self.assertIsNone(nav.neighbour(tabs, "Logs / Debug", +1))

    def test_an_unknown_tab_counts_as_the_first(self) -> None:
        self.assertEqual(nav.neighbour(nav.PREPARE_TABS, None, +1), "Preprocess & Inspect")
        self.assertIsNone(nav.neighbour(nav.PREPARE_TABS, "nope", -1))


class TestTrackedStrip(unittest.TestCase):
    def test_an_ordinary_run_tracks_the_open_tab(self) -> None:
        with _Fake(_FakeStreamlit()) as fake:
            nav.render_page_tabs(nav.RUN_MODEL_PAGE)
            call = fake.tabs_calls[-1]
            self.assertEqual(call["on_change"], "rerun")
            self.assertEqual(call["labels"], list(nav.RUN_MODEL_TABS))
            fake.session_state[call["key"]] = "Command & Run"     # the user clicked a tab
            nav.render_page_tabs(nav.RUN_MODEL_PAGE)
            self.assertEqual(nav.active_tab(nav.RUN_MODEL_PAGE), "Command & Run")

    def test_a_busy_run_does_not_track_and_keeps_the_tab(self) -> None:
        """A tracked strip reruns on a click, and a rerun stops a blocking job."""
        with _Fake(_FakeStreamlit()) as fake:
            nav.request_tab("Command & Run", nav.RUN_MODEL_PAGE)
            nav.render_page_tabs(nav.RUN_MODEL_PAGE)
            tracked_key = fake.tabs_calls[-1]["key"]
            nav.render_page_tabs(nav.RUN_MODEL_PAGE, busy=True)
            busy = fake.tabs_calls[-1]
            self.assertNotIn("on_change", busy)
            self.assertNotEqual(busy["key"], tracked_key)
            self.assertEqual(busy["default"], "Command & Run")
            # The tracked widget was not drawn on the busy run, so Streamlit
            # drops its state; the next ordinary run reopens the same tab.
            fake.session_state.pop(tracked_key)
            nav.render_page_tabs(nav.RUN_MODEL_PAGE)
            self.assertEqual(fake.tabs_calls[-1]["default"], "Command & Run")

    def test_a_new_visit_opens_the_first_tab(self) -> None:
        with _Fake(_FakeStreamlit()) as fake:
            nav.request_tab("Logs / Debug", nav.RUN_MODEL_PAGE)
            nav.render_page_tabs(nav.RUN_MODEL_PAGE)
            self.assertEqual(nav.active_tab(nav.RUN_MODEL_PAGE), "Logs / Debug")
            fake.session_state[SCROLL_SERIAL_KEY] = 1               # another page was opened
            self.assertIsNone(nav.active_tab(nav.RUN_MODEL_PAGE))
            fake.session_state.pop(fake.tabs_calls[-1]["key"])        # its widget state is gone
            nav.render_page_tabs(nav.RUN_MODEL_PAGE)
            self.assertEqual(fake.tabs_calls[-1]["default"], "Configuration")

    def test_pages_do_not_share_requests(self) -> None:
        with _Fake(_FakeStreamlit()):
            nav.request_tab("Logs / Debug", nav.RUN_MODEL_PAGE)
            self.assertIsNone(nav.pending_tab(nav.PREPARE_PAGE))
            self.assertEqual(nav.pending_tab(nav.RUN_MODEL_PAGE), "Logs / Debug")

    def test_any_pressed_reads_only_a_true_button(self) -> None:
        self.assertTrue(nav.any_pressed({"run_batteryml_btn": True}, ("run_batteryml_btn",)))
        self.assertFalse(nav.any_pressed({"run_batteryml_btn": False}, ("run_batteryml_btn",)))
        self.assertFalse(nav.any_pressed({"other": True}, ("run_batteryml_btn",)))


class TestAppWiring(unittest.TestCase):
    """Read from app.py's syntax tree: which page passes what."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.tree = ast.parse(APP.read_text(encoding="utf-8"))
        cls.functions = {n.name: n for n in ast.walk(cls.tree) if isinstance(n, ast.FunctionDef)}
        cls.assigned = {}
        for node in cls.tree.body:
            if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                cls.assigned[node.target.id] = ast.literal_eval(node.value)

    def _calls(self, function: str, name: str) -> List[ast.Call]:
        return [
            n for n in ast.walk(self.functions[function])
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == name
        ]

    def test_both_tabbed_pages_hand_their_page_to_the_bottom_navigation(self) -> None:
        for function, page in (("render_prepare_data_page", "PREPARE_PAGE"), ("render_run_model_page", "RUN_MODEL_PAGE")):
            with self.subTest(function=function):
                args = [ast.unparse(c.args[0]) for c in self._calls(function, "render_bottom_step_navigation") if c.args]
                self.assertEqual(args, [page])

    def test_every_busy_key_is_a_real_button_and_the_run_buttons_are_listed(self) -> None:
        source = APP.read_text(encoding="utf-8")
        for name, required in (("PREPARE_BUSY_KEYS", "prep_run"), ("RUN_MODEL_BUSY_KEYS", "run_batteryml_btn")):
            keys = self.assigned[name]
            self.assertIn(required, keys)
            for key in keys:
                self.assertIn(f'key="{key}"', source, f"{name} lists {key!r}, which no widget uses")

    def test_run_model_builds_its_tabs_through_the_helper(self) -> None:
        self.assertEqual(len(self._calls("render_run_model_page", "render_page_tabs")), 1)
        literal_tabs = [
            n for n in ast.walk(self.functions["render_run_model_page"])
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "tabs"
        ]
        self.assertEqual(literal_tabs, [])


try:
    from streamlit.testing.v1 import AppTest
except ImportError:  # pragma: no cover - streamlit always ships it at the pinned version
    AppTest = None

if AppTest is not None:
    from tests.test_quick_experience_fixes import _RunModelHarness  # noqa: E402
else:  # pragma: no cover
    _RunModelHarness = object


@unittest.skipIf(AppTest is None, "streamlit.testing is not available")
class TestRunModelPrevNext(_RunModelHarness, unittest.TestCase):
    """The real page under AppTest. Run is never pressed (see the harness)."""

    def active(self, at, page: str = nav.RUN_MODEL_PAGE) -> str:
        stored = at.session_state[f"{page}_tab_active"]
        return stored["tab"]

    def click(self, at, key: str) -> None:
        at.button(key=key).click().run()
        self.assertHealthy(at)

    def test_next_walks_the_tabs_then_leaves_for_review_results(self) -> None:
        at = self.open_run_model(cuda_ok=False)
        self.assertEqual(self.active(at), "Configuration")
        for expected in ("Dataset Readiness", "Command & Run", "Logs / Debug"):
            self.click(at, "bottom_next_step")
            self.assertEqual(at.session_state["current_step"], "run_model")
            self.assertEqual(self.active(at), expected)
        self.click(at, "bottom_next_step")
        self.assertEqual(at.session_state["current_step"], "review_results")

    def test_prev_walks_back_then_opens_prepare_data_on_its_last_tab(self) -> None:
        at = self.open_run_model(cuda_ok=False)
        self.click(at, "bottom_next_step")
        self.assertEqual(self.active(at), "Dataset Readiness")
        self.click(at, "bottom_prev_step")
        self.assertEqual(self.active(at), "Configuration")
        self.assertEqual(at.session_state["current_step"], "run_model")
        self.click(at, "bottom_prev_step")
        self.assertEqual(at.session_state["current_step"], "prepare_data")
        self.assertEqual(self.active(at, nav.PREPARE_PAGE), nav.PREPARE_TABS[-1])

    def test_the_sidebar_pager_still_moves_whole_pages(self) -> None:
        at = self.open_run_model(cuda_ok=False)
        self.click(at, "sidebar_next_step")
        self.assertEqual(at.session_state["current_step"], "review_results")


if __name__ == "__main__":
    unittest.main()
