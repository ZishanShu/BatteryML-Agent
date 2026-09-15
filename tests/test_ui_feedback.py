"""Transient success feedback: shown once per event, never on a rerun.

The deduplication logic is pure (:func:`should_notify` takes the mapping it
records into), so these tests need no Streamlit runtime. The Streamlit-bound
half is checked in `tests/test_agent_web_ui.py`.

Run from the repository root:

    conda run -n batteryml python -m unittest tests.test_ui_feedback -v
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _streamlit_available() -> bool:
    try:
        import streamlit  # noqa: F401
    except Exception:
        return False
    return True


STREAMLIT_AVAILABLE = _streamlit_available()

FEEDBACK_PATH = REPO_ROOT / "batteryml_web" / "ui" / "feedback.py"


@unittest.skipUnless(STREAMLIT_AVAILABLE, "streamlit is not importable in this environment")
class TestDeduplication(unittest.TestCase):
    def setUp(self) -> None:
        from batteryml_web.ui import feedback

        self.feedback = feedback

    def test_first_occurrence_notifies_and_reruns_do_not(self) -> None:
        seen: dict[str, str] = {}
        self.assertTrue(self.feedback.should_notify(seen, "run_completed", "sig-1"))
        for _ in range(10):  # ten Streamlit reruns
            self.assertFalse(self.feedback.should_notify(seen, "run_completed", "sig-1"))

    def test_a_new_occurrence_notifies_again(self) -> None:
        seen: dict[str, str] = {}
        self.feedback.should_notify(seen, "run_completed", "sig-1")
        self.assertTrue(self.feedback.should_notify(seen, "run_completed", "sig-2"))
        self.assertFalse(self.feedback.should_notify(seen, "run_completed", "sig-2"))

    def test_events_are_tracked_independently(self) -> None:
        seen: dict[str, str] = {}
        self.assertTrue(self.feedback.should_notify(seen, "plan_ready", "x"))
        self.assertTrue(self.feedback.should_notify(seen, "run_completed", "x"))

    def test_empty_event_key_is_a_programming_error(self) -> None:
        with self.assertRaises(ValueError):
            self.feedback.should_notify({}, "  ", "sig")

    def test_reset_lets_the_same_occurrence_report_again(self) -> None:
        seen: dict[str, str] = {}
        self.feedback.should_notify(seen, "run_completed", "sig-1")
        seen.pop("run_completed", None)
        self.assertTrue(self.feedback.should_notify(seen, "run_completed", "sig-1"))

    def test_plan_signature_ignores_device_and_smoke(self) -> None:
        """Fiddling with a device or the smoke box must not toast."""
        base = {
            "dataset": "MATR",
            "task": "RUL prediction",
            "model": "batlinet",
            "config_path": "configs/external/batlinet/matr_1.yaml",
            "can_execute": True,
            "epochs": None,
            "command": ["python", "x", "--device", "cpu"],
        }
        changed_device = dict(base, command=["python", "x", "--device", "cuda:0"])
        changed_smoke = dict(base, epochs=1)
        self.assertEqual(
            self.feedback.plan_signature(base), self.feedback.plan_signature(changed_device)
        )
        self.assertEqual(
            self.feedback.plan_signature(base), self.feedback.plan_signature(changed_smoke)
        )

    def test_plan_signature_changes_with_the_subject(self) -> None:
        base = {
            "dataset": "MATR",
            "task": "RUL prediction",
            "model": "batlinet",
            "config_path": "a.yaml",
            "can_execute": True,
        }
        for field, value in (
            ("model", "cptransformer"),
            ("dataset", "HUST"),
            ("task", "SOH estimation"),
            ("config_path", "b.yaml"),
            ("can_execute", False),
        ):
            with self.subTest(field=field):
                self.assertNotEqual(
                    self.feedback.plan_signature(base),
                    self.feedback.plan_signature(dict(base, **{field: value})),
                )

    def test_plan_signature_of_a_non_plan_is_empty(self) -> None:
        self.assertEqual(self.feedback.plan_signature(None), "")
        self.assertEqual(self.feedback.plan_signature("not a plan"), "")

    def test_unserializable_signatures_do_not_raise(self) -> None:
        self.assertIsInstance(self.feedback.signature_of(object()), str)


class TestFeedbackConvention(unittest.TestCase):
    """Source-level checks, so they run without Streamlit installed."""

    def setUp(self) -> None:
        self.source = FEEDBACK_PATH.read_text(encoding="utf-8")

    def test_the_helper_never_writes_a_persistent_success_box(self) -> None:
        self.assertNotIn("st.success(", self.source)
        self.assertIn("st.toast", self.source)

    def test_every_required_event_has_a_key(self) -> None:
        for name in (
            "EVENT_PROJECT_CREATED",
            "EVENT_QUICKSTART_READY",
            "EVENT_REQUEST_ANALYZED",
            "EVENT_PLAN_READY",
            "EVENT_RUN_COMPLETED",
            "EVENT_RESULT_LOADED",
            "EVENT_EXPLANATION_READY",
        ):
            self.assertIn(name, self.source)

    def test_the_app_reports_each_required_event(self) -> None:
        app = (REPO_ROOT / "batteryml_web" / "app.py").read_text(encoding="utf-8")
        agent = (
            REPO_ROOT / "batteryml_web" / "ui" / "agent_workflow.py"
        ).read_text(encoding="utf-8")
        combined = app + agent
        for name in (
            "EVENT_PROJECT_CREATED",
            "EVENT_QUICKSTART_READY",
            "EVENT_REQUEST_ANALYZED",
            "EVENT_PLAN_READY",
            "EVENT_RUN_COMPLETED",
            "EVENT_RESULT_LOADED",
            "EVENT_EXPLANATION_READY",
        ):
            self.assertIn(name, combined, f"{name} is never reported")

    def test_ordinary_successes_no_longer_use_a_green_box(self) -> None:
        app = (REPO_ROOT / "batteryml_web" / "app.py").read_text(encoding="utf-8")
        for gone in (
            'st.success("✅ Project created successfully")',
            'st.success("MATR demo loaded into the workspace settings.")',
            'st.success("Run completed successfully.")',
        ):
            self.assertNotIn(gone, app)

    def test_a_failure_is_never_rendered_as_a_success(self) -> None:
        app = (REPO_ROOT / "batteryml_web" / "app.py").read_text(encoding="utf-8")
        for line in app.splitlines():
            if "st.success(" in line and " else " in line:
                self.fail(f"a success box carries a failure branch: {line.strip()}")


class TestStreamlitClientConfig(unittest.TestCase):
    """C: end users must not be offered the developer toolbar."""

    def setUp(self) -> None:
        self.path = REPO_ROOT / ".streamlit" / "config.toml"

    def test_config_exists_and_sets_viewer_toolbar(self) -> None:
        self.assertTrue(self.path.is_file(), ".streamlit/config.toml is missing")
        text = self.path.read_text(encoding="utf-8")
        self.assertIn("[client]", text)
        self.assertIn('toolbarMode = "viewer"', text)

    def test_config_parses_as_toml_where_a_parser_exists(self) -> None:
        try:
            import tomllib
        except ModuleNotFoundError:  # pragma: no cover - Python < 3.11
            self.skipTest("tomllib is unavailable")
        data = tomllib.loads(self.path.read_text(encoding="utf-8"))
        self.assertEqual(data["client"]["toolbarMode"], "viewer")

    def test_no_javascript_keyboard_interception_was_added(self) -> None:
        app = (REPO_ROOT / "batteryml_web" / "app.py").read_text(encoding="utf-8")
        for forbidden in ("addEventListener", "keydown", "<script"):
            self.assertNotIn(forbidden, app)


if __name__ == "__main__":
    unittest.main(verbosity=2)
