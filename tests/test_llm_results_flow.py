"""LLM settings from Results, and the one session-level LLM state (UI-14).

Regression tests C19–C31 of the global UI/UX unification, plus the existing
safety contracts D32–D36 re-checked against the code this change touched.

No provider is ever contacted. Every LLM entry point the pages use is a fake
— ``run_llm_prompt`` and ``explain_run_result`` (Review Results) and
``test_llm_connection`` (Connect) — and any outbound HTTP attempt fails the
test (``urllib.request.urlopen`` is replaced with a tripwire). Runs are
synthetic manifests in a temporary projects area; the API key is a made-up
string used only to prove it never leaves the session.

    python -m unittest tests.test_llm_results_flow -v
"""

from __future__ import annotations

import logging
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any, Dict, List
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from batteryml_web.core import run_manifest as rm  # noqa: E402
from batteryml_web.ui import llm_state, settings_panel  # noqa: E402
from tests.test_agent_web_ui import FakeStreamlit, _fake_streamlit, _Rerun  # noqa: E402
from tests.test_quick_experience_fixes import AppTest, _PageHarness  # noqa: E402
from tests.test_review_results_presentation import RUNS, _seed_project  # noqa: E402

APP = REPO_ROOT / "batteryml_web" / "app.py"

#: A made-up key with a shape a redactor must catch. Never a real credential.
SYNTHETIC_KEY = "sk-synthetic-NEVERWRITE-0123456789abcdef0123"
SETTINGS = {
    "llm_provider_preset": "Custom OpenAI-compatible",
    "llm_base_url": "https://llm.example.invalid/v1",
    "llm_api_key": SYNTHETIC_KEY,
    "llm_model": "synthetic-model",
}

DIALOG_KEY_FIELD = f"{settings_panel.DIALOG_FORM_PREFIX}_api_key"


def _no_network(*args, **kwargs):
    raise AssertionError("a test tried to open a network connection")


@unittest.skipIf(AppTest is None, "streamlit.testing is not available")
class _ResultsHarness(_PageHarness):
    """Review Results on a synthetic project, with every LLM call faked."""

    def setUp(self) -> None:
        super().setUp()
        from batteryml_web.agent import llm_client, result_agent
        from batteryml_web.core import project_context

        area = self.tmp / "web_projects"
        self.stack.enter_context(mock.patch.object(project_context, "PROJECTS_ROOT", str(area)))
        self.review_project = area / "review_project"
        self.dirs = _seed_project(self.review_project)

        self.prompts: List[str] = []
        self.explains: List[Any] = []
        self.connects: List[Dict[str, str]] = []
        self.prompt_error: Exception | None = None
        self.connect_ok = True

        def fake_prompt(prompt, **kwargs):
            self.prompts.append(str(prompt))
            if self.prompt_error is not None:
                raise self.prompt_error
            run_line = next((l for l in str(prompt).splitlines() if l.startswith("Run ID:")), "Run ID: ?")
            return f"SUMMARY-FOR::{run_line.split(':', 1)[1].strip()}"

        def fake_explain(*args, **kwargs):
            self.explains.append(args)
            return {"summary": "synthetic explanation", "observations": [], "caveats": []}

        def fake_connect(**kwargs):
            self.connects.append({k: v for k, v in kwargs.items() if k != "api_key"})
            return (True, "Connected (synthetic).") if self.connect_ok else (False, "HTTP Error 401: Unauthorized")

        self.stack.enter_context(mock.patch.object(llm_client, "run_llm_prompt", fake_prompt))
        self.stack.enter_context(mock.patch.object(result_agent, "explain_run_result", fake_explain))
        self.stack.enter_context(mock.patch.object(settings_panel, "test_llm_connection", fake_connect))
        self.stack.enter_context(mock.patch("urllib.request.urlopen", _no_network))

    def page(self, step: str = "review_results", **session: Any) -> "AppTest":
        at = AppTest.from_file(str(APP), default_timeout=300)
        at.session_state["project_root"] = str(self.review_project)
        at.session_state["project_name"] = self.review_project.name
        at.session_state["current_step"] = step
        for key, value in session.items():
            at.session_state[key] = value
        at.run()
        self.assertHealthy(at)
        return at

    @staticmethod
    def connected() -> Dict[str, Any]:
        return {**SETTINGS, "llm_status": llm_state.LLM_STATUS_CONNECTED}

    @staticmethod
    def dialog_open(at) -> bool:
        return any(t.key == DIALOG_KEY_FIELD for t in at.text_input)

    @staticmethod
    def texts(at) -> List[str]:
        out: List[str] = []
        for group in (at.markdown, at.caption, at.code, at.error, at.warning, at.info, at.success):
            out.extend(str(e.value) for e in group)
        out.extend(str(e.label) for e in at.expander)
        return out

    def click(self, at, key: str) -> None:
        at.button(key=key).click().run()
        self.assertHealthy(at)


class TestC19ToC23ResultsAndSettings(_ResultsHarness, unittest.TestCase):
    def test_c19_connected_results_never_ask_for_credentials(self) -> None:
        at = self.page(**self.connected())
        self.assertFalse(self.dialog_open(at))
        self.assertNotIn(settings_panel.REVEAL_REQUEST_KEY, at.session_state)
        self.assertTrue(any("LLM assistant status: Connected" in c.value for c in at.caption))
        # Summarize uses the connected LLM for the selected result, once.
        self.click(at, "review_summarize_results")
        self.assertEqual(len(self.prompts), 1)
        self.assertFalse(self.dialog_open(at))

    def test_c20_arriving_at_results_never_opens_settings(self) -> None:
        """Not configured, from the sidebar: results only, no dialog."""
        at = self.page()
        self.assertFalse(self.dialog_open(at))
        self.assertIn("Run history", [h.value for h in at.subheader])
        at.run()
        self.assertFalse(self.dialog_open(at))
        self.assertEqual(self.prompts, [])

    def test_c20_open_review_results_after_a_quick_run_does_not_open_settings(self) -> None:
        """Quick experience → Open Review Results is navigation, not an LLM request.

        It used to open Settings once after a successful run; Settings now
        opens only from Summarize results / Explain metrics (or Configure LLM).
        """
        for status in ("success", "failed"):
            with self.subTest(last_run_status=status):
                at = self.page(step="home", home_entry="quick", last_run_status=status)
                self.click(at, "home_quick_open_review")
                self.assertEqual(at.session_state["current_step"], "review_results")
                self.assertFalse(self.dialog_open(at))
                self.assertNotIn(settings_panel.REVEAL_REQUEST_KEY, at.session_state)
        # The first request for an interpretation is what opens it.
        self.click(at, "review_summarize_results")
        self.assertTrue(self.dialog_open(at))
        self.assertEqual(self.prompts, [])

    def test_c21_the_interpretation_is_shown_open_and_once(self) -> None:
        """Summarize's answer is drawn directly below the buttons, not in a collapsed expander."""
        at = self.page(**self.connected())
        self.click(at, "review_summarize_results")
        run_answers = [m.value for m in at.markdown if "SUMMARY-FOR::" in str(m.value)]
        self.assertEqual(len(run_answers), 1, "the answer must appear exactly once on the page")
        self.assertFalse([e for e in at.expander if "LLM Assistant Output" in str(e.label)])
        self.assertTrue(any("LLM Assistant Output · Result interpretation" in str(m.value) for m in at.markdown))

    def test_c22_configure_llm_opens_the_shared_form(self) -> None:
        at = self.page(**self.connected())
        self.click(at, "review_configure_llm")
        self.assertTrue(self.dialog_open(at))
        prefix = settings_panel.DIALOG_FORM_PREFIX
        keys = {t.key for t in at.text_input}
        for field in ("base_url", "api_key", "model"):
            self.assertIn(f"{prefix}_{field}", keys)
        self.assertIn(f"{prefix}_provider", {s.key for s in at.selectbox})
        buttons = {b.key: b.label for b in at.button}
        self.assertEqual(buttons[f"{prefix}_save"], settings_panel.SAVE_LABEL)
        self.assertEqual(buttons[f"{prefix}_connect"], settings_panel.CONNECT_LABEL)
        # The saved values are shown, the key masked; nothing is retyped (Case C).
        self.assertEqual(at.text_input(key=f"{prefix}_base_url").value, SETTINGS["llm_base_url"])
        self.assertEqual(at.text_input(key=f"{prefix}_api_key").proto.type, 1)   # PASSWORD
        self.assertEqual(self.prompts, [])

    def test_c23_summarize_and_explain_without_a_connection_open_settings_and_send_nothing(self) -> None:
        for status in (llm_state.LLM_STATUS_NOT_CONFIGURED, llm_state.LLM_STATUS_CONFIGURED,
                       llm_state.LLM_STATUS_FAILED):
            for button in ("review_summarize_results", "review_explain_metrics"):
                with self.subTest(status=status, button=button):
                    session = {} if status == llm_state.LLM_STATUS_NOT_CONFIGURED else {**SETTINGS, "llm_status": status}
                    at = self.page(**session)
                    self.click(at, button)
                    self.assertTrue(self.dialog_open(at))
                    self.assertEqual(self.prompts, [])
                    self.assertEqual(self.explains, [])
                    self.assertEqual(self.connects, [], "opening Settings must not test the connection")


class TestC24ToC26SharedState(_ResultsHarness, unittest.TestCase):
    def test_c24_connect_once_and_every_page_shares_it(self) -> None:
        at = self.page(step="home")
        at.text_input(key="sidebar_llm_base_url").input(SETTINGS["llm_base_url"])
        at.text_input(key="sidebar_llm_api_key").input(SYNTHETIC_KEY)
        at.text_input(key="sidebar_llm_model").input(SETTINGS["llm_model"])
        self.click(at, "sidebar_llm_connect")
        self.assertEqual(at.session_state["llm_status"], llm_state.LLM_STATUS_CONNECTED)
        self.assertEqual(len(self.connects), 1)
        for step in ("agent_workflow", "run_model", "review_results", "home"):
            with self.subTest(step=step):
                at.session_state["current_step"] = step
                at.run()
                self.assertHealthy(at)
                self.assertEqual(at.session_state["llm_status"], llm_state.LLM_STATUS_CONNECTED)
                self.assertEqual(at.session_state["llm_api_key"], SYNTHETIC_KEY)
                self.assertFalse(self.dialog_open(at))
                self.assertFalse(any(settings_panel.CONFIGURE_IN_SETTINGS_MESSAGE in t for t in self.texts(at)))
        # Results recognised the connection: Explain runs through it.
        at.session_state["current_step"] = "review_results"
        at.run()
        self.click(at, "review_explain_metrics")
        self.assertEqual(len(self.explains), 1)
        self.assertEqual(len(self.connects), 1, "no page re-tested the connection")

    def test_c25_switching_project_keeps_the_llm_state(self) -> None:
        at = self.page(**self.connected())
        other = self.review_project.parent / "another_project"
        other.mkdir()
        at.session_state["project_root"] = str(other)
        at.session_state["project_name"] = other.name
        at.session_state["current_project"] = other.name
        at.run()
        self.assertHealthy(at)
        for key, value in self.connected().items():
            self.assertEqual(at.session_state[key], value, key)
        # And no project-loading code path touches the LLM keys.
        for path in (REPO_ROOT / "batteryml_web" / "core" / "project_context.py",):
            self.assertNotIn("llm_", path.read_text(encoding="utf-8"))

    def test_c26_reruns_do_not_repeat_a_request(self) -> None:
        at = self.page(**self.connected())
        self.click(at, "review_summarize_results")
        self.assertEqual(len(self.prompts), 1)
        for _ in range(3):
            at.run()
            self.assertHealthy(at)
        self.assertEqual(len(self.prompts), 1)
        self.click(at, "review_explain_metrics")
        at.run()
        self.assertEqual(len(self.explains), 1)


class TestC27C28Verification(unittest.TestCase):
    """The status rules, on the one store, through a fake session."""

    def _connected_session(self) -> Dict[str, Any]:
        return {**SETTINGS, "llm_status": llm_state.LLM_STATUS_CONNECTED, "llm_status_message": "ok"}

    def test_c27_a_changed_provider_model_or_key_is_no_longer_verified(self) -> None:
        changes = {
            "model": dict(model="other-model"),
            "base_url": dict(base_url="https://other.example.invalid/v1"),
            "key": dict(api_key="sk-synthetic-other-0000000000"),
            "provider": dict(provider="OpenAI"),
        }
        for name, change in changes.items():
            with self.subTest(change=name):
                session = self._connected_session()
                with _fake_streamlit(FakeStreamlit(session)):
                    args = dict(base_url=SETTINGS["llm_base_url"], api_key=SYNTHETIC_KEY,
                                model=SETTINGS["llm_model"], provider=SETTINGS["llm_provider_preset"])
                    args.update(change)
                    llm_state.set_llm_settings(**args)
                    self.assertEqual(llm_state.get_llm_status(), llm_state.LLM_STATUS_CONFIGURED)
                    self.assertFalse(llm_state.is_llm_connected())
                    self.assertFalse(settings_panel.llm_action_allowed())

    def test_c28_an_unchanged_save_keeps_the_connection(self) -> None:
        session = self._connected_session()
        session.update({"t_base_url": SETTINGS["llm_base_url"], "t_api_key": SYNTHETIC_KEY,
                        "t_model": SETTINGS["llm_model"]})
        with _fake_streamlit(FakeStreamlit(session, {settings_panel.SAVE_LABEL})):
            action = settings_panel.render_llm_settings_form(key_prefix="t")
        self.assertEqual(action, "save")
        self.assertEqual(session["llm_status"], llm_state.LLM_STATUS_CONNECTED)


class TestC29C30Provenance(_ResultsHarness, unittest.TestCase):
    def test_c29_an_old_401_is_not_shown_as_the_live_state(self) -> None:
        at = self.page(**self.connected())
        self.prompt_error = RuntimeError("HTTP Error 401: Unauthorized")
        self.click(at, "review_summarize_results")
        self.assertEqual(at.session_state["llm_status"], llm_state.LLM_STATUS_FAILED)
        warnings = [w.value for w in at.warning if "The last request for this result failed" in w.value]
        self.assertTrue(warnings, "the failed request is not reported")
        self.assertTrue(any("not a live connection check" in c.value for c in at.caption))
        # The user fixes it and connects (sidebar, fake test passes).
        self.prompt_error = None
        self.click(at, "sidebar_llm_connect")
        self.assertEqual(at.session_state["llm_status"], llm_state.LLM_STATUS_CONNECTED)
        at.run()
        self.assertFalse([w.value for w in at.warning if "The last request for this result failed" in w.value])
        self.assertFalse(any("401" in t for t in self.texts(at)))
        self.assertTrue(any("LLM assistant status: Connected" in c.value for c in at.caption))
        self.assertEqual(len(self.prompts), 1, "Connect must not resend the failed request")

    def test_c30_an_explanation_stays_with_its_run(self) -> None:
        run_a, run_b = RUNS[0][0], RUNS[1][0]
        at = self.page(**self.connected(), run_history_selection=run_a)
        self.click(at, "review_summarize_results")
        self.assertTrue(any(f"SUMMARY-FOR::{run_a}" in t for t in self.texts(at)))
        self.assertIn(run_a, at.session_state["review_llm_output_identity"])
        self.assertEqual(at.session_state["review_llm_output_llm_settings"],
                         "|".join([SETTINGS["llm_provider_preset"], SETTINGS["llm_base_url"],
                                   SETTINGS["llm_model"], "key"]))
        at.selectbox(key="run_history_selection").set_value(run_b).run()
        self.assertHealthy(at)
        self.assertFalse(any(f"SUMMARY-FOR::{run_a}" in t for t in self.texts(at)),
                         "run A's summary is shown under run B")
        at.selectbox(key="run_history_selection").set_value(run_a).run()
        self.assertFalse(any("SUMMARY-FOR::" in t for t in self.texts(at)), "an old answer was relabelled")
        self.assertEqual(len(self.prompts), 1, "nothing is regenerated automatically")


class TestC31KeyNeverLeaves(_ResultsHarness, unittest.TestCase):
    def test_c31_the_key_is_not_written_logged_or_shown(self) -> None:
        records: List[str] = []

        class _Capture(logging.Handler):
            def emit(self, record: logging.LogRecord) -> None:
                records.append(self.format(record))

        handler = _Capture(level=logging.DEBUG)
        root = logging.getLogger()
        previous = root.level
        root.addHandler(handler)
        root.setLevel(logging.DEBUG)
        self.addCleanup(root.removeHandler, handler)
        self.addCleanup(root.setLevel, previous)

        at = self.page(**self.connected())
        self.click(at, "review_summarize_results")                         # ok
        self.prompt_error = RuntimeError(f"HTTP Error 401: invalid api key {SYNTHETIC_KEY}")
        self.click(at, "review_summarize_results")                         # error echoing the key
        self.connect_ok = False
        self.click(at, "review_configure_llm")
        at.run()
        self.click(at, "sidebar_llm_connect")                              # failed connect

        shown = "\n".join(self.texts(at))
        self.assertNotIn(SYNTHETIC_KEY, shown)
        self.assertNotIn(SYNTHETIC_KEY[10:30], shown)
        self.assertNotIn(SYNTHETIC_KEY, "\n".join(records))
        for path in self.tmp.rglob("*"):
            if path.is_file():
                with self.subTest(path=str(path.relative_to(self.tmp))):
                    self.assertNotIn(SYNTHETIC_KEY.encode(), path.read_bytes())
        # The only displayable summary carries a boolean, never the value.
        with _fake_streamlit(FakeStreamlit(dict(SETTINGS))):
            summary = llm_state.llm_settings_summary()
            self.assertNotIn(SYNTHETIC_KEY, repr(summary))
            self.assertNotIn(SYNTHETIC_KEY, llm_state.llm_settings_fingerprint())


class TestDialogBody(unittest.TestCase):
    """The dialog itself, through the fake: Close and Connect close it; nothing else is sent."""

    def test_close_forgets_the_request_and_redraws(self) -> None:
        session = {settings_panel.REVEAL_REQUEST_KEY: settings_panel.REVEAL_CONFIGURE}
        with _fake_streamlit(FakeStreamlit(session, {settings_panel.CLOSE_LABEL})):
            with self.assertRaises(_Rerun):
                settings_panel._settings_dialog_body(settings_panel.REVEAL_CONFIGURE)
        self.assertNotIn(settings_panel.REVEAL_REQUEST_KEY, session)

    def test_a_successful_connect_closes_it_without_a_request(self) -> None:
        session = {"dialog_llm_base_url": SETTINGS["llm_base_url"], "dialog_llm_api_key": SYNTHETIC_KEY,
                   "dialog_llm_model": SETTINGS["llm_model"],
                   settings_panel.REVEAL_REQUEST_KEY: settings_panel.REVEAL_ACTION}
        with mock.patch.object(settings_panel, "test_llm_connection", lambda **_: (True, "ok")), \
                mock.patch("batteryml_web.agent.llm_client.run_llm_prompt", _no_network):
            with _fake_streamlit(FakeStreamlit(session, {settings_panel.CONNECT_LABEL})):
                with self.assertRaises(_Rerun):
                    settings_panel._settings_dialog_body(settings_panel.REVEAL_ACTION)
        self.assertEqual(session["llm_status"], llm_state.LLM_STATUS_CONNECTED)
        self.assertNotIn(settings_panel.REVEAL_REQUEST_KEY, session)

    def test_case_d_shows_a_short_redacted_failure_and_keeps_the_fields(self) -> None:
        session = {**SETTINGS, "llm_status": llm_state.LLM_STATUS_FAILED,
                   "llm_status_message": f"HTTP Error 401: Unauthorized ({SYNTHETIC_KEY})"}
        fake = FakeStreamlit(session)
        with _fake_streamlit(fake):
            settings_panel._render_settings_body("t")
        captions = "\n".join(fake.rec["captions"])
        self.assertIn("Last connection test failed", captions)
        self.assertNotIn(SYNTHETIC_KEY, captions)
        self.assertEqual(session["t_base_url"], SETTINGS["llm_base_url"])   # fields kept
        self.assertEqual(session["llm_status"], llm_state.LLM_STATUS_FAILED)   # not retried

    def test_no_automatic_reveal_remains(self) -> None:
        """Settings opens only for Configure LLM or an LLM action without a connection."""
        source = Path(settings_panel.__file__).read_text(encoding="utf-8")
        for gone in ("REVEAL_ARRIVAL", "RESULTS_INTENT_KEY", "consume_results_intent", "mark_results_intent"):
            self.assertNotIn(gone, source)
            self.assertNotIn(gone, APP.read_text(encoding="utf-8"))
        self.assertEqual(set(settings_panel.REVEAL_NOTES), {settings_panel.REVEAL_ACTION})

    def test_one_form_one_call_site(self) -> None:
        source = Path(settings_panel.__file__).read_text(encoding="utf-8")
        self.assertEqual(source.count("render_llm_settings_form(key_prefix="), 1)
        self.assertNotIn("render_llm_settings_form", APP.read_text(encoding="utf-8"))
        self.assertIn("_render_settings_body(DIALOG_FORM_PREFIX", source)
        self.assertIn("_render_settings_body(SIDEBAR_FORM_PREFIX", source)


class TestD32ToD36ExistingSafety(unittest.TestCase):
    """The backend contracts this change sits beside, checked directly."""

    def test_d32_an_soh_task_cannot_run_an_rul_config(self) -> None:
        from batteryml_web.core import quick_experience as qx
        from batteryml_web.core.config_task import TASK_RUL, TASK_SOH

        plan = qx.build_quick_plan(
            task=TASK_SOH, config_path=REPO_ROOT / "configs" / "baselines" / "sklearn" / "xgb" / "matr_1.yaml",
            device="cpu", repo_root=REPO_ROOT, allowed_datasets=["MATR"], cuda_ok=False, require_data=False)
        self.assertFalse(plan.valid)
        self.assertEqual(plan.config_task, TASK_RUL)
        self.assertTrue(any("trains for" in e and TASK_RUL in e for e in plan.errors), plan.errors)

    def test_d33_yaml_upload_stays_safe(self) -> None:
        from batteryml_web.core import config_upload as cu

        config, reason = cu.parse_uploaded_yaml(b"model: !!python/object/apply:os.system ['echo unsafe']\n")
        self.assertIsNone(config)
        self.assertIn("tag", reason)
        self.assertIsNotNone(cu.check_upload_name("../escape.yaml"))
        self.assertIsNone(cu.parse_uploaded_yaml(b"")[0])

    def test_d34_cpu_and_cuda_stay_guarded(self) -> None:
        from batteryml_web.core import quick_experience as qx
        from batteryml_web.ui import agent_workflow as aw

        self.assertIsNone(qx.device_problem("cpu", cuda_ok=False))
        self.assertIn("Choose cpu explicitly", qx.device_problem("cuda", cuda_ok=False, cuda_detail="x"))
        plan = {"can_execute": True, "command": ["python", "run"], "reason": ""}
        reason = aw.gpu_mode_block(mode="GPU", gpu_options=["cuda:0"], chosen_gpu="cuda:0",
                                   cuda_status=(False, "sm_86 unsupported"))
        blocked = aw.block_plan(plan, reason)
        self.assertFalse(aw.run_is_enabled(blocked))
        self.assertEqual(blocked["command"], plan["command"])            # never rewritten to cpu

    def test_d35_quick_runs_stay_in_the_project_history(self) -> None:
        from batteryml_web.core import direct_runs
        from batteryml_web.ui import run_history as rh

        tmp = Path(tempfile.mkdtemp(prefix="synthetic_quick_history_"))
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        run_id = "run_20261012_120000_xgb_matr_seed0_abcdef"
        run_dir = direct_runs.direct_run_dir(tmp, "quickstart", run_id)
        Path(run_dir).mkdir(parents=True, exist_ok=True)
        rm.write_manifest(run_dir, {"run_id": run_id, "status": rm.STATUS_COMPLETED,
                                    "created_at": "2026-10-12T12:00:00+00:00", "dataset": "MATR"})
        runs = rm.list_runs(tmp)
        self.assertEqual([r["run_id"] for r in runs], [run_id])
        self.assertEqual(rh.history_rows(runs)[0]["Status"], rm.STATUS_COMPLETED)
        source = APP.read_text(encoding="utf-8")
        self.assertIn('direct_runs.direct_run_dir(workspace_root, "quickstart", run_id)', source)

    def test_d36_readiness_and_config_selection(self) -> None:
        from batteryml_web.core import quick_experience as qx
        from batteryml_web.core.config_task import TASK_RUL, TASK_SOH

        # "Runnable" means a protocol config exists for it (data aside); the
        # readiness verdict is carried separately and never upgraded.
        choices = qx.dataset_choices({"SNL": "RAW_ONLY", "HUST": "RAW_ONLY", "MATR": qx.STATUS_READY},
                                     [("MATR", "RUL"), ("MATR", "SOH"), ("HUST", "RUL")], default="MATR")
        self.assertEqual([c.name for c in choices], ["MATR", "HUST", "SNL"])
        self.assertEqual([c.runnable for c in choices], [True, True, False])
        self.assertEqual([c.status for c in choices], [qx.STATUS_READY, "RAW_ONLY", "RAW_ONLY"])
        self.assertEqual(set(choices[0].tasks), {TASK_RUL, TASK_SOH})
        with self.assertRaises(ValueError):
            qx.dataset_choices({"MATR": qx.STATUS_READY}, [("SNL", "RUL")], default="MATR")
        self.assertEqual(qx.reconcile_selection("gone.yaml", ["a.yaml", "b.yaml"], default="b.yaml"),
                         ("b.yaml", "gone.yaml"))
        self.assertEqual(qx.reconcile_selection("a.yaml", ["a.yaml"]), ("a.yaml", None))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
