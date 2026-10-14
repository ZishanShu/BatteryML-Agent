"""Agent Workflow: deterministic alternatives, and "Explain this workflow".

The user asked "Use ElasticNet on HUST to predict soh." and saw "HUST READY,
0 model(s) runnable, 23 blocked" with nothing that said why or what to do
instead; "Explain this workflow" appeared to do nothing. These tests pin the
fix (UI-15, UI-16). HUST has had SOH configurations since SOH-14, so the
"ready data, no SOH configuration" case is now exercised on SNL, which has
RUL configurations only:

* A1–A9: the alternatives come from the real configuration registry and the
  real readiness/validation layer, evaluated against temporary repository
  roots that hold the real ``configs/`` and ``synthetic_`` processed files.
  Nothing is invented: a combination is runnable only if readiness says so.
* B10–B20: the Explain interaction on the real page (``streamlit.testing``),
  with a fake LLM client and a network tripwire. No paid request is made.

Every dataset file here is synthetic; no model is trained.

    python -m unittest tests.test_agent_alternatives -v
"""

from __future__ import annotations

import os
import pickle
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

from batteryml_web.agent import alternatives as alt  # noqa: E402
from batteryml_web.agent import llm_client, workflow  # noqa: E402
from batteryml_web.agent.catalog import build_intent_catalog  # noqa: E402
from batteryml_web.agent.llm_intent import resolve_model_request  # noqa: E402
from batteryml_web.agent.prompts import build_agent_workflow_context  # noqa: E402
from batteryml_web.ui import agent_workflow as aw  # noqa: E402

try:
    from streamlit.testing.v1 import AppTest
except ImportError:  # pragma: no cover
    AppTest = None

APP = REPO_ROOT / "batteryml_web" / "app.py"
SOH = "SOH estimation"
RUL = "RUL prediction"
REQUEST = "Use ElasticNet on SNL to predict soh."  # SNL: RUL configs only (HUST has SOH since SOH-14)
SECRET = "sk-synthetic-NEVERSHOW-0123456789abcdef"


def _make_root(*, processed=(), raw=()) -> Path:
    """A temporary repository root: the real configs, synthetic data files."""
    from batteryml.preprocess import DOWNLOAD_LINKS

    root = Path(tempfile.mkdtemp(prefix="synthetic_repo_"))
    (root / "configs").symlink_to(REPO_ROOT / "configs")
    for dataset in processed:
        folder = root / "data" / "processed" / dataset
        folder.mkdir(parents=True)
        for index in range(4):
            (folder / f"synthetic_cell_{index}.pkl").write_bytes(pickle.dumps({"synthetic": True}))
    for dataset in raw:
        folder = root / "data" / "raw" / dataset
        folder.mkdir(parents=True)
        for item in DOWNLOAD_LINKS[dataset]:
            (folder / item[1]).write_bytes(b"synthetic")
    return root


class _InRoot(unittest.TestCase):
    """Readiness against one synthetic root (the validator resolves data
    paths from the working directory, so the test runs inside the root)."""

    PROCESSED: tuple = ("MATR", "SNL")
    RAW: tuple = ()

    def setUp(self) -> None:
        self.root = _make_root(processed=self.PROCESSED, raw=self.RAW)
        self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)
        previous = os.getcwd()
        os.chdir(self.root)
        self.addCleanup(os.chdir, previous)
        self.catalog = build_intent_catalog()

    def context(self, dataset: str, task: str) -> Dict[str, Any]:
        return build_agent_workflow_context(dataset, task, repo_root=self.root)

    def alternatives(self, dataset: str, task: str, model: str | None = "ElasticNet") -> Dict[str, Any]:
        context = self.context(dataset, task)
        request = resolve_model_request(model, context, self.catalog)
        return alt.build_workflow_alternatives(
            context, model_request=request, context_for=self.context, repo_root=self.root
        )


class TestA1ToA4RegistryDerived(_InRoot):
    def test_a1_hust_ready_but_nothing_runs_soh(self) -> None:
        context = self.context("SNL", SOH)
        self.assertEqual(context["dataset_status"], "READY")
        self.assertEqual(context["runnable_models"], [])
        result = self.alternatives("SNL", SOH)
        self.assertEqual(result["diagnosis"]["code"], alt.NO_CONFIGURATION)
        self.assertIn("SNL data is ready, but no registered SOH estimation configuration", result["diagnosis"]["message"])
        self.assertEqual(result["requested"]["model"], "elastic_net")

    def test_a2_no_hust_soh_configuration_is_invented(self) -> None:
        configured = alt.configured_triples()
        self.assertFalse([t for t in configured if t[0] == "SNL" and t[1] == SOH])
        result = self.alternatives("SNL", SOH)
        self.assertEqual(result["same_dataset_task"], [])
        self.assertFalse([a for a in result["actions"] if a["dataset"] == "SNL" and a["task"] == SOH])
        block = alt.alternatives_prompt_block(result)
        self.assertNotIn("SNL / SOH estimation", block)

    def test_a3_alternatives_are_the_registry_and_readiness_answer(self) -> None:
        configured = alt.configured_triples()
        result = self.alternatives("SNL", SOH)
        expected_datasets = sorted({d for d, t, _ in configured if t == SOH and d != "SNL"})
        self.assertEqual([e["dataset"] for e in result["same_task_other_datasets"]], expected_datasets)
        for entry in result["same_task_other_datasets"]:
            registered = {m for d, t, m in configured if d == entry["dataset"] and t == SOH}
            readiness = {m["model"] for m in self.context(entry["dataset"], SOH)["runnable_models"]}
            self.assertEqual({m["model"] for m in entry["runnable"]}, readiness)
            self.assertLessEqual(readiness, registered)
        expected_tasks = sorted({t for d, t, _ in configured if d == "SNL" and t != SOH})
        self.assertEqual([e["task"] for e in result["same_dataset_other_tasks"]], expected_tasks)

    def test_a4_matr_soh_models_are_offered_when_ready(self) -> None:
        result = self.alternatives("SNL", SOH)
        matr = next(e for e in result["same_task_other_datasets"] if e["dataset"] == "MATR")
        self.assertEqual(matr["dataset_status"], "READY")
        self.assertTrue(matr["runnable"])
        action = next(a for a in result["actions"] if a["dataset"] == "MATR")
        self.assertEqual(action["model"], "elastic_net")         # the requested model, runnable there
        self.assertIn("on MATR for SOH estimation", action["label"])


class TestA4NotReadyMeansNotOffered(_InRoot):
    PROCESSED = ("SNL",)

    def test_matr_soh_is_named_but_never_offered_without_data(self) -> None:
        result = self.alternatives("SNL", SOH)
        matr = next(e for e in result["same_task_other_datasets"] if e["dataset"] == "MATR")
        self.assertEqual(matr["runnable"], [])
        registered = {m for d, t, m in alt.configured_triples() if d == "MATR" and t == SOH}
        self.assertEqual(set(matr["configured_not_runnable"]), registered)
        self.assertFalse([a for a in result["actions"] if a["dataset"] == "MATR"])
        self.assertIn("none of those datasets is ready here", result["diagnosis"]["message"])


class TestA6RawOnly(_InRoot):
    PROCESSED = ("SNL",)
    RAW = ("MATR",)

    def test_raw_only_is_never_advertised_as_runnable(self) -> None:
        self.assertEqual(self.context("MATR", SOH)["dataset_status"], "RAW_ONLY")
        result = self.alternatives("SNL", SOH)
        matr = next(e for e in result["same_task_other_datasets"] if e["dataset"] == "MATR")
        self.assertEqual((matr["dataset_status"], matr["runnable"]), ("RAW_ONLY", []))
        self.assertFalse([a for a in result["actions"] if a["dataset"] == "MATR"])
        own = self.alternatives("MATR", SOH)
        self.assertEqual(own["diagnosis"]["code"], alt.DATASET_NOT_READY)
        self.assertIn("raw data is present but has not been preprocessed", own["diagnosis"]["message"])


class TestModelNamesWithoutSeparators(unittest.TestCase):
    """"ElasticNet" names elastic_net: the request in the report was not even
    recognised as asking for a model, so nothing could be said about it."""

    def test_written_without_a_separator(self) -> None:
        from batteryml_web.agent.catalog import resolve_catalog_model
        from batteryml_web.agent.llm_intent import find_catalog_model_mention

        catalog = build_intent_catalog()
        self.assertEqual(find_catalog_model_mention(REQUEST, catalog), "elastic_net")
        self.assertEqual(find_catalog_model_mention("use elastic-net on MATR", catalog), "elastic_net")
        self.assertEqual(resolve_catalog_model("ElasticNet", catalog), "elastic_net")
        self.assertIsNone(resolve_catalog_model("ElasticNetPlusPlus", catalog))   # still exact, never fuzzy


class TestA5HonestEmpty(unittest.TestCase):
    def test_no_alternative_says_so(self) -> None:
        context = {"dataset": "SNL", "task": SOH, "dataset_status": "READY",
                   "runnable_models": [], "blocked_models": []}
        result = alt.build_workflow_alternatives(
            context, configured=[], context_for=lambda d, t: self.fail("nothing to look up"))
        self.assertEqual(result["diagnosis"]["code"], alt.NO_CONFIGURATION)
        self.assertIn("No dataset currently has a registered SOH estimation configuration",
                      result["diagnosis"]["message"])
        self.assertEqual((result["actions"], result["same_task_other_datasets"],
                          result["same_dataset_other_tasks"]), ([], [], []))

    def test_a_runnable_request_needs_no_alternatives(self) -> None:
        context = {"dataset": "MATR", "task": SOH, "dataset_status": "READY",
                   "runnable_models": [{"model": "ridge", "model_display": "Ridge"}], "blocked_models": []}
        result = alt.build_workflow_alternatives(
            context, model_request={"requested": "ridge", "status": "runnable", "model_key": "ridge"},
            configured=[("MATR", SOH, "ridge")], context_for=lambda d, t: self.fail("not needed"))
        self.assertEqual(result["diagnosis"]["code"], alt.RUNNABLE)
        self.assertEqual(result["actions"], [])


# --- the real page ---------------------------------------------------------------


def _no_network(*args, **kwargs):
    raise AssertionError("a test tried to open a network connection")


@unittest.skipIf(AppTest is None, "streamlit.testing is not available")
class _Page(_InRoot):
    """Agent Workflow on the synthetic root, every LLM call faked."""

    def setUp(self) -> None:
        super().setUp()
        self.prompts: List[Dict[str, Any]] = []
        self.reply: Any = "SYNTHETIC EXPLANATION: see configs/invented/hust_soh.yaml"
        self.launches: List[Any] = []

        def fake_llm(prompt, **kwargs):
            self.prompts.append({"prompt": prompt, **{k: v for k, v in kwargs.items() if k != "api_key"}})
            if isinstance(self.reply, Exception):
                raise self.reply
            return self.reply

        real_context = build_agent_workflow_context
        patches = [
            mock.patch.object(workflow, "run_llm_prompt", fake_llm),
            mock.patch.object(llm_client, "run_llm_prompt", fake_llm),
            mock.patch.object(aw, "build_agent_workflow_context",
                              lambda d, t, **kw: real_context(d, t, repo_root=self.root, **kw)),
            mock.patch.object(aw, "_launch_run", lambda **kw: self.launches.append(kw)),
            mock.patch("urllib.request.urlopen", _no_network),
        ]
        for patch in patches:
            patch.start()
            self.addCleanup(patch.stop)

    def page(self, *, connected: bool, request: str = REQUEST) -> "AppTest":
        at = AppTest.from_file(str(APP), default_timeout=300)
        at.session_state["current_step"] = "agent_workflow"
        at.session_state["agent_pending_request"] = request
        if connected:
            for key, value in {"llm_base_url": "https://llm.example.invalid/v1", "llm_api_key": SECRET,
                               "llm_model": "synthetic-model", "llm_status": "connected"}.items():
                at.session_state[key] = value
        at.run()
        self.assertFalse(at.exception, [str(e.value)[:300] for e in at.exception])
        return at

    @staticmethod
    def after_button(at, label: str, count: int = 4) -> List[str]:
        main = [str(getattr(n, "label", "") or getattr(n, "value", "")) for n in at.main]
        index = main.index(label)
        return main[index + 1:index + 1 + count]

    def click(self, at, key: str) -> None:
        at.button(key=key).click().run()
        self.assertFalse(at.exception, [str(e.value)[:300] for e in at.exception])


class TestA7ToA9AlternativesOnThePage(_Page):
    def test_a9_without_an_llm_the_page_still_says_what_to_do(self) -> None:
        at = self.page(connected=False)
        warnings = " ".join(w.value for w in at.warning)
        self.assertIn("SNL data is ready, but no registered SOH estimation configuration", warnings)
        labels = [b.label for b in at.button if (b.key or "").startswith("agent_alternative_")]
        self.assertIn("Use Elastic Net Regression on MATR for SOH estimation", labels)
        self.assertIn("Change task to RUL prediction with Elastic Net Regression", labels)
        self.assertEqual(self.prompts, [])

    def test_a7_a8_choosing_an_alternative_reanalyses_and_never_runs(self) -> None:
        at = self.page(connected=False)
        at.session_state["agent_selected_config"] = "configs/stale/previous.yaml"
        at.session_state["agent_config_choice_elastic_net"] = "configs/stale/previous.yaml"
        key = next(b.key for b in at.button if b.label.startswith("Use Elastic Net Regression on MATR"))
        self.click(at, key)
        context = at.session_state["agent_workflow_context"]
        self.assertEqual((context["dataset"], context["task"]), ("MATR", SOH))
        self.assertEqual(at.session_state["agent_request_input"], "Use Elastic Net Regression on MATR for SOH estimation.")
        self.assertEqual(at.session_state["agent_selected_model"], "elastic_net")
        self.assertNotEqual(at.session_state["agent_selected_config"] if "agent_selected_config" in at.session_state else None,
                            "configs/stale/previous.yaml")
        self.assertEqual(self.launches, [], "choosing an alternative started a run")
        self.assertNotIn("agent_execution_result", at.session_state)
        self.assertTrue(any(b.key == "agent_run" for b in at.button))       # offered, not pressed


class TestB10ToB18Explain(_Page):
    def test_b10_rendering_spends_nothing(self) -> None:
        at = self.page(connected=True)
        at.run()
        self.assertEqual(self.prompts, [])

    def test_b11_to_b14_one_click_one_request_shown_under_the_button(self) -> None:
        at = self.page(connected=True)
        self.click(at, "agent_explain_workflow")
        self.assertEqual(len(self.prompts), 1)
        below = self.after_button(at, aw.EXPLAIN_WORKFLOW_LABEL)
        self.assertTrue(any("SYNTHETIC EXPLANATION" in text for text in below), below)
        self.assertFalse([e for e in at.expander if e.label == "Agent explanation"])
        for _ in range(3):
            at.run()
        self.assertEqual(len(self.prompts), 1)
        self.assertTrue(any("SYNTHETIC EXPLANATION" in text for text in self.after_button(at, aw.EXPLAIN_WORKFLOW_LABEL)))

    def test_b15_without_a_connection_the_shared_settings_open(self) -> None:
        at = self.page(connected=False)
        self.click(at, "agent_explain_workflow")
        self.assertEqual(self.prompts, [])
        below = self.after_button(at, aw.EXPLAIN_WORKFLOW_LABEL)
        self.assertTrue(any(aw.EXPLAIN_NEEDS_LLM in text for text in below), below)
        self.assertIn(aw.OPEN_LLM_SETTINGS_LABEL, below)
        self.assertTrue([t for t in at.text_input if t.key == "dialog_llm_api_key"], "the shared settings did not open")

    def test_b16_a_provider_failure_is_shown_redacted_and_not_retried(self) -> None:
        self.reply = RuntimeError(f"LLM API request failed with status 401: invalid key {SECRET}")
        at = self.page(connected=True)
        self.click(at, "agent_explain_workflow")
        warnings = [w.value for w in at.warning if "could not be generated" in w.value]
        self.assertTrue(warnings)
        page_text = " ".join(str(getattr(n, "value", "")) for n in at.main)
        self.assertNotIn(SECRET, page_text)
        self.assertIn("Workflow analysis", [h.value for h in at.subheader])   # deterministic page intact
        at.run()
        self.assertEqual(len(self.prompts), 1, "a failure must not be retried automatically")

    def test_b17_b18_an_unrunnable_workflow_is_explained_with_the_backend_alternatives(self) -> None:
        at = self.page(connected=True)
        self.click(at, "agent_explain_workflow")
        sent = self.prompts[0]
        stored = at.session_state[aw.ALTERNATIVES_KEY]
        self.assertIn(alt.alternatives_prompt_block(stored), sent["prompt"])
        self.assertIn("diagnosis: no_configuration", sent["prompt"])
        self.assertIn("never as an execution plan", sent["system_prompt"])
        self.assertIn("never offer to start training", sent["system_prompt"])


class TestB19B20Boundaries(_Page):
    def test_b19_a_new_request_drops_the_old_explanation(self) -> None:
        at = self.page(connected=True)
        self.click(at, "agent_explain_workflow")
        key = next(b.key for b in at.button if b.label.startswith("Use Elastic Net Regression on MATR"))
        self.click(at, key)
        self.assertNotIn(aw.WORKFLOW_EXPLANATION_KEY, at.session_state)
        text = " ".join(str(getattr(n, "value", "")) for n in at.main)
        self.assertNotIn("SYNTHETIC EXPLANATION", text)
        self.assertEqual(len(self.prompts), 1)

    def test_b20_no_invented_configuration_reaches_a_plan(self) -> None:
        at = self.page(connected=True)
        self.click(at, "agent_explain_workflow")
        self.assertNotIn("agent_execution_plan", at.session_state)             # SNL SOH: nothing to plan
        key = next(b.key for b in at.button if b.label.startswith("Use Elastic Net Regression on MATR"))
        self.click(at, key)
        plan = at.session_state["agent_execution_plan"]
        command = " ".join(str(part) for part in plan.get("command") or [])
        self.assertNotIn("invented", command)
        runnable = {m["model"]: m["validated_config"]
                    for m in at.session_state["agent_workflow_context"]["runnable_models"]}
        self.assertIn(str(runnable["elastic_net"]), command)
        self.assertEqual(self.launches, [])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
