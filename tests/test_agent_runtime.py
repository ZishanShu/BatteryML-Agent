"""Backend-only tests for the agent runtime flow.

No real external LLM API calls: the LLM boundary is a stub injected via
``guide_workflow(..., llm_call=...)``, and
:func:`workflow_guidance_preview` exercises the same path without any LLM
at all. Registry-backed assertions skip where ``batteryml`` cannot be
imported.

Run from the repository root:

    conda run -n batteryml python -m unittest tests.test_agent_runtime -v
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tests.test_readiness import REGISTRIES_AVAILABLE  # noqa: E402
from batteryml_web.agent.intent import (  # noqa: E402
    resolve_task,
    resolve_dataset,
    resolve_workflow_intent,
)
from batteryml_web.agent.workflow import (  # noqa: E402
    SYSTEM_PROMPT,
    build_workflow_guidance_prompt,
    guide_workflow,
    workflow_guidance_preview,
)


def _stub_llm_factory(seen: dict) -> object:
    def stub(prompt: str, *, system_prompt: str = "", **kwargs: object) -> str:
        seen["prompt"] = prompt
        seen["system_prompt"] = system_prompt
        return "STUBBED-GUIDANCE: use a runnable model."
    return stub


def _runnable_names(context: dict) -> set[str]:
    return {str(item.get("model")) for item in context.get("runnable_models", [])}


def _blocked_names(context: dict) -> set[str]:
    return {str(item.get("model")) for item in context.get("blocked_models", [])}


class TestIntentResolution(unittest.TestCase):
    """Dataset/task phrases fold onto the platform vocabulary."""

    def test_lifetime_request_resolves_matr_rul(self) -> None:
        intent = resolve_workflow_intent("I want to use MATR to predict battery lifetime.")
        self.assertEqual(intent["dataset"], "MATR")
        self.assertEqual(intent["task"], "RUL prediction")
        self.assertFalse(intent["needs_dataset_clarification"])
        self.assertFalse(intent["needs_task_clarification"])

    def test_task_phrase_variants(self) -> None:
        for phrase, expected in (
            ("cycle life prediction on HUST", "RUL prediction"),
            ("RUL for MATR", "RUL prediction"),
            ("remaining useful life of CALCE", "RUL prediction"),
            ("SOH of MATR", "SOH estimation"),
            ("state of health for HUST", "SOH estimation"),
            ("forecast the degradation trajectory for MATR",
             "Degradation trajectory forecasting"),
            ("future capacity trajectory of MATR",
             "Degradation trajectory forecasting"),
        ):
            with self.subTest(phrase=phrase):
                self.assertEqual(resolve_task(phrase), expected)

    def test_dataset_aliases(self) -> None:
        for phrase, expected in (
            ("use matr data", "MATR"),
            ("HUST cells", "HUST"),
            ("MICH archive", "MICH"),
        ):
            with self.subTest(phrase=phrase):
                self.assertEqual(resolve_dataset(phrase), expected)

    def test_unknown_dataset_does_not_become_matr(self) -> None:
        intent = resolve_workflow_intent("Predict battery lifetime with ZZZ cells.")
        self.assertIsNone(intent["dataset"])
        self.assertEqual(intent["task"], "RUL prediction")
        self.assertTrue(intent["needs_dataset_clarification"])

    def test_empty_request_rejected(self) -> None:
        with self.assertRaises(ValueError):
            resolve_workflow_intent("   ")


class TestWorkflowRuntime(unittest.TestCase):
    """Intent -> compact context -> injected prompt -> stubbed LLM."""

    @unittest.skipUnless(REGISTRIES_AVAILABLE, "batteryml is not importable in this environment")
    def test_matr_lifetime_runtime(self) -> None:
        seen: dict = {}
        result = guide_workflow(
            "I want to use MATR to predict battery lifetime.",
            base_url="http://stub",
            api_key="stub",
            model="stub",
            llm_call=_stub_llm_factory(seen),
        )
        self.assertEqual(result["dataset"], "MATR")
        self.assertEqual(result["task"], "RUL prediction")
        runnable = _runnable_names(result["workflow_context"])
        self.assertIn("cptransformer", runnable)
        self.assertIn("batlinet", runnable)
        blocked = _blocked_names(result["workflow_context"])
        self.assertIn("PBT", blocked)
        self.assertNotIn("PBT", runnable)
        self.assertNotIn("BatteryMFormer", runnable)
        self.assertNotIn("BatteryMFormer", blocked)
        # The stub saw the injected authoritative context.
        self.assertIn("cptransformer", seen["prompt"])
        self.assertIn("PBT", seen["prompt"])
        self.assertNotIn("BatteryMFormer", seen["prompt"])
        self.assertIn("authoritative", seen["system_prompt"])
        self.assertEqual(result["guidance"], "STUBBED-GUIDANCE: use a runnable model.")
        self.assertEqual(len(result["prompt_messages"]), 2)

    @unittest.skipUnless(REGISTRIES_AVAILABLE, "batteryml is not importable in this environment")
    def test_trajectory_runtime_exposes_batterymformer(self) -> None:
        seen: dict = {}
        result = guide_workflow(
            "Forecast the degradation trajectory for MATR.",
            base_url="http://stub",
            api_key="stub",
            model="stub",
            llm_call=_stub_llm_factory(seen),
        )
        self.assertEqual(result["task"], "Degradation trajectory forecasting")
        self.assertIn("BatteryMFormer", _blocked_names(result["workflow_context"]))
        self.assertNotIn("BatteryMFormer", _runnable_names(result["workflow_context"]))
        self.assertNotIn("cptransformer", _runnable_names(result["workflow_context"]))

    @unittest.skipUnless(REGISTRIES_AVAILABLE, "batteryml is not importable in this environment")
    def test_mich_lifetime_no_executable_command(self) -> None:
        preview = workflow_guidance_preview("Predict battery lifetime with MICH.")
        context = preview["workflow_context"]
        self.assertEqual(context["dataset_status"], "RAW_ONLY")
        self.assertFalse(context["preprocessing_supported"])
        user_text = preview["prompt_messages"][1]["content"]
        self.assertNotIn("data/raw/MICH", user_text)
        self.assertIn("not yet wired", user_text)

    @unittest.skipUnless(REGISTRIES_AVAILABLE, "batteryml is not importable in this environment")
    def test_no_fabricated_config_paths(self) -> None:
        result = workflow_guidance_preview(
            "I want to use MATR to predict battery lifetime."
        )
        runnable = {
            item["model"]: item
            for item in result["workflow_context"]["runnable_models"]
        }
        for name in ("cptransformer", "batlinet"):
            validated = runnable[name]["validated_config"]
            self.assertTrue(validated)
            path = Path(validated)
            self.assertTrue(
                (REPO_ROOT / validated).is_file()
                if not path.is_absolute()
                else path.is_file()
            )
            self.assertIn("configs/", validated.replace("\\", "/"))
        user_text = result["prompt_messages"][1]["content"]
        for name in ("cptransformer", "batlinet"):
            self.assertIn(runnable[name]["validated_config"].split("configs/")[1], user_text)

    def test_unknown_dataset_never_calls_llm(self) -> None:
        calls: list = []

        def stub(prompt: str, **kwargs: object) -> str:
            calls.append(prompt)
            return "SHOULD-NOT-HAPPEN"

        result = guide_workflow(
            "Predict battery lifetime with ZZZ cells.",
            base_url="http://stub",
            api_key="stub",
            model="stub",
            llm_call=stub,
        )
        self.assertIsNone(result["dataset"])
        self.assertIsNone(result["workflow_context"])
        self.assertEqual(calls, [])
        self.assertIn("dataset", result["guidance"].lower())

    def test_prompt_states_authority_rules(self) -> None:
        preview = workflow_guidance_preview.__doc__  # sanity: preview path exists
        self.assertTrue(preview)
        context = {
            "dataset": "MATR",
            "dataset_status": "READY",
            "task": "RUL prediction",
            "preprocessing_supported": False,
            "runnable_models": [
                {
                    "model": "xgb",
                    "model_display": "XGBoost",
                    "status": "READY",
                    "validated_config": "configs/x.yaml",
                }
            ],
            "blocked_models": [
                {
                    "model": "PBT",
                    "status": "NEEDS_ASSETS",
                    "missing": ["checkpoint"],
                    "recommended_action": "Provide assets.",
                }
            ],
            "recommended_next_steps": ["Pick xgb."],
        }
        prompt = build_workflow_guidance_prompt("Use MATR for RUL.", context)
        self.assertIn("configs/x.yaml", prompt)
        self.assertIn("NEEDS_ASSETS", prompt)
        self.assertIn(SYSTEM_PROMPT.split(".")[0][:40], SYSTEM_PROMPT)


if __name__ == "__main__":
    unittest.main(verbosity=2)
