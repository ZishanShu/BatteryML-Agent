"""Tests for LLM-assisted intent resolution and the catalog it is bound to.

Milestone D. No network: every LLM response is a stub. The point of these
tests is the refusal policy — the LLM may interpret, and the backend decides
— so most of them feed the resolver an answer that is wrong in a specific
way and assert that the wrong part is rejected rather than used.

Run from the repository root:

    conda run -n batteryml python -m unittest tests.test_llm_intent -v
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from batteryml_web.agent.catalog import (  # noqa: E402
    CANONICAL_TASKS,
    build_intent_catalog,
    closest_model_keys,
    resolve_catalog_dataset,
    resolve_catalog_model,
    resolve_catalog_task,
)
from batteryml_web.agent.llm_intent import (  # noqa: E402
    FORBIDDEN_KEYS,
    MAX_RATIONALE_CHARS,
    SYSTEM_PROMPT,
    build_intent_prompt,
    extract_json_object,
    filter_suggestions,
    find_catalog_model_mention,
    resolve_model_request,
    resolve_workflow_intent_with_llm,
    validate_llm_intent,
)

TASK_RUL = "RUL prediction"
TASK_SOH = "SOH estimation"
TASK_TRAJ = "Degradation trajectory forecasting"

CATALOG = {
    "datasets": [
        {"dataset": "MATR", "aliases": ["MATR", "matr"], "status": "READY",
         "preprocessing_supported": True},
        {"dataset": "HUST", "aliases": ["HUST", "hust"], "status": "READY",
         "preprocessing_supported": True},
        {"dataset": "MICH", "aliases": ["MICH", "mich"], "status": "RAW_ONLY",
         "preprocessing_supported": False},
    ],
    "tasks": [
        {"task": TASK_RUL, "aliases": ["RUL", "battery lifetime", "寿命预测"]},
        {"task": TASK_SOH, "aliases": ["SOH", "state of health"]},
        {"task": TASK_TRAJ, "aliases": ["trajectory"]},
    ],
    "models": [
        {"model_key": "cptransformer", "model_display": "CPTransformer — CyclePatch Transformer",
         "aliases": ["cptransformer", "CPTransformerRULPredictor"], "datasets": ["MATR"],
         "tasks": [TASK_RUL], "has_validated_config": True, "external": False},
        {"model_key": "batlinet", "model_display": "BatLiNet — Inter-cell Lifetime Prediction",
         "aliases": ["batlinet", "BatLiNetRULPredictor"], "datasets": ["MATR"],
         "tasks": [TASK_RUL], "has_validated_config": True, "external": False},
        {"model_key": "xgb", "model_display": "XGBoost — Extreme Gradient Boosting",
         "aliases": ["xgb", "xgboost"], "datasets": ["MATR", "HUST"],
         "tasks": [TASK_RUL], "has_validated_config": True, "external": False},
        {"model_key": "PBT", "model_display": "PBT", "aliases": ["PBT", "pbt"],
         "datasets": [], "tasks": [TASK_RUL], "has_validated_config": False, "external": True},
    ],
}

CONTEXT = {
    "dataset": "MATR",
    "dataset_status": "READY",
    "task": TASK_RUL,
    "preprocessing_supported": True,
    "runnable_models": [
        {"model": "cptransformer", "model_display": "CPTransformer — CyclePatch Transformer",
         "status": "READY", "validated_config": "configs/external/cptransformer/matr_1.yaml"},
        {"model": "batlinet", "model_display": "BatLiNet — Inter-cell Lifetime Prediction",
         "status": "READY", "validated_config": "configs/external/batlinet/matr_1.yaml"},
        {"model": "xgb", "model_display": "XGBoost — Extreme Gradient Boosting",
         "status": "READY", "validated_config": "configs/baselines/sklearn/xgb/matr_1.yaml"},
    ],
    "blocked_models": [
        {"model": "PBT", "status": "NEEDS_ASSETS", "missing": ["pretrained checkpoint"],
         "recommended_action": "Provide the pretrained assets."},
    ],
    "recommended_next_steps": ["Pick a runnable model."],
}


def _stub(response: str):
    calls: list = []

    def call(prompt, **kwargs):
        calls.append({"prompt": prompt, "kwargs": kwargs})
        return response

    call.calls = calls  # type: ignore[attr-defined]
    return call


def _resolve(request: str, response: str | None = None, **kwargs):
    stub = _stub(response if response is not None else "{}")
    result = resolve_workflow_intent_with_llm(
        request, catalog=CATALOG, llm_call=stub, **kwargs
    )
    return result, stub


class TestCatalogFromTheRealBackend(unittest.TestCase):
    """The catalog is derived, never a second hand-written list."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.catalog = build_intent_catalog()

    def test_datasets_come_from_the_platform_list(self) -> None:
        names = {entry["dataset"] for entry in self.catalog["datasets"]}
        for expected in ("MATR", "HUST", "CALCE", "MICH"):
            self.assertIn(expected, names)
        for entry in self.catalog["datasets"]:
            self.assertIn(
                entry["status"],
                {"READY", "RAW_ONLY", "INCOMPLETE_PROCESSED", "MISSING"},
            )

    def test_tasks_are_exactly_the_three_platform_tasks(self) -> None:
        self.assertEqual(
            [entry["task"] for entry in self.catalog["tasks"]], list(CANONICAL_TASKS)
        )

    def test_models_come_from_the_config_registry_and_external_table(self) -> None:
        by_key = {entry["model_key"]: entry for entry in self.catalog["models"]}
        for expected in ("cptransformer", "batlinet", "xgb"):
            self.assertIn(expected, by_key)
            self.assertIn("MATR", by_key[expected]["datasets"])
            self.assertTrue(by_key[expected]["has_validated_config"])
            self.assertFalse(by_key[expected]["external"])
        self.assertIn("PBT", by_key)
        self.assertTrue(by_key["PBT"]["external"])
        self.assertEqual(by_key["PBT"]["tasks"], [TASK_RUL])
        self.assertEqual(by_key["BatteryMFormer"]["tasks"], [TASK_TRAJ])

    def test_class_names_and_display_names_resolve(self) -> None:
        self.assertEqual(
            resolve_catalog_model("BatLiNetRULPredictor", self.catalog), "batlinet"
        )
        self.assertEqual(resolve_catalog_model("XGBoost", self.catalog), "xgb")
        self.assertIsNone(resolve_catalog_model("SomeNewModel", self.catalog))

    def test_catalog_carries_no_config_paths(self) -> None:
        """A path in the catalog would be a path the LLM could echo back."""
        blob = repr(self.catalog)
        self.assertNotIn(".yaml", blob)
        self.assertNotIn("configs/", blob)


class TestCatalogResolution(unittest.TestCase):
    def test_dataset_aliases(self) -> None:
        self.assertEqual(resolve_catalog_dataset("matr", CATALOG), "MATR")
        self.assertEqual(resolve_catalog_dataset("MATR", CATALOG), "MATR")
        self.assertIsNone(resolve_catalog_dataset("NOT_A_DATASET", CATALOG))
        self.assertIsNone(resolve_catalog_dataset(None, CATALOG))

    def test_task_aliases_including_chinese(self) -> None:
        self.assertEqual(resolve_catalog_task("rul", CATALOG), TASK_RUL)
        self.assertEqual(resolve_catalog_task("寿命预测", CATALOG), TASK_RUL)
        self.assertEqual(resolve_catalog_task("state of health", CATALOG), TASK_SOH)
        self.assertIsNone(resolve_catalog_task("clustering", CATALOG))

    def test_closest_names_are_suggestions_not_matches(self) -> None:
        self.assertIsNone(resolve_catalog_model("cptransfomer", CATALOG))
        self.assertIn("cptransformer", closest_model_keys("cptransfomer", CATALOG))

    def test_model_mention_in_mixed_language_text(self) -> None:
        self.assertEqual(
            find_catalog_model_mention("MATR 上用 CPTransformer 做寿命预测", CATALOG),
            "cptransformer",
        )
        self.assertEqual(
            find_catalog_model_mention("用CPTransformer做预测", CATALOG), "cptransformer"
        )
        self.assertEqual(
            find_catalog_model_mention("Try BatLiNet on my MATR cells", CATALOG), "batlinet"
        )
        self.assertIsNone(find_catalog_model_mention("run something on MATR", CATALOG))

    def test_short_aliases_do_not_match_inside_words(self) -> None:
        self.assertIsNone(find_catalog_model_mention("please use xgbfoo here", CATALOG))


class TestJsonExtraction(unittest.TestCase):
    def test_plain_object(self) -> None:
        self.assertEqual(extract_json_object('{"dataset": "MATR"}'), {"dataset": "MATR"})

    def test_markdown_fence_and_surrounding_prose(self) -> None:
        self.assertEqual(
            extract_json_object('Sure!\n```json\n{"dataset": "MATR"}\n```\nHope that helps.'),
            {"dataset": "MATR"},
        )

    def test_invalid_or_absent_json_is_none_not_a_guess(self) -> None:
        for text in ("", "no json here", "{not json}", "[1, 2, 3]", "null"):
            with self.subTest(text=text):
                self.assertIsNone(extract_json_object(text))


class TestValidation(unittest.TestCase):
    def test_a_clean_answer_passes_through(self) -> None:
        result = validate_llm_intent(
            {
                "dataset": "matr",
                "task": "rul",
                "requested_model": "CPTransformerRULPredictor",
                "needs_clarification": False,
                "clarification_question": None,
                "suggested_model_keys": ["batlinet", "xgb"],
                "user_facing_rationale": "MATR has validated RUL configs.",
            },
            CATALOG,
        )
        self.assertEqual(result["dataset"], "MATR")
        self.assertEqual(result["task"], TASK_RUL)
        self.assertEqual(result["requested_model"], "cptransformer")
        self.assertEqual(result["suggested_model_keys"], ["batlinet", "xgb"])
        self.assertEqual(result["rejected"], [])
        self.assertFalse(result["needs_clarification"])

    def test_invented_dataset_is_rejected_not_accepted(self) -> None:
        result = validate_llm_intent({"dataset": "MATR_V2", "task": "rul"}, CATALOG)
        self.assertIsNone(result["dataset"])
        self.assertTrue(result["needs_clarification"])
        self.assertIn(
            {"field": "dataset", "value": "MATR_V2", "kind": "unknown_dataset"},
            result["rejected"],
        )

    def test_invented_task_and_model_are_rejected(self) -> None:
        result = validate_llm_intent(
            {"dataset": "MATR", "task": "anomaly detection", "requested_model": "GPT-5"},
            CATALOG,
        )
        self.assertIsNone(result["task"])
        self.assertIsNone(result["requested_model"])
        self.assertEqual(result["requested_model_name"], "GPT-5")
        kinds = {item["kind"] for item in result["rejected"]}
        self.assertEqual(kinds, {"unknown_task", "unknown_model"})

    def test_backend_owned_fields_are_dropped_and_reported(self) -> None:
        payload = {"dataset": "MATR", "task": "rul"}
        for key in FORBIDDEN_KEYS:
            payload[key] = "anything"
        result = validate_llm_intent(payload, CATALOG)
        rejected_fields = {item["field"] for item in result["rejected"]}
        self.assertEqual(rejected_fields, set(FORBIDDEN_KEYS))
        blob = repr({k: v for k, v in result.items() if k != "rejected"})
        self.assertNotIn("configs/", blob)
        self.assertNotIn(".yaml", blob)
        self.assertTrue(result["warnings"])

    def test_unknown_suggestions_are_dropped_individually(self) -> None:
        result = validate_llm_intent(
            {
                "dataset": "MATR",
                "task": "rul",
                "suggested_model_keys": ["batlinet", "MegaNet9000", "xgb"],
            },
            CATALOG,
        )
        self.assertEqual(result["suggested_model_keys"], ["batlinet", "xgb"])
        self.assertIn(
            {"field": "suggested_model_keys", "value": "MegaNet9000", "kind": "unknown_model"},
            result["rejected"],
        )

    def test_missing_dataset_or_task_forces_clarification(self) -> None:
        result = validate_llm_intent(
            {"dataset": None, "task": "rul", "needs_clarification": False}, CATALOG
        )
        self.assertTrue(result["needs_clarification"])

    def test_rationale_is_capped_and_flattened(self) -> None:
        result = validate_llm_intent(
            {"dataset": "MATR", "task": "rul", "user_facing_rationale": "x " * 2000},
            CATALOG,
        )
        self.assertLessEqual(len(result["user_facing_rationale"]), MAX_RATIONALE_CHARS)

    def test_a_non_object_response_is_refused_wholesale(self) -> None:
        for payload in (None, [], "MATR", 7):
            with self.subTest(payload=payload):
                result = validate_llm_intent(payload, CATALOG)
                self.assertIsNone(result["dataset"])
                self.assertTrue(result["needs_clarification"])
                self.assertTrue(result["rejected"])

    def test_suggestions_that_are_not_a_list_are_refused(self) -> None:
        result = validate_llm_intent(
            {"dataset": "MATR", "task": "rul", "suggested_model_keys": "batlinet"}, CATALOG
        )
        self.assertEqual(result["suggested_model_keys"], [])
        self.assertIn("not_a_list", {item["kind"] for item in result["rejected"]})


class TestResolutionRouting(unittest.TestCase):
    def test_obvious_request_takes_the_deterministic_fast_path(self) -> None:
        result, stub = _resolve("Use CPTransformer on MATR to predict battery lifetime.")
        self.assertEqual(result["source"], "deterministic")
        self.assertEqual(stub.calls, [])  # no tokens spent
        self.assertEqual(result["dataset"], "MATR")
        self.assertEqual(result["task"], TASK_RUL)
        self.assertEqual(result["requested_model"], "cptransformer")

    def test_informal_request_goes_to_the_llm(self) -> None:
        response = (
            '{"dataset": "MATR", "task": "RUL prediction", "requested_model": null,'
            ' "needs_clarification": false, "suggested_model_keys": ["xgb"],'
            ' "user_facing_rationale": "Cycle aging on MATR is the RUL task."}'
        )
        result, stub = _resolve(
            "Can you run a model for cycle aging on the MATR dataset?", response
        )
        self.assertEqual(len(stub.calls), 1)
        self.assertEqual(result["source"], "llm")
        self.assertEqual(result["dataset"], "MATR")
        self.assertEqual(result["task"], TASK_RUL)
        self.assertEqual(result["suggested_model_keys"], ["xgb"])

    def test_llm_failure_falls_back_to_the_deterministic_resolver(self) -> None:
        def boom(prompt, **kwargs):
            raise RuntimeError("502 from the gateway")

        result = resolve_workflow_intent_with_llm(
            "something vague about MATR", catalog=CATALOG, llm_call=boom
        )
        self.assertEqual(result["source"], "deterministic_fallback")
        self.assertEqual(result["dataset"], "MATR")
        self.assertIn("502", " ".join(result["warnings"]))

    def test_unparsable_reply_is_discarded(self) -> None:
        result, _ = _resolve("something vague about MATR", "I think you want MATR!")
        self.assertEqual(result["source"], "deterministic_fallback")
        self.assertIn("not valid JSON", " ".join(result["warnings"]))

    def test_without_credentials_the_shortfall_is_stated(self) -> None:
        result = resolve_workflow_intent_with_llm(
            "Use transformer on MATR to predict life", catalog=CATALOG
        )
        self.assertFalse(result["llm_available"])
        self.assertEqual(result["source"], "deterministic_fallback")
        self.assertIn(
            "advanced natural-language understanding is unavailable",
            " ".join(result["warnings"]),
        )
        # The deterministic half still works: the dataset is resolved, and a
        # model name that is not in the catalog stays unresolved rather than
        # being folded into a near neighbour.
        self.assertEqual(result["dataset"], "MATR")
        self.assertIsNone(result["requested_model"])
        named = resolve_workflow_intent_with_llm(
            "Use CPTransformer on MATR for life", catalog=CATALOG
        )
        self.assertEqual(named["requested_model"], "cptransformer")

    def test_empty_request_is_a_programming_error(self) -> None:
        with self.assertRaises(ValueError):
            resolve_workflow_intent_with_llm("   ", catalog=CATALOG)

    def test_no_dataset_is_never_silently_defaulted_to_matr(self) -> None:
        response = (
            '{"dataset": null, "task": "RUL prediction", "requested_model": null,'
            ' "needs_clarification": true,'
            ' "clarification_question": "Which dataset should I use?",'
            ' "suggested_model_keys": [], "user_facing_rationale": "No dataset was named."}'
        )
        result, _ = _resolve("I want future capacity prediction with the best model", response)
        self.assertIsNone(result["dataset"])
        self.assertTrue(result["needs_clarification"])
        self.assertIn("Which dataset", result["clarification_question"])

    def test_explicit_model_survives_a_vague_llm_answer(self) -> None:
        response = (
            '{"dataset": "MATR", "task": "RUL prediction", "requested_model": null,'
            ' "needs_clarification": false, "suggested_model_keys": [],'
            ' "user_facing_rationale": "ok"}'
        )
        result, _ = _resolve("run BatLiNet for the aging thing on MATR cells", response)
        self.assertEqual(result["requested_model"], "batlinet")

    def test_result_never_carries_a_config_path_or_command(self) -> None:
        response = (
            '{"dataset": "MATR", "task": "RUL prediction",'
            ' "config_path": "configs/external/cptransformer/matr_1.yaml",'
            ' "command": "python bin/batteryml.py run ...",'
            ' "requested_model": "cptransformer", "needs_clarification": false,'
            ' "suggested_model_keys": [], "user_facing_rationale": "ok"}'
        )
        result, _ = _resolve("cycle aging on the MATR cells please", response)
        payload = {k: v for k, v in result.items() if k != "rejected"}
        blob = repr(payload)
        self.assertNotIn(".yaml", blob)
        self.assertNotIn("bin/batteryml.py", blob)
        self.assertNotIn("can_execute", result)


class TestChineseRequests(unittest.TestCase):
    def test_mixed_language_requests_resolve(self) -> None:
        for text, dataset, task in (
            ("用 MATR 数据集预测电池寿命", "MATR", TASK_RUL),
            ("MATR 上用 CPTransformer 做寿命预测", "MATR", TASK_RUL),
            ("帮我用 MATR 做电池寿命预测", "MATR", TASK_RUL),
        ):
            with self.subTest(text=text):
                result, stub = _resolve(text)
                self.assertEqual(result["dataset"], dataset)
                self.assertEqual(result["task"], task)
                self.assertEqual(stub.calls, [])  # deterministic handles these

    def test_recommendation_request_without_a_dataset_asks_for_one(self) -> None:
        response = (
            '{"dataset": null, "task": "RUL prediction", "requested_model": null,'
            ' "needs_clarification": true,'
            ' "clarification_question": "请问使用哪个数据集？",'
            ' "suggested_model_keys": ["cptransformer", "batlinet"],'
            ' "user_facing_rationale": "需要先确定数据集。"}'
        )
        result, stub = _resolve("帮我推荐一个能做电池寿命预测的模型", response)
        self.assertEqual(len(stub.calls), 1)
        self.assertIsNone(result["dataset"])
        self.assertTrue(result["needs_clarification"])
        self.assertEqual(result["suggested_model_keys"], ["cptransformer", "batlinet"])

    def test_trajectory_request_resolves_the_task_without_a_dataset(self) -> None:
        result, _ = _resolve(
            "我想预测未来容量衰减轨迹",
            '{"dataset": null, "task": "Degradation trajectory forecasting",'
            ' "requested_model": null, "needs_clarification": true,'
            ' "clarification_question": "哪个数据集?", "suggested_model_keys": [],'
            ' "user_facing_rationale": "ok"}',
        )
        self.assertEqual(result["task"], TASK_TRAJ)
        self.assertIsNone(result["dataset"])


class TestModelRequestAgainstReadiness(unittest.TestCase):
    """Readiness, not the LLM, decides what a requested model means."""

    def test_runnable_model_is_selectable(self) -> None:
        result = resolve_model_request("cptransformer", CONTEXT, CATALOG)
        self.assertEqual(result["status"], "runnable")
        self.assertEqual(result["model_key"], "cptransformer")
        self.assertFalse(result["requires_confirmation"])

    def test_blocked_model_is_never_silently_replaced(self) -> None:
        result = resolve_model_request("PBT", CONTEXT, CATALOG)
        self.assertEqual(result["status"], "blocked")
        self.assertEqual(result["model_key"], "PBT")
        self.assertIn("NEEDS_ASSETS", result["reason"])
        self.assertTrue(result["requires_confirmation"])
        self.assertEqual(result["alternatives"], ["cptransformer", "batlinet", "xgb"])

    def test_unknown_model_is_reported_as_unregistered(self) -> None:
        result = resolve_model_request("SomeNewModel", CONTEXT, CATALOG)
        self.assertEqual(result["status"], "unknown")
        self.assertIsNone(result["model_key"])
        self.assertIn("not currently registered", result["reason"])
        self.assertTrue(result["requires_confirmation"])

    def test_near_miss_name_suggests_but_does_not_select(self) -> None:
        result = resolve_model_request("cptransfomer", CONTEXT, CATALOG)
        self.assertEqual(result["status"], "unknown")
        self.assertIsNone(result["model_key"])
        self.assertIn("cptransformer", result["alternatives"])
        self.assertTrue(result["requires_confirmation"])

    def test_no_request_needs_no_confirmation(self) -> None:
        result = resolve_model_request(None, CONTEXT, CATALOG)
        self.assertEqual(result["status"], "none")
        self.assertFalse(result["requires_confirmation"])

    def test_suggestions_are_split_by_the_backend(self) -> None:
        split = filter_suggestions(["cptransformer", "PBT", "MegaNet"], CONTEXT)
        self.assertEqual(split["runnable"], ["cptransformer"])
        self.assertEqual(split["blocked"], ["PBT"])
        self.assertEqual(split["unknown"], ["MegaNet"])

    def test_nothing_runnable_means_no_alternatives_are_invented(self) -> None:
        empty_context = dict(CONTEXT, runnable_models=[])
        result = resolve_model_request("PBT", empty_context, CATALOG)
        self.assertEqual(result["alternatives"], [])


class TestExecutionSafety(unittest.TestCase):
    """Free-form LLM output can never reach an executor from this module."""

    def setUp(self) -> None:
        import ast

        self.source = (
            REPO_ROOT / "batteryml_web" / "agent" / "llm_intent.py"
        ).read_text(encoding="utf-8")
        self.tree = ast.parse(self.source)

    def _called_names(self) -> set[str]:
        """Names actually called in the module, prose in docstrings excluded."""
        import ast

        names: set[str] = set()
        for node in ast.walk(self.tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if isinstance(func, ast.Name):
                names.add(func.id)
            elif isinstance(func, ast.Attribute):
                prefix = func.value.id if isinstance(func.value, ast.Name) else ""
                names.add(f"{prefix}.{func.attr}" if prefix else func.attr)
        return names

    def _imported_modules(self) -> set[str]:
        import ast

        modules: set[str] = set()
        for node in ast.walk(self.tree):
            if isinstance(node, ast.Import):
                modules.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                modules.add(node.module)
                modules.update(alias.name for alias in node.names)
        return modules

    def test_module_never_plans_or_executes(self) -> None:
        called = self._called_names()
        for forbidden in (
            "execute_plan",
            "execute_plan_streaming",
            "build_execution_plan",
            "plan_for_request",
            "Popen",
            "run",
            "system",
        ):
            self.assertNotIn(forbidden, called, f"llm_intent must not call {forbidden}")
        imported = self._imported_modules()
        for forbidden in ("subprocess", "os", "batteryml_web.agent.executor"):
            self.assertNotIn(forbidden, imported)

    def test_module_does_not_read_readiness_directly(self) -> None:
        """Readiness reaches this layer as a context the caller passes in."""
        called = self._called_names()
        for forbidden in ("get_workflow_context", "assess_model", "assess_dataset"):
            self.assertNotIn(forbidden, called)

    def test_no_chain_of_thought_is_requested(self) -> None:
        prompt = build_intent_prompt("anything", CATALOG).lower()
        system = SYSTEM_PROMPT.lower()
        for phrase in ("step by step", "step-by-step", "chain of thought", "reasoning:"):
            self.assertNotIn(phrase, prompt)
            self.assertNotIn(phrase, system)
        self.assertIn("do not include reasoning steps", system)

    def test_prompt_contains_only_catalog_identifiers(self) -> None:
        prompt = build_intent_prompt("anything", CATALOG)
        self.assertNotIn(".yaml", prompt)
        self.assertNotIn("configs/", prompt)
        self.assertIn("cptransformer", prompt)
        self.assertIn("PBT", prompt)


if __name__ == "__main__":
    unittest.main(verbosity=2)
