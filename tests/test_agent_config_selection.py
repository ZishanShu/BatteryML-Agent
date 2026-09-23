"""Agent config selection honors explicit split requests (STEP 7C-FIX).

Regression test for the real-server bug where
"Use CNN on MATR with the MATR primary test split (matr_1) to predict
battery lifetime." resolved to ``configs/baselines/nn_models/cnn/clo.yaml``:
the planner took the first valid config in sort order ("Clo" sorts before
"Official BatteryML split 1") and the "(matr_1)" hint was discarded during
intent parsing.

Behavioral contract asserted here (no training process is launched by any
test in this module — every path ends in plan construction at most):

1-3. MATR + CNN + explicit matr_1 / "primary test split" / both -> matr_1.yaml
4-5. MATR + CNN + explicit matr_2 / "secondary test split" -> matr_2.yaml
6.   invalid explicit split/config -> deterministic refusal, no silent fallback
7.   generic MATR + CNN -> documented default OR explicit ambiguity, never an
     arbitrary first-list pick
8-9. HUST / SNL + CNN stay associated with their own dataset configs
10.  covered by the untouched STEP 2 registry suite (run separately)
11.  no test here starts a subprocess
"""

from __future__ import annotations

import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

try:
    from batteryml_web.core import readiness as readiness_mod

    readiness_mod.get_config_registry(str(REPO_ROOT)).entries
    REGISTRIES_AVAILABLE = True
    REGISTRY_SKIP_MESSAGE = ""
except Exception as exc:  # noqa: BLE001 - skip reason only
    REGISTRIES_AVAILABLE = False
    REGISTRY_SKIP_MESSAGE = f"registries unavailable: {exc}"

require_registries = unittest.skipUnless(REGISTRIES_AVAILABLE, REGISTRY_SKIP_MESSAGE)

CNN_MATR_1 = "configs/baselines/nn_models/cnn/matr_1.yaml"
CNN_MATR_2 = "configs/baselines/nn_models/cnn/matr_2.yaml"
CNN_CLO = "configs/baselines/nn_models/cnn/clo.yaml"


def _tail(path) -> str:
    tail = str(path or "").replace("\\", "/").split("configs/", 1)[-1]
    return tail if tail.startswith("configs/") else f"configs/{tail}"


def _plan(request: str):
    from batteryml_web.agent.execution import plan_for_request

    return plan_for_request(request, repo_root=str(REPO_ROOT), device="cpu", seed=0, epochs=1)


@require_registries
class TestExplicitSplitHintWins(unittest.TestCase):
    def test_matr_1_token_resolves_matr_1(self) -> None:
        plan = _plan("Use CNN on MATR with matr_1 to predict battery lifetime.")
        self.assertEqual(_tail(plan["config_path"]), CNN_MATR_1)
        self.assertTrue(plan["can_execute"])

    def test_primary_test_split_resolves_matr_1(self) -> None:
        plan = _plan("Use CNN on MATR with the MATR primary test split to predict battery lifetime.")
        self.assertEqual(_tail(plan["config_path"]), CNN_MATR_1)
        self.assertTrue(plan["can_execute"])

    def test_primary_test_split_with_matr_1_resolves_matr_1(self) -> None:
        plan = _plan(
            "Use CNN on MATR with the MATR primary test split (matr_1) "
            "to predict battery lifetime."
        )
        self.assertEqual(_tail(plan["config_path"]), CNN_MATR_1)
        self.assertTrue(plan["can_execute"])
        self.assertNotEqual(_tail(plan["config_path"]), CNN_CLO)

    def test_matr_2_token_resolves_matr_2(self) -> None:
        plan = _plan("Use CNN on MATR with matr_2 to predict battery lifetime.")
        self.assertEqual(_tail(plan["config_path"]), CNN_MATR_2)
        self.assertTrue(plan["can_execute"])

    def test_secondary_test_split_resolves_matr_2(self) -> None:
        plan = _plan("Use CNN on MATR with the MATR secondary test split to predict battery lifetime.")
        self.assertEqual(_tail(plan["config_path"]), CNN_MATR_2)
        self.assertTrue(plan["can_execute"])


@require_registries
class TestInvalidSplitRefused(unittest.TestCase):
    def test_unknown_split_has_no_silent_fallback(self) -> None:
        plan = _plan("Use CNN on MATR with matr_3 to predict battery lifetime.")
        self.assertFalse(plan["can_execute"])
        self.assertIsNone(plan["config_path"])
        self.assertIsNone(plan["command"])
        self.assertIn("clo.yaml", str(plan.get("config_choices")))
        self.assertIn("matr_1.yaml", str(plan.get("config_choices")))

    def test_unknown_config_filename_refused(self) -> None:
        plan = _plan("Use CNN on MATR with config matr_9.yaml to predict battery lifetime.")
        self.assertFalse(plan["can_execute"])
        self.assertIsNone(plan["config_path"])


@require_registries
class TestGenericRequestNeverPicksArbitrarily(unittest.TestCase):
    def test_generic_matr_cnn_is_default_or_ambiguity(self) -> None:
        plan = _plan("Use CNN on MATR to predict battery lifetime.")
        choices = [_tail(c) for c in (plan.get("config_choices") or [])]
        self.assertGreaterEqual(len(choices), 2)
        self.assertIn(CNN_CLO, choices)
        self.assertIn(CNN_MATR_1, choices)
        self.assertIn(CNN_MATR_2, choices)
        if plan["config_path"] is None:
            self.assertFalse(plan["can_execute"])
            self.assertEqual(plan.get("config_selection"), "ambiguous")
        else:
            # A repository-documented default is the only acceptable silent pick.
            self.assertTrue(plan["can_execute"])
            self.assertIn(_tail(plan["config_path"]), choices)


@require_registries
class TestOtherDatasetsUnaffected(unittest.TestCase):
    def test_hust_cnn_stays_hust(self) -> None:
        plan = _plan("Use CNN on HUST to predict battery lifetime.")
        self.assertEqual(_tail(plan["config_path"]), "configs/baselines/nn_models/cnn/hust.yaml")
        self.assertTrue(plan["can_execute"])

    def test_snl_cnn_stays_snl_associated(self) -> None:
        from batteryml_web.core.config_registry import get_config_registry

        registry = get_config_registry(str(REPO_ROOT))
        entries = registry.get_configs_for_selection("SNL", "RUL prediction", "cnn")
        self.assertTrue(entries)
        for entry in entries:
            self.assertEqual(entry.dataset, "SNL")
            self.assertIn("snl", entry.config_path.as_posix())


@require_registries
class TestSplitIdentityIsStructural(unittest.TestCase):
    def test_split_keys_come_from_splitter_class(self) -> None:
        from batteryml_web.core.config_registry import get_config_registry

        registry = get_config_registry(str(REPO_ROOT))
        by_name = {
            e.config_path.name: e
            for e in registry.get_configs_for_selection("MATR", "RUL prediction", "cnn")
        }
        self.assertEqual(by_name["matr_1.yaml"].split_key, "matr_primary")
        self.assertEqual(by_name["matr_2.yaml"].split_key, "matr_secondary")
        self.assertEqual(by_name["clo.yaml"].split_key, "matr_clo")

    def test_intent_carries_requested_split(self) -> None:
        from batteryml_web.agent.intent import resolve_workflow_intent

        intent = resolve_workflow_intent(
            "Use CNN on MATR with the MATR primary test split (matr_1) "
            "to predict battery lifetime."
        )
        self.assertEqual(intent["dataset"], "MATR")
        self.assertEqual(intent["requested_split"], "matr_primary")
        generic = resolve_workflow_intent("Use CNN on MATR to predict battery lifetime.")
        self.assertIsNone(generic["requested_split"])
        self.assertFalse(generic["mentions_config_or_split"])


if __name__ == "__main__":
    unittest.main()
