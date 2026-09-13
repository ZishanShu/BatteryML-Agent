"""Backend-only tests for deterministic execution planning.

No training, no subprocess, no downloads: plans are built and the real
:func:`config_validator.load_and_validate_config_file` is run with
``require_data=True`` as the final gate. Registry-backed assertions skip
where ``batteryml`` cannot be imported.

Run from the repository root:

    conda run -n batteryml python -m unittest tests.test_agent_execution -v
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tests.test_readiness import REGISTRIES_AVAILABLE  # noqa: E402
from batteryml_web.agent.execution import (  # noqa: E402
    WRAPPER,
    build_execution_plan,
    find_model_entry,
    plan_for_request,
)


class TestRunnablePlans(unittest.TestCase):
    """CPTransformer / BatLiNet on MATR: validated config, plan, no run."""

    @unittest.skipUnless(REGISTRIES_AVAILABLE, "batteryml is not importable in this environment")
    def test_cptransformer_matr_plan(self) -> None:
        plan = plan_for_request("Use CPTransformer on MATR to predict battery lifetime.")

        self.assertEqual(plan["dataset"], "MATR")
        self.assertEqual(plan["task"], "RUL prediction")
        self.assertEqual(plan["model"], "cptransformer")
        self.assertEqual(plan["model_status"], "READY")
        self.assertTrue(plan["config_path"].endswith(
            "configs/external/cptransformer/matr_1.yaml"
        ))
        self.assertTrue(plan["config_valid"], plan["validation_errors"])
        self.assertTrue(plan["can_execute"])
        self.assertEqual(plan["validation_errors"], [])
        command = plan["command"]
        self.assertIsInstance(command, list)
        # Exact CLI contract: wrapper, positional config, train+eval+device+seed+workspace.
        self.assertEqual(command[1], WRAPPER)
        self.assertEqual(command[2], "run")
        self.assertEqual(command[3], plan["config_path"])
        self.assertIn("--train", command)
        self.assertIn("--eval", command)
        self.assertIn("--device", command)
        self.assertIn("--seed", command)
        self.assertIn("--workspace", command)
        self.assertNotIn("--config", command)

    @unittest.skipUnless(REGISTRIES_AVAILABLE, "batteryml is not importable in this environment")
    def test_batlinet_matr_plan(self) -> None:
        plan = plan_for_request("Use BatLiNet on MATR to predict battery lifetime.")

        self.assertEqual(plan["model"], "batlinet")
        self.assertEqual(plan["model_status"], "READY")
        self.assertTrue(plan["config_path"].endswith(
            "configs/external/batlinet/matr_1.yaml"
        ))
        self.assertTrue(plan["config_valid"], plan["validation_errors"])
        self.assertTrue(plan["can_execute"])
        self.assertIsInstance(plan["command"], list)

    @unittest.skipUnless(REGISTRIES_AVAILABLE, "batteryml is not importable in this environment")
    def test_model_alias_resolution(self) -> None:
        from batteryml_web.agent.prompts import build_agent_workflow_context

        context = build_agent_workflow_context("MATR", "RUL prediction")
        entry, side = find_model_entry(context, "CPTransformerRULPredictor")
        self.assertEqual(side, "runnable")
        self.assertEqual(entry["model"], "cptransformer")


class TestBlockedPlans(unittest.TestCase):
    """Blocked models and datasets never yield an executable plan."""

    @unittest.skipUnless(REGISTRIES_AVAILABLE, "batteryml is not importable in this environment")
    def test_pbt_blocked(self) -> None:
        plan = plan_for_request("Use PBT on MATR to predict battery lifetime.")

        self.assertEqual(plan["model"], "PBT")
        self.assertEqual(plan["model_status"], "NEEDS_ASSETS")
        self.assertFalse(plan["can_execute"])
        self.assertIsNone(plan["config_path"])
        self.assertTrue(plan["command"] in (None, []))

    @unittest.skipUnless(REGISTRIES_AVAILABLE, "batteryml is not importable in this environment")
    def test_batterymformer_trajectory_blocked(self) -> None:
        plan = plan_for_request("Forecast the degradation trajectory for MATR with BatteryMFormer.")

        self.assertEqual(plan["task"], "Degradation trajectory forecasting")
        self.assertEqual(plan["model"], "BatteryMFormer")
        self.assertEqual(plan["model_status"], "NEEDS_DATA")
        self.assertFalse(plan["can_execute"])
        self.assertIsNone(plan["config_path"])
        self.assertTrue(plan["command"] in (None, []))

    @unittest.skipUnless(REGISTRIES_AVAILABLE, "batteryml is not importable in this environment")
    def test_mich_raw_only_blocked(self) -> None:
        plan = plan_for_request("Use XGBoost on MICH to predict battery lifetime.")

        self.assertEqual(plan["dataset"], "MICH")
        self.assertEqual(plan["dataset_status"], "RAW_ONLY")
        self.assertFalse(plan["can_execute"])
        self.assertIsNone(plan["command"])
        self.assertNotIn("data/raw/MICH", str(plan))

    @unittest.skipUnless(REGISTRIES_AVAILABLE, "batteryml is not importable in this environment")
    def test_unknown_model_cannot_execute(self) -> None:
        plan = build_execution_plan("MATR", "RUL prediction", "definitely_not_a_model")

        self.assertFalse(plan["can_execute"])
        self.assertIsNone(plan["config_path"])
        self.assertTrue(plan["command"] in (None, []))

    @unittest.skipUnless(REGISTRIES_AVAILABLE, "batteryml is not importable in this environment")
    def test_plan_never_invents_config_paths(self) -> None:
        from batteryml_web.agent.prompts import build_agent_workflow_context

        context = build_agent_workflow_context("MATR", "RUL prediction")
        allowed = {
            str(e.get("validated_config"))
            for e in context["runnable_models"]
        }
        for name in ("cptransformer", "batlinet"):
            plan = build_execution_plan("MATR", "RUL prediction", name)
            self.assertIn(plan["config_path"], allowed)


class TestValidatorIsFinalGate(unittest.TestCase):
    """An invalid config refuses execution even against mocked READY state."""

    def test_invalid_config_forces_no_execution(self) -> None:
        import tempfile

        bogus = Path(tempfile.gettempdir()) / "__agent_exec_bogus__.yaml"
        bogus.write_text("model: {}\n", encoding="utf-8")
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
                    "validated_config": str(bogus),
                }
            ],
            "blocked_models": [],
            "recommended_next_steps": [],
        }
        plan = build_execution_plan(
            "MATR", "RUL prediction", "xgb", workflow_context=context
        )

        self.assertEqual(plan["config_path"], str(bogus))
        self.assertFalse(plan["config_valid"])
        self.assertTrue(plan["validation_errors"])
        self.assertFalse(plan["can_execute"])
        self.assertTrue(plan["command"] in (None, []))

    def test_empty_identifiers_rejected(self) -> None:
        with self.assertRaises(ValueError):
            build_execution_plan("", "RUL prediction", "xgb")


if __name__ == "__main__":
    unittest.main(verbosity=2)
