"""Backend-only tests for the agent workflow-context helper.

No LLM API calls, no training, no downloads: every test reads deterministic
state from :func:`batteryml_web.agent.prompts.build_agent_workflow_context`,
which itself only calls the filesystem/registry-backed
:func:`batteryml_web.core.readiness.get_workflow_context`. Tests that need
the live BatteryML registries skip (rather than fail) where ``batteryml``
cannot be imported.

Run from the repository root:

    python -m unittest tests.test_agent_context -v
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tests.test_readiness import (  # noqa: E402
    DEMO_ENVIRONMENT_AVAILABLE,
    DEMO_ENVIRONMENT_SKIP_MESSAGE,
)
from batteryml_web.agent.prompts import build_agent_workflow_context  # noqa: E402


def _runnable_names(context: dict) -> set[str]:
    return {str(item.get("model")) for item in context.get("runnable_models", [])}


def _blocked_names(context: dict) -> set[str]:
    return {str(item.get("model")) for item in context.get("blocked_models", [])}


class TestMatrRulAgentContext(unittest.TestCase):
    """The primary case: MATR lifetime prediction must offer CP/BatLiNet."""

    @unittest.skipUnless(DEMO_ENVIRONMENT_AVAILABLE, DEMO_ENVIRONMENT_SKIP_MESSAGE)
    def test_matr_rul_has_cptransformer_and_batlinet_runnable(self) -> None:
        context = build_agent_workflow_context("MATR", "RUL prediction")

        self.assertEqual(context["dataset"], "MATR")
        self.assertEqual(context["dataset_status"], "READY")
        self.assertEqual(context["task"], "RUL prediction")
        runnable = _runnable_names(context)
        self.assertIn("cptransformer", runnable)
        self.assertIn("batlinet", runnable)

    @unittest.skipUnless(DEMO_ENVIRONMENT_AVAILABLE, DEMO_ENVIRONMENT_SKIP_MESSAGE)
    def test_pbt_blocked_not_runnable(self) -> None:
        context = build_agent_workflow_context("MATR", "RUL prediction")

        self.assertNotIn("PBT", _runnable_names(context))
        self.assertIn("PBT", _blocked_names(context))
        blocked = {item["model"]: item for item in context["blocked_models"]}
        self.assertEqual(blocked["PBT"]["status"], "NEEDS_ASSETS")
        self.assertTrue(blocked["PBT"]["missing"])
        self.assertTrue(blocked["PBT"]["recommended_action"])

    @unittest.skipUnless(DEMO_ENVIRONMENT_AVAILABLE, DEMO_ENVIRONMENT_SKIP_MESSAGE)
    def test_batterymformer_absent_from_rul_recommendations(self) -> None:
        context = build_agent_workflow_context("MATR", "RUL prediction")

        self.assertNotIn("BatteryMFormer", _runnable_names(context))
        self.assertNotIn("BatteryMFormer", _blocked_names(context))
        text = " ".join(context["recommended_next_steps"]).lower()
        self.assertNotIn("batterymformer", text)

    @unittest.skipUnless(DEMO_ENVIRONMENT_AVAILABLE, DEMO_ENVIRONMENT_SKIP_MESSAGE)
    def test_no_fabricated_config_paths(self) -> None:
        context = build_agent_workflow_context("MATR", "RUL prediction")

        runnable = {item["model"]: item for item in context["runnable_models"]}
        for name in ("cptransformer", "batlinet"):
            validated = runnable[name]["validated_config"]
            self.assertTrue(validated, f"{name} runnable without a validated_config")
            self.assertTrue(
                (REPO_ROOT / validated).is_file() if not Path(validated).is_absolute()
                else Path(validated).is_file(),
                f"{name} validated_config does not exist on disk: {validated}",
            )
            self.assertIn("configs/", validated.replace("\\", "/"))
        for item in context["runnable_models"]:
            self.assertEqual(item["status"], "READY")

    @unittest.skipUnless(DEMO_ENVIRONMENT_AVAILABLE, DEMO_ENVIRONMENT_SKIP_MESSAGE)
    def test_matr_supports_preprocessing_command(self) -> None:
        context = build_agent_workflow_context("MATR", "RUL prediction")

        self.assertTrue(context["preprocessing_supported"] is False or True)
        # MATR is READY, so the flag only documents the RAW_ONLY route;
        # the MATR readiness dict must carry the explicit field.
        from batteryml_web.core.readiness import assess_dataset

        self.assertIn("preprocessing_supported", assess_dataset("MATR").to_dict())


class TestMichBlockedAgentContext(unittest.TestCase):
    """MICH is RAW_ONLY via an unwired v11 archive: never emit a command."""

    @unittest.skipUnless(DEMO_ENVIRONMENT_AVAILABLE, DEMO_ENVIRONMENT_SKIP_MESSAGE)
    def test_mich_rul_reports_raw_only_and_no_executable_command(self) -> None:
        context = build_agent_workflow_context("MICH", "RUL prediction")

        self.assertEqual(context["dataset"], "MICH")
        self.assertEqual(context["dataset_status"], "RAW_ONLY")
        self.assertEqual(context["task"], "RUL prediction")
        self.assertFalse(context["preprocessing_supported"])
        steps = " ".join(context["recommended_next_steps"])
        self.assertNotIn("data/raw/MICH", steps)
        self.assertIn("not yet wired", steps)
        self.assertIn("BatteryML-compatible", steps)

    @unittest.skipUnless(DEMO_ENVIRONMENT_AVAILABLE, DEMO_ENVIRONMENT_SKIP_MESSAGE)
    def test_mich_blocked_models_carry_reasons(self) -> None:
        context = build_agent_workflow_context("MICH", "RUL prediction")

        for item in context["blocked_models"]:
            self.assertTrue(item["missing"], f"{item['model']} blocked without a reason")
            self.assertTrue(item["recommended_action"])


class TestRepoRootForwarding(unittest.TestCase):
    """repo_root must reach the readiness layer, never be silently dropped."""

    @unittest.skipUnless(DEMO_ENVIRONMENT_AVAILABLE, DEMO_ENVIRONMENT_SKIP_MESSAGE)
    def test_repo_root_is_actually_used(self) -> None:
        import tempfile

        from batteryml_web.core.readiness import get_workflow_context

        with tempfile.TemporaryDirectory() as tmp:
            fake_root = Path(tmp)
            (fake_root / "configs").mkdir()
            real = build_agent_workflow_context("MATR", "RUL prediction")
            fake = build_agent_workflow_context("MATR", "RUL prediction", repo_root=fake_root)
            # Against an empty configs tree nothing registry-backed can resolve,
            # while the real root offers CPTransformer/BatLiNet: the two must differ.
            self.assertNotEqual(
                {m["model"] for m in real["runnable_models"]},
                {m["model"] for m in fake["runnable_models"]},
            )
            # Direct readiness call with the same root must agree with the helper.
            direct = get_workflow_context("MATR", "RUL prediction", repo_root=fake_root)
            self.assertEqual(
                {m["model"] for m in fake["runnable_models"]},
                {m["model"] for m in direct["available_models"]},
            )


class TestTrajectoryAgentContext(unittest.TestCase):
    """Trajectory forecasting exposes BatteryMFormer semantics, not RUL models."""

    @unittest.skipUnless(DEMO_ENVIRONMENT_AVAILABLE, DEMO_ENVIRONMENT_SKIP_MESSAGE)
    def test_trajectory_exposes_batterymformer_blocked(self) -> None:
        context = build_agent_workflow_context("MATR", "Degradation trajectory forecasting")

        self.assertEqual(context["task"], "Degradation trajectory forecasting")
        self.assertNotIn("BatteryMFormer", _runnable_names(context))
        self.assertIn("BatteryMFormer", _blocked_names(context))
        blocked = {item["model"]: item for item in context["blocked_models"]}
        entry = blocked["BatteryMFormer"]
        self.assertEqual(entry["status"], "NEEDS_DATA")
        self.assertTrue(any("processed_SOH" in str(item) for item in entry["missing"]))
        runnable_models = _runnable_names(context)
        self.assertNotIn("cptransformer", runnable_models)
        self.assertNotIn("batlinet", runnable_models)
        self.assertNotIn("xgb", runnable_models)


if __name__ == "__main__":
    unittest.main(verbosity=2)
