"""Backend-only tests for the planner-owned --epochs smoke override.

No training, no subprocess, no downloads. ``epochs`` lives in the planner
(:func:`build_execution_plan` / :func:`plan_for_request`); the executor only
validates the resulting argv deterministically.

Run from the repository root:

    conda run -n batteryml python -m unittest tests.test_agent_epochs -v
"""

from __future__ import annotations

import sys
import unittest

REPO_ROOT = __import__("pathlib").Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tests.test_readiness import (  # noqa: E402
    REGISTRIES_AVAILABLE,
    REGISTRY_SKIP_MESSAGE,
)
from batteryml_web.agent.execution import build_execution_plan, plan_for_request  # noqa: E402
from batteryml_web.agent.executor import execute_plan  # noqa: E402


class TestEpochsOverride(unittest.TestCase):
    @unittest.skipUnless(REGISTRIES_AVAILABLE, REGISTRY_SKIP_MESSAGE)
    def test_none_preserves_old_behavior(self) -> None:
        plan = plan_for_request("Use CPTransformer on MATR to predict battery lifetime.")
        self.assertTrue(plan["can_execute"])
        self.assertNotIn("--epochs", plan["command"])
        self.assertIsNone(plan.get("epochs"))

    @unittest.skipUnless(REGISTRIES_AVAILABLE, REGISTRY_SKIP_MESSAGE)
    def test_epochs_one_appends_flag(self) -> None:
        plan = plan_for_request(
            "Use CPTransformer on MATR to predict battery lifetime.",
            epochs=1,
            seed=0,
            workspace="runs/agent_smoke_cptransformer",
        )
        self.assertTrue(plan["can_execute"])
        command = plan["command"]
        self.assertIn("--train", command)
        self.assertIn("--eval", command)
        self.assertIn("--epochs", command)
        self.assertEqual(command[command.index("--epochs") + 1], "1")
        self.assertEqual(plan.get("epochs"), 1)
        # Workspace honored exactly.
        self.assertIn("runs/agent_smoke_cptransformer", command)

    @unittest.skipUnless(REGISTRIES_AVAILABLE, REGISTRY_SKIP_MESSAGE)
    def test_build_execution_plan_epochs(self) -> None:
        plan = build_execution_plan("MATR", "RUL prediction", "cptransformer", epochs=2)
        self.assertTrue(plan["can_execute"])
        self.assertIn("--epochs", plan["command"])
        self.assertEqual(plan["command"][plan["command"].index("--epochs") + 1], "2")

    def test_rejections(self) -> None:
        for bad in (0, -1, True, False, "1", 1.5, [1]):
            with self.subTest(epochs=bad):
                with self.assertRaises(ValueError):
                    plan_for_request(
                        "Use CPTransformer on MATR to predict battery lifetime.",
                        epochs=bad,  # type: ignore[arg-type]
                    )

    @unittest.skipUnless(REGISTRIES_AVAILABLE, REGISTRY_SKIP_MESSAGE)
    def test_executor_accepts_valid_epochs_dry_run(self) -> None:
        calls: list = []
        plan = plan_for_request(
            "Use CPTransformer on MATR to predict battery lifetime.",
            epochs=1,
            workspace="runs/agent_smoke_cptransformer",
        )

        def runner(argv, **kwargs):
            calls.append(argv)
            raise AssertionError("dry-run must not call subprocess")

        result = execute_plan(plan, dry_run=True, runner=runner)
        self.assertEqual(calls, [])
        self.assertTrue(result["success"])
        self.assertIn("--epochs", result["command"])
        self.assertEqual(result["command"][result["command"].index("--epochs") + 1], "1")

    def test_no_arbitrary_flag_override(self) -> None:
        import inspect

        from batteryml_web.agent import execution as execution_module

        source = inspect.getsource(execution_module.build_execution_plan)
        # The planner exposes exactly one flag override: epochs.
        self.assertIn("epochs", source)
        sig = inspect.signature(execution_module.build_execution_plan)
        self.assertNotIn("extra_flags", sig.parameters)
        self.assertNotIn("cli_flags", sig.parameters)
        self.assertNotIn("extra_args", sig.parameters)


if __name__ == "__main__":
    unittest.main(verbosity=2)
