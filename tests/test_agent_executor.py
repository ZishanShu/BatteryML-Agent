"""Backend-only tests for the deterministic execution controller.

No training, no real subprocess, no downloads: ``subprocess.run`` is
replaced by an injected ``runner`` stub, and dry-run mode never starts a
process at all. Registry-backed plan assertions skip where ``batteryml``
cannot be imported.

Run from the repository root:

    conda run -n batteryml python -m unittest tests.test_agent_executor -v
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tests.test_readiness import (  # noqa: E402
    REGISTRIES_AVAILABLE,
    REGISTRY_SKIP_MESSAGE,
)
from batteryml_web.agent.execution import build_execution_plan, plan_for_request  # noqa: E402
from batteryml_web.agent.executor import (  # noqa: E402
    discover_artifacts,
    execute_plan,
    validate_plan_for_execution,
)


def _runner_factory(calls: list, returncode: int = 0):
    def runner(argv, **kwargs):
        calls.append({"argv": list(argv), "kwargs": dict(kwargs)})
        return SimpleNamespace(
            returncode=returncode,
            stdout=f"<stdout rc={returncode}>",
            stderr="" if returncode == 0 else "<stderr boom>",
        )

    return runner


def _mutate(plan: dict, **overrides: object) -> dict:
    mutated = dict(plan)
    mutated.update(overrides)
    return mutated


class TestDryRunValidPlans(unittest.TestCase):
    """Dry-run validates and returns argv without starting a process."""

    @unittest.skipUnless(REGISTRIES_AVAILABLE, REGISTRY_SKIP_MESSAGE)
    def test_cptransformer_dry_run(self) -> None:
        calls: list = []
        plan = plan_for_request("Use CPTransformer on MATR to predict battery lifetime.")
        self.assertTrue(plan["can_execute"])

        result = execute_plan(plan, dry_run=True, runner=_runner_factory(calls))

        self.assertEqual(calls, [])
        self.assertFalse(result["started"])
        self.assertTrue(result["completed"])
        self.assertTrue(result["success"])
        self.assertIsNone(result["returncode"])
        self.assertEqual(result["command"], plan["command"])
        self.assertIsInstance(result["command"], list)

    @unittest.skipUnless(REGISTRIES_AVAILABLE, REGISTRY_SKIP_MESSAGE)
    def test_batlinet_dry_run(self) -> None:
        calls: list = []
        plan = plan_for_request("Use BatLiNet on MATR to predict battery lifetime.")
        self.assertTrue(plan["can_execute"])

        result = execute_plan(plan, dry_run=True, runner=_runner_factory(calls))

        self.assertEqual(calls, [])
        self.assertEqual(result["command"], plan["command"])
        self.assertTrue(result["success"])


class TestMockedExecution(unittest.TestCase):
    """Injected runner: honest success/failure propagation, no retry."""

    @unittest.skipUnless(REGISTRIES_AVAILABLE, REGISTRY_SKIP_MESSAGE)
    def test_mocked_success(self) -> None:
        import tempfile

        calls: list = []
        with tempfile.TemporaryDirectory(dir=str(REPO_ROOT / "runs")) as workspace:
            ws = Path(workspace)
            (ws / "predictions_seed_0.pkl").write_bytes(b"x")
            plan = plan_for_request("Use CPTransformer on MATR to predict battery lifetime.")
            plan = _mutate(plan, command=[*plan["command"][:5], str(ws), *plan["command"][6:]])

            result = execute_plan(plan, runner=_runner_factory(calls, returncode=0))

        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["argv"], result["command"])
        self.assertFalse(calls[0]["kwargs"].get("shell", True))
        self.assertTrue(result["started"])
        self.assertTrue(result["completed"])
        self.assertTrue(result["success"])
        self.assertEqual(result["returncode"], 0)
        self.assertEqual(result["stdout"], "<stdout rc=0>")
        self.assertIn(str(ws), str(result["workspace"]))

    @unittest.skipUnless(REGISTRIES_AVAILABLE, REGISTRY_SKIP_MESSAGE)
    def test_mocked_nonzero_preserved(self) -> None:
        calls: list = []
        plan = plan_for_request("Use CPTransformer on MATR to predict battery lifetime.")

        result = execute_plan(plan, runner=_runner_factory(calls, returncode=3))

        self.assertEqual(len(calls), 1)
        self.assertTrue(result["started"])
        self.assertTrue(result["completed"])
        self.assertFalse(result["success"])
        self.assertEqual(result["returncode"], 3)
        self.assertEqual(result["stdout"], "<stdout rc=3>")
        self.assertEqual(result["stderr"], "<stderr boom>")
        self.assertTrue(result["error"])
        self.assertEqual(
            result["artifacts"],
            {k: [] for k in result["artifacts"]},
        )


class TestBlockedPlansRejected(unittest.TestCase):
    """Blocked plans never reach subprocess."""

    @unittest.skipUnless(REGISTRIES_AVAILABLE, REGISTRY_SKIP_MESSAGE)
    def test_pbt_rejected(self) -> None:
        calls: list = []
        plan = plan_for_request("Use PBT on MATR to predict battery lifetime.")
        self.assertFalse(plan["can_execute"])

        result = execute_plan(plan, runner=_runner_factory(calls))

        self.assertEqual(calls, [])
        self.assertFalse(result["started"])
        self.assertFalse(result["success"])

    @unittest.skipUnless(REGISTRIES_AVAILABLE, REGISTRY_SKIP_MESSAGE)
    def test_mich_rejected(self) -> None:
        calls: list = []
        plan = plan_for_request("Use XGBoost on MICH to predict battery lifetime.")
        self.assertFalse(plan["can_execute"])

        result = execute_plan(plan, runner=_runner_factory(calls))

        self.assertEqual(calls, [])
        self.assertFalse(result["started"])
        self.assertNotIn("data/raw/MICH", str(result))

    @unittest.skipUnless(REGISTRIES_AVAILABLE, REGISTRY_SKIP_MESSAGE)
    def test_batterymformer_rejected(self) -> None:
        calls: list = []
        plan = plan_for_request("Forecast the degradation trajectory for MATR with BatteryMFormer.")
        self.assertFalse(plan["can_execute"])

        result = execute_plan(plan, runner=_runner_factory(calls))

        self.assertEqual(calls, [])
        self.assertFalse(result["started"])


class TestForgedPlansRejected(unittest.TestCase):
    """Validation rules catch mutated/forged plans before subprocess."""

    @unittest.skipUnless(REGISTRIES_AVAILABLE, REGISTRY_SKIP_MESSAGE)
    def test_forged_can_execute_with_invalid_config_rejected(self) -> None:
        calls: list = []
        plan = plan_for_request("Use CPTransformer on MATR to predict battery lifetime.")
        forged = _mutate(plan, config_valid=False, can_execute=True)

        result = execute_plan(forged, runner=_runner_factory(calls))

        self.assertEqual(calls, [])
        self.assertFalse(result["started"])

    @unittest.skipUnless(REGISTRIES_AVAILABLE, REGISTRY_SKIP_MESSAGE)
    def test_shell_string_command_rejected(self) -> None:
        calls: list = []
        plan = plan_for_request("Use CPTransformer on MATR to predict battery lifetime.")
        forged = _mutate(plan, command=" ".join(plan["command"]))

        result = execute_plan(forged, runner=_runner_factory(calls))

        self.assertEqual(calls, [])
        self.assertFalse(result["started"])
        self.assertIn("shell", str(result["error"]).lower())

    @unittest.skipUnless(REGISTRIES_AVAILABLE, REGISTRY_SKIP_MESSAGE)
    def test_config_mismatch_rejected(self) -> None:
        calls: list = []
        plan = plan_for_request("Use CPTransformer on MATR to predict battery lifetime.")
        forged_command = list(plan["command"])
        forged_command[3] = "configs/some_other_model.yaml"
        forged = _mutate(plan, command=forged_command)

        result = execute_plan(forged, runner=_runner_factory(calls))

        self.assertEqual(calls, [])
        self.assertFalse(result["started"])
        self.assertIn("config", str(result["error"]).lower())

    @unittest.skipUnless(REGISTRIES_AVAILABLE, REGISTRY_SKIP_MESSAGE)
    def test_workspace_outside_runs_rejected(self) -> None:
        calls: list = []
        plan = plan_for_request("Use CPTransformer on MATR to predict battery lifetime.")
        forged_command = list(plan["command"])
        index = forged_command.index("--workspace")
        forged_command[index + 1] = "/tmp/evil_workspace"
        forged = _mutate(plan, command=forged_command)

        result = execute_plan(forged, runner=_runner_factory(calls))

        self.assertEqual(calls, [])
        self.assertFalse(result["started"])


class TestArtifactDiscovery(unittest.TestCase):
    """Artifact paths are reported only for files that really exist."""

    def test_missing_workspace_yields_empty(self) -> None:
        artifacts = discover_artifacts("runs/__definitely_no_such_workspace__")

        self.assertEqual(
            artifacts,
            {k: [] for k in artifacts},
        )

    def test_existing_files_discovered(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory(dir=str(REPO_ROOT / "runs")) as workspace:
            ws = Path(workspace)
            (ws / "metrics.csv").write_text("Metric,Value\nRMSE,1.0\n", encoding="utf-8")
            (ws / "predictions_seed_0.pkl").write_bytes(b"x")
            (ws / "run_config.yaml").write_text("model: {}\n", encoding="utf-8")

            artifacts = discover_artifacts(workspace)

            self.assertTrue(any(p.endswith("metrics.csv") for p in artifacts["metrics_files"]))
            self.assertTrue(artifacts["prediction_files"])
            self.assertTrue(artifacts["config_files"])
            for paths in artifacts.values():
                for path in paths:
                    self.assertTrue(Path(path).is_file())

    def test_validate_plan_ok_shape(self) -> None:
        plan = build_execution_plan.__doc__  # sanity: planner API present
        self.assertTrue(plan)
        checked = validate_plan_for_execution({"can_execute": False})
        self.assertFalse(checked["ok"])
        self.assertTrue(checked["error"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
