"""Run Model and Home-demo runs appear in Review Results.

The reported gap: Review Results lists a project's runs from the ``run.json``
files under it (``run_manifest.list_runs``), and only Agent Workflow wrote
one. A run started from **Run Model → Run BatteryML** or **Home → Run MATR
Demo** was therefore never listed — even though Home sends the user to Review
Results right after the demo. ``batteryml_web/core/direct_runs.py`` gives
those runs a folder and a manifest of their own.

Pure filesystem tests on a temporary folder; nothing runs BatteryML.

Run from the repository root::

    python -m unittest tests.test_direct_runs -v
"""

from __future__ import annotations

import ast
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from batteryml_web.core import direct_runs as dr  # noqa: E402
from batteryml_web.core import run_manifest as rm  # noqa: E402
from batteryml_web.core.run_identity import new_run_id  # noqa: E402

APP_PATH = REPO_ROOT / "batteryml_web" / "app.py"


class _Case(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="direct-runs-"))
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.project = self.tmp / "runs" / "web_projects" / "my_battery_project"

    def begin(self, source: str = "run_model", run_id: str | None = None) -> Path:
        run_id = run_id or new_run_id(dataset="MATR", model="xgb", seed=0)
        run_dir = dr.direct_run_dir(self.project, source, run_id)
        dr.begin_direct_run(
            run_dir,
            workspace=f"runs/web_projects/my_battery_project/{source}/{run_id}",
            run_id=run_id,
            project="my_battery_project",
            dataset="MATR",
            task="RUL prediction",
            model_key="xgb",
            model_display="XGBoost — Extreme Gradient Boosting",
            config_path="configs/baselines/sklearn/xgb/matr_1.yaml",
            seed=0,
            device="cpu",
            command=["python", "batteryml_web/core/run_batteryml_cli.py", "run"],
            git_commit="abc123",
        )
        return run_dir


class TestTheRunIsListed(_Case):
    def test_a_completed_run_model_run_is_in_the_project_history(self) -> None:
        run_dir = self.begin()
        dr.finish_direct_run(
            run_dir,
            return_code=0,
            elapsed_seconds=12.3456,
            output="Seed is set to 0.\n",
            artifacts={"prediction_files": [str(run_dir / "predictions_seed_0_x.pkl")]},
            metrics={"RMSE": 101.5},
        )
        runs = rm.list_runs(self.project)
        self.assertEqual(len(runs), 1)
        run = runs[0]
        self.assertEqual(run["status"], rm.STATUS_COMPLETED)
        self.assertEqual(run["metrics"], {"RMSE": 101.5})
        self.assertEqual(run["return_code"], 0)
        self.assertEqual(run["elapsed_seconds"], 12.346)
        self.assertEqual(run["dataset"], "MATR")
        self.assertEqual(run["model_key"], "xgb")
        self.assertEqual(run["seed"], "0")
        self.assertEqual(run["git_commit"], "abc123")
        self.assertEqual(
            run["artifacts"]["prediction_files"], [str(run_dir / "predictions_seed_0_x.pkl")]
        )
        self.assertEqual(
            (run_dir / dr.OUTPUT_LOG).read_text(encoding="utf-8"), "Seed is set to 0.\n"
        )

    def test_the_demo_and_run_model_runs_are_listed_side_by_side(self) -> None:
        first = self.begin("quickstart")
        dr.finish_direct_run(first, return_code=0, elapsed_seconds=1, output="", metrics={})
        second = self.begin("run_model")
        dr.finish_direct_run(second, return_code=0, elapsed_seconds=1, output="", metrics={})
        dirs = {Path(run["run_dir"]).parent.name for run in rm.list_runs(self.project)}
        self.assertEqual(dirs, {"quickstart", "run_model"})

    def test_two_runs_of_the_same_selection_do_not_overwrite_each_other(self) -> None:
        """The old shared <project>/workspace folder made this impossible."""
        for _ in range(2):
            run_dir = self.begin()
            dr.finish_direct_run(run_dir, return_code=0, elapsed_seconds=1, output="")
        self.assertEqual(len(rm.list_runs(self.project)), 2)


class TestHonestOutcomes(_Case):
    def test_a_failed_run_is_failed_and_has_no_metrics(self) -> None:
        run_dir = self.begin()
        dr.finish_direct_run(
            run_dir,
            return_code=1,
            elapsed_seconds=3,
            output="AssertionError: data/processed/MATR\n",
            artifacts={"prediction_files": ["x.pkl"]},
            metrics={"RMSE": 1.0},
        )
        run = rm.list_runs(self.project)[0]
        self.assertEqual(run["status"], rm.STATUS_FAILED)
        self.assertEqual(run["metrics"], {})
        self.assertEqual(run["artifacts"], {})
        self.assertIn("exited with code 1", run["failure"]["message"])
        self.assertIn(dr.OUTPUT_LOG, run["failure"]["message"])

    def test_a_run_that_never_finished_is_reported_interrupted(self) -> None:
        """A closed tab or a crash mid-run: recorded, not invented."""
        self.begin()
        run = rm.list_runs(self.project, live_run_ids=())[0]
        self.assertEqual(run["status"], rm.STATUS_INTERRUPTED)

    def test_the_record_exists_before_the_run_starts(self) -> None:
        run_dir = self.begin()
        manifest = rm.read_manifest(run_dir)
        self.assertEqual(manifest["status"], rm.STATUS_RUNNING)
        self.assertIsNotNone(manifest["started_at"])

    def test_a_metrics_read_failure_is_recorded_not_hidden(self) -> None:
        run_dir = self.begin()
        dr.finish_direct_run(
            run_dir, return_code=0, elapsed_seconds=1, output="",
            metrics={}, metrics_error="KeyError: 'scores'",
        )
        manifest = rm.read_manifest(run_dir)
        self.assertEqual(manifest["status"], rm.STATUS_COMPLETED)
        self.assertEqual(manifest["metrics_error"], "KeyError: 'scores'")

    def test_a_recorded_run_is_never_overwritten(self) -> None:
        run_id = new_run_id(dataset="MATR", model="xgb", seed=0)
        self.begin(run_id=run_id)
        with self.assertRaises(FileExistsError):
            self.begin(run_id=run_id)


class TestRunFolders(unittest.TestCase):
    def test_the_folder_layout(self) -> None:
        run_id = new_run_id(dataset="MATR", model="xgb", seed=0)
        self.assertEqual(
            dr.direct_run_dir("runs/web_projects/p", "run_model", run_id).as_posix(),
            f"runs/web_projects/p/run_model/{run_id}",
        )

    def test_an_unknown_source_or_a_foreign_id_is_refused(self) -> None:
        run_id = new_run_id(dataset="MATR", model="xgb", seed=0)
        with self.assertRaises(ValueError):
            dr.direct_run_dir("p", "agent", run_id)
        for bad in ("../escape", "workspace", "", "run_x/../../y"):
            with self.subTest(run_id=bad), self.assertRaises(ValueError):
                dr.direct_run_dir("p", "run_model", bad)


class TestReservedRunId(unittest.TestCase):
    SELECTION = dict(dataset="MATR", model="xgb", seed=0, device="cpu",
                     config_path="configs/baselines/sklearn/xgb/matr_1.yaml")

    def test_stable_across_reruns_of_the_same_selection(self) -> None:
        session: dict = {}
        first = dr.reserve_run_id(session, key="k", **self.SELECTION)
        second = dr.reserve_run_id(session, key="k", **self.SELECTION)
        self.assertEqual(first, second)

    def test_a_changed_selection_gets_a_new_id(self) -> None:
        session: dict = {}
        first = dr.reserve_run_id(session, key="k", **self.SELECTION)
        changed = dict(self.SELECTION, seed=1)
        second = dr.reserve_run_id(session, key="k", **changed)
        self.assertNotEqual(first, second)
        self.assertIn("seed1", second)

    def test_release_makes_the_next_run_a_new_run(self) -> None:
        session: dict = {}
        first = dr.reserve_run_id(session, key="k", **self.SELECTION)
        dr.release_run_id(session, key="k")
        self.assertNotEqual(first, dr.reserve_run_id(session, key="k", **self.SELECTION))


class TestBothButtonsRecordTheirRuns(unittest.TestCase):
    """Static: begin before the run, finish after it, at both call sites."""

    def _calls_in_order(self, function_name: str) -> list[str]:
        tree = ast.parse(APP_PATH.read_text(encoding="utf-8"))
        function = next(
            node for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name == function_name
        )
        calls = [
            (node.lineno, node.col_offset, node.func.id)
            for node in ast.walk(function)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
            and node.func.id in {"_begin_page_run", "run_command_with_progress", "_finish_page_run"}
        ]
        return [name for _, _, name in sorted(calls)]

    def test_run_model_page(self) -> None:
        self.assertEqual(
            self._calls_in_order("render_run_model_page"),
            ["_begin_page_run", "run_command_with_progress", "_finish_page_run"],
        )

    def test_home_demo(self) -> None:
        self.assertEqual(
            self._calls_in_order("render_home_page"),
            ["_begin_page_run", "run_command_with_progress", "_finish_page_run"],
        )

    def test_no_run_is_written_to_the_old_shared_workspace(self) -> None:
        source = APP_PATH.read_text(encoding="utf-8")
        self.assertNotIn('/ "workspace"', source)


if __name__ == "__main__":
    unittest.main()
