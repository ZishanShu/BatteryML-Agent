"""Run identity, manifests, history and real cancellation.

These tests use the real filesystem (in temporary directories) and, for
cancellation, real subprocesses: a cancel that only flips a label would pass
any test built on a mock, and a fake Cancel button is precisely the failure
this milestone exists to prevent. Nothing here trains a model, imports
Streamlit, or needs a GPU.

Run from the repository root:

    conda run -n batteryml python -m unittest tests.test_run_lifecycle -v
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from batteryml_web.core import run_manifest as rm  # noqa: E402
from batteryml_web.core.project_context import (  # noqa: E402
    agent_workspace,
    resolve_project,
    run_workspace,
)
from batteryml_web.core.run_identity import (  # noqa: E402
    describe_run_id,
    is_run_id,
    new_run_id,
    safe_token,
)
from batteryml_web.core.run_manager import RunHandle, RunManager  # noqa: E402


class TestRunIdentity(unittest.TestCase):
    """A run id names an execution, not a configuration."""

    def test_repeating_the_same_configuration_still_makes_distinct_ids(self) -> None:
        ids = {
            new_run_id(dataset="MATR", model="batlinet", seed=0)
            for _ in range(200)
        }
        self.assertEqual(len(ids), 200)

    def test_ids_of_the_same_second_are_still_distinct(self) -> None:
        fixed = datetime(2026, 9, 16, 18, 57, 42, tzinfo=timezone.utc)
        first = new_run_id(dataset="MATR", model="cnn", seed=0, now=fixed)
        second = new_run_id(dataset="MATR", model="cnn", seed=0, now=fixed)
        self.assertNotEqual(first, second)
        self.assertTrue(first.startswith("run_20260916_185742_cnn_matr_seed0_"))

    def test_the_id_is_safe_as_a_single_path_segment(self) -> None:
        run_id = new_run_id(dataset="../../etc", model="a/b c", seed="../x")
        for bad in ("/", "\\", "..", " ", ":"):
            self.assertNotIn(bad, run_id)
        self.assertTrue(is_run_id(run_id))

    def test_seed_zero_is_recorded_as_zero_not_as_missing(self) -> None:
        """Seed 0 is the platform default; a falsy test would lose it."""
        self.assertEqual(safe_token(0), "0")
        run_id = new_run_id(dataset="MATR", model="xgb", seed=0)
        self.assertIn("_seed0_", run_id)
        self.assertEqual(describe_run_id(run_id)["seed"], "0")

    def test_a_foreign_directory_name_is_not_mistaken_for_a_run(self) -> None:
        for name in ("agent_matr_batlinet", "checkpoints", "", "run_x"):
            self.assertFalse(is_run_id(name))


class TestRunWorkspaceStaysInsideTheProject(unittest.TestCase):
    """A2: every run belongs to the loaded project, under its own directory."""

    def test_each_run_gets_its_own_directory_under_the_model_workspace(self) -> None:
        project = resolve_project(project_name="alpha")
        base = agent_workspace(project, "MATR", "batlinet")
        first = run_workspace(project, "MATR", "batlinet", "run_a")
        second = run_workspace(project, "MATR", "batlinet", "run_b")
        self.assertTrue(first.startswith(base + "/"))
        self.assertNotEqual(first, second)

    def test_alpha_never_writes_into_beta(self) -> None:
        run_id = new_run_id(dataset="MATR", model="batlinet", seed=0)
        alpha = run_workspace(resolve_project(project_name="alpha"), "MATR", "batlinet", run_id)
        beta = run_workspace(resolve_project(project_name="beta"), "MATR", "batlinet", run_id)
        self.assertIn("/alpha/", alpha)
        self.assertNotIn("/beta/", alpha)
        self.assertIn("/beta/", beta)
        self.assertNotIn("/alpha/", beta)

    def test_a_traversing_run_id_cannot_escape_the_project(self) -> None:
        project = resolve_project(project_name="alpha")
        path = run_workspace(project, "MATR", "batlinet", "../../../etc/passwd")
        self.assertTrue(path.startswith("runs/web_projects/alpha/"))
        self.assertNotIn("..", path)


class _ManifestCase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.project_root = Path(self._tmp.name) / "runs" / "web_projects" / "alpha"
        self.addCleanup(self._tmp.cleanup)

    def _make(self, run_id: str, **kwargs) -> Path:
        run_dir = self.project_root / "agent_matr_batlinet" / run_id
        manifest = rm.build_manifest(
            run_id=run_id,
            project="alpha",
            workspace=str(run_dir),
            dataset="MATR",
            task="RUL prediction",
            model_key="batlinet",
            seed=0,
            **kwargs,
        )
        rm.write_manifest(run_dir, manifest)
        return run_dir


class TestManifestPersistence(_ManifestCase):
    """A3: the record survives the process that wrote it."""

    def test_a_manifest_round_trips(self) -> None:
        run_dir = self._make("run_1")
        loaded = rm.read_manifest(run_dir)
        self.assertEqual(loaded["run_id"], "run_1")
        self.assertEqual(loaded["status"], rm.STATUS_PLANNED)
        self.assertEqual(loaded["seed"], "0")

    def test_the_write_is_atomic_and_leaves_no_scratch_file(self) -> None:
        run_dir = self._make("run_1")
        for index in range(20):
            rm.update_manifest(run_dir, {"elapsed_seconds": index})
            # A reader between writes always sees whole JSON.
            json.loads((run_dir / rm.MANIFEST_NAME).read_text(encoding="utf-8"))
        leftovers = [p.name for p in run_dir.iterdir() if p.name.startswith(".run-manifest-")]
        self.assertEqual(leftovers, [])

    def test_a_truncated_manifest_is_reported_missing_not_raised(self) -> None:
        run_dir = self._make("run_1")
        (run_dir / rm.MANIFEST_NAME).write_text('{"run_id": "run_1"', encoding="utf-8")
        self.assertIsNone(rm.read_manifest(run_dir))
        # ...and one damaged run does not take the history down with it.
        self.assertEqual(rm.list_runs(self.project_root), [])

    def test_no_secret_field_can_reach_the_manifest(self) -> None:
        run_dir = self._make("run_1")
        text = (run_dir / rm.MANIFEST_NAME).read_text(encoding="utf-8")
        for secret in ("api_key", "llm_api_key", "token", "password"):
            self.assertNotIn(secret, text.lower())


class TestStatusModel(_ManifestCase):
    """A4: status is recorded and its transitions are checked."""

    def test_the_happy_path_is_planned_running_completed(self) -> None:
        run_dir = self._make("run_1")
        rm.update_manifest(run_dir, {}, status=rm.STATUS_RUNNING)
        rm.update_manifest(run_dir, {}, status=rm.STATUS_COMPLETED)
        self.assertEqual(rm.read_manifest(run_dir)["status"], rm.STATUS_COMPLETED)

    def test_running_may_fail_or_be_cancelled(self) -> None:
        for target in (rm.STATUS_FAILED, rm.STATUS_CANCELLED, rm.STATUS_INTERRUPTED):
            with self.subTest(target=target):
                self.assertTrue(rm.can_transition(rm.STATUS_RUNNING, target))

    def test_a_terminal_run_cannot_be_reopened(self) -> None:
        run_dir = self._make("run_1")
        rm.update_manifest(run_dir, {}, status=rm.STATUS_RUNNING)
        rm.update_manifest(run_dir, {}, status=rm.STATUS_CANCELLED)
        with self.assertRaises(ValueError):
            rm.update_manifest(run_dir, {}, status=rm.STATUS_RUNNING)
        with self.assertRaises(ValueError):
            rm.update_manifest(run_dir, {}, status=rm.STATUS_COMPLETED)

    def test_planned_can_never_jump_straight_to_completed(self) -> None:
        self.assertFalse(rm.can_transition(rm.STATUS_PLANNED, rm.STATUS_COMPLETED))

    def test_a_stale_running_manifest_becomes_interrupted_not_completed(self) -> None:
        """A8/C6: the app restarted; this run cannot be confirmed alive."""
        run_dir = self._make("run_1")
        rm.update_manifest(run_dir, {}, status=rm.STATUS_RUNNING)
        listed = rm.list_runs(self.project_root, live_run_ids=[])
        self.assertEqual(listed[0]["status"], rm.STATUS_INTERRUPTED)
        self.assertNotEqual(listed[0]["status"], rm.STATUS_COMPLETED)
        self.assertEqual(listed[0]["reconciled_from"], rm.STATUS_RUNNING)
        # Reporting is not rewriting: the file still says RUNNING.
        self.assertEqual(rm.read_manifest(run_dir)["status"], rm.STATUS_RUNNING)

    def test_a_tracked_running_manifest_is_still_reported_running(self) -> None:
        run_dir = self._make("run_1")
        rm.update_manifest(run_dir, {}, status=rm.STATUS_RUNNING)
        listed = rm.list_runs(self.project_root, live_run_ids=["run_1"])
        self.assertEqual(listed[0]["status"], rm.STATUS_RUNNING)


class TestRunHistory(_ManifestCase):
    """A7: history comes from the manifests on disk."""

    def test_history_is_newest_first(self) -> None:
        for index, run_id in enumerate(("run_a", "run_b", "run_c")):
            run_dir = self._make(run_id)
            rm.update_manifest(run_dir, {"created_at": f"2026-09-1{index}T10:00:00+00:00"})
        order = [entry["run_id"] for entry in rm.list_runs(self.project_root)]
        self.assertEqual(order, ["run_c", "run_b", "run_a"])

    def test_history_survives_a_completely_fresh_process_state(self) -> None:
        """Nothing is read from memory: a new call sees the same runs."""
        self._make("run_1")
        self._make("run_2")
        first = rm.list_runs(self.project_root)
        second = rm.list_runs(self.project_root)
        self.assertEqual(len(first), 2)
        self.assertEqual(
            [e["run_id"] for e in first], [e["run_id"] for e in second]
        )

    def test_one_project_never_lists_another_projects_runs(self) -> None:
        self._make("run_alpha")
        beta_root = self.project_root.parent / "beta"
        rm.write_manifest(
            beta_root / "agent_matr_batlinet" / "run_beta",
            rm.build_manifest(
                run_id="run_beta", project="beta", workspace="w", dataset="MATR",
                task="RUL prediction", model_key="batlinet", seed=0,
            ),
        )
        alpha_ids = [e["run_id"] for e in rm.list_runs(self.project_root)]
        beta_ids = [e["run_id"] for e in rm.list_runs(beta_root)]
        self.assertEqual(alpha_ids, ["run_alpha"])
        self.assertEqual(beta_ids, ["run_beta"])

    def test_a_named_run_is_found_only_inside_its_own_project(self) -> None:
        self._make("run_alpha")
        self.assertIsNotNone(rm.find_run(self.project_root, "run_alpha"))
        self.assertIsNone(rm.find_run(self.project_root, "run_beta"))

    def test_history_names_no_model_or_metric_of_its_own(self) -> None:
        """A7: every displayed value comes from the manifest."""
        run_dir = self._make("run_1", model_display="BatLiNet — Inter-cell")
        rm.update_manifest(run_dir, {}, status=rm.STATUS_RUNNING)
        rm.update_manifest(
            run_dir, {"metrics": {"RMSE": 716.48}}, status=rm.STATUS_COMPLETED
        )
        entry = rm.list_runs(self.project_root)[0]
        self.assertEqual(entry["metrics"]["RMSE"], 716.48)
        source = Path(rm.__file__).read_text(encoding="utf-8")
        for hardcoded in ("BatLiNet", "CPTransformer", "RMSE", "716"):
            self.assertNotIn(hardcoded, source)


def _process_is_running(pid: int) -> bool:
    """Whether ``pid`` is a *live* process, as opposed to a zombie.

    ``os.kill(pid, 0)`` is the obvious check and the wrong one: a killed
    child whose parent has died stays signalable until something reaps it,
    so that test passes for a process that is already dead. Linux reports the
    real answer in ``/proc/<pid>/stat``, where state ``Z`` is a zombie.
    """
    try:
        os.kill(pid, 0)
    except (ProcessLookupError, PermissionError):
        return False
    stat_path = Path(f"/proc/{pid}/stat")
    if not stat_path.exists():
        return False
    try:
        return stat_path.read_text(encoding="utf-8").split()[2] != "Z"
    except (OSError, IndexError):
        return False


class TestRealCancellation(unittest.TestCase):
    """A5: Cancel kills a real process, and only that process.

    Built on actual subprocesses. A mock would happily accept a Cancel that
    changes a label and leaves the training running, which is the exact
    defect this must rule out.
    """

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.run_dir = Path(self._tmp.name) / "run_1"
        self.run_dir.mkdir(parents=True)
        self.addCleanup(self._tmp.cleanup)
        self.manager = RunManager()

    def _spawn(self, seconds: int = 120) -> subprocess.Popen:
        process = subprocess.Popen(
            [sys.executable, "-c", f"import time; time.sleep({seconds})"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=False,
            start_new_session=True,
        )
        self.addCleanup(self._reap, process)
        return process

    @staticmethod
    def _reap(process: subprocess.Popen) -> None:
        if process.poll() is None:
            process.kill()
        try:
            process.wait(timeout=5)
        except Exception:
            pass

    def _register(self, run_id: str, process: subprocess.Popen) -> RunHandle:
        handle = RunHandle(run_id, self.run_dir, {})
        handle.status = rm.STATUS_RUNNING
        handle._process = process
        self.manager._runs[run_id] = handle
        return handle

    def test_cancel_actually_terminates_the_process(self) -> None:
        process = self._spawn()
        self._register("run_1", process)
        self.assertIsNone(process.poll(), "precondition: the child is running")

        self.assertTrue(self.manager.cancel("run_1", grace_s=10))
        self.assertIsNotNone(process.poll(), "the child must really be dead")

    def test_cancel_targets_only_the_requested_run(self) -> None:
        victim = self._spawn()
        bystander = self._spawn()
        self._register("victim", victim)
        self._register("bystander", bystander)

        self.manager.cancel("victim", grace_s=10)
        self.assertIsNotNone(victim.poll(), "the requested run must stop")
        self.assertIsNone(
            bystander.poll(), "another run must be untouched by a cancel"
        )

    def test_a_child_that_ignores_sigterm_is_still_stopped(self) -> None:
        stubborn = subprocess.Popen(
            [
                sys.executable,
                "-c",
                "import signal, time\n"
                "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
                "time.sleep(120)\n",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=False,
            start_new_session=True,
        )
        self.addCleanup(self._reap, stubborn)
        time.sleep(0.8)  # let the handler install
        self._register("run_1", stubborn)
        self.manager.cancel("run_1", grace_s=2)
        self.assertIsNotNone(stubborn.poll(), "SIGKILL must follow SIGTERM")

    def test_cancelling_an_unknown_run_is_a_no_op(self) -> None:
        self.assertFalse(self.manager.cancel("nobody"))

    def test_the_whole_process_group_of_the_run_is_signalled(self) -> None:
        """A child's own children must not survive its cancellation."""
        parent = subprocess.Popen(
            [
                sys.executable,
                "-c",
                "import subprocess, sys, time\n"
                "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(120)'])\n"
                "print(child.pid, flush=True)\n"
                "time.sleep(120)\n",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=False,
            start_new_session=True,
        )
        self.addCleanup(self._reap, parent)
        grandchild_pid = int(parent.stdout.readline().decode().strip())
        self.assertEqual(
            os.getpgid(grandchild_pid),
            os.getpgid(parent.pid),
            "precondition: the grandchild shares the run's process group",
        )
        self._register("run_1", parent)
        self.manager.cancel("run_1", grace_s=5)
        time.sleep(1.0)
        self.assertFalse(
            _process_is_running(grandchild_pid),
            "the grandchild should have been signalled with the group",
        )


class TestNoUnsafeProcessHandling(unittest.TestCase):
    """The blast radius of a cancel is one process group, never a pattern."""

    SOURCES = (
        "batteryml_web/core/run_manager.py",
        "batteryml_web/agent/streaming.py",
        "batteryml_web/agent/executor.py",
    )

    def _text(self) -> str:
        """The run path's code, with docstrings dropped.

        Scanned through the AST rather than as raw text: these modules
        document *why* they never use a broad kill, and reading that prose
        as behaviour would fail the test for explaining itself.
        """
        import ast

        chunks: list[str] = []
        for name in self.SOURCES:
            tree = ast.parse((REPO_ROOT / name).read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(
                    node,
                    (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef),
                ):
                    body = getattr(node, "body", [])
                    if (
                        body
                        and isinstance(body[0], ast.Expr)
                        and isinstance(body[0].value, ast.Constant)
                        and isinstance(body[0].value.value, str)
                    ):
                        node.body = body[1:] or [ast.Pass()]
            chunks.append(ast.unparse(tree))
        return "\n".join(chunks)

    def test_no_shell_execution_anywhere_on_the_run_path(self) -> None:
        text = self._text()
        self.assertNotIn("shell=True", text)
        self.assertNotIn("os.system", text)

    def test_no_broad_process_killing(self) -> None:
        text = self._text()
        for forbidden in ("pkill", "killall", "taskkill", "kill -9"):
            self.assertNotIn(forbidden, text)

    def test_the_run_manager_imports_no_streamlit(self) -> None:
        import ast

        tree = ast.parse(
            (REPO_ROOT / "batteryml_web/core/run_manager.py").read_text(encoding="utf-8")
        )
        imported: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported += [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                imported.append(str(node.module))
        self.assertNotIn("streamlit", imported)
        self.assertFalse([name for name in imported if "streamlit" in name])


if __name__ == "__main__":
    unittest.main()
