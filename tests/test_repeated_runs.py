"""Running a second time must be as ordinary as running the first.

The defect these tests exist for: the page executed once and then offered no
way to execute again. Two independent causes, both fixed and both pinned
below.

1. **The lifecycle could be wedged by a reporting failure.**
   ``_finish_active_run`` used to read the run's artifacts first and clear
   ``ACTIVE_RUN_KEY`` last. Anything that raised in between reached the page
   error boundary with the key still set, so every later script run re-entered
   the same failing branch — and the Run button is drawn only when no run is
   active. Now the run is released before any fallible work.

2. **The staleness guard fired unconditionally.**
   It compared the plan's argv, which carries ``--workspace .../<run_id>``,
   against the argv stored at launch. The reserved run id is re-minted on the
   rerun that follows every launch, so the two could never match again and the
   finished run's results were deleted before the page drew them. The
   signature is now taken from the experiment, not from the run's identity.

The end-to-end cases use a **real** ``RunManager``, a real subprocess and a
real manifest on disk, in a temporary repository root. A stub manager cannot
show that a terminal run is reaped while its worker thread is still finishing,
and that window is precisely where the reaping used to silently do nothing.

The child process deliberately emits no progress events at all — it is the
sklearn shape (a Ridge fit reports no epochs), so these tests also cover
"terminal detection must not depend on the event stream".

Run from the repository root::

    conda run -n batteryml python -m unittest tests.test_repeated_runs -v
"""

from __future__ import annotations

import ast
import json
import sys
import tempfile
import textwrap
import time
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from batteryml_web.core import run_manifest as rm  # noqa: E402
from batteryml_web.core.run_identity import new_run_id  # noqa: E402
from batteryml_web.core.run_manager import RunManager  # noqa: E402
from batteryml_web.ui import agent_workflow as ui  # noqa: E402

from tests.test_agent_web_ui import (  # noqa: E402
    FakeStreamlit,
    _fake_streamlit,
    _Rerun,
)

#: What the validator demands the second argv element be.
WRAPPER = "batteryml_web/core/run_batteryml_cli.py"

#: Long enough that the page observes a RUNNING snapshot before the terminal
#: one, short enough that the suite stays fast.
CHILD_SLEEP_S = 0.4

TIMEOUT_S = 30.0


def _write_fake_repo(root: Path) -> Path:
    """A minimal tree the plan validator accepts, with a harmless wrapper.

    The wrapper is a stand-in, not a mock of BatteryML: the point of these
    tests is the *lifecycle around* an execution, so the child only has to be
    a real process that really exits. It prints one line and no progress
    counter, which is the sklearn shape.
    """
    (root / "runs").mkdir(parents=True, exist_ok=True)
    (root / "configs").mkdir(parents=True, exist_ok=True)
    wrapper = root / WRAPPER
    wrapper.parent.mkdir(parents=True, exist_ok=True)
    wrapper.write_text(
        textwrap.dedent(
            f"""
            import sys, time
            time.sleep({CHILD_SLEEP_S})
            print("fit complete")
            sys.exit(0)
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )
    config = root / "configs" / "ridge.yaml"
    config.write_text("model:\n  name: 'RidgeRULPredictor'\n", encoding="utf-8")
    return config


def _plan(config: Path, workspace: str) -> dict:
    """An executable plan of the exact shape build_execution_plan produces."""
    return {
        "dataset": "MATR",
        "task": "SOH estimation",
        "model": "ridge",
        "model_display": "Ridge",
        "config_path": str(config),
        "config_valid": True,
        "model_status": "READY",
        "dataset_status": "READY",
        "can_execute": True,
        "epochs": None,
        "command_workspace": workspace,
        "command": [
            sys.executable,
            WRAPPER,
            "run",
            str(config),
            "--workspace",
            workspace,
            "--train",
            "--eval",
            "--device",
            "cpu",
            "--seed",
            "0",
        ],
    }


def _run_to_completion(manager: RunManager, *, root: Path, config: Path,
                       run_id: str) -> dict:
    """Start one run through the real manager and wait for a terminal status.

    Returns the last snapshot. Polls the *status*, never the event stream:
    that is the contract a model reporting no progress has to satisfy.
    """
    workspace = f"runs/agent_matr_ridge/{run_id}"
    run_dir = root / workspace
    plan = _plan(config, workspace)
    rm.write_manifest(
        run_dir,
        rm.build_manifest(
            run_id=run_id,
            project="alpha",
            workspace=workspace,
            dataset="MATR",
            task="SOH estimation",
            model_key="ridge",
            config_path=str(config),
            seed=0,
            device="cpu",
            command=plan["command"],
        ),
    )
    manager.start(run_id=run_id, run_dir=run_dir, plan=plan, repo_root=root)
    deadline = time.monotonic() + TIMEOUT_S
    snapshot = manager.snapshot(run_id) or {}
    while not snapshot.get("finished"):
        if time.monotonic() > deadline:
            raise AssertionError(f"run {run_id} never reached a terminal status")
        time.sleep(0.02)
        snapshot = manager.snapshot(run_id) or {}
    return snapshot


class TestRepeatedExecution(unittest.TestCase):
    """A. and H. — the same plan, run twice, through the real manager."""

    def test_the_same_plan_run_twice_produces_two_distinct_runs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = _write_fake_repo(root)
            manager = RunManager()

            run_a = new_run_id(dataset="MATR", model="ridge", seed=0)
            first = _run_to_completion(manager, root=root, config=config,
                                       run_id=run_a)
            self.assertEqual(first["status"], rm.STATUS_COMPLETED)

            run_b = new_run_id(dataset="MATR", model="ridge", seed=0)
            # A. the second run of an identical plan is a different run.
            self.assertNotEqual(run_b, run_a)
            second = _run_to_completion(manager, root=root, config=config,
                                        run_id=run_b)
            self.assertEqual(second["status"], rm.STATUS_COMPLETED)

            # H. two directories, two manifests, the first untouched.
            dir_a = root / "runs" / "agent_matr_ridge" / run_a
            dir_b = root / "runs" / "agent_matr_ridge" / run_b
            self.assertNotEqual(dir_a, dir_b)
            for directory, run_id in ((dir_a, run_a), (dir_b, run_b)):
                manifest = json.loads(
                    (directory / rm.MANIFEST_NAME).read_text(encoding="utf-8")
                )
                self.assertEqual(manifest["run_id"], run_id)
                self.assertEqual(manifest["status"], rm.STATUS_COMPLETED)

    def test_a_completed_run_does_not_block_starting_another(self) -> None:
        """B. — a terminal run is never treated as one still in flight."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = _write_fake_repo(root)
            manager = RunManager()
            run_a = new_run_id(dataset="MATR", model="ridge", seed=0)
            _run_to_completion(manager, root=root, config=config, run_id=run_a)
            self.assertNotIn(run_a, manager.live_run_ids())
            run_b = new_run_id(dataset="MATR", model="ridge", seed=0)
            _run_to_completion(manager, root=root, config=config, run_id=run_b)
            self.assertNotIn(run_b, manager.live_run_ids())


class TestTerminalReaping(unittest.TestCase):
    """E., F. — a run reporting nothing still ends, and is still reaped."""

    def test_a_run_with_no_progress_events_still_reaches_completed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = _write_fake_repo(root)
            manager = RunManager()
            run_id = new_run_id(dataset="MATR", model="ridge", seed=0)
            snapshot = _run_to_completion(manager, root=root, config=config,
                                          run_id=run_id)
            self.assertEqual(snapshot["status"], rm.STATUS_COMPLETED)
            # The child emitted no epoch counter — no training_progress
            # event, no total to count towards — and the run still reached a
            # terminal status. The executor's own closing stage event is not
            # a progress report and is not what ended the run.
            event = snapshot.get("event") or {}
            self.assertNotEqual(event.get("type"), "training_progress")
            self.assertIsNone(event.get("total"))
            self.assertIn(
                "fit complete", "\n".join(snapshot.get("tail") or [])
            )

    def test_forget_reaps_a_terminal_run_while_its_worker_finishes(self) -> None:
        """The window that made ``forget`` a silent no-op.

        The worker publishes the terminal status and result together, then
        writes the manifest and calls back. A page reacting to that snapshot
        arrives while the thread is still alive, and the old ``is_alive()``
        gate dropped nothing at all in exactly that window.
        """
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = _write_fake_repo(root)
            manager = RunManager()
            run_id = new_run_id(dataset="MATR", model="ridge", seed=0)
            _run_to_completion(manager, root=root, config=config, run_id=run_id)
            handle = manager.get(run_id)
            self.assertIsNotNone(handle)
            # Force the exact window: terminal status, thread still "alive".
            handle._thread = _AlwaysAliveThread()
            manager.forget(run_id)
            self.assertIsNone(manager.get(run_id))

    def test_forget_leaves_a_run_that_is_still_executing(self) -> None:
        """Non-vacuity: reaping is about terminal status, not about anything."""
        manager = RunManager()
        from batteryml_web.core.run_manager import RunHandle

        handle = RunHandle("run_x", Path("/tmp/none"), {})
        handle.status = rm.STATUS_RUNNING
        handle._thread = _AlwaysAliveThread()
        manager._runs["run_x"] = handle
        manager.forget("run_x")
        self.assertIsNotNone(manager.get("run_x"))


class _AlwaysAliveThread:
    """Stands in for a worker thread that has not returned yet."""

    def is_alive(self) -> bool:
        return True


class _Handle:
    """The manager handle ``_finish_active_run`` reads."""

    def __init__(self, result: dict, run_dir: Path) -> None:
        self.result = result
        self.run_dir = run_dir


class _StubManager:
    """Enough manager for the session-state half of the lifecycle."""

    def __init__(self, handle=None, snapshots=None) -> None:
        self._handle = handle
        self._snapshots = dict(snapshots or {})
        self.forgotten: list[str] = []

    def get(self, run_id):
        return self._handle

    def snapshot(self, run_id):
        return self._snapshots.get(str(run_id))

    def forget(self, run_id):
        self.forgotten.append(str(run_id))

    def cancel(self, run_id, **kwargs):
        return True

    def live_run_ids(self):
        return []


class _LifecycleCase(unittest.TestCase):
    """Shared plumbing for driving the real page functions with a fake st."""

    def _finish(self, session: dict, *, handle, snapshot, context_error=None,
                context=None):
        manager = _StubManager(handle=handle)
        real_manager, real_context = ui.get_run_manager, ui.build_result_context

        def _context(*args, **kwargs):
            if context_error is not None:
                raise context_error
            return context

        ui.get_run_manager = lambda: manager
        ui.build_result_context = _context
        try:
            fake = FakeStreamlit(session)
            with _fake_streamlit(fake):
                ui._finish_active_run("run_1", snapshot, smoke=False)
        finally:
            ui.get_run_manager = real_manager
            ui.build_result_context = real_context
        return manager


class TestTerminalRunIsReleased(_LifecycleCase):
    """B., C., D. — the wedge, from the page's side."""

    def test_a_completed_run_stops_being_the_active_run(self) -> None:
        session = {ui.ACTIVE_RUN_KEY: "run_1"}
        manager = self._finish(
            session,
            handle=_Handle(
                {"success": True, "workspace": "runs/w/run_1", "returncode": 0},
                Path("/tmp/none"),
            ),
            snapshot={"status": rm.STATUS_COMPLETED, "finished": True},
            context={"success": True, "workspace": "runs/w/run_1"},
        )
        self.assertNotIn(ui.ACTIVE_RUN_KEY, session)
        self.assertEqual(manager.forgotten, ["run_1"])
        self.assertEqual(session["last_run_status"], "completed")

    def test_a_failed_result_read_still_releases_the_run(self) -> None:
        """D. — the exact wedge: reporting fails, the lifecycle does not.

        Before the fix this left ``ACTIVE_RUN_KEY`` set, and since the Run
        button is drawn only when no run is active, the page never offered a
        second run again.
        """
        session = {ui.ACTIVE_RUN_KEY: "run_1"}
        self._finish(
            session,
            handle=_Handle(
                {"success": True, "workspace": "runs/w/run_1", "returncode": 0},
                Path("/tmp/none"),
            ),
            snapshot={"status": rm.STATUS_COMPLETED, "finished": True},
            context_error=ValueError("the predictions artifact is unreadable"),
        )
        self.assertNotIn(ui.ACTIVE_RUN_KEY, session)
        # Loud, not silent: the reason is kept and shown.
        self.assertIn("unreadable", session[ui.RESULT_ERROR_KEY])
        # And it says nothing false about the run itself.
        self.assertNotIn("agent_result_context", session)

    def test_a_new_finish_clears_an_earlier_read_failure(self) -> None:
        session = {ui.ACTIVE_RUN_KEY: "run_1", ui.RESULT_ERROR_KEY: "stale"}
        self._finish(
            session,
            handle=_Handle(
                {"success": True, "workspace": "runs/w/run_1", "returncode": 0},
                Path("/tmp/none"),
            ),
            snapshot={"status": rm.STATUS_COMPLETED, "finished": True},
            context={"success": True, "workspace": "runs/w/run_1"},
        )
        self.assertNotIn(ui.RESULT_ERROR_KEY, session)

    def test_a_cancelled_run_is_released_with_no_result_context(self) -> None:
        session = {ui.ACTIVE_RUN_KEY: "run_1", "cancelling_run_1": True}
        self._finish(
            session,
            handle=_Handle({"success": False, "returncode": -15}, Path("/tmp/none")),
            snapshot={"status": rm.STATUS_CANCELLED, "finished": True},
            context_error=AssertionError("must not be called for a cancelled run"),
        )
        self.assertNotIn(ui.ACTIVE_RUN_KEY, session)
        self.assertNotIn("cancelling_run_1", session)
        self.assertNotIn("agent_result_context", session)


class TestActiveRunRendering(unittest.TestCase):
    """C. — the caller always learns whether it must draw the Run button."""

    def _render(self, session: dict, manager) -> bool | None:
        real_manager = ui.get_run_manager
        ui.get_run_manager = lambda: manager
        try:
            fake = FakeStreamlit(session)
            with _fake_streamlit(fake):
                try:
                    return ui._render_active_run(
                        "run_1", plan={}, smoke=False, device="cpu"
                    )
                except _Rerun:
                    # A finished run reruns the page; it has already released
                    # the run, which is what the caller needs.
                    return False
        finally:
            ui.get_run_manager = real_manager

    def test_a_running_run_reports_itself_as_in_flight(self) -> None:
        manager = _StubManager(
            snapshots={
                "run_1": {
                    "status": rm.STATUS_RUNNING,
                    "finished": False,
                    "elapsed_s": 1.0,
                    "event": {"stage": "Training"},
                    "tail": [],
                }
            }
        )
        session: dict = {ui.ACTIVE_RUN_KEY: "run_1"}
        self.assertTrue(self._render(session, manager))
        self.assertEqual(session[ui.ACTIVE_RUN_KEY], "run_1")

    def test_an_untracked_run_is_released_instead_of_spinning(self) -> None:
        manager = _StubManager(snapshots={})
        session: dict = {ui.ACTIVE_RUN_KEY: "run_1", "cancelling_run_1": True}
        self.assertFalse(self._render(session, manager))
        self.assertNotIn(ui.ACTIVE_RUN_KEY, session)
        self.assertNotIn("cancelling_run_1", session)


class TestExecutedPlanSignature(unittest.TestCase):
    """G. — the guard must not fire for a plan nobody changed."""

    def _plan(self, workspace: str) -> dict:
        return {
            "dataset": "MATR",
            "task": "SOH estimation",
            "model": "ridge",
            "config_path": "configs/soh/ridge/matr.yaml",
            "epochs": None,
            "command": [
                "python", WRAPPER, "run", "configs/soh/ridge/matr.yaml",
                "--workspace", workspace, "--train", "--eval",
            ],
        }

    def test_the_signature_ignores_the_runs_own_identity(self) -> None:
        """The whole defect, in one assertion.

        Two renders of the same prepared plan differ only in the reserved run
        id inside ``--workspace``. An argv signature made them different
        plans, which erased the finished run's results on the rerun that
        followed every launch.
        """
        first = ui.executed_plan_signature(
            self._plan("runs/agent_matr_ridge/run_A"),
            device="cpu", seed=0, smoke=False,
        )
        second = ui.executed_plan_signature(
            self._plan("runs/agent_matr_ridge/run_B"),
            device="cpu", seed=0, smoke=False,
        )
        self.assertEqual(first, second)

    def test_a_real_change_of_experiment_still_changes_the_signature(self) -> None:
        """Non-vacuity: it is not simply constant."""
        base = self._plan("runs/agent_matr_ridge/run_A")
        reference = ui.executed_plan_signature(
            base, device="cpu", seed=0, smoke=False
        )
        variants = {
            "seed": ui.executed_plan_signature(base, device="cpu", seed=1, smoke=False),
            "device": ui.executed_plan_signature(base, device="cuda:0", seed=0, smoke=False),
            "smoke": ui.executed_plan_signature(base, device="cpu", seed=0, smoke=True),
        }
        for name, signature in variants.items():
            with self.subTest(changed=name):
                self.assertNotEqual(reference, signature)
        for field, value in (
            ("dataset", "HUST"), ("task", "RUL prediction"),
            ("model", "xgboost"), ("config_path", "configs/other.yaml"),
            ("epochs", 1),
        ):
            with self.subTest(changed=field):
                changed = dict(base)
                changed[field] = value
                self.assertNotEqual(
                    reference,
                    ui.executed_plan_signature(
                        changed, device="cpu", seed=0, smoke=False
                    ),
                )

    def test_a_missing_plan_has_no_signature(self) -> None:
        self.assertEqual(
            ui.executed_plan_signature(None, device="cpu", seed=0, smoke=False), ""
        )


class TestSourceContract(unittest.TestCase):
    """The orderings that make the fixes structural rather than incidental."""

    def _function(self, name: str) -> ast.FunctionDef:
        tree = ast.parse(Path(ui.__file__).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == name:
                return node
        raise AssertionError(f"{name} not found in {ui.__file__}")

    def test_the_run_is_released_before_any_artifact_is_read(self) -> None:
        """Statement order is the fix; a comment saying so is not."""
        function = self._function("_finish_active_run")
        release_line = None
        report_line = None
        for node in ast.walk(function):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "pop"
                and node.args
                and isinstance(node.args[0], ast.Name)
                and node.args[0].id == "ACTIVE_RUN_KEY"
            ):
                release_line = node.lineno
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id in {"_result_context_for", "_store_json_state"}
            ):
                report_line = node.lineno if report_line is None else min(
                    report_line, node.lineno
                )
        self.assertIsNotNone(release_line)
        self.assertIsNotNone(report_line)
        self.assertLess(release_line, report_line)

    def test_the_reporting_half_is_guarded(self) -> None:
        function = self._function("_finish_active_run")
        self.assertTrue(
            any(isinstance(node, ast.Try) for node in ast.walk(function)),
            "a failure to read results must not escape into the page boundary",
        )

    def test_the_run_signature_comes_from_the_experiment_not_the_argv(self) -> None:
        """The stale comparison must not come back.

        Asserted on the assignment itself rather than on the presence of a
        string: the page still records the argv in the manifest, which is
        correct and must not be what this test trips over.
        """
        tree = ast.parse(Path(ui.__file__).read_text(encoding="utf-8"))
        assignments = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "current_signature"
                for target in node.targets
            )
        ]
        self.assertEqual(len(assignments), 1)
        value = assignments[0].value
        self.assertIsInstance(value, ast.Call)
        self.assertIsInstance(value.func, ast.Name)
        self.assertEqual(value.func.id, "executed_plan_signature")


if __name__ == "__main__":
    unittest.main()
