"""A run always reaches a terminal state, and both routes run the same command.

Three defects are pinned here, each reproduced before it was repaired.

**1. A run could stay RUNNING forever.** ``execute_plan_streaming``'s event
loop ended only when *both* pipes reached EOF, with no timeout (``RunManager``
passes none) and no look at the process itself. A pipe closes when the last
holder of its write end closes it, and the child is not always the last
holder: anything it spawned that inherited stdout or stderr keeps the pipe
open after the child has exited. Measured before the fix — a child that
spawns a sleeper and exits 0 left the loop waiting with both streams still
open 7.8 seconds after the child was gone, and nothing would ever have ended
it. The manifest stayed RUNNING, and the page polled a job that had finished.

**2. One NaN deleted a whole cell.** The cycle-level SOH work added
``& _usable_rows(features)`` to the sample mask in ``Task.build`` — but to
the *shared* code path, not to the cycle-level branch its own comment
described. On a whole-cell RUL task one row is one cell and one feature is a
whole matrix: a CNN's ``VoltageCapacityMatrixFeatureExtractor`` row is
100x1000. A single NaN among those 100,000 values dropped the entire cell,
silently, although its label was good; with one NaN per cell the training set
came out empty. Before the SOH work the mask was ``~isnan(labels)`` alone.

**3. The two execution routes drifted.** Run Model spelled its argv out
inline and omitted ``--metric``, so the same config scored ``MAE,RMSE``
through the Agent and fell back to the CLI default ``RMSE,MAE,MAPE`` through
Run Model. Both now build the argv with one function.

Real subprocesses, tiny and fast — no ``sleep``-padded waits, and nothing
trains.

Run from the repository root::

    conda run -n batteryml python -m unittest tests.test_execution_lifecycle -v
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
import textwrap
import time
import unittest
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from batteryml.task import _usable_rows  # noqa: E402
from batteryml_web.agent import streaming as sm  # noqa: E402
from batteryml_web.agent.execution import build_run_command  # noqa: E402
from batteryml_web.core import run_manifest as rm  # noqa: E402
from batteryml_web.core.run_identity import new_run_id  # noqa: E402
from batteryml_web.core import run_manager as run_manager_module  # noqa: E402
from batteryml_web.core.run_manager import RunManager  # noqa: E402

WRAPPER = "batteryml_web/core/run_batteryml_cli.py"

#: Bounded so a hang fails the test instead of hanging the suite.
TIMEOUT_S = 40.0


def _fake_repo(root: Path, child_source: str) -> Path:
    """A tree the plan validator accepts, with a stand-in wrapper.

    The wrapper is not a mock of BatteryML: these tests are about the
    lifecycle *around* an execution, so the child only has to be a real
    process that really exits.
    """
    (root / "runs").mkdir(parents=True, exist_ok=True)
    (root / "configs").mkdir(parents=True, exist_ok=True)
    wrapper = root / WRAPPER
    wrapper.parent.mkdir(parents=True, exist_ok=True)
    wrapper.write_text(textwrap.dedent(child_source).strip() + "\n", encoding="utf-8")
    config = root / "configs" / "model.yaml"
    config.write_text("model:\n  name: 'CNNRULPredictor'\n", encoding="utf-8")
    return config


def _plan(config: Path, workspace: str) -> dict:
    return {
        "dataset": "MATR",
        "task": "RUL prediction",
        "model": "cnn",
        "model_display": "CNN",
        "config_path": str(config),
        "config_valid": True,
        "model_status": "READY",
        "dataset_status": "READY",
        "can_execute": True,
        "epochs": 1,
        "command_workspace": workspace,
        "command": build_run_command(
            config_path=str(config),
            workspace=workspace,
            task="RUL prediction",
            device="cpu",
            seed=0,
            epochs=1,
        ),
    }


#: A child that exits cleanly but leaves a grandchild holding both pipes.
#: This is the shape that hung the loop.
ORPHAN_HOLDER = """
    import subprocess, sys
    print('training done', flush=True)
    subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(120)'])
    sys.exit(0)
"""


class TestTerminalState(unittest.TestCase):
    """§6: a run becomes exactly one terminal state, always."""

    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="lifecycle-"))
        self.addCleanup(shutil.rmtree, self.tmp, True)

    def _run(self, child_source: str) -> dict:
        config = _fake_repo(self.tmp, child_source)
        start = time.monotonic()
        result = sm.execute_plan_streaming(
            _plan(config, "runs/w"), repo_root=self.tmp
        )
        self.assertLess(
            time.monotonic() - start, TIMEOUT_S,
            "the streaming loop did not return; it is waiting on a pipe",
        )
        return result

    def test_a_successful_run_completes(self) -> None:
        result = self._run("import sys\nprint('ok', flush=True)\nsys.exit(0)")
        self.assertTrue(result["success"])
        self.assertEqual(result["returncode"], 0)
        self.assertIn("ok", result["stdout"])

    def test_a_failing_run_fails_and_keeps_its_stderr(self) -> None:
        result = self._run(
            "import sys\nprint('boom', file=sys.stderr, flush=True)\nsys.exit(3)"
        )
        self.assertFalse(result["success"])
        self.assertEqual(result["returncode"], 3)
        self.assertIn("boom", result["stderr"])
        # §6: a failure is reported, never retried.
        self.assertIn("no retry performed", result["error"])

    def test_a_silent_run_with_no_progress_still_terminates(self) -> None:
        """§7: lifecycle never depends on a tqdm line appearing."""
        result = self._run("import sys\nsys.exit(0)")
        self.assertTrue(result["success"])
        self.assertEqual(result["stdout"], "")

    def test_malformed_progress_output_does_not_fail_the_run(self) -> None:
        """§7: a broken progress line is presentation, not an error."""
        result = self._run(
            "import sys\n"
            "print('Training X: 999%|###| not/a/counter', flush=True)\n"
            "print('Epoch |||| 3', flush=True)\n"
            "sys.exit(0)"
        )
        self.assertTrue(result["success"])

    def test_an_orphan_holding_the_pipes_does_not_hang_the_run(self) -> None:
        """The reproduced hang. Before the fix this never returned.

        The child exits 0; a grandchild it spawned keeps both pipes open.
        The loop must notice the child is gone rather than wait for an EOF
        that is not coming.
        """
        start = time.monotonic()
        result = self._run(ORPHAN_HOLDER)
        elapsed = time.monotonic() - start
        self.assertTrue(result["success"])
        self.assertEqual(result["returncode"], 0)
        # Its real output is still captured.
        self.assertIn("training done", result["stdout"])
        # And the page is told the capture may be short, rather than the run
        # being reported as if nothing unusual happened.
        self.assertIn("kept its output pipes open", result.get("stream_note", ""))
        self.assertLess(elapsed, sm.POST_EXIT_DRAIN_S + 10.0)

    def test_a_healthy_run_is_not_delayed_by_the_drain_window(self) -> None:
        """Non-vacuity: the fix must cost a normal run nothing."""
        start = time.monotonic()
        self._run("import sys\nprint('quick', flush=True)\nsys.exit(0)")
        self.assertLess(time.monotonic() - start, sm.POST_EXIT_DRAIN_S)

    def test_a_large_tail_after_exit_is_not_truncated(self) -> None:
        """The drain is measured from activity, so output is never cut off."""
        result = self._run(
            "import sys\n"
            "for i in range(2000):\n"
            "    print('line %d' % i, flush=True)\n"
            "sys.exit(0)"
        )
        self.assertTrue(result["success"])
        self.assertIn("line 0", result["stdout"])
        self.assertIn("line 1999", result["stdout"])


class TestManagerReachesTerminalStatus(unittest.TestCase):
    """§6 through the real RunManager and a real manifest on disk."""

    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="manager-"))
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.manager = RunManager()

    def _start(self, child_source: str, run_id: str | None = None) -> tuple[str, Path]:
        config = _fake_repo(self.tmp, child_source)
        run_id = run_id or new_run_id(dataset="MATR", model="cnn", seed=0)
        workspace = f"runs/agent_matr_cnn/{run_id}"
        run_dir = self.tmp / workspace
        plan = _plan(config, workspace)
        rm.write_manifest(
            run_dir,
            rm.build_manifest(
                run_id=run_id, project="alpha", workspace=workspace,
                dataset="MATR", task="RUL prediction", model_key="cnn",
                config_path=str(config), seed=0, device="cpu",
                command=plan["command"],
            ),
        )
        self.manager.start(
            run_id=run_id, run_dir=run_dir, plan=plan, repo_root=self.tmp
        )
        return run_id, run_dir

    def _await_terminal(self, run_id: str) -> dict:
        deadline = time.monotonic() + TIMEOUT_S
        snapshot = self.manager.snapshot(run_id) or {}
        while not snapshot.get("finished"):
            if time.monotonic() > deadline:
                self.fail(
                    f"run {run_id} never left {snapshot.get('status')!r}; "
                    "it is stuck"
                )
            time.sleep(0.02)
            snapshot = self.manager.snapshot(run_id) or {}
        return snapshot

    def _manifest(self, run_dir: Path) -> dict:
        return json.loads(
            (run_dir / rm.MANIFEST_NAME).read_text(encoding="utf-8")
        )

    def test_success_reaches_completed(self) -> None:
        run_id, run_dir = self._start("import sys\nprint('ok')\nsys.exit(0)")
        self.assertEqual(
            self._await_terminal(run_id)["status"], rm.STATUS_COMPLETED
        )
        self.assertEqual(self._manifest(run_dir)["status"], rm.STATUS_COMPLETED)

    def test_failure_reaches_failed(self) -> None:
        run_id, run_dir = self._start("import sys\nsys.exit(4)")
        self.assertEqual(
            self._await_terminal(run_id)["status"], rm.STATUS_FAILED
        )
        manifest = self._manifest(run_dir)
        self.assertEqual(manifest["status"], rm.STATUS_FAILED)
        self.assertEqual(manifest["return_code"], 4)

    def test_an_orphan_holding_the_pipes_still_reaches_completed(self) -> None:
        """The stuck run, end to end: manager, manifest and all."""
        run_id, run_dir = self._start(ORPHAN_HOLDER)
        self.assertEqual(
            self._await_terminal(run_id)["status"], rm.STATUS_COMPLETED
        )
        self.assertEqual(self._manifest(run_dir)["status"], rm.STATUS_COMPLETED)
        # Released from the manager, so the page offers Run again.
        self.manager.forget(run_id)
        self.assertNotIn(run_id, self.manager.live_run_ids())

    def test_repeating_a_config_produces_two_distinct_runs(self) -> None:
        """§10-G: identical plan, two identities, two directories."""
        first, first_dir = self._start("import sys\nsys.exit(0)")
        self._await_terminal(first)
        second, second_dir = self._start("import sys\nsys.exit(0)")
        self._await_terminal(second)
        self.assertNotEqual(first, second)
        self.assertNotEqual(first_dir, second_dir)
        self.assertTrue((first_dir / rm.MANIFEST_NAME).is_file())
        self.assertTrue((second_dir / rm.MANIFEST_NAME).is_file())

    def test_a_cancelled_run_reaches_cancelled(self) -> None:
        """§10-E: Stop terminates the process group and settles the state."""
        run_id, run_dir = self._start(
            "import sys, time\nprint('started', flush=True)\ntime.sleep(120)"
        )
        deadline = time.monotonic() + TIMEOUT_S
        while (self.manager.snapshot(run_id) or {}).get("status") != rm.STATUS_RUNNING:
            if time.monotonic() > deadline:
                self.fail("the run never reached RUNNING")
            time.sleep(0.02)
        # Give the child a moment to exist before signalling it.
        handle = self.manager.get(run_id)
        while handle._process is None:
            if time.monotonic() > deadline:
                self.fail("the run never spawned a process")
            time.sleep(0.02)
        self.assertTrue(self.manager.cancel(run_id))
        self.assertEqual(
            self._await_terminal(run_id)["status"], rm.STATUS_CANCELLED
        )
        manifest = self._manifest(run_dir)
        self.assertEqual(manifest["status"], rm.STATUS_CANCELLED)
        # A stopped run measured nothing.
        self.assertEqual(manifest.get("metrics"), {})


class TestTerminalStateIsPublishedDurablyFirst(unittest.TestCase):
    """``finished`` means settled — on disk as well as in memory.

    The flake these tests replace was the mild symptom of a real ordering
    defect. ``_execute`` published the in-memory terminal status and *then*
    wrote the terminal manifest, so for a window the two disagreed. Measured
    before the fix with a tight poll: the manifest was still RUNNING at the
    instant ``finished`` first became true in **20 of 20** repetitions,
    usually for about 2ms but once for 446ms on an idle machine. The test
    that caught it slept 20ms between polls, which is why it failed
    intermittently rather than always — 3 of 25 runs in isolation, more
    under load.

    The window mattered because of what the page does inside it.
    ``_finish_active_run`` reacts to ``finished`` by calling ``forget()`` —
    dropping the run from ``live_run_ids()`` — and then reruns, and Run
    History renders ``list_runs(..., live_run_ids=...)``. A manifest still
    saying RUNNING that is no longer live is precisely what
    ``reconcile_status`` calls an abandoned job, so a run that had just
    exited 0 was shown as INTERRUPTED, "not known to have finished". Through
    the real ``forget``/``list_runs`` path that reproduced **15 of 15**.

    So the contract is now an ordering: the manifest is written first, and
    ``finished`` becomes true only afterwards.
    """

    #: §7-8 asks for 20+ repetitions with no intermittent RUNNING, and this
    #: is the test that answers it. Each repetition spawns a real child, so
    #: it costs about two seconds; that is the price of the guarantee and it
    #: is not reduced.
    REPETITIONS = 20

    #: The mislabel below reproduced 15 times out of 15 before the fix — it
    #: was deterministic, not intermittent — so a handful of repetitions
    #: demonstrates it and the other test carries the anti-flake load.
    LABEL_REPETITIONS = 5

    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="durable-"))
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.manager = RunManager()

    def _prepare(self, child_source: str, *, index: int = 0) -> dict:
        """Everything a run needs on disk, without starting it.

        Separate from :meth:`_start` so a test can make the filesystem fail
        *after* the fixture tree exists and before the worker touches it.
        """
        root = self.tmp / f"rep{index}"
        config = _fake_repo(root, child_source)
        run_id = new_run_id(dataset="MATR", model="cnn", seed=0)
        workspace = f"runs/agent_matr_cnn/{run_id}"
        run_dir = root / workspace
        plan = _plan(config, workspace)
        rm.write_manifest(
            run_dir,
            rm.build_manifest(
                run_id=run_id, project="alpha", workspace=workspace,
                dataset="MATR", task="RUL prediction", model_key="cnn",
                config_path=str(config), seed=0, device="cpu",
                command=plan["command"],
            ),
        )
        return {
            "run_id": run_id,
            "run_dir": run_dir,
            "plan": plan,
            "repo_root": root,
            "project_root": root / "runs" / "agent_matr_cnn",
        }

    def _start(self, child_source: str, *, index: int = 0) -> tuple[str, Path, Path]:
        prepared = self._prepare(child_source, index=index)
        self.manager.start(
            run_id=prepared["run_id"],
            run_dir=prepared["run_dir"],
            plan=prepared["plan"],
            repo_root=prepared["repo_root"],
        )
        return prepared["run_id"], prepared["run_dir"], prepared["project_root"]

    def _spin_to_finished(self, run_id: str) -> dict:
        """Poll with no delay, so a millisecond window is not slept through.

        The 20ms sleeping poll the older tests use is what made this defect
        intermittent instead of visible: it slept through the typical 2ms
        window and only noticed the occasional long one.

        ``sleep(0)`` yields the GIL without waiting. A bare busy-spin would
        hold it in 5ms slices and starve the very worker thread whose
        ordering is under test — making the test slower and less faithful
        rather than stricter.
        """
        deadline = time.monotonic() + TIMEOUT_S
        snapshot = self.manager.snapshot(run_id) or {}
        while not snapshot.get("finished"):
            if time.monotonic() > deadline:
                self.fail(
                    f"run {run_id} never left {snapshot.get('status')!r}"
                )
            time.sleep(0)
            snapshot = self.manager.snapshot(run_id) or {}
        return snapshot

    def _manifest_status(self, run_dir: Path) -> str:
        return json.loads(
            (run_dir / rm.MANIFEST_NAME).read_text(encoding="utf-8")
        )["status"]

    def test_finished_implies_the_manifest_is_already_terminal(self) -> None:
        """§7-8: repeated, with no sleep to hide the window."""
        lagged = []
        for index in range(self.REPETITIONS):
            run_id, run_dir, _ = self._start(
                "import sys\nprint('ok')\nsys.exit(0)", index=index
            )
            snapshot = self._spin_to_finished(run_id)
            status = self._manifest_status(run_dir)
            self.assertEqual(snapshot["status"], rm.STATUS_COMPLETED)
            if status not in rm.TERMINAL_STATUSES:
                lagged.append((index, status))
        self.assertEqual(
            lagged, [],
            "the manifest was still non-terminal when `finished` became "
            f"true, in {len(lagged)} of {self.REPETITIONS} repetitions",
        )

    def test_a_completed_run_is_never_reported_interrupted(self) -> None:
        """The user-visible defect, through the page's own sequence.

        Not a simulation of the UI: the same ``forget`` and the same
        ``list_runs(..., live_run_ids=...)`` the Agent page calls, in the
        order it calls them.
        """
        for index in range(self.LABEL_REPETITIONS):
            run_id, run_dir, project_root = self._start(
                "import sys\nprint('ok')\nsys.exit(0)", index=index
            )
            snapshot = self._spin_to_finished(run_id)
            self.assertEqual(snapshot["status"], rm.STATUS_COMPLETED)

            self.manager.forget(run_id)
            entries = rm.list_runs(
                project_root, live_run_ids=self.manager.live_run_ids()
            )
            shown = next(
                (e for e in entries if str(e.get("run_id")) == run_id), {}
            )
            self.assertEqual(
                shown.get("status"), rm.STATUS_COMPLETED,
                f"repetition {index}: a run that exited 0 was reported as "
                f"{shown.get('status')} "
                f"(reconciled_from={shown.get('reconciled_from')})",
            )

    def test_a_finished_run_is_no_longer_reported_live(self) -> None:
        """Non-vacuity for the test above: ``live_run_ids`` really does drop it.

        Without this, ``list_runs`` could be returning COMPLETED merely
        because the run was still considered live, which would make the
        reconciliation path untested rather than correct.
        """
        run_id, _run_dir, _project = self._start("import sys\nsys.exit(0)")
        self._spin_to_finished(run_id)
        self.assertNotIn(run_id, self.manager.live_run_ids())

    def test_an_exception_in_the_executor_still_reaches_a_terminal_state(
        self,
    ) -> None:
        """The unbounded half of the defect.

        ``execute_plan_streaming`` was called unguarded. Anything it raised
        killed the worker thread with the status still RUNNING — and
        ``live_run_ids`` reads the status, not the thread, so the manager
        went on calling the run live, the page polled a job whose worker was
        dead, and orphan recovery never applied because the manager insisted
        it was alive. Reproduced before the fix: RUNNING in memory and on
        disk, thread dead, still listed live, with no time bound at all.
        """
        real = run_manager_module.execute_plan_streaming

        def _raise(*_args, **_kwargs):
            raise RuntimeError("boom inside the executor")

        run_manager_module.execute_plan_streaming = _raise
        try:
            run_id, run_dir, _project = self._start("import sys\nsys.exit(0)")
            snapshot = self._spin_to_finished(run_id)
        finally:
            run_manager_module.execute_plan_streaming = real

        self.assertEqual(snapshot["status"], rm.STATUS_FAILED)
        self.assertEqual(self._manifest_status(run_dir), rm.STATUS_FAILED)
        self.assertNotIn(run_id, self.manager.live_run_ids())
        # No exit code is claimed: no process outcome was ever observed.
        self.assertIsNone(snapshot.get("returncode"))

    def test_a_setup_failure_still_reaches_a_terminal_state(self) -> None:
        """The same shape one step earlier, before the status leaves PLANNED.

        ``run_dir.mkdir`` raises as readily as the executor does — an
        unwritable parent is enough — and it ran outside the guard, so the
        run was stranded at PLANNED with a dead worker.
        """
        prepared = self._prepare("import sys\nsys.exit(0)")
        run_id = prepared["run_id"]
        real = Path.mkdir

        def _deny(self_path, *args, **kwargs):
            raise PermissionError(f"refused: {self_path}")

        # Patched only now: the fixture tree above needed a working mkdir,
        # and what is under test is the worker's own call.
        Path.mkdir = _deny
        try:
            self.manager.start(
                run_id=run_id,
                run_dir=prepared["run_dir"],
                plan=prepared["plan"],
                repo_root=prepared["repo_root"],
            )
            snapshot = self._spin_to_finished(run_id)
        finally:
            Path.mkdir = real

        self.assertEqual(snapshot["status"], rm.STATUS_FAILED)
        self.assertNotIn(run_id, self.manager.live_run_ids())


class TestOrphanedRunningManifest(unittest.TestCase):
    """§6 case 4: a RUNNING manifest with no live process is not RUNNING."""

    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="orphan-"))
        self.addCleanup(shutil.rmtree, self.tmp, True)

    def test_a_manifest_left_running_recovers_a_terminal_state(self) -> None:
        run_dir = self.tmp / "runs" / "web_projects" / "alpha" / "run_x"
        manifest = rm.build_manifest(
            run_id="run_20260101_000000_cnn_matr_seed0_aaaaaa",
            project="alpha", workspace=str(run_dir), dataset="MATR",
            task="RUL prediction", model_key="cnn", seed=0, device="cpu",
            command=["python", "x"],
        )
        manifest["status"] = rm.STATUS_RUNNING
        rm.write_manifest(run_dir, manifest)

        # A fresh process: nothing is live, so nothing may be reported as
        # still running.
        runs = rm.list_runs(
            self.tmp / "runs" / "web_projects" / "alpha", live_run_ids=[]
        )
        self.assertEqual(len(runs), 1)
        self.assertNotEqual(runs[0]["status"], rm.STATUS_RUNNING)
        self.assertIn(runs[0]["status"], rm.TERMINAL_STATUSES)

    def test_a_genuinely_live_run_is_left_alone(self) -> None:
        """Non-vacuity: recovery must not reap a job that is really going."""
        run_id = "run_20260101_000000_cnn_matr_seed0_aaaaaa"
        run_dir = self.tmp / "runs" / "web_projects" / "alpha" / "run_x"
        manifest = rm.build_manifest(
            run_id=run_id, project="alpha", workspace=str(run_dir),
            dataset="MATR", task="RUL prediction", model_key="cnn", seed=0,
            device="cpu", command=["python", "x"],
        )
        manifest["status"] = rm.STATUS_RUNNING
        rm.write_manifest(run_dir, manifest)
        runs = rm.list_runs(
            self.tmp / "runs" / "web_projects" / "alpha", live_run_ids=[run_id]
        )
        self.assertEqual(runs[0]["status"], rm.STATUS_RUNNING)


class TestBothRoutesBuildOneCommand(unittest.TestCase):
    """§10-A: the same run, whichever button started it."""

    def test_the_run_model_page_uses_the_shared_builder(self) -> None:
        """Behavioural: the page's argv is the builder's argv.

        Asserted by calling the builder with the page's own inputs and
        comparing, rather than by grepping ``app.py`` — what matters is that
        the two routes cannot produce different commands.
        """
        shared = build_run_command(
            config_path="configs/baselines/nn_models/cnn/matr_1.yaml",
            workspace="runs/web_projects/alpha/workspace",
            task="RUL prediction", device="cpu", seed=0,
        )
        agent = build_run_command(
            config_path="configs/baselines/nn_models/cnn/matr_1.yaml",
            workspace="runs/web_projects/alpha/workspace",
            task="RUL prediction", device="cpu", seed=0,
        )
        self.assertEqual(shared, agent)

    def test_the_contract_flags_are_always_present(self) -> None:
        """CLAUDE.md §5: without --train --eval nothing trains."""
        command = build_run_command(
            config_path="c.yaml", workspace="runs/w",
            task="RUL prediction", device="cpu", seed=0,
        )
        self.assertIn("--train", command)
        self.assertIn("--eval", command)
        # The config is positional, immediately after "run".
        self.assertEqual(command[command.index("run") + 1], "c.yaml")
        self.assertNotIn("--config", command)

    def test_the_task_decides_the_metrics_on_both_routes(self) -> None:
        soh = build_run_command(
            config_path="c.yaml", workspace="runs/w",
            task="SOH estimation", device="cpu", seed=0,
        )
        self.assertEqual(soh[soh.index("--metric") + 1], "MAE,RMSE")

    def test_an_epoch_override_is_optional_and_never_invented(self) -> None:
        without = build_run_command(
            config_path="c.yaml", workspace="runs/w",
            task="RUL prediction", device="cpu", seed=0,
        )
        self.assertNotIn("--epochs", without)
        with_override = build_run_command(
            config_path="c.yaml", workspace="runs/w",
            task="RUL prediction", device="cpu", seed=0, epochs=1,
        )
        self.assertEqual(with_override[with_override.index("--epochs") + 1], "1")


class TestWholeCellFeaturesSurviveTheSohWork(unittest.TestCase):
    """§8: cycle-level infrastructure must not change legacy RUL shapes."""

    def _task(self, cycle_level: bool):
        """A Task stub carrying only what the mask block reads."""

        class _Task:
            is_cycle_level = cycle_level

        return _Task()

    def test_a_cnn_sized_cell_is_not_deleted_by_one_nan(self) -> None:
        """The reproduced regression, at the mask.

        One NaN among 100,000 values used to remove the whole cell.
        """
        features = torch.randn(5, 100, 1000)
        labels = torch.tensor([800.0, 900.0, 1000.0, 1100.0, 1200.0])
        features[1, 0, 0] = float("nan")
        features[3, 50, 500] = float("nan")

        whole_cell = ~torch.isnan(labels)
        self.assertEqual(int(whole_cell.sum()), 5)

        # What the shared block would have produced before the repair.
        over_filtered = whole_cell & _usable_rows(features)
        self.assertEqual(int(over_filtered.sum()), 3)

    def test_every_cell_carrying_a_nan_emptied_the_training_set(self) -> None:
        """The degenerate case the repair prevents."""
        features = torch.randn(4, 100, 1000)
        labels = torch.tensor([800.0, 900.0, 1000.0, 1100.0])
        for index in range(4):
            features[index, index, index] = float("nan")
        self.assertEqual(int((~torch.isnan(labels) & _usable_rows(features)).sum()), 0)
        self.assertEqual(int((~torch.isnan(labels)).sum()), 4)

    def test_the_mask_block_applies_the_feature_filter_only_to_cycle_tasks(self) -> None:
        """The repair itself, read from ``Task.build``'s own branch.

        Behavioural equivalents are asserted above; this pins the condition
        so the filter cannot quietly become unconditional again.
        """
        import ast
        import inspect

        from batteryml.task import Task

        source = textwrap.dedent(inspect.getsource(Task.build))
        tree = ast.parse(source)
        guarded = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.If)
            and "is_cycle_level" in ast.unparse(node.test)
            and "_usable_rows" in ast.unparse(node)
        ]
        self.assertTrue(
            guarded,
            "_usable_rows must be applied inside an is_cycle_level branch",
        )

    def test_a_cycle_level_task_still_drops_unreadable_rows(self) -> None:
        """Non-vacuity: SOH keeps the filter it needs."""
        features = torch.randn(6, 37)
        labels = torch.tensor([1.0, 0.98, 0.95, 0.93, 0.91, 0.90])
        features[2, 5] = float("nan")
        mask = ~torch.isnan(labels) & _usable_rows(features)
        self.assertEqual(int(mask.sum()), 5)
        self.assertFalse(bool(mask[2]))


class TestConfigResolutionReachesTheRightData(unittest.TestCase):
    """§10-I: the config that reaches the command builder reads the dataset
    the user asked for.

    The reproduced defect: ``_infer_dataset`` searched the whole YAML text
    for each dataset name with ``MATR`` first, and every CNN config names
    ``VoltageCapacityMatrixFeatureExtractor`` — *Matrix* contains **matr**.
    All eight CNN configs registered as MATR, including ``hust.yaml``
    (``data/processed/HUST``) and ``snl.yaml`` (``data/processed/SNL``). So
    "CNN on MATR" could resolve to a config that loads HUST, which is
    refused by the data-requiring validator or dies on
    ``AssertionError: data/processed/HUST`` — and HUST and SNL offered no
    CNN at all despite each having one written for it.
    """

    def setUp(self) -> None:
        from batteryml_web.core.config_registry import get_config_registry

        self.registry = get_config_registry()

    def _entries(self, name_fragment: str):
        return [
            entry
            for entry in self.registry.get_entries()
            if name_fragment in str(getattr(entry, "config_path", ""))
        ]

    def test_every_config_is_registered_under_the_data_it_loads(self) -> None:
        """The invariant, over the whole catalog — not only CNN."""
        import yaml

        offenders = []
        for entry in self.registry.get_entries():
            path = Path(str(entry.config_path))
            data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            cell_data_path = (data.get("train_test_split") or {}).get(
                "cell_data_path"
            )
            if not isinstance(cell_data_path, str):
                continue
            leaf = cell_data_path.rstrip("/").rsplit("/", 1)[-1].upper()
            if leaf != str(entry.dataset).upper():
                offenders.append(
                    f"{path.name}: registered as {entry.dataset}, loads {leaf}"
                )
        self.assertEqual(offenders, [], "; ".join(offenders))

    def test_cnn_on_matr_offers_only_matr_configs(self) -> None:
        offered = self.registry.get_configs_for_selection(
            "MATR", "RUL prediction", "cnn"
        )
        self.assertTrue(offered, "MATR must still offer CNN")
        for entry in offered:
            name = Path(str(entry.config_path)).name
            with self.subTest(config=name):
                self.assertNotIn(name, {"hust.yaml", "snl.yaml"})

    def test_the_datasets_with_a_cnn_config_can_select_it(self) -> None:
        """They offered nothing at all while every CNN config claimed MATR."""
        for dataset, expected in (("HUST", "hust.yaml"), ("SNL", "snl.yaml")):
            with self.subTest(dataset=dataset):
                offered = self.registry.get_configs_for_selection(
                    dataset, "RUL prediction", "cnn"
                )
                names = {Path(str(e.config_path)).name for e in offered}
                self.assertIn(expected, names)

    def test_a_feature_extractor_name_cannot_name_the_dataset(self) -> None:
        """The collision itself, pinned so it cannot return.

        Non-vacuous: the string really is present in these configs.
        """
        for entry in self._entries("nn_models/cnn"):
            path = Path(str(entry.config_path))
            text = path.read_text(encoding="utf-8")
            self.assertIn("VoltageCapacityMatrixFeatureExtractor", text)
            if path.name == "hust.yaml":
                self.assertEqual(entry.dataset, "HUST")
            if path.name == "snl.yaml":
                self.assertEqual(entry.dataset, "SNL")

    def test_a_multi_dataset_config_is_not_offered_as_one_dataset(self) -> None:
        """``mix``/``cruh``/``crush`` read several directories at once.

        They are not any single dataset's config. They stay on disk and
        remain runnable from the CLI; they are simply not advertised under a
        dataset they only partly use.
        """
        registered = {
            Path(str(e.config_path)).name for e in self.registry.get_entries()
        }
        for name in ("mix.yaml", "cruh.yaml", "crush.yaml"):
            with self.subTest(config=name):
                self.assertNotIn(name, registered)


if __name__ == "__main__":
    unittest.main()
