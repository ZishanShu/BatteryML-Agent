"""Runs a validated plan in the background, so the page stays interactive.

The problem this solves is structural, not cosmetic.
:func:`batteryml_web.agent.streaming.execute_plan_streaming` blocks its
caller until the child exits. Called straight from a Streamlit script, it
blocks the script run — so a Cancel button drawn in that same call can never
receive a click, and the only honest "Cancel" would have been a button that
changes a label and lets the training continue. That is exactly the fake
this must not be.

So execution moves one step away from the page::

    UI  ->  RunManager  ->  execute_plan_streaming  ->  subprocess

The manager owns a worker thread per run. The page starts a run, returns
immediately, and afterwards only *polls* structured state. Nothing here
imports Streamlit, so the manager is equally callable from a test or a
script.

Cancellation is real. The child is started in its own session
(``start_new_session=True``), which puts it in its own process group, so a
cancel signals **that group and no other**: ``os.killpg`` on the group this
manager created. There is no ``pkill``, no ``killall``, no pattern matching
against process names — a name pattern could match another user's training
job on a shared server. SIGTERM first, and SIGKILL only after a grace period
for a child that ignores it.

A cancelled run is not a finished run: its output and any artifacts already
written are kept and labelled partial, its manifest goes to ``CANCELLED``,
and no result context is built from it. Metrics from a killed training run
would be meaningless, so none are offered.

**One ordering holds this together**: ``snapshot()["finished"]`` becomes
true only after the manifest is already terminal, so the in-memory view and
the durable record can never be caught disagreeing. See :meth:`_finalize`
and ``docs/RUN_LIFECYCLE.md``.
"""

from __future__ import annotations

import os
import signal
import subprocess
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any, Callable, Deque, Dict, List, Optional

from ..agent.executor import _empty_artifacts
from ..agent.streaming import execute_plan_streaming
from . import run_manifest as rm

#: Lines of recent output the manager keeps in memory for the UI. Bounded on
#: purpose: the full log goes to the run directory, and an unbounded buffer
#: in a long training run is a memory leak the page would carry.
LIVE_TAIL_LINES = 40

#: Seconds a cancelled child is given to exit on SIGTERM before SIGKILL.
CANCEL_GRACE_S = 10.0

#: Seconds to wait for the child to be reaped after SIGKILL. A kill cannot
#: be refused, but it is delivered asynchronously, so reporting "stopped"
#: without waiting would sometimes be reporting on a signal in flight.
KILL_WAIT_S = 5.0

#: Incremental log written while the run is in flight. The authoritative
#: per-channel logs are written when it ends; events do not carry a channel,
#: so inventing one here would be a lie about which stream a line came from.
LIVE_LOG = "run.log"

CANCELLED_MESSAGE = "Cancelled by user"


def _failure_result(plan: Dict[str, Any], exc: BaseException) -> Dict[str, Any]:
    """The result shape for a run whose executor never returned one.

    Same keys :func:`execute_plan_streaming` returns, so every consumer —
    the manifest writer, the result builder, the page — reads it without a
    special case. ``success`` is False and ``returncode`` is None, because
    no process outcome was observed: the failure is the executor's, and
    claiming an exit code would invent one.
    """
    return {
        "started": False,
        "completed": False,
        "success": False,
        "returncode": None,
        "command": [str(part) for part in (plan.get("command") or [])],
        "workspace": plan.get("command_workspace"),
        "stdout": "",
        "stderr": "",
        "error": f"The run failed before it could report a result "
        f"({type(exc).__name__}: {exc}).",
        "artifacts": _empty_artifacts(),
        "dry_run": False,
    }


class RunHandle:
    """Live state for one run. Mutated by the worker, read by the poller."""

    def __init__(self, run_id: str, run_dir: Path, plan: Dict[str, Any]) -> None:
        self.run_id = str(run_id)
        self.run_dir = Path(run_dir)
        self.plan = plan
        self.status: str = rm.STATUS_PLANNED
        self.event: Dict[str, Any] = {}
        self.tail: Deque[str] = deque(maxlen=LIVE_TAIL_LINES)
        self.result: Optional[Dict[str, Any]] = None
        self.started_monotonic: Optional[float] = None
        self.cancel_requested = False
        self._process: Optional[Any] = None
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None

    # --- snapshot ------------------------------------------------------
    def snapshot(self) -> Dict[str, Any]:
        """A plain, JSON-safe view. The page never touches the handle."""
        with self._lock:
            elapsed = (
                time.monotonic() - self.started_monotonic
                if self.started_monotonic is not None
                else 0.0
            )
            return {
                "run_id": self.run_id,
                "run_dir": str(self.run_dir),
                "status": self.status,
                "event": dict(self.event),
                "tail": list(self.tail),
                "elapsed_s": elapsed,
                "cancel_requested": self.cancel_requested,
                "finished": self.status in rm.TERMINAL_STATUSES,
                "returncode": (self.result or {}).get("returncode"),
            }

    def is_alive(self) -> bool:
        thread = self._thread
        return bool(thread and thread.is_alive())


class RunManager:
    """Tracks the runs this process started. One instance per application."""

    def __init__(self) -> None:
        self._runs: Dict[str, RunHandle] = {}
        self._lock = threading.Lock()

    # --- lifecycle ------------------------------------------------------
    def start(
        self,
        *,
        run_id: str,
        run_dir: str | Path,
        plan: Dict[str, Any],
        repo_root: str | Path,
        env: Optional[Dict[str, str]] = None,
        on_finished: Optional[Callable[[str, Dict[str, Any]], None]] = None,
    ) -> RunHandle:
        """Start ``plan`` in the background and return immediately.

        The manifest must already exist (the caller creates it at PLANNED, so
        a run is on disk before anything executes and a crash between the two
        leaves a record rather than a silent gap).
        """
        directory = Path(run_dir)
        handle = RunHandle(run_id, directory, plan)
        with self._lock:
            existing = self._runs.get(str(run_id))
            if (
                existing is not None
                and existing.is_alive()
                and existing.status not in rm.TERMINAL_STATUSES
            ):
                raise RuntimeError(f"Run {run_id} is already running.")
            self._runs[str(run_id)] = handle

        thread = threading.Thread(
            target=self._execute,
            args=(handle, Path(repo_root), env, on_finished),
            name=f"batteryml-run-{run_id}",
            daemon=True,
        )
        handle._thread = thread
        thread.start()
        return handle

    def _execute(
        self,
        handle: RunHandle,
        repo_root: Path,
        env: Optional[Dict[str, str]],
        on_finished: Optional[Callable[[str, Dict[str, Any]], None]],
    ) -> None:
        # Every path out of this method, exception or not, reaches exactly
        # one ``_finalize``. The setup is inside the guard too: ``mkdir`` on
        # an unwritable parent raises as readily as the executor does, and it
        # runs *before* the status leaves PLANNED, so leaving it outside
        # stranded the run one state earlier with the same result — a dead
        # thread that ``live_run_ids`` still calls live.
        #
        # ``result`` is bound first so the ``finally`` can never raise
        # NameError over the exception it is finalizing.
        started_monotonic = time.monotonic()
        result: Dict[str, Any] = _failure_result(
            handle.plan, RuntimeError("The run ended without reporting a result.")
        )
        try:
            result = self._run_to_completion(handle, repo_root, env)
        except BaseException as exc:  # noqa: BLE001 - see below
            # The executor raising is the one way a run could stay RUNNING
            # *forever*. This call was not guarded: an exception killed the
            # worker thread with ``handle.status`` still RUNNING, and because
            # ``live_run_ids`` reads the status rather than the thread, the
            # manager went on reporting the run as live. The page polled a
            # job whose worker was dead, and orphan recovery never applied
            # because the manager insisted it was alive. Reproduced by making
            # the executor raise: RUNNING in memory and on disk, worker
            # thread dead, still listed live, with no time bound at all.
            #
            # ``BaseException`` deliberately, and not re-raised: this is a
            # daemon worker thread, so a ``KeyboardInterrupt`` or
            # ``SystemExit`` delivered here has nowhere to propagate except
            # out of the thread — which is precisely the silent death that
            # stranded the run. The run is recorded FAILED, carrying the
            # exception text, which is what actually happened to it.
            result = _failure_result(handle.plan, exc)
        finally:
            self._finalize(
                handle,
                result,
                elapsed=time.monotonic() - started_monotonic,
                on_finished=on_finished,
            )

    def _run_to_completion(
        self,
        handle: RunHandle,
        repo_root: Path,
        env: Optional[Dict[str, str]],
    ) -> Dict[str, Any]:
        """Mark the run RUNNING and drive it to the executor's result.

        Raises whatever the setup or the executor raises; :meth:`_execute`
        owns the guard, so there is exactly one place that decides what an
        unexpected failure means.
        """
        handle.run_dir.mkdir(parents=True, exist_ok=True)
        live_log = handle.run_dir / LIVE_LOG
        with handle._lock:
            handle.status = rm.STATUS_RUNNING
            handle.started_monotonic = time.monotonic()
        try:
            rm.update_manifest(
                handle.run_dir,
                {"started_at": rm.utc_now_iso()},
                status=rm.STATUS_RUNNING,
            )
        except (FileNotFoundError, ValueError):
            # The manifest is the caller's responsibility; a missing or
            # already-terminal one must not stop the run from being tracked.
            pass

        def on_event(event: Dict[str, Any]) -> None:
            message = str(event.get("message") or "")
            # A heartbeat repeats the last observed state and says nothing
            # new. Appending it would fill both the live tail and run.log
            # with hundreds of identical "Running." lines and push the run's
            # real output out of view — which is exactly what a log is for.
            # The event itself is still stored, so elapsed time and the last
            # counter keep updating while the run is quiet.
            if event.get("heartbeat"):
                with handle._lock:
                    handle.event = dict(event)
                return
            with handle._lock:
                handle.event = dict(event)
                if message:
                    handle.tail.append(message)
            if message:
                # Appended as it arrives, so a run killed mid-flight still
                # leaves readable output behind.
                try:
                    with live_log.open("a", encoding="utf-8") as stream:
                        stream.write(message + "\n")
                except OSError:
                    pass

        def launcher(argv: List[str], **kwargs: Any) -> Any:
            # start_new_session puts the child in its own process group, which
            # is what makes a targeted group-kill possible without touching
            # anything else on the machine.
            kwargs["start_new_session"] = True
            process = subprocess.Popen(argv, **kwargs)
            with handle._lock:
                handle._process = process
            if handle.cancel_requested:
                # Cancelled between the click and the spawn.
                self._signal_process(process, signal.SIGTERM)
            return process

        return execute_plan_streaming(
            handle.plan,
            on_event=on_event,
            repo_root=repo_root,
            env=env,
            popen=launcher,
        )

    def _finalize(
        self,
        handle: RunHandle,
        result: Dict[str, Any],
        *,
        elapsed: float,
        on_finished: Optional[Callable[[str, Dict[str, Any]], None]],
    ) -> None:
        """Settle one run: durable record first, then the in-memory flag.

        **The ordering here is the contract.** ``snapshot()["finished"]`` is
        what every consumer reacts to, so it must not become true until the
        run is genuinely settled — and the manifest is the durable half of
        that. The previous order published the in-memory terminal status
        first and wrote the manifest afterwards, which left a window where
        the two sources of truth disagreed. Measured: the manifest was still
        RUNNING at the instant ``finished`` first became true in 20 of 20
        repetitions, typically for ~2ms but once for 446ms on an idle
        machine.

        That window is not academic, because of what the page does inside
        it. ``_finish_active_run`` reacts to ``finished`` by calling
        ``forget()`` — which drops the handle, so the run leaves
        ``live_run_ids()`` — and then reruns the script, and Run History
        renders ``list_runs(..., live_run_ids=...)``. A manifest still
        saying RUNNING that is no longer live is exactly what
        ``reconcile_status`` treats as an abandoned job, so a run that had
        just exited 0 was reported as INTERRUPTED — "not known to have
        finished". Reproduced through the real ``forget``/``list_runs``
        path: 15 of 15.

        Writing the manifest first closes it. ``finished`` now implies the
        durable record is already terminal, so whatever a consumer does next
        — poll, forget, re-read, restart — it sees one settled state.

        The in-memory publication is in a ``finally``: a failed manifest
        write must not be able to strand the run at RUNNING, which would
        trade a millisecond window for the permanent one this also fixes.
        """
        cancelled = handle.cancel_requested
        if cancelled:
            status = rm.STATUS_CANCELLED
        elif result.get("success"):
            status = rm.STATUS_COMPLETED
        else:
            status = rm.STATUS_FAILED

        try:
            self._write_logs(handle.run_dir, result)
            changes: Dict[str, Any] = {
                "finished_at": rm.utc_now_iso(),
                "elapsed_seconds": round(elapsed, 3),
                "return_code": result.get("returncode"),
                "artifacts": result.get("artifacts") or {},
                "command": [str(part) for part in (result.get("command") or [])],
            }
            if cancelled:
                changes["failure"] = {
                    "category": "cancelled_by_user",
                    "message": CANCELLED_MESSAGE,
                }
                # Explicitly: no metrics. A run stopped part-way has not
                # measured anything, and its artifacts are partial.
                changes["metrics"] = {}
                changes["partial_artifacts"] = True
            elif not result.get("success"):
                changes["failure"] = {
                    "category": "execution_failed",
                    "message": str(result.get("error") or "")
                    or f"The run exited with code {result.get('returncode')}.",
                }
            try:
                rm.update_manifest(handle.run_dir, changes, status=status)
            except (FileNotFoundError, ValueError):
                # No manifest, or one already terminal. Neither can strand a
                # run: there is nothing on disk left claiming RUNNING.
                pass
        finally:
            with handle._lock:
                handle.status = status
                handle.result = result

        if on_finished is not None:
            try:
                on_finished(handle.run_id, result)
            except Exception:  # noqa: BLE001 - a callback bug must not crash the worker
                pass

    @staticmethod
    def _write_logs(run_dir: Path, result: Dict[str, Any]) -> None:
        """Persist the separated streams once the run has ended.

        Written from what the streaming executor actually captured — which
        includes everything produced before a cancellation, because the
        pumps drain the pipes after the child dies.
        """
        for name, key in ((rm.STDOUT_LOG, "stdout"), (rm.STDERR_LOG, "stderr")):
            try:
                (run_dir / name).write_text(str(result.get(key) or ""), encoding="utf-8")
            except OSError:
                pass

    # --- cancellation ---------------------------------------------------
    @staticmethod
    def _signal_process(process: Any, sig: int) -> None:
        """Signal one child's process group, and nothing else.

        The group id is read from the child's own pid, so the blast radius is
        the process tree this manager started. A ``ProcessLookupError`` means
        it already exited, which is success for our purposes.
        """
        pid = getattr(process, "pid", None)
        if pid is None:
            return
        # ``killpg``/``getpgid`` are POSIX-only.  Windows still needs a
        # targeted cancellation path, so fall back to the Popen handle rather
        # than attempting to call an API that does not exist on that host.
        kill_group = getattr(os, "killpg", None)
        get_group = getattr(os, "getpgid", None)
        if kill_group is None or get_group is None:
            try:
                process.send_signal(sig)
            except Exception:  # noqa: BLE001
                pass
            return
        try:
            kill_group(get_group(pid), sig)
        except (ProcessLookupError, PermissionError, OSError):
            # Fall back to the single process; still targeted, never broad.
            try:
                process.send_signal(sig)
            except Exception:  # noqa: BLE001
                pass

    def cancel(self, run_id: str, *, grace_s: float = CANCEL_GRACE_S) -> bool:
        """Terminate exactly the run named by ``run_id``.

        Returns True when a live process was signalled. SIGTERM first so the
        child can close its files; SIGKILL only if it is still alive after
        ``grace_s``.
        """
        handle = self.get(run_id)
        if handle is None:
            return False
        with handle._lock:
            handle.cancel_requested = True
            process = handle._process
        if process is None:
            return False
        if process.poll() is not None:
            return False

        self._signal_process(process, signal.SIGTERM)
        if self._wait_for_exit(process, grace_s):
            return True

        # It ignored SIGTERM. SIGKILL cannot be ignored, but it is still
        # asynchronous: the caller is told the run stopped, so wait for the
        # process to actually be reaped rather than reporting on a signal we
        # merely sent.
        self._signal_process(process, signal.SIGKILL)
        self._wait_for_exit(process, KILL_WAIT_S)
        return True

    @staticmethod
    def _wait_for_exit(process: Any, timeout_s: float) -> bool:
        """Poll until the child is reaped, or ``timeout_s`` elapses."""
        deadline = time.monotonic() + max(0.0, timeout_s)
        while time.monotonic() < deadline:
            if process.poll() is not None:
                return True
            time.sleep(0.05)
        return process.poll() is not None

    # --- inspection ------------------------------------------------------
    def get(self, run_id: str) -> Optional[RunHandle]:
        with self._lock:
            return self._runs.get(str(run_id))

    def snapshot(self, run_id: str) -> Optional[Dict[str, Any]]:
        handle = self.get(run_id)
        return handle.snapshot() if handle is not None else None

    def live_run_ids(self) -> List[str]:
        """Runs this process is still tracking as not-yet-terminal.

        What :func:`run_manifest.list_runs` needs in order to tell a genuinely
        running run from a stale ``RUNNING`` manifest left by a previous
        process.
        """
        with self._lock:
            return [
                run_id
                for run_id, handle in self._runs.items()
                if handle.status not in rm.TERMINAL_STATUSES
            ]

    def forget(self, run_id: str) -> None:
        """Drop a finished run from memory. Its manifest stays on disk.

        Terminal status, not thread liveness, is the condition. A caller
        that reacts to the terminal snapshot arrives while the worker thread
        is still returning, so gating on ``is_alive()`` made this a silent
        no-op in exactly that window — the window every page uses — and
        handles accumulated for the lifetime of the process.

        **Calling this is safe the moment a snapshot says finished**, which
        is the point of the ordering in :meth:`_finalize`: the manifest is
        already terminal by then, so dropping the handle cannot leave the
        run looking abandoned. It could before. ``forget`` here, followed by
        Run History reading ``list_runs(..., live_run_ids=...)``, is what
        turned a run that had just exited 0 into a reported INTERRUPTED.

        Dropping the handle does not touch the worker: it holds its own
        reference. What is dropped is only this manager's ability to hand
        out a snapshot of a run that has already ended, which is what the
        manifest is for.
        """
        with self._lock:
            handle = self._runs.get(str(run_id))
            if handle is None:
                return
            if handle.status in rm.TERMINAL_STATUSES or not handle.is_alive():
                del self._runs[str(run_id)]


#: The application's manager. A module-level instance because Streamlit
#: re-executes the script on every interaction: a manager stored in session
#: state would be rebuilt constantly and would lose track of its own threads.
_MANAGER: Optional[RunManager] = None
_MANAGER_LOCK = threading.Lock()


def get_run_manager() -> RunManager:
    with _MANAGER_LOCK:
        global _MANAGER
        if _MANAGER is None:
            _MANAGER = RunManager()
        return _MANAGER
