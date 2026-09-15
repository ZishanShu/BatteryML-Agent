"""Streaming execution of a validated plan, with structured progress events.

``execute_plan`` blocks until the BatteryML wrapper exits, which makes the
Streamlit page look frozen for the whole run. This module runs the *same*
validated plan through ``subprocess.Popen`` and reports what the run is
doing while it does it.

Contract, deliberately identical to :func:`batteryml_web.agent.executor.execute_plan`:

* the plan is re-validated by the same
  :func:`~batteryml_web.agent.executor.validate_plan_for_execution` gate;
  nothing that gate refuses ever starts;
* the command is an argv list executed with ``shell=False``;
* stdout and stderr are preserved **in full**, carriage returns included, so
  a tqdm bar and a traceback both survive intact;
* a non-zero return code is reported honestly and **nothing is retried**;
* no metric is ever read out of the output. Scientific results come only
  from :func:`batteryml_web.agent.results.build_result_context`, which reads
  the canonical ``scores`` saved by BatteryML's own evaluator.

What is added is a callback: ``on_event(event)`` is invoked with small
JSON-serializable dicts as the run progresses::

    {"type": "stage" | "training_progress" | "output" | "completed",
     "stage": str | None,
     "current": int | None,
     "total": int | None,
     "fraction": float | None,
     "message": str}

``fraction`` is filled in **only** when the run itself printed a counter.
Where no machine-readable progress exists the event carries ``None`` and the
caller shows a stage and an elapsed time instead of an invented percentage.

Streamlit independence: the reader threads never call the callback. They
only decode bytes into a queue; ``on_event`` is invoked from the thread that
called :func:`execute_plan_streaming`, so a Streamlit caller can update its
widgets directly and a headless caller sees no threading at all.
"""

from __future__ import annotations

import codecs
import os
import queue
import re
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from .executor import _empty_artifacts, _repo_root, discover_artifacts, validate_plan_for_execution

EventCallback = Callable[[Dict[str, Any]], None]

#: Stage names the UI may display. Ordered as the run reaches them.
STAGE_VALIDATING = "Validating plan"
STAGE_LOADING = "Loading data"
STAGE_TRAINING = "Training"
STAGE_EVALUATING = "Evaluating"
STAGE_READING = "Reading results"
STAGE_COMPLETE = "Complete"
STAGE_FAILED = "Failed"

STAGES: Tuple[str, ...] = (
    STAGE_VALIDATING,
    STAGE_LOADING,
    STAGE_TRAINING,
    STAGE_EVALUATING,
    STAGE_READING,
    STAGE_COMPLETE,
)

#: A tqdm line: "<desc>:  10%|#         | 1/10 [00:06<00:54,  6.80s/it]".
#: Anchored on the percent-bar-counter shape, so a plain sentence that merely
#: contains "1/10" is not mistaken for progress.
_TQDM_RE = re.compile(
    r"^(?P<desc>.*?):\s*(?P<percent>\d{1,3})%\|[^|]*\|\s*(?P<current>\d+)\s*/\s*(?P<total>\d+)"
)

#: tqdm descriptions that mean training. ``Traning`` is the upstream typo in
#: ``batteryml/models/nn_model.py``; ``Training <Model>`` is what the
#: extension models in ``batteryml_web/models/`` emit.
_TRAINING_DESC_PREFIXES: Tuple[str, ...] = ("training", "traning")

#: Lines BatteryML really prints while it is building the dataset.
#: ``batteryml/task.py`` emits the two tqdm bars; ``batteryml/pipeline.py``
#: prints the seed line and the cache line.
_LOADING_MARKERS: Tuple[str, ...] = (
    "reading train data",
    "reading test data",
    "load datasets from cache",
    "seed is set to",
)

_READ_CHUNK = 65536

#: The message of the periodic "still running" event. A heartbeat carries the
#: last *observed* stage and progress and never advances either, so a caller
#: can refresh its elapsed-time display without a percentage being invented.
HEARTBEAT_MESSAGE = "Running."

#: How long the event loop waits for output before emitting a heartbeat.
_HEARTBEAT_INTERVAL_S = 0.2

#: Segments longer than this are truncated in the *event* only; the stored
#: stdout/stderr keep every byte.
_EVENT_MESSAGE_CHARS = 500


def parse_progress_line(line: str) -> Optional[Dict[str, Any]]:
    """Read a tqdm counter out of one output segment.

    Returns ``{"desc", "percent", "current", "total", "fraction"}`` or
    ``None`` when the segment carries no machine-readable counter. ``None``
    is the honest answer: a fabricated percentage is worse than none.

    ``fraction`` is ``current / total`` computed from the counter, not from
    tqdm's own rounded percent, and is ``None`` when ``total`` is 0.
    """
    match = _TQDM_RE.match(str(line or "").strip())
    if match is None:
        return None
    current = int(match.group("current"))
    total = int(match.group("total"))
    return {
        "desc": match.group("desc").strip(),
        "percent": int(match.group("percent")),
        "current": current,
        "total": total,
        "fraction": (current / total) if total > 0 else None,
    }


def is_training_progress(desc: str) -> bool:
    """Whether a tqdm description belongs to the training loop."""
    lowered = str(desc or "").strip().lower()
    return any(lowered.startswith(prefix) for prefix in _TRAINING_DESC_PREFIXES)


def classify_stage(line: str) -> Optional[str]:
    """The stage one output segment proves the run has reached, or None.

    Only markers the pipeline actually prints are recognised. Anything else —
    a traceback, a warning, an unrecognised log line — returns ``None`` so
    the caller keeps the last stage it genuinely observed.
    """
    text = str(line or "").strip().lower()
    if not text:
        return None
    progress = parse_progress_line(line)
    if progress is not None and is_training_progress(progress["desc"]):
        return STAGE_TRAINING
    if any(marker in text for marker in _LOADING_MARKERS):
        return STAGE_LOADING
    return None


def _decode_pump(
    stream: Any,
    channel: str,
    events: "queue.Queue[Tuple[str, str, str]]",
    raw_parts: List[str],
) -> None:
    """Read one pipe to EOF, storing raw text and queueing segments.

    Runs in its own thread and never touches the callback, Streamlit, or any
    shared mutable state other than its own ``raw_parts`` list and the queue.

    Reads through :func:`os.read` rather than ``readline`` because tqdm ends
    each update with a carriage return and no newline: ``readline`` would
    block until the bar finished, which is exactly the freeze this module
    exists to remove. Segments are split on ``\\r`` **and** ``\\n``; the raw
    text keeps both.
    """
    decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
    pending = ""
    fileno = stream.fileno()
    try:
        while True:
            chunk = os.read(fileno, _READ_CHUNK)
            if not chunk:
                break
            text = decoder.decode(chunk)
            if not text:
                continue
            raw_parts.append(text)
            pending += text
            segments = re.split(r"[\r\n]", pending)
            pending = segments.pop()
            for segment in segments:
                if segment.strip():
                    events.put(("segment", channel, segment))
    except (OSError, ValueError) as exc:
        # The pipe was closed under us (kill/timeout). Report it as a
        # segment instead of dying silently in a daemon thread.
        events.put(("segment", channel, f"[stream {channel} closed: {type(exc).__name__}: {exc}]"))
    finally:
        flushed = decoder.decode(b"", final=True)
        if flushed:
            raw_parts.append(flushed)
        tail = pending + flushed
        if tail.strip():
            events.put(("segment", channel, tail))
        # Popen does not close its pipes on wait() (only communicate() does),
        # so the reader closes the one it owns and the descriptor is released
        # as soon as the stream ends.
        try:
            stream.close()
        except OSError:
            pass
        events.put(("eof", channel, ""))


class _ProgressState:
    """The last progress the run actually reported. Never extrapolated."""

    def __init__(self) -> None:
        self.stage: str = STAGE_VALIDATING
        self.current: Optional[int] = None
        self.total: Optional[int] = None
        self.fraction: Optional[float] = None
        self.training_started: bool = False

    def observe(self, segment: str) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """Fold one segment in. Returns ``(stage_changed, progress or None)``."""
        progress = parse_progress_line(segment)
        stage = classify_stage(segment)
        changed = False

        if progress is not None and is_training_progress(progress["desc"]):
            self.training_started = True
            self.current = progress["current"]
            self.total = progress["total"]
            self.fraction = progress["fraction"]
            if self.stage != STAGE_TRAINING:
                self.stage = STAGE_TRAINING
                changed = True
            elif (
                progress["total"] > 0
                and progress["current"] >= progress["total"]
            ):
                # The CLI always runs --train then --eval (CLAUDE.md §5), so
                # once the training counter is full the remaining work before
                # exit is evaluation. Inferred from the command contract, not
                # from a printed marker: BatteryML prints nothing when
                # evaluation begins.
                self.stage = STAGE_EVALUATING
                self.current = None
                self.total = None
                self.fraction = None
                changed = True
            return changed, progress

        if stage is not None and stage != self.stage:
            # Never walk backwards: a late "Reading test data" bar during
            # evaluation must not reset the stage to Loading data.
            if not (stage == STAGE_LOADING and self.training_started):
                self.stage = stage
                self.current = None
                self.total = None
                self.fraction = None
                changed = True
        return changed, progress


def _event(
    event_type: str,
    state: _ProgressState,
    message: str,
    *,
    current: Optional[int] = None,
    total: Optional[int] = None,
    fraction: Optional[float] = None,
) -> Dict[str, Any]:
    return {
        "type": event_type,
        "stage": state.stage,
        "current": current,
        "total": total,
        "fraction": fraction,
        "message": message[:_EVENT_MESSAGE_CHARS],
    }


def _emit(on_event: Optional[EventCallback], event: Dict[str, Any]) -> None:
    """Deliver one event. A callback that raises must not kill the run."""
    if on_event is None:
        return
    try:
        on_event(event)
    except Exception as exc:  # noqa: BLE001 - a display bug must not stop a run
        # Reported once, on the run's own error channel, rather than hidden:
        # the subprocess keeps going and its output is still captured.
        print(f"[progress callback raised {type(exc).__name__}: {exc}]")


def execute_plan_streaming(
    plan: Dict[str, Any],
    *,
    on_event: Optional[EventCallback] = None,
    timeout: Optional[float] = None,
    env: Optional[Dict[str, str]] = None,
    repo_root: str | Path | None = None,
    cwd: str | Path | None = None,
    popen: Optional[Callable[..., Any]] = None,
) -> Dict[str, Any]:
    """Run a validated plan, reporting progress through ``on_event``.

    Args:
        plan: dict from
            :func:`batteryml_web.agent.execution.build_execution_plan`;
            re-validated here by the executor's own gate.
        on_event: optional callback invoked **on the calling thread** with
            the structured events documented in the module docstring.
        timeout: seconds before the child is killed (None = no limit).
        env: environment mapping for the child (None = inherit).
        repo_root: repository root for workspace containment and ``cwd``.
        cwd: working directory for the child (defaults to the repo root).
        popen: injectable ``subprocess.Popen`` replacement for tests.

    Returns the same dict shape as
    :func:`batteryml_web.agent.executor.execute_plan` — ``started``,
    ``completed``, ``success``, ``returncode``, ``command``, ``workspace``,
    ``stdout``, ``stderr``, ``error``, ``artifacts``, ``dry_run`` — plus
    ``final_stage`` and ``elapsed_s``. ``stdout``/``stderr`` hold the
    complete output, carriage returns included.
    """
    root = Path(repo_root) if repo_root is not None else _repo_root()
    state = _ProgressState()
    _emit(on_event, _event("stage", state, "Validating the execution plan."))

    checked = validate_plan_for_execution(plan, repo_root=root)
    base: Dict[str, Any] = {
        "started": False,
        "completed": False,
        "success": False,
        "returncode": None,
        "command": checked.get("argv", []),
        "workspace": checked.get("workspace"),
        "stdout": "",
        "stderr": "",
        "error": checked.get("error"),
        "artifacts": _empty_artifacts(),
        "dry_run": False,
        "final_stage": STAGE_FAILED,
        "elapsed_s": 0.0,
    }
    if not checked["ok"]:
        state.stage = STAGE_FAILED
        _emit(on_event, _event("completed", state, str(checked.get("error") or "refused")))
        return base

    argv: List[str] = checked["argv"]
    launcher = popen if popen is not None else subprocess.Popen
    started_at = time.monotonic()
    state.stage = STAGE_LOADING
    _emit(on_event, _event("stage", state, "Starting the BatteryML wrapper."))

    try:
        proc = launcher(
            argv,
            cwd=str(cwd) if cwd is not None else str(root),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
            env=env,
            shell=False,
        )
    except OSError as exc:
        base["error"] = f"execution could not start ({type(exc).__name__}: {exc})."
        state.stage = STAGE_FAILED
        _emit(on_event, _event("completed", state, str(base["error"])))
        return base

    base["started"] = True
    stdout_parts: List[str] = []
    stderr_parts: List[str] = []
    events: "queue.Queue[Tuple[str, str, str]]" = queue.Queue()
    threads = [
        threading.Thread(
            target=_decode_pump,
            args=(proc.stdout, "stdout", events, stdout_parts),
            daemon=True,
            name="batteryml-stdout",
        ),
        threading.Thread(
            target=_decode_pump,
            args=(proc.stderr, "stderr", events, stderr_parts),
            daemon=True,
            name="batteryml-stderr",
        ),
    ]
    for thread in threads:
        thread.start()

    open_streams = len(threads)
    timed_out = False
    while open_streams > 0:
        if timeout is not None and (time.monotonic() - started_at) > timeout and not timed_out:
            timed_out = True
            proc.kill()
        try:
            kind, channel, payload = events.get(timeout=_HEARTBEAT_INTERVAL_S)
        except queue.Empty:
            # No output for a moment: still alive, so the caller can refresh
            # its elapsed-time display without any progress being invented.
            _emit(
                on_event,
                _event(
                    "stage",
                    state,
                    HEARTBEAT_MESSAGE,
                    current=state.current,
                    total=state.total,
                    fraction=state.fraction,
                ),
            )
            continue
        if kind == "eof":
            open_streams -= 1
            continue
        changed, progress = state.observe(payload)
        if progress is not None and is_training_progress(progress["desc"]):
            _emit(
                on_event,
                _event(
                    "training_progress",
                    state,
                    payload,
                    current=progress["current"],
                    total=progress["total"],
                    fraction=progress["fraction"],
                ),
            )
        elif progress is not None:
            _emit(
                on_event,
                _event(
                    "stage",
                    state,
                    payload,
                    current=progress["current"],
                    total=progress["total"],
                    fraction=progress["fraction"],
                ),
            )
        elif changed:
            _emit(on_event, _event("stage", state, payload))
        else:
            _emit(on_event, _event("output", state, payload))

    for thread in threads:
        thread.join(timeout=5)
    returncode = proc.wait()
    elapsed = time.monotonic() - started_at

    base["completed"] = not timed_out
    base["returncode"] = returncode
    base["stdout"] = "".join(stdout_parts)
    base["stderr"] = "".join(stderr_parts)
    base["elapsed_s"] = elapsed
    base["success"] = (returncode == 0) and not timed_out

    if timed_out:
        base["error"] = f"execution timed out after {timeout}s; nothing retried."
    elif base["success"]:
        state.stage = STAGE_READING
        _emit(on_event, _event("stage", state, "Reading result artifacts."))
        base["artifacts"] = discover_artifacts(str(checked["workspace"]), repo_root=root)
        base["error"] = None
    else:
        base["error"] = (
            f"wrapper exited with returncode {returncode}; "
            "no retry performed, no metrics fabricated."
        )

    state.stage = STAGE_COMPLETE if base["success"] else STAGE_FAILED
    base["final_stage"] = state.stage
    _emit(
        on_event,
        _event(
            "completed",
            state,
            base["error"] or f"Finished in {elapsed:.1f}s.",
            fraction=1.0 if base["success"] else None,
        ),
    )
    return base
