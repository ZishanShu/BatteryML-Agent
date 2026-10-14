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
     "message": str,
     "heartbeat": bool,
     "indeterminate": bool}

``fraction`` is filled in **only** when the run itself printed a counter.
Where no machine-readable progress exists the event carries ``None`` and the
caller shows a stage and an elapsed time instead of an invented percentage.

``heartbeat`` marks the periodic tick emitted while the run is quiet. It
repeats the last observed stage and counter and advances neither, so a
caller can refresh an elapsed time without anything being invented — and a
caller that keeps a log must skip it, or the log fills with identical lines.

``indeterminate`` says the current stage is one the run never puts a counter
on, so a caller can draw a spinner immediately rather than waiting for a bar
that is not coming.

The stages are named after the tqdm bars BatteryML actually prints
(``batteryml/task.py``, ``batteryml/models/nn_model.py``) plus two boundaries
inferred from the CLI's own call order; see
:meth:`_ProgressState._stage_after_completed_counter` for exactly which two
and why each follows from the code.

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
#:
#: The four data stages are named after the tqdm bars BatteryML actually
#: prints, rather than collapsed into one "Loading data". A run that spends a
#: minute reading cells and a minute expanding them into cycle samples was
#: reporting the same three words for both, which told the user nothing about
#: which half they were waiting on.
STAGE_VALIDATING = "Validating plan"
STAGE_PREPARING = "Preparing"
STAGE_LOADING = "Loading data"
STAGE_READING_TRAIN = "Reading train data"
STAGE_READING_TEST = "Reading test data"
STAGE_EXTRACTING_TRAIN = "Extracting train samples"
STAGE_EXTRACTING_TEST = "Extracting test samples"
STAGE_TRAINING = "Training"
STAGE_EVALUATING = "Evaluating"
STAGE_READING = "Reading results"
STAGE_COMPLETE = "Complete"
STAGE_FAILED = "Failed"

STAGES: Tuple[str, ...] = (
    STAGE_VALIDATING,
    STAGE_PREPARING,
    STAGE_LOADING,
    STAGE_READING_TRAIN,
    STAGE_READING_TEST,
    STAGE_EXTRACTING_TRAIN,
    STAGE_EXTRACTING_TEST,
    STAGE_TRAINING,
    STAGE_EVALUATING,
    STAGE_READING,
    STAGE_COMPLETE,
)

#: Stages that never carry a counter, so a caller knows to draw a spinner and
#: an elapsed time rather than waiting for a bar that will not arrive.
INDETERMINATE_STAGES: Tuple[str, ...] = (
    STAGE_VALIDATING,
    STAGE_PREPARING,
    STAGE_LOADING,
    STAGE_EVALUATING,
    STAGE_READING,
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

#: tqdm description -> stage, for the bars BatteryML actually prints. Each key
#: is the literal ``desc=`` argument at the cited line, lowercased; matching is
#: by prefix so a description the pipeline extends keeps resolving.
#:
#: * ``batteryml/task.py:79``  ``desc='Reading train data'``
#: * ``batteryml/task.py:81``  ``desc='Reading test data'``
#: * ``batteryml/task.py:150`` ``desc=f'Extracting {split_name} cycle samples'``
#:
#: Longest key first, so "extracting train" is tested before any shorter key
#: that could also prefix-match it.
_DESC_STAGES: Tuple[Tuple[str, str], ...] = (
    ("extracting train cycle samples", STAGE_EXTRACTING_TRAIN),
    ("extracting test cycle samples", STAGE_EXTRACTING_TEST),
    ("reading train data", STAGE_READING_TRAIN),
    ("reading test data", STAGE_READING_TEST),
)

#: Prose lines ``batteryml/pipeline.py`` prints around dataset construction.
#: These carry no counter, so they place the run in an indeterminate stage.
_LOADING_MARKERS: Tuple[str, ...] = (
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

#: How long the loop keeps reading after the child has exited and gone quiet
#: before it stops waiting for EOF. Generous on purpose: a healthy run EOFs
#: within milliseconds of exiting, so this window is only ever spent in the
#: pathological case where something the child spawned still holds the pipes.
#: Measured from the last byte received, not from the exit, so a large
#: buffered tail still arriving keeps pushing the deadline out and is never
#: truncated.
POST_EXIT_DRAIN_S = 5.0

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


def stage_for_description(desc: str) -> Optional[str]:
    """The stage a tqdm description names, or None for an unknown one.

    Training is checked first because its descriptions are open-ended
    (``Training BatLiNet``), then the four fixed data-stage descriptions.
    """
    lowered = str(desc or "").strip().lower()
    if not lowered:
        return None
    if is_training_progress(lowered):
        return STAGE_TRAINING
    for prefix, stage in _DESC_STAGES:
        if lowered.startswith(prefix):
            return stage
    return None


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
    if progress is not None:
        stage = stage_for_description(progress["desc"])
        if stage is not None:
            return stage
    if any(marker in text for marker in _LOADING_MARKERS):
        return STAGE_LOADING
    return None


def is_indeterminate_stage(stage: str) -> bool:
    """Whether a stage is one the run never puts a counter on.

    A caller uses this to decide between a progress bar and a spinner
    *before* a counter arrives, instead of waiting for one that never will.
    """
    return str(stage or "") in INDETERMINATE_STAGES


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


#: Stages a run only ever passes through once, in this order. Used solely to
#: refuse a *backwards* stage transition — a late "Reading test data" bar
#: emitted during evaluation must not drag the display back to loading. A
#: stage absent from this tuple (Complete, Failed) is set explicitly and is
#: not subject to the ordering rule.
_STAGE_ORDER: Tuple[str, ...] = (
    STAGE_VALIDATING,
    STAGE_PREPARING,
    STAGE_LOADING,
    STAGE_READING_TRAIN,
    STAGE_READING_TEST,
    STAGE_EXTRACTING_TRAIN,
    STAGE_EXTRACTING_TEST,
    STAGE_TRAINING,
    STAGE_EVALUATING,
    STAGE_READING,
)


def _stage_rank(stage: str) -> Optional[int]:
    try:
        return _STAGE_ORDER.index(str(stage))
    except ValueError:
        return None


class _ProgressState:
    """The last progress the run actually reported. Never extrapolated."""

    def __init__(self) -> None:
        self.stage: str = STAGE_VALIDATING
        self.current: Optional[int] = None
        self.total: Optional[int] = None
        self.fraction: Optional[float] = None
        self.training_started: bool = False
        #: The stage that printed the counter just observed, which is not
        #: always :attr:`stage`: a *full* counter both belongs to the stage
        #: that printed it and proves the run has moved past that stage. The
        #: event for that line must be attributed to the stage that earned
        #: it, or a finished 42/42 extraction would be drawn as a training
        #: bar sitting at 100% before training had begun.
        self.counter_stage: Optional[str] = None

    def _clear_counter(self) -> None:
        self.current = None
        self.total = None
        self.fraction = None

    def _advance_to(self, stage: str) -> bool:
        """Move to ``stage`` unless that would walk backwards.

        Returns whether the stage changed. The counter is cleared on a real
        change, because a counter belongs to the stage that printed it: a
        41/41 left over from reading is not 41/41 of extracting.
        """
        if stage == self.stage:
            return False
        here, there = _stage_rank(self.stage), _stage_rank(stage)
        if here is not None and there is not None and there < here:
            return False
        self.stage = stage
        self._clear_counter()
        return True

    def observe(
        self, segment: str
    ) -> Tuple[bool, Optional[Dict[str, Any]], Optional[str]]:
        """Fold one segment in.

        Returns ``(stage_changed, progress or None, entered or None)``, where
        ``entered`` is a stage the run is *inferred* to have begun because a
        counter finished — see
        :meth:`_stage_after_completed_counter`. The caller announces it as
        its own event, because nothing in the output announces it and a stage
        the display never reports is a stage the user never sees.
        """
        progress = parse_progress_line(segment)
        changed = False
        self.counter_stage = None

        if progress is not None:
            stage = stage_for_description(progress["desc"])
            if stage is None:
                # A counter whose description we do not recognise. It is not
                # evidence about where the run is, so it moves no stage — and
                # it is reported without its numbers, because a counter
                # attributed to a stage that did not print it is worse than
                # no counter.
                return False, progress, None
            changed = self._advance_to(stage)
            if not changed and stage != self.stage:
                # The advance was refused as backwards: a bar from an earlier
                # phase printed late. Its counter describes neither the stage
                # it names nor the stage the run is in, so it is not
                # attributed to either.
                return False, progress, None
            if stage == STAGE_TRAINING:
                self.training_started = True
            # Retained on the state, not only on the event, so the
            # heartbeats between two counter updates keep reporting the
            # last real counter instead of blanking the bar every 0.2s.
            self.current = progress["current"]
            self.total = progress["total"]
            self.fraction = progress["fraction"]
            self.counter_stage = stage
            entered = None
            if progress["total"] > 0 and progress["current"] >= progress["total"]:
                finished = self._stage_after_completed_counter(stage)
                if finished is not None and self._advance_to(finished):
                    changed = True
                    entered = finished
            return changed, progress, entered

        stage = classify_stage(segment)
        if stage is not None:
            changed = self._advance_to(stage)
        return changed, progress, None

    @staticmethod
    def _stage_after_completed_counter(stage: str) -> Optional[str]:
        """What a *full* counter proves the run has moved on to, if anything.

        Both entries are inferred from the command contract rather than from
        a printed marker, because BatteryML prints nothing at either
        boundary. They are the only two inferences in this module, and each
        follows from code that cannot reorder without this being updated:

        * ``bin/batteryml.py:115 run()`` calls ``pipeline.train()`` and then
          ``pipeline.evaluate()``, and the wrapper always passes both
          ``--train`` and ``--eval`` (CLAUDE.md §5). So a full training
          counter means evaluation is what remains.
        * ``Task.build`` (``batteryml/task.py:70``) reads train cells, reads
          test cells, extracts train samples, then extracts test samples —
          and returns. ``Pipeline.train`` then calls ``model.fit``. So a full
          *test extraction* counter means the dataset is built and fitting is
          what remains. This is what keeps a one-shot sklearn fit — Ridge,
          Lasso, Elastic Net, Random Forest, XGBoost — from sitting on
          "Extracting test samples" for its whole duration: those models emit
          no training counter at all, and none is waited for.
        """
        if stage == STAGE_TRAINING:
            return STAGE_EVALUATING
        if stage == STAGE_EXTRACTING_TEST:
            return STAGE_TRAINING
        return None


def _event(
    event_type: str,
    state: _ProgressState,
    message: str,
    *,
    current: Optional[int] = None,
    total: Optional[int] = None,
    fraction: Optional[float] = None,
    heartbeat: bool = False,
    stage: Optional[str] = None,
) -> Dict[str, Any]:
    """One structured progress event.

    ``heartbeat`` marks the periodic "still alive" tick. It is a flag rather
    than a message a consumer has to recognise by its text: a log that
    appended every heartbeat filled with hundreds of identical
    ``"Running."`` lines and pushed the run's real output out of view. A
    consumer that keeps a log skips these; one that draws an elapsed time
    uses them.

    ``stage`` overrides the state's current stage for this one event, and is
    used only to attribute a counter to the stage that printed it.
    """
    effective = stage or state.stage
    return {
        "type": event_type,
        "stage": effective,
        "current": current,
        "total": total,
        "fraction": fraction,
        "message": message[:_EVENT_MESSAGE_CHARS],
        "heartbeat": bool(heartbeat),
        "indeterminate": is_indeterminate_stage(effective),
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


def _stream_process(
    argv: List[str],
    *,
    cwd: str,
    env: Optional[Dict[str, str]],
    timeout: Optional[float],
    popen: Optional[Callable[..., Any]],
    state: "_ProgressState",
    on_event: Optional[EventCallback],
    base: Dict[str, Any],
) -> Optional[bool]:
    """Spawn ``argv`` and stream its progress into ``on_event``.

    Shared by :func:`execute_plan_streaming` (Agent Workflow, after its plan
    gate) and :func:`stream_command` (the protocol pages, whose callers have
    already validated the command). Fills ``started``, ``completed``,
    ``returncode``, ``stdout``, ``stderr``, ``elapsed_s``, ``success`` and,
    when relevant, ``error`` / ``stream_note`` on ``base``.

    Returns None when the process could not start (``base`` then carries the
    error and a ``completed`` event has been emitted), otherwise whether the
    run timed out.
    """
    launcher = popen if popen is not None else subprocess.Popen
    started_at = time.monotonic()
    # "Preparing", not "Loading data": between the spawn and the first tqdm
    # bar the child is importing torch and building the config, which is not
    # reading any data. The first real bar moves it on.
    state.stage = STAGE_PREPARING
    _emit(on_event, _event("stage", state, "Starting the BatteryML wrapper."))

    try:
        proc = launcher(
            argv,
            cwd=cwd,
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
        return None

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
    abandoned_streams = False
    #: When the child was first observed to have exited. The loop below ends
    #: on EOF from both pipes, which is the normal and complete ending — but
    #: it must not *depend* on it. A pipe is closed when the last holder of
    #: the write end closes it, and the child is not always the last holder:
    #: anything it spawned that inherited its stdout or stderr keeps the pipe
    #: open after the child itself has exited. This loop would then wait for
    #: an EOF that is not coming, with no timeout (RunManager passes none),
    #: so the run stayed RUNNING for the lifetime of the server process and
    #: the page polled a job that had already finished. Reproduced: a child
    #: that spawns a sleeper and exits leaves both streams open indefinitely.
    #: Last time anything at all arrived from either pipe. The drain window
    #: is measured from *activity*, not from the exit, so a child that exits
    #: while a large buffered tail is still being read is never truncated:
    #: every segment that arrives pushes the deadline out again.
    last_activity = time.monotonic()
    while open_streams > 0:
        if timeout is not None and (time.monotonic() - started_at) > timeout and not timed_out:
            timed_out = True
            proc.kill()
        if (
            proc.poll() is not None
            and (time.monotonic() - last_activity) > POST_EXIT_DRAIN_S
        ):
            # The child is gone and its output has been quiet for the whole
            # drain window; whatever still holds these pipes is not the run.
            # Stop waiting and let the terminal status be decided by the
            # return code, which is already available.
            abandoned_streams = True
            break
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
                    heartbeat=True,
                ),
            )
            continue
        last_activity = time.monotonic()
        if kind == "eof":
            open_streams -= 1
            continue
        changed, progress, entered = state.observe(payload)
        if progress is not None and state.counter_stage is not None:
            # Attributed to the stage that printed this counter, which is not
            # always the stage the run is now in: a full counter is both the
            # end of its own stage and the proof the next one has begun.
            _emit(
                on_event,
                _event(
                    "training_progress"
                    if is_training_progress(progress["desc"])
                    else "stage",
                    state,
                    payload,
                    current=progress["current"],
                    total=progress["total"],
                    fraction=progress["fraction"],
                    stage=state.counter_stage,
                ),
            )
        elif progress is not None:
            # An unrecognised bar: its text is kept, its numbers are not
            # attached to a stage that did not produce them.
            _emit(on_event, _event("output", state, payload))
        elif changed:
            _emit(on_event, _event("stage", state, payload))
        else:
            _emit(on_event, _event("output", state, payload))
        if entered is not None:
            # The inferred boundary, announced in its own event and carrying
            # no counter. Without this the run would enter Training (or
            # Evaluating) silently, and the display would keep showing the
            # completed bar of the stage before it until the next heartbeat.
            _emit(
                on_event,
                _event("stage", state, f"{entered} (inferred from the CLI's call order)."),
            )

    if not abandoned_streams:
        # Only when EOF really arrived. If the loop gave up on the pipes, the
        # pumps are blocked on a read nothing will end, and joining them —
        # even with a timeout — would spend that timeout per thread for
        # nothing. They are daemons, so they cost nothing to leave.
        for thread in threads:
            thread.join(timeout=5)
    # The child has already exited in every path that reaches here (the loop
    # ends on EOF, on the abandoned-streams break, or after a kill), so this
    # reaps it rather than waiting on it.
    returncode = proc.wait()
    elapsed = time.monotonic() - started_at

    base["completed"] = not timed_out
    base["returncode"] = returncode
    base["stdout"] = "".join(stdout_parts)
    base["stderr"] = "".join(stderr_parts)
    base["elapsed_s"] = elapsed
    base["success"] = (returncode == 0) and not timed_out

    if abandoned_streams:
        # Said plainly, because the captured output may be short of whatever
        # the other holder of the pipe wrote afterwards. The run's own
        # outcome is not in doubt: the return code below is the child's.
        base["stream_note"] = (
            f"The run exited with code {returncode}, but something it "
            f"started kept its output pipes open for more than "
            f"{POST_EXIT_DRAIN_S:.0f}s. The run was recorded from its exit "
            "code; any output written after that point is not captured."
        )

    return timed_out


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
    timed_out = _stream_process(
        argv,
        cwd=str(cwd) if cwd is not None else str(root),
        env=env,
        timeout=timeout,
        popen=popen,
        state=state,
        on_event=on_event,
        base=base,
    )
    if timed_out is None:
        return base
    returncode = base["returncode"]
    elapsed = base["elapsed_s"]

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


def stream_command(
    argv: List[str],
    *,
    cwd: str | Path,
    on_event: Optional[EventCallback] = None,
    env: Optional[Dict[str, str]] = None,
    timeout: Optional[float] = None,
    popen: Optional[Callable[..., Any]] = None,
) -> Dict[str, Any]:
    """Run an argv the caller has already validated, with the same events.

    The protocol pages (Run Model, Quick experience) build and gate their own
    BatteryML command, so they do not go through the Agent plan gate — but
    they report progress exactly as Agent Workflow does: the same stages, the
    same counters parsed from BatteryML's own tqdm output, the same heartbeat,
    the same ``completed`` event. Nothing here reads a metric or retries.

    Returns ``started``, ``completed``, ``success``, ``returncode``,
    ``command``, ``stdout``, ``stderr``, ``error``, ``final_stage`` and
    ``elapsed_s``.
    """
    state = _ProgressState()
    base: Dict[str, Any] = {
        "started": False,
        "completed": False,
        "success": False,
        "returncode": None,
        "command": [str(part) for part in argv],
        "stdout": "",
        "stderr": "",
        "error": None,
        "final_stage": STAGE_FAILED,
        "elapsed_s": 0.0,
    }
    timed_out = _stream_process(
        [str(part) for part in argv],
        cwd=str(cwd),
        env=env,
        timeout=timeout,
        popen=popen,
        state=state,
        on_event=on_event,
        base=base,
    )
    if timed_out is None:
        return base
    if timed_out:
        base["error"] = f"execution timed out after {timeout}s; nothing retried."
    elif not base["success"]:
        base["error"] = (
            f"BatteryML exited with return code {base['returncode']}; "
            "no retry performed, no metrics fabricated."
        )
    state.stage = STAGE_COMPLETE if base["success"] else STAGE_FAILED
    base["final_stage"] = state.stage
    _emit(
        on_event,
        _event(
            "completed",
            state,
            base["error"] or f"Finished in {base['elapsed_s']:.1f}s.",
            fraction=1.0 if base["success"] else None,
        ),
    )
    return base
