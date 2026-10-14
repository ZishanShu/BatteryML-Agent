from __future__ import annotations

import html
import os
import queue
import re
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Optional, Sequence

import streamlit as st


# tqdm uses ANSI cursor controls when it refreshes a bar in a terminal.
# Streamlit is not a terminal, so those controls would appear literally as
# sequences such as ``\x1b[A`` in the browser.
_ANSI_ESCAPE_RE = re.compile(r"\x1b(?:\[[0-?]*[ -/]*[@-~]|[@-_])")


def _strip_terminal_controls(text: str) -> str:
    """Remove terminal-only ANSI controls while preserving line breaks."""
    return _ANSI_ESCAPE_RE.sub("", text).replace("\x08", "")


def get_safe_runtime_env() -> dict:
    env = os.environ.copy()
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("OPENBLAS_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    env.setdefault("VECLIB_MAXIMUM_THREADS", "1")
    env.setdefault("NUMEXPR_NUM_THREADS", "1")
    env.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    env.setdefault("PYTHONFAULTHANDLER", "1")
    return env


def _format_elapsed(seconds: float) -> str:
    total = max(0, int(seconds))
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def _render_progress_log(log_box: Any, logs: list[str]) -> None:
    """Render the current tqdm-style update as one refreshing line.

    tqdm advances its display with carriage returns rather than emitting a
    new log record.  Streamlit's text iterator can expose those updates as a
    series of lines, so rendering every captured chunk makes one progress bar
    grow into a wall of repeated percentages.  Keep the complete chunks in
    ``logs`` for the durable log and show only the newest non-empty carriage
    return segment here.
    """
    output = "Waiting for command output..."
    for chunk in reversed(logs):
        clean_chunk = _strip_terminal_controls(chunk)
        segments = [part.strip() for part in re.split(r"[\r\n]+", clean_chunk) if part.strip()]
        if segments:
            output = segments[-1]
            break
    escaped = html.escape(output)
    log_box.markdown(
        "<div class=\"bml-progress-log-scroll\" aria-label=\"Latest command output\">"
        f"<pre>{escaped}</pre>"
        "</div>",
        unsafe_allow_html=True,
    )


def _run_with_live_stages(
    command: Sequence[str],
    cwd: Optional[Path],
    log_path: Path,
    host: Any,
) -> dict:
    """A model run drawn exactly as Agent Workflow draws one.

    Same event stream (:func:`batteryml_web.agent.streaming.stream_command`:
    stages and counters parsed from BatteryML's own tqdm output, a heartbeat
    while it is quiet) and the same renderer (``_LiveProgress``: a stage line
    with elapsed time and, once measurable, an ETA; a bar only when the run
    printed a counter; the raw output behind a technical-log disclosure).
    Run Model and Quick experience used a fixed bar that never moved and
    showed the command twice; now all three run surfaces read alike.
    """
    # Imported here: agent_workflow imports this module at load time.
    from batteryml_web.agent.streaming import stream_command
    from batteryml_web.ui.agent_workflow import _LiveProgress

    log_path.parent.mkdir(parents=True, exist_ok=True)
    status_slot = host.empty()
    bar_slot = host.empty()
    log_slot = host.empty()
    bar_slot.progress(0)
    live = _LiveProgress(status_slot, bar_slot, log_slot)

    result = stream_command(
        [str(part) for part in command],
        cwd=str(cwd) if cwd else os.getcwd(),
        on_event=live.on_event,
        env=get_safe_runtime_env(),
    )
    output = _strip_terminal_controls(
        str(result.get("stdout") or "")
        + (("\n" + str(result.get("stderr"))) if result.get("stderr") else "")
    )
    lines = output.splitlines(keepends=True)
    output_tail = "".join(lines[-200:])
    return_code = result.get("returncode")
    elapsed = float(result.get("elapsed_s") or 0.0)
    success = bool(result.get("success"))

    # The live technical log gives way to the final one below.
    log_slot.empty()
    if success:
        bar_slot.progress(100)
    else:
        if result.get("error") and not result.get("started"):
            host.error(str(result["error"]))
        if return_code in {139, -11} or "segmentation fault" in output_tail.lower():
            host.error(
                "Native dependency crash / segmentation fault detected. Possible causes: xgboost, libomp, numpy, sklearn, torch, or cache incompatibility."
            )
        host.error(f"Return code: {return_code}")
    host.caption(f"Log file: `{log_path}`")
    details = host.container(key="bml_progress_log_details")
    details.expander("Logs", expanded=not success).code(
        output_tail or "No logs captured.", language="text"
    )

    log_path.write_text(
        f"COMMAND: {' '.join(str(part) for part in command)}\nReturn code: {return_code}\nElapsed: {elapsed:.2f}s\n\n{output}",
        encoding="utf-8",
    )
    return {
        "success": success,
        "return_code": return_code,
        "elapsed": elapsed,
        "log_path": str(log_path),
        "output_tail": output_tail,
        "output": output,
    }


def run_command_with_progress(
    command: Sequence[str],
    cwd: Optional[Path],
    log_path: Path,
    steps: list[tuple[int, str]],
    container: Any = None,
    live_stages: bool = False,
) -> dict:
    """Run ``command``, drawing its progress into ``container``.

    ``container`` is where every element this function draws goes — the bar,
    the status line, the elapsed time, the command box and the log. It
    defaults to ``st``, i.e. whatever container is current at the call site.

    It exists because the two Run buttons live in ``st.columns`` so that they
    sit beside another control, and anything drawn under a button inherits
    that column's width: the run progress bar came out at 66% of the content
    area on Run Model and 49% on the Home demo. A progress bar is a
    full-width element — it is read as a proportion of the page — so the
    caller hands in a container created *outside* the columns and the bar
    spans the content area while the buttons stay side by side.

    Nothing here sets a pixel width. The bar fills whatever container it is
    given, so it stays responsive and cannot overflow.
    """
    # Every element below is drawn through this, never through `st` directly,
    # so a caller's container is honoured for all of them or for none.
    host = container if container is not None else st
    if live_stages:
        # A BatteryML model run: drawn like Agent Workflow's (stages, real
        # counters, ETA, technical log). ``steps`` is not used — every stage
        # shown comes from the run's own output.
        return _run_with_live_stages(command, cwd, log_path, host)

    start = time.perf_counter()
    log_path.parent.mkdir(parents=True, exist_ok=True)

    progress = host.progress(0)
    status_text = host.empty()
    elapsed_text = host.empty()
    cmd_box = host.empty()
    log_surface = host.container(key="bml_progress_log")
    log_box = log_surface.empty()
    logs: list[str] = []

    cmd_box.code(" ".join(str(part) for part in command), language="bash")

    def update(percent: int, message: str) -> None:
        progress.progress(max(0, min(percent, 100)))
        status_text.info(message)
        elapsed_text.caption(f"Elapsed time: {_format_elapsed(time.perf_counter() - start)}")

    update(steps[0][0] if steps else 0, steps[0][1] if steps else "Starting...")

    process = subprocess.Popen(
        [str(part) for part in command],
        cwd=str(cwd) if cwd else None,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=get_safe_runtime_env(),
    )

    update(steps[1][0] if len(steps) > 1 else 10, steps[1][1] if len(steps) > 1 else "Running command...")
    running_percent = steps[1][0] if len(steps) > 1 else 10

    # Output is read on a helper thread so this (the script) thread can
    # refresh the page every second even while BatteryML prints nothing —
    # BatLiNet's feature extraction runs for minutes without a line, and the
    # page then sat on "Check dataset" with "Elapsed time: 00:00:00", which
    # reads as frozen. Only this thread touches Streamlit.
    lines: "queue.Queue[Optional[str]]" = queue.Queue()

    def _pump(stream) -> None:
        try:
            for raw in stream:
                lines.put(raw)
        finally:
            lines.put(None)

    reader = None
    if process.stdout is not None:
        reader = threading.Thread(target=_pump, args=(process.stdout,), daemon=True)
        reader.start()
        stream_open = True
        last_tick = 0.0
        while stream_open:
            try:
                raw = lines.get(timeout=1.0)
            except queue.Empty:
                raw = ""
            if raw is None:
                stream_open = False
            elif raw:
                logs.append(_strip_terminal_controls(raw))
                _render_progress_log(log_box, logs)
            now = time.perf_counter()
            if stream_open and now - last_tick >= 1.0:
                last_tick = now
                # A truthful status: the run is in progress; the bar holds
                # where it is because BatteryML reports no percentage.
                update(running_percent, "BatteryML is running. Output appears below as it is printed.")
        reader.join(timeout=5)

    return_code = process.wait()
    output_tail = "".join(logs[-200:])
    elapsed = time.perf_counter() - start

    for percent, message in steps[2:]:
        update(percent, message)
        time.sleep(0.03)

    success = return_code == 0
    if success:
        progress.progress(100)
        status_text.success(steps[-1][1] if steps else "Done")
    else:
        progress.progress(100)
        status_text.error("Task failed. Please check logs below.")
        if return_code in {139, -11} or "segmentation fault" in output_tail.lower():
            host.error(
                "Native dependency crash / segmentation fault detected. Possible causes: xgboost, libomp, numpy, sklearn, torch, or cache incompatibility."
            )
        host.error(f"Return code: {return_code}")

    elapsed_text.caption(f"Elapsed time: {_format_elapsed(elapsed)}")
    host.caption(f"Log file: `{log_path}`")
    with host.container(key="bml_progress_log_details"):
        with host.expander("Logs", expanded=not success):
            st.code(output_tail or "No logs captured.", language="text")

    # The whole output, not the on-screen tail: this file is the run's log,
    # and a failure's cause is often far above its last 200 lines.
    output = "".join(logs)
    log_path.write_text(
        f"COMMAND: {' '.join(str(part) for part in command)}\nReturn code: {return_code}\nElapsed: {elapsed:.2f}s\n\n{output}",
        encoding="utf-8",
    )

    return {
        "success": success,
        "return_code": return_code,
        "elapsed": elapsed,
        "log_path": str(log_path),
        "output_tail": output_tail,
        "output": output,
    }


def run_data_command_with_progress(
    command: Sequence[str],
    cwd: Optional[Path],
    log_path: Path,
    steps: list[tuple[int, str]],
    container: Any = None,
) -> dict:
    """Run a download or preprocessing command through the same progress UI.

    Keeping data preparation as a named operation makes the intent at the
    workflow surface clear while reusing the tested command runner below it.
    """
    return run_command_with_progress(
        command=command,
        cwd=cwd,
        log_path=log_path,
        steps=steps,
        container=container,
    )
