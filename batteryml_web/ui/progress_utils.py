from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path
from typing import Any, Optional, Sequence

import streamlit as st


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


def run_command_with_progress(
    command: Sequence[str],
    cwd: Optional[Path],
    log_path: Path,
    steps: list[tuple[int, str]],
    container: Any = None,
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
    start = time.perf_counter()
    log_path.parent.mkdir(parents=True, exist_ok=True)

    # Every element below is drawn through this, never through `st` directly,
    # so a caller's container is honoured for all of them or for none.
    host = container if container is not None else st

    progress = host.progress(0)
    status_text = host.empty()
    elapsed_text = host.empty()
    cmd_box = host.empty()
    log_box = host.empty()
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

    if process.stdout is not None:
        for line in process.stdout:
            logs.append(line)
            log_box.code("".join(logs[-80:]), language="text")

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
    with host.expander("Logs", expanded=not success):
        st.code(output_tail or "No logs captured.", language="text")

    log_path.write_text(
        f"COMMAND: {' '.join(str(part) for part in command)}\nReturn code: {return_code}\nElapsed: {elapsed:.2f}s\n\n{output_tail}",
        encoding="utf-8",
    )

    return {
        "success": success,
        "return_code": return_code,
        "elapsed": elapsed,
        "log_path": str(log_path),
        "output_tail": output_tail,
    }
