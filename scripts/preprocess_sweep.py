#!/usr/bin/env python
"""Preprocess every registered dataset that has raw data, one at a time, without stopping.

Iterates the datasets BatteryML ships a preprocessor for, skips those with no
raw data, and invokes preprocessing through ``batteryml_web/core/run_batteryml_cli.py``
per CLAUDE.md §5 — never ``python bin/batteryml.py``, which shadows the
``batteryml`` package.

Per dataset it captures stdout and stderr to ``logs/preprocess/<DATASET>.log``,
records wall-clock duration and the resulting ``.pkl`` count, and writes a row
to ``logs/preprocess/sweep_report.csv``. One dataset failing never stops the
sweep.

Failure handling, per CLAUDE.md §7: if a dataset fails and this script created
its ``data/processed/<DATASET>`` directory, the directory is removed and the
removal is reported. An empty ``data/processed/<DATASET>`` is never left
behind and never created to silence an assertion — an empty directory
satisfies ``BaseTrainTestSplitter``'s existence assert and then yields an
empty cell list.

Usage::

    python scripts/preprocess_sweep.py                      # every dataset with raw data
    python scripts/preprocess_sweep.py --dry-run            # print the commands, run nothing
    python scripts/preprocess_sweep.py --only MATR --only HUST
    python scripts/preprocess_sweep.py --skip RWTH
    python scripts/preprocess_sweep.py --force              # redo datasets already processed
    python scripts/preprocess_sweep.py --timeout 7200

Resumable: a dataset whose ``data/processed/<DATASET>`` already contains
``.pkl`` files is reported as ``already_processed`` and skipped unless
``--force`` is given. Re-running after an interruption therefore picks up where
the previous run stopped.

Exit codes: ``0`` if no dataset failed, ``1`` if any did, ``2`` for a usage
error (``--only`` naming an unknown dataset).
"""

from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.verify.common import known_datasets  # noqa: E402

WRAPPER = Path("batteryml_web") / "core" / "run_batteryml_cli.py"
RAW_ROOT = Path("data") / "raw"
PROCESSED_ROOT = Path("data") / "processed"
LOG_ROOT = Path("logs") / "preprocess"
REPORT_NAME = "sweep_report.csv"
PROCESSED_GLOB = "*.pkl"

DEFAULT_TIMEOUT_S = 4 * 60 * 60

# run_batteryml_cli.py exits with this when preprocessing produced no output;
# its stderr already names the raw path, the glob patterns that matched nothing
# and any directory it cleaned up, so the sweep surfaces that rather than
# repeating the diagnosis.
WRAPPER_EXIT_NO_OUTPUT = 3

# Outcomes, in the order a reader should act on them.
OUTCOME_OK = "ok"
OUTCOME_FAILED = "failed"
OUTCOME_TIMEOUT = "timeout"
OUTCOME_NO_RAW = "skipped_no_raw"
OUTCOME_ALREADY = "already_processed"
OUTCOME_EXCLUDED = "skipped_by_filter"
OUTCOME_DRY_RUN = "dry_run"

FAILURE_OUTCOMES = {OUTCOME_FAILED, OUTCOME_TIMEOUT}

CSV_HEADER = [
    "dataset",
    "outcome",
    "returncode",
    "duration_s",
    "raw_file_count",
    "raw_mib",
    "processed_pkl_count",
    "processed_mib",
    "removed_partial_output",
    "log_path",
    "command",
    "note",
]


def _dir_file_count_and_bytes(path: Path, pattern: str = "*") -> tuple[int, int]:
    """Number of files and total bytes under ``path``. ``(0, 0)`` if absent."""
    if not path.is_dir():
        return 0, 0
    count = 0
    total = 0
    for entry in path.rglob(pattern):
        if not entry.is_file():
            continue
        count += 1
        try:
            total += entry.stat().st_size
        except OSError:
            continue
    return count, total


def _mib(num_bytes: int) -> float:
    return round(num_bytes / (1024 * 1024), 3)


def preprocess_command(dataset: str) -> list[str]:
    """The exact command this sweep runs for one dataset (CLAUDE.md §5)."""
    return [
        sys.executable,
        str(WRAPPER),
        "preprocess",
        dataset,
        str(RAW_ROOT / dataset),
        str(PROCESSED_ROOT / dataset),
    ]


def _write_log_header(log_path: Path, dataset: str, command: list[str]) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(
        f"# BatteryML preprocessing log\n"
        f"# dataset: {dataset}\n"
        f"# started (UTC): {datetime.now(timezone.utc).isoformat(timespec='seconds')}\n"
        f"# cwd: {REPO_ROOT}\n"
        f"# command: {' '.join(command)}\n"
        f"# ---\n",
        encoding="utf-8",
    )


def _append_log(log_path: Path, text: str) -> None:
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(text)


def preprocess_one(
    dataset: str,
    *,
    force: bool,
    dry_run: bool,
    timeout_s: int,
) -> dict[str, Any]:
    """Preprocess one dataset. Never raises; always returns a report row."""
    raw_dir = REPO_ROOT / RAW_ROOT / dataset
    processed_dir = REPO_ROOT / PROCESSED_ROOT / dataset
    log_path = REPO_ROOT / LOG_ROOT / f"{dataset}.log"
    command = preprocess_command(dataset)

    raw_count, raw_bytes = _dir_file_count_and_bytes(raw_dir)
    row: dict[str, Any] = {
        "dataset": dataset,
        "outcome": OUTCOME_OK,
        "returncode": "",
        "duration_s": 0.0,
        "raw_file_count": raw_count,
        "raw_mib": _mib(raw_bytes),
        "processed_pkl_count": 0,
        "processed_mib": 0.0,
        "removed_partial_output": False,
        "log_path": (LOG_ROOT / f"{dataset}.log").as_posix(),
        "command": " ".join(command),
        "note": "",
    }

    if raw_count == 0:
        row["outcome"] = OUTCOME_NO_RAW
        row["log_path"] = ""
        row["note"] = (
            f"no raw files under {(RAW_ROOT / dataset).as_posix()}; this sweep never downloads. "
            f"Fetch the raw data first (batteryml download {dataset} {(RAW_ROOT / dataset).as_posix()})."
        )
        return row

    existing_pkl, existing_bytes = _dir_file_count_and_bytes(processed_dir, PROCESSED_GLOB)
    if existing_pkl and not force:
        row["outcome"] = OUTCOME_ALREADY
        row["processed_pkl_count"] = existing_pkl
        row["processed_mib"] = _mib(existing_bytes)
        row["log_path"] = ""
        row["note"] = (
            f"{(PROCESSED_ROOT / dataset).as_posix()} already holds {existing_pkl} "
            f"{PROCESSED_GLOB} file(s); pass --force to redo it."
        )
        return row

    if dry_run:
        row["outcome"] = OUTCOME_DRY_RUN
        row["log_path"] = ""
        row["note"] = "dry run: command not executed"
        return row

    # Track whether this run is the thing that brings the directory into
    # existence, so a failure can undo exactly what it created and nothing else.
    created_processed_dir = not processed_dir.exists()

    _write_log_header(log_path, dataset, command)
    started = time.monotonic()
    returncode: int | None = None
    outcome = OUTCOME_OK
    note = ""

    try:
        proc = subprocess.run(
            command,
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
        returncode = proc.returncode
        _append_log(log_path, proc.stdout or "")
        if proc.stderr:
            _append_log(log_path, "\n# --- stderr ---\n" + proc.stderr)
        if returncode != 0:
            outcome = OUTCOME_FAILED
            detail = (proc.stderr or "").strip().splitlines()
            reason = " ".join(line.strip() for line in detail[:4]) if detail else ""
            note = f"exit {returncode}; see {row['log_path']}" + (f". {reason}" if reason else "")
    except subprocess.TimeoutExpired as exc:
        outcome = OUTCOME_TIMEOUT
        note = f"no output within {timeout_s}s; raise --timeout or investigate"
        _append_log(
            log_path,
            (exc.stdout or "") if isinstance(exc.stdout, str) else "",
        )
        _append_log(log_path, f"\n# --- TIMEOUT after {timeout_s}s ---\n")
    except OSError as exc:
        outcome = OUTCOME_FAILED
        note = f"could not start the wrapper: {type(exc).__name__}: {exc}"
        _append_log(log_path, f"\n# --- {note} ---\n")

    duration = round(time.monotonic() - started, 2)
    pkl_count, pkl_bytes = _dir_file_count_and_bytes(processed_dir, PROCESSED_GLOB)

    # A zero-exit run that produced nothing is a failure, whatever the CLI said.
    if outcome == OUTCOME_OK and pkl_count == 0:
        outcome = OUTCOME_FAILED
        note = (
            f"the command exited 0 but wrote no {PROCESSED_GLOB} file to "
            f"{(PROCESSED_ROOT / dataset).as_posix()}; see {row['log_path']}. The "
            f"wrapper normally exits {WRAPPER_EXIT_NO_OUTPUT} with a diagnosis in that "
            f"case, so reaching this branch means the run bypassed it."
        )

    removed = False
    if outcome in FAILURE_OUTCOMES and pkl_count == 0 and created_processed_dir:
        # Remove only a directory this run created, and only when it holds no
        # BatteryData. Leaving it would satisfy BaseTrainTestSplitter's
        # existence assert and silently train on nothing (CLAUDE.md §7).
        try:
            shutil.rmtree(processed_dir)
            removed = True
            note = (
                f"{note}. Removed the partial output directory "
                f"{(PROCESSED_ROOT / dataset).as_posix()} that this run created"
            ).lstrip(". ")
            _append_log(
                log_path,
                f"\n# --- removed partial output {(PROCESSED_ROOT / dataset).as_posix()} ---\n",
            )
        except OSError as exc:
            note = (
                f"{note}. Could not remove the partial output directory "
                f"{(PROCESSED_ROOT / dataset).as_posix()}: {type(exc).__name__}: {exc}. "
                f"Remove it by hand — an empty processed directory silently yields an "
                f"empty cell list"
            ).lstrip(". ")

    row.update(
        {
            "outcome": outcome,
            "returncode": "" if returncode is None else returncode,
            "duration_s": duration,
            "processed_pkl_count": pkl_count,
            "processed_mib": _mib(pkl_bytes),
            "removed_partial_output": removed,
            "note": note,
        }
    )
    return row


def _write_report(rows: list[dict[str, Any]]) -> Path:
    report_path = REPO_ROOT / LOG_ROOT / REPORT_NAME
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(CSV_HEADER)
        for row in rows:
            writer.writerow([row[column] for column in CSV_HEADER])
    return report_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--only",
        action="append",
        default=[],
        metavar="DATASET",
        help="Preprocess only this dataset. Repeatable.",
    )
    parser.add_argument(
        "--skip",
        action="append",
        default=[],
        metavar="DATASET",
        help="Skip this dataset. Repeatable. Applied after --only.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run datasets whose processed directory already holds .pkl files.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the command for each dataset and run nothing.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_TIMEOUT_S,
        metavar="SECONDS",
        help=f"Per-dataset timeout (default: {DEFAULT_TIMEOUT_S}s).",
    )
    args = parser.parse_args(argv)

    if not (REPO_ROOT / WRAPPER).exists():
        print(
            f"error: {WRAPPER.as_posix()} does not exist. CLAUDE.md §5 requires every "
            "BatteryML invocation to go through the wrapper; running bin/batteryml.py "
            "directly shadows the batteryml package.",
            file=sys.stderr,
        )
        return 2

    datasets, dataset_source = known_datasets()
    if not datasets:
        print("error: no datasets are registered; nothing to sweep.", file=sys.stderr)
        return 2

    known = {name.upper(): name for name in datasets}
    requested = [value.strip().upper() for value in args.only if value.strip()]
    unknown = [value for value in requested if value not in known]
    if unknown:
        print(
            f"error: --only names unknown dataset(s): {', '.join(unknown)}. "
            f"Known: {', '.join(sorted(known.values()))}.",
            file=sys.stderr,
        )
        return 2

    skipped = {value.strip().upper() for value in args.skip if value.strip()}
    unknown_skip = sorted(value for value in skipped if value not in known)
    if unknown_skip:
        print(
            f"warning: --skip names unknown dataset(s), ignored: {', '.join(unknown_skip)}.",
            file=sys.stderr,
        )

    selected = [known[value] for value in requested] if requested else sorted(datasets)

    print(f"datasets from {dataset_source}")
    print(f"sweeping {len(selected)} dataset(s): {', '.join(selected)}")
    if args.dry_run:
        print("dry run: no command will be executed")
    print("")

    rows: list[dict[str, Any]] = []
    for dataset in selected:
        if dataset.upper() in skipped:
            rows.append(
                {
                    **{column: "" for column in CSV_HEADER},
                    "dataset": dataset,
                    "outcome": OUTCOME_EXCLUDED,
                    "duration_s": 0.0,
                    "raw_file_count": 0,
                    "raw_mib": 0.0,
                    "processed_pkl_count": 0,
                    "processed_mib": 0.0,
                    "removed_partial_output": False,
                    "command": "",
                    "note": "excluded by --skip",
                }
            )
            print(f"  [{OUTCOME_EXCLUDED}] {dataset}")
            continue

        print(f"  -> {dataset} ...", flush=True)
        row = preprocess_one(
            dataset, force=args.force, dry_run=args.dry_run, timeout_s=args.timeout
        )
        rows.append(row)
        detail = f" ({row['processed_pkl_count']} pkl, {row['duration_s']}s)" if row["duration_s"] else ""
        print(f"  [{row['outcome']}] {dataset}{detail}")
        if row["note"]:
            print(f"        {row['note']}")
        if args.dry_run:
            print(f"        {row['command']}")

    report_path = _write_report(rows)

    failed = [row["dataset"] for row in rows if row["outcome"] in FAILURE_OUTCOMES]
    succeeded = [row["dataset"] for row in rows if row["outcome"] == OUTCOME_OK]
    removed = [row["dataset"] for row in rows if row["removed_partial_output"]]

    print("")
    print(f"{len(succeeded)} preprocessed, {len(failed)} failed, of {len(rows)} dataset(s)")
    if removed:
        print(
            f"removed the partial output directory this run created for: {', '.join(removed)}"
        )
    if failed:
        print(f"failed: {', '.join(failed)} — see {(LOG_ROOT).as_posix()}/<DATASET>.log")
    print(f"report: {report_path.relative_to(REPO_ROOT).as_posix()}")

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
