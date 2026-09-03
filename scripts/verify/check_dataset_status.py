"""DATA-01 — per-dataset raw/processed state.

For every dataset BatteryML ships a preprocessor for, reports: raw present,
raw file count and size, processed present and non-empty, and whether the
processed output is older than the raw input.

Emits ``dataset_status.csv`` and ``dataset_status.json``. Never downloads
anything and never creates a directory under ``data/`` — creating an empty
``data/processed/<DATASET>`` to silence an assertion is forbidden by
CLAUDE.md §7, and this check exists partly to catch that having happened.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .common import (
    FAIL,
    PASS,
    SKIP,
    CheckResult,
    dir_stats,
    known_datasets,
    relative_to_repo,
    repo_root,
    run_check,
    write_csv,
    write_json,
)

CHECK_ID = "DATA-01"
CHECK_NAME = "dataset_status"

# BatteryData files that preprocessing writes.
PROCESSED_GLOB = "*.pkl"

CSV_HEADER = [
    "dataset",
    "raw_path",
    "raw_present",
    "raw_file_count",
    "raw_mib",
    "raw_newest_utc",
    "processed_path",
    "processed_present",
    "processed_pkl_count",
    "processed_mib",
    "processed_newest_utc",
    "processed_older_than_raw",
    "state",
]

# Per-dataset states, in the order a user would act on them.
STATE_READY = "ready"
STATE_STALE = "processed_older_than_raw"
STATE_EMPTY_PROCESSED = "processed_dir_empty"
STATE_NEEDS_PREPROCESS = "needs_preprocess"
STATE_NO_RAW = "raw_missing"


def _iso(timestamp: float | None) -> str | None:
    if not timestamp:
        return None
    return datetime.fromtimestamp(timestamp, tz=timezone.utc).isoformat(timespec="seconds")


def _dataset_row(root: Path, dataset: str) -> dict[str, Any]:
    """Collect the raw/processed facts for one dataset. Reads only; never writes."""
    raw_dir = root / "data" / "raw" / dataset
    processed_dir = root / "data" / "processed" / dataset

    raw = dir_stats(raw_dir)
    processed = dir_stats(processed_dir, PROCESSED_GLOB)

    raw_present = bool(raw.get("exists") and raw.get("is_dir") and raw.get("file_count"))
    processed_present = bool(processed.get("exists") and processed.get("is_dir"))
    processed_count = int(processed.get("file_count") or 0)

    raw_newest = raw.get("newest_mtime")
    processed_newest = processed.get("newest_mtime")
    stale = bool(
        raw_newest and processed_newest and processed_newest < raw_newest
    )

    if not processed_present or processed_count == 0:
        if processed_present and processed_count == 0:
            state = STATE_EMPTY_PROCESSED
        elif raw_present:
            state = STATE_NEEDS_PREPROCESS
        else:
            state = STATE_NO_RAW
    elif stale:
        state = STATE_STALE
    else:
        state = STATE_READY

    return {
        "dataset": dataset,
        "raw_path": relative_to_repo(raw_dir),
        "raw_present": raw_present,
        "raw_file_count": int(raw.get("file_count") or 0),
        "raw_mib": raw.get("total_mib") or 0.0,
        "raw_newest_utc": _iso(raw_newest),
        "processed_path": relative_to_repo(processed_dir),
        "processed_present": processed_present,
        "processed_pkl_count": processed_count,
        "processed_mib": processed.get("total_mib") or 0.0,
        "processed_newest_utc": _iso(processed_newest),
        "processed_older_than_raw": stale,
        "state": state,
    }


def check(output_dir: Path | None = None) -> CheckResult:
    """Report raw/processed state for every dataset with a registered preprocessor."""
    root = repo_root()
    output_dir = Path(output_dir) if output_dir else root / "runs" / "verify"

    datasets, source = known_datasets()
    if not datasets:
        return CheckResult(
            CHECK_ID,
            CHECK_NAME,
            FAIL,
            "No datasets could be discovered. Expected either "
            "batteryml.preprocess.SUPPORTED_SOURCES['DATASETS'] or "
            "batteryml/preprocess/preprocess_<NAME>.py files.",
            {"dataset_source": source},
        )

    rows = [_dataset_row(root, dataset) for dataset in sorted(datasets)]

    data_dir = root / "data"
    payload = {
        "check_id": CHECK_ID,
        "dataset_source": source,
        "data_dir": relative_to_repo(data_dir),
        "data_dir_exists": data_dir.exists(),
        "processed_glob": PROCESSED_GLOB,
        "datasets": rows,
    }
    artifacts = [
        write_json(output_dir / "dataset_status.json", payload),
        write_csv(
            output_dir / "dataset_status.csv",
            CSV_HEADER,
            [[row[column] for column in CSV_HEADER] for row in rows],
        ),
    ]

    by_state: dict[str, list[str]] = {}
    for row in rows:
        by_state.setdefault(row["state"], []).append(row["dataset"])

    evidence = {
        "dataset_source": source,
        "data_dir_exists": data_dir.exists(),
        "dataset_count": len(rows),
        "by_state": by_state,
        "datasets": rows,
    }

    if not data_dir.exists():
        return CheckResult(
            CHECK_ID,
            CHECK_NAME,
            SKIP,
            f"{relative_to_repo(data_dir)} does not exist, so no dataset has raw or "
            f"processed files to report on. {len(rows)} dataset(s) are registered. "
            "This check never downloads; place raw data under data/raw/<DATASET>/ first.",
            evidence,
            artifacts,
        )

    # An empty processed directory is DATA-02's failure to report: it owns that
    # condition over the whole data/processed/ tree, not only the registered
    # datasets seen here, so it is reported once rather than by both checks.
    empty = by_state.get(STATE_EMPTY_PROCESSED, [])

    ready = by_state.get(STATE_READY, [])
    stale = by_state.get(STATE_STALE, [])
    needs = by_state.get(STATE_NEEDS_PREPROCESS, [])
    no_raw = by_state.get(STATE_NO_RAW, [])

    summary = (
        f"{len(ready)} ready, {len(stale)} stale, {len(needs)} awaiting preprocessing, "
        f"{len(no_raw)} without raw data, of {len(rows)} registered dataset(s)."
    )
    if stale:
        summary += f" Stale (processed older than raw): {', '.join(stale)} — re-run preprocessing."
    if empty:
        summary += (
            f" {len(empty)} processed directory/directories are empty "
            f"({', '.join(empty)}); DATA-02 reports that as a failure."
        )
    return CheckResult(CHECK_ID, CHECK_NAME, PASS, summary, evidence, artifacts)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for dataset_status.csv and dataset_status.json (default: runs/verify/).",
    )
    args = parser.parse_args(argv)

    result = run_check(CHECK_ID, CHECK_NAME, check, output_dir=args.output_dir)
    print(f"[{result.status}] {result.check_id} {result.name}: {result.message}")
    for artifact in result.artifacts:
        print(f"       wrote {artifact}")
    return 1 if result.status == FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
