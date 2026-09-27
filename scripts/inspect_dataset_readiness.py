"""Print what the platform sees on disk for each dataset. Read-only.

Run this on the server when the website's dataset status disagrees with what
you believe you downloaded. It answers one question:

    is this a code bug, or is the data somewhere the platform has never been
    told about?

For every dataset it prints each raw location the platform looks at, whether
that location's signature holds, the processed directory, and the readiness
status that follows.

**It only reads.** Nothing is deleted, copied, renamed, extracted,
downloaded or preprocessed. The only filesystem calls are existence checks,
file counts and — for an archive — a ZIP header read.

Usage, from the repository root::

    python scripts/inspect_dataset_readiness.py
    python scripts/inspect_dataset_readiness.py MATR XJTU
    python scripts/inspect_dataset_readiness.py --json

``--json`` prints the same facts as one JSON document, for pasting back.
"""

from __future__ import annotations

import argparse
import json
import sys
import zipfile
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from batteryml_web.core.readiness import (  # noqa: E402
    BATTERYLIFE_RAW_PARTS,
    PLATFORM_DATASETS,
    PROCESSED_GLOB,
    _dir_file_count,
    _native_preprocessor_datasets,
    _recognised_raw_layouts,
    _repo_root,
    assess_dataset,
)

FOUND = "FOUND"
ABSENT = "absent"


def _candidate_rows(name: str, root: Path) -> list[dict[str, Any]]:
    """Every raw location the platform checks, and what is actually there.

    Reported whether or not the signature holds, because "the platform never
    looks here" and "the platform looks here and finds nothing" are
    different findings and the second is the one that needs a data move.
    """
    v11 = root.joinpath(*BATTERYLIFE_RAW_PARTS)
    native = root / "data" / "raw" / name
    extracted = v11 / name
    archive = v11 / f"{name}.zip"

    rows: list[dict[str, Any]] = []

    rows.append(
        {
            "label": "raw native",
            "path": str(native),
            "exists": native.exists(),
            "kind": "directory" if native.is_dir() else ("file" if native.is_file() else "-"),
            "files": _dir_file_count(native) if native.is_dir() else 0,
        }
    )
    rows.append(
        {
            "label": "raw BatteryLife (extracted)",
            "path": str(extracted),
            "exists": extracted.exists(),
            "kind": "directory" if extracted.is_dir() else ("file" if extracted.is_file() else "-"),
            "files": _dir_file_count(extracted) if extracted.is_dir() else 0,
        }
    )
    readable_zip = archive.is_file() and zipfile.is_zipfile(archive)
    rows.append(
        {
            "label": "raw BatteryLife (archive)",
            "path": str(archive),
            "exists": archive.exists(),
            "kind": (
                "readable zip"
                if readable_zip
                else ("NOT a readable zip" if archive.is_file() else "-")
            ),
            "files": 1 if readable_zip else 0,
        }
    )
    return rows


def inspect(name: str, root: Path) -> dict[str, Any]:
    """One dataset's raw candidates, processed directory and status."""
    report = assess_dataset(name)
    processed = root / "data" / "processed" / name
    recognised = _recognised_raw_layouts(name, root=root)
    return {
        "dataset": name,
        "status": report.status.value,
        "candidates": _candidate_rows(name, root),
        "recognised_layouts": [
            {
                "kind": layout.kind,
                "path": str(layout.path),
                "files": layout.file_count,
                "preprocessable": layout.preprocessable,
            }
            for layout in recognised
        ],
        "processed_path": str(processed),
        "processed_exists": processed.is_dir(),
        "processed_files": _dir_file_count(processed, PROCESSED_GLOB)
        if processed.is_dir()
        else 0,
        "preprocessing_supported": report.preprocessing_supported,
        "reason": report.reason,
    }


def _print_human(entry: dict[str, Any]) -> None:
    print(f"{entry['dataset']}")
    for row in entry["candidates"]:
        mark = FOUND if row["files"] else ABSENT
        detail = f"{row['kind']}, {row['files']} file(s)" if row["exists"] else "not present"
        print(f"    {row['label']:29} {mark:6}  {detail}")
        print(f"        {row['path']}")
    processed = (
        f"{entry['processed_files']} {PROCESSED_GLOB} file(s)"
        if entry["processed_exists"]
        else "not present"
    )
    print(f"    {'processed':29} {FOUND if entry['processed_files'] else ABSENT:6}  {processed}")
    print(f"        {entry['processed_path']}")
    print(f"    status                        {entry['status']}")
    if entry["status"] == "RAW_ONLY":
        print(
            "    preprocessing                 "
            + (
                "supported by BatteryML"
                if entry["preprocessing_supported"]
                else "NOT available for this layout (raw detected only)"
            )
        )
    print()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Read-only dataset readiness inspection. Changes nothing.",
    )
    parser.add_argument(
        "datasets",
        nargs="*",
        help="Datasets to inspect. Default: every declared platform dataset.",
    )
    parser.add_argument(
        "--json", action="store_true", help="Emit JSON instead of a table."
    )
    args = parser.parse_args()

    root = _repo_root()
    names = list(args.datasets) or list(PLATFORM_DATASETS)
    entries = [inspect(name, root) for name in names]

    if args.json:
        print(json.dumps({"repo_root": str(root), "datasets": entries}, indent=2))
        return 0

    print(f"repository root: {root}")
    print(f"BatteryLife raw root: {root.joinpath(*BATTERYLIFE_RAW_PARTS)}")
    print(
        "datasets BatteryML can preprocess itself: "
        + ", ".join(sorted(_native_preprocessor_datasets()) or ["(batteryml not importable)"])
    )
    print()
    for entry in entries:
        _print_human(entry)

    counts: dict[str, int] = {}
    for entry in entries:
        counts[entry["status"]] = counts.get(entry["status"], 0) + 1
    print("summary: " + ", ".join(f"{k} {v}" for k, v in sorted(counts.items())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
