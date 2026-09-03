"""DATA-02 — no directory under data/processed/ exists while holding no BatteryData.

An empty ``data/processed/<X>/`` is the single most dangerous state this
repository can be in, because it does not look like an error:

* ``BaseTrainTestSplitter.__init__`` asserts only that the path *exists*
  (``batteryml/train_test_split/base.py:25``), which an empty directory
  satisfies;
* it then globs ``*.pkl`` and gets nothing, so ``self._file_list`` is empty;
* training proceeds on zero cells instead of failing.

`CLAUDE.md` §7 forbids creating one, but the tooling used to create them by
accident: ``bin/batteryml.py`` makes the output directory *before* running the
preprocessor, and a preprocessor that matches no raw files returns ``(0, 0)``
and exits 0. ``batteryml_web/core/run_batteryml_cli.py`` now refuses that case
and cleans up after itself, and this check is the standing guarantee that no
such directory survives anywhere.

Deliberately scans **every** directory under ``data/processed/``, not only the
registered datasets that DATA-01 knows about: a directory left by a typo, an
abandoned custom upload, or a renamed dataset is exactly as dangerous and
exactly as invisible. DATA-01 owns the per-dataset inventory and reports this
condition as a state; the failure lives here so it is reported once.

Never deletes anything — it names what to remove and why.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from .common import (
    FAIL,
    PASS,
    SKIP,
    CheckResult,
    known_datasets,
    relative_to_repo,
    repo_root,
    run_check,
    write_json,
)

CHECK_ID = "DATA-02"
CHECK_NAME = "empty_processed_dirs"

PROCESSED_ROOT = Path("data") / "processed"
PROCESSED_GLOB = "*.pkl"


def _inspect(directory: Path, registered: set[str]) -> dict[str, Any]:
    """Facts about one directory under data/processed/. Reads only."""
    try:
        entries = list(directory.iterdir())
    except OSError as exc:
        return {
            "directory": relative_to_repo(directory),
            "name": directory.name,
            "registered_dataset": directory.name in registered,
            "error": f"{type(exc).__name__}: {exc}",
            "processed_pkl_count": 0,
            "other_file_count": 0,
            "empty": False,
        }

    pkl = [p for p in entries if p.is_file() and p.suffix == ".pkl"]
    others = [p for p in entries if p.is_file() and p.suffix != ".pkl"]
    subdirectories = [p for p in entries if p.is_dir()]
    nested_pkl = sum(1 for p in directory.rglob(PROCESSED_GLOB) if p.is_file()) - len(pkl)

    return {
        "directory": relative_to_repo(directory),
        "name": directory.name,
        "registered_dataset": directory.name in registered,
        "processed_pkl_count": len(pkl),
        "other_file_count": len(others),
        "subdirectory_count": len(subdirectories),
        "nested_pkl_count": nested_pkl,
        "empty": len(pkl) == 0,
        "error": None,
    }


def check(output_dir: Path | None = None) -> CheckResult:
    """Fail if any directory under data/processed/ holds no .pkl file."""
    root = repo_root()
    output_dir = Path(output_dir) if output_dir else root / "runs" / "verify"
    processed_root = root / PROCESSED_ROOT

    registered = set(known_datasets()[0])

    if not processed_root.is_dir():
        return CheckResult(
            CHECK_ID,
            CHECK_NAME,
            SKIP,
            f"{relative_to_repo(processed_root)} does not exist, so there is no "
            "directory that could be empty. Nothing to check.",
            {"processed_root": relative_to_repo(processed_root), "exists": False},
        )

    directories = sorted(p for p in processed_root.iterdir() if p.is_dir())
    records = [_inspect(directory, registered) for directory in directories]
    empty = [record for record in records if record["empty"] and not record["error"]]
    unreadable = [record for record in records if record["error"]]

    payload = {
        "check_id": CHECK_ID,
        "processed_root": relative_to_repo(processed_root),
        "directory_count": len(records),
        "empty_count": len(empty),
        "directories": records,
    }
    artifacts = [write_json(output_dir / "empty_processed_dirs.json", payload)]

    evidence = {
        "processed_root": relative_to_repo(processed_root),
        "directory_count": len(records),
        "empty": [record["directory"] for record in empty],
        "unreadable": [
            {"directory": record["directory"], "error": record["error"]}
            for record in unreadable
        ],
        "directories": records,
    }

    if unreadable:
        first = unreadable[0]
        return CheckResult(
            CHECK_ID,
            CHECK_NAME,
            FAIL,
            f"{len(unreadable)} directory/directories under "
            f"{relative_to_repo(processed_root)} could not be read. First: "
            f"{first['directory']} — {first['error']}.",
            evidence,
            artifacts,
        )

    if empty:
        names = ", ".join(record["directory"] for record in empty)
        nested = [record for record in empty if record["nested_pkl_count"]]
        detail = ""
        if nested:
            detail = (
                f" {len(nested)} of them hold .pkl files in subdirectories rather than "
                f"at the top level, where the splitter looks: "
                f"{', '.join(record['directory'] for record in nested)}."
            )
        return CheckResult(
            CHECK_ID,
            CHECK_NAME,
            FAIL,
            f"{len(empty)} directory/directories under "
            f"{relative_to_repo(processed_root)} contain no {PROCESSED_GLOB} file: "
            f"{names}.{detail} An empty processed directory is worse than a missing "
            "one: it satisfies BaseTrainTestSplitter's existence assert "
            "(batteryml/train_test_split/base.py:25) and then yields an empty cell "
            "list, so training proceeds on no data instead of failing. Either "
            "preprocess the dataset (python scripts/preprocess_sweep.py --only <NAME>) "
            "or remove the directory (rmdir succeeds only if it is truly empty).",
            evidence,
            artifacts,
        )

    if not records:
        return CheckResult(
            CHECK_ID,
            CHECK_NAME,
            PASS,
            f"{relative_to_repo(processed_root)} exists but contains no dataset "
            "directories; nothing can be silently empty.",
            evidence,
            artifacts,
        )

    return CheckResult(
        CHECK_ID,
        CHECK_NAME,
        PASS,
        f"All {len(records)} directory/directories under "
        f"{relative_to_repo(processed_root)} contain at least one {PROCESSED_GLOB} file.",
        evidence,
        artifacts,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for empty_processed_dirs.json (default: runs/verify/).",
    )
    args = parser.parse_args(argv)

    result = run_check(CHECK_ID, CHECK_NAME, check, output_dir=args.output_dir)
    print(f"[{result.status}] {result.check_id} {result.name}: {result.message}")
    for artifact in result.artifacts:
        print(f"       wrote {artifact}")
    return 1 if result.status == FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
