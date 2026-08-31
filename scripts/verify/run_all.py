"""Run every verification check and write verification_report.json plus a readable summary.

Usage::

    python -m scripts.verify.run_all
    python -m scripts.verify.run_all --only config_validity
    python -m scripts.verify.run_all --only CFG-01 --only ENV-01
    python -m scripts.verify.run_all --output-dir runs/web_projects/demo
    python -m scripts.verify.run_all --list

Exit codes:

* ``0`` — no check failed (PASS and SKIP only);
* ``1`` — at least one check returned FAIL;
* ``2`` — the ``--only`` filter matched no check (a usage error, not a result).

SKIP never fails the run: a check that cannot observe anything because the data
is absent has not found a defect. Use ``--strict`` to treat SKIP as failure when
running somewhere the data is expected to be present.
"""

from __future__ import annotations

import argparse
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from . import (
    check_cli_contract,
    check_config_validity,
    check_dataset_status,
    check_environment,
    check_registry_integrity,
    check_split_requirements,
)
from .common import FAIL, PASS, SKIP, CheckResult, relative_to_repo, repo_root, run_check, write_json, write_text

# Ordered so that the cheapest and most foundational checks report first: a
# broken environment explains most of what follows.
CHECKS: tuple[tuple[str, str, Any, bool], ...] = (
    (check_environment.CHECK_ID, check_environment.CHECK_NAME, check_environment.check, True),
    (check_cli_contract.CHECK_ID, check_cli_contract.CHECK_NAME, check_cli_contract.check, False),
    (check_dataset_status.CHECK_ID, check_dataset_status.CHECK_NAME, check_dataset_status.check, True),
    (check_config_validity.CHECK_ID, check_config_validity.CHECK_NAME, check_config_validity.check, True),
    (check_split_requirements.CHECK_ID, check_split_requirements.CHECK_NAME, check_split_requirements.check, True),
    (check_registry_integrity.CHECK_ID, check_registry_integrity.CHECK_NAME, check_registry_integrity.check, True),
)

DEFAULT_OUTPUT_DIR = Path("runs") / "verify"

_STATUS_ORDER = {FAIL: 0, SKIP: 1, PASS: 2}


def _selected(
    filters: list[str],
) -> tuple[list[tuple[str, str, Callable[..., CheckResult], bool]], list[str]]:
    """Apply ``--only`` filters by check id or name (case-insensitive)."""
    if not filters:
        return list(CHECKS), []

    wanted = {value.strip().lower() for value in filters if value.strip()}
    selected = [
        entry for entry in CHECKS if entry[0].lower() in wanted or entry[1].lower() in wanted
    ]
    matched = {entry[0].lower() for entry in selected} | {entry[1].lower() for entry in selected}
    unmatched = sorted(value for value in wanted if value not in matched)
    return selected, unmatched


def _summary_text(results: list[CheckResult], counts: dict[str, int], output_dir: Path) -> str:
    """A summary a human reads top to bottom without opening the JSON."""
    generated = datetime.now(timezone.utc).isoformat(timespec="seconds")
    lines = [
        "BatteryML verification summary",
        "=" * 60,
        f"generated (UTC)   {generated}",
        f"repository        {relative_to_repo(repo_root()) or repo_root().as_posix()}",
        f"python            {platform.python_version()} ({sys.executable})",
        f"output directory  {relative_to_repo(output_dir)}",
        "",
        f"{counts[PASS]} passed, {counts[FAIL]} failed, {counts[SKIP]} skipped "
        f"of {len(results)} check(s)",
        "",
    ]

    for result in sorted(results, key=lambda item: (_STATUS_ORDER[item.status], item.check_id)):
        lines.append(f"[{result.status:<4}] {result.check_id}  {result.name}  ({result.duration_s:.2f}s)")
        lines.append(f"        {result.message}")
        for artifact in result.artifacts:
            lines.append(f"        wrote: {artifact}")
        lines.append("")

    failed = [result for result in results if result.status == FAIL]
    if failed:
        lines.append("Act on these first:")
        for result in failed:
            lines.append(f"  - {result.check_id} {result.name}: {result.message}")
        lines.append("")

    skipped = [result for result in results if result.status == SKIP]
    if skipped:
        lines.append("Skipped (nothing observable, not a defect):")
        for result in skipped:
            lines.append(f"  - {result.check_id} {result.name}: {result.message}")
        lines.append("")

    lines.append(
        "Nothing in this report is a claim about pipeline behaviour on real data. "
        "These checks inspect the repository, the environment and the filesystem only; "
        "they never train, evaluate, download, or write into data/."
    )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--only",
        action="append",
        default=[],
        metavar="ID_OR_NAME",
        help="Run only this check, by id (ENV-01) or name (environment). Repeatable.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help=f"Where checks write their artifacts (default: {DEFAULT_OUTPUT_DIR.as_posix()}/).",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Treat SKIP as failure. Use where the data is expected to be present.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Print only the final counts; the summary file is written either way.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List the available checks and exit without running anything.",
    )
    args = parser.parse_args(argv)

    if args.list:
        for check_id, name, _, _ in CHECKS:
            print(f"{check_id}  {name}")
        return 0

    output_dir = Path(args.output_dir) if args.output_dir else repo_root() / DEFAULT_OUTPUT_DIR
    selected, unmatched = _selected(args.only)
    if unmatched:
        available = ", ".join(f"{check_id}/{name}" for check_id, name, _, _ in CHECKS)
        print(
            f"--only matched no check: {', '.join(unmatched)}. Available: {available}",
            file=sys.stderr,
        )
        return 2

    results: list[CheckResult] = []
    for check_id, name, function, takes_output_dir in selected:
        kwargs = {"output_dir": output_dir} if takes_output_dir else {}
        result = run_check(check_id, name, function, **kwargs)
        results.append(result)
        if not args.quiet:
            print(f"[{result.status:<4}] {result.check_id} {result.name}: {result.message}")

    counts = {status: sum(1 for result in results if result.status == status) for status in (PASS, FAIL, SKIP)}

    report = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "repo_root": repo_root().as_posix(),
        "python": {"version": platform.python_version(), "executable": sys.executable},
        "platform": platform.platform(),
        "output_dir": relative_to_repo(output_dir),
        "filters": args.only,
        "strict": bool(args.strict),
        "counts": counts,
        "checks": [result.to_dict() for result in results],
    }

    report_path = write_json(output_dir / "verification_report.json", report)
    summary_path = write_text(
        output_dir / "verification_summary.txt", _summary_text(results, counts, output_dir)
    )

    if not args.quiet:
        print("")
    print(
        f"{counts[PASS]} passed, {counts[FAIL]} failed, {counts[SKIP]} skipped "
        f"of {len(results)} check(s)"
    )
    print(f"report:  {report_path}")
    print(f"summary: {summary_path}")

    if counts[FAIL]:
        return 1
    if args.strict and counts[SKIP]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
