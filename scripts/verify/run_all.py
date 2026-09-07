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
import importlib
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

from .common import FAIL, PASS, SKIP, CheckResult, relative_to_repo, repo_root, run_check, write_json, write_text

# (check id, name, module in this package, whether check() takes output_dir).
#
# Module names, not imported modules: a check whose own imports fail must not
# take the whole harness down with it. Three checks import pyyaml, and a broken
# environment is exactly when the harness is most needed — importing them here
# meant `--only ENV-01` died with ModuleNotFoundError instead of reporting the
# missing package. Each module is imported when it is about to run, and an
# import failure becomes a SKIP for that check alone.
#
# Ordered so that the cheapest and most foundational checks report first: a
# broken environment explains most of what follows.
CHECKS: tuple[tuple[str, str, str, bool], ...] = (
    ("ENV-01", "environment", "check_environment", True),
    ("CLI-01", "cli_contract", "check_cli_contract", False),
    ("DATA-01", "dataset_status", "check_dataset_status", True),
    ("CFG-01", "config_validity", "check_config_validity", True),
    # After CFG-01: that establishes the configs are sound, this establishes the
    # execution gate agrees with them.
    ("VAL-01", "validator_contract", "check_validator_contract", True),
    ("SPLIT-01", "split_requirements", "check_split_requirements", True),
    ("REG-01", "registry_integrity", "check_registry_integrity", True),
    ("EXT-01", "extension_contract", "extension_contract", True),
)

DEFAULT_OUTPUT_DIR = Path("runs") / "verify"

_STATUS_ORDER = {FAIL: 0, SKIP: 1, PASS: 2}


def _load_check(check_id: str, name: str, module_name: str) -> tuple[object | None, CheckResult | None]:
    """Import one check module, or return the SKIP result explaining why not.

    A check module that cannot be imported has observed nothing, so it skips
    rather than fails; ENV-01 is the check that reports the missing package
    loudly, and it has no third-party imports of its own precisely so that it
    still runs here.
    """
    try:
        return importlib.import_module(f".{module_name}", __package__), None
    except Exception as exc:  # noqa: BLE001 - reported as a result, never hidden
        missing = getattr(exc, "name", None)
        detail = (
            f"needs the package {missing!r}, which is not installed"
            if isinstance(exc, ModuleNotFoundError) and missing
            else f"raised {type(exc).__name__}: {exc}"
        )
        return None, CheckResult(
            check_id,
            name,
            SKIP,
            f"scripts/verify/{module_name}.py could not be imported: it {detail}. "
            "Install the dependencies with `pip install -r requirements.txt` and re-run; "
            "ENV-01 reports exactly which are missing. If the package is present, this "
            "is a defect in the check module — see evidence.traceback.",
            {
                "module": f"scripts.verify.{module_name}",
                "exception_type": type(exc).__name__,
                "exception": str(exc),
                "missing_module": missing,
            },
        )


def _selected(
    filters: list[str],
) -> tuple[list[tuple[str, str, str, bool]], list[str]]:
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
        for check_id, name, module_name, _ in CHECKS:
            print(f"{check_id}  {name}  (scripts/verify/{module_name}.py)")
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
    for check_id, name, module_name, takes_output_dir in selected:
        module, skip_result = _load_check(check_id, name, module_name)
        if module is None:
            result = skip_result
        else:
            kwargs = {"output_dir": output_dir} if takes_output_dir else {}
            result = run_check(check_id, name, module.check, **kwargs)
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
