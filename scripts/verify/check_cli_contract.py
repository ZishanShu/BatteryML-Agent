"""CLI-01 — the wrapper's ``run`` subcommand still has the argument shape we depend on.

CLAUDE.md §5 records five facts the user verified on the server, two of which
this check guards:

* ``config_path`` is a **positional** argument — ``run --config configs/...``
  fails with ``unrecognized arguments: --config``;
* ``--train`` and ``--eval`` are mandatory — without them BatteryML prints the
  workspace path and exits.

Inspecting ``--help`` only. This check never trains, never evaluates, never
touches ``data/`` and never creates a workspace.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

from .common import FAIL, PASS, SKIP, CheckResult, relative_to_repo, repo_root, run_check

CHECK_ID = "CLI-01"
CHECK_NAME = "cli_contract"

WRAPPER = Path("batteryml_web") / "core" / "run_batteryml_cli.py"

# Flags the web layer and the documented command block rely on.
REQUIRED_RUN_FLAGS = ("--train", "--eval", "--device", "--seed", "--workspace")

# The positional the config path is passed as.
REQUIRED_RUN_POSITIONAL = "config"

# A run subcommand that grew a --config option would silently change the
# documented command block, so its absence is part of the contract.
FORBIDDEN_RUN_FLAGS = ("--config",)

_IMPORT_FAILURE_RE = re.compile(r"ModuleNotFoundError|ImportError|No module named", re.IGNORECASE)

HELP_TIMEOUT_S = 120


def _run_help(root: Path, subcommand: list[str]) -> dict[str, Any]:
    """Invoke the wrapper with ``--help`` and capture the result.

    Returns a dict with ``argv``, ``returncode``, ``stdout``, ``stderr`` and,
    on failure to even start the process, ``spawn_error``.
    """
    argv = [sys.executable, str(WRAPPER), *subcommand, "--help"]
    record: dict[str, Any] = {"argv": " ".join(argv), "cwd": relative_to_repo(root) or "."}
    try:
        proc = subprocess.run(
            argv,
            cwd=str(root),
            capture_output=True,
            text=True,
            timeout=HELP_TIMEOUT_S,
        )
    except subprocess.TimeoutExpired:
        record["spawn_error"] = f"--help did not return within {HELP_TIMEOUT_S}s"
        return record
    except OSError as exc:
        record["spawn_error"] = f"{type(exc).__name__}: {exc}"
        return record

    record["returncode"] = proc.returncode
    record["stdout"] = proc.stdout
    record["stderr"] = proc.stderr
    return record


def _positional_names(help_text: str) -> list[str]:
    """Names listed under argparse's 'positional arguments' section.

    Argparse writes each argument at a fixed indent and wraps its help text at
    a deeper indent, so only lines at the *first* indent seen in the section
    name an argument. Section headers are flush left.
    """
    names: list[str] = []
    inside = False
    argument_indent: int | None = None

    for line in help_text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        indent = len(line) - len(line.lstrip())

        if indent == 0:
            header = stripped.lower().rstrip(":")
            inside = header in {"positional arguments", "positional"}
            argument_indent = None
            continue

        if not inside:
            continue
        if argument_indent is None:
            argument_indent = indent
        if indent != argument_indent:
            # Wrapped help text for the argument above.
            continue
        token = stripped.split()[0].rstrip(",")
        if not token.startswith("-"):
            names.append(token)

    return names


def check() -> CheckResult:
    """Confirm the run subcommand's positional/flag shape from --help alone."""
    root = repo_root()
    wrapper_path = root / WRAPPER

    if not wrapper_path.exists():
        return CheckResult(
            CHECK_ID,
            CHECK_NAME,
            FAIL,
            f"CLI wrapper {WRAPPER.as_posix()} does not exist. CLAUDE.md §5 requires every "
            "BatteryML invocation to go through it; `python bin/batteryml.py` shadows the "
            "batteryml package and fails with ModuleNotFoundError.",
            {"expected_wrapper": WRAPPER.as_posix()},
        )

    top_level = _run_help(root, [])
    run_help = _run_help(root, ["run"])

    evidence: dict[str, Any] = {
        "wrapper": WRAPPER.as_posix(),
        "top_level_help": {k: v for k, v in top_level.items() if k != "stdout"},
        "run_help_argv": run_help.get("argv"),
        "run_help_returncode": run_help.get("returncode"),
    }

    if "spawn_error" in run_help:
        return CheckResult(
            CHECK_ID,
            CHECK_NAME,
            FAIL,
            f"Could not run `{run_help['argv']}`: {run_help['spawn_error']}.",
            evidence,
        )

    combined = (run_help.get("stdout") or "") + (run_help.get("stderr") or "")

    if run_help.get("returncode") != 0:
        # A dependency that will not import is ENV-01's finding, not this
        # check's: the argument shape is simply unobservable here.
        if _IMPORT_FAILURE_RE.search(combined):
            evidence["stderr_tail"] = (run_help.get("stderr") or "").strip()[-1200:]
            return CheckResult(
                CHECK_ID,
                CHECK_NAME,
                SKIP,
                "The CLI wrapper could not start because an import failed, so --help "
                "produced no argument list. Fix the environment first (see ENV-01), then "
                "re-run this check.",
                evidence,
            )
        evidence["stdout_tail"] = (run_help.get("stdout") or "").strip()[-1200:]
        evidence["stderr_tail"] = (run_help.get("stderr") or "").strip()[-1200:]
        return CheckResult(
            CHECK_ID,
            CHECK_NAME,
            FAIL,
            f"`{run_help['argv']}` exited {run_help['returncode']}. `run --help` must exit 0; "
            "see evidence.stderr_tail.",
            evidence,
        )

    help_text = run_help.get("stdout") or ""
    positionals = _positional_names(help_text)
    missing_flags = [flag for flag in REQUIRED_RUN_FLAGS if flag not in help_text]
    present_forbidden = [flag for flag in FORBIDDEN_RUN_FLAGS if flag in help_text]
    positional_ok = REQUIRED_RUN_POSITIONAL in positionals

    evidence.update(
        {
            "positional_arguments": positionals,
            "required_flags": list(REQUIRED_RUN_FLAGS),
            "missing_flags": missing_flags,
            "forbidden_flags_present": present_forbidden,
            "config_is_positional": positional_ok,
            "run_help": help_text,
        }
    )

    problems: list[str] = []
    if not positional_ok:
        problems.append(
            f"`{REQUIRED_RUN_POSITIONAL}` is not listed as a positional argument "
            f"(saw {positionals or 'none'})"
        )
    if missing_flags:
        problems.append(f"missing flag(s): {', '.join(missing_flags)}")
    if present_forbidden:
        problems.append(
            f"`run` now accepts {', '.join(present_forbidden)}, which contradicts the "
            "documented positional-config command block"
        )

    if problems:
        return CheckResult(
            CHECK_ID,
            CHECK_NAME,
            FAIL,
            "The run subcommand's argument shape has drifted from CLAUDE.md §5: "
            + "; ".join(problems)
            + ". Update CLAUDE.md §5 and every documented command block together, or revert "
            "the CLI change.",
            evidence,
        )

    return CheckResult(
        CHECK_ID,
        CHECK_NAME,
        PASS,
        f"`run` takes `{REQUIRED_RUN_POSITIONAL}` positionally and accepts "
        f"{', '.join(REQUIRED_RUN_FLAGS)}; no --config option is offered.",
        evidence,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.parse_args(argv)
    result = run_check(CHECK_ID, CHECK_NAME, check)
    print(f"[{result.status}] {result.check_id} {result.name}: {result.message}")
    return 1 if result.status == FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
