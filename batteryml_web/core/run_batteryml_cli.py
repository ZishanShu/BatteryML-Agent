"""The single entry point for every BatteryML invocation (`CLAUDE.md` §5).

Running ``python bin/batteryml.py`` directly fails with
``ModuleNotFoundError: ... 'batteryml' is not a package``, because ``bin/``
shadows the package. This wrapper puts the repository root on ``sys.path``
first, then delegates.

It also enforces two preconditions and one postcondition that upstream does
not, all of them here rather than in ``batteryml/`` or ``bin/`` so the
divergence from microsoft/BatteryML stays at zero:

* **Extension models are registered** before the CLI runs, so a model defined
  in ``batteryml_web/models/`` is usable (see that package's docstring).
* **A declared-but-unimplemented preprocessor is refused up front.**
  ``SUPPORTED_SOURCES`` advertises eight cycler formats; only ``ARBIN`` and
  ``NEWARE`` have a preprocessor. Upstream accepts the other six as valid
  ``input_type`` choices and then dies inside ``PREPROCESSORS.build`` with a
  bare ``KeyError`` — after the user has already staged raw data.
* **Preprocessing that writes nothing is a failure.** Upstream's preprocessors
  locate cells with a non-recursive ``Path.glob(...)``; when nothing matches,
  the cell set is empty, ``process()`` returns ``(0, 0)``,
  ``BasePreprocessor.__call__`` prints "Successfully processed 0 batteries" and
  the CLI exits **0**. Worse, ``bin/batteryml.py`` creates the output directory
  *before* processing, so each such run leaves an empty
  ``data/processed/<DATASET>/`` behind — which then satisfies
  ``BaseTrainTestSplitter``'s ``assert path.exists()`` and yields an empty cell
  list, so a later training run proceeds on nothing. That is precisely the trap
  `CLAUDE.md` §7 forbids, created by the tooling itself.

Exit codes:

===== ==========================================================
  0    the underlying CLI succeeded
  1    the underlying CLI failed (its own error is on stderr)
  2    a wrapper precondition failed: extension registration, or an
       unimplemented preprocessor was requested
  3    preprocessing ran but produced no BatteryData
===== ==========================================================
"""

from __future__ import annotations

import re
import runpy
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

#: BatteryData files a preprocessor writes into its output directory.
PROCESSED_GLOB = "*.pkl"

#: Wrapper-level precondition failure.
EXIT_PRECONDITION = 2
#: Preprocessing produced no output.
EXIT_NO_OUTPUT = 3

#: Options of `preprocess` that consume the following token, so the positional
#: scan does not mistake their value for a positional argument.
_VALUE_OPTIONS = {"--config"}

#: Options that consume nothing.
_FLAG_OPTIONS = {"-q", "--quiet", "--silent"}

_HELP_OPTIONS = {"-h", "--help"}

_GLOB_PATTERN = re.compile(r"\.glob\(\s*[fr]?['\"]([^'\"]+)['\"]")


@dataclass(frozen=True)
class PreprocessInvocation:
    """A fully-determined ``preprocess`` command line."""

    input_type: str
    raw_dir: Path
    output_dir: Path


def parse_preprocess_invocation(argv: list[str]) -> PreprocessInvocation | None:
    """Recognize an unambiguous ``preprocess`` invocation, else return None.

    Deliberately conservative: this only mirrors enough of
    ``bin/batteryml.py``'s parser to identify the three positionals
    (``input_type``, ``raw_dir``, ``output_dir``). Anything it cannot read with
    certainty — a help request, a different subcommand, an unexpected number of
    positionals — returns None and the command is passed through untouched, so
    a parser change upstream degrades to today's behaviour rather than to a
    wrong guess.
    """
    if not argv or argv[0] != "preprocess":
        return None
    if any(token in _HELP_OPTIONS for token in argv):
        return None

    positionals: list[str] = []
    index = 1
    while index < len(argv):
        token = argv[index]
        if token in _VALUE_OPTIONS:
            index += 2
            continue
        if token.startswith("--") and "=" in token and token.split("=", 1)[0] in _VALUE_OPTIONS:
            index += 1
            continue
        if token in _FLAG_OPTIONS:
            index += 1
            continue
        if token.startswith("-"):
            # An option this wrapper does not know about. Do not guess.
            return None
        positionals.append(token)
        index += 1

    if len(positionals) != 3:
        return None
    input_type, raw_dir, output_dir = positionals
    return PreprocessInvocation(input_type, Path(raw_dir), Path(output_dir))


def unimplemented_preprocessor_error(input_type: str) -> str | None:
    """Message to print if ``input_type`` is advertised but has no preprocessor.

    Returns None when the format is implemented, or when the question cannot be
    answered (batteryml not importable) — in which case the CLI's own error is
    the better one to show.
    """
    try:
        from batteryml import preprocess as preprocess_package
        from batteryml.builders import PREPROCESSORS
    except Exception:  # noqa: BLE001 - the CLI will report the import failure itself
        return None

    expected = f"{input_type}Preprocessor"
    if expected in PREPROCESSORS.class_mapping:
        return None

    sources = getattr(preprocess_package, "SUPPORTED_SOURCES", {}) or {}
    advertised = {name for names in sources.values() for name in names}
    if input_type not in advertised:
        # Not advertised either; argparse's own "invalid choice" error is clearer.
        return None

    implemented = sorted(
        name[: -len("Preprocessor")]
        for name in PREPROCESSORS.class_mapping
        if name.endswith("Preprocessor")
    )
    return (
        f"error: the input format {input_type!r} is declared as supported but is not "
        f"implemented.\n"
        f"  batteryml/preprocess/__init__.py lists it in SUPPORTED_SOURCES, so the "
        f"command line accepts it, but no {expected} class is registered, so nothing "
        f"can read those files.\n"
        f"  This is an upstream gap in microsoft/BatteryML, not a fault in your data "
        f"or your command.\n"
        f"  Formats that are actually implemented: {', '.join(implemented)}.\n"
        f"  To use {input_type!r} data, export it from your cycler software to one of "
        f"those formats first, or supply a cycler column-mapping config (see "
        f"configs/cyclers/) for a format that is implemented."
    )


def discovery_globs(input_type: str) -> list[str]:
    """The glob patterns ``preprocess_<input_type>.py`` searches with.

    Read from source at failure time, so the message stays true if upstream
    changes a pattern. Returns an empty list when the module cannot be read.
    """
    directory = Path(__file__).resolve().parents[2] / "batteryml" / "preprocess"
    for candidate in (
        directory / f"preprocess_{input_type}.py",
        directory / f"preprocess_{input_type.lower()}.py",
    ):
        if candidate.is_file():
            try:
                text = candidate.read_text(encoding="utf-8")
            except OSError:
                return []
            seen: list[str] = []
            for pattern in _GLOB_PATTERN.findall(text):
                if pattern not in seen:
                    seen.append(pattern)
            return seen
    return []


def describe_raw_directory(raw_dir: Path) -> str:
    """One line on what is actually in the raw directory, for the failure message."""
    if not raw_dir.is_dir():
        return f"  The raw directory {raw_dir} does not exist or is not a directory."

    try:
        entries = list(raw_dir.iterdir())
    except OSError as exc:
        return f"  Could not inspect {raw_dir}: {type(exc).__name__}: {exc}."

    top_files = [p for p in entries if p.is_file()]
    subdirectories = [p for p in entries if p.is_dir()]
    nested = sum(1 for p in raw_dir.rglob("*") if p.is_file()) - len(top_files)

    if not top_files and not nested:
        return f"  {raw_dir} contains no files at all."
    if not top_files and nested:
        names = ", ".join(sorted(p.name for p in subdirectories)[:5])
        return (
            f"  All {nested} raw file(s) are inside subdirectories ({names}), and the "
            f"search is not recursive — move the files up one level into {raw_dir}."
        )
    if nested:
        return (
            f"  {raw_dir} holds {len(top_files)} file(s) at the top level and {nested} "
            f"deeper in subdirectories; only the top level is searched."
        )
    return (
        f"  {raw_dir} holds {len(top_files)} top-level file(s), none matching. Compare "
        f"their names against the patterns above."
    )


def no_output_message(invocation: PreprocessInvocation, removed: bool) -> str:
    """Explain that preprocessing produced nothing, and why it might have."""
    patterns = discovery_globs(invocation.input_type)
    pattern_line = (
        f"  It searched {invocation.raw_dir} for: "
        + ", ".join(repr(p) for p in patterns)
        + " (not recursively)."
        if patterns
        else f"  It searched {invocation.raw_dir} and matched nothing."
    )
    lines = [
        f"error: preprocessing {invocation.input_type} produced no {PROCESSED_GLOB} file.",
        pattern_line,
        describe_raw_directory(invocation.raw_dir),
    ]
    if removed:
        lines.append(
            f"  Removed the empty output directory {invocation.output_dir} that this "
            f"command created. An empty processed directory is worse than no directory: "
            f"it satisfies the training pipeline's existence check and then yields an "
            f"empty cell list, so training would proceed on no data."
        )
    elif invocation.output_dir.is_dir():
        lines.append(
            f"  {invocation.output_dir} already existed and is left as it was. If it is "
            f"empty, remove it: an empty processed directory satisfies the training "
            f"pipeline's existence check and then yields an empty cell list."
        )
    return "\n".join(lines)


def _count_processed(output_dir: Path) -> int:
    if not output_dir.is_dir():
        return 0
    return sum(1 for _ in output_dir.glob(PROCESSED_GLOB))


def _remove_if_created_and_empty(output_dir: Path, created_here: bool) -> bool:
    """Remove ``output_dir`` only if this invocation created it and it holds no output."""
    if not created_here or not output_dir.is_dir():
        return False
    if _count_processed(output_dir):
        return False
    try:
        shutil.rmtree(output_dir)
        return True
    except OSError as exc:
        print(
            f"warning: could not remove the empty output directory {output_dir}: "
            f"{type(exc).__name__}: {exc}. Remove it by hand — an empty processed "
            f"directory silently yields an empty cell list.",
            file=sys.stderr,
        )
        return False


def _exit_code(exc: SystemExit) -> int:
    if exc.code is None:
        return 0
    if isinstance(exc.code, int):
        return exc.code
    print(exc.code, file=sys.stderr)
    return 1


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]

    # Put repo root before bin/ so that "batteryml" resolves to the package,
    # not bin/batteryml.py.
    sys.path.insert(0, str(repo_root))

    # Register models defined outside upstream BatteryML before the CLI runs.
    # bin/batteryml.py imports only batteryml.*, so without this an extension
    # model's @MODELS.register() never fires and the run fails later inside
    # MODELS.build() with a bare "<X> is not registered to Models!".
    # Imported here rather than at module scope so that the repo root is on
    # sys.path first. No-op while batteryml_web/models declares no models.
    from batteryml_web.models import (
        ExtensionRegistrationError,
        register_extension_models,
    )

    try:
        register_extension_models()
    except ExtensionRegistrationError as exc:
        # The message already names the module and the corrective action, so a
        # traceback would only bury it. Any other exception is an unexpected
        # bug and is deliberately left to propagate with its traceback intact.
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(EXIT_PRECONDITION)

    argv = sys.argv[1:]
    invocation = parse_preprocess_invocation(argv)

    if invocation is not None:
        refusal = unimplemented_preprocessor_error(invocation.input_type)
        if refusal is not None:
            print(refusal, file=sys.stderr)
            raise SystemExit(EXIT_PRECONDITION)

    # Whether the output directory exists *before* the CLI runs decides what may
    # be cleaned up afterwards: only what this invocation created.
    created_here = invocation is not None and not invocation.output_dir.exists()

    cli_path = repo_root / "bin" / "batteryml.py"

    # Preserve user-provided CLI arguments.
    sys.argv = [str(cli_path)] + argv

    exit_code = 0
    try:
        runpy.run_path(str(cli_path), run_name="__main__")
    except SystemExit as exc:
        exit_code = _exit_code(exc)
    except BaseException:
        # The command failed with a traceback. Do not leave a directory this
        # invocation created behind before the traceback propagates.
        if invocation is not None:
            _remove_if_created_and_empty(invocation.output_dir, created_here)
        raise

    if invocation is None:
        raise SystemExit(exit_code)

    if _count_processed(invocation.output_dir):
        raise SystemExit(exit_code)

    removed = _remove_if_created_and_empty(invocation.output_dir, created_here)
    if exit_code == 0:
        # Upstream reported success while writing nothing. This is the failure
        # the whole postcondition exists for.
        print(no_output_message(invocation, removed), file=sys.stderr)
        raise SystemExit(EXIT_NO_OUTPUT)

    # The CLI already reported its own failure; only the cleanup is news.
    if removed:
        print(
            f"note: removed the empty output directory {invocation.output_dir} that this "
            f"command created.",
            file=sys.stderr,
        )
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
