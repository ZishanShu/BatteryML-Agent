"""Shared types and helpers for the verification checks.

Design constraints (CLAUDE.md §0, §6):

* the environment that authors these checks cannot run the pipeline, so the
  checks must be defensive by construction and must never assume that data,
  a GPU, or even a third-party dependency is present;
* a check reports; it does not repair, download, or write into ``data/``.
"""

from __future__ import annotations

import ast
import json
import os
import sys
import time
import traceback
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable, Iterable

PASS = "PASS"
FAIL = "FAIL"
SKIP = "SKIP"

STATUSES = (PASS, FAIL, SKIP)

# Dataset names BatteryML ships a preprocessor for. Used only as the
# dependency-free fallback when `batteryml` cannot be imported; the live list
# is read from batteryml.preprocess.SUPPORTED_SOURCES whenever possible.
_PREPROCESSOR_FILE_PREFIX = "preprocess_"


@dataclass
class CheckResult:
    """The single structured value every check returns.

    Attributes:
        check_id: stable identifier, e.g. ``ENV-01``. Quoted in reports.
        name: short machine-readable name, e.g. ``environment``.
        status: one of ``PASS`` / ``FAIL`` / ``SKIP``.
        message: one line a human can act on. For FAIL it names the offending
            file/field and the corrective action; for SKIP it says why.
        evidence: structured detail. Serialized verbatim into
            ``verification_report.json``.
        artifacts: paths the check wrote, relative to the repo root.
        duration_s: wall-clock seconds the check took.
    """

    check_id: str
    name: str
    status: str
    message: str
    evidence: dict[str, Any] = field(default_factory=dict)
    artifacts: list[str] = field(default_factory=list)
    duration_s: float = 0.0

    def __post_init__(self) -> None:
        if self.status not in STATUSES:
            raise ValueError(
                f"CheckResult({self.check_id}): status {self.status!r} is not one of {STATUSES}."
            )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def repo_root() -> Path:
    """Absolute path of the repository root.

    ``scripts/verify/common.py`` -> ``scripts/verify`` -> ``scripts`` -> root.
    """
    return Path(__file__).resolve().parents[2]


def run_check(
    check_id: str,
    name: str,
    fn: Callable[..., CheckResult],
    *args: Any,
    **kwargs: Any,
) -> CheckResult:
    """Invoke a check function, converting any escaping exception into a FAIL.

    This is the only place the harness swallows an exception, and it never
    swallows it silently: the exception type, message and full traceback are
    preserved in the result's evidence.
    """
    started = time.monotonic()
    try:
        result = fn(*args, **kwargs)
        if not isinstance(result, CheckResult):
            raise TypeError(
                f"check {check_id} returned {type(result).__name__}, expected CheckResult."
            )
    except Exception as exc:  # noqa: BLE001 - deliberate top-level barrier
        result = CheckResult(
            check_id=check_id,
            name=name,
            status=FAIL,
            message=(
                f"{name} raised {type(exc).__name__}: {exc}. This is a defect in the "
                f"check itself or in the code it inspects; see evidence.traceback."
            ),
            evidence={
                "exception_type": type(exc).__name__,
                "exception": str(exc),
                "traceback": traceback.format_exc(),
            },
        )
    result.duration_s = round(time.monotonic() - started, 4)
    return result


def write_json(path: Path, payload: Any) -> str:
    """Write ``payload`` as indented UTF-8 JSON; return the repo-relative path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )
    return relative_to_repo(path)


def write_text(path: Path, text: str) -> str:
    """Write ``text`` as UTF-8; return the repo-relative path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return relative_to_repo(path)


def write_csv(path: Path, header: list[str], rows: Iterable[list[Any]]) -> str:
    """Write a CSV with the stdlib csv module; return the repo-relative path."""
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for row in rows:
            writer.writerow(row)
    return relative_to_repo(path)


def relative_to_repo(path: Path) -> str:
    """Repo-relative POSIX path, or the absolute path if it is outside the repo."""
    path = Path(path).resolve()
    try:
        return path.relative_to(repo_root()).as_posix()
    except ValueError:
        return path.as_posix()


def dir_stats(path: Path, pattern: str = "*") -> dict[str, Any]:
    """Count files, total bytes and newest mtime under ``path``.

    Returns ``{"exists": False}`` when the directory is absent, so callers can
    report SKIP instead of raising.
    """
    if not path.exists():
        return {"exists": False, "is_dir": False}
    if not path.is_dir():
        return {"exists": True, "is_dir": False}

    file_count = 0
    total_bytes = 0
    newest_mtime = 0.0
    for entry in path.rglob(pattern):
        if not entry.is_file():
            continue
        file_count += 1
        try:
            stat = entry.stat()
        except OSError:
            # A broken symlink or a file removed mid-scan. Count it, but do
            # not let it distort the size or mtime figures.
            continue
        total_bytes += stat.st_size
        newest_mtime = max(newest_mtime, stat.st_mtime)

    return {
        "exists": True,
        "is_dir": True,
        "file_count": file_count,
        "total_bytes": total_bytes,
        "total_mib": round(total_bytes / (1024 * 1024), 3),
        "newest_mtime": newest_mtime or None,
    }


def try_import(module_name: str) -> tuple[Any | None, str | None]:
    """Import a module, returning ``(module, None)`` or ``(None, reason)``.

    Used by every check that would prefer live registry data but must still
    produce a result when a heavy dependency (torch, xgboost) is absent.
    """
    root = repo_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    try:
        return __import__(module_name, fromlist=["__name__"]), None
    except Exception as exc:  # noqa: BLE001 - reported, never hidden
        return None, f"{type(exc).__name__}: {exc}"


def registered_names_from_source(registry_attr: str) -> dict[str, str]:
    """Find every class registered with ``@<registry_attr>.register()`` by parsing source.

    This is the dependency-free fallback for the live registries in
    ``batteryml.builders``: it reads ``batteryml/**/*.py`` with ``ast`` and
    never imports anything, so it works without torch installed.

    Args:
        registry_attr: the registry object name as written in the decorator,
            e.g. ``"MODELS"`` for ``@MODELS.register()``.

    Returns:
        Mapping of registered name -> repo-relative ``path:line`` of the class.

    Raises:
        ValueError: if two classes register the same name, which the live
            Registry would also reject (batteryml/utils/registry.py:38-40).
    """
    found: dict[str, str] = {}
    root = repo_root()
    for source_path in sorted((root / "batteryml").rglob("*.py")):
        # registry.py's docstring contains an example @MODELS.register() that
        # is not a real registration.
        if source_path.name == "registry.py":
            continue
        try:
            tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
        except (OSError, SyntaxError) as exc:
            raise ValueError(f"cannot parse {relative_to_repo(source_path)}: {exc}") from exc

        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue
            for decorator in node.decorator_list:
                registered = _register_decorator_name(decorator, registry_attr)
                if registered is None:
                    continue
                name = registered or node.name
                location = f"{relative_to_repo(source_path)}:{node.lineno}"
                if name in found:
                    raise ValueError(
                        f"duplicate registration {name!r} in {registry_attr}: "
                        f"{found[name]} and {location}"
                    )
                found[name] = location
    return found


def _register_decorator_name(decorator: ast.expr, registry_attr: str) -> str | None:
    """Return the explicit name for ``@REGISTRY.register(...)``, "" for the default, else None."""
    if not isinstance(decorator, ast.Call):
        return None
    func = decorator.func
    if not isinstance(func, ast.Attribute) or func.attr != "register":
        return None
    if not isinstance(func.value, ast.Name) or func.value.id != registry_attr:
        return None

    # @REGISTRY.register('explicit_name') or @REGISTRY.register(name='...')
    for arg in decorator.args:
        if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
            return arg.value
    for keyword in decorator.keywords:
        if keyword.arg == "name" and isinstance(keyword.value, ast.Constant):
            if isinstance(keyword.value.value, str):
                return keyword.value.value
    return ""


def supported_sources() -> tuple[dict[str, list[str]], str]:
    """Every ``input_type`` the CLI advertises, grouped, and where the list came from.

    Prefers the live ``batteryml.preprocess.SUPPORTED_SOURCES``. Falls back to
    reading the same literal out of ``batteryml/preprocess/__init__.py`` with
    ``ast``, so the advertised-vs-registered comparison still works when
    ``batteryml`` cannot be imported (typically because torch is absent).
    A filename scan is the last resort; it can only see what exists, so it
    cannot detect an advertised source that has no preprocessor.
    """
    module, reason = try_import("batteryml.preprocess")
    if module is not None:
        sources = getattr(module, "SUPPORTED_SOURCES", None)
        if isinstance(sources, dict) and sources:
            return (
                {group: list(names) for group, names in sources.items()},
                "batteryml.preprocess.SUPPORTED_SOURCES",
            )

    init_path = repo_root() / "batteryml" / "preprocess" / "__init__.py"
    if init_path.exists():
        try:
            tree = ast.parse(init_path.read_text(encoding="utf-8"), filename=str(init_path))
        except (OSError, SyntaxError) as exc:
            tree = None
            reason = f"{reason}; cannot parse {relative_to_repo(init_path)}: {exc}"
        if tree is not None:
            for node in ast.walk(tree):
                if not isinstance(node, ast.Assign):
                    continue
                if not any(
                    isinstance(target, ast.Name) and target.id == "SUPPORTED_SOURCES"
                    for target in node.targets
                ):
                    continue
                try:
                    literal = ast.literal_eval(node.value)
                except ValueError:
                    continue
                if isinstance(literal, dict) and literal:
                    return (
                        {str(group): [str(n) for n in names] for group, names in literal.items()},
                        f"ast read of {relative_to_repo(init_path)} SUPPORTED_SOURCES "
                        f"(import failed: {reason})",
                    )

    names = []
    preprocess_dir = repo_root() / "batteryml" / "preprocess"
    for source_path in sorted(preprocess_dir.glob(f"{_PREPROCESSOR_FILE_PREFIX}*.py")):
        stem = source_path.stem[len(_PREPROCESSOR_FILE_PREFIX):]
        # Cycler preprocessors are lower-case filenames (arbin, neware);
        # dataset preprocessors are upper-case (MATR, UL_PUR).
        if stem.upper() == stem:
            names.append(stem)
    return (
        {"DATASETS": names},
        f"filename scan of batteryml/preprocess/ (import and ast read failed: {reason})",
    )


def known_datasets() -> tuple[list[str], str]:
    """The dataset names BatteryML can preprocess, and where the list came from."""
    sources, source = supported_sources()
    return list(sources.get("DATASETS", [])), source


def env_flag(name: str) -> bool:
    """Read a boolean environment flag; anything but 1/true/yes is False."""
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}
