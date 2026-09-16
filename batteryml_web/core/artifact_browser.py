"""Safe, run-scoped access to the files a run actually produced.

A download control that takes a path from the page is a file-disclosure bug
waiting to happen: ``../../../etc/passwd``, or a symlink pointing out of the
run directory, and the server hands it over. So every path is resolved and
then *proved* to be inside the run's own directory before anything is read.

Two specific defences, because one is not enough:

* ``Path.resolve()`` collapses ``..`` and follows symlinks, so the check is
  made against the real location rather than the spelling of the path. A
  symlink inside the run directory pointing at ``/etc`` therefore fails;
* containment is checked with ``relative_to``, not ``str.startswith``:
  ``/runs/web_projects/alpha-evil`` starts with ``/runs/web_projects/alpha``
  as a string but is a different directory.

Listing is discovery — only files that exist are returned, so the page never
offers a download for something that was never produced.

No Streamlit here; the page calls this and renders the result.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

#: Files this size or smaller may be offered for download. Larger artifacts
#: are listed with their size and left on disk rather than read into memory.
MAX_DOWNLOAD_BYTES = 64 * 1024 * 1024

#: Grouping for display, by extension. A file matching nothing is "other";
#: nothing is dropped, because a file that exists should be visible.
CATEGORIES: Dict[str, tuple[str, ...]] = {
    "predictions": (".pkl", ".csv"),
    "figures": (".png", ".jpg", ".jpeg", ".svg", ".pdf"),
    "logs": (".log", ".txt"),
    "config": (".yaml", ".yml"),
    "manifest": (".json",),
    "checkpoints": (".ckpt", ".pt", ".pth", ".bin"),
}


class UnsafePathError(ValueError):
    """Raised when a requested path is not inside the run directory."""


def resolve_within(root: str | Path, candidate: str | Path) -> Path:
    """Resolve ``candidate`` and prove it lies inside ``root``.

    Raises :class:`UnsafePathError` otherwise. This is the only way the rest
    of the platform should turn a page-supplied path into a file to read.
    """
    base = Path(root).resolve()
    target = Path(candidate)
    if not target.is_absolute():
        target = base / target
    resolved = target.resolve()
    try:
        resolved.relative_to(base)
    except ValueError as exc:
        raise UnsafePathError(
            f"{candidate!r} resolves to {resolved}, which is outside the run "
            f"directory {base}. Refusing to read it."
        ) from exc
    return resolved


def categorise(path: Path) -> str:
    suffix = path.suffix.lower()
    for name, suffixes in CATEGORIES.items():
        if suffix in suffixes:
            return name
    return "other"


def list_artifacts(run_dir: str | Path) -> List[Dict[str, Any]]:
    """Every file really present under ``run_dir``, with size and category.

    Entries whose resolved location escapes the run directory (a symlink
    planted there, say) are skipped rather than listed, so the page cannot
    offer them at all.
    """
    root = Path(run_dir)
    if not root.is_dir():
        return []
    base = root.resolve()
    entries: List[Dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        try:
            resolved = path.resolve()
            resolved.relative_to(base)
        except (ValueError, OSError):
            continue
        try:
            size = resolved.stat().st_size
        except OSError:
            continue
        entries.append(
            {
                "name": path.name,
                "relative_path": path.relative_to(root).as_posix(),
                "category": categorise(path),
                "size_bytes": size,
                "downloadable": size <= MAX_DOWNLOAD_BYTES,
            }
        )
    return entries


def read_artifact(
    run_dir: str | Path,
    relative_path: str,
    *,
    max_bytes: int = MAX_DOWNLOAD_BYTES,
) -> Optional[bytes]:
    """Bytes of one artifact, or None when it is too large to serve.

    Raises :class:`UnsafePathError` for anything outside the run directory.
    The size is checked before the read, so an oversized file is refused
    rather than loaded into memory and then rejected.
    """
    resolved = resolve_within(run_dir, relative_path)
    if not resolved.is_file():
        return None
    if resolved.stat().st_size > max_bytes:
        return None
    return resolved.read_bytes()
