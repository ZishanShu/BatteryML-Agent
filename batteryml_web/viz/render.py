"""The only module in this package that writes files, plus the figure manifest.

Two jobs:

* **Save a figure deterministically.** ``CLAUDE.md`` §8 requires identical
  config + seed + data to produce identical artifacts. Matplotlib stamps a
  creation date into SVG and PDF output and salts generated SVG element ids, so
  a byte-identical rerun needs both suppressed — see :func:`_deterministic_output`.
  Nothing else in the package touches the filesystem or global state.

* **Record where every figure came from.** ``figure_manifest.json`` names, for
  each figure: the artifact it was drawn from and that artifact's SHA-256, the
  git commit of the code that drew it, the seed, the label space and unit, and
  the figure's own SHA-256. A figure that cannot be traced back to an artifact
  is not evidence, and this file is what makes the trace cheap.

The manifest carries **no wall-clock timestamp**, deliberately: a timestamp
would make every rerun differ and destroy the determinism check that the file
exists to support. Run timing belongs in ``run_metadata.json``.
"""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
import os
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from matplotlib.figure import Figure

MANIFEST_VERSION = 1
MANIFEST_FILENAME = "figure_manifest.json"
SUPPORTED_FORMATS = ("png", "svg", "pdf")
SOFTWARE_TAG = "batteryml_web.viz"

# Fixed salt for matplotlib's generated SVG element ids. Without it the ids are
# salted per process and two identical runs produce different bytes.
SVG_HASHSALT = "batteryml-web-viz"

# Matplotlib's documented mechanism for reproducible output: when this is set,
# the SVG and PDF backends stamp it instead of the current time. Used in
# preference to passing ``Date: None`` / ``CreationDate: None`` through
# ``metadata``, because whether those backends treat an explicit ``None`` as
# "omit" or as a literal has varied, and this environment cannot test which.
SOURCE_DATE_EPOCH = "0"


class VizRenderError(RuntimeError):
    """A figure could not be written, or was written somewhere unusable."""


@dataclass(frozen=True)
class RenderedFigure:
    """One figure file on disk.

    Attributes:
        figure_id: stable identity, e.g. ``parity``. Quoted by the manuscript's
            per-phase output lists via the filename it produces.
        filename: path relative to the directory holding the manifest.
        path: absolute path actually written.
        fmt: ``png``, ``svg`` or ``pdf``.
        sha256: hash of the written bytes.
        size_bytes: size of the written file.
        details: figure-specific facts worth recording — point count, whether
            cell identity was available, what the annotation said.
    """

    figure_id: str
    filename: str
    path: Path
    fmt: str
    sha256: str
    size_bytes: int
    details: dict[str, Any] = field(default_factory=dict)

    def to_manifest_entry(self) -> dict[str, Any]:
        """The manifest form: no absolute path, so the file stays portable."""
        entry = asdict(self)
        entry.pop("path")
        return entry


@dataclass(frozen=True)
class SkippedFigure:
    """A figure that was deliberately not drawn, and why.

    A skipped figure is recorded, never silently omitted: the absence of a file
    is otherwise indistinguishable from a crash.
    """

    figure_id: str
    reason: str


def _attach_canvas(fig: Figure, fmt: str) -> None:
    """Give ``fig`` the backend canvas that can write ``fmt``.

    Attached explicitly rather than relying on matplotlib's internal canvas
    switching, so the behaviour does not depend on the matplotlib version.

    Raises:
        VizRenderError: for an unsupported format, or when the backend module
            cannot be imported.
    """
    if fmt == "png":
        from matplotlib.backends.backend_agg import FigureCanvasAgg as canvas_class
    elif fmt == "svg":
        from matplotlib.backends.backend_svg import FigureCanvasSVG as canvas_class
    elif fmt == "pdf":
        from matplotlib.backends.backend_pdf import FigureCanvasPdf as canvas_class
    else:
        raise VizRenderError(
            f"unsupported figure format {fmt!r}. Supported: {', '.join(SUPPORTED_FORMATS)}."
        )
    canvas_class(fig)


def _metadata_for(fmt: str) -> dict[str, Any]:
    """Per-format metadata, using only keys with a constant value.

    Nothing here suppresses a date: that is :func:`_deterministic_output`'s job.
    These entries only replace matplotlib's own version-bearing Creator /
    Software string with a constant one, so a matplotlib upgrade does not
    change the bytes for that reason alone.
    """
    if fmt == "png":
        return {"Software": SOFTWARE_TAG}
    return {"Creator": SOFTWARE_TAG}


@contextlib.contextmanager
def _deterministic_output():
    """Make matplotlib's output byte-stable, then put the process back.

    Two sources of per-run variation:

    * the SVG and PDF backends stamp the current time, unless
      ``SOURCE_DATE_EPOCH`` is set — matplotlib's documented switch for
      reproducible output;
    * the SVG backend salts generated element ids from ``svg.hashsalt``.

    Both are process-global, so this is the one place in the package that
    touches global state, and it restores what it found. Applied to every
    format, including PNG, so the behaviour does not depend on the caller
    knowing which formats carry a timestamp.
    """
    import matplotlib

    previous = os.environ.get("SOURCE_DATE_EPOCH")
    os.environ["SOURCE_DATE_EPOCH"] = SOURCE_DATE_EPOCH
    try:
        with matplotlib.rc_context({"svg.hashsalt": SVG_HASHSALT}):
            yield
    finally:
        if previous is None:
            os.environ.pop("SOURCE_DATE_EPOCH", None)
        else:
            os.environ["SOURCE_DATE_EPOCH"] = previous


def render_to_bytes(fig: Figure, *, fmt: str = "png", dpi: float = 300.0) -> bytes:
    """Render a figure to bytes without touching the filesystem.

    Used by :func:`save_figure`, so the hash recorded in the manifest is the
    hash of exactly the bytes that were written, and by the ``FIG-01`` check,
    which renders figures to memory to prove the builders work and that two
    identical renders produce identical bytes.

    Args:
        fig: the figure to render.
        fmt: ``png``, ``svg`` or ``pdf``.
        dpi: raster resolution for ``png``.

    Returns:
        The encoded figure.

    Raises:
        VizRenderError: for an unsupported format or a backend failure.
    """
    fmt = str(fmt).strip().lower()
    _attach_canvas(fig, fmt)
    buffer = io.BytesIO()
    try:
        with _deterministic_output():
            fig.savefig(
                buffer, format=fmt, dpi=dpi,
                facecolor=fig.get_facecolor(), metadata=_metadata_for(fmt),
            )
    except Exception as exc:
        raise VizRenderError(
            f"matplotlib failed to render a {fmt} figure: {type(exc).__name__}: {exc}."
        ) from exc
    payload = buffer.getvalue()
    if not payload:
        raise VizRenderError(
            f"matplotlib produced zero bytes for a {fmt} figure. Nothing was written."
        )
    return payload


def save_figure(
    fig: Figure,
    output_path: Path,
    *,
    figure_id: str,
    fmt: str = "png",
    dpi: float = 300.0,
    manifest_dir: Path | None = None,
    details: dict[str, Any] | None = None,
) -> RenderedFigure:
    """Write one figure and return its manifest record.

    Args:
        fig: the figure to write. Not closed here — the caller owns it.
        output_path: destination file, including the extension.
        figure_id: stable id for the manifest.
        fmt: output format; must match ``output_path``'s suffix.
        dpi: raster resolution for ``png``.
        manifest_dir: directory the manifest will live in, used to compute the
            relative filename. Defaults to the figure's own directory.
        details: extra facts to record alongside the figure.

    Returns:
        The :class:`RenderedFigure` describing what was written.

    Raises:
        VizRenderError: for a format mismatch, an unwritable destination, or a
            backend failure. Never leaves a partially written file unreported.
    """
    fmt = str(fmt).strip().lower()
    output_path = Path(output_path)
    suffix = output_path.suffix.lower().lstrip(".")
    if suffix != fmt:
        raise VizRenderError(
            f"{output_path}: suffix {suffix!r} does not match the requested format {fmt!r}."
        )

    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise VizRenderError(
            f"cannot create the output directory {output_path.parent}: {exc}."
        ) from exc

    payload = render_to_bytes(fig, fmt=fmt, dpi=dpi)
    try:
        output_path.write_bytes(payload)
    except OSError as exc:
        raise VizRenderError(f"cannot write {output_path}: {exc}.") from exc

    base = Path(manifest_dir) if manifest_dir is not None else output_path.parent
    try:
        relative = output_path.resolve().relative_to(Path(base).resolve()).as_posix()
    except ValueError:
        relative = output_path.as_posix()

    return RenderedFigure(
        figure_id=figure_id,
        filename=relative,
        path=output_path,
        fmt=fmt,
        sha256=hashlib.sha256(payload).hexdigest(),
        size_bytes=len(payload),
        details=dict(details or {}),
    )


def git_commit(repo_root: Path) -> str:
    """The commit the drawing code was at, or a stated reason it is unknown.

    Never raises: a figure written outside a git checkout is still a valid
    figure, it just carries less provenance, and that fact belongs in the
    manifest rather than in a traceback.
    """
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return f"unavailable: {type(exc).__name__}: {exc}"
    if completed.returncode != 0:
        detail = completed.stderr.strip() or f"git exited {completed.returncode}"
        return f"unavailable: {detail}"
    commit = completed.stdout.strip()
    return commit or "unavailable: git printed nothing"


def build_manifest(
    *,
    figures: list[RenderedFigure],
    skipped: list[SkippedFigure],
    source: dict[str, Any],
    theme: str,
    dpi: float,
    repo_root: Path,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble the manifest for one figure-generation run.

    Args:
        figures: what was written.
        skipped: what was not, each with a reason.
        source: provenance of the inputs — artifact path and hash, config path
            and hash, seed, dataset, model.
        theme: the theme the figures were drawn against.
        dpi: the raster resolution used.
        repo_root: where to ask git for the commit.
        extra: anything else worth recording, e.g. the recorded scores and the
            score-consistency check.

    Returns:
        A JSON-serialisable dict.
    """
    import matplotlib

    manifest: dict[str, Any] = {
        "manifest_version": MANIFEST_VERSION,
        "generator": SOFTWARE_TAG,
        "git_commit": git_commit(repo_root),
        "matplotlib_version": matplotlib.__version__,
        "theme": theme,
        "dpi": dpi,
        "source": dict(source),
        "figures": [item.to_manifest_entry() for item in figures],
        "skipped": [asdict(item) for item in skipped],
    }
    if extra:
        manifest.update(extra)
    return manifest


def write_manifest(directory: Path, manifest: dict[str, Any]) -> Path:
    """Write ``figure_manifest.json`` into ``directory`` and return its path."""
    directory = Path(directory)
    try:
        directory.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise VizRenderError(f"cannot create {directory}: {exc}.") from exc
    path = directory / MANIFEST_FILENAME
    path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=False, default=str) + "\n",
        encoding="utf-8",
    )
    return path


def read_manifest(directory: Path) -> dict[str, Any]:
    """Read a manifest back.

    Args:
        directory: the directory holding ``figure_manifest.json``.

    Returns:
        The parsed manifest.

    Raises:
        VizRenderError: if it is missing or is not a JSON object.
    """
    path = Path(directory) / MANIFEST_FILENAME
    if not path.is_file():
        raise VizRenderError(
            f"{path} does not exist. Generate figures first with "
            "`python -m batteryml_web.viz.cli results ...`."
        )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise VizRenderError(f"{path} is not valid JSON: {exc}.") from exc
    if not isinstance(payload, dict):
        raise VizRenderError(
            f"{path} holds a {type(payload).__name__}; a manifest must be a JSON object."
        )
    return payload
