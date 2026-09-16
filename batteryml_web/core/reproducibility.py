"""What a run needs to be reproducible, and the bundle that carries it.

A reviewer should be able to take one zip and see what was run, on what, with
which code, and what came out. That is the whole purpose here.

Two hard boundaries.

**Secrets never enter.** The environment section is built from an explicit
allow-list of package names and interpreter facts — never by iterating
``os.environ``, which is how API keys, proxy credentials and tokens end up
in "just the environment, for reproducibility" files. There is no code path
from a credential to a bundle.

**Size is respected.** The raw dataset is never bundled, and checkpoints are
excluded by default because they are routinely gigabytes. Anything left out
is *named* in the README with the path where it really lives, so an omission
is a documented pointer rather than a silent gap. The zip is streamed file by
file, so nothing is duplicated into memory.

No Streamlit. Nothing here computes or alters a metric.
"""

from __future__ import annotations

import json
import platform
import sys
import zipfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from . import run_manifest as rm

#: Packages whose versions are worth recording. An allow-list, not a scan of
#: everything installed: a full freeze is noise, and iterating the
#: environment is the habit that leaks credentials.
RECORDED_PACKAGES: tuple[str, ...] = (
    "batteryml",
    "torch",
    "numpy",
    "scipy",
    "pandas",
    "scikit-learn",
    "xgboost",
    "streamlit",
    "matplotlib",
    "pyyaml",
)

BUNDLE_README = "README.txt"
BUNDLE_MANIFEST = "run_manifest.json"
BUNDLE_METRICS = "metrics.json"
BUNDLE_ENVIRONMENT = "environment.json"
BUNDLE_GIT = "git_commit.txt"
BUNDLE_PREDICTIONS_CSV = "predictions.csv"

#: Suffixes never included by default. Checkpoints are large and the raw
#: data is not ours to redistribute.
EXCLUDED_SUFFIXES: frozenset[str] = frozenset({".ckpt", ".pt", ".pth", ".bin"})

#: A single file larger than this is referenced in the README instead of
#: copied, so a bundle cannot silently become a multi-gigabyte download.
MAX_BUNDLED_BYTES = 64 * 1024 * 1024


def package_versions(names: Iterable[str] = RECORDED_PACKAGES) -> Dict[str, str]:
    """Installed version per named package; "not installed" when absent.

    Uses ``importlib.metadata`` rather than importing the packages, so
    recording a version never triggers a CUDA context or a slow import.
    """
    from importlib import metadata

    versions: Dict[str, str] = {}
    for name in names:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = "not installed"
        except Exception as exc:  # noqa: BLE001 - never fail a bundle over this
            versions[name] = f"unavailable ({type(exc).__name__})"
    return versions


def environment_snapshot() -> Dict[str, Any]:
    """Interpreter and package facts. No environment variables, ever."""
    return {
        "python_version": sys.version.split()[0],
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "packages": package_versions(),
        "note": (
            "Package versions come from importlib.metadata. Environment "
            "variables are deliberately not recorded: they carry credentials."
        ),
    }


def dataset_fingerprint(processed_dir: str | Path) -> Dict[str, Any]:
    """A cheap, honest fingerprint of a processed dataset directory.

    File count, total size and the sorted file names — metadata only. Content
    hashing of a processed battery dataset would read gigabytes on every
    call, so it is not done here and not claimed. A directory that does not
    exist reports that it does not exist rather than an empty fingerprint
    that would look like a real one.
    """
    directory = Path(processed_dir)
    if not directory.is_dir():
        return {"available": False, "reason": f"{processed_dir} does not exist.",
                "file_count": None, "total_bytes": None, "files": []}
    files = sorted(path for path in directory.rglob("*") if path.is_file())
    return {
        "available": True,
        "reason": "",
        "file_count": len(files),
        "total_bytes": sum(path.stat().st_size for path in files),
        "files": [path.name for path in files],
        "method": "file list and sizes; contents are not hashed",
    }


def _bundle_candidates(run_dir: Path) -> List[Path]:
    """Files under ``run_dir`` eligible for the bundle.

    Discovered, not assumed: only files that actually exist are listed, so a
    bundle never contains a placeholder for something that was never
    produced.
    """
    if not run_dir.is_dir():
        return []
    return sorted(
        path
        for path in run_dir.rglob("*")
        if path.is_file() and path.suffix.lower() not in EXCLUDED_SUFFIXES
    )


def predictions_csv_rows(view: Dict[str, Any], limit: int = 100000) -> Optional[str]:
    """The observed/predicted table as CSV text, or None.

    Converted only when the run really holds paired observed and predicted
    values; a run without ground truth exports no CSV rather than a column of
    predictions labelled as results.
    """
    if not view or not view.get("available"):
        return None
    ids = view.get("sample_ids")
    header = ("cell_id" if ids else "index") + ",observed,predicted,residual"
    lines = [header]
    for index in range(min(int(limit), int(view.get("count") or 0))):
        label = ids[index] if ids else str(index)
        lines.append(
            f"{label},{view['observed'][index]!r},"
            f"{view['predicted'][index]!r},{view['residuals'][index]!r}"
        )
    return "\n".join(lines) + "\n"


def build_readme(
    manifest: Dict[str, Any],
    *,
    included: List[str],
    omitted: List[Dict[str, str]],
) -> str:
    """Plain-text README naming what is here and what deliberately is not."""
    lines = [
        "BatteryML run bundle",
        "====================",
        "",
        f"Run ID:    {manifest.get('run_id')}",
        f"Project:   {manifest.get('project')}",
        f"Status:    {manifest.get('status')}",
        f"Dataset:   {manifest.get('dataset')}",
        f"Task:      {manifest.get('task')}",
        f"Model:     {manifest.get('model_display') or manifest.get('model_key')}",
        f"Seed:      {manifest.get('seed')}",
        f"Device:    {manifest.get('device')}",
        f"Commit:    {manifest.get('git_commit') or 'not recorded'}",
        f"Directory: {manifest.get('workspace')}",
        "",
        "Metrics in metrics.json are the values BatteryML itself saved. They",
        "were not recomputed when this bundle was written.",
        "",
        "Included files:",
    ]
    lines += [f"  {name}" for name in included] or ["  (none)"]
    lines += ["", "Deliberately not included:"]
    if omitted:
        lines += [f"  {item['path']} — {item['reason']}" for item in omitted]
    else:
        lines.append("  (nothing was omitted)")
    lines += [
        "",
        "The raw battery dataset is never bundled: it is large and is not",
        "ours to redistribute. The run's own config snapshot and manifest",
        "record which dataset was used so it can be obtained separately.",
        "",
        "No API key, token or environment variable appears in this bundle.",
        "",
    ]
    return "\n".join(lines)


def write_bundle(
    run_dir: str | Path,
    destination: str | Path,
    *,
    manifest: Optional[Dict[str, Any]] = None,
    predictions_csv: Optional[str] = None,
    max_bytes: int = MAX_BUNDLED_BYTES,
) -> Dict[str, Any]:
    """Write the reproducibility zip for one run.

    Only files under ``run_dir`` are considered, so a bundle for run A can
    never contain run B's artifacts: the candidate list is rooted at this
    run's own directory, and each entry is stored under its path *relative to
    that root*.

    Returns a summary with the included names and the omissions (with
    reasons), which the README also carries.
    """
    root = Path(run_dir).resolve()
    target = Path(destination)
    target.parent.mkdir(parents=True, exist_ok=True)
    record = manifest or rm.read_manifest(root) or {}

    included: List[str] = []
    omitted: List[Dict[str, str]] = []
    candidates = _bundle_candidates(root)

    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in candidates:
            relative = path.relative_to(root).as_posix()
            try:
                size = path.stat().st_size
            except OSError:
                continue
            if size > max_bytes:
                omitted.append({
                    "path": str(path),
                    "reason": f"{size} bytes exceeds the {max_bytes}-byte bundle limit; "
                              "it remains in the run directory",
                })
                continue
            # Streamed by zipfile from the file on disk; never read whole
            # into memory first.
            archive.write(path, arcname=relative)
            included.append(relative)

        for path in root.rglob("*"):
            if path.is_file() and path.suffix.lower() in EXCLUDED_SUFFIXES:
                omitted.append({
                    "path": str(path),
                    "reason": "checkpoints are excluded by default; it remains "
                              "in the run directory",
                })

        archive.writestr(BUNDLE_MANIFEST, json.dumps(record, indent=2, sort_keys=True))
        archive.writestr(
            BUNDLE_METRICS,
            json.dumps(record.get("metrics") or {}, indent=2, sort_keys=True),
        )
        archive.writestr(
            BUNDLE_ENVIRONMENT, json.dumps(environment_snapshot(), indent=2, sort_keys=True)
        )
        archive.writestr(BUNDLE_GIT, str(record.get("git_commit") or "not recorded") + "\n")
        if predictions_csv:
            archive.writestr(BUNDLE_PREDICTIONS_CSV, predictions_csv)
            included.append(BUNDLE_PREDICTIONS_CSV)
        archive.writestr(
            BUNDLE_README,
            build_readme(record, included=sorted(included), omitted=omitted),
        )

    return {
        "path": str(target),
        "included": sorted(included),
        "omitted": omitted,
        "run_id": record.get("run_id"),
    }
