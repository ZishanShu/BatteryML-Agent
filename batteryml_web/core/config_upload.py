"""Accept an uploaded run config safely — validate first, then store it once.

Used by Home → Quick experience → "Apply uploaded configuration". Applying
a config only *selects* it; nothing here starts a run (the Quick experience
plan, :mod:`quick_experience`, gates execution separately).

An uploaded YAML is untrusted input that will later be handed to BatteryML,
which unpickles every file under the config's ``cell_data_path``. So, in this
order, an upload must:

1. carry a plain ``.yaml`` / ``.yml`` file name (no path parts, no ``..``) and
   be at most :data:`MAX_UPLOAD_BYTES`;
2. parse with ``yaml.safe_load`` into a mapping — any language-specific tag
   (``!!python/object``…) is refused;
3. pass the platform's config validator (every class name resolves in the
   live BatteryML registries) with a task established by its registered
   label annotator (:func:`config_task.resolve_config_task`), on a dataset
   and task the caller offers;
4. set no file or directory option except ``train_test_split.cell_data_path``,
   which must stay inside ``data/processed`` after symlinks are resolved — so
   an upload can neither redirect a run's workspace nor point BatteryML at
   pickles outside the processed-data tree — and no device option (the device
   is the page's choice, validated against the machine like any other run).

Only then is it written, normalized by ``yaml.safe_dump``, to
``<managed root>/uploads/configs/<random id>.yaml`` with exclusive create: the
original file name never chooses a path, and nothing — repository config or
earlier upload — is ever overwritten.

Streamlit-free.
"""

from __future__ import annotations

import hashlib
import json
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import yaml

from .config_task import resolve_config_task

#: Generous for a run config (the repository's largest is ~2 KB).
MAX_UPLOAD_BYTES = 256 * 1024
ALLOWED_SUFFIXES = (".yaml", ".yml")
MANAGED_SUBDIR = Path("uploads") / "configs"
PROCESSED_SUBDIR = Path("data") / "processed"

_SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._ -]{0,127}$")
#: A key whose value names a file or directory. ``checkpoint_freq`` and the
#: like are not matched: only a name *ending* in a path word is.
_PATH_KEY = re.compile(
    r"(^|_)(path|paths|file|filepath|filename|dir|directory|workspace|root|url|uri)$",
    re.IGNORECASE,
)
_ALLOWED_PATH_KEYS = {("train_test_split", "cell_data_path")}
#: The compute device is a CLI argument chosen on the page ("Compute device")
#: and validated there; a config may not choose it behind the page's back.
_DEVICE_KEY = re.compile(r"^(device|devices|gpu|gpus|gpu_ids|cuda)$", re.IGNORECASE)


@dataclass(frozen=True)
class UploadOutcome:
    """The result of :func:`stage_uploaded_config`.

    ``path`` is set only when ``ok``; ``errors`` is non-empty only when not.
    """

    ok: bool
    errors: Tuple[str, ...] = ()
    path: Optional[Path] = None
    task: Optional[str] = None
    model_key: Optional[str] = None
    dataset: Optional[str] = None
    warnings: Tuple[str, ...] = field(default_factory=tuple)


def check_upload_name(name: Any) -> Optional[str]:
    """Why an uploaded file name is unacceptable, or None."""
    text = str(name or "")
    if not text:
        return "The uploaded file has no name."
    if "/" in text or "\\" in text or "\x00" in text or ".." in text:
        return f"The file name {text!r} contains a path; upload a plain .yaml file."
    if not _SAFE_NAME.match(text):
        return (f"The file name {text!r} contains unsupported characters; use letters, "
                "digits, '.', '_', '-' or spaces.")
    if Path(text).suffix.lower() not in ALLOWED_SUFFIXES:
        return f"The file {text!r} is not a .yaml or .yml file."
    return None


def parse_uploaded_yaml(raw: bytes) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """``(mapping, None)`` or ``(None, reason)``. Never constructs objects."""
    if len(raw) > MAX_UPLOAD_BYTES:
        return None, (f"The file is {len(raw)} bytes; uploaded configs are limited to "
                      f"{MAX_UPLOAD_BYTES} bytes.")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        return None, f"The file is not UTF-8 text ({exc})."
    if not text.strip():
        return None, "The file is empty."
    try:
        loaded = yaml.safe_load(text)
    except yaml.constructor.ConstructorError as exc:
        return None, (f"The file uses a YAML tag that is not allowed in a config "
                      f"({str(exc).splitlines()[0]}). Only plain YAML values are accepted.")
    except yaml.YAMLError as exc:
        return None, f"The file is not valid YAML: {exc}"
    if not isinstance(loaded, dict):
        return None, (f"The file parsed to {type(loaded).__name__}; a run config must be "
                      "a mapping of sections (model, train_test_split, feature, label).")
    return loaded, None


def _walk(node: Any, trail: Tuple[str, ...] = ()) -> Iterable[Tuple[Tuple[str, ...], Any]]:
    if isinstance(node, Mapping):
        for key, value in node.items():
            here = trail + (str(key),)
            yield here, value
            yield from _walk(value, here)
    elif isinstance(node, list):
        for item in node:
            yield from _walk(item, trail)


def _as_paths(value: Any) -> List[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list) and all(isinstance(v, str) for v in value):
        return list(value)
    return []


def unsafe_options(config: Mapping[str, Any], *, repo_root: Path) -> List[str]:
    """Options an uploaded config may not set. Empty list = none found."""
    problems: List[str] = []
    for trail, _value in _walk(config):
        if _DEVICE_KEY.match(trail[-1]):
            problems.append(
                f"`{'.'.join(trail)}` chooses a compute device; the device is selected on the "
                "page (Compute device) and checked against this machine there.")
            continue
        if _PATH_KEY.search(trail[-1]) and tuple(trail[-2:]) not in _ALLOWED_PATH_KEYS:
            problems.append(
                f"`{'.'.join(trail)}` sets a file or directory; an uploaded config may only "
                "name its data through train_test_split.cell_data_path."
            )
    split = config.get("train_test_split")
    declared = split.get("cell_data_path") if isinstance(split, Mapping) else None
    paths = _as_paths(declared)
    if declared is not None and not paths:
        problems.append("`train_test_split.cell_data_path` must be a string or a list of strings.")
    processed_root = (Path(repo_root) / PROCESSED_SUBDIR).resolve()
    for text in paths:
        candidate = Path(text)
        if candidate.is_absolute() or ".." in candidate.parts:
            problems.append(
                f"`train_test_split.cell_data_path` = {text!r} must be a relative path "
                f"inside {PROCESSED_SUBDIR.as_posix()}/ (no absolute path, no '..').")
            continue
        resolved = (Path(repo_root) / candidate).resolve()
        if resolved != processed_root and processed_root not in resolved.parents:
            problems.append(
                f"`train_test_split.cell_data_path` = {text!r} resolves outside "
                f"{PROCESSED_SUBDIR.as_posix()}/ ({resolved}); uploaded configs may only "
                "read processed data.")
    return problems


def anchor_data_paths(config: Mapping[str, Any], repo_root: Path) -> Dict[str, Any]:
    """A copy whose cell_data_path is resolved against the repository root.

    The validator checks data paths relative to the process directory; the
    app's working directory is not guaranteed to be the repository.
    """
    copy = dict(config)
    split = copy.get("train_test_split")
    if isinstance(split, Mapping):
        paths = _as_paths(split.get("cell_data_path"))
        if paths:
            anchored = [str(Path(repo_root) / p) if not Path(p).is_absolute() else p for p in paths]
            copy["train_test_split"] = {
                **dict(split),
                "cell_data_path": anchored if isinstance(split.get("cell_data_path"), list) else anchored[0],
            }
    return copy


def _managed_dir(managed_root: Path) -> Path:
    """``<managed_root>/uploads/configs``, refusing any symlinked component."""
    root = Path(managed_root)
    target = root / MANAGED_SUBDIR
    probe = root
    for part in MANAGED_SUBDIR.parts:
        probe = probe / part
        if probe.is_symlink():
            raise OSError(f"{probe} is a symbolic link; uploads are never written through one.")
    target.mkdir(parents=True, exist_ok=True)
    if root.resolve() not in target.resolve().parents:
        raise OSError(f"{target} resolves outside {root}; refusing to write the upload.")
    return target


def stage_uploaded_config(
    raw: bytes,
    original_name: str,
    *,
    managed_root: Path,
    repo_root: Path,
    allowed_datasets: Sequence[str],
    allowed_tasks: Sequence[str],
) -> UploadOutcome:
    """Validate an upload and, only if it is acceptable, store it under a new id."""
    from .config_semantics import infer_config_semantics
    from .config_validator import validate_config

    name_error = check_upload_name(original_name)
    if name_error:
        return UploadOutcome(False, (name_error,))
    config, parse_error = parse_uploaded_yaml(raw)
    if parse_error or config is None:
        return UploadOutcome(False, (parse_error or "The file could not be read.",))

    errors: List[str] = []
    report = validate_config(anchor_data_paths(config, repo_root), require_data=False)
    errors.extend(report["errors"])
    resolution = resolve_config_task(config)
    if resolution.error:
        errors.append(resolution.error)
    elif not resolution.structural:
        errors.append(
            "The config's label annotator does not establish its task; a declared "
            "`task:` alone is not enough to run it.")
    elif resolution.task not in allowed_tasks:
        errors.append(f"The config trains for {resolution.task!r}; this page runs "
                      f"{' or '.join(allowed_tasks)}.")
    semantics = infer_config_semantics(config)
    if semantics.dataset not in allowed_datasets:
        errors.append(f"The config reads dataset {semantics.dataset or 'unknown'!r}; this page "
                      f"runs {', '.join(allowed_datasets)}.")
    errors.extend(unsafe_options(config, repo_root=repo_root))
    if errors:
        return UploadOutcome(False, tuple(dict.fromkeys(errors)))

    try:
        directory = _managed_dir(managed_root)
        upload_id = uuid.uuid4().hex
        path = directory / f"{upload_id}.yaml"
        with open(path, "x", encoding="utf-8") as handle:  # exclusive: never overwrite
            handle.write(yaml.safe_dump(config, sort_keys=False))
        with open(directory / f"{upload_id}.json", "x", encoding="utf-8") as handle:
            json.dump({
                "original_name": str(original_name),
                "sha256": hashlib.sha256(raw).hexdigest(),
                "uploaded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "task": resolution.task,
                "model_key": semantics.model_key,
                "dataset": semantics.dataset,
            }, handle, indent=2)
    except OSError as exc:
        return UploadOutcome(False, (f"The upload could not be stored: {exc}",))
    return UploadOutcome(True, (), path, resolution.task, semantics.model_key, semantics.dataset,
                         tuple(report["warnings"]))


def is_managed_upload(path: Path, managed_root: Path) -> bool:
    """Whether ``path`` is a file this module stored under ``managed_root``."""
    try:
        directory = (Path(managed_root) / MANAGED_SUBDIR).resolve()
        resolved = Path(path).resolve()
    except OSError:
        return False
    return resolved.parent == directory and resolved.suffix == ".yaml" and resolved.is_file()


__all__ = [
    "ALLOWED_SUFFIXES",
    "MAX_UPLOAD_BYTES",
    "UploadOutcome",
    "check_upload_name",
    "is_managed_upload",
    "parse_uploaded_yaml",
    "stage_uploaded_config",
    "unsafe_options",
    "anchor_data_paths",
]
