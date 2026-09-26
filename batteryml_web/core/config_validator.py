"""Validate a BatteryML run config before anything is executed.

`CLAUDE.md` §3 makes validators the gate on execution: nothing runs until one
says it may. That only works if the validator's idea of "supported" matches
what the pipeline can actually build, so every declared class name here is
resolved against the **live** ``batteryml.builders`` registries rather than a
hand-maintained whitelist.

The registries are populated exactly the way a real run populates them —
importing ``batteryml`` for the core implementations and calling
``batteryml_web.models.register_extension_models()`` for the extension models,
which is what ``batteryml_web/core/run_batteryml_cli.py`` does before handing
control to the CLI. A config this module accepts is therefore a config the
wrapper can build.

This module has no side effects. It never writes a config, never creates a
directory, never touches ``data/``, and never executes anything.
"""

from __future__ import annotations

import tempfile
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List

import yaml

#: Sections a run config must declare, each carrying a ``name``.
REQUIRED_SECTIONS = ("model", "train_test_split", "feature", "label")

#: Section -> the ``batteryml.builders`` registry its ``name`` must resolve in.
#: ``feature_transformation`` and ``label_transformation`` are optional: whether
#: a config declares one is a separate decision, so their absence is never an
#: error. When present, they must still resolve.
SECTION_REGISTRY = {
    "model": "MODELS",
    "train_test_split": "TRAIN_TEST_SPLITTERS",
    "feature": "FEATURE_EXTRACTORS",
    "label": "LABEL_ANNOTATORS",
    "feature_transformation": "DATA_TRANSFORMATIONS",
    "label_transformation": "DATA_TRANSFORMATIONS",
}

#: Tasks the platform supports, in the form reported back to the caller.
SUPPORTED_TASKS = ("RUL", "SOH")

#: Label annotator -> the task it implies. Used to derive the task from the
#: config itself rather than trusting the caller's label.
TASK_BY_ANNOTATOR = {
    "RULLabelAnnotator": "RUL",
    "SOHLabelAnnotator": "SOH",
}

#: BatteryData files that preprocessing writes into ``data/processed/<DATASET>``.
PROCESSED_GLOB = "*.pkl"


def _extension_key(model_name: str) -> str:
    """The extension module name behind a registered predictor class name.

    Reuses ``config_registry.MODEL_CLASS_KEYS`` — the table the rest of the
    platform already uses for this mapping — rather than keeping a second
    copy that could drift from it. An unknown class name yields "", which
    matches no extension.
    """
    from .config_registry import MODEL_CLASS_KEYS

    return MODEL_CLASS_KEYS.get(str(model_name).strip().lower(), "")


def _unavailable_external_models() -> Dict[str, str]:
    """Extensions skipped because their vendored source is absent, if any.

    Never raises: this only enriches an error message, so a failure to
    import the models package must not replace a useful validation error
    with an unrelated traceback.
    """
    try:
        from ..models import unavailable_extension_models

        return dict(unavailable_extension_models())
    except Exception:  # noqa: BLE001 - message enrichment only, see docstring
        return {}


@lru_cache(maxsize=1)
def _load_registries() -> tuple[Dict[str, frozenset], str | None]:
    """Registered names per registry, or the reason they cannot be read.

    Cached: importing ``batteryml`` is expensive, and registration cannot
    change without a fresh interpreter anyway, since Python caches modules.

    Returns:
        ``(registries, None)`` on success, where ``registries`` maps a registry
        attribute name to the frozenset of names registered in it; or
        ``({}, reason)`` when the registries cannot be read at all.
    """
    try:
        import batteryml  # noqa: F401 - imported for its registration side effects
        from batteryml import builders
    except Exception as exc:  # noqa: BLE001 - reported to the caller, never hidden
        return {}, (
            f"batteryml is not importable ({type(exc).__name__}: {exc}). "
            "Install the dependencies with `pip install -r requirements.txt`; "
            "no config can be validated, and no run can start, until it imports."
        )

    try:
        from ..models import register_extension_models

        register_extension_models()
    except Exception as exc:  # noqa: BLE001 - reported to the caller, never hidden
        return {}, (
            f"extension models could not be registered ({type(exc).__name__}: {exc}). "
            "batteryml_web/core/run_batteryml_cli.py performs the same registration "
            "before every run, so this would fail the run too. See "
            "batteryml_web/models/__init__.py."
        )

    registries: Dict[str, frozenset] = {}
    for attribute in sorted(set(SECTION_REGISTRY.values())):
        registry = getattr(builders, attribute, None)
        mapping = getattr(registry, "class_mapping", None)
        if not isinstance(mapping, dict):
            return {}, (
                f"batteryml.builders.{attribute} has no class_mapping; the registry "
                "API has changed and this validator needs updating."
            )
        registries[attribute] = frozenset(mapping)
    return registries, None


def registry_status() -> tuple[Dict[str, frozenset], str | None]:
    """Public view of the registries this validator resolves names against.

    Exposed so a caller can tell "the registries are unavailable" from "this
    config is wrong" before interpreting a rejection. ``VAL-01``
    (``scripts/verify/check_validator_contract.py``) uses it to report a skip
    rather than refusing every config for the same reason.

    Returns:
        ``(registries, None)`` when readable, else ``({}, reason)``.
    """
    return _load_registries()


def _as_path(value: Any) -> Path | None:
    if value is None:
        return None
    return Path(str(value))


def _normalize_task(raw: Any) -> tuple[str | None, bool]:
    """Fold a caller-supplied task label to ``RUL`` / ``SOH``.

    Accepts both the bare form (``"RUL"``) and the UI's descriptive form
    (``"RUL prediction"``, ``"SOH estimation"``; see ``batteryml_web/app.py:1048``).

    Returns:
        ``(task, recognized)``. When not recognized, ``task`` is the uppercased
        input so the caller can see what was rejected.
    """
    text = str(raw or "").strip().upper()
    if not text:
        return None, True
    for supported in SUPPORTED_TASKS:
        if text == supported or text.startswith(supported + " "):
            return supported, True
    return text, False


def _section_names(section: Any) -> List[str]:
    """Every class name a section declares, including nested transformations."""
    if not isinstance(section, dict):
        return []
    names: List[str] = []
    name = section.get("name")
    if isinstance(name, str) and name:
        names.append(name)
    nested = section.get("transformations")
    if isinstance(nested, list):
        for item in nested:
            names.extend(_section_names(item))
    return names


def _declared_paths(config: Dict[str, Any]) -> List[str]:
    """The ``cell_data_path`` entries a config declares, normalized to a list.

    The field is a string for single-dataset presets and a list for the
    multi-dataset ones (``cruh``, ``crush``, ``mix``).
    """
    split = config.get("train_test_split")
    if not isinstance(split, dict):
        return []
    value = split.get("cell_data_path")
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [str(item) for item in value]
    return []


def _writability_error(output_dir: Path) -> str | None:
    """Whether ``output_dir`` could be written, without creating it.

    Probes the nearest existing ancestor instead of creating the directory, so
    that validating a config leaves nothing behind.
    """
    probe = output_dir
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    if not probe.is_dir():
        return (
            f"Output directory {output_dir} cannot be created: {probe} exists but is "
            "not a directory."
        )
    try:
        with tempfile.NamedTemporaryFile(dir=probe, prefix=".batteryml_write_test_"):
            pass
    except OSError as exc:
        return (
            f"Output directory {output_dir} is not writable: {type(exc).__name__}: "
            f"{exc} (probed the nearest existing ancestor, {probe})."
        )
    return None


def validate_config(
    config: Dict[str, Any],
    *,
    require_data: bool = True,
) -> Dict[str, Any]:
    """Decide whether a run config may be executed.

    Args:
        config: the parsed config, optionally carrying the extra keys
            ``config_path``, ``task`` and ``output_dir`` / ``workspace`` that
            :func:`load_and_validate_config_file` adds.
        require_data: when True (the default, and the right setting before a
            run), a declared ``cell_data_path`` that is missing or holds no
            ``*.pkl`` is an error, because the run provably cannot succeed:
            ``BaseTrainTestSplitter`` asserts the path exists
            (``batteryml/train_test_split/base.py:25``) and an existing but
            empty directory passes that assert and then yields an empty cell
            list. Set False to check a config's structure before the data is
            preprocessed; the same findings are then reported as warnings.

    Returns:
        A report with ``valid``, ``errors`` and ``warnings``, plus the resolved
        ``task``, ``model``, ``split_strategy``, ``dataset_path`` (the first
        declared path, kept for backward compatibility) and ``dataset_paths``.
    """
    errors: List[str] = []
    warnings: List[str] = []

    config_path = _as_path(config.get("config_path"))
    if config_path is not None and not config_path.exists():
        errors.append(f"Config file does not exist: {config_path}")

    # --- task -------------------------------------------------------------
    label = config.get("label")
    annotator = label.get("name") if isinstance(label, dict) else None
    derived_task = TASK_BY_ANNOTATOR.get(str(annotator)) if annotator else None

    requested_task, recognized = _normalize_task(config.get("task"))
    if not recognized:
        errors.append(
            f"Unsupported task: {requested_task}. Supported: {', '.join(SUPPORTED_TASKS)}."
        )
    if requested_task and derived_task and recognized and requested_task != derived_task:
        warnings.append(
            f"Requested task {requested_task} disagrees with the config's label "
            f"annotator {annotator!r}, which implies {derived_task}. The run will "
            f"follow the config, not the request."
        )
    task = derived_task or (requested_task if recognized else None)

    # --- structure --------------------------------------------------------
    for section_name in REQUIRED_SECTIONS:
        section = config.get(section_name)
        if section is None:
            errors.append(f"Missing required section `{section_name}`.")
        elif not isinstance(section, dict):
            errors.append(
                f"Section `{section_name}` is {type(section).__name__}, expected a mapping."
            )
        elif not section.get("name"):
            errors.append(f"Section `{section_name}` has no `name`.")

    model = config.get("model")
    model_name = model.get("name") if isinstance(model, dict) else None
    split = config.get("train_test_split")
    split_name = split.get("name") if isinstance(split, dict) else None

    # --- every declared name must resolve to a registered implementation ---
    registries, registry_error = _load_registries()
    if registry_error is not None:
        errors.append(registry_error)
    else:
        unavailable = _unavailable_external_models()
        for section_name, registry_attr in SECTION_REGISTRY.items():
            registered = registries.get(registry_attr, frozenset())
            for name in _section_names(config.get(section_name)):
                if name in registered:
                    continue
                # "Not registered" is true but misleading for a model whose
                # wrapper is present and correct and whose *vendored source*
                # was never cloned: it sends the reader to fix a config or
                # write an implementation, when the actual action is to
                # vendor a third-party repository. Name the real cause.
                source = unavailable.get(_extension_key(name))
                if source is not None:
                    errors.append(
                        f"`{section_name}.name` = {name!r} is unavailable in this "
                        f"checkout: it wraps a third-party source that is not "
                        f"vendored here ({source} is missing, and third_party/ is "
                        f"gitignored). The config and the wrapper are both fine. "
                        f"Vendor that repository to enable this model, or use one "
                        f"of the models that need no external source."
                    )
                    continue
                errors.append(
                    f"`{section_name}.name` = {name!r} is not registered in "
                    f"{registry_attr}. Registered: "
                    f"{', '.join(sorted(registered)) or '(none)'}."
                )

    # --- data readiness ---------------------------------------------------
    dataset_paths = _declared_paths(config)
    record = errors if require_data else warnings
    for declared in dataset_paths:
        path = Path(declared)
        if not path.exists():
            record.append(
                f"Declared cell_data_path does not exist: {declared}. "
                f"Run preprocessing first "
                f"(python scripts/preprocess_sweep.py --only {path.name})."
            )
        elif not path.is_dir():
            record.append(f"Declared cell_data_path is not a directory: {declared}.")
        elif not any(path.glob(PROCESSED_GLOB)):
            record.append(
                f"Declared cell_data_path {declared} holds no {PROCESSED_GLOB} file. "
                "An empty processed directory satisfies BaseTrainTestSplitter's "
                "existence assert and then yields an empty cell list, so the run would "
                "train on nothing. Run preprocessing, or remove the empty directory."
            )

    # --- output directory -------------------------------------------------
    output_dir = _as_path(config.get("output_dir") or config.get("workspace"))
    if output_dir is not None:
        writability_error = _writability_error(output_dir)
        if writability_error is not None:
            errors.append(writability_error)

    return {
        "valid": not errors,
        "errors": errors,
        "warnings": warnings,
        "task": task,
        "model": model_name,
        "split_strategy": split_name,
        "dataset_path": dataset_paths[0] if dataset_paths else None,
        "dataset_paths": dataset_paths,
        "require_data": require_data,
    }


def load_and_validate_config_file(
    path: str | Path,
    task: str | None = None,
    output_dir: str | Path | None = None,
    *,
    require_data: bool = True,
) -> Dict[str, Any]:
    """Read a config file from disk and validate it.

    Args:
        path: the config file.
        task: optional task label to cross-check against the config's own label
            annotator. Accepts ``"RUL"`` or ``"RUL prediction"``. When omitted,
            the task is derived from the config.
        output_dir: optional workspace, checked for writability.
        require_data: see :func:`validate_config`.
    """
    path = Path(path)
    if not path.exists():
        return {
            "valid": False,
            "errors": [f"Config file does not exist: {path}"],
            "warnings": [],
            "task": None,
            "model": None,
            "split_strategy": None,
            "dataset_path": None,
            "dataset_paths": [],
            "require_data": require_data,
        }

    try:
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        return {
            "valid": False,
            "errors": [f"Config file is not valid YAML: {path}: {exc}"],
            "warnings": [],
            "task": None,
            "model": None,
            "split_strategy": None,
            "dataset_path": None,
            "dataset_paths": [],
            "require_data": require_data,
        }

    if loaded is None:
        loaded = {}
    if not isinstance(loaded, dict):
        return {
            "valid": False,
            "errors": [
                f"Config file {path} parsed to {type(loaded).__name__}, expected a "
                "mapping of top-level sections."
            ],
            "warnings": [],
            "task": None,
            "model": None,
            "split_strategy": None,
            "dataset_path": None,
            "dataset_paths": [],
            "require_data": require_data,
        }

    config = dict(loaded)
    config["config_path"] = str(path)
    if task is not None:
        config["task"] = task
    if output_dir is not None:
        config["output_dir"] = str(output_dir)
    return validate_config(config, require_data=require_data)
