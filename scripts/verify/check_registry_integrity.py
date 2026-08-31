"""REG-01 — every registered dataset and model resolves to a real implementation.

Checked as failures:

* a dataset advertised in ``SUPPORTED_SOURCES`` has no
  ``<NAME>Preprocessor`` registered in ``PREPROCESSORS`` — ``bin/batteryml.py``
  accepts it as ``input_type`` and then builds ``f'{input_type}Preprocessor'``,
  so the mismatch surfaces as a ``KeyError`` only once the user has already
  supplied raw data;
* two classes register the same name in one registry;
* a registered name resolves to nothing.

Reported as an **observation**, not a failure:

* tasks and devices. No BatteryML model declares either — ``BaseModel``
  (``batteryml/models/base.py:11``) carries only ``workspace``, and no
  subclass adds a ``tasks`` or ``devices`` attribute. The requirement cannot
  be satisfied against the current core, so this check reports the *derived*
  task and device support (task from the predictor class suffix and from the
  configs that reference it; devices from whether the class is an NNModel or
  a SklearnModel) and flags every model that carries no declaration. Turning
  this into a failure would require either an upstream change or a
  declaration table in ``batteryml_web/``; that is a design decision, not
  something this harness should force.
"""

from __future__ import annotations

import argparse
import ast
from pathlib import Path
from typing import Any

import yaml

from .common import (
    FAIL,
    PASS,
    CheckResult,
    load_live_registries,
    registered_names_from_source,
    relative_to_repo,
    repo_root,
    run_check,
    supported_sources,
    try_import,
    write_json,
)

CHECK_ID = "REG-01"
CHECK_NAME = "registry_integrity"

REGISTRY_ATTRS = (
    "MODELS",
    "PREPROCESSORS",
    "FEATURE_EXTRACTORS",
    "LABEL_ANNOTATORS",
    "TRAIN_TEST_SPLITTERS",
    "DATA_TRANSFORMATIONS",
)

# Base class -> device support. Derived, not declared; see the module docstring.
NN_BASES = {"NNModel"}
SKLEARN_BASES = {"SklearnModel"}

DEVICES_NN = ["cpu", "cuda"]
DEVICES_SKLEARN = ["cpu"]


def _model_source_roots() -> list[Path]:
    """Directories that may define a model class.

    ``batteryml_web/models/`` is included because an extension model registered
    there reaches the same MODELS registry (see
    ``batteryml_web/models/__init__.py``). Scanning only ``batteryml/models/``
    would leave every extension model's base chain unresolvable, which this
    check reports as a failure.
    """
    root = repo_root()
    return [root / "batteryml" / "models", root / "batteryml_web" / "models"]


def _model_bases() -> dict[str, dict[str, Any]]:
    """Model class name -> ``{"bases": [...], "defined_in": "path:line"}``.

    Parsed from source with ``ast``; nothing is imported, so this works
    without torch installed.
    """
    bases: dict[str, dict[str, Any]] = {}
    for models_dir in _model_source_roots():
        if not models_dir.is_dir():
            continue
        for source_path in sorted(models_dir.rglob("*.py")):
            tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
            for node in ast.walk(tree):
                if not isinstance(node, ast.ClassDef):
                    continue
                names = []
                for base in node.bases:
                    if isinstance(base, ast.Name):
                        names.append(base.id)
                    elif isinstance(base, ast.Attribute):
                        names.append(base.attr)
                bases[node.name] = {
                    "bases": names,
                    "defined_in": f"{relative_to_repo(source_path)}:{node.lineno}",
                }
    return bases


def _resolve_device_support(model: str, bases: dict[str, dict[str, Any]]) -> tuple[list[str], str]:
    """Derive device support by walking the class's base chain."""
    seen: set[str] = set()
    queue = [model]
    while queue:
        current = queue.pop()
        if current in seen:
            continue
        seen.add(current)
        if current in NN_BASES:
            return DEVICES_NN, f"derived: {model} inherits {current}"
        if current in SKLEARN_BASES:
            return DEVICES_SKLEARN, f"derived: {model} inherits {current}"
        entry = bases.get(current)
        if entry:
            queue.extend(entry["bases"])
    return [], "not derivable: neither NNModel nor SklearnModel in the base chain"


def _origin(defined_in: str | None) -> str:
    """Whether a model class lives in upstream core or in the extension point."""
    if not defined_in:
        return "unknown"
    if defined_in.startswith("batteryml_web/models/"):
        return "extension"
    if defined_in.startswith("batteryml/"):
        return "upstream"
    return "unknown"


def _extension_state() -> dict[str, Any]:
    """What the extension package declares, and whether it is self-consistent.

    ``batteryml_web.models`` is deliberately import-light (importlib and
    pathlib only), so it can be read here without torch installed.
    """
    state: dict[str, Any] = {
        "importable": False,
        "declared_modules": [],
        "unlisted_modules": [],
        "missing_modules": [],
        "error": None,
    }
    package, reason = try_import("batteryml_web.models")
    if package is None:
        state["error"] = reason
        return state
    state["importable"] = True
    state["declared_modules"] = list(getattr(package, "EXTENSION_MODULES", ()))
    try:
        state["unlisted_modules"] = list(package.unlisted_modules())
        state["missing_modules"] = list(package.missing_modules())
    except Exception as exc:  # noqa: BLE001 - reported, never hidden
        state["error"] = f"{type(exc).__name__}: {exc}"
    return state


def _tasks_from_configs(root: Path) -> dict[str, set[str]]:
    """Model class name -> the task labels of the configs that reference it."""
    tasks: dict[str, set[str]] = {}
    configs_root = root / "configs"
    if not configs_root.is_dir():
        return tasks
    for path in sorted(configs_root.rglob("*.yaml")):
        try:
            loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError):
            continue
        if not isinstance(loaded, dict):
            continue
        model = loaded.get("model")
        label = loaded.get("label")
        if not isinstance(model, dict) or not model.get("name"):
            continue
        annotator = label.get("name") if isinstance(label, dict) else None
        if annotator == "SOHLabelAnnotator":
            task = "SOH"
        elif annotator == "RULLabelAnnotator":
            task = "RUL"
        else:
            continue
        tasks.setdefault(str(model["name"]), set()).add(task)
    return tasks


def _live_registries() -> tuple[dict[str, dict[str, str]] | None, str]:
    """The live builders registries, or None with the reason they cannot be read.

    Delegates to common.load_live_registries, which imports every registering
    module -- batteryml, batteryml.preprocess and batteryml_web.models -- before
    reading any class_mapping. Reading it without importing batteryml.preprocess
    first made this check order-dependent: PREPROCESSORS was empty unless
    DATA-01 had already imported that module as a side effect, and every
    advertised input_type was then reported unresolved.
    """
    return load_live_registries(REGISTRY_ATTRS)


def check(output_dir: Path | None = None) -> CheckResult:
    """Confirm registered names resolve, are unique, and report task/device support."""
    root = repo_root()
    output_dir = Path(output_dir) if output_dir else root / "runs" / "verify"

    errors: list[str] = []

    # The extension point must be self-consistent before anything else is
    # trusted: an unlisted module file is a model that will never register.
    extension = _extension_state()
    if not extension["importable"]:
        errors.append(
            f"batteryml_web.models is not importable ({extension['error']}). "
            "The CLI wrapper imports it before every run, so this breaks every "
            "invocation, not just extension models."
        )
    else:
        if extension["unlisted_modules"]:
            errors.append(
                f"{len(extension['unlisted_modules'])} module file(s) in "
                f"batteryml_web/models/ are not listed in EXTENSION_MODULES: "
                f"{', '.join(extension['unlisted_modules'])}. An unlisted module is "
                f"never imported, so its @MODELS.register() never fires. Add the "
                f"name(s) to EXTENSION_MODULES, or prefix the file with '_' if it is "
                f"a helper rather than a model."
            )
        if extension["missing_modules"]:
            errors.append(
                f"EXTENSION_MODULES lists {len(extension['missing_modules'])} module(s) "
                f"with no file in batteryml_web/models/: "
                f"{', '.join(extension['missing_modules'])}. Create the file(s) or "
                f"remove the name(s)."
            )
        if extension["error"]:
            errors.append(f"batteryml_web.models is inconsistent: {extension['error']}")

    live, live_reason = _live_registries()
    # registered_names_from_source raises ValueError on a duplicate registration,
    # which is exactly the condition this check is meant to surface.
    try:
        from_source = {attr: registered_names_from_source(attr) for attr in REGISTRY_ATTRS}
        duplicate_error = None
    except ValueError as exc:
        from_source = {}
        duplicate_error = str(exc)
        errors.append(f"duplicate registration: {exc}")

    registries = live if live is not None else from_source
    registry_source = live_reason if live is not None else f"source scan ({live_reason})"

    for attribute, mapping in sorted(registries.items()):
        for name, target in sorted(mapping.items()):
            if not target:
                errors.append(f"{attribute}[{name!r}] resolves to nothing")

    # Datasets and cyclers advertised by the CLI vs preprocessors that exist.
    advertised, dataset_source = supported_sources()
    preprocessors = registries.get("PREPROCESSORS", {})

    unresolved_sources: list[dict[str, str]] = []
    for group, names in sorted(advertised.items()):
        for name in names:
            expected = f"{name}Preprocessor"
            if expected not in preprocessors:
                unresolved_sources.append(
                    {
                        "group": group,
                        "input_type": name,
                        "expected_class": expected,
                    }
                )
    if unresolved_sources:
        listed = ", ".join(item["input_type"] for item in unresolved_sources)
        errors.append(
            f"{len(unresolved_sources)} input_type(s) advertised by "
            f"batteryml/preprocess/__init__.py SUPPORTED_SOURCES have no registered "
            f"preprocessor: {listed}"
        )

    # Task and device support: derived, because nothing declares it.
    bases = _model_bases()
    config_tasks = _tasks_from_configs(root)
    models: list[dict[str, Any]] = []
    undeclared: list[str] = []
    for name in sorted(registries.get("MODELS", {})):
        devices, device_basis = _resolve_device_support(name, bases)
        suffix_task = (
            "SOH" if name.endswith("SOHPredictor")
            else "RUL" if name.endswith("RULPredictor")
            else None
        )
        tasks = sorted(config_tasks.get(name, set()) | ({suffix_task} if suffix_task else set()))
        defined_in = (bases.get(name) or {}).get("defined_in")
        models.append(
            {
                "model": name,
                "origin": _origin(defined_in),
                "defined_in": defined_in,
                "declares_tasks": False,
                "declares_devices": False,
                "derived_tasks": tasks,
                "derived_devices": devices,
                "device_basis": device_basis,
                "referenced_by_configs": sorted(config_tasks.get(name, set())) or [],
            }
        )
        undeclared.append(name)
        if not devices:
            errors.append(
                f"MODELS[{name!r}] device support is not derivable: {device_basis}"
            )

    payload = {
        "check_id": CHECK_ID,
        "registry_source": registry_source,
        "dataset_source": dataset_source,
        "registered": {attr: sorted(mapping) for attr, mapping in sorted(registries.items())},
        "registered_counts": {attr: len(mapping) for attr, mapping in sorted(registries.items())},
        "advertised_input_types": advertised,
        "unresolved_input_types": unresolved_sources,
        "extension_point": extension,
        "models": models,
        "duplicate_error": duplicate_error,
    }
    artifacts = [write_json(output_dir / "registry_integrity.json", payload)]

    observation = (
        f"No model declares tasks or devices: BaseModel (batteryml/models/base.py:11) "
        f"carries only `workspace` and no subclass adds `tasks` or `devices`. Task and "
        f"device support for all {len(undeclared)} model(s) is derived instead "
        f"(class-name suffix plus referencing configs; NNModel vs SklearnModel base). "
        f"Satisfying the requirement as written needs either an upstream change or a "
        f"declaration table in batteryml_web/ — a design decision, not a harness fix."
    )

    extension_models = [entry["model"] for entry in models if entry["origin"] == "extension"]
    evidence = {
        "registry_source": registry_source,
        "registered_counts": payload["registered_counts"],
        "unresolved_input_types": unresolved_sources,
        "extension_point": extension,
        "extension_models": extension_models,
        "models_without_declared_tasks_or_devices": undeclared,
        "derived_model_support": models,
        "observations": [observation],
        "errors": errors,
    }

    if errors:
        return CheckResult(
            CHECK_ID,
            CHECK_NAME,
            FAIL,
            f"{len(errors)} registry problem(s). First: {errors[0]}. "
            "See evidence.errors for the full list.",
            evidence,
            artifacts,
        )

    counts = ", ".join(
        f"{attr.lower()}={len(mapping)}" for attr, mapping in sorted(registries.items())
    )
    extension_note = (
        f"{len(extension_models)} extension model(s) from batteryml_web/models/: "
        f"{', '.join(extension_models)}"
        if extension_models
        else "extension point declares no models"
    )
    return CheckResult(
        CHECK_ID,
        CHECK_NAME,
        PASS,
        f"All registered names resolve and are unique ({counts}) [{registry_source}]. "
        f"{extension_note}. "
        "Observation: no model declares tasks or devices; both are derived.",
        evidence,
        artifacts,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for registry_integrity.json (default: runs/verify/).",
    )
    args = parser.parse_args(argv)

    result = run_check(CHECK_ID, CHECK_NAME, check, output_dir=args.output_dir)
    print(f"[{result.status}] {result.check_id} {result.name}: {result.message}")
    for artifact in result.artifacts:
        print(f"       wrote {artifact}")
    return 1 if result.status == FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
