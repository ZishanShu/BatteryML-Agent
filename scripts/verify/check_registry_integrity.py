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
    registered_names_from_source,
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


def _model_bases() -> dict[str, list[str]]:
    """Model class name -> its base class names, parsed from source (no import)."""
    bases: dict[str, list[str]] = {}
    models_dir = repo_root() / "batteryml" / "models"
    if not models_dir.is_dir():
        return bases
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
            bases[node.name] = names
    return bases


def _resolve_device_support(model: str, bases: dict[str, list[str]]) -> tuple[list[str], str]:
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
        queue.extend(bases.get(current, []))
    return [], "not derivable: neither NNModel nor SklearnModel in the base chain"


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
    """The live builders registries, or None with the reason it could not be read."""
    builders, reason = try_import("batteryml.builders")
    if builders is None:
        return None, f"import failed: {reason}"
    # batteryml.builders defines the registries but the classes register
    # themselves on import of batteryml itself.
    _, package_reason = try_import("batteryml")
    live: dict[str, dict[str, str]] = {}
    for attribute in REGISTRY_ATTRS:
        registry = getattr(builders, attribute, None)
        mapping = getattr(registry, "class_mapping", None)
        if not isinstance(mapping, dict):
            return None, f"batteryml.builders.{attribute} has no class_mapping"
        live[attribute] = {
            name: f"{getattr(cls, '__module__', '?')}.{getattr(cls, '__name__', '?')}"
            for name, cls in mapping.items()
        }
    if package_reason:
        return live, f"live registries (partial: importing batteryml raised {package_reason})"
    return live, "live batteryml.builders registries"


def check(output_dir: Path | None = None) -> CheckResult:
    """Confirm registered names resolve, are unique, and report task/device support."""
    root = repo_root()
    output_dir = Path(output_dir) if output_dir else root / "runs" / "verify"

    errors: list[str] = []

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
        models.append(
            {
                "model": name,
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

    evidence = {
        "registry_source": registry_source,
        "registered_counts": payload["registered_counts"],
        "unresolved_input_types": unresolved_sources,
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
    return CheckResult(
        CHECK_ID,
        CHECK_NAME,
        PASS,
        f"All registered names resolve and are unique ({counts}) [{registry_source}]. "
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
