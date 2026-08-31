"""CFG-01 — every config under configs/ parses, has its required keys, and names things that exist.

Checked as failures:

* the file does not parse as YAML, or does not parse to a mapping;
* a required top-level section is missing, or its ``name`` is missing;
* a declared class name does not resolve to a registered implementation
  (model, splitter, feature extractor, label annotator, data transformation).

Reported as **observations**, never failures:

* a missing ``feature_transformation`` — whether a config declares one is a
  blocked decision and is not this harness's to judge;
* a ``cell_data_path`` that does not exist on disk — the checks are
  data-optional by design, and DATA-01 owns dataset presence.

Cycler column-map files (``configs/cyclers/``) describe raw-file columns, not
runs; they are listed as skipped rather than judged against the run schema.
"""

from __future__ import annotations

import argparse
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
    write_json,
)

CHECK_ID = "CFG-01"
CHECK_NAME = "config_validity"

# Sections a run config must declare, each with a `name`.
REQUIRED_SECTIONS = ("model", "train_test_split", "feature", "label")

# Section -> the builders registry its `name` must resolve in.
SECTION_REGISTRY = {
    "model": "MODELS",
    "train_test_split": "TRAIN_TEST_SPLITTERS",
    "feature": "FEATURE_EXTRACTORS",
    "label": "LABEL_ANNOTATORS",
    "feature_transformation": "DATA_TRANSFORMATIONS",
    "label_transformation": "DATA_TRANSFORMATIONS",
}

# Directories under configs/ that are not run configs.
NON_RUN_DIRS = ("cyclers",)


def _load_registries() -> tuple[dict[str, dict[str, str]], str]:
    """Registered names per registry, and where they came from.

    Prefers the live ``batteryml.builders`` registries (which is what the
    pipeline actually looks names up in). Falls back to parsing
    ``@REGISTRY.register()`` decorators out of the source when ``batteryml``
    cannot be imported, so the check still works without torch installed.
    """
    attributes = set(SECTION_REGISTRY.values())
    live, source = load_live_registries(attributes)
    if live is not None:
        return live, source

    return (
        {attribute: registered_names_from_source(attribute) for attribute in attributes},
        f"source scan of @REGISTRY.register() decorators ({source})",
    )


def _section_names(section: Any) -> list[str]:
    """Every class name a config section declares, including nested transformations."""
    if not isinstance(section, dict):
        return []
    names: list[str] = []
    name = section.get("name")
    if isinstance(name, str) and name:
        names.append(name)
    nested = section.get("transformations")
    if isinstance(nested, list):
        for item in nested:
            names.extend(_section_names(item))
    return names


def _declared_paths(config: dict[str, Any]) -> list[str]:
    """The cell_data_path entries a config declares, normalized to a list."""
    split = config.get("train_test_split")
    if not isinstance(split, dict):
        return []
    value = split.get("cell_data_path")
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [str(item) for item in value]
    return []


def _inspect_config(
    root: Path,
    path: Path,
    registries: dict[str, dict[str, str]],
) -> dict[str, Any]:
    """Parse and validate one config file. Never raises."""
    relative = relative_to_repo(path)
    record: dict[str, Any] = {
        "config": relative,
        "errors": [],
        "observations": [],
        "declared_paths": [],
        "missing_paths": [],
    }

    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        record["errors"].append(f"cannot read file: {type(exc).__name__}: {exc}")
        return record

    try:
        loaded = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        record["errors"].append(f"YAML parse error: {exc}")
        return record

    if not isinstance(loaded, dict):
        record["errors"].append(
            f"parsed to {type(loaded).__name__}, expected a mapping of top-level sections"
        )
        return record

    record["sections"] = sorted(loaded)

    for section_name in REQUIRED_SECTIONS:
        section = loaded.get(section_name)
        if section is None:
            record["errors"].append(f"missing required section `{section_name}`")
            continue
        if not isinstance(section, dict):
            record["errors"].append(
                f"section `{section_name}` is {type(section).__name__}, expected a mapping"
            )
            continue
        if not section.get("name"):
            record["errors"].append(f"section `{section_name}` has no `name`")

    if "feature_transformation" not in loaded:
        record["no_feature_transformation"] = True
        record["observations"].append(
            "no `feature_transformation` section (blocked decision — reported, not judged)"
        )

    for section_name, registry_attr in SECTION_REGISTRY.items():
        section = loaded.get(section_name)
        registry = registries.get(registry_attr, {})
        for name in _section_names(section):
            if name not in registry:
                record["errors"].append(
                    f"`{section_name}.name` = {name!r} is not registered in {registry_attr}"
                )

    model = loaded.get("model")
    if isinstance(model, dict):
        record["model_name"] = model.get("name")

    for declared in _declared_paths(loaded):
        record["declared_paths"].append(declared)
        if not (root / declared).exists():
            record["missing_paths"].append(declared)
    if record["missing_paths"]:
        record["observations"].append(
            "declared cell_data_path not present on disk: "
            + ", ".join(record["missing_paths"])
            + " (data-optional — see DATA-01)"
        )

    return record


def check(output_dir: Path | None = None) -> CheckResult:
    """Parse and validate every YAML under configs/."""
    root = repo_root()
    output_dir = Path(output_dir) if output_dir else root / "runs" / "verify"
    configs_root = root / "configs"

    if not configs_root.is_dir():
        return CheckResult(
            CHECK_ID,
            CHECK_NAME,
            FAIL,
            f"{relative_to_repo(configs_root)} does not exist; there are no configs to validate.",
            {"configs_root": relative_to_repo(configs_root)},
        )

    registries, registry_source = _load_registries()

    run_configs: list[dict[str, Any]] = []
    skipped: list[str] = []
    for path in sorted(configs_root.rglob("*.yaml")):
        relative_parts = path.relative_to(configs_root).parts
        if relative_parts and relative_parts[0] in NON_RUN_DIRS:
            skipped.append(relative_to_repo(path))
            continue
        run_configs.append(_inspect_config(root, path, registries))

    failing = [record for record in run_configs if record["errors"]]
    # Kept apart so the blocked-decision observation is not buried under the
    # data-absence observation, which fires on every config when data/ is empty.
    without_feature_transformation = [
        record["config"] for record in run_configs if record.get("no_feature_transformation")
    ]
    with_missing_paths = [record["config"] for record in run_configs if record["missing_paths"]]

    payload = {
        "check_id": CHECK_ID,
        "configs_root": relative_to_repo(configs_root),
        "registry_source": registry_source,
        "registered_counts": {name: len(mapping) for name, mapping in sorted(registries.items())},
        "run_config_count": len(run_configs),
        "skipped_non_run_configs": skipped,
        "configs": run_configs,
    }
    artifacts = [write_json(output_dir / "config_validity.json", payload)]

    evidence = {
        "registry_source": registry_source,
        "run_config_count": len(run_configs),
        "skipped_non_run_configs": skipped,
        "failing_count": len(failing),
        "failing": [
            {"config": record["config"], "errors": record["errors"]} for record in failing
        ],
        "configs_without_feature_transformation": without_feature_transformation,
        "configs_with_missing_declared_paths": with_missing_paths,
    }

    if failing:
        first = failing[0]
        return CheckResult(
            CHECK_ID,
            CHECK_NAME,
            FAIL,
            f"{len(failing)} of {len(run_configs)} run config(s) are invalid. "
            f"First: {first['config']} — {first['errors'][0]}. "
            "Fix the config or register the missing implementation; see "
            "evidence.failing for the full list.",
            evidence,
            artifacts,
        )

    return CheckResult(
        CHECK_ID,
        CHECK_NAME,
        PASS,
        f"All {len(run_configs)} run config(s) parse, declare the required sections "
        f"({', '.join(REQUIRED_SECTIONS)}) and name registered implementations "
        f"[{registry_source}]. Observations: "
        f"{len(without_feature_transformation)} without feature_transformation "
        f"(blocked decision), {len(with_missing_paths)} whose declared cell_data_path "
        f"is absent (see DATA-01). {len(skipped)} cycler column-map file(s) skipped.",
        evidence,
        artifacts,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for config_validity.json (default: runs/verify/).",
    )
    args = parser.parse_args(argv)

    result = run_check(CHECK_ID, CHECK_NAME, check, output_dir=args.output_dir)
    print(f"[{result.status}] {result.check_id} {result.name}: {result.message}")
    for artifact in result.artifacts:
        print(f"       wrote {artifact}")
    return 1 if result.status == FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
