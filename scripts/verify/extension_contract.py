"""EXT-01 — enforce the external-method contract and catalog gate."""

from __future__ import annotations

import argparse
import ast
import inspect
from pathlib import Path
from typing import Any

from .common import FAIL, PASS, CheckResult, relative_to_repo, repo_root, run_check, write_json

CHECK_ID = "EXT-01"
CHECK_NAME = "extension_contract"
MODEL_REGISTRY = "MODELS"
REQUIRED_MODEL_METHODS = ("fit", "predict", "dump_checkpoint", "load_checkpoint", "to")
MODEL_BASES = {"BaseModel", "NNModel", "SklearnModel"}


def _registered_models_from_source(root: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for source_root in (root / "batteryml" / "models", root / "batteryml_web" / "models"):
        if not source_root.is_dir():
            continue
        for path in sorted(source_root.rglob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if not isinstance(node, ast.ClassDef):
                    continue
                names: list[str] = []
                for decorator in node.decorator_list:
                    if not isinstance(decorator, ast.Call):
                        continue
                    function = decorator.func
                    if not isinstance(function, ast.Attribute) or function.attr != "register":
                        continue
                    if not isinstance(function.value, ast.Name) or function.value.id != MODEL_REGISTRY:
                        continue
                    if decorator.args and isinstance(decorator.args[0], ast.Constant):
                        names.append(str(decorator.args[0].value))
                    else:
                        names.append(node.name)
                for name in names:
                    result[name] = {
                        "class": node.name,
                        "path": relative_to_repo(path),
                        "line": node.lineno,
                        "bases": [base.id for base in node.bases if isinstance(base, ast.Name)],
                        "methods": {item.name for item in node.body if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))},
                    }
    return result


def _source_has_model_base(name: str, classes: dict[str, dict[str, Any]], seen: set[str] | None = None) -> bool:
    seen = seen or set()
    if name in seen:
        return False
    seen.add(name)
    info = classes.get(name)
    if info is None:
        return name in MODEL_BASES
    return any(base in MODEL_BASES or _source_has_model_base(base, classes, seen) for base in info["bases"])


def _live_model_failures() -> tuple[dict[str, Any], list[str]]:
    try:
        import batteryml  # noqa: F401
        import batteryml.models  # noqa: F401
        from batteryml import builders
        from batteryml_web.models import register_extension_models

        register_extension_models()
        mapping = dict(builders.MODELS.class_mapping)
    except Exception as exc:  # noqa: BLE001 - report the actual gate failure
        return {}, [f"model registry could not be imported: {type(exc).__name__}: {exc}"]

    failures: list[str] = []
    evidence: dict[str, Any] = {}
    for name, cls in sorted(mapping.items()):
        methods: dict[str, Any] = {}
        for method_name in REQUIRED_MODEL_METHODS:
            method = getattr(cls, method_name, None)
            if not callable(method):
                failures.append(f"MODELS[{name!r}] has no callable {method_name}")
                continue
            try:
                methods[method_name] = list(inspect.signature(method).parameters)
            except (TypeError, ValueError) as exc:
                failures.append(f"MODELS[{name!r}].{method_name} cannot be inspected: {exc}")
        bases = {base.__name__ for base in inspect.getmro(cls)}
        if not bases & MODEL_BASES:
            failures.append(f"MODELS[{name!r}] does not inherit BaseModel/NNModel/SklearnModel")
        evidence[name] = {"module": cls.__module__, "class": cls.__name__, "methods": methods}
    return evidence, failures


def check(output_dir: Path | None = None) -> CheckResult:
    root = repo_root()
    output_dir = Path(output_dir) if output_dir else root / "runs" / "verify"
    failures: list[str] = []
    source_models = _registered_models_from_source(root)
    source_classes: dict[str, dict[str, Any]] = {}
    for item in source_models.values():
        source_classes[item["class"]] = item
    for registered_name, item in sorted(source_models.items()):
        if not _source_has_model_base(item["class"], source_classes):
            failures.append(f"{registered_name!r} at {item['path']} does not inherit a model base")
        missing = sorted(set(REQUIRED_MODEL_METHODS) - set(item["methods"]))
        # Inherited methods are accepted here; the live check below verifies the effective class.
        if missing and not _source_has_model_base(item["class"], source_classes):
            failures.append(f"{registered_name!r} is missing model methods: {', '.join(missing)}")

    live_evidence, live_failures = _live_model_failures()
    failures.extend(live_failures)

    from batteryml_web.core.catalog import build_catalog
    catalog = build_catalog(root)
    for issue in catalog.issues:
        failures.append(f"catalog could not parse {relative_to_repo(issue.config_path)}: {issue.detail}")
    if catalog.registry_error:
        failures.append(f"catalog registry import failed: {catalog.registry_error}")
    for entry in catalog.entries:
        if not entry.model_name or not entry.feature_name or not entry.label_name:
            failures.append(f"catalog entry {relative_to_repo(entry.config_path)} has unresolved required names")
        if entry.runnable and entry.failures:
            failures.append(f"catalog entry {relative_to_repo(entry.config_path)} claims runnable with gate failures")
        if not entry.runnable and entry.status.value == "runnable":
            failures.append(f"catalog entry {relative_to_repo(entry.config_path)} has non-runnable state runnable")

    payload = {
        "check_id": CHECK_ID,
        "source_registered_models": source_models,
        "live_models": live_evidence,
        "catalog": catalog.to_dict(include_debug=True),
        "failures": failures,
    }
    artifacts = [write_json(output_dir / "extension_contract.json", payload)]
    evidence = {
        "source_model_count": len(source_models),
        "live_model_count": len(live_evidence),
        "catalog_entry_count": len(catalog.entries),
        "failure_count": len(failures),
        "failures": failures,
    }
    if failures:
        return CheckResult(
            CHECK_ID,
            CHECK_NAME,
            FAIL,
            f"{len(failures)} extension-contract failure(s). First: {failures[0]}",
            evidence,
            artifacts,
        )
    return CheckResult(
        CHECK_ID,
        CHECK_NAME,
        PASS,
        f"{len(source_models)} registered model(s) and {len(catalog.entries)} catalog entries satisfy the contract.",
        evidence,
        artifacts,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args(argv)
    result = run_check(CHECK_ID, CHECK_NAME, check, output_dir=args.output_dir)
    print(f"[{result.status}] {result.check_id} {result.name}: {result.message}")
    return 1 if result.status == FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
