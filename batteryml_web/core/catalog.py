"""Structured catalog and execution-readiness gate for BatteryML methods.

The catalog is derived exclusively from run configs under ``configs/**/*.yaml``.
It does not render UI and has no Streamlit dependency.
"""

from __future__ import annotations

import inspect
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml


class MethodCategory(str, Enum):
    CLASSICAL_ML = "classical_ml"
    NEURAL = "neural"
    PROBABILISTIC = "probabilistic"
    GENERATIVE = "generative"
    TRAJECTORY = "trajectory"


class CatalogStatus(str, Enum):
    RUNNABLE = "runnable"
    IMPORT_FAILED = "import_failed"
    MISSING_SCHEMA_FIELDS = "missing_schema_fields"
    CONFIG_INVALID = "config_invalid"
    NEEDS_PREPROCESSING = "needs_preprocessing"
    MISSING_CHECKPOINT = "missing_checkpoint"
    MISSING_ASSET = "missing_asset"


@dataclass(frozen=True)
class GateFailure:
    status: CatalogStatus
    condition: str
    detail: str
    path: str | None = None


@dataclass(frozen=True)
class CatalogEntry:
    config_path: Path
    model_name: str | None
    feature_name: str | None
    label_name: str | None
    split_name: str | None
    cell_data_paths: tuple[Path, ...]
    datasets: tuple[str, ...]
    dataset: str
    task: str
    device: str
    devices: tuple[str, ...]
    category: MethodCategory
    debug_only: bool
    required_checkpoints: tuple[Path, ...]
    required_assets: tuple[Path, ...]
    failures: tuple[GateFailure, ...] = field(default_factory=tuple)

    @property
    def status(self) -> CatalogStatus:
        return self.failures[0].status if self.failures else CatalogStatus.RUNNABLE

    @property
    def runnable(self) -> bool:
        return not self.failures

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["config_path"] = self.config_path.as_posix()
        value["cell_data_paths"] = [path.as_posix() for path in self.cell_data_paths]
        value["required_checkpoints"] = [path.as_posix() for path in self.required_checkpoints]
        value["required_assets"] = [path.as_posix() for path in self.required_assets]
        value["category"] = self.category.value
        value["status"] = self.status.value
        value["runnable"] = self.runnable
        value["failures"] = [
            {**asdict(item), "status": item.status.value} for item in self.failures
        ]
        return value


@dataclass(frozen=True)
class CatalogIssue:
    config_path: Path
    status: CatalogStatus
    detail: str

    def to_dict(self) -> dict[str, str]:
        return {
            "config_path": self.config_path.as_posix(),
            "status": self.status.value,
            "detail": self.detail,
        }


@dataclass(frozen=True)
class MethodCatalog:
    entries: tuple[CatalogEntry, ...]
    issues: tuple[CatalogIssue, ...] = field(default_factory=tuple)
    registry_error: str | None = None

    def methods(
        self,
        *,
        include_debug: bool = False,
        runnable_only: bool = False,
    ) -> tuple[CatalogEntry, ...]:
        """Return config-derived method entries, excluding debug entries by default."""
        return tuple(
            entry
            for entry in self.entries
            if (include_debug or not entry.debug_only)
            and (not runnable_only or entry.runnable)
        )

    def to_dict(self, *, include_debug: bool = False) -> dict[str, Any]:
        visible = self.methods(include_debug=include_debug)
        return {
            "entries": [entry.to_dict() for entry in visible],
            "issues": [issue.to_dict() for issue in self.issues],
            "registry_error": self.registry_error,
            "counts": {
                "scanned_run_configs": len(self.entries),
                "visible_methods": len(visible),
                "runnable_methods": sum(entry.runnable for entry in visible),
                "debug_methods": sum(entry.debug_only for entry in self.entries),
                "scan_issues": len(self.issues),
            },
        }


_REQUIRED_SECTIONS = ("model", "train_test_split", "feature", "label")
_NON_RUN_ROOTS = {"cyclers"}
_CATEGORY_VALUES = {item.value: item for item in MethodCategory}
_PATH_EXCLUSIONS = {"cell_data_path", "workspace", "output_dir", "config_path"}


def _mapping_name(config: Mapping[str, Any], section: str) -> str | None:
    value = config.get(section)
    if not isinstance(value, Mapping):
        return None
    name = value.get("name")
    return str(name).strip() if isinstance(name, str) and name.strip() else None


def _cell_data_paths(config: Mapping[str, Any]) -> tuple[Path, ...]:
    split = config.get("train_test_split")
    value = split.get("cell_data_path") if isinstance(split, Mapping) else None
    if isinstance(value, str) and value.strip():
        return (Path(value),)
    if isinstance(value, list):
        return tuple(Path(str(item)) for item in value if str(item).strip())
    return ()


def _task(label_name: str | None) -> str:
    upper = (label_name or "").upper()
    if "SOH" in upper:
        return "SOH"
    if "RUL" in upper:
        return "RUL"
    if "SOC" in upper:
        return "SOC"
    return "unknown"


def _required_files(config: Mapping[str, Any]) -> tuple[tuple[Path, ...], tuple[Path, ...]]:
    checkpoints: list[Path] = []
    assets: list[Path] = []

    def visit(value: Any, key: str = "") -> None:
        if isinstance(value, Mapping):
            for child_key, child in value.items():
                visit(child, str(child_key).lower())
            return
        if isinstance(value, list):
            for child in value:
                visit(child, key)
            return
        if not isinstance(value, str) or not value.strip() or key in _PATH_EXCLUSIONS:
            return
        if "checkpoint" in key or key in {"ckpt", "ckpt_to_resume"}:
            checkpoints.append(Path(value))
        elif key in {"external_asset", "external_assets"} or key.endswith("_filepath"):
            assets.append(Path(value))

    visit(config)
    return tuple(dict.fromkeys(checkpoints)), tuple(dict.fromkeys(assets))


def _load_registries() -> tuple[dict[str, dict[str, type]], str | None]:
    try:
        import batteryml  # noqa: F401
        import batteryml.feature  # noqa: F401
        import batteryml.label  # noqa: F401
        import batteryml.train_test_split  # noqa: F401
        import batteryml.preprocess  # noqa: F401
        from batteryml import builders
        from batteryml_web.models import register_extension_models

        register_extension_models()
        registries: dict[str, dict[str, type]] = {}
        for name in ("MODELS", "FEATURE_EXTRACTORS", "LABEL_ANNOTATORS", "TRAIN_TEST_SPLITTERS", "DATA_TRANSFORMATIONS"):
            registry = getattr(builders, name)
            registries[name] = dict(registry.class_mapping)
        return registries, None
    except Exception as exc:  # noqa: BLE001 - retained as a gate failure
        return {}, f"{type(exc).__name__}: {exc}"


def _category(config: Mapping[str, Any], model_class: type | None) -> MethodCategory:
    model = config.get("model")
    declared = model.get("category") if isinstance(model, Mapping) else None
    if declared is None and model_class is not None:
        declared = getattr(model_class, "category", None)
    if declared is not None:
        normalized = str(getattr(declared, "value", declared)).strip().lower()
        if normalized in _CATEGORY_VALUES:
            return _CATEGORY_VALUES[normalized]

    if model_class is not None:
        try:
            from batteryml.models.nn_model import NNModel
            if issubclass(model_class, NNModel):
                return MethodCategory.NEURAL
        except Exception:  # noqa: BLE001 - registry import status is reported separately
            pass
        qualified = f"{model_class.__module__}.{model_class.__name__}".lower()
        if "gaussian_process" in qualified or "probabilistic" in qualified:
            return MethodCategory.PROBABILISTIC
        if "generative" in qualified:
            return MethodCategory.GENERATIVE
        if "trajectory" in qualified:
            return MethodCategory.TRAJECTORY
    return MethodCategory.CLASSICAL_ML


def _devices(model_class: type | None) -> tuple[str, ...]:
    if model_class is None:
        return ("cpu",)
    try:
        from batteryml.models.nn_model import NNModel
        if issubclass(model_class, NNModel):
            return ("cpu", "cuda")
    except Exception:  # noqa: BLE001 - registry import status is reported separately
        pass
    return ("cpu",)


def _debug_only(config: Mapping[str, Any], model_name: str | None, model_class: type | None) -> bool:
    model = config.get("model")
    declared = model.get("debug_only") if isinstance(model, Mapping) else None
    if declared is None and model_class is not None:
        declared = getattr(model_class, "debug_only", None)
    if declared is not None:
        return bool(declared)
    return bool(model_name and model_name.lower().startswith("dummy"))


def _validate_signatures(model_class: type) -> list[str]:
    failures: list[str] = []
    required = {
        "fit": ("self", "dataset", "timestamp"),
        "predict": ("self", "dataset", "data_type"),
        "dump_checkpoint": ("self", "path"),
        "load_checkpoint": ("self", "path"),
        "to": ("self", "device"),
    }
    for method_name, prefix in required.items():
        method = getattr(model_class, method_name, None)
        if not callable(method):
            failures.append(f"model {model_class.__name__} has no callable {method_name}")
            continue
        try:
            actual = tuple(inspect.signature(method).parameters)
        except (TypeError, ValueError) as exc:
            failures.append(f"cannot inspect {model_class.__name__}.{method_name}: {exc}")
            continue
        if actual[: len(prefix)] != prefix:
            failures.append(
                f"{model_class.__name__}.{method_name}{actual} must start with {prefix}"
            )
    return failures


def _structural_failures(config: Mapping[str, Any]) -> list[GateFailure]:
    failures: list[GateFailure] = []
    for section in _REQUIRED_SECTIONS:
        value = config.get(section)
        if not isinstance(value, Mapping):
            failures.append(GateFailure(
                CatalogStatus.MISSING_SCHEMA_FIELDS,
                "config_schema",
                f"missing mapping section `{section}`",
            ))
        elif not _mapping_name(config, section):
            failures.append(GateFailure(
                CatalogStatus.MISSING_SCHEMA_FIELDS,
                "config_schema",
                f"section `{section}` has no non-empty `name`",
            ))
    if not _cell_data_paths(config):
        failures.append(GateFailure(
            CatalogStatus.MISSING_SCHEMA_FIELDS,
            "config_schema",
            "`train_test_split.cell_data_path` must be a path or non-empty path list",
        ))
    return failures


def _entry_from_config(
    path: Path,
    config: Mapping[str, Any],
    registries: Mapping[str, Mapping[str, type]],
    registry_error: str | None,
) -> CatalogEntry:
    model_name = _mapping_name(config, "model")
    feature_name = _mapping_name(config, "feature")
    label_name = _mapping_name(config, "label")
    split_name = _mapping_name(config, "train_test_split")
    declared_paths = _cell_data_paths(config)
    paths = tuple(
        item if item.is_absolute() else path.parents[3] / item
        for item in declared_paths
    )
    datasets = tuple(dict.fromkeys(item.name for item in declared_paths))
    dataset = datasets[0] if len(datasets) == 1 else "+".join(datasets) or "unknown"
    declared_checkpoints, declared_assets = _required_files(config)
    root = path.parents[3]
    checkpoints = tuple(
        item if item.is_absolute() else root / item for item in declared_checkpoints
    )
    assets = tuple(item if item.is_absolute() else root / item for item in declared_assets)
    failures = _structural_failures(config)

    model_class = registries.get("MODELS", {}).get(model_name or "")
    if registry_error:
        failures.append(GateFailure(
            CatalogStatus.IMPORT_FAILED,
            "registry_import",
            f"registries could not be imported: {registry_error}",
        ))
    else:
        resolutions = (
            ("model", model_name, "MODELS"),
            ("feature extractor", feature_name, "FEATURE_EXTRACTORS"),
            ("label annotator", label_name, "LABEL_ANNOTATORS"),
            ("splitter", split_name, "TRAIN_TEST_SPLITTERS"),
        )
        for label, name, registry_name in resolutions:
            if name and name not in registries.get(registry_name, {}):
                failures.append(GateFailure(
                    CatalogStatus.IMPORT_FAILED,
                    f"{label}_registration",
                    f"{name!r} is not registered in {registry_name}",
                ))
        if model_class is not None:
            for detail in _validate_signatures(model_class):
                failures.append(GateFailure(
                    CatalogStatus.CONFIG_INVALID,
                    "model_interface",
                    detail,
                ))

    if not failures:
        try:
            from batteryml_web.core.config_validator import load_and_validate_config_file
            validation = load_and_validate_config_file(path, require_data=False)
        except Exception as exc:  # noqa: BLE001 - retained as a gate failure
            failures.append(GateFailure(
                CatalogStatus.IMPORT_FAILED,
                "config_validator",
                f"validator import or execution failed: {type(exc).__name__}: {exc}",
            ))
        else:
            if not validation.get("valid"):
                failures.append(GateFailure(
                    CatalogStatus.CONFIG_INVALID,
                    "config_validation",
                    "; ".join(str(item) for item in validation.get("errors", [])),
                ))

    for data_path in paths:
        if not data_path.is_dir() or not any(data_path.glob("*.pkl")):
            failures.append(GateFailure(
                CatalogStatus.NEEDS_PREPROCESSING,
                "processed_dataset",
                f"processed dataset is absent or has no *.pkl files: {data_path}",
                data_path.as_posix(),
            ))
    for checkpoint in checkpoints:
        if not checkpoint.is_file():
            failures.append(GateFailure(
                CatalogStatus.MISSING_CHECKPOINT,
                "required_checkpoint",
                f"required checkpoint does not exist: {checkpoint}",
                checkpoint.as_posix(),
            ))
    for asset in assets:
        if not asset.exists():
            failures.append(GateFailure(
                CatalogStatus.MISSING_ASSET,
                "required_external_asset",
                f"required external asset does not exist: {asset}",
                asset.as_posix(),
            ))

    model_section = config.get("model")
    requested_device = "cpu"
    if isinstance(model_section, Mapping) and model_section.get("device"):
        requested_device = str(model_section["device"])
    supported_devices = _devices(model_class)
    if requested_device not in supported_devices:
        failures.append(GateFailure(
            CatalogStatus.CONFIG_INVALID,
            "device_support",
            f"device {requested_device!r} is not supported; expected one of {supported_devices}",
        ))

    return CatalogEntry(
        config_path=path,
        model_name=model_name,
        feature_name=feature_name,
        label_name=label_name,
        split_name=split_name,
        cell_data_paths=paths,
        datasets=datasets,
        dataset=dataset,
        task=_task(label_name),
        device=requested_device,
        devices=supported_devices,
        category=_category(config, model_class),
        debug_only=_debug_only(config, model_name, model_class),
        required_checkpoints=checkpoints,
        required_assets=assets,
        failures=tuple(failures),
    )


def build_catalog(root_dir: Path | str | None = None) -> MethodCatalog:
    """Scan ``configs/**/*.yaml`` and return a structured, non-rendering catalog."""
    root = Path(root_dir) if root_dir is not None else Path(__file__).resolve().parents[2]
    configs_root = root / "configs"
    registries, registry_error = _load_registries()
    entries: list[CatalogEntry] = []
    issues: list[CatalogIssue] = []

    if not configs_root.is_dir():
        return MethodCatalog(
            (),
            (CatalogIssue(configs_root, CatalogStatus.CONFIG_INVALID, "configs directory does not exist"),),
            registry_error,
        )

    for path in sorted(configs_root.rglob("*.yaml")):
        relative_parts = path.relative_to(configs_root).parts
        if relative_parts and relative_parts[0] in _NON_RUN_ROOTS:
            continue
        try:
            loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError) as exc:
            issues.append(CatalogIssue(
                path,
                CatalogStatus.CONFIG_INVALID,
                f"cannot read valid YAML: {type(exc).__name__}: {exc}",
            ))
            continue
        if not isinstance(loaded, Mapping):
            issues.append(CatalogIssue(
                path,
                CatalogStatus.CONFIG_INVALID,
                f"top level is {type(loaded).__name__}, expected mapping",
            ))
            continue
        entries.append(_entry_from_config(path, loaded, registries, registry_error))

    return MethodCatalog(tuple(entries), tuple(issues), registry_error)


# Compatibility entry points for backend callers. No catalog is cached because
# filesystem readiness is part of the result and can change after preprocessing.
def get_catalog(
    root_dir: Path | str | None = None,
    force_rebuild: bool = False,
    include_debug: bool = False,
) -> MethodCatalog:
    del force_rebuild, include_debug
    return build_catalog(root_dir)


def refresh_catalog(
    root_dir: Path | str | None = None,
    include_debug: bool = False,
) -> MethodCatalog:
    del include_debug
    return build_catalog(root_dir)
