"""Backend readiness layer for datasets and models.

One structured answer to two questions the platform, its validators, its
frontend and the future LLM agent all need to ask:

* Can this dataset be used for a run right now?
* Can this model be run on this dataset right now?

Readiness is discovered from the filesystem and the live BatteryML
registries — never assumed, never faked (CLAUDE.md §7). A dataset that only
exists as raw files is reported as ``RAW_ONLY``, not as a failure; a model
whose pretrained assets are missing is reported as ``NEEDS_ASSETS`` with the
exact missing paths. Every report carries a ``recommended_action`` a human
or the agent can act on.

Design constraints (CLAUDE.md §3, §6):

* No Streamlit import: this module must stay usable by the backend, the
  verification harness, the agent and the tests in any environment.
* No heavy imports at module scope. ``batteryml`` is imported lazily through
  :func:`config_validator.registry_status` (which also registers the
  extension models exactly the way a run does), so a machine without torch
  still gets dataset readiness and external-model readiness.
* Third-party sources are never modified or executed here: external-model
  readiness inspects files and conda environments only. Execution stays with
  ``batteryml_web/external_runners`` and the CLI wrapper.

Agent-facing entry point: :func:`get_workflow_context`.
"""

from __future__ import annotations

import shutil
import subprocess
import zipfile
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Sequence

from .config_registry import ConfigRegistry, get_config_registry
from .config_validator import load_and_validate_config_file, registry_status

#: BatteryData files preprocessing writes into ``data/processed/<DATASET>``.
PROCESSED_GLOB = "*.pkl"

#: Task labels used across the platform. Registry-model tasks come from the
#: config's label annotator (via ConfigRegistry); external models declare
#: theirs here, in ``EXTERNAL_MODELS``.
TASK_RUL = "RUL prediction"
TASK_SOH = "SOH estimation"
TASK_TRAJECTORY = "Degradation trajectory forecasting"

#: Datasets the platform declares, whether or not anything is on disk yet.
#: A declared dataset with no local data reports MISSING — that is a state,
#: not an error. Unioned with whatever exists under data/raw and
#: data/processed by :func:`list_dataset_readiness`.
PLATFORM_DATASETS: tuple[str, ...] = (
    "CALCE",
    "HNEI",
    "HUST",
    "ISU_ILCC",
    "MATR",
    "MICH",
    "MICH_EXP",
    "RWTH",
    "SDU",
    "SNL",
    "Stanford_2",
    "Tongji",
    "UL_PUR",
    "XJTU",
)

#: Where the BatteryLife v11 distribution is unpacked, relative to the repo
#: root. Already the platform's convention: the external-model readiness path
#: below names these same archives, and the agent's RAW_ONLY notice quotes
#: this path.
BATTERYLIFE_RAW_PARTS: tuple[str, ...] = ("data", "raw", "BatteryLife", "v11")

def _native_preprocessor_datasets() -> frozenset[str]:
    """Datasets BatteryML has a registered preprocessor for.

    Read from ``batteryml.preprocess.SUPPORTED_SOURCES`` rather than listed
    here, so this cannot drift from what the CLI can actually preprocess.
    Falls back to an empty set when batteryml cannot be imported — readiness
    must keep working on a machine without torch (module docstring).
    """
    try:
        from batteryml.preprocess import SUPPORTED_SOURCES
    except Exception:  # noqa: BLE001 - readiness degrades, never fails
        return frozenset()
    return frozenset(str(name).upper() for name in SUPPORTED_SOURCES.get("DATASETS", ()))


@dataclass(frozen=True)
class RawLayout:
    """One recognised on-disk shape for a dataset's raw data.

    A layout is only *recognised* when its signature holds — the directory
    existing is never enough. ``preprocessable`` says whether BatteryML's own
    preprocessing route can consume this layout; a layout that is merely
    detected reports RAW_ONLY and emits no preprocess command (CLAUDE.md §7
    and the rule that downloaded is not runnable).
    """

    #: "native" (data/raw/<NAME>) or "batterylife" (the v11 distribution).
    kind: str
    #: The path the signature was found at, for the details panel.
    path: Path
    #: How many regular files prove it.
    file_count: int
    #: Whether BatteryML can preprocess *this* layout.
    preprocessable: bool
    #: One line naming what was found, for the reason string.
    description: str


def _archive_layout(path: Path, *, kind: str) -> RawLayout | None:
    """A readable ZIP archive, or None.

    The signature is ``zipfile.is_zipfile``, not the ``.zip`` suffix: a
    truncated or half-downloaded file carries the name but not the data, and
    must not be reported as present.
    """
    if not path.is_file() or not zipfile.is_zipfile(path):
        return None
    return RawLayout(
        kind=kind,
        path=path,
        file_count=1,
        preprocessable=False,
        description=f"archive {path.name}",
    )


def _directory_layout(
    path: Path, *, kind: str, preprocessable: bool
) -> RawLayout | None:
    """A directory holding at least one regular file, or None.

    Deliberately *non-empty*, not merely present: an empty
    ``data/raw/<NAME>`` is the state left behind by a failed or abandoned
    download, and reporting it as raw data available would send a user to
    preprocess nothing. The count is over regular files, so a tree of empty
    subdirectories does not qualify either.

    Deliberately **not** a content contract. ``third_party/BatteryLife`` is
    absent from this checkout, so the file names inside an extracted v11
    dataset cannot be proven here; requiring a guessed name would trade this
    bug for a stricter one. Non-empty is what can be established, and it is
    enough to tell "downloaded" from "not downloaded".
    """
    if not path.is_dir():
        return None
    count = _dir_file_count(path)
    if count == 0:
        return None
    return RawLayout(
        kind=kind,
        path=path,
        file_count=count,
        preprocessable=preprocessable,
        description=f"{count} file(s) in {_display_path(path)}",
    )


def _recognised_raw_layouts(
    name: str,
    *,
    root: Path,
    raw_root: str | Path | None = None,
) -> list[RawLayout]:
    """Every recognised raw layout for ``name``, best first.

    Three shapes are recognised, and the order is the order of usefulness:

    1. ``<raw_root>/<NAME>/`` — BatteryML's native raw directory. Only this
       one is ever ``preprocessable``, and only for a dataset BatteryML has
       a registered preprocessor for.
    2. ``data/raw/BatteryLife/v11/<NAME>/`` — the v11 archive **extracted**,
       which is the state the external-model readiness path already requires
       (it calls ``_dir_has_file`` on ``root_path``).
    3. ``data/raw/BatteryLife/v11/<NAME>.zip`` — the archive as downloaded.

    Shape 2 is what this function was added for. The previous code
    recognised only shape 3, and only for a hardcoded three of the seven
    BatteryLife datasets this platform declares, so a dataset the user had
    downloaded *and extracted* — the normal end state — reported MISSING,
    as did one whose archive was simply not on that list.
    """
    layouts: list[RawLayout] = []

    native_dir = (
        Path(raw_root) / name if raw_root else root / "data" / "raw" / name
    )
    native = _directory_layout(
        native_dir,
        kind="native",
        preprocessable=name.upper() in _native_preprocessor_datasets(),
    )
    if native is not None:
        layouts.append(native)

    # The BatteryLife distribution lives at a fixed repository path; it is
    # not relative to a caller-supplied raw_root, and it is checked whether
    # or not one was given. Gating it on "raw_root is None" is what made the
    # detection vanish for any caller that named its own raw directory.
    v11 = root.joinpath(*BATTERYLIFE_RAW_PARTS)
    extracted = _directory_layout(
        v11 / name, kind="batterylife", preprocessable=False
    )
    if extracted is not None:
        layouts.append(extracted)
    archive = _archive_layout(v11 / f"{name}.zip", kind="batterylife")
    if archive is not None:
        layouts.append(archive)
    return layouts


#: External (non-BatteryML-pipeline) models and the runner that owns them.
#: ``task`` is fixed by the model's actual output semantics: PBT predicts a
#: scalar lifetime; BatteryMFormer forecasts a future SOH/capacity trajectory
#: and must never be presented as a scalar RUL model.
EXTERNAL_MODELS: dict[str, dict[str, str]] = {
    "PBT": {"task": TASK_RUL, "runner": "PBTRunner"},
    "BatteryMFormer": {"task": TASK_TRAJECTORY, "runner": "BatteryMFormerRunner"},
}

_DEFAULT_RECOMMENDED_ACTION = {
    "READY": "Dataset is ready for model execution.",
    "RAW_ONLY": "Run preprocessing before model execution.",
    "INCOMPLETE_PROCESSED": (
        "Processed directory exists but holds no usable "
        f"{PROCESSED_GLOB} files. Run preprocessing, or remove the empty "
        "directory so it cannot satisfy a mere existence assert."
    ),
    "MISSING": (
        "Dataset is not available locally. Select another dataset, or place "
        "raw data under data/raw/<DATASET>/ and preprocess it."
    ),
}


class DatasetStatus(str, Enum):
    READY = "READY"
    RAW_ONLY = "RAW_ONLY"
    INCOMPLETE_PROCESSED = "INCOMPLETE_PROCESSED"
    MISSING = "MISSING"


class ModelStatus(str, Enum):
    READY = "READY"
    NEEDS_ASSETS = "NEEDS_ASSETS"
    NEEDS_DATA = "NEEDS_DATA"
    DEPENDENCY_UNAVAILABLE = "DEPENDENCY_UNAVAILABLE"
    UNREGISTERED = "UNREGISTERED"
    CONFIG_INVALID = "CONFIG_INVALID"


@dataclass(frozen=True)
class DatasetReadiness:
    dataset: str
    status: DatasetStatus
    raw_available: bool
    processed_available: bool
    processed_sample_count: int
    reason: str
    recommended_action: str
    raw_path: str
    processed_path: str
    preprocessing_supported: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "status": self.status.value,
            "raw_available": self.raw_available,
            "processed_available": self.processed_available,
            "processed_sample_count": self.processed_sample_count,
            "reason": self.reason,
            "recommended_action": self.recommended_action,
            "raw_path": self.raw_path,
            "processed_path": self.processed_path,
            "preprocessing_supported": self.preprocessing_supported,
        }


@dataclass(frozen=True)
class ModelReadiness:
    model: str
    model_display: str
    task: str
    registered: bool
    ready: bool
    status: ModelStatus
    missing: tuple[str, ...]
    reason: str
    recommended_action: str
    config_paths: tuple[str, ...] = ()
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "model_display": self.model_display,
            "task": self.task,
            "registered": self.registered,
            "ready": self.ready,
            "status": self.status.value,
            "missing": list(self.missing),
            "reason": self.reason,
            "recommended_action": self.recommended_action,
            "config_paths": list(self.config_paths),
            "details": dict(self.details),
        }


def _dir_file_count(path: Path, pattern: str = "*") -> int:
    if not path.is_dir():
        return 0
    return sum(1 for entry in path.rglob(pattern) if entry.is_file())


def _dir_has_file(path: Path, pattern: str = "*") -> bool:
    return any(entry.is_file() for entry in path.rglob(pattern)) if path.is_dir() else False



def assess_dataset(
    dataset: str,
    *,
    raw_root: str | Path | None = None,
    processed_root: str | Path | None = None,
    processed_glob: str = PROCESSED_GLOB,
) -> DatasetReadiness:
    """Report one dataset's readiness from the filesystem alone."""

    root = _repo_root()
    name = str(dataset).strip()

    if not name:
        raise ValueError("dataset name must be a non-empty string.")

    raw_dir = (
        Path(raw_root) / name
        if raw_root
        else root / "data" / "raw" / name
    )
    processed_dir = (
        Path(processed_root) / name
        if processed_root
        else root / "data" / "processed" / name
    )

    # Every recognised raw shape for this dataset, best first: the native
    # raw directory, the extracted BatteryLife v11 directory, then the v11
    # archive. A layout counts only when its signature holds — an empty
    # directory and a half-downloaded archive are both "not here".
    layouts = _recognised_raw_layouts(name, root=root, raw_root=raw_root)
    best = layouts[0] if layouts else None

    raw_path = best.path if best is not None else raw_dir
    raw_count = best.file_count if best is not None else 0
    uses_alternate_raw_source = best is not None and best.kind != "native"

    # A file that carries the archive's name but is not a readable ZIP is
    # worth saying out loud: it is the shape of an interrupted download, and
    # it is not the same finding as "nothing was ever downloaded".
    v11_archive = root.joinpath(*BATTERYLIFE_RAW_PARTS) / f"{name}.zip"
    invalid_alt_archive: Path | None = (
        v11_archive
        if v11_archive.is_file() and not zipfile.is_zipfile(v11_archive)
        else None
    )

    processed_count = _dir_file_count(
        processed_dir,
        processed_glob,
    )

    raw_dir_exists = raw_dir.is_dir()
    processed_dir_exists = processed_dir.is_dir()

    raw_available = raw_count > 0
    processed_available = processed_count > 0

    if processed_available:
        status = DatasetStatus.READY
        reason = (
            f"{processed_count} usable {processed_glob} file(s) in "
            f"{_display_path(processed_dir)}"
        )

    elif processed_dir_exists:
        status = DatasetStatus.INCOMPLETE_PROCESSED

        detail = (
            f"with raw data present at {_display_path(raw_path)}"
            if raw_available
            else "and no raw data available"
        )

        reason = (
            f"Processed directory {_display_path(processed_dir)} exists but "
            f"holds no usable {processed_glob} file, {detail}. "
            "An empty processed directory satisfies BaseTrainTestSplitter's "
            "existence assert and would yield an empty cell list."
        )

    elif raw_available:
        status = DatasetStatus.RAW_ONLY

        reason = (
            f"{raw_count} raw source(s) at {_display_path(raw_path)} "
            f"but no usable processed {processed_glob} output"
        )

    else:
        status = DatasetStatus.MISSING
        existing = []

        if raw_dir_exists:
            existing.append(
                f"raw directory {raw_dir} is empty"
            )

        if processed_dir_exists:
            existing.append(
                f"processed directory {processed_dir} is empty"
            )

        if invalid_alt_archive is not None:
            existing.append(
                f"alternate raw archive {invalid_alt_archive} exists "
                "but is not a readable ZIP"
            )

        reason = (
            "No local data found"
            + (
                f" ({'; '.join(existing)})"
                if existing
                else ""
            )
        )

    return DatasetReadiness(
        dataset=name,
        status=status,
        raw_available=raw_available,
        processed_available=processed_available,
        processed_sample_count=processed_count,
        reason=reason,
        recommended_action=_DEFAULT_RECOMMENDED_ACTION[
            status.value
        ],
        raw_path=str(raw_path),
        processed_path=str(processed_dir),
        # Only a raw layout BatteryML can actually preprocess earns this.
        # A BatteryLife v11 dataset — extracted or archived — is detected but
        # has no BatteryML preprocessor, so it stays False and no preprocess
        # command is ever offered for it. Detected is not runnable.
        preprocessing_supported=(
            status is DatasetStatus.RAW_ONLY
            and best is not None
            and best.preprocessable
        ),
    )

def list_dataset_readiness(
    datasets: Sequence[str] | None = None,
    *,
    raw_root: str | Path | None = None,
    processed_root: str | Path | None = None,
) -> list[DatasetReadiness]:
    """Readiness for the declared platform datasets plus anything on disk.

    When ``datasets`` is omitted the list is the union of
    :data:`PLATFORM_DATASETS` and the directory names found under
    ``data/raw`` and ``data/processed``, so nothing present locally can be
    silently left out.
    """
    if datasets is None:
        root = _repo_root()
        raw_dir = Path(raw_root) if raw_root else root / "data" / "raw"
        processed_dir = Path(processed_root) if processed_root else root / "data" / "processed"
        on_disk = {
            entry.name
            for entry in list(raw_dir.glob("*")) + list(processed_dir.glob("*"))
            if entry.is_dir() and not entry.name.startswith("_")
        }
        datasets = sorted(set(PLATFORM_DATASETS) | on_disk)
    return [assess_dataset(name, raw_root=raw_root, processed_root=processed_root) for name in datasets]


def assess_registry_model(
    model_key: str,
    *,
    dataset: str | None = None,
    task: str | None = None,
    repo_root: str | Path | None = None,
    registry: ConfigRegistry | None = None,
    split_hint: str | None = None,
) -> ModelReadiness:
    """Readiness of one config-backed model from the live registries.

    A model is READY for a (dataset, task) pair when at least one of its
    configs for that dataset passes the deterministic validator
    (``config_validator.load_and_validate_config_file`` with data required) —
    the same gate a run goes through. Registration is checked against the
    live ``MODELS`` registry populated the way ``run_batteryml_cli.py``
    populates it. A model whose configs all target other datasets is reported
    as NEEDS_DATA for the requested dataset, never as READY.
    """
    root = Path(repo_root) if repo_root else _repo_root()
    config_reg = registry if registry is not None else get_config_registry(str(root))

    all_for_model = [e for e in config_reg.get_entries(dataset=None, task=task) if e.model_key == model_key]
    entries = [e for e in all_for_model if dataset is None or e.dataset == dataset]

    display = next((e.model_display for e in entries or all_for_model), model_key)
    tasks = sorted({e.task for e in all_for_model}) or ([task] if task else [])

    if not all_for_model:
        return ModelReadiness(
            model=model_key,
            model_display=model_key,
            task=task or (tasks[0] if tasks else TASK_RUL),
            registered=False,
            ready=False,
            status=ModelStatus.UNREGISTERED,
            missing=(f"no run config under configs/ declares model key '{model_key}'",),
            reason=(
                f"No config under configs/ resolves to model key '{model_key}', so "
                "the model cannot be built by the pipeline."
            ),
            recommended_action=(
                "Add a validated config for this model (see docs/EXTENDING_MODELS.md) "
                "before it can be selected."
            ),
        )

    model_names = sorted({str((e.metadata.get("yaml") or {}).get("model", {}).get("name", "")).strip() for e in all_for_model} - {""})

    registries, registry_error = registry_status()
    if registry_error is not None:
        return ModelReadiness(
            model=model_key,
            model_display=display,
            task=tasks[0] if tasks else TASK_RUL,
            registered=False,
            ready=False,
            status=ModelStatus.DEPENDENCY_UNAVAILABLE,
            missing=(registry_error,),
            reason="The BatteryML registries could not be imported, so no model can be built.",
            recommended_action=(
                "Install the dependencies (`pip install -r requirements.txt`) and re-run; "
                "see the message in `missing` for the exact import failure."
            ),
            config_paths=tuple(e.config_path.as_posix() for e in all_for_model),
        )

    models_registry = registries.get("MODELS", frozenset())
    registered_names = [name for name in model_names if name in models_registry]
    registered = bool(registered_names)

    if not entries:
        # The model exists and may run elsewhere, but nothing is wired for the
        # requested dataset. Reporting READY here would let the agent offer a
        # run that has no config and no data behind it.
        configured_for = sorted({e.dataset for e in all_for_model})
        return ModelReadiness(
            model=model_key,
            model_display=display,
            task=tasks[0] if tasks else TASK_RUL,
            registered=registered,
            ready=False,
            status=ModelStatus.NEEDS_DATA,
            missing=(
                f"no run config declares dataset '{dataset}' for model '{model_key}' "
                f"(configured only for: {', '.join(configured_for)})",
            ),
            reason=(
                f"'{model_key}' has no config for dataset '{dataset}'; its validated "
                f"configs target {', '.join(configured_for)}."
            ),
            recommended_action=(
                f"Generate a validated config for '{model_key}' on '{dataset}' and "
                "preprocess that dataset first."
            ),
            details={"configured_for_datasets": configured_for},
        )

    validations = [
        (entry, load_and_validate_config_file(entry.config_path, require_data=True))
        for entry in entries
    ]
    valid = [(entry, report) for entry, report in validations if report.get("valid")]

    if valid and registered:
        from .config_registry import select_config_for_hint

        selection = select_config_for_hint(
            [entry for entry, _ in valid], split_hint
        )
        chosen = selection["entry"]
        by_path = {entry.config_path.as_posix(): report for entry, report in valid}
        if chosen is not None:
            best_entry, best_report = chosen, by_path[chosen.config_path.as_posix()]
            return ModelReadiness(
                model=model_key,
                model_display=display,
                task=tasks[0] if tasks else TASK_RUL,
                registered=True,
                ready=True,
                status=ModelStatus.READY,
                missing=(),
                reason=(
                    f"Registered in MODELS ({', '.join(registered_names)}); config "
                    f"{_display_path(best_entry.config_path)} passes the execution gate "
                    f"with its declared data present ({', '.join(best_report.get('dataset_paths', [])) or 'n/a'})."
                ),
                recommended_action="Select this model's config and run it through the validated workflow.",
                config_paths=tuple(entry.config_path.as_posix() for entry, _ in valid),
                details={
                    "validated_config": best_entry.config_path.as_posix(),
                    "config_selection": selection["status"],
                    "config_choices": [
                        e.config_path.as_posix() for e in selection["choices"]
                    ],
                },
            )
        # Several distinct protocols (or an invalid explicit request): READY
        # as a model, but with no validated config — the user must choose.
        # Silently taking the first list item here is what once resolved
        # "matr_1" to clo.yaml.
        return ModelReadiness(
            model=model_key,
            model_display=display,
            task=tasks[0] if tasks else TASK_RUL,
            registered=True,
            ready=True,
            status=ModelStatus.READY,
            missing=(),
            reason=selection["message"],
            recommended_action=selection["message"],
            config_paths=tuple(entry.config_path.as_posix() for entry, _ in valid),
            details={
                "validated_config": None,
                "config_selection": selection["status"],
                "config_choices": [
                    e.config_path.as_posix() for e in selection["choices"]
                ],
            },
        )

    # Not ready: pick the config with the fewest errors and report honestly.
    entry, report = min(validations, key=lambda item: len(item[1].get("errors", [])))
    errors = tuple(str(e) for e in report.get("errors", []))
    data_errors = [e for e in errors if "cell_data_path" in e or "pkl" in e]
    if not registered:
        status = ModelStatus.UNREGISTERED
        action = (
            "The config's model class is not registered in MODELS. Register it "
            "through batteryml_web/models/ (docs/EXTENDING_MODELS.md) before use."
        )
    elif data_errors and len(data_errors) == len(errors):
        status = ModelStatus.NEEDS_DATA
        action = "Run preprocessing for the declared dataset(s), then re-check readiness."
    elif errors:
        status = ModelStatus.CONFIG_INVALID
        action = "Fix the config errors listed in `missing` before running."
    else:
        status = ModelStatus.NEEDS_DATA
        action = "Provide the missing inputs listed in `missing` before running."

    return ModelReadiness(
        model=model_key,
        model_display=display,
        task=tasks[0] if tasks else TASK_RUL,
        registered=registered,
        ready=False,
        status=status,
        missing=errors or (f"no runnable config for model '{model_key}'",),
        reason=(
            f"No config for '{model_key}' passes the execution gate"
            + (f" for dataset '{dataset}'" if dataset else "")
            + "."
        ),
        recommended_action=action,
        config_paths=tuple(e.config_path.as_posix() for e in entries),
    )


def _pbt_environment_available(environment: str = "pbt311") -> tuple[bool, str | None]:
    """Whether conda and the named environment exist. Never executes the env."""
    conda = shutil.which("conda")
    if conda is None:
        return False, "conda executable not found on PATH (required for the pbt311 environment)."
    try:
        listing = subprocess.run(
            [conda, "env", "list"],
            capture_output=True,
            text=True,
            check=False,
            timeout=60,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return False, f"could not list conda environments: {type(exc).__name__}: {exc}"
    if listing.returncode != 0:
        return False, f"`conda env list` failed: {listing.stderr.strip()}"
    for line in listing.stdout.splitlines():
        tokens = line.split()
        if tokens and tokens[0] == environment:
            return True, None
    return False, f"conda environment '{environment}' not found in `conda env list`."



def _discovered_pbt_config(
    source_root: Path,
) -> dict[str, Any]:
    """Discover real local PBT assets without assuming author paths."""

    assets = source_root / "pretrained_assets"
    config: dict[str, Any] = {"assets": {}}

    # Actual Zenodo archive extraction location.
    extracted = assets / "checkpoints" / "PBTs"

    valid_runs: list[Path] = []

    if extracted.is_dir():
        valid_runs = sorted(
            run
            for run in extracted.iterdir()
            if run.is_dir()
            and (run / "args.json").is_file()
            and (run / "model.safetensors").is_file()
        )

    if valid_runs:
        config["assets"]["pretrained_checkpoint"] = str(
            extracted
        )
        config["assets"]["args_path"] = str(
            valid_runs[0] / "args.json"
        )

        scalers = [
            run / "label_scaler"
            for run in valid_runs
            if (run / "label_scaler").is_file()
        ]

        if scalers:
            config["assets"]["label_scaler"] = str(
                scalers[0]
            )

    pca = sorted(
        source_root.glob("**/*pca*.pkl")
    )

    if pca:
        config["assets"]["pca_path"] = str(pca[0])

    embeddings = assets / "condition_embeddings"

    required_embeddings = (
        "training_DKP_embed_all_Llama.pkl",
        "validation_DKP_embed_all_Llama.pkl",
        "testing_DKP_embed_all_Llama.pkl",
    )

    if (
        embeddings.is_dir()
        and all(
            (embeddings / filename).is_file()
            for filename in required_embeddings
        )
    ):
        config["assets"][
            "aging_condition_embeddings"
        ] = str(embeddings)

    # A PBT-compatible MIX_large dataset has not yet been prepared.
    candidate = source_root / "dataset"

    if candidate.is_dir() and _dir_has_file(candidate):
        config["dataset_path"] = str(candidate)

    return config


def _pbt_asset_missing(
    source_root: Path,
    config: Mapping[str, Any],
) -> tuple[list[str], list[str]]:
    """Return real missing PBT runtime inputs."""

    missing: list[str] = []
    notes: list[str] = []

    assets_cfg = config.get("assets", {})

    if not assets_cfg.get("pretrained_checkpoint"):
        missing.append(
            "pretrained checkpoint with args.json + "
            "model.safetensors under "
            "third_party/PBT/pretrained_assets/"
            "checkpoints/PBTs/"
        )

    if not assets_cfg.get("label_scaler"):
        missing.append(
            "checkpoint label_scaler"
        )

    if not assets_cfg.get("pca_path"):
        missing.append(
            "PCA artifact "
            "(MIX_large_pca_0.99_Llama.pkl); "
            "this is distinct from checkpoint label_scaler"
        )

    if not assets_cfg.get(
        "aging_condition_embeddings"
    ):
        missing.append(
            "precomputed aging-condition embeddings: "
            "training/validation/testing_DKP_embed_all_Llama.pkl"
        )

    if not assets_cfg.get("args_path"):
        notes.append(
            "checkpoint args.json unavailable; "
            "run arguments must be supplied explicitly"
        )

    # Source inspection shows PBT inference constructs AutoConfig and
    # AutoTokenizer from LLM_path, while the actual prompt embeddings are
    # loaded from precomputed pickle files. Full LLM weights have not been
    # demonstrated as a runtime requirement.
    if not assets_cfg.get("LLM_path"):
        missing.append(
            "local HuggingFace config/tokenizer metadata "
            "for LLM_path; full LLM weight files are not "
            "currently demonstrated as required by PBT inference"
        )

    if not config.get("dataset_path"):
        missing.append(
            "PBT-compatible BatteryLife MIX_large dataset layout; "
            "BatteryML data/processed/*.pkl must not be assumed compatible"
        )

    return missing, notes

def assess_external_model(
    name: str,
    *,
    repo_root: str | Path | None = None,
    config: Mapping[str, Any] | None = None,
    check_environment: bool = True,
    probe_environment: bool = False,
) -> ModelReadiness:
    """Readiness of an external (non-pipeline) model: PBT, BatteryMFormer.

    Filesystem inspection only. Third-party sources are read, never modified
    and never executed; with ``probe_environment=True`` the PBT import is
    additionally probed inside its conda environment through
    :class:`external_runners.pbt.PBTRunner`.
    """
    root = Path(repo_root) if repo_root else _repo_root()
    spec = EXTERNAL_MODELS.get(name)
    if spec is None:
        raise ValueError(
            f"Unknown external model {name!r}. Known: {', '.join(sorted(EXTERNAL_MODELS))}. "
            "For BatteryML-pipeline models use assess_registry_model()."
        )
    task = spec["task"]

    if name == "PBT":
        source_root = root / "third_party" / "PBT"
        discovered = _discovered_pbt_config(source_root)
        merged = {**discovered, **(dict(config) if config else {})}
        if config and "assets" in config:
            merged["assets"] = {**discovered.get("assets", {}), **config["assets"]}

        missing: list[str] = []
        source_files = ("models/PBT.py", "run_main.py", "finetune_model.py", "evaluate_model.py")
        for relative in source_files:
            if not (source_root / relative).is_file():
                missing.append(f"third_party/PBT/{relative}")

        env_error: str | None = None
        if check_environment:
            env_ok, env_error = _pbt_environment_available()
            if not env_ok:
                missing.append(env_error)
            elif probe_environment:
                from ..external_runners import PBTRunner

                probe = PBTRunner(root, merged).readiness_check()
                for item in probe.get("missing", []):
                    if "import PBT" in str(item):
                        missing.append(str(item))

        asset_missing, notes = _pbt_asset_missing(source_root, merged)
        missing.extend(asset_missing)

        if missing:
            if env_error is not None:
                status = ModelStatus.DEPENDENCY_UNAVAILABLE
            elif any("checkpoint" in item or "PCA" in item or "LLM checkpoint" in item for item in asset_missing):
                status = ModelStatus.NEEDS_ASSETS
            elif any("dataset" in item for item in asset_missing):
                status = ModelStatus.NEEDS_DATA
            else:
                status = ModelStatus.NEEDS_ASSETS
        else:
            status = ModelStatus.READY

        return ModelReadiness(
            model=name,
            model_display=name,
            task=task,
            registered=True,
            ready=not missing,
            status=status,
            missing=tuple(missing),
            reason=(
                "Source and isolated environment are available; no pretrained "
                "assets are missing."
                if not missing
                else f"{len(missing)} required input(s) missing for PBT execution."
            ),
            recommended_action=(
                "Select this model and run it through its external runner."
                if not missing
                else "Provide the pretrained checkpoint and required assets before execution; "
                "the source in third_party/PBT must not be modified to fake readiness."
            ),
            details={"source_root": str(source_root), "environment": "pbt311", "notes": notes},
        )

    if name == "BatteryMFormer":
        source_root = root / "third_party" / "BatteryMFormer"
        missing = []
        for relative in ("models/BatteryMFormer.py", "run_main.py"):
            if not (source_root / relative).is_file():
                missing.append(f"third_party/BatteryMFormer/{relative}")
        if shutil.which("python") is None and shutil.which("python3") is None:
            missing.append("python executable")

        discovered: dict[str, Any] = {}
        prompt_embeddings = source_root / "data_provider" / "prompt_embeddings" / "Qwen3_total.pkl"
        if prompt_embeddings.is_file():
            discovered["prompt_embeddings_path"] = str(prompt_embeddings)
        else:
            missing.append(f"prompt embedding pickle: {prompt_embeddings}")
        split_json = source_root / "data_provider" / "split_json" / "pure_ood" / "Liion_split_seed2024.json"
        if split_json.is_file():
            discovered["split_json_path"] = str(split_json)
        else:
            missing.append(f"evaluation split file: {split_json}")

        merged = {**discovered, **(dict(config) if config else {})}
        root_path = merged.get("root_path")
        root_path_ok = bool(root_path) and _dir_has_file(Path(str(root_path)), pattern="*")
        processed_soh = merged.get("processed_SOH_path")
        processed_soh_ok = bool(processed_soh) and _dir_has_file(Path(str(processed_soh)))
        if not root_path_ok:
            missing.append(
                "BatteryLife v11 dataset directory (root_path). It exists only as "
                "unextracted archives under data/raw/BatteryLife/v11/ "
                "(CALCE.zip, Life labels.zip, MICH.zip, MICH_EXP.zip, Stanford_2.zip)."
            )
        if not processed_soh_ok:
            missing.append(
                "processed_SOH directory with processed SOH series "
                "(processed_SOH_path) — not present anywhere in the repository."
            )

        status = ModelStatus.READY if not missing else ModelStatus.NEEDS_DATA
        return ModelReadiness(
            model=name,
            model_display=name,
            task=task,
            registered=True,
            ready=not missing,
            status=status,
            missing=tuple(missing),
            reason=(
                "Source, prompt embeddings and split files are available; dataset "
                "inputs are missing."
                if missing
                else "All declared inputs are available."
            ),
            recommended_action=(
                "Select this model for degradation trajectory forecasting."
                if not missing
                else "Unpack/prepare the BatteryLife v11 data and processed_SOH "
                "directory before execution; do not substitute another dataset."
            ),
            details={"source_root": str(source_root)},
        )

    raise ValueError(f"External model {name!r} has no readiness implementation.")


def assess_model(
    name: str,
    *,
    dataset: str | None = None,
    task: str | None = None,
    repo_root: str | Path | None = None,
    registry: ConfigRegistry | None = None,
) -> ModelReadiness:
    """Readiness of any known model, dispatching to the right backend."""
    token = str(name).strip()
    if not token:
        raise ValueError("model name must be a non-empty string.")
    if token in EXTERNAL_MODELS:
        return assess_external_model(token, repo_root=repo_root)
    for external in EXTERNAL_MODELS:
        if token.lower() == external.lower():
            return assess_external_model(external, repo_root=repo_root)

    from .config_registry import MODEL_CLASS_KEYS, MODEL_KEY_ALIASES, normalize_model_token

    normalized = normalize_model_token(token)
    model_key = MODEL_KEY_ALIASES.get(normalized)
    if model_key is None:
        model_key = MODEL_CLASS_KEYS.get(normalized.replace(" ", "").lower())
    if model_key is None:
        model_key = token.replace(" ", "_").lower()
    return assess_registry_model(
        model_key,
        dataset=dataset,
        task=task,
        repo_root=repo_root,
        registry=registry,
    )


def list_model_readiness(
    *,
    dataset: str | None = None,
    task: str | None = None,
    repo_root: str | Path | None = None,
    include_external: bool = True,
) -> list[ModelReadiness]:
    """Readiness for every known model (config-backed plus external)."""
    root = Path(repo_root) if repo_root else _repo_root()
    config_reg = get_config_registry(str(root))
    model_keys = sorted({entry.model_key for entry in config_reg.entries})
    reports = [
        assess_registry_model(key, dataset=dataset, task=task, repo_root=root, registry=config_reg)
        for key in model_keys
    ]
    if include_external:
        reports.extend(
            assess_external_model(name, repo_root=root) for name in sorted(EXTERNAL_MODELS)
        )
    return reports


def _normalize_task_label(task: str | None) -> str:
    """Fold task spellings onto the three platform tasks."""
    if task is None:
        return TASK_RUL
    text = str(task).strip().lower()
    if not text:
        return TASK_RUL
    if text.startswith("rul"):
        return TASK_RUL
    if text.startswith("soh"):
        return TASK_SOH
    if "trajectory" in text or "forecast" in text:
        return TASK_TRAJECTORY
    raise ValueError(
        f"Unsupported task {task!r}. Supported: {TASK_RUL!r}, {TASK_SOH!r}, "
        f"{TASK_TRAJECTORY!r} (case-insensitive prefixes 'rul', 'soh', 'trajectory')."
    )


def get_workflow_context(
    dataset: str | None = None,
    task: str | None = None,
    *,
    repo_root: str | Path | None = None,
    split_hint: str | None = None,
) -> dict[str, Any]:
    """The structured state the LLM agent must consult instead of guessing.

    The agent reads this; it never decides from filenames whether data or
    assets exist, and it never manufactures a model or a prediction.

    Returns a dict with:

    * ``dataset`` — :meth:`DatasetReadiness.to_dict` for the dataset
      (default ``MATR``, the primary worked example);
    * ``task`` — the normalized task label;
    * ``available_models`` — READY models whose task matches;
    * ``blocked_models`` — known models for the task that are not READY,
      each with its missing inputs and recommended action;
    * ``recommended_next_steps`` — ordered, actionable strings.
    """
    root = Path(repo_root) if repo_root else _repo_root()
    normalized_task = _normalize_task_label(task)
    dataset_name = str(dataset).strip() if dataset else "MATR"
    if not dataset_name:
        raise ValueError("dataset must be a non-empty string when provided.")

    data_readiness = assess_dataset(
        dataset_name,
        raw_root=(
            (root / "data" / "raw")
            if repo_root is not None
            else None
        ),
        processed_root=(
            (root / "data" / "processed")
            if repo_root is not None
            else None
        ),
    )

    config_reg = get_config_registry(str(root))
    reports: list[ModelReadiness] = []
    if normalized_task in (TASK_RUL, TASK_SOH):
        # Registry models are scalar RUL/SOH predictors by construction (their
        # configs declare RULLabelAnnotator / SOHLabelAnnotator). Asking them
        # about trajectory forecasting would only produce misleading
        # UNREGISTERED rows.
        model_keys = sorted({entry.model_key for entry in config_reg.entries})
        reports.extend(
            assess_registry_model(key, dataset=dataset_name, task=normalized_task, repo_root=root, registry=config_reg, split_hint=split_hint)
            for key in model_keys
        )
    reports.extend(
        assess_external_model(name, repo_root=root)
        for name in sorted(EXTERNAL_MODELS)
        if EXTERNAL_MODELS[name]["task"] == normalized_task
    )

    available = [r.to_dict() for r in reports if r.ready and r.status is ModelStatus.READY]
    blocked = [r.to_dict() for r in reports if not (r.ready and r.status is ModelStatus.READY)]

    steps: list[str] = []
    if data_readiness.status is DatasetStatus.READY:
        if available:
            steps.append(
                f"Dataset '{dataset_name}' is READY. Choose one of the available "
                f"models ({', '.join(item['model_display'] for item in available)}) and "
                "one of its validated configs."
            )
        else:
            steps.append(
                f"Dataset '{dataset_name}' is READY but no model for task "
                f"'{normalized_task}' is executable yet."
            )
    elif data_readiness.status is DatasetStatus.RAW_ONLY:
        if data_readiness.preprocessing_supported:
            steps.append(
                f"Dataset '{dataset_name}' is RAW_ONLY: run preprocessing "
                "(python batteryml_web/core/run_batteryml_cli.py preprocess "
                f"{dataset_name} data/raw/{dataset_name} data/processed/{dataset_name}) "
                "before any model run."
            )
        else:
            steps.append(
                f"Dataset '{dataset_name}' is RAW_ONLY: BatteryLife v11 "
                f"{dataset_name} data are present locally "
                f"({_display_path(Path(data_readiness.raw_path))}), but this "
                "archive is not yet wired to the BatteryML preprocessing "
                "route. Prepare/unpack an approved BatteryML-compatible "
                f"{dataset_name} input before model execution."
            )
    elif data_readiness.status is DatasetStatus.INCOMPLETE_PROCESSED:
        steps.append(
            f"Dataset '{dataset_name}' has a processed directory without usable "
            f"{PROCESSED_GLOB} files: run preprocessing, or remove the empty "
            "directory so it cannot pass a mere existence check."
        )
    else:
        steps.append(
            f"Dataset '{dataset_name}' is MISSING locally: select a READY dataset "
            f"such as MATR, or place raw data under data/raw/{dataset_name}/ and "
            "preprocess it."
        )

    if normalized_task is TASK_TRAJECTORY:
        steps.append(
            "Degradation trajectory forecasting is served by external models only "
            "(BatteryMFormer); it is not a scalar RUL task and must not run through "
            "RULLabelAnnotator."
        )
    external_blocked = [item for item in blocked if item["model"] in EXTERNAL_MODELS]
    other_blocked = [item for item in blocked if item["model"] not in EXTERNAL_MODELS]
    for item in external_blocked + other_blocked[:2]:
        steps.append(f"{item['model_display']} is {item['status']}: {item['recommended_action']}")
    if len(other_blocked) > 2:
        steps.append(
            f"{len(other_blocked) - 2} further known models are blocked for this "
            "dataset/task; see blocked_models for each one's missing inputs."
        )
    if available:
        steps.append(
            "Run the chosen config through the validator "
            "(batteryml_web/core/config_validator) and the CLI wrapper, then read "
            "metrics from the run workspace. The agent explains results; it never "
            "computes or edits them."
        )

    return {
        "dataset": data_readiness.to_dict(),
        "task": normalized_task,
        "available_models": available,
        "blocked_models": blocked,
        "recommended_next_steps": steps,
    }


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _display_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(_repo_root()).as_posix()
    except ValueError:
        return str(path)
