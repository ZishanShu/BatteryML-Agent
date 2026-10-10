"""User-facing semantics for a BatteryML YAML configuration.

The execution layer needs the exact values from a config, while the UI needs
to explain those values in plain language.  This module keeps that translation
small, deterministic and independent of Streamlit: it reads only the YAML
mapping (and, when available, a :class:`ConfigEntry`) and returns the dataset,
task and model that a user is about to run.

The helpers deliberately do not inspect arbitrary text in a config.  Dataset
comes from ``train_test_split.cell_data_path``, task from an explicit
``task``/label annotator, and model from ``model.name``.  That avoids the
substring collisions that the registry has already had to fix (for example,
``eol_soh`` is a RUL threshold, not an SOH task).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml

from .config_registry import (
    MODEL_CLASS_KEYS,
    MODEL_DISPLAY_NAMES,
    MODEL_KEY_ALIASES,
    ConfigEntry,
    normalize_model_token,
)
from .readiness import TASK_RUL, TASK_SOH, TASK_TRAJECTORY


_UNKNOWN_DATASET = "Unknown dataset"
_UNKNOWN_TASK = "Unknown task"
_UNKNOWN_MODEL = "Unknown model"


@dataclass(frozen=True)
class ConfigSemantics:
    """The user-readable meaning of a validated or uploaded config.

    ``None`` means the config did not declare that value.  UI code can use the
    ``*_label`` properties when it needs a visible placeholder, while backend
    code can keep the distinction between an absent value and a literal name.
    ``model_key`` is the canonical key used by the execution registry;
    ``model_display`` is the professional name shown to the user.
    """

    dataset: str | None = None
    task: str | None = None
    model_key: str | None = None
    model_display: str | None = None

    @property
    def dataset_label(self) -> str:
        return self.dataset or _UNKNOWN_DATASET

    @property
    def task_label(self) -> str:
        return self.task or _UNKNOWN_TASK

    @property
    def model_label(self) -> str:
        return self.model_display or self.model_key or _UNKNOWN_MODEL

    @property
    def model(self) -> str | None:
        """The readable model value, retained under the short UI name."""

        return self.model_display or self.model_key

    def to_dict(self) -> dict[str, str | None]:
        """Return a serialisable representation suitable for session state."""

        return {
            "dataset": self.dataset,
            "task": self.task,
            "model_key": self.model_key,
            "model_display": self.model_display,
        }


def model_display_name(entry_or_key: ConfigEntry | str | None) -> str | None:
    """Resolve the display name for a :class:`ConfigEntry` or model key.

    ``ConfigEntry.model_display`` is authoritative when present.  The fallback
    path covers uploaded configs and callers that only have a canonical key,
    while retaining the same aliases as :mod:`config_registry`.
    """

    if isinstance(entry_or_key, ConfigEntry):
        if str(entry_or_key.model_display or "").strip():
            return str(entry_or_key.model_display).strip()
        key = entry_or_key.model_key
    elif entry_or_key is None:
        return None
    else:
        key = str(entry_or_key)

    raw = str(key or "").strip()
    if not raw:
        return None

    normalized = normalize_model_token(raw)
    # MODEL_DISPLAY_NAMES has aliases such as ``xgboost`` and ``xgb``.  The
    # normalized lookup preserves its separator/case-insensitive behaviour.
    for candidate, display in MODEL_DISPLAY_NAMES.items():
        if normalize_model_token(candidate) == normalized:
            return display

    canonical = MODEL_KEY_ALIASES.get(normalized)
    if canonical:
        for candidate, display in MODEL_DISPLAY_NAMES.items():
            if normalize_model_token(candidate) == normalize_model_token(canonical):
                return display

    # A key that is not in the registry is still better shown as a readable
    # title than as an empty cell.  The exact key remains available separately.
    return raw.replace("_", " ").replace("-", " ").strip().title()


def infer_config_semantics(
    config: Mapping[str, Any] | None,
    *,
    entry: ConfigEntry | None = None,
    source_path: str | Path | None = None,
) -> ConfigSemantics:
    """Infer dataset, task and model meaning from a YAML mapping.

    ``entry`` supplies registry-derived values when the config has already
    been catalogued.  Explicit YAML values still win for fields that the
    entry cannot provide, which makes the function useful for an uploaded
    config before it is written to the registry.
    """

    data = config if isinstance(config, Mapping) else {}

    dataset = _dataset_from_yaml(data)
    if not dataset and entry is not None:
        dataset = _clean_optional(entry.dataset)
    if not dataset and source_path:
        dataset = _dataset_from_path(source_path)

    task = _task_from_yaml(data)
    if not task and entry is not None:
        task = _clean_optional(entry.task)

    model_key, model_display = _model_from_yaml(data)
    if not model_key and entry is not None:
        model_key = _clean_optional(entry.model_key)
    if not model_display:
        model_display = model_display_name(entry if entry is not None else model_key)

    return ConfigSemantics(
        dataset=dataset,
        task=task,
        model_key=model_key,
        model_display=model_display,
    )


def load_config_semantics(
    path: str | Path,
    *,
    entry: ConfigEntry | None = None,
) -> ConfigSemantics:
    """Read one YAML file and return its user-facing semantics.

    Malformed or unreadable files return an all-unknown result; validation is
    owned by :mod:`config_validator`, so a display helper should never turn a
    page render into a crash.
    """

    config_path = Path(path)
    try:
        loaded = yaml.safe_load(config_path.read_text(encoding="utf-8", errors="ignore"))
    except (OSError, UnicodeError, yaml.YAMLError):
        loaded = {}
    return infer_config_semantics(
        loaded if isinstance(loaded, Mapping) else {},
        entry=entry,
        source_path=config_path,
    )


def _clean_optional(value: Any) -> str | None:
    text = str(value or "").strip()
    return text or None


def _dataset_from_yaml(data: Mapping[str, Any]) -> str | None:
    declared = _clean_optional(data.get("dataset"))
    if declared:
        return declared.upper()

    split = data.get("train_test_split")
    if not isinstance(split, Mapping):
        return None
    paths = split.get("cell_data_path")
    if isinstance(paths, (list, tuple)):
        names = [_dataset_leaf(path) for path in paths]
        names = [name for name in names if name]
        if names:
            return ", ".join(dict.fromkeys(names))
        return None
    return _dataset_leaf(paths)


def _dataset_leaf(value: Any) -> str | None:
    text = _clean_optional(value)
    if not text:
        return None
    # Configs use POSIX paths, but uploaded files on Windows may use ``\\``.
    leaf = re.split(r"[/\\]", text.rstrip("/\\"))[-1].strip()
    return leaf.upper() if leaf else None


def _task_from_yaml(data: Mapping[str, Any]) -> str | None:
    declared = _clean_optional(data.get("task"))
    if declared:
        canonical = _canonical_task(declared)
        return canonical or declared

    label = data.get("label")
    label_name = label.get("name") if isinstance(label, Mapping) else label
    name = _clean_optional(label_name)
    if not name:
        return None
    token = re.sub(r"[^a-z0-9]", "", name.lower())
    if token.startswith("rul") or "remainingusefullife" in token:
        return TASK_RUL
    if token.startswith("soh") or "stateofhealth" in token:
        return TASK_SOH
    if "trajectory" in token or "forecast" in token:
        return TASK_TRAJECTORY
    return name


def _canonical_task(value: str) -> str | None:
    token = re.sub(r"[^a-z0-9]", "", value.lower())
    if token.startswith("rul") or "remainingusefullife" in token:
        return TASK_RUL
    if token.startswith("soh") or "stateofhealth" in token:
        return TASK_SOH
    if "trajectory" in token or "forecast" in token:
        return TASK_TRAJECTORY
    return None


def _model_from_yaml(data: Mapping[str, Any]) -> tuple[str | None, str | None]:
    model = data.get("model")
    raw = model.get("name") if isinstance(model, Mapping) else model
    raw_text = _clean_optional(raw)
    if not raw_text:
        return None, None

    normalized = normalize_model_token(raw_text)
    compact = re.sub(r"[^a-z0-9]", "", raw_text.lower())
    key = MODEL_KEY_ALIASES.get(normalized) or MODEL_CLASS_KEYS.get(compact)
    if key is None:
        key = raw_text.lower().replace(" ", "_")
    return key, model_display_name(key)


def _dataset_from_path(path: str | Path) -> str | None:
    stem = Path(path).stem.upper()
    # Only fall back for the repository's conventional dataset-prefixed
    # filenames (``matr_1.yaml``).  Treating every config stem (``cnn.yaml``)
    # as a dataset would be more confusing than leaving the field unknown.
    known = {
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
        "STANFORD_2",
        "TONGJI",
        "UL_PUR",
        "XJTU",
    }
    for candidate in sorted(known, key=len, reverse=True):
        if stem == candidate or stem.startswith(f"{candidate}_") or stem.startswith(f"{candidate}-"):
            return candidate
    return None


__all__ = [
    "ConfigSemantics",
    "infer_config_semantics",
    "load_config_semantics",
    "model_display_name",
]
