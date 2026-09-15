"""The canonical catalog the LLM is allowed to choose from.

Built entirely from the live backend — ``ConfigRegistry`` for the configs
that exist, ``readiness`` for what is on disk, and the registry's own alias
tables for the names a user might type. There is deliberately no second,
hand-maintained list: a catalog that drifts from the registry would let the
agent offer a model the planner cannot build.

What the catalog is for:

* it is the *menu* handed to the LLM, so the model has real names to map a
  free-text request onto instead of inventing one;
* it is the *whitelist* the LLM's answer is checked against
  (:mod:`batteryml_web.agent.llm_intent`). A dataset, task or model that is
  not in the catalog is rejected, not accepted with a warning.

What the catalog is **not**: it is not a readiness verdict. Dataset status
here is the filesystem state (READY / RAW_ONLY / …) and a model entry says
only which datasets and tasks it has configs for. Whether a model may run
now is decided per (dataset, task) by
:func:`batteryml_web.agent.prompts.build_agent_workflow_context`, and
nothing in this module may be used to bypass it.

No Streamlit, no network, no subprocess.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from ..core.readiness import (
    EXTERNAL_MODELS,
    PLATFORM_DATASETS,
    TASK_RUL,
    TASK_SOH,
    TASK_TRAJECTORY,
)
from .execution import canonical_model_key

#: The three platform tasks, with the spellings a user may type. Kept beside
#: the canonical labels so the LLM can be told exactly what to return.
TASK_ALIASES: Dict[str, List[str]] = {
    TASK_RUL: [
        "RUL",
        "RUL prediction",
        "remaining useful life",
        "cycle life",
        "battery lifetime",
        "lifetime prediction",
        "寿命预测",
        "电池寿命预测",
        "剩余寿命预测",
    ],
    TASK_SOH: [
        "SOH",
        "SOH estimation",
        "state of health",
        "健康状态估计",
        "SOH 估计",
    ],
    TASK_TRAJECTORY: [
        "trajectory",
        "degradation trajectory forecasting",
        "capacity trajectory",
        "future capacity",
        "容量衰减轨迹预测",
        "退化轨迹预测",
    ],
}

CANONICAL_TASKS: tuple[str, ...] = (TASK_RUL, TASK_SOH, TASK_TRAJECTORY)


def _dataset_aliases(name: str) -> List[str]:
    """Spellings of one dataset name that all mean the same dataset."""
    lowered = name.lower()
    variants = {
        name,
        lowered,
        name.replace("_", " "),
        name.replace("_", "-"),
        lowered.replace("_", " "),
        lowered.replace("_", "-"),
    }
    return sorted(variants)


def build_dataset_catalog(
    *, repo_root: str | Path | None = None
) -> List[Dict[str, Any]]:
    """Every declared dataset with its aliases and current filesystem status.

    Status comes from :func:`batteryml_web.core.readiness.list_dataset_readiness`,
    so a dataset that exists only as raw files reports ``RAW_ONLY`` and one
    that is absent reports ``MISSING``. Neither is an error; both are facts
    the agent must not talk around.
    """
    from ..core.readiness import list_dataset_readiness

    root = Path(repo_root) if repo_root is not None else None
    statuses: Dict[str, Dict[str, Any]] = {}
    for report in list_dataset_readiness(
        raw_root=(root / "data" / "raw") if root is not None else None,
        processed_root=(root / "data" / "processed") if root is not None else None,
    ):
        statuses[report.dataset] = report.to_dict()

    names = sorted(set(PLATFORM_DATASETS) | set(statuses))
    catalog: List[Dict[str, Any]] = []
    for name in names:
        report = statuses.get(name, {})
        catalog.append(
            {
                "dataset": name,
                "aliases": _dataset_aliases(name),
                "status": report.get("status", "MISSING"),
                "preprocessing_supported": bool(report.get("preprocessing_supported", False)),
            }
        )
    return catalog


def build_model_catalog(
    *, repo_root: str | Path | None = None
) -> List[Dict[str, Any]]:
    """Every model the platform knows, with aliases and config coverage.

    Config-backed models come from :class:`ConfigRegistry` — one entry per
    ``model_key``, listing the datasets and tasks its configs actually
    declare. External models come from
    :data:`batteryml_web.core.readiness.EXTERNAL_MODELS`, which fixes their
    task by their real output semantics.

    ``has_validated_config`` says only that a config file exists for that
    model; whether it *passes* the validator with its data present is a
    readiness question, answered elsewhere.
    """
    from ..core.config_registry import (
        MODEL_CLASS_KEYS,
        MODEL_DISPLAY_NAMES,
        MODEL_KEY_ALIASES,
        get_config_registry,
    )

    root = Path(repo_root) if repo_root is not None else None
    registry = get_config_registry(str(root) if root is not None else None)

    by_key: Dict[str, Dict[str, Any]] = {}
    for entry in registry.entries:
        record = by_key.setdefault(
            entry.model_key,
            {
                "model_key": entry.model_key,
                "model_display": entry.model_display,
                "aliases": set(),
                "datasets": set(),
                "tasks": set(),
                "has_validated_config": False,
                "external": False,
            },
        )
        record["datasets"].add(entry.dataset)
        record["tasks"].add(entry.task)
        if entry.config_exists:
            record["has_validated_config"] = True

    for alias, target in MODEL_KEY_ALIASES.items():
        if target in by_key:
            by_key[target]["aliases"].add(alias)
    for class_name, target in MODEL_CLASS_KEYS.items():
        key = canonical_model_key(target)
        if key in by_key:
            by_key[key]["aliases"].add(class_name)
    for token, display in MODEL_DISPLAY_NAMES.items():
        key = canonical_model_key(token)
        if key in by_key:
            by_key[key]["aliases"].add(token)
            by_key[key].setdefault("model_display", display)

    for name, spec in EXTERNAL_MODELS.items():
        by_key[name] = {
            "model_key": name,
            "model_display": name,
            "aliases": {name, name.lower()},
            "datasets": set(),
            "tasks": {spec["task"]},
            "has_validated_config": False,
            "external": True,
        }

    catalog: List[Dict[str, Any]] = []
    for key in sorted(by_key):
        record = by_key[key]
        record["aliases"].add(key)
        catalog.append(
            {
                "model_key": record["model_key"],
                "model_display": record["model_display"] or record["model_key"],
                "aliases": sorted(str(a) for a in record["aliases"]),
                "datasets": sorted(record["datasets"]),
                "tasks": sorted(record["tasks"]),
                "has_validated_config": bool(record["has_validated_config"]),
                "external": bool(record["external"]),
            }
        )
    return catalog


def build_intent_catalog(*, repo_root: str | Path | None = None) -> Dict[str, Any]:
    """The whole menu: datasets, tasks and models, all from the live backend."""
    return {
        "datasets": build_dataset_catalog(repo_root=repo_root),
        "tasks": [
            {"task": task, "aliases": list(TASK_ALIASES[task])}
            for task in CANONICAL_TASKS
        ],
        "models": build_model_catalog(repo_root=repo_root),
    }


def _normalized(text: str) -> str:
    return " ".join(str(text or "").replace("_", " ").replace("-", " ").lower().split())


def resolve_catalog_dataset(
    value: Any, catalog: Dict[str, Any]
) -> Optional[str]:
    """Fold a dataset name onto a canonical key, or None when unknown."""
    wanted = _normalized(value)
    if not wanted:
        return None
    for entry in catalog.get("datasets", []):
        if _normalized(entry.get("dataset")) == wanted:
            return str(entry.get("dataset"))
        if any(_normalized(alias) == wanted for alias in entry.get("aliases", [])):
            return str(entry.get("dataset"))
    return None


def resolve_catalog_task(value: Any, catalog: Dict[str, Any]) -> Optional[str]:
    """Fold a task name onto one of the three canonical labels, or None."""
    wanted = _normalized(value)
    if not wanted:
        return None
    for entry in catalog.get("tasks", []):
        if _normalized(entry.get("task")) == wanted:
            return str(entry.get("task"))
        if any(_normalized(alias) == wanted for alias in entry.get("aliases", [])):
            return str(entry.get("task"))
    return None


def resolve_catalog_model(value: Any, catalog: Dict[str, Any]) -> Optional[str]:
    """Fold a model name onto a canonical ``model_key``, or None when unknown.

    Exact catalog names and aliases first, then the registry's own folding
    (:func:`batteryml_web.agent.execution.canonical_model_key`) so class
    names like ``BatLiNetRULPredictor`` resolve the same way the planner
    resolves them. Never fuzzy-matches: an unknown name stays unknown.
    """
    wanted = _normalized(value)
    if not wanted:
        return None
    for entry in catalog.get("models", []):
        if _normalized(entry.get("model_key")) == wanted:
            return str(entry.get("model_key"))
        if _normalized(entry.get("model_display")) == wanted:
            return str(entry.get("model_key"))
        if any(_normalized(alias) == wanted for alias in entry.get("aliases", [])):
            return str(entry.get("model_key"))
    folded = canonical_model_key(str(value))
    for entry in catalog.get("models", []):
        if str(entry.get("model_key")) == folded:
            return str(entry.get("model_key"))
    return None


def models_for_task(catalog: Dict[str, Any], task: str) -> List[Dict[str, Any]]:
    """Catalog entries whose configs declare ``task`` (or that are external to it)."""
    return [
        entry
        for entry in catalog.get("models", [])
        if task in entry.get("tasks", [])
    ]


def closest_model_keys(
    value: Any, catalog: Dict[str, Any], *, limit: int = 3
) -> List[str]:
    """Catalog model keys whose names are textually closest to ``value``.

    Used only to *suggest* alternatives for an unknown name in a message the
    user must confirm. It never selects a model: a near-miss is still a miss
    (milestone D requirement 7).
    """
    import difflib

    wanted = _normalized(value)
    if not wanted:
        return []
    names: Dict[str, str] = {}
    for entry in catalog.get("models", []):
        key = str(entry.get("model_key"))
        for name in [key, str(entry.get("model_display"))] + list(entry.get("aliases", [])):
            names[_normalized(name)] = key
    # 0.7 rather than difflib's 0.6 default: at 0.6 "transfomer" also
    # suggested "random" (ratio 0.625), which helps nobody.
    matches = difflib.get_close_matches(wanted, list(names), n=limit * 2, cutoff=0.7)
    ordered: List[str] = []
    for match in matches:
        key = names[match]
        if key not in ordered:
            ordered.append(key)
    return ordered[:limit]
