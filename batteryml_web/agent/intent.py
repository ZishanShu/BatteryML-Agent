"""Deterministic natural-language intent resolution for workflow guidance.

Smallest possible resolver: match dataset names and task phrases against the
request text. No NLP framework, no guessing — when the dataset or the task
cannot be resolved, the caller gets an explicit clarification request instead
of a silent default. In particular an explicit natural-language request must
never silently become MATR.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from ..core.readiness import TASK_RUL, TASK_SOH, TASK_TRAJECTORY

#: Dataset names recognized in free text, longest first so "mich_exp" wins
#: over "mich". Matching is case-insensitive on word boundaries.
KNOWN_DATASETS: List[str] = [
    "MICH_EXP",
    "MATR",
    "HUST",
    "MICH",
    "CALCE",
    "RWTH",
    "HNEI",
    "SNL",
    "UL_PUR",
]

_TRAJECTORY_PHRASES: List[str] = [
    "degradation trajectory",
    "future capacity trajectory",
    "capacity trajectory",
    "trajectory",
]

_SOH_PHRASES: List[str] = [
    "state of health",
]

_RUL_PHRASES: List[str] = [
    "remaining useful life",
    "cycle life",
    "battery lifetime",
    "battery life",
    "lifetime",
]

_SOH_TOKEN = re.compile(r"\bsoh\b")
_RUL_TOKEN = re.compile(r"\brul\b")


def _dataset_patterns() -> List[tuple[str, "re.Pattern[str]"]]:
    return [
        (name, re.compile(r"\b" + re.escape(name.lower().replace("_", r"[\s_\-]?")) + r"\b"))
        for name in KNOWN_DATASETS
    ]


def resolve_dataset(text: str) -> Optional[str]:
    """Return the dataset named in ``text``, or None when none is named."""
    lowered = str(text or "").lower()
    for name, pattern in _dataset_patterns():
        if pattern.search(lowered):
            return name
    return None


def resolve_task(text: str) -> Optional[str]:
    """Return the platform task named in ``text``, or None when none is named."""
    lowered = str(text or "").lower()
    if any(phrase in lowered for phrase in _TRAJECTORY_PHRASES):
        return TASK_TRAJECTORY
    if any(phrase in lowered for phrase in _SOH_PHRASES) or _SOH_TOKEN.search(lowered):
        return TASK_SOH
    if any(phrase in lowered for phrase in _RUL_PHRASES) or _RUL_TOKEN.search(lowered):
        return TASK_RUL
    return None


def resolve_workflow_intent(text: str) -> Dict[str, Any]:
    """Resolve a natural-language workflow request to dataset + task.

    Returns a dict with ``dataset`` / ``task`` (each None when unresolvable)
    plus ``needs_dataset_clarification`` / ``needs_task_clarification``
    flags and a ``message`` suitable for asking the user. Nothing is
    defaulted: an unresolvable dataset stays None so the caller cannot offer
    MATR for a request that never named it.
    """
    if not str(text or "").strip():
        raise ValueError("resolve_workflow_intent: request text is empty.")
    dataset = resolve_dataset(text)
    task = resolve_task(text)
    needs_dataset = dataset is None
    needs_task = task is None
    if needs_dataset and needs_task:
        message = (
            "Could not determine the dataset or the task from the request. "
            f"Known datasets: {', '.join(KNOWN_DATASETS)}. "
            "Tasks: 'RUL prediction' (battery lifetime), 'SOH estimation' "
            "(state of health), 'Degradation trajectory forecasting'."
        )
    elif needs_dataset:
        message = (
            "Could not determine the dataset from the request. "
            f"Known datasets: {', '.join(KNOWN_DATASETS)}."
        )
    elif needs_task:
        message = (
            f"Dataset resolved to {dataset}, but the task is unclear. "
            "Tasks: 'RUL prediction' (battery lifetime), 'SOH estimation' "
            "(state of health), 'Degradation trajectory forecasting'."
        )
    else:
        message = f"Resolved dataset={dataset}, task={task}."
    return {
        "dataset": dataset,
        "task": task,
        "needs_dataset_clarification": needs_dataset,
        "needs_task_clarification": needs_task,
        "message": message,
    }
