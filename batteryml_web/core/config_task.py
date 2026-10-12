"""Which task a BatteryML run config trains for — decided from its structure.

**The** task-inference rule of the platform. The config validator (the gate
on execution), the config registry (the catalog), the config semantics shown
on Home, the uploaded-config check and the Quick experience run plan all call
:func:`resolve_config_task`; none of them keeps a second copy.

The authority is the label annotator: it *is* what the run trains against.
It is matched by its exact registered class name — never by substring,
prefix or file name — so ``label: {name: RULLabelAnnotator, eol_soh: 0.8}``
stays RUL (``eol_soh`` is RUL's end-of-life threshold, not an SOH task) and a
file called ``soh_something.yaml`` proves nothing.

An explicit top-level ``task:`` is optional (the native BatteryML format has
none; only the SOH presets declare one). When present it must be one of a
small set of exact spellings and must agree with the annotator; a
disagreement is an error, never resolved by picking one side.

Streamlit-free and side-effect free.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional

#: Canonical task labels (the values ``readiness.TASK_RUL`` / ``TASK_SOH``
#: carry; ``tests/test_config_task.py`` keeps the two in step).
TASK_RUL = "RUL prediction"
TASK_SOH = "SOH estimation"
SUPPORTED_TASK_LABELS = (TASK_RUL, TASK_SOH)

#: Registered label annotator class name -> the task it trains for.
#: Exact keys of BatteryML's ``LABEL_ANNOTATORS`` registry.
TASK_BY_LABEL_ANNOTATOR = {
    "RULLabelAnnotator": TASK_RUL,
    "SOHLabelAnnotator": TASK_SOH,
    "CycleSOHLabelAnnotator": TASK_SOH,
}

#: The short form the config validator reports (``"RUL"`` / ``"SOH"``).
SHORT_TASK = {TASK_RUL: "RUL", TASK_SOH: "SOH"}

#: Exact (case-insensitive, whitespace-trimmed) spellings accepted for an
#: explicitly declared ``task:``. Nothing else is guessed.
_DECLARED_TASKS = {
    "rul": TASK_RUL,
    "rul prediction": TASK_RUL,
    "remaining useful life": TASK_RUL,
    "remaining useful life prediction": TASK_RUL,
    "soh": TASK_SOH,
    "soh estimation": TASK_SOH,
    "state of health": TASK_SOH,
    "state of health estimation": TASK_SOH,
}

SOURCE_ANNOTATOR = "label annotator"
SOURCE_DECLARED = "declared task"


def canonical_task(value: Any) -> Optional[str]:
    """``"RUL prediction"`` / ``"SOH estimation"`` for an exact known spelling, else None."""
    text = " ".join(str(value or "").strip().lower().split())
    return _DECLARED_TASKS.get(text)


@dataclass(frozen=True)
class TaskResolution:
    """What a config says it trains for.

    ``task`` is None whenever ``error`` is set. ``structural`` is True only
    when the label annotator itself established the task — the condition an
    execution gate requires, because the annotator is what actually runs.
    """

    task: Optional[str]
    source: Optional[str]
    annotator: Optional[str]
    declared: Optional[str]
    error: Optional[str] = None

    @property
    def structural(self) -> bool:
        return self.task is not None and self.source == SOURCE_ANNOTATOR

    @property
    def short(self) -> Optional[str]:
        return SHORT_TASK.get(self.task) if self.task else None


def resolve_config_task(config: Optional[Mapping[str, Any]]) -> TaskResolution:
    """Resolve the task of a parsed config. Pure; never raises."""
    data = config if isinstance(config, Mapping) else {}
    label = data.get("label")
    raw_annotator = label.get("name") if isinstance(label, Mapping) else None
    annotator = str(raw_annotator).strip() if raw_annotator else None
    raw_declared = data.get("task")
    declared_text = str(raw_declared).strip() if raw_declared not in (None, "") else None

    structural = TASK_BY_LABEL_ANNOTATOR.get(annotator) if annotator else None
    declared = canonical_task(declared_text) if declared_text else None

    if declared_text and declared is None:
        return TaskResolution(
            None, None, annotator, declared_text,
            error=(f"The config declares task {declared_text!r}, which is not a supported "
                   f"task. Supported: {', '.join(SUPPORTED_TASK_LABELS)}."),
        )
    if structural and declared and structural != declared:
        return TaskResolution(
            None, None, annotator, declared_text,
            error=(f"The config declares task {declared!r} but its label annotator "
                   f"{annotator!r} trains for {structural!r}. Fix the config; the "
                   "platform never picks one of the two."),
        )
    if structural:
        return TaskResolution(structural, SOURCE_ANNOTATOR, annotator, declared_text)
    if declared:
        return TaskResolution(declared, SOURCE_DECLARED, annotator, declared_text)
    if annotator:
        return TaskResolution(
            None, None, annotator, None,
            error=(f"The label annotator {annotator!r} does not identify a supported "
                   f"task. Supported annotators: {', '.join(sorted(TASK_BY_LABEL_ANNOTATOR))}."),
        )
    return TaskResolution(
        None, None, None, None,
        error="The config has no `label.name`, so its task cannot be determined.",
    )


__all__ = [
    "SHORT_TASK",
    "SUPPORTED_TASK_LABELS",
    "TASK_BY_LABEL_ANNOTATOR",
    "TASK_RUL",
    "TASK_SOH",
    "TaskResolution",
    "canonical_task",
    "resolve_config_task",
]
