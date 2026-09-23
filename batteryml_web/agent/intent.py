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
#:
#: Dataset names are Latin in every language the platform supports, so a
#: mixed Chinese/English request such as "用 MATR 数据集预测电池寿命" resolves
#: its dataset here unchanged; only the task phrases needed translating.
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
    # Forecasting phrased without the word "trajectory". These ask what
    # happens *next*, which is neither a current-cycle SOH estimate nor a
    # lifetime; they are listed so they resolve to trajectory rather than
    # falling through to whichever other table matches a stray word.
    "future degradation",
    "forecast degradation",
    "how will capacity change",
    "capacity over the next",
    "trajectory",
    # Chinese. Listed here, not in a separate resolver, so the offline route
    # (no API key, CLAUDE.md §1 Route B) understands the same phrases the
    # LLM-assisted route does. Longest first: 容量衰减轨迹 must win over 轨迹.
    "未来容量衰减轨迹",
    "容量衰减轨迹",
    "衰减轨迹",
    "退化轨迹",
    "容量轨迹",
]

_SOH_PHRASES: List[str] = [
    "state of health",
    # "How healthy is this cell now / at cycle N" is the canonical SOH
    # question asked without the acronym. "battery health" alone is enough:
    # the phrase is about condition, where the lifetime family asks about
    # duration, and the two tables cannot both match the same words.
    "battery health",
    "cell health",
    "health at cycle",
    "health now",
    "current health",
    "健康状态",
    "健康度",
    "荷电健康",
    "电池健康",
    "当前健康",
]

_RUL_PHRASES: List[str] = [
    "remaining useful life",
    # "How many cycles are left" is the lifetime question asked in plain
    # words. It is about how long the cell lasts, not how healthy it is now,
    # and must never reach the SOH route.
    "cycles are left",
    "how many cycles",
    "cycles left",
    "cycles remaining",
    "remaining cycles",
    "how much longer",
    "cycle life",
    "battery lifetime",
    "battery life",
    "lifetime",
    "剩余使用寿命",
    "剩余寿命",
    "电池寿命",
    "循环寿命",
    "使用寿命",
    "寿命预测",
    "寿命",
]

_SOH_TOKEN = re.compile(r"\bsoh\b")
_RUL_TOKEN = re.compile(r"\brul\b")

#: Phrasings that ask the platform to choose a model rather than naming one.
#: English and Chinese, matched case-insensitively as substrings; Chinese has
#: no word boundaries, and the English entries are long enough that a
#: substring match cannot fire inside an unrelated word.
_RECOMMENDATION_PHRASES: List[str] = [
    "recommend",
    "recommendation",
    "suggest a model",
    "suggest which",
    "which model",
    "what model",
    "best model",
    "best available model",
    "help me choose",
    "choose a model",
    "pick a model",
    "推荐",
    "选哪个模型",
    "用哪个模型",
    "哪个模型",
    "该用什么模型",
]


def detect_recommendation_request(text: str) -> bool:
    """Whether the request asks the platform to choose a model.

    A recommendation request is answered differently from a request that
    names a model: the platform proposes from what is runnable instead of
    resolving one name. Detected here, deterministically, so the offline
    route behaves the same way the LLM-assisted one does.
    """
    lowered = str(text or "").lower()
    if not lowered:
        return False
    return any(phrase in lowered for phrase in _RECOMMENDATION_PHRASES)


def _dataset_patterns() -> List[tuple[str, "re.Pattern[str]"]]:
    return [
        (name, re.compile(r"\b" + re.escape(name.lower().replace("_", r"[\s_\-]?")) + r"\b"))
        for name in KNOWN_DATASETS
    ]


#: Task phrase tables, ordered longest-match-first within each family so a
#: specific phrase cannot be shadowed by a shorter one it contains.
_TRAJECTORY_PHRASES.sort(key=len, reverse=True)
_SOH_PHRASES.sort(key=len, reverse=True)
_RUL_PHRASES.sort(key=len, reverse=True)


def resolve_dataset(text: str) -> Optional[str]:
    """Return the dataset named in ``text``, or None when none is named."""
    lowered = str(text or "").lower()
    for name, pattern in _dataset_patterns():
        if pattern.search(lowered):
            return name
    return None


#: "…until end of life at 80% SOH" names the *threshold* an RUL label uses,
#: exactly as ``eol_soh`` does in a config. The sentence is a lifetime
#: question, so this mention is removed before the SOH tables are consulted —
#: the same decoy, in natural language rather than YAML.
#:
#: The gap between the two may contain a decimal point ("at 0.8 SOH") but not
#: a sentence end, so the mention is only stripped while it is still part of
#: the same clause. A period followed by whitespace stops the match; a period
#: inside a number does not.
_EOL_SOH_CONTEXT = re.compile(
    r"\b(?:end[\s-]of[\s-]life|eol)\b(?:(?!\.\s)[^;!?]){0,32}?\bsoh\b"
)


def resolve_task(text: str) -> Optional[str]:
    """Return the platform task named in ``text``, or None when none is named."""
    lowered = _EOL_SOH_CONTEXT.sub(" ", str(text or "").lower())
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
    from ..core.config_registry import hint_mentions_config_or_split, resolve_split_hint

    requested_split = resolve_split_hint(text)
    mentions_config = hint_mentions_config_or_split(text)
    return {
        "dataset": dataset,
        "task": task,
        "needs_dataset_clarification": needs_dataset,
        "needs_task_clarification": needs_task,
        "message": message,
        #: Canonical split identity explicitly named in the request
        #: (``matr_primary`` / ``matr_secondary`` / ``matr_clo``), or None
        #: for a generic request. ``mentions_config_or_split`` distinguishes
        #: "named something that matches nothing" from "named nothing".
        "requested_split": requested_split,
        "mentions_config_or_split": mentions_config,
    }
