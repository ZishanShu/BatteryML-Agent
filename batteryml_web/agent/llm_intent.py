"""LLM-assisted intent resolution, with the backend as the final authority.

The deterministic resolver in :mod:`batteryml_web.agent.intent` is precise
but rigid: it matches phrases. A user writing "Can you run a model for
cycle aging on the MATR cells?", "Try BatLiNet on my MATR cells", or
"帮我推荐一个能做电池寿命预测的模型" means something the phrase table does not
cover. This module lets an LLM *interpret* such a request, and then refuses
to take its word for anything.

The pipeline, in order:

    LLM interpretation
      -> structured candidate intent (JSON only)
      -> canonical catalog resolution   (batteryml_web/agent/catalog.py)
      -> readiness                      (build_agent_workflow_context)
      -> validator                      (build_execution_plan)
      -> user confirmation              (the Run button)
      -> execution

What the LLM may do: map free text onto canonical dataset / task / model
identifiers that already exist, ask one clarifying question, and write a
short user-facing sentence of rationale.

What the LLM may never do, enforced here rather than requested in a prompt:

* invent a dataset, task or model — anything that does not resolve against
  the catalog is rejected and recorded, never passed on;
* supply a config path, a command, a metric or a workspace — those keys are
  dropped and reported even if the model returns them;
* declare anything executable. Nothing in this module consults readiness,
  builds a plan, or calls an executor. ``needs_clarification`` and
  ``suggested_model_keys`` are advisory inputs to a UI that must still go
  through the validator and an explicit user action.

If no LLM is configured, or the call or the parse fails, the deterministic
resolver's answer is returned with ``llm_available`` False so the caller can
say plainly that advanced natural-language understanding is unavailable.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from .catalog import (
    build_intent_catalog,
    closest_model_keys,
    resolve_catalog_dataset,
    resolve_catalog_model,
    resolve_catalog_task,
)
from .intent import detect_recommendation_request, resolve_workflow_intent

#: Keys an answer must never carry. Each names something only the backend
#: may decide; seeing one is a signal the model tried to skip the gate.
FORBIDDEN_KEYS: tuple[str, ...] = (
    "config",
    "config_path",
    "configs",
    "command",
    "argv",
    "workspace",
    "metrics",
    "rmse",
    "mae",
    "mape",
    "prediction",
    "predictions",
    "can_execute",
    "epochs",
    "device",
)

#: Caps on the free text the model is allowed to contribute to the UI.
MAX_RATIONALE_CHARS = 600
MAX_QUESTION_CHARS = 300
MAX_SUGGESTIONS = 5

#: Aliases shorter than this are matched on word boundaries only. Chinese
#: text has no spaces, so a longer alias is also matched as a substring;
#: doing that for "rf" or "cnn" would match inside unrelated words.
_SUBSTRING_ALIAS_MIN_LEN = 5

SYSTEM_PROMPT = (
    "You map a battery-ageing workflow request onto a fixed catalog of "
    "datasets, tasks and models. "
    "Reply with a single JSON object and nothing else: no prose, no markdown "
    "fence, no explanation outside the JSON. "
    "Use only identifiers that appear verbatim in the catalog you are given. "
    "If the request names something that is not in the catalog, set the "
    "corresponding field to null and say so in user_facing_rationale; never "
    "substitute a similar name silently. "
    "Never output a config path, a file path, a command, a metric, a device or "
    "a number of epochs — the backend owns all of those. "
    "Never state that a model can run: readiness is decided after you, by the "
    "platform. "
    "Do not include reasoning steps. user_facing_rationale is at most two short "
    "sentences written for the user."
)


def _catalog_block(catalog: Dict[str, Any]) -> str:
    """Render the catalog compactly enough to fit in a prompt."""
    dataset_lines = "\n".join(
        f"- {entry['dataset']} (status={entry.get('status')})"
        for entry in catalog.get("datasets", [])
    ) or "- (none)"
    task_lines = "\n".join(
        f"- {entry['task']}" for entry in catalog.get("tasks", [])
    ) or "- (none)"
    model_lines = []
    for entry in catalog.get("models", []):
        aliases = [a for a in entry.get("aliases", []) if a != entry["model_key"]][:4]
        datasets = ", ".join(entry.get("datasets", [])) or "none"
        tasks = ", ".join(entry.get("tasks", [])) or "none"
        model_lines.append(
            f"- {entry['model_key']} | {entry['model_display']} | tasks: {tasks} "
            f"| configs for datasets: {datasets}"
            + (f" | also called: {', '.join(aliases)}" if aliases else "")
            + (" | external runner" if entry.get("external") else "")
        )
    models = "\n".join(model_lines) or "- (none)"
    return (
        "CATALOG (the only identifiers you may use)\n"
        f"datasets:\n{dataset_lines}\n"
        f"tasks (use the label exactly):\n{task_lines}\n"
        f"models (use model_key exactly):\n{models}"
    )


def build_intent_prompt(user_request: str, catalog: Dict[str, Any]) -> str:
    """The user-side prompt: the request, the catalog, and the exact schema."""
    return (
        f"User request (may be English, Chinese, or mixed):\n{str(user_request).strip()}\n\n"
        f"{_catalog_block(catalog)}\n\n"
        "Return exactly this JSON shape:\n"
        "{\n"
        '  "dataset": "<dataset name from the catalog>" or null,\n'
        '  "task": "<task label from the catalog>" or null,\n'
        '  "requested_model": "<model_key the user explicitly asked for>" or null,\n'
        '  "recommendation_requested": true or false,\n'
        '  "needs_clarification": true or false,\n'
        '  "clarification_question": "<one short question>" or null,\n'
        '  "suggested_model_keys": ["<model_key>", ...],\n'
        '  "user_facing_rationale": "<at most two short sentences>"\n'
        "}\n\n"
        "Rules:\n"
        "- If the user did not name a dataset, dataset is null and "
        "needs_clarification is true. Never assume a default dataset.\n"
        "- If the user did not name a task, task is null and "
        "needs_clarification is true, unless the wording clearly implies one of "
        "the catalog tasks.\n"
        "- requested_model is only for a model the user actually named. If the "
        "name is not in the catalog, set it to null and name it in the rationale.\n"
        "- recommendation_requested is true when the user asks which model to use, "
        "or asks for the best or a suggested model, rather than naming one.\n"
        "- suggested_model_keys is your recommendation; it may be empty. The "
        "platform will drop any model that is not currently runnable, so "
        "suggest freely but never claim any of them can run."
    )


def extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    """Parse the first JSON object in ``text``; None when there is none.

    Tolerates a markdown fence and surrounding prose, because a model that
    ignores "JSON only" is common and the alternative — treating the whole
    reply as a failure — throws away a usable answer. It does not tolerate
    invalid JSON: a parse failure is reported, never patched up.
    """
    raw = str(text or "").strip()
    if not raw:
        return None
    fenced = re.search(r"```(?:json)?\s*(.*?)```", raw, re.DOTALL)
    if fenced:
        raw = fenced.group(1).strip()
    start = raw.find("{")
    end = raw.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        parsed = json.loads(raw[start : end + 1])
    except (ValueError, TypeError):
        return None
    return parsed if isinstance(parsed, dict) else None


def _clean_text(value: Any, limit: int) -> Optional[str]:
    if not isinstance(value, str):
        return None
    text = " ".join(value.split())
    if not text:
        return None
    return text[:limit]


def validate_llm_intent(
    payload: Any,
    catalog: Dict[str, Any],
) -> Dict[str, Any]:
    """Check one parsed LLM answer against the canonical catalog.

    Pure and side-effect free, so the whole rejection policy is testable
    without a network. Returns the same shape
    :func:`resolve_workflow_intent_with_llm` returns, minus ``source`` and
    ``llm_available``, with every unusable value replaced by ``None`` and
    recorded under ``rejected``.

    ``rejected`` entries are ``{"field", "value", "kind"}`` — the caller
    shows them so a user can see that a hallucinated name was refused rather
    than quietly ignored.
    """
    rejected: List[Dict[str, Any]] = []
    warnings: List[str] = []
    if not isinstance(payload, dict):
        return {
            "dataset": None,
            "task": None,
            "requested_model": None,
            "requested_model_name": None,
            "recommendation_requested": False,
            "needs_clarification": True,
            "clarification_question": None,
            "suggested_model_keys": [],
            "user_facing_rationale": "",
            "rejected": [{"field": "<response>", "value": None, "kind": "not_an_object"}],
            "warnings": ["The model did not return a JSON object."],
        }

    for key in FORBIDDEN_KEYS:
        if key in payload:
            rejected.append({"field": key, "value": payload[key], "kind": "forbidden_field"})
    if rejected:
        warnings.append(
            "The model returned backend-owned field(s) "
            f"({', '.join(item['field'] for item in rejected)}); they were discarded. "
            "Config paths, commands, devices and metrics come from the platform only."
        )

    dataset = resolve_catalog_dataset(payload.get("dataset"), catalog)
    if payload.get("dataset") is not None and dataset is None:
        rejected.append(
            {"field": "dataset", "value": payload.get("dataset"), "kind": "unknown_dataset"}
        )

    task = resolve_catalog_task(payload.get("task"), catalog)
    if payload.get("task") is not None and task is None:
        rejected.append({"field": "task", "value": payload.get("task"), "kind": "unknown_task"})

    requested_raw = payload.get("requested_model")
    requested_model = resolve_catalog_model(requested_raw, catalog)
    requested_model_name = None
    if requested_raw is not None and requested_model is None:
        requested_model_name = _clean_text(requested_raw, 120) or str(requested_raw)[:120]
        rejected.append(
            {"field": "requested_model", "value": requested_raw, "kind": "unknown_model"}
        )

    suggested: List[str] = []
    raw_suggestions = payload.get("suggested_model_keys")
    if raw_suggestions is None:
        raw_suggestions = []
    if not isinstance(raw_suggestions, (list, tuple)):
        rejected.append(
            {"field": "suggested_model_keys", "value": raw_suggestions, "kind": "not_a_list"}
        )
        raw_suggestions = []
    for candidate in list(raw_suggestions)[: MAX_SUGGESTIONS * 2]:
        key = resolve_catalog_model(candidate, catalog)
        if key is None:
            rejected.append(
                {"field": "suggested_model_keys", "value": candidate, "kind": "unknown_model"}
            )
            continue
        if key not in suggested:
            suggested.append(key)
    suggested = suggested[:MAX_SUGGESTIONS]

    needs = payload.get("needs_clarification")
    if not isinstance(needs, bool):
        if needs is not None:
            warnings.append("needs_clarification was not a boolean; treated as false.")
        needs = False
    if dataset is None or task is None:
        needs = True

    recommendation = payload.get("recommendation_requested")
    if not isinstance(recommendation, bool):
        if recommendation is not None:
            warnings.append(
                "recommendation_requested was not a boolean; treated as false."
            )
        recommendation = False

    return {
        "dataset": dataset,
        "task": task,
        "requested_model": requested_model,
        "requested_model_name": requested_model_name,
        "recommendation_requested": bool(recommendation),
        "needs_clarification": bool(needs),
        "clarification_question": _clean_text(
            payload.get("clarification_question"), MAX_QUESTION_CHARS
        ),
        "suggested_model_keys": suggested,
        "user_facing_rationale": _clean_text(
            payload.get("user_facing_rationale"), MAX_RATIONALE_CHARS
        )
        or "",
        "rejected": rejected,
        "warnings": warnings,
    }


def find_catalog_model_mention(text: str, catalog: Dict[str, Any]) -> Optional[str]:
    """Deterministically match a model name in free text against the catalog.

    Longest alias first on a word boundary; for aliases of at least
    ``_SUBSTRING_ALIAS_MIN_LEN`` characters a plain substring match is also
    accepted, because Chinese text has no spaces and "用CPTransformer做预测"
    offers no word boundary. Short aliases ("rf", "cnn", "mlp") stay
    boundary-only so they cannot match inside an unrelated word.
    """
    lowered = str(text or "").lower()
    if not lowered:
        return None
    candidates: List[tuple[int, str]] = []
    for entry in catalog.get("models", []):
        key = str(entry.get("model_key"))
        names = {key, str(entry.get("model_display"))} | set(entry.get("aliases", []))
        for alias in names:
            alias_lower = str(alias).strip().lower()
            if not alias_lower:
                continue
            # A multi-word alias also matches written without its separator
            # ("ElasticNet", "elastic-net" for "elastic net"), still on word
            # boundaries, so "Use ElasticNet on HUST" names elastic_net.
            pattern = r"[\s_\-]*".join(re.escape(part) for part in re.split(r"[\s_\-]+", alias_lower))
            if re.search(r"\b" + pattern + r"\b", lowered):
                candidates.append((len(alias_lower), key))
            elif len(alias_lower) >= _SUBSTRING_ALIAS_MIN_LEN and alias_lower in lowered:
                candidates.append((len(alias_lower), key))
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0][1]


def _deterministic_result(
    user_request: str,
    catalog: Dict[str, Any],
    *,
    source: str,
    llm_available: bool,
    warnings: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """The deterministic resolver's answer in the structured result shape."""
    intent = resolve_workflow_intent(user_request)
    dataset = intent.get("dataset")
    task = intent.get("task")
    mention = find_catalog_model_mention(user_request, catalog)
    return {
        "source": source,
        "llm_available": bool(llm_available),
        "dataset": dataset,
        "task": task,
        "requested_model": mention,
        "requested_model_name": None,
        "recommendation_requested": detect_recommendation_request(user_request),
        "needs_clarification": dataset is None or task is None,
        "clarification_question": (
            str(intent.get("message")) if (dataset is None or task is None) else None
        ),
        "suggested_model_keys": [],
        "user_facing_rationale": "",
        "rejected": [],
        "warnings": list(warnings or []),
        "message": str(intent.get("message")),
    }


def resolve_workflow_intent_with_llm(
    user_request: str,
    *,
    base_url: str = "",
    api_key: str = "",
    model: str = "",
    repo_root: str | Path | None = None,
    timeout: int = 60,
    llm_call: Optional[Callable[..., str]] = None,
    catalog: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Resolve one request to canonical identifiers, using the LLM only if needed.

    Args:
        user_request: the user's free text.
        base_url / api_key / model: LLM credentials. When any is missing and
            no ``llm_call`` is injected, the deterministic answer is returned
            with ``llm_available`` False.
        repo_root: forwarded to the catalog builder (testability).
        timeout: per-request LLM timeout in seconds.
        llm_call: injectable ``(prompt, *, system_prompt, ...) -> str`` for
            tests; production passes None and the real client is used.
        catalog: injectable catalog (tests); built from the backend otherwise.

    Returns a dict with ``source`` (``"deterministic"``, ``"llm"`` or
    ``"deterministic_fallback"``), ``llm_available``, ``dataset``, ``task``,
    ``requested_model``, ``requested_model_name`` (the unusable name the
    model asked for, when it was rejected), ``needs_clarification``,
    ``clarification_question``, ``suggested_model_keys``,
    ``user_facing_rationale``, ``rejected``, ``warnings`` and ``message``.

    Every identifier in the result exists in the catalog. None of them means
    anything is runnable: that is decided afterwards, by readiness and the
    validator.
    """
    if not str(user_request or "").strip():
        raise ValueError("resolve_workflow_intent_with_llm: user_request is empty.")
    working_catalog = (
        catalog if catalog is not None else build_intent_catalog(repo_root=repo_root)
    )

    caller = llm_call
    credentials_present = bool(
        str(base_url).strip() and str(api_key).strip() and str(model).strip()
    )
    llm_available = caller is not None or credentials_present

    deterministic = resolve_workflow_intent(user_request)
    wants_recommendation = detect_recommendation_request(user_request)
    if deterministic.get("dataset") and deterministic.get("task"):
        # Fast path: the phrase table already answered, so no tokens are
        # spent and the offline route and the LLM route agree by construction.
        #
        # One exception: a request that asks *which* model to use is asking
        # for a judgement the phrase table cannot make. When an LLM is
        # available it is consulted even though the dataset and task are
        # already resolved; its suggestions are still filtered by readiness
        # afterwards, so it decides nothing.
        if not (wants_recommendation and llm_available):
            return _deterministic_result(
                user_request,
                working_catalog,
                source="deterministic",
                llm_available=llm_available,
            )

    if caller is None:
        if not credentials_present:
            return _deterministic_result(
                user_request,
                working_catalog,
                source="deterministic_fallback",
                llm_available=False,
                warnings=[
                    "No LLM is configured, so advanced natural-language "
                    "understanding is unavailable. The deterministic resolver "
                    "matched what it could; name the dataset and the task "
                    "explicitly if it asks."
                ],
            )
        from .llm_client import run_llm_prompt

        caller = run_llm_prompt

    prompt = build_intent_prompt(user_request, working_catalog)
    try:
        raw = caller(
            prompt,
            base_url=base_url,
            api_key=api_key,
            model=model,
            system_prompt=SYSTEM_PROMPT,
            timeout=timeout,
        )
    except TypeError:
        # A minimal stub may accept only (prompt, *, system_prompt).
        raw = caller(prompt, system_prompt=SYSTEM_PROMPT)
    except Exception as exc:  # noqa: BLE001 - reported, and the run continues offline
        return _deterministic_result(
            user_request,
            working_catalog,
            source="deterministic_fallback",
            llm_available=False,
            warnings=[
                f"The LLM request failed ({type(exc).__name__}: {exc}); the "
                "deterministic resolver was used instead."
            ],
        )

    payload = extract_json_object(raw)
    if payload is None:
        fallback = _deterministic_result(
            user_request,
            working_catalog,
            source="deterministic_fallback",
            llm_available=True,
            warnings=[
                "The LLM reply was not valid JSON, so it was discarded and the "
                "deterministic resolver was used instead."
            ],
        )
        return fallback

    validated = validate_llm_intent(payload, working_catalog)
    validated["source"] = "llm"
    validated["llm_available"] = True
    validated["message"] = (
        validated.get("clarification_question")
        or validated.get("user_facing_rationale")
        or str(deterministic.get("message"))
    )
    if validated.get("requested_model") is None and validated.get("requested_model_name") is None:
        # The model named nothing; fall back to the deterministic mention so an
        # explicit "Use BatLiNet" is never lost to a vague LLM answer.
        validated["requested_model"] = find_catalog_model_mention(
            user_request, working_catalog
        )
    # The deterministic detector is additive, never subtractive: a request that
    # plainly asks for a recommendation stays one even if the model said false.
    validated["recommendation_requested"] = bool(
        validated.get("recommendation_requested")
    ) or wants_recommendation
    # Also fall back to the resolved dataset/task where the model left them
    # null but the phrase table had them, so an LLM answer is never worse than
    # the deterministic one it replaced.
    if validated.get("dataset") is None and deterministic.get("dataset"):
        validated["dataset"] = deterministic["dataset"]
    if validated.get("task") is None and deterministic.get("task"):
        validated["task"] = deterministic["task"]
    if validated.get("dataset") and validated.get("task"):
        validated["needs_clarification"] = bool(
            validated.get("clarification_question")
        ) and bool(validated.get("needs_clarification"))
    return validated


def resolve_model_request(
    requested: Optional[str],
    workflow_context: Dict[str, Any],
    catalog: Dict[str, Any],
) -> Dict[str, Any]:
    """Decide what an explicitly requested model means for this context.

    The readiness context is authoritative: a model is ``runnable`` only if
    it appears under ``runnable_models`` there. A requested model that is
    blocked, or applies to another dataset/task, or is not in the catalog at
    all, is reported as such together with the runnable alternatives — and
    ``requires_confirmation`` is True, because milestone D forbids silently
    substituting one model for another.

    Returns ``{"requested", "status", "model_key", "model_display",
    "reason", "alternatives", "requires_confirmation"}`` with ``status`` in
    ``{"none", "runnable", "blocked", "unknown"}``.
    """
    from .execution import find_model_entry

    runnable_keys = [str(e.get("model")) for e in workflow_context.get("runnable_models", [])]
    name = str(requested or "").strip()
    if not name:
        return {
            "requested": None,
            "status": "none",
            "model_key": None,
            "model_display": None,
            "reason": "",
            "alternatives": runnable_keys,
            "requires_confirmation": False,
        }

    entry, side = find_model_entry(workflow_context, name)
    if entry is not None and side == "runnable":
        return {
            "requested": name,
            "status": "runnable",
            "model_key": str(entry.get("model")),
            "model_display": entry.get("model_display"),
            "reason": "",
            "alternatives": [key for key in runnable_keys if key != str(entry.get("model"))],
            "requires_confirmation": False,
        }
    if entry is not None:
        missing = "; ".join(str(item) for item in entry.get("missing") or [])
        return {
            "requested": name,
            "status": "blocked",
            "model_key": str(entry.get("model")),
            # Blocked context entries carry no display name; the canonical key
            # is the honest label rather than None.
            "model_display": entry.get("model_display") or str(entry.get("model")),
            "reason": (
                f"{entry.get('model')} is {entry.get('status')} for this dataset and "
                f"task: {entry.get('recommended_action') or missing or 'see blocked_models'}"
            ),
            "alternatives": runnable_keys,
            "requires_confirmation": True,
        }

    catalog_key = resolve_catalog_model(name, catalog)
    if catalog_key is not None:
        return {
            "requested": name,
            "status": "blocked",
            "model_key": catalog_key,
            "model_display": catalog_key,
            "reason": (
                f"{catalog_key} is a known model, but it is not available for "
                f"dataset {workflow_context.get('dataset')} and task "
                f"{workflow_context.get('task')}."
            ),
            "alternatives": runnable_keys,
            "requires_confirmation": True,
        }

    return {
        "requested": name,
        "status": "unknown",
        "model_key": None,
        "model_display": None,
        "reason": (
            f"'{name}' is not currently registered on this platform. See "
            "docs/EXTENDING_MODELS.md to add it."
        ),
        "alternatives": [
            key for key in closest_model_keys(name, catalog) if key in runnable_keys
        ]
        or runnable_keys,
        "requires_confirmation": True,
    }


def filter_suggestions(
    suggested_keys: List[str],
    workflow_context: Dict[str, Any],
) -> Dict[str, List[str]]:
    """Split LLM suggestions into runnable / blocked / unknown, by the backend.

    The LLM may suggest anything in the catalog; only the readiness context
    decides what is offered as executable. A blocked suggestion is kept
    separately so it can be *mentioned* with its reason (PBT, for example)
    and never presented as runnable.
    """
    from .execution import find_model_entry

    runnable: List[str] = []
    blocked: List[str] = []
    unknown: List[str] = []
    for key in suggested_keys or []:
        name = str(key).strip()
        if not name:
            continue
        entry, side = find_model_entry(workflow_context, name)
        if entry is None:
            unknown.append(name)
        elif side == "runnable":
            if str(entry.get("model")) not in runnable:
                runnable.append(str(entry.get("model")))
        elif str(entry.get("model")) not in blocked:
            blocked.append(str(entry.get("model")))
    return {"runnable": runnable, "blocked": blocked, "unknown": unknown}
