"""Agent runtime: natural language -> deterministic context -> LLM explanation.

Smallest backend-only flow wiring the existing pieces together:

1. :func:`batteryml_web.agent.intent.resolve_workflow_intent` resolves the
   dataset and task from the user's request (deterministic, no guessing).
2. :func:`batteryml_web.agent.prompts.build_agent_workflow_context` builds
   the compact authoritative readiness context for that pair.
3. The compact context is injected into the existing
   :func:`batteryml_web.agent.llm_client.run_llm_prompt` call path as data;
   the LLM explains and recommends among runnable models only.

Per `CLAUDE.md` §3 the LLM must never fabricate config paths, metrics,
runnable models, or predictions, nor perform preprocessing, training or
evaluation itself. The system prompt states that the deterministic context
is authoritative; this module additionally returns the deterministic result
alongside the LLM prose so a caller never has to parse prose for state.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from .intent import resolve_workflow_intent
from .llm_client import run_llm_prompt
from .prompts import build_agent_workflow_context

SYSTEM_PROMPT = (
    "You are a BatteryML workflow guidance assistant. "
    "A deterministic workflow context is provided below and is authoritative: "
    "only models listed under runnable_models may be offered as executable now; "
    "models under blocked_models may be explained but must never be presented "
    "as executable; validated_config values must be copied exactly and never "
    "invented; recommended_next_steps may be explained but must not be replaced "
    "with invented commands. "
    "Never fabricate config paths, metrics, or predictions. "
    "Never numerically predict battery lifetime yourself. "
    "Never perform preprocessing, training, or evaluation yourself. "
    "Provide explanation and recommendation only."
)


def _render_context_block(workflow_context: Dict[str, Any]) -> str:
    """Render the compact context as JSON the LLM can read but not rewrite."""
    runnable_lines = "\n".join(
        f"- {item.get('model_display') or item.get('model')} "
        f"(status={item.get('status')}, validated_config={item.get('validated_config')})"
        for item in workflow_context.get("runnable_models", [])
    ) or "- (none)"
    blocked_lines = "\n".join(
        f"- {item.get('model')} (status={item.get('status')}; "
        f"missing: {'; '.join(str(m) for m in item.get('missing', []))}; "
        f"action: {item.get('recommended_action')})"
        for item in workflow_context.get("blocked_models", [])
    ) or "- (none)"
    steps_lines = "\n".join(
        f"- {step}" for step in workflow_context.get("recommended_next_steps", [])
    ) or "- (none)"
    return (
        "Deterministic workflow context (authoritative, do not contradict):\n"
        f"dataset: {workflow_context.get('dataset')} "
        f"(status={workflow_context.get('dataset_status')}, "
        f"preprocessing_supported={workflow_context.get('preprocessing_supported')})\n"
        f"task: {workflow_context.get('task')}\n"
        f"runnable_models:\n{runnable_lines}\n"
        f"blocked_models:\n{blocked_lines}\n"
        f"recommended_next_steps:\n{steps_lines}"
    )


def build_workflow_guidance_prompt(
    user_request: str,
    workflow_context: Dict[str, Any],
) -> str:
    """Assemble the user prompt: original request plus authoritative context."""
    return (
        f"User request:\n{user_request.strip()}\n\n"
        f"{_render_context_block(workflow_context)}\n\n"
        "Respond with workflow guidance: explain the dataset readiness, "
        "recommend among runnable_models only, explain blocked_models without "
        "offering them as executable, and restate the relevant "
        "recommended_next_steps. Copy any validated_config value exactly."
    )


def guide_workflow(
    user_request: str,
    *,
    base_url: str,
    api_key: str,
    model: str,
    repo_root: str | Path | None = None,
    timeout: int = 60,
    llm_call: Optional[Callable[..., str]] = None,
) -> Dict[str, Any]:
    """Resolve, ground, and explain one natural-language workflow request.

    Args:
        user_request: free text such as "I want to use MATR to predict
            battery lifetime."
        base_url / api_key / model: forwarded to
            :func:`~batteryml_web.agent.llm_client.run_llm_prompt` unless
            ``llm_call`` is given.
        repo_root: forwarded to the readiness layer (testability).
        timeout: per-request LLM timeout in seconds.
        llm_call: injectable LLM boundary ``(prompt, *, system_prompt,
            **kwargs) -> str``; tests pass a stub, production passes None
            to use the real client.

    Returns:
        ``{"dataset", "task", "workflow_context", "prompt_messages",
        "guidance"}``. When intent cannot be resolved, the LLM is never
        called and ``guidance`` holds the clarification message instead.
    """
    if not str(user_request or "").strip():
        raise ValueError("guide_workflow: user_request is empty.")

    intent = resolve_workflow_intent(user_request)
    dataset = intent.get("dataset")
    task = intent.get("task")

    if dataset is None or task is None:
        prompt_messages: List[Dict[str, str]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": str(user_request).strip()},
        ]
        return {
            "dataset": dataset,
            "task": task,
            "workflow_context": None,
            "prompt_messages": prompt_messages,
            "guidance": intent["message"],
        }

    workflow_context = build_agent_workflow_context(dataset, task, repo_root=repo_root)
    prompt = build_workflow_guidance_prompt(user_request, workflow_context)
    prompt_messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]

    caller = llm_call if llm_call is not None else run_llm_prompt
    try:
        guidance = caller(
            prompt,
            base_url=base_url,
            api_key=api_key,
            model=model,
            system_prompt=SYSTEM_PROMPT,
            timeout=timeout,
        )
    except TypeError:
        # A minimal stub may accept only (prompt, *, system_prompt).
        guidance = caller(prompt, system_prompt=SYSTEM_PROMPT)

    return {
        "dataset": dataset,
        "task": task,
        "workflow_context": workflow_context,
        "prompt_messages": prompt_messages,
        "guidance": str(guidance),
    }


def workflow_guidance_preview(
    user_request: str,
    *,
    repo_root: str | Path | None = None,
) -> Dict[str, Any]:
    """Resolve intent and context without calling any LLM.

    Returns the same shape as :func:`guide_workflow` minus the network call:
    ``guidance`` is None so tests and offline callers can assert exactly what
    the LLM would have been told.
    """
    if not str(user_request or "").strip():
        raise ValueError("workflow_guidance_preview: user_request is empty.")
    intent = resolve_workflow_intent(user_request)
    dataset = intent.get("dataset")
    task = intent.get("task")
    if dataset is None or task is None:
        return {
            "dataset": dataset,
            "task": task,
            "workflow_context": None,
            "prompt_messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": str(user_request).strip()},
            ],
            "guidance": None,
        }
    workflow_context = build_agent_workflow_context(dataset, task, repo_root=repo_root)
    prompt = build_workflow_guidance_prompt(user_request, workflow_context)
    return {
        "dataset": dataset,
        "task": task,
        "workflow_context": workflow_context,
        "prompt_messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "guidance": None,
    }
