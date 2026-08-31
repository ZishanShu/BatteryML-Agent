"""LLM-assisted explanation of an already-computed result.

Per `CLAUDE.md` §3 the LLM must never produce a numeric prediction, metric or
benchmark value. This module is explanation only: metrics are computed by the
BatteryML evaluator, passed in here as data, and returned unchanged alongside
the prose. :func:`explain_results` never recomputes, rounds, or edits a number.
"""

from __future__ import annotations

import json
from typing import Any

from .llm_client import run_llm_prompt
from .prompts import build_results_summary_prompt

SYSTEM_PROMPT = (
    "You are a results explanation assistant. Return JSON only. "
    "Never restate, recompute, round, or invent a metric value."
)

LOG_EXCERPT_CHARS = 6000


def _extract_json(text: str, *, source: str) -> dict[str, Any]:
    """Parse a JSON object out of an LLM reply, tolerating a code fence."""
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if len(lines) < 2:
            raise ValueError(
                f"{source}: reply opened a code fence but contained no body: {stripped[:200]!r}"
            )
        lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()

    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"{source}: reply was not valid JSON ({exc}). First 500 characters: {stripped[:500]!r}"
        ) from exc

    if not isinstance(parsed, dict):
        raise ValueError(
            f"{source}: reply parsed to {type(parsed).__name__}, expected a JSON object."
        )
    return parsed


def explain_results(
    metrics: dict[str, Any],
    log_text: str,
    *,
    base_url: str,
    api_key: str,
    model: str,
    timeout: int = 60,
) -> dict[str, Any]:
    """Ask the configured LLM to explain metrics that BatteryML already computed.

    Args:
        metrics: metric name -> value, as read from the run's ``metrics.csv``
            or from the evaluator's scores dict. Passed through verbatim.
        log_text: the run's ``training_log.txt``; only the first
            ``LOG_EXCERPT_CHARS`` characters are sent.
        base_url: OpenAI-compatible API base URL.
        api_key: API key for that endpoint.
        model: model identifier to request.
        timeout: per-request timeout in seconds.

    Returns:
        A dict with the LLM's ``explanation`` and ``caveats`` strings plus a
        ``metrics`` key holding the caller's input object unchanged, so a UI
        can render prose and numbers side by side without the prose being the
        source of any number.

    Raises:
        ValueError: if ``metrics`` is empty, if credentials are missing, or if
            the reply is not a JSON object.
        RuntimeError: if the API call itself fails (raised by
            :func:`~batteryml_web.agent.llm_client.run_llm_prompt`).
    """
    if not metrics:
        raise ValueError(
            "explain_results: metrics is empty. Read the run's metrics.csv first; "
            "this function explains computed metrics and never derives them."
        )

    prompt = build_results_summary_prompt(
        {
            "metrics": metrics,
            "log_excerpt": log_text[:LOG_EXCERPT_CHARS],
        }
    )
    content = run_llm_prompt(
        prompt,
        base_url=base_url,
        api_key=api_key,
        model=model,
        system_prompt=SYSTEM_PROMPT,
        timeout=timeout,
    )
    parsed = _extract_json(content, source="explain_results")

    # The caller's metrics object wins unconditionally: whatever the model
    # echoed back under this key is discarded.
    parsed["metrics"] = metrics
    return parsed
