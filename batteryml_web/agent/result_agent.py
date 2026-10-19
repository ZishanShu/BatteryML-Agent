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
from .result_analysis import (
    ANALYSIS_SYSTEM_PROMPT,
    build_result_analysis_prompt,
)
from .result_analysis_v2 import (
    ANALYSIS_SYSTEM_PROMPT_V2,
    CONTEXT_VERSION as CONTEXT_VERSION_V2,
    build_result_analysis_prompt_v2,
)
from .results import build_result_interpretation_prompt

RESULT_SYSTEM_PROMPT = (
    "You are a run-result interpretation assistant. Return JSON only. "
    "The structured run result below is authoritative: restate its metrics "
    "exactly, never recompute, round, or invent a metric value, never claim "
    "a missing metric, never claim benchmark performance, convergence, or "
    "scientific superiority from a smoke run, and interpret MAPE as a ratio "
    "(e.g. 0.662 means about 66.2%) while preserving the raw number."
)

# Backwards-compatible alias: existing callers import SYSTEM_PROMPT.
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


def explain_run_result(
    result_context: dict[str, Any],
    *,
    base_url: str,
    api_key: str,
    model: str,
    timeout: int = 60,
    llm_call: Any | None = None,
    analysis_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Explain a structured run result from :func:`results.build_result_context`.

    Reuses the existing explanation boundary (:func:`explain_results` rules):
    metrics pass through verbatim and the reply is parsed as JSON. Unlike
    :func:`explain_results`, this entry point accepts the whole structured
    context — including the empty-metrics case, which it reports honestly
    instead of raising, and the smoke-run warning, which the prompt carries
    so the LLM restates it rather than claiming benchmark performance.

    Args:
        result_context: dict from
            :func:`batteryml_web.agent.results.build_result_context`.
        base_url / api_key / model / timeout: forwarded to the LLM client
            unless ``llm_call`` is given.
        llm_call: injectable ``(prompt, *, system_prompt, **kwargs) -> str``
            boundary for tests.
        analysis_context: the richer deterministic context from
            :func:`~batteryml_web.agent.result_analysis.build_analysis_context`.
            When present it selects the analysis prompt, which asks for
            interpretation, diagnostics, a gated benchmark comparison and
            next steps instead of a restatement of the configuration.

    Returns:
        A dict with the LLM's parsed reply plus ``result_context`` holding
        the caller's input object unchanged, so prose is never the source
        of any number. ``analysis_context`` is echoed back the same way when
        one was supplied, so the benchmark the UI renders and the benchmark
        the model was shown are the same object.

    Raises:
        ValueError: if ``result_context`` is empty or a failed run carries
            fabricated metrics.
        RuntimeError: if the API call itself fails.
    """
    if not result_context:
        raise ValueError(
            "explain_run_result: result_context is empty. Build it with "
            "batteryml_web.agent.results.build_result_context first."
        )
    metrics = dict(result_context.get("metrics") or {})
    if not result_context.get("success") and metrics:
        raise ValueError(
            "explain_run_result: failed run carries metrics; refusing to "
            "interpret fabricated values."
        )

    # An enriched analysis context gets the analysis prompt; a bare result
    # context keeps the interpretation prompt. Two shapes rather than two
    # code paths: the caller decides how much evidence it can supply, and a
    # caller that supplies none still works exactly as before.
    if analysis_context and analysis_context.get("context_version") == CONTEXT_VERSION_V2:
        prompt = build_result_analysis_prompt_v2(analysis_context)
        system_prompt = ANALYSIS_SYSTEM_PROMPT_V2
    elif analysis_context:
        prompt = build_result_analysis_prompt(analysis_context)
        system_prompt = ANALYSIS_SYSTEM_PROMPT
    else:
        prompt = build_result_interpretation_prompt(result_context)
        system_prompt = RESULT_SYSTEM_PROMPT
    caller = llm_call if llm_call is not None else run_llm_prompt
    try:
        content = caller(
            prompt,
            base_url=base_url,
            api_key=api_key,
            model=model,
            system_prompt=system_prompt,
            timeout=timeout,
        )
    except TypeError:
        content = caller(prompt, system_prompt=system_prompt)
    parsed = _extract_json(content, source="explain_run_result")

    # The caller's context wins unconditionally: whatever the model echoed
    # back under these keys is discarded.
    parsed["metrics"] = metrics
    parsed["result_context"] = result_context
    if analysis_context:
        parsed["analysis_context"] = analysis_context
    return parsed
