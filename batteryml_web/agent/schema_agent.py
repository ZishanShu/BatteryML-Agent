"""LLM-assisted schema mapping suggestions.

Suggestion only. Per `CLAUDE.md` §3 the agent output path is always
``suggest -> user review -> validator -> execute``: this module returns a
proposed mapping and never writes it to disk, never validates it as accepted,
and never touches raw data. The caller is responsible for putting the result in
front of the user and then through
:func:`batteryml_web.core.schema_validator.validate_schema_mapping`.
"""

from __future__ import annotations

import json
from typing import Any

from .llm_client import run_llm_prompt
from .prompts import build_schema_mapping_prompt

SYSTEM_PROMPT = "You are a schema mapping assistant. Return JSON only."


def _extract_json(text: str, *, source: str) -> dict[str, Any]:
    """Parse a JSON object out of an LLM reply, tolerating a code fence.

    Args:
        text: the raw assistant message.
        source: label used in error messages so the caller can tell which
            call produced an unusable reply.

    Raises:
        ValueError: if the reply is not a JSON object.
    """
    stripped = text.strip()
    if stripped.startswith("```"):
        # Drop the opening fence (with an optional language tag) and the
        # closing fence, keeping everything between them.
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


def suggest_schema_mapping(
    dataset_summary: dict[str, Any],
    *,
    base_url: str,
    api_key: str,
    model: str,
    timeout: int = 60,
) -> dict[str, Any]:
    """Ask the configured LLM to map raw columns onto BatteryML schema fields.

    Args:
        dataset_summary: column names, dtypes, missing-value counts and a
            preview, as produced by
            :func:`batteryml_web.core.dataset_tools.inspect_tabular_file`.
        base_url: OpenAI-compatible API base URL.
        api_key: API key for that endpoint.
        model: model identifier to request.
        timeout: per-request timeout in seconds.

    Returns:
        The parsed suggestion. Always contains a ``mapping`` key
        (BatteryML field -> raw column name or empty string) and usually a
        ``rationale`` key.

    Raises:
        ValueError: if ``dataset_summary`` is empty, if credentials are
            missing, or if the reply is not a JSON object carrying ``mapping``.
        RuntimeError: if the API call itself fails (raised by
            :func:`~batteryml_web.agent.llm_client.run_llm_prompt`).
    """
    if not dataset_summary:
        raise ValueError(
            "suggest_schema_mapping: dataset_summary is empty. Inspect the uploaded "
            "file first so the prompt carries column names and dtypes."
        )

    content = run_llm_prompt(
        build_schema_mapping_prompt(dataset_summary),
        base_url=base_url,
        api_key=api_key,
        model=model,
        system_prompt=SYSTEM_PROMPT,
        timeout=timeout,
    )
    parsed = _extract_json(content, source="suggest_schema_mapping")

    mapping = parsed.get("mapping")
    if not isinstance(mapping, dict):
        raise ValueError(
            "suggest_schema_mapping: reply is missing a `mapping` object. "
            f"Keys present: {sorted(parsed)}."
        )
    return parsed
