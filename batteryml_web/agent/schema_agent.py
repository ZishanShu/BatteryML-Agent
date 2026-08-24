from __future__ import annotations

import json
from typing import Any, Dict, List

from .llm_client import LLMClient
from .prompts import build_schema_mapping_prompt


def _extract_json(text: str) -> Dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        text = text.replace("json\n", "", 1)
    return json.loads(text)


def suggest_schema_mapping(dataset_summary: Dict[str, Any], llm_client: Any | None = None) -> Dict[str, Any]:
    client = llm_client if isinstance(llm_client, LLMClient) else llm_client
    if client is None:
        raise ValueError("llm_client is required for schema suggestion")

    prompt = build_schema_mapping_prompt(dataset_summary)
    response = client.chat_completions(
        messages=[
            {"role": "system", "content": "You are a schema mapping assistant. Return JSON only."},
            {"role": "user", "content": prompt},
        ],
        temperature=0,
    )
    content = response["choices"][0]["message"]["content"]
    parsed = _extract_json(content)
    if "mapping" not in parsed:
        raise ValueError("LLM output missing `mapping` field.")
    return parsed
