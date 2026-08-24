from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from .llm_client import LLMClient


def explain_results(metrics: Dict[str, Any], log_text: str, llm_client: Any | None = None) -> Dict[str, Any]:
    client = llm_client if isinstance(llm_client, LLMClient) else llm_client
    if client is None:
        raise ValueError("llm_client is required")

    prompt = (
        "Explain the results briefly based on these metrics and logs. Return JSON only with keys: explanation, caveats. "
        "Do not modify metrics.\n\n"
        f"Metrics: {json.dumps(metrics, ensure_ascii=False)}\n\n"
        f"Log excerpt: {log_text[:6000]}"
    )
    response = client.chat_completions(
        messages=[
            {"role": "system", "content": "You are a results explanation assistant. Return JSON only."},
            {"role": "user", "content": prompt},
        ],
        temperature=0,
    )
    content = response["choices"][0]["message"]["content"]
    return json.loads(content)
