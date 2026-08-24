from __future__ import annotations

from pathlib import Path

import requests



def _join_base_url(base_url: str, path: str) -> str:
    base = str(base_url or "").strip().rstrip("/")
    suffix = path if path.startswith("/") else f"/{path}"
    return f"{base}{suffix}"


def run_llm_prompt(
    prompt: str,
    *,
    base_url: str,
    api_key: str,
    model: str,
    system_prompt: str | None = None,
    timeout: int = 60,
) -> str:
    if not base_url or not api_key or not model:
        raise ValueError("Missing Base URL, API key, or model.")
    endpoint = _join_base_url(base_url, "/chat/completions")
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    response = requests.post(
        endpoint,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={"model": model, "messages": messages, "temperature": 0.2},
        timeout=timeout,
    )
    if response.status_code >= 400:
        raise RuntimeError(f"LLM API request failed with status {response.status_code}: {response.text[:1200]}")
    data = response.json()
    try:
        content = data["choices"][0]["message"]["content"]
    except Exception as exc:
        raise RuntimeError(f"Unexpected LLM API response format: {data}") from exc
    if not content or not str(content).strip():
        raise RuntimeError("LLM API returned an empty response.")
    return str(content).strip()


def test_llm_connection(*, base_url: str, api_key: str, model: str) -> tuple[bool, str]:
    try:
        text = run_llm_prompt(
            "Reply with exactly: OK",
            base_url=base_url,
            api_key=api_key,
            model=model,
            system_prompt="You are a connection test assistant.",
            timeout=30,
        )
        return True, "LLM connection verified." if "OK" in text else f"LLM responded, but not with exact OK: {text[:200]}"
    except Exception as exc:
        return False, f"LLM connection failed: {exc}"


def save_llm_output(output_text: str, output_path: Path, *, title: str = "BatteryML LLM Assistant Output", model: str = "", base_url: str = "") -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    text = (
        f"# {title}\n\n"
        f"- Model: {model}\n"
        f"- Base URL: {base_url}\n"
        "- Safety: Suggestion only. Requires user confirmation before applying.\n\n"
        "## Output\n\n"
        f"{output_text}\n"
    )
    output_path.write_text(text, encoding="utf-8")
    return output_path
