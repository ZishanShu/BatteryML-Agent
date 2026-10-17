from __future__ import annotations

import copy
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict
from urllib.parse import urlsplit, urlunsplit

import requests



#: The path this client appends to the configured Base URL. The Base URL is
#: the OpenAI-compatible *root* — it already carries whatever version segment
#: the provider uses — so nothing here adds a ``/v1``.
CHAT_COMPLETIONS_PATH = "/chat/completions"


def _join_base_url(base_url: str, path: str) -> str:
    """Build the endpoint from the configured Base URL.

    The contract is that the Base URL is the API root, so
    ``https://host/api/v1`` becomes ``https://host/api/v1/chat/completions``
    and never ``.../api/v1/v1/chat/completions``. A version segment is never
    inserted, because the user's Base URL already decides it.

    Two things a user actually does are absorbed rather than failed on: a
    trailing slash, and pasting the full chat-completions URL copied out of a
    provider's documentation. The second is why the suffix is stripped before
    it is appended — otherwise that paste silently produces
    ``.../chat/completions/chat/completions``, which 404s with a message that
    explains nothing.
    """
    base = str(base_url or "").strip().rstrip("/")
    suffix = path if path.startswith("/") else f"/{path}"
    if base.lower().endswith(suffix.lower()):
        base = base[: -len(suffix)].rstrip("/")
    return f"{base}{suffix}"


def run_llm_prompt(
    prompt: str,
    *,
    base_url: str,
    api_key: str,
    model: str,
    system_prompt: str | None = None,
    timeout: int = 60,
    exchange: Dict[str, Any] | None = None,
) -> str:
    """Send one chat-completions request and return the reply text.

    ``exchange``, when given, is filled in place with what was actually sent
    and received, for an explanation record: the endpoint, the JSON body
    (model, messages, generation settings), the timeout, the HTTP status, the
    response's own ``model``/``id``/``finish_reason``/``usage`` when the
    provider returns them, and the raw ``message.content``. The request
    **headers are never copied into it** — they carry the API key — and any
    occurrence of the key in a provider's error text is replaced. Passing
    nothing changes nothing about the request.
    """
    record = exchange if exchange is not None else {}
    if not base_url or not api_key or not model:
        record["stage"] = "not_sent"
        raise ValueError("Missing Base URL, API key, or model.")
    endpoint = _join_base_url(base_url, CHAT_COMPLETIONS_PATH)
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    # ``stream`` is sent explicitly rather than left to the provider's
    # default: this client reads one whole JSON body, so a provider that
    # defaults to streaming would hand back a server-sent-event stream
    # that ``response.json()`` cannot parse. Nothing provider-specific
    # is added beyond the standard chat-completions fields.
    body = {
        "model": model,
        "messages": messages,
        "stream": False,
        "temperature": 0.2,
    }
    record.update(
        {
            "stage": "sending",
            "endpoint": _endpoint_without_query(endpoint),
            "request_body": copy.deepcopy(body),
            "timeout_s": timeout,
            "sent_at": _utc_now(),
        }
    )
    started = time.perf_counter()
    try:
        response = requests.post(
            endpoint,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json=body,
            timeout=timeout,
        )
    except Exception as exc:
        record.update(
            {
                "stage": "transport_error",
                "received_at": _utc_now(),
                "elapsed_s": round(time.perf_counter() - started, 3),
                "error": _scrub(f"{type(exc).__name__}: {exc}", api_key),
            }
        )
        raise
    record.update(
        {
            "received_at": _utc_now(),
            "elapsed_s": round(time.perf_counter() - started, 3),
            "http_status": response.status_code,
        }
    )
    if response.status_code >= 400:
        record["stage"] = "http_error"
        record["error_body"] = _scrub(response.text[:1200], api_key)
        raise RuntimeError(f"LLM API request failed with status {response.status_code}: {response.text[:1200]}")
    try:
        data = response.json()
    except Exception:
        record["stage"] = "unparseable_body"
        record["error_body"] = _scrub(response.text[:1200], api_key)
        raise
    if isinstance(data, dict):
        record["response_meta"] = {
            "model": data.get("model"),
            "id": data.get("id"),
            "created": data.get("created"),
            "system_fingerprint": data.get("system_fingerprint"),
            "usage": data.get("usage"),
        }
    try:
        choice = data["choices"][0]
        message = choice["message"]
        content = message["content"]
    except Exception as exc:
        record["stage"] = "unexpected_format"
        record["error_body"] = _scrub(str(data)[:1200], api_key)
        raise RuntimeError(f"Unexpected LLM API response format: {data}") from exc
    record["response_meta"]["finish_reason"] = choice.get("finish_reason") if isinstance(choice, dict) else None
    record["raw_content"] = content
    # ``message.content`` only. Reasoning-style providers also return
    # ``reasoning_content`` — the model's working, not its answer — and some
    # return it alongside a perfectly good ``content``. Falling back to it
    # when ``content`` is empty would put the model's scratch work on the
    # page as if it were the reply, so an empty answer is reported as one.
    if not content or not str(content).strip():
        record["stage"] = "empty_content"
        raise RuntimeError("LLM API returned an empty response.")
    record["stage"] = "received"
    return str(content).strip()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _endpoint_without_query(endpoint: str) -> str:
    """The endpoint with any query string and userinfo removed.

    Some gateways take a key as ``?key=…`` or ``user:pass@host``; neither may
    reach a record.
    """
    parts = urlsplit(endpoint)
    host = parts.hostname or ""
    if parts.port:
        host = f"{host}:{parts.port}"
    return urlunsplit((parts.scheme, host, parts.path, "", ""))


def _scrub(text: str, api_key: str) -> str:
    """Provider text with the key, and anything key-shaped, removed."""
    from .failure_taxonomy import redact

    cleaned = str(text or "")
    if api_key:
        cleaned = cleaned.replace(api_key, "[redacted]")
    return redact(cleaned)


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
