"""A traceable record of one LLM result explanation (EXPL-01).

Review Results → **Explain metrics** sends a prompt built from deterministic
run facts (:mod:`.result_analysis`) to the configured OpenAI-compatible
model and shows the parsed answer. Until this module, the answer lived only
in the browser session: nothing on disk said which run it explained, what
the model was actually sent, which model answered, or with what settings.

A record here is written for **every** click that reaches the request path,
including a failed one, and contains:

* the result's identity — project, run id, run directory, the page's
  identity key — and the run's own manifest fields (dataset, task, model,
  seed, device, status, the commit the run was trained at);
* the code — the commit and dirty state of the checkout that sent the
  request;
* the files — path, size and SHA-256 of the run config, the configs
  BatteryML saved in the workspace, the selected prediction artifact and
  ``run.json``;
* the call — endpoint (no query string or userinfo), provider label,
  requested model, the response's own model identifier when the provider
  returns one, response id, finish reason, token usage, timing, and the
  generation settings exactly as sent (``temperature``, ``stream``, the
  timeout) with an explicit list of settings that were **not** sent;
* the exact ``messages`` array sent, and the deterministic analysis context
  the user prompt was built from;
* the raw reply text, the parsed JSON, the markdown the page showed — or,
  on failure, the stage it failed at and the redacted error.

**Never stored:** the API key or any request header. The record is checked
for the key before it is written, and refused if the key appears.

Nothing here talks to a model or recomputes a metric. Writing is the only
side effect, under ``<project>/llm_explanations/<run id>/<record id>/``,
outside the run directory so run history and bundles are unchanged.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

SCHEMA = "batteryml-agent/llm-explanation-record@1"

#: The action this record documents, as named on screen.
EXPLAIN_METRICS_ACTION = "Review Results → Explain metrics"

#: Folder, under the project root, that holds every explanation record.
RECORDS_DIRNAME = "llm_explanations"
RECORD_FILENAME = "record.json"

#: Chat-completions settings this client does not send, so a reader knows
#: the provider's defaults applied to them rather than assuming a value.
NOT_SENT_SETTINGS = (
    "max_tokens",
    "top_p",
    "seed",
    "presence_penalty",
    "frequency_penalty",
    "stop",
    "response_format",
)

OUTCOME_ANSWER = "answer"
OUTCOME_FAILED = "failed"


class RecordLeakError(ValueError):
    """The record would contain the API key; it is not written."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def new_record_id(now: Optional[datetime] = None) -> str:
    stamp = (now or datetime.now(timezone.utc)).strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}_{uuid.uuid4().hex[:8]}"


def file_identity(path: Any) -> Dict[str, Any]:
    """Path, existence, size, modification time and SHA-256 of one file."""
    if not path:
        return {"path": None, "exists": False}
    target = Path(str(path))
    entry: Dict[str, Any] = {"path": str(target), "exists": target.is_file()}
    if not entry["exists"]:
        return entry
    digest = hashlib.sha256()
    with target.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    stat = target.stat()
    entry.update(
        {
            "size_bytes": stat.st_size,
            "modified_utc": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(timespec="seconds"),
            "sha256": digest.hexdigest(),
        }
    )
    return entry


def code_identity(repo_root: Path) -> Dict[str, Any]:
    """Commit and dirty state of the checkout, or a stated reason it is unknown."""

    def _git(*args: str) -> tuple[int, str, str]:
        try:
            done = subprocess.run(
                ["git", *args], cwd=str(repo_root), capture_output=True, text=True, timeout=10, check=False
            )
        except (OSError, subprocess.SubprocessError) as exc:
            return -1, "", f"{type(exc).__name__}: {exc}"
        return done.returncode, done.stdout.strip(), done.stderr.strip()

    code, head, err = _git("rev-parse", "HEAD")
    if code != 0 or not head:
        return {"commit": None, "unavailable_reason": err or f"git exited {code}"}
    code, status, err = _git("status", "--porcelain", "--untracked-files=no")
    changed = [line[3:] for line in status.splitlines() if line.strip()] if code == 0 else None
    return {
        "commit": head,
        "tracked_files_modified": None if changed is None else bool(changed),
        "modified_tracked_files": changed,
    }


def _workspace_configs(run_dir: Any) -> List[Dict[str, Any]]:
    """The ``config_*.yaml`` files BatteryML saved into the run workspace."""
    if not run_dir or not Path(str(run_dir)).is_dir():
        return []
    return [file_identity(path) for path in sorted(Path(str(run_dir)).rglob("config_*.yaml"))]


def _json_safe(value: Any) -> Any:
    """A JSON-serialisable copy; unknown objects become their ``repr``."""
    if isinstance(value, Mapping):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    if isinstance(value, float):
        return value if value == value and value not in (float("inf"), float("-inf")) else repr(value)
    try:
        import numpy as np  # local: the record module must import without numpy

        if isinstance(value, np.generic):
            return _json_safe(value.item())
    except ImportError:
        pass
    return repr(value)


def _response_model_note(response_meta: Mapping[str, Any], exchange: Mapping[str, Any]) -> Optional[str]:
    """Why the answering model is not stated, when it is not."""
    if response_meta.get("model"):
        return None
    status = exchange.get("http_status")
    if isinstance(status, int) and status < 400:
        return "The provider's response carried no `model` field; only the requested model is known."
    return "No successful response was received; the answering model is unknown."


def build_record(
    *,
    exchange: Mapping[str, Any],
    analysis_context: Mapping[str, Any],
    manifest: Mapping[str, Any],
    project_name: str,
    project_root: str,
    run_id: str,
    run_dir: str,
    identity_key: str,
    predictions_file: Optional[str],
    provider: str,
    requested_model: str,
    repo_root: Path,
    started_at: str,
    finished_at: str,
    parsed: Optional[Mapping[str, Any]] = None,
    rendered: Optional[str] = None,
    error: Optional[str] = None,
    action: str = EXPLAIN_METRICS_ACTION,
    record_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Assemble the record for one Explain-metrics click.

    ``exchange`` is what :func:`~batteryml_web.agent.llm_client.run_llm_prompt`
    filled in. ``parsed`` is the model's JSON with the platform's echoed
    contexts removed by the caller or here; ``error`` is set on failure.
    """
    request_body = dict(exchange.get("request_body") or {})
    messages = request_body.pop("messages", None)
    response_meta = dict(exchange.get("response_meta") or {})
    answer = None
    if parsed is not None:
        # The caller's echo of its own inputs is not part of the answer.
        answer = {k: v for k, v in parsed.items() if k not in {"metrics", "result_context", "analysis_context"}}

    sent = messages is not None
    outcome = OUTCOME_ANSWER if (error is None and parsed is not None) else OUTCOME_FAILED
    return {
        "schema": SCHEMA,
        "record_id": record_id or new_record_id(),
        "action": action,
        "context_version": str(analysis_context.get("context_version") or "v1"),
        "outcome": outcome,
        "result": {
            "project": project_name,
            "project_root": project_root,
            "run_id": run_id,
            "run_dir": run_dir,
            "identity_key": identity_key,
            "dataset": manifest.get("dataset"),
            "task": manifest.get("task"),
            "model_key": manifest.get("model_key"),
            "seed": manifest.get("seed"),
            "device": manifest.get("device"),
            "run_status": manifest.get("status"),
            "run_git_commit": manifest.get("git_commit"),
            "run_command": manifest.get("command"),
            "run_metrics_from_manifest": _json_safe(manifest.get("metrics")),
        },
        "code": code_identity(repo_root),
        "files": {
            "run_manifest": file_identity(Path(run_dir) / "run.json" if run_dir else None),
            "config": file_identity(manifest.get("config_path")),
            "workspace_configs": _workspace_configs(run_dir),
            "predictions": file_identity(predictions_file),
        },
        "call": {
            "started_at": started_at,
            "finished_at": finished_at,
            "sent_at": exchange.get("sent_at"),
            "received_at": exchange.get("received_at"),
            "elapsed_s": exchange.get("elapsed_s"),
            "request_sent": sent,
            "stage": exchange.get("stage"),
            "http_status": exchange.get("http_status"),
            "provider": provider,
            "endpoint": exchange.get("endpoint"),
            "requested_model": request_body.get("model") or requested_model,
            "response_model": response_meta.get("model"),
            "response_model_note": _response_model_note(response_meta, exchange),
            "response_id": response_meta.get("id"),
            "response_created": response_meta.get("created"),
            "system_fingerprint": response_meta.get("system_fingerprint"),
            "finish_reason": response_meta.get("finish_reason"),
            "usage": _json_safe(response_meta.get("usage")),
            "generation_settings_sent": {k: v for k, v in request_body.items() if k != "model"},
            "timeout_s": exchange.get("timeout_s"),
            "settings_not_sent_provider_defaults_apply": list(NOT_SENT_SETTINGS),
        },
        "request": {
            "messages": _json_safe(messages),
            "analysis_context": _json_safe(analysis_context),
        },
        "response": {
            "raw_content": exchange.get("raw_content"),
            "parsed": _json_safe(answer),
            "rendered_markdown": rendered,
        },
        "error": None if error is None else {"stage": exchange.get("stage"), "message": error,
                                             "provider_body": exchange.get("error_body")},
        "redaction": "No API key or request header is stored. The record was checked for the key before writing.",
        "human_review": {"status": "NOT STARTED", "reviewer": None, "completed_at": None},
    }


def assert_no_secret(record: Mapping[str, Any], secrets: Iterable[str]) -> None:
    """Refuse a record in which any non-empty secret appears verbatim."""
    text = json.dumps(record, ensure_ascii=False)
    for secret in secrets:
        if secret and len(secret) >= 4 and secret in text:
            raise RecordLeakError(
                "The explanation record contained the API key and was not written. "
                "Report this; nothing was saved."
            )
    if re.search(r'"authorization"\s*:', text, re.IGNORECASE):
        raise RecordLeakError("The explanation record contained an Authorization header and was not written.")


def record_dir(project_root: str | Path, run_id: str, record_id: str) -> Path:
    safe_run = re.sub(r"[^A-Za-z0-9_.-]", "_", run_id or "no_run_id")
    return Path(project_root) / RECORDS_DIRNAME / safe_run / record_id


def write_record(record: Mapping[str, Any], target_dir: Path, *, secrets: Iterable[str]) -> Path:
    """Write ``record.json`` atomically after the secret check; return its path."""
    assert_no_secret(record, list(secrets))
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / RECORD_FILENAME
    if target.exists():
        raise FileExistsError(f"{target} already exists; a record is never overwritten.")
    handle, tmp = tempfile.mkstemp(dir=str(target_dir), suffix=".tmp")
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump(record, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
        os.replace(tmp, target)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
    return target


def load_record(path: str | Path) -> Dict[str, Any]:
    """Read a record and check it is one."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict) or data.get("schema") != SCHEMA:
        raise ValueError(f"{path} is not a {SCHEMA} record.")
    return data


def list_records(project_root: str | Path) -> List[Path]:
    root = Path(project_root) / RECORDS_DIRNAME
    return sorted(root.rglob(RECORD_FILENAME)) if root.is_dir() else []
