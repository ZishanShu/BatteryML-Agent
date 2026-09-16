"""One vocabulary for "something went wrong", across execution and the LLM.

The page should tell a battery scientist what happened and what they can do,
without making them read a Python traceback to find out — and without hiding
the traceback from someone who wants it.

Design rules, all of them about not guessing:

* **Execution failures are not re-classified here.**
  :func:`batteryml_web.agent.run_policy.classify_execution_failure` already
  recognises CUDA OOM, an unavailable CUDA device, a failed start, a timeout
  and a non-zero exit, and it never retries. This module maps that verdict
  into the common shape rather than re-deriving it from the same text with
  slightly different markers, which is how two classifiers drift apart.
* **Only conservative evidence classifies.** A category is assigned from a
  specific marker — a CUDA OOM message, an HTTP 401, a JSON parse failure —
  never from a vague string that merely co-occurs. Anything unrecognised is
  :data:`CATEGORY_UNKNOWN`, which is a useful answer: it says "we do not
  know", and the technical details are still there.
* **Nothing is auto-repaired.** No category's recommended action is taken
  automatically. Training is never retried, a device is never silently
  swapped, and a parameter is never quietly reduced. The recommendation is
  text for the user to act on.

Secrets: :func:`redact` removes anything key-shaped from text that reaches a
message, and the structured fields are built from categories rather than from
raw request bodies, so a credential has no route into the UI or a log.

No Streamlit, no network.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Optional

from .run_policy import classify_execution_failure

# --- categories -----------------------------------------------------------

CATEGORY_NONE = "none"
CATEGORY_CUDA_OOM = "cuda_out_of_memory"
CATEGORY_CUDA_UNAVAILABLE = "cuda_device_unavailable"
CATEGORY_DATASET_MISSING = "dataset_unavailable"
CATEGORY_MODEL_ASSETS = "model_assets_missing"
CATEGORY_CONFIG_INVALID = "config_invalid"
CATEGORY_PROCESS_FAILED = "process_failed"
CATEGORY_CANCELLED = "cancelled_by_user"
CATEGORY_LLM_NOT_CONFIGURED = "llm_not_configured"
CATEGORY_LLM_AUTH = "llm_authentication_failed"
CATEGORY_LLM_TIMEOUT = "llm_timeout_or_network"
CATEGORY_LLM_INVALID_RESPONSE = "llm_invalid_response"
CATEGORY_UNKNOWN = "unclassified"

#: Patterns that identify an authentication problem. Deliberately narrow:
#: a 401/403, or an explicit mention of an invalid key.
_AUTH_MARKERS = (
    re.compile(r"\b401\b"),
    re.compile(r"\b403\b"),
    re.compile(r"unauthor", re.IGNORECASE),
    re.compile(r"invalid[_ ]?api[_ ]?key", re.IGNORECASE),
    re.compile(r"authentication", re.IGNORECASE),
)

_TIMEOUT_MARKERS = (
    re.compile(r"timed? ?out", re.IGNORECASE),
    re.compile(r"\bconnection (refused|reset|error|aborted)\b", re.IGNORECASE),
    re.compile(r"name or service not known", re.IGNORECASE),
    re.compile(r"temporary failure in name resolution", re.IGNORECASE),
    re.compile(r"\b(502|503|504)\b"),
)

#: Anything resembling a bearer token or API key, removed before display.
_SECRET_PATTERN = re.compile(
    r"(sk-[A-Za-z0-9_\-]{6,}|Bearer\s+[A-Za-z0-9._\-]{8,}|"
    r"api[_-]?key[\"'\s:=]+[A-Za-z0-9._\-]{6,})",
    re.IGNORECASE,
)

REDACTED = "[redacted]"


def redact(text: Any) -> str:
    """Remove anything key-shaped from text destined for a user or a log."""
    return _SECRET_PATTERN.sub(REDACTED, str(text or ""))


def _finding(
    category: str,
    short_title: str,
    user_message: str,
    recommended_action: str,
    *,
    technical_details_reference: Optional[str] = None,
    evidence_line: Optional[str] = None,
    retried: bool = False,
) -> Dict[str, Any]:
    return {
        "category": category,
        "short_title": short_title,
        "user_message": redact(user_message),
        "recommended_action": redact(recommended_action),
        "technical_details_reference": technical_details_reference,
        "evidence_line": redact(evidence_line) if evidence_line else None,
        # Always False for execution. Recorded so the page can state it.
        "retried": bool(retried),
    }


def no_failure() -> Dict[str, Any]:
    return _finding(CATEGORY_NONE, "", "", "")


# --- execution ------------------------------------------------------------

#: How run_policy's kinds map onto this vocabulary. A kind with no entry
#: becomes CATEGORY_UNKNOWN rather than being forced into a neighbour.
_EXECUTION_CATEGORY: Dict[str, str] = {
    "cuda_oom": CATEGORY_CUDA_OOM,
    "cuda_unavailable": CATEGORY_CUDA_UNAVAILABLE,
    "not_started": CATEGORY_PROCESS_FAILED,
    "timeout": CATEGORY_PROCESS_FAILED,
    "process_failed": CATEGORY_PROCESS_FAILED,
}

_EXECUTION_TITLE: Dict[str, str] = {
    CATEGORY_CUDA_OOM: "GPU memory is insufficient for this run.",
    CATEGORY_CUDA_UNAVAILABLE: "The selected GPU is not available.",
    CATEGORY_PROCESS_FAILED: "The run did not finish.",
    CATEGORY_UNKNOWN: "The run failed for an unrecognised reason.",
}


def classify_run_failure(
    execution_result: Optional[Dict[str, Any]],
    *,
    device: str = "cpu",
    inventory: Optional[Dict[str, Any]] = None,
    smoke: bool = False,
    status: Optional[str] = None,
    log_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Classify one finished run.

    A run the user cancelled is reported as cancelled and is never dressed up
    as a crash — the process exited because it was asked to.
    """
    if str(status or "").upper() == "CANCELLED":
        return _finding(
            CATEGORY_CANCELLED,
            "Cancelled by user.",
            "This run was stopped before it finished, so it produced no result.",
            "Start a new run when you are ready. Any files it had already "
            "written are kept in the run directory and are partial.",
            technical_details_reference=log_path,
        )

    verdict = classify_execution_failure(
        execution_result, device=device, inventory=inventory, smoke=smoke
    )
    kind = str(verdict.get("kind") or "none")
    if kind == "none":
        return no_failure()

    category = _EXECUTION_CATEGORY.get(kind, CATEGORY_UNKNOWN)
    actions = [str(item) for item in (verdict.get("actions") or [])]
    return _finding(
        category,
        _EXECUTION_TITLE.get(category, _EXECUTION_TITLE[CATEGORY_UNKNOWN]),
        str(verdict.get("headline") or ""),
        # The existing policy's own suggestions, offered — never taken.
        "\n".join(f"- {action}" for action in actions)
        or "Open the technical details and the run log to see what the run reported.",
        technical_details_reference=log_path,
        evidence_line=verdict.get("evidence_line"),
        retried=bool(verdict.get("retried")),
    )


# --- readiness ------------------------------------------------------------


def classify_readiness_block(
    *,
    dataset_status: Optional[str] = None,
    model_status: Optional[str] = None,
    validation_errors: Optional[list] = None,
) -> Dict[str, Any]:
    """Why a plan cannot run, in the same vocabulary as a failure.

    Reported from the backend's own readiness verdicts; this decides nothing
    about readiness itself.
    """
    if validation_errors:
        first = str(validation_errors[0])
        return _finding(
            CATEGORY_CONFIG_INVALID,
            "The configuration was refused by the validator.",
            first,
            "Fix the configuration the validator names, then rebuild the plan. "
            "Nothing is executed until it passes.",
        )
    status = str(dataset_status or "").upper()
    if status in {"RAW_ONLY", "MISSING", "EMPTY"}:
        return _finding(
            CATEGORY_DATASET_MISSING,
            "The dataset is not ready for training.",
            f"The dataset reports status {status}, so training data is not "
            "available in processed form.",
            "Run preprocessing for this dataset before running a model. "
            "Training reads data/processed/<DATASET>, never data/raw.",
        )
    model = str(model_status or "").upper()
    if model in {"NEEDS_ASSETS", "NEEDS_DATA"}:
        return _finding(
            CATEGORY_MODEL_ASSETS,
            "This model is missing the assets it needs.",
            f"The model reports status {model}.",
            "Provide the assets the readiness report names. The model stays "
            "non-executable until they are present.",
        )
    return no_failure()


# --- the LLM --------------------------------------------------------------


def classify_llm_failure(
    error: Any,
    *,
    configured: bool = True,
    response_text: Optional[str] = None,
) -> Dict[str, Any]:
    """Classify an LLM problem, conservatively.

    ``configured=False`` is not an error at all in the platform's terms — the
    assistant is optional and every page works without it — so it is reported
    as a configuration state with an action, not as a fault.
    """
    if not configured:
        return _finding(
            CATEGORY_LLM_NOT_CONFIGURED,
            "The assistant is not configured.",
            "No LLM endpoint is set, so assistance is unavailable. The "
            "deterministic workflow is unaffected and remains fully usable.",
            "Add a Base URL, API key and model in Settings, then press "
            "Connect. Nothing else about the platform depends on it.",
        )
    if error is None and response_text is not None:
        return _finding(
            CATEGORY_LLM_INVALID_RESPONSE,
            "The assistant's reply could not be used.",
            "The endpoint replied, but not with the structured answer this "
            "step requires, so the reply was discarded.",
            "Try again, or continue without assistance — the deterministic "
            "workflow does not need it. A model that cannot return JSON "
            "reliably is a poor fit for this step.",
        )

    text = redact(error)
    if any(pattern.search(text) for pattern in _AUTH_MARKERS):
        return _finding(
            CATEGORY_LLM_AUTH,
            "The assistant rejected the credentials.",
            "The endpoint refused the request as unauthenticated or "
            "unauthorized.",
            "Check the API key and Base URL in Settings, then press Connect. "
            "This is not retried automatically: repeating a rejected "
            "credential can lock an account.",
            evidence_line=text[:400],
        )
    if any(pattern.search(text) for pattern in _TIMEOUT_MARKERS):
        return _finding(
            CATEGORY_LLM_TIMEOUT,
            "The assistant could not be reached.",
            "The request timed out or the endpoint could not be contacted.",
            "Check the endpoint and the network, then try again. The "
            "deterministic workflow continues to work offline.",
            evidence_line=text[:400],
        )
    return _finding(
        CATEGORY_UNKNOWN,
        "The assistant failed for an unrecognised reason.",
        "The request did not succeed and the cause was not recognised, so it "
        "is not being guessed at.",
        "Open the details below and try again, or continue without "
        "assistance.",
        evidence_line=text[:400],
    )


def explanation_failure_is_not_a_run_failure(finding: Dict[str, Any]) -> bool:
    """Whether ``finding`` concerns the assistant rather than the run.

    Used by the page to keep a computed result intact when only its optional
    explanation failed. A run's status is never changed because an
    explanation could not be produced.
    """
    return str(finding.get("category", "")).startswith("llm_")
