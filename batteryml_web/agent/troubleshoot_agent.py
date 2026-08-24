from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from core.config_validator import load_and_validate_config_file
from core.schema_validator import validate_schema_mapping
from .llm_client import LLMClient

FAILURE_CATEGORIES = [
    "Dataset missing",
    "Schema mismatch",
    "Config error",
    "Memory error",
    "Training instability",
    "Evaluation error",
    "Other",
]


def _classify_log(log_text: str) -> str:
    text = (log_text or "").lower()
    if any(k in text for k in ["file not found", "no such file", "dataset missing", "not exist"]):
        return "Dataset missing"
    if any(k in text for k in ["schema", "column", "mapping", "missing_required_fields"]):
        return "Schema mismatch"
    if any(k in text for k in ["config", "yaml", "keyerror", "unsupported task", "unsupported model"]):
        return "Config error"
    if any(k in text for k in ["cuda out of memory", "out of memory", "oom"]):
        return "Memory error"
    if any(k in text for k in ["nan", "diverged", "instability", "loss exploded"]):
        return "Training instability"
    if any(k in text for k in ["metric", "evaluate", "prediction", "evaluation"]):
        return "Evaluation error"
    return "Other"


def diagnose_failure(
    training_log_text: str,
    config_path: str | Path,
    schema_mapping_path: str | Path | None = None,
    llm_client: Any | None = None,
) -> Dict[str, Any]:
    config_path = Path(config_path)
    schema_mapping_path = Path(schema_mapping_path) if schema_mapping_path else None

    if llm_client is not None and isinstance(llm_client, LLMClient) and llm_client.is_configured():
        prompt = {
            "log_excerpt": training_log_text[:6000],
            "config_path": str(config_path),
            "schema_mapping_path": str(schema_mapping_path) if schema_mapping_path else None,
            "allowed_categories": FAILURE_CATEGORIES,
            "instruction": "Return JSON only with failure_category, likely cause, suggested_fix, validator_to_rerun, safe_to_continue.",
        }
        response = llm_client.chat_completions(
            messages=[
                {"role": "system", "content": "You diagnose BatteryML workflow failures. Return JSON only."},
                {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)},
            ],
            temperature=0,
        )
        content = response["choices"][0]["message"]["content"]
        parsed = json.loads(content)
        return parsed

    category = _classify_log(training_log_text)
    if category == "Dataset missing":
        likely_cause = "The dataset path referenced by the config does not exist or is inaccessible."
        suggested_fix = "Check the dataset path in the config and ensure the processed data directory exists."
        validator = "config_validator"
        safe = False
    elif category == "Schema mismatch":
        likely_cause = "The uploaded/raw columns do not satisfy the required BatteryML schema mapping."
        suggested_fix = "Review schema mapping, ensure required fields are mapped, then rerun schema validation."
        validator = "schema_validator"
        safe = False
    elif category == "Config error":
        likely_cause = "The generated config contains unsupported or missing parameters."
        suggested_fix = "Regenerate the config from the supported template and rerun config validation."
        validator = "config_validator"
        safe = False
    elif category == "Memory error":
        likely_cause = "The model or batch setting likely exceeded available memory."
        suggested_fix = "Reduce model complexity, batch size, or switch to CPU / smaller split."
        validator = "config_validator"
        safe = False
    elif category == "Training instability":
        likely_cause = "Training likely diverged or produced invalid values."
        suggested_fix = "Lower learning rate, simplify model, or inspect data quality before rerunning training."
        validator = "config_validator"
        safe = False
    elif category == "Evaluation error":
        likely_cause = "The evaluation step failed to read predictions or compute metrics."
        suggested_fix = "Ensure training completed and rerun evaluation after verifying output artifacts."
        validator = "results_validator"
        safe = False
    else:
        likely_cause = "The failure did not match known patterns in the log."
        suggested_fix = "Inspect the full log and rerun the relevant validator before retrying."
        validator = "manual_review"
        safe = False

    return {
        "failure_category": category,
        "likely_cause": likely_cause,
        "suggested_fix": suggested_fix,
        "validator_to_rerun": validator,
        "safe_to_continue": safe,
        "source": "rules",
    }


def run_validator_after_fix(category: str, config_path: str | Path, schema_mapping_path: str | Path | None = None) -> Dict[str, Any]:
    category = category or "Other"
    if category == "Schema mismatch" and schema_mapping_path:
        mapping_path = Path(schema_mapping_path)
        if mapping_path.exists():
            mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
            report = validate_schema_mapping(mapping)
            return report
        return {"valid": False, "errors": [f"Schema mapping file not found: {mapping_path}"], "warnings": []}

    report = load_and_validate_config_file(config_path, task="RUL", output_dir=None)
    return report
