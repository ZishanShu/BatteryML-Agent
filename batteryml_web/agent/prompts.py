from __future__ import annotations

from typing import Any, Dict, List

from core.schema_validator import OPTIONAL_FIELDS, REQUIRED_FIELDS


def _join_list(items: List[str]) -> str:
    return "\n".join(f"- {item}" for item in items)


def build_schema_columns_prompt(dataset_summary: Dict[str, Any]) -> str:
    return (
        "Describe the columns in this dataset for a BatteryML workflow. "
        "Return JSON only with keys: column_descriptions, data_quality_notes.\n\n"
        f"Columns: {dataset_summary.get('columns', [])}\n"
        f"Dtypes: {dataset_summary.get('dtypes', {})}\n"
        f"Missing values: {dataset_summary.get('missing_values', {})}\n"
        f"Preview: {dataset_summary.get('preview', [])}"
    )


def build_schema_mapping_prompt(dataset_summary: Dict[str, Any]) -> str:
    return (
        "Given the following dataset summary, map BatteryML schema fields to the best matching raw columns. "
        "Return JSON only with the structure: {\"mapping\": {field: column_name_or_empty_string}, \"rationale\": {field: explanation}}. "
        "Do not include markdown, code fences, or extra text.\n\n"
        f"Required schema fields: {REQUIRED_FIELDS}\n"
        f"Optional schema fields: {OPTIONAL_FIELDS}\n"
        f"Columns: {dataset_summary.get('columns', [])}\n"
        f"Dtypes: {dataset_summary.get('dtypes', {})}\n"
        f"Missing values: {dataset_summary.get('missing_values', {})}\n"
        f"Dataset summary: {dataset_summary}"
    )


def build_data_quality_prompt(dataset_summary: Dict[str, Any]) -> str:
    return (
        "Explain obvious data quality issues in this BatteryML dataset summary. "
        "Return JSON only with keys: issues, severity, recommended_checks.\n\n"
        f"Summary: {dataset_summary}"
    )


def build_config_explain_prompt(config_summary: Dict[str, Any]) -> str:
    return (
        "Explain this BatteryML config in simple terms. Return JSON only with keys: overview, key_components, caveats.\n\n"
        f"Config summary: {config_summary}"
    )


def build_model_task_prompt(config_summary: Dict[str, Any]) -> str:
    return (
        "Suggest whether the current BatteryML task/model pair looks reasonable and propose alternatives if needed. "
        "Return JSON only with keys: suggested_task, suggested_model, rationale.\n\n"
        f"Config summary: {config_summary}"
    )


def build_log_explain_prompt(log_text: str) -> str:
    return (
        "Explain this BatteryML run log. Return JSON only with keys: summary, likely_error, next_action.\n\n"
        f"Log excerpt:\n{log_text[:8000]}"
    )


def build_results_summary_prompt(results_summary: Dict[str, Any]) -> str:
    return (
        "Summarize these BatteryML results. Return JSON only with keys: summary, caveats, interpretation.\n\n"
        f"Results summary: {results_summary}"
    )
