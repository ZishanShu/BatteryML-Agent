from __future__ import annotations

from typing import Any, Dict, List

REQUIRED_FIELDS = [
    "cycle_index",
    "voltage",
    "current",
    "discharge_capacity",
]

OPTIONAL_FIELDS = [
    "timestamp",
    "charge_capacity",
    "temperature",
    "cell_id",
]

ALL_SCHEMA_FIELDS = REQUIRED_FIELDS + OPTIONAL_FIELDS


def validate_schema_mapping(mapping: Dict[str, Any]) -> Dict[str, Any]:
    """Validate a user-provided schema mapping against required BatteryML fields."""
    mapping = mapping or {}
    mapped_fields = mapping.get("mapping", mapping)

    missing_required_fields: List[str] = []
    invalid_targets: Dict[str, str] = {}

    for field in REQUIRED_FIELDS:
        target = mapped_fields.get(field)
        if not target:
            missing_required_fields.append(field)
        elif not isinstance(target, str):
            invalid_targets[field] = "Target column must be a string."

    for field in mapped_fields:
        if field not in ALL_SCHEMA_FIELDS:
            invalid_targets[field] = "Unknown BatteryML schema field."

    valid = not missing_required_fields and not invalid_targets
    return {
        "valid": valid,
        "missing_required_fields": missing_required_fields,
        "invalid_targets": invalid_targets,
        "required_fields": REQUIRED_FIELDS,
        "optional_fields": OPTIONAL_FIELDS,
        "details": "Schema mapping is valid." if valid else "Schema mapping failed validation.",
    }
