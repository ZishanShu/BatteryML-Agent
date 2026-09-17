"""Compatibility re-export of the training-layer checkpoint schema.

The schema moved to :mod:`batteryml.training_checkpoint`. It belongs to the
training layer: ``Pipeline.train`` both writes and reads these files, and
core BatteryML must not import the web application to do so — which is what
it had to do while this module owned the format.

Nothing else changed. There is still exactly one schema, one serializer and
one validator; this module only keeps the old import path working, so
existing callers and tests need no edit and no checkpoint on disk is
affected.

New code should import :mod:`batteryml.training_checkpoint` directly.
"""

from __future__ import annotations

from batteryml.training_checkpoint import (
    ALLOWED_OVERRIDES,
    BLOCKING_IDENTITY_FIELDS,
    KIND_FUTURE_VERSION,
    KIND_RESUMABLE,
    KIND_UNREADABLE,
    KIND_WEIGHTS_ONLY,
    SCHEMA_KEY,
    SCHEMA_VERSION,
    UNREADABLE_MESSAGE,
    WEIGHTS_ONLY_MESSAGE,
    build_checkpoint,
    capture_rng_state,
    classify_checkpoint,
    config_fingerprint,
    is_resumable,
    load_raw,
    restore_rng_state,
    save_checkpoint,
    split_fingerprint,
    validate_resume,
)

#: Re-exported verbatim; this list is what the old import path guarantees.
__all__ = [
    "ALLOWED_OVERRIDES",
    "BLOCKING_IDENTITY_FIELDS",
    "KIND_FUTURE_VERSION",
    "KIND_RESUMABLE",
    "KIND_UNREADABLE",
    "KIND_WEIGHTS_ONLY",
    "SCHEMA_KEY",
    "SCHEMA_VERSION",
    "UNREADABLE_MESSAGE",
    "WEIGHTS_ONLY_MESSAGE",
    "build_checkpoint",
    "capture_rng_state",
    "classify_checkpoint",
    "config_fingerprint",
    "is_resumable",
    "load_raw",
    "restore_rng_state",
    "save_checkpoint",
    "split_fingerprint",
    "validate_resume",
]
