"""Compact facts about one prediction artifact, at three boundaries.

The user reported prediction values that "appear identical or repeated", and
the cause was not knowable by reading code: it could have been a stale
selection, the wrong artifact, display rounding, a broadcasting or indexing
error, a doubled inverse transform, or a model that genuinely predicts a
constant. Those have opposite repairs, and guessing between them is how a
display bug gets "fixed" by changing the numbers.

So this module measures instead, at the three places a value can change
shape:

    saved artifact  ->  result-view data  ->  rendered table / plot

:func:`artifact_diagnostics` reads the file through the existing CPU-safe
unpickler — there is exactly one of those, and this adds no second one. It
reports what is in the file at full precision and never rounds, sorts,
rescales or repairs anything. If the saved predictions really are constant,
that is what it says, and the constant is preserved: manufacturing variation
to make a plot look healthier would turn a reporting bug into a scientific
one.

:func:`compare_boundaries` then checks that what the view layer produced
still matches the file, which is what distinguishes "the artifact is
constant" from "the artifact varies and something downstream flattened it".

Nothing here is a metric. Counts and extremes of stored values are a
description of a file, and no number produced here is ever presented as a
result.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from .results import load_prediction_payload

#: How many raw values to carry for eyeballing, before any formatting.
SAMPLE_VALUES = 8

#: Distinct-value count at or below which a series is called constant-ish.
#: One is genuinely constant; the field is reported as a number either way,
#: so no threshold decides anything on its own.
CONSTANT_DISTINCT = 1


def _describe_container(values: Any) -> Dict[str, Any]:
    """Shape and dtype as the container itself reports them.

    Read from the object rather than inferred, so a torch tensor of shape
    (N, 1) is described as (N, 1) and not as a flat list of N — the
    difference between those two is one of the candidate causes.
    """
    shape = getattr(values, "shape", None)
    if shape is not None:
        try:
            shape = tuple(int(dimension) for dimension in shape)
        except (TypeError, ValueError):
            shape = None
    if shape is None and isinstance(values, (list, tuple)):
        shape = (len(values),)
        if values and isinstance(values[0], (list, tuple)):
            shape = (len(values), len(values[0]))
    dtype = getattr(values, "dtype", None)
    return {
        "shape": shape,
        "dtype": str(dtype) if dtype is not None else type(values).__name__,
    }


def _flatten(values: Any) -> Optional[List[float]]:
    """Every scalar in ``values``, in stored order, or None.

    Order is preserved exactly. Nothing is sorted here: sorting predictions
    without carrying their labels and sample identities along is one of the
    misalignment failures this module exists to detect, so it never does it.
    """
    candidate = values
    for attribute in ("detach", "cpu"):
        if hasattr(candidate, attribute):
            try:
                candidate = getattr(candidate, attribute)()
            except Exception:  # noqa: BLE001 - a diagnostic must not raise
                return None
    if hasattr(candidate, "tolist"):
        try:
            candidate = candidate.tolist()
        except Exception:  # noqa: BLE001
            return None
    if isinstance(candidate, (int, float)) and not isinstance(candidate, bool):
        # A single scalar where a series was expected: the broadcasting case.
        return [float(candidate)]
    if not isinstance(candidate, (list, tuple)):
        return None
    flat: List[float] = []
    stack: List[Any] = list(candidate)
    while stack:
        item = stack.pop(0)
        if isinstance(item, (list, tuple)):
            stack = list(item) + stack
            continue
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            return None
        flat.append(float(item))
    return flat


def describe_series(values: Any, name: str = "") -> Dict[str, Any]:
    """Full-precision facts about one stored array.

    Returns ``{"name", "shape", "dtype", "count", "finite", "non_finite",
    "distinct", "constant", "min", "max", "sample"}``. ``min``/``max`` are
    the stored floats, unrounded; ``sample`` is the first few values exactly
    as stored, before any display formatting touches them.
    """
    described = _describe_container(values)
    flat = _flatten(values)
    if flat is None:
        return {
            "name": name,
            "shape": described["shape"],
            "dtype": described["dtype"],
            "count": 0,
            "finite": 0,
            "non_finite": 0,
            "distinct": 0,
            "constant": False,
            "min": None,
            "max": None,
            "sample": [],
            "readable": False,
        }
    finite = [value for value in flat if math.isfinite(value)]
    distinct = len(set(finite))
    return {
        "name": name,
        "shape": described["shape"],
        "dtype": described["dtype"],
        "count": len(flat),
        "finite": len(finite),
        "non_finite": len(flat) - len(finite),
        "distinct": distinct,
        # Reported, not acted on. A constant series is preserved as it is.
        "constant": bool(finite) and distinct <= CONSTANT_DISTINCT,
        "min": min(finite) if finite else None,
        "max": max(finite) if finite else None,
        "sample": flat[:SAMPLE_VALUES],
        "readable": True,
    }


def _identity_summary(records: Any) -> Dict[str, Any]:
    """Length and a couple of representative ``(cell, cycle)`` identities."""
    if not isinstance(records, (list, tuple)) or not records:
        return {"length": 0, "representative": [], "distinct_cells": 0}
    representative: List[str] = []
    cells: set = set()
    for record in records:
        if not isinstance(record, dict):
            continue
        cell = str(record.get("cell_id"))
        cells.add(cell)
        if len(representative) < 3:
            representative.append(f"{cell} @ cycle {record.get('cycle_index')}")
    return {
        "length": len(records),
        "representative": representative,
        "distinct_cells": len(cells),
    }


def artifact_diagnostics(
    path: str | Path,
    *,
    run_id: str = "",
) -> Dict[str, Any]:
    """Read one artifact and describe exactly what is stored in it.

    Read through :func:`~batteryml_web.agent.results.load_prediction_payload`
    — the platform's single CPU-safe unpickler — so this adds no second
    loader and inherits its CUDA remapping. A file that cannot be read is
    reported as unreadable; it is never replaced by another file.
    """
    target = Path(path)
    result: Dict[str, Any] = {
        "run_id": str(run_id),
        "path": str(target),
        "readable": False,
        "reason": "",
        "prediction": None,
        "observed": None,
        "sample_index": {"length": 0, "representative": [], "distinct_cells": 0},
        "scores": {},
        "task_metadata": {},
    }
    if not target.is_file():
        result["reason"] = f"No file at {target}."
        return result
    payload = load_prediction_payload(target)
    if not isinstance(payload, dict):
        result["reason"] = (
            f"{target.name} could not be read as a BatteryML prediction payload."
        )
        return result

    result["readable"] = True
    result["prediction"] = describe_series(payload.get("prediction"), "prediction")

    bundle = payload.get("data")
    test_data = getattr(bundle, "test_data", None)
    result["observed"] = describe_series(getattr(test_data, "label", None), "observed")
    # Stated because it changes what the plotted values mean: a view applies
    # the same inverse transform BatteryML's own evaluation applies before
    # scoring, and applying it twice is one of the candidate causes.
    result["label_transformation"] = type(
        getattr(bundle, "label_transformation", None)
    ).__name__ if getattr(bundle, "label_transformation", None) is not None else None

    result["sample_index"] = _identity_summary(payload.get("sample_index"))
    scores = payload.get("scores")
    result["scores"] = dict(scores) if isinstance(scores, dict) else {}
    metadata = payload.get("task_metadata")
    result["task_metadata"] = dict(metadata) if isinstance(metadata, dict) else {}
    return result


def compare_boundaries(
    stored: Optional[Sequence[float]],
    displayed: Optional[Sequence[float]],
) -> Dict[str, Any]:
    """Whether the view's values still match the file's, value for value.

    The point of the comparison is to separate two very different findings
    that look identical on screen:

    * ``distinct_stored`` is 1 — the model really did predict one value, and
      the display is faithful. Nothing here may change that;
    * ``distinct_stored`` is many while ``distinct_displayed`` is 1 —
      something between the file and the screen flattened it, which is a bug
      in this platform.

    ``aligned`` is True only when the two series are equal element-wise and
    in the same order, so a reordering that lost its labels is visible too.
    """
    left = list(stored or [])
    right = list(displayed or [])
    distinct_stored = len({value for value in left if math.isfinite(value)})
    distinct_displayed = len({value for value in right if math.isfinite(value)})
    aligned = len(left) == len(right) and all(
        (math.isnan(a) and math.isnan(b)) or a == b for a, b in zip(left, right)
    )
    return {
        "count_stored": len(left),
        "count_displayed": len(right),
        "distinct_stored": distinct_stored,
        "distinct_displayed": distinct_displayed,
        "aligned": aligned,
        # The one conclusion worth naming, because it is the actionable one.
        "flattened_by_display": distinct_stored > 1 and distinct_displayed <= 1,
        "genuinely_constant": distinct_stored <= 1 and len(left) > 0,
    }


def diagnostics_rows(diagnostics: Dict[str, Any]) -> List[Dict[str, str]]:
    """The compact table the details area shows. Full precision throughout."""
    if not diagnostics.get("readable"):
        return [{"Field": "Artifact", "Value": str(diagnostics.get("reason") or "Unreadable")}]
    prediction = diagnostics.get("prediction") or {}
    observed = diagnostics.get("observed") or {}
    identity = diagnostics.get("sample_index") or {}
    rows = [
        {"Field": "Run ID", "Value": str(diagnostics.get("run_id") or "Not recorded")},
        {"Field": "Artifact", "Value": str(diagnostics.get("path") or "")},
        {"Field": "Prediction shape", "Value": str(prediction.get("shape"))},
        {"Field": "Prediction dtype", "Value": str(prediction.get("dtype"))},
        {
            "Field": "Finite / non-finite",
            "Value": f"{prediction.get('finite')} / {prediction.get('non_finite')}",
        },
        {"Field": "Distinct values", "Value": str(prediction.get("distinct"))},
        # repr, not a format string: the whole point is to show the stored
        # float rather than a rounded reading of it.
        {"Field": "Min (full precision)", "Value": repr(prediction.get("min"))},
        {"Field": "Max (full precision)", "Value": repr(prediction.get("max"))},
        {
            "Field": "First stored values",
            "Value": ", ".join(repr(value) for value in prediction.get("sample") or []),
        },
        {"Field": "Observed shape", "Value": str(observed.get("shape"))},
        {
            "Field": "Label transformation",
            "Value": str(diagnostics.get("label_transformation") or "None recorded"),
        },
        {"Field": "Sample identities", "Value": str(identity.get("length"))},
    ]
    if identity.get("representative"):
        rows.append(
            {
                "Field": "Example identities",
                "Value": "; ".join(identity["representative"]),
            }
        )
    return rows
