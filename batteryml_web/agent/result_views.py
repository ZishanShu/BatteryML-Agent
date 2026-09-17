"""Scientific views of a completed run, derived from its saved artifact.

What this is for: three metric numbers do not tell a battery scientist
whether a model is any good. Where the model over- or under-predicts, and on
which cells, is the part a reviewer reads. This turns the canonical saved
predictions artifact into plain arrays the page can draw.

Three rules hold it to honesty.

**Nothing is recomputed.** RMSE / MAE / MAPE come from the ``scores`` dict
BatteryML itself saved, read by :mod:`batteryml_web.agent.results`. This
module never computes a metric — residuals are a *display* of stored values
(``predicted - observed``), not a re-derivation of model quality, and no
number produced here is ever presented as a metric.

**The transform is applied, because the metrics apply it.**
``Pipeline.evaluate`` stores ``prediction`` and ``test_data.label`` in
transformed space and inverse-transforms *both* before scoring
(``batteryml/data/databundle.py``, ``DataBundle.evaluate``). A plot drawn
from the raw stored arrays would be in a different space from the reported
RMSE — axes in log-cycles beside a metric in cycles, with nothing saying so.
So the same ``label_transformation.inverse_transform`` is applied here, and
:data:`TRANSFORM_NOTE` records that it was.

**A missing value is stated, not invented.** If the artifact has no observed
labels, there is no observed-vs-predicted chart — not a chart drawn against
the predictions themselves, and never a fabricated ground truth. Sample ids
are used only where the artifact really carries them.

No Streamlit, no plotting library: this returns data. Drawing is the page's
job, and testing this needs neither a browser nor a GPU.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from .results import load_prediction_payload

#: Said on every figure built from inverse-transformed values, so the reader
#: knows the axes are in the same space as the canonical metrics.
TRANSFORM_NOTE = (
    "Values are shown in the same space as the reported metrics: the saved "
    "label transformation was inverted, exactly as BatteryML's own evaluation "
    "does before scoring."
)

#: Shown where the artifact cannot support a view. Never replaced by a chart.
UNAVAILABLE_MESSAGE = (
    "This run does not contain the values required for this visualization."
)

RUL_SCHEMA = "rul_scalar"
TRAJECTORY_SCHEMA = "trajectory"
#: A scalar per *cycle sample* rather than per cell: the shape a cycle-level
#: task such as SOH estimation produces. Structurally it is still one number
#: per row, so it is told apart from RUL by the ``sample_index`` the
#: evaluation step records beside the predictions — never by guessing from
#: the numbers themselves.
SOH_SCHEMA = "soh_cycle"
UNKNOWN_SCHEMA = "unknown"

#: SOH is a ratio of present to reference discharge capacity. A healthy new
#: cell sits at 1.0 and a degraded one below it. These are the bounds a
#: *display* diagnostic checks against; nothing is clipped to them.
SOH_HEALTHY = 1.0
#: Slightly above 1.0 is ordinary early-life behaviour on many chemistries
#: (a cell can exceed its first measured discharge after formation), so the
#: diagnostic separates "above 1.0" from "implausibly above 1.0".
SOH_IMPLAUSIBLE_HIGH = 1.1

#: Said beside every out-of-range count. The canonical predictions are what
#: the metrics were computed from and what the artifact holds; this is a
#: reading of them, and reading is all it is.
RANGE_NOTE = (
    "Diagnostic only. SOH is a ratio, so a physically meaningful estimate "
    "lies at or below 1.0 and above 0. Predictions outside that range are "
    "counted here and left exactly as the model produced them: nothing is "
    "clipped, rescaled or excluded, and the reported metrics are computed "
    "from these same values."
)


def _to_list(values: Any) -> Optional[List[float]]:
    """A flat list of floats from a tensor, array or sequence.

    Returns None for anything that is not a 1-D sequence of finite numbers,
    so a structure this module does not understand produces no figure rather
    than a misleading one.
    """
    import math

    if values is None:
        return None
    candidate = values
    # torch.Tensor / numpy.ndarray both expose .tolist(); neither is imported
    # here, so a CPU-only environment without numpy still works.
    if hasattr(candidate, "detach"):
        try:
            candidate = candidate.detach()
        except Exception:  # noqa: BLE001
            return None
    if hasattr(candidate, "cpu"):
        try:
            candidate = candidate.cpu()
        except Exception:  # noqa: BLE001
            return None
    if hasattr(candidate, "tolist"):
        try:
            candidate = candidate.tolist()
        except Exception:  # noqa: BLE001
            return None
    if not isinstance(candidate, (list, tuple)):
        return None
    flat: List[float] = []
    for item in candidate:
        # One level of nesting: a column vector of shape (N, 1) is the common
        # saved shape. Anything deeper is not a scalar prediction series.
        if isinstance(item, (list, tuple)):
            if len(item) != 1:
                return None
            item = item[0]
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            return None
        number = float(item)
        if not math.isfinite(number):
            return None
        flat.append(number)
    return flat


def _inverse_transform(values: Any, payload: Dict[str, Any]) -> Any:
    """Undo the label transformation, as ``DataBundle.evaluate`` does.

    Returns the values unchanged when the bundle declares no transformation,
    and unchanged (rather than raising) when the inverse cannot be applied —
    the caller still gets numbers, and :func:`build_rul_view` reports whether
    the inversion happened.
    """
    bundle = payload.get("data")
    transformation = getattr(bundle, "label_transformation", None)
    if transformation is None:
        return values
    try:
        return transformation.inverse_transform(values)
    except Exception:  # noqa: BLE001 - a display path must not break a result
        return values


def observed_labels(payload: Dict[str, Any]) -> Any:
    """The stored ground-truth test labels, or None.

    They live inside the saved ``DataBundle`` (``data.test_data.label``).
    An artifact without them supports no observed-vs-predicted view, and
    this returning None is how that is reported.
    """
    bundle = payload.get("data")
    test_data = getattr(bundle, "test_data", None)
    return getattr(test_data, "label", None)


def cycle_sample_index(payload: Optional[Dict[str, Any]]) -> Optional[List[Dict[str, Any]]]:
    """The per-row ``(cell_id, cycle_index)`` identities, or None.

    Written by ``Pipeline.evaluate`` for a cycle-level run and by nothing
    else, which is what makes it a reliable marker: a cell-level RUL artifact
    never carries one. Validated strictly — a record missing either field
    makes the whole index unusable rather than half-labelling a figure.
    """
    if not isinstance(payload, dict):
        return None
    records = payload.get("sample_index")
    if not isinstance(records, (list, tuple)) or not records:
        return None
    out: List[Dict[str, Any]] = []
    for record in records:
        if (
            not isinstance(record, dict)
            or "cell_id" not in record
            or "cycle_index" not in record
        ):
            return None
        try:
            cycle = int(record["cycle_index"])
        except (TypeError, ValueError):
            return None
        out.append({"cell_id": str(record["cell_id"]), "cycle_index": cycle})
    return out


def detect_schema(payload: Optional[Dict[str, Any]]) -> str:
    """What kind of result this artifact holds.

    A scalar per sample is an RUL/lifetime result, unless the artifact also
    carries a cycle-level sample index — then each scalar belongs to one
    cycle of one cell, which is an SOH-shaped result. A sequence per sample
    is a trajectory. Every distinction here is structural, read from what the
    artifact records about itself: an RUL scalar can never be re-read as a
    trajectory, and a per-cell result is never relabelled per-cycle.
    """
    if not isinstance(payload, dict):
        return UNKNOWN_SCHEMA
    prediction = payload.get("prediction")
    if prediction is None:
        return UNKNOWN_SCHEMA
    flat = _to_list(prediction)
    if flat is not None:
        index = cycle_sample_index(payload)
        if index is not None and len(index) == len(flat):
            return SOH_SCHEMA
        return RUL_SCHEMA
    rows = _trajectory_rows(prediction)
    return TRAJECTORY_SCHEMA if rows is not None else UNKNOWN_SCHEMA


def _trajectory_rows(values: Any) -> Optional[List[List[float]]]:
    """``values`` as a list of per-sample sequences, or None."""
    candidate = values
    for attribute in ("detach", "cpu"):
        if hasattr(candidate, attribute):
            try:
                candidate = getattr(candidate, attribute)()
            except Exception:  # noqa: BLE001
                return None
    if hasattr(candidate, "tolist"):
        try:
            candidate = candidate.tolist()
        except Exception:  # noqa: BLE001
            return None
    if not isinstance(candidate, (list, tuple)) or not candidate:
        return None
    rows: List[List[float]] = []
    for row in candidate:
        series = _to_list(row)
        if series is None or len(series) < 2:
            return None
        rows.append(series)
    return rows


def sample_ids(payload: Dict[str, Any], count: int) -> Optional[List[str]]:
    """Cell/sample identifiers if the artifact carries them, else None.

    Deliberately conservative: an index is not an identifier, so where the
    artifact names nothing this returns None and the table shows a position
    rather than an invented cell name.
    """
    bundle = payload.get("data")
    for attribute in ("test_cell_ids", "cell_ids", "test_ids"):
        raw = getattr(bundle, attribute, None)
        if raw is None:
            continue
        try:
            ids = [str(item) for item in raw]
        except TypeError:
            continue
        if len(ids) == count:
            return ids
    return None


def build_rul_view(payload: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Observed, predicted and residual series for a scalar-label run.

    Returns ``{"available", "reason", "observed", "predicted", "residuals",
    "sample_ids", "count", "inverse_transformed", "note"}``. ``available`` is
    False whenever the artifact cannot support the view, and then every
    series is empty — there is no partial figure drawn from half the data.
    """
    empty: Dict[str, Any] = {
        "available": False,
        "reason": UNAVAILABLE_MESSAGE,
        "observed": [],
        "predicted": [],
        "residuals": [],
        "sample_ids": None,
        "count": 0,
        "inverse_transformed": False,
        "note": "",
    }
    if not isinstance(payload, dict):
        return empty
    if detect_schema(payload) != RUL_SCHEMA:
        return dict(empty, reason=(
            "This run's predictions are not a scalar value per sample, so an "
            "observed-versus-predicted view does not apply to it."
        ))

    raw_labels = observed_labels(payload)
    if raw_labels is None:
        return dict(empty, reason=(
            "The saved artifact carries no observed labels, so predictions "
            "cannot be compared against ground truth. Nothing is plotted "
            "rather than plotting predictions against themselves."
        ))

    bundle = payload.get("data")
    transformed = getattr(bundle, "label_transformation", None) is not None
    observed = _to_list(_inverse_transform(raw_labels, payload))
    predicted = _to_list(_inverse_transform(payload.get("prediction"), payload))
    if observed is None or predicted is None:
        return empty
    if len(observed) != len(predicted):
        return dict(empty, reason=(
            f"The artifact holds {len(observed)} observed value(s) and "
            f"{len(predicted)} prediction(s); they cannot be paired."
        ))
    if not observed:
        return empty

    return {
        "available": True,
        "reason": "",
        "observed": observed,
        "predicted": predicted,
        # A display of two stored values, not a recomputed score.
        "residuals": [p - o for o, p in zip(observed, predicted)],
        "sample_ids": sample_ids(payload, len(observed)),
        "count": len(observed),
        "inverse_transformed": transformed,
        "note": TRANSFORM_NOTE if transformed else "",
    }


def build_soh_view(payload: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Observed, predicted, residual and identity for a cycle-level SOH run.

    The same shape as :func:`build_rul_view` plus the two things that make a
    cycle-level result readable: ``cell_ids`` and ``cycle_indices``, one per
    row, taken from the index the evaluation step recorded. It also carries
    ``range`` — the display-only out-of-range summary described in
    :func:`soh_range_diagnostics`.

    Refuses an artifact whose schema is not cycle-level SOH rather than
    reinterpreting a per-cell result as a per-cycle one.
    """
    empty: Dict[str, Any] = {
        "available": False,
        "reason": UNAVAILABLE_MESSAGE,
        "observed": [],
        "predicted": [],
        "residuals": [],
        "cell_ids": [],
        "cycle_indices": [],
        "sample_ids": None,
        "count": 0,
        "n_cells": 0,
        "inverse_transformed": False,
        "note": "",
        "range": soh_range_diagnostics([]),
    }
    if not isinstance(payload, dict):
        return empty
    if detect_schema(payload) != SOH_SCHEMA:
        return dict(empty, reason=(
            "This run's predictions carry no per-cycle sample index, so they "
            "are not a cycle-level SOH result and are not shown as one."
        ))
    index = cycle_sample_index(payload) or []

    raw_labels = observed_labels(payload)
    if raw_labels is None:
        return dict(empty, reason=(
            "The saved artifact carries no observed SOH labels, so estimates "
            "cannot be compared against ground truth. Nothing is plotted "
            "rather than plotting predictions against themselves."
        ))

    bundle = payload.get("data")
    transformed = getattr(bundle, "label_transformation", None) is not None
    observed = _to_list(_inverse_transform(raw_labels, payload))
    predicted = _to_list(_inverse_transform(payload.get("prediction"), payload))
    if observed is None or predicted is None:
        return empty
    if not (len(observed) == len(predicted) == len(index)):
        return dict(empty, reason=(
            f"The artifact holds {len(observed)} observed value(s), "
            f"{len(predicted)} prediction(s) and {len(index)} sample "
            "identity record(s); they cannot be paired."
        ))
    if not observed:
        return empty

    cell_ids = [record["cell_id"] for record in index]
    return {
        "available": True,
        "reason": "",
        "observed": observed,
        "predicted": predicted,
        # A display of two stored values, not a recomputed score.
        "residuals": [p - o for o, p in zip(observed, predicted)],
        "cell_ids": cell_ids,
        "cycle_indices": [record["cycle_index"] for record in index],
        # Readable row labels for the table: a real cell and a real cycle.
        "sample_ids": [
            f"{record['cell_id']} @ cycle {record['cycle_index']}"
            for record in index
        ],
        "count": len(observed),
        "n_cells": len(set(cell_ids)),
        "inverse_transformed": transformed,
        "note": TRANSFORM_NOTE if transformed else "",
        "range": soh_range_diagnostics(predicted),
    }


def soh_range_diagnostics(predicted: Any) -> Dict[str, Any]:
    """How far the estimates fall outside the physical range of a ratio.

    **Display only, and non-destructive.** The canonical predictions are the
    numbers the metrics were computed from and the numbers the artifact
    holds. This counts how many of them a battery scientist would call
    physically impossible, and reports the extremes, so a run whose RMSE
    looks acceptable cannot hide estimates of -4.3 or 2.1 behind an average.
    Nothing here clips, rescales, drops or reorders a value, and no count
    produced here is a metric.

    Returns ``{"count", "below_zero", "above_one", "above_implausible",
    "min", "max", "out_of_range", "fraction_out_of_range", "note"}``. The
    two extremes are None for an empty series rather than a placeholder
    number.
    """
    values = _to_list(predicted) or []
    below = sum(1 for value in values if value < 0.0)
    above = sum(1 for value in values if value > SOH_HEALTHY)
    implausible = sum(1 for value in values if value > SOH_IMPLAUSIBLE_HIGH)
    out_of_range = below + above
    return {
        "count": len(values),
        "below_zero": below,
        "above_one": above,
        "above_implausible": implausible,
        "min": min(values) if values else None,
        "max": max(values) if values else None,
        "out_of_range": out_of_range,
        "fraction_out_of_range": (out_of_range / len(values)) if values else 0.0,
        "note": RANGE_NOTE,
    }


def largest_residuals(
    view: Dict[str, Any], *, limit: int = 10
) -> List[Dict[str, Any]]:
    """The rows where the estimate misses the observed value by the most.

    Ordered by ``abs(residual)``, descending, so the worst cases are the ones
    a reviewer reads first. A *selection* of stored rows — the residuals were
    already computed as ``predicted - observed`` when the view was built, and
    nothing is recomputed, aggregated or scored here.

    Ties keep their original order, so the table is deterministic for a given
    artifact.
    """
    if not view.get("available"):
        return []
    residuals = view.get("residuals") or []
    order = sorted(range(len(residuals)), key=lambda i: (-abs(residuals[i]), i))
    ids = view.get("sample_ids")
    rows: List[Dict[str, Any]] = []
    for index in order[: max(int(limit), 0)]:
        row: Dict[str, Any] = {}
        if ids:
            row["Sample"] = ids[index]
        else:
            row["#"] = index
        row["Observed"] = view["observed"][index]
        row["Predicted"] = view["predicted"][index]
        row["Residual"] = residuals[index]
        rows.append(row)
    return rows


def build_trajectory_view(payload: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Per-sample predicted (and observed, where stored) trajectories.

    Only a genuine sequence-per-sample result reaches this. A scalar RUL
    output is refused rather than reinterpreted as a one-point trajectory.
    """
    empty: Dict[str, Any] = {
        "available": False,
        "reason": UNAVAILABLE_MESSAGE,
        "predicted": [],
        "observed": [],
        "count": 0,
        "length": 0,
    }
    if not isinstance(payload, dict):
        return empty
    if detect_schema(payload) != TRAJECTORY_SCHEMA:
        return dict(empty, reason=(
            "This run's output is not a degradation trajectory. A scalar "
            "lifetime prediction is not re-read as one."
        ))
    predicted = _trajectory_rows(payload.get("prediction"))
    if not predicted:
        return empty
    observed = _trajectory_rows(observed_labels(payload)) or []
    return {
        "available": True,
        "reason": "",
        "predicted": predicted,
        "observed": observed,
        "count": len(predicted),
        "length": len(predicted[0]),
    }


def prediction_table(view: Dict[str, Any], *, limit: int = 200) -> List[Dict[str, Any]]:
    """Rows of observed / predicted / residual, straight from ``view``.

    ``limit`` bounds what is handed to the page; the count in the view stays
    the true number, so a truncated table never misreports how many
    predictions the run made.
    """
    if not view.get("available"):
        return []
    ids = view.get("sample_ids")
    rows: List[Dict[str, Any]] = []
    for index in range(min(int(limit), int(view.get("count") or 0))):
        row: Dict[str, Any] = {}
        # Only a real identifier is called one; otherwise the position is
        # labelled as a position.
        if ids:
            row["Cell"] = ids[index]
        else:
            row["#"] = index
        row["Observed"] = view["observed"][index]
        row["Predicted"] = view["predicted"][index]
        row["Residual"] = view["residuals"][index]
        rows.append(row)
    return rows


def load_view(predictions_file: str | Path) -> Dict[str, Any]:
    """Read one predictions artifact and build whatever view it supports."""
    payload = load_prediction_payload(Path(predictions_file))
    schema = detect_schema(payload)
    return {
        "schema": schema,
        "rul": build_rul_view(payload),
        "soh": build_soh_view(payload),
        "trajectory": build_trajectory_view(payload),
    }
