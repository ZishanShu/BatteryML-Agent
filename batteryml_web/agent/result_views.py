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

#: One lifetime / remaining-useful-life number per **cell**. The ordinary
#: BatteryML early-cycle lifetime result: parity, residual and table views
#: are meaningful, and a RUL-vs-cycle curve does not exist, because the run
#: never estimated RUL at more than one cycle of a cell.
RUL_SCHEMA = "rul_scalar"
#: Repeated RUL estimates for the **same cell at several observed cycles** —
#: the only shape for which "remaining useful life vs cycle" is a real
#: measurement rather than a drawing. Requires RUL task semantics *and*
#: genuine per-cell, per-cycle identity; see :func:`schema_evidence`.
RUL_CYCLEWISE_SCHEMA = "rul_cyclewise"
TRAJECTORY_SCHEMA = "trajectory"
#: A scalar per *cycle sample* rather than per cell: the shape a cycle-level
#: task such as SOH estimation produces. Structurally it is still one number
#: per row, so it is told apart from RUL by the ``sample_index`` the
#: evaluation step records beside the predictions — never by guessing from
#: the numbers themselves.
SOH_SCHEMA = "soh_cycle"
#: One SOH number per **cell**, from the whole-cell ``SOHLabelAnnotator``.
#: Not a cycle-wise SOH result and not an RUL result: naming it either would
#: put it under a view that answers a different question.
SOH_SCALAR_SCHEMA = "soh_scalar"
UNKNOWN_SCHEMA = "unknown"

#: The cycle index ``SampleIndex.for_whole_cells`` writes for a row that is
#: about a cell rather than a cycle. Mirrored from
#: ``batteryml.data.sample_index`` rather than imported, so this module keeps
#: working against an artifact unpickled without the training stack present.
WHOLE_CELL_CYCLE = -1

#: Task labels as ``Pipeline._task_metadata`` records them, folded to the two
#: families this module distinguishes.
_TASK_FAMILY_RUL = "RUL"
_TASK_FAMILY_SOH = "SOH"

#: Label annotator class name -> task family. The annotator *is* what the run
#: trained against, so it cannot disagree with the task the way a filename or
#: a model name can. Mirrors ``config_task.TASK_BY_LABEL_ANNOTATOR``.
_ANNOTATOR_FAMILIES = {
    "rullabelannotator": _TASK_FAMILY_RUL,
    "sohlabelannotator": _TASK_FAMILY_SOH,
    "cyclesohlabelannotator": _TASK_FAMILY_SOH,
}

#: Said wherever a scalar-per-cell RUL result is shown, if anything is said at
#: all. Deliberately a caption rather than a warning: nothing is wrong with
#: such a run, and there is nothing for the reader to fix.
RUL_SCALAR_CURVE_NOTE = (
    "This run produces one RUL/lifetime estimate per cell, so a cycle-wise "
    "RUL curve is not available."
)

#: Said on a cycle-wise RUL curve.
RUL_CURVE_NOTE = (
    "Remaining useful life against the cycle each estimate was made at. Both "
    "series are the values the artifact stores: no remaining life is "
    "reconstructed from a cell's final lifetime, and points from different "
    "cells are never joined."
)

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
    "Diagnostic only. SOH is a ratio to the recorded reference capacity: a "
    "value below 0 is not physical, while a value above 1.0 can occur when a "
    "later cycle exceeds the reference and is not by itself invalid. Values "
    "are counted here and left exactly as the model produced them: nothing "
    "is clipped, rescaled or excluded, and the reported metrics are computed "
    "from these same values."
)


def _to_list(values: Any, *, finite_only: bool = True) -> Optional[List[float]]:
    """A flat list of floats from a tensor, array or sequence.

    Returns None for anything that is not a 1-D sequence of numbers, so a
    structure this module does not understand produces no figure rather than
    a misleading one. With ``finite_only`` (the default) a single NaN or inf
    also returns None; the view builders pass ``finite_only=False`` and then
    omit non-finite *rows* through one shared mask, counted and reported
    (:func:`_paired_series`), instead of hiding the whole result.
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
        if finite_only and not math.isfinite(number):
            return None
        flat.append(number)
    return flat


#: The label transformation was recorded and inverted exactly once.
TRANSFORM_INVERTED = "inverted"
#: The bundle records that no label transformation was used, so the stored
#: values already are the target values.
TRANSFORM_NONE = "none_recorded"
#: The bundle does not say whether a transformation was used. Nothing is
#: plotted in physical units from such an artifact.
TRANSFORM_MISSING = "metadata_missing"
#: A transformation is recorded but its inverse raised. The stored values are
#: in transformed space and are never labelled as physical units.
TRANSFORM_FAILED = "inverse_failed"


def transformation_chain(payload: Optional[Dict[str, Any]]) -> List[str]:
    """Class names of the recorded label transformation, outermost first.

    A ``SequentialDataTransformation`` is expanded into its members, in the
    order they were applied. Empty when none is recorded.
    """
    if not isinstance(payload, dict):
        return []
    transformation = getattr(payload.get("data"), "label_transformation", None)
    if transformation is None:
        return []
    members = getattr(transformation, "transformations", None)
    if isinstance(members, (list, tuple)) and members:
        return [type(member).__name__ for member in members]
    return [type(transformation).__name__]


def _invert(values: Any, payload: Dict[str, Any]) -> tuple:
    """``(values, status, detail)``: the recorded inverse, applied once.

    The single place a label transformation is undone for display. Unlike
    :func:`_inverse_transform` it never hands back untransformed values as
    if they were inverted: the status says what happened, and the view
    builders refuse a physical-unit view for :data:`TRANSFORM_MISSING` and
    :data:`TRANSFORM_FAILED`.
    """
    bundle = payload.get("data")
    if bundle is None or not hasattr(bundle, "label_transformation"):
        return values, TRANSFORM_MISSING, (
            "The saved artifact does not record whether its labels were "
            "transformed, so the stored values cannot be placed in physical "
            "units."
        )
    transformation = bundle.label_transformation
    if transformation is None:
        return values, TRANSFORM_NONE, ""
    try:
        return transformation.inverse_transform(values), TRANSFORM_INVERTED, ""
    except Exception as exc:  # noqa: BLE001 - reported as a refusal, never as data
        return values, TRANSFORM_FAILED, (
            f"The recorded label transformation ({type(transformation).__name__}) "
            f"could not be inverted ({type(exc).__name__}: {exc}). The stored "
            "values are in transformed space and are not plotted as physical "
            "units."
        )


def _paired_series(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Observed and predicted target values, paired row by row.

    Returns ``{"ok", "reason", "observed", "predicted", "positions",
    "total_rows", "omitted_nonfinite", "transform_status"}``. ``positions``
    are the artifact row numbers of the kept rows, so every other per-row
    field (cell id, cycle) is filtered with the same mask. A row is omitted
    only when its observed **or** predicted value is NaN/inf after
    inversion; the count is reported and the saved metrics are never
    recomputed on the reduced set.
    """
    import math

    out: Dict[str, Any] = {
        "ok": False, "reason": UNAVAILABLE_MESSAGE, "observed": [],
        "predicted": [], "positions": [], "total_rows": 0,
        "omitted_nonfinite": 0, "transform_status": "",
    }
    raw_labels = observed_labels(payload)
    if raw_labels is None:
        return out
    observed_raw, status, detail = _invert(raw_labels, payload)
    out["transform_status"] = status
    if status in (TRANSFORM_MISSING, TRANSFORM_FAILED):
        out["reason"] = detail
        return out
    predicted_raw, status_p, detail_p = _invert(payload.get("prediction"), payload)
    if status_p != status:
        out["reason"] = detail_p or UNAVAILABLE_MESSAGE
        return out
    observed = _to_list(observed_raw, finite_only=False)
    predicted = _to_list(predicted_raw, finite_only=False)
    if observed is None or predicted is None:
        return out
    if len(observed) != len(predicted):
        out["reason"] = (
            f"The artifact holds {len(observed)} observed value(s) and "
            f"{len(predicted)} prediction(s); they cannot be paired."
        )
        return out
    out["total_rows"] = len(observed)
    keep = [
        i for i in range(len(observed))
        if math.isfinite(observed[i]) and math.isfinite(predicted[i])
    ]
    out["omitted_nonfinite"] = len(observed) - len(keep)
    if not keep:
        out["reason"] = (
            f"All {len(observed)} row(s) have a non-finite observed or "
            "predicted value, so nothing can be plotted."
        )
        return out
    out.update(
        ok=True, reason="",
        observed=[observed[i] for i in keep],
        predicted=[predicted[i] for i in keep],
        positions=keep,
    )
    return out


def _omission_note(paired: Dict[str, Any]) -> str:
    omitted = int(paired.get("omitted_nonfinite") or 0)
    if not omitted:
        return ""
    return (
        f"{omitted} of {paired['total_rows']} row(s) have a non-finite observed "
        "or predicted value and are not plotted. The saved metrics are shown "
        "as recorded and were not recomputed without them."
    )


#: What one plotted value is, by target. ``unit`` is the canonical unit of
#: the stored (inverse-transformed) value; nothing here rescales it.
TARGET_LIFETIME = "lifetime"
TARGET_RUL = "rul"
TARGET_SOH = "soh"
TARGET_UNKNOWN = "unknown"


def target_spec(payload: Optional[Dict[str, Any]], schema: str,
                *, label_annotator: Optional[str] = None) -> Dict[str, Any]:
    """What the plotted number *is*, read from what the run recorded.

    Returns ``{"kind", "quantity", "unit", "residual_unit", "granularity",
    "definition", "reference"}``. ``label_annotator`` is an optional hint
    from saved provenance (the run's own config copy) for an artifact that
    predates ``task_metadata``; the artifact's own record wins.

    ``RULLabelAnnotator`` counts cycles from the first recorded cycle until
    discharge capacity falls to ``eol_soh`` of nominal
    (``batteryml/label/rul.py``). One such value per cell is that cell's
    **cycle life**, not the remaining life at some observation cycle, so a
    whole-cell result is named "Battery lifetime" even though the platform's
    task is called RUL prediction.
    """
    metadata = payload.get("task_metadata") if isinstance(payload, dict) else None
    metadata = metadata if isinstance(metadata, dict) else {}
    annotator = str(metadata.get("label_annotator") or label_annotator or "").strip()
    family = task_family(payload) or _ANNOTATOR_FAMILIES.get(annotator.lower())
    spec: Dict[str, Any] = {
        "kind": TARGET_UNKNOWN, "quantity": "Target value",
        "unit": "unit not recorded", "residual_unit": "unit not recorded",
        "granularity": "one point per stored row", "definition": "",
        "reference": "",
        "residual_note": "",
        "label_annotator": annotator or None,
    }
    if schema in (SOH_SCHEMA, SOH_SCALAR_SCHEMA) or family == _TASK_FAMILY_SOH:
        mode = str(metadata.get("reference_capacity_mode") or "").strip()
        spec.update(
            kind=TARGET_SOH, quantity="SOH",
            unit="ratio", residual_unit="ratio",
            residual_note=(
                "An SOH residual ×100 is in percentage points, not a relative "
                "percentage error."
            ),
            granularity=(
                "one point per observed cell and cycle"
                if schema == SOH_SCHEMA else "one point per cell"
            ),
            definition=(
                "SOH is the ratio of a cycle's discharge capacity to the "
                "recorded reference capacity"
                + (f" ({mode})." if mode else " (reference mode not recorded).")
                + " Values above 1.0 can occur when a later cycle exceeds the "
                "reference; they are shown as stored."
            ),
            reference=mode or "not recorded",
        )
        return spec
    if schema == RUL_CYCLEWISE_SCHEMA:
        spec.update(
            kind=TARGET_RUL, quantity="Remaining useful life", unit="cycles",
            residual_unit="cycles",
            granularity="one point per observed cell and cycle",
            definition="Remaining useful life at the observed cycle, as stored.",
        )
        return spec
    if annotator.lower() == "rullabelannotator":
        spec.update(
            kind=TARGET_LIFETIME, quantity="Battery lifetime", unit="cycles",
            residual_unit="cycles", granularity="one point per cell",
            definition=(
                "RULLabelAnnotator counts cycles from the first recorded cycle "
                "to end of life (discharge capacity at or below eol_soh of "
                "nominal), so each value is a cell's cycle life, not remaining "
                "life at an observation cycle."
            ),
        )
        return spec
    if family == _TASK_FAMILY_RUL:
        spec.update(
            kind=TARGET_LIFETIME, quantity="RUL/lifetime target",
            unit="cycles", residual_unit="cycles",
            granularity="one point per cell",
            definition=(
                f"Recorded task is RUL; label annotator {annotator or 'not recorded'}."
            ),
        )
    return spec


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


def task_family(payload: Optional[Dict[str, Any]]) -> Optional[str]:
    """``"RUL"``, ``"SOH"`` or None, from what the artifact records.

    Read from ``task_metadata``, which ``Pipeline.evaluate`` writes beside the
    predictions: its ``task`` when the config declared one, and otherwise the
    label annotator's class name — the component the run actually trained
    against. A model's name is never consulted: ``RidgeRULPredictor`` trains
    the SOH route in ``configs/soh/ridge/matr.yaml``, and ``XGBoostRUL...``
    the XGBoost one, so the ``RUL`` in a class name says nothing about the
    task.
    """
    if not isinstance(payload, dict):
        return None
    metadata = payload.get("task_metadata")
    if not isinstance(metadata, dict):
        return None
    declared = str(metadata.get("task") or "").strip().lower()
    if declared.startswith("soh") or "state of health" in declared:
        return _TASK_FAMILY_SOH
    if declared.startswith("rul") or "remaining useful life" in declared:
        return _TASK_FAMILY_RUL
    annotator = str(metadata.get("label_annotator") or "").strip().lower()
    return _ANNOTATOR_FAMILIES.get(annotator)


def schema_evidence(payload: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Everything the classification is allowed to look at, in one place.

    Separated from :func:`detect_schema` so the *reasoning* is inspectable: a
    view that refuses to draw a curve can say which piece of evidence was
    missing, and a test can assert on the evidence rather than only on the
    verdict.

    Returns a dict with:

    * ``task_family`` — ``"RUL"``, ``"SOH"`` or None;
    * ``rows`` — the number of scalar predictions, or None if not scalar;
    * ``has_index`` — a well-formed ``sample_index`` of the right length;
    * ``whole_cell_index`` — the index exists but every row is marked
      :data:`WHOLE_CELL_CYCLE`, i.e. one row per cell;
    * ``repeated_cells`` — at least one cell id appears more than once;
    * ``max_cycles_per_cell`` — the largest number of *distinct* observed
      cycles recorded for any one cell;
    * ``cycle_level`` — the conjunction that makes a per-cycle view
      meaningful at all: a real index, no whole-cell sentinel, a repeated
      cell, and at least two distinct cycles for some cell.

    ``cycle_level`` is deliberately not "more rows than cells" and not
    "a sample_index exists". Both of those are satisfied by artifacts that
    hold nothing per-cycle: a whole-cell RUL run carries one index record per
    cell with ``cycle_index = -1``, and would pass either test.
    """
    evidence: Dict[str, Any] = {
        "task_family": task_family(payload),
        "rows": None,
        "has_index": False,
        "whole_cell_index": False,
        "repeated_cells": False,
        "max_cycles_per_cell": 0,
        "cycle_level": False,
    }
    if not isinstance(payload, dict):
        return evidence
    # A NaN row is still a scalar row: it is omitted from figures later,
    # counted, and must not turn the whole result into "unknown".
    flat = _to_list(payload.get("prediction"), finite_only=False)
    if flat is None:
        return evidence
    evidence["rows"] = len(flat)

    index = cycle_sample_index(payload)
    if index is None or len(index) != len(flat):
        return evidence
    evidence["has_index"] = True

    cycles_by_cell: Dict[str, set] = {}
    counts: Dict[str, int] = {}
    whole_cell_rows = 0
    for record in index:
        cell = str(record["cell_id"])
        cycle = int(record["cycle_index"])
        counts[cell] = counts.get(cell, 0) + 1
        if cycle == WHOLE_CELL_CYCLE:
            whole_cell_rows += 1
            continue
        cycles_by_cell.setdefault(cell, set()).add(cycle)

    evidence["whole_cell_index"] = whole_cell_rows == len(index)
    evidence["repeated_cells"] = any(count > 1 for count in counts.values())
    evidence["max_cycles_per_cell"] = max(
        (len(cycles) for cycles in cycles_by_cell.values()), default=0
    )
    evidence["cycle_level"] = bool(
        whole_cell_rows == 0
        and evidence["repeated_cells"]
        and evidence["max_cycles_per_cell"] >= 2
    )
    return evidence


def detect_schema(payload: Optional[Dict[str, Any]]) -> str:
    """What kind of result this artifact holds.

    Two questions, answered separately and never conflated: **which task**
    the run answered, and **at what granularity** it recorded the answer.

    ============ ============ ==========================
    Task family  Cycle-level  Schema
    ============ ============ ==========================
    RUL          no           :data:`RUL_SCHEMA`
    RUL          yes          :data:`RUL_CYCLEWISE_SCHEMA`
    SOH          no           :data:`SOH_SCALAR_SCHEMA`
    SOH          yes          :data:`SOH_SCHEMA`
    unknown      no           :data:`RUL_SCHEMA`
    unknown      yes          :data:`SOH_SCHEMA`
    ============ ============ ==========================

    A sequence per sample is a trajectory, whatever the task says, and is
    never re-read as a series of per-cycle RUL estimates.

    The two "unknown" rows preserve the contract that predates
    ``task_metadata``: an artifact with no index is read as RUL, and one with
    a cycle-level index as SOH — the only cycle-level task the platform has.
    An unidentified task can therefore never produce
    :data:`RUL_CYCLEWISE_SCHEMA`, which is the point: a cycle-wise RUL curve
    is drawn only where the artifact says, in its own recorded task, that its
    numbers are remaining useful life.
    """
    if not isinstance(payload, dict):
        return UNKNOWN_SCHEMA
    prediction = payload.get("prediction")
    if prediction is None:
        return UNKNOWN_SCHEMA
    flat = _to_list(prediction, finite_only=False)
    if flat is None:
        rows = _trajectory_rows(prediction)
        return TRAJECTORY_SCHEMA if rows is not None else UNKNOWN_SCHEMA

    evidence = schema_evidence(payload)
    family = evidence["task_family"]
    if family == _TASK_FAMILY_RUL:
        return RUL_CYCLEWISE_SCHEMA if evidence["cycle_level"] else RUL_SCHEMA
    if family == _TASK_FAMILY_SOH:
        return SOH_SCHEMA if evidence["cycle_level"] else SOH_SCALAR_SCHEMA
    return SOH_SCHEMA if evidence["cycle_level"] else RUL_SCHEMA


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


#: Schemas whose rows are one scalar each, so parity, residual and table
#: views apply. Whether a *cycle curve* also applies is a separate question,
#: answered by :func:`build_rul_cycle_view` and by ``supports_cycle_curve``.
SCALAR_SCHEMAS = (RUL_SCHEMA, RUL_CYCLEWISE_SCHEMA, SOH_SCALAR_SCHEMA)


def build_rul_view(payload: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Observed, predicted and residual series for a scalar-label run.

    Returns ``{"available", "reason", "observed", "predicted", "residuals",
    "sample_ids", "count", "inverse_transformed", "note", "schema",
    "supports_cycle_curve"}``. ``available`` is False whenever the artifact
    cannot support the view, and then every series is empty — there is no
    partial figure drawn from half the data.

    Accepts every schema whose rows are one scalar each — a per-cell lifetime,
    a per-cell SOH, and the cycle-wise RUL case, where several rows belong to
    one cell. Parity and residual are meaningful for all three; the extra
    ``supports_cycle_curve`` flag is what tells the page whether the
    *additional* cycle curve exists, and it is never inferred from the
    numbers.
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
        "schema": UNKNOWN_SCHEMA,
        "supports_cycle_curve": False,
    }
    if not isinstance(payload, dict):
        return empty
    schema = detect_schema(payload)
    empty["schema"] = schema
    if schema not in SCALAR_SCHEMAS:
        return dict(empty, reason=(
            "This run's predictions are not a scalar value per sample, so an "
            "observed-versus-predicted view does not apply to it."
        ))

    if observed_labels(payload) is None:
        return dict(empty, reason=(
            "The saved artifact carries no observed labels, so predictions "
            "cannot be compared against ground truth. Nothing is plotted "
            "rather than plotting predictions against themselves."
        ))

    paired = _paired_series(payload)
    if not paired["ok"]:
        return dict(empty, reason=paired["reason"])
    observed, predicted, keep = paired["observed"], paired["predicted"], paired["positions"]
    transformed = paired["transform_status"] == TRANSFORM_INVERTED

    # Identity, filtered by the same mask as the values. A cycle-wise result
    # names the cell *and* the cycle of every row, because a bare cell id
    # would repeat and identify nothing; a whole-cell index names the cell.
    index = cycle_sample_index(payload)
    if index is not None and len(index) != paired["total_rows"]:
        index = None
    cell_ids: Optional[List[str]] = None
    if index is not None:
        cell_ids = [str(index[i]["cell_id"]) for i in keep]
    if index is not None and schema == RUL_CYCLEWISE_SCHEMA:
        ids: Optional[List[str]] = [
            f"{index[i]['cell_id']} @ cycle {index[i]['cycle_index']}" for i in keep
        ]
    elif index is not None and all(
        int(index[i]["cycle_index"]) == WHOLE_CELL_CYCLE for i in keep
    ):
        ids = list(cell_ids or [])
    else:
        all_ids = sample_ids(payload, paired["total_rows"])
        ids = [all_ids[i] for i in keep] if all_ids else None
        if all_ids and cell_ids is None:
            cell_ids = list(ids or [])

    notes = [TRANSFORM_NOTE if transformed else "", _omission_note(paired)]
    return {
        "available": True,
        "reason": "",
        "observed": observed,
        "predicted": predicted,
        # A display of two stored values, not a recomputed score.
        "residuals": [p - o for o, p in zip(observed, predicted)],
        "sample_ids": ids,
        "cell_ids": cell_ids,
        "row_positions": keep,
        "count": len(observed),
        "total_rows": paired["total_rows"],
        "omitted_nonfinite": paired["omitted_nonfinite"],
        "inverse_transformed": transformed,
        "transform_status": paired["transform_status"],
        "transform_chain": transformation_chain(payload),
        "note": " ".join(part for part in notes if part),
        "schema": schema,
        "supports_cycle_curve": schema == RUL_CYCLEWISE_SCHEMA,
        "target": target_spec(payload, schema),
    }


def build_rul_cycle_view(payload: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Remaining useful life against the cycle each estimate was made at.

    **This is the only place a RUL-vs-cycle curve comes from, and it exists
    only for an artifact that genuinely holds repeated RUL estimates for the
    same cell at several observed cycles.** A per-cell lifetime result has no
    such curve; one is never synthesised for it, and no remaining life is
    reconstructed as ``final_lifetime - cycle`` from anything.

    Returns ``{"available", "reason", "cells", "count", "n_cells",
    "inverse_transformed", "note"}``. Each entry of ``cells`` is::

        {"cell_id": str,
         "cycles": [int, ...],          # ascending, distinct rows kept
         "observed": [float, ...],      # true remaining life, as stored
         "predicted": [float, ...]}     # estimated remaining life, as stored

    Cells are separate entries precisely so a caller cannot join two cells
    into one line: there is no flat series here to draw by accident. Within a
    cell the rows are sorted by cycle, because the storage order is the order
    the extractor produced and a line drawn in that order would zig-zag.
    """
    empty: Dict[str, Any] = {
        "available": False,
        "reason": UNAVAILABLE_MESSAGE,
        "cells": [],
        "count": 0,
        "n_cells": 0,
        "inverse_transformed": False,
        "note": "",
    }
    if not isinstance(payload, dict):
        return empty
    schema = detect_schema(payload)
    if schema != RUL_CYCLEWISE_SCHEMA:
        return dict(empty, reason=(
            RUL_SCALAR_CURVE_NOTE if schema == RUL_SCHEMA else
            "This run's artifact does not contain per-cell, per-cycle "
            "remaining-useful-life estimates, so a cycle-wise RUL curve is "
            "not available for it."
        ))

    index = cycle_sample_index(payload) or []
    if observed_labels(payload) is None:
        return dict(empty, reason=(
            "The saved artifact carries no observed remaining-life labels. "
            "A true-RUL line is never reconstructed from a cell's final "
            "lifetime, so nothing is drawn."
        ))
    paired = _paired_series(payload)
    if not paired["ok"]:
        return dict(empty, reason=paired["reason"])
    if paired["total_rows"] != len(index):
        return dict(empty, reason=(
            f"The artifact holds {paired['total_rows']} observed/predicted "
            f"pair(s) and {len(index)} sample identity record(s); they cannot "
            "be paired."
        ))
    observed, predicted, keep = paired["observed"], paired["predicted"], paired["positions"]

    # Insertion order, so the cells listed are the ones the run evaluated
    # first rather than an alphabetical sample of them. ``j`` indexes the
    # kept rows; ``keep[j]`` is the artifact row, whose identity it carries.
    order: List[str] = []
    rows_by_cell: Dict[str, List[int]] = {}
    for j, position in enumerate(keep):
        cell = str(index[position]["cell_id"])
        if cell not in rows_by_cell:
            rows_by_cell[cell] = []
            order.append(cell)
        rows_by_cell[cell].append(j)

    cells: List[Dict[str, Any]] = []
    for cell in order:
        rows = sorted(rows_by_cell[cell], key=lambda j: int(index[keep[j]]["cycle_index"]))
        cells.append({
            "cell_id": cell,
            "cycles": [int(index[keep[j]]["cycle_index"]) for j in rows],
            "observed": [observed[j] for j in rows],
            "predicted": [predicted[j] for j in rows],
        })

    transformed = paired["transform_status"] == TRANSFORM_INVERTED
    notes = [RUL_CURVE_NOTE, TRANSFORM_NOTE if transformed else "", _omission_note(paired)]
    return {
        "available": True,
        "reason": "",
        "cells": cells,
        "count": len(observed),
        "total_rows": paired["total_rows"],
        "omitted_nonfinite": paired["omitted_nonfinite"],
        "n_cells": len(cells),
        "inverse_transformed": transformed,
        "note": " ".join(part for part in notes if part),
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

    if observed_labels(payload) is None:
        return dict(empty, reason=(
            "The saved artifact carries no observed SOH labels, so estimates "
            "cannot be compared against ground truth. Nothing is plotted "
            "rather than plotting predictions against themselves."
        ))

    paired = _paired_series(payload)
    if not paired["ok"]:
        return dict(empty, reason=paired["reason"])
    if paired["total_rows"] != len(index):
        return dict(empty, reason=(
            f"The artifact holds {paired['total_rows']} observed/predicted "
            f"pair(s) and {len(index)} sample identity record(s); they cannot "
            "be paired."
        ))
    observed, predicted, keep = paired["observed"], paired["predicted"], paired["positions"]
    transformed = paired["transform_status"] == TRANSFORM_INVERTED

    # Every per-row field goes through the same mask as the values.
    kept_index = [index[i] for i in keep]
    cell_ids = [record["cell_id"] for record in kept_index]
    notes = [TRANSFORM_NOTE if transformed else "", _omission_note(paired)]
    return {
        "available": True,
        "reason": "",
        "observed": observed,
        "predicted": predicted,
        # A display of two stored values, not a recomputed score.
        "residuals": [p - o for o, p in zip(observed, predicted)],
        "cell_ids": cell_ids,
        "cycle_indices": [record["cycle_index"] for record in kept_index],
        # Readable row labels for the table: a real cell and a real cycle.
        "sample_ids": [
            f"{record['cell_id']} @ cycle {record['cycle_index']}"
            for record in kept_index
        ],
        "row_positions": keep,
        "count": len(observed),
        "total_rows": paired["total_rows"],
        "omitted_nonfinite": paired["omitted_nonfinite"],
        "n_cells": len(set(cell_ids)),
        "inverse_transformed": transformed,
        "transform_status": paired["transform_status"],
        "transform_chain": transformation_chain(payload),
        "note": " ".join(part for part in notes if part),
        "range": soh_range_diagnostics(predicted),
        "schema": SOH_SCHEMA,
        "target": target_spec(payload, SOH_SCHEMA),
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


def attach_artifact_context(
    view: Optional[Dict[str, Any]],
    payload: Optional[Dict[str, Any]],
    artifact: str | Path,
) -> Optional[Dict[str, Any]]:
    """Record which exact artifact a view was built from, in place.

    Adds ``artifact_path`` (so the page can name the file and offer the
    export command for exactly it, and key its widgets to it) and, for an
    artifact that predates ``task_metadata``, refines ``target`` with the
    label annotator from the run's own config copy beside the artifact —
    saved provenance, never a guess. The artifact's own record still wins.
    """
    if not isinstance(view, dict):
        return view
    view["artifact_path"] = str(artifact)
    # The scores BatteryML saved, verbatim, for captions and exports. Copied,
    # never computed: this module has no formula for any of them.
    scores = payload.get("scores") if isinstance(payload, dict) else None
    view["saved_metrics"] = {
        str(name): value for name, value in (scores or {}).items()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    } if isinstance(scores, dict) else {}
    target = view.get("target")
    if isinstance(target, dict) and not target.get("label_annotator"):
        from .artifact_provenance import artifact_provenance

        try:
            hint = artifact_provenance(artifact, hash_files=False).get("label_annotator")
        except FileNotFoundError:
            hint = None
        if hint:
            view["target"] = target_spec(
                payload, str(view.get("schema") or ""), label_annotator=str(hint)
            )
    return view


def load_view(predictions_file: str | Path) -> Dict[str, Any]:
    """Read one predictions artifact and build whatever view it supports.

    ``rul_cycles`` is built for every artifact and refuses itself for all but
    a genuine cycle-wise RUL one, so a caller that draws it unconditionally
    still draws nothing where no such curve exists.
    """
    path = Path(predictions_file)
    payload = load_prediction_payload(path)
    schema = detect_schema(payload)
    views = {
        "schema": schema,
        "rul": build_rul_view(payload),
        "rul_cycles": build_rul_cycle_view(payload),
        "soh": build_soh_view(payload),
        "trajectory": build_trajectory_view(payload),
        "artifact_path": str(path),
    }
    for key in ("rul", "soh"):
        attach_artifact_context(views[key], payload, path)
    return views
