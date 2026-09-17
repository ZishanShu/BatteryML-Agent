"""Describe a result artifact safely, including the .pkl ones.

The Predictions tab used to say "parser not implemented for this format" for
every ``.pkl`` — which is almost every artifact this platform produces. That
message was wrong twice over: the platform *does* know this payload shape
(it is the one ``Pipeline.evaluate`` writes), and where it genuinely does not
know a file, "not implemented" says nothing a user can act on.

So there are two answers here and no third:

* **a recognised BatteryML prediction payload** — described through the same
  reader and the same inverse-transform the result views use
  (:mod:`batteryml_web.agent.result_views`), so a preview can never disagree
  with the figures or the metrics beside it;
* **anything else** — described *structurally*: top-level type, keys, lengths
  and tensor shapes, and nothing more. No guessed meaning, no field renamed
  into something familiar, no numbers presented as predictions.

Safety: reading a pickle executes what it references, so this only ever
unpickles through
:func:`batteryml_web.agent.results.load_prediction_payload`, the platform's
existing CPU-safe reader — a second unpickler with subtly different rules is
exactly what must not exist. Nothing here computes a metric.

No Streamlit.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from .result_views import (
    RUL_SCHEMA,
    TRAJECTORY_SCHEMA,
    build_rul_view,
    build_trajectory_view,
    detect_schema,
)
from .results import load_prediction_payload

#: Formats this platform can preview, stated so the UI can show it rather
#: than claiming something the code does not do.
SUPPORTED_PREVIEW_FORMATS: tuple[str, ...] = (
    "BatteryML prediction payloads (.pkl) written by Pipeline.evaluate — "
    "prediction, scores, data, seed",
    "CSV tables (.csv)",
    "JSON documents (.json)",
)

KIND_BATTERYML_PREDICTION = "batteryml_prediction"
KIND_UNSUPPORTED_PICKLE = "unsupported_pickle"
KIND_UNREADABLE = "unreadable"
KIND_TABLE = "table"
KIND_JSON = "json"
KIND_OTHER = "other"

UNSUPPORTED_PICKLE_MESSAGE = (
    "This .pkl artifact was found, but its structure is not a supported "
    "prediction-view format. Its structure is summarised below; nothing in "
    "it has been interpreted as a prediction."
)

UNREADABLE_MESSAGE = (
    "This .pkl artifact could not be read back. It may have been written by "
    "a different environment, or be incomplete."
)

#: Keys the BatteryML payload carries. Presence of `prediction` is what makes
#: a payload a prediction payload; the rest enrich the description.
_PAYLOAD_KEYS = ("prediction", "scores", "data", "seed")


def _shape_of(value: Any) -> Optional[str]:
    """A readable shape/length for a value, without importing torch/numpy."""
    shape = getattr(value, "shape", None)
    if shape is not None:
        try:
            return "×".join(str(int(dim)) for dim in shape) or "scalar"
        except (TypeError, ValueError):
            return str(shape)
    if isinstance(value, (list, tuple)):
        return f"{len(value)} item(s)"
    if isinstance(value, dict):
        return f"{len(value)} key(s)"
    return None


def describe_structure(payload: Any, *, max_keys: int = 40) -> Dict[str, Any]:
    """Type, keys and shapes of ``payload`` — structure only, no meaning.

    This is what an unrecognised artifact gets. Every entry is something the
    object itself reports; nothing is inferred about what a field means.
    """
    described: Dict[str, Any] = {
        "type": type(payload).__name__,
        "length": None,
        "fields": [],
    }
    if isinstance(payload, dict):
        described["length"] = len(payload)
        for key in list(payload)[:max_keys]:
            value = payload[key]
            described["fields"].append(
                {
                    "name": str(key),
                    "type": type(value).__name__,
                    "shape": _shape_of(value) or "—",
                }
            )
    elif isinstance(payload, (list, tuple)):
        described["length"] = len(payload)
        if payload:
            described["fields"].append(
                {
                    "name": "item[0]",
                    "type": type(payload[0]).__name__,
                    "shape": _shape_of(payload[0]) or "—",
                }
            )
    else:
        described["shape"] = _shape_of(payload)
    return described


def _is_prediction_payload(payload: Any) -> bool:
    """Whether this is the payload ``Pipeline.evaluate`` writes.

    Requires a dict carrying ``prediction`` *and* a shape the view layer
    recognises. A dict that merely has a key called "prediction" but holds
    something unrecognisable is treated as unsupported, not forced into a
    view it does not fit.
    """
    if not isinstance(payload, dict) or "prediction" not in payload:
        return False
    return detect_schema(payload) in (RUL_SCHEMA, TRAJECTORY_SCHEMA)


def preview_prediction_file(path: str | Path) -> Dict[str, Any]:
    """Describe one prediction artifact for the Predictions panel.

    Returns ``{"kind", "path", "message", "schema", "summary", "structure",
    "rul", "trajectory"}``. ``summary`` holds only facts read from the file:
    how many predictions, whether ground truth is present, the stored seed,
    and whether the metrics BatteryML saved are present — never a metric
    computed here.
    """
    target = Path(path)
    result: Dict[str, Any] = {
        "kind": KIND_OTHER,
        "path": str(target),
        "name": target.name,
        "message": "",
        "schema": None,
        "summary": {},
        "structure": {},
        "rul": None,
        "trajectory": None,
    }

    suffix = target.suffix.lower()
    if suffix == ".csv":
        result["kind"] = KIND_TABLE
        return result
    if suffix == ".json":
        result["kind"] = KIND_JSON
        return result
    if suffix != ".pkl":
        return result

    payload = load_prediction_payload(target)
    if payload is None:
        result["kind"] = KIND_UNREADABLE
        result["message"] = UNREADABLE_MESSAGE
        return result

    result["structure"] = describe_structure(payload)

    if not _is_prediction_payload(payload):
        result["kind"] = KIND_UNSUPPORTED_PICKLE
        result["message"] = UNSUPPORTED_PICKLE_MESSAGE
        return result

    schema = detect_schema(payload)
    result["kind"] = KIND_BATTERYML_PREDICTION
    result["schema"] = schema

    rul = build_rul_view(payload) if schema == RUL_SCHEMA else None
    trajectory = build_trajectory_view(payload) if schema == TRAJECTORY_SCHEMA else None
    result["rul"] = rul
    result["trajectory"] = trajectory

    prediction = payload.get("prediction")
    scores = payload.get("scores")
    count: Optional[int] = None
    if rul and rul.get("available"):
        count = int(rul["count"])
    elif trajectory and trajectory.get("available"):
        count = int(trajectory["count"])
    elif hasattr(prediction, "__len__"):
        try:
            count = len(prediction)
        except TypeError:
            count = None

    result["summary"] = {
        "Output kind": (
            "Scalar prediction per cell (lifetime / RUL)"
            if schema == RUL_SCHEMA
            else "Sequence per cell (degradation trajectory)"
        ),
        "Predictions": "Not available" if count is None else str(count),
        "Ground truth stored": "Yes" if _has_ground_truth(rul, trajectory) else "No",
        "Seed recorded": str(payload.get("seed")) if payload.get("seed") is not None else "Not recorded",
        # Presence only. The values themselves stay with the canonical
        # result context; this panel does not restate or recompute them.
        "Saved metrics present": "Yes" if isinstance(scores, dict) and scores else "No",
    }
    return result


def _has_ground_truth(rul: Optional[Dict[str, Any]], trajectory: Optional[Dict[str, Any]]) -> bool:
    if rul is not None:
        return bool(rul.get("available"))
    if trajectory is not None:
        return bool(trajectory.get("observed"))
    return False


def preview_many(paths: List[str]) -> List[Dict[str, Any]]:
    """Describe several artifacts, skipping none and inventing none."""
    return [preview_prediction_file(path) for path in paths]
