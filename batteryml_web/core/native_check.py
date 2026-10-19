"""Read-only verification of one saved BatteryML run (NATIVE-01).

Given a run directory and the prediction artifact BatteryML saved there,
establish from the artifact itself — never from a summary — :

* **file identity**: SHA-256 and size of ``run.json``, the configs and the
  prediction artifact, and whether they match the fingerprints an LLM
  explanation record holds for the same run;
* **what is stored**: the payload's keys, the prediction and test-label
  tensors (shape, dtype, finite count, range) and whether a sample index
  with cell ids was saved;
* **the fitted label transformation**: its classes in order and the fitted
  parameters it carries (e.g. ZScore ``_mean``/``_std``), read, never refit
  to substitute;
* **the metric replay**: the stored arrays inverse-transformed once by the
  stored transformation; RMSE/MAE/MAPE recomputed with BatteryML's own
  ``DataBundle.evaluate`` and, independently, in float64; each compared with
  the saved scores and the run manifest; targets equal to zero counted
  (MAPE's denominator);
* optionally, **test identity**: the split and labels rebuilt from the
  processed cells with the run's own config (no feature extraction, no
  training), then compared in order with the stored targets; and the label
  transformation refit on the rebuilt training labels, compared with the
  stored fitted state — as a check, never as a replacement.

Nothing here trains, predicts, calls an LLM, or writes inside the run
directory. A status is one of the strings below; a value that could not be
read is ``None`` with a ``NOT_RECORDED``/``NOT_VERIFIED`` status, never 0.

Unpickling: the artifact is loaded with the platform's CPU-safe loader, and
only after its SHA-256 matched a recorded fingerprint, or when the caller
explicitly accepts an unrecorded file it produced itself.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

SCHEMA = "batteryml-agent/native-run-check@1"
REPORT_FILENAME = "native_check.json"
CHECKS_DIRNAME = "native_checks"

MATCH = "MATCH"
MISMATCH = "MISMATCH"
NOT_RECORDED = "NOT_RECORDED"
NOT_VERIFIED = "NOT_VERIFIED"
NOT_PERFORMED = "NOT_PERFORMED"
VERIFIED = "VERIFIED"

METRICS = ("RMSE", "MAE", "MAPE")

#: Replaying BatteryML's own evaluator on the stored objects runs the same
#: float32 operations that produced the saved score, so it is expected to be
#: identical; the tolerance only absorbs a different BLAS summation order.
EVALUATOR_RTOL = 1e-6
#: The independent float64 recomputation differs from a float32 score by
#: rounding only.
INDEPENDENT_RTOL = 1e-4
#: Rebuilt labels are integers (cycle counts); a stored target went through
#: log → z-score → inverse in float32, so it is compared with a relative tolerance.
LABEL_RTOL = 1e-4


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def file_entry(path: Optional[Path], recorded: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    """Identity of one file, and its comparison with a recorded fingerprint."""
    if path is None:
        return {"path": None, "exists": False, "fingerprint_status": NOT_RECORDED}
    target = Path(path)
    entry: Dict[str, Any] = {"path": str(target), "exists": target.is_file()}
    if entry["exists"]:
        entry["size_bytes"] = target.stat().st_size
        entry["sha256"] = sha256_file(target)
    expected = (recorded or {}).get("sha256")
    if not expected:
        entry["fingerprint_status"] = NOT_RECORDED
    elif not entry["exists"]:
        entry["fingerprint_status"] = "FILE_MISSING"
    else:
        entry["recorded_sha256"] = expected
        entry["fingerprint_status"] = MATCH if entry["sha256"] == expected else MISMATCH
    return entry


def _flat(values: Any) -> Optional[List[float]]:
    if values is None:
        return None
    if hasattr(values, "detach"):
        values = values.detach().cpu().reshape(-1).tolist()
    elif hasattr(values, "tolist"):
        values = values.reshape(-1).tolist() if hasattr(values, "reshape") else values.tolist()
    try:
        return [float(v) for v in values]
    except (TypeError, ValueError):
        return None


def tensor_facts(values: Any) -> Dict[str, Any]:
    """Shape, dtype and range of a stored array, at full precision."""
    if values is None:
        return {"present": False, "status": NOT_RECORDED}
    flat = _flat(values)
    shape = getattr(values, "shape", None)
    facts: Dict[str, Any] = {
        "present": True,
        "type": type(values).__name__,
        "shape": [int(d) for d in shape] if shape is not None else None,
        "dtype": str(getattr(values, "dtype", "")) or None,
    }
    if flat is None:
        facts["status"] = "UNREADABLE"
        return facts
    finite = [v for v in flat if math.isfinite(v)]
    facts.update(
        {
            "count": len(flat),
            "finite": len(finite),
            "non_finite": len(flat) - len(finite),
            "min": min(finite) if finite else None,
            "max": max(finite) if finite else None,
            "mean": (sum(finite) / len(finite)) if finite else None,
        }
    )
    return facts


def transform_state(transform: Any) -> Optional[Dict[str, Any]]:
    """Class names in order and the fitted parameters a transformation stores."""
    if transform is None:
        return None
    state: Dict[str, Any] = {"class": type(transform).__name__}
    children = getattr(transform, "transformations", None)
    if isinstance(children, (list, tuple)):
        state["order"] = [type(child).__name__ for child in children]
        state["children"] = [transform_state(child) for child in children]
        return state
    for attribute in ("base", "_mean", "_std"):
        if hasattr(transform, attribute):
            value = getattr(transform, attribute)
            flat = _flat(value) if value is not None and not isinstance(value, (int, float)) else value
            state[attribute.lstrip("_")] = flat
    fitted = [k for k in ("mean", "std") if k in state]
    state["fitted"] = None if not fitted else all(state[k] is not None for k in fitted)
    return state


def _independent(target: Sequence[float], prediction: Sequence[float], metric: str) -> Optional[float]:
    pairs = list(zip(target, prediction))
    if not pairs:
        return None
    if metric == "RMSE":
        return math.sqrt(sum((t - p) ** 2 for t, p in pairs) / len(pairs))
    if metric == "MAE":
        return sum(abs(t - p) for t, p in pairs) / len(pairs)
    if metric == "MAPE":
        if any(t == 0 for t, _ in pairs):
            return None
        return sum(abs((t - p) / t) for t, p in pairs) / len(pairs)
    return None


def _close(a: Optional[float], b: Optional[float], rtol: float) -> Optional[bool]:
    if a is None or b is None or not math.isfinite(a) or not math.isfinite(b):
        return None
    return abs(a - b) <= rtol * max(abs(a), abs(b), 1e-12)


def replay_metrics(
    bundle: Any,
    prediction: Any,
    stored_scores: Mapping[str, Any],
    manifest_metrics: Optional[Mapping[str, Any]] = None,
    metrics: Sequence[str] = METRICS,
) -> Dict[str, Any]:
    """Recompute the saved metrics from the stored objects; compare.

    ``bundle.evaluate`` is BatteryML's evaluator from the code now imported
    (its file and hash are reported by the caller). The arrays are inverted
    exactly once, by the stored label transformation, for the independent
    float64 computation and for the reported cycle-scale ranges.
    """
    transform = getattr(bundle, "label_transformation", None)
    target_stored = getattr(getattr(bundle, "test_data", None), "label", None)
    if target_stored is None or prediction is None:
        return {"status": NOT_PERFORMED, "reason": "stored test labels or prediction missing"}
    import torch

    with torch.no_grad():
        prediction_tensor = prediction if torch.is_tensor(prediction) else torch.as_tensor(prediction)
        if transform is not None:
            target_native = transform.inverse_transform(target_stored.clone())
            prediction_native = transform.inverse_transform(prediction_tensor.clone())
        else:
            target_native, prediction_native = target_stored, prediction_tensor
    target_list = _flat(target_native) or []
    prediction_list = _flat(prediction_native) or []
    rows: Dict[str, Any] = {}
    for metric in metrics:
        stored = stored_scores.get(metric)
        stored = float(stored) if isinstance(stored, (int, float)) else None
        try:
            evaluator = float(bundle.evaluate(prediction_tensor, metric))
            evaluator_error = None
        except Exception as exc:  # noqa: BLE001 - reported, never hidden
            evaluator, evaluator_error = None, f"{type(exc).__name__}: {exc}"
        independent = _independent(target_list, prediction_list, metric)
        manifest = (manifest_metrics or {}).get(metric)
        manifest = float(manifest) if isinstance(manifest, (int, float)) else None
        rows[metric] = {
            "stored_score": stored,
            "run_manifest_value": manifest,
            "evaluator_replay": evaluator,
            "evaluator_error": evaluator_error,
            "independent_float64": independent,
            "evaluator_equals_stored": _close(evaluator, stored, EVALUATOR_RTOL),
            "independent_matches_stored": _close(independent, stored, INDEPENDENT_RTOL),
            "manifest_equals_stored": None if manifest is None or stored is None else manifest == stored,
        }
    zero_targets = sum(1 for t in target_list if t == 0)
    replayed = [r["evaluator_equals_stored"] for r in rows.values() if r["stored_score"] is not None]
    return {
        "status": VERIFIED if replayed and all(replayed) else NOT_VERIFIED,
        "pairs": len(target_list),
        "inverse_transform_applied_once": transform is not None,
        "metrics": rows,
        "mape_zero_target_count": zero_targets,
        "mape_definition": "mean(|(target - prediction) / target|), a ratio (not multiplied by 100)",
        "target_native_scale": tensor_facts(target_native),
        "prediction_native_scale": tensor_facts(prediction_native),
        "target_native_values": target_list,
        "prediction_native_values": prediction_list,
    }


def array_scale(transform: Any, replay: Mapping[str, Any]) -> Dict[str, Any]:
    """Which scale the stored arrays are on, and how that was established."""
    if transform is None:
        return {"stored_arrays": "native (no label transformation stored)", "status": NOT_VERIFIED}
    order = (transform_state(transform) or {}).get("order") or [type(transform).__name__]
    if replay.get("status") == VERIFIED:
        return {
            "stored_arrays": f"label-transformed ({' then '.join(order)}); metrics are on the inverse-transformed scale",
            "status": VERIFIED,
            "evidence": "BatteryML's evaluator, which inverts both arrays once with the stored transformation, "
                        "reproduces every stored score.",
        }
    return {"stored_arrays": None, "status": NOT_VERIFIED,
            "evidence": "The evaluator replay did not reproduce the stored scores."}


def compare_labels(
    rebuilt_ids: Sequence[str],
    rebuilt_labels: Sequence[float],
    stored_targets: Sequence[float],
    saved_ids: Optional[Sequence[str]] = None,
    rtol: float = LABEL_RTOL,
) -> Dict[str, Any]:
    """Order-sensitive comparison of rebuilt test labels with stored targets."""
    result: Dict[str, Any] = {
        "rebuilt_count": len(rebuilt_labels),
        "stored_count": len(stored_targets),
        "rebuilt_cell_ids": list(rebuilt_ids),
    }
    if len(rebuilt_labels) != len(stored_targets):
        result.update(status=MISMATCH, reason="different counts")
        return result
    worst = 0.0
    mismatches: List[Dict[str, Any]] = []
    for index, (cell, rebuilt, stored) in enumerate(zip(rebuilt_ids, rebuilt_labels, stored_targets)):
        relative = abs(rebuilt - stored) / max(abs(rebuilt), 1e-12)
        worst = max(worst, relative)
        if relative > rtol:
            mismatches.append({"index": index, "cell_id": cell, "rebuilt": rebuilt, "stored": stored})
    result["max_relative_difference"] = worst
    result["mismatches"] = mismatches
    if saved_ids is not None:
        result["saved_cell_ids_equal_rebuilt"] = list(saved_ids) == list(rebuilt_ids)
    result["status"] = MATCH if not mismatches and result.get("saved_cell_ids_equal_rebuilt", True) else MISMATCH
    return result


def compare_fitted_state(stored: Optional[Mapping[str, Any]], refit: Optional[Mapping[str, Any]], rtol: float = 1e-5) -> Dict[str, Any]:
    """Whether a refit on rebuilt training labels reproduces the stored state."""
    if not stored or not refit:
        return {"status": NOT_PERFORMED}

    def _numbers(state: Mapping[str, Any]) -> List[float]:
        out: List[float] = []
        for child in state.get("children") or [state]:
            for key in ("mean", "std"):
                values = child.get(key)
                if isinstance(values, list):
                    out.extend(values)
        return out

    a, b = _numbers(stored), _numbers(refit)
    same = len(a) == len(b) and all(_close(x, y, rtol) for x, y in zip(a, b))
    return {"status": MATCH if same else MISMATCH, "stored": a, "refit_on_rebuilt_training_labels": b,
            "note": "Refit only to check the stored state; the stored state is what the run used."}


def scientific_target(annotator: Any) -> Dict[str, Any]:
    """What the label annotator computes, with its parameters and source hash.

    The definition is written for the annotator classes this repository
    ships, from their source; anything else is NOT_RECORDED rather than
    guessed from a class name.
    """
    name = type(annotator).__name__
    entry: Dict[str, Any] = {"label_annotator": name}
    try:
        source = inspect.getsourcefile(type(annotator))
        entry["source_file"] = source
        entry["source_sha256"] = sha256_file(Path(source)) if source else None
    except (TypeError, OSError):
        entry["source_file"] = entry["source_sha256"] = None
    if name == "RULLabelAnnotator":
        params = {k: getattr(annotator, k, None) for k in ("eol_soh", "pad_eol", "min_rul_limit")}
        entry.update(
            {
                "name": "total cycle life",
                "unit": "cycles",
                "parameters": params,
                "definition": (
                    "Counted from the first recorded cycle (label starts at 1 and increases by one per cycle) "
                    f"until the cycle whose maximum discharge capacity is <= {params['eol_soh']} x nominal "
                    "capacity; a cell that never reaches it gets its count + 1 when pad_eol is true; labels "
                    f"<= {params['min_rul_limit']} become NaN and are excluded. No observation point is "
                    "subtracted."
                ),
                "status": "FROM_IMPLEMENTATION_SOURCE",
            }
        )
    elif name == "CycleSOHLabelAnnotator":
        params = {k: getattr(annotator, k, None) for k in
                  ("reference_capacity_mode", "reference_n_cycles", "min_cycle", "max_cycle", "cycle_stride")}
        entry.update(
            {
                "name": "state of health at an observed cycle",
                "unit": "ratio",
                "parameters": params,
                "definition": "Q_discharge(cell, t) / Q_ref(cell) for each sampled observed cycle t.",
                "status": "FROM_IMPLEMENTATION_SOURCE",
            }
        )
    else:
        entry.update({"name": None, "unit": None, "status": NOT_RECORDED})
    return entry


def default_output_dir(run_dir: Path, stamp: str) -> Path:
    """``<project>/native_checks/<run id>/<stamp>``; the run dir is never written."""
    run_dir = Path(run_dir)
    return run_dir.parent.parent / CHECKS_DIRNAME / run_dir.name / stamp


def find_reports(project_root: Path, run_id: str) -> List[Path]:
    folder = Path(project_root) / CHECKS_DIRNAME / run_id
    return sorted(folder.glob(f"*/{REPORT_FILENAME}")) if folder.is_dir() else []


def matching_report(project_root: Path, run_id: str, predictions_sha256: Optional[str]) -> Optional[Dict[str, Any]]:
    """The newest report for this run made from exactly this prediction file."""
    if not predictions_sha256:
        return None
    for path in reversed(find_reports(project_root, run_id)):
        try:
            report = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        files = (report.get("files") or {}) if isinstance(report, dict) else {}
        if (files.get("predictions") or {}).get("sha256") == predictions_sha256:
            report["_report_path"] = str(path)
            report["_report_sha256"] = sha256_file(path)
            return report
    return None
