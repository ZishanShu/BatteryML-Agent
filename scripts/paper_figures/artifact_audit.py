"""Row-identity audit of one prediction artifact, plus whole-cell identity recovery.

Answers, from the artifact itself (nothing assumed about why rows repeat):

* payload keys; raw shapes of ``prediction``, ``data.test_data.label`` and
  ``data.test_data.feature``; whether ``sample_index`` / ``task_metadata`` exist;
* total rows, distinct cell ids, the distribution of rows per cell, a full
  example of a repeated cell (its records, cycle index, observed/predicted in
  label units) and whether its observed value is constant;
* the saved scores next to the same scores recomputed by calling the
  artifact's own ``DataBundle.evaluate`` (the BatteryML evaluator, exactly as
  ``Pipeline.evaluate`` called it), and next to a row-pooled recomputation;
  a shape mismatch that would make the evaluator broadcast is reported;
* optionally, how ``collect`` turned it into ``samples.csv`` rows.

``--recover-identity`` (whole-cell lifetime artifacts only) rebuilds the test
split exactly as ``batteryml/task.py`` does — the run's own config copy, its
split class over the processed data, its label annotator, the NaN-label mask —
and accepts the rebuilt cell order only if every rebuilt label equals the
artifact's own observed label (after the recorded inverse transform) within
tolerance. Only then is a sidecar written; the artifact is never modified.

Usage (repository root)::

    python -m scripts.paper_figures.artifact_audit --artifact <predictions.pkl> \\
        [--collected <collect output dir> --entry <entry id>] \\
        [--recover-identity --sidecar <new identity.json>] [--output <new audit.json>]
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

from scripts.paper_figures.common import REPO_ROOT, code_version, command_line, read_csv

SIDECAR_KIND = "fig3_identity_sidecar"
SIDECAR_VERSION = 1
#: Labels are cycle counts stored as float32 and round-tripped through the
#: recorded transformation, so a rebuilt label must match to this tolerance.
LABEL_ATOL = 1e-2
LABEL_RTOL = 1e-5


class AuditError(RuntimeError):
    """The artifact or its context cannot be audited; the message says why."""


def _shape(value: Any) -> Any:
    shape = getattr(value, "shape", None)
    if shape is not None:
        return [int(s) for s in shape]
    if isinstance(value, (list, tuple)):
        return [len(value)]
    return None if value is None else type(value).__name__


def _flat(values: Any) -> List[float]:
    from batteryml_web.agent.result_views import _to_list  # noqa: PLC2701 - same flattening as the views

    out = _to_list(values, finite_only=False)
    if out is None:
        raise AuditError("values are not a flat numeric series")
    return out


def _inverse(payload: Dict[str, Any], values: Any) -> Any:
    transformation = getattr(payload.get("data"), "label_transformation", None)
    return values if transformation is None else transformation.inverse_transform(values)


def audit_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Pure (no file access): everything the artifact says about its rows."""
    import torch

    from batteryml_web.agent.result_views import (
        WHOLE_CELL_CYCLE, cycle_sample_index, detect_schema, observed_labels,
        transformation_chain,
    )

    bundle = payload.get("data")
    test = getattr(bundle, "test_data", None)
    label = observed_labels(payload)
    prediction = payload.get("prediction")
    report: Dict[str, Any] = {
        "payload_keys": sorted(str(k) for k in payload),
        "schema": detect_schema(payload),
        "prediction_shape": _shape(prediction),
        "observed_label_shape": _shape(label),
        "test_feature_shape": _shape(getattr(test, "feature", None)),
        "has_sample_index": isinstance(payload.get("sample_index"), (list, tuple)),
        "sample_index_rows": (len(payload["sample_index"])
                              if isinstance(payload.get("sample_index"), (list, tuple)) else None),
        "task_metadata": payload.get("task_metadata"),
        "seed": payload.get("seed"),
        "saved_scores": payload.get("scores"),
        "label_transformation": transformation_chain(payload),
    }
    if label is None or prediction is None:
        report["verdict"] = "no observed labels or no predictions; nothing to pair"
        return report
    observed = _flat(_inverse(payload, label))
    predicted = _flat(_inverse(payload, prediction))
    report["total_rows"] = len(predicted)
    report["observed_rows"] = len(observed)

    # The evaluator, exactly as Pipeline.evaluate called it on these objects.
    recomputed: Dict[str, Any] = {}
    for metric in ("RMSE", "MAE"):
        try:
            recomputed[metric] = float(bundle.evaluate(prediction, metric))
        except Exception as exc:  # noqa: BLE001 - reported, never substituted
            recomputed[metric] = f"evaluator failed: {type(exc).__name__}: {exc}"
    report["evaluator_recomputed"] = recomputed
    shapes_differ = _shape(label) != _shape(prediction)
    report["evaluator_broadcast_risk"] = (
        f"observed {_shape(label)} vs prediction {_shape(prediction)}: the evaluator "
        "subtracts them elementwise, so differing shapes broadcast" if shapes_differ else "none")
    if len(observed) == len(predicted):
        pairs = [(o, p) for o, p in zip(observed, predicted) if math.isfinite(o) and math.isfinite(p)]
        if pairs:
            report["row_pooled_recomputed"] = {
                "RMSE": math.sqrt(sum((p - o) ** 2 for o, p in pairs) / len(pairs)),
                "MAE": sum(abs(p - o) for o, p in pairs) / len(pairs),
                "rows_used": len(pairs),
            }
    saved = payload.get("scores") if isinstance(payload.get("scores"), dict) else {}
    report["saved_vs_evaluator"] = {
        m: (None if not isinstance(recomputed.get(m), float) or not isinstance(saved.get(m), (int, float))
            else abs(float(saved[m]) - recomputed[m]))
        for m in ("RMSE", "MAE")
    }

    index = cycle_sample_index(payload)
    if index is None:
        report["identity"] = "not recorded: the artifact has no usable sample_index"
        report["verdict"] = (
            "rows carry no cell identity; a collector that writes a placeholder id for "
            "every row would make them look like one repeated cell")
        return report
    if len(index) != len(predicted):
        report["identity"] = (f"sample_index has {len(index)} records for {len(predicted)} "
                              "predictions; it cannot be paired")
        report["verdict"] = report["identity"]
        return report
    counts = Counter(r["cell_id"] for r in index)
    report["identity"] = "recorded"
    report["distinct_cells"] = len(counts)
    report["rows_per_cell_distribution"] = {str(k): v for k, v in sorted(Counter(counts.values()).items())}
    report["whole_cell_rows"] = sum(1 for r in index if r["cycle_index"] == WHOLE_CELL_CYCLE)
    repeated = [cell for cell, n in counts.items() if n > 1]
    report["repeated_cells"] = len(repeated)
    if repeated:
        cell = repeated[0]
        rows = [i for i, r in enumerate(index) if r["cell_id"] == cell]
        report["repeated_example"] = [
            {"row": i, **index[i], "observed": observed[i] if i < len(observed) else None,
             "predicted": predicted[i]} for i in rows[:20]]
        spread = {}
        for c in repeated:
            values = [observed[i] for i, r in enumerate(index) if r["cell_id"] == c and i < len(observed)]
            spread[c] = max(values) - min(values) if values else None
        report["observed_constant_within_repeated_cell"] = all(
            s is not None and abs(s) <= LABEL_ATOL for s in spread.values())
        report["observed_spread_by_repeated_cell"] = dict(list(spread.items())[:20])
        report["verdict"] = ("cells repeat in the recorded index; the rows are distinct "
                             "observations and are not averaged, deduplicated or truncated")
    else:
        report["verdict"] = "one row per cell"
    _ = torch  # torch is imported so tensors in the payload resolve on CPU
    return report


def samples_csv_view(collected_dir: Path, entry_id: str) -> Dict[str, Any]:
    rows = [r for r in read_csv(Path(collected_dir) / "samples.csv") if r["entry_id"] == entry_id]
    ids = [r["cell_id"] for r in rows]
    return {
        "rows": len(rows),
        "distinct_cell_id_values": len(set(ids)),
        "placeholder_ids": sum(1 for i in ids if i in ("", "NA", "None")),
        "first_rows": rows[:5],
    }


def _config_sections(config_copy: Path) -> Dict[str, Any]:
    import yaml

    raw = yaml.safe_load(Path(config_copy).read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or "train_test_split" not in raw or "label" not in raw:
        raise AuditError(f"{config_copy} lacks train_test_split/label sections.")
    return raw


def rebuild_whole_cell_labels(config_copy: Path) -> Dict[str, Any]:
    """The test cells and labels Task.build would produce (whole-cell path)."""
    import torch

    import batteryml.label  # noqa: F401 - registers label annotators
    import batteryml.train_test_split  # noqa: F401 - registers splitters
    from batteryml.builders import LABEL_ANNOTATORS, TRAIN_TEST_SPLITTERS
    from batteryml.data import BatteryData
    from batteryml.label.base import BaseCycleLabelAnnotator

    config = _config_sections(config_copy)
    splitter = TRAIN_TEST_SPLITTERS.build(dict(config["train_test_split"]), "raise")
    annotator = LABEL_ANNOTATORS.build(dict(config["label"]))
    if isinstance(annotator, BaseCycleLabelAnnotator):
        raise AuditError("identity recovery applies to whole-cell label annotators only.")
    _, test_list = splitter.split()
    cells = [BatteryData.load(path) for path in test_list]
    labels = annotator(cells)
    keep = ~torch.isnan(labels)
    return {
        "split_class": config["train_test_split"].get("name"),
        "label_annotator": config["label"].get("name"),
        "cell_ids": [str(c.cell_id) for c, k in zip(cells, keep.tolist()) if k],
        "labels": [float(v) for v in labels[keep].tolist()],
        "dropped_nan_label_cells": [str(c.cell_id) for c, k in zip(cells, keep.tolist()) if not k],
    }


def match_identity(observed: List[float], rebuilt: Dict[str, Any]) -> Dict[str, Any]:
    """Pure: accept the rebuilt order only if every label matches row by row."""
    labels = rebuilt["labels"]
    if len(labels) != len(observed):
        return {"status": "mismatch",
                "reason": f"rebuilt {len(labels)} test cells, artifact has {len(observed)} rows"}
    if len(set(rebuilt["cell_ids"])) != len(rebuilt["cell_ids"]):
        return {"status": "mismatch", "reason": "rebuilt cell ids are not unique"}
    worst = 0.0
    for i, (o, r) in enumerate(zip(observed, labels)):
        if not (math.isfinite(o) and math.isfinite(r)):
            return {"status": "mismatch", "reason": f"row {i}: non-finite label"}
        diff = abs(o - r)
        worst = max(worst, diff)
        if diff > LABEL_ATOL + LABEL_RTOL * abs(r):
            return {"status": "mismatch",
                    "reason": f"row {i}: artifact label {o} != rebuilt {r} for {rebuilt['cell_ids'][i]}"}
    return {"status": "verified", "max_abs_label_diff": worst, "rows": len(labels)}


def recover_identity(artifact: Path, payload: Dict[str, Any]) -> Dict[str, Any]:
    from batteryml_web.agent.artifact_provenance import artifact_provenance
    from batteryml_web.agent.result_views import observed_labels

    record = artifact_provenance(artifact)
    if record["config_copy"] == "not recorded":
        raise AuditError(f"{artifact.parent}: no single config copy; the run's split cannot be rebuilt.")
    config_copy = artifact.parent / record["config_copy"]
    rebuilt = rebuild_whole_cell_labels(config_copy)
    observed = _flat(_inverse(payload, observed_labels(payload)))
    verdict = match_identity(observed, rebuilt)
    return {
        "kind": SIDECAR_KIND, "version": SIDECAR_VERSION,
        "artifact": str(artifact), "artifact_sha256": record["artifact_sha256"],
        "config_copy": str(config_copy), "config_copy_sha256": record["config_copy_sha256"],
        "split_class": rebuilt["split_class"], "label_annotator": rebuilt["label_annotator"],
        "method": ("rebuilt the test split from the run's config copy over the processed data "
                   "(split class order, label annotator, NaN-label mask as in batteryml/task.py) "
                   "and matched every rebuilt label to the artifact's observed label"),
        "tolerance": {"atol": LABEL_ATOL, "rtol": LABEL_RTOL},
        "dropped_nan_label_cells": rebuilt["dropped_nan_label_cells"],
        "cell_ids": rebuilt["cell_ids"] if verdict["status"] == "verified" else [],
        **verdict,
    }


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--collected", type=Path, default=None)
    parser.add_argument("--entry", default=None)
    parser.add_argument("--recover-identity", action="store_true")
    parser.add_argument("--sidecar", type=Path, default=None,
                        help="Where to write a verified identity sidecar (must not exist).")
    parser.add_argument("--output", type=Path, default=None, help="Audit JSON (must not exist).")
    args = parser.parse_args(argv)
    from batteryml_web.agent.results import load_prediction_payload

    for target in (args.sidecar, args.output):
        if target is not None and target.exists():
            print(f"error: {target} exists; choose a new path.", file=sys.stderr)
            return 1
    payload = load_prediction_payload(args.artifact)
    if payload is None:
        print(f"error: {args.artifact} is not a readable prediction artifact.", file=sys.stderr)
        return 1
    report: Dict[str, Any] = {"artifact": str(args.artifact), **audit_payload(payload)}
    if args.collected and args.entry:
        report["samples_csv"] = samples_csv_view(args.collected, args.entry)
    if args.recover_identity:
        try:
            recovery = recover_identity(args.artifact, payload)
        except (AuditError, AssertionError, OSError, ValueError, KeyError) as exc:
            recovery = {"status": "error", "reason": f"{type(exc).__name__}: {exc}"}
        report["identity_recovery"] = {k: v for k, v in recovery.items() if k != "cell_ids"}
        if recovery.get("status") == "verified" and args.sidecar is not None:
            recovery["created_by"] = command_line(argv, "scripts.paper_figures.artifact_audit")
            recovery["code_version"] = code_version()
            args.sidecar.parent.mkdir(parents=True, exist_ok=True)
            args.sidecar.write_text(json.dumps(recovery, indent=2) + "\n", encoding="utf-8")
            report["identity_recovery"]["sidecar"] = str(args.sidecar)
    text = json.dumps(report, indent=2, sort_keys=True, default=str)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    _ = REPO_ROOT
    raise SystemExit(main())
