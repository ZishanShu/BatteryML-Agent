"""Measured CPU inference latency and inference-file size for the SOH group.

Run by ``fig3_experiments.py --stages measure`` in a child process whose
thread environment is fixed BEFORE NumPy/BLAS load (OMP/MKL/OpenBLAS/
NumExpr/VECLIB = 1); this module refuses to time anything otherwise.

Protocol (this study's, not a general standard), from the spec file:

* one trained checkpoint per model (the seed the plan fixes), loaded through
  the model class's own ``load_checkpoint``;
* one fixed batch of real test samples: positions chosen by rank only
  (evenly spaced over the shared, identically ordered test sample index),
  fixed before any timing and identical for every model;
* input = the artifact's own already-transformed test features for those
  rows, flattened exactly as ``SklearnModel.predict`` does, in memory;
* timed region = ``estimator.predict(X)`` only. Excluded for every model:
  disk I/O, checkpoint deserialization, feature extraction, the feature
  transformation, the inverse label transformation, metric computation;
* before timing, ``estimator.predict(X)`` must reproduce the artifact's saved
  predictions for those rows (same label space) — the timed checkpoint is the
  one that produced the reported metrics;
* ``warmup`` untimed calls, then ``repeats`` timed calls with
  ``time.perf_counter_ns``; every timing is saved.

Inference-file size = bytes of (a) the model checkpoint file the run wrote
(not ``latest.ckpt``, a byte copy of it) and (b) the fitted feature and label
transformations the prediction path needs, serialized here from the
artifact's own objects (pickle protocol 4) because the run stores them only
inside the prediction artifact. Logs, predictions and optimizer state are not
counted. This is file size on disk, not RAM.

Outputs (new directory): ``latency_raw.csv``, ``latency_summary.csv``,
``latency.csv`` (fig3d_latency.template.csv format), ``inference_files.csv``,
``radar_metrics.csv``, ``radar_config.json``, ``measurement_metadata.json``.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from scripts.paper_figures.common import REPO_ROOT, code_version, read_json, write_csv

THREAD_ENV = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
              "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")
PICKLE_PROTOCOL = 4
MIB = float(2 ** 20)
TIMING_BOUNDARY = ("estimator.predict on an in-memory batch of already-transformed test "
                   "features; label inverse transform, feature transformation, disk I/O, "
                   "checkpoint deserialization and metric computation are outside the timer")
RAW_COLUMNS = ["entry_id", "model", "repeat", "elapsed_ns", "elapsed_ms"]
SUMMARY_COLUMNS = ["entry_id", "model", "batch_size", "repeats", "warmup_runs",
                   "mean_ms_per_batch", "sd_ms_per_batch", "median_ms_per_batch",
                   "min_ms_per_batch", "max_ms_per_batch",
                   "batch_amortized_mean_ms_per_sample"]
LATENCY_COLUMNS = [
    "model", "latency_value", "latency_unit", "statistic", "sd", "batch_size", "device",
    "hardware", "cpu_threads", "warmup_runs", "repeats", "timing_boundary",
    "includes_data_transfer", "includes_feature_computation", "includes_postprocessing",
    "gpu_sync", "measurement_kind", "source_file",
]
FILE_COLUMNS = ["entry_id", "model", "role", "path", "bytes", "origin"]


class MeasureError(RuntimeError):
    """A measurement that cannot be made as specified; the message says why."""


# --- pure helpers --------------------------------------------------------------

def batch_positions(n_rows: int, batch_size: int) -> List[int]:
    """Evenly spaced row positions, by rank only (never by error)."""
    if batch_size < 1 or n_rows < batch_size:
        raise MeasureError(f"cannot take a batch of {batch_size} from {n_rows} test rows")
    if batch_size == 1:
        return [0]
    positions = [round(i * (n_rows - 1) / (batch_size - 1)) for i in range(batch_size)]
    if len(set(positions)) != batch_size:
        raise MeasureError("evenly spaced positions collided; n_rows too small")
    return positions


def timing_stats(elapsed_ns: Sequence[int]) -> Dict[str, float]:
    """Mean, sample SD (n-1), median, min, max — in milliseconds per call."""
    ms = sorted(v / 1e6 for v in elapsed_ns)
    n = len(ms)
    if n < 2:
        raise MeasureError("at least two timed repeats are needed for an SD")
    mean = sum(ms) / n
    sd = math.sqrt(sum((v - mean) ** 2 for v in ms) / (n - 1))
    median = ms[n // 2] if n % 2 else (ms[n // 2 - 1] + ms[n // 2]) / 2
    return {"mean": mean, "sd": sd, "median": median, "min": ms[0], "max": ms[-1], "n": n}


def reference_radar(entries: List[Dict[str, Any]], *, reference_model: str,
                    batch_size: int) -> Dict[str, Any]:
    """radar_config + long-format metric rows from measured raw values."""
    rows = []
    for e in entries:
        rows += [
            {"model": e["model"], "dimension": "SOH_MAE", "value": e["saved_mae"],
             "unit": "ratio", "source": f"{e['artifact']} scores['MAE']", "n_repeats": 1},
            {"model": e["model"], "dimension": "inference_latency",
             "value": e["mean_ms_per_batch"], "unit": "ms/batch",
             "source": "latency_raw.csv", "n_repeats": e["repeats"]},
            {"model": e["model"], "dimension": "inference_files_size",
             "value": e["inference_mib"], "unit": "MiB",
             "source": "inference_files.csv", "n_repeats": 1},
        ]
    config = {
        "display_name": "SOH prediction error and deployment-cost trade-offs",
        "normalization": {"mode": "reference_relative", "reference_model": reference_model,
                          "formula": "score = r / (r + x)"},
        "dimensions": [
            {"name": "SOH_MAE", "unit": "ratio", "direction": "lower_is_better",
             "definition": "cycle-pooled SOH MAE on the test split (BatteryML saved score)",
             "source": "prediction artifact scores['MAE']"},
            {"name": "inference_latency", "unit": "ms/batch", "direction": "lower_is_better",
             "definition": f"mean wall-clock time of estimator.predict on one fixed batch of "
                           f"{batch_size} test samples, CPU, 1 thread",
             "source": "latency_raw.csv (this measurement)"},
            {"name": "inference_files_size", "unit": "MiB", "direction": "lower_is_better",
             "definition": "bytes of the model checkpoint plus the fitted feature and label "
                           "transformations needed for inference, / 2^20",
             "source": "inference_files.csv (this measurement)"},
        ],
    }
    return {"config": config, "rows": rows}


# --- measurement (touches files, models, clocks) --------------------------------

def _check_thread_env() -> None:
    wrong = {k: os.environ.get(k) for k in THREAD_ENV if os.environ.get(k) != "1"}
    if wrong:
        raise MeasureError(
            f"thread environment not fixed to 1 before start: {wrong}. Run through "
            "fig3_experiments.py --stages measure, which sets it for this process.")


def _hardware() -> Dict[str, Any]:
    model = "not readable"
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.is_file():
        for line in cpuinfo.read_text(errors="replace").splitlines():
            if line.lower().startswith("model name"):
                model = line.split(":", 1)[1].strip()
                break
    affinity = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None
    return {"cpu_model": model, "logical_cpus": os.cpu_count(), "cpus_in_affinity": affinity,
            "machine": platform.machine(), "platform": platform.platform(),
            "python": platform.python_version()}


def _checkpoint(directory: Path) -> Path:
    candidates = sorted(p for p in directory.glob("*.ckpt")
                        if p.name != "latest.ckpt" and not p.name.endswith(".resume.ckpt"))
    if len(candidates) != 1:
        raise MeasureError(
            f"{directory}: expected exactly one model checkpoint besides latest.ckpt, found "
            f"{[p.name for p in candidates]}; the timed model must be unambiguous.")
    return candidates[0]


def _load_model(artifact: Path):
    import yaml

    from batteryml.builders import MODELS
    from batteryml_web.agent.artifact_provenance import artifact_provenance
    from batteryml_web.models import register_extension_models

    register_extension_models()
    record = artifact_provenance(artifact)
    if record["config_copy"] == "not recorded":
        raise MeasureError(f"{artifact.parent}: no single config copy; the model cannot be rebuilt.")
    config = yaml.safe_load((artifact.parent / record["config_copy"]).read_text(encoding="utf-8"))
    model_config = dict(config["model"])
    model = MODELS.build(model_config)
    checkpoint = _checkpoint(artifact.parent)
    model.load_checkpoint(str(checkpoint))
    estimator = getattr(model, "model", None)
    if estimator is None or not hasattr(estimator, "predict"):
        raise MeasureError(f"{model_config.get('name')}: not a scikit-learn style model; "
                           "this CPU protocol does not apply to it.")
    return estimator, checkpoint, record


def _single_thread(estimator) -> Dict[str, Any]:
    applied = {}
    params = estimator.get_params() if hasattr(estimator, "get_params") else {}
    for key in ("n_jobs", "nthread"):
        if key in params:
            estimator.set_params(**{key: 1})
            applied[key] = 1
    return applied


def measure(spec: Dict[str, Any], output: Path) -> Dict[str, Any]:
    _check_thread_env()
    import pickle

    import numpy as np
    import torch
    from threadpoolctl import threadpool_info, threadpool_limits

    from batteryml_web.agent.result_views import cycle_sample_index
    from batteryml_web.agent.results import load_prediction_payload

    torch.set_num_threads(1)
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise MeasureError(f"{output} exists and is not empty; choose a new directory.")
    output.mkdir(parents=True, exist_ok=True)
    protocol = spec["protocol"]
    batch, warmup, repeats = int(protocol["batch_size"]), int(protocol["warmup"]), int(protocol["repeats"])

    loaded = []
    for entry in spec["entries"]:
        artifact = Path(entry["artifact"])
        payload = load_prediction_payload(artifact)
        if payload is None:
            raise MeasureError(f"{artifact} is not a readable prediction artifact.")
        index = cycle_sample_index(payload)
        if index is None:
            raise MeasureError(f"{artifact}: no sample_index; the batch keys cannot be fixed.")
        loaded.append((entry, artifact, payload, index))
    keys = [(r["cell_id"], r["cycle_index"]) for r in loaded[0][3]]
    for entry, _, _, index in loaded[1:]:
        if [(r["cell_id"], r["cycle_index"]) for r in index] != keys:
            raise MeasureError(f"{entry['entry_id']}: test sample keys differ from "
                               f"{loaded[0][0]['entry_id']}; the models were not evaluated on "
                               "the same samples.")
    positions = batch_positions(len(keys), batch)

    raw_rows, summary, latency_rows, file_rows, radar_entries, checks = [], [], [], [], [], {}
    hardware = _hardware()
    for entry, artifact, payload, _ in loaded:
        estimator, checkpoint, record = _load_model(artifact)
        threads = _single_thread(estimator)
        features = payload["data"].test_data.feature.to("cpu")
        X = np.ascontiguousarray(features.view(len(features), -1).numpy()[positions])
        saved = payload["prediction"].reshape(-1).to("cpu").numpy()[positions]
        with threadpool_limits(limits=1):
            check = np.asarray(estimator.predict(X)).reshape(-1)
            if not np.allclose(check, saved, rtol=1e-4, atol=1e-5):
                raise MeasureError(
                    f"{entry['entry_id']}: checkpoint {checkpoint.name} does not reproduce the "
                    f"saved predictions (max diff {float(np.max(np.abs(check - saved)))}); "
                    "it is not the model behind the reported metrics.")
            for _ in range(warmup):
                estimator.predict(X)
            elapsed = []
            for _ in range(repeats):
                start = time.perf_counter_ns()
                estimator.predict(X)
                elapsed.append(time.perf_counter_ns() - start)
            pools = threadpool_info()
        checks[entry["entry_id"]] = {
            "checkpoint": str(checkpoint), "max_abs_diff_vs_saved": float(np.max(np.abs(check - saved))),
            "input_shape": list(X.shape), "input_dtype": str(X.dtype),
            "estimator": type(estimator).__name__, "thread_params_set": threads,
            "threadpools_during_timing": [
                {k: p.get(k) for k in ("user_api", "internal_api", "num_threads")} for p in pools],
        }
        stats = timing_stats(elapsed)
        raw_rows += [{"entry_id": entry["entry_id"], "model": entry["model"], "repeat": i,
                      "elapsed_ns": ns, "elapsed_ms": ns / 1e6} for i, ns in enumerate(elapsed)]
        summary.append({"entry_id": entry["entry_id"], "model": entry["model"],
                        "batch_size": batch, "repeats": repeats, "warmup_runs": warmup,
                        "mean_ms_per_batch": stats["mean"], "sd_ms_per_batch": stats["sd"],
                        "median_ms_per_batch": stats["median"], "min_ms_per_batch": stats["min"],
                        "max_ms_per_batch": stats["max"],
                        "batch_amortized_mean_ms_per_sample": stats["mean"] / batch})
        latency_rows.append({
            "model": entry["model"], "latency_value": stats["mean"], "latency_unit": "ms_per_batch",
            "statistic": "mean", "sd": stats["sd"], "batch_size": batch, "device": "cpu",
            "hardware": hardware["cpu_model"], "cpu_threads": 1, "warmup_runs": warmup,
            "repeats": repeats, "timing_boundary": TIMING_BOUNDARY,
            "includes_data_transfer": "false", "includes_feature_computation": "false",
            "includes_postprocessing": "false", "gpu_sync": "false",
            "measurement_kind": "inference", "source_file": str(output / "latency_raw.csv"),
        })
        # Inference files: the checkpoint as written, plus the fitted transformations.
        files = [{"role": "model checkpoint", "path": checkpoint, "origin": "written by the run"}]
        target_dir = output / "inference_files" / entry["entry_id"]
        target_dir.mkdir(parents=True)
        for role in ("feature_transformation", "label_transformation"):
            obj = getattr(payload["data"], role, None)
            if obj is None:
                continue
            obj = obj.to("cpu") if hasattr(obj, "to") else obj
            path = target_dir / f"{role}.pkl"
            path.write_bytes(pickle.dumps(obj, protocol=PICKLE_PROTOCOL))
            files.append({"role": f"fitted {role.replace('_', ' ')}", "path": path,
                          "origin": f"serialized here from the artifact (pickle protocol {PICKLE_PROTOCOL})"})
        total = 0
        for f in files:
            size = Path(f["path"]).stat().st_size
            total += size
            file_rows.append({"entry_id": entry["entry_id"], "model": entry["model"],
                              "role": f["role"], "path": str(f["path"]), "bytes": size,
                              "origin": f["origin"]})
        scores = payload.get("scores") or {}
        mae = scores.get("MAE")
        if not isinstance(mae, (int, float)) or not math.isfinite(float(mae)):
            raise MeasureError(f"{entry['entry_id']}: saved MAE missing or non-finite.")
        radar_entries.append({"model": entry["model"], "artifact": str(artifact),
                              "saved_mae": float(mae), "mean_ms_per_batch": stats["mean"],
                              "repeats": repeats, "inference_mib": total / MIB})

    write_csv(output / "latency_raw.csv", raw_rows, RAW_COLUMNS)
    write_csv(output / "latency_summary.csv", summary, SUMMARY_COLUMNS)
    write_csv(output / "latency.csv", latency_rows, LATENCY_COLUMNS)
    write_csv(output / "inference_files.csv", file_rows, FILE_COLUMNS)
    radar = reference_radar(radar_entries, reference_model=spec["reference_model"], batch_size=batch)
    write_csv(output / "radar_metrics.csv", radar["rows"],
              ["model", "dimension", "value", "unit", "source", "n_repeats"])
    (output / "radar_config.json").write_text(json.dumps(radar["config"], indent=2) + "\n",
                                              encoding="utf-8")
    metadata = {
        "protocol": protocol, "timing_boundary": TIMING_BOUNDARY,
        "units": {"primary": "ms per batch (batch_size as recorded)",
                  "batch_amortized_ms_per_sample": "mean_ms_per_batch / batch_size; not a "
                                                   "batch_size=1 online latency"},
        "batch_positions": positions, "batch_keys": [list(keys[p]) for p in positions],
        "batch_selection": "evenly spaced ranks over the shared test sample index; fixed "
                           "before timing; independent of any error",
        "hardware": hardware, "thread_env": {k: os.environ.get(k) for k in THREAD_ENV},
        "torch_threads": torch.get_num_threads(), "checks": checks,
        "file_size_unit": "MiB = bytes / 2^20 (file size on disk, not RAM)",
        "code_version": code_version(), "spec": spec,
    }
    (output / "measurement_metadata.json").write_text(
        json.dumps(metadata, indent=2, default=str) + "\n", encoding="utf-8")
    return metadata


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        measure(read_json(args.spec), args.output)
    except MeasureError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"measured into {args.output}")
    return 0


if __name__ == "__main__":
    _ = REPO_ROOT
    raise SystemExit(main())
