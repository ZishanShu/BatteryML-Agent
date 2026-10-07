"""Fig. 3 data: list candidate runs, then collect an explicit selection.

Two subcommands, both visualization-only (no training, no inference, no
download, no deletion):

``list-runs``
    Read-only inventory of every ``predictions_seed_*.pkl`` under ``--runs-root``
    with what is recorded beside it (``run.json``, ``config_<ts>.yaml``). Nothing
    is unpickled and nothing is chosen: rows are sorted by path, never by date
    or error.

``collect``
    Reads the selection JSON you wrote (see ``fig3_selection.example.json``),
    loads each named artifact through the platform's CPU-safe loader
    (``batteryml_web.agent.results.load_prediction_payload``), builds the same
    view the website uses (``agent/result_views.py``: the recorded label
    transformation inverted exactly once, one finite mask over observed and
    predicted, identity filtered by the same mask), and writes standardized
    files into a NEW directory:

    * ``runs.csv``      one row per selected artifact: identity, protocol,
                        target, saved scores verbatim, counts, provenance;
    * ``samples.csv``   one row per plotted pair: observed, predicted,
                        ``abs_error = |predicted - observed|`` in target units;
    * ``published_rmse.csv``  only if the selection asks for the published
                        BatteryML reference table (kept separate from local runs);
    * ``latency.csv`` / ``radar_metrics.csv`` / ``radar_config.json``  copied
                        from the files the selection names, after validation;
    * ``collected_metadata.json``.

Usage (from the repository root)::

    python -m scripts.paper_figures.collect_fig3_data list-runs --runs-root runs
    python -m scripts.paper_figures.collect_fig3_data collect \\
        --selection scripts/paper_figures/fig3_selection.json \\
        --output runs/paper_figures/fig3_data
"""

from __future__ import annotations

import argparse
import shutil
from collections import Counter
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from scripts.paper_figures.common import (
    REPO_ROOT, command_line, code_version, read_csv, read_json, write_csv,
)

NR = "not recorded"

RUN_COLUMNS = [
    "entry_id", "model", "model_label", "dataset", "dataset_source", "protocol",
    "protocol_source", "split_class", "schema", "task_kind", "quantity", "unit",
    "label_annotator", "soh_reference", "seed", "smoke", "epochs_override",
    "smoke_evidence", "run_id", "run_manifest", "artifact", "artifact_sha256",
    "config_identity", "config_path", "config_copy", "model_class",
    "saved_RMSE", "saved_MAE", "saved_MAPE_ratio", "metric_source",
    "sample_weighting", "stored_rows", "plotted_rows", "omitted_nonfinite",
    "n_cells", "transform_status", "transform_chain",
    "identity_status", "identity_source", "rows_per_cell_max",
    "determinism", "observation_budget",
]
#: identity_status values. Only these two let a row stand for one named cell.
IDENTITY_KNOWN = ("recorded", "recovered")
SAMPLE_COLUMNS = ["entry_id", "artifact_row", "cell_id", "cycle_index",
                  "observed", "predicted", "abs_error"]
PUBLISHED_COLUMNS = ["model", "model_label", "dataset", "protocol", "central",
                     "dispersion", "seeds", "censored", "censor_threshold",
                     "metric", "unit", "source"]


class CollectError(RuntimeError):
    """A selection that cannot be collected; the message names the entry."""


# --- list-runs -----------------------------------------------------------------

def list_runs(runs_root: Path) -> List[Dict[str, Any]]:
    """Every prediction artifact under ``runs_root`` with its recorded context."""
    from batteryml_web.agent.artifact_provenance import artifact_provenance, smoke_evidence
    from batteryml_web.core.benchmarks import subset_for_splitter

    rows = []
    for path in sorted(Path(runs_root).rglob("predictions_seed_*.pkl")):
        record = artifact_provenance(path, hash_files=False)
        label = record.get("label") if isinstance(record.get("label"), dict) else {}
        rows.append({
            "artifact": str(path),
            "run_id": record["run_id"],
            "model": record["model_key"] if record["model_key"] != NR else record["model_class"],
            "dataset": record["dataset"],
            "task": record["task"],
            "label_annotator": record.get("label_annotator") or NR,
            "split_class": record["split"],
            "protocol": subset_for_splitter(record["split"]) or NR,
            "seed": record["seed"],
            "smoke": record["smoke"],
            "epochs_override": record["epochs_override"],
            "config_copy_epochs": record["config_copy_epochs"],
            "status": record["status"],
            "smoke_evidence": smoke_evidence(record),
            "reference_capacity_mode": label.get("reference_capacity_mode", NR),
        })
    return rows


LIST_COLUMNS = ["artifact", "run_id", "model", "dataset", "task", "label_annotator",
                "split_class", "protocol", "seed", "smoke", "epochs_override",
                "config_copy_epochs", "status", "reference_capacity_mode", "smoke_evidence"]


# --- collect -------------------------------------------------------------------

def _resolve(path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else Path.cwd() / path


def _check(declared: Optional[str], recorded: Any, what: str, entry: str) -> tuple:
    """(value, source). A declared value that contradicts the record is refused."""
    recorded_ok = recorded not in (None, "", NR)
    if declared and recorded_ok and str(declared) != str(recorded):
        raise CollectError(
            f"entry {entry!r}: selection says {what}={declared!r} but the run records "
            f"{recorded!r}. Fix the selection; a recorded value is never overridden."
        )
    if recorded_ok:
        return str(recorded), "recorded"
    if declared:
        return str(declared), "declared in selection (not recorded beside the artifact)"
    raise CollectError(f"entry {entry!r}: {what} is neither recorded nor declared.")


def collect_entry(entry: Dict[str, Any]) -> Dict[str, Any]:
    """One selected artifact -> a runs.csv row and its samples.csv rows."""
    from batteryml_web.agent.artifact_provenance import artifact_provenance, smoke_evidence
    from batteryml_web.agent.result_views import (
        RUL_CYCLEWISE_SCHEMA, SOH_SCHEMA, TRANSFORM_FAILED, TRANSFORM_MISSING, attach_artifact_context,
        build_rul_view, build_soh_view, cycle_sample_index, detect_schema,
    )
    from batteryml_web.agent.results import load_prediction_payload
    from batteryml_web.core.benchmarks import subset_for_splitter

    entry_id = str(entry.get("id") or "").strip()
    if not entry_id:
        raise CollectError("every runs[] entry needs a unique 'id'.")
    if not entry.get("artifact") or not entry.get("model"):
        raise CollectError(f"entry {entry_id!r}: 'artifact' and 'model' are required.")
    path = _resolve(str(entry["artifact"]))
    if not path.is_file():
        raise CollectError(f"entry {entry_id!r}: artifact {path} does not exist.")

    record = artifact_provenance(path)
    payload = load_prediction_payload(path)
    if payload is None:
        raise CollectError(f"entry {entry_id!r}: {path} is not a readable prediction artifact.")
    schema = detect_schema(payload)
    view = build_soh_view(payload) if schema == SOH_SCHEMA else build_rul_view(payload)
    if not view.get("available"):
        raise CollectError(f"entry {entry_id!r}: {view.get('reason')}")
    if view.get("transform_status") in (TRANSFORM_MISSING, TRANSFORM_FAILED):
        raise CollectError(f"entry {entry_id!r}: label space unknown ({view['transform_status']}).")
    attach_artifact_context(view, payload, path)
    target = view.get("target") or {}
    if target.get("unit") in (None, "", "unit not recorded"):
        raise CollectError(
            f"entry {entry_id!r}: the target unit cannot be established from the "
            "recorded label annotator; it is not guessed."
        )

    dataset, dataset_source = _check(entry.get("dataset"), record["dataset"], "dataset", entry_id)
    split_protocol = subset_for_splitter(record["split"]) if record["split"] != NR else None
    protocol, protocol_source = _check(entry.get("protocol"), split_protocol, "protocol", entry_id)
    if entry.get("model_key"):
        _check(entry.get("model_key"), record["model_key"], "model_key", entry_id)
    seed = record["seed"] if record["seed"] != NR else payload.get("seed")
    saved = view.get("saved_metrics") or {}
    label = record.get("label") if isinstance(record.get("label"), dict) else {}
    per_cycle = schema in (SOH_SCHEMA, RUL_CYCLEWISE_SCHEMA)
    run_row = {
        "entry_id": entry_id,
        "model": str(entry["model"]),
        "model_label": str(entry.get("model_label") or entry["model"]),
        "dataset": dataset, "dataset_source": dataset_source,
        "protocol": protocol, "protocol_source": protocol_source,
        "split_class": record["split"], "schema": schema,
        "task_kind": target.get("kind"), "quantity": target.get("quantity"),
        "unit": target.get("unit"),
        "label_annotator": target.get("label_annotator") or record.get("label_annotator") or NR,
        "soh_reference": target.get("reference") or label.get("reference_capacity_mode") or NR,
        "seed": seed, "smoke": record["smoke"], "epochs_override": record["epochs_override"],
        "smoke_evidence": smoke_evidence(record),
        "run_id": record["run_id"], "run_manifest": record["run_manifest"],
        "artifact": str(path), "artifact_sha256": record["artifact_sha256"],
        # Same model name with a different config is a different configuration.
        "config_identity": (record["config_copy_sha256"] if record["config_copy_sha256"] != NR
                            else record["config_path"]),
        "config_path": record["config_path"], "config_copy": record["config_copy"],
        "model_class": record["model_class"],
        "saved_RMSE": saved.get("RMSE"), "saved_MAE": saved.get("MAE"),
        "saved_MAPE_ratio": saved.get("MAPE"),
        "metric_source": "payload['scores'] (BatteryML Pipeline.evaluate; verbatim)",
        "sample_weighting": ("cycle-pooled: every stored cell-cycle row weighs the same"
                             if per_cycle else "one stored row per cell"),
        "stored_rows": view.get("total_rows"), "plotted_rows": view.get("count"),
        "omitted_nonfinite": view.get("omitted_nonfinite"),
        "n_cells": len(set(view.get("cell_ids") or [])) or NR,
        "transform_status": view.get("transform_status"),
        "transform_chain": "|".join(view.get("transform_chain") or []) or "none",
    }
    positions = view.get("row_positions") or list(range(view["count"]))
    index = cycle_sample_index(payload)
    index_ok = index is not None and len(index) == view.get("total_rows")
    cells = view.get("cell_ids")
    if index_ok:
        identity = ("recorded", "sample_index written by Pipeline.evaluate")
    elif cells:
        identity = ("recorded", "cell ids carried by the saved DataBundle")
    else:
        identity = ("not_recorded",
                    "the artifact carries no sample_index and no cell ids; rows are "
                    "unnamed (never given a placeholder id)")
    cycles = view.get("cycle_indices")
    if cycles is None:
        # Scalar views carry no cycle list; take it from the recorded index
        # through the same row positions (-1 = whole cell), never invent it.
        cycles = ([index[p]["cycle_index"] for p in positions] if index_ok
                  else [None] * view["count"])
    if entry.get("identity_sidecar"):
        recovered = load_identity_sidecar(_resolve(str(entry["identity_sidecar"])), record,
                                          int(view.get("total_rows") or 0), entry_id)
        named = [recovered["cell_ids"][p] for p in positions]
        if cells and [str(c) for c in cells] != named:
            raise CollectError(
                f"entry {entry_id!r}: the identity sidecar disagrees with the identity the "
                "artifact records; the recorded identity is never overridden.")
        if not cells:
            cells = named
            cycles = [-1] * view["count"]
            identity = ("recovered", (
                f"identity sidecar {entry['identity_sidecar']}: {recovered['method']}; "
                f"max |label difference| {recovered.get('max_abs_label_diff')}"))
    if not cells:
        cells = [""] * view["count"]
    counts = Counter(str(c) for c in cells) if identity[0] != "not_recorded" else None
    run_row.update({
        "identity_status": identity[0], "identity_source": identity[1],
        "rows_per_cell_max": max(counts.values()) if counts else None,
        "n_cells": len(counts) if counts else NR,
        "determinism": str(entry.get("determinism") or NR),
        "observation_budget": str(entry.get("observation_budget") or NR),
    })
    samples = [
        {"entry_id": entry_id, "artifact_row": positions[i], "cell_id": cells[i],
         "cycle_index": cycles[i], "observed": view["observed"][i],
         "predicted": view["predicted"][i],
         "abs_error": abs(view["predicted"][i] - view["observed"][i])}
        for i in range(view["count"])
    ]
    return {"run": run_row, "samples": samples}


def load_identity_sidecar(path: Path, record: Dict[str, Any], total_rows: int,
                          entry_id: str) -> Dict[str, Any]:
    """A verified identity sidecar bound to exactly this artifact, or CollectError."""
    from scripts.paper_figures.artifact_audit import SIDECAR_KIND

    if not path.is_file():
        raise CollectError(f"entry {entry_id!r}: identity sidecar {path} does not exist.")
    sidecar = read_json(path)
    problems = []
    if sidecar.get("kind") != SIDECAR_KIND:
        problems.append("not an identity sidecar")
    if sidecar.get("status") != "verified":
        problems.append(f"status is {sidecar.get('status')!r}, not 'verified'")
    if sidecar.get("artifact_sha256") != record["artifact_sha256"]:
        problems.append("it was made for a different artifact (SHA-256 differs)")
    ids = sidecar.get("cell_ids") or []
    if len(ids) != total_rows:
        problems.append(f"it names {len(ids)} cells for {total_rows} artifact rows")
    if len(set(ids)) != len(ids):
        problems.append("its cell ids repeat")
    if problems:
        raise CollectError(f"entry {entry_id!r}: identity sidecar {path} refused: "
                           + "; ".join(problems))
    return sidecar


def published_rows(spec: Dict[str, Any]) -> List[Dict[str, Any]]:
    """The published README benchmark (RMSE, cycles) for the selected models/subsets."""
    from batteryml_web.core import benchmarks as bm

    models = [str(m) for m in spec.get("models") or []]
    subsets = [str(s) for s in spec.get("protocols") or bm.SUBSETS]
    rows = []
    for record in bm.all_records():
        if (models and record.model_key not in models) or record.dataset_subset not in subsets:
            continue
        rows.append({
            "model": record.model_key, "model_label": bm.ROW_BY_MODEL_KEY.get(record.model_key),
            "dataset": "published", "protocol": record.dataset_subset,
            "central": record.value.central, "dispersion": record.value.dispersion,
            "seeds": record.value.seeds, "censored": record.value.censored,
            "censor_threshold": record.value.censor_threshold,
            "metric": record.metric, "unit": record.unit, "source": record.source,
        })
    return rows


def collect(selection_path: Path, output: Path, argv: Optional[List[str]] = None) -> Dict[str, Any]:
    from scripts.paper_figures.fig3c_multicriteria_radar import validate_radar_inputs
    from scripts.paper_figures.fig3d_inference_latency import validate_latency_rows
    from batteryml_web.agent.artifact_provenance import sha256_file

    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise CollectError(f"{output} exists and is not empty; choose a new output directory.")
    selection = read_json(selection_path)
    entries = selection.get("runs") or []
    ids = [str(e.get("id")) for e in entries]
    if len(ids) != len(set(ids)):
        raise CollectError("runs[].id values must be unique.")
    order = [str(m) for m in selection.get("model_order") or []]
    runs, samples = [], []
    for entry in entries:
        collected_entry = collect_entry(entry)
        if collected_entry["run"]["model"] not in order:
            raise CollectError(
                f"entry {entry.get('id')!r}: model {collected_entry['run']['model']!r} is not "
                "in model_order; list it so its colour and legend position are fixed."
            )
        runs.append(collected_entry["run"])
        samples.extend(collected_entry["samples"])

    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "runs.csv", runs, RUN_COLUMNS)
    write_csv(output / "samples.csv", samples, SAMPLE_COLUMNS)
    notes: List[str] = []
    published = selection.get("published_reference") or {}
    if published.get("include"):
        write_csv(output / "published_rmse.csv", published_rows(published), PUBLISHED_COLUMNS)
    inputs: Dict[str, Any] = {}
    if selection.get("latency_csv"):
        source = _resolve(str(selection["latency_csv"]))
        problems = validate_latency_rows(read_csv(source))
        inputs["latency_csv"] = {"path": str(source), "sha256": sha256_file(source),
                                 "problems": problems}
        shutil.copyfile(source, output / "latency.csv")
    radar = selection.get("radar") or {}
    if radar.get("metrics_csv") and radar.get("config"):
        metrics_path, config_path = _resolve(radar["metrics_csv"]), _resolve(radar["config"])
        inputs["radar"] = {
            "metrics_csv": str(metrics_path), "metrics_sha256": sha256_file(metrics_path),
            "config": str(config_path), "config_sha256": sha256_file(config_path),
            "problems": validate_radar_inputs(read_csv(metrics_path), read_json(config_path))[1],
        }
        shutil.copyfile(metrics_path, output / "radar_metrics.csv")
        shutil.copyfile(config_path, output / "radar_config.json")
    metadata = {
        "selection_file": str(selection_path),
        "selection_sha256": sha256_file(Path(selection_path)),
        "model_order": order,
        "model_labels": {r["model"]: r["model_label"] for r in runs} | dict(selection.get("model_labels") or {}),
        "entries": [r["entry_id"] for r in runs],
        "published_reference": published,
        "inputs": inputs,
        "notes": notes,
        "command": command_line(argv, "scripts.paper_figures.collect_fig3_data"),
        "code_version": code_version(),
        "processing": (
            "Each artifact read by agent.results.load_prediction_payload (CPU-safe); "
            "observed/predicted from agent.result_views (recorded label transformation "
            "inverted once; one finite mask; identity filtered by the same mask); "
            "abs_error = |predicted - observed|; saved scores copied verbatim."
        ),
    }
    import json

    (output / "collected_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    return metadata


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m scripts.paper_figures.collect_fig3_data",
                                     description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    lister = sub.add_parser("list-runs", help="Read-only inventory of prediction artifacts.")
    lister.add_argument("--runs-root", type=Path, default=Path("runs"))
    lister.add_argument("--output", type=Path, default=None,
                        help="Optional CSV path for the inventory (must not exist).")
    collector = sub.add_parser("collect", help="Collect an explicit selection into a new directory.")
    collector.add_argument("--selection", type=Path, required=True)
    collector.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "list-runs":
            rows = list_runs(args.runs_root)
            if args.output is not None:
                if args.output.exists():
                    raise CollectError(f"{args.output} exists; choose a new file.")
                args.output.parent.mkdir(parents=True, exist_ok=True)
                write_csv(args.output, rows, LIST_COLUMNS)
            print(f"{len(rows)} prediction artifact(s) under {args.runs_root} (sorted by path; none chosen):")
            for row in rows:
                print(" | ".join(f"{k}={row[k]}" for k in
                                 ("artifact", "run_id", "model", "task", "protocol", "seed", "smoke")))
            return 0
        metadata = collect(args.selection, args.output, argv)
        print(f"Collected {len(metadata['entries'])} artifact(s) into {args.output}")
        return 0
    except (CollectError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
