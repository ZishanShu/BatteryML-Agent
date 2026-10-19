"""Verify one saved BatteryML run from its own files, read-only (NATIVE-01).

No training, no prediction, no LLM call, and nothing is written inside the
run directory. The report goes to a NEW folder (default
``<project>/native_checks/<run id>/<UTC stamp>/``).

    # Fingerprints, payload structure, fitted transform and metric replay
    python -m scripts.native_run_check --run-dir /abs/path/runs/web_projects/<project>/quickstart/<run id> \\
        --record /abs/path/.../llm_explanations/<run id>/<record id>/record.json

    # ... plus test-cell identity: rebuild split and labels from data/processed (no features, no training)
    python -m scripts.native_run_check --run-dir ... --record ... --rebuild-identities

Run from the repository root with the run's environment active. The report
states which ``batteryml`` implementation and commit performed the replay:
that is the CURRENT checkout, not a claim about the code that trained the run.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from batteryml_web.core import native_check as nc  # noqa: E402

PACKAGES = ("torch", "numpy", "scikit-learn", "xgboost", "pandas", "scipy", "streamlit", "pyyaml")


def _git(*args: str) -> Optional[str]:
    try:
        done = subprocess.run(["git", *args], cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=20, check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    return done.stdout.strip() if done.returncode == 0 else None


def current_environment() -> Dict[str, Any]:
    import batteryml
    import batteryml.data.databundle as databundle

    versions = {}
    for name in PACKAGES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    status = _git("status", "--porcelain", "--untracked-files=no")
    return {
        "scope": "CURRENT environment and checkout that performed this check; not the training environment",
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "packages": versions,
        "batteryml_module": batteryml.__file__,
        "evaluator_source": databundle.__file__,
        "evaluator_source_sha256": nc.sha256_file(Path(databundle.__file__)),
        "git_head": _git("rev-parse", "HEAD"),
        "git_tracked_files_modified": None if status is None else bool(status),
    }


def _pick_predictions(run_dir: Path, explicit: Optional[str]) -> Path:
    if explicit:
        path = Path(explicit)
        if not path.is_file():
            raise SystemExit(f"--predictions {path} does not exist.")
        return path
    found = sorted(run_dir.glob("predictions_seed_*.pkl"))
    if len(found) != 1:
        raise SystemExit(
            f"{run_dir} holds {len(found)} predictions_seed_*.pkl file(s); pass --predictions with the one to check."
        )
    return found[0]


def rebuild_identities(config: Dict[str, Any]) -> Dict[str, Any]:
    """Split and labels from the processed cells, with the run's config. No features, no training."""
    import torch
    from batteryml.builders import DATA_TRANSFORMATIONS, LABEL_ANNOTATORS, TRAIN_TEST_SPLITTERS
    from batteryml.data import BatteryData

    split_config = dict(config["train_test_split"])
    cell_path = Path(split_config.get("cell_data_path", ""))
    if not cell_path.is_absolute():
        split_config["cell_data_path"] = str(REPO_ROOT / cell_path)
    splitter = TRAIN_TEST_SPLITTERS.build(split_config, "raise")
    train_paths, test_paths = splitter.split()
    annotator = LABEL_ANNOTATORS.build(config["label"])

    def _labels(paths: List[Any]) -> Dict[str, Any]:
        cells = [BatteryData.load(path) for path in paths]
        labels = annotator(cells)
        ids = [cell.cell_id for cell in cells]
        keep = (~torch.isnan(labels)).tolist()
        return {
            "all_ids": ids,
            "kept_ids": [cell for cell, k in zip(ids, keep) if k],
            "excluded_ids": [cell for cell, k in zip(ids, keep) if not k],
            "kept_labels": [float(v) for v, k in zip(labels.tolist(), keep) if k],
        }

    train, test = _labels(train_paths), _labels(test_paths)
    refit_state = None
    if config.get("label_transformation"):
        transform = DATA_TRANSFORMATIONS.build(config["label_transformation"])
        transform.fit(torch.tensor(train["kept_labels"], dtype=torch.float32))
        refit_state = nc.transform_state(transform)
    return {
        "method": "split + label annotator rebuilt from data/processed with the run's config and the CURRENT code; "
                  "no feature extraction, no training",
        "cell_data_path": split_config["cell_data_path"],
        "train": train,
        "test": test,
        "scientific_target": nc.scientific_target(annotator),
        "refit_state": refit_state,
    }


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-dir", required=True, help="absolute path of the run directory")
    parser.add_argument("--record", help="an LLM explanation record.json whose fingerprints the files must match")
    parser.add_argument("--predictions", help="the prediction artifact, if the run holds more than one")
    parser.add_argument("--rebuild-identities", action="store_true",
                        help="rebuild split and labels from data/processed to check test-cell identity and order")
    parser.add_argument("--accept-unrecorded", action="store_true",
                        help="load an artifact that no record fingerprints (only for a file you produced yourself)")
    parser.add_argument("--output", help="a NEW folder for the report (default under <project>/native_checks/)")
    args = parser.parse_args(argv)

    run_dir = Path(args.run_dir)
    if not run_dir.is_absolute() or not run_dir.is_dir():
        raise SystemExit(f"--run-dir must be an existing absolute directory: {run_dir}")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = Path(args.output) if args.output else nc.default_output_dir(run_dir, stamp)
    if output.exists():
        raise SystemExit(f"{output} already exists; a report is never overwritten.")
    if run_dir.resolve() in output.resolve().parents or output.resolve() == run_dir.resolve():
        raise SystemExit("--output must be outside the run directory.")

    record = json.loads(Path(args.record).read_text(encoding="utf-8-sig")) if args.record else {}
    record_files = record.get("files") or {}
    predictions_path = _pick_predictions(run_dir, args.predictions)
    manifest_path = run_dir / "run.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {}
    config_path = Path(manifest["config_path"]) if manifest.get("config_path") else None
    if config_path is not None and not config_path.is_absolute():
        config_path = REPO_ROOT / config_path
    workspace_configs = sorted(run_dir.glob("config_*.yaml"))

    files = {
        "run_manifest": nc.file_entry(manifest_path, record_files.get("run_manifest")),
        "config": nc.file_entry(config_path, record_files.get("config")),
        "workspace_configs": [
            nc.file_entry(path, next((w for w in record_files.get("workspace_configs") or []
                                      if Path(str(w.get("path"))).name == path.name), None))
            for path in workspace_configs
        ],
        "predictions": nc.file_entry(predictions_path, record_files.get("predictions")),
    }
    report: Dict[str, Any] = {
        "schema": nc.SCHEMA,
        "checked_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "run_dir": str(run_dir),
        "run_id": manifest.get("run_id") or run_dir.name,
        "linked_record_id": record.get("record_id"),
        "run_manifest_fields": {k: manifest.get(k) for k in
                                ("dataset", "task", "model_key", "seed", "device", "status", "git_commit", "command", "metrics")},
        "files": files,
        "environment_current": current_environment(),
        "boundaries": [
            "Read-only: nothing was trained, predicted or written in the run directory; no LLM was called.",
            "The replay used the CURRENT checkout's BatteryML code (environment_current); it shows that the stored "
            "objects reproduce the stored scores, not which code trained the run.",
        ],
    }

    fingerprint = files["predictions"]["fingerprint_status"]
    if fingerprint == nc.MISMATCH or (fingerprint != nc.MATCH and not args.accept_unrecorded):
        report["payload"] = {"status": nc.NOT_PERFORMED,
                             "reason": f"prediction fingerprint {fingerprint}; the file was not loaded"}
    else:
        from batteryml_web.agent.results import load_prediction_payload

        payload = load_prediction_payload(predictions_path)
        if not isinstance(payload, dict):
            report["payload"] = {"status": "UNREADABLE"}
        else:
            bundle = payload.get("data")
            test_data = getattr(bundle, "test_data", None)
            transform = getattr(bundle, "label_transformation", None)
            saved_index = payload.get("sample_index")
            bundle_index = getattr(test_data, "sample_index", None)
            saved_ids = [r.get("cell_id") for r in saved_index] if isinstance(saved_index, list) else (
                list(bundle_index.cell_ids) if bundle_index is not None else None)
            replay = nc.replay_metrics(bundle, payload.get("prediction"), payload.get("scores") or {},
                                       manifest.get("metrics"))
            report["payload"] = {
                "status": "READ",
                "keys": sorted(str(k) for k in payload.keys()),
                "prediction_stored": nc.tensor_facts(payload.get("prediction")),
                "test_label_stored": nc.tensor_facts(getattr(test_data, "label", None)),
                "seed": payload.get("seed"),
                "stored_scores": payload.get("scores"),
                "sample_index_in_payload": isinstance(saved_index, list),
                "sample_index_in_bundle": bundle_index is not None,
                "saved_cell_ids": saved_ids,
                "saved_cell_ids_status": "RECORDED_IN_ARTIFACT" if saved_ids else nc.NOT_RECORDED,
            }
            report["label_transformation"] = {"stored_state": nc.transform_state(transform)}
            report["metric_replay"] = replay
            report["array_scale"] = nc.array_scale(transform, replay)
            if args.rebuild_identities:
                import yaml

                if config_path is None or not config_path.is_file():
                    report["identity"] = {"status": nc.NOT_PERFORMED, "reason": "run config not found"}
                else:
                    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
                    rebuilt = rebuild_identities(config)
                    comparison = nc.compare_labels(rebuilt["test"]["kept_ids"], rebuilt["test"]["kept_labels"],
                                                   replay.get("target_native_values") or [], saved_ids)
                    report["identity"] = {**comparison, "method": rebuilt["method"],
                                          "cell_data_path": rebuilt["cell_data_path"],
                                          "test_excluded_ids": rebuilt["test"]["excluded_ids"],
                                          "train_kept_ids": rebuilt["train"]["kept_ids"],
                                          "train_excluded_ids": rebuilt["train"]["excluded_ids"]}
                    report["fitted_state_check"] = nc.compare_fitted_state(
                        report["label_transformation"]["stored_state"], rebuilt["refit_state"])
                    report["scientific_target"] = {**rebuilt["scientific_target"],
                                                   "rebuilt_labels_match_stored_targets": comparison["status"]}
            else:
                report["identity"] = {"status": nc.NOT_PERFORMED, "saved_cell_ids_status":
                                      report["payload"]["saved_cell_ids_status"]}

    output.mkdir(parents=True)
    target = output / nc.REPORT_FILENAME
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=repr) + "\n", encoding="utf-8")
    replay = report.get("metric_replay") or {}
    print(f"report: {target}")
    print(f"predictions fingerprint: {fingerprint}")
    print(f"metric replay: {replay.get('status', nc.NOT_PERFORMED)}")
    for name, row in (replay.get("metrics") or {}).items():
        print(f"  {name}: stored={row['stored_score']!r} evaluator={row['evaluator_replay']!r} "
              f"independent={row['independent_float64']!r}")
    print(f"array scale: {(report.get('array_scale') or {}).get('status', nc.NOT_PERFORMED)}")
    print(f"test identity: {(report.get('identity') or {}).get('status', nc.NOT_PERFORMED)}")
    if "fitted_state_check" in report:
        print(f"fitted state vs refit on rebuilt training labels: {report['fitted_state_check']['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
