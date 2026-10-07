"""Fig. 3 experiment campaign: audit -> plan -> reuse/run -> measure -> collect -> plot.

One resumable entry point for the experiments behind Fig. 3, run on the
server that holds the processed data. Every stage writes into one campaign
directory; nothing outside it is modified (old runs, predictions, metrics,
caches and run.json files are only read).

Stages (``--stages``, comma-separated, in this order):

``audit``    environment, processed data, configs, GPUs; every existing
             prediction artifact under ``--runs-root`` with its reuse
             eligibility; optional row-identity audits of an existing
             selection (``--existing-selection``).
``plan``     write ``experiment_plan.json`` ONCE from the fixed protocol
             below and the audit. An existing plan is loaded, never rewritten.
``run``      reuse what the plan marks reusable; launch the remaining
             training tasks one at a time through the BatteryML CLI wrapper,
             each with its own run.json (run_manifest schema), output.log and
             acceptance check. Resumable: accepted tasks are skipped.
``measure``  CPU inference latency + inference-file size for the SOH group
             (``fig3_measure.py`` in a child process with 1 thread).
``collect``  identity recovery where an artifact carries no cell identity,
             seed-effect check, selection file, ``collect_fig3_data.collect``.
``plot``     panels A–E as separate files (``plot_fig3_all``).
``report``   experiment table, metrics summary, stage status.

Protocol (preregistered here, never adjusted after seeing test errors):

* SOH, MATR1 (configs/soh/*/matr.yaml): Ridge, Lasso, ElasticNet, RF, XGBoost.
* Lifetime, MATR1: Ridge, RF, XGBoost (configs/baselines/sklearn/*/matr_1.yaml),
  CNN (configs/baselines/nn_models/cnn/matr_1.yaml), extension BatLiNet and
  CPTransformer (configs/external/*/matr_1.yaml).
* Lifetime, HUST: Ridge, RF, XGBoost, CNN (``hust.yaml``) when the processed
  data exist; no HUST split/config is invented for the extension models.
* Seeds 0, 1, 2 for models whose randomness the pipeline seed reaches; one
  fit (seed 0) for deterministic estimators. Fixed epoch budgets from the
  configs, final-epoch weights, no test-based checkpoint selection, no
  ``--epochs`` override (no smoke runs).

Usage (repository root, conda env ``batteryml``)::

    python -m scripts.paper_figures.fig3_experiments \\
        --campaign runs/paper_figures/fig3_campaign_v1 \\
        --stages audit,plan,run,measure,collect,plot,report \\
        --existing-selection runs/paper_figures/fig3_local_LfPNvddj/fig3_selection.json
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from scripts.paper_figures.common import REPO_ROOT, code_version, read_json, write_csv

PROTOCOL_VERSION = "fig3-protocol-v1"
SEEDS = (0, 1, 2)
STAGES = ("audit", "plan", "run", "measure", "collect", "plot", "report")
MODEL_ORDER = ["ridge", "lasso", "elastic_net", "rf", "xgb", "cnn", "batlinet", "cptransformer"]

#: How each model's randomness relates to the pipeline seed ([STATIC] reading
#: of the wrapper sources; the collect stage re-checks it on the predictions).
MODELS: Dict[str, Dict[str, Any]] = {
    "ridge": {"label": "Ridge", "kind": "sklearn", "seeded": False,
              "seed_note": "sklearn Ridge: closed-form solve, no random state"},
    "lasso": {"label": "Lasso", "kind": "sklearn", "seeded": False,
              "seed_note": "LassoCV: unshuffled 5-fold CV, cyclic coordinate descent"},
    "elastic_net": {"label": "ElasticNet", "kind": "sklearn", "seeded": False,
                    "seed_note": "ElasticNetCV: unshuffled 5-fold CV, cyclic coordinate descent"},
    "rf": {"label": "Random forest", "kind": "sklearn", "seeded": True,
           "seed_note": "RandomForestRegressor(random_state=None) draws bootstrap and feature "
                        "subsets from NumPy's global RNG, which Pipeline.set_seed seeds"},
    "xgb": {"label": "XGBoost", "kind": "sklearn", "seeded": False,
            "seed_note": "XGBRegressor keeps its own default random_state and subsample=1; "
                         "the pipeline seed does not reach the estimator"},
    "cnn": {"label": "CNN", "kind": "torch", "seeded": True,
            "seed_note": "torch.manual_seed via Pipeline.set_seed (init, shuffling)"},
    "batlinet": {"label": "BatLiNet", "kind": "torch", "seeded": True,
                 "seed_note": "torch.manual_seed (init, permutation, support sampling)"},
    "cptransformer": {"label": "CPTransformer", "kind": "torch", "seeded": True,
                      "seed_note": "torch.manual_seed (init, shuffling)",
                      "requires": "third_party/BatteryLife"},
}

#: The comparison groups, in execution order (SOH first: C and D need it).
GROUPS: List[Dict[str, Any]] = [
    {"group": "soh", "task": "SOH estimation (CycleSOHLabelAnnotator)", "dataset": "MATR",
     "protocol": "MATR1",
     "configs": {m: f"configs/soh/{m}/matr.yaml"
                 for m in ("ridge", "lasso", "elastic_net", "rf", "xgb")}},
    {"group": "lifetime", "task": "battery lifetime / cycle life (RULLabelAnnotator)",
     "dataset": "MATR", "protocol": "MATR1",
     "configs": {"ridge": "configs/baselines/sklearn/ridge/matr_1.yaml",
                 "rf": "configs/baselines/sklearn/rf/matr_1.yaml",
                 "xgb": "configs/baselines/sklearn/xgb/matr_1.yaml",
                 "cnn": "configs/baselines/nn_models/cnn/matr_1.yaml",
                 "batlinet": "configs/external/batlinet/matr_1.yaml",
                 "cptransformer": "configs/external/cptransformer/matr_1.yaml"}},
    {"group": "lifetime", "task": "battery lifetime / cycle life (RULLabelAnnotator)",
     "dataset": "HUST", "protocol": "HUST",
     "configs": {"ridge": "configs/baselines/sklearn/ridge/hust.yaml",
                 "rf": "configs/baselines/sklearn/rf/hust.yaml",
                 "xgb": "configs/baselines/sklearn/xgb/hust.yaml",
                 "cnn": "configs/baselines/nn_models/cnn/hust.yaml",
                 "batlinet": None, "cptransformer": None}},
]

#: Measurement protocol for D and C (this study's protocol, not a standard).
LATENCY_PROTOCOL = {"device": "cpu", "threads": 1, "batch_size": 32, "warmup": 20,
                    "repeats": 100, "seed": 0, "group": "soh", "dataset": "MATR",
                    "protocol": "MATR1", "reference_model": "ridge"}

# Task statuses.
PLANNED, RUNNING, ACCEPTED, REUSED = "PLANNED", "RUNNING", "ACCEPTED", "REUSED"
FAILED, REJECTED, BLOCKED, NOT_PLANNED = "FAILED", "REJECTED", "BLOCKED", "NOT_PLANNED"
DONE = (ACCEPTED, REUSED)

#: Failure output that means the shared data (not one model) is broken.
SHARED_DATA_ERRORS = (
    r"AssertionError: .*data/processed", r"SampleIndexError", r"No cycle-level SOH sample",
    r"UnpicklingError", r"_pickle\.UnpicklingError", r"EOFError: Ran out of input",
)


class CampaignError(RuntimeError):
    """A stage cannot proceed; the message names what and why."""


# --- small utilities ---------------------------------------------------------------

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json_atomic(path: Path, data: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=str(path.parent),
                                         prefix=f".{path.name}.", suffix=".tmp", delete=False)
    with handle as stream:
        stream.write(json.dumps(data, indent=2, sort_keys=True, default=str) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(handle.name, path)


def next_numbered(parent: Path, prefix: str, suffix: str = "") -> Path:
    """parent/<prefix>_NNN<suffix> that does not exist yet (history is never overwritten)."""
    parent.mkdir(parents=True, exist_ok=True)
    pattern = rf"{re.escape(prefix)}_(\d+){re.escape(suffix)}"
    taken = [int(m.group(1)) for p in parent.glob(f"{prefix}_*")
             if (m := re.fullmatch(pattern, p.name))]
    return parent / f"{prefix}_{(max(taken) + 1 if taken else 1):03d}{suffix}"


def latest_numbered(parent: Path, prefix: str) -> Optional[Path]:
    found = sorted(p for p in parent.glob(f"{prefix}_*")
                   if re.fullmatch(rf"{re.escape(prefix)}_\d+", p.name))
    return found[-1] if found else None


def log(campaign: Path, message: str) -> None:
    line = f"{time.strftime('%Y-%m-%dT%H:%M:%S')} {message}"
    print(line, flush=True)
    with open(Path(campaign) / "stage_log.txt", "a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def _yaml(path: Path) -> Dict[str, Any]:
    import yaml

    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise CampaignError(f"{path} is not a YAML mapping.")
    return data


# --- pure planning logic ---------------------------------------------------------------

def observation_budget(feature: Dict[str, Any]) -> str:
    """What the model may observe of each test cell, from the config's feature section."""
    name = str(feature.get("name") or "")
    if name == "VoltageCapacityMatrixFeatureExtractor":
        return f"cycles <= {feature.get('max_cycle_index')} (Qd(V) diff vs cycle {feature.get('diff_base')})"
    if name == "BatLiNetFeatureExtractor":
        return f"cycles {feature.get('min_cycle_index')}-{feature.get('max_cycle_index')}"
    if name == "BatteryLifeCycleCurveFeatureExtractor":
        return f"first {feature.get('early_cycle_threshold')} cycles"
    if name == "SOHVoltageCurveFeatureExtractor":
        return (f"voltage curve of the estimated cycle ({feature.get('n_voltage_points')} points, "
                f"{feature.get('n_history_cycles')} history cycle(s))")
    return f"{name or 'unknown extractor'}: {json.dumps(feature, sort_keys=True)}"


def training_budget(model_cfg: Dict[str, Any], kind: str) -> Dict[str, Any]:
    if kind == "torch":
        return {"epochs": model_cfg.get("epochs"), "batch_size": model_cfg.get("batch_size"),
                "termination": "fixed epoch budget from the config; final-epoch weights are "
                               "evaluated; no validation split, no test-based selection",
                "note": "the training loop prints test RMSE every evaluate_freq epochs "
                        "(monitoring output only; nothing is selected from it)"}
    return {"epochs": None, "termination": "single fit call (closed form / internal CV / "
                                           "fixed n_estimators); no checkpoint selection"}


def seeds_for(model: str) -> Tuple[int, ...]:
    return SEEDS if MODELS[model]["seeded"] else (0,)


def task_id(group: str, dataset: str, protocol: str, model: str, seed: int) -> str:
    return f"{group}-{dataset}-{protocol}-{model}-seed{seed}"


def eligible_reuse(candidate: Dict[str, Any], task: Dict[str, Any]) -> List[str]:
    """Pure. Reasons the candidate artifact may NOT stand for the task ([] = eligible)."""
    reasons = []
    if candidate.get("run_json") is None:
        reasons.append("no run.json (status, seed and smoke flag not recorded)")
        return reasons
    if str(candidate.get("status")).upper() != "COMPLETED":
        reasons.append(f"status {candidate.get('status')!r}")
    if candidate.get("return_code") != 0:
        reasons.append(f"return_code {candidate.get('return_code')!r}")
    if candidate.get("smoke") is not False:
        reasons.append(f"smoke {candidate.get('smoke')!r}")
    budget = (task.get("training_budget") or {}).get("epochs")
    override = candidate.get("epochs_override")
    if override not in (None, "", "None") and (budget is None or str(override) != str(budget)):
        # An --epochs equal to the config's own budget changes nothing; any
        # other value (e.g. a 1-epoch smoke run) is a different training run.
        reasons.append(f"epochs override {override!r} (planned budget {budget!r})")
    if str(candidate.get("seed")) != str(task["seed"]):
        reasons.append(f"seed {candidate.get('seed')!r} != {task['seed']}")
    if candidate.get("config_copy_sha256") != task["config_sha256"]:
        reasons.append("config copy differs from the planned config (SHA-256)")
    if candidate.get("predictions_for_seed", 1) != 1:
        reasons.append(f"{candidate.get('predictions_for_seed')} prediction files for this seed "
                       "in one workspace (ambiguous)")
    if task.get("needs_checkpoint") and not candidate.get("model_checkpoint"):
        reasons.append("no unambiguous model checkpoint (needed for the latency measurement)")
    return reasons


def choose_reuse(candidates: List[Dict[str, Any]], task: Dict[str, Any]) -> Dict[str, Any]:
    """Earliest-finished eligible candidate (a preregistered rule, never by score)."""
    assessed = [{"artifact": c["artifact"], "reasons": eligible_reuse(c, task),
                 "finished_at": c.get("finished_at") or ""} for c in candidates]
    eligible = sorted((a for a in assessed if not a["reasons"]),
                      key=lambda a: (a["finished_at"], a["artifact"]))
    return {"chosen": eligible[0]["artifact"] if eligible else None,
            "rule": "earliest finished eligible run (not chosen by any score)",
            "assessed": assessed}


def build_plan(*, repo_root: Path, candidates: List[Dict[str, Any]], cpu_threads: int,
               data_counts: Dict[str, int], present_paths: Iterable[str]) -> Dict[str, Any]:
    """Pure apart from reading config files under repo_root."""
    present = set(present_paths)
    tasks, not_planned = [], []
    for group in GROUPS:
        for model in MODEL_ORDER:
            if model not in group["configs"]:
                continue
            config = group["configs"][model]
            where = f"{group['group']}/{group['dataset']}/{group['protocol']}/{model}"
            if config is None:
                not_planned.append({"what": where, "reason": "no config for this dataset in the "
                                    "repository; no split or adapter is invented"})
                continue
            config_path = repo_root / config
            if not config_path.is_file():
                not_planned.append({"what": where, "reason": f"{config} does not exist"})
                continue
            cfg = _yaml(config_path)
            info = MODELS[model]
            blockers = []
            data_path = str(cfg["train_test_split"].get("cell_data_path"))
            if data_counts.get(data_path, 0) == 0:
                blockers.append(f"{data_path} has no processed cell files")
            if info.get("requires") and info["requires"] not in present:
                blockers.append(f"{info['requires']} is not present")
            for seed in seeds_for(model):
                task = {
                    "task_id": task_id(group["group"], group["dataset"], group["protocol"], model, seed),
                    "group": group["group"], "task": group["task"],
                    "dataset": group["dataset"], "protocol": group["protocol"],
                    "split_class": cfg["train_test_split"].get("name"),
                    "fixed_split_identity": (
                        f"{cfg['train_test_split'].get('name')} over {data_path}: the test cohort "
                        "is fixed by the split class and does not depend on the seed"),
                    "model": model, "model_class": cfg["model"].get("name"),
                    "config": config, "config_sha256": sha256_file(config_path),
                    "feature": cfg.get("feature"), "label": cfg.get("label"),
                    "observation_budget": observation_budget(cfg.get("feature") or {}),
                    "seed": seed, "seed_policy": ("seeded" if info["seeded"] else "deterministic"),
                    "seed_note": info["seed_note"],
                    "training_budget": training_budget(cfg["model"], info["kind"]),
                    "device": ("one free CUDA GPU (no DDP)" if info["kind"] == "torch"
                               else f"cpu, {cpu_threads} threads"),
                    "kind": info["kind"],
                    "needs_checkpoint": (group["group"] == LATENCY_PROTOCOL["group"]
                                         and seed == LATENCY_PROTOCOL["seed"]),
                    "blocked_at_plan": blockers,
                }
                same = [c for c in candidates if c.get("config_copy_sha256") == task["config_sha256"]
                        or c.get("model") == model]
                task["reuse"] = choose_reuse(same, task)
                task["action"] = ("reuse" if task["reuse"]["chosen"] else
                                  "blocked" if blockers else "new run")
                tasks.append(task)
    return {"protocol_version": PROTOCOL_VERSION, "seeds": list(SEEDS),
            "model_order": MODEL_ORDER, "latency_protocol": LATENCY_PROTOCOL,
            "tasks": tasks, "not_planned": not_planned}


def pick_free_gpu(gpu_csv: str, apps_csv: str, *, max_used_mib: float = 500.0,
                  max_util: float = 5.0) -> Optional[Dict[str, Any]]:
    """Pure. Lowest-index GPU with no compute process and (almost) no memory in use."""
    busy = {line.split(",")[0].strip() for line in apps_csv.splitlines() if line.strip()}
    for line in gpu_csv.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 5:
            continue
        index, uuid, used, total, util = parts[:5]
        try:
            used_f, util_f = float(used), float(util)
        except ValueError:
            continue
        if uuid in busy or used_f > max_used_mib or util_f > max_util:
            continue
        return {"index": int(index), "uuid": uuid, "memory_used_mib": used_f,
                "memory_total_mib": float(total), "utilization": util_f}
    return None


def classify_failure(output_tail: str) -> str:
    return ("shared_data" if any(re.search(p, output_tail) for p in SHARED_DATA_ERRORS)
            else "task")


def epochs_completed(run_dir: Path, log_text: str, planned: Optional[int]) -> Optional[int]:
    """Largest completed epoch evidenced by epoch checkpoints or the loop's own print.

    The run directory holds exactly one training run, so every epoch
    checkpoint in it is this run's. The seed in the file name is NOT used:
    NNModel.fit names its weight checkpoints with its own ``seed`` parameter,
    which defaults to 0 and is not passed by Pipeline.train
    (batteryml/models/nn_model.py), so seed-1/2 runs also write "seed_0".
    """
    found = []
    for path in run_dir.glob("*_epoch_*.ckpt"):
        m = re.search(r"_epoch_(\d+)\.(?:resume\.)?ckpt$", path.name)
        if m:
            found.append(int(m.group(1)))
    if planned:
        if re.search(rf"\[{planned}/{planned}\]", log_text):
            found.append(planned)
    return max(found) if found else None


def initial_state(plan: Dict[str, Any]) -> Dict[str, Any]:
    tasks = {}
    for t in plan["tasks"]:
        status = REUSED if t["action"] == "reuse" else BLOCKED if t["action"] == "blocked" else PLANNED
        tasks[t["task_id"]] = {"status": status, "attempts": [],
                               "artifact": t["reuse"]["chosen"] if status == REUSED else None,
                               "reason": "; ".join(t["blocked_at_plan"]) if status == BLOCKED else ""}
    return {"tasks": tasks}


def runnable(plan: Dict[str, Any], state: Dict[str, Any], *, retry_failed: bool,
             only_groups: Optional[Sequence[str]] = None,
             blocked_groups: Iterable[Tuple[str, str]] = ()) -> List[Dict[str, Any]]:
    """Pure. Tasks this invocation should attempt, in plan order."""
    blocked = set(blocked_groups)
    out = []
    for task in plan["tasks"]:
        entry = state["tasks"][task["task_id"]]
        if only_groups and task["group"] not in only_groups:
            continue
        if (task["group"], task["dataset"]) in blocked:
            continue
        status = entry["status"]
        if status in DONE or status == RUNNING:
            continue
        if status in (FAILED, REJECTED) and not retry_failed:
            continue
        if status == BLOCKED and task["blocked_at_plan"]:
            continue
        out.append(task)
    return out


# --- audit (reads the server) --------------------------------------------------------

def _run_text(command: List[str]) -> Tuple[int, str]:
    try:
        done = subprocess.run(command, capture_output=True, text=True, timeout=60, check=False)
    except (OSError, subprocess.SubprocessError) as exc:
        return -1, f"{type(exc).__name__}: {exc}"
    return done.returncode, done.stdout + done.stderr


def gpu_query() -> Dict[str, Any]:
    code, gpus = _run_text(["nvidia-smi", "--query-gpu=index,uuid,memory.used,memory.total,"
                            "utilization.gpu,name", "--format=csv,noheader,nounits"])
    code2, apps = _run_text(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid",
                             "--format=csv,noheader"])
    return {"available": code == 0, "gpus_csv": gpus if code == 0 else "",
            "apps_csv": apps if code2 == 0 else "", "error": "" if code == 0 else gpus}


def environment_audit(repo_root: Path) -> Dict[str, Any]:
    versions = {}
    for module in ("numpy", "torch", "sklearn", "xgboost", "threadpoolctl"):
        try:
            versions[module] = str(__import__(module).__version__)
        except Exception as exc:  # noqa: BLE001 - recorded as provenance
            versions[module] = f"not importable ({type(exc).__name__})"
    data = {}
    for name in ("MATR", "HUST"):
        path = repo_root / "data" / "processed" / name
        data[f"data/processed/{name}"] = (len(list(path.glob("*.pkl"))) if path.is_dir() else 0)
    cuda = {}
    try:
        import torch

        cuda = {"cuda_available": bool(torch.cuda.is_available()),
                "device_count": int(torch.cuda.device_count())}
    except Exception as exc:  # noqa: BLE001
        cuda = {"error": f"{type(exc).__name__}: {exc}"}
    git_code, git_head = _run_text(["git", "-C", str(repo_root), "rev-parse", "HEAD"])
    return {
        "repo_root": str(repo_root), "is_git_repo": (repo_root / ".git").exists(),
        "git_head": git_head.strip() if git_code == 0 else "unknown",
        "conda_env": os.environ.get("CONDA_DEFAULT_ENV"), "python": sys.executable,
        "versions": versions, "processed_cell_files": data, "cuda": cuda, "gpus": gpu_query(),
        "paths_present": [p for p in ("third_party/BatteryLife", "cache")
                          if (repo_root / p).exists()],
        "cpu_count": os.cpu_count(), "code_version": code_version(),
    }


def scan_existing(runs_root: Path, exclude: Path) -> List[Dict[str, Any]]:
    """Every prediction artifact with the facts reuse is decided on (no unpickling)."""
    from batteryml_web.agent.artifact_provenance import artifact_provenance
    from batteryml_web.core import run_manifest as rm

    exclude = exclude.resolve()
    rows = []
    for path in sorted(Path(runs_root).rglob("predictions_seed_*.pkl")):
        if exclude in path.resolve().parents:
            continue
        record = artifact_provenance(path, hash_files=False)
        manifest = rm.read_manifest(path.parent)
        seed = re.match(r"predictions_seed_(\d+)_", path.name)
        same_seed = len(list(path.parent.glob(f"predictions_seed_{seed.group(1)}_*.pkl"))) if seed else 0
        configs = sorted(path.parent.glob("config_*.yaml"))
        checkpoints = sorted(p for p in path.parent.glob("*.ckpt")
                             if p.name != "latest.ckpt" and "_epoch_" not in p.name
                             and not p.name.endswith(".resume.ckpt"))
        rows.append({
            "artifact": str(path), "run_json": "run.json" if manifest else None,
            "run_id": record["run_id"], "model": record["model_key"], "dataset": record["dataset"],
            "status": (manifest or {}).get("status"), "return_code": (manifest or {}).get("return_code"),
            "smoke": record["smoke"] if manifest else None,
            "epochs_override": (manifest or {}).get("epochs"),
            "seed": (manifest or {}).get("seed"), "finished_at": (manifest or {}).get("finished_at"),
            "config_copy_sha256": sha256_file(configs[0]) if len(configs) == 1 else None,
            "config_copies": len(configs), "split_class": record["split"],
            "predictions_for_seed": same_seed,
            "model_checkpoint": str(checkpoints[0]) if len(checkpoints) == 1 else None,
        })
    return rows


AUDIT_COLUMNS = ["artifact", "run_json", "run_id", "model", "dataset", "status", "return_code",
                 "smoke", "epochs_override", "seed", "finished_at", "config_copy_sha256",
                 "config_copies", "split_class", "predictions_for_seed", "model_checkpoint"]


def stage_audit(campaign: Path, args) -> None:
    from scripts.paper_figures import artifact_audit as aa
    from batteryml_web.agent.results import load_prediction_payload

    out = campaign / "audit"
    out.mkdir(parents=True, exist_ok=True)
    env = environment_audit(REPO_ROOT)
    write_json_atomic(out / "environment.json", env)
    log(campaign, f"audit: git {env['git_head'][:12]}, conda env {env['conda_env']!r}, "
                  f"processed {env['processed_cell_files']}, cuda {env['cuda']}")
    if not env["is_git_repo"] or env["processed_cell_files"].get("data/processed/MATR", 0) == 0:
        log(campaign, "audit: WARNING this checkout has no git repo or no processed MATR "
                      "data; it is not the experiment server. Planning will block every task.")
    existing = scan_existing(Path(args.runs_root), campaign)
    write_csv(out / "existing_artifacts.csv", existing, AUDIT_COLUMNS)
    log(campaign, f"audit: {len(existing)} existing prediction artifact(s) listed")
    if args.existing_selection:
        selection = read_json(Path(args.existing_selection))
        for entry in selection.get("runs") or []:
            artifact = Path(entry["artifact"])
            payload = load_prediction_payload(artifact)
            if payload is None:
                report = {"artifact": str(artifact), "error": "not a readable prediction artifact"}
            else:
                report = {"artifact": str(artifact), "entry": entry.get("id"),
                          **aa.audit_payload(payload)}
                if report.get("schema") == "rul_scalar":
                    try:
                        recovery = aa.recover_identity(artifact, payload)
                    except (aa.AuditError, AssertionError, OSError, ValueError, KeyError) as exc:
                        recovery = {"status": "error", "reason": f"{type(exc).__name__}: {exc}"}
                    report["identity_recovery"] = {k: v for k, v in recovery.items() if k != "cell_ids"}
                    if recovery.get("cell_ids") and report.get("identity") == "recorded":
                        from batteryml_web.agent.result_views import cycle_sample_index

                        recorded = [r["cell_id"] for r in cycle_sample_index(payload)]
                        report["identity_recovery"]["equals_recorded_identity"] = (
                            recorded == recovery["cell_ids"])
            collected = Path(args.existing_selection).parent / "data"
            if (collected / "samples.csv").is_file() and entry.get("id"):
                report["samples_csv"] = aa.samples_csv_view(collected, entry["id"])
            target = out / "artifact_audits" / f"{entry.get('id') or artifact.stem}.json"
            write_json_atomic(target, report)
            log(campaign, f"audit: {entry.get('id')}: identity={report.get('identity')}, "
                          f"rows={report.get('total_rows')}, cells={report.get('distinct_cells')}, "
                          f"verdict={report.get('verdict')}")


# --- plan --------------------------------------------------------------------------

def stage_plan(campaign: Path, args) -> Dict[str, Any]:
    plan_path = campaign / "experiment_plan.json"
    if plan_path.is_file():
        plan = read_json(plan_path)
        log(campaign, f"plan: loaded existing plan ({len(plan['tasks'])} tasks); it is never "
                      "rewritten. Start a new campaign directory for a different plan.")
        return plan
    env_path = campaign / "audit" / "environment.json"
    existing_path = campaign / "audit" / "existing_artifacts.csv"
    if not env_path.is_file() or not existing_path.is_file():
        raise CampaignError("run the audit stage before plan.")
    env = read_json(env_path)
    with open(existing_path, newline="", encoding="utf-8") as handle:
        candidates = []
        for row in csv.DictReader(handle):
            row = {k: (None if v in ("", "NA") else v) for k, v in row.items()}
            row["return_code"] = int(row["return_code"]) if row.get("return_code") not in (None,) else None
            row["smoke"] = {"True": True, "False": False}.get(str(row.get("smoke")), row.get("smoke"))
            row["predictions_for_seed"] = int(row.get("predictions_for_seed") or 0)
            candidates.append(row)
    plan = build_plan(repo_root=REPO_ROOT, candidates=candidates, cpu_threads=args.cpu_threads,
                      data_counts=env["processed_cell_files"],
                      present_paths=env.get("paths_present") or [])
    plan.update({"created_by": " ".join([sys.executable, "-m",
                                         "scripts.paper_figures.fig3_experiments"] + sys.argv[1:]),
                 "git_head": env["git_head"], "environment": "audit/environment.json"})
    write_json_atomic(plan_path, plan)
    write_json_atomic(campaign / "state" / "tasks.json", initial_state(plan))
    counts: Dict[str, int] = {}
    for t in plan["tasks"]:
        counts[t["action"]] = counts.get(t["action"], 0) + 1
    log(campaign, f"plan: wrote {plan_path} ({len(plan['tasks'])} tasks: {counts}; "
                  f"{len(plan['not_planned'])} not planned)")
    return plan


# --- run ---------------------------------------------------------------------------

class Lock:
    def __init__(self, path: Path) -> None:
        self.path = path

    def __enter__(self):
        if self.path.exists():
            try:
                pid = int(self.path.read_text().strip())
                os.kill(pid, 0)
                raise CampaignError(f"another runner (pid {pid}) holds {self.path}.")
            except (ValueError, ProcessLookupError):
                pass  # stale lock from a process that no longer exists
            except PermissionError:
                raise CampaignError(f"{self.path} is held by a process of another user.")
        self.path.write_text(str(os.getpid()))
        return self

    def __exit__(self, *exc):
        if self.path.exists() and self.path.read_text().strip() == str(os.getpid()):
            self.path.unlink()


def subprocess_launcher(command: List[str], env: Dict[str, str], log_path: Path) -> int:
    with open(log_path, "w", encoding="utf-8") as handle:
        process = subprocess.Popen(command, cwd=str(REPO_ROOT), env=env, stdout=handle,
                                   stderr=subprocess.STDOUT)
        try:
            return process.wait()
        except KeyboardInterrupt:
            process.send_signal(signal.SIGINT)
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
            raise


def thread_env(threads: int) -> Dict[str, str]:
    return {k: str(threads) for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                                      "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")}


def accept_run(run_dir: Path, task: Dict[str, Any], return_code: int) -> Dict[str, Any]:
    """Acceptance of one finished training run. Never edits any file it reads."""
    from batteryml_web.agent.result_views import detect_schema
    from batteryml_web.agent.results import load_prediction_payload

    problems, facts = [], {"return_code": return_code}
    if return_code != 0:
        problems.append(f"exit code {return_code}")
    predictions = sorted(run_dir.glob(f"predictions_seed_{task['seed']}_*.pkl"))
    if len(predictions) != 1:
        problems.append(f"{len(predictions)} prediction file(s) for seed {task['seed']}")
        return {"accepted": False, "problems": problems, **facts}
    facts["artifact"] = str(predictions[0])
    payload = load_prediction_payload(predictions[0])
    if payload is None:
        problems.append("prediction artifact unreadable")
        return {"accepted": False, "problems": problems, **facts}
    scores = payload.get("scores") or {}
    facts["metrics"] = {k: scores.get(k) for k in ("RMSE", "MAE", "MAPE")}
    for key in ("RMSE", "MAE"):
        value = scores.get(key)
        if not isinstance(value, (int, float)) or value != value or value in (float("inf"), float("-inf")):
            problems.append(f"saved {key} missing or non-finite")
    if str(payload.get("seed")) != str(task["seed"]):
        problems.append(f"artifact seed {payload.get('seed')} != {task['seed']}")
    configs = sorted(run_dir.glob("config_*.yaml"))
    if len(configs) != 1 or sha256_file(configs[0]) != task["config_sha256"]:
        problems.append("config copy missing or different from the planned config")
    schema = detect_schema(payload)
    facts["schema"] = schema
    expected = "soh_cycle" if task["group"] == "soh" else "rul_scalar"
    if schema != expected:
        problems.append(f"schema {schema} != expected {expected}")
    log_text = (run_dir / "output.log").read_text(errors="replace") if (run_dir / "output.log").is_file() else ""
    facts["convergence_warnings"] = len(re.findall(r"ConvergenceWarning", log_text))
    if task["kind"] == "torch":
        planned = task["training_budget"].get("epochs")
        done = epochs_completed(run_dir, log_text, planned)
        facts["epochs_completed"] = done
        facts["termination"] = ("fixed epoch budget reached" if planned and done == planned
                                else "budget not evidenced")
        if not planned or done != planned:
            problems.append(f"completed epochs {done} not evidenced equal to the budget {planned}")
    else:
        checkpoints = [p for p in run_dir.glob("*.ckpt") if p.name != "latest.ckpt"]
        facts["model_checkpoint"] = str(checkpoints[0]) if len(checkpoints) == 1 else None
        facts["iterations"] = "single fit"
        if len(checkpoints) != 1:
            problems.append(f"{len(checkpoints)} model checkpoint(s) besides latest.ckpt")
    return {"accepted": not problems, "problems": problems, **facts}


def execute_task(campaign: Path, task: Dict[str, Any], state: Dict[str, Any], *,
                 device: str, extra_env: Dict[str, str],
                 launcher: Callable[[List[str], Dict[str, str], Path], int] = subprocess_launcher
                 ) -> Dict[str, Any]:
    from batteryml_web.core import run_manifest as rm
    from batteryml_web.core.run_identity import new_run_id

    entry = state["tasks"][task["task_id"]]
    run_id = new_run_id(dataset=task["dataset"], model=task["model"], seed=task["seed"])
    run_dir = campaign / "runs" / run_id
    workspace = os.path.relpath(run_dir, REPO_ROOT)
    command = [sys.executable, "batteryml_web/core/run_batteryml_cli.py", "run", task["config"],
               "--workspace", workspace, "--train", "--eval", "--device", device,
               "--seed", str(task["seed"]), "--skip_if_executed", "False"]
    _, git_head = _run_text(["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"])
    rm.write_manifest(run_dir, rm.build_manifest(
        run_id=run_id, project="paper_fig3", workspace=workspace, dataset=task["dataset"],
        task=task["group"], model_key=task["model"], model_display=MODELS[task["model"]]["label"],
        config_path=task["config"], seed=task["seed"], epochs=None, device=device, smoke=False,
        command=command, git_commit=git_head.strip() or None))
    rm.update_manifest(run_dir, {"started_at": rm.utc_now_iso()}, status=rm.STATUS_RUNNING)
    attempt = {"run_id": run_id, "run_dir": str(run_dir), "device": device,
               "env": extra_env, "started_at": rm.utc_now_iso(), "command": command}
    entry["status"] = RUNNING
    entry["attempts"].append(attempt)
    save_state(campaign, state)
    env = {**os.environ, **extra_env}
    started = time.monotonic()
    try:
        code = launcher(command, env, run_dir / "output.log")
    except KeyboardInterrupt:
        rm.update_manifest(run_dir, {"finished_at": rm.utc_now_iso()}, status=rm.STATUS_INTERRUPTED)
        attempt.update(status="INTERRUPTED", finished_at=rm.utc_now_iso())
        entry["status"] = PLANNED
        save_state(campaign, state)
        raise
    elapsed = time.monotonic() - started
    verdict = accept_run(run_dir, task, code)
    rm.update_manifest(
        run_dir,
        {"finished_at": rm.utc_now_iso(), "elapsed_seconds": round(elapsed, 3),
         "return_code": code,
         "artifacts": {"prediction_files": [verdict["artifact"]]} if verdict.get("artifact") else {},
         "metrics": (verdict.get("metrics") or {}) if code == 0 else {}},
        status=rm.STATUS_COMPLETED if code == 0 else rm.STATUS_FAILED)
    attempt.update(finished_at=rm.utc_now_iso(), return_code=code, elapsed_seconds=elapsed,
                   acceptance=verdict)
    write_json_atomic(run_dir / "fig3_task_record.json",
                      {"task": task, "attempt": attempt, "plan_protocol": PROTOCOL_VERSION})
    if verdict["accepted"]:
        entry.update(status=ACCEPTED, artifact=verdict["artifact"], reason="")
    else:
        tail = (run_dir / "output.log").read_text(errors="replace")[-20000:] \
            if (run_dir / "output.log").is_file() else ""
        entry.update(status=FAILED if code != 0 else REJECTED,
                     reason="; ".join(verdict["problems"]), failure_class=classify_failure(tail))
    save_state(campaign, state)
    return entry


def load_state(campaign: Path) -> Dict[str, Any]:
    path = campaign / "state" / "tasks.json"
    if not path.is_file():
        raise CampaignError("no state/tasks.json; run the plan stage first.")
    return read_json(path)


def save_state(campaign: Path, state: Dict[str, Any]) -> None:
    write_json_atomic(campaign / "state" / "tasks.json", state)


def recover_interrupted(campaign: Path, state: Dict[str, Any]) -> List[str]:
    """A task left RUNNING by a dead runner: its run becomes INTERRUPTED, the task PLANNED."""
    from batteryml_web.core import run_manifest as rm

    recovered = []
    for tid, entry in state["tasks"].items():
        if entry["status"] != RUNNING:
            continue
        attempt = entry["attempts"][-1] if entry["attempts"] else {}
        run_dir = Path(attempt.get("run_dir", ""))
        manifest = rm.read_manifest(run_dir) if run_dir.name else None
        if manifest and manifest.get("status") == rm.STATUS_RUNNING:
            rm.update_manifest(run_dir, {"finished_at": rm.utc_now_iso()},
                               status=rm.STATUS_INTERRUPTED)
        attempt["status"] = "INTERRUPTED"
        entry["status"] = PLANNED
        recovered.append(tid)
    if recovered:
        save_state(campaign, state)
    return recovered


def stage_run(campaign: Path, args, plan: Dict[str, Any],
              launcher: Callable[[List[str], Dict[str, str], Path], int] = subprocess_launcher,
              gpu_probe: Callable[[], Dict[str, Any]] = gpu_query) -> None:
    state = load_state(campaign)
    for tid in recover_interrupted(campaign, state):
        log(campaign, f"run: {tid} was left RUNNING by an earlier runner; its run is marked "
                      "INTERRUPTED and the task will run again with a new run id")
    blocked_groups = {(t["group"], t["dataset"]) for t in plan["tasks"]
                      if state["tasks"][t["task_id"]].get("failure_class") == "shared_data"}
    todo = runnable(plan, state, retry_failed=args.retry_failed, only_groups=args.only_groups,
                    blocked_groups=blocked_groups)
    log(campaign, f"run: {len(todo)} task(s) to attempt (max {args.max_tasks or 'all'})")
    launched = 0
    for task in todo:
        if args.max_tasks and launched >= args.max_tasks:
            break
        if (task["group"], task["dataset"]) in blocked_groups:
            continue
        entry = state["tasks"][task["task_id"]]
        if task["kind"] == "torch":
            if args.gpu == "none":
                entry.update(status=BLOCKED, reason="deep model; --gpu none")
                save_state(campaign, state)
                continue
            if args.gpu == "auto":
                probe = gpu_probe()
                gpu = pick_free_gpu(probe["gpus_csv"], probe["apps_csv"]) if probe["available"] else None
                if gpu is None:
                    entry.update(status=BLOCKED, reason="no free GPU right now (rerun later)")
                    save_state(campaign, state)
                    log(campaign, f"run: {task['task_id']} waits: no free GPU")
                    continue
                index = gpu["index"]
            else:
                index = int(args.gpu)
            env = {**thread_env(args.cpu_threads), "CUDA_VISIBLE_DEVICES": str(index)}
            device = "cuda"
        else:
            env = {**thread_env(args.cpu_threads), "CUDA_VISIBLE_DEVICES": ""}
            device = "cpu"
        log(campaign, f"run: start {task['task_id']} on {device} {env.get('CUDA_VISIBLE_DEVICES')!r}")
        result = execute_task(campaign, task, state, device=device, extra_env=env, launcher=launcher)
        launched += 1
        log(campaign, f"run: {task['task_id']} -> {result['status']} {result.get('reason') or ''}")
        if result.get("failure_class") == "shared_data":
            blocked_groups.add((task["group"], task["dataset"]))
            log(campaign, f"run: shared-data failure; remaining {task['group']}/{task['dataset']} "
                          "tasks are not started in this invocation")


# --- measure -----------------------------------------------------------------------

def done_tasks(plan: Dict[str, Any], state: Dict[str, Any]) -> List[Tuple[Dict[str, Any], str]]:
    return [(t, state["tasks"][t["task_id"]]["artifact"]) for t in plan["tasks"]
            if state["tasks"][t["task_id"]]["status"] in DONE]


def stage_measure(campaign: Path, args, plan: Dict[str, Any]) -> Optional[Path]:
    from scripts.paper_figures.fig3d_inference_latency import validate_latency_rows
    from scripts.paper_figures.common import read_csv

    lp = plan["latency_protocol"]
    state = load_state(campaign)
    wanted = [t for t in plan["tasks"] if t["group"] == lp["group"] and t["dataset"] == lp["dataset"]
              and t["protocol"] == lp["protocol"] and t["seed"] == lp["seed"]]
    missing = [t["task_id"] for t in wanted if state["tasks"][t["task_id"]]["status"] not in DONE]
    if missing:
        log(campaign, f"measure: skipped; these seed-{lp['seed']} SOH tasks are not done: {missing}")
        return None
    entries = [{"entry_id": t["task_id"], "model": t["model"],
                "artifact": state["tasks"][t["task_id"]]["artifact"]} for t in wanted]
    out = next_numbered(campaign / "measurements", "measure")
    spec = {"protocol": {k: lp[k] for k in ("device", "threads", "batch_size", "warmup", "repeats")},
            "reference_model": lp["reference_model"], "entries": entries}
    spec_path = out.parent / f"{out.name}_spec.json"
    write_json_atomic(spec_path, spec)
    env = {**os.environ, **thread_env(1), "CUDA_VISIBLE_DEVICES": ""}
    command = [sys.executable, "-m", "scripts.paper_figures.fig3_measure", "--spec", str(spec_path),
               "--output", str(out)]
    log(campaign, f"measure: {' '.join(command)}")
    done = subprocess.run(command, cwd=str(REPO_ROOT), env=env, capture_output=True, text=True,
                          check=False)
    (out.parent / f"{out.name}.log").write_text(done.stdout + done.stderr, encoding="utf-8")
    if done.returncode != 0:
        log(campaign, f"measure: FAILED (exit {done.returncode}); see {out.parent / (out.name + '.log')}")
        return None
    problems = validate_latency_rows(read_csv(out / "latency.csv"))
    if problems:
        log(campaign, f"measure: latency rows fail validation: {problems}")
        return None
    log(campaign, f"measure: wrote {out}")
    return out


# --- collect -----------------------------------------------------------------------

def seed_effect(arrays: Dict[int, List[float]]) -> Dict[str, Any]:
    """Pure. Whether predictions differ between seeds (identical = seed had no effect)."""
    seeds = sorted(arrays)
    base = arrays[seeds[0]]
    identical = all(arrays[s] == base for s in seeds[1:])
    return {"seeds": seeds, "identical_predictions": identical}


def build_selection(plan: Dict[str, Any], state: Dict[str, Any], *,
                    sidecars: Dict[str, str], drop: Dict[str, str],
                    measurement: Optional[Path]) -> Dict[str, Any]:
    """Pure. The selection file collect_fig3_data reads, from done tasks only."""
    runs = []
    for task, artifact in done_tasks(plan, state):
        if task["task_id"] in drop:
            continue
        entry = {"id": task["task_id"], "model": task["model"], "artifact": artifact,
                 "dataset": task["dataset"], "protocol": task["protocol"],
                 "determinism": task["seed_policy"], "observation_budget": task["observation_budget"]}
        if task["task_id"] in sidecars:
            entry["identity_sidecar"] = sidecars[task["task_id"]]
        runs.append(entry)
    selection = {"model_order": MODEL_ORDER,
                 "model_labels": {m: MODELS[m]["label"] for m in MODEL_ORDER},
                 "runs": runs, "published_reference": {"include": False},
                 "dropped_entries": drop, "plan_protocol": plan["protocol_version"]}
    if measurement is not None:
        selection["latency_csv"] = str(measurement / "latency.csv")
        selection["radar"] = {"metrics_csv": str(measurement / "radar_metrics.csv"),
                              "config": str(measurement / "radar_config.json")}
    return selection


def stage_collect(campaign: Path, args, plan: Dict[str, Any]) -> Optional[Path]:
    from scripts.paper_figures import artifact_audit as aa
    from scripts.paper_figures.collect_fig3_data import CollectError, collect
    from batteryml_web.agent.results import load_prediction_payload

    state = load_state(campaign)
    done = done_tasks(plan, state)
    if not done:
        log(campaign, "collect: no completed task yet")
        return None
    sidecars, drop, notes = {}, {}, {}
    arrays: Dict[Tuple[str, str, str, str], Dict[int, List[float]]] = {}
    for task, artifact in done:
        payload = load_prediction_payload(Path(artifact))
        if payload is None:
            drop[task["task_id"]] = "artifact unreadable at collect time"
            continue
        report = aa.audit_payload(payload)
        key = (task["group"], task["dataset"], task["protocol"], task["model"])
        arrays.setdefault(key, {})[task["seed"]] = aa._flat(payload["prediction"])
        if task["group"] == "lifetime" and report.get("identity") != "recorded":
            target = campaign / "derived" / "identity" / f"{task['task_id']}.json"
            if target.is_file():
                sidecars[task["task_id"]] = str(target)
                continue
            try:
                recovery = aa.recover_identity(Path(artifact), payload)
            except (aa.AuditError, AssertionError, OSError, ValueError, KeyError) as exc:
                recovery = {"status": "error", "reason": f"{type(exc).__name__}: {exc}"}
            if recovery.get("status") == "verified":
                recovery["created_by"] = "fig3_experiments collect stage"
                write_json_atomic(target, recovery)
                sidecars[task["task_id"]] = str(target)
            notes[task["task_id"]] = f"identity: {recovery.get('status')} {recovery.get('reason', '')}"
    for key, by_seed in arrays.items():
        if len(by_seed) > 1:
            effect = seed_effect(by_seed)
            if effect["identical_predictions"]:
                # Identical replicates would fake an error bar: keep seed 0 only.
                for task, _ in done:
                    if (task["group"], task["dataset"], task["protocol"], task["model"]) == key \
                            and task["seed"] != min(by_seed):
                        drop[task["task_id"]] = ("predictions identical to the lowest seed: the "
                                                 "seed did not reach the model; kept once as n=1")
    measurement = latest_numbered(campaign / "measurements", "measure")
    if measurement is not None and not (measurement / "latency.csv").is_file():
        measurement = None
    selection = build_selection(plan, state, sidecars=sidecars, drop=drop, measurement=measurement)
    selection["collect_notes"] = notes
    sel_path = next_numbered(campaign / "selection", "fig3_selection", ".json")
    write_json_atomic(sel_path, selection)
    out = next_numbered(campaign / "collected", "collected")
    try:
        collect(sel_path, out, argv=["collect", "--selection", str(sel_path), "--output", str(out)])
    except CollectError as exc:
        log(campaign, f"collect: FAILED: {exc}")
        return None
    log(campaign, f"collect: {len(selection['runs'])} entries into {out} (selection {sel_path}); "
                  f"dropped {drop or 'none'}; identity sidecars {list(sidecars) or 'none'}")
    return out


# --- plot / report -----------------------------------------------------------------

def stage_plot(campaign: Path, args) -> Optional[Path]:
    from scripts.paper_figures.plot_fig3_all import plot_all, status_table

    collected = latest_numbered(campaign / "collected", "collected")
    if collected is None:
        log(campaign, "plot: nothing collected yet")
        return None
    out = next_numbered(campaign / "figures", "figures")
    results = plot_all(collected, out, e_dataset="MATR", e_protocol="MATR1", e_seed="0",
                       argv=["--data", str(collected), "--output", str(out)])
    log(campaign, f"plot: {out}\n{status_table(results)}")
    return out


TABLE_COLUMNS = ["task_id", "group", "dataset", "protocol", "model", "seed", "seed_policy",
                 "config", "observation_budget", "status", "action", "run_id", "device",
                 "epochs_budget", "epochs_completed", "return_code", "RMSE", "MAE", "artifact",
                 "reason"]


def stage_report(campaign: Path, plan: Dict[str, Any]) -> None:
    state = load_state(campaign)
    rows = []
    for t in plan["tasks"]:
        e = state["tasks"][t["task_id"]]
        last = e["attempts"][-1] if e["attempts"] else {}
        acceptance = last.get("acceptance") or {}
        metrics = acceptance.get("metrics") or {}
        rows.append({
            "task_id": t["task_id"], "group": t["group"], "dataset": t["dataset"],
            "protocol": t["protocol"], "model": t["model"], "seed": t["seed"],
            "seed_policy": t["seed_policy"], "config": t["config"],
            "observation_budget": t["observation_budget"], "status": e["status"],
            "action": t["action"], "run_id": last.get("run_id") or (
                "reused" if e["status"] == REUSED else ""),
            "device": last.get("device", ""), "epochs_budget": t["training_budget"].get("epochs"),
            "epochs_completed": acceptance.get("epochs_completed", ""),
            "return_code": last.get("return_code", ""), "RMSE": metrics.get("RMSE"),
            "MAE": metrics.get("MAE"), "artifact": e.get("artifact") or "",
            "reason": e.get("reason", ""),
        })
    write_csv(campaign / "experiment_table.csv", rows, TABLE_COLUMNS)
    counts: Dict[str, int] = {}
    for r in rows:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    log(campaign, f"report: {campaign / 'experiment_table.csv'} — {counts}")


# --- entry point -------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--stages", default=",".join(STAGES))
    parser.add_argument("--runs-root", default="runs")
    parser.add_argument("--existing-selection", default=None)
    parser.add_argument("--cpu-threads", type=int, default=max(1, min(8, (os.cpu_count() or 2) // 2)))
    parser.add_argument("--gpu", default="auto", help="auto | none | <index>")
    parser.add_argument("--max-tasks", type=int, default=0, help="0 = no limit")
    parser.add_argument("--only-groups", default="", help="e.g. soh or soh,lifetime")
    parser.add_argument("--retry-failed", action="store_true")
    args = parser.parse_args(argv)
    args.only_groups = [g for g in args.only_groups.split(",") if g]
    stages = [s.strip() for s in args.stages.split(",") if s.strip()]
    unknown = [s for s in stages if s not in STAGES]
    if unknown:
        print(f"error: unknown stage(s) {unknown}; choose from {STAGES}", file=sys.stderr)
        return 2
    # Configs name data/processed/... and cache/ relative to the repository
    # root, exactly as the BatteryML CLI resolves them.
    os.chdir(REPO_ROOT)
    campaign = args.campaign if args.campaign.is_absolute() else REPO_ROOT / args.campaign
    campaign.mkdir(parents=True, exist_ok=True)
    try:
        with Lock(campaign / ".runner.lock"):
            plan = None
            for stage in STAGES:
                if stage not in stages:
                    continue
                if stage == "audit":
                    stage_audit(campaign, args)
                    continue
                plan = plan or stage_plan(campaign, args)
                if stage == "run":
                    stage_run(campaign, args, plan)
                elif stage == "measure":
                    stage_measure(campaign, args, plan)
                elif stage == "collect":
                    stage_collect(campaign, args, plan)
                elif stage == "plot":
                    stage_plot(campaign, args)
                elif stage == "report":
                    stage_report(campaign, plan)
    except CampaignError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("interrupted; rerun the same command to resume", file=sys.stderr)
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
