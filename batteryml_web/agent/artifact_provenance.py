"""What is recorded about one prediction artifact, read from beside it only.

A figure is only as trustworthy as the statement of where its numbers came
from. This module answers that for **one exact artifact path** and nothing
else: it hashes the file and reads the two records a run leaves next to it —

* ``run.json`` (written by the platform for Agent Workflow, Run Model and
  MATR-demo runs): run id, model, config path, seed, epochs, smoke flag,
  status, command;
* ``config_<timestamp>.yaml`` (copied by ``Pipeline.train``): the label
  annotator and its parameters, the split, the model class.

It never searches another directory, never picks "the newest run", and never
infers a fact from a directory name. A legacy CLI workspace has no
``run.json``; then the run-level fields say "not recorded" rather than being
guessed — in particular a directory called ``..._smoke_...`` is **not**
evidence of a smoke run, and a Ridge fit has no epochs to infer.

One trap is recorded rather than worked around: ``Pipeline.train`` copies the
**original** config file before applying a ``--epochs`` override, so the copy
of a 1-epoch smoke run still shows the config's full epoch count. Only
``run.json`` (``smoke``, ``epochs``, ``command``) records the override.

No pickle is read here; the artifact is hashed as bytes.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

NOT_RECORDED = "not recorded"

#: Said wherever the config copy's epoch count is shown.
CONFIG_COPY_EPOCHS_NOTE = (
    "config_<ts>.yaml is the original config file copied before any --epochs "
    "override, so its epoch count does not show a smoke override; run.json "
    "does."
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_run_manifest(directory: Path) -> Optional[Dict[str, Any]]:
    path = directory / "run.json"
    if not path.is_file():
        return None
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return loaded if isinstance(loaded, dict) else None


def _read_config_copy(path: Path) -> Optional[Dict[str, Any]]:
    import yaml

    try:
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError):
        return None
    return loaded if isinstance(loaded, dict) else None


def _named_section(config: Dict[str, Any], key: str) -> Dict[str, Any]:
    section = config.get(key)
    return section if isinstance(section, dict) else {}


def _transformation_names(section: Dict[str, Any]) -> List[str]:
    if not section:
        return []
    members = section.get("transformations")
    if isinstance(members, list) and members:
        return [str(member.get("name")) for member in members if isinstance(member, dict)]
    return [str(section.get("name"))] if section.get("name") else []


def artifact_provenance(artifact: str | Path, *, hash_files: bool = True) -> Dict[str, Any]:
    """Everything recorded about ``artifact``, from its own directory.

    ``hash_files=False`` skips the SHA-256 digests (the page reads this on
    every rerun; the export always hashes).

    Raises:
        FileNotFoundError: ``artifact`` is not a file. The caller named one
            exact artifact; there is no fallback to another.
    """
    path = Path(artifact)
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} is not a file. Pass the exact predictions_seed_<seed>_<ts>.pkl "
            "path; nothing searches for another artifact."
        )
    directory = path.parent
    manifest = _read_run_manifest(directory)
    configs = sorted(directory.glob("config_*.yaml"))

    record: Dict[str, Any] = {
        "artifact_path": str(path),
        "artifact_sha256": sha256_file(path) if hash_files else NOT_RECORDED,
        "artifact_size_bytes": path.stat().st_size,
        "run_manifest": "run.json" if manifest is not None else NOT_RECORDED,
        "run_id": NOT_RECORDED,
        "model_key": NOT_RECORDED,
        "model_display": NOT_RECORDED,
        "dataset": NOT_RECORDED,
        "task": NOT_RECORDED,
        "config_path": NOT_RECORDED,
        "seed": NOT_RECORDED,
        "epochs_override": NOT_RECORDED,
        "smoke": NOT_RECORDED,
        "status": NOT_RECORDED,
        "command": NOT_RECORDED,
        "config_copies": [config.name for config in configs],
        "config_copy": NOT_RECORDED,
        "config_copy_sha256": NOT_RECORDED,
        "model_class": NOT_RECORDED,
        "config_copy_epochs": NOT_RECORDED,
        "split": NOT_RECORDED,
        "label": NOT_RECORDED,
        "label_annotator": None,
        "label_transformation_config": [],
        "notes": [],
    }
    if manifest is not None:
        for key in ("run_id", "model_key", "model_display", "dataset", "task",
                    "config_path", "seed", "status"):
            if manifest.get(key) not in (None, ""):
                record[key] = manifest[key]
        if manifest.get("epochs") not in (None, ""):
            record["epochs_override"] = manifest["epochs"]
        if "smoke" in manifest:
            record["smoke"] = bool(manifest["smoke"])
        if manifest.get("command"):
            record["command"] = manifest["command"]
    else:
        record["notes"].append(
            "No run.json beside the artifact: run id, seed, smoke status and "
            "epochs are not recorded for it. A directory name is not used as "
            "evidence of any of them."
        )

    if len(configs) == 1:
        config = _read_config_copy(configs[0])
        record["config_copy"] = configs[0].name
        record["config_copy_sha256"] = sha256_file(configs[0]) if hash_files else NOT_RECORDED
        if config is not None:
            model = _named_section(config, "model")
            label = _named_section(config, "label")
            split = _named_section(config, "train_test_split")
            record["model_class"] = model.get("name") or NOT_RECORDED
            if "epochs" in model:
                record["config_copy_epochs"] = model.get("epochs")
                record["notes"].append(CONFIG_COPY_EPOCHS_NOTE)
            record["split"] = split.get("name") or NOT_RECORDED
            if label:
                record["label"] = dict(label)
                record["label_annotator"] = label.get("name")
            record["label_transformation_config"] = _transformation_names(
                _named_section(config, "label_transformation")
            )
    elif len(configs) > 1:
        record["notes"].append(
            f"{len(configs)} config copies sit beside the artifact; none is "
            "chosen, because the copy's timestamp does not identify which "
            "training run produced this artifact."
        )
    else:
        record["notes"].append("No config copy beside the artifact.")
    return record


def smoke_evidence(record: Dict[str, Any]) -> str:
    """One sentence on whether this is a smoke run, from records only."""
    smoke = record.get("smoke")
    if smoke is True:
        epochs = record.get("epochs_override")
        return (
            f"run.json records smoke=true (epochs override {epochs}). "
            "Workflow validation only; not benchmark performance."
        )
    if smoke is False:
        epochs = record.get("epochs_override")
        if epochs not in (None, NOT_RECORDED):
            return f"run.json records smoke=false with an epochs override of {epochs}."
        return "run.json records smoke=false and no epochs override (config epochs)."
    return (
        "Smoke status not recorded (no run.json beside the artifact). Do not "
        "infer it from the directory name."
    )
