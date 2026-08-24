from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List

import yaml

SUPPORTED_TASKS = {"SOH", "RUL"}
SUPPORTED_MODELS = {
    "XGBoostRULPredictor",
    "DummyRULPredictor",
}
SUPPORTED_SPLITS = {
    "MATRPrimaryTestTrainTestSplitter",
}


def _as_path(value: Any) -> Path | None:
    if value is None:
        return None
    path = Path(str(value))
    return path


def validate_config(config: Dict[str, Any]) -> Dict[str, Any]:
    errors: List[str] = []
    warnings: List[str] = []

    config_path = _as_path(config.get("config_path"))
    if config_path is not None and not config_path.exists():
        errors.append(f"Config file does not exist: {config_path}")

    task = str(config.get("task", "")).upper()
    if task and task not in SUPPORTED_TASKS:
        errors.append(f"Unsupported task: {task}")

    model = config.get("model", {})
    model_name = model.get("name") if isinstance(model, dict) else None
    if model_name not in SUPPORTED_MODELS:
        errors.append(f"Unsupported model: {model_name}")

    split = config.get("train_test_split", {})
    split_name = split.get("name") if isinstance(split, dict) else None
    if split_name not in SUPPORTED_SPLITS:
        errors.append(f"Unsupported split strategy: {split_name}")

    dataset_path = split.get("cell_data_path") if isinstance(split, dict) else None
    if dataset_path is not None:
        dataset_path = Path(str(dataset_path))
        if not dataset_path.exists():
            warnings.append(f"Dataset path does not exist locally yet: {dataset_path}")

    output_dir = _as_path(config.get("output_dir") or config.get("workspace"))
    if output_dir is not None:
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
            writable_test = output_dir / ".write_test"
            writable_test.write_text("ok", encoding="utf-8")
            writable_test.unlink(missing_ok=True)
        except Exception as exc:
            errors.append(f"Output directory is not writable: {exc}")

    valid = not errors
    return {
        "valid": valid,
        "errors": errors,
        "warnings": warnings,
        "task": task,
        "model": model_name,
        "split_strategy": split_name,
        "dataset_path": str(dataset_path) if dataset_path is not None else None,
    }


def load_and_validate_config_file(path: str | Path, task: str, output_dir: str | Path | None = None) -> Dict[str, Any]:
    path = Path(path)
    if not path.exists():
        return {
            "valid": False,
            "errors": [f"Config file does not exist: {path}"],
            "warnings": [],
        }
    with path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}
    config["config_path"] = str(path)
    config["task"] = task
    if output_dir is not None:
        config["output_dir"] = str(output_dir)
    return validate_config(config)
