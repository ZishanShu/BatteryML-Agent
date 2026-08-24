from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml


def _base_template(dataset: str, split_strategy: str) -> Dict[str, Any]:
    dataset = dataset.upper()
    split_strategy = split_strategy.upper()

    if dataset != "MATR":
        raise ValueError("Phase 5 quick-start currently only supports MATR.")

    split_name_map = {
        "PRIMARY": "MATRPrimaryTestTrainTestSplitter",
        "DEFAULT": "MATRPrimaryTestTrainTestSplitter",
    }
    split_name = split_name_map.get(split_strategy, split_strategy)

    return {
        "train_test_split": {
            "name": split_name,
            "cell_data_path": "data/processed/MATR",
        },
        "feature": {
            "name": "VoltageCapacityMatrixFeatureExtractor",
            "diff_base": 8,
            "max_cycle_index": 98,
            "cycles_to_keep": 98,
            "use_precalculated_qdlin": True,
        },
        "feature_transformation": {"name": "ZScoreDataTransformation"},
        "label_transformation": {
            "name": "SequentialDataTransformation",
            "transformations": [
                {"name": "LogScaleDataTransformation"},
                {"name": "ZScoreDataTransformation"},
            ],
        },
    }


def build_quickstart_config(dataset: str, task: str, model: str) -> Dict[str, Any]:
    return build_task_config(dataset=dataset, task=task, model=model, split_strategy="PRIMARY")


def build_task_config(dataset: str, task: str, model: str, split_strategy: str) -> Dict[str, Any]:
    task = task.upper()
    model_name = model
    config = _base_template(dataset, split_strategy)
    config["model"] = {"name": model_name}

    if task == "RUL":
        config["label"] = {"name": "RULLabelAnnotator"}
    elif task == "SOH":
        config["label"] = {"name": "SOHLabelAnnotator"}
    else:
        raise ValueError(f"Unsupported task: {task}")

    return config


def save_config_yaml(config: Dict[str, Any], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False)
    return path
