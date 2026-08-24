#!/usr/bin/env python
"""Manually reproduce BatteryML training/evaluation without the batteryml CLI."""

from __future__ import annotations

import argparse
import pickle
import shutil
from pathlib import Path

import torch

from batteryml.pipeline import build_dataset, load_config, set_seed, timestamp
from batteryml.builders import MODELS

# Import modules for registry side effects.
import batteryml.data.transformation.log_scale  # noqa: F401
import batteryml.data.transformation.sequential  # noqa: F401
import batteryml.data.transformation.z_score  # noqa: F401
import batteryml.feature  # noqa: F401
import batteryml.label  # noqa: F401
import batteryml.models.rul_predictors  # noqa: F401
import batteryml.train_test_split  # noqa: F401


def train_and_evaluate(
    config_path: Path,
    workspace: Path | None,
    seed: int,
    device: str,
    metrics: list[str],
    epochs: int | None,
    ckpt_to_resume: Path | None,
    save_train_predictions: bool,
) -> dict:
    set_seed(seed)
    config = load_config(str(config_path), str(workspace) if workspace is not None else None)

    if epochs is not None:
        config["model"]["epochs"] = epochs

    dataset, raw_data = build_dataset(config, device=device)

    model = MODELS.build(config["model"])
    if model.workspace is None:
        model.workspace = config["workspace"]
    if ckpt_to_resume is not None:
        model.load_checkpoint(str(ckpt_to_resume))
    if torch.__version__ >= "2" and isinstance(model, torch.nn.Module):
        model = torch.compile(model)
    model = model.to(device)

    run_ts = timestamp()
    if model.workspace is not None:
        shutil.copyfile(config_path, model.workspace / f"manual_config_{run_ts}.yaml")

    model.fit(dataset, timestamp=f"manual_seed_{seed}_{run_ts}")

    test_prediction = model.predict(dataset, data_type="test")
    scores = {metric: dataset.evaluate(test_prediction, metric, data_type="test") for metric in metrics}
    result = {
        "scores": scores,
        "test_prediction": test_prediction.detach().cpu(),
        "seed": seed,
        "config_path": str(config_path),
        "raw_data": raw_data,
    }

    print("\nTest scores")
    print("-" * 80)
    for metric, value in scores.items():
        print(f"{metric}: {value}")

    if save_train_predictions:
        train_prediction = model.predict(dataset, data_type="train")
        train_scores = {metric: dataset.evaluate(train_prediction, metric, data_type="train") for metric in metrics}
        result["train_prediction"] = train_prediction.detach().cpu()
        result["train_scores"] = train_scores
        print("\nTrain scores")
        print("-" * 80)
        for metric, value in train_scores.items():
            print(f"{metric}: {value}")

    if config["workspace"] is not None:
        output_path = config["workspace"] / f"manual_predictions_seed_{seed}_{run_ts}.pkl"
        with open(output_path, "wb") as fout:
            pickle.dump({**result, "data": dataset.to("cpu")}, fout)
        print(f"\nSaved predictions to {output_path}")

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/baselines/sklearn/xgb/matr_1.yaml",
        type=Path,
        help="BatteryML YAML config to reproduce.",
    )
    parser.add_argument(
        "--workspace",
        default="workspace/manual_xgb_matr_1",
        type=Path,
        help="Directory for checkpoints, copied config and predictions. Use 'none' to disable.",
    )
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--metrics", default="RMSE,MAE,MAPE")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--ckpt-to-resume", type=Path, default=None)
    parser.add_argument("--save-train-predictions", action="store_true")
    args = parser.parse_args()

    workspace = None if str(args.workspace).lower() == "none" else args.workspace
    metrics = [item.strip() for item in args.metrics.split(",") if item.strip()]

    train_and_evaluate(
        config_path=args.config,
        workspace=workspace,
        seed=args.seed,
        device=args.device,
        metrics=metrics,
        epochs=args.epochs,
        ckpt_to_resume=args.ckpt_to_resume,
        save_train_predictions=args.save_train_predictions,
    )


if __name__ == "__main__":
    main()
