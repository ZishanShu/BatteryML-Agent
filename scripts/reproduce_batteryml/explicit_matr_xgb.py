#!/usr/bin/env python
"""Fully explicit MATR XGBoost RUL reproduction without Pipeline/builders."""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

from batteryml.data.transformation.log_scale import LogScaleDataTransformation
from batteryml.data.transformation.sequential import SequentialDataTransformation
from batteryml.data.transformation.z_score import ZScoreDataTransformation
from batteryml.feature.voltage_capacity_matrix import VoltageCapacityMatrixFeatureExtractor
from batteryml.label.rul import RULLabelAnnotator
from batteryml.models.rul_predictors.xgb import XGBoostRULPredictor
from batteryml.task import Task
from batteryml.train_test_split.MATR_split import (
    MATRCLOTestTrainTestSplitter,
    MATRPrimaryTestTrainTestSplitter,
    MATRSecondaryTestTrainTestSplitter,
)

SPLITTERS = {
    "matr_1": MATRPrimaryTestTrainTestSplitter,
    "matr_2": MATRSecondaryTestTrainTestSplitter,
    "clo": MATRCLOTestTrainTestSplitter,
}


def build_matr_task(processed_matr_dir: Path, split: str) -> Task:
    splitter_cls = SPLITTERS[split]
    splitter = splitter_cls(str(processed_matr_dir))

    feature_extractor = VoltageCapacityMatrixFeatureExtractor(
        diff_base=8,
        max_cycle_index=98,
        cycles_to_keep=98,
        use_precalculated_qdlin=True,
    )
    label_annotator = RULLabelAnnotator()
    feature_transformation = ZScoreDataTransformation()
    label_transformation = SequentialDataTransformation(
        transformations=[
            LogScaleDataTransformation(),
            ZScoreDataTransformation(),
        ]
    )

    return Task(
        train_test_splitter=splitter,
        feature_extractor=feature_extractor,
        label_annotator=label_annotator,
        feature_transformation=feature_transformation,
        label_transformation=label_transformation,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-matr-dir", default="data/processed/MATR", type=Path)
    parser.add_argument("--split", choices=sorted(SPLITTERS), default="matr_1")
    parser.add_argument("--workspace", default="workspace/explicit_matr_xgb", type=Path)
    parser.add_argument("--metrics", default="RMSE,MAE,MAPE")
    args = parser.parse_args()

    args.workspace.mkdir(parents=True, exist_ok=True)

    task = build_matr_task(args.processed_matr_dir, args.split)
    dataset = task.build()

    model = XGBoostRULPredictor(workspace=args.workspace)
    model.fit(dataset, timestamp=f"explicit_{args.split}")

    prediction = model.predict(dataset, data_type="test")
    metrics = [item.strip() for item in args.metrics.split(",") if item.strip()]
    scores = {metric: dataset.evaluate(prediction, metric, data_type="test") for metric in metrics}

    print("\nExplicit MATR XGBoost scores")
    print("-" * 80)
    for metric, value in scores.items():
        print(f"{metric}: {value}")

    output_path = args.workspace / f"explicit_{args.split}_predictions.pkl"
    with open(output_path, "wb") as fout:
        pickle.dump(
            {
                "prediction": prediction.detach().cpu(),
                "scores": scores,
                "split": args.split,
                "dataset": dataset.to("cpu"),
            },
            fout,
        )
    print(f"\nSaved predictions to {output_path}")


if __name__ == "__main__":
    main()
