#!/usr/bin/env python
"""Prepare BatteryML raw files and run preprocessors without using the CLI."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from batteryml.preprocess import DOWNLOAD_LINKS, download_file
from batteryml.preprocess.preprocess_CALCE import CALCEPreprocessor
from batteryml.preprocess.preprocess_HNEI import HNEIPreprocessor
from batteryml.preprocess.preprocess_HUST import HUSTPreprocessor
from batteryml.preprocess.preprocess_MATR import MATRPreprocessor
from batteryml.preprocess.preprocess_RWTH import RWTHPreprocessor
from batteryml.preprocess.preprocess_SNL import SNLPreprocessor
from batteryml.preprocess.preprocess_UL_PUR import UL_PURPreprocessor


PREPROCESSORS = {
    "MATR": MATRPreprocessor,
    "HUST": HUSTPreprocessor,
    "CALCE": CALCEPreprocessor,
    "RWTH": RWTHPreprocessor,
    "SNL": SNLPreprocessor,
    "HNEI": HNEIPreprocessor,
    "UL_PUR": UL_PURPreprocessor,
}

MATR_FILENAMES = [
    "MATR_batch_20170512.mat",
    "MATR_batch_20170630.mat",
    "MATR_batch_20180412.mat",
    "MATR_batch_20190124.mat",
]

BATTERY_ARCHIVE_DATASETS = {"SNL", "HNEI", "UL_PUR"}


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def bootstrap_existing_matr_files(raw_root: Path, project_root: Path, move: bool) -> None:
    raw_matr_dir = raw_root / "MATR"
    ensure_dir(raw_matr_dir)

    for filename in MATR_FILENAMES:
        target = raw_matr_dir / filename
        if target.exists():
            continue

        candidates = [project_root / "data" / filename, project_root / filename]
        source = next((path for path in candidates if path.exists()), None)
        if source is None:
            continue

        if move:
            print(f"Move {source} -> {target}")
            shutil.move(str(source), str(target))
        else:
            print(f"Copy {source} -> {target}")
            shutil.copy2(source, target)


def download_dataset(dataset: str, raw_dir: Path) -> None:
    if dataset not in DOWNLOAD_LINKS:
        print(
            f"Skip download for {dataset}: no direct link in BatteryML. "
            "Download it manually from Battery Archive and put files under "
            f"{raw_dir}."
        )
        return

    ensure_dir(raw_dir)
    for item in DOWNLOAD_LINKS[dataset]:
        if len(item) == 2:
            url, filename = item
            total_length = None
        else:
            url, filename, total_length = item
        download_file(url, raw_dir / filename, total_length=total_length)


def preprocess_dataset(dataset: str, raw_dir: Path, processed_dir: Path, silent: bool) -> None:
    ensure_dir(processed_dir)
    preprocessor_cls = PREPROCESSORS[dataset]
    processor = preprocessor_cls(output_dir=processed_dir, silent=silent)
    processor(raw_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["MATR"],
        choices=sorted(PREPROCESSORS),
        help="Datasets to prepare.",
    )
    parser.add_argument("--raw-root", default="data/raw", type=Path)
    parser.add_argument("--processed-root", default="data/processed", type=Path)
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download raw files when BatteryML has direct download links.",
    )
    parser.add_argument(
        "--preprocess",
        action="store_true",
        help="Run dataset-specific preprocessors and generate BatteryData .pkl files.",
    )
    parser.add_argument(
        "--bootstrap-local-matr",
        action="store_true",
        help="Copy local data/MATR_batch_*.mat files into data/raw/MATR before preprocessing.",
    )
    parser.add_argument(
        "--move-local-matr",
        action="store_true",
        help="Move instead of copy when --bootstrap-local-matr is used.",
    )
    parser.add_argument("--silent", action="store_true", help="Suppress preprocessor progress logs.")
    args = parser.parse_args()

    project_root = Path.cwd()

    if args.bootstrap_local_matr:
        bootstrap_existing_matr_files(args.raw_root, project_root, move=args.move_local_matr)

    if not args.download and not args.preprocess:
        print("Nothing to do. Use --download and/or --preprocess.")
        return

    for dataset in args.datasets:
        raw_dir = args.raw_root / dataset
        processed_dir = args.processed_root / dataset

        if args.download:
            download_dataset(dataset, raw_dir)

        if args.preprocess:
            if dataset in BATTERY_ARCHIVE_DATASETS and not raw_dir.exists():
                raise FileNotFoundError(
                    f"{dataset} raw directory does not exist: {raw_dir}. "
                    "Download it manually from Battery Archive first."
                )
            preprocess_dataset(dataset, raw_dir, processed_dir, silent=args.silent)


if __name__ == "__main__":
    main()
