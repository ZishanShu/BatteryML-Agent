#!/usr/bin/env python
"""Check raw/processed data availability for BatteryML reproduction."""

from __future__ import annotations

import argparse
from pathlib import Path


PUBLIC_DOWNLOADABLE = {
    "MATR": [
        "MATR_batch_20170512.mat",
        "MATR_batch_20170630.mat",
        "MATR_batch_20180412.mat",
        "MATR_batch_20190124.mat",
    ],
    "HUST": ["hust_data.zip"],
    "CALCE": [
        "CS2_33.zip",
        "CS2_34.zip",
        "CS2_35.zip",
        "CS2_36.zip",
        "CS2_37.zip",
        "CS2_38.zip",
        "CX2_16.zip",
        "CX2_33.zip",
        "CX2_34.zip",
        "CX2_35.zip",
        "CX2_36.zip",
        "CX2_37.zip",
        "CX2_38.zip",
    ],
    "RWTH": ["RWTH.zip"],
}

BATTERY_ARCHIVE = {
    "SNL": ["*_timeseries.csv", "*_cycle_data.csv"],
    "HNEI": ["*HNEI*timeseries.csv", "*HNEI*cycle_data.csv"],
    "UL_PUR": ["*UL-PUR_N*timeseries.csv", "*UL-PUR_N*cycle_data.csv"],
}

BENCHMARK_REQUIREMENTS = {
    "matr_1/matr_2": ["MATR"],
    "hust": ["HUST"],
    "snl": ["SNL"],
    "cruh": ["CALCE", "RWTH", "UL_PUR", "HNEI"],
    "crush": ["CALCE", "RWTH", "UL_PUR", "SNL", "HNEI"],
    "mix": ["CALCE", "RWTH", "UL_PUR", "HNEI", "MATR", "HUST"],
}


def mark(ok: bool) -> str:
    return "OK" if ok else "MISSING"


def count_processed(processed_root: Path, dataset: str) -> int:
    processed_dir = processed_root / dataset
    if not processed_dir.exists():
        return 0
    return len(list(processed_dir.glob("*.pkl")))


def has_raw(raw_root: Path, dataset: str) -> bool:
    raw_dir = raw_root / dataset
    if dataset in PUBLIC_DOWNLOADABLE:
        return all((raw_dir / name).exists() for name in PUBLIC_DOWNLOADABLE[dataset])
    if dataset in BATTERY_ARCHIVE:
        return all(any(raw_dir.glob(pattern)) for pattern in BATTERY_ARCHIVE[dataset])
    return False


def print_raw_status(raw_root: Path) -> None:
    print("\nRaw data status")
    print("-" * 80)
    for dataset in ["MATR", "HUST", "CALCE", "RWTH", "SNL", "HNEI", "UL_PUR"]:
        exists = has_raw(raw_root, dataset)
        source = "BatteryML download" if dataset in PUBLIC_DOWNLOADABLE else "Battery Archive manual download"
        print(f"{dataset:<8} {mark(exists):<8} source={source} path={raw_root / dataset}")
        if not exists and dataset in PUBLIC_DOWNLOADABLE:
            missing = [name for name in PUBLIC_DOWNLOADABLE[dataset] if not (raw_root / dataset / name).exists()]
            for name in missing[:5]:
                print(f"          missing: {name}")
            if len(missing) > 5:
                print(f"          ... {len(missing) - 5} more")


def print_processed_status(processed_root: Path) -> None:
    print("\nProcessed data status")
    print("-" * 80)
    for dataset in ["MATR", "HUST", "CALCE", "RWTH", "SNL", "HNEI", "UL_PUR"]:
        count = count_processed(processed_root, dataset)
        print(f"{dataset:<8} {mark(count > 0):<8} pkl_count={count:<4} path={processed_root / dataset}")


def print_benchmark_readiness(processed_root: Path) -> None:
    print("\nBenchmark readiness based on processed .pkl files")
    print("-" * 80)
    for name, datasets in BENCHMARK_REQUIREMENTS.items():
        missing = [dataset for dataset in datasets if count_processed(processed_root, dataset) == 0]
        if missing:
            print(f"{name:<14} MISSING processed datasets: {', '.join(missing)}")
        else:
            print(f"{name:<14} OK")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", default="data/raw", type=Path)
    parser.add_argument("--processed-root", default="data/processed", type=Path)
    args = parser.parse_args()

    print_raw_status(args.raw_root)
    print_processed_status(args.processed_root)
    print_benchmark_readiness(args.processed_root)


if __name__ == "__main__":
    main()
